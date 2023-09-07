from typing import List, Union
import pytorch_lightning as pl
import torch
from botorch.acquisition import (
    ExpectedImprovement,
    NoisyExpectedImprovement,
    UpperConfidenceBound,
)
from botorch.acquisition.monte_carlo import qUpperConfidenceBound
from botorch.sampling.normal import SobolQMCNormalSampler
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from chaos.data.utils import torch_delete_rows
from chaos.plotting.bo_plotting import BOPlotter
from chaos.utils import instantiate_class
import wandb
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from chaos.gprotorch.metrics import (
    negative_log_predictive_density,
    mean_standardized_log_loss,
    quantile_coverage_error,
)

from chaos.results import compute_top5_metrics


class BoModule(pl.LightningModule):
    def __init__(
        self,
        data: pl.LightningDataModule,
        model_config: dict = None,
        acquisition_class: str = "ucb",
        beta: float = 0.1,
        top_n: List[int] = [1, 3, 5, 10],
        batch_size: int = 1,
        finetuning: bool = True,
        enable_plotting: bool = True,
        enable_logging_images: bool = False,
    ):
        super().__init__()
        self.top_count = None
        self.data = data
        self.model_config = model_config
        self.acquisition = None
        self.top_n = top_n
        self.finetuning = finetuning
        self.acquisition_class = acquisition_class
        self.beta = beta
        self.batch_size = batch_size
        self.enable_plotting = enable_plotting
        self.enable_logging_images = enable_logging_images
        self.plotting_utils = BOPlotter() if enable_plotting else None
        self.prev_state = None

        self.save_hyperparameters(
            ignore=["kernel", "model", "mll", "data", "acquisition_class", "beta"]
        )
        self.automatic_optimization = False

    def log_diversity_metrics(self, data_matrix):
        kernel_operator = self.model.covar_module(data_matrix)
        kernel_matrix = kernel_operator.evaluate()
        kernel_matrix = kernel_matrix.detach().cpu().numpy()
        np.fill_diagonal(kernel_matrix, np.nan)

        scaler = MinMaxScaler()

        kernel_matrix_normalized = scaler.fit_transform(kernel_matrix)

        # Compute the average similarity
        avg_similarity = np.nanmean(kernel_matrix_normalized)

        self.log("average_similarity", avg_similarity)

    # BO iteration
    def training_step(self, batch, batch_idx):
        train_x, train_y = self.data.train_x, self.data.train_y
        self.log("train/best_so_far", torch.max(train_y))
        self.model = instantiate_class(
            self.model_config, train_x=train_x, train_y=train_y
        )
        if self.acquisition_class != "random":
            self.model.fit(train_x, train_y)

        heldout_x, heldout_y = self.data.heldout_x, self.data.heldout_y
        new_x, new_y, suggestion_ids = self.optimize_acqf_and_get_observation(
            heldout_x, heldout_y
        )
        if self.enable_plotting:
            if self.global_step % 100 == 0:
                predictions_valid, var_valid = self.model.predict(
                    heldout_x, observation_noise=True, return_var=True
                )
                predictions_train, var_train = self.model.predict(
                    train_x, observation_noise=True, return_var=True
                )
                if self.enable_logging_images:
                    pred_vs_gt_fig = self.plotting_utils.plot_predicted_vs_actual(
                        predictions_train,
                        train_y,
                        var_train.sqrt(),
                        predictions_valid,
                        heldout_y,
                        var_valid.sqrt(),
                    )
                    residuals_figure = self.plotting_utils.plot_residuals(
                        predictions_valid, heldout_y
                    )
                    self.logger.experiment.log(
                        {"pred-vs-gt": wandb.Image(pred_vs_gt_fig)}
                    )
                    self.logger.experiment.log(
                        {"residuals": wandb.Image(residuals_figure)}
                    )

                mse_valid = mean_squared_error(heldout_y, predictions_valid)
                r2_valid = r2_score(heldout_y, predictions_valid)
                mae_valid = mean_absolute_error(heldout_y, predictions_valid)

                mse_train = mean_squared_error(train_y, predictions_train)
                r2_train = r2_score(train_y, predictions_train)
                mae_train = mean_absolute_error(train_y, predictions_train)

                pred_dist_valid = self.model.posterior(
                    heldout_x, observation_noise=True
                )
                pred_dist_train = self.model.posterior(train_x, observation_noise=True)

                # Compute GP-specific uncertainty metrics
                nlpd_valid = negative_log_predictive_density(pred_dist_valid, heldout_y)
                msll_valid = mean_standardized_log_loss(pred_dist_valid, heldout_y)
                qce_valid = quantile_coverage_error(pred_dist_valid, heldout_y)

                nlpd_train = negative_log_predictive_density(pred_dist_train, train_y)
                msll_train = mean_standardized_log_loss(pred_dist_train, train_y)
                qce_train = quantile_coverage_error(pred_dist_train, train_y)

                # Compute metrics for top 5%, bottom 5%, and all
                nlpd_top_5, r2_top_5, mae_top_5 = compute_top5_metrics(
                    self.model, self.data, "top_5"
                )
                nlpd_bottom_5, r2_bottom_5, mae_bottom_5 = compute_top5_metrics(
                    self.model, self.data, "bottom_5"
                )
                nlpd_all, r2_all, mae_all = compute_top5_metrics(
                    self.model, self.data, "all"
                )

                self.logger.experiment.log(
                    {
                        "train/mse": mse_train,
                        "train/mae": mae_train,
                        "train/r2": r2_train,
                        "valid/mse": mse_valid,
                        "valid/mae": mae_valid,
                        "valid/r2": r2_valid,
                        "train/nlpd": nlpd_train,
                        "train/msll": msll_train,
                        "train/qce": qce_train,
                        "valid/nlpd": nlpd_valid,
                        "valid/msll": msll_valid,
                        "valid/qce": qce_valid,
                        "NLPD_top_5": nlpd_top_5,
                        "R2_top_5": r2_top_5,
                        "MAE_top_5": mae_top_5,
                        "NLPD_bottom_5": nlpd_bottom_5,
                        "R2_bottom_5": r2_bottom_5,
                        "MAE_bottom_5": mae_bottom_5,
                        "NLPD_all": nlpd_all,
                        "R2_all": r2_all,
                        "MAE_all": mae_all,
                    }
                )

                self.log_diversity_metrics(self.data.train_x)
                self.log_model_parameters()

        self.update_data(new_x, new_y, suggestion_ids)

    def on_train_start(self) -> None:
        self.data.log_data_stats(self.logger)

    def on_train_end(self) -> None:
        self.data.log_top_n_counts(self.logger)
        self.data.log_quantile_counts(self.logger)
        self.logger.experiment.finish()

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return self.data.train_dataloader()

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return self.data.val_dataloader()

    def configure_optimizers(self):
        pass

    def construct_acquisition(self):
        if self.acquisition_class == "ei":
            self.acquisition = ExpectedImprovement(
                model=self.model,
                best_f=self.data.train_y.max(),
            )

        elif self.acquisition_class == "nei":
            self.acquisition = NoisyExpectedImprovement(
                model=self.model,
                X_observed=self.data.train_x,
                num_fantasies=100,
            )

        elif self.acquisition_class == "ucb":
            self.acquisition = UpperConfidenceBound(
                model=self.model,
                beta=self.beta,
            )

        elif self.acquisition_class == "qucb":
            sampler = SobolQMCNormalSampler(1024)
            self.acquisition = qUpperConfidenceBound(self.model, self.beta, sampler)

    def optimize_acquisition(self, heldout_x, batch_size=4):
        self.model.eval()
        acq_vals = self.acquisition(heldout_x.unsqueeze(-2))
        best_idxs = torch.argsort(acq_vals, descending=True)[:batch_size]
        self.log("sum_acq_values", acq_vals.sum())
        self.log("suggestion_idx", best_idxs[0])

        return best_idxs

    def optimize_acqf_and_get_observation(self, heldout_x, heldout_y):
        if self.acquisition_class == "random":
            best_idxs = torch.randperm(len(heldout_y))[: self.batch_size]
        else:
            self.construct_acquisition()
            best_idxs = self.optimize_acquisition(heldout_x, batch_size=self.batch_size)
        return heldout_x[best_idxs], heldout_y[best_idxs], best_idxs

    def update_data(self, new_x, new_y, best_idxs):
        self.data.train_indexes.extend(best_idxs)
        self.log("train/suggestion", torch.max(new_y))

        # update heldout set points
        self.data.heldout_x = torch_delete_rows(self.data.heldout_x, best_idxs)
        self.data.heldout_y = torch_delete_rows(self.data.heldout_y, best_idxs)

        # update training set points
        self.data.train_x = torch.cat([self.data.train_x, new_x])
        self.data.train_y = torch.cat([self.data.train_y, new_y])

    def log_model_parameters(self):
        for name, param in self.model.named_hyperparameters():
            transformed_name = name.replace("raw_", "")
            attr = self.model
            for part in transformed_name.split("."):
                attr = getattr(attr, part)
            value = attr.cpu().detach().numpy()

            self.logger.experiment.log({transformed_name: value})
