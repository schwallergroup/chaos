import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
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

from chaos.data.module import BaseDataModule
from chaos.data.module import Featurizer
from chaos.initialization.initializers import BOInitializer

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
from pytorch_lightning import seed_everything

seed_everything(1)

featurizer = Featurizer(task="reaction_optimization", representation="drfp", nBits=256)
initializer = BOInitializer(n_clusters=10, method="true_random")
data = BaseDataModule(
    "data/reactions/bh/bh_reaction_1.csv",
    input_column="rxn",
    target_column="yield",
    featurizer=featurizer,
    initializer=initializer,
)

model_config = model_config = {
    "class_path": "chaos.surrogate_models.gp.SimpleGP",
    "init_args": {
        "likelihood": {
            "class_path": "gpytorch.likelihoods.GaussianLikelihood",
        },
        "covar_module": {
            "class_path": "chaos.surrogate_models.gp.MyAdditiveKernel",
            "init_args": {
                "kernel_config": [
                    {
                        "class_path": "gpytorch.kernels.rq_kernel.RQKernel",
                        # "init_args": {"nu": 0.5}
                    },
                    {
                        "class_path": "gpytorch.kernels.matern_kernel.MaternKernel"
                        # "chaos.gprotorch.kernels.fingerprint_kernels.tanimoto_kernel.TanimotoKernel" #'gpytorch.kernels.linear_kernel.LinearKernel'
                    },
                ]
            },
        },
        "standardize": True,
        "normalize": False,
        "initial_noise_val": 1.0,
        "noise_constraint": 1.0e-05,
    },
}


class GPRegression(pl.LightningModule):
    def __init__(
        self,
        data: pl.LightningDataModule,
        model_config: dict,
        enable_plotting: bool = True,
    ):
        super().__init__()
        self.top_count = None
        self.data = data
        self.model_config = model_config
        self.enable_plotting = enable_plotting
        self.plotting_utils = BOPlotter() if enable_plotting else None
        self.prev_state = None
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

    def training_step(self, batch, batch_idx):
        train_x, train_y = self.data.train_x, self.data.train_y
        self.model = instantiate_class(
            self.model_config, train_x=train_x, train_y=train_y
        )

        self.model.fit(train_x, train_y)

        heldout_x, heldout_y = self.data.heldout_x, self.data.heldout_y

        if self.enable_plotting:
            if self.global_step % 100 == 0:
                predictions_valid, var_valid = self.model.predict(
                    heldout_x, observation_noise=True, return_var=True
                )
                predictions_train, var_train = self.model.predict(
                    train_x, observation_noise=True, return_var=True
                )
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

                mse_valid = mean_squared_error(heldout_y, predictions_valid)
                r2_valid = r2_score(heldout_y, predictions_valid)
                mae_valid = mean_absolute_error(heldout_y, predictions_valid)

                mse_train = mean_squared_error(heldout_y, predictions_valid)
                r2_train = r2_score(heldout_y, predictions_valid)
                mae_train = mean_absolute_error(heldout_y, predictions_valid)

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

                self.logger.experiment.summary["train/mse"] = mse_train
                self.logger.experiment.summary["train/r2"] = r2_train
                self.logger.experiment.summary["train/mae"] = mae_train

                self.logger.experiment.summary["valid/mse"] = mse_valid
                self.logger.experiment.summary["valid/r2"] = r2_valid
                self.logger.experiment.summary["valid/mae"] = mae_valid

                self.logger.experiment.summary["train/nlpd"] = nlpd_train
                self.logger.experiment.summary["train/msll"] = msll_train
                self.logger.experiment.summary["train/qce"] = qce_train

                self.logger.experiment.summary["valid/nlpd"] = nlpd_valid
                self.logger.experiment.summary["valid/msll"] = msll_valid
                self.logger.experiment.summary["valid/qce"] = qce_valid

                self.logger.experiment.log({"pred-vs-gt": wandb.Image(pred_vs_gt_fig)})
                self.logger.experiment.log({"residuals": wandb.Image(residuals_figure)})
                self.log_diversity_metrics(self.data.train_x)
                self.log_model_parameters()

    def on_train_start(self) -> None:
        self.data.log_data_stats(self.logger)

    def on_train_end(self) -> None:
        self.logger.experiment.finish()

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return self.data.train_dataloader()

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return self.data.val_dataloader()

    def configure_optimizers(self):
        pass

    def log_model_parameters(self):
        for name, param in self.model.named_hyperparameters():
            transformed_name = name.replace("raw_", "")
            attr = self.model
            for part in transformed_name.split("."):
                attr = getattr(attr, part)
            value = attr.cpu().detach().numpy()

            self.logger.experiment.log({transformed_name: value})


if __name__ == "__main__":
    gpregressor = GPRegression(data=data, model_config=model_config)
    logger = (
        WandbLogger(project="additives-rebuttal")
        if gpregressor.enable_plotting
        else None
    )
    trainer = Trainer(
        max_epochs=1,
        logger=logger,
        log_every_n_steps=1,
        num_sanity_val_steps=0,
        min_epochs=1,
        max_steps=-1,
        accelerator="cpu",
        devices=1,
    )
    trainer.fit(gpregressor)
    wandb.finish()
