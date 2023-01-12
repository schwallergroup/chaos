import copy
from typing import Any, List, Optional, Union

import botorch.acquisition
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import plotly
import plotly.express as px
import pytorch_lightning as pl
import torch
import wandb
from additive_bo.data.utils import torch_delete_rows
from additive_bo.gprotorch.kernels.fingerprint_kernels.tanimoto_kernel import (
    TanimotoKernel,
)
from additive_bo.surrogate_models.gp import (
    GP,
    CustomHeteroskedasticGP,
    FixedGP,
    HeteroskedasticGP,
)
from botorch import fit_gpytorch_model
from botorch.acquisition import (
    ExpectedImprovement,
    NoisyExpectedImprovement,
    UpperConfidenceBound,
)
from gpytorch import ExactMarginalLogLikelihood
from pytorch_lightning.utilities.types import (
    EVAL_DATALOADERS,
    TRAIN_DATALOADERS,
)
from sklearn.manifold import TSNE
from torchmetrics import MeanAbsoluteError, MeanSquaredError, R2Score


class BoModule(pl.LightningModule):
    def __init__(
        self,
        data: pl.LightningDataModule,
        model: Union[GP, FixedGP, HeteroskedasticGP, CustomHeteroskedasticGP],
        acquisition_class: str = "ucb",
        beta: float = 0.1,
        top_n: List[int] = None,
    ):
        super().__init__()
        print(model.noise_val, "NOISE VALUE")
        self.top_count = None
        self.data = data
        self.model = model
        self.acquisition = None
        self.top_n = top_n
        # print(
        #     "x_dimensions",
        #     self.trainer.datamodule.train_x.shape,
        #     self.trainer.datamodule.heldout_x.shape,
        # )

        self.acquisition_class = acquisition_class
        self.beta = beta

        self.save_hyperparameters(
            ignore=["kernel", "model", "mll", "data", "acquisition_class", "beta"]
        )
        self.automatic_optimization = False

    def initialize_mll(self, likelihood, model, state_dict=None):
        """
        Initialise model and loss function.

        Args:
            state_dict: current state dict used to speed up fitting

        """
        self.mll = ExactMarginalLogLikelihood(likelihood, model)
        # load state dict if it is passed
        if state_dict is not None:
            self.model.load_state_dict(state_dict)

    def count_top_n_in_init_data(self, n):
        top_nth_yield = self.data.get_nth_largest_yield(n=n)  # self.trainer.datamodule.
        mask = self.data.train_y >= top_nth_yield  # self.trainer.datamodule.
        return sum(mask)

    def on_train_start(self) -> None:
        self.logger.experiment.summary[
            "global_optimum"
        ] = self.data.objective_optimum  # self.trainer.datamodule.
        self.logger.experiment.config.update(
            {"surrogate_model.init_args.noise_val": self.model.noise_val},
            allow_val_change=True,
        )
        self.top_count = [self.count_top_n_in_init_data(n) for n in self.top_n]

    def visualize_latent_space(self, method="pca"):
        fig = self.data.plot_latent_space(method=method)  # self.trainer.datamodule.
        self.logger.log_image(key="latent-space", images=[wandb.Image(fig)])

    def plot_predicted_vs_true_observation_noise(self):
        # fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        with torch.no_grad():
            posterior_train = self.model.posterior(
                self.data.train_x, observation_noise=True  # self.trainer.datamodule.
            )
            posterior_test = self.model.posterior(
                self.data.heldout_x, observation_noise=True  # self.trainer.datamodule.
            )

            mean_train = posterior_train.mean
            mean_test = posterior_test.mean

            pred_error = (
                (self.data.heldout_y.squeeze() - mean_test.squeeze())
                ** 2  # self.trainer.datamodule.
            ).mean()
            print(f"Cross-validation error: {pred_error : 4.2}")

            # get lower and upper confidence bounds
            # lower, upper = posterior.mvn.confidence_region()

            # scatterplot of predicted versus test
            _, axes = plt.subplots(1, 1, figsize=(6, 4))
            # plt.plot([-1.5, 1.5], [-1.5, 1.5], 'k', label="true objective", linewidth=2)
            # plt.figure()
            axes.scatter(
                self.data.heldout_y.squeeze(),  # self.trainer.datamodule.
                mean_test.squeeze(),
                color="blue",
            )
            axes.scatter(
                self.data.train_y.squeeze(),
                mean_train.squeeze(),
                color="red",  # self.trainer.datamodule.
            )
            axes.ticklabel_format(useOffset=False)

            plt.xlabel("Actual")
            plt.ylabel("Predicted")

            self.logger.log_image(key="pred-vs-actual", images=[wandb.Image(axes)])

    def training_step(self, batch, batch_idx):
        # self.visualize_latent_space('tsne')
        # train_x, train_y = batch
        # train_x, train_y = train_x.squeeze(0), train_y.squeeze(0)

        train_x, train_y = (
            self.data.train_x,  # self.trainer.datamodule
            self.data.train_y,  # self.trainer.datamodule
        )
        self.log("train/best_so_far", torch.max(train_y))
        prev_state = self.model.state_dict()
        self.model = self.model.reinit(train_x=train_x, train_y=train_y)
        self.initialize_mll(likelihood=self.model.likelihood, model=self.model)
        self.model.load_state_dict(
            {k: v for k, v in prev_state.items() if "outcome_transform" not in k},
            strict=False,
        )

        fit_gpytorch_model(self.mll, max_retries=1)

        for param_name, param in self.model.named_parameters():
            try:
                self.log(param_name, param)
            except ValueError:
                pass

                # self.log(param_name, wandb.Histogram(param.detach()))
        self.log("model_output_scale", self.model.covar_module.outputscale)
        try:
            self.log("likelihood_noise", self.model.likelihood.noise)
        except:
            pass

        # mean_values = []
        # uncertainties = []

        # self._optimize_acqf_and_get_observation()

        # ovo gde
        # reinit model
        # prev_state = self.model.state_dict()
        # self.model = self.model.reinit(train_x=self.trainer.datamodule.train_x,
        #                                train_y=self.trainer.datamodule.train_y)

        # self.initialize_mll()
        # self.model.load_state_dict({k: v for k, v in prev_state.items() if "outcome_transform" not in k}, strict=False)

        # heldout_x, heldout_y = (
        #     self.trainer.datamodule.heldout_x,
        #     self.trainer.datamodule.heldout_y,
        # )

        self.plot_predicted_vs_true_observation_noise()

        # self.optimize_acqf_and_get_observation(heldout_x, heldout_y)

        # prev_state = self.model.state_dict()
        # self.model.load_state_dict(
        #     {k: v for k, v in prev_state.items() if "outcome_transform" not in k},
        #     strict=False,
        # )

        heldout_x, heldout_y = (
            self.data.heldout_x,  # self.trainer.datamodule.
            self.data.heldout_y,  # self.trainer.datamodule.
        )
        self.optimize_acqf_and_get_observation(heldout_x, heldout_y)

    # def validation_step(self, batch, *args, **kwargs):
    #     heldout_x, heldout_y = batch
    #     heldout_x, heldout_y = heldout_x.squeeze(0), heldout_y.squeeze(0)

    #     # heldout_x, heldout_y = (
    #     #     self.trainer.datamodule.heldout_x,
    #     #     self.trainer.datamodule.heldout_y,
    #     # )

    #     self.optimize_acqf_and_get_observation(heldout_x, heldout_y)
    #     # prev_state = self.model.state_dict()
    #     # self.model.load_state_dict(
    #     #     {k: v for k, v in prev_state.items() if "outcome_transform" not in k},
    #     #     strict=False,
    #     # )

    def on_train_end(self) -> None:
        for i, top_n_count in enumerate(self.top_count):
            self.logger.experiment.summary[f"top_{self.top_n[i]}_count"] = top_n_count
        self.logger.experiment.finish()

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return self.data.train_dataloader()  # self.trainer.datamodule

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return self.data.val_dataloader()  # self.trainer.datamodule

    # def on_validation_end(self) -> None:
    #     # reinit model
    #     self._initialize_mll(state_dict=self.model.state_dict())

    # def validation_step(self, batch, *args, **kwargs) -> Optional[STEP_OUTPUT]:
    #     with torch.no_grad():
    #         valid_x, valid_y = batch
    #         valid_x, valid_y = (
    #             valid_x.squeeze(),
    #             valid_y.squeeze(),
    #         )
    #         # self.model.eval()
    #         # self.mll.eval()
    #         # mean and variance GP prediction
    #         f_pred = self.model.posterior(valid_x, observation_noise=True)
    #
    #         y_pred = f_pred.mean
    #         # y_var = f_pred.variance
    #
    #         # Transform back to real data space to compute metrics and detach gradients. Must unsqueeze dimension
    #         # to make compatible with inverse_transform in scikit-learn version > 1
    #
    #         # y_pred = self.trainer.datamodule.y_scaler.inverse_transform(y_pred.detach().unsqueeze(dim=1))
    #         # valid_y = self.trainer.datamodule.y_scaler.inverse_transform(valid_y.detach().unsqueeze(dim=1))
    #
    #         # valid_y_stan = self.model.outcome_transform(valid_y)
    #         # Compute R^2, RMSE and MAE on Test set
    #
    #         score = r2_score(valid_y, y_pred)
    #         rmse = np.sqrt(mean_squared_error(valid_y, y_pred))
    #         mae = mean_absolute_error(valid_y, y_pred)
    #
    #         self.log("valid/r2_score", score)
    #         self.log("valid/nrmse", rmse/torch.std(valid_y))
    #         self.log("valid/nmae", mae/torch.std(valid_y))
    #
    #     # reinit model
    #     # self._initialize_mll(state_dict=self.model.state_dict())

    def configure_optimizers(self):
        # self.trainer.reset_train_dataloader()
        # print(self.trainer.datamodule)
        # print(self.train_dataloader.loaders)
        pass

    def construct_acquisition(self):
        if self.acquisition_class == "ei":
            self.acquisition = ExpectedImprovement(
                model=self.model,
                best_f=self.data.train_y.max(),  # self.trainer.datamodule
            )

        elif self.acquisition_class == "nei":
            self.acquisition = NoisyExpectedImprovement(
                model=self.model,
                X_observed=self.data.train_x,  # self.trainer.datamodule
                num_fantasies=100,
            )

        elif self.acquisition_class == "ucb":
            self.acquisition = UpperConfidenceBound(
                model=self.model,
                beta=self.beta,
            )

    def optimize_acquisition(self, heldout_x, heldout_y):
        acq_vals = []
        # Loop over the discrete set of points to evaluate the acquisition function at.
        for i in range(len(heldout_y)):
            acq_vals.append(
                self.acquisition(heldout_x[i].unsqueeze(-2))
            )  # use unsqueeze to append batch dimension

        # observe new values
        acq_vals = torch.tensor(acq_vals)
        best_idx = torch.argmax(acq_vals)
        self.log("sum_acq_values", acq_vals.sum())
        self.log("suggestion_idx", best_idx)

        return best_idx

    # todo
    # def update_train_heldout_data()

    def optimize_acqf_and_get_observation(self, heldout_x, heldout_y):

        if self.acquisition_class == "random":
            best_idx = torch.randperm(len(heldout_y))[0]
        else:
            self.construct_acquisition()
            best_idx = self.optimize_acquisition(heldout_x, heldout_y)

        new_x = heldout_x[best_idx].unsqueeze(-2)  # add batch dimension
        new_y = heldout_y[best_idx].unsqueeze(-1)  # add output dimension

        self.data.train_indexes.append(best_idx)  # self.trainer.datamodule

        for i, n in enumerate(self.top_n):
            if new_y >= self.data.get_nth_largest_yield(n):  # self.trainer.datamodule
                self.top_count[i] += 1
        self.log("train/suggestion", new_y)

        # update heldout set points
        # delete the selected input and value from the heldout set.
        self.data.heldout_x = torch_delete_rows(self.data.heldout_x, [best_idx])
        self.data.heldout_y = torch_delete_rows(self.data.heldout_y, [best_idx])

        # update training points
        # self.trainer.datamodule.train_x = torch.cat([self.trainer.datamodule.train_x, new_x.to('cpu')])
        # self.trainer.datamodule.train_y = torch.cat([self.trainer.datamodule.train_y, new_y.to('cpu')])

        self.data.train_x = torch.cat(  # self.trainer.datamodule
            [self.data.train_x, new_x]
        )
        self.data.train_y = torch.cat(  # self.trainer.datamodule
            [self.data.train_y, new_y]
        )
