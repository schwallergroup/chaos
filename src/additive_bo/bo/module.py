from typing import List

import gpytorch
import pytorch_lightning as pl
import torch
from botorch import fit_gpytorch_mll
from botorch.acquisition import (
    ExpectedImprovement,
    NoisyExpectedImprovement,
    UpperConfidenceBound,
)
from botorch.acquisition.monte_carlo import qUpperConfidenceBound
from botorch.sampling.normal import SobolQMCNormalSampler
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS

from additive_bo.data.utils import torch_delete_rows
from additive_bo.surrogate_models.gp import GP


class BoModule(pl.LightningModule):
    def __init__(
        self,
        data: pl.LightningDataModule,
        model: GP,
        acquisition_class: str = "ucb",
        beta: float = 0.1,
        top_n: List[int] = [1, 3, 5, 10],
        batch_size: int = 1,
        finetuning: bool = True,
    ):
        super().__init__()
        self.top_count = None
        self.data = data
        self.model = model
        self.acquisition = None
        self.top_n = top_n
        self.finetuning = finetuning
        self.acquisition_class = acquisition_class
        self.beta = beta
        self.batch_size = batch_size

        self.save_hyperparameters(
            ignore=["kernel", "model", "mll", "data", "acquisition_class", "beta"]
        )
        self.automatic_optimization = False

    def on_train_start(self) -> None:
        self.prepare_logging_and_counters()

    # BO iteration
    def training_step(self, batch, batch_idx):
        train_x, train_y = self.data.train_x, self.data.train_y
        self.log("train/best_so_far", torch.max(train_y))

        if self.acquisition_class != "random":
            self.model.train()
            self.train_surrogate(train_x, train_y)
            self.log_model_parameters()

        heldout_x, heldout_y = self.data.heldout_x, self.data.heldout_y
        new_x, new_y, suggestion_ids = self.optimize_acqf_and_get_observation(
            heldout_x, heldout_y
        )
        self.update_data(new_x, new_y, suggestion_ids)

    def train_surrogate(self, train_x, train_y):
        prev_state = self.model.state_dict()
        prev_state = {
            k: v for k, v in prev_state.items() if "outcome_transform" not in k
        }
        prev_state = prev_state if self.finetuning else None
        self.model.fit(train_x, train_y, state_dict=prev_state)

    def on_train_end(self) -> None:
        self.save_summary()

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

        for i, n in enumerate(self.top_n):
            for yi in new_y:
                if yi >= self.data.get_nth_largest_yield(n):
                    self.top_count[i] += 1

        self.log("train/suggestion", torch.max(new_y))

        # update heldout set points
        self.data.heldout_x = torch_delete_rows(self.data.heldout_x, best_idxs)
        self.data.heldout_y = torch_delete_rows(self.data.heldout_y, best_idxs)

        # update training set points
        self.data.train_x = torch.cat([self.data.train_x, new_x])
        self.data.train_y = torch.cat([self.data.train_y, new_y])

    def prepare_logging_and_counters(self):
        self.logger.experiment.summary["global_optimum"] = self.data.objective_optimum
        self.logger.experiment.config.update(
            {"surrogate_model.init_args.noise_val": self.model.noise_val},
            allow_val_change=True,
        )
        self.top_count = [self.count_top_n_in_init_data(n) for n in self.top_n]

    def save_summary(self):
        for i, top_n_count in enumerate(self.top_count):
            self.logger.experiment.summary[f"top_{self.top_n[i]}_count"] = top_n_count
        self.logger.experiment.finish()

    def log_model_parameters(self):
        for param_name, param in self.model.named_parameters():
            self.try_log(param_name, param)

        try:
            self.log(
                "lengthscale", self.model.covar_module.base_kernel.lengthscale.squeeze()
            )
        except:
            pass

    def try_log(self, name, value):
        try:
            self.log(name, value.item())
        except ValueError:
            pass

    def count_top_n_in_init_data(self, n):
        top_nth_yield = self.data.get_nth_largest_yield(n=n)
        mask = self.data.train_y >= top_nth_yield
        return sum(mask)
