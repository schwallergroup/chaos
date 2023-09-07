from itertools import chain
from typing import Optional, Union, List

import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader
from chaos.gprotorch.dataloader.mol_prop import DataLoaderMP
from chaos.gprotorch.dataloader.reaction_loader import ReactionLoader

from chaos.data.dataset import SingleSampleDataset
from chaos.data.utils import torch_delete_rows
from chaos.gprotorch.dataloader import DataLoaderMP, ReactionLoader
from chaos.initialization.initializers import BOInitializer
from chaos.data.utils import find_duplicates, find_nan_rows
from abc import ABC


class Featurizer:
    def __init__(
        self,
        task: str = "molecular_optimization",
        representation: str = "fingerprints",
        bond_radius: int = 3,
        nBits: int = 2048,
        **kwargs,
    ):
        self.representation = representation
        self.bond_radius = bond_radius
        self.nBits = nBits
        if task == "molecular_optimization":
            self.loader = DataLoaderMP()
        elif task == "reaction_optimization":
            self.loader = ReactionLoader()
        else:
            raise ValueError("Invalid task specified")

    def featurize(self, data):
        self.loader.features = (
            data.to_list() if isinstance(self.loader, DataLoaderMP) else data
        )
        self.loader.featurize(
            representation=self.representation,
            bond_radius=self.bond_radius,
            nBits=self.nBits,
        )
        return self.loader.features


class BaseDataModule(pl.LightningDataModule, ABC):
    def __init__(
        self,
        data_path: str,
        input_column: Union[str, List[str]] = "input",
        target_column: str = "target",
        init_sample_size: int = 20,
        featurizer: Featurizer = Featurizer(),
        initializer: BOInitializer = None,
    ) -> None:
        self.data_path = data_path
        self.target_column = target_column
        self.input_column = input_column
        self.init_sample_size = init_sample_size
        self.featurizer = featurizer
        self.initializer = (
            initializer
            if initializer is not None
            else BOInitializer(method="true_random", n_clusters=init_sample_size)
        )

        self.setup()

    def load_data(self):
        self.data = pd.read_csv(self.data_path)

    def featurize_data(self):
        x = self.featurizer.featurize(self.data[self.input_column])
        y = self.data[self.target_column].values

        self.x = torch.from_numpy(x).to(torch.float64)
        self.y = torch.from_numpy(y).to(torch.float64).unsqueeze(-1)

    def preprocess_data(self):
        nan_rows = find_nan_rows(self.x)
        self.x = torch_delete_rows(self.x, nan_rows)
        self.y = torch_delete_rows(self.y, nan_rows)

        duplicates = find_duplicates(self.x)
        self.x = torch_delete_rows(self.x, duplicates)
        self.y = torch_delete_rows(self.y, duplicates)

    def split_data(self):
        init_indexes, _ = self.initializer.fit(self.x, exclude=None)

        print(f"Selected reactions: {init_indexes}")
        self.train_indexes = init_indexes
        self.train_x = self.x[init_indexes]
        self.heldout_x = torch_delete_rows(self.x, init_indexes)

        self.train_y = self.y[init_indexes]
        self.heldout_y = torch_delete_rows(self.y, init_indexes)

    def calculate_data_metrics(self):
        target_stat_max = torch.max(self.y)
        target_stat_mean = torch.mean(self.y)
        target_stat_std = torch.std(self.y)
        target_stat_var = torch.var(self.y)
        input_stat_feature_dimension = self.x.shape[-1]
        input_stat_n_points = self.x.shape[0]

        target_q75 = torch.quantile(self.y, 0.75)
        target_q90 = torch.quantile(self.y, 0.9)
        target_q95 = torch.quantile(self.y, 0.95)
        target_q99 = torch.quantile(self.y, 0.99)

        top_3_values, _ = torch.topk(self.y, 3, dim=0)
        top_5_values, _ = torch.topk(self.y, 5, dim=0)
        top_10_values, _ = torch.topk(self.y, 10, dim=0)

        top_1 = torch.max(self.y)
        top_3 = top_3_values[-1]
        top_5 = top_5_values[-1]
        top_10 = top_10_values[-1]

        self.data_metrics = {
            "target_stat_max": target_stat_max,
            "target_stat_mean": target_stat_mean,
            "target_stat_std": target_stat_std,
            "target_stat_var": target_stat_var,
            "input_stat_feature_dimension": input_stat_feature_dimension,
            "input_stat_n_points": input_stat_n_points,
            "target_q75": target_q75,
            "target_q90": target_q90,
            "target_q95": target_q95,
            "target_q99": target_q99,
            "top_1": top_1,
            "top_3": top_3,
            "top_5": top_5,
            "top_10": top_10,
        }

    def log_data_metrics(self, logger):
        self.log_data_stats(self.data_metrics, logger)
        self.log_top_n_counts(self.data_metrics, logger)
        self.log_quantile_counts(self.data_metrics, logger)

    def log_data_stats(self, logger):
        for key, value in self.data_metrics.items():
            if "stat" in key:
                logger.experiment.summary[key] = (
                    value.item() if torch.is_tensor(value) else value
                )

    def log_top_n_counts(self, logger):
        for n in [1, 3, 5, 10]:
            threshold = self.data_metrics[f"top_{n}"]
            count = (self.train_y >= threshold).sum().item()
            logger.experiment.summary[f"top_{n}_count"] = count

    def log_quantile_counts(self, logger):
        for q in [0.75, 0.9, 0.95, 0.99]:
            threshold = self.data_metrics[f"target_q{int(q * 100)}"]
            count = (self.train_y >= threshold).sum().item()
            logger.experiment.summary[f"quantile_{int(q * 100)}_count"] = count

    def update_results(self, experiment_results, experiment_indexes):
        # Ensure target column exists
        if self.target_column not in self.data.columns:
            self.data[self.target_column] = 0.0

        # Update the target column with the results
        self.data.loc[experiment_indexes, self.target_column] = experiment_results
        self.y = torch.tensor(self.data[self.target_column].values).to(torch.float64)
        self.train_y = self.y[self.train_indexes]

    def setup(self, stage: Optional[str] = None) -> None:
        self.load_data()
        self.featurize_data()
        self.preprocess_data()
        self.split_data()
        self.calculate_data_metrics()

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        train_dataset = SingleSampleDataset(self.train_x, self.train_y)
        return DataLoader(train_dataset, num_workers=4)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        valid_dataset = SingleSampleDataset(self.heldout_x, self.heldout_y)
        return DataLoader(valid_dataset, num_workers=4)

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        batch = [item.to(device) for item in batch]
        return batch
