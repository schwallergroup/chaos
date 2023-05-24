from itertools import chain
from typing import Optional

import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader

from chaos.data.dataset import SingleSampleDataset
from chaos.data.utils import torch_delete_rows
from chaos.gprotorch.dataloader import DataLoaderMP, ReactionLoader
from chaos.initialization.initializers import BOInitializer


class BOAdditivesDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_path: str = "data/additives_reactions.csv",
        reaction_plate: int = 1,
        init_sample_size: int = 20,
        featurize_column: str = "reaction_smiles",
        exclude_n_largest: int = 0,
        scale_by_baseline: bool = False,
        init_selection_method: BOInitializer = None,
        representation: str = "drfp",
        feature_dimension: int = 2048,
        bond_radius: int = 3,
    ):
        super().__init__()
        self.objective_optimum: int = None
        self.additives_reactions: pd.DataFrame = None
        self.base_reactions: pd.DataFrame = None
        self.data_path = data_path
        self.reaction_plate = reaction_plate
        self.representation = representation
        self.bond_radius = bond_radius
        self.feature_dimension = feature_dimension
        self.init_sample_size = init_sample_size
        self.featurize_column = featurize_column
        self.exclude_n_largest = exclude_n_largest
        self.scale_by_baseline = scale_by_baseline
        self.init_selection_method = init_selection_method

        self.save_hyperparameters()
        self.setup()

    def setup(self, stage: Optional[str] = None) -> None:
        self.setup_data_by_reaction_plate()
        self.x, self.y = self.featurize(self.representation)

        self.remove_duplicates()  # will edit additives dataframe
        print(self.x.shape, "X shape")
        self.remove_nan_rows()  # will edit additives dataframe

        baseline_reaction_index = self.get_baseline_reaction()
        high_yield_rxn_indexes = self.get_n_largest_yield_reactions(
            n=self.exclude_n_largest
        )

        init_indexes = self.init_selection_method.fit(
            self.x, exclude=baseline_reaction_index + high_yield_rxn_indexes
        )

        print(f"Selected reactions: {init_indexes}")
        self.train_indexes = init_indexes
        self.train_x = self.x[init_indexes]
        self.train_y = self.y[init_indexes]

        self.heldout_x = torch_delete_rows(
            self.x, baseline_reaction_index + init_indexes
        )
        self.heldout_y = torch_delete_rows(
            self.y, baseline_reaction_index + init_indexes
        )

        self.objective_optimum = torch.max(self.y)

    def set_features(self, loader, column):
        loader.features = (
            self.additives_reactions[column].to_list()
            if isinstance(loader, DataLoaderMP)
            else self.additives_reactions[column]
        )

    def set_labels(self, loader):
        loader.labels = self.additives_reactions["UV210_Prod AreaAbs"].to_numpy()

    def get_featurized_data(self, loader, representation, bond_radius=None, nBits=None):
        loader.featurize(representation, bond_radius=bond_radius, nBits=nBits)
        return loader.features, loader.labels

    def featurize(self, representation):
        """
        Featurize reaction or additive smiles using defined representation.

        """
        x, y = None, None
        if representation == "ohe":
            reaction_loader = ReactionLoader()
            self.set_features(reaction_loader, "Additive_Smiles")
            self.get_featurized_data(reaction_loader, "ohe")
            self.set_labels(reaction_loader)

            x = reaction_loader.features
            y = reaction_loader.labels
            print(x.shape, "x shape")

        elif self.featurize_column in ["Additive_Smiles", "reaction_smiles"]:
            loader = (
                DataLoaderMP()
                if self.featurize_column == "Additive_Smiles"
                else ReactionLoader()
            )
            self.set_features(loader, self.featurize_column)
            self.get_featurized_data(
                loader,
                representation,
                bond_radius=self.bond_radius,
                nBits=self.feature_dimension,
            )
            self.set_labels(loader)

            x = loader.features
            y = loader.labels

        y = y.reshape(-1, 1)

        return torch.from_numpy(x).to(torch.float64), torch.from_numpy(y).to(
            torch.float64
        )

    def get_baseline_reaction(self):
        return self.additives_reactions[
            self.additives_reactions[self.featurize_column].str.contains("N#N")
        ].index.tolist()

    def train_test_split(self, init_indexes, baseline_reaction_index):
        self.train_indexes = init_indexes
        self.train_x = self.x[init_indexes]  # init_indexes
        self.train_y = self.y[init_indexes]

        self.heldout_x = torch_delete_rows(
            self.x, baseline_reaction_index + init_indexes
        )
        self.heldout_y = torch_delete_rows(
            self.y, baseline_reaction_index + init_indexes
        )

    def setup_data_by_reaction_plate(self):
        data = pd.read_csv(self.data_path)
        reaction_data = data[data["Plate"] == self.reaction_plate]
        self.base_reactions = reaction_data[reaction_data["Additive_Smiles"] == "N#N"]

        additive_cols = ["reaction_smiles", "Additive_Smiles", "UV210_Prod AreaAbs"]
        grouped_data = reaction_data[additive_cols].groupby(
            ["reaction_smiles", "Additive_Smiles"]
        )

        if self.scale_by_baseline:
            self.additives_reactions = grouped_data.apply(
                lambda x: x[["UV210_Prod AreaAbs"]].mean()
                / self.base_reactions["UV210_Prod AreaAbs"].mean()
            ).reset_index()
            self.base_reactions = (
                self.base_reactions["UV210_Prod AreaAbs"]
                / self.base_reactions["UV210_Prod AreaAbs"].mean()
            )
        else:
            self.additives_reactions = grouped_data.apply(
                lambda x: x[["UV210_Prod AreaAbs"]].mean()
            ).reset_index()
            self.base_reactions = self.base_reactions["UV210_Prod AreaAbs"]

    def remove_nan_rows(self):
        mask = torch.isnan(self.x).any(dim=1)
        indices_to_delete = mask.nonzero().flatten().tolist()
        self.x = torch_delete_rows(self.x, indices_to_delete)
        self.y = torch_delete_rows(self.y, indices_to_delete)
        self.additives_reactions = self.additives_reactions.drop(
            index=indices_to_delete
        ).reset_index(drop=True)

    def remove_duplicates(self):
        __, inv, counts = torch.unique(
            self.x, return_inverse=True, return_counts=True, dim=0
        )
        duplicates = tuple(
            [
                torch.where(inv == i)[0].tolist()
                for i, c in enumerate(counts)
                if counts[i] > 1
            ]
        )
        if len(duplicates) > 0:
            indices_to_delete = list(chain(*[x[1:] for x in duplicates]))
            self.x = torch_delete_rows(self.x, indices_to_delete)
            self.y = torch_delete_rows(self.y, indices_to_delete)
            self.additives_reactions = self.additives_reactions.drop(
                index=indices_to_delete
            ).reset_index(drop=True)

    def get_n_largest_yield_reactions(self, n=10):
        high_yield_mol_indexes = (
            self.additives_reactions["UV210_Prod AreaAbs"].nlargest(n=n).index.tolist()
        )
        return high_yield_mol_indexes

    def get_nth_largest_yield(self, n=10):
        return self.additives_reactions["UV210_Prod AreaAbs"].nlargest(n=n).iloc[-1]

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        train_dataset = SingleSampleDataset(self.train_x, self.train_y)
        return DataLoader(train_dataset, num_workers=4)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        valid_dataset = SingleSampleDataset(self.heldout_x, self.heldout_y)
        return DataLoader(valid_dataset, num_workers=4)

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        batch = [item.to(device) for item in batch]
        return batch
