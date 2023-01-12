from itertools import chain
from typing import Optional

import matplotlib.lines as mlines
import numpy as np
import pandas as pd
import plotly.express as px
import pytorch_lightning as pl
import torch
from additive_bo.data.dataset import DynamicSet, SingleSampleDataset
from additive_bo.data.utils import torch_delete_rows
from additive_bo.data_init_selection.clustering import BOInitDataSelection
from additive_bo.gprotorch.dataloader import DataLoaderMP, ReactionLoader
from matplotlib import pyplot as plt
from pytorch_lightning.utilities.types import (
    EVAL_DATALOADERS,
    TRAIN_DATALOADERS,
)
from scipy.stats import sem
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader


class BOAdditivesDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_path: str = "data/additives_reactions.csv",
        reaction_plate: int = 1,
        base_additive: str = "N#N",
        representation: str = "drfp",
        feature_dimension: int = 2048,
        # todo change init sample to float
        init_sample_size: int = 20,
        featurize_column: str = "reaction_smiles",
        exclude_n_largest: int = 0,
        scale_by_baseline: bool = False,
        init_selection_method: BOInitDataSelection = None,
        noise_calc: str = "se",
    ):
        super().__init__()
        self.noise_calc = noise_calc
        self.objective_optimum: int = None
        # todo do I need scaling here
        self.x: torch.tensor = None
        self.y: torch.tensor = None
        self.heldout_x: torch.tensor = None
        self.heldout_y: torch.tensor = None
        self.train_x: torch.tensor = None
        self.train_y: torch.tensor = None
        self.additives_reactions: pd.DataFrame = None
        self.base_reactions: pd.DataFrame = None

        self.data_path = data_path
        self.reaction_plate = reaction_plate
        self.base_additive = base_additive
        self.representation = representation
        self.feature_dimension = feature_dimension
        self.init_sample_size = init_sample_size
        self.featurize_column = featurize_column
        self.exclude_n_largest = exclude_n_largest
        self.scale_by_baseline = scale_by_baseline
        self.init_selection_method = init_selection_method

        self.save_hyperparameters()
        self.setup()

    def setup_data_by_reaction_plate(self):
        data = pd.read_csv(self.data_path)
        reaction_data = data[data["Plate"] == self.reaction_plate]
        self.base_reactions = reaction_data[reaction_data["Additive_Smiles"] == "N#N"]

        # todo do I need to scale by base reaction
        if self.scale_by_baseline:
            self.additives_reactions = (
                reaction_data[[self.featurize_column, "UV210_Prod AreaAbs"]]
                .groupby(self.featurize_column)
                .apply(
                    lambda x: x[["UV210_Prod AreaAbs"]].mean()
                    / self.base_reactions["UV210_Prod AreaAbs"].mean()
                )
                .reset_index()
            )
            self.base_reactions = (
                self.base_reactions["UV210_Prod AreaAbs"]
                / self.base_reactions["UV210_Prod AreaAbs"].mean()
            )

        else:
            self.additives_reactions = (
                reaction_data[[self.featurize_column, "UV210_Prod AreaAbs"]]
                .groupby(self.featurize_column)
                .apply(lambda x: x[["UV210_Prod AreaAbs"]].mean())
                .reset_index()
            )

    def calculate_noise_error(self):
        if self.noise_calc == "se":
            self.noise = sem(self.base_reactions.values)
        elif self.noise_calc == "var":
            self.noise = self.base_reactions.var()
        elif self.noise_calc == "std":
            self.noise = self.base_reactions.std()

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

    def featurize(self):
        """
        Featurize reaction or additive smiles using defined representation.

        """
        if self.featurize_column == "Additive_Smiles":
            loader = DataLoaderMP()
            loader.features = self.additives_reactions[self.featurize_column].to_list()
            loader.featurize(
                self.representation, bond_radius=3, nBits=self.feature_dimension
            )
        elif self.featurize_column == "reaction_smiles":
            loader = ReactionLoader()
            loader.features = self.additives_reactions[self.featurize_column]
            loader.featurize(self.representation, nBits=self.feature_dimension)
        loader.labels = (
            self.additives_reactions["UV210_Prod AreaAbs"].to_numpy().reshape(-1, 1)
        )

        x = loader.features
        y = loader.labels

        self.x = torch.from_numpy(x).to(torch.float64)  # .to('cuda')
        self.y = torch.from_numpy(y).to(torch.float64)  # .to('cuda')

    def get_baseline_reaction(self):
        return self.additives_reactions[
            self.additives_reactions[self.featurize_column].str.contains(
                self.base_additive
            )
        ].index[0]

    def setup(self, stage: Optional[str] = None) -> None:
        self.setup_data_by_reaction_plate()
        self.calculate_noise_error()
        self.featurize()
        self.remove_duplicates()

        baseline_reaction_index = self.get_baseline_reaction()
        high_yield_rxn_indexes = self.get_n_largest_yield_reactions(
            n=self.exclude_n_largest
        )
        init_indexes = self.init_selection_method.fit(
            self.x, exclude=[baseline_reaction_index] + high_yield_rxn_indexes
        )
        # exclude=high_yield_rxn_indexes)

        print(f"Selected reactions: {init_indexes}")
        # train_indexes = [baseline_reaction_index] + init_indexes
        self.train_indexes = init_indexes
        self.train_x = self.x[init_indexes]  # init_indexes
        self.train_y = self.y[init_indexes]

        self.heldout_x = torch_delete_rows(
            self.x, [baseline_reaction_index] + init_indexes
        )  # x[heldout_indices]
        self.heldout_y = torch_delete_rows(
            self.y, [baseline_reaction_index] + init_indexes
        )  # y[heldout_indices]

        # print(self.trainer, "SELF TRAINER JEBEM TI MATER")
        # self.train_x = self.train_x.to('cuda')
        # self.train_y = self.train_y.to('cuda')

        # self.heldout_x = self.heldout_x.to('cuda')
        # self.heldout_y = self.heldout_y.to('cuda')

        # shuffle_indices = torch.randperm(self.heldout_x.size()[0])
        #
        # self.heldout_x = self.heldout_x[shuffle_indices]
        # self.heldout_y = self.heldout_y[shuffle_indices]

        self.objective_optimum = torch.max(self.y)

    def plot_latent_space(self, method="pca"):

        # fig = plt.figure()
        # for x, y, col, m, s in zip(components[:, 0][sorted_indices], components[:, 1][sorted_indices],
        #                            np.array(colors)[sorted_indices], np.array(markers)[sorted_indices],
        #                            np.array(sizes)[sorted_indices]):
        #     # print(x, y, col, m)
        #     plt.scatter(x, y, color=col, marker=m, cmap='YlOrRd', s=s, edgecolors='black')
        # sm = plt.cm.ScalarMappable(cmap=plt.cm.get_cmap('YlOrRd'))
        # fig.colorbar(sm)
        # cmap = plt.cm.get_cmap('YlOrRd')
        # star = mlines.Line2D([], [], color=cmap(1.0), marker='*', linestyle='None',
        #                      markersize=10, label='Global optimum')
        # red_square = mlines.Line2D([], [], color=cmap(1.0), marker='s', linestyle='None',
        #                            markersize=10, label='Heldout data points')
        # purple_triangle = mlines.Line2D([], [], color=cmap(1.0), marker='o', linestyle='None',
        #                                 markersize=10, label='Initial data points')
        # plt.xlabel('Principal component 1')
        # plt.ylabel('Principal component 2')
        # plt.legend(handles=[star, red_square, purple_triangle])
        # fig.legend()

        if method == "tsne":
            feature_reduction = TSNE(
                n_components=2,
                learning_rate=300.0,
                metric="jaccard",
                init="random",
                random_state=0,
            )
        elif method == "pca":
            feature_reduction = PCA(n_components=2)

        x_embedded = feature_reduction.fit_transform(self.x)

        mask = [
            0
            if idx not in self.train_indexes and idx != torch.argmax(self.y)
            else 1
            if idx in self.train_indexes
            else 2
            for idx in range(self.x.shape[0])
        ]

        sorted_indices = np.array(mask).argsort()

        # labels = [str(idx) if idx in self.init_selection_method.selected_reactions else "" for idx in
        #           range(self.x.shape[0])]
        sizes = [
            200 if idx in self.train_indexes + [torch.argmax(self.y)] else 25
            for idx in range(self.x.shape[0])
        ]
        markers = [
            "*"
            if idx == torch.argmax(self.y)
            else "o"
            if idx in self.train_indexes
            else "s"
            for idx in range(self.x.shape[0])
        ]

        # fig = px.scatter(x_embedded, x=0, y=1, color=self.y.squeeze(), text=labels, size=sizes)

        norm = plt.Normalize()
        colors = plt.cm.OrRd(norm(self.y.squeeze()))  # plt.cm.OrRd(self.y.squeeze())

        fig = plt.figure()
        for x, y, col, m, s in zip(
            x_embedded[:, 0][sorted_indices],
            x_embedded[:, 1][sorted_indices],
            np.array(colors)[sorted_indices],
            np.array(markers)[sorted_indices],
            np.array(sizes)[sorted_indices],
        ):
            plt.scatter(
                x, y, color=col, marker=m, cmap="YlOrRd", s=s, edgecolors="black"
            )
        sm = plt.cm.ScalarMappable(cmap=plt.cm.get_cmap("YlOrRd"))
        fig.colorbar(sm)
        cmap = plt.cm.get_cmap("YlOrRd")
        star = mlines.Line2D(
            [],
            [],
            color=cmap(1.0),
            marker="*",
            linestyle="None",
            markersize=10,
            label="Global optimum",
        )
        red_square = mlines.Line2D(
            [],
            [],
            color=cmap(1.0),
            marker="s",
            linestyle="None",
            markersize=10,
            label="Heldout data points",
        )
        purple_triangle = mlines.Line2D(
            [],
            [],
            color=cmap(1.0),
            marker="o",
            linestyle="None",
            markersize=10,
            label="Initial data points",
        )
        plt.xlabel(f"{method}-1")
        plt.ylabel(f"{method}-2")
        plt.legend(handles=[star, red_square, purple_triangle])
        fig.legend()
        return fig

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


# class BOPhotoswitchDataModule(pl.LightningDataModule):
#     def __init__(
#         self,
#         data_path: str = "data/photoswitches.csv",
#         representation: str = "fingerprints",
#         feature_dimension: int = 2048,
#         init_sample: int = 20,
#         normalize: bool = False,
#         standardize: bool = False,
#     ):
#         super().__init__()
#         self.data_path = data_path
#         self.representation = representation
#         self.feature_dimension = feature_dimension
#         self.init_sample = init_sample
#         self.normalize = normalize
#         self.standardize = standardize
#         self.save_hyperparameters()
#         self.setup()
#
#     def _featurize(self):
#         loader = DataLoaderMP()
#         loader.load_benchmark("Photoswitch", self.data_path)
#         loader.featurize(
#             self.representation, bond_radius=3, nBits=self.feature_dimension
#         )
#         x = loader.features
#         y = loader.labels
#
#         # if self.normalize:
#         #     x = (x - np.min(x)) / (np.max(x) - np.min(x))
#         # if self.standardize:
#         #     y = (y - y.mean()) / (y.std())
#
#         x = torch.from_numpy(x).to(torch.float64)
#         y = torch.from_numpy(y).to(torch.float64)
#
#         return x, y
#
#     def setup(self, stage: Optional[str] = None) -> None:
#         x, y = self._featurize()
#         print(max(y), "max y")
#         print(torch.max(x), "max x")
#
#         init_indexes = torch.randperm(x.size()[0])[: self.init_sample]
#         all_indices = torch.arange(0, x.shape[0]).tolist()
#         heldout_indices = [idx for idx in all_indices if idx not in init_indexes]
#
#         self.train_x = x[init_indexes]
#         self.train_y = y[init_indexes]
#
#         self.heldout_x = x[heldout_indices]
#         self.heldout_y = y[heldout_indices]
#
#         self.objective_optimum = torch.max(self.heldout_y)
#
#     def train_dataloader(self) -> TRAIN_DATALOADERS:
#         train_dataset = SingleSampleDataset(self.train_x, self.train_y)
#         return DataLoader(train_dataset, num_workers=24)
#
#     def val_dataloader(self) -> EVAL_DATALOADERS:
#         valid_dataset = SingleSampleDataset(self.heldout_x, self.heldout_y)
#         return DataLoader(valid_dataset, num_workers=24)
