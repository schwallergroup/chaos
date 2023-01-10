import random

import numpy as np
from rdkit.ML.Cluster.Butina import ClusterData
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn_extra.cluster import KMedoids

from additive_bo.data.utils import torch_delete_rows


class BOInitDataSelection:
    def __init__(
        self,
        init_method: str = "k-medoids++",
        metric: str = "jaccard",
        n_clusters: int = 20,
        seed: int = None,
    ):
        self.selected_reactions = None
        self.init_method = init_method
        self.metric = metric
        self.n_clusters = n_clusters
        self.seed = seed

    def fit(self, x, exclude: list = None):
        x_init = torch_delete_rows(x, exclude)
        if self.init_method == "believer":
            distances = pairwise_distances(x_init, metric=self.metric)
            prev_selected_point = random.randint(0, distances.shape[0] - 1)
            print(prev_selected_point, "initial selected point")
            init_indices_from_clusters = [prev_selected_point]
            # distances = np.delete(distances, selected_point, axis=0)
            # distances = np.delete(distances, selected_point, axis=1)

            for i in range(self.n_clusters - 1):
                selected_point = np.argmax(distances[prev_selected_point, :])
                distances[:, selected_point] = -1
                prev_selected_point = selected_point
                init_indices_from_clusters.append(selected_point)
                # distances = np.delete(distances, selected_point, axis=0)
                # distances = np.delete(distances, selected_point, axis=1)

        elif self.init_method == "believer-v2":
            distances = pairwise_distances(x_init, metric=self.metric)
            selected_point = random.randint(0, distances.shape[0] - 1)
            init_indices_from_clusters = [selected_point]
            distances = np.delete(distances, selected_point, axis=0)
            distances = np.delete(distances, selected_point, axis=1)

            for i in range(self.n_clusters - 1):
                selected_point = np.argmax(distances[selected_point, :])
                init_indices_from_clusters.append(selected_point)
                distances = np.delete(distances, selected_point, axis=0)
                distances = np.delete(distances, selected_point, axis=1)

        elif self.init_method == "butina":
            distances = pairwise_distances(x_init, metric=self.metric)
            dists = []
            for i in range(distances.shape[0]):
                for j in range(i):
                    dists.append(distances[i, j])
            clusters = ClusterData(
                dists, x_init.shape[0], distThresh=0.05, isDistData=True
            )
            init_indices_from_clusters = [c[0] for c in clusters[: self.n_clusters]]

        elif self.init_method == "pca_kmedoids":
            x_pca = PCA(n_components=2).fit_transform(x_init)
            # find indices of x_init_set in x
            kmedoids = KMedoids(
                n_clusters=self.n_clusters,
                init="k-medoids++",
                random_state=self.seed,
                metric=self.metric,
                max_iter=5000,
            ).fit(x_pca)
            init_indices_from_clusters = kmedoids.medoid_indices_.tolist()

        else:
            # find indices of x_init_set in x
            kmedoids = KMedoids(
                n_clusters=self.n_clusters,
                init=self.init_method,
                random_state=self.seed,
                metric=self.metric,
                max_iter=5000,
            ).fit(x_init)
            init_indices_from_clusters = kmedoids.medoid_indices_.tolist()

        init_reactions_indexes = []
        for init_index in init_indices_from_clusters:
            row = x_init.numpy()[init_index, :]
            orig_index = np.where((x.numpy() == row).all(axis=1))[0]
            init_reactions_indexes.append(orig_index[0])

        self.selected_reactions = init_reactions_indexes
        return init_reactions_indexes


class BOInitDataSelectionButinaClustering:
    def __init__(
        self,
        init_method: str = "k-medoids++",
        metric: str = "jaccard",
        n_clusters: int = 20,
        seed: int = None,
    ):
        self.selected_reactions = None
        self.init_method = init_method
        self.metric = metric
        self.n_clusters = n_clusters
        self.seed = seed

    def fit(self, x, exclude: list = None):
        # find indices of x_init_set in x
        x_init = torch_delete_rows(x, exclude)
        kmedoids = KMedoids(
            n_clusters=self.n_clusters,
            init=self.init_method,
            random_state=self.seed,
            metric=self.metric,
        ).fit(x_init)
        init_indices_from_clusters = kmedoids.medoid_indices_.tolist()

        init_reactions_indexes = []
        for init_index in init_indices_from_clusters:
            row = x_init.numpy()[init_index, :]
            orig_index = np.where((x.numpy() == row).all(axis=1))[0]
            init_reactions_indexes.append(orig_index[0])

        self.selected_reactions = init_reactions_indexes
        return init_reactions_indexes
