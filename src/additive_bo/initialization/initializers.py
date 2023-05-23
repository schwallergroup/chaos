import random
from abc import ABC, abstractmethod

import numpy as np
import torch
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn_extra.cluster import KMedoids


def get_original_index(row, x):
    orig_index = np.where((x.numpy() == row).all(axis=1))[0]
    return orig_index[0]


def torch_delete_rows(input_tensor, indices):
    if indices is None:
        return input_tensor
    mask = torch.ones(input_tensor.size(0), dtype=torch.bool)
    mask[indices] = False
    return input_tensor[mask]


class Initializer(ABC):
    @abstractmethod
    def fit(self, x, exclude=None):
        pass


class MaxMinInitializer(Initializer):
    def __init__(self, metric="jaccard", n_clusters=20, **kwargs):
        self.metric = metric
        self.n_clusters = n_clusters

    def fit(self, x, exclude=None):
        x_init = torch_delete_rows(x, exclude)
        distances = cdist(x_init, x_init, metric=self.metric)
        selected_indices = [random.randint(0, distances.shape[0] - 1)]
        for _ in range(self.n_clusters - 1):
            coverage = np.min(distances[:, selected_indices], axis=1)
            selected_index = np.argmax(coverage)
            selected_indices.append(selected_index)
        return selected_indices


class RandomInitializer(Initializer):
    def __init__(self, n_clusters=20, **kwargs):
        self.n_clusters = n_clusters

    def fit(self, x, exclude=None):
        x_init = torch_delete_rows(x, exclude)
        return torch.randperm(len(x_init))[: self.n_clusters].tolist()


class KMeansInitializer(Initializer):
    def __init__(self, n_clusters=20, seed=None, **kwargs):
        self.n_clusters = n_clusters
        self.seed = seed

    def fit(self, x, exclude=None):
        x_init = torch_delete_rows(x, exclude)
        x_init_normalized = StandardScaler().fit_transform(x_init)
        kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.seed,
            max_iter=5000,
        ).fit(x_init_normalized)
        centroids = kmeans.cluster_centers_
        return [
            np.argmin(np.linalg.norm(x_init - centroid, axis=1))
            for centroid in centroids
        ]


class KMedoidsInitializer(Initializer):
    def __init__(
        self,
        n_clusters=20,
        seed=None,
        metric="jaccard",
        init_method="k-medoids++",
        **kwargs,
    ):
        self.n_clusters = n_clusters
        self.seed = seed
        self.metric = metric
        self.init_method = init_method

    def fit(self, x, exclude=None):
        x_init = torch_delete_rows(x, exclude)
        kmedoids = KMedoids(
            n_clusters=self.n_clusters,
            init=self.init_method,
            random_state=self.seed,
            metric=self.metric,
            max_iter=5000,
        ).fit(x_init)
        return kmedoids.medoid_indices_.tolist()


class BOInitializer:
    def __init__(
        self,
        init_method: str = "true_random",
        metric: str = None,
        n_clusters: int = None,
    ):
        self.n_clusters = n_clusters
        self.init_methods = {
            "maxmin": MaxMinInitializer,
            "true_random": RandomInitializer,
            "kmeans": KMeansInitializer,
            "kmedoids": KMedoidsInitializer,
        }
        if init_method not in self.init_methods:
            raise ValueError(f"Unknown init_method: {init_method}")
        print(init_method)
        self.initializer = self.init_methods[init_method](
            metric=metric, n_clusters=n_clusters
        )
        self.selected_reactions = None

    def fit(self, x, exclude: list = None):
        x_init = torch_delete_rows(x, exclude)
        self.selected_reactions = [
            get_original_index(x_init.numpy()[i], x)
            for i in self.initializer.fit(x, exclude)
        ]
        return self.selected_reactions
