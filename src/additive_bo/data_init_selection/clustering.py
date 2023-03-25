import random

import numpy as np
import torch
from additive_bo.data.utils import torch_delete_rows
from pyDOE import lhs
from rdkit.ML.Cluster.Butina import ClusterData
from scipy.spatial.distance import cdist

# from scipy.optimize import latin_hypercube
from scipy.stats import qmc, uniform
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn_extra.cluster import KMedoids
from sobol_seq import i4_sobol_generate


def get_original_index(row, x):
    orig_index = np.where((x.numpy() == row).all(axis=1))[0]
    return orig_index[0]


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
        labels = []
        x_init = torch_delete_rows(x, exclude)
        clusters = {}
        if self.init_method == "believer":
            distances = pairwise_distances(x_init, metric=self.metric)
            prev_selected_point = random.randint(0, distances.shape[0] - 1)
            print(prev_selected_point, "initial selected point")
            init_indices_from_clusters = [prev_selected_point]

            ix = np.argmax(distances[prev_selected_point, :])
            print(
                ix,
                "<- the real next selected point",
                "the distance to that point",
                distances[prev_selected_point, ix],
            )

            distances = np.delete(distances, prev_selected_point, axis=0)
            distances = np.delete(distances, prev_selected_point, axis=1)

            ix = np.argmax(distances[prev_selected_point, :])
            print(
                ix,
                "<- the fake next selected point",
                "the distance to the fake point",
                distances[prev_selected_point, ix],
            )

            for i in range(self.n_clusters - 1):
                selected_point = np.argmax(distances[prev_selected_point, :])
                # distances[:, selected_point] = -1
                prev_selected_point = selected_point
                init_indices_from_clusters.append(selected_point)

                ix = np.argmax(distances[prev_selected_point, :])
                print(
                    ix,
                    "<- the real next selected point",
                    "the distance to that point",
                    distances[prev_selected_point, ix],
                )

                distances = np.delete(distances, selected_point, axis=0)
                distances = np.delete(distances, selected_point, axis=1)

                ix = np.argmax(distances[prev_selected_point, :])
                print(
                    ix,
                    "<- the fake next selected point",
                    "the distance to the fake point",
                    distances[prev_selected_point, ix],
                )

        elif self.init_method == "not-believer":
            distances = pairwise_distances(x_init, metric=self.metric)
            prev_selected_point = random.randint(0, distances.shape[0] - 1)
            print(prev_selected_point, "initial selected point")
            init_indices_from_clusters = [prev_selected_point]
            closest_indices = np.argsort(distances[prev_selected_point, :])[
                1 : self.n_clusters
            ]
            init_indices_from_clusters.extend(closest_indices)

        elif self.init_method == "random-from-clusters":
            # use KMeans clustering to identify clusters
            kmedoids = KMedoids(
                n_clusters=self.n_clusters,
                init="random",
                random_state=self.seed,
                metric=self.metric,
                max_iter=5000,
            )
            kmedoids.fit(x_init)
            labels = kmedoids.labels_
            n_samples = 1
            # randomly select n_samples from each cluster
            selected_indices = []
            for cluster_id in range(self.n_clusters):
                cluster_indices = np.where(labels == cluster_id)[0]
                if len(cluster_indices) <= n_samples:
                    selected_indices.extend(cluster_indices)
                else:
                    selected_indices.extend(
                        np.random.choice(cluster_indices, size=n_samples, replace=False)
                    )

            init_indices_from_clusters = selected_indices

        elif self.init_method == "cluster-believer":
            kmedoids = KMedoids(
                n_clusters=3,
                init="random",
                random_state=self.seed,
                metric=self.metric,
                max_iter=5000,
            )
            kmedoids.fit(x_init)
            cluster_centers = kmedoids.cluster_centers_

            # Select a few points from each cluster
            num_points = 10
            init_indices_from_clusters = [random.randint(0, x_init.shape[0] - 1)]
            for center in cluster_centers:
                distances = cdist(x_init, center.reshape(1, -1), metric=self.metric)
                closest_indices = np.argsort(distances.flatten())[
                    : num_points // len(cluster_centers)
                ]
                init_indices_from_clusters.extend(closest_indices)

        elif self.init_method == "cluster-believer-3":
            kmedoids = KMedoids(
                n_clusters=self.n_clusters - 1,
                init="random",
                random_state=self.seed,
                metric=self.metric,
                max_iter=5000,
            )
            kmedoids.fit(x_init)
            labels = kmedoids.labels_

            selected_indices = [random.randint(0, x_init.shape[0] - 1)]
            distances = cdist(x_init, x_init, metric=self.metric)

            # Select one point from each cluster
            for i in range(self.n_clusters - 1):
                # Get the indices of the data points in the i-th cluster
                indices = np.where(labels == i)[0]

                # Select the point in the cluster that is farthest from the previously selected points

                coverage = np.min(distances[:, selected_indices], axis=1)
                coverage[~indices] = -1
                selected_index = np.argmax(coverage)

                # Add the selected point to the list of selected points
                selected_indices.append(selected_index)

            init_indices_from_clusters = selected_indices

        elif self.init_method == "believer-3":
            # Compute pairwise distances between all pairs of points in the dataset
            distances = cdist(x_init, x_init, metric=self.metric)

            # Initialize a list of selected points with one randomly chosen point
            selected_indices = [random.randint(0, distances.shape[0] - 1)]

            # Select the remaining points using a greedy algorithm
            for i in range(self.n_clusters - 1):
                # Compute the coverage of the remaining dataset for each point
                coverage = np.min(distances[:, selected_indices], axis=1)

                # Select the point with the maximum coverage
                selected_index = np.argmax(coverage)

                # Add the selected point to the list of selected points
                selected_indices.append(selected_index)

            # Return the selected points as a NumPy array
            init_indices_from_clusters = selected_indices

        elif self.init_method == "cluster-believer-2":
            prev_selected_point = random.randint(0, x_init.shape[0] - 1)

            kmedoids = KMedoids(
                n_clusters=self.n_clusters - 1,
                init="random",
                random_state=self.seed,
                metric=self.metric,
                max_iter=5000,
            )

            kmedoids.fit(x_init)
            cluster_centers = kmedoids.cluster_centers_
            labels = kmedoids.labels_

            for i in range(self.n_clusters - 1):
                # Get the indices of the data points in the i-th cluster
                indices = np.where(labels == i)[0]

                # Select the point in the cluster that is farthest from the previously selected points
                max_distances = np.max(distances[indices], axis=1)
                selected_point_index = indices[np.argmax(max_distances)]

                # Add the selected point to the list of selected points
                selected_points.append(selected_point_index)

        elif self.init_method == "max-diversity-sampling-2":
            init_indices_from_clusters = [random.randint(0, x_init.shape[0] - 1)]

            for i in range(self.n_clusters - 1):
                distances = cdist(
                    x_init, x_init[init_indices_from_clusters], metric=self.metric
                )
                # min_distances = np.min(distances, axis=1)
                min_distances = np.sum(distances, axis=1)
                new_index = np.argmax(min_distances)
                while new_index in init_indices_from_clusters:
                    min_distances[new_index] = -np.inf
                    new_index = np.argmax(min_distances)
                init_indices_from_clusters.append(new_index)

        elif self.init_method == "sobol":
            bounds = [
                (torch.min(x_init[:, i]), torch.max(x_init[:, i]))
                for i in range(x_init.shape[-1])
            ]
            sobol_seq = i4_sobol_generate(x_init.shape[1], self.n_clusters)
            # Map the sample to the input space of the chemical reactions data matrix
            input_points = uniform(*np.transpose(bounds)).ppf(sobol_seq)
            # output_points = np.apply_along_axis(lambda x: x_init[np.argmin(cdist(x_init, np.array([x]), metric=self.metric))], 1, input_points)
            init_indices_from_clusters = np.apply_along_axis(
                lambda x: np.argmin(cdist(x_init, np.array([x]), metric=self.metric)),
                1,
                input_points,
            )
            init_indices_from_clusters = init_indices_from_clusters.tolist()

        elif self.init_method == "latin-hypercube":
            bounds = [
                (torch.min(x_init[:, i]), torch.max(x_init[:, i]))
                for i in range(x_init.shape[-1])
            ]
            # lower_bounds = [b[0].item() for b in bounds]
            # upper_bounds = [b[1].item()+1e-7 for b in bounds]

            # strength=1, scramble=True, optimization='random-cd',
            # lhs = qmc.LatinHypercube(d=len(bounds), seed=self.seed)

            input_points = lhs(
                len(bounds), samples=self.n_clusters, criterion="maximin"
            )
            # scale to bounds
            input_points = (
                input_points * (np.array(bounds)[:, 1] - np.array(bounds)[:, 0])
                + np.array(bounds)[:, 0]
            )

            # if self.metric == 'jaccard':
            #     input_points = np.where(input_points <= 0.5, 0, 1)

            # sample = lhs.random(self.n_clusters)

            # input_points = qmc.scale(sample, lower_bounds, upper_bounds)
            # print(np.array(input_points[0,:]).reshape(-1,1).shape)
            # print(np.argmin(cdist(x_init, np.array(input_points[0,:]).reshape(1, -1), metric='euclidean')), 'sfsdf')
            init_indices_from_clusters = np.apply_along_axis(
                lambda x: np.argmin(cdist(x_init, np.array([x]), metric="euclidean")),
                1,
                input_points,
            )
            init_indices_from_clusters = init_indices_from_clusters.tolist()

        elif self.init_method == "pca-importance":
            # Perform PCA on the data matrix to find the most important dimensions
            pca = PCA(5)
            pca.fit(x_init)

            # Extract the first two principal components
            pcs = [pca.components_[i] for i in range(len(pca.components_))]

            # Calculate the scores of each data point on the first two principal components
            pc_scores = np.dot(x_init, np.transpose(pcs))

            # Select the points with the highest scores on the first two principal components
            init_indices_from_clusters = np.argsort(np.sum(pc_scores**2, axis=1))[
                -self.n_clusters :
            ]
            init_indices_from_clusters = init_indices_from_clusters.tolist()

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
            x_pca = PCA(n_components=50).fit_transform(x_init)
            # find indices of x_init_set in x
            kmedoids = KMedoids(
                n_clusters=self.n_clusters,
                init="random",
                random_state=self.seed,
                metric=self.metric,
                max_iter=5000,
            ).fit(x_pca)
            init_indices_from_clusters = kmedoids.medoid_indices_.tolist()

        elif self.init_method == "true_random":
            init_indices_from_clusters = torch.randperm(len(x_init))[: self.n_clusters]

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

            labels = kmedoids.labels_
            cluster_centers_indices = kmedoids.medoid_indices_
            cluster_centers = kmedoids.cluster_centers_

            # print(cluster_centers, 'cluster centers')

            # Create a dictionary to store the cluster centers and points
            clusters = {}
            print(kmedoids.inertia_, "inertia")
            for i, (label, center) in enumerate(
                zip(range(len(cluster_centers_indices)), cluster_centers)
            ):
                # if label not in clusters:
                #     clusters[label] = []
                center_index = cluster_centers_indices[label]

                # compute the distance of each point in the cluster to the center
                # distances = np.linalg.norm(x_init - x_init[center_index, :], axis=1)
                distances = cdist(x_init, center.reshape(1, -1), metric=self.metric)
                # todo which distance metric should be used for which representation

                # get the indices of the points in the current cluster
                cluster_indices = np.where(labels == label)[0]
                # sort the indices by the distances of the corresponding points to the center
                sorted_indices = sorted(cluster_indices, key=lambda i: distances[i])
                sorted_original_indices = [
                    get_original_index(x_init.numpy()[i], x) for i in sorted_indices
                ]
                # add the sorted indices to the dictionary
                # sorted_cluster_points[label] = sorted_indices
                # clusters[label].append(get_original_index(x_init.numpy()[i], x))
                clusters[label] = sorted_original_indices

                # if label not in clusters:
                #     clusters[label] = []
                # clusters[label].append(get_original_index(x_init.numpy()[i], x))

            # Add the cluster centers to the dictionary
            # for label, center in zip(range(kmedoids.n_clusters), kmedoids.cluster_centers_):
            #     clusters[label].append(get_original_index(center, x))

            # Print the dictionary
            # print(clusters)

        init_reactions_indexes = []
        for init_index in init_indices_from_clusters:
            row = x_init.numpy()[init_index, :]
            orig_index = get_original_index(
                row, x
            )  # np.where((x.numpy() == row).all(axis=1))[0]
            init_reactions_indexes.append(orig_index)
        print(init_indices_from_clusters, " indices before ")
        print(init_reactions_indexes, "indices after")
        self.selected_reactions = init_reactions_indexes
        return init_reactions_indexes, clusters


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
