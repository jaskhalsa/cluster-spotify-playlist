from collections import defaultdict
from typing import Dict, List, Tuple, Union

import numpy as np
from kneed import KneeLocator
from numpy import ndarray
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture


class ClusterHandler:
    features: ndarray
    algorithm: str
    pca_k: int
    random_state: int
    model: Union[GaussianMixture, KMeans]

    def __init__(
        self,
        features: ndarray,
        algorithm: str = "kmeans",
        pca_k: Union[int, None] = None,
        random_state: int = 12345,
    ) -> None:
        if pca_k:
            self.features = PCA(n_components=pca_k).fit_transform(features)
        else:
            self.features = features

        self.algorithm = algorithm
        self.pca_k = pca_k
        self.random_state = random_state

    def __get_model(self, k: int) -> Union[GaussianMixture, KMeans]:
        if self.algorithm == "gmm":
            return GaussianMixture(n_components=k, random_state=self.random_state)
        return KMeans(n_clusters=k, random_state=self.random_state, init="k-means++")

    def __get_centroids(self, model: Union[GaussianMixture, KMeans]) -> np.ndarray:
        if self.algorithm == "gmm":
            return model.means_
        return model.cluster_centers_

    def get_inertias(self, k_min: int, k_max: int) -> List[Tuple[int, float]]:
        """
        This function gets the inertias from k_min to k_max
        and returns these as a tuple with the k that the inertias
        correspond to.
        Args:
            k_min (int): min cluster size
            k_max (int): max cluster size
        Returns:
            List[Tuple[int, float]]: List of tuples containing (k, inertia)
        """
        inertias = []

        for k in range(k_min, k_max + 1):
            model = self.__get_model(k).fit(self.features)

            inertias.append((k, model.inertia_))

        return inertias

    @staticmethod
    def calculate_optimal_cluster_given_inertias(
        unordered_inertias: List[Tuple[int, float]], knee_locator_cls: KneeLocator
    ) -> int:
        """
        This function calculates the optimum cluster size given a range of inertias.
        It computes the optimum cluster size using the elbow method.
        The shape of the function is convex and decreasing in ascending order of
        cluster sizes.
        Args:
            unordered_inertias (List[Tuple[int, float]]): inertias
            knee_locator_cls (KneeLocator): The class for the knee_locator
        Returns:
            int: optimum cluster size
        """
        sorted_inertias = sorted(unordered_inertias, key=lambda x: x[0])
        x, y = zip(*sorted_inertias)
        kneedle = knee_locator_cls(x, y, S=1.0, curve="convex", direction="decreasing")
        optimum_k = int(kneedle.elbow)
        return optimum_k

    def cluster(self, k: int) -> Dict[int, List[Tuple[int, float]]]:
        """
        Clusters and returns the alias and their distances for each
        cluster centroid
        Args:
            k (int): cluster size
        Returns:
            Dict[int, List[Tuple[int, float]]]: Returns a dict of cluster centroid
                and a list of tuples with (alias_index, distance_to_centroid)
        """
        self.model = self.__get_model(k).fit(self.features)
        cluster_args = self.get_items_for_each_cluster()
        return cluster_args

    def get_items_for_each_cluster(self) -> Dict[int, List[int]]:
        """
        This assigns a centroid for the features.
        It then returns a dict of cluster centroids and a list of tuples
        corresponding to (alias_index, distance_to_centroid)
        Returns:
            Dict[int, List[int]]: Returns a dict of cluster centroids and a list of tuples
                corresponding to (alias_index, distance_to_centroid)
        """
        labels = self.model.fit_predict(self.features)
        centroids = self.__get_centroids(self.model)
        args = defaultdict(list)

        for i, feature in enumerate(self.features):
            label = labels[i]
            centroid = centroids[label]
            value = np.linalg.norm(feature - centroid)
            args[label].append((i, value))

        for centroid in args:
            args[centroid] = sorted(args[centroid], key=lambda x: x[1])

        return args
