#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Python implementation of Self-Organising Map (SOM).
"""


import numpy as np
import numba


class SOM(object):
    def __init__(
        self, height: int, width: int, max_iter: int, alpha_start=0.1, seed=1234
    ):
        # Reproducibility
        np.random.seed(seed)
        # Network dimension
        self._height = height
        self._width = width
        # Maximum iteration
        self._max_iter = max_iter
        # Initial alpha (learning rate) at start
        self._alpha_start = alpha_start
        # Initial neighbour radius
        self._sigma = max(width, height) / 2.0
        # Initial time constant
        self._lambda = max_iter / np.log(max(width, height) / 2.0)
        # Indicator whether the map weights have been initialized
        self._initialized = False

        # Array of learning rates
        self._alphas = None
        # Array of neighbour radius
        self._sigmas = None
        # Weight matrix
        self._map = np.array([])

    def _initialize_weights(self, data: np.ndarray) -> None:
        """Initialize the node weights from Gaussian distribution.

        Returns: initialized map

        Args:
            data: given input data.
        """
        self._map = np.random.normal(size=(self._height, self._width, data.shape[1]))
        self._initialized = True

    def _calc_closet_node(self, input_vect: np.ndarray) -> np.ndarray:
        """Calculate the node that is closest to the input data point.

        Args:
            input_vect: input data point

        Returns:
            index of the node (BMU) in SOM
        """
        dist_mat = np.sqrt(np.sum((self._map - input_vect) ** 2, axis=2))
        bmu_idx = np.where(dist_mat == np.amin(dist_mat))
        return np.array(bmu_idx).reshape(-1)

    def _fit_on_data(self, input_vect: np.ndarray, epoch: int) -> None:
        """For each iteration,
        adapt SOM weights towards a given data point.

        Args:
            input_vect: input vector for current data point.
        """
        # Find BMU
        bmu_idx = self._calc_closet_node(input_vect)
        for x in range(self._height):
            for y in range(self._width):
                self._map[x, y, :] = self._update_node_weights(
                    input_vect,
                    node_weights=self._map[x, y, :],
                    node_idx=np.array([x, y]),
                    bmu_idx=bmu_idx,
                    radius=self._sigmas[epoch],
                    learning_rate=self._alphas[epoch],
                )
        print(
            f"Epoch={epoch}, BMU_idx={[bmu_idx[0], bmu_idx[1]]}, radius="
            f"{self._sigmas[epoch]}, learning_rate={self._alphas[epoch]}"
        )

    @staticmethod
    @numba.jit(nopython=True)
    def _update_node_weights(
        input_vect: np.ndarray,
        node_weights: np.ndarray,
        node_idx: np.ndarray,
        bmu_idx: np.ndarray,
        radius: float,
        learning_rate: float,
    ) -> np.ndarray:
        """Update weights of the given neighbour node according to BMU.

        Args:
            input_vect: input data point.
            node_weights: weights vector for a node in the network.
            node_idx: x and y coordinates for the given node.
            bmu_idx: x and y coordinates for the BMU node.
            radius: the radius for this iteration.
            learning_rate: the learning rate for this iteration.

        Returns:
            Updated weights for the given neighbour node.
        """
        dist = np.sqrt(np.sum((node_idx - bmu_idx) ** 2))

        if dist <= radius:
            # Calculate influence for neighbour node
            influence = np.exp(-dist / (2 * (radius ** 2)))

            # Update weight matrix
            new_weights = node_weights + (
                learning_rate * influence * (input_vect - node_weights)
            )
            return new_weights
        else:
            return node_weights

    def _calc_decays_per_iteration(self) -> np.ndarray:
        """Calculate decay weight per iteration.

        Returns: An array of decay weights.

        """
        epoch_lst = np.linspace(0, 1, self._max_iter)
        return np.exp(-epoch_lst / self._lambda)

    def fit(self, data: np.ndarray) -> None:
        """Train the SOM on the given data for several iterations.

        Args:
            data: input data array.
        """
        if not self._initialized:
            self._initialize_weights(data)

        # Randomly select a data point to pass into SOM
        indx = np.random.choice(np.arange(len(data)), self._max_iter)

        # Set learning rate decays for given number of iteration
        self._alphas = self._alpha_start * self._calc_decays_per_iteration()
        # Set neighbourhood radius decays for given number of iteration
        self._sigmas = self._sigma * self._calc_decays_per_iteration()

        for i in range(self._max_iter):
            self._fit_on_data(data[indx[i]], i)

    @property
    def get_map(self):
        return self._map
