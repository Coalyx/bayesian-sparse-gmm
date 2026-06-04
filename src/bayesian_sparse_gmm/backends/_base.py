from abc import ABC, abstractmethod

import numpy as np


class ComputeBackend(ABC):
    """Abstract base class for all compute backends."""

    @abstractmethod
    def compute_cluster_log_probs(
        self, X: np.ndarray, mu: np.ndarray, log_w: np.ndarray, sigma2: np.ndarray
    ) -> np.ndarray:
        """Step 2: Compute log probability of cluster assignment for all samples and clusters.

        Returns shape (n, K_max)
        """
        pass

    @abstractmethod
    def compute_sufficient_stats(
        self, X: np.ndarray, z: np.ndarray, K_max: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Step 4: Compute cluster sizes and feature sums per cluster.

        Returns (n_k, sum_x)
        """
        pass

    @abstractmethod
    def sample_cluster_means(
        self,
        sum_x: np.ndarray,
        n_k: np.ndarray,
        tau2: np.ndarray,
        sigma2: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Step 4b: Sample cluster means mu[k,j] ~ Normal."""
        pass

    @abstractmethod
    def sample_inverse_gaussian(
        self, mu_abs: np.ndarray, lam: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        """Step 4a: Sample tau^2 via Inverse Gaussian."""
        pass
