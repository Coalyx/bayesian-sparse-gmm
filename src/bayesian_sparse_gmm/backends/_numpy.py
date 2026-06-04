import numpy as np
from ._base import ComputeBackend
from ..utils import sample_inverse_gaussian as utils_sample_inverse_gaussian

class NumpyBackend(ComputeBackend):
    """NumPy-based compute backend for the Bayesian Sparse GMM."""

    def compute_cluster_log_probs(
        self, X: np.ndarray, mu: np.ndarray, log_w: np.ndarray, sigma2: np.ndarray
    ) -> np.ndarray:
        """Step 2: Compute log probability of cluster assignment for all samples and clusters."""
        # Scale X and mu by sqrt(sigma2) to compute diagonal Mahalanobis distance efficiently via GEMM trick
        std = np.sqrt(sigma2)
        X_scaled = X / std
        mu_scaled = mu / std
        x_sq = np.sum(X_scaled**2, axis=1, keepdims=True)
        mu_sq = np.sum(mu_scaled**2, axis=1, keepdims=True).T
        dist = x_sq - 2.0 * np.dot(X_scaled, mu_scaled.T) + mu_sq
        dist = np.maximum(dist, 0.0)
        return log_w - 0.5 * dist

    def compute_sufficient_stats(
        self, X: np.ndarray, z: np.ndarray, K_max: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Step 4: Compute cluster sizes and feature sums per cluster."""
        n_k = np.bincount(z, minlength=K_max)
        sum_x = np.zeros((K_max, X.shape[1]), dtype=X.dtype)
        np.add.at(sum_x, z, X)
        return n_k, sum_x

    def sample_cluster_means(
        self, sum_x: np.ndarray, n_k: np.ndarray,
        tau2: np.ndarray, sigma2: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        """Step 4b: Sample cluster means mu[k,j] ~ Normal."""
        post_var = 1.0 / (n_k[:, np.newaxis] / sigma2[np.newaxis, :] + 1.0 / tau2)
        post_mean = post_var * (sum_x / sigma2[np.newaxis, :])
        return rng.normal(loc=post_mean, scale=np.sqrt(post_var))

    def sample_inverse_gaussian(
        self, mu_abs: np.ndarray, lam: np.ndarray,
        rng: np.random.Generator
    ) -> np.ndarray:
        """Step 4a: Sample tau^2 via Inverse Gaussian."""
        inv_mean = lam / (mu_abs + 1e-10)
        shape = lam ** 2
        inv_tau2 = utils_sample_inverse_gaussian(inv_mean, shape, rng)
        return 1.0 / inv_tau2
