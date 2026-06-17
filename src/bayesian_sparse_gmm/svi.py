from typing import Optional

import numpy as np
from scipy.special import digamma
from tqdm import tqdm

from .backends._base import ComputeBackend
from .config import HyperParams, SVIConfig
from .state import VariationalState


class SVIOptimizer:
    """Stochastic Variational Inference Optimizer using Natural Gradients.

    Implements coordinate ascent variational inference (CAVI) combined with
    stochastic optimization for the Bayesian Sparse GMM.
    """

    def __init__(
        self, config: SVIConfig, hyperparams: HyperParams, backend: ComputeBackend
    ):
        self.config = config
        self.hyperparams = hyperparams
        self.backend = backend

    def initialize(self, X: np.ndarray, rng: np.random.Generator) -> VariationalState:
        """Initialize the variational state using K-Means++ or random assignment.

        Args:
            X: Data matrix of shape (n_samples, p_features).
            rng: Random number generator.

        Returns:
            Initial variational state.
        """
        n, p = X.shape
        K_max = self.config.K_max
        hp = self.hyperparams

        self.beta_theta = (p ** (1 + hp.kappa)) / np.maximum(np.log(p), 1e-10)

        try:
            from sklearn.cluster import KMeans

            kmeans = KMeans(
                n_clusters=K_max,
                init="k-means++",
                n_init=1,
                random_state=rng.integers(0, 2**31 - 1),
            )
            kmeans.fit(X)
            expected_mu = kmeans.cluster_centers_.copy()
        except Exception:
            expected_mu = rng.normal(size=(K_max, p))

        global_N_k = np.ones(K_max) * (n / K_max)
        global_sum_x = expected_mu * global_N_k[:, np.newaxis]

        expected_sigma2 = np.ones(p)

        dirichlet_alpha = np.ones(K_max) * (hp.alpha / K_max) + global_N_k
        expected_w = dirichlet_alpha / np.sum(dirichlet_alpha)

        pi_xi = np.full(p, hp.theta)
        expected_theta = hp.theta

        s_active = int(np.sum(pi_xi))
        beta_a = 1.0 + s_active
        beta_b = self.beta_theta + p - s_active

        return VariationalState(
            expected_mu=expected_mu,
            expected_sigma2=expected_sigma2,
            expected_w=expected_w,
            pi_xi=pi_xi,
            expected_theta=expected_theta,
            dirichlet_alpha=dirichlet_alpha,
            beta_a=beta_a,
            beta_b=beta_b,
            global_N_k=global_N_k,
            global_sum_x=global_sum_x,
            epoch=0,
            iteration=0,
            K_active=K_max,
        )

    def _get_learning_rate(self, iteration: int) -> float:
        """Compute the learning rate rho_t for natural gradient update."""
        return (iteration + self.config.delay_rho) ** (-self.config.forgetting_rate)

    def _local_step(
        self, X_batch: np.ndarray, state: VariationalState
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute expected sufficient statistics for the mini-batch (E-step).

        Args:
            X_batch: Mini-batch data of shape (batch_size, p_features).
            state: Current variational state.

        Returns:
            Tuple of (expected_n_k, expected_sum_x) for the batch.
        """
        expected_log_w = digamma(state.dirichlet_alpha) - digamma(
            np.sum(state.dirichlet_alpha)
        )

        log_probs = self.backend.compute_cluster_log_probs(
            X_batch, state.expected_mu, expected_log_w, state.expected_sigma2
        )

        max_log = np.max(log_probs, axis=1, keepdims=True)
        r_ik = np.exp(log_probs - max_log)
        r_ik /= np.sum(r_ik, axis=1, keepdims=True)

        return self.backend.compute_expected_sufficient_stats(
            X_batch, r_ik, self.config.K_max
        )

    def _update_feature_inclusion(
        self, state: VariationalState
    ) -> tuple[np.ndarray, float, float, float]:
        """Update variational parameters for feature selection (xi and theta)."""
        hp = self.hyperparams
        p = state.expected_mu.shape[1]
        active_clusters = np.where(state.global_N_k > 0.01)[0]

        if len(active_clusters) == 0:
            return (
                np.full(p, state.expected_theta),
                state.expected_theta,
                state.beta_a,
                state.beta_b,
            )

        sum_abs_mu = np.sum(np.abs(state.expected_mu[active_clusters, :]), axis=0)
        safe_theta = np.clip(state.expected_theta, 1e-15, 1.0 - 1e-15)

        log_slab = (
            np.log(safe_theta)
            + len(active_clusters) * np.log(hp.lambda_1 / 2.0)
            - hp.lambda_1 * sum_abs_mu
        )
        log_spike = (
            np.log(1.0 - safe_theta)
            + len(active_clusters) * np.log(hp.lambda_0 / 2.0)
            - hp.lambda_0 * sum_abs_mu
        )

        log_denom = np.logaddexp(log_slab, log_spike)
        pi_xi = np.exp(log_slab - log_denom)

        expected_s_active = np.sum(pi_xi)
        beta_a = 1.0 + expected_s_active
        beta_b = self.beta_theta + p - expected_s_active
        expected_theta = beta_a / (beta_a + beta_b)

        return pi_xi, expected_theta, beta_a, beta_b

    def _global_step(
        self,
        X_batch: np.ndarray,
        batch_n_k: np.ndarray,
        batch_sum_x: np.ndarray,
        state: VariationalState,
        full_n: int,
        iteration: int,
    ) -> VariationalState:
        """Update global variational parameters using natural gradients (M-step)."""
        hp = self.hyperparams
        batch_size = X_batch.shape[0]
        rho = self._get_learning_rate(iteration)

        scaling = full_n / batch_size
        scaled_n_k = batch_n_k * scaling
        scaled_sum_x = batch_sum_x * scaling

        new_alpha = (hp.alpha / self.config.K_max) + scaled_n_k
        dirichlet_alpha = (1.0 - rho) * state.dirichlet_alpha + rho * new_alpha
        expected_w = dirichlet_alpha / np.sum(dirichlet_alpha)

        global_N_k = (1.0 - rho) * state.global_N_k + rho * scaled_n_k
        global_sum_x = (1.0 - rho) * state.global_sum_x + rho * scaled_sum_x

        # Compute expected penalty for each feature based on current inclusion probs
        C_j = state.pi_xi * hp.lambda_1 + (1.0 - state.pi_xi) * hp.lambda_0

        expected_mu = np.zeros_like(state.expected_mu)
        for k in range(self.config.K_max):
            if global_N_k[k] > 1e-10:
                bar_x = global_sum_x[k] / global_N_k[k]
                # Soft-thresholding to shrink means towards exactly 0
                penalty = C_j / global_N_k[k]
                expected_mu[k] = np.sign(bar_x) * np.maximum(0.0, np.abs(bar_x) - penalty)
            else:
                expected_mu[k] = state.expected_mu[k]

        # Now that expected_mu is properly shrunken, update feature inclusion
        pi_xi, expected_theta, beta_a, beta_b = self._update_feature_inclusion(state)

        if hp.use_identity_covariance:
            expected_sigma2 = np.ones(state.expected_sigma2.shape)
        else:
            expected_sigma2 = state.expected_sigma2

        K_active = int(np.sum(global_N_k > 0.01))

        return VariationalState(
            expected_mu=expected_mu,
            expected_sigma2=expected_sigma2,
            expected_w=expected_w,
            pi_xi=pi_xi,
            expected_theta=expected_theta,
            dirichlet_alpha=dirichlet_alpha,
            beta_a=beta_a,
            beta_b=beta_b,
            global_N_k=global_N_k,
            global_sum_x=global_sum_x,
            epoch=state.epoch,
            iteration=iteration,
            K_active=K_active,
        )

    def optimize(self, X: np.ndarray, seed: Optional[int] = None) -> VariationalState:
        """Run SVI optimization over the dataset for configured epochs.

        Args:
            X: Full data matrix.
            seed: Random seed.

        Returns:
            Final variational state.
        """
        rng = np.random.default_rng(seed)
        state = self.initialize(X, rng)

        n, _ = X.shape
        batch_size = min(self.config.batch_size, n)

        indices = np.arange(n)
        iterator = tqdm(
            range(self.config.epochs),
            disable=not self.config.verbose,
            desc="SVI Epochs",
        )
        iteration = 1

        for epoch in iterator:
            rng.shuffle(indices)

            for start_idx in range(0, n, batch_size):
                end_idx = min(start_idx + batch_size, n)
                batch_indices = indices[start_idx:end_idx]
                X_batch = X[batch_indices]

                batch_n_k, batch_sum_x = self._local_step(X_batch, state)
                state = self._global_step(
                    X_batch,
                    batch_n_k,
                    batch_sum_x,
                    state,
                    full_n=n,
                    iteration=iteration,
                )
                iteration += 1

            state.epoch = epoch + 1

        return state
