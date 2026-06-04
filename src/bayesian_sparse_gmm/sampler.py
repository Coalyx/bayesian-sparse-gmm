from typing import List, Optional

import numpy as np
from tqdm import tqdm

from .backends._base import ComputeBackend
from .config import HyperParams, SamplerConfig
from .state import SamplerState


class GibbsSampler:
    """Gibbs Sampler orchestrator for Bayesian Sparse GMM."""

    def __init__(
        self, config: SamplerConfig, hyperparams: HyperParams, backend: ComputeBackend
    ):
        self.config = config
        self.hyperparams = hyperparams
        self.backend = backend

    def initialize(self, X: np.ndarray, rng: np.random.Generator) -> SamplerState:
        """Initialize the sampler state using K-Means++ or random assignment."""
        n, p = X.shape
        K_max = self.config.K_max

        try:
            from sklearn.cluster import KMeans

            kmeans = KMeans(
                n_clusters=K_max,
                init="k-means++",
                n_init=1,
                random_state=rng.integers(0, 2**31 - 1),
            )
            kmeans.fit(X)
            z = kmeans.labels_
            mu = kmeans.cluster_centers_.copy()

        except Exception:
            z = rng.choice(K_max, size=n)
            mu = rng.normal(size=(K_max, p))

        w = np.ones(K_max) / K_max
        gamma = np.ones((K_max, p), dtype=np.int32)
        theta = self.hyperparams.theta
        tau2 = np.ones((K_max, p))
        sigma2 = np.ones(p)

        return SamplerState(
            z=z,
            w=w,
            mu=mu,
            gamma=gamma,
            theta=theta,
            tau2=tau2,
            sigma2=sigma2,
            iteration=0,
        )

    def sample_step(
        self, X: np.ndarray, state: SamplerState, rng: np.random.Generator
    ) -> SamplerState:
        """Execute one complete Gibbs sampling iteration."""
        K_max = self.config.K_max
        n, p = X.shape
        hp = self.hyperparams

        # STEP 1: Update cluster assignments (z) using the previous mixing weights
        threshold = 1.0 / (2.0 * n)
        w_safe = np.where(state.w < threshold, 1e-300, state.w)
        log_w = np.log(w_safe)
        log_probs = self.backend.compute_cluster_log_probs(
            X, state.mu, log_w, state.sigma2
        )
        max_log = np.max(log_probs, axis=1, keepdims=True)
        probs = np.exp(log_probs - max_log)
        probs /= np.sum(probs, axis=1, keepdims=True)

        cumsum = np.cumsum(probs, axis=1)
        u = rng.uniform(size=(n, 1))
        z = np.sum(cumsum < u, axis=1)
        z = np.clip(z, 0, K_max - 1)

        # STEP 2: Update cluster mixing weights (w) using the new cluster assignments
        n_k = np.bincount(z, minlength=K_max)
        w = rng.dirichlet(hp.alpha + n_k)

        # STEP 3: Update feature inclusion indicators (gamma) using active cluster mask
        active_mask = n_k > 0
        active_K = int(np.sum(active_mask))
        if active_K == 0:
            active_K = K_max
            active_mask = np.ones(K_max, dtype=bool)

        gamma = np.zeros((K_max, p), dtype=np.int32)
        if state.iteration < self.config.warm_up_iters:
            # During warm-up, force gamma to 1 for all active clusters to allow means to migrate
            gamma[active_mask] = 1
        else:
            # Local feature selection: sample gamma_kj independently for active clusters
            active_mu = state.mu[active_mask]
            log_laplace_slab = (
                np.log(hp.lambda_1) - np.log(2.0)
            ) - hp.lambda_1 * np.abs(active_mu)
            log_laplace_spike = (
                np.log(hp.lambda_0) - np.log(2.0)
            ) - hp.lambda_0 * np.abs(active_mu)

            safe_theta = np.clip(state.theta, 1e-15, 1.0 - 1e-15)
            log_P_slab = np.log(safe_theta) + log_laplace_slab
            log_P_spike = np.log(1.0 - safe_theta) + log_laplace_spike

            max_log = np.maximum(log_P_slab, log_P_spike)
            prob_slab = np.exp(log_P_slab - max_log)
            prob_spike = np.exp(log_P_spike - max_log)
            p_slab = prob_slab / (prob_slab + prob_spike)

            gamma[active_mask] = rng.binomial(1, p_slab)

        # STEP 3b: Update feature-specific variances (sigma2)
        # sigma2_j ~ Inverse-Gamma(a_sigma + N / 2, b_sigma + 0.5 * sum_i (X_ij - mu_{Z_i, j})^2)
        residuals_sq = (X - state.mu[z]) ** 2
        sum_residuals_sq = np.sum(residuals_sq, axis=0)

        shape_sig = hp.a_sigma + 0.5 * n
        scale_sig = 1.0 / (hp.b_sigma + 0.5 * sum_residuals_sq)

        gamma_sig_sample = rng.gamma(shape_sig, scale_sig)
        sigma2 = 1.0 / np.maximum(gamma_sig_sample, 1e-15)

        # STEP 4: Update cluster means (mu) and auxiliary variables (tau2)
        lam = np.where(gamma == 1, hp.lambda_1, hp.lambda_0)
        tau2 = self.backend.sample_inverse_gaussian(np.abs(state.mu), lam, rng)

        n_k_new, sum_x = self.backend.compute_sufficient_stats(X, z, K_max)
        mu = self.backend.sample_cluster_means(sum_x, n_k_new, tau2, sigma2, rng)

        return SamplerState(
            z=z,
            w=w,
            mu=mu,
            gamma=gamma,
            theta=state.theta,
            tau2=tau2,
            sigma2=sigma2,
            iteration=state.iteration + 1,
        )

    def run(self, X: np.ndarray, seed: Optional[int] = None) -> List[SamplerState]:
        """Run the Gibbs sampler chain and return thinned post-burn-in states."""
        rng = np.random.default_rng(seed)
        state = self.initialize(X, rng)

        n_iter = self.config.n_iter
        burn_in = self.config.burn_in
        thinning = self.config.thinning
        verbose = self.config.verbose

        states = []
        iterator = tqdm(range(1, n_iter + 1), disable=not verbose, desc="Gibbs Sampler")
        for i in iterator:
            state = self.sample_step(X, state, rng)

            if i > burn_in and (i - burn_in) % thinning == 0:
                states.append(state)

        return states
