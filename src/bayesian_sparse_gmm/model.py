from typing import Any, Dict, Optional

import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils.validation import check_array, check_is_fitted

from .backends import select_backend
from .config import HyperParams, SamplerConfig
from .sampler import GibbsSampler
from .utils import log_sum_exp


class BayesianSparseGMM(BaseEstimator, ClusterMixin):
    """Bayesian Sparse Gaussian Mixture Model for high-dimensional clustering.

    Parameters
    ----------
    K_max : int, default=20
        Maximum number of clusters.
    n_iter : int, default=5000
        Number of Gibbs sampler iterations.
    burn_in : int, default=1000
        Number of burn-in iterations to discard.
    thinning : int, default=1
        Thinning interval for MCMC samples.
    lambda_0 : float, default=100.0
        Spike prior parameter (large value for sparse features).
    lambda_1 : float, default=1.0
        Slab prior parameter (small value for active features).
    alpha : float, default=1.0
        Dirichlet prior parameter for mixing weights.
    theta : float, default=0.5
        Prior probability of a feature being informative (Slab).
    kappa : float, default=0.1
        Sparsity aggressiveness parameter.
    lambda_pois : float, default=2.0
        Truncated Poisson rate for K prior.
    backend : str, default='auto'
        Computation backend: 'numpy', 'numba', or 'auto'.
    n_jobs : int, default=-1
        Parallel jobs for Numba.
    random_state : int, optional
        Seed for the random number generator.
    verbose : int, default=0
        Progress reporting interval.
    """

    def __init__(
        self,
        K_max: int = 20,
        n_iter: int = 5000,
        burn_in: int = 1000,
        thinning: int = 1,
        warm_up_iters: int = 50,
        lambda_0: float = 100.0,
        lambda_1: float = 1.0,
        alpha: float = 1.0,
        theta: float = 0.5,
        kappa: float = 0.1,
        lambda_pois: float = 2.0,
        a_sigma: float = 1.0,
        b_sigma: float = 1.0,
        backend: str = "auto",
        n_jobs: int = -1,
        random_state: Optional[int] = None,
        verbose: int = 0,
    ):
        self.K_max = K_max
        self.n_iter = n_iter
        self.burn_in = burn_in
        self.thinning = thinning
        self.warm_up_iters = warm_up_iters
        self.lambda_0 = lambda_0
        self.lambda_1 = lambda_1
        self.alpha = alpha
        self.theta = theta
        self.kappa = kappa
        self.lambda_pois = lambda_pois
        self.a_sigma = a_sigma
        self.b_sigma = b_sigma
        self.backend = backend
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X: np.ndarray, y: Any = None) -> "BayesianSparseGMM":
        """Fit the GMM model using Gibbs sampling."""
        X = check_array(X, dtype=[np.float64, np.float32])

        if self.alpha < 1.0:
            raise ValueError(
                f"alpha must be >= 1.0 (paper §2 constraint). Got alpha={self.alpha}. "
                "Values < 1 produce degenerate Dirichlet priors."
            )
        if self.lambda_0 <= self.lambda_1:
            raise ValueError(
                "lambda_0 must be > lambda_1 (spike > slab). "
                f"Got lambda_0={self.lambda_0}, lambda_1={self.lambda_1}."
            )

        config = SamplerConfig(
            K_max=self.K_max,
            n_iter=self.n_iter,
            burn_in=self.burn_in,
            thinning=self.thinning,
            warm_up_iters=self.warm_up_iters,
            backend=self.backend,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            verbose=self.verbose,
        )
        hyperparams = HyperParams(
            lambda_0=self.lambda_0,
            lambda_1=self.lambda_1,
            alpha=self.alpha,
            theta=self.theta,
            kappa=self.kappa,
            lambda_pois=self.lambda_pois,
            a_sigma=self.a_sigma,
            b_sigma=self.b_sigma,
        )

        self.backend_ = select_backend(config.backend)
        sampler = GibbsSampler(config, hyperparams, self.backend_)

        self.states_ = sampler.run(X, seed=self.random_state)

        from .postprocessing import align_labels

        self.states_ = align_labels(self.states_)

        self.w_ = np.mean([state.w for state in self.states_], axis=0)
        self.means_ = np.mean([state.mu for state in self.states_], axis=0)

        # Joint xi is 1D (p,) — posterior inclusion probability per feature
        self.feature_probabilities_ = np.mean(
            [state.xi for state in self.states_], axis=0
        )
        self.selected_features_ = np.where(self.feature_probabilities_ > 0.5)[0]

        # Final label assignment based on mode over samples
        z_samples = np.array([state.z for state in self.states_])
        labels = np.empty(X.shape[0], dtype=int)
        for i in range(X.shape[0]):
            labels[i] = np.argmax(np.bincount(z_samples[:, i]))
        self.labels_ = labels

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict posterior probability of each cluster for each sample."""
        X = check_array(X, dtype=[np.float64, np.float32])
        check_is_fitted(self, "states_")

        n = X.shape[0]
        threshold = 1.0 / (2.0 * n)
        all_probs = []
        for state in self.states_:
            w_safe = np.where(state.w < threshold, 1e-300, state.w)
            log_w = np.log(w_safe)
            log_probs = self.backend_.compute_cluster_log_probs(
                X, state.mu, log_w, state.sigma2
            )

            max_log = np.max(log_probs, axis=1, keepdims=True)
            probs = np.exp(log_probs - max_log)
            probs /= np.sum(probs, axis=1, keepdims=True)
            all_probs.append(probs)

        return np.mean(all_probs, axis=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict cluster index for each sample."""
        return np.argmax(self.predict_proba(X), axis=1)

    def score(self, X: np.ndarray, y: Any = None) -> float:
        """Compute the average GMM log-likelihood of the dataset."""
        X = check_array(X, dtype=[np.float64, np.float32])
        check_is_fitted(self, "states_")

        n, p = X.shape
        threshold = 1.0 / (2.0 * n)

        log_liks = []
        for state in self.states_:
            w_safe = np.where(state.w < threshold, 1e-300, state.w)
            log_w = np.log(w_safe)
            log_probs = self.backend_.compute_cluster_log_probs(
                X, state.mu, log_w, state.sigma2
            )

            const = -0.5 * p * np.log(2.0 * np.pi) - 0.5 * np.sum(np.log(state.sigma2))
            sample_log_lik = log_sum_exp(log_probs, axis=1) + const
            log_liks.append(np.mean(sample_log_lik))

        return float(np.mean(log_liks))

    @property
    def n_clusters_(self) -> int:
        """Number of active clusters."""
        check_is_fitted(self, "labels_")
        return len(np.unique(self.labels_))

    @property
    def feature_probabilities_2d_(self) -> np.ndarray:
        """Deprecated: 2D feature probabilities (retained for backward compatibility)."""
        check_is_fitted(self, "feature_probabilities_")
        return np.broadcast_to(
            self.feature_probabilities_, (self.K_max, len(self.feature_probabilities_))
        )

    @property
    def trace_(self) -> Dict[str, np.ndarray]:
        """Full trace of MCMC samples."""
        check_is_fitted(self, "states_")
        return {
            "z": np.array([state.z for state in self.states_]),
            "w": np.array([state.w for state in self.states_]),
            "mu": np.array([state.mu for state in self.states_]),
            "xi": np.array([state.xi for state in self.states_]),
            "theta": np.array([state.theta for state in self.states_]),
            "sigma2": np.array([state.sigma2 for state in self.states_]),
        }
