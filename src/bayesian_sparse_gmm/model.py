import warnings
from typing import Any, Dict, Optional

import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils.validation import check_array, check_is_fitted

from .backends import select_backend
from .config import HyperParams, SamplerConfig, SVIConfig
from .sampler import GibbsSampler
from .svi import SVIOptimizer
from .utils import log_sum_exp


class BayesianSparseGMM(BaseEstimator, ClusterMixin):
    """Bayesian Sparse Gaussian Mixture Model for high-dimensional clustering.

    Supports both exact MCMC via Gibbs Sampling and scalable Stochastic Variational Inference (SVI).

    Parameters
    ----------
    K_max : int, default=20
        Maximum number of clusters.
    optimizer : str, default='default'
        Optimization method: 'default' (Gibbs MCMC) or 'svi' (Natural Gradient SVI).
    n_iter : int, default=5000
        Number of Gibbs sampler iterations (MCMC only).
    burn_in : int, default=1000
        Number of burn-in iterations to discard (MCMC only).
    thinning : int, default=1
        Thinning interval for MCMC samples (MCMC only).
    warm_up_iters : int, default=50
        Warm-up iterations for feature selection (MCMC only).
    epochs : int, default=100
        Total passes over the dataset (SVI only).
    batch_size : int, default=256
        Mini-batch size for stochastic updates (SVI only).
    delay_rho : float, default=1.0
        Learning rate delay parameter tau_0 (SVI only).
    forgetting_rate : float, default=0.75
        Learning rate forgetting parameter kappa (SVI only).
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
        Truncated Poisson rate for K prior (MCMC only).
    use_identity_covariance : bool, default=True
        If True, use fixed identity covariance I_p (paper §6 default).
    backend : str, default='auto'
        Computation backend: 'numpy', 'numba', 'cuda', or 'auto'.
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
        optimizer: str = "default",
        n_iter: int = 5000,
        burn_in: int = 1000,
        thinning: int = 1,
        warm_up_iters: int = 50,
        epochs: int = 100,
        batch_size: int = 256,
        delay_rho: float = 1.0,
        forgetting_rate: float = 0.75,
        lambda_0: float = 100.0,
        lambda_1: float = 1.0,
        alpha: float = 1.0,
        theta: float = 0.5,
        kappa: float = 0.1,
        lambda_pois: float = 2.0,
        a_sigma: float = 1.0,
        b_sigma: float = 1.0,
        use_identity_covariance: bool = True,
        backend: str = "auto",
        n_jobs: int = -1,
        random_state: Optional[int] = None,
        verbose: int = 0,
    ):
        self.K_max = K_max
        self.optimizer = optimizer
        self.n_iter = n_iter
        self.burn_in = burn_in
        self.thinning = thinning
        self.warm_up_iters = warm_up_iters
        self.epochs = epochs
        self.batch_size = batch_size
        self.delay_rho = delay_rho
        self.forgetting_rate = forgetting_rate
        self.lambda_0 = lambda_0
        self.lambda_1 = lambda_1
        self.alpha = alpha
        self.theta = theta
        self.kappa = kappa
        self.lambda_pois = lambda_pois
        self.a_sigma = a_sigma
        self.b_sigma = b_sigma
        self.use_identity_covariance = use_identity_covariance
        self.backend = backend
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X: np.ndarray, y: Any = None) -> "BayesianSparseGMM":
        """Fit the GMM model."""
        X = check_array(X, dtype=[np.float64, np.float32])

        if self.optimizer not in ["default", "svi"]:
            raise ValueError("optimizer must be 'default' (MCMC) or 'svi'.")

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

        if self.use_identity_covariance:
            feat_var = np.var(X, axis=0)
            mean_var = float(np.mean(feat_var))
            if mean_var < 0.5 or mean_var > 2.0:
                warnings.warn(
                    f"Feature variance mean={mean_var:.4f} deviates significantly "
                    "from 1.0. With use_identity_covariance=True, data should be "
                    "standardized to unit variance (e.g. StandardScaler).",
                    UserWarning,
                    stacklevel=2,
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
            use_identity_covariance=self.use_identity_covariance,
        )

        self.backend_ = select_backend(self.backend)

        if self.optimizer == "default":
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
            sampler = GibbsSampler(config, hyperparams, self.backend_)
            self.states_ = sampler.run(X, seed=self.random_state)
            from .postprocessing import align_labels

            self.states_ = align_labels(self.states_, X)

            self.w_ = np.mean([state.w for state in self.states_], axis=0)
            self.means_ = np.mean([state.mu for state in self.states_], axis=0)
            self.feature_probabilities_ = np.mean(
                [state.xi for state in self.states_], axis=0
            )

            z_samples = np.array([state.z for state in self.states_])
            labels = np.empty(X.shape[0], dtype=int)
            for i in range(X.shape[0]):
                labels[i] = np.argmax(np.bincount(z_samples[:, i]))
            self.labels_ = labels

            K_samples = [state.K_active for state in self.states_]
            self.K_hat_ = int(np.argmax(np.bincount(K_samples)))

        else:
            config = SVIConfig(
                K_max=self.K_max,
                epochs=self.epochs,
                batch_size=self.batch_size,
                delay_rho=self.delay_rho,
                forgetting_rate=self.forgetting_rate,
                backend=self.backend,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
                verbose=self.verbose,
            )
            optimizer_algo = SVIOptimizer(config, hyperparams, self.backend_)
            self.state_ = optimizer_algo.optimize(X, seed=self.random_state)

            self.w_ = self.state_.expected_w
            self.means_ = self.state_.expected_mu
            self.feature_probabilities_ = self.state_.pi_xi
            self.labels_ = self.predict(X)
            self.K_hat_ = self.state_.K_active

        self.selected_features_ = np.where(self.feature_probabilities_ > 0.5)[0]
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict expected posterior probability of each cluster for each sample."""
        X = check_array(X, dtype=[np.float64, np.float32])
        n = X.shape[0]
        threshold = 1.0 / (2.0 * n)

        if self.optimizer == "default":
            check_is_fitted(self, "states_")
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
        else:
            check_is_fitted(self, "state_")
            w_safe = np.where(
                self.state_.expected_w < threshold, 1e-300, self.state_.expected_w
            )
            log_w = np.log(w_safe)
            log_probs = self.backend_.compute_cluster_log_probs(
                X, self.state_.expected_mu, log_w, self.state_.expected_sigma2
            )
            max_log = np.max(log_probs, axis=1, keepdims=True)
            probs = np.exp(log_probs - max_log)
            probs /= np.sum(probs, axis=1, keepdims=True)
            return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict cluster index for each sample."""
        return np.argmax(self.predict_proba(X), axis=1)

    def score(self, X: np.ndarray, y: Any = None) -> float:
        """Compute the average GMM log-likelihood of the dataset."""
        X = check_array(X, dtype=[np.float64, np.float32])
        n, p = X.shape
        threshold = 1.0 / (2.0 * n)

        if self.optimizer == "default":
            check_is_fitted(self, "states_")
            log_liks = []
            for state in self.states_:
                w_safe = np.where(state.w < threshold, 1e-300, state.w)
                log_w = np.log(w_safe)
                log_probs = self.backend_.compute_cluster_log_probs(
                    X, state.mu, log_w, state.sigma2
                )
                const = -0.5 * p * np.log(2.0 * np.pi) - 0.5 * np.sum(
                    np.log(state.sigma2)
                )
                sample_log_lik = log_sum_exp(log_probs, axis=1) + const
                log_liks.append(np.mean(sample_log_lik))
            return float(np.mean(log_liks))
        else:
            check_is_fitted(self, "state_")
            w_safe = np.where(
                self.state_.expected_w < threshold, 1e-300, self.state_.expected_w
            )
            log_w = np.log(w_safe)
            log_probs = self.backend_.compute_cluster_log_probs(
                X, self.state_.expected_mu, log_w, self.state_.expected_sigma2
            )
            const = -0.5 * p * np.log(2.0 * np.pi) - 0.5 * np.sum(
                np.log(self.state_.expected_sigma2)
            )
            sample_log_lik = log_sum_exp(log_probs, axis=1) + const
            return float(np.mean(sample_log_lik))

    @property
    def n_clusters_(self) -> int:
        """Number of clusters."""
        check_is_fitted(self, "K_hat_")
        return self.K_hat_

    @property
    def feature_probabilities_2d_(self) -> np.ndarray:
        """Deprecated: 2D feature probabilities (retained for backward compatibility)."""
        check_is_fitted(self, "feature_probabilities_")
        return np.broadcast_to(
            self.feature_probabilities_, (self.K_max, len(self.feature_probabilities_))
        )

    @property
    def trace_(self) -> Dict[str, np.ndarray]:
        """Full trace of MCMC samples (only valid for MCMC optimizer)."""
        check_is_fitted(self, "states_")
        if self.optimizer != "default":
            raise ValueError("trace_ is only available when optimizer='default'")
        return {
            "z": np.array([state.z for state in self.states_]),
            "w": np.array([state.w for state in self.states_]),
            "mu": np.array([state.mu for state in self.states_]),
            "xi": np.array([state.xi for state in self.states_]),
            "theta": np.array([state.theta for state in self.states_]),
            "sigma2": np.array([state.sigma2 for state in self.states_]),
        }
