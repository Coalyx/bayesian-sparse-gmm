import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils.validation import check_array, check_is_fitted
from typing import Optional, Dict, Any

from .config import SamplerConfig, HyperParams
from .state import SamplerState
from .backends import select_backend
from .sampler import GibbsSampler
from .utils import log_sum_exp

class BayesianSparseGMM(BaseEstimator, ClusterMixin):
    """Bayesian Sparse Gaussian Mixture Model for high-dimensional clustering.
    
    Parameters
    ----------
    K_max : int, default=15
        Maximum number of clusters.
    n_iter : int, default=2000
        Number of Gibbs sampler iterations.
    burn_in : int, default=500
        Number of burn-in iterations to discard.
    thinning : int, default=1
        Thinning interval for MCMC samples.
    lambda_0 : float, default=1000.0
        Spike prior parameter (large value for sparse features).
    lambda_1 : float, default=0.1
        Slab prior parameter (small value for active features).
    alpha : float, default=0.01
        Dirichlet prior parameter for mixing weights.
    a : float, default=1.0
        Beta prior parameter a for sparsity probability.
    b : float, default=100.0
        Beta prior parameter b for sparsity probability.
    backend : str, default='auto'
        Computation backend: 'numpy', 'numba', or 'auto'.
    n_jobs : int, default=-1
        Number of parallel jobs (for Numba backend).
    random_state : int, optional
        Seed for the random number generator.
    verbose : int, default=0
        Progress reporting interval.
    """

    def __init__(
        self,
        K_max: int = 15,
        n_iter: int = 2000,
        burn_in: int = 500,
        thinning: int = 1,
        lambda_0: float = 1000.0,
        lambda_1: float = 0.1,
        alpha: float = 0.01,
        a: float = 1.0,
        b: float = 100.0,
        backend: str = "auto",
        n_jobs: int = -1,
        random_state: Optional[int] = None,
        verbose: int = 0,
    ):
        self.K_max = K_max
        self.n_iter = n_iter
        self.burn_in = burn_in
        self.thinning = thinning
        self.lambda_0 = lambda_0
        self.lambda_1 = lambda_1
        self.alpha = alpha
        self.a = a
        self.b = b
        self.backend = backend
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X: np.ndarray, y: Any = None) -> "BayesianSparseGMM":
        """Fit the GMM model using Gibbs sampling."""
        X = check_array(X, dtype=[np.float64, np.float32])
        
        # Build configuration and hyperparameter objects
        config = SamplerConfig(
            K_max=self.K_max,
            n_iter=self.n_iter,
            burn_in=self.burn_in,
            thinning=self.thinning,
            backend=self.backend,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            verbose=self.verbose,
        )
        hyperparams = HyperParams(
            lambda_0=self.lambda_0,
            lambda_1=self.lambda_1,
            alpha=self.alpha,
            a=self.a,
            b=self.b,
        )
        
        # Instantiate backend and sampler
        self.backend_ = select_backend(config.backend)
        sampler = GibbsSampler(config, hyperparams, self.backend_)
        
        # Run sampler
        self.states_ = sampler.run(X, seed=self.random_state)
        
        from .postprocessing import align_labels
        self.states_ = align_labels(self.states_)
        
        # Compute and cache point estimates
        self.w_ = np.mean([state.w for state in self.states_], axis=0)
        self.means_ = np.mean([state.mu for state in self.states_], axis=0)
        self.feature_probabilities_ = np.mean([state.gamma for state in self.states_], axis=0)
        self.selected_features_ = np.where(self.feature_probabilities_ > 0.5)[0]
        
        # Compute MAP labels via mode of z samples
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
        
        all_probs = []
        for state in self.states_:
            log_w = np.log(np.maximum(state.w, 1e-15))
            log_probs = self.backend_.compute_cluster_log_probs(X, state.mu, log_w)
            
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
        
        p = X.shape[1]
        const = -0.5 * p * np.log(2.0 * np.pi)
        
        log_liks = []
        for state in self.states_:
            log_w = np.log(np.maximum(state.w, 1e-15))
            log_probs = self.backend_.compute_cluster_log_probs(X, state.mu, log_w)
            
            sample_log_lik = log_sum_exp(log_probs, axis=1) + const
            log_liks.append(np.mean(sample_log_lik))
            
        return float(np.mean(log_liks))

    @property
    def n_clusters_(self) -> int:
        """Number of active clusters."""
        check_is_fitted(self, "labels_")
        return len(np.unique(self.labels_))

    @property
    def trace_(self) -> Dict[str, np.ndarray]:
        """Full trace of MCMC samples."""
        check_is_fitted(self, "states_")
        return {
            "z": np.array([state.z for state in self.states_]),
            "w": np.array([state.w for state in self.states_]),
            "mu": np.array([state.mu for state in self.states_]),
            "gamma": np.array([state.gamma for state in self.states_]),
            "theta": np.array([state.theta for state in self.states_]),
        }
