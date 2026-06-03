import numpy as np
from typing import Optional, List
from tqdm import tqdm
from .config import SamplerConfig, HyperParams
from .state import SamplerState
from .backends._base import ComputeBackend

class GibbsSampler:
    """Gibbs Sampler orchestrator for Bayesian Sparse GMM."""

    def __init__(self, config: SamplerConfig, hyperparams: HyperParams, backend: ComputeBackend):
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
                init='k-means++', 
                n_init=1, 
                random_state=rng.integers(0, 2**31 - 1)
            )
            kmeans.fit(X)
            z = kmeans.labels_
            mu = kmeans.cluster_centers_.copy()
            
            # Shrink cluster centers to avoid the initialization vicious cycle.
            sum_abs_mu = np.sum(np.abs(mu), axis=0)
            S_init_mean = np.mean(sum_abs_mu)
            target_S = 0.5
            if S_init_mean > target_S:
                mu = mu * (target_S / S_init_mean)

        except Exception:
            z = rng.choice(K_max, size=n)
            mu = rng.normal(size=(K_max, p))
            
        w = np.ones(K_max) / K_max
        gamma = rng.binomial(1, 0.1, size=p).astype(np.int32)
        theta = 0.1
        tau2 = np.ones((K_max, p))
        
        return SamplerState(
            z=z, w=w, mu=mu, gamma=gamma, theta=theta, tau2=tau2, iteration=0
        )

    def sample_step(self, X: np.ndarray, state: SamplerState, rng: np.random.Generator) -> SamplerState:
        """Execute one complete Gibbs sampling iteration."""
        K_max = self.config.K_max
        n, p = X.shape
        hp = self.hyperparams
        
        # STEP 1: Update cluster mixing weights (w)
        n_k = np.bincount(state.z, minlength=K_max)
        w = rng.dirichlet(hp.alpha + n_k)
        
        # STEP-2: Update cluster assignments (z)
        log_w = np.log(np.maximum(w, 1e-15))
        log_probs = self.backend.compute_cluster_log_probs(X, state.mu, log_w)
        max_log = np.max(log_probs, axis=1, keepdims=True)
        probs = np.exp(log_probs - max_log)
        probs /= np.sum(probs, axis=1, keepdims=True)
        
        # Vectorized categorical sampling
        cumsum = np.cumsum(probs, axis=1)
        u = rng.uniform(size=(n, 1))
        z = np.sum(cumsum < u, axis=1)
        z = np.clip(z, 0, K_max - 1)
        
        # STEP 3: Update feature inclusion indicators (gamma)
        sum_abs_mu = np.sum(np.abs(state.mu), axis=0)
        
        log_laplace_slab = K_max * (np.log(hp.lambda_1) - np.log(2.0)) - hp.lambda_1 * sum_abs_mu
        log_laplace_spike = K_max * (np.log(hp.lambda_0) - np.log(2.0)) - hp.lambda_0 * sum_abs_mu
        
        safe_theta = np.clip(state.theta, 1e-15, 1.0 - 1e-15)
        log_P_slab = np.log(safe_theta) + log_laplace_slab
        log_P_spike = np.log(1.0 - safe_theta) + log_laplace_spike
        
        max_log = np.maximum(log_P_slab, log_P_spike)
        prob_slab = np.exp(log_P_slab - max_log)
        prob_spike = np.exp(log_P_spike - max_log)
        
        p_slab = prob_slab / (prob_slab + prob_spike)
        gamma = rng.binomial(1, p_slab)
        
        # STEP 4: Update cluster means (mu) and auxiliary variables (tau2)
        lam = np.where(gamma == 1, hp.lambda_1, hp.lambda_0)[np.newaxis, :]
        tau2 = self.backend.sample_inverse_gaussian(np.abs(state.mu), lam, rng)
        
        n_k_new, sum_x = self.backend.compute_sufficient_stats(X, z, K_max)
        mu = self.backend.sample_cluster_means(sum_x, n_k_new, tau2, rng)
        
        # STEP 5: Update sparsity probability (theta)
        num_active = np.sum(gamma)
        theta = rng.beta(hp.a + num_active, hp.b + p - num_active)
        
        return SamplerState(
            z=z,
            w=w,
            mu=mu,
            gamma=gamma,
            theta=theta,
            tau2=tau2,
            iteration=state.iteration + 1
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
