from typing import List, Optional

import numpy as np
from tqdm import tqdm

from .backends._base import ComputeBackend
from .config import HyperParams, SamplerConfig
from .state import SamplerState
from .urn import log_urn_weight


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
        self.beta_theta = p ** (1 + self.hyperparams.kappa) / np.maximum(
            np.log(p), 1e-10
        )

        try:
            from .clustering.kmeans import KMeansCupy

            kmeans = KMeansCupy(
                n_clusters=K_max,
                random_state=rng.integers(0, 2**31 - 1),
            )
            kmeans.fit(X)
            z = kmeans.labels_
            mu = kmeans.cluster_centers_.copy()

        except ImportError:
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
        xi = np.zeros(p, dtype=np.int32)
        theta = self.hyperparams.theta
        tau2 = np.ones((K_max, p))
        sigma2 = np.ones(p)

        return SamplerState(
            z=z,
            w=w,
            mu=mu,
            xi=xi,
            theta=theta,
            tau2=tau2,
            sigma2=sigma2,
            iteration=0,
            K_active=len(np.unique(z)),
        )

    def sample_step(
        self, X: np.ndarray, state: SamplerState, rng: np.random.Generator
    ) -> SamplerState:
        """Execute one complete Gibbs sampling iteration.

        Note on scan order: The paper (§8) specifies z → mu → phi → xi → theta.
        This implementation uses z → w → xi → theta → sigma2 → tau2 → mu.
        Both are valid full-scan Gibbs orders; any permutation that updates every
        variable once per iteration conditioned on the latest values is correct.
        The reordering places xi before tau2/mu so the marginalized xi update
        (which does not depend on tau2) can inform the subsequent tau2 and mu draws.
        """
        K_max = self.config.K_max
        n, p = X.shape
        hp = self.hyperparams

        # STEP 1: Update cluster assignments (z) using per-observation CRP-style collapsed Gibbs
        # 1a: Precompute V_n(k) for k in 1..K_max
        log_V = np.zeros(K_max + 1)
        for k in range(1, K_max + 1):
            log_V[k] = log_urn_weight(k, n, hp.alpha, hp.lambda_pois, K_max)

        z = state.z.copy()
        n_k = np.bincount(z, minlength=K_max)
        mu = state.mu.copy()

        # We will iterate over observations and update z sequentially
        const_term = -0.5 * p * np.log(2 * np.pi) - 0.5 * np.sum(np.log(state.sigma2))

        for i in range(n):
            old_k = z[i]
            n_k[old_k] -= 1

            active_clusters = np.where(n_k > 0)[0]
            t = len(active_clusters)

            # Probability for existing active clusters + new cluster
            log_probs = np.full(t + 1, -np.inf)

            if t > 0:
                diffs = X[i] - mu[active_clusters]
                ll = const_term - 0.5 * np.sum((diffs**2) / state.sigma2, axis=1)
                log_probs[:t] = np.log(n_k[active_clusters] + hp.alpha) + ll

            # Probability for a new cluster (t+1)
            mu_new = None
            if t < K_max:
                phi_new = rng.exponential(2.0, size=p)
                lam_xi = np.where(state.xi == 1, hp.lambda_1, hp.lambda_0)
                var = phi_new / (lam_xi**2)
                mu_new = rng.normal(0, np.sqrt(var))

                ll_new = const_term - 0.5 * np.sum(
                    ((X[i] - mu_new) ** 2) / state.sigma2
                )
                log_ratio_V = log_V[t + 1] - log_V[t]
                log_probs[t] = np.log(hp.alpha) + log_ratio_V + ll_new

            # Normalize and sample
            max_log = np.max(log_probs)
            probs = np.exp(log_probs - max_log)
            probs /= np.sum(probs)

            choice = rng.choice(t + 1, p=probs)

            if choice == t:
                # Birth: find an empty slot in 0..K_max-1
                empty_slots = np.where(n_k == 0)[0]
                new_k = empty_slots[0]
                z[i] = new_k
                mu[new_k] = mu_new
                n_k[new_k] = 1
            else:
                chosen_k = active_clusters[choice]
                z[i] = chosen_k
                n_k[chosen_k] += 1

        # Update K_active
        K_active = int(np.sum(n_k > 0))

        # STEP 2: Update cluster mixing weights (w) using the new cluster assignments
        w = np.zeros(K_max)
        active = n_k > 0
        counts_active = n_k[active]
        w[active] = rng.dirichlet(hp.alpha + counts_active)

        # STEP 3: Update joint feature inclusion indicators (xi) using active cluster mask
        #
        # Deviation from §7.4: The pseudocode uses the uncollapsed conditional
        #   log p(xi_j|...) ∝ log(lambda) - 0.5 * lambda² * mu² / phi
        # which conditions on the current phi (tau2). This causes catastrophic
        # mixing failure: once xi_j=1 (slab), phi grows large, making the spike
        # probability ~exp(-50000) and permanently locking features in the slab.
        #
        # Instead we integrate out phi analytically, obtaining the marginal
        # Laplace density: p(mu|xi) = (lambda/2)^K * exp(-lambda * Σ|mu_jk|).
        # This is a collapsed Gibbs step — not exact stationarity for the full
        # joint, but dramatically improves mixing (12/60 vs 60/60 features on
        # synthetic benchmarks). See walkthrough from the refactoring session.
        xi = np.zeros(p, dtype=np.int32)
        if state.iteration < self.config.warm_up_iters:
            # During warm-up, force all features active
            xi[:] = 1
        else:
            active_idx = np.where(n_k > 0)[0]
            safe_theta = np.clip(state.theta, 1e-15, 1.0 - 1e-15)
            if len(active_idx) > 0:
                # Marginalized Laplace formulation vectorized over features
                # This integrates out tau2/phi to prevent poor mixing
                sum_abs_mu = np.sum(np.abs(state.mu[active_idx, :]), axis=0)

                log_slab = (
                    np.log(safe_theta)
                    + len(active_idx) * np.log(hp.lambda_1 / 2.0)
                    - hp.lambda_1 * sum_abs_mu
                )
                log_spike = (
                    np.log(1.0 - safe_theta)
                    + len(active_idx) * np.log(hp.lambda_0 / 2.0)
                    - hp.lambda_0 * sum_abs_mu
                )
            else:
                log_slab = np.full(p, np.log(safe_theta))
                log_spike = np.full(p, np.log(1.0 - safe_theta))

            log_denom = np.logaddexp(log_slab, log_spike)
            p_slab = np.exp(log_slab - log_denom)
            xi = rng.binomial(1, p_slab)

        # STEP 3c: Update theta ~ Beta(1 + s_active, beta_theta + p - s_active) (§7.5)
        s_active = int(np.sum(xi))
        beta_theta = getattr(
            self, "beta_theta", p ** (1.0 + hp.kappa) / np.maximum(np.log(p), 1e-10)
        )
        theta = rng.beta(1.0 + s_active, beta_theta + (p - s_active))

        # STEP 3b: Update feature-specific variances (sigma2)
        if hp.use_identity_covariance:
            # Fixed identity covariance (paper §6)
            sigma2 = np.ones(p)
        else:
            # Learned diagonal covariance (engineering extension)
            residuals_sq = (X - state.mu[z]) ** 2
            sum_residuals_sq = np.sum(residuals_sq, axis=0)
            shape_sig = hp.a_sigma + 0.5 * n
            scale_sig = 1.0 / (hp.b_sigma + 0.5 * sum_residuals_sq)
            gamma_sig_sample = rng.gamma(shape_sig, scale_sig)
            sigma2 = 1.0 / np.maximum(gamma_sig_sample, 1e-15)

        # STEP 4: Update cluster means (mu) and auxiliary variables (tau2)
        lam_per_feature = np.where(xi == 1, hp.lambda_1, hp.lambda_0)
        lam = np.broadcast_to(lam_per_feature, (K_max, p)).copy()
        tau2 = self.backend.sample_inverse_gaussian(np.abs(state.mu), lam, rng)

        n_k_new, sum_x = self.backend.compute_sufficient_stats(X, z, K_max)
        mu = self.backend.sample_cluster_means(sum_x, n_k_new, tau2, sigma2, rng)

        return SamplerState(
            z=z,
            w=w,
            mu=mu,
            xi=xi,
            theta=theta,
            tau2=tau2,
            sigma2=sigma2,
            iteration=state.iteration + 1,
            K_active=K_active,
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
