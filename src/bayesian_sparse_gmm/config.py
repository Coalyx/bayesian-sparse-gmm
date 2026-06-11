from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class HyperParams:
    """Hyperparameters for the Bayesian GMM."""

    lambda_0: float = 100.0  # Spike rate (lambda_0 >> lambda_1 for sparsity)
    lambda_1: float = 1.0  # Slab rate
    alpha: float = 1.0  # Dirichlet concentration (alpha >= 1.0)
    theta: float = 0.5  # Prior probability of informative features
    kappa: float = 0.1  # Sparsity aggressiveness for beta_theta
    lambda_pois: float = 2.0  # Truncated Poisson rate for K prior
    a_sigma: float = 1.0
    b_sigma: float = 1.0


@dataclass(frozen=True)
class SamplerConfig:
    """Configuration options for the Gibbs sampler."""

    K_max: int = 20  # Conservative cluster count upper bound
    n_iter: int = 5000  # Total iterations (burn_in + post-burn_in)
    burn_in: int = 1000  # Burn-in iterations to discard
    thinning: int = 1
    warm_up_iters: int = 50
    backend: str = "auto"
    n_jobs: int = -1
    random_state: Optional[int] = None
    verbose: int = 0
