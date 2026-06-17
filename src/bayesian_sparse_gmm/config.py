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
    use_identity_covariance: bool = True  # paper §6 default: fixed I_p


@dataclass(frozen=True)
class SVIConfig:
    """Configuration options for Stochastic Variational Inference (SVI)."""

    K_max: int = 20  # Conservative cluster count upper bound
    epochs: int = 100  # Total passes over the dataset
    batch_size: int = 256  # Mini-batch size
    delay_rho: float = 1.0  # Learning rate delay parameter (tau_0)
    forgetting_rate: float = 0.75  # Learning rate forgetting parameter (kappa)
    backend: str = "auto"
    n_jobs: int = -1
    random_state: Optional[int] = None
    verbose: int = 0


@dataclass(frozen=True)
class SamplerConfig:
    """Configuration options for the Gibbs sampler (MCMC)."""

    K_max: int = 20
    n_iter: int = 5000
    burn_in: int = 1000
    thinning: int = 1
    warm_up_iters: int = 50
    backend: str = "auto"
    n_jobs: int = -1
    random_state: Optional[int] = None
    verbose: int = 0
