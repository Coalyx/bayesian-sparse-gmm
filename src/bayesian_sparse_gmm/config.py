from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class HyperParams:
    """Hyperparameters for the Bayesian GMM."""
    lambda_0: float = 100.0
    lambda_1: float = 0.1
    alpha: float = 1.0
    a: float = 1.0
    b: float = 1.0

@dataclass(frozen=True)
class SamplerConfig:
    """Configuration options for the Gibbs sampler."""
    K_max: int = 15
    n_iter: int = 2000
    burn_in: int = 500
    thinning: int = 1
    backend: str = "auto"
    n_jobs: int = -1
    random_state: Optional[int] = None
    verbose: int = 0
