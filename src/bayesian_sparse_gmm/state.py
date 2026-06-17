from dataclasses import dataclass

import numpy as np


@dataclass
class VariationalState:
    """Variational parameters for the Bayesian Sparse GMM SVI optimizer."""

    expected_mu: np.ndarray
    expected_sigma2: np.ndarray
    expected_w: np.ndarray
    pi_xi: np.ndarray
    expected_theta: float

    dirichlet_alpha: np.ndarray
    beta_a: float
    beta_b: float
    global_N_k: np.ndarray
    global_sum_x: np.ndarray

    epoch: int
    iteration: int
    K_active: int = 0


@dataclass
class SamplerState:
    """MCMC sampler state at a single iteration."""

    z: np.ndarray
    w: np.ndarray
    mu: np.ndarray
    xi: np.ndarray
    theta: float
    tau2: np.ndarray
    sigma2: np.ndarray
    iteration: int
    K_active: int = 0
