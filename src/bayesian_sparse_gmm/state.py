from dataclasses import dataclass

import numpy as np


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
