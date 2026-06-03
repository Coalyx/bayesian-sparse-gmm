import numpy as np

def log_sum_exp(x: np.ndarray, axis: int = -1, keepdims: bool = False) -> np.ndarray:
    """Compute log of sum of exponentials in a numerically stable way."""
    x_max = np.max(x, axis=axis, keepdims=True)
    res = x_max + np.log(np.sum(np.exp(x - x_max), axis=axis, keepdims=True))
    if not keepdims:
        res = np.squeeze(res, axis=axis)
    return res

def sample_inverse_gaussian(mean: np.ndarray, shape: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Sample from Inverse Gaussian distribution using Michael et al. (1976)."""
    mean = np.minimum(mean, 1e5)
    y = rng.normal(size=mean.shape) ** 2
    mean2 = mean ** 2
    term = np.sqrt(np.maximum(0.0, 4.0 * mean * shape * y + mean2 * (y ** 2)))
    x1 = mean / (1.0 + (mean * y) / (2.0 * shape) + term / (2.0 * shape))
    u = rng.uniform(size=mean.shape)
    
    cond = mean / (mean + x1 + 1e-15)
    mask = u <= cond
    
    res = np.empty_like(mean)
    res[mask] = x1[mask]
    res[~mask] = mean2[~mask] / (x1[~mask] + 1e-15)
    return res
