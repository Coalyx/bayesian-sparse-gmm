import numpy as np
import warnings
from numba import njit, prange, cuda
from ._base import ComputeBackend

# =====================================================================
# Numba CPU parallel kernels
# =====================================================================

@njit(parallel=True, fastmath=True, cache=True)
def _compute_cluster_log_probs_numba(X, mu, log_w):
    n, p = X.shape
    K_max = mu.shape[0]
    log_probs = np.empty((n, K_max), dtype=X.dtype)
    for i in prange(n):
        for k in range(K_max):
            dist = 0.0
            for j in range(p):
                diff = X[i, j] - mu[k, j]
                dist += diff * diff
            log_probs[i, k] = log_w[k] - 0.5 * dist
    return log_probs

@njit(parallel=True, fastmath=True, cache=True)
def _compute_sufficient_stats_numba(X, z, K_max):
    n, p = X.shape
    n_k = np.zeros(K_max, dtype=np.int64)
    for i in range(n):
        n_k[z[i]] += 1
        
    sum_x = np.zeros((K_max, p), dtype=X.dtype)
    for j in prange(p):
        for i in range(n):
            k = z[i]
            sum_x[k, j] += X[i, j]
            
    return n_k, sum_x

@njit(parallel=True, fastmath=True, cache=True)
def _sample_cluster_means_numba(sum_x, n_k, tau2, noise):
    K_max, p = sum_x.shape
    mu = np.empty((K_max, p), dtype=sum_x.dtype)
    for k in prange(K_max):
        nk = n_k[k]
        for j in range(p):
            post_var = 1.0 / (nk + 1.0 / tau2[k, j])
            post_mean = post_var * sum_x[k, j]
            mu[k, j] = post_mean + np.sqrt(post_var) * noise[k, j]
    return mu

@njit(parallel=True, fastmath=True, cache=True)
def _sample_inverse_gaussian_numba(mu_abs, lam, y_noise, u_noise):
    K_max, p = mu_abs.shape
    tau2 = np.empty((K_max, p), dtype=mu_abs.dtype)
    for k in prange(K_max):
        for j in range(p):
            mean = lam[0, j] / (mu_abs[k, j] + 1e-10)
            if mean > 1e5:
                mean = 1e5
            sh = lam[0, j] ** 2
            mean2 = mean ** 2
            y = y_noise[k, j]
            term = np.sqrt(np.maximum(0.0, 4.0 * mean * sh * y + mean2 * (y ** 2)))
            x1 = mean / (1.0 + (mean * y) / (2.0 * sh) + term / (2.0 * sh))
            u = u_noise[k, j]
            
            cond = mean / (mean + x1 + 1e-15)
            if u <= cond:
                inv_tau2 = x1
            else:
                inv_tau2 = mean2 / (x1 + 1e-15)
            tau2[k, j] = 1.0 / inv_tau2
    return tau2

# =====================================================================
# Numba CUDA GPU kernels
# =====================================================================

@cuda.jit
def _compute_cluster_log_probs_cuda(X, mu, log_w, out):
    i = cuda.grid(1)
    n = X.shape[0]
    K_max = mu.shape[0]
    p = X.shape[1]
    if i < n:
        for k in range(K_max):
            dist = 0.0
            for j in range(p):
                diff = X[i, j] - mu[k, j]
                dist += diff * diff
            out[i, k] = log_w[k] - 0.5 * dist

@cuda.jit
def _compute_sufficient_stats_cuda(X, z, n_k, sum_x):
    i = cuda.grid(1)
    n = X.shape[0]
    p = X.shape[1]
    if i < n:
        k = z[i]
        cuda.atomic.add(n_k, k, 1)
        for j in range(p):
            cuda.atomic.add(sum_x, (k, j), X[i, j])

@cuda.jit
def _sample_cluster_means_cuda(sum_x, n_k, tau2, noise, out):
    k = cuda.grid(1)
    K_max = sum_x.shape[0]
    p = sum_x.shape[1]
    if k < K_max:
        nk = n_k[k]
        for j in range(p):
            post_var = 1.0 / (nk + 1.0 / tau2[k, j])
            post_mean = post_var * sum_x[k, j]
            out[k, j] = post_mean + np.sqrt(post_var) * noise[k, j]

@cuda.jit
def _sample_inverse_gaussian_cuda(mu_abs, lam, y_noise, u_noise, out_tau2):
    k = cuda.grid(1)
    K_max = mu_abs.shape[0]
    p = mu_abs.shape[1]
    if k < K_max:
        for j in range(p):
            mean = lam[0, j] / (mu_abs[k, j] + 1e-10)
            if mean > 1e5:
                mean = 1e5
            sh = lam[0, j] ** 2
            mean2 = mean ** 2
            y = y_noise[k, j]
            term = np.sqrt(np.maximum(0.0, 4.0 * mean * sh * y + mean2 * (y ** 2)))
            x1 = mean / (1.0 + (mean * y) / (2.0 * sh) + term / (2.0 * sh))
            u = u_noise[k, j]
            
            cond = mean / (mean + x1 + 1e-15)
            if u <= cond:
                inv_tau2 = x1
            else:
                inv_tau2 = mean2 / (x1 + 1e-15)
            out_tau2[k, j] = 1.0 / inv_tau2

# =====================================================================
# Numba backend class wrapper
# =====================================================================

class NumbaBackend(ComputeBackend):
    """Numba-accelerated compute backend supporting CPU parallel and GPU CUDA."""

    def __init__(self, use_cuda: bool = False):
        self.use_cuda = use_cuda
        if use_cuda:
            if not cuda.is_available():
                warnings.warn("CUDA is requested but Numba CUDA is not available. Falling back to CPU multi-core.")
                self.use_cuda = False

    def compute_cluster_log_probs(
        self, X: np.ndarray, mu: np.ndarray, log_w: np.ndarray
    ) -> np.ndarray:
        if self.use_cuda:
            n, p = X.shape
            K_max = mu.shape[0]
            out = np.empty((n, K_max), dtype=X.dtype)
            
            d_X = cuda.to_device(X)
            d_mu = cuda.to_device(mu)
            d_log_w = cuda.to_device(log_w)
            d_out = cuda.to_device(out)
            
            threads_per_block = 256
            blocks_per_grid = (n + threads_per_block - 1) // threads_per_block
            _compute_cluster_log_probs_cuda[blocks_per_grid, threads_per_block](d_X, d_mu, d_log_w, d_out)
            
            d_out.copy_to_host(out)
            return out
        else:
            return _compute_cluster_log_probs_numba(X, mu, log_w)

    def compute_sufficient_stats(
        self, X: np.ndarray, z: np.ndarray, K_max: int
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.use_cuda:
            n, p = X.shape
            n_k = np.zeros(K_max, dtype=np.int64)
            sum_x = np.zeros((K_max, p), dtype=X.dtype)
            
            d_X = cuda.to_device(X)
            d_z = cuda.to_device(z)
            d_n_k = cuda.to_device(n_k)
            d_sum_x = cuda.to_device(sum_x)
            
            threads_per_block = 256
            blocks_per_grid = (n + threads_per_block - 1) // threads_per_block
            _compute_sufficient_stats_cuda[blocks_per_grid, threads_per_block](d_X, d_z, d_n_k, d_sum_x)
            
            d_n_k.copy_to_host(n_k)
            d_sum_x.copy_to_host(sum_x)
            return n_k, sum_x
        else:
            return _compute_sufficient_stats_numba(X, z, K_max)

    def sample_cluster_means(
        self, sum_x: np.ndarray, n_k: np.ndarray,
        tau2: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        noise = rng.normal(size=sum_x.shape)
        if self.use_cuda:
            K_max, p = sum_x.shape
            out = np.empty((K_max, p), dtype=sum_x.dtype)
            
            d_sum_x = cuda.to_device(sum_x)
            d_n_k = cuda.to_device(n_k)
            d_tau2 = cuda.to_device(tau2)
            d_noise = cuda.to_device(noise)
            d_out = cuda.to_device(out)
            
            threads_per_block = 64
            blocks_per_grid = (K_max + threads_per_block - 1) // threads_per_block
            _sample_cluster_means_cuda[blocks_per_grid, threads_per_block](
                d_sum_x, d_n_k, d_tau2, d_noise, d_out
            )
            
            d_out.copy_to_host(out)
            return out
        else:
            return _sample_cluster_means_numba(sum_x, n_k, tau2, noise)

    def sample_inverse_gaussian(
        self, mu_abs: np.ndarray, lam: np.ndarray,
        rng: np.random.Generator
    ) -> np.ndarray:
        y_noise = rng.normal(size=mu_abs.shape) ** 2
        u_noise = rng.uniform(size=mu_abs.shape)
        if self.use_cuda:
            K_max, p = mu_abs.shape
            out_tau2 = np.empty((K_max, p), dtype=mu_abs.dtype)
            
            d_mu_abs = cuda.to_device(mu_abs)
            d_lam = cuda.to_device(lam)
            d_y_noise = cuda.to_device(y_noise)
            d_u_noise = cuda.to_device(u_noise)
            d_out_tau2 = cuda.to_device(out_tau2)
            
            threads_per_block = 64
            blocks_per_grid = (K_max + threads_per_block - 1) // threads_per_block
            _sample_inverse_gaussian_cuda[blocks_per_grid, threads_per_block](
                d_mu_abs, d_lam, d_y_noise, d_u_noise, d_out_tau2
            )
            
            d_out_tau2.copy_to_host(out_tau2)
            return out_tau2
        else:
            return _sample_inverse_gaussian_numba(mu_abs, lam, y_noise, u_noise)
