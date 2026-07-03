import numpy as np

from ._base import ComputeBackend

try:
    import cupy as cp

    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None


class CUDABackend(ComputeBackend):
    """CuPy-based GPU accelerated compute backend."""

    def __init__(self):
        if not CUPY_AVAILABLE:
            raise ImportError(
                "CuPy is not installed. Install with: pip install cupy-cuda12x"
            )

    def _get_optimal_chunk_size(
        self, num_features: int, num_clusters: int, dtype
    ) -> int:
        """Dynamically compute optimal chunk size based on available VRAM."""
        try:
            free_mem, _ = cp.cuda.Device().mem_info
            # Use 50% of available free memory for safety
            usable_mem = free_mem * 0.5

            bytes_per_element = np.dtype(dtype).itemsize
            # Max elements per sample across operations is roughly D + 2*K + 5
            elements_per_sample = num_features + 2 * num_clusters + 5
            bytes_per_sample = elements_per_sample * bytes_per_element

            if bytes_per_sample == 0:
                return 16384

            chunk_size = int(usable_mem / bytes_per_sample)
            return max(1024, min(chunk_size, 1024 * 1024))
        except Exception:
            return 16384

    def compute_cluster_log_probs(
        self, X: np.ndarray, mu: np.ndarray, log_w: np.ndarray, sigma2: np.ndarray
    ) -> np.ndarray:
        """Compute diagonal-Mahalanobis log-probabilities on GPU."""
        CHUNK_SIZE = self._get_optimal_chunk_size(X.shape[1], mu.shape[0], X.dtype)
        n_samples = X.shape[0]

        if n_samples <= CHUNK_SIZE:
            X_gpu = cp.asarray(X)
            mu_gpu = cp.asarray(mu)
            log_w_gpu = cp.asarray(log_w)
            sigma2_gpu = cp.asarray(sigma2)

            std = cp.sqrt(sigma2_gpu)
            X_scaled = X_gpu / std
            mu_scaled = mu_gpu / std

            x_sq = cp.sum(X_scaled**2, axis=1, keepdims=True)
            mu_sq = cp.sum(mu_scaled**2, axis=1, keepdims=True).T
            dist = x_sq - 2.0 * cp.dot(X_scaled, mu_scaled.T) + mu_sq
            dist = cp.maximum(dist, 0.0)

            return cp.asnumpy(log_w_gpu - 0.5 * dist)
        else:
            mu_gpu = cp.asarray(mu)
            log_w_gpu = cp.asarray(log_w)
            sigma2_gpu = cp.asarray(sigma2)
            std = cp.sqrt(sigma2_gpu)
            mu_scaled = mu_gpu / std
            mu_sq = cp.sum(mu_scaled**2, axis=1, keepdims=True).T

            result = np.empty((n_samples, mu.shape[0]), dtype=X.dtype)

            for i in range(0, n_samples, CHUNK_SIZE):
                X_chunk = cp.asarray(X[i : i + CHUNK_SIZE])
                X_scaled = X_chunk / std
                x_sq = cp.sum(X_scaled**2, axis=1, keepdims=True)
                dist = x_sq - 2.0 * cp.dot(X_scaled, mu_scaled.T) + mu_sq
                dist = cp.maximum(dist, 0.0)
                result[i : i + CHUNK_SIZE] = cp.asnumpy(log_w_gpu - 0.5 * dist)

            return result

    def compute_expected_sufficient_stats(
        self, X: np.ndarray, r_ik: np.ndarray, K_max: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute expected sufficient statistics on GPU."""
        CHUNK_SIZE = self._get_optimal_chunk_size(X.shape[1], K_max, X.dtype)
        n_samples = X.shape[0]

        if n_samples <= CHUNK_SIZE:
            X_gpu = cp.asarray(X)
            r_ik_gpu = cp.asarray(r_ik)

            expected_n_k = cp.sum(r_ik_gpu, axis=0)
            expected_sum_x = cp.dot(r_ik_gpu.T, X_gpu)

            return cp.asnumpy(expected_n_k), cp.asnumpy(expected_sum_x)
        else:
            expected_n_k = cp.zeros(K_max, dtype=r_ik.dtype)
            expected_sum_x = cp.zeros((K_max, X.shape[1]), dtype=X.dtype)

            for i in range(0, n_samples, CHUNK_SIZE):
                X_chunk = cp.asarray(X[i : i + CHUNK_SIZE])
                r_ik_chunk = cp.asarray(r_ik[i : i + CHUNK_SIZE])

                expected_n_k += cp.sum(r_ik_chunk, axis=0)
                expected_sum_x += cp.dot(r_ik_chunk.T, X_chunk)

            return cp.asnumpy(expected_n_k), cp.asnumpy(expected_sum_x)

    def compute_sufficient_stats(
        self, X: np.ndarray, z: np.ndarray, K_max: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute cluster sizes and feature sums per cluster on GPU."""
        CHUNK_SIZE = self._get_optimal_chunk_size(X.shape[1], K_max, X.dtype)
        n_samples = X.shape[0]

        if n_samples <= CHUNK_SIZE:
            X_gpu = cp.asarray(X)
            z_gpu = cp.asarray(z)

            n_k = cp.bincount(z_gpu, minlength=K_max)
            sum_x = cp.zeros((K_max, X.shape[1]), dtype=X_gpu.dtype)
            cp.add.at(sum_x, z_gpu, X_gpu)

            return cp.asnumpy(n_k), cp.asnumpy(sum_x)
        else:
            n_k = cp.zeros(K_max, dtype=cp.int64)
            sum_x = cp.zeros((K_max, X.shape[1]), dtype=X.dtype)

            for i in range(0, n_samples, CHUNK_SIZE):
                X_chunk = cp.asarray(X[i : i + CHUNK_SIZE])
                z_chunk = cp.asarray(z[i : i + CHUNK_SIZE])

                n_k += cp.bincount(z_chunk, minlength=K_max)
                cp.add.at(sum_x, z_chunk, X_chunk)

            return cp.asnumpy(n_k), cp.asnumpy(sum_x)

    def sample_cluster_means(
        self,
        sum_x: np.ndarray,
        n_k: np.ndarray,
        tau2: np.ndarray,
        sigma2: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Sample cluster means mu[k,j] ~ Normal on GPU."""
        sum_x_gpu = cp.asarray(sum_x)
        n_k_gpu = cp.asarray(n_k)
        tau2_gpu = cp.asarray(tau2)
        sigma2_gpu = cp.asarray(sigma2)

        noise = rng.normal(size=sum_x.shape)
        noise_gpu = cp.asarray(noise)

        post_var = 1.0 / (n_k_gpu[:, cp.newaxis] / sigma2_gpu + 1.0 / tau2_gpu)
        post_mean = post_var * (sum_x_gpu / sigma2_gpu)
        mu_gpu = post_mean + cp.sqrt(post_var) * noise_gpu

        return cp.asnumpy(mu_gpu)

    def sample_inverse_gaussian(
        self, mu_abs: np.ndarray, lam: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        """Sample tau^2 via Inverse Gaussian on GPU."""
        mu_abs_gpu = cp.asarray(mu_abs)
        lam_gpu = cp.asarray(lam)

        y_noise = rng.normal(size=mu_abs.shape) ** 2
        u_noise = rng.uniform(size=mu_abs.shape)

        y_gpu = cp.asarray(y_noise)
        u_gpu = cp.asarray(u_noise)

        inv_mean = cp.minimum(lam_gpu / (mu_abs_gpu + 1e-10), 1e5)
        shape = lam_gpu**2
        inv_mean2 = inv_mean**2

        term = cp.sqrt(
            cp.maximum(0.0, 4.0 * inv_mean * shape * y_gpu + inv_mean2 * (y_gpu**2))
        )
        x1 = inv_mean / (
            1.0 + (inv_mean * y_gpu) / (2.0 * shape) + term / (2.0 * shape)
        )

        cond = inv_mean / (inv_mean + x1 + 1e-15)
        mask = u_gpu <= cond

        res = cp.where(mask, x1, inv_mean2 / (x1 + 1e-15))
        return cp.asnumpy(1.0 / res)
