import numpy as np
import pytest

from bayesian_sparse_gmm.backends import select_backend
from bayesian_sparse_gmm.backends._cuda import CUPY_AVAILABLE, CUDABackend
from bayesian_sparse_gmm.backends._numpy import NumpyBackend

try:
    from bayesian_sparse_gmm.backends._numba import NumbaBackend, _probe_numba_cuda

    NUMBA_AVAILABLE = True
    NUMBA_CUDA_AVAILABLE = _probe_numba_cuda()
except ImportError:
    NUMBA_AVAILABLE = False
    NUMBA_CUDA_AVAILABLE = False


# --- Fixtures ---


@pytest.fixture
def sample_data():
    """Minimal test data for backend equivalence checks."""
    rng = np.random.default_rng(42)
    X = rng.normal(size=(10, 4)).astype(np.float64)
    mu = rng.normal(size=(3, 4)).astype(np.float64)
    log_w = np.log(np.array([0.5, 0.3, 0.2]))
    sigma2 = np.abs(rng.normal(size=4)) + 0.1
    z = rng.choice(3, size=10)
    return X, mu, log_w, sigma2, z


# --- select_backend routing ---


def test_cuda_preference_selects_gpu_backend():
    """'cuda' should resolve to a GPU backend or fall back gracefully."""
    backend = select_backend("cuda")
    if CUPY_AVAILABLE:
        assert isinstance(backend, CUDABackend)
    elif NUMBA_AVAILABLE and NUMBA_CUDA_AVAILABLE:
        assert isinstance(backend, NumbaBackend)
        assert backend.use_cuda is True
    elif NUMBA_AVAILABLE:
        # Numba available but CUDA probe failed — falls back to CPU Numba
        assert isinstance(backend, NumbaBackend)
        assert backend.use_cuda is False
    else:
        assert isinstance(backend, NumpyBackend)


def test_cupy_import_error_when_unavailable():
    if not CUPY_AVAILABLE:
        with pytest.raises(ImportError):
            CUDABackend()


# --- CuPy backend equivalence (only when CuPy installed) ---


@pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not installed")
class TestCUDABackendEquivalence:
    """Verify CuPy backend produces identical results to NumPy backend."""

    def test_compute_cluster_log_probs(self, sample_data):
        X, mu, log_w, sigma2, _ = sample_data
        ref = NumpyBackend().compute_cluster_log_probs(X, mu, log_w, sigma2)
        got = CUDABackend().compute_cluster_log_probs(X, mu, log_w, sigma2)
        np.testing.assert_allclose(got, ref, rtol=1e-6)

    def test_compute_sufficient_stats(self, sample_data):
        X, _, _, _, z = sample_data
        n_ref, s_ref = NumpyBackend().compute_sufficient_stats(X, z, 3)
        n_got, s_got = CUDABackend().compute_sufficient_stats(X, z, 3)
        np.testing.assert_array_equal(n_got, n_ref)
        np.testing.assert_allclose(s_got, s_ref, rtol=1e-6)

    def test_sample_cluster_means(self, sample_data):
        X, mu, _, sigma2, _ = sample_data
        tau2 = np.abs(mu) + 0.1
        n_k = np.array([4, 3, 3])
        sum_x = np.random.default_rng(0).normal(size=(3, 4))

        rng1 = np.random.default_rng(99)
        rng2 = np.random.default_rng(99)
        ref = NumpyBackend().sample_cluster_means(sum_x, n_k, tau2, sigma2, rng1)
        got = CUDABackend().sample_cluster_means(sum_x, n_k, tau2, sigma2, rng2)
        np.testing.assert_allclose(got, ref, rtol=1e-6)


# --- Numba CUDA backend equivalence (only when Numba CUDA available) ---


@pytest.mark.skipif(not NUMBA_CUDA_AVAILABLE, reason="Numba CUDA not available")
class TestNumbaCUDABackendEquivalence:
    """Verify Numba CUDA backend produces identical results to Numba CPU backend."""

    def test_compute_cluster_log_probs(self, sample_data):
        X, mu, log_w, sigma2, _ = sample_data
        cpu = NumbaBackend(use_cuda=False)
        gpu = NumbaBackend(use_cuda=True)

        ref = cpu.compute_cluster_log_probs(X, mu, log_w, sigma2)
        got = gpu.compute_cluster_log_probs(X, mu, log_w, sigma2)
        np.testing.assert_allclose(got, ref, rtol=1e-5)

    def test_compute_sufficient_stats(self, sample_data):
        X, _, _, _, z = sample_data
        cpu = NumbaBackend(use_cuda=False)
        gpu = NumbaBackend(use_cuda=True)

        n_ref, s_ref = cpu.compute_sufficient_stats(X, z, 3)
        n_got, s_got = gpu.compute_sufficient_stats(X, z, 3)
        np.testing.assert_array_equal(n_got, n_ref)
        np.testing.assert_allclose(s_got, s_ref, rtol=1e-5)

    def test_sample_cluster_means(self, sample_data):
        X, mu, _, sigma2, _ = sample_data
        tau2 = np.abs(mu) + 0.1
        n_k = np.array([4, 3, 3])
        sum_x = np.random.default_rng(0).normal(size=(3, 4))
        cpu = NumbaBackend(use_cuda=False)
        gpu = NumbaBackend(use_cuda=True)

        rng1 = np.random.default_rng(99)
        rng2 = np.random.default_rng(99)
        ref = cpu.sample_cluster_means(sum_x, n_k, tau2, sigma2, rng1)
        got = gpu.sample_cluster_means(sum_x, n_k, tau2, sigma2, rng2)
        np.testing.assert_allclose(got, ref, rtol=1e-5)
