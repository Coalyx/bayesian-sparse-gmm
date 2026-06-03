import pytest
import numpy as np
from bayesian_sparse_gmm.backends import select_backend
from bayesian_sparse_gmm.backends._cuda import CUPY_AVAILABLE, CUDABackend
from bayesian_sparse_gmm.backends._numpy import NumpyBackend

def test_cuda_backend_availability_and_fallback():
    if not CUPY_AVAILABLE:
        # Check that instantiating CUDABackend directly raises ImportError
        with pytest.raises(ImportError):
            CUDABackend()
            
        # Check that select_backend("cuda") falls back gracefully (either to Numba or NumPy)
        backend = select_backend("cuda")
        assert isinstance(backend, NumpyBackend) or hasattr(backend, "use_cuda")
    else:
        # If CuPy is installed, run equivalence tests
        cuda_backend = CUDABackend()
        numpy_backend = NumpyBackend()
        
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        mu = np.array([[1.0, 2.0], [0.0, 0.0]])
        log_w = np.array([0.0, -1.0])
        z = np.array([0, 1, 0])
        K_max = 2
        
        # 1. compute_cluster_log_probs
        res_np = numpy_backend.compute_cluster_log_probs(X, mu, log_w)
        res_cu = cuda_backend.compute_cluster_log_probs(X, mu, log_w)
        assert np.allclose(res_np, res_cu)
        
        # 2. compute_sufficient_stats
        n_np, sum_np = numpy_backend.compute_sufficient_stats(X, z, K_max)
        n_cu, sum_cu = cuda_backend.compute_sufficient_stats(X, z, K_max)
        assert np.array_equal(n_np, n_cu)
        assert np.allclose(sum_np, sum_cu)
