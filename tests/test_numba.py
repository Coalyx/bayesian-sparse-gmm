import numpy as np
from bayesian_sparse_gmm.backends._numpy import NumpyBackend
from bayesian_sparse_gmm.backends._numba import NumbaBackend
from bayesian_sparse_gmm.model import BayesianSparseGMM

def test_numba_backend_equivalence():
    numpy_backend = NumpyBackend()
    numba_backend = NumbaBackend(use_cuda=False)
    
    # Test data
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    mu = np.array([[1.0, 2.0], [0.0, 0.0]])
    log_w = np.array([0.0, -1.0])
    z = np.array([0, 1, 0])
    K_max = 2
    
    # 1. compute_cluster_log_probs
    res_np = numpy_backend.compute_cluster_log_probs(X, mu, log_w, np.ones(2))
    res_nb = numba_backend.compute_cluster_log_probs(X, mu, log_w, np.ones(2))
    assert np.allclose(res_np, res_nb)
    
    # 2. compute_sufficient_stats
    n_np, sum_np = numpy_backend.compute_sufficient_stats(X, z, K_max)
    n_nb, sum_nb = numba_backend.compute_sufficient_stats(X, z, K_max)
    assert np.array_equal(n_np, n_nb)
    assert np.allclose(sum_np, sum_nb)
    
    # 3. sample_cluster_means
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    tau2 = np.array([[1.0, 0.5], [2.0, 1.5]])
    
    # Pre-seeding guarantees identical random draws
    mu_np = numpy_backend.sample_cluster_means(sum_np, n_np, tau2, np.ones(2), rng1)
    mu_nb = numba_backend.sample_cluster_means(sum_nb, n_nb, tau2, np.ones(2), rng2)
    assert np.allclose(mu_np, mu_nb)
    
    # 4. sample_inverse_gaussian
    rng3 = np.random.default_rng(42)
    rng4 = np.random.default_rng(42)
    mu_abs = np.abs(mu)
    lam = np.array([[1.5, 2.5]])
    
    tau_np = numpy_backend.sample_inverse_gaussian(mu_abs, lam, rng3)
    tau_nb = numba_backend.sample_inverse_gaussian(mu_abs, lam, rng4)
    assert np.allclose(tau_np, tau_nb)

def test_model_with_numba_backend():
    X = np.random.normal(size=(50, 4))
    gmm = BayesianSparseGMM(
        K_max=3,
        n_iter=30,
        burn_in=10,
        thinning=1,
        backend="numba",
        random_state=42
    )
    gmm.fit(X)
    assert gmm.means_.shape == (3, 4)
    assert len(gmm.states_) == 20
