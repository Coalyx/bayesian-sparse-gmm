import numpy as np

from bayesian_sparse_gmm.backends._numpy import NumpyBackend


def test_numpy_compute_cluster_log_probs():
    backend = NumpyBackend()
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    mu = np.array([[1.0, 2.0], [0.0, 0.0]])
    log_w = np.array([0.0, -1.0])

    # Expected distance:
    # Row 0 to Mu 0: (1-1)^2 + (2-2)^2 = 0
    # Row 0 to Mu 1: (1-0)^2 + (2-0)^2 = 5
    # Row 1 to Mu 0: (3-1)^2 + (4-2)^2 = 8
    # Row 1 to Mu 1: (3-0)^2 + (4-0)^2 = 25
    # Expected log probs = log_w - 0.5 * dist:
    # Row 0: [0.0 - 0.5*0, -1.0 - 0.5*5] = [0.0, -3.5]
    # Row 1: [0.0 - 0.5*8, -1.0 - 0.5*25] = [-4.0, -13.5]
    expected = np.array([[0.0, -3.5], [-4.0, -13.5]])
    actual = backend.compute_cluster_log_probs(X, mu, log_w, np.ones(2))
    assert np.allclose(actual, expected)


def test_numpy_compute_sufficient_stats():
    backend = NumpyBackend()
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    z = np.array([0, 1, 0])
    K_max = 3

    n_k, sum_x = backend.compute_sufficient_stats(X, z, K_max)
    # Expected count: Cluster 0: 2, Cluster 1: 1, Cluster 2: 0
    assert np.array_equal(n_k, [2, 1, 0])
    # Expected sum_x:
    # Cluster 0: Row 0 + Row 2 = [6.0, 8.0]
    # Cluster 1: Row 1 = [3.0, 4.0]
    # Cluster 2: [0.0, 0.0]
    assert np.allclose(sum_x, [[6.0, 8.0], [3.0, 4.0], [0.0, 0.0]])


def test_numpy_sample_cluster_means():
    backend = NumpyBackend()
    rng = np.random.default_rng(42)
    sum_x = np.array([[10.0, 20.0]])
    n_k = np.array([5])
    tau2 = np.array([[1.0, 0.5]])

    # post_var = 1 / (n_k + 1/tau2) = 1 / (5 + 1) = 1/6 for col 0
    # post_var = 1 / (5 + 2) = 1/7 for col 1
    # post_mean = post_var * sum_x:
    # col 0: 10/6 = 1.6666
    # col 1: 20/7 = 2.8571
    samples = []
    for _ in range(10000):
        samples.append(backend.sample_cluster_means(sum_x, n_k, tau2, np.ones(2), rng))

    samples = np.array(samples)
    sample_mean = np.mean(samples, axis=0)
    assert np.allclose(sample_mean, [[10.0 / 6.0, 20.0 / 7.0]], rtol=0.05)


def test_numpy_sample_inverse_gaussian():
    backend = NumpyBackend()
    rng = np.random.default_rng(42)
    mu_abs = np.array([[2.0, 3.0]])
    lam = np.array([[1.5, 2.5]])

    # Mean of 1/tau^2 is inv_mean = lam / mu_abs
    # Since tau^2 = 1 / inv_tau2, expected mean of 1 / tau2 is inv_mean
    samples = []
    for _ in range(10000):
        tau2 = backend.sample_inverse_gaussian(mu_abs, lam, rng)
        samples.append(1.0 / tau2)

    samples = np.array(samples)
    sample_mean = np.mean(samples, axis=0)
    expected_mean = lam / (mu_abs + 1e-10)
    assert np.allclose(sample_mean, expected_mean, rtol=0.05)
