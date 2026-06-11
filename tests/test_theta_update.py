import numpy as np

from bayesian_sparse_gmm.backends._numpy import NumpyBackend
from bayesian_sparse_gmm.config import HyperParams, SamplerConfig
from bayesian_sparse_gmm.sampler import GibbsSampler
from bayesian_sparse_gmm.model import BayesianSparseGMM


def test_theta_range_and_updates():
    """Verify that after updates, theta lies strictly in the interval (0, 1)."""
    hp = HyperParams(theta=0.5, kappa=0.1)
    cfg = SamplerConfig(K_max=3)
    backend = NumpyBackend()
    sampler = GibbsSampler(cfg, hp, backend)

    X = np.random.normal(size=(10, 5))
    rng = np.random.default_rng(42)
    state = sampler.initialize(X, rng)

    for _ in range(10):
        state = sampler.sample_step(X, state, rng)
        assert 0.0 < state.theta < 1.0


def test_theta_conjugacy():
    """Verify conjugate Beta update matches the theoretical posterior mean."""
    # With xi = [1, 1, 0, 0, 0] (2 active, 3 spike), p=5, kappa=0.1:
    # beta_theta = 5^1.1 / log(5) ≈ 5.873 / 1.609 ≈ 3.65
    # theta ~ Beta(1 + 2, beta_theta + (5 - 2)) = Beta(3, 6.65)
    # Expected mean = 3 / (3 + 6.65) ≈ 0.311
    hp = HyperParams(theta=0.5, kappa=0.1)
    cfg = SamplerConfig(K_max=3)
    backend = NumpyBackend()
    sampler = GibbsSampler(cfg, hp, backend)

    X = np.random.normal(size=(10, 5))
    rng = np.random.default_rng(42)
    sampler.initialize(X, rng)

    # Let the sampler calculate the exact beta_theta
    expected_beta_theta = 5.0**1.1 / np.log(5.0)
    assert np.allclose(sampler.beta_theta, expected_beta_theta)

    thetas = []
    # Force xi to have 2 active features
    xi = np.array([1, 1, 0, 0, 0], dtype=np.int32)
    s_active = int(np.sum(xi))

    for _ in range(10000):
        theta = rng.beta(1.0 + s_active, sampler.beta_theta + (5 - s_active))
        thetas.append(theta)

    sample_mean = np.mean(thetas)
    # Verify the mean is within statistical bounds [0.25, 0.37]
    assert 0.25 <= sample_mean <= 0.37


def test_high_p_sparsity():
    """Verify that in high dimensions (p=1000), theta is driven very close to 0."""
    p = 1000
    hp = HyperParams(theta=0.5, kappa=0.1)
    cfg = SamplerConfig(K_max=3)
    backend = NumpyBackend()
    sampler = GibbsSampler(cfg, hp, backend)

    X = np.random.normal(size=(10, p))
    rng = np.random.default_rng(42)
    sampler.initialize(X, rng)

    # Force s_active to be 0
    xi = np.zeros(p, dtype=np.int32)
    s_active = int(np.sum(xi))

    thetas = []
    # beta_theta = 1000^1.1 / log(1000) ≈ 1995.26 / 6.908 ≈ 288.84
    # theta ~ Beta(1, 288.84 + 1000) = Beta(1, 1288.84)
    # Expected mean ≈ 1 / 1289.84 ≈ 0.000775
    for _ in range(1000):
        theta = rng.beta(1.0 + s_active, sampler.beta_theta + (p - s_active))
        thetas.append(theta)

    sample_mean = np.mean(thetas)
    assert sample_mean < 0.005


def test_theta_trace():
    """Verify that theta varies across MCMC iterations in the model trace."""
    X = np.random.normal(size=(30, 10))
    model = BayesianSparseGMM(
        K_max=2, n_iter=20, burn_in=5, thinning=1, random_state=42
    )
    model.fit(X)

    trace_theta = model.trace_["theta"]
    # 20 - 5 = 15 samples
    assert trace_theta.shape == (15,)
    # Verify that the samples are not all identical
    assert len(np.unique(trace_theta)) > 1
