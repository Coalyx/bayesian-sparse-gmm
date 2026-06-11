import numpy as np

from bayesian_sparse_gmm.backends._numpy import NumpyBackend
from bayesian_sparse_gmm.config import HyperParams, SamplerConfig
from bayesian_sparse_gmm.sampler import GibbsSampler


def test_xi_shape_and_binary_values():
    """Verify that xi is 1D with shape (p,) and all elements are binary (0 or 1)."""
    hp = HyperParams()
    cfg = SamplerConfig(K_max=3, warm_up_iters=0)
    backend = NumpyBackend()
    sampler = GibbsSampler(cfg, hp, backend)

    # 10 samples, 5 features
    X = np.random.normal(size=(10, 5))
    rng = np.random.default_rng(42)

    state = sampler.initialize(X, rng)
    assert state.xi.shape == (5,)
    assert np.all(np.isin(state.xi, [0, 1]))

    next_state = sampler.sample_step(X, state, rng)
    assert next_state.xi.shape == (5,)
    assert np.all(np.isin(next_state.xi, [0, 1]))


def test_xi_warm_up_behavior():
    """Verify that during the warm-up period, xi is forced to be all 1s."""
    hp = HyperParams()
    cfg = SamplerConfig(K_max=3, warm_up_iters=5)
    backend = NumpyBackend()
    sampler = GibbsSampler(cfg, hp, backend)

    X = np.random.normal(size=(10, 5))
    rng = np.random.default_rng(42)

    state = sampler.initialize(X, rng)
    # At iteration 0 (which is < warm_up_iters), sample_step should force xi = 1
    next_state = sampler.sample_step(X, state, rng)
    assert np.array_equal(next_state.xi, np.ones(5, dtype=np.int32))


def test_xi_signal_aggregation_recovery():
    """Verify that xi correctly aggregates cluster signals and recovers signal features."""
    # We construct a synthetic dataset where:
    # - Feature 0 has cluster-specific differences (signal)
    # - Feature 1 has no cluster-specific differences (noise)
    rng = np.random.default_rng(42)
    X_c0 = rng.normal(loc=-5.0, scale=0.1, size=(20, 2))
    X_c1 = rng.normal(loc=5.0, scale=0.1, size=(20, 2))
    # Override feature 1 to be completely noise centered at 0 across both clusters
    X_c0[:, 1] = rng.normal(loc=0.0, scale=1.0, size=20)
    X_c1[:, 1] = rng.normal(loc=0.0, scale=1.0, size=20)

    X = np.vstack([X_c0, X_c1])

    # Sampler configuration: lambda_0 is large (spike) and lambda_1 is small (slab)
    hp = HyperParams(lambda_0=100.0, lambda_1=0.1, theta=0.5)
    cfg = SamplerConfig(K_max=2, warm_up_iters=0)
    backend = NumpyBackend()
    sampler = GibbsSampler(cfg, hp, backend)

    state = sampler.initialize(X, rng)
    # Manually configure state to force high cluster separation and active labels
    state.z = np.concatenate([np.zeros(20, dtype=int), np.ones(20, dtype=int)])
    state.mu = np.array([[-5.0, 0.0], [5.0, 0.0]])
    state.tau2 = np.ones((2, 2))

    next_state = sampler.sample_step(X, state, rng)

    # Feature 0 has significant means of -5.0 and 5.0.
    # The normal-scale-mixture term log N(mu; 0, tau2 / lambda^2) will strongly favor the slab (lambda_1 = 0.1) over spike (lambda_0 = 100.0).
    # Feature 1 has means near 0.0. The term will favor the spike (lambda_0 = 100.0).
    # Therefore, xi[0] should be 1, and xi[1] should be 0.
    assert next_state.xi[0] == 1
    assert next_state.xi[1] == 0
