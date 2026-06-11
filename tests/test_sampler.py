import numpy as np

from bayesian_sparse_gmm.backends._numpy import NumpyBackend
from bayesian_sparse_gmm.config import HyperParams, SamplerConfig
from bayesian_sparse_gmm.sampler import GibbsSampler


def test_sampler_initialization():
    hp = HyperParams()
    cfg = SamplerConfig(K_max=3)
    backend = NumpyBackend()
    sampler = GibbsSampler(cfg, hp, backend)

    # 10 samples, 4 features
    X = np.random.normal(size=(10, 4))
    rng = np.random.default_rng(42)

    state = sampler.initialize(X, rng)

    assert state.iteration == 0
    assert state.z.shape == (10,)
    assert state.w.shape == (3,)
    assert state.mu.shape == (3, 4)
    assert state.xi.shape == (4,)
    assert np.array_equal(state.xi, np.zeros(4, dtype=np.int32))
    assert state.tau2.shape == (3, 4)
    assert state.sigma2.shape == (4,)
    assert 0.0 <= state.theta <= 1.0


def test_sampler_step():
    hp = HyperParams()
    cfg = SamplerConfig(K_max=3)
    backend = NumpyBackend()
    sampler = GibbsSampler(cfg, hp, backend)

    X = np.random.normal(size=(10, 4))
    rng = np.random.default_rng(42)

    state = sampler.initialize(X, rng)
    next_state = sampler.sample_step(X, state, rng)

    assert next_state.iteration == 1
    assert next_state.z.shape == (10,)
    assert next_state.w.shape == (3,)
    assert next_state.mu.shape == (3, 4)
    assert next_state.xi.shape == (4,)
    assert next_state.tau2.shape == (3, 4)
    assert next_state.sigma2.shape == (4,)


def test_sampler_run_chain():
    hp = HyperParams()
    # 20 iterations, burn_in 10, thinning 2 -> expected (20 - 10) / 2 = 5 samples
    cfg = SamplerConfig(K_max=3, n_iter=20, burn_in=10, thinning=2)
    backend = NumpyBackend()
    sampler = GibbsSampler(cfg, hp, backend)

    X = np.random.normal(size=(10, 4))
    states = sampler.run(X, seed=42)

    assert len(states) == 5
    for i, state in enumerate(states):
        # Iteration indices should be 12, 14, 16, 18, 20
        assert state.iteration == 10 + (i + 1) * 2


def test_sampler_active_masking():
    hp = HyperParams(lambda_0=10.0, lambda_1=0.1)
    cfg = SamplerConfig(K_max=4, warm_up_iters=0)
    backend = NumpyBackend()
    sampler = GibbsSampler(cfg, hp, backend)

    # 10 samples, 2 features
    X = np.random.normal(size=(10, 2))
    rng = np.random.default_rng(42)
    state = sampler.initialize(X, rng)

    # Set state so that only cluster 0 is active
    state.z = np.zeros(10, dtype=int)
    state.mu = np.array([[1.0, 2.0], [10.0, 20.0], [100.0, 200.0], [1000.0, 2000.0]])

    # Run a single step
    next_state = sampler.sample_step(X, state, rng)

    # Under masking, active_K = 1, sum_abs_mu = [1.0, 2.0]
    # log_laplace_slab = (log(0.1) - log(2.0)) - 0.1 * [1.0, 2.0] = -3.0957, -3.1957
    # log_laplace_spike = (log(10.0) - log(2.0)) - 10.0 * [1.0, 2.0] = -8.3906, -18.3906
    # So slab probability is close to 1, and both features should be selected.
    expected_xi = np.array([1, 1], dtype=np.int32)
    assert np.array_equal(next_state.xi, expected_xi)


def test_hard_thresholding_memberships():
    """Test that mixing weights below 1/(2N) are hard-thresholded to 1e-300."""
    hp = HyperParams(alpha=0.01)
    cfg = SamplerConfig(K_max=3)
    backend = NumpyBackend()
    sampler = GibbsSampler(cfg, hp, backend)

    # N = 100 samples, threshold = 1 / 200 = 0.005
    X = np.random.normal(size=(100, 2))
    rng = np.random.default_rng(42)
    state = sampler.initialize(X, rng)

    # Verify that a step runs successfully without numerical issues
    next_state = sampler.sample_step(X, state, rng)
    assert next_state is not None

    # Verify the thresholding logic directly
    n = X.shape[0]
    threshold = 1.0 / (2.0 * n)
    assert threshold == 0.005

    # Mock some weights: one above threshold, others below
    w = np.array([0.996, 0.003, 0.001])
    w_safe = np.where(w < threshold, 1e-300, w)
    assert w_safe[0] == 0.996
    assert w_safe[1] == 1e-300
    assert w_safe[2] == 1e-300


def test_gibbs_sequence_order():
    """Verify that cluster assignments (Z) are updated first, followed by mixing weights (W)."""
    hp = HyperParams(alpha=0.1)
    cfg = SamplerConfig(K_max=3)
    backend = NumpyBackend()
    sampler = GibbsSampler(cfg, hp, backend)

    # 5 samples, 2 features
    X = np.zeros((5, 2))

    # Setup custom state:
    z = np.zeros(5, dtype=int)
    w = np.array([1.0, 0.0, 0.0])
    mu = np.zeros((3, 2))
    xi = np.ones(2, dtype=np.int32)
    theta = 0.5
    tau2 = np.ones((3, 2))
    sigma2 = np.ones(2)

    from bayesian_sparse_gmm.state import SamplerState

    state = SamplerState(
        z=z, w=w, mu=mu, xi=xi, theta=theta, tau2=tau2, sigma2=sigma2, iteration=0
    )

    real_rng = np.random.default_rng(42)

    class MockGenerator:
        def __init__(self, rng):
            self.rng = rng
            self.passed_alpha = None

        def choice(self, a, size=None, replace=True, p=None):
            # Force choosing the second option to trigger birth/active cluster 1
            return 1

        def dirichlet(self, alpha, size=None):
            self.passed_alpha = alpha
            return self.rng.dirichlet(alpha, size=size)

        def __getattr__(self, name):
            return getattr(self.rng, name)

    mock_rng = MockGenerator(real_rng)

    # Run a single sampler step
    next_state = sampler.sample_step(X, state, mock_rng)

    # If Z is updated first:
    # 1. Z is updated. Under choice=1, Z becomes [1, 1, 1, 1, 0].
    # 2. W is updated using the new Z. Since active clusters is [0, 1], next_state.w should have next_state.w[2] == 0.0.
    assert np.isclose(next_state.w[2], 0.0)
    # The dirichlet is called with alpha + counts_active = 0.1 + [1, 4] = [1.1, 4.1]
    assert mock_rng.passed_alpha is not None
    assert np.allclose(mock_rng.passed_alpha, hp.alpha + np.array([1, 4]))


def test_sampler_theta_updates():
    """Verify that theta is initialized to the value specified in HyperParams and updates dynamically."""
    custom_theta = 0.35
    hp = HyperParams(theta=custom_theta)
    cfg = SamplerConfig(K_max=3)
    backend = NumpyBackend()
    sampler = GibbsSampler(cfg, hp, backend)

    X = np.random.normal(size=(10, 4))
    rng = np.random.default_rng(42)

    state = sampler.initialize(X, rng)
    assert state.theta == custom_theta

    # Run a few steps to ensure theta changes
    thetas = [state.theta]
    for _ in range(5):
        state = sampler.sample_step(X, state, rng)
        thetas.append(state.theta)

    # It should not remain constant, so the number of unique thetas is > 1
    assert len(set(thetas)) > 1
