import numpy as np
from bayesian_sparse_gmm.config import HyperParams, SamplerConfig
from bayesian_sparse_gmm.backends._numpy import NumpyBackend
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
    assert state.gamma.shape == (4,)
    assert state.tau2.shape == (3, 4)
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
    assert next_state.gamma.shape == (4,)
    assert next_state.tau2.shape == (3, 4)

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
