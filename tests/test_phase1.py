import numpy as np
from scipy.special import logsumexp
from scipy.stats import invgauss
from bayesian_sparse_gmm.config import HyperParams, SamplerConfig
from bayesian_sparse_gmm.state import SamplerState
from bayesian_sparse_gmm.utils import log_sum_exp, sample_inverse_gaussian

def test_config():
    hp = HyperParams()
    assert hp.lambda_0 == 1000.0
    assert hp.lambda_1 == 0.1
    assert hp.alpha == 0.01
    assert hp.b == 100.0
    
    cfg = SamplerConfig(K_max=10)
    assert cfg.K_max == 10
    assert cfg.n_jobs == -1

def test_state():
    z = np.zeros(10, dtype=int)
    w = np.ones(5) / 5.0
    mu = np.zeros((5, 3))
    gamma = np.ones(3, dtype=int)
    tau2 = np.ones((5, 3))
    
    state = SamplerState(
        z=z, w=w, mu=mu, gamma=gamma, theta=0.5, tau2=tau2, iteration=0
    )
    assert state.iteration == 0
    assert state.theta == 0.5

def test_log_sum_exp():
    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    expected = logsumexp(x, axis=-1)
    actual = log_sum_exp(x, axis=-1)
    assert np.allclose(actual, expected)

def test_sample_inverse_gaussian():
    rng = np.random.default_rng(42)
    mean = np.array([2.0, 3.0])
    shape = np.array([1.5, 2.5])
    
    # Draw many samples to test distribution mean
    samples = []
    for _ in range(10000):
        samples.append(sample_inverse_gaussian(mean, shape, rng))
    
    samples = np.array(samples)
    sample_mean = np.mean(samples, axis=0)
    
    # Inverse Gaussian mean should be close to theoretical mean
    assert np.allclose(sample_mean, mean, rtol=0.05)
