import numpy as np

from bayesian_sparse_gmm.diagnostics import effective_sample_size, gelman_rubin


def test_rhat_stationary_and_divergent():
    # 1. Stationary chain (split R-hat should be close to 1.0)
    rng = np.random.default_rng(42)
    stationary = rng.normal(loc=0.0, scale=1.0, size=500)
    rhat_stat = gelman_rubin(stationary)
    assert np.isclose(rhat_stat, 1.0, atol=0.05)

    # 2. Divergent chain: first half is different from second half (R-hat should be > 1.1)
    divergent = np.concatenate(
        [
            rng.normal(loc=-2.0, scale=1.0, size=250),
            rng.normal(loc=2.0, scale=1.0, size=250),
        ]
    )
    rhat_div = gelman_rubin(divergent)
    assert rhat_div > 1.1


def test_ess_iid_and_correlated():
    # 1. Independent samples (ESS should be close to sample size)
    rng = np.random.default_rng(42)
    iid = rng.normal(size=500)
    ess_iid = effective_sample_size(iid)
    # ESS for i.i.d should be close to 500
    assert 400 <= ess_iid <= 600

    # 2. Highly correlated samples (AR(1) process with high phi)
    correlated = np.empty(500)
    correlated[0] = 0.0
    for i in range(1, 500):
        correlated[i] = 0.95 * correlated[i - 1] + rng.normal()
    ess_corr = effective_sample_size(correlated)
    # ESS should be much smaller than 500
    assert ess_corr < 100


def test_multidimensional_trace():
    rng = np.random.default_rng(42)
    # Trace with 100 samples, shape (100, 3, 2)
    trace = rng.normal(size=(100, 3, 2))

    rhat = gelman_rubin(trace)
    ess = effective_sample_size(trace)

    # Outputs should preserve shape (3, 2)
    assert rhat.shape == (3, 2)
    assert ess.shape == (3, 2)
    assert np.all(rhat >= 0.9)
    assert np.all(ess > 0.0)
