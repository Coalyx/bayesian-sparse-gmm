import numpy as np

from bayesian_sparse_gmm.urn import (
    log_rising_factorial,
    log_urn_weight,
    truncated_poisson_pmf,
    urn_weight,
)


def test_log_rising_factorial():
    assert np.isclose(log_rising_factorial(5.0, 0), 0.0)
    assert np.isclose(log_rising_factorial(5.0, 1), np.log(5.0))
    assert np.isclose(log_rising_factorial(5.0, 3), np.log(5.0 * 6.0 * 7.0))


def test_truncated_poisson_pmf():
    lambda_pois = 2.0
    K_max = 5
    pmfs = [truncated_poisson_pmf(k, lambda_pois, K_max) for k in range(1, K_max + 1)]
    assert np.isclose(sum(pmfs), 1.0)
    assert pmfs[1] >= pmfs[0]  # for lambda=2.0, PMF at 2 is >= PMF at 1
    assert truncated_poisson_pmf(0, lambda_pois, K_max) == 0.0
    assert truncated_poisson_pmf(K_max + 1, lambda_pois, K_max) == 0.0


def test_urn_weight():
    alpha = 1.0
    lambda_pois = 2.0
    K_max = 10
    n = 20

    weights = [urn_weight(k, n, alpha, lambda_pois, K_max) for k in range(1, K_max + 1)]

    assert all(w >= 0 for w in weights)
    assert urn_weight(0, n, alpha, lambda_pois, K_max) == 0.0
    assert urn_weight(K_max + 1, n, alpha, lambda_pois, K_max) == 0.0

    for k in range(1, K_max + 1):
        # We need to be careful with floating point 0
        if weights[k - 1] > 0:
            assert np.isclose(
                np.log(weights[k - 1]), log_urn_weight(k, n, alpha, lambda_pois, K_max)
            )
