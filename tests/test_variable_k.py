import numpy as np

from bayesian_sparse_gmm.model import BayesianSparseGMM


def test_variable_k_birth_death():
    rng = np.random.default_rng(42)
    X1 = rng.normal(-10, 1, size=(50, 3))
    X2 = rng.normal(10, 1, size=(50, 3))
    X = np.vstack([X1, X2])

    model = BayesianSparseGMM(
        K_max=10, n_iter=100, burn_in=50, random_state=42, lambda_pois=2.0
    )
    model.fit(X)

    active_clusters = [state.K_active for state in model.states_]
    assert len(active_clusters) == 50
    # The mode K_active should be around 2
    assert np.median(active_clusters) in [1, 2, 3]


def test_variable_k_convergence():
    rng = np.random.default_rng(42)
    X1 = rng.normal(-10, 1, size=(20, 2))
    X2 = rng.normal(0, 1, size=(20, 2))
    X3 = rng.normal(10, 1, size=(20, 2))
    X = np.vstack([X1, X2, X3])

    model = BayesianSparseGMM(
        K_max=10, n_iter=200, burn_in=150, random_state=42, lambda_pois=3.0
    )
    model.fit(X)

    active_clusters = [state.K_active for state in model.states_]
    mode_K = int(np.argmax(np.bincount(active_clusters)))
    assert mode_K in [2, 3, 4]
