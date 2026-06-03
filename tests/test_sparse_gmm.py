from bayesian_sparse_gmm import BayesianSparseGMM


def test_initialization():
    """Test that the model initializes with correct parameters."""
    model = BayesianSparseGMM(n_components=5, alpha_0=0.05, max_iter=50)
    assert model.n_components == 5
    assert model.alpha_0 == 0.05
    assert model.max_iter == 50
    assert model.weights_ is None
    assert model.means_ is None
    assert model.covariances_ is None
    assert not model.converged_
