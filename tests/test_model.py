import numpy as np
from sklearn.datasets import make_blobs
from bayesian_sparse_gmm.model import BayesianSparseGMM

def test_model_scikit_learn_api():
    # Generate simple clustering dataset: 3 centers in 4D space, 2 dimensions are noise
    X, y = make_blobs(n_samples=50, centers=3, n_features=4, random_state=42)
    # Make two features sparse (noise-like) by setting them to small scale
    X[:, 2:] = np.random.normal(scale=0.01, size=(50, 2))
    
    # Instantiate GMM
    gmm = BayesianSparseGMM(
        K_max=5,
        n_iter=50,
        burn_in=20,
        thinning=2,
        random_state=42,
        verbose=0
    )
    
    # Check parameters
    assert gmm.K_max == 5
    assert gmm.n_iter == 50
    
    # Fit GMM
    gmm.fit(X)
    
    # Check fitted attributes
    assert hasattr(gmm, "states_")
    assert len(gmm.states_) == 15  # (50 - 20) / 2 = 15
    assert gmm.means_.shape == (5, 4)
    assert gmm.w_.shape == (5,)
    assert gmm.feature_probabilities_.shape == (4,)
    assert gmm.labels_.shape == (50,)
    assert gmm.n_clusters_ <= 5
    
    # Check trace
    trace = gmm.trace_
    assert "z" in trace
    assert trace["z"].shape == (15, 50)
    assert trace["mu"].shape == (15, 5, 4)
    assert trace["w"].shape == (15, 5)
    assert trace["gamma"].shape == (15, 4)
    
    # Predict
    labels = gmm.predict(X)
    assert labels.shape == (50,)
    assert np.all(labels >= 0) and np.all(labels < 5)
    
    # Predict proba
    proba = gmm.predict_proba(X)
    assert proba.shape == (50, 5)
    assert np.allclose(np.sum(proba, axis=1), 1.0)
    
    # Score
    score = gmm.score(X)
    assert isinstance(score, float)


def test_model_custom_theta():
    X = np.random.normal(size=(20, 3))
    # Test setting custom theta
    gmm = BayesianSparseGMM(K_max=3, theta=0.4, n_iter=10, burn_in=2, random_state=42)
    assert gmm.theta == 0.4
    gmm.fit(X)
    assert hasattr(gmm, "states_")
    for state in gmm.states_:
        assert state.theta == 0.4

