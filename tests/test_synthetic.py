import numpy as np
from sklearn.metrics import adjusted_rand_score

from bayesian_sparse_gmm.model import BayesianSparseGMM


def test_synthetic_sparse_recovery():
    """Verify that BayesianSparseGMM can recover the true clusters and features
    on a small synthetic sparse dataset.
    """
    rng = np.random.default_rng(42)
    n = 100
    p = 20
    K_true = 3
    n_signal = 5

    # Generate synthetic data
    X = rng.normal(0, 1, size=(n, p))
    true_labels = rng.integers(0, K_true, size=n)

    # Add signal to the first `n_signal` features
    for k in range(K_true):
        mask = true_labels == k
        X[mask, :n_signal] += rng.normal(
            loc=rng.choice([-5, 5]), scale=0.5, size=(mask.sum(), n_signal)
        )

    gmm = BayesianSparseGMM(
        K_max=5,
        n_iter=50,
        burn_in=20,
        warm_up_iters=0,
        lambda_0=50.0,
        lambda_1=1.0,
        random_state=42,
        verbose=0,
    )
    gmm.fit(X)

    # Check that ARI is reasonably high
    ari = adjusted_rand_score(true_labels, gmm.labels_)
    assert ari > 0.40, f"Expected good clustering performance, got ARI={ari:.3f}"

    # Check feature selection (should select the first 5 features)
    selected = gmm.selected_features_
    precision = len([f for f in selected if f < n_signal]) / max(len(selected), 1)
    recall = len([f for f in selected if f < n_signal]) / n_signal

    assert (
        recall >= 0.8
    ), f"Expected to recover most signal features, got recall={recall:.3f}"
    assert (
        precision > 0.5
    ), f"Expected high precision on selected features, got precision={precision:.3f}"
