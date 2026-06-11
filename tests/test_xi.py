import numpy as np

from bayesian_sparse_gmm.model import BayesianSparseGMM


def test_xi_drops_noise_features():
    """Verify that the model drops all noise features (xi goes to 0)
    when there is no cluster signal in the data.
    """
    rng = np.random.default_rng(42)
    # Pure standard normal noise data (no real clusters)
    # n=1000 ensures random cluster means are small (~0.05)
    X = rng.normal(0, 1, size=(1000, 10))

    gmm = BayesianSparseGMM(
        K_max=3,
        n_iter=50,
        burn_in=20,
        warm_up_iters=0,
        lambda_0=10.0,  # lower spike penalty so the threshold is higher (log(10)/10 = 0.23 > 0.05)
        lambda_1=1.0,
        random_state=42,
        verbose=0,
    )
    gmm.fit(X)

    # Since the data is pure noise, all feature probabilities should be pushed towards 0
    # Before the Laplace fix, they were stuck at 1.0 due to normal-scale mixture conditioning.
    mean_prob = np.mean(gmm.feature_probabilities_)
    assert mean_prob < 0.2, (
        f"Expected noise features to be dropped (prob < 0.2), but got mean probability {mean_prob:.3f}. "
        "Feature selection (xi) might be stuck in the slab."
    )
    assert (
        len(gmm.selected_features_) < 5
    ), f"Expected most features to be dropped, but kept {len(gmm.selected_features_)}/10 features."
