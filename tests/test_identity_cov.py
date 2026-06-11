"""Tests for Module 7: post-processing and evaluation alignment."""

import numpy as np

from bayesian_sparse_gmm.postprocessing import _find_reference_state, align_labels
from bayesian_sparse_gmm.state import SamplerState

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_state(mu, z, K_max=None):
    """Build a minimal SamplerState for testing postprocessing logic."""
    n, p = mu.shape[0], mu.shape[1]
    if K_max is None:
        K_max = n
    return SamplerState(
        z=np.array(z, dtype=int),
        w=np.ones(K_max) / K_max,
        mu=mu.copy(),
        xi=np.zeros(p, dtype=np.int32),
        theta=0.5,
        tau2=np.ones((K_max, p)),
        sigma2=np.ones(p),
        iteration=1,
        K_active=len(np.unique(z)),
    )


# ---------------------------------------------------------------------------
# Change 1: Label-switching reference selection
# ---------------------------------------------------------------------------


class TestFindReferenceState:
    """_find_reference_state picks the state with the lowest reconstruction error."""

    def test_perfect_reconstruction_chosen(self):
        """State where mu[z] == X should have zero error and be selected."""
        rng = np.random.default_rng(0)
        n, p = 20, 3
        X = rng.normal(size=(n, p))

        # state_a: perfect reconstruction — mu = unique rows, z maps them back
        unique_rows = X.copy()  # one cluster per point
        z_perfect = np.arange(n)
        mu_perfect = unique_rows
        state_perfect = _make_state(mu_perfect, z_perfect, K_max=n)

        # state_b: random mu — bad reconstruction
        mu_random = rng.normal(size=(n, p))
        state_random = _make_state(mu_random, z_perfect, K_max=n)

        idx = _find_reference_state([state_random, state_perfect], X)
        assert idx == 1

    def test_returns_int(self):
        rng = np.random.default_rng(1)
        n, p = 5, 2
        X = rng.normal(size=(n, p))
        mu = rng.normal(size=(3, p))
        states = [_make_state(mu, [0, 1, 2, 0, 1], K_max=3)]
        idx = _find_reference_state(states, X)
        assert isinstance(idx, int)

    def test_single_state_returns_zero(self):
        rng = np.random.default_rng(2)
        X = rng.normal(size=(4, 2))
        mu = rng.normal(size=(2, 2))
        states = [_make_state(mu, [0, 1, 0, 1], K_max=2)]
        assert _find_reference_state(states, X) == 0


class TestAlignLabels:
    def test_accepts_X_parameter(self):
        rng = np.random.default_rng(3)
        X = rng.normal(size=(6, 2))
        mu = rng.normal(size=(2, 2))
        states = [_make_state(mu, [0, 0, 0, 1, 1, 1], K_max=2)] * 3
        result = align_labels(states, X)
        assert len(result) == 3

    def test_empty_states(self):
        X = np.zeros((0, 2))
        assert align_labels([], X) == []


# ---------------------------------------------------------------------------
# Change 2: K estimator via posterior mode
# ---------------------------------------------------------------------------


class TestKHatEstimator:
    """K_hat_ should equal the posterior mode of K_active across MCMC samples."""

    def test_mode_is_correct(self):
        from bayesian_sparse_gmm.model import BayesianSparseGMM

        rng = np.random.default_rng(42)
        # Simple 2-cluster data
        X = np.vstack(
            [
                rng.normal([0, 0], 0.3, size=(20, 2)),
                rng.normal([5, 5], 0.3, size=(20, 2)),
            ]
        )

        model = BayesianSparseGMM(
            K_max=5, n_iter=100, burn_in=30, random_state=42, verbose=0
        )
        model.fit(X)

        # K_hat_ is the argmax of bincount of K_active samples
        K_samples = [state.K_active for state in model.states_]
        expected = int(np.argmax(np.bincount(K_samples)))
        assert model.K_hat_ == expected

    def test_n_clusters_returns_k_hat(self):
        from bayesian_sparse_gmm.model import BayesianSparseGMM

        rng = np.random.default_rng(7)
        X = rng.normal(size=(20, 3))
        model = BayesianSparseGMM(K_max=4, n_iter=50, burn_in=10, random_state=7)
        model.fit(X)
        assert model.n_clusters_ == model.K_hat_


# ---------------------------------------------------------------------------
# Change 3: Identity covariance mode
# ---------------------------------------------------------------------------


class TestIdentityCovariance:
    """With use_identity_covariance=True, sigma2 must stay ones(p) every step."""

    def _run_sampler(self, use_identity_covariance: bool):
        from bayesian_sparse_gmm.backends._numpy import NumpyBackend
        from bayesian_sparse_gmm.config import HyperParams, SamplerConfig
        from bayesian_sparse_gmm.sampler import GibbsSampler

        rng = np.random.default_rng(0)
        n, p, K = 30, 4, 3
        X = rng.normal(size=(n, p))

        config = SamplerConfig(K_max=K, n_iter=5, burn_in=0)
        hp = HyperParams(use_identity_covariance=use_identity_covariance)
        sampler = GibbsSampler(config, hp, NumpyBackend())
        states = sampler.run(X, seed=0)
        return states, p

    def test_identity_mode_sigma2_is_ones(self):
        states, p = self._run_sampler(use_identity_covariance=True)
        for state in states:
            np.testing.assert_array_equal(
                state.sigma2,
                np.ones(p),
                err_msg="sigma2 must be ones(p) in identity covariance mode",
            )

    def test_learned_mode_sigma2_varies(self):
        states, p = self._run_sampler(use_identity_covariance=False)
        # At least some sigma2 values should differ from 1.0
        all_ones = all(np.allclose(s.sigma2, np.ones(p)) for s in states)
        assert not all_ones, "Learned sigma2 should differ from ones(p)"

    def test_both_modes_produce_valid_clustering(self):
        from bayesian_sparse_gmm.model import BayesianSparseGMM

        rng = np.random.default_rng(99)
        X = np.vstack(
            [
                rng.normal([0, 0], 0.5, size=(15, 2)),
                rng.normal([4, 4], 0.5, size=(15, 2)),
            ]
        )

        for flag in (True, False):
            model = BayesianSparseGMM(
                K_max=4,
                n_iter=80,
                burn_in=20,
                use_identity_covariance=flag,
                random_state=99,
            )
            model.fit(X)
            assert model.K_hat_ >= 1
            assert len(model.labels_) == len(X)
