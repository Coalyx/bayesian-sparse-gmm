"""
Tests for joblib serialization / deserialization correctness.

These tests verify that:
1. model.predict(X) is *bit-identical* before and after a joblib save/load.
2. Predictions are invariant to the test-batch size (single sample vs. full
   batch vs. arbitrary subset).
3. Backend-change warnings are emitted correctly when the runtime environment
   does not support the backend that was used during training.
"""

import io
import pickle
import warnings

import joblib
import numpy as np
import pytest
from sklearn.datasets import make_blobs

from bayesian_sparse_gmm.model import BayesianSparseGMM


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fit_mcmc(X, **kwargs):
    defaults = dict(
        K_max=3,
        optimizer="default",
        n_iter=60,
        burn_in=20,
        thinning=2,
        random_state=0,
        backend="numpy",
    )
    defaults.update(kwargs)
    model = BayesianSparseGMM(**defaults)
    model.fit(X)
    return model


def _fit_svi(X, **kwargs):
    defaults = dict(
        K_max=3,
        optimizer="svi",
        epochs=15,
        batch_size=32,
        random_state=0,
        backend="numpy",
    )
    defaults.update(kwargs)
    model = BayesianSparseGMM(**defaults)
    model.fit(X)
    return model


def _roundtrip_joblib(model):
    """Serialize to an in-memory buffer and deserialize -- no disk I/O needed."""
    buf = io.BytesIO()
    joblib.dump(model, buf)
    buf.seek(0)
    return joblib.load(buf)


def _roundtrip_pickle(model):
    """Alternative pickle round-trip for cross-validation."""
    return pickle.loads(pickle.dumps(model))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def blobs_dataset():
    """Small 3-cluster dataset with 2 informative and 2 noise features."""
    X, _ = make_blobs(n_samples=80, centers=3, n_features=4, random_state=7)
    X[:, 2:] = np.random.default_rng(7).normal(scale=0.05, size=(80, 2))
    return X.astype(np.float64)


# ---------------------------------------------------------------------------
# Bug #1 regression: threshold must use n_train_, not n_test
# ---------------------------------------------------------------------------


class TestThresholdConsistency:
    """
    Ensure predict() and predict_proba() produce the same result regardless
    of the test-batch size -- the root cause of the joblib drift bug.
    """

    def test_predict_single_sample_equals_full_batch_mcmc(self, blobs_dataset):
        """Single-sample predict must match multi-sample predict for MCMC."""
        X = blobs_dataset
        model = _fit_mcmc(X)

        pred_full = model.predict(X)
        for i in range(len(X)):
            pred_single = model.predict(X[[i]])
            assert pred_single[0] == pred_full[i], (
                f"Sample {i}: single-sample predict={pred_single[0]} "
                f"!= full-batch predict={pred_full[i]}. "
                "Likely cause: threshold uses test-set n instead of n_train_."
            )

    def test_predict_single_sample_equals_full_batch_svi(self, blobs_dataset):
        """Single-sample predict must match multi-sample predict for SVI."""
        X = blobs_dataset
        model = _fit_svi(X)

        pred_full = model.predict(X)
        for i in range(len(X)):
            pred_single = model.predict(X[[i]])
            assert pred_single[0] == pred_full[i], (
                f"Sample {i}: single-sample predict={pred_single[0]} "
                f"!= full-batch predict={pred_full[i]}."
            )

    def test_predict_proba_sum_to_one_any_batch_mcmc(self, blobs_dataset):
        """predict_proba rows must sum to 1 for any batch size."""
        X = blobs_dataset
        model = _fit_mcmc(X)

        for batch_size in [1, 5, 10, len(X)]:
            proba = model.predict_proba(X[:batch_size])
            assert proba.shape == (batch_size, model.K_max)
            np.testing.assert_allclose(
                np.sum(proba, axis=1),
                np.ones(batch_size),
                atol=1e-12,
                err_msg=(
                    f"predict_proba rows do not sum to 1 for batch_size={batch_size}"
                ),
            )

    def test_predict_proba_sum_to_one_any_batch_svi(self, blobs_dataset):
        """predict_proba rows must sum to 1 for any batch size (SVI)."""
        X = blobs_dataset
        model = _fit_svi(X)

        for batch_size in [1, 5, 10, len(X)]:
            proba = model.predict_proba(X[:batch_size])
            np.testing.assert_allclose(
                np.sum(proba, axis=1),
                np.ones(batch_size),
                atol=1e-12,
                err_msg=(
                    f"SVI predict_proba rows do not sum to 1 for batch_size={batch_size}"
                ),
            )

    def test_n_train_stored_after_fit(self, blobs_dataset):
        """n_train_ must be set to training-set size after fit."""
        X = blobs_dataset
        model = _fit_mcmc(X)
        assert hasattr(model, "n_train_"), "model must expose n_train_ after fit()"
        assert model.n_train_ == len(X)


# ---------------------------------------------------------------------------
# Bug #1 regression: joblib round-trip must produce identical predictions
# ---------------------------------------------------------------------------


class TestJoblibRoundTrip:
    """
    Verify that predict / predict_proba / score are bit-identical before and
    after a joblib serialization round-trip.
    """

    def test_mcmc_predict_identical_after_roundtrip(self, blobs_dataset):
        X = blobs_dataset
        model = _fit_mcmc(X)

        pred_before = model.predict(X)
        model2 = _roundtrip_joblib(model)
        pred_after = model2.predict(X)

        np.testing.assert_array_equal(
            pred_before,
            pred_after,
            err_msg="MCMC predict labels differ after joblib round-trip.",
        )

    def test_mcmc_predict_proba_identical_after_roundtrip(self, blobs_dataset):
        X = blobs_dataset
        model = _fit_mcmc(X)

        proba_before = model.predict_proba(X)
        model2 = _roundtrip_joblib(model)
        proba_after = model2.predict_proba(X)

        np.testing.assert_array_equal(
            proba_before,
            proba_after,
            err_msg="MCMC predict_proba differs after joblib round-trip.",
        )

    def test_mcmc_score_identical_after_roundtrip(self, blobs_dataset):
        X = blobs_dataset
        model = _fit_mcmc(X)

        score_before = model.score(X)
        model2 = _roundtrip_joblib(model)
        score_after = model2.score(X)

        assert score_before == score_after, (
            f"MCMC score differs after joblib round-trip: "
            f"{score_before} vs {score_after}"
        )

    def test_svi_predict_identical_after_roundtrip(self, blobs_dataset):
        X = blobs_dataset
        model = _fit_svi(X)

        pred_before = model.predict(X)
        model2 = _roundtrip_joblib(model)
        pred_after = model2.predict(X)

        np.testing.assert_array_equal(
            pred_before,
            pred_after,
            err_msg="SVI predict labels differ after joblib round-trip.",
        )

    def test_svi_predict_proba_identical_after_roundtrip(self, blobs_dataset):
        X = blobs_dataset
        model = _fit_svi(X)

        proba_before = model.predict_proba(X)
        model2 = _roundtrip_joblib(model)
        proba_after = model2.predict_proba(X)

        np.testing.assert_array_equal(
            proba_before,
            proba_after,
            err_msg="SVI predict_proba differs after joblib round-trip.",
        )

    def test_mcmc_single_sample_predict_identical_after_roundtrip(self, blobs_dataset):
        """
        Specifically tests the Bug #1 scenario: predict a single sample both
        before and after the round-trip and ensure they are equal.
        This was the most broken case (threshold=0.5 for n=1).
        """
        X = blobs_dataset
        model = _fit_mcmc(X)

        # Predict single sample before save
        pred_before = model.predict(X[[0]])

        model2 = _roundtrip_joblib(model)

        # Predict single sample after load
        pred_after = model2.predict(X[[0]])

        assert pred_before[0] == pred_after[0], (
            f"Single-sample predict after joblib round-trip: "
            f"before={pred_before[0]}, after={pred_after[0]}"
        )

    def test_pickle_roundtrip_mcmc(self, blobs_dataset):
        """Ensure pickle (which joblib uses internally) also gives same results."""
        X = blobs_dataset
        model = _fit_mcmc(X)

        pred_before = model.predict(X)
        model2 = _roundtrip_pickle(model)
        pred_after = model2.predict(X)

        np.testing.assert_array_equal(
            pred_before,
            pred_after,
            err_msg="MCMC predict differs after pickle round-trip.",
        )

    def test_n_train_preserved_after_roundtrip(self, blobs_dataset):
        """n_train_ must survive serialization unchanged."""
        X = blobs_dataset
        model = _fit_mcmc(X)
        model2 = _roundtrip_joblib(model)

        assert model2.n_train_ == model.n_train_, (
            f"n_train_ changed after round-trip: "
            f"before={model.n_train_}, after={model2.n_train_}"
        )

    def test_backend_class_preserved_after_roundtrip(self, blobs_dataset):
        """backend_class_ must survive serialization unchanged."""
        X = blobs_dataset
        model = _fit_mcmc(X)
        model2 = _roundtrip_joblib(model)

        assert model2.backend_class_ == model.backend_class_


# ---------------------------------------------------------------------------
# Bug #2: Backend-change warning
# ---------------------------------------------------------------------------


class TestBackendChangeWarning:
    """
    Verify that a clear UserWarning is emitted when the model is loaded in an
    environment where the originally-used backend is unavailable.
    """

    def test_backend_mismatch_emits_warning(self, blobs_dataset):
        """
        Simulate the CUDA->numpy fallback scenario by manually patching
        backend_class_ to pretend the model was trained with CUDABackend.
        """
        X = blobs_dataset
        model = _fit_mcmc(X, backend="numpy")

        # Serialize the model
        buf = io.BytesIO()
        joblib.dump(model, buf)
        buf.seek(0)
        loaded: BayesianSparseGMM = joblib.load(buf)

        # Manually set backend_class_ to a different value to trigger the warning
        loaded.backend_class_ = "CUDABackend"

        # Re-trigger __setstate__ by pickling/unpickling the patched object
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            pickle.loads(pickle.dumps(loaded))

        backend_warnings = [
            x for x in w
            if issubclass(x.category, UserWarning) and "CUDABackend" in str(x.message)
        ]
        assert len(backend_warnings) >= 1, (
            "Expected a UserWarning about backend mismatch (CUDABackend->NumpyBackend) "
            "but none was emitted."
        )

    def test_no_warning_when_backend_unchanged(self, blobs_dataset):
        """No backend-mismatch warning when backend is the same after round-trip."""
        X = blobs_dataset
        model = _fit_mcmc(X, backend="numpy")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _roundtrip_joblib(model)

        backend_warnings = [
            x for x in w
            if issubclass(x.category, UserWarning)
            and "backend" in str(x.message).lower()
        ]
        assert len(backend_warnings) == 0, (
            f"Unexpected backend warning when backend did not change: "
            f"{[str(x.message) for x in backend_warnings]}"
        )

    def test_backend_class_set_after_fit(self, blobs_dataset):
        """backend_class_ must be set to the correct class name after fit()."""
        X = blobs_dataset
        model = _fit_mcmc(X, backend="numpy")
        assert hasattr(model, "backend_class_")
        assert model.backend_class_ == "NumpyBackend"
