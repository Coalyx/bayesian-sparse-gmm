"""
Correctness verification for CuPy KMeans CUDA optimization.

Tests compare the CuPy/CUDA implementation against scikit-learn's KMeans
across multiple dimensions: label quality, center accuracy, numerical stability,
edge cases, and dtype handling.

Requires: CuPy + CUDA GPU. Tests are automatically skipped on CPU-only machines.
"""

import numpy as np
import pytest
from sklearn.cluster import KMeans as SklearnKMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from bayesian_sparse_gmm.clustering.kmeans import KMeansCupy

cp = pytest.importorskip(
    "cupy", reason="CuPy not installed — skipping KMeans CUDA tests"
)

SEED = 42


class TestBasicClusteringQuality:
    """Test clustering quality against ground truth and sklearn."""

    def test_ari_vs_ground_truth(self):
        """ARI vs ground truth should be reasonable for n_init=1."""
        X, true_labels = make_blobs(
            n_samples=3000, centers=5, cluster_std=0.8, random_state=SEED
        )
        X = X.astype(np.float32)

        km = KMeansCupy(n_clusters=5, max_iter=300, random_state=SEED)
        km.fit(X)

        ari = adjusted_rand_score(true_labels, km.labels_)
        assert ari >= 0.60, f"ARI={ari:.4f} too low"

    def test_nmi_vs_ground_truth(self):
        """NMI vs ground truth should be high."""
        X, true_labels = make_blobs(
            n_samples=3000, centers=5, cluster_std=0.8, random_state=SEED
        )
        X = X.astype(np.float32)

        km = KMeansCupy(n_clusters=5, max_iter=300, random_state=SEED)
        km.fit(X)

        nmi = normalized_mutual_info_score(true_labels, km.labels_)
        assert nmi >= 0.85, f"NMI={nmi:.4f} too low"

    def test_matches_sklearn(self):
        """CuPy should match sklearn when using equivalent config."""
        X, _ = make_blobs(n_samples=2000, centers=4, cluster_std=1.0, random_state=SEED)
        X = X.astype(np.float32)

        sk = SklearnKMeans(
            n_clusters=4, init="k-means++", n_init=1, max_iter=300, random_state=SEED
        )
        sk.fit(X)

        cupy_km = KMeansCupy(n_clusters=4, max_iter=300, random_state=SEED)
        cupy_km.fit(X)

        ari = adjusted_rand_score(sk.labels_, cupy_km.labels_)
        assert ari >= 0.80, f"ARI={ari:.4f} — CuPy diverged from sklearn"

    def test_inertia_comparable_to_sklearn(self):
        """CuPy inertia should be within reasonable range of sklearn."""
        X, _ = make_blobs(n_samples=2000, centers=4, cluster_std=1.0, random_state=SEED)
        X = X.astype(np.float32)

        sk = SklearnKMeans(
            n_clusters=4, init="k-means++", n_init=1, max_iter=300, random_state=SEED
        )
        sk.fit(X)

        cupy_km = KMeansCupy(n_clusters=4, max_iter=300, random_state=SEED)
        cupy_km.fit(X)

        def compute_inertia(X, labels, centers):
            return sum(
                np.dot(X[i] - centers[labels[i]], X[i] - centers[labels[i]])
                for i in range(len(X))
            )

        sk_inertia = compute_inertia(X, sk.labels_, sk.cluster_centers_)
        cupy_inertia = compute_inertia(X, cupy_km.labels_, cupy_km.cluster_centers_)
        ratio = cupy_inertia / sk_inertia
        assert 0.8 <= ratio <= 1.5, f"Inertia ratio={ratio:.4f} out of range"


class TestConvergenceAndStability:
    """Test convergence behavior and numerical stability."""

    def test_convergence(self):
        """Short run and long run should produce similar results."""
        X, _ = make_blobs(n_samples=1000, centers=3, cluster_std=0.5, random_state=SEED)
        X = X.astype(np.float32)

        km5 = KMeansCupy(n_clusters=3, max_iter=5, random_state=SEED)
        km5.fit(X)

        km300 = KMeansCupy(n_clusters=3, max_iter=300, random_state=SEED)
        km300.fit(X)

        ari = adjusted_rand_score(km5.labels_, km300.labels_)
        assert ari >= 0.90, f"ARI={ari:.4f} — poor convergence"

    def test_reproducibility_same_seed(self):
        """Same seed should produce identical results."""
        X, _ = make_blobs(n_samples=1000, centers=4, random_state=SEED)
        X = X.astype(np.float32)

        km1 = KMeansCupy(n_clusters=4, max_iter=100, random_state=123)
        km1.fit(X)

        km2 = KMeansCupy(n_clusters=4, max_iter=100, random_state=123)
        km2.fit(X)

        assert np.array_equal(km1.labels_, km2.labels_), "Labels differ with same seed"
        np.testing.assert_allclose(
            km1.cluster_centers_, km2.cluster_centers_, atol=1e-5
        )

    def test_atomicadd_accuracy(self):
        """atomicAdd accumulation should produce accurate center estimates."""
        rng = np.random.RandomState(SEED)
        centers_true = np.array([[0, 0], [10, 10], [20, 20]], dtype=np.float32)

        X_parts = [
            rng.randn(1000, 2).astype(np.float32) * 0.01 + c for c in centers_true
        ]
        X = np.vstack(X_parts)

        km = KMeansCupy(n_clusters=3, max_iter=300, random_state=SEED)
        km.fit(X)

        from scipy.spatial.distance import cdist

        dists = cdist(km.cluster_centers_, centers_true)
        max_err = 0.0
        for _ in range(3):
            idx = np.unravel_index(np.argmin(dists), dists.shape)
            max_err = max(max_err, dists[idx])
            dists[idx[0], :] = np.inf
            dists[:, idx[1]] = np.inf

        assert max_err < 0.5, f"Max center error={max_err:.6f} too large"


class TestDtypeHandling:
    """Test float32, float64, and integer input handling."""

    def test_float64_clustering(self):
        """Float64 input should work correctly."""
        X, true_labels = make_blobs(
            n_samples=1000, centers=3, cluster_std=1.0, random_state=SEED
        )
        X = X.astype(np.float64)

        km = KMeansCupy(n_clusters=3, max_iter=300, random_state=SEED)
        km.fit(X)

        ari = adjusted_rand_score(true_labels, km.labels_)
        assert ari >= 0.85, f"Float64 ARI={ari:.4f} too low"
        assert km.cluster_centers_.dtype == np.float64

    def test_integer_input_autocast(self):
        """Integer input should be auto-cast to float32."""
        X, _ = make_blobs(n_samples=500, centers=3, random_state=SEED)
        X_int = (X * 10).astype(np.int32)

        km = KMeansCupy(n_clusters=3, max_iter=300, random_state=SEED)
        km.fit(X_int)

        assert km.cluster_centers_.dtype == np.float32

    def test_non_contiguous_input(self):
        """Fortran-order input should be handled correctly."""
        X, true_labels = make_blobs(n_samples=1000, centers=3, random_state=SEED)
        X_f = np.asfortranarray(X.astype(np.float32))
        assert not X_f.flags.c_contiguous

        km = KMeansCupy(n_clusters=3, max_iter=300, random_state=SEED)
        km.fit(X_f)

        ari = adjusted_rand_score(true_labels, km.labels_)
        assert ari >= 0.85, f"Non-contiguous ARI={ari:.4f} too low"


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_cluster(self):
        """Single cluster should assign all labels to 0."""
        X = np.random.RandomState(SEED).randn(500, 3).astype(np.float32)

        km = KMeansCupy(n_clusters=1, max_iter=100, random_state=SEED)
        km.fit(X)

        assert np.all(km.labels_ == 0)
        center_error = np.linalg.norm(km.cluster_centers_[0] - X.mean(axis=0))
        assert center_error < 0.5, f"Center error={center_error:.6f}"

    def test_two_well_separated_clusters(self):
        """Two well-separated clusters should be perfectly identified."""
        rng = np.random.RandomState(SEED)
        c1 = rng.randn(500, 2).astype(np.float32) + np.array([10, 10], dtype=np.float32)
        c2 = rng.randn(500, 2).astype(np.float32) + np.array(
            [-10, -10], dtype=np.float32
        )
        X = np.vstack([c1, c2])
        true_labels = np.array([0] * 500 + [1] * 500)

        km = KMeansCupy(n_clusters=2, max_iter=300, random_state=SEED)
        km.fit(X)

        ari = adjusted_rand_score(true_labels, km.labels_)
        assert ari >= 0.99, f"ARI={ari:.4f} — clusters not perfectly separated"

    def test_labels_range(self):
        """All labels should be in [0, n_clusters)."""
        X, _ = make_blobs(n_samples=2000, centers=6, random_state=SEED)
        X = X.astype(np.float32)

        km = KMeansCupy(n_clusters=6, max_iter=300, random_state=SEED)
        km.fit(X)

        assert np.all((km.labels_ >= 0) & (km.labels_ < 6))
        assert len(km.labels_) == 2000

    def test_empty_cluster_handling(self):
        """More clusters than natural groups should not crash or produce NaN."""
        rng = np.random.RandomState(SEED)
        c1 = rng.randn(500, 2).astype(np.float32) * 0.1 + np.array(
            [5, 5], dtype=np.float32
        )
        c2 = rng.randn(500, 2).astype(np.float32) * 0.1 + np.array(
            [-5, -5], dtype=np.float32
        )
        X = np.vstack([c1, c2])

        km = KMeansCupy(n_clusters=5, max_iter=300, random_state=SEED)
        km.fit(X)

        assert not np.any(np.isnan(km.cluster_centers_))
        assert not np.any(np.isnan(km.labels_))


class TestHighDimensional:
    """Stress test CUDA kernels with large dimensions."""

    def test_d200(self):
        """d=200 should cluster correctly."""
        X, true_labels = make_blobs(
            n_samples=5000,
            centers=10,
            n_features=200,
            cluster_std=3.0,
            random_state=SEED,
        )
        X = X.astype(np.float32)

        km = KMeansCupy(n_clusters=10, max_iter=300, random_state=SEED)
        km.fit(X)

        ari = adjusted_rand_score(true_labels, km.labels_)
        assert ari >= 0.70, f"ARI={ari:.4f} too low for d=200"
        assert km.cluster_centers_.shape == (10, 200)

    def test_d500(self):
        """d=500 should cluster correctly with no NaN/Inf."""
        X, true_labels = make_blobs(
            n_samples=2000,
            centers=5,
            n_features=500,
            cluster_std=5.0,
            random_state=SEED,
        )
        X = X.astype(np.float32)

        km = KMeansCupy(n_clusters=5, max_iter=100, random_state=SEED)
        km.fit(X)

        ari = adjusted_rand_score(true_labels, km.labels_)
        assert ari >= 0.60, f"ARI={ari:.4f} too low for d=500"
        assert np.all(np.isfinite(km.cluster_centers_))


class TestNInit:
    """Test n_init parameter for multiple initialization runs."""

    def test_n_init_improves_or_matches_quality(self):
        """n_init=10 should produce equal or better inertia than n_init=1."""
        X, true_labels = make_blobs(
            n_samples=3000, centers=5, cluster_std=0.8, random_state=SEED
        )
        X = X.astype(np.float32)

        km1 = KMeansCupy(n_clusters=5, n_init=1, max_iter=300, random_state=SEED)
        km1.fit(X)

        km10 = KMeansCupy(n_clusters=5, n_init=10, max_iter=300, random_state=SEED)
        km10.fit(X)

        assert (
            km10.inertia_ <= km1.inertia_ + 1e-6
        ), f"n_init=10 inertia ({km10.inertia_:.1f}) > n_init=1 ({km1.inertia_:.1f})"
        assert km10.inertia_ is not None and km10.inertia_ > 0

    def test_n_init_backward_compat(self):
        """n_init=1 (explicit) should produce identical results to default."""
        X, _ = make_blobs(n_samples=1000, centers=4, random_state=SEED)
        X = X.astype(np.float32)

        km_explicit = KMeansCupy(n_clusters=4, n_init=1, max_iter=100, random_state=123)
        km_explicit.fit(X)

        km_default = KMeansCupy(n_clusters=4, max_iter=100, random_state=123)
        km_default.fit(X)

        assert np.array_equal(km_explicit.labels_, km_default.labels_)
        np.testing.assert_allclose(
            km_explicit.cluster_centers_, km_default.cluster_centers_, atol=1e-5
        )

    def test_n_init_reproducibility(self):
        """n_init > 1 with same seed should produce near-identical results."""
        X, _ = make_blobs(n_samples=1000, centers=4, random_state=SEED)
        X = X.astype(np.float32)

        km1 = KMeansCupy(n_clusters=4, n_init=5, max_iter=100, random_state=42)
        km1.fit(X)

        km2 = KMeansCupy(n_clusters=4, n_init=5, max_iter=100, random_state=42)
        km2.fit(X)

        # atomicAdd float32 is non-deterministic in accumulation order,
        # so check ARI and inertia rather than exact equality
        assert (
            abs(km1.inertia_ - km2.inertia_) < 1.0
        ), f"Inertia diff={abs(km1.inertia_ - km2.inertia_):.6f}"
        ari = adjusted_rand_score(km1.labels_, km2.labels_)
        assert ari >= 0.99, f"ARI between runs={ari:.4f}"


class TestSharedMemoryKernel:
    """Test shared memory CUDA kernel optimization."""

    def test_shmem_path_correct(self):
        """Small centers should use shared memory and cluster correctly."""
        X, true_labels = make_blobs(
            n_samples=2000, centers=5, n_features=10, cluster_std=1.0, random_state=SEED
        )
        X = X.astype(np.float32)

        km = KMeansCupy(n_clusters=5, max_iter=300, random_state=SEED)
        # 5 * 10 * 4 = 200 bytes << 48KB — shmem path guaranteed
        km.fit(X)

        ari = adjusted_rand_score(true_labels, km.labels_)
        assert ari >= 0.60, f"Shared memory kernel ARI={ari:.4f} too low"

    def test_global_memory_fallback(self):
        """Large centers should fall back to global memory without crash."""
        # 100 * 500 * 4 = 200KB > 48KB — forces global memory path
        X, _ = make_blobs(
            n_samples=5000,
            centers=20,
            n_features=500,
            cluster_std=5.0,
            random_state=SEED,
        )
        X = X.astype(np.float32)

        km = KMeansCupy(n_clusters=100, max_iter=50, random_state=SEED)
        km.fit(X)

        assert not np.any(np.isnan(km.cluster_centers_))
        assert np.all(np.isfinite(km.cluster_centers_))
