import argparse
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import (
    fetch_20newsgroups_vectorized,
    fetch_olivetti_faces,
    load_digits,
    load_wine,
)
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    v_measure_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from bayesian_sparse_gmm.diagnostics import effective_sample_size, gelman_rubin
from bayesian_sparse_gmm.model import BayesianSparseGMM


def run_olivetti_benchmark(backend="numba"):
    """Evaluate on Olivetti Faces (p=4096, K=40)."""
    print("\n" + "=" * 60)
    print("OLIVETTI FACES BENCHMARK (n=400, p=4096, K=40)")
    print("=" * 60)

    faces = fetch_olivetti_faces(shuffle=True, random_state=42)
    X, y = faces.data, faces.target
    # Standardize: identity covariance requires unit-variance features
    X_scaled = StandardScaler().fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    print(
        f"Train: {X_train.shape} | Test: {X_test.shape} | Classes: {len(np.unique(y))}"
    )

    # --- BSGMM ---
    t0 = time.time()
    p = X_train.shape[1]
    gmm = BayesianSparseGMM(
        K_max=40,
        n_iter=500,
        burn_in=200,
        thinning=2,
        lambda_0=100.0,
        lambda_1=1.0,
        alpha=1.0,
        theta=0.5,
        a_sigma=1.0,
        b_sigma=1.0,
        backend=backend,
        random_state=42,
        verbose=1,
    )
    gmm.fit(X_train)
    bsgmm_time = time.time() - t0

    # --- KMeans baseline ---
    t0 = time.time()
    kmeans = KMeans(n_clusters=40, random_state=42, n_init=10)
    kmeans.fit(X_train)
    kmeans_time = time.time() - t0

    # --- Clustering metrics ---
    ari_b = adjusted_rand_score(y_train, gmm.labels_)
    ari_k = adjusted_rand_score(y_train, kmeans.labels_)
    ami_b = adjusted_mutual_info_score(y_train, gmm.labels_)
    ami_k = adjusted_mutual_info_score(y_train, kmeans.labels_)
    vm_b = v_measure_score(y_train, gmm.labels_)
    vm_k = v_measure_score(y_train, kmeans.labels_)

    print(
        f"\nBSGMM  ({bsgmm_time:.1f}s) - ARI: {ari_b:.4f} | AMI: {ami_b:.4f} | V: {vm_b:.4f}"
    )
    print(
        f"KMeans ({kmeans_time:.1f}s) - ARI: {ari_k:.4f} | AMI: {ami_k:.4f} | V: {vm_k:.4f}"
    )

    # --- Probabilistic fit ---
    test_ll = gmm.score(X_test)
    print(f"\nTest Log-Likelihood: {test_ll:.4f}")

    # --- Feature selection ---
    n_sel = len(gmm.selected_features_)
    p_total = X.shape[1]
    print(f"Features kept: {n_sel}/{p_total} ({n_sel/p_total:.2%})")

    # --- Convergence ---
    trace = gmm.trace_
    rhat_t = gelman_rubin(trace["theta"])
    ess_t = effective_sample_size(trace["theta"])
    rhat_g = gelman_rubin(trace["xi"])
    ess_g = effective_sample_size(trace["xi"])
    print(f"\nTheta  - R-hat: {rhat_t:.4f} | ESS: {ess_t:.1f}")
    print(f"Xi     - R-hat mean: {np.mean(rhat_g):.4f} max: {np.max(rhat_g):.4f}")
    print(f"         ESS   mean: {np.mean(ess_g):.1f} min: {np.min(ess_g):.1f}")

    # --- Visualization ---
    os.makedirs("./visualize", exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    prob_img = gmm.feature_probabilities_.reshape(64, 64)
    im0 = axes[0].imshow(prob_img, cmap="hot", interpolation="nearest")
    axes[0].set_title("P(xi=1 | X)")
    fig.colorbar(im0, ax=axes[0], fraction=0.046)

    mask_img = (gmm.feature_probabilities_ > 0.5).reshape(64, 64).astype(float)
    axes[1].imshow(mask_img, cmap="gray", interpolation="nearest")
    axes[1].set_title(f"Selected Mask ({n_sel}/{p_total})")

    axes[2].imshow(faces.images[0] * mask_img, cmap="gray", interpolation="nearest")
    axes[2].set_title("Face x Mask Overlay")

    plt.tight_layout()
    plt.savefig("./visualize/olivetti_features.png", dpi=150)
    plt.close()
    print("Saved './visualize/olivetti_features.png'")


def run_text_benchmark(backend="numba"):
    """Evaluate on 20 Newsgroups text data (LSA-reduced)."""
    print("\n" + "=" * 60)
    print("20 NEWSGROUPS TEXT BENCHMARK (LSA-reduced)")
    print("=" * 60)

    newsgroups = fetch_20newsgroups_vectorized(subset="all")
    X_sparse, y = newsgroups.data, newsgroups.target

    # LSA dimensionality reduction (preserves Gaussian-like structure)
    svd = TruncatedSVD(n_components=100, random_state=42)
    X_lsa = svd.fit_transform(X_sparse)
    # Standardize LSA features to ensure unit variance per dimension
    X_lsa = StandardScaler().fit_transform(X_lsa)
    print(f"LSA shape: {X_lsa.shape} | Topics: {len(np.unique(y))}")

    # --- BSGMM ---
    t0 = time.time()
    gmm = BayesianSparseGMM(
        K_max=25,
        n_iter=300,
        burn_in=100,
        thinning=2,
        lambda_0=100.0,
        lambda_1=1.0,
        alpha=1.0,
        theta=0.5,
        a_sigma=1.0,
        b_sigma=1.0,
        backend=backend,
        random_state=42,
        verbose=1,
    )
    gmm.fit(X_lsa)
    bsgmm_time = time.time() - t0

    # --- KMeans ---
    kmeans = KMeans(n_clusters=20, random_state=42, n_init=10)
    kmeans.fit(X_lsa)

    ari_b = adjusted_rand_score(y, gmm.labels_)
    ari_k = adjusted_rand_score(y, kmeans.labels_)
    ami_b = adjusted_mutual_info_score(y, gmm.labels_)
    ami_k = adjusted_mutual_info_score(y, kmeans.labels_)

    n_sel = len(gmm.selected_features_)
    print(
        f"\nBSGMM  ({bsgmm_time:.1f}s) - ARI: {ari_b:.4f} | AMI: {ami_b:.4f} | Features: {n_sel}/100"
    )
    print(f"KMeans           - ARI: {ari_k:.4f} | AMI: {ami_k:.4f}")


def run_synthetic_sparse_benchmark(backend="numba"):
    """Synthetic data: 6 clusters in 60-dim space, only 10/60 features are informative."""
    print("\n" + "=" * 60)
    print("SYNTHETIC SPARSE SIGNAL BENCHMARK (n=600, p=60, K=6, signal=10)")
    print("=" * 60)

    rng_d = np.random.default_rng(0)
    K_true, n_per_k, p_total, p_signal = 6, 100, 60, 10
    means = np.zeros((K_true, p_total))
    means[:, :p_signal] = rng_d.normal(0, 3.0, size=(K_true, p_signal))
    X_parts = [
        rng_d.normal(means[k], 1.0, size=(n_per_k, p_total)) for k in range(K_true)
    ]
    y_parts = [np.full(n_per_k, k) for k in range(K_true)]
    X_raw = np.vstack(X_parts)
    y = np.hstack(y_parts)
    shuf = rng_d.permutation(len(y))
    X_raw, y = X_raw[shuf], y[shuf]
    print(
        f"Data: {X_raw.shape} | True K={K_true} | Signal={p_signal}/{p_total} features"
    )

    X = StandardScaler().fit_transform(X_raw)

    t0 = time.time()
    gmm = BayesianSparseGMM(
        K_max=10,
        n_iter=500,
        burn_in=100,
        thinning=1,
        lambda_0=100.0,
        lambda_1=1.0,
        alpha=1.0,
        theta=0.5,
        a_sigma=1.0,
        b_sigma=1.0,
        backend=backend,
        random_state=42,
        verbose=1,
    )
    gmm.fit(X)
    bsgmm_time = time.time() - t0

    t0 = time.time()
    km = KMeans(n_clusters=K_true, random_state=42, n_init=10)
    km.fit(X)
    km_time = time.time() - t0

    ari_b = adjusted_rand_score(y, gmm.labels_)
    ari_k = adjusted_rand_score(y, km.labels_)
    ami_b = adjusted_mutual_info_score(y, gmm.labels_)
    n_sel = len(gmm.selected_features_)
    true_sig = set(range(p_signal))
    sel = set(gmm.selected_features_)
    prec = len(sel & true_sig) / max(n_sel, 1)
    rec = len(sel & true_sig) / p_signal
    print(
        f"\nBSGMM  ({bsgmm_time:.1f}s) - ARI: {ari_b:.4f} | AMI: {ami_b:.4f} | Features: {n_sel}/{p_total}"
    )
    print(
        f"KMeans ({km_time:.1f}s) - ARI: {ari_k:.4f} | AMI: {adjusted_mutual_info_score(y, km.labels_):.4f}"
    )
    print(f"Feature Selection  - Precision: {prec:.2%} | Recall: {rec:.2%}")

    X_2d = PCA(n_components=2, random_state=42).fit_transform(X)
    pal = plt.cm.tab10(np.linspace(0, 0.9, 10))
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Synthetic Sparse Signal Benchmark", fontsize=13, fontweight="bold")

    for k in range(K_true):
        m = y == k
        axes[0].scatter(
            X_2d[m, 0], X_2d[m, 1], c=[pal[k]], s=18, alpha=0.75, label=f"C{k}"
        )
    axes[0].set_title(f"Ground Truth (K={K_true})")
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    axes[0].legend(fontsize=7, ncol=2)

    for idx, k in enumerate(np.unique(gmm.labels_)):
        m = gmm.labels_ == k
        axes[1].scatter(X_2d[m, 0], X_2d[m, 1], c=[pal[idx % 10]], s=18, alpha=0.75)
    axes[1].set_title(f"BSGMM (ARI={ari_b:.3f}, K={len(np.unique(gmm.labels_))})")
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")

    bar_c = ["#e74c3c" if i in true_sig else "#bdc3c7" for i in range(p_total)]
    axes[2].bar(range(p_total), gmm.feature_probabilities_, color=bar_c, alpha=0.85)
    axes[2].axhline(0.5, color="k", ls="--", lw=1, label="threshold")
    axes[2].set_xlabel("Feature index")
    axes[2].set_ylabel("P(ξ=1|X)")
    axes[2].set_title(
        f"Feature Importance (red=signal | Prec={prec:.0%}, Rec={rec:.0%})"
    )
    axes[2].legend(fontsize=9)

    plt.tight_layout()
    os.makedirs("./visualize", exist_ok=True)
    plt.savefig("./visualize/synthetic_sparse.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved './visualize/synthetic_sparse.png'")


def run_digits_benchmark(backend="numba"):
    """Sklearn Digits: 1797 samples, 64 features (8x8 images), K=10 digit classes."""
    print("\n" + "=" * 60)
    print("SKLEARN DIGITS BENCHMARK (n=1797, p=64, K=10)")
    print("=" * 60)

    digits = load_digits()
    X_raw, y = digits.data, digits.target
    # Standardize: identity covariance requires unit-variance features
    X = StandardScaler().fit_transform(X_raw)
    print(f"Data: {X.shape} | Classes: {len(np.unique(y))}")

    t0 = time.time()
    gmm = BayesianSparseGMM(
        K_max=15,
        n_iter=600,
        burn_in=150,
        thinning=2,
        lambda_0=100.0,
        lambda_1=1.0,
        alpha=1.0,
        theta=0.5,
        a_sigma=1.0,
        b_sigma=1.0,
        backend=backend,
        random_state=42,
        verbose=1,
    )
    gmm.fit(X)
    bsgmm_time = time.time() - t0

    t0 = time.time()
    km = KMeans(n_clusters=10, random_state=42, n_init=10)
    km.fit(X)
    km_time = time.time() - t0

    ari_b = adjusted_rand_score(y, gmm.labels_)
    ari_k = adjusted_rand_score(y, km.labels_)
    ami_b = adjusted_mutual_info_score(y, gmm.labels_)
    ami_k = adjusted_mutual_info_score(y, km.labels_)
    n_sel = len(gmm.selected_features_)
    print(
        f"\nBSGMM  ({bsgmm_time:.1f}s) - ARI: {ari_b:.4f} | AMI: {ami_b:.4f} | Features: {n_sel}/64"
    )
    print(f"KMeans ({km_time:.1f}s) - ARI: {ari_k:.4f} | AMI: {ami_k:.4f}")

    X_2d = PCA(n_components=2, random_state=42).fit_transform(X)
    pal = plt.cm.tab10(np.linspace(0, 0.9, 10))

    fig = plt.figure(figsize=(18, 9))
    fig.suptitle("Sklearn Digits Benchmark", fontsize=13, fontweight="bold")

    # Top-left: PCA scatter colored by true digit
    ax1 = fig.add_subplot(2, 3, 1)
    for k in range(10):
        m = y == k
        ax1.scatter(X_2d[m, 0], X_2d[m, 1], c=[pal[k]], s=10, alpha=0.6, label=str(k))
    ax1.set_title("Ground Truth Digits")
    ax1.set_xlabel("PC1")
    ax1.set_ylabel("PC2")
    ax1.legend(fontsize=7, ncol=2, markerscale=1.5)

    # Top-middle: PCA scatter colored by BSGMM clusters
    ax2 = fig.add_subplot(2, 3, 2)
    unique_labels = np.unique(gmm.labels_)
    n_pred = len(unique_labels)
    pal2 = plt.cm.tab20(np.linspace(0, 1, max(n_pred, 2)))
    for idx, k in enumerate(unique_labels):
        m = gmm.labels_ == k
        ax2.scatter(X_2d[m, 0], X_2d[m, 1], c=[pal2[idx % 20]], s=10, alpha=0.6)
    ax2.set_title(f"BSGMM (ARI={ari_b:.3f}, K={n_pred})")
    ax2.set_xlabel("PC1")
    ax2.set_ylabel("PC2")

    # Top-right: Feature importance heatmap (8x8)
    ax3 = fig.add_subplot(2, 3, 3)
    prob_img = gmm.feature_probabilities_.reshape(8, 8)
    im = ax3.imshow(prob_img, cmap="hot", vmin=0, vmax=1)
    ax3.set_title(f"P(γ=1|X) heatmap ({n_sel}/64 active)")
    fig.colorbar(im, ax=ax3, fraction=0.046)

    # Bottom: One representative sample image per predicted cluster (up to 10)
    unique_clusters = np.unique(gmm.labels_)[:10]
    for idx, k in enumerate(unique_clusters):
        ax = fig.add_subplot(2, len(unique_clusters), len(unique_clusters) + idx + 1)
        sample_idx = np.where(gmm.labels_ == k)[0][0]
        ax.imshow(digits.images[sample_idx], cmap="gray_r", interpolation="nearest")
        ax.set_title(f"C{k}\n(true={y[sample_idx]})", fontsize=8)
        ax.axis("off")

    plt.tight_layout()
    os.makedirs("./visualize", exist_ok=True)
    plt.savefig("./visualize/digits_clusters.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved './visualize/digits_clusters.png'")


def run_wine_benchmark(backend="numba"):
    """Wine dataset: 178 samples, 13 chemical features, K=3 cultivars."""
    print("\n" + "=" * 60)
    print("WINE BENCHMARK (n=178, p=13, K=3 cultivars)")
    print("=" * 60)

    wine = load_wine()
    X_raw, y = wine.data, wine.target
    feat_names = wine.feature_names
    X = StandardScaler().fit_transform(X_raw)
    print(f"Data: {X.shape} | Classes: {len(np.unique(y))}")

    t0 = time.time()
    gmm = BayesianSparseGMM(
        K_max=6,
        n_iter=500,
        burn_in=100,
        thinning=1,
        lambda_0=100.0,
        lambda_1=1.0,
        alpha=1.0,
        theta=0.5,
        a_sigma=1.0,
        b_sigma=1.0,
        backend=backend,
        random_state=42,
        verbose=1,
    )
    gmm.fit(X)
    bsgmm_time = time.time() - t0

    t0 = time.time()
    km = KMeans(n_clusters=3, random_state=42, n_init=10)
    km.fit(X)
    km_time = time.time() - t0

    ari_b = adjusted_rand_score(y, gmm.labels_)
    ari_k = adjusted_rand_score(y, km.labels_)
    ami_b = adjusted_mutual_info_score(y, gmm.labels_)
    ami_k = adjusted_mutual_info_score(y, km.labels_)
    n_sel = len(gmm.selected_features_)
    print(
        f"\nBSGMM  ({bsgmm_time:.1f}s) - ARI: {ari_b:.4f} | AMI: {ami_b:.4f} | Features: {n_sel}/13"
    )
    print(f"KMeans ({km_time:.1f}s) - ARI: {ari_k:.4f} | AMI: {ami_k:.4f}")
    print(f"Selected features: {[feat_names[i] for i in gmm.selected_features_]}")

    X_2d = PCA(n_components=2, random_state=42).fit_transform(X)
    pal = plt.cm.Set1(np.linspace(0, 0.6, 3))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Wine Dataset Benchmark (3 cultivars)", fontsize=13, fontweight="bold")

    # Panel 1: Ground truth PCA
    for k in range(3):
        m = y == k
        axes[0].scatter(
            X_2d[m, 0], X_2d[m, 1], c=[pal[k]], s=40, alpha=0.8, label=f"Cultivar {k+1}"
        )
    axes[0].set_title(f"Ground Truth (K=3)")
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    axes[0].legend()

    # Panel 2: BSGMM predicted
    unique_labels_w = np.unique(gmm.labels_)
    n_pred = len(unique_labels_w)
    pal2 = plt.cm.Set2(np.linspace(0, 0.7, max(n_pred, 2)))
    for idx, k in enumerate(unique_labels_w):
        m = gmm.labels_ == k
        axes[1].scatter(
            X_2d[m, 0],
            X_2d[m, 1],
            c=[pal2[idx % len(pal2)]],
            s=40,
            alpha=0.8,
            label=f"C{k}",
        )
    axes[1].set_title(f"BSGMM (ARI={ari_b:.3f}, K={n_pred})")
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")
    axes[1].legend()

    # Panel 3: Feature importance bar chart
    sorted_idx = np.argsort(gmm.feature_probabilities_)[::-1]
    bar_c = [
        "#e74c3c" if i in gmm.selected_features_ else "#bdc3c7" for i in sorted_idx
    ]
    axes[2].barh(
        range(len(feat_names)),
        gmm.feature_probabilities_[sorted_idx],
        color=bar_c,
        alpha=0.85,
    )
    axes[2].set_yticks(range(len(feat_names)))
    axes[2].set_yticklabels([feat_names[i] for i in sorted_idx], fontsize=8)
    axes[2].axvline(0.5, color="k", ls="--", lw=1)
    axes[2].set_xlabel("P(ξ=1|X)")
    axes[2].set_title(f"Feature Importance ({n_sel}/13 selected)")

    plt.tight_layout()
    os.makedirs("./visualize", exist_ok=True)
    plt.savefig("./visualize/wine_clusters.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved './visualize/wine_clusters.png'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bayesian Sparse GMM benchmarks")
    parser.add_argument(
        "--backend",
        default="numba",
        choices=["numpy", "numba", "cuda", "auto"],
        help="Compute backend (default: numba)",
    )
    args = parser.parse_args()

    run_olivetti_benchmark(backend=args.backend)
    run_text_benchmark(backend=args.backend)
    run_synthetic_sparse_benchmark(backend=args.backend)
    run_digits_benchmark(backend=args.backend)
    run_wine_benchmark(backend=args.backend)
