import argparse
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
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

    # --- DBSCAN baseline ---
    t0 = time.time()
    # eps=20.0 is a heuristic for 4096-dim scaled data, often needs tuning
    dbscan = DBSCAN(eps=20.0, min_samples=3)
    dbscan.fit(X_train)
    dbscan_time = time.time() - t0

    # --- BSGMM (SVI) ---
    t0 = time.time()
    gmm_svi = BayesianSparseGMM(
        K_max=40,
        optimizer="svi",
        epochs=50,
        batch_size=128,
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
    gmm_svi.fit(X_train)
    svi_time = time.time() - t0

    # --- Clustering metrics ---
    ari_b = adjusted_rand_score(y_train, gmm.labels_)
    ari_s = adjusted_rand_score(y_train, gmm_svi.labels_)
    ari_k = adjusted_rand_score(y_train, kmeans.labels_)
    ari_d = adjusted_rand_score(y_train, dbscan.labels_)
    
    ami_b = adjusted_mutual_info_score(y_train, gmm.labels_)
    ami_s = adjusted_mutual_info_score(y_train, gmm_svi.labels_)
    ami_k = adjusted_mutual_info_score(y_train, kmeans.labels_)
    ami_d = adjusted_mutual_info_score(y_train, dbscan.labels_)
    
    vm_b = v_measure_score(y_train, gmm.labels_)
    vm_s = v_measure_score(y_train, gmm_svi.labels_)
    vm_k = v_measure_score(y_train, kmeans.labels_)
    vm_d = v_measure_score(y_train, dbscan.labels_)

    print(
        f"\nBSGMM(MCMC)({bsgmm_time:.1f}s) - ARI: {ari_b:.4f} | AMI: {ami_b:.4f} | V: {vm_b:.4f}"
    )
    print(
        f"BSGMM(SVI) ({svi_time:.1f}s) - ARI: {ari_s:.4f} | AMI: {ami_s:.4f} | V: {vm_s:.4f}"
    )
    print(
        f"KMeans     ({kmeans_time:.1f}s) - ARI: {ari_k:.4f} | AMI: {ami_k:.4f} | V: {vm_k:.4f}"
    )
    print(
        f"DBSCAN     ({dbscan_time:.1f}s) - ARI: {ari_d:.4f} | AMI: {ami_d:.4f} | V: {vm_d:.4f} | Clusters: {len(np.unique(dbscan.labels_))}"
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

    # 1. Feature Selection Plot
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    prob_img = gmm.feature_probabilities_.reshape(64, 64)
    im0 = axes[0].imshow(prob_img, cmap="hot", interpolation="nearest")
    axes[0].set_title(f"MCMC P(xi=1|X) ({n_sel}/{p_total})")
    fig.colorbar(im0, ax=axes[0], fraction=0.046)
    
    prob_img_svi = gmm_svi.feature_probabilities_.reshape(64, 64)
    im1 = axes[1].imshow(prob_img_svi, cmap="hot", interpolation="nearest")
    axes[1].set_title(f"SVI P(xi=1|X)")
    fig.colorbar(im1, ax=axes[1], fraction=0.046)
    
    mask_img = (gmm.feature_probabilities_ > 0.5).reshape(64, 64).astype(float)
    axes[2].imshow(mask_img, cmap="gray", interpolation="nearest")
    axes[2].set_title(f"MCMC Selected Mask")

    axes[3].imshow(faces.images[0] * mask_img, cmap="gray", interpolation="nearest")
    axes[3].set_title("Face x MCMC Mask Overlay")

    plt.tight_layout()
    plt.savefig("./visualize/olivetti_features.png", dpi=150)
    plt.close()
    
    # 2. Clusters Plot (PCA 2D)
    X_2d = PCA(n_components=2, random_state=42).fit_transform(X_train)
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    fig.suptitle("Olivetti Faces Clusters (PCA 2D Projection)", fontsize=14, fontweight="bold")
    
    pal = plt.cm.tab20(np.linspace(0, 1, 40))
    
    # Ground Truth
    for k in np.unique(y_train):
        m = y_train == k
        axes[0].scatter(X_2d[m, 0], X_2d[m, 1], c=[pal[k % 40]], s=20, alpha=0.7)
    axes[0].set_title("Ground Truth")
    
    # KMeans
    for k in np.unique(kmeans.labels_):
        m = kmeans.labels_ == k
        axes[1].scatter(X_2d[m, 0], X_2d[m, 1], c=[pal[k % 40]], s=20, alpha=0.7)
    axes[1].set_title(f"KMeans (ARI={ari_k:.3f})")
    
    # DBSCAN
    for k in np.unique(dbscan.labels_):
        m = dbscan.labels_ == k
        c = 'k' if k == -1 else pal[k % 40]
        axes[2].scatter(X_2d[m, 0], X_2d[m, 1], c=[c], s=20, alpha=0.7)
    axes[2].set_title(f"DBSCAN (ARI={ari_d:.3f})")
    
    # BSGMM MCMC
    for k in np.unique(gmm.labels_):
        m = gmm.labels_ == k
        axes[3].scatter(X_2d[m, 0], X_2d[m, 1], c=[pal[k % 40]], s=20, alpha=0.7)
    axes[3].set_title(f"BSGMM MCMC (ARI={ari_b:.3f})")
    
    # BSGMM SVI
    for k in np.unique(gmm_svi.labels_):
        m = gmm_svi.labels_ == k
        axes[4].scatter(X_2d[m, 0], X_2d[m, 1], c=[pal[k % 40]], s=20, alpha=0.7)
    axes[4].set_title(f"BSGMM SVI (ARI={ari_s:.3f})")
    
    plt.tight_layout()
    plt.savefig("./visualize/olivetti_clusters.png", dpi=150)
    plt.close()
    print("Saved './visualize/olivetti_clusters.png' and './visualize/olivetti_features.png'")


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

    # --- BSGMM (SVI) ---
    t0 = time.time()
    gmm_svi = BayesianSparseGMM(
        K_max=25, optimizer="svi", epochs=50, batch_size=128,
        lambda_0=100.0, lambda_1=1.0, alpha=1.0, theta=0.5,
        a_sigma=1.0, b_sigma=1.0, backend=backend, random_state=42, verbose=0
    )
    gmm_svi.fit(X_lsa)
    svi_time = time.time() - t0

    # --- KMeans ---
    t0 = time.time()
    kmeans = KMeans(n_clusters=20, random_state=42, n_init=10)
    kmeans.fit(X_lsa)
    km_time = time.time() - t0
    
    # --- DBSCAN ---
    t0 = time.time()
    dbscan = DBSCAN(eps=0.5, min_samples=5) # LSA scaled, typical eps
    dbscan.fit(X_lsa)
    db_time = time.time() - t0

    ari_b = adjusted_rand_score(y, gmm.labels_)
    ari_s = adjusted_rand_score(y, gmm_svi.labels_)
    ari_k = adjusted_rand_score(y, kmeans.labels_)
    ari_d = adjusted_rand_score(y, dbscan.labels_)
    
    ami_b = adjusted_mutual_info_score(y, gmm.labels_)
    ami_s = adjusted_mutual_info_score(y, gmm_svi.labels_)
    ami_k = adjusted_mutual_info_score(y, kmeans.labels_)
    ami_d = adjusted_mutual_info_score(y, dbscan.labels_)

    n_sel_b = len(gmm.selected_features_)
    n_sel_s = len(gmm_svi.selected_features_)
    
    print(f"\nBSGMM (MCMC) ({bsgmm_time:.1f}s) - ARI: {ari_b:.4f} | AMI: {ami_b:.4f} | Feats: {n_sel_b}/100")
    print(f"BSGMM (SVI)  ({svi_time:.1f}s) - ARI: {ari_s:.4f} | AMI: {ami_s:.4f} | Feats: {n_sel_s}/100")
    print(f"KMeans       ({km_time:.1f}s) - ARI: {ari_k:.4f} | AMI: {ami_k:.4f}")
    print(f"DBSCAN       ({db_time:.1f}s) - ARI: {ari_d:.4f} | AMI: {ami_d:.4f} | Clusters: {len(np.unique(dbscan.labels_))}")


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
    gmm_svi = BayesianSparseGMM(
        K_max=10, optimizer="svi", epochs=50, batch_size=64,
        lambda_0=100.0, lambda_1=1.0, alpha=1.0, theta=0.5,
        a_sigma=1.0, b_sigma=1.0, backend=backend, random_state=42, verbose=0
    )
    gmm_svi.fit(X)
    svi_time = time.time() - t0

    t0 = time.time()
    km = KMeans(n_clusters=K_true, random_state=42, n_init=10)
    km.fit(X)
    km_time = time.time() - t0
    
    t0 = time.time()
    dbscan = DBSCAN(eps=2.0, min_samples=5)
    dbscan.fit(X)
    db_time = time.time() - t0

    ari_b = adjusted_rand_score(y, gmm.labels_)
    ari_s = adjusted_rand_score(y, gmm_svi.labels_)
    ari_k = adjusted_rand_score(y, km.labels_)
    ari_d = adjusted_rand_score(y, dbscan.labels_)
    
    ami_b = adjusted_mutual_info_score(y, gmm.labels_)
    ami_s = adjusted_mutual_info_score(y, gmm_svi.labels_)
    ami_k = adjusted_mutual_info_score(y, km.labels_)
    ami_d = adjusted_mutual_info_score(y, dbscan.labels_)
    
    n_sel_b = len(gmm.selected_features_)
    n_sel_s = len(gmm_svi.selected_features_)
    
    true_sig = set(range(p_signal))
    sel_b = set(gmm.selected_features_)
    sel_s = set(gmm_svi.selected_features_)
    
    prec_b = len(sel_b & true_sig) / max(n_sel_b, 1)
    rec_b = len(sel_b & true_sig) / p_signal
    prec_s = len(sel_s & true_sig) / max(n_sel_s, 1)
    rec_s = len(sel_s & true_sig) / p_signal
    
    print(f"\nBSGMM (MCMC) ({bsgmm_time:.1f}s) - ARI: {ari_b:.4f} | AMI: {ami_b:.4f} | Feats: {n_sel_b}/{p_total} (P:{prec_b:.0%} R:{rec_b:.0%})")
    print(f"BSGMM (SVI)  ({svi_time:.1f}s) - ARI: {ari_s:.4f} | AMI: {ami_s:.4f} | Feats: {n_sel_s}/{p_total} (P:{prec_s:.0%} R:{rec_s:.0%})")
    print(f"KMeans       ({km_time:.1f}s) - ARI: {ari_k:.4f} | AMI: {ami_k:.4f}")
    print(f"DBSCAN       ({db_time:.1f}s) - ARI: {ari_d:.4f} | AMI: {ami_d:.4f} | Clusters: {len(np.unique(dbscan.labels_))}")

    X_2d = PCA(n_components=2, random_state=42).fit_transform(X)
    pal = plt.cm.tab10(np.linspace(0, 0.9, 10))
    fig, axes = plt.subplots(1, 4, figsize=(24, 5))
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

    unique_db = np.unique(dbscan.labels_)
    pal_db = plt.cm.tab20(np.linspace(0, 1, max(len(unique_db), 2)))
    for idx, k in enumerate(unique_db):
        m = dbscan.labels_ == k
        c = 'k' if k == -1 else pal_db[idx % 20]
        axes[1].scatter(X_2d[m, 0], X_2d[m, 1], c=[c], s=18, alpha=0.75)
    axes[1].set_title(f"DBSCAN (ARI={ari_d:.3f}, K={len(unique_db)})")
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")

    for idx, k in enumerate(np.unique(gmm_svi.labels_)):
        m = gmm_svi.labels_ == k
        axes[2].scatter(X_2d[m, 0], X_2d[m, 1], c=[pal[idx % 10]], s=18, alpha=0.75)
    axes[2].set_title(f"BSGMM SVI (ARI={ari_s:.3f}, K={len(np.unique(gmm_svi.labels_))})")
    axes[2].set_xlabel("PC1")
    axes[2].set_ylabel("PC2")

    bar_c = ["#e74c3c" if i in true_sig else "#bdc3c7" for i in range(p_total)]
    axes[3].bar(range(p_total), gmm_svi.feature_probabilities_, color=bar_c, alpha=0.85)
    axes[3].axhline(0.5, color="k", ls="--", lw=1, label="threshold")
    axes[3].set_xlabel("Feature index")
    axes[3].set_ylabel("P(ξ=1|X)")
    axes[3].set_title(
        f"SVI Feature Importance (red=signal | P={prec_s:.0%}, R={rec_s:.0%})"
    )
    axes[3].legend(fontsize=9)

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
    gmm_svi = BayesianSparseGMM(
        K_max=15, optimizer="svi", epochs=50, batch_size=256,
        lambda_0=100.0, lambda_1=1.0, alpha=1.0, theta=0.5,
        a_sigma=1.0, b_sigma=1.0, backend=backend, random_state=42, verbose=0
    )
    gmm_svi.fit(X)
    svi_time = time.time() - t0

    t0 = time.time()
    km = KMeans(n_clusters=10, random_state=42, n_init=10)
    km.fit(X)
    km_time = time.time() - t0
    
    t0 = time.time()
    dbscan = DBSCAN(eps=4.0, min_samples=10)
    dbscan.fit(X)
    db_time = time.time() - t0

    ari_b = adjusted_rand_score(y, gmm.labels_)
    ari_s = adjusted_rand_score(y, gmm_svi.labels_)
    ari_k = adjusted_rand_score(y, km.labels_)
    ari_d = adjusted_rand_score(y, dbscan.labels_)
    
    ami_b = adjusted_mutual_info_score(y, gmm.labels_)
    ami_s = adjusted_mutual_info_score(y, gmm_svi.labels_)
    ami_k = adjusted_mutual_info_score(y, km.labels_)
    ami_d = adjusted_mutual_info_score(y, dbscan.labels_)
    
    n_sel_b = len(gmm.selected_features_)
    n_sel_s = len(gmm_svi.selected_features_)
    
    print(f"\nBSGMM (MCMC) ({bsgmm_time:.1f}s) - ARI: {ari_b:.4f} | AMI: {ami_b:.4f} | Feats: {n_sel_b}/64")
    print(f"BSGMM (SVI)  ({svi_time:.1f}s) - ARI: {ari_s:.4f} | AMI: {ami_s:.4f} | Feats: {n_sel_s}/64")
    print(f"KMeans       ({km_time:.1f}s) - ARI: {ari_k:.4f} | AMI: {ami_k:.4f}")
    print(f"DBSCAN       ({db_time:.1f}s) - ARI: {ari_d:.4f} | AMI: {ami_d:.4f} | Clusters: {len(np.unique(dbscan.labels_))}")

    X_2d = PCA(n_components=2, random_state=42).fit_transform(X)
    pal = plt.cm.tab10(np.linspace(0, 0.9, 10))

    fig = plt.figure(figsize=(24, 9))
    fig.suptitle("Sklearn Digits Benchmark", fontsize=13, fontweight="bold")

    # Top-left: PCA scatter colored by true digit
    ax1 = fig.add_subplot(2, 4, 1)
    for k in range(10):
        m = y == k
        ax1.scatter(X_2d[m, 0], X_2d[m, 1], c=[pal[k]], s=10, alpha=0.6, label=str(k))
    ax1.set_title("Ground Truth Digits")
    ax1.set_xlabel("PC1")
    ax1.set_ylabel("PC2")
    ax1.legend(fontsize=7, ncol=2, markerscale=1.5)

    # Top-mid-left: DBSCAN
    ax_db = fig.add_subplot(2, 4, 2)
    unique_db = np.unique(dbscan.labels_)
    pal_db = plt.cm.tab20(np.linspace(0, 1, max(len(unique_db), 2)))
    for idx, k in enumerate(unique_db):
        m = dbscan.labels_ == k
        c = 'k' if k == -1 else pal_db[idx % 20]
        ax_db.scatter(X_2d[m, 0], X_2d[m, 1], c=[c], s=10, alpha=0.6)
    ax_db.set_title(f"DBSCAN (ARI={ari_d:.3f}, K={len(unique_db)})")
    ax_db.set_xlabel("PC1")
    ax_db.set_ylabel("PC2")

    # Top-mid-right: PCA scatter colored by BSGMM clusters
    ax2 = fig.add_subplot(2, 4, 3)
    unique_labels = np.unique(gmm_svi.labels_)
    n_pred = len(unique_labels)
    pal2 = plt.cm.tab20(np.linspace(0, 1, max(n_pred, 2)))
    for idx, k in enumerate(unique_labels):
        m = gmm_svi.labels_ == k
        ax2.scatter(X_2d[m, 0], X_2d[m, 1], c=[pal2[idx % 20]], s=10, alpha=0.6)
    ax2.set_title(f"BSGMM SVI (ARI={ari_s:.3f}, K={n_pred})")
    ax2.set_xlabel("PC1")
    ax2.set_ylabel("PC2")

    # Top-right: Feature importance heatmap (8x8)
    ax3 = fig.add_subplot(2, 4, 4)
    prob_img = gmm_svi.feature_probabilities_.reshape(8, 8)
    im = ax3.imshow(prob_img, cmap="hot", vmin=0, vmax=1)
    ax3.set_title(f"SVI P(γ=1|X) heatmap ({n_sel_s}/64 active)")
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
    gmm_svi = BayesianSparseGMM(
        K_max=6, optimizer="svi", epochs=50, batch_size=32,
        lambda_0=100.0, lambda_1=1.0, alpha=1.0, theta=0.5,
        a_sigma=1.0, b_sigma=1.0, backend=backend, random_state=42, verbose=0
    )
    gmm_svi.fit(X)
    svi_time = time.time() - t0

    t0 = time.time()
    km = KMeans(n_clusters=3, random_state=42, n_init=10)
    km.fit(X)
    km_time = time.time() - t0
    
    t0 = time.time()
    dbscan = DBSCAN(eps=2.0, min_samples=5)
    dbscan.fit(X)
    db_time = time.time() - t0

    ari_b = adjusted_rand_score(y, gmm.labels_)
    ari_s = adjusted_rand_score(y, gmm_svi.labels_)
    ari_k = adjusted_rand_score(y, km.labels_)
    ari_d = adjusted_rand_score(y, dbscan.labels_)
    
    ami_b = adjusted_mutual_info_score(y, gmm.labels_)
    ami_s = adjusted_mutual_info_score(y, gmm_svi.labels_)
    ami_k = adjusted_mutual_info_score(y, km.labels_)
    ami_d = adjusted_mutual_info_score(y, dbscan.labels_)
    
    n_sel_b = len(gmm.selected_features_)
    n_sel_s = len(gmm_svi.selected_features_)
    
    print(f"\nBSGMM (MCMC) ({bsgmm_time:.1f}s) - ARI: {ari_b:.4f} | AMI: {ami_b:.4f} | Feats: {n_sel_b}/13")
    print(f"BSGMM (SVI)  ({svi_time:.1f}s) - ARI: {ari_s:.4f} | AMI: {ami_s:.4f} | Feats: {n_sel_s}/13")
    print(f"KMeans       ({km_time:.1f}s) - ARI: {ari_k:.4f} | AMI: {ami_k:.4f}")
    print(f"DBSCAN       ({db_time:.1f}s) - ARI: {ari_d:.4f} | AMI: {ami_d:.4f} | Clusters: {len(np.unique(dbscan.labels_))}")
    print(f"MCMC Selected features: {[feat_names[i] for i in gmm.selected_features_]}")

    X_2d = PCA(n_components=2, random_state=42).fit_transform(X)
    pal = plt.cm.Set1(np.linspace(0, 0.6, 3))

    fig, axes = plt.subplots(1, 4, figsize=(24, 5))
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

    # Panel 2: DBSCAN
    unique_db = np.unique(dbscan.labels_)
    pal_db = plt.cm.Set2(np.linspace(0, 0.7, max(len(unique_db), 2)))
    for idx, k in enumerate(unique_db):
        m = dbscan.labels_ == k
        c = 'k' if k == -1 else pal_db[idx % len(pal_db)]
        axes[1].scatter(
            X_2d[m, 0], X_2d[m, 1], c=[c], s=40, alpha=0.8, label=f"C{k}" if k != -1 else "Noise",
        )
    axes[1].set_title(f"DBSCAN (ARI={ari_d:.3f}, K={len(unique_db)})")
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")
    axes[1].legend()

    # Panel 3: BSGMM (SVI) predicted
    unique_labels_w = np.unique(gmm_svi.labels_)
    n_pred = len(unique_labels_w)
    pal2 = plt.cm.Set2(np.linspace(0, 0.7, max(n_pred, 2)))
    for idx, k in enumerate(unique_labels_w):
        m = gmm_svi.labels_ == k
        axes[2].scatter(
            X_2d[m, 0], X_2d[m, 1], c=[pal2[idx % len(pal2)]], s=40, alpha=0.8, label=f"C{k}",
        )
    axes[2].set_title(f"BSGMM SVI (ARI={ari_s:.3f}, K={n_pred})")
    axes[2].set_xlabel("PC1")
    axes[2].set_ylabel("PC2")
    axes[2].legend()

    # Panel 4: Feature importance bar chart
    sorted_idx = np.argsort(gmm_svi.feature_probabilities_)[::-1]
    bar_c = [
        "#e74c3c" if i in gmm_svi.selected_features_ else "#bdc3c7" for i in sorted_idx
    ]
    axes[3].barh(
        range(len(feat_names)),
        gmm_svi.feature_probabilities_[sorted_idx],
        color=bar_c,
        alpha=0.85,
    )
    axes[3].set_yticks(range(len(feat_names)))
    axes[3].set_yticklabels([feat_names[i] for i in sorted_idx], fontsize=8)
    axes[3].axvline(0.5, color="k", ls="--", lw=1)
    axes[3].set_xlabel("P(ξ=1|X)")
    axes[3].set_title(f"SVI Feature Importance ({n_sel_s}/13 selected)")


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
