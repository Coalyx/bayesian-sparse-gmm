import os
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces, fetch_20newsgroups_vectorized
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, v_measure_score
from sklearn.model_selection import train_test_split

from bayesian_sparse_gmm.model import BayesianSparseGMM
from bayesian_sparse_gmm.diagnostics import gelman_rubin, effective_sample_size

def run_olivetti_benchmark():
    """Evaluate on Olivetti Faces (p=4096, K=40)."""
    print("\n" + "=" * 60)
    print("OLIVETTI FACES BENCHMARK (n=400, p=4096, K=40)")
    print("=" * 60)

    faces = fetch_olivetti_faces(shuffle=True, random_state=42)
    X, y = faces.data, faces.target

    p = X.shape[1]
    X_scaled = X / np.sqrt(p)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {X_train.shape} | Test: {X_test.shape} | Classes: {len(np.unique(y))}")

    # --- BSGMM ---
    t0 = time.time()
    gmm = BayesianSparseGMM(
        K_max=40,
        n_iter=500,
        burn_in=200,
        thinning=2,
        lambda_0=1000.0,
        lambda_1=0.1,
        backend="numba",
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

    print(f"\nBSGMM  ({bsgmm_time:.1f}s) - ARI: {ari_b:.4f} | AMI: {ami_b:.4f} | V: {vm_b:.4f}")
    print(f"KMeans ({kmeans_time:.1f}s) - ARI: {ari_k:.4f} | AMI: {ami_k:.4f} | V: {vm_k:.4f}")

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
    rhat_g = gelman_rubin(trace["gamma"])
    ess_g = effective_sample_size(trace["gamma"])
    print(f"\nTheta  - R-hat: {rhat_t:.4f} | ESS: {ess_t:.1f}")
    print(f"Gamma  - R-hat mean: {np.mean(rhat_g):.4f} max: {np.max(rhat_g):.4f}")
    print(f"         ESS   mean: {np.mean(ess_g):.1f} min: {np.min(ess_g):.1f}")

    # --- Visualization ---
    os.makedirs("./visualize", exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    prob_img = gmm.feature_probabilities_.reshape(64, 64)
    im0 = axes[0].imshow(prob_img, cmap="hot", interpolation="nearest")
    axes[0].set_title("P(gamma=1 | X)")
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


def run_text_benchmark():
    """Evaluate on 20 Newsgroups text data (LSA-reduced)."""
    print("\n" + "=" * 60)
    print("20 NEWSGROUPS TEXT BENCHMARK (LSA-reduced)")
    print("=" * 60)

    newsgroups = fetch_20newsgroups_vectorized(subset='all')
    X_sparse, y = newsgroups.data, newsgroups.target

    # LSA dimensionality reduction (preserves Gaussian-like structure)
    svd = TruncatedSVD(n_components=100, random_state=42)
    X_lsa = svd.fit_transform(X_sparse)
    X_lsa = normalize(X_lsa)  # L2-normalize rows
    print(f"LSA shape: {X_lsa.shape} | Topics: {len(np.unique(y))}")

    # --- BSGMM ---
    t0 = time.time()
    gmm = BayesianSparseGMM(
        K_max=25,
        n_iter=300,
        burn_in=100,
        thinning=2,
        lambda_0=500.0,
        lambda_1=0.1,
        backend="numba",
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
    print(f"\nBSGMM  ({bsgmm_time:.1f}s) - ARI: {ari_b:.4f} | AMI: {ami_b:.4f} | Features: {n_sel}/100")
    print(f"KMeans           - ARI: {ari_k:.4f} | AMI: {ami_k:.4f}")


if __name__ == "__main__":
    run_olivetti_benchmark()
    run_text_benchmark()
