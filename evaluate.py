import os
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces, fetch_20newsgroups_vectorized
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, v_measure_score
from sklearn.model_selection import train_test_split

from bayesian_sparse_gmm.model import BayesianSparseGMM
from bayesian_sparse_gmm.diagnostics import gelman_rubin, effective_sample_size

def run_olivetti_benchmark():
    """Evaluate performance on Olivetti Faces dataset."""
    print("\n" + "=" * 50)
    print("RUNNING OLIVETTI FACES BENCHMARK")
    print("=" * 50)
    
    # Load and scale data
    faces = fetch_olivetti_faces(shuffle=True, random_state=42)
    X = faces.data
    y = faces.target
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")
    print(f"Number of classes: {len(np.unique(y))}")
    
    # Fit Bayesian Sparse GMM
    t0 = time.time()
    gmm = BayesianSparseGMM(
        K_max=40,
        n_iter=150,
        burn_in=50,
        thinning=2,
        backend="numba",
        random_state=42,
        verbose=1
    )
    gmm.fit(X_train)
    bsgmm_time = time.time() - t0
    print(f"BSGMM fitting took {bsgmm_time:.2f} seconds")
    
    # Fit KMeans Baseline
    t0 = time.time()
    kmeans = KMeans(n_clusters=40, random_state=42, n_init=10)
    kmeans.fit(X_train)
    kmeans_time = time.time() - t0
    
    # Evaluation: Clustering
    y_pred_bsgmm = gmm.labels_
    y_pred_kmeans = kmeans.labels_
    
    ari_bsgmm = adjusted_rand_score(y_train, y_pred_bsgmm)
    ari_kmeans = adjusted_rand_score(y_train, y_pred_kmeans)
    
    ami_bsgmm = adjusted_mutual_info_score(y_train, y_pred_bsgmm)
    ami_kmeans = adjusted_mutual_info_score(y_train, y_pred_kmeans)
    
    v_bsgmm = v_measure_score(y_train, y_pred_bsgmm)
    v_kmeans = v_measure_score(y_train, y_pred_kmeans)
    
    print("\n--- CLUSTERING METRICS (TRAIN) ---")
    print(f"BSGMM  - ARI: {ari_bsgmm:.4f} | AMI: {ami_bsgmm:.4f} | V-Measure: {v_bsgmm:.4f}")
    print(f"KMeans - ARI: {ari_kmeans:.4f} | AMI: {ami_kmeans:.4f} | V-Measure: {v_kmeans:.4f}")
    
    # Evaluation: Test Log-Likelihood
    test_score = gmm.score(X_test)
    print(f"\nBSGMM Test Average Log-Likelihood: {test_score:.4f}")
    
    # Evaluation: Feature Selection & Sparsity
    sparsity = len(gmm.selected_features_) / X.shape[1]
    print(f"Sparsity Ratio (features kept): {sparsity:.4f} ({len(gmm.selected_features_)} / {X.shape[1]})")
    
    # Evaluation: MCMC Convergence Diagnostics
    trace = gmm.trace_
    rhat_theta = gelman_rubin(trace["theta"])
    ess_theta = effective_sample_size(trace["theta"])
    
    rhat_gamma = gelman_rubin(trace["gamma"])
    ess_gamma = effective_sample_size(trace["gamma"])
    
    print("\n--- CONVERGENCE DIAGNOSTICS ---")
    print(f"Theta - R-hat: {rhat_theta:.4f} | ESS: {ess_theta:.1f}")
    print(f"Gamma - Mean R-hat: {np.mean(rhat_gamma):.4f} | Max R-hat: {np.max(rhat_gamma):.4f}")
    print(f"Gamma - Mean ESS: {np.mean(ess_gamma):.1f} | Min ESS: {np.min(ess_gamma):.1f}")
    
    # Visualization: Feature selection heatmap
    os.makedirs("./visualize", exist_ok=True)
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Posterior Selection Probability P(gamma=1)")
    prob_image = gmm.feature_probabilities_.reshape(64, 64)
    plt.imshow(prob_image, cmap="hot", interpolation="nearest")
    plt.colorbar()
    
    plt.subplot(1, 2, 2)
    plt.title("Selected Feature Mask (> 0.5)")
    mask_image = (gmm.feature_probabilities_ > 0.5).reshape(64, 64)
    plt.imshow(mask_image, cmap="gray", interpolation="nearest")
    plt.colorbar()
    
    plt.tight_layout()
    plt.savefig("./visualize/olivetti_features.png", dpi=150)
    plt.close()
    print("Saved feature selection heatmap to './visualize/olivetti_features.png'")

def run_text_benchmark():
    """Evaluate performance on fetch_20newsgroups dataset."""
    print("\n" + "=" * 50)
    print("RUNNING 20 NEWSGROUPS TEXT BENCHMARK")
    print("=" * 50)
    
    # Load text subset
    newsgroups = fetch_20newsgroups_vectorized(subset='all')
    X_text = newsgroups.data
    y_text = newsgroups.target
    
    # Slice to a subset
    X_sub = X_text[:1000, :5000].toarray()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_sub)
    
    print(f"Text dataset shape: {X_scaled.shape}")
    print(f"Number of topic clusters: {len(np.unique(y_text[:1000]))}")
    
    # Fit BSGMM
    t0 = time.time()
    gmm = BayesianSparseGMM(
        K_max=20,
        n_iter=100,
        burn_in=30,
        thinning=2,
        backend="numba",
        random_state=42,
        verbose=1
    )
    gmm.fit(X_scaled)
    bsgmm_time = time.time() - t0
    print(f"BSGMM fitting took {bsgmm_time:.2f} seconds")
    
    # Fit KMeans Baseline
    kmeans = KMeans(n_clusters=20, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    
    # Evaluation
    ari_bsgmm = adjusted_rand_score(y_text[:1000], gmm.labels_)
    ari_kmeans = adjusted_rand_score(y_text[:1000], kmeans.labels_)
    
    sparsity = len(gmm.selected_features_) / X_scaled.shape[1]
    
    print("\n--- TEXT BENCHMARK RESULTS ---")
    print(f"BSGMM  - ARI: {ari_bsgmm:.4f} | Sparsity: {sparsity:.4f}")
    print(f"KMeans - ARI: {ari_kmeans:.4f}")

if __name__ == "__main__":
    run_olivetti_benchmark()
    run_text_benchmark()
