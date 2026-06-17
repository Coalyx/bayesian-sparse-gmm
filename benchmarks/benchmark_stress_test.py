import time
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, v_measure_score
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler

from bayesian_sparse_gmm.model import BayesianSparseGMM

def generate_complex_arcs(n_samples=10000, n_features=1024, noise_level=0.15):
    """
    Generate a complex dataset with arc-shaped clusters mixed together, embedded in high dimensions.
    """
    rng = np.random.default_rng(42)
    # Create 4 arcs (moons) that intersect
    n_per_pair = n_samples // 2
    
    # Pair 1: Standard moons
    X1, y1 = make_moons(n_samples=n_per_pair, noise=noise_level, random_state=42)
    
    # Pair 2: Rotated and shifted moons
    X2, y2 = make_moons(n_samples=n_per_pair, noise=noise_level, random_state=43)
    y2 += 2 # Cluster labels 2 and 3
    
    # Rotate by 60 degrees
    theta = np.radians(60)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    X2 = X2 @ R
    
    # Shift to overlap with Pair 1
    X2[:, 0] += 0.3
    X2[:, 1] += 0.3
    
    X_2d = np.vstack((X1, X2))
    y = np.concatenate((y1, y2))
    
    # Embed in high-dimensional space
    X = np.zeros((n_samples, n_features))
    X[:, 0] = X_2d[:, 0]
    X[:, 1] = X_2d[:, 1]
    
    # Add a few derived non-linear signal features
    X[:, 2] = X_2d[:, 0] * X_2d[:, 1]
    X[:, 3] = X_2d[:, 0] ** 2
    X[:, 4] = X_2d[:, 1] ** 2
    
    # Add complex pure noise features
    noise_features = rng.normal(0, 1.5, size=(n_samples, n_features - 5))
    X[:, 5:] = noise_features
    
    # Randomly permute the features so the signal is not just in the first 5 dims
    feature_shuf = rng.permutation(n_features)
    X = X[:, feature_shuf]
    
    # Shuffle samples
    sample_shuf = rng.permutation(n_samples)
    X, y, X_2d = X[sample_shuf], y[sample_shuf], X_2d[sample_shuf]
    
    # Scale
    X = StandardScaler().fit_transform(X)
    
    return X, y, X_2d, feature_shuf

def run_stress_test_benchmark(backend="cuda"):
    n_samples = 10000
    n_features = 1024
    
    print(f"\n=======================================================")
    print(f"STRESS TEST BENCHMARK: ARC-SHAPED CLUSTERS")
    print(f"n={n_samples}, p={n_features}, backend={backend.upper()}")
    print(f"=======================================================\n")
    
    print("Generating complex dataset...")
    X, y, X_2d, feature_shuf = generate_complex_arcs(n_samples, n_features)
    
    # Find original signal feature indices
    signal_indices = [np.where(feature_shuf == i)[0][0] for i in range(5)]
    print(f"Signal feature indices: {signal_indices}")
    
    # --- DBSCAN ---
    # In 1024 dimensions, distance values are large. 
    # For N(0,1) variables, expected Euclidean distance is roughly sqrt(2*1024) ~ 45
    # We set a somewhat large eps and min_samples to see how it handles it.
    print("\nRunning DBSCAN ...")
    t0 = time.time()
    dbscan = DBSCAN(eps=40.0, min_samples=10) 
    dbscan.fit(X)
    t_dbscan = time.time() - t0
    
    # --- BSGMM (SVI) ---
    print("\nRunning Bayesian Sparse GMM (SVI) ...")
    t0 = time.time()
    gmm = BayesianSparseGMM(
        K_max=10,
        optimizer="svi",
        epochs=100,
        batch_size=512,
        lambda_0=1000.0,
        lambda_1=0.1,        
        theta=0.5,      
        backend=backend,
        random_state=42,
        use_identity_covariance=True,
        verbose=1
    )
    gmm.fit(X)
    t_bsgmm = time.time() - t0
    
    # --- Results ---
    ari_db = adjusted_rand_score(y, dbscan.labels_)
    ami_db = adjusted_mutual_info_score(y, dbscan.labels_)
    v_db = v_measure_score(y, dbscan.labels_)
    
    ari_gmm = adjusted_rand_score(y, gmm.labels_)
    ami_gmm = adjusted_mutual_info_score(y, gmm.labels_)
    v_gmm = v_measure_score(y, gmm.labels_)
    
    print("\n--- RESULTS ---")
    print(f"DBSCAN Time: {t_dbscan:.2f} seconds")
    print(f"BSGMM Time:  {t_bsgmm:.2f} seconds")
    print(f"Speedup:     {t_dbscan/t_bsgmm:.2f}x (Note: DBSCAN doesn't scale well with dimensions)")
    
    print(f"\nDBSCAN Clusters found: {len(np.unique(dbscan.labels_))} (Noise points: {np.sum(dbscan.labels_ == -1)})")
    print(f"BSGMM Clusters found:  {gmm.K_hat_}")
    
    print(f"\nDBSCAN - ARI: {ari_db:.4f} | AMI: {ami_db:.4f} | V: {v_db:.4f}")
    print(f"BSGMM  - ARI: {ari_gmm:.4f} | AMI: {ami_gmm:.4f} | V: {v_gmm:.4f}")
    
    n_sel = len(gmm.selected_features_)
    print(f"\nBSGMM Selected Features: {n_sel}/{n_features}")
    
    # Check if BSGMM found the true signal features
    true_sig = set(signal_indices)
    sel = set(gmm.selected_features_)
    found = true_sig.intersection(sel)
    print(f"Signal features found: {found} (out of {true_sig})")
    
    # --- Visualization ---
    os.makedirs("./visualize", exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Stress Test: Mixed Arcs (n={n_samples}, p={n_features})", fontsize=14, fontweight='bold')
    
    # Ground Truth
    pal = plt.cm.tab10(np.linspace(0, 0.9, 10))
    for k in np.unique(y):
        m = y == k
        axes[0].scatter(X_2d[m, 0], X_2d[m, 1], c=[pal[k % 10]], s=5, alpha=0.5)
    axes[0].set_title("Ground Truth (2D Projection)")
    
    # DBSCAN
    unique_db = np.unique(dbscan.labels_)
    pal_db = plt.cm.tab20(np.linspace(0, 1, max(len(unique_db), 2)))
    for idx, k in enumerate(unique_db):
        m = dbscan.labels_ == k
        color = 'k' if k == -1 else pal_db[idx % 20]
        axes[1].scatter(X_2d[m, 0], X_2d[m, 1], c=[color], s=5, alpha=0.5)
    axes[1].set_title(f"DBSCAN (ARI={ari_db:.3f})")
    
    # BSGMM
    unique_gmm = np.unique(gmm.labels_)
    pal_gmm = plt.cm.tab20(np.linspace(0, 1, max(len(unique_gmm), 2)))
    for idx, k in enumerate(unique_gmm):
        m = gmm.labels_ == k
        axes[2].scatter(X_2d[m, 0], X_2d[m, 1], c=[pal_gmm[idx % 20]], s=5, alpha=0.5)
    axes[2].set_title(f"BSGMM SVI (ARI={ari_gmm:.3f}, {n_sel} feats)")
    
    plt.tight_layout()
    plt.savefig("./visualize/stress_test_arcs.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved './visualize/stress_test_arcs.png'")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="cuda", type=str)
    args = parser.parse_args()
    
    run_stress_test_benchmark(backend=args.backend)
