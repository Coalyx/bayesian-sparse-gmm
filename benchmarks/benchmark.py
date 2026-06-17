import time
import numpy as np
from sklearn.preprocessing import StandardScaler
from bayesian_sparse_gmm.model import BayesianSparseGMM

def run_benchmark(n=5000, p=50, K_true=5, signal_features=10, backend="cuda"):
    print(f"\n=======================================================")
    print(f"BENCHMARK: SVI vs MCMC (n={n}, p={p}, K={K_true})")
    print(f"Backend: {backend.upper()}")
    print(f"=======================================================\n")
    
    # Generate Synthetic Sparse Data
    rng = np.random.default_rng(42)
    means = np.zeros((K_true, p))
    means[:, :signal_features] = rng.normal(0, 3.0, size=(K_true, signal_features))
    
    n_per_k = n // K_true
    X_parts = [rng.normal(means[k], 1.0, size=(n_per_k, p)) for k in range(K_true)]
    X_raw = np.vstack(X_parts)
    y = np.repeat(np.arange(K_true), n_per_k)
    
    # Shuffle
    shuf = rng.permutation(len(y))
    X_raw, y = X_raw[shuf], y[shuf]
    
    # Standardize
    X = StandardScaler().fit_transform(X_raw)
    
    # --- MCMC (Default) ---
    print("Running MCMC (Gibbs Sampling) ...")
    t0 = time.time()
    gmm_mcmc = BayesianSparseGMM(
        K_max=10,
        optimizer="default",
        n_iter=100,      # Small number for benchmarking
        burn_in=20,
        backend=backend,
        random_state=42,
        use_identity_covariance=True
    )
    gmm_mcmc.fit(X)
    t_mcmc = time.time() - t0
    
    # --- SVI ---
    print("Running SVI (Natural Gradients) ...")
    t0 = time.time()
    gmm_svi = BayesianSparseGMM(
        K_max=10,
        optimizer="svi",
        epochs=10,       # 10 passes over the dataset
        batch_size=256,
        backend=backend,
        random_state=42,
        use_identity_covariance=True
    )
    gmm_svi.fit(X)
    t_svi = time.time() - t0
    
    print("\n--- RESULTS ---")
    print(f"MCMC Time: {t_mcmc:.2f} seconds")
    print(f"SVI Time:  {t_svi:.2f} seconds")
    print(f"Speedup:   {t_mcmc/t_svi:.2f}x")
    
    print(f"\nMCMC Found Clusters: {gmm_mcmc.K_hat_}")
    print(f"SVI Found Clusters:  {gmm_svi.K_hat_}")
    
    n_sel_mcmc = len(gmm_mcmc.selected_features_)
    n_sel_svi = len(gmm_svi.selected_features_)
    print(f"MCMC Selected Features: {n_sel_mcmc}/{p}")
    print(f"SVI Selected Features:  {n_sel_svi}/{p}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="cuda", type=str)
    args = parser.parse_args()
    
    run_benchmark(n=10000, p=100, K_true=10, signal_features=15, backend=args.backend)
