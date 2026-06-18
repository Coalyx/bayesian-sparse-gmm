import os
import sys
import time
import tracemalloc
from typing import Dict, Any, List

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

# Ensure src is in the path to import bayesian_sparse_gmm
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from bayesian_sparse_gmm.model import BayesianSparseGMM

# Import data pipeline
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from benchmarks.data_pipeline import DataPipelineOrchestrator

from benchmarks.metrics import misclustering_rate, compute_bic, gap_statistic


def profile_execution(func, *args, **kwargs) -> tuple:
    """Executes a function and measures time and peak memory (RAM)."""
    tracemalloc.start()
    start_time = time.time()
    
    result = func(*args, **kwargs)
    
    execution_time = time.time() - start_time
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    # Convert peak memory to MB
    peak_memory_mb = peak / (1024 * 1024)
    
    return result, execution_time, peak_memory_mb


def run_benchmark(datasets_filter=None) -> pd.DataFrame:
    """
    Runs the benchmark suite on all available datasets and models.
    """
    print("Loading datasets...")
    orchestrator = DataPipelineOrchestrator(data_dir=os.path.join(os.path.dirname(__file__), '..', 'data'))
    
    # Temporary patch to skip huge OpenML datasets if needed, or just load all.
    datasets = orchestrator.load_all()
    
    results = []

    for name, ds in datasets.items():
        if datasets_filter and not any(f in name for f in datasets_filter):
            continue
            
        print(f"\n--- Benchmarking Dataset: {name} ({ds.category}) ---")
        X = ds.X
        y_true = ds.y
        n_samples, n_features = X.shape
        K_expected = ds.K_expected
        
        if K_expected <= 0:
            K_expected = 2 # fallback
        
        # Calculate optimal K via Gap Statistics
        print(f"[{name}] Computing Gap Statistics...")
        K_max_eval = max(10, K_expected + 5)
        # Handle small datasets
        if n_samples < K_max_eval:
            K_max_eval = max(2, n_samples // 2)
            
        opt_k, gaps = gap_statistic(X, K_max=K_max_eval, n_refs=3, random_state=42)
        print(f"[{name}] Gap Statistic Optimal K: {opt_k} (Expected: {K_expected})")
        
        models = {
            "K-Means": lambda: KMeans(n_clusters=K_expected, random_state=42, n_init=10).fit_predict(X),
            # DBSCAN needs careful eps tuning, using a basic heuristic:
            "DBSCAN": lambda: DBSCAN(eps=0.5 * np.sqrt(n_features), min_samples=5).fit_predict(X),
            "GMM": lambda: GaussianMixture(n_components=K_expected, covariance_type='full', random_state=42).fit_predict(X),
            # BayesianSparseGMM automatically infers K
            "BayesianSparseGMM": lambda: BayesianSparseGMM(
                K_max=K_max_eval, n_iter=100, burn_in=50, backend="cuda", random_state=42, verbose=1
            ).fit(X).labels_,
            "BayesianSparseGMM (SVI)": lambda: BayesianSparseGMM(
                K_max=K_max_eval, optimizer="svi", epochs=50, batch_size=min(128, n_samples), backend="cuda", random_state=42, verbose=1
            ).fit(X).labels_
        }

        for model_name, model_func in models.items():
            print(f"  Running {model_name}...")
            try:
                y_pred, exec_time, peak_mem = profile_execution(model_func)
                
                # Check for noise-only predictions (e.g. DBSCAN might cluster everything as noise)
                unique_pred = np.unique(y_pred)
                n_clusters_found = len([l for l in unique_pred if l != -1])
                
                if n_clusters_found < 1:
                    ari = 0.0
                    nmi = 0.0
                    mcr = 1.0
                    bic = np.inf
                else:
                    ari = adjusted_rand_score(y_true, y_pred)
                    nmi = normalized_mutual_info_score(y_true, y_pred)
                    mcr = misclustering_rate(y_true, y_pred)
                    bic = compute_bic(X, y_pred)
                    
                results.append({
                    "Dataset": name,
                    "Category": ds.category,
                    "N": n_samples,
                    "D": n_features,
                    "K_Expected": K_expected,
                    "Gap_Opt_K": opt_k,
                    "Model": model_name,
                    "K_Found": n_clusters_found,
                    "ARI": ari,
                    "NMI": nmi,
                    "MCR": mcr,
                    "BIC": bic,
                    "Time_s": exec_time,
                    "Memory_MB": peak_mem
                })
            except Exception as e:
                print(f"  [ERROR] {model_name} failed on {name}: {e}")
                results.append({
                    "Dataset": name,
                    "Category": ds.category,
                    "N": n_samples,
                    "D": n_features,
                    "K_Expected": K_expected,
                    "Gap_Opt_K": opt_k,
                    "Model": model_name,
                    "K_Found": 0,
                    "ARI": np.nan,
                    "NMI": np.nan,
                    "MCR": np.nan,
                    "BIC": np.nan,
                    "Time_s": np.nan,
                    "Memory_MB": np.nan
                })
                
    df_results = pd.DataFrame(results)
    
    # Save raw results
    os.makedirs(os.path.join(os.path.dirname(__file__), 'results'), exist_ok=True)
    csv_path = os.path.join(os.path.dirname(__file__), 'results', 'benchmark_results.csv')
    df_results.to_csv(csv_path, index=False)
    print(f"\nSaved raw benchmark results to {csv_path}")
    
    return df_results

if __name__ == "__main__":
    run_benchmark()
