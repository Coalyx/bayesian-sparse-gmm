import time
import numpy as np
from sklearn.cluster import KMeans
import sys
import os

# Add src to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from bayesian_sparse_gmm.clustering.kmeans import KMeansCupy

def benchmark_kmeans():
    print("Generating random dataset...")
    n_samples = 100000
    n_features = 100
    n_clusters = 50
    rng = np.random.default_rng(42)
    X = rng.normal(size=(n_samples, n_features)).astype(np.float32)

    print(f"Dataset shape: {X.shape}, Clusters: {n_clusters}")
    print("-" * 40)

    # Scikit-Learn KMeans
    print("Running sklearn KMeans...")
    sklearn_kmeans = KMeans(n_clusters=n_clusters, init="k-means++", n_init=1, max_iter=100, random_state=42)
    
    start_time = time.time()
    sklearn_kmeans.fit(X)
    sklearn_time = time.time() - start_time
    print(f"sklearn KMeans time: {sklearn_time:.4f} seconds")

    # CuPy KMeans
    print("\nRunning CuPy KMeans...")
    cupy_kmeans = KMeansCupy(n_clusters=n_clusters, max_iter=100, random_state=42)
    
    # Run once to compile / initialize CUDA context
    cupy_kmeans.fit(X[:100])
    
    start_time = time.time()
    cupy_kmeans.fit(X)
    cupy_time = time.time() - start_time
    print(f"CuPy KMeans time: {cupy_time:.4f} seconds")
    
    print("-" * 40)
    print(f"Speedup: {sklearn_time / cupy_time:.2f}x")

if __name__ == "__main__":
    benchmark_kmeans()
