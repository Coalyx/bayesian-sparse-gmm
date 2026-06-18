import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# Add src to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from bayesian_sparse_gmm.clustering.kmeans import KMeansCupy

def visualize_kmeans():
    print("Generating 2D blobs dataset...")
    n_samples = 2000
    n_clusters = 5
    X, true_labels = make_blobs(n_samples=n_samples, centers=n_clusters, cluster_std=1.0, random_state=42)
    X = X.astype(np.float32)

    # Scikit-Learn KMeans
    print("Running sklearn KMeans...")
    sklearn_kmeans = KMeans(n_clusters=n_clusters, init="k-means++", n_init=1, max_iter=300, random_state=42)
    sklearn_kmeans.fit(X)
    sklearn_labels = sklearn_kmeans.labels_
    sklearn_centers = sklearn_kmeans.cluster_centers_

    # CuPy KMeans
    print("Running CuPy KMeans...")
    cupy_kmeans = KMeansCupy(n_clusters=n_clusters, max_iter=300, random_state=42)
    cupy_kmeans.fit(X)
    cupy_labels = cupy_kmeans.labels_
    cupy_centers = cupy_kmeans.cluster_centers_

    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot Sklearn
    axes[0].scatter(X[:, 0], X[:, 1], c=sklearn_labels, cmap='viridis', s=10, alpha=0.6)
    axes[0].scatter(sklearn_centers[:, 0], sklearn_centers[:, 1], c='red', marker='X', s=200, edgecolors='black')
    axes[0].set_title("Scikit-Learn KMeans")
    
    # Plot CuPy
    axes[1].scatter(X[:, 0], X[:, 1], c=cupy_labels, cmap='viridis', s=10, alpha=0.6)
    axes[1].scatter(cupy_centers[:, 0], cupy_centers[:, 1], c='red', marker='X', s=200, edgecolors='black')
    axes[1].set_title("CuPy KMeans++")

    plt.suptitle("KMeans Clustering Comparison", fontsize=16)
    plt.tight_layout()
    
    output_dir = os.path.join(os.path.dirname(__file__), "visualize")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "kmeans_comparison.png")
    plt.savefig(output_path, dpi=150)
    print(f"Visualization saved to {output_path}")

if __name__ == "__main__":
    visualize_kmeans()
