# Bayesian Sparse GMM

Bayesian Sparse Gaussian Mixture Model (GMM) implementation in Python.

## Installation

To install the latest release:

```bash
pip install bayesian-sparse-gmm
```

Or for development (editable mode):

```bash
git clone https://github.com/Coalyx/bayesian-sparse-gmm.git
cd bayesian-sparse-gmm
pip install -e .
```

## Quick Start

```python
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from bayesian_sparse_gmm import BayesianSparseGMM

# Append noise dimensions to true clusters to verify that the model successfully performs feature selection.
rng = np.random.default_rng(42)
X_clean, _ = make_blobs(n_samples=200, centers=3, n_features=2, cluster_std=0.5, random_state=42)
X_noise = rng.normal(loc=0.0, scale=1.0, size=(200, 8))
X = np.hstack([X_clean, X_noise])

# Standardize features to satisfy the zero-mean assumptions in the prior structure.
X = StandardScaler().fit_transform(X)

model = BayesianSparseGMM(
    K_max=5,
    n_iter=300,
    burn_in=100,
    lambda_0=10.0,
    lambda_1=0.05,
    random_state=42,
    verbose=0
)
model.fit(X)

print(f"Number of active clusters: {model.n_clusters_}")
print(f"Selected informative features: {model.selected_features_}")
print(f"Feature inclusion probabilities: {model.feature_probabilities_.round(3)}")

labels = model.predict(X)
```

## GPU / CUDA Acceleration

The model supports three backends, selected via the `backend` parameter:

| Backend | Value | Description |
|---------|-------|-------------|
| NumPy | `"numpy"` | Pure NumPy — always available, no extras needed |
| Numba CPU | `"numba"` | Parallel CPU via Numba JIT (**default**) |
| CUDA GPU | `"cuda"` | GPU via CuPy or Numba CUDA |
| Auto | `"auto"` | Numba CPU if available, else NumPy |

### Install GPU support

```bash
# CuPy (recommended — more flexible with driver versions)
pip install "bayesian-sparse-gmm[cuda]"
# or install directly for your CUDA version, e.g.:
pip install cupy-cuda12x   # CUDA 12.x
pip install cupy-cuda11x   # CUDA 11.x
```

### Run with GPU (Python API)

```python
model = BayesianSparseGMM(
    K_max=10,
    n_iter=500,
    backend="cuda",   # GPU via CuPy → Numba CUDA → NumPy fallback
    random_state=42,
)
model.fit(X)
```


> [!NOTE]
> **Graceful fallback.** When `backend="cuda"` is requested, the library
> probes available backends in order: CuPy → Numba CUDA → NumPy. If a GPU
> backend is unavailable or incompatible, it falls back automatically and
> emits a `UserWarning` explaining why.

> [!WARNING]
> **Numba CUDA / PTX version mismatch.** Numba bundles its own CUDA
> toolkit. If the PTX version Numba generates is newer than what the
> installed NVIDIA driver supports (e.g., `Unsupported .version 8.7;
> current version is '8.2'`), Numba CUDA will be disabled automatically.
> Solutions:
> - **Preferred:** Install CuPy (`pip install cupy-cuda12x`) — it ships
>   its own CUDA runtime and handles driver compatibility independently.
> - **Alternative:** Update the NVIDIA driver (≥ 545 for CUDA 12.3+).
> - **Alternative:** Pin Numba to a version matching your driver's CUDA
>   support level.

---

## Development and Testing

Install development dependencies:

```bash
pip install -e ".[dev]"
```

Run tests using `pytest`:

```bash
pytest
```

## Algorithm Overview

Bayesian Sparse Gaussian Mixture Model (BSGMM) is a robust clustering algorithm designed specifically for high-dimensional data where the number of features significantly exceeds the number of samples ($p \gg n$). It integrates a Spike-and-Slab LASSO prior to perform simultaneous clustering and feature selection.

### Suitable Use Cases

1. **High-Dimensional Clustering ($p \gg n$)**: When dealing with datasets where traditional clustering algorithms (like K-Means or standard GMM) fail due to the "curse of dimensionality". Examples include bioinformatics (e.g., single-cell RNA-seq, genomics), text mining (high-dimensional TF-IDF matrices), and high-resolution images.
2. **Automatic Feature Selection (Interpretability)**: When the goal is not only to cluster the samples but also to identify which specific features (biomarkers, keywords, pixels) drive the cluster assignments. The model automatically shrinks noisy features to exactly zero.
3. **Unknown Number of Clusters**: When the true number of clusters is unknown. BSGMM can dynamically infer the optimal number of clusters from the data (bounded by `K_max`).
4. **Weak Signals in Noisy Backgrounds**: When the discriminative signal is weak and dispersed among thousands of irrelevant features, the model's sparsity mechanism is highly effective at pooling signals.

## Hyperparameters

Understanding the key hyperparameters is crucial for fine-tuning the model's sparsity and clustering behavior:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `K_max` | `int` | `15` | The maximum possible number of clusters. The algorithm will automatically find the active number of clusters $K \le K_{max}$. Should be set safely higher than the expected number of true clusters. |
| `lambda_0` | `float` | `1000.0` | **Spike rate** of the Spike-and-Slab LASSO prior. A larger value aggressively forces non-informative (noise) features closer to zero. Must satisfy `lambda_0 >> lambda_1`. |
| `lambda_1` | `float` | `0.1` | **Slab rate**. A smaller value allows informative features to deviate freely from zero to capture the cluster structure. |
| `alpha` | `float` | `1.0` | Dirichlet concentration parameter for the cluster weight prior. Controls the prior belief over the distribution of cluster sizes. |
| `theta` | `float` | `0.1` | Prior probability of a feature being included in the active set (the slab component). Smaller values induce stronger sparsity (fewer features selected). |
| `burn_in` | `int` | `500` | Number of initial MCMC iterations discarded to allow the Markov chain to converge to the stationary distribution. |
| `n_iter` | `int` | `1000` | Total number of MCMC iterations. The number of samples used for posterior inference is `n_iter - burn_in`. |

*Tip: For extremely high-dimensional datasets with heavy noise, tuning `lambda_0` to be larger and `theta` to be smaller will encourage more aggressive feature selection.*

## Reference

```bib
@article{JMLR:v26:23-0142,
  author  = {Dapeng Yao and Fangzheng Xie and Yanxun Xu},
  title   = {Bayesian Sparse Gaussian Mixture Model for Clustering in High Dimensions},
  journal = {Journal of Machine Learning Research},
  year    = {2025},
  volume  = {26},
  number  = {21},
  pages   = {1--50},
  url     = {http://jmlr.org/papers/v26/23-0142.html}
}
```