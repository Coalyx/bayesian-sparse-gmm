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

## Development and Testing

Install development dependencies:

```bash
pip install -e ".[dev]"
```

Run tests using `pytest`:

```bash
pytest
```

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