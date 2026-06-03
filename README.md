# Bayesian Sparse GMM

Bayesian Sparse Gaussian Mixture Model (GMM) implementation in Python.

This model employs a sparsity-inducing prior (e.g., a Dirichlet distribution with parameter $\alpha_0 < 1$) over mixture component weights to automatically determine/prune the number of active components.

## Installation

To install the latest release:

```bash
pip install bayesian-sparse-gmm
```

Or for development (editable mode):

```bash
git clone https://github.com/your-username/bayesian-sparse-gmm.git
cd bayesian-sparse-gmm
pip install -e .
```

## Quick Start

```python
import numpy as np
from bayesian_sparse_gmm import BayesianSparseGMM

# Generate some synthetic data
X = np.concatenate([
    np.random.normal(0, 1, (100, 2)),
    np.random.normal(5, 1, (100, 2))
])

# Initialize model with maximum 10 components
model = BayesianSparseGMM(n_components=10, alpha_0=0.1)

# Fit the model (to be implemented)
# model.fit(X)

# Print estimated component weights
# print(model.weights_)
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
