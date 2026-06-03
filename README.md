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

## Citation

If you use this project in your research, please cite the following paper:

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