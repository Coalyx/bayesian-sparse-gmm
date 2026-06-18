# Clustering Benchmark Dataset Report

This report summarizes the statistical properties of the standardized and validated datasets loaded by the data pipeline.

## Dataset Statistics Table

| Dataset Name | Category | Samples ($N$) | Dimensions ($D$) | Expected $K$ | Observed $K$ | Imbalance Ratio | Outliers (%) | Description |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `custom_gmm_base` | Custom | 600 | 2 | 3 | 3 | 1.00 | 0.0% | Custom simulated overlapping Gaussian Mixture Model in 2D |
| `custom_gmm_base_noise_05` | Custom | 630 | 2 | 3 | 3 | 1.00 | 4.8% | Custom simulated overlapping Gaussian Mixture Model in 2D with 5% injected background noise |
| `custom_gmm_base_noise_15` | Custom | 690 | 2 | 3 | 3 | 1.00 | 13.0% | Custom simulated overlapping Gaussian Mixture Model in 2D with 15% injected background noise |
| `custom_gmm_base_noise_30` | Custom | 780 | 2 | 3 | 3 | 1.00 | 23.1% | Custom simulated overlapping Gaussian Mixture Model in 2D with 30% injected background noise |
| `clustpy_banknotes` | Public Benchmark | 1,372 | 4 | 2 | 2 | 1.25 | 0.0% | UCI banknote authentication dataset |
| `clustpy_seeds` | Public Benchmark | 210 | 7 | 3 | 3 | 1.00 | 0.0% | UCI wheat seeds dataset |
| `gagolewski_atom` | Public Benchmark | 800 | 3 | 2 | 2 | 1.00 | 0.0% | Dataset 'atom' from the Gagolewski 'fcps' benchmark battery |
| `gagolewski_x2` | Public Benchmark | 120 | 2 | 3 | 3 | 1.67 | 0.0% | Dataset 'x2' from the Gagolewski 'wut' benchmark battery |
| `openml_balance-scale` | Public Benchmark | 625 | 4 | 3 | 3 | 5.88 | 0.0% | OpenML-CC18 classification dataset balance-scale (ID: 11) |
| `openml_kr-vs-kp` | Public Benchmark | 3,196 | 36 | 2 | 2 | 1.09 | 0.0% | OpenML-CC18 classification dataset kr-vs-kp (ID: 3) |
| `openml_mfeat-factors` | Public Benchmark | 2,000 | 216 | 10 | 10 | 1.00 | 0.0% | OpenML-CC18 classification dataset mfeat-factors (ID: 12) |
| `gagolewski_unbalance` | Real-World | 6,500 | 2 | 8 | 8 | 20.00 | 0.0% | Dataset 'unbalance' from the Gagolewski 'sipu' benchmark battery |
| `gagolewski_worms_64` | Real-World | 105,000 | 64 | 25 | 25 | 1.00 | 0.0% | Dataset 'worms_64' from the Gagolewski 'sipu' benchmark battery |
| `tenx_mouse_brain` | Real-World | 1,243 | 500 | 1 | 1 | 1.00 | 0.0% | 1k cells from embryonic mouse brain (filtered to 500 highly variable genes) |
| `toy_blobs_aniso` | Toy | 500 | 2 | 3 | 3 | 1.01 | 0.0% | Anisotropically (linearly) transformed Gaussian blobs |
| `toy_blobs_varied` | Toy | 500 | 2 | 3 | 3 | 1.01 | 0.0% | Gaussian blobs with significantly different cluster variances |
| `toy_circles` | Toy | 500 | 2 | 2 | 2 | 1.00 | 0.0% | Concentric circles dataset with background noise |
| `toy_moons` | Toy | 500 | 2 | 2 | 2 | 1.00 | 0.0% | Two interleaving half-moon shapes with background noise |
| `toy_no_structure` | Toy | 500 | 2 | 1 | 1 | 1.00 | 0.0% | Uniformly distributed random noise in 2D space |


## Pipeline Integrity and Preprocessing Notes

### Preprocessing Pipeline
1. **Missing Values**: Any dataset containing missing values (NaNs) is imputed using standard mean-imputation per feature.
2. **Standardization**: Feature matrices are scaled using zero-mean unit-variance scaling: $X_{std} = \frac{X - \mu}{\sigma}$. This standard scaling ensures that variances are equalized across features, making the dataset compatible with spherical models like isotropic Gaussian Mixture Models.
3. **Contiguous Label Encoding**: Target labels are mapped to contiguous integers from $0$ to $K_{observed}-1$, while preserving `-1` labels for injected noise/outliers.

### Validation Checks
All datasets must pass the following validation checks before registration:
- **Dimensionality**: Feature matrix is strictly 2D, target labels are strictly 1D.
- **Sample Alignment**: The number of samples in $X$ matches the number of labels in $y$.
- **Data Types**: Features are numeric; target labels are integer arrays.
- **Completeness**: No NaN or infinite values are allowed in the preprocessed feature matrices.
- **Zero-Variance Features**: Alerts are printed if any feature column has zero variance across all samples.
