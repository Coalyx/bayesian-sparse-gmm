# Benchmarks for Bayesian Sparse GMM

This directory contains the benchmarking suite to evaluate the performance of `BayesianSparseGMM` against other standard clustering algorithms (K-Means, DBSCAN, Standard GMM) on various synthetic and real-world datasets.

## Directory Structure

- `runner.py`: The main script to execute the full benchmark suite across all datasets.
- `benchmark.py` & `run_benchmark.py`: Additional or alternative scripts for running benchmarks.
- `benchmark_stress_test.py`: Script for stress testing the models under varying constraints (e.g., high dimensions, large sample sizes).
- `run_olivetti_only.py`: Specific script for benchmarking on the Olivetti faces dataset.
- `data_pipeline.py`: Handles downloading, caching, and preprocessing datasets (supports synthetic data and OpenML).
- `evaluate.py` & `metrics.py`: Implements custom evaluation metrics such as Adjusted Rand Index (ARI), Normalized Mutual Information (NMI), Misclustering Rate (MCR), Bayesian Information Criterion (BIC), and Gap Statistics.
- `visualize.py`: Generates plots and visual comparisons from the benchmark results.
- `results/`: Directory where the raw benchmark outputs (like `benchmark_results.csv`) are saved.
- `visualize/`: Directory where the generated visualization plots are saved.
- `DATA_REPORT.md`: A comprehensive report detailing the dataset characteristics, evaluation metrics, and analysis of benchmark results.

## How to Run

### 1. Run the Standard Benchmark Suite

To run the main benchmarks across the default set of datasets, use:

```bash
# Make sure you are in the project root directory
python benchmarks/runner.py
```

This script will:
1. Load datasets via `data_pipeline.py`.
2. Run baseline models (K-Means, DBSCAN, GMM) and `BayesianSparseGMM` (both standard and SVI modes).
3. Compute all evaluation metrics.
4. Save the compiled results into `benchmarks/results/benchmark_results.csv`.

### 2. Generate Visualizations

Once you have generated the benchmark results, you can create visual plots to analyze the performance:

```bash
python benchmarks/visualize.py
```

The plots will be saved into the `benchmarks/visualize/` directory.

### 3. Run Specific Tests

If you want to run targeted benchmarks, you can execute specific scripts:

- **Stress Testing** (evaluating performance with large data):
  ```bash
  python benchmarks/benchmark_stress_test.py
  ```

- **Olivetti Faces Dataset Only** (for evaluating high-dimensional image data):
  ```bash
  python benchmarks/run_olivetti_only.py
  ```

## Notes
- The models utilize GPU acceleration if a CUDA backend is available.
- If you want to benchmark a specific dataset, you can modify the `datasets_filter` argument within `runner.py`.
