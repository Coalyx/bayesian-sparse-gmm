"""Clustering Benchmark Data Pipeline and Validation.

This module provides a clean, modular, and robust pipeline for downloading,
loading, preprocessing, validating, and reporting statistics on various
clustering datasets.
"""

import logging
import os
import time
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Optional/conditional imports with error handling
try:
    import openml
except ImportError:
    openml = None

try:
    import clustpy
    import clustpy.data as cpd
except ImportError:
    clustpy = None
    cpd = None

try:
    import clustbench
except ImportError:
    clustbench = None

try:
    import scanpy as sc
except ImportError:
    sc = None

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("DataPipeline")


@dataclass
class Dataset:
    """Standardized representation of a dataset in the pipeline."""

    name: str
    X: np.ndarray
    y: np.ndarray
    K_expected: int
    category: str  # 'Toy', 'Public Benchmark', 'Real-World', 'Custom'
    description: str


class DataPreprocessor:
    """Preprocesses dataset features and labels to standard format."""

    @staticmethod
    def preprocess(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Imputes NaNs, scales features to zero-mean unit-variance, and encodes labels.

        Args:
            X: Raw feature matrix.
            y: Raw class labels.

        Returns:
            Preprocessed (X, y) tuple.
        """
        # 1. Handle missing values (NaNs)
        if np.isnan(X).any():
            logger.info("Found missing values. Imputing with column mean...")
            imputer = SimpleImputer(strategy="mean")
            X = imputer.fit_transform(X)

        # 2. Standardize features to zero mean and unit variance
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # 3. Standardize labels (contiguous integers 0..K-1)
        # We preserve noise/outlier labels of -1 if they exist
        unique_labels = np.unique(y)
        has_noise = (
            -1 in unique_labels or "-1" in unique_labels or -1.0 in unique_labels
        )

        # Clean labels to integer array
        y_clean = np.zeros(y.shape[0], dtype=np.int32)

        label_map = {}
        idx = 0

        # Keep -1 as noise if it exists
        if has_noise:
            label_map[-1] = -1
            label_map["-1"] = -1
            label_map[-1.0] = -1

        for val in unique_labels:
            # Map normal labels (not noise) to 0..K-1
            if val not in (-1, "-1", -1.0):
                label_map[val] = idx
                idx += 1

        for i, val in enumerate(y):
            # Resolve to standard mapped label
            y_clean[i] = label_map.get(val, label_map.get(str(val), -1))

        return X_scaled, y_clean


class DataValidator:
    """Validates data integrity and type/shape alignment."""

    @staticmethod
    def validate(dataset: Dataset) -> bool:
        """Validates the dataset integrity.

        Args:
            dataset: Dataset instance to validate.

        Returns:
            True if dataset is valid, raises ValueError otherwise.
        """
        X, y = dataset.X, dataset.y

        # 1. Shape check
        if X.ndim != 2:
            raise ValueError(
                f"[{dataset.name}] Feature matrix X must be 2D. Got shape {X.shape}."
            )
        if y.ndim != 1:
            raise ValueError(
                f"[{dataset.name}] Label array y must be 1D. Got shape {y.shape}."
            )
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"[{dataset.name}] Dimension mismatch: X has {X.shape[0]} samples, "
                f"but y has {y.shape[0]} labels."
            )
        if X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError(f"[{dataset.name}] Dataset is empty: shape {X.shape}.")

        # 2. Type check
        if not np.issubdtype(X.dtype, np.number):
            raise ValueError(
                f"[{dataset.name}] Feature matrix X must be numeric. Got type {X.dtype}."
            )
        if not np.issubdtype(y.dtype, np.integer):
            raise ValueError(
                f"[{dataset.name}] Label array y must contain integers. Got type {y.dtype}."
            )

        # 3. NaNs and Infs checks
        if np.isnan(X).any():
            raise ValueError(
                f"[{dataset.name}] Feature matrix X contains NaNs after preprocessing."
            )
        if np.isinf(X).any():
            raise ValueError(
                f"[{dataset.name}] Feature matrix X contains Infinite values."
            )

        # 4. Variance check
        variances = np.var(X, axis=0)
        zero_var_cols = np.where(variances == 0)[0]
        if len(zero_var_cols) > 0:
            logger.warning(
                f"[{dataset.name}] Features {zero_var_cols.tolist()} have zero variance."
            )

        # 5. Expected clusters check
        unique_labels = np.unique(y)
        # Exclude noise class (-1) when checking expected cluster count
        n_clusters = len([lbl for lbl in unique_labels if lbl != -1])
        if n_clusters != dataset.K_expected and dataset.K_expected > 0:
            logger.warning(
                f"[{dataset.name}] Expected K={dataset.K_expected} clusters, "
                f"but found {n_clusters} classes in labels."
            )

        logger.info(f"[{dataset.name}] Integrity validation passed. Shape: {X.shape}.")
        return True


# =====================================================================
# Dataset Loaders
# =====================================================================


class ToyDatasetLoader:
    """Generates 2D synthetic toy datasets using scikit-learn."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def load_circles(self) -> Dataset:
        """Generates noisy concentric circles."""
        X, y = make_circles(
            n_samples=500, noise=0.05, factor=0.5, random_state=self.random_state
        )
        return Dataset(
            name="toy_circles",
            X=X,
            y=y,
            K_expected=2,
            category="Toy",
            description="Concentric circles dataset with background noise",
        )

    def load_moons(self) -> Dataset:
        """Generates noisy interleaving half circles."""
        X, y = make_moons(n_samples=500, noise=0.05, random_state=self.random_state)
        return Dataset(
            name="toy_moons",
            X=X,
            y=y,
            K_expected=2,
            category="Toy",
            description="Two interleaving half-moon shapes with background noise",
        )

    def load_blobs_aniso(self) -> Dataset:
        """Generates anisotropic (skewed) blobs."""
        X, y = make_blobs(
            n_samples=500, centers=3, cluster_std=1.0, random_state=self.random_state
        )
        # Transform blobs anisotropically
        rng = np.random.default_rng(self.random_state)
        transformation = rng.normal(size=(2, 2))
        X_aniso = np.dot(X, transformation)
        return Dataset(
            name="toy_blobs_aniso",
            X=X_aniso,
            y=y,
            K_expected=3,
            category="Toy",
            description="Anisotropically (linearly) transformed Gaussian blobs",
        )

    def load_blobs_varied(self) -> Dataset:
        """Generates blobs with varied variances."""
        X, y = make_blobs(
            n_samples=500,
            centers=3,
            cluster_std=[1.0, 2.5, 0.5],
            random_state=self.random_state,
        )
        return Dataset(
            name="toy_blobs_varied",
            X=X,
            y=y,
            K_expected=3,
            category="Toy",
            description="Gaussian blobs with significantly different cluster variances",
        )

    def load_no_structure(self) -> Dataset:
        """Generates uniform random noise (no structure)."""
        rng = np.random.default_rng(self.random_state)
        X = rng.uniform(low=-1.0, high=1.0, size=(500, 2))
        y = np.zeros(500, dtype=np.int32)  # No structure, single default cluster
        return Dataset(
            name="toy_no_structure",
            X=X,
            y=y,
            K_expected=1,
            category="Toy",
            description="Uniformly distributed random noise in 2D space",
        )


class OpenMLCC18Loader:
    """Downloads and loads representative datasets from OpenML-CC18 suite."""

    def __init__(self, cache_dir: str = "./data"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        if openml is not None:
            openml.config.cache_directory = os.path.abspath(cache_dir)

    def load_dataset_by_id(self, dataset_id: int, name: str) -> Dataset:
        """Downloads a specific dataset by its OpenML ID."""
        if openml is None:
            raise ImportError("openml package is required but not installed.")

        logger.info(f"Fetching OpenML dataset: {name} (ID: {dataset_id})...")
        ds = openml.datasets.get_dataset(dataset_id, download_data=True)
        X, y, _, _ = ds.get_data(
            target=ds.default_target_attribute, dataset_format="array"
        )

        # Resolve target/labels
        if y is None:
            y = np.zeros(X.shape[0], dtype=np.int32)

        unique_labels = len(np.unique(y))

        return Dataset(
            name=f"openml_{name}",
            X=X,
            y=y,
            K_expected=unique_labels,
            category="Public Benchmark",
            description=f"OpenML-CC18 classification dataset {name} (ID: {dataset_id})",
        )


class ClustPyUCILoader:
    """Loads UCI benchmark datasets via ClustPy."""

    def __init__(self, cache_dir: str = "./data"):
        self.cache_dir = cache_dir
        # Redirect ClustPy cache if needed (defaults to ~/Downloads/clustpy_datafiles)
        # Note: ClustPy downloads to user's download directory by default.

    def load_banknotes(self) -> Dataset:
        """Loads the UCI banknotes dataset."""
        if cpd is None:
            raise ImportError("clustpy is required but not installed.")

        logger.info("Loading ClustPy UCI dataset: banknotes...")
        res = cpd.load_banknotes()
        return Dataset(
            name="clustpy_banknotes",
            X=res.data,
            y=res.target,
            K_expected=2,
            category="Public Benchmark",
            description="UCI banknote authentication dataset",
        )

    def load_seeds(self) -> Dataset:
        """Loads the UCI seeds dataset."""
        if cpd is None:
            raise ImportError("clustpy is required but not installed.")

        logger.info("Loading ClustPy UCI dataset: seeds...")
        res = cpd.load_seeds()
        return Dataset(
            name="clustpy_seeds",
            X=res.data,
            y=res.target,
            K_expected=3,
            category="Public Benchmark",
            description="UCI wheat seeds dataset",
        )


class GagolewskiLoader:
    """Loads wut, sipu, and fcps datasets using the clustbench library."""

    def __init__(self):
        self.data_url = "https://github.com/gagolews/clustering-data-v1/raw/v1.1.0"

    def _load_gagolewski(self, battery: str, name: str, K: int) -> Dataset:
        if clustbench is None:
            raise ImportError("clustering-benchmarks (clustbench) is required.")

        logger.info(f"Loading {battery}/{name} dataset from Gagolewski suite...")
        ds = clustbench.load_dataset(battery, name, url=self.data_url)

        # Ground truth labels in clustbench is a list of arrays (multiple expert partitions).
        # We take the first partition as the default ground truth labels.
        y = (
            ds.labels[0]
            if isinstance(ds.labels, list) and len(ds.labels) > 0
            else np.zeros(ds.data.shape[0])
        )

        return Dataset(
            name=f"gagolewski_{name}",
            X=ds.data,
            y=y,
            K_expected=K,
            category="Real-World" if battery == "sipu" else "Public Benchmark",
            description=f"Dataset '{name}' from the Gagolewski '{battery}' benchmark battery",
        )

    def load_x2(self) -> Dataset:
        return self._load_gagolewski("wut", "x2", K=3)

    def load_unbalance(self) -> Dataset:
        return self._load_gagolewski("sipu", "unbalance", K=8)

    def load_worms_64(self) -> Dataset:
        return self._load_gagolewski("sipu", "worms_64", K=25)

    def load_atom(self) -> Dataset:
        return self._load_gagolewski("fcps", "atom", K=2)


class TenxBrainLoader:
    """Downloads and loads 10x Genomics mouse brain dataset."""

    def __init__(self, cache_dir: str = "./data"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.url = "https://cf.10xgenomics.com/samples/cell-exp/3.0.0/neuron_1k_v3/neuron_1k_v3_filtered_feature_bc_matrix.h5"
        self.filename = os.path.join(
            cache_dir, "neuron_1k_v3_filtered_feature_bc_matrix.h5"
        )

    def load(self) -> Dataset:
        """Downloads, reads HDF5 file, filters genes, and returns a Dataset."""
        if sc is None:
            raise ImportError("scanpy package is required but not installed.")

        # Download if not present
        if not os.path.exists(self.filename):
            logger.info("Downloading 10x Genomics mouse brain 1k dataset...")
            headers = {"User-Agent": "Mozilla/5.0"}
            req = urllib.request.Request(self.url, headers=headers)
            with urllib.request.urlopen(req) as response:
                with open(self.filename, "wb") as f:
                    f.write(response.read())
            logger.info("Download completed successfully.")
        else:
            logger.info("Using cached HDF5 file for 10x Genomics mouse brain.")

        # Read the 10x h5 file
        adata = sc.read_10x_h5(self.filename)

        # Basic preprocessing for single-cell data before clustering:
        # Filter cells and genes to make it clean
        sc.pp.filter_cells(adata, min_genes=200)
        sc.pp.filter_genes(adata, min_cells=3)

        # Normalize and log-transform
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

        # Extract highly variable genes to reduce dimensionality for clustering (e.g. 500 genes)
        sc.pp.highly_variable_genes(adata, n_top_genes=500)
        adata_subset = adata[:, adata.var["highly_variable"]]

        # Convert sparse matrix to dense array
        X = (
            adata_subset.X.toarray()
            if hasattr(adata_subset.X, "toarray")
            else adata_subset.X
        )

        # We don't have explicit ground truth labels for the cells, so we assign zeros
        # as a placeholder or we can use cell cycle / library size as mock labels.
        # Here we just use a default label array.
        y = np.zeros(X.shape[0], dtype=np.int32)

        return Dataset(
            name="tenx_mouse_brain",
            X=X,
            y=y,
            K_expected=1,
            category="Real-World",
            description="1k cells from embryonic mouse brain (filtered to 500 highly variable genes)",
        )


class CustomNoisyLoader:
    """Generates custom overlapping Gaussian mixtures and injects uniform outlier noise."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def generate_gmm(self, n_samples: int = 600, n_features: int = 2) -> Dataset:
        """Simulates an overlapping Gaussian Mixture Model in 2D."""
        rng = np.random.default_rng(self.random_state)

        # 3 clusters with overlapping means
        means = [np.array([0.0, 0.0]), np.array([1.2, 1.2]), np.array([0.6, -0.6])]
        cov = np.array([[1.0, 0.15], [0.15, 1.0]])

        samples_per_k = n_samples // len(means)
        X_parts = []
        y_parts = []

        for k, mean in enumerate(means):
            X_k = rng.multivariate_normal(mean, cov, size=samples_per_k)
            X_parts.append(X_k)
            y_parts.append(np.full(samples_per_k, k, dtype=np.int32))

        X = np.vstack(X_parts)
        y = np.hstack(y_parts)

        # Shuffle
        shuf = rng.permutation(X.shape[0])
        X, y = X[shuf], y[shuf]

        return Dataset(
            name="custom_gmm_base",
            X=X,
            y=y,
            K_expected=3,
            category="Custom",
            description="Custom simulated overlapping Gaussian Mixture Model in 2D",
        )

    def inject_outliers(self, dataset: Dataset, noise_ratio: float) -> Dataset:
        """Injects uniform background noise into a dataset."""
        rng = np.random.default_rng(self.random_state)
        X, y = dataset.X, dataset.y
        n_samples = X.shape[0]
        n_noise = int(n_samples * noise_ratio)

        # Find feature bounding box
        min_vals = X.min(axis=0)
        max_vals = X.max(axis=0)

        # Generate uniform noise within bounding box
        X_noise = rng.uniform(min_vals, max_vals, size=(n_noise, X.shape[1]))
        y_noise = np.full(n_noise, -1, dtype=np.int32)  # Noise labeled as -1

        X_noisy = np.vstack([X, X_noise])
        y_noisy = np.hstack([y, y_noise])

        # Shuffle all
        shuf = rng.permutation(X_noisy.shape[0])
        X_noisy, y_noisy = X_noisy[shuf], y_noisy[shuf]

        pct = int(noise_ratio * 100)
        return Dataset(
            name=f"{dataset.name}_noise_{pct:02d}",
            X=X_noisy,
            y=y_noisy,
            K_expected=dataset.K_expected,
            category="Custom",
            description=f"{dataset.description} with {pct}% injected background noise",
        )


# =====================================================================
# Pipeline Orchestrator & CLI Runner
# =====================================================================


class DataPipelineOrchestrator:
    """Manages the full dataset loading, preprocessing, and validation lifecycle."""

    def __init__(self, data_dir: str = "./data", random_state: int = 42):
        self.data_dir = data_dir
        self.random_state = random_state
        self.datasets: Dict[str, Dataset] = {}

        # Initialize loaders
        self.toy_loader = ToyDatasetLoader(random_state)
        self.openml_loader = OpenMLCC18Loader(data_dir)
        self.clustpy_loader = ClustPyUCILoader(data_dir)
        self.gagolewski_loader = GagolewskiLoader()
        self.tenx_loader = TenxBrainLoader(data_dir)
        self.custom_loader = CustomNoisyLoader(random_state)

    def load_all(self) -> Dict[str, Dataset]:
        """Loads and processes all specified datasets."""
        # 1. Load Toy Datasets
        self._register_dataset(self.toy_loader.load_circles())
        self._register_dataset(self.toy_loader.load_moons())
        self._register_dataset(self.toy_loader.load_blobs_aniso())
        self._register_dataset(self.toy_loader.load_blobs_varied())
        self._register_dataset(self.toy_loader.load_no_structure())

        # 2. Load Public Benchmark Datasets (OpenML & ClustPy)
        # We select specific small/medium datasets from OpenML-CC18 suite
        try:
            self._register_dataset(
                self.openml_loader.load_dataset_by_id(11, "balance-scale")
            )
            self._register_dataset(self.openml_loader.load_dataset_by_id(3, "kr-vs-kp"))
            self._register_dataset(
                self.openml_loader.load_dataset_by_id(12, "mfeat-factors")
            )
        except Exception as e:
            logger.error(f"Error loading OpenML datasets: {e}")

        try:
            self._register_dataset(self.clustpy_loader.load_banknotes())
            self._register_dataset(self.clustpy_loader.load_seeds())
        except Exception as e:
            logger.error(f"Error loading ClustPy UCI datasets: {e}")

        # 3. Load Real-World/Gagolewski Benchmark Datasets
        try:
            self._register_dataset(self.gagolewski_loader.load_x2())
            self._register_dataset(self.gagolewski_loader.load_unbalance())
            self._register_dataset(self.gagolewski_loader.load_worms_64())
            self._register_dataset(self.gagolewski_loader.load_atom())
        except Exception as e:
            logger.error(f"Error loading Gagolewski datasets: {e}")

        try:
            self._register_dataset(self.tenx_loader.load())
        except Exception as e:
            logger.error(f"Error loading 10x Genomics mouse brain dataset: {e}")

        # 4. Load Custom Noisy Datasets (GMM & Noise Injection)
        gmm_base = self.custom_loader.generate_gmm()
        self._register_dataset(gmm_base)
        self._register_dataset(self.custom_loader.inject_outliers(gmm_base, 0.05))
        self._register_dataset(self.custom_loader.inject_outliers(gmm_base, 0.15))
        self._register_dataset(self.custom_loader.inject_outliers(gmm_base, 0.30))

        return self.datasets

    def _register_dataset(self, dataset: Dataset) -> None:
        """Preprocesses, validates, and registers a dataset in the pipeline."""
        logger.info(f"Processing and registering dataset: {dataset.name}...")

        # Preprocess features & labels
        X_prep, y_prep = DataPreprocessor.preprocess(dataset.X, dataset.y)
        dataset.X = X_prep
        dataset.y = y_prep

        # Validate integrity
        DataValidator.validate(dataset)

        self.datasets[dataset.name] = dataset


def compute_statistics(datasets: Dict[str, Dataset]) -> List[Dict[str, Any]]:
    """Computes statistical properties for each loaded dataset."""
    stats_list = []

    for name, ds in datasets.items():
        X, y = ds.X, ds.y
        n_samples, n_features = X.shape

        # Count classes (excluding noise/outliers labeled -1)
        unique_labels = np.unique(y)
        normal_labels = [lbl for lbl in unique_labels if lbl != -1]
        k_observed = len(normal_labels)

        # Calculate class imbalance (majority size / minority size)
        if k_observed > 1:
            class_counts = [np.sum(y == lbl) for lbl in normal_labels]
            maj_count = max(class_counts)
            min_count = min(class_counts)
            imbalance_ratio = maj_count / min_count if min_count > 0 else float("nan")
        else:
            imbalance_ratio = 1.0

        # Count outliers (labeled -1)
        n_outliers = np.sum(y == -1)
        outlier_percentage = (n_outliers / n_samples) * 100

        stats_list.append(
            {
                "Name": name,
                "Category": ds.category,
                "Samples": n_samples,
                "Dimensions": n_features,
                "Expected K": ds.K_expected,
                "Observed K": k_observed,
                "Imbalance Ratio": imbalance_ratio,
                "Outliers %": outlier_percentage,
                "Description": ds.description,
            }
        )

    return stats_list


def generate_markdown_report(
    stats: List[Dict[str, Any]], filepath: str = "DATA_REPORT.md"
) -> None:
    """Generates the DATA_REPORT.md file containing the pipeline summary."""
    df = pd.DataFrame(stats)

    # Sort by Category and Name
    df = df.sort_values(by=["Category", "Name"]).reset_index(drop=True)

    with open(filepath, "w") as f:
        f.write("# Clustering Benchmark Dataset Report\n\n")
        f.write(
            "This report summarizes the statistical properties of the standardized and validated datasets loaded by the data pipeline.\n\n"
        )

        # Write statistical table
        f.write("## Dataset Statistics Table\n\n")
        f.write(
            "| Dataset Name | Category | Samples ($N$) | Dimensions ($D$) | Expected $K$ | Observed $K$ | Imbalance Ratio | Outliers (%) | Description |\n"
        )
        f.write(
            "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |\n"
        )

        for _, row in df.iterrows():
            imb_str = (
                f"{row['Imbalance Ratio']:.2f}"
                if not pd.isna(row["Imbalance Ratio"])
                else "N/A"
            )
            outlier_str = (
                f"{row['Outliers %']:.1f}%" if row["Outliers %"] > 0 else "0.0%"
            )
            f.write(
                f"| `{row['Name']}` | {row['Category']} | {row['Samples']:,} | {row['Dimensions']:,} | "
                f"{row['Expected K']} | {row['Observed K']} | {imb_str} | {outlier_str} | {row['Description']} |\n"
            )

        f.write("\n\n## Pipeline Integrity and Preprocessing Notes\n\n")
        f.write("### Preprocessing Pipeline\n")
        f.write(
            "1. **Missing Values**: Any dataset containing missing values (NaNs) is imputed using standard mean-imputation per feature.\n"
        )
        f.write(
            "2. **Standardization**: Feature matrices are scaled using zero-mean unit-variance scaling: $X_{std} = \\frac{X - \\mu}{\\sigma}$. This standard scaling ensures that variances are equalized across features, making the dataset compatible with spherical models like isotropic Gaussian Mixture Models.\n"
        )
        f.write(
            "3. **Contiguous Label Encoding**: Target labels are mapped to contiguous integers from $0$ to $K_{observed}-1$, while preserving `-1` labels for injected noise/outliers.\n\n"
        )

        f.write("### Validation Checks\n")
        f.write(
            "All datasets must pass the following validation checks before registration:\n"
        )
        f.write(
            "- **Dimensionality**: Feature matrix is strictly 2D, target labels are strictly 1D.\n"
        )
        f.write(
            "- **Sample Alignment**: The number of samples in $X$ matches the number of labels in $y$.\n"
        )
        f.write(
            "- **Data Types**: Features are numeric; target labels are integer arrays.\n"
        )
        f.write(
            "- **Completeness**: No NaN or infinite values are allowed in the preprocessed feature matrices.\n"
        )
        f.write(
            "- **Zero-Variance Features**: Alerts are printed if any feature column has zero variance across all samples.\n"
        )

    logger.info(f"Generated data report at: {os.path.abspath(filepath)}")


if __name__ == "__main__":
    t_start = time.time()
    logger.info("Starting data pipeline execution...")

    # Run pipeline
    orchestrator = DataPipelineOrchestrator(data_dir="./data")
    datasets = orchestrator.load_all()

    # Compute statistics and generate markdown report
    stats = compute_statistics(datasets)
    generate_markdown_report(stats, "DATA_REPORT.md")

    elapsed = time.time() - t_start
    logger.info(f"Data pipeline finished successfully in {elapsed:.2f} seconds!")
