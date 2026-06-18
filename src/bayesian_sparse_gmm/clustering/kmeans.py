import os

import numpy as np

try:
    import cupy as cp

    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None


class KMeansCupy:
    """A highly optimized K-Means implementation using custom CUDA C++ kernels via CuPy.

    Features:
        - KMeans++ initialization with CUDA-accelerated distance updates
        - Lloyd's algorithm with fused assign + accumulate CUDA kernel
        - Shared memory optimization for centers when they fit in device shmem
        - n_init support: runs multiple times, keeps best result (lowest inertia)
    """

    # Conservative shared memory limit (48KB, guaranteed on CC 2.0+)
    _DEFAULT_MAX_SHMEM = 48 * 1024

    def __init__(
        self, n_clusters=8, n_init=1, max_iter=300, tol=1e-4, random_state=None
    ):
        """Initialize the KMeans parameters.

        Args:
            n_clusters: Number of clusters.
            n_init: Number of times the algorithm will be run with different
                centroid seeds. The final result will be the best output of
                n_init consecutive runs in terms of inertia.
            max_iter: Maximum number of Lloyd iterations per run.
            tol: Convergence tolerance based on center shift.
            random_state: Random seed for reproducibility.
        """
        if not CUPY_AVAILABLE:
            raise ImportError("CuPy is not installed.")
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        self._load_cuda_kernels()

    def _load_cuda_kernels(self):
        """Compile and load the custom CUDA kernels from kmeans.cu."""
        kernel_path = os.path.join(os.path.dirname(__file__), "kernels", "kmeans.cu")
        with open(kernel_path, "r") as f:
            cuda_source = f.read()
        self.module = cp.RawModule(code=cuda_source)

        # KMeans++ distance update kernels
        self.kernel_update_dist_f32 = self.module.get_function(
            "update_distances_float32"
        )
        self.kernel_update_dist_f64 = self.module.get_function(
            "update_distances_float64"
        )

        # Lloyd assign + accumulate kernels (global memory)
        self.kernel_assign_f32 = self.module.get_function(
            "assign_and_accumulate_float32"
        )
        self.kernel_assign_f64 = self.module.get_function(
            "assign_and_accumulate_float64"
        )

        # Lloyd assign + accumulate kernels (shared memory for centers)
        self.kernel_assign_shmem_f32 = self.module.get_function(
            "assign_accumulate_shmem_float32"
        )
        self.kernel_assign_shmem_f64 = self.module.get_function(
            "assign_accumulate_shmem_float64"
        )

    def _get_max_shared_mem(self):
        """Query maximum shared memory per block for the current device."""
        try:
            props = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
            return props.get("sharedMemPerBlock", self._DEFAULT_MAX_SHMEM)
        except Exception:
            return self._DEFAULT_MAX_SHMEM

    def _kmeans_plusplus(self, X, rng):
        """K-Means++ initialization algorithm using optimized CUDA kernels."""
        n_samples, n_features = X.shape
        dtype = X.dtype
        centers = cp.empty((self.n_clusters, n_features), dtype=dtype)

        # Randomly choose the first center
        initial_idx = rng.randint(0, n_samples)
        centers[0] = X[initial_idx]

        if self.n_clusters == 1:
            return centers

        # Distances to the closest center
        # Initialize with max float values
        if dtype == np.float32:
            min_dist_sq = cp.full(n_samples, cp.finfo(cp.float32).max, dtype=cp.float32)
            update_kernel = self.kernel_update_dist_f32
        else:
            min_dist_sq = cp.full(n_samples, cp.finfo(cp.float64).max, dtype=cp.float64)
            update_kernel = self.kernel_update_dist_f64

        block_size = 256
        grid_size = (n_samples + block_size - 1) // block_size

        for c in range(1, self.n_clusters):
            # Update min_dist_sq with the newest center
            update_kernel(
                (grid_size,),
                (block_size,),
                (X, centers[c - 1], min_dist_sq, n_samples, n_features),
            )

            # Choose the next center
            dist_sum = cp.sum(min_dist_sq)
            if dist_sum > 0:
                probs = min_dist_sq / dist_sum
            else:
                probs = cp.ones(n_samples, dtype=dtype) / n_samples

            cumulative_probs = cp.cumsum(probs)
            r = rng.rand()
            next_idx = int(cp.searchsorted(cumulative_probs, r))
            next_idx = min(next_idx, n_samples - 1)
            centers[c] = X[next_idx]

        return centers

    def _compute_inertia(self, X, centers, labels):
        """Compute inertia (sum of squared distances to assigned centers)."""
        assigned_centers = centers[labels]
        return float(cp.sum((X - assigned_centers) ** 2))

    def _single_lloyd(self, X_cp, rng):
        """Run a single K-Means: KMeans++ init + Lloyd iterations.

        Returns:
            Tuple of (centers, labels, inertia).
        """
        n_samples, n_features = X_cp.shape
        dtype = X_cp.dtype

        centers = self._kmeans_plusplus(X_cp, rng)
        labels = cp.zeros(n_samples, dtype=cp.int32)

        # Determine whether to use shared memory kernel
        dtype_size = 4 if dtype == cp.float32 else 8
        shmem_required = self.n_clusters * n_features * dtype_size
        max_shmem = self._get_max_shared_mem()
        use_shmem = shmem_required <= max_shmem

        if use_shmem:
            assign_kernel = (
                self.kernel_assign_shmem_f32
                if dtype == cp.float32
                else self.kernel_assign_shmem_f64
            )
        else:
            assign_kernel = (
                self.kernel_assign_f32
                if dtype == cp.float32
                else self.kernel_assign_f64
            )

        block_size = 256
        grid_size = (n_samples + block_size - 1) // block_size

        for _ in range(self.max_iter):
            new_centers_sum = cp.zeros_like(centers)
            new_centers_count = cp.zeros(self.n_clusters, dtype=cp.int32)

            # E-step & M-step (accumulation) combined in single kernel
            if use_shmem:
                assign_kernel(
                    (grid_size,),
                    (block_size,),
                    (
                        X_cp,
                        centers,
                        labels,
                        new_centers_sum,
                        new_centers_count,
                        n_samples,
                        n_features,
                        self.n_clusters,
                    ),
                    shared_mem=shmem_required,
                )
            else:
                assign_kernel(
                    (grid_size,),
                    (block_size,),
                    (
                        X_cp,
                        centers,
                        labels,
                        new_centers_sum,
                        new_centers_count,
                        n_samples,
                        n_features,
                        self.n_clusters,
                    ),
                )

            # M-step (averaging)
            counts = new_centers_count[:, cp.newaxis]
            mask = (counts > 0).flatten()

            new_centers = cp.empty_like(centers)
            if cp.any(mask):
                new_centers[mask] = new_centers_sum[mask] / counts[mask]

            # Keep previous centers for empty clusters
            empty_mask = ~mask
            if cp.any(empty_mask):
                new_centers[empty_mask] = centers[empty_mask]

            # Check convergence
            center_shift = cp.sum((centers - new_centers) ** 2)
            centers = new_centers
            if center_shift < self.tol:
                break

        inertia = self._compute_inertia(X_cp, centers, labels)
        return centers, labels, inertia

    def fit(self, X):
        """Compute K-Means clustering.

        Runs the algorithm n_init times and keeps the best result
        (lowest inertia).
        """
        # Ensure contiguous arrays in GPU memory
        X_cp = cp.asarray(X)
        if not X_cp.flags.c_contiguous:
            X_cp = cp.ascontiguousarray(X_cp)

        dtype = X_cp.dtype
        if dtype not in [cp.float32, cp.float64]:
            X_cp = X_cp.astype(cp.float32)

        # Generate deterministic seeds for each init run
        # When n_init==1, use original seed directly for backward compatibility
        if self.n_init == 1:
            seeds = [self.random_state]
        else:
            if self.random_state is not None:
                base_rng = np.random.RandomState(self.random_state)
                seeds = [int(base_rng.randint(0, 2**31)) for _ in range(self.n_init)]
            else:
                seeds = [None] * self.n_init

        best_inertia = None
        best_centers = None
        best_labels = None

        for seed in seeds:
            rng = cp.random.RandomState(seed)
            centers, labels, inertia = self._single_lloyd(X_cp, rng)

            if best_inertia is None or inertia < best_inertia:
                best_inertia = inertia
                best_centers = centers
                best_labels = labels

        self.cluster_centers_ = cp.asnumpy(best_centers)
        self.labels_ = cp.asnumpy(best_labels)
        self.inertia_ = best_inertia
        return self
