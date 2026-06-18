extern "C" {

// K-Means++ Distance Update (Float32)
__global__ void update_distances_float32(
    const float* __restrict__ X,
    const float* __restrict__ new_center,
    float* __restrict__ min_dist_sq,
    int n_samples,
    int n_features) 
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_samples) {
        float dist_sq = 0.0f;
        for (int j = 0; j < n_features; j++) {
            float diff = X[i * n_features + j] - new_center[j];
            dist_sq += diff * diff;
        }
        if (dist_sq < min_dist_sq[i]) {
            min_dist_sq[i] = dist_sq;
        }
    }
}

// Lloyd Iteration Assign and Accumulate (Float32)
__global__ void assign_and_accumulate_float32(
    const float* __restrict__ X,
    const float* __restrict__ centers,
    int* __restrict__ labels,
    float* __restrict__ new_centers_sum,
    int* __restrict__ new_centers_count,
    int n_samples,
    int n_features,
    int n_clusters)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_samples) return;

    float min_dist_sq = 3.402823466e+38f; // FLT_MAX
    int best_cluster = 0;

    for (int k = 0; k < n_clusters; k++) {
        float dist_sq = 0.0f;
        for (int j = 0; j < n_features; j++) {
            float diff = X[i * n_features + j] - centers[k * n_features + j];
            dist_sq += diff * diff;
        }
        if (dist_sq < min_dist_sq) {
            min_dist_sq = dist_sq;
            best_cluster = k;
        }
    }

    labels[i] = best_cluster;

    atomicAdd(&new_centers_count[best_cluster], 1);
    for (int j = 0; j < n_features; j++) {
        atomicAdd(&new_centers_sum[best_cluster * n_features + j], X[i * n_features + j]);
    }
}

// K-Means++ Distance Update (Float64)
__global__ void update_distances_float64(
    const double* __restrict__ X,
    const double* __restrict__ new_center,
    double* __restrict__ min_dist_sq,
    int n_samples,
    int n_features) 
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_samples) {
        double dist_sq = 0.0;
        for (int j = 0; j < n_features; j++) {
            double diff = X[i * n_features + j] - new_center[j];
            dist_sq += diff * diff;
        }
        if (dist_sq < min_dist_sq[i]) {
            min_dist_sq[i] = dist_sq;
        }
    }
}

// Lloyd Iteration Assign and Accumulate (Float64)
__global__ void assign_and_accumulate_float64(
    const double* __restrict__ X,
    const double* __restrict__ centers,
    int* __restrict__ labels,
    double* __restrict__ new_centers_sum,
    int* __restrict__ new_centers_count,
    int n_samples,
    int n_features,
    int n_clusters)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_samples) return;

    double min_dist_sq = 1.7976931348623158e+308; // DBL_MAX
    int best_cluster = 0;

    for (int k = 0; k < n_clusters; k++) {
        double dist_sq = 0.0;
        for (int j = 0; j < n_features; j++) {
            double diff = X[i * n_features + j] - centers[k * n_features + j];
            dist_sq += diff * diff;
        }
        if (dist_sq < min_dist_sq) {
            min_dist_sq = dist_sq;
            best_cluster = k;
        }
    }

    labels[i] = best_cluster;

    // Note: CUDA natively supports atomicAdd for double since Compute Capability 6.0
    // We assume >= CC 6.0 (Pascal and newer)
    atomicAdd(&new_centers_count[best_cluster], 1);
    for (int j = 0; j < n_features; j++) {
        atomicAdd(&new_centers_sum[best_cluster * n_features + j], X[i * n_features + j]);
    }
}

// ============================================================================
// Shared Memory Variants
// Centers are cooperatively loaded into shared memory by all threads in a block,
// reducing global memory reads from O(blockDim * n_clusters * n_features) to
// O(n_clusters * n_features) per block.
// ============================================================================

// Shared Memory Lloyd Assign + Accumulate (Float32)
__global__ void assign_accumulate_shmem_float32(
    const float* __restrict__ X,
    const float* __restrict__ centers,
    int* __restrict__ labels,
    float* __restrict__ new_centers_sum,
    int* __restrict__ new_centers_count,
    int n_samples,
    int n_features,
    int n_clusters)
{
    extern __shared__ unsigned char s_mem[];
    float* s_centers = (float*)s_mem;

    // Cooperative loading: all threads in block load centers into shared memory
    int total_elements = n_clusters * n_features;
    for (int idx = threadIdx.x; idx < total_elements; idx += blockDim.x) {
        s_centers[idx] = centers[idx];
    }
    __syncthreads();

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_samples) return;

    float min_dist_sq = 3.402823466e+38f; // FLT_MAX
    int best_cluster = 0;

    for (int k = 0; k < n_clusters; k++) {
        float dist_sq = 0.0f;
        for (int j = 0; j < n_features; j++) {
            float diff = X[i * n_features + j] - s_centers[k * n_features + j];
            dist_sq += diff * diff;
        }
        if (dist_sq < min_dist_sq) {
            min_dist_sq = dist_sq;
            best_cluster = k;
        }
    }

    labels[i] = best_cluster;

    atomicAdd(&new_centers_count[best_cluster], 1);
    for (int j = 0; j < n_features; j++) {
        atomicAdd(&new_centers_sum[best_cluster * n_features + j], X[i * n_features + j]);
    }
}

// Shared Memory Lloyd Assign + Accumulate (Float64)
__global__ void assign_accumulate_shmem_float64(
    const double* __restrict__ X,
    const double* __restrict__ centers,
    int* __restrict__ labels,
    double* __restrict__ new_centers_sum,
    int* __restrict__ new_centers_count,
    int n_samples,
    int n_features,
    int n_clusters)
{
    extern __shared__ unsigned char s_mem[];
    double* s_centers_d = (double*)s_mem;

    int total_elements = n_clusters * n_features;
    for (int idx = threadIdx.x; idx < total_elements; idx += blockDim.x) {
        s_centers_d[idx] = centers[idx];
    }
    __syncthreads();

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_samples) return;

    double min_dist_sq = 1.7976931348623158e+308; // DBL_MAX
    int best_cluster = 0;

    for (int k = 0; k < n_clusters; k++) {
        double dist_sq = 0.0;
        for (int j = 0; j < n_features; j++) {
            double diff = X[i * n_features + j] - s_centers_d[k * n_features + j];
            dist_sq += diff * diff;
        }
        if (dist_sq < min_dist_sq) {
            min_dist_sq = dist_sq;
            best_cluster = k;
        }
    }

    labels[i] = best_cluster;

    atomicAdd(&new_centers_count[best_cluster], 1);
    for (int j = 0; j < n_features; j++) {
        atomicAdd(&new_centers_sum[best_cluster * n_features + j], X[i * n_features + j]);
    }
}

}
