import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances


def misclustering_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes the Mis-Clustering Rate (MCR) using the Hungarian (Kuhn-Munkres) algorithm
    to find the optimal bipartite matching between true and predicted labels.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.

    Returns:
        MCR score between 0.0 and 1.0 (lower is better).
    """
    # Exclude noise labels (-1) from the MCR calculation if necessary,
    # or treat them as a distinct class. For simplicity, we match all labels.
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")

    unique_true = np.unique(y_true)
    unique_pred = np.unique(y_pred)

    n_classes = max(len(unique_true), len(unique_pred))
    
    # Map labels to 0 .. n_classes-1
    true_map = {val: i for i, val in enumerate(unique_true)}
    pred_map = {val: i for i, val in enumerate(unique_pred)}
    
    cost_matrix = np.zeros((n_classes, n_classes), dtype=int)
    for i in range(len(y_true)):
        t_idx = true_map[y_true[i]]
        p_idx = pred_map[y_pred[i]]
        cost_matrix[t_idx, p_idx] += 1

    # linear_sum_assignment minimizes the cost, so we pass negative of counts to maximize matching
    row_ind, col_ind = linear_sum_assignment(cost_matrix.max() - cost_matrix)
    
    correct_matches = cost_matrix[row_ind, col_ind].sum()
    mcr = 1.0 - (correct_matches / len(y_true))
    return mcr


def compute_bic(X: np.ndarray, labels: np.ndarray, K: int = None) -> float:
    """
    Computes a simplified Bayesian Information Criterion (BIC) for clustering.
    Assumes a spherical Gaussian distribution for each cluster with a common scalar variance.
    
    BIC = -2 * log_likelihood + p * log(N)
    where p is the number of parameters.
    
    Args:
        X: Feature matrix of shape (N, D).
        labels: Cluster labels for each data point.
        K: Number of clusters. If None, inferred from unique labels.
        
    Returns:
        BIC value (lower is better). Returns np.inf if clustering is invalid.
    """
    N, D = X.shape
    unique_labels = np.unique(labels)
    # Exclude noise (-1) from cluster centers computation
    clusters = [l for l in unique_labels if l != -1]
    
    if K is None:
        K = len(clusters)
        
    if K == 0 or len(clusters) == 0:
        return np.inf

    # Calculate cluster centers and sizes
    centers = np.zeros((K, D))
    n_k = np.zeros(K)
    
    for i, c in enumerate(clusters):
        mask = (labels == c)
        n_k[i] = np.sum(mask)
        centers[i] = np.mean(X[mask], axis=0)
        
    # Estimate common variance
    variance = 0.0
    for i, c in enumerate(clusters):
        mask = (labels == c)
        variance += np.sum((X[mask] - centers[i]) ** 2)
    
    # Ensure variance is strictly positive to avoid log(0)
    variance = max(variance / N, 1e-6)
    
    # Log-likelihood under spherical Gaussian assumption
    log_likelihood = 0.0
    for i, c in enumerate(clusters):
        if n_k[i] > 0:
            log_likelihood += n_k[i] * np.log(n_k[i] / N)
            log_likelihood -= (n_k[i] * D / 2) * np.log(2 * np.pi * variance)
            # The distance term cancels out when summed up if variance is the sample variance
            log_likelihood -= (n_k[i] * D) / 2

    # Number of parameters:
    # (K - 1) for class probabilities + K * D for means + 1 for common variance
    p = (K - 1) + (K * D) + 1
    
    bic = -2 * log_likelihood + p * np.log(N)
    return bic


def compute_inertia(X: np.ndarray, labels: np.ndarray) -> float:
    """Computes within-cluster sum of squares (inertia)."""
    unique_labels = np.unique(labels)
    clusters = [l for l in unique_labels if l != -1]
    
    inertia = 0.0
    for c in clusters:
        mask = (labels == c)
        if np.sum(mask) > 0:
            center = np.mean(X[mask], axis=0)
            inertia += np.sum((X[mask] - center) ** 2)
            
    # Small epsilon to avoid log(0) later
    return max(inertia, 1e-10)


def gap_statistic(X: np.ndarray, K_max: int, n_refs: int = 5, random_state: int = 42) -> tuple:
    """
    Computes Gap Statistics using Monte Carlo simulated null reference (uniform box)
    and K-Means as the base clustering algorithm to find the optimal K.

    Args:
        X: Feature matrix.
        K_max: Maximum number of clusters to evaluate.
        n_refs: Number of reference datasets to generate.
        random_state: Random state for reproducibility.

    Returns:
        optimal_K: Estimated optimal number of clusters.
        gaps: Array of gap values for K=1 to K_max.
    """
    if X.shape[0] < K_max:
        K_max = X.shape[0] // 2

    rng = np.random.default_rng(random_state)
    gaps = np.zeros(K_max)
    s_k = np.zeros(K_max)
    
    # Bounding box for uniform reference
    min_vals = np.min(X, axis=0)
    max_vals = np.max(X, axis=0)

    for k in range(1, K_max + 1):
        # 1. Base clustering on true data
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=5)
        kmeans.fit(X)
        disp_true = np.log(compute_inertia(X, kmeans.labels_))

        # 2. Reference datasets
        ref_disps = np.zeros(n_refs)
        for i in range(n_refs):
            X_ref = rng.uniform(min_vals, max_vals, size=X.shape)
            km_ref = KMeans(n_clusters=k, random_state=random_state, n_init=5)
            km_ref.fit(X_ref)
            ref_disps[i] = np.log(compute_inertia(X_ref, km_ref.labels_))
            
        mean_ref_disp = np.mean(ref_disps)
        sd_ref_disp = np.std(ref_disps)
        
        gaps[k - 1] = mean_ref_disp - disp_true
        s_k[k - 1] = sd_ref_disp * np.sqrt(1 + 1 / n_refs)

    # 3. Find optimal K using the Gap Statistic criteria
    # K is chosen as the smallest k such that Gap(k) >= Gap(k+1) - s_{k+1}
    optimal_k = 1
    for k in range(K_max - 1):
        if gaps[k] >= gaps[k + 1] - s_k[k + 1]:
            optimal_k = k + 1
            break
    else:
        # If the condition is never met, choose the K that maximizes the gap
        optimal_k = np.argmax(gaps) + 1

    return optimal_k, gaps
