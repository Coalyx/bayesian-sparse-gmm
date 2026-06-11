import numpy as np
from scipy.special import gammaln

def log_rising_factorial(x: float, m: int) -> float:
    """log(x * (x+1) * ... * (x+m-1)) computed in log space for stability.

    This is equivalent to log(Gamma(x+m) / Gamma(x)).
    """
    if m == 0:
        return 0.0
    return float(gammaln(x + m) - gammaln(x))

def truncated_poisson_pmf(K: int, lambda_pois: float, K_max: int) -> float:
    """Truncated Poisson(lambda_pois) on {1, ..., K_max}."""
    if K < 1 or K > K_max:
        return 0.0
    log_probs = np.arange(1, K_max + 1) * np.log(lambda_pois) - gammaln(np.arange(1, K_max + 1) + 1.0)
    max_log = np.max(log_probs)
    probs = np.exp(log_probs - max_log)
    probs /= np.sum(probs)
    return float(probs[K - 1])

def log_urn_weight(k: int, n: int, alpha: float, lambda_pois: float, K_max: int) -> float:
    """Compute log V_n(k) from the exchangeable partition probability.

    V_n(k) = sum_{K=k}^{K_max} p_K(K) * rising_factorial(K, k) / rising_factorial(alpha*K, n)
    """
    if k > K_max or k < 1:
        return -np.inf
        
    log_p_K_unnorm = np.arange(1, K_max + 1) * np.log(lambda_pois) - gammaln(np.arange(1, K_max + 1) + 1.0)
    max_log_p = np.max(log_p_K_unnorm)
    probs = np.exp(log_p_K_unnorm - max_log_p)
    probs /= np.sum(probs)
    
    term_logs = []
    for K in range(k, K_max + 1):
        if probs[K - 1] <= 0:
            continue
        log_p = np.log(probs[K - 1])
        log_rf_K_k = log_rising_factorial(K, k)
        log_rf_alpha_K_n = log_rising_factorial(alpha * K, n)
        
        term_logs.append(log_p + log_rf_K_k - log_rf_alpha_K_n)
        
    if not term_logs:
        return -np.inf
        
    max_term = np.max(term_logs)
    sum_exp = np.sum(np.exp(term_logs - max_term))
    
    return float(max_term + np.log(sum_exp))

def urn_weight(k: int, n: int, alpha: float, lambda_pois: float, K_max: int) -> float:
    """Compute V_n(k) from the exchangeable partition probability."""
    return float(np.exp(log_urn_weight(k, n, alpha, lambda_pois, K_max)))
