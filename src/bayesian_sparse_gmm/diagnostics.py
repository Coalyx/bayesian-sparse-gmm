import numpy as np

def _prepare_chains(trace: np.ndarray) -> tuple[np.ndarray, tuple]:
    """Prepare input trace array to be of shape (M, N, D) and return original shape."""
    orig_shape = trace.shape
    
    # 1D: (samples,) -> (1, samples, 1)
    if trace.ndim == 1:
        return trace[np.newaxis, :, np.newaxis], ()
        
    # 2D: (samples, D) -> (1, samples, D)
    if trace.ndim == 2:
        return trace[np.newaxis, :, :], (orig_shape[1],)
        
    # 3D: (samples, K, P) -> (1, samples, K*P)
    if trace.ndim == 3:
        samples, k, p = trace.shape
        return trace.reshape(1, samples, k * p), (k, p)
        
    # 4D: (chains, samples, K, P) -> (chains, samples, K*P)
    if trace.ndim == 4:
        chains, samples, k, p = trace.shape
        return trace.reshape(chains, samples, k * p), (k, p)
        
    raise ValueError(f"Unsupported trace dimensions: {trace.ndim}")

def gelman_rubin(trace: np.ndarray) -> np.ndarray:
    """Compute Gelman-Rubin R-hat statistic for trace variables.
    
    Supports single or multiple chains. For single chain inputs, split-R-hat is computed.
    """
    chains, param_shape = _prepare_chains(trace)
    M, N, D = chains.shape
    
    if N < 4:
        # Too few samples to split/estimate
        rhat_flat = np.ones(D)
        return rhat_flat.reshape(param_shape) if param_shape else rhat_flat[0]
        
    if M < 2:
        # Split-R-hat: split each chain in half
        N_half = N // 2
        chains_split = np.empty((2 * M, N_half, D), dtype=chains.dtype)
        for m in range(M):
            chains_split[2 * m] = chains[m, :N_half]
            chains_split[2 * m + 1] = chains[m, N_half:2 * N_half]
        chains = chains_split
        M, N, D = chains.shape
        
    # Means per chain
    chain_means = np.mean(chains, axis=1)
    overall_mean = np.mean(chain_means, axis=0)
    
    # Within-chain variance
    chain_vars = np.var(chains, axis=1, ddof=1)
    W = np.mean(chain_vars, axis=0)
    
    # Between-chain variance
    B = (N / (M - 1.0)) * np.sum((chain_means - overall_mean) ** 2, axis=0)
    
    # Estimated marginal variance
    var_theta = ((N - 1.0) / N) * W + (1.0 / N) * B
    
    rhat_flat = np.empty_like(var_theta)
    zero_w = W == 0.0
    rhat_flat[zero_w] = 1.0
    rhat_flat[~zero_w] = np.sqrt(var_theta[~zero_w] / W[~zero_w])
    
    if param_shape:
        return rhat_flat.reshape(param_shape)
    return rhat_flat[0]

def effective_sample_size(trace: np.ndarray) -> np.ndarray:
    """Compute Effective Sample Size (ESS) for MCMC chains."""
    chains, param_shape = _prepare_chains(trace)
    M, N, D = chains.shape
    
    ess_flat = np.empty(D)
    
    for d in range(D):
        rhos = []
        for m in range(M):
            x = chains[m, :, d]
            mean = np.mean(x)
            var = np.var(x)
            if var == 0.0:
                rhos.append(np.zeros(N))
                continue
            
            xp = x - mean
            corr = np.correlate(xp, xp, mode='full')
            autocorr = corr[N-1:] / (N * var)
            rhos.append(autocorr)
            
        avg_rho = np.mean(rhos, axis=0)
        
        sum_rho = 0.0
        for t in range(1, N - 1, 2):
            val = avg_rho[t] + avg_rho[t+1]
            if val < 0:
                break
            sum_rho += avg_rho[t] + avg_rho[t+1]
            
        ess_flat[d] = (M * N) / (1.0 + 2.0 * sum_rho)
        
    if param_shape:
        return ess_flat.reshape(param_shape)
    return ess_flat[0]
