import numpy as np
from typing import List
from scipy.optimize import linear_sum_assignment
from .state import SamplerState

def align_labels(states: List[SamplerState]) -> List[SamplerState]:
    """Align cluster labels across MCMC iterations to solve label switching.
    
    Uses the final state as the reference and applies the Hungarian algorithm
    (linear sum assignment) to match clusters in each state to the reference based
    on the Euclidean distance of their cluster means.
    """
    if not states:
        return states
        
    # Use the last state as the reference
    ref_mu = states[-1].mu
    K_max = ref_mu.shape[0]
    
    for state in states:
        # Compute cost matrix: squared Euclidean distance between means
        # state.mu is (K_max, p), ref_mu is (K_max, p)
        # diff is (K_max, K_max, p)
        diff = state.mu[:, np.newaxis, :] - ref_mu[np.newaxis, :, :]
        cost_matrix = np.sum(diff ** 2, axis=2)
        
        # row_ind corresponds to indices in state.mu
        # col_ind corresponds to indices in ref_mu
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        new_mu = np.empty_like(state.mu)
        new_w = np.empty_like(state.w)
        new_tau2 = np.empty_like(state.tau2)
        new_z = np.empty_like(state.z)
        
        for r, c in zip(row_ind, col_ind):
            new_mu[c] = state.mu[r]
            new_w[c] = state.w[r]
            new_tau2[c] = state.tau2[r]
            new_z[state.z == r] = c
            
        state.mu = new_mu
        state.w = new_w
        state.tau2 = new_tau2
        state.z = new_z
        
    return states
