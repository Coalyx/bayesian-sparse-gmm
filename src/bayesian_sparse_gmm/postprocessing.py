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
        
    ref_idx = int(np.argmax([np.std(s.w) for s in states]))
    ref_mu = states[ref_idx].mu
    K_max = ref_mu.shape[0]
    
    for state in states:
        diff = state.mu[:, np.newaxis, :] - ref_mu[np.newaxis, :, :]
        cost_matrix = np.sum(diff ** 2, axis=2)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        new_mu = np.empty_like(state.mu)
        new_w = np.empty_like(state.w)
        new_tau2 = np.empty_like(state.tau2)
        new_gamma = np.empty_like(state.gamma)
        new_z = np.empty_like(state.z)
        
        for r, c in zip(row_ind, col_ind):
            new_mu[c] = state.mu[r]
            new_w[c] = state.w[r]
            new_tau2[c] = state.tau2[r]
            new_gamma[c] = state.gamma[r]
            new_z[state.z == r] = c
            
        state.mu = new_mu
        state.w = new_w
        state.tau2 = new_tau2
        state.gamma = new_gamma
        state.z = new_z
        
    return states
