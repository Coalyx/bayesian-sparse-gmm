from typing import List

import numpy as np
from scipy.optimize import linear_sum_assignment

from .state import SamplerState


def _find_reference_state(states: List[SamplerState], X: np.ndarray) -> int:
    """Find b* = argmin_b ||X - mu^(b) @ L^(b).T||_F^2.

    L^(b) is the assignment matrix where L[i,k] = 1 if z[i] == k, else 0.
    Equivalently, the reconstructed X is state.mu[state.z].
    """
    best_idx = 0
    best_error = np.inf

    for b, state in enumerate(states):
        X_hat = state.mu[state.z]  # shape (n, p)
        error = np.sum((X - X_hat) ** 2)
        if error < best_error:
            best_error = error
            best_idx = b

    return best_idx


def align_labels(states: List[SamplerState], X: np.ndarray) -> List[SamplerState]:
    """Align cluster labels across MCMC iterations to solve label switching.

    Selects the reference iteration as the one minimising reconstruction error
    ||X - mu^(b) @ L^(b).T||_F^2 (paper §9), then applies the Hungarian
    algorithm to match every other state to that reference.

    Parameters
    ----------
    states:
        Post-burn-in sampler states.
    X:
        Original data matrix of shape (n, p).

    Returns
    -------
    List[SamplerState]
        States with permuted cluster indices for label-switching alignment.
    """
    if not states:
        return states

    ref_idx = _find_reference_state(states, X)
    ref_mu = states[ref_idx].mu

    for state in states:
        diff = state.mu[:, np.newaxis, :] - ref_mu[np.newaxis, :, :]
        cost_matrix = np.sum(diff**2, axis=2)
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
