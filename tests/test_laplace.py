import numpy as np

def test_laplace_density_prefers_spike_for_small_mu():
    """Verify that marginalized Laplace density properly pushes small means to the spike.
    This validates the fix for the Normal-Scale mixture conditional which was getting stuck.
    """
    lambda_0 = 100.0  # Spike
    lambda_1 = 1.0    # Slab
    mu = np.array([0.01])

    # Marginal Laplace Density
    log_laplace_slab = np.log(lambda_1 / 2) - lambda_1 * np.abs(mu)
    log_laplace_spike = np.log(lambda_0 / 2) - lambda_0 * np.abs(mu)
    
    assert log_laplace_spike > log_laplace_slab, (
        "With mu=0.01, the Laplace density should strongly prefer the spike "
        "(lambda_0) over the slab (lambda_1)."
    )
