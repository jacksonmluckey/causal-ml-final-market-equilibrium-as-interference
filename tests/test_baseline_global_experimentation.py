"""
Test baseline global experimentation implementation.

This test verifies that the two-phase baseline global experimentation
algorithm works correctly: random exploration followed by exploitation
of the learned optimal payment.
"""

import numpy as np
import pytest
from demo.method.baseline_global_experimentation import (
    run_baseline_global_learning,
    fit_utility_spline
)
from demo.method import (
    create_queue_allocation,
    create_linear_revenue,
    create_logistic_choice,
    create_lognormal_costs,
    SupplierParameters,
)


def test_fit_utility_spline():
    """Test that spline fitting finds reasonable maximum."""
    # Create synthetic data with known maximum at p=5
    p_values = np.linspace(0, 10, 20)
    # Quadratic with maximum at p=5
    u_values = 100 - (p_values - 5)**2 + np.random.normal(0, 1, 20)

    spline, p_optimal = fit_utility_spline(
        payments=p_values,
        utilities=u_values,
        p_bounds=(0.0, 10.0),
        smoothing=1.0
    )

    # Should find optimal around p=5
    assert 4.0 <= p_optimal <= 6.0, f"Expected optimal around 5, got {p_optimal}"


def test_baseline_global_learning_runs():
    """Test that baseline global learning completes without errors."""
    # Setup simple experiment
    rng = np.random.default_rng(42)

    # Create allocation function
    allocation = create_queue_allocation(L=8)

    # Create revenue function
    revenue_fn = create_linear_revenue(gamma=100.0, allocation=allocation)

    # Create supplier parameters
    choice = create_logistic_choice(alpha=1.0)
    costs = create_lognormal_costs()
    supplier_params = SupplierParameters(
        choice=choice,
        private_features=costs
    )

    # Run baseline with small parameters
    T = 30
    T_explore = 20
    n = 100

    result = run_baseline_global_learning(
        T=T,
        T_explore=T_explore,
        n=n,
        revenue_fn=revenue_fn,
        allocation=allocation,
        supplier_params=supplier_params,
        d_a=0.5,
        p_bounds=(0.5, 5.0),
        rng=rng,
        verbose=False
    )

    # Check that we got results
    assert result.results is not None
    assert len(result.results.timepoints) == T
    assert result.results.final_payment is not None

    # Check phases
    # First T_explore should have random payments
    exploration_payments = [tp.p for tp in result.results.timepoints[:T_explore]]
    # They should be diverse (not all the same)
    assert len(set(np.round(exploration_payments, 2))) > 5, "Exploration should sample diverse payments"

    # Last phase should all use the learned payment
    exploitation_payments = [tp.p for tp in result.results.timepoints[T_explore:]]
    if len(exploitation_payments) > 0:
        # All should be the same (the learned optimal)
        assert all(np.isclose(p, result.results.final_payment, atol=1e-6)
                   for p in exploitation_payments), "Exploitation should use learned payment"


def test_baseline_requires_T_explore():
    """Test that T_explore parameter is required."""
    allocation = create_queue_allocation(L=8)
    revenue_fn = create_linear_revenue(gamma=100.0, allocation=allocation)
    choice = create_logistic_choice(alpha=1.0)
    costs = create_lognormal_costs()
    supplier_params = SupplierParameters(choice=choice, private_features=costs)

    with pytest.raises(ValueError, match="T_explore must be provided"):
        run_baseline_global_learning(
            T=100,
            T_explore=None,
            n=100,
            revenue_fn=revenue_fn,
            allocation=allocation,
            supplier_params=supplier_params,
            d_a=0.5
        )


def test_baseline_T_explore_validation():
    """Test that T_explore must be <= T."""
    allocation = create_queue_allocation(L=8)
    revenue_fn = create_linear_revenue(gamma=100.0, allocation=allocation)
    choice = create_logistic_choice(alpha=1.0)
    costs = create_lognormal_costs()
    supplier_params = SupplierParameters(choice=choice, private_features=costs)

    with pytest.raises(ValueError, match="T_explore.*must be <= T"):
        run_baseline_global_learning(
            T=50,
            T_explore=100,  # Invalid: T_explore > T
            n=100,
            revenue_fn=revenue_fn,
            allocation=allocation,
            supplier_params=supplier_params,
            d_a=0.5
        )
