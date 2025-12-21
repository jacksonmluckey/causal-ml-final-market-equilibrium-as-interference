"""
Test global experimentation with multiple bandit strategies.

This test suite verifies:
1. Baseline strategy (two-phase explore/exploit)
2. Epsilon-greedy strategy (continuous explore/exploit)
3. Wrapper function correctly dispatches to strategies
4. Shared utilities work correctly
"""

import numpy as np
import pytest
from demo.method import (
    run_global_experimentation,
    fit_utility_spline,
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
    u_values = 100 - (p_values - 5) ** 2 + np.random.normal(0, 1, 20)

    spline, p_optimal = fit_utility_spline(
        payments=p_values, utilities=u_values, p_bounds=(0.0, 10.0), smoothing=1.0
    )

    # Should find optimal around p=5
    assert 4.0 <= p_optimal <= 6.0, f"Expected optimal around 5, got {p_optimal}"


def test_baseline_strategy_runs():
    """Test that baseline strategy completes without errors."""
    # Setup simple experiment
    rng = np.random.default_rng(42)

    # Create allocation function
    allocation = create_queue_allocation(L=8)

    # Create revenue function
    revenue_fn = create_linear_revenue(gamma=100.0, allocation=allocation)

    # Create supplier parameters
    choice = create_logistic_choice(alpha=1.0)
    costs = create_lognormal_costs()
    supplier_params = SupplierParameters(choice=choice, private_features=costs)

    # Run baseline with small parameters
    T = 30
    T_explore = 20
    n = 100

    result = run_global_experimentation(
        strategy="baseline",
        T=T,
        T_explore=T_explore,
        n=n,
        revenue_fn=revenue_fn,
        allocation=allocation,
        supplier_params=supplier_params,
        d_a=0.5,
        p_bounds=(0.5, 5.0),
        rng=rng,
        verbose=False,
    )

    # Check that we got results
    assert result.results is not None
    assert len(result.results.timepoints) == T
    assert result.results.final_payment is not None

    # Check phases
    # First T_explore should have random payments
    exploration_payments = [tp.p for tp in result.results.timepoints[:T_explore]]
    # They should be diverse (not all the same)
    assert len(set(np.round(exploration_payments, 2))) > 5, (
        "Exploration should sample diverse payments"
    )

    # Last phase should all use the learned payment
    exploitation_payments = [tp.p for tp in result.results.timepoints[T_explore:]]
    if len(exploitation_payments) > 0:
        # All should be the same (the learned optimal)
        assert all(
            np.isclose(p, result.results.final_payment, atol=1e-6)
            for p in exploitation_payments
        ), "Exploitation should use learned payment"


def test_baseline_requires_T_explore():
    """Test that T_explore parameter is required for baseline strategy."""
    allocation = create_queue_allocation(L=8)
    revenue_fn = create_linear_revenue(gamma=100.0, allocation=allocation)
    choice = create_logistic_choice(alpha=1.0)
    costs = create_lognormal_costs()
    supplier_params = SupplierParameters(choice=choice, private_features=costs)

    with pytest.raises(ValueError, match="T_explore must be provided"):
        run_global_experimentation(
            strategy="baseline",
            T=100,
            T_explore=None,
            n=100,
            revenue_fn=revenue_fn,
            allocation=allocation,
            supplier_params=supplier_params,
            d_a=0.5,
        )


def test_baseline_T_explore_validation():
    """Test that T_explore must be <= T."""
    allocation = create_queue_allocation(L=8)
    revenue_fn = create_linear_revenue(gamma=100.0, allocation=allocation)
    choice = create_logistic_choice(alpha=1.0)
    costs = create_lognormal_costs()
    supplier_params = SupplierParameters(choice=choice, private_features=costs)

    with pytest.raises(ValueError, match="T_explore.*must be <= T"):
        run_global_experimentation(
            strategy="baseline",
            T=50,
            T_explore=100,  # Invalid: T_explore > T
            n=100,
            revenue_fn=revenue_fn,
            allocation=allocation,
            supplier_params=supplier_params,
            d_a=0.5,
        )


def test_epsilon_greedy_strategy_runs():
    """Test that epsilon-greedy strategy completes without errors."""
    rng = np.random.default_rng(42)

    allocation = create_queue_allocation(L=8)
    revenue_fn = create_linear_revenue(gamma=100.0, allocation=allocation)
    choice = create_logistic_choice(alpha=1.0)
    costs = create_lognormal_costs()
    supplier_params = SupplierParameters(choice=choice, private_features=costs)

    T = 50
    n = 100

    result = run_global_experimentation(
        strategy="epsilon_greedy",
        T=T,
        n=n,
        p_init=2.0,  # Required for epsilon-greedy
        epsilon=0.2,
        revenue_fn=revenue_fn,
        allocation=allocation,
        supplier_params=supplier_params,
        d_a=0.5,
        p_bounds=(0.5, 5.0),
        rng=rng,
        verbose=False,
    )

    # Check that we got results
    assert result.results is not None
    assert len(result.results.timepoints) == T
    assert result.results.final_payment is not None


def test_epsilon_greedy_requires_p_init():
    """Test that p_init parameter is required for epsilon-greedy."""
    allocation = create_queue_allocation(L=8)
    revenue_fn = create_linear_revenue(gamma=100.0, allocation=allocation)
    choice = create_logistic_choice(alpha=1.0)
    costs = create_lognormal_costs()
    supplier_params = SupplierParameters(choice=choice, private_features=costs)

    with pytest.raises(ValueError, match="p_init must be provided"):
        run_global_experimentation(
            strategy="epsilon_greedy",
            T=50,
            n=100,
            p_init=None,  # Missing required parameter
            revenue_fn=revenue_fn,
            allocation=allocation,
            supplier_params=supplier_params,
            d_a=0.5,
        )


def test_epsilon_greedy_with_linear_decay():
    """Test epsilon-greedy with linear decay."""
    rng = np.random.default_rng(42)

    allocation = create_queue_allocation(L=8)
    revenue_fn = create_linear_revenue(gamma=100.0, allocation=allocation)
    choice = create_logistic_choice(alpha=1.0)
    costs = create_lognormal_costs()
    supplier_params = SupplierParameters(choice=choice, private_features=costs)

    result = run_global_experimentation(
        strategy="epsilon_greedy",
        T=50,
        n=100,
        p_init=2.0,
        epsilon=0.3,
        epsilon_decay="linear",
        revenue_fn=revenue_fn,
        allocation=allocation,
        supplier_params=supplier_params,
        d_a=0.5,
        p_bounds=(0.5, 5.0),
        rng=rng,
        verbose=False,
    )

    assert result.results is not None
    assert len(result.results.timepoints) == 50


def test_epsilon_greedy_exploration_strategies():
    """Test epsilon-greedy with different exploration strategies."""
    rng = np.random.default_rng(42)

    allocation = create_queue_allocation(L=8)
    revenue_fn = create_linear_revenue(gamma=100.0, allocation=allocation)
    choice = create_logistic_choice(alpha=1.0)
    costs = create_lognormal_costs()
    supplier_params = SupplierParameters(choice=choice, private_features=costs)

    # Test adaptive_step
    result1 = run_global_experimentation(
        strategy="epsilon_greedy",
        T=30,
        n=100,
        p_init=2.0,
        epsilon=0.2,
        exploration_strategy="adaptive_step",
        step_size_pct=0.1,
        revenue_fn=revenue_fn,
        allocation=allocation,
        supplier_params=supplier_params,
        d_a=0.5,
        p_bounds=(0.5, 5.0),
        rng=rng,
    )
    assert result1.results is not None

    # Test uniform
    result2 = run_global_experimentation(
        strategy="epsilon_greedy",
        T=30,
        n=100,
        p_init=2.0,
        epsilon=0.2,
        exploration_strategy="uniform",
        revenue_fn=revenue_fn,
        allocation=allocation,
        supplier_params=supplier_params,
        d_a=0.5,
        p_bounds=(0.5, 5.0),
        rng=rng,
    )
    assert result2.results is not None


def test_invalid_strategy_raises_error():
    """Test that invalid strategy raises ValueError."""
    allocation = create_queue_allocation(L=8)
    revenue_fn = create_linear_revenue(gamma=100.0, allocation=allocation)
    choice = create_logistic_choice(alpha=1.0)
    costs = create_lognormal_costs()
    supplier_params = SupplierParameters(choice=choice, private_features=costs)

    with pytest.raises(ValueError, match="Unknown strategy"):
        run_global_experimentation(
            strategy="invalid_strategy",
            T=50,
            n=100,
            revenue_fn=revenue_fn,
            allocation=allocation,
            supplier_params=supplier_params,
            d_a=0.5,
        )
