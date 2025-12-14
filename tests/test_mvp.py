"""
Tests for refactored demo allocation and market_platform modules.

This test file verifies that the functional refactoring maintains
the same behavior as the original class-based implementation.
"""

import pytest
import numpy as np
from demo import (
    # Allocation
    AllocationFunction,
    create_queue_allocation,
    create_linear_allocation,
    create_smooth_linear_allocation,
    compute_expected_allocation,
    # Supplier
    create_logistic_choice,
    create_lognormal_costs,
    SupplierParameters,
    compute_activation_probability,
    # Market Platform
    create_linear_revenue,
    MarketParameters,
    find_equilibrium_supply_mu,
    simulate_market_period,
)


class TestAllocationFunctions:
    """Test allocation function creation and evaluation."""

    def test_queue_allocation_creation(self):
        """Test creating queue allocation function."""
        allocation = create_queue_allocation(L=8)
        assert isinstance(allocation, AllocationFunction)
        assert allocation.name == "M/M/1 Queue (L=8)"
        assert allocation.omega is not None
        assert allocation.omega_prime is not None

    def test_queue_allocation_properties(self):
        """Test queue allocation function properties."""
        allocation = create_queue_allocation(L=8)

        # Test boundary conditions
        assert allocation(0) == 0.0  # ω(0) = 0
        assert allocation(1.0) == pytest.approx(1 - 1/8)  # ω(1) = 1 - 1/L
        assert allocation(10.0) == pytest.approx(1.0)  # ω(∞) → 1

        # Test non-decreasing
        x_values = [0.1, 0.5, 1.0, 1.5, 2.0]
        omega_values = [allocation(x) for x in x_values]
        for i in range(len(omega_values) - 1):
            assert omega_values[i] <= omega_values[i+1]

    def test_linear_allocation(self):
        """Test linear allocation function."""
        allocation = create_linear_allocation()

        assert allocation(0.0) == 0.0
        assert allocation(0.5) == 0.5
        assert allocation(1.0) == 1.0
        assert allocation(2.0) == 1.0

    def test_smooth_linear_allocation(self):
        """Test smooth linear allocation function."""
        allocation = create_smooth_linear_allocation()

        assert allocation(0.0) == pytest.approx(0.0)
        # 1 - exp(-x) approaches 1 as x → ∞
        assert allocation(10.0) == pytest.approx(1.0, abs=0.01)

    def test_omega_derivative(self):
        """Test allocation function derivatives."""
        allocation = create_queue_allocation(L=8)

        # Derivative should be positive for valid x
        assert allocation.derivative(0.5) > 0
        assert allocation.derivative(1.0) > 0

        # Derivative should approach 1 as x → 0
        assert allocation.derivative(0.001) == pytest.approx(1.0, abs=0.01)

    def test_expected_allocation(self):
        """Test expected allocation computation."""
        allocation = create_queue_allocation(L=8)

        d_a = 0.4
        mu = 0.5

        q = compute_expected_allocation(allocation, mu, d_a)

        # q should equal ω(d_a/μ)
        x = d_a / mu
        expected = allocation(x)
        assert q == pytest.approx(expected)

    def test_allocation_invalid_capacity(self):
        """Test that invalid queue capacity raises error."""
        with pytest.raises(ValueError, match="Queue capacity L must be at least 2"):
            create_queue_allocation(L=1)


class TestSupplierParameters:
    """Test supplier choice and parameter creation."""

    def test_logistic_choice_creation(self):
        """Test creating logistic choice function."""
        choice = create_logistic_choice(alpha=1.0)
        assert choice.name == "Logistic (α=1.0)"
        assert choice.f is not None
        assert choice.f_prime is not None

    def test_logistic_choice_properties(self):
        """Test logistic choice function properties."""
        choice = create_logistic_choice(alpha=1.0)

        # Test that choice probability is in [0, 1]
        for x in [0, 10, 20, 30, 50]:
            for b in [5, 15, 25]:
                prob = choice(x, b)
                assert 0 <= prob <= 1

        # Test monotonicity: higher revenue → higher probability
        b = 20
        prob_low = choice(10, b)
        prob_high = choice(30, b)
        assert prob_high > prob_low

    def test_lognormal_costs(self):
        """Test lognormal cost distribution."""
        costs = create_lognormal_costs(log_mean=0.0, log_std=1.0, scale=20.0)
        assert costs.name == "LogNormal(μ=0.0, σ=1.0, scale=20.0)"

        # Sample should return positive values
        samples = costs.sample(100)
        assert all(s > 0 for s in samples)
        assert len(samples) == 100


    def test_activation_probability(self):
        """Test activation probability computation."""
        choice = create_logistic_choice(alpha=1.0)
        costs = create_lognormal_costs()
        params = SupplierParameters(
            choice=choice,
            private_features=costs,
        )

        # Test that activation increases with revenue
        prob_low = compute_activation_probability(params, 10, n_monte_carlo=10000)
        prob_high = compute_activation_probability(params, 50, n_monte_carlo=10000)

        assert 0 <= prob_low <= 1
        assert 0 <= prob_high <= 1
        assert prob_high > prob_low


class TestMarketPlatform:
    """Test market platform functions."""

    def test_linear_revenue_creation(self):
        """Test creating linear revenue function."""
        allocation = create_queue_allocation(L=8)
        revenue_fn = create_linear_revenue(gamma=100.0, allocation=allocation)

        assert revenue_fn.name == "Linear (γ=100.0)"
        assert revenue_fn.r is not None
        assert revenue_fn.r_prime is not None

    def test_linear_revenue_properties(self):
        """Test linear revenue function properties."""
        allocation = create_queue_allocation(L=8)
        gamma = 100.0
        revenue_fn = create_linear_revenue(gamma, allocation)

        # r(x) should equal γ * ω(x)
        for x in [0.5, 1.0, 1.5]:
            r_val = revenue_fn.r(x)
            expected = gamma * allocation(x)
            assert r_val == pytest.approx(expected)


    def test_equilibrium_computation(self):
        """Test equilibrium activation rate computation."""
        allocation = create_queue_allocation(L=8)
        choice = create_logistic_choice(alpha=1.0)
        costs = create_lognormal_costs()
        d_a = 0.4
        p = 30.0
        mu = find_equilibrium_supply_mu(p, d_a, choice, costs, allocation)
        # mu should be in (0, 1)
        assert 0 < mu < 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])