"""
Unit tests for experiment helper functions.

Tests the helper functions in src/demo/method/experiment.py that were
extracted during the refactoring to eliminate code duplication.
"""

import numpy as np
import pytest
from demo.method import (
    # Experiment helpers
    setup_rng,
    extract_demand_from_params,
    sample_current_state,
    compute_equilibrium_allocation,
    compute_weighted_average_payment,
    build_experiment_results,
    ExperimentParams,
    # Supporting dataclasses
    DemandConfig,
    CurrentDemandState,
    EquilibriumAllocation,
    # For test setup
    create_queue_allocation,
    create_logistic_choice,
    create_lognormal_costs,
    create_linear_revenue,
    SupplierParameters,
    DemandParameters,
    GlobalState,
    TimePointData,
)


class TestSetupRng:
    """Test RNG initialization helper."""

    def test_setup_rng_with_seed(self):
        """Test that RNG with seed produces deterministic results."""
        rng1 = setup_rng(None, 42)
        rng2 = setup_rng(None, 42)

        # Same seed should produce same sequence
        val1 = rng1.uniform(0, 1)
        val2 = rng2.uniform(0, 1)
        assert val1 == val2

    def test_setup_rng_without_seed(self):
        """Test that RNG without seed produces different sequences."""
        rng1 = setup_rng(None, None)
        rng2 = setup_rng(None, None)

        # Different RNGs should produce different sequences
        vals1 = [rng1.uniform(0, 1) for _ in range(10)]
        vals2 = [rng2.uniform(0, 1) for _ in range(10)]
        assert vals1 != vals2

    def test_setup_rng_preserves_existing(self):
        """Test that providing an existing RNG returns it unchanged."""
        rng_original = np.random.default_rng(123)
        val1 = rng_original.uniform(0, 1)

        rng_returned = setup_rng(rng_original, 999)  # seed should be ignored
        val2 = rng_returned.uniform(0, 1)

        # Should be the same RNG instance, continuing the same sequence
        assert rng_returned is rng_original
        assert val1 != val2  # Different draws from same sequence


class TestExtractDemandFromParams:
    """Test demand parameter extraction."""

    def test_extract_demand_float(self):
        """Test extraction when demand is a float (fixed d_a)."""
        allocation = create_queue_allocation(L=8)
        revenue_fn = create_linear_revenue(gamma=100.0, allocation=allocation)
        choice = create_logistic_choice(alpha=2.0)
        costs = create_lognormal_costs(log_mean=-1.0, log_std=0.5, scale=20.0)
        supplier_params = SupplierParameters(choice=choice, private_features=costs)

        params = ExperimentParams(
            T=100,
            n=1000,
            p_init=50.0,
            revenue_fn=revenue_fn,
            p_bounds=(0.0, 200.0),
            allocation=allocation,
            supplier_params=supplier_params,
            demand=0.5,  # Float
            eta=1.0,
            experiment_type="local",
            zeta=0.1,
        )

        config = extract_demand_from_params(params)
        assert isinstance(config, DemandConfig)
        assert config.d_a == 0.5
        assert config.demand_params is None

    def test_extract_demand_params(self):
        """Test extraction when demand is DemandParameters (stochastic)."""
        allocation = create_queue_allocation(L=8)
        revenue_fn = create_linear_revenue(gamma=100.0, allocation=allocation)
        choice = create_logistic_choice(alpha=2.0)
        costs = create_lognormal_costs(log_mean=-1.0, log_std=0.5, scale=20.0)
        supplier_params = SupplierParameters(choice=choice, private_features=costs)

        demand_params = DemandParameters(
            states={
                "Low": GlobalState(name="Low", d_a=0.3, probability=0.5),
                "High": GlobalState(name="High", d_a=0.7, probability=0.5),
            }
        )

        params = ExperimentParams(
            T=100,
            n=1000,
            p_init=50.0,
            revenue_fn=revenue_fn,
            p_bounds=(0.0, 200.0),
            allocation=allocation,
            supplier_params=supplier_params,
            demand=demand_params,  # DemandParameters
            eta=1.0,
            experiment_type="local",
            zeta=0.1,
        )

        config = extract_demand_from_params(params)
        assert isinstance(config, DemandConfig)
        assert config.d_a is None
        assert config.demand_params is demand_params


class TestSampleCurrentState:
    """Test state sampling helper."""

    def test_sample_current_state_fixed(self):
        """Test sampling with fixed demand (no DemandParameters)."""
        rng = np.random.default_rng(42)

        current = sample_current_state(None, 0.5, rng)

        assert isinstance(current, CurrentDemandState)
        assert current.state is None
        assert current.d_a == 0.5

    def test_sample_current_state_stochastic(self):
        """Test sampling with stochastic demand model."""
        demand_params = DemandParameters(
            states={
                "Low": GlobalState(name="Low", d_a=0.3, probability=0.3),
                "High": GlobalState(name="High", d_a=0.7, probability=0.7),
            }
        )
        rng = np.random.default_rng(42)

        # Sample multiple times to test stochasticity
        samples = [sample_current_state(demand_params, None, rng) for _ in range(100)]

        # All should have valid states
        assert all(isinstance(s, CurrentDemandState) for s in samples)
        assert all(s.state is not None for s in samples)
        assert all(s.d_a in [0.3, 0.7] for s in samples)

        # Should have mix of states (with high probability)
        d_a_values = [s.d_a for s in samples]
        assert 0.3 in d_a_values  # Low state appeared
        assert 0.7 in d_a_values  # High state appeared

    def test_sample_current_state_requires_one_parameter(self):
        """Test that either demand_params or d_a must be provided."""
        rng = np.random.default_rng(42)

        with pytest.raises(
            ValueError, match="Must provide either demand_params or d_a"
        ):
            sample_current_state(None, None, rng)


class TestComputeEquilibriumAllocation:
    """Test equilibrium computation helper."""

    def test_compute_equilibrium_allocation(self):
        """Test that equilibrium allocation is computed correctly."""
        allocation = create_queue_allocation(L=8)
        choice = create_logistic_choice(alpha=2.0)
        costs = create_lognormal_costs(log_mean=-1.0, log_std=0.5, scale=20.0)
        supplier_params = SupplierParameters(choice=choice, private_features=costs)

        p = 50.0
        d_a = 0.5

        eq = compute_equilibrium_allocation(p, d_a, supplier_params, allocation)

        assert isinstance(eq, EquilibriumAllocation)
        assert 0 <= eq.mu_eq <= 1  # Supply fraction
        assert eq.q_eq >= 0  # Expected allocation


class TestComputeWeightedAveragePayment:
    """Test weighted average payment computation."""

    def test_compute_weighted_average_payment(self):
        """Test weighted average formula: (2/T(T+1)) Σ t·p_t."""
        # Create simple timepoints with known payments
        timepoints = [
            TimePointData(t=1, p=10.0, D=100, T=50, S=40.0, U=100.0),
            TimePointData(t=2, p=20.0, D=100, T=50, S=40.0, U=100.0),
            TimePointData(t=3, p=30.0, D=100, T=50, S=40.0, U=100.0),
        ]

        weighted_avg = compute_weighted_average_payment(timepoints)

        # Manual calculation: (2/(3*4)) * (1*10 + 2*20 + 3*30)
        # = (2/12) * (10 + 40 + 90) = (1/6) * 140 = 23.333...
        expected = (2.0 / (3 * 4)) * (1 * 10 + 2 * 20 + 3 * 30)
        assert abs(weighted_avg - expected) < 1e-10

    def test_compute_weighted_average_payment_empty(self):
        """Test with empty timepoints list."""
        weighted_avg = compute_weighted_average_payment([])
        assert weighted_avg == 0.0


class TestBuildExperimentResults:
    """Test experiment results building."""

    def test_build_experiment_results(self):
        """Test that experiment results are built correctly."""
        timepoints = [
            TimePointData(t=1, p=10.0, D=100, T=50, S=40.0, U=100.0),
            TimePointData(t=2, p=20.0, D=100, T=50, S=40.0, U=150.0),
            TimePointData(t=3, p=30.0, D=100, T=50, S=40.0, U=200.0),
        ]

        results = build_experiment_results(
            timepoints=timepoints, final_payment=30.0, weighted_average_payment=25.0
        )

        assert results.final_payment == 30.0
        assert results.weighted_average_payment == 25.0
        assert results.average_payment == 20.0  # (10+20+30)/3
        assert results.total_utility == 450.0  # 100+150+200
        assert results.mean_utility == 150.0  # 450/3
        assert len(results.timepoints) == 3

    def test_build_experiment_results_no_weighted_average(self):
        """Test building results without weighted average (for global experiments)."""
        timepoints = [
            TimePointData(t=1, p=10.0, D=100, T=50, S=40.0, U=100.0),
        ]

        results = build_experiment_results(
            timepoints=timepoints, final_payment=10.0, weighted_average_payment=None
        )

        assert results.weighted_average_payment is None
        assert results.final_payment == 10.0
