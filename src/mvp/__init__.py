"""
Stochastic Market Model

Implementation of the model from:
Wager & Xu (2021) "Experimenting in Equilibrium"

This package implements a centralized marketplace model where:
- A platform sets payments to attract suppliers
- Suppliers decide whether to become active based on expected revenue
- Demand is randomly allocated among active suppliers
- The platform aims to maximize utility (revenue - payments)

Main components:
- demand: Demand generation given global state
- allocation: Regular allocation functions ω(x)
- supplier: Supplier choice behavior
- platform: Platform utility and market simulation
"""

from .utils import numerical_derivative

from .demand import (
    DemandModel,
    GlobalState,
    sample_state,
    sample_demand,
    expected_demand,
)

from .allocation import (
    AllocationFunction,
    create_queue_allocation,
    create_linear_allocation,
    create_smooth_linear_allocation,
    compute_omega,
    compute_omega_derivative,
    compute_finite_allocation,
    compute_expected_allocation,
    compute_expected_allocation_derivative,
    compute_total_demand_served,
    compute_utilization,
)

from .supplier import (
    ChoiceFunction,
    PrivateFeatureDistribution,
    SupplierParameters,
    create_logistic_choice,
    create_lognormal_costs,
    create_uniform_costs,
    compute_activation_probability,
    compute_activation_sensitivity,
    sample_supplier_activations,
)

from .market_platform import (
    RevenueFunction,
    MarketOutcome,
    MarketParameters,
    create_linear_revenue,
    compute_platform_utility,
    compute_platform_utility_derivative,
    compute_mean_field_utility,
    simulate_market_period,
)

from .find_equilibrium import (
    MeanFieldEquilibrium,
    solve_equilibrium_supply,
    compute_mean_field_equilibrium,
)

from .mean_field import (
    MarketParameters as MeanFieldMarketParameters,
)

__all__ = [
    # Utils
    'numerical_derivative',
    # Demand
    'DemandModel', 'GlobalState',
    'sample_state', 'sample_demand', 'expected_demand',
    # Allocation
    'AllocationFunction',
    'create_queue_allocation', 'create_linear_allocation', 'create_smooth_linear_allocation',
    'compute_omega', 'compute_omega_derivative',
    'compute_finite_allocation', 'compute_expected_allocation',
    'compute_expected_allocation_derivative',
    'compute_total_demand_served', 'compute_utilization',
    # Supplier
    'ChoiceFunction', 'PrivateFeatureDistribution', 'SupplierParameters',
    'create_logistic_choice', 'create_lognormal_costs', 'create_uniform_costs',
    'compute_activation_probability', 'compute_activation_sensitivity',
    'sample_supplier_activations',
    # Market Platform
    'RevenueFunction', 'MarketOutcome', 'MarketParameters',
    'create_linear_revenue',
    'compute_platform_utility', 'compute_platform_utility_derivative',
    'compute_mean_field_utility',
    'simulate_market_period',
    # Equilibrium
    'MeanFieldEquilibrium', 'MeanFieldMarketParameters',
    'solve_equilibrium_supply', 'compute_mean_field_equilibrium',
]
