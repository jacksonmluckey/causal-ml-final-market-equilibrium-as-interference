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
    DemandParams,
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
    create_simple_allocation,
    compute_expected_allocation,
    compute_expected_allocation_derivative,
    compute_total_demand_served
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

from .market import (
    MarketParameters,
)

from .revenue import (
    RevenueFunction,
    create_linear_revenue,
    compute_total_revenue,
    compute_revenue_per_supplier,
    compute_revenue_derivative,
)

from .market_platform import (
    MarketOutcome,
    simulate_market_period,
)

from .platform_utility import (
    compute_platform_utility,
    compute_platform_utility_derivative,
        compute_mean_field_utility,
)

from .find_equilibrium import (
    MeanFieldEquilibrium,
    find_equilibrium_supply_mu,
    compute_mean_field_equilibrium,
)

from .marginal_response import (
    MarginalResponseAnalysis,
    compute_marginal_response,
    compute_supply_gradient,
    analyze_marginal_response,
    compute_utility_gradient,
    analyze_payment_range,
)

__all__ = [
    # Utils
    'numerical_derivative',
    # Demand
    'DemandParams', 'GlobalState',
    'sample_state', 'sample_demand', 'expected_demand',
    # Allocation
    'AllocationFunction',
    'create_queue_allocation', 'create_linear_allocation', 'create_smooth_linear_allocation',
    'create_simple_allocation', 'compute_expected_allocation',
    'compute_expected_allocation_derivative',
    'compute_total_demand_served',
    # Supplier
    'ChoiceFunction', 'PrivateFeatureDistribution', 'SupplierParameters',
    'create_logistic_choice', 'create_lognormal_costs', 'create_uniform_costs',
    'compute_activation_probability', 'compute_activation_sensitivity',
    'sample_supplier_activations',
    # Market Parameters
    'MarketParameters',
    # Revenue
    'RevenueFunction',
    'create_linear_revenue',
    'compute_total_revenue', 'compute_revenue_per_supplier', 'compute_revenue_derivative',
    # Market Platform
    'MarketOutcome',
    'compute_platform_utility', 'compute_platform_utility_derivative',
    'compute_mean_field_utility',
    'simulate_market_period',
    # Equilibrium
    'MeanFieldEquilibrium',
    'find_equilibrium_supply_mu', 'compute_mean_field_equilibrium',
    # Marginal Response
    'MarginalResponseAnalysis',
    'compute_marginal_response', 'compute_supply_gradient',
    'analyze_marginal_response', 'compute_utility_gradient',
    'analyze_payment_range',
]
