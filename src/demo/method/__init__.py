r"""
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
    DemandParameters,
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
    compute_total_demand_served,
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

from .revenue import RevenueFunction, create_linear_revenue

from .platform_utility import (
    compute_platform_utility,
    compute_platform_utility_derivative,
    compute_realized_utility,
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
    # compute_utility_gradient,
    # analyze_payment_range,
)

from .experiment import (
    ExperimentParams,
    DemandConfig,
    CurrentDemandState,
    EquilibriumAllocation,
    setup_rng,
    extract_demand_from_params,
    sample_current_state,
    compute_equilibrium_allocation,
    compute_weighted_average_payment,
    build_experiment_results,
    run_global_one_timepoint,
)

from .experiment_results import (
    TimePointData,
    ExperimentResults,
    Experiment,
    experiment_to_dataframe,
    compare_experiments,
    compute_cumulative_regret,
    analyze_convergence,
)

from .local_experimentation import (
    GradientEstimate,
    OptimizationState,
    generate_payment_perturbations,
    run_local_experiment,
    estimate_delta_hat,
    estimate_upsilon_hat,
    estimate_gamma_hat,
    estimate_utility_gradient,
    initialize_optimizer,
    mirror_descent_update,
    run_learning_algorithm,
    compute_experimentation_cost,
    compute_optimal_zeta,
)

from .global_experimentation import (
    run_global_experimentation,
)

from .bandit_experimentation import (
    fit_utility_spline,
    find_best_payment_from_history,
    sample_exploration_payment,
    compute_epsilon,
)


# Backwards compatibility alias
def run_baseline_global_learning(**kwargs):
    """
    Backwards-compatible alias for baseline global experimentation.

    This function calls run_global_experimentation with strategy="baseline".
    Use run_global_experimentation directly for new code.
    """
    return run_global_experimentation(strategy="baseline", **kwargs)


__all__ = [
    # Utils
    "numerical_derivative",
    # Demand
    "DemandParameters",
    "GlobalState",
    "sample_state",
    "sample_demand",
    "expected_demand",
    # Allocation
    "AllocationFunction",
    "create_queue_allocation",
    "create_linear_allocation",
    "create_smooth_linear_allocation",
    "create_simple_allocation",
    "compute_expected_allocation",
    "compute_expected_allocation_derivative",
    "compute_total_demand_served",
    # Supplier
    "ChoiceFunction",
    "PrivateFeatureDistribution",
    "SupplierParameters",
    "create_logistic_choice",
    "create_lognormal_costs",
    "create_uniform_costs",
    "compute_activation_probability",
    "compute_activation_sensitivity",
    "sample_supplier_activations",
    # Revenue
    "RevenueFunction",
    "create_linear_revenue",
    # Platform Utility
    "compute_platform_utility",
    "compute_platform_utility_derivative",
    "compute_realized_utility",
    # Equilibrium
    "MeanFieldEquilibrium",
    "find_equilibrium_supply_mu",
    "compute_mean_field_equilibrium",
    # Marginal Response
    "MarginalResponseAnalysis",
    "compute_marginal_response",
    "compute_supply_gradient",
    "analyze_marginal_response",
    # Experiment Setup & Helpers
    "ExperimentParams",
    "DemandConfig",
    "CurrentDemandState",
    "EquilibriumAllocation",
    "setup_rng",
    "extract_demand_from_params",
    "sample_current_state",
    "compute_equilibrium_allocation",
    "compute_weighted_average_payment",
    "build_experiment_results",
    "run_global_one_timepoint",
    # Experiment Results
    "TimePointData",
    "ExperimentResults",
    "Experiment",
    "experiment_to_dataframe",
    "compare_experiments",
    "compute_cumulative_regret",
    "analyze_convergence",
    # Local Experimentation
    "GradientEstimate",
    "OptimizationState",
    "generate_payment_perturbations",
    "run_local_experiment",
    "estimate_delta_hat",
    "estimate_upsilon_hat",
    "estimate_gamma_hat",
    "estimate_utility_gradient",
    "initialize_optimizer",
    "mirror_descent_update",
    "run_learning_algorithm",
    "compute_experimentation_cost",
    "compute_optimal_zeta",
    # Global Experimentation
    "run_global_experimentation",
    # Bandit Utilities
    "fit_utility_spline",
    "find_best_payment_from_history",
    "sample_exploration_payment",
    "compute_epsilon",
    # Backwards Compatibility
    "run_baseline_global_learning",
]
