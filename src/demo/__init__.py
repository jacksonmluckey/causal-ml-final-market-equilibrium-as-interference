"""
demo package - re-exports from demo.method
"""

from .method import (
    # Utils
    numerical_derivative,
    # Demand
    DemandParameters,
    GlobalState,
    sample_state,
    sample_demand,
    expected_demand,
    # Allocation
    AllocationFunction,
    create_queue_allocation,
    create_linear_allocation,
    create_smooth_linear_allocation,
    create_simple_allocation,
    compute_expected_allocation,
    compute_expected_allocation_derivative,
    compute_total_demand_served,
    # Supplier
    ChoiceFunction,
    PrivateFeatureDistribution,
    SupplierParameters,
    create_logistic_choice,
    create_lognormal_costs,
    create_uniform_costs,
    compute_activation_probability,
    compute_activation_sensitivity,
    sample_supplier_activations,
    # Revenue
    RevenueFunction,
    create_linear_revenue,
    # Platform Utility
    compute_platform_utility,
    compute_platform_utility_derivative,
    compute_realized_utility,
    # Equilibrium
    MeanFieldEquilibrium,
    find_equilibrium_supply_mu,
    compute_mean_field_equilibrium,
    # Marginal Response
    MarginalResponseAnalysis,
    compute_marginal_response,
    compute_supply_gradient,
    analyze_marginal_response,
    # Experiment Setup & Helpers
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
    # Experiment Results
    TimePointData,
    ExperimentResults,
    Experiment,
    experiment_to_dataframe,
    compare_experiments,
    compute_cumulative_regret,
    analyze_convergence,
    # Local Experimentation
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
    # Global Experimentation
    run_global_experimentation,
    # Bandit Utilities
    fit_utility_spline,
    find_best_payment_from_history,
    sample_exploration_payment,
    compute_epsilon,
)


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
]
