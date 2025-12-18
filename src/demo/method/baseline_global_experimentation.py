r"""
Baseline Global Experimentation (Two-Phase)

This is a simple baseline for comparison with local experimentation.
Instead of adaptively updating payments based on gradient estimates,
this algorithm uses a two-phase approach:

Phase 1 (Exploration): Randomly sample payments from [p_min, p_max]
Phase 2 (Exploitation): Use the learned optimal payment

This represents a naive approach that doesn't leverage the structure
of the problem for gradient estimation.
"""

import numpy as np
from scipy.interpolate import UnivariateSpline
from typing import Optional, List, Tuple, overload

from .allocation import AllocationFunction
from .revenue import RevenueFunction
from .supplier import SupplierParameters
from .demand import DemandParameters, GlobalState
from .experiment_results import (
    TimePointData,
    ExperimentResults,
    Experiment
)
from .experiment import (
    ExperimentParams,
    setup_rng,
    extract_demand_from_params,
    sample_current_state,
    build_experiment_results,
    run_global_one_timepoint
)


def fit_utility_spline(
    payments: np.ndarray,
    utilities: np.ndarray,
    p_bounds: Tuple[float, float],
    smoothing: Optional[float] = None
) -> Tuple[UnivariateSpline, float]:
    r"""
    Fit a smooth spline to utility observations and find the maximizer.

    Parameters
    ----------
    payments : np.ndarray
        Array of payment values from exploration
    utilities : np.ndarray
        Array of corresponding utilities
    p_bounds : Tuple[float, float]
        Payment bounds [p_min, p_max]
    smoothing : Optional[float]
        Smoothing parameter for spline (if None, uses default)

    Returns
    -------
    spline : UnivariateSpline
        Fitted spline function
    p_optimal : float
        Learned optimal payment (maximizer of spline)
    """
    # Fit univariate spline
    if smoothing is None:
        # Use default smoothing based on number of points
        smoothing = len(payments) * 0.1

    # Sort by payment for spline fitting
    sort_idx = np.argsort(payments)
    p_sorted = payments[sort_idx]
    u_sorted = utilities[sort_idx]

    # Fit spline (k=3 for cubic spline if we have enough points)
    k = min(3, len(payments) - 1)
    spline = UnivariateSpline(p_sorted, u_sorted, k=k, s=smoothing)

    # Find maximizer by evaluating on a fine grid
    p_min, p_max = p_bounds
    p_grid = np.linspace(p_min, p_max, 1000)
    u_grid = spline(p_grid)

    # Find the payment with highest utility
    best_idx = np.argmax(u_grid)
    p_optimal = p_grid[best_idx]

    return spline, float(p_optimal)


@overload
def run_baseline_global_learning(
    *,
    params: ExperimentParams,
    T_explore: int,
    rng: Optional[np.random.Generator] = None,
    verbose: bool = False
) -> Experiment: ...


@overload
def run_baseline_global_learning(
    T: int,
    T_explore: int,
    n: int,
    revenue_fn: RevenueFunction,
    allocation: AllocationFunction,
    supplier_params: SupplierParameters,
    d_a: Optional[float] = None,
    demand_params: Optional[DemandParameters] = None,
    p_bounds: Tuple[float, float] = (0.0, float('inf')),
    rng: Optional[np.random.Generator] = None,
    rng_seed: Optional[int] = None,
    verbose: bool = False,
    *,
    params: None = None
) -> Experiment: ...


def run_baseline_global_learning(
    T: Optional[int] = None,
    T_explore: Optional[int] = None,
    n: Optional[int] = None,
    revenue_fn: Optional[RevenueFunction] = None,
    allocation: Optional[AllocationFunction] = None,
    supplier_params: Optional[SupplierParameters] = None,
    d_a: Optional[float] = None,
    demand_params: Optional[DemandParameters] = None,
    p_bounds: Optional[Tuple[float, float]] = None,
    rng: Optional[np.random.Generator] = None,
    rng_seed: Optional[int] = None,
    verbose: bool = False,
    *,
    params: Optional[ExperimentParams] = None
) -> Experiment:
    r"""
    Run baseline global experimentation with two-phase approach.

    Phase 1 (Exploration, t=1 to T_explore):
        - Randomly sample payment p_t ~ Uniform(p_min, p_max)
        - Observe utility U_t
        - Store (p_t, U_t)

    Learning Step:
        - Fit spline: Û(p) = spline({p_t}, {U_t})
        - Find maximizer: p̂ = argmax Û(p)

    Phase 2 (Exploitation, t=T_explore+1 to T):
        - Deploy learned payment: p_t = p̂
        - Observe utility U_t

    Parameters
    ----------
    T : int
        Total number of periods
    T_explore : int
        Number of exploration periods (≤ T)
    n : int
        Number of suppliers
    revenue_fn : RevenueFunction
        Platform revenue function
    allocation : AllocationFunction
        The allocation function
    supplier_params : SupplierParameters
        Supplier behavior parameters
    d_a : Optional[float]
        Fixed expected demand per supplier
    demand_params : Optional[DemandParameters]
        Demand model (if using states)
    p_bounds : Tuple[float, float]
        Payment bounds [p_min, p_max]
    rng : Optional[np.random.Generator]
        Random number generator
    rng_seed : Optional[int]
        Random seed
    verbose : bool
        Whether to print progress

    Returns
    -------
    Experiment
        Complete experiment with parameters and results

    Notes
    -----
    Can be called in two ways:
    1. With ExperimentParams: run_baseline_global_learning(params=exp_params, T_explore=50)
    2. With individual parameters: run_baseline_global_learning(T=100, T_explore=50, n=1000, ...)
    """
    # Extract parameters from ExperimentParams if provided
    if params is not None:
        T = params.T
        n = params.n
        revenue_fn = params.revenue_fn
        allocation = params.allocation
        supplier_params = params.supplier_params
        p_bounds = params.p_bounds
        rng_seed = params.rng_seed

        # Extract demand parameters
        demand_config = extract_demand_from_params(params)
        d_a = demand_config.d_a
        demand_params = demand_config.demand_params
    else:
        # Validate that required parameters are provided (except T_explore which is checked separately)
        if any(x is None for x in [T, n, revenue_fn, allocation, supplier_params]):
            raise ValueError(
                "When params is not provided, all of T, n, "
                "revenue_fn, allocation, and supplier_params must be provided"
            )
        if p_bounds is None:
            p_bounds = (0.0, float('inf'))

    # Check T_explore separately to provide specific error message
    if T_explore is None:
        raise ValueError("T_explore must be provided")

    if T_explore > T:
        raise ValueError(f"T_explore ({T_explore}) must be <= T ({T})")

    # Setup RNG
    rng = setup_rng(rng, rng_seed)

    p_bounds = (float(p_bounds[0]), float(p_bounds[1]))
    p_min, p_max = p_bounds

    # Create experiment parameters (only if not already provided)
    if params is None:
        params = ExperimentParams(
            T=T,
            n=n,
            p_init=None,  # Not used in baseline
            revenue_fn=revenue_fn,
            p_bounds=p_bounds,
            allocation=allocation,
            supplier_params=supplier_params,
            demand=demand_params if demand_params is not None else d_a,
            eta=None,  # Not used in baseline
            experiment_type="baseline_global",
            zeta=None,
            alpha=None,
            delta=None,
            rng_seed=rng_seed,
            store_detailed_data=False
        )

    timepoints: List[TimePointData] = []
    exploration_payments: List[float] = []
    exploration_utilities: List[float] = []

    # =========================================================================
    # PHASE 1: EXPLORATION
    # =========================================================================
    if verbose:
        print(f"Phase 1: Exploration (t=1 to {T_explore})")

    for t in range(1, T_explore + 1):
        if verbose and t % 20 == 0:
            print(f"  Period {t}/{T_explore}")

        # Step 1: Draw payment uniformly from [p_min, p_max]
        p_t = rng.uniform(p_min, p_max)

        # Step 2: Sample state for this period
        current = sample_current_state(demand_params, d_a, rng)
        state = current.state
        current_d_a = current.d_a

        # Step 3: Run experiment with payment p_t
        timepoint = run_global_one_timepoint(
            n=n,
            p=p_t,
            revenue_fn=revenue_fn,
            allocation=allocation,
            supplier_params=supplier_params,
            t=t,
            d_a=current_d_a,
            demand_params=demand_params,
            state=state,
            rng=rng
        )

        # Step 4: Store (p_t, U_t)
        exploration_payments.append(p_t)
        exploration_utilities.append(timepoint.U)
        timepoints.append(timepoint)

    # =========================================================================
    # LEARNING STEP
    # =========================================================================
    if verbose:
        print(f"\nLearning Step: Fitting spline to {T_explore} observations")

    payments_array = np.array(exploration_payments)
    utilities_array = np.array(exploration_utilities)

    spline, p_hat = fit_utility_spline(
        payments=payments_array,
        utilities=utilities_array,
        p_bounds=p_bounds
    )

    if verbose:
        print(f"  Learned optimal payment: p̂ = {p_hat:.4f}")

    # =========================================================================
    # PHASE 2: EXPLOITATION
    # =========================================================================
    if verbose:
        print(f"\nPhase 2: Exploitation (t={T_explore+1} to {T})")

    for t in range(T_explore + 1, T + 1):
        if verbose and t % 20 == 0:
            print(f"  Period {t}/{T}")

        # Step 1: Sample state for this period
        current = sample_current_state(demand_params, d_a, rng)
        state = current.state
        current_d_a = current.d_a

        # Step 2: Deploy learned payment p̂
        timepoint = run_global_one_timepoint(
            n=n,
            p=p_hat,
            revenue_fn=revenue_fn,
            allocation=allocation,
            supplier_params=supplier_params,
            t=t,
            d_a=current_d_a,
            demand_params=demand_params,
            state=state,
            rng=rng
        )

        timepoints.append(timepoint)

    # =========================================================================
    # BUILD RESULTS
    # =========================================================================
    results = build_experiment_results(
        timepoints=timepoints,
        final_payment=p_hat,
        weighted_average_payment=None  # Not used for baseline
    )

    return Experiment(params=params, results=results)
