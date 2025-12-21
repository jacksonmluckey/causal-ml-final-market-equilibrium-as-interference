r"""
Global Experimentation with Multiple Bandit Strategies

This module provides a unified interface to different bandit-based strategies
for global experimentation:

1. Baseline (strategy="baseline"):
   Two-phase approach with separate exploration and exploitation phases.
   Phase 1: Randomly sample payments from [p_min, p_max]
   Learn: Fit spline to observations, find optimal
   Phase 2: Deploy learned optimal payment

2. Epsilon-Greedy (strategy="epsilon_greedy"):
   At each timestep, explore with probability ε, exploit with 1-ε.
   Continuously updates best payment as better ones are found.

Future possible strategies:
- UCB (Upper Confidence Bound)
- Thompson Sampling
"""

import numpy as np
from typing import Optional, List, Tuple, overload

from .allocation import AllocationFunction
from .revenue import RevenueFunction
from .supplier import SupplierParameters
from .demand import DemandParameters
from .experiment_results import TimePointData, Experiment
from .experiment import (
    ExperimentParams,
    setup_rng,
    extract_demand_from_params,
    sample_current_state,
    build_experiment_results,
    run_global_one_timepoint,
)
from .bandit_experimentation import (
    fit_utility_spline,
    find_best_payment_from_history,
    sample_exploration_payment,
    compute_epsilon,
)


@overload
def run_global_experimentation(
    *,
    strategy: str,
    params: ExperimentParams,
    T_explore: Optional[int] = None,
    epsilon: Optional[float] = None,
    epsilon_decay: Optional[str] = None,
    decay_rate: Optional[float] = None,
    exploration_strategy: Optional[str] = None,
    step_size_pct: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
    verbose: bool = False,
) -> Experiment: ...


@overload
def run_global_experimentation(
    strategy: str,
    T: int,
    n: int,
    revenue_fn: RevenueFunction,
    allocation: AllocationFunction,
    supplier_params: SupplierParameters,
    d_a: Optional[float] = None,
    demand_params: Optional[DemandParameters] = None,
    p_bounds: Tuple[float, float] = (0.0, float("inf")),
    p_init: Optional[float] = None,
    T_explore: Optional[int] = None,
    epsilon: Optional[float] = None,
    epsilon_decay: Optional[str] = None,
    decay_rate: Optional[float] = None,
    exploration_strategy: Optional[str] = None,
    step_size_pct: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
    rng_seed: Optional[int] = None,
    verbose: bool = False,
    *,
    params: None = None,
) -> Experiment: ...


def run_global_experimentation(
    strategy: str,
    T: Optional[int] = None,
    n: Optional[int] = None,
    revenue_fn: Optional[RevenueFunction] = None,
    allocation: Optional[AllocationFunction] = None,
    supplier_params: Optional[SupplierParameters] = None,
    d_a: Optional[float] = None,
    demand_params: Optional[DemandParameters] = None,
    p_bounds: Optional[Tuple[float, float]] = None,
    p_init: Optional[float] = None,
    T_explore: Optional[int] = None,
    epsilon: Optional[float] = None,
    epsilon_decay: Optional[str] = None,
    decay_rate: Optional[float] = None,
    exploration_strategy: Optional[str] = None,
    step_size_pct: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
    rng_seed: Optional[int] = None,
    verbose: bool = False,
    *,
    params: Optional[ExperimentParams] = None,
) -> Experiment:
    r"""
    Run global experimentation with specified bandit strategy.

    Parameters
    ----------
    strategy : str
        Bandit strategy: "baseline" or "epsilon_greedy"
    T : int
        Total number of periods
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
    p_init : Optional[float]
        Initial payment (required for epsilon_greedy, optional for baseline)
    T_explore : Optional[int]
        Number of exploration periods (required for baseline)
    epsilon : Optional[float]
        Exploration probability (for epsilon_greedy, default=0.1)
    epsilon_decay : Optional[str]
        Decay schedule: None, "linear", or "exponential" (for epsilon_greedy)
    decay_rate : Optional[float]
        Decay rate for exponential schedule (for epsilon_greedy)
    exploration_strategy : Optional[str]
        Exploration method: "adaptive_step" or "uniform" (for epsilon_greedy, default="adaptive_step")
    step_size_pct : Optional[float]
        Step size as % of p_best for adaptive_step (for epsilon_greedy, default=0.1)
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
    1. With ExperimentParams: run_global_experimentation(strategy="baseline", params=exp_params, T_explore=50)
    2. With individual parameters: run_global_experimentation(strategy="baseline", T=100, T_explore=50, n=1000, ...)
    """
    # Dispatch to strategy-specific implementation
    if strategy == "baseline":
        return _run_baseline_strategy(
            T=T,
            n=n,
            revenue_fn=revenue_fn,
            allocation=allocation,
            supplier_params=supplier_params,
            d_a=d_a,
            demand_params=demand_params,
            p_bounds=p_bounds,
            T_explore=T_explore,
            rng=rng,
            rng_seed=rng_seed,
            verbose=verbose,
            params=params,
        )
    elif strategy == "epsilon_greedy":
        return _run_epsilon_greedy_strategy(
            T=T,
            n=n,
            revenue_fn=revenue_fn,
            allocation=allocation,
            supplier_params=supplier_params,
            d_a=d_a,
            demand_params=demand_params,
            p_bounds=p_bounds,
            p_init=p_init,
            epsilon=epsilon,
            epsilon_decay=epsilon_decay,
            decay_rate=decay_rate,
            exploration_strategy=exploration_strategy,
            step_size_pct=step_size_pct,
            rng=rng,
            rng_seed=rng_seed,
            verbose=verbose,
            params=params,
        )
    # elif strategy == "ucb":
    #     # TODO: Implement UCB strategy
    #     raise NotImplementedError("UCB strategy not yet implemented")
    # elif strategy == "thompson":
    #     # TODO: Implement Thompson Sampling strategy
    #     raise NotImplementedError("Thompson Sampling strategy not yet implemented")
    else:
        raise ValueError(
            f"Unknown strategy: {strategy}. Choose from: 'baseline', 'epsilon_greedy'"
        )


def _run_baseline_strategy(
    T: Optional[int] = None,
    n: Optional[int] = None,
    revenue_fn: Optional[RevenueFunction] = None,
    allocation: Optional[AllocationFunction] = None,
    supplier_params: Optional[SupplierParameters] = None,
    d_a: Optional[float] = None,
    demand_params: Optional[DemandParameters] = None,
    p_bounds: Optional[Tuple[float, float]] = None,
    T_explore: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
    rng_seed: Optional[int] = None,
    verbose: bool = False,
    *,
    params: Optional[ExperimentParams] = None,
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
    """
    # Extract parameters from ExperimentParams if provided
    if params is not None:
        T = params.T
        if params.T_explore is not None:
            T_explore = params.T_explore
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
        # Validate that required parameters are provided
        if any(x is None for x in [T, n, revenue_fn, allocation, supplier_params]):
            raise ValueError(
                "When params is not provided, all of T, n, "
                "revenue_fn, allocation, and supplier_params must be provided"
            )
        if p_bounds is None:
            p_bounds = (0.0, float("inf"))

    # Check T_explore separately
    if T_explore is None:
        raise ValueError("T_explore must be provided for baseline strategy")

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
            experiment_type="global",
            zeta=None,
            alpha=None,
            delta=None,
            rng_seed=rng_seed,
            store_detailed_data=False,
            T_explore=T_explore,
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
            rng=rng,
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
        payments=payments_array, utilities=utilities_array, p_bounds=p_bounds
    )

    if verbose:
        print(f"  Learned optimal payment: p̂ = {p_hat:.4f}")

    # =========================================================================
    # PHASE 2: EXPLOITATION
    # =========================================================================
    if verbose:
        print(f"\nPhase 2: Exploitation (t={T_explore + 1} to {T})")

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
            rng=rng,
        )

        timepoints.append(timepoint)

    # =========================================================================
    # BUILD RESULTS
    # =========================================================================
    results = build_experiment_results(
        timepoints=timepoints,
        final_payment=p_hat,
        weighted_average_payment=None,  # Not used for baseline
    )

    return Experiment(params=params, results=results)


def _run_epsilon_greedy_strategy(
    T: Optional[int] = None,
    n: Optional[int] = None,
    revenue_fn: Optional[RevenueFunction] = None,
    allocation: Optional[AllocationFunction] = None,
    supplier_params: Optional[SupplierParameters] = None,
    d_a: Optional[float] = None,
    demand_params: Optional[DemandParameters] = None,
    p_bounds: Optional[Tuple[float, float]] = None,
    p_init: Optional[float] = None,
    epsilon: Optional[float] = None,
    epsilon_decay: Optional[str] = None,
    decay_rate: Optional[float] = None,
    exploration_strategy: Optional[str] = None,
    step_size_pct: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
    rng_seed: Optional[int] = None,
    verbose: bool = False,
    *,
    params: Optional[ExperimentParams] = None,
) -> Experiment:
    r"""
    Run epsilon-greedy global experimentation.

    At each timestep t:
        1. Compute epsilon_t (with decay if enabled)
        2. With probability epsilon_t: EXPLORE (sample new payment)
           With probability 1 - epsilon_t: EXPLOIT (use best payment so far)
        3. Observe utility
        4. Update best payment if better utility found

    This strategy continuously balances exploration and exploitation,
    unlike the baseline which has separate phases.
    """
    # Extract parameters from ExperimentParams if provided
    if params is not None:
        T = params.T
        n = params.n
        p_init = params.p_init
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
        # Validate that required parameters are provided
        if any(x is None for x in [T, n, revenue_fn, allocation, supplier_params]):
            raise ValueError(
                "When params is not provided, all of T, n, "
                "revenue_fn, allocation, and supplier_params must be provided"
            )
        if p_bounds is None:
            p_bounds = (0.0, float("inf"))

    # Check p_init
    if p_init is None:
        raise ValueError("p_init must be provided for epsilon_greedy strategy")

    # Set defaults for epsilon-greedy parameters
    if epsilon is None:
        epsilon = 0.1
    if exploration_strategy is None:
        exploration_strategy = "adaptive_step"
    if step_size_pct is None:
        step_size_pct = 0.1

    # Setup RNG
    rng = setup_rng(rng, rng_seed)

    p_bounds = (float(p_bounds[0]), float(p_bounds[1]))
    p_min, p_max = p_bounds

    # Create experiment parameters (only if not already provided)
    if params is None:
        params = ExperimentParams(
            T=T,
            n=n,
            p_init=p_init,
            revenue_fn=revenue_fn,
            p_bounds=p_bounds,
            allocation=allocation,
            supplier_params=supplier_params,
            demand=demand_params if demand_params is not None else d_a,
            eta=None,  # Not used in epsilon-greedy
            experiment_type="global",
            zeta=None,
            alpha=None,
            delta=None,
            rng_seed=rng_seed,
            store_detailed_data=False,
        )

    # History tracking
    payment_history: List[float] = []
    utility_history: List[float] = []
    timepoints: List[TimePointData] = []

    # Start with p_init
    p_best = p_init

    if verbose:
        decay_str = f" with {epsilon_decay} decay" if epsilon_decay else ""
        print(
            f"Epsilon-Greedy Strategy (ε={epsilon}{decay_str}, strategy={exploration_strategy})"
        )

    # =========================================================================
    # MAIN LOOP
    # =========================================================================
    for t in range(1, T + 1):
        if verbose and t % 20 == 0:
            print(f"  Period {t}/{T}")

        # Step 1: Compute current epsilon (with decay if enabled)
        epsilon_t = compute_epsilon(epsilon, epsilon_decay, decay_rate, t, T)

        # Step 2: Decide explore vs exploit
        if rng.uniform() < epsilon_t:
            # EXPLORE: Sample new payment
            p_t = sample_exploration_payment(
                p_best=p_best,
                p_bounds=p_bounds,
                strategy=exploration_strategy,
                step_size_pct=step_size_pct,
                rng=rng,
            )
        else:
            # EXPLOIT: Use best payment from history
            p_t = p_best

        # Step 3: Sample state for this period
        current = sample_current_state(demand_params, d_a, rng)
        state = current.state
        current_d_a = current.d_a

        # Step 4: Run experiment with payment p_t
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
            rng=rng,
        )

        # Step 5: Update history
        payment_history.append(p_t)
        utility_history.append(timepoint.U)
        timepoints.append(timepoint)

        # Step 6: Update best payment if we found a better one
        if t == 1 or timepoint.U > max(utility_history[:-1]):
            p_best = p_t
        else:
            # Use the payment with historically highest utility
            p_best = find_best_payment_from_history(payment_history, utility_history)

    # =========================================================================
    # BUILD RESULTS
    # =========================================================================
    final_payment = p_best

    if verbose:
        print(f"\nFinal learned payment: p̂ = {final_payment:.4f}")

    results = build_experiment_results(
        timepoints=timepoints,
        final_payment=final_payment,
        weighted_average_payment=None,
    )

    return Experiment(params=params, results=results)
