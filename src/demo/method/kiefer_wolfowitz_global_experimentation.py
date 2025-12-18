r"""
Global Experimentation using Kiefer-Wolfowitz Algorithm

Section 4.4 of Wager & Xu (2021): Comparison with global experimentation.

In global experimentation, each period t we choose a SINGLE payment $p_t$
for all suppliers and observe aggregate utility $U_t$. This is a standard
continuous-armed bandit problem.

This implementation uses the Kiefer-Wolfowitz stochastic approximation algorithm
for gradient estimation via finite differences.

Key result (Shamir 2013): Even with strongly concave utility, no algorithm
can achieve expected regret growing slower than $\sqrt{T}$. This gives $1/\sqrt{T}$ rate
of decay in errors, compared to $1/T$ for local experimentation.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Tuple, overload

from .allocation import AllocationFunction
from .revenue import RevenueFunction
from .platform_utility import compute_realized_utility
from .supplier import SupplierParameters, sample_supplier_activations
from .demand import DemandParameters, GlobalState, sample_state, sample_demand
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
    compute_equilibrium_allocation,
    build_experiment_results,
    run_global_one_timepoint
)


@overload
def run_kiefer_wolfowitz_global_learning(
    *,
    params: ExperimentParams,
    rng: Optional[np.random.Generator] = None
) -> Experiment: ...


@overload
def run_kiefer_wolfowitz_global_learning(
    T: int,
    n: int,
    p_init: float,
    eta: float,
    delta: float,
    revenue_fn: RevenueFunction,
    allocation: AllocationFunction,
    supplier_params: SupplierParameters,
    d_a: Optional[float] = None,
    demand_params: Optional[DemandParameters] = None,
    p_bounds: Tuple[float, float] = (0.0, float('inf')),
    rng: Optional[np.random.Generator] = None,
    rng_seed: Optional[int] = None,
    *,
    params: None = None
) -> Experiment: ...


def run_kiefer_wolfowitz_global_learning(
    T: Optional[int] = None,
    n: Optional[int] = None,
    p_init: Optional[float] = None,
    eta: Optional[float] = None,
    delta: Optional[float] = None,
    revenue_fn: Optional[RevenueFunction] = None,
    allocation: Optional[AllocationFunction] = None,
    supplier_params: Optional[SupplierParameters] = None,
    d_a: Optional[float] = None,
    demand_params: Optional[DemandParameters] = None,
    p_bounds: Optional[Tuple[float, float]] = None,
    rng: Optional[np.random.Generator] = None,
    rng_seed: Optional[int] = None,
    *,
    params: Optional[ExperimentParams] = None
) -> Experiment:
    r"""
    Run global experimentation using the Kiefer-Wolfowitz algorithm.

    This implements the Kiefer-Wolfowitz stochastic approximation algorithm
    for gradient estimation via finite differences:
        $p_{t+1} = p_t + \eta_t \cdot (U(p_t + \delta) - U(p_t - \delta)) / (2\delta)$

    With step sizes $\eta_t = \eta/t^{2/3}$ and perturbation $\delta_t = \delta/t^{1/3}$,
    this achieves $O(T^{-1/3})$ convergence - worse than local experimentation's $O(T^{-1})$.

    Parameters
    ----------
    T : int
        Number of periods (uses 2 experiments per gradient estimate)
    n : int
        Number of suppliers
    p_init : float
        Initial payment
    eta : float
        Step size parameter
    delta : float
        Finite difference step size
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
        Payment bounds
    rng : Optional[np.random.Generator]
        Random number generator
    rng_seed : Optional[int]
        Random seed

    Returns
    -------
    Experiment
        Complete experiment with parameters and results

    Notes
    -----
    Can be called in two ways:
    1. With ExperimentParams: run_kiefer_wolfowitz_global_learning(params=exp_params)
    2. With individual parameters: run_kiefer_wolfowitz_global_learning(T=100, n=1000, ...)
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
        eta = params.eta
        delta = params.delta
        rng_seed = params.rng_seed

        # Extract demand parameters
        demand_config = extract_demand_from_params(params)
        d_a = demand_config.d_a
        demand_params = demand_config.demand_params
    else:
        # Validate that required parameters are provided
        if any(x is None for x in [T, n, p_init, eta, delta, revenue_fn, allocation, supplier_params]):
            raise ValueError(
                "When params is not provided, all of T, n, p_init, eta, delta, "
                "revenue_fn, allocation, and supplier_params must be provided"
            )
        if p_bounds is None:
            p_bounds = (0.0, float('inf'))

    # Setup RNG
    rng = setup_rng(rng, rng_seed)

    p_bounds = (float(p_bounds[0]), float(p_bounds[1]))

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
            eta=eta,
            experiment_type="global",
            zeta=None,
            alpha=None,
            delta=delta,
            rng_seed=rng_seed,
            store_detailed_data=False
        )

    from .find_equilibrium import find_equilibrium_supply_mu

    p = np.clip(p_init, p_bounds[0], p_bounds[1])
    timepoints: List[TimePointData] = []
    current_d_a = d_a

    for t in range(1, T + 1):
        # Decreasing step sizes for convergence
        eta_t = eta / (t ** (2/3))
        delta_t = delta / (t ** (1/3))

        # Sample state for this period
        current = sample_current_state(demand_params, d_a, rng)
        state = current.state
        current_d_a = current.d_a

        # Kiefer-Wolfowitz two-point finite difference gradient estimate
        # Evaluate utility at p+δ and p-δ to estimate the gradient
        p_plus = np.clip(p + delta_t, p_bounds[0], p_bounds[1])
        p_minus = np.clip(p - delta_t, p_bounds[0], p_bounds[1])

        # Run experiments at p+ and p-
        # We'll store the experiment at the current p for this timepoint
        timepoint = run_global_one_timepoint(
            n=n,
            p=p,
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

        # Compute utilities for gradient estimate
        U_plus = _compute_realized_utility(
            n, p_plus, current_d_a, revenue_fn, allocation, supplier_params,
            demand_params, state, rng
        )
        U_minus = _compute_realized_utility(
            n, p_minus, current_d_a, revenue_fn, allocation, supplier_params,
            demand_params, state, rng
        )

        # Finite difference gradient estimate
        grad_est = (U_plus - U_minus) / (2 * delta_t)

        # Gradient ascent step
        p = p + eta_t * grad_est
        p = np.clip(p, p_bounds[0], p_bounds[1])

    # Build results
    results = build_experiment_results(
        timepoints=timepoints,
        final_payment=p,
        weighted_average_payment=None  # Not used for global
    )

    return Experiment(params=params, results=results)


def _compute_realized_utility(
    n: int,
    p: float,
    d_a: float,
    revenue_fn: RevenueFunction,
    allocation: AllocationFunction,
    supplier_params: SupplierParameters,
    demand_params: Optional[DemandParameters],
    state: Optional[GlobalState],
    rng: np.random.Generator
) -> float:
    r"""Helper to compute realized utility for a single experiment."""
    eq = compute_equilibrium_allocation(p, d_a, supplier_params, allocation)
    q_eq = eq.q_eq
    
    # Sample activations
    payments = np.full(n, p)
    internal_seed = int(rng.integers(0, 2**31))
    Z = sample_supplier_activations(
        n=n, payments=payments, expected_allocation=q_eq,
        params=supplier_params, seed=internal_seed
    )
    T = int(Z.sum())
    
    # Sample demand
    if demand_params is not None and state is not None:
        D = sample_demand(demand_params, state, n, rng)
    else:
        D = int(round(n * d_a))
    
    # Compute utility
    return compute_realized_utility(D, T, p, revenue_fn, allocation, n)