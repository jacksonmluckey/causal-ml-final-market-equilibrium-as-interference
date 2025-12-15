"""
Global Experimentation (Bandit Baseline)

Section 4.4 of Wager & Xu (2021): Comparison with global experimentation.

In global experimentation, each period t we choose a SINGLE payment $p_t$
for all suppliers and observe aggregate utility $U_t$. This is a standard
continuous-armed bandit problem.

Key result (Shamir 2013): Even with strongly concave utility, no algorithm
can achieve expected regret growing slower than $\sqrt{T}$. This gives $1/\sqrt{T}$ rate
of decay in errors, compared to $1/T$ for local experimentation.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Tuple, overload

from .allocation import AllocationFunction
from .supplier import SupplierParameters, sample_supplier_activations
from .demand import DemandParameters, GlobalState, sample_state, sample_demand
from .experiment_results import (
    TimePointData,
    ExperimentParams,
    ExperimentResults,
    Experiment
)


def run_global_experiment(
    n: int,
    p: float,
    gamma: float,
    allocation: AllocationFunction,
    supplier_params: SupplierParameters,
    t: int = 0,
    d_a: Optional[float] = None,
    demand_params: Optional[DemandParameters] = None,
    state: Optional[GlobalState] = None,
    rng: Optional[np.random.Generator] = None
) -> TimePointData:
    """
    Run one period of global experimentation.

    All suppliers receive the SAME payment p (no perturbations).

    Parameters
    ----------
    n : int
        Number of suppliers
    p : float
        Payment level for all suppliers
    gamma : float
        Platform revenue per unit
    allocation : AllocationFunction
        The allocation function ω
    supplier_params : SupplierParameters
        Supplier behavior parameters
    t : int
        Time period (1-indexed)
    d_a : Optional[float]
        Expected demand per supplier (if not using demand_params)
    demand_params : Optional[DemandParameters]
        Demand model parameters
    state : Optional[GlobalState]
        Current global state
    rng : Optional[np.random.Generator]
        Random number generator

    Returns
    -------
    TimePointData
        Data from this experimental period (gradient fields set to None)
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # Compute equilibrium allocation
    from .find_equilibrium import find_equilibrium_supply_mu
    current_d_a = state.d_a if state else d_a
    
    mu_eq = find_equilibrium_supply_mu(
        p=p,
        d_a=current_d_a,
        choice=supplier_params.choice,
        private_features=supplier_params.private_features,
        allocation=allocation
    )
    q_eq = allocation(current_d_a / mu_eq) if mu_eq > 0 else 0.0
    
    # All suppliers get same payment
    payments = np.full(n, p)
    
    # Sample activations
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
        D = int(round(n * current_d_a))
    
    # Compute realized utility
    if T > 0:
        actual_q = allocation(D / T)
        S = T * actual_q
    else:
        S = 0.0

    U = gamma * S - p * S  # Revenue minus payments

    return TimePointData(
        t=t,
        p=p,
        D=D,
        T=T,
        S=S,
        U=U,
        state=state,
        gradient_estimate=None,  # Not used in global
        delta_hat=None,
        upsilon_hat=None,
        zeta=None,
        epsilon=None,
        Z=None
    )


@overload
def run_global_learning(
    *,
    params: ExperimentParams,
    rng: Optional[np.random.Generator] = None
) -> Experiment: ...


@overload
def run_global_learning(
    T: int,
    n: int,
    p_init: float,
    eta: float,
    delta: float,
    gamma: float,
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


def run_global_learning(
    T: Optional[int] = None,
    n: Optional[int] = None,
    p_init: Optional[float] = None,
    eta: Optional[float] = None,
    delta: Optional[float] = None,
    gamma: Optional[float] = None,
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
    """
    Run global experimentation using finite-difference gradient estimates.

    Uses Kiefer-Wolfowitz style stochastic approximation:
        $p_{t+1} = p_t + \eta_t \cdot (U(p_t + \delta) - U(p_t - \delta)) / (2\delta)$

    With step sizes $\eta_t = \eta/t^{2/3}$ and perturbation $\delta_t = \delta/t^{1/3}$,
    this achieves $O(T^{-1/3})$ convergence - worse than local's $O(T^{-1})$.

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
    gamma : float
        Platform revenue per unit
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
    1. With ExperimentParams: run_global_learning(params=exp_params)
    2. With individual parameters: run_global_learning(T=100, n=1000, ...)
    """
    # Extract parameters from ExperimentParams if provided
    if params is not None:
        T = params.T
        n = params.n
        p_init = params.p_init
        gamma = params.gamma
        allocation = params.allocation
        supplier_params = params.supplier_params
        p_bounds = params.p_bounds
        eta = params.eta
        delta = params.delta
        rng_seed = params.rng_seed

        # Extract demand parameters
        if isinstance(params.demand, DemandParameters):
            demand_params = params.demand
            d_a = None
        else:
            d_a = params.demand
            demand_params = None
    else:
        # Validate that required parameters are provided
        if any(x is None for x in [T, n, p_init, eta, delta, gamma, allocation, supplier_params]):
            raise ValueError(
                "When params is not provided, all of T, n, p_init, eta, delta, "
                "gamma, allocation, and supplier_params must be provided"
            )
        if p_bounds is None:
            p_bounds = (0.0, float('inf'))

    # Setup RNG
    if rng is None:
        if rng_seed is not None:
            rng = np.random.default_rng(rng_seed)
        else:
            rng = np.random.default_rng()

    p_bounds = (float(p_bounds[0]), float(p_bounds[1]))

    # Create experiment parameters (only if not already provided)
    if params is None:
        params = ExperimentParams(
            T=T,
            n=n,
            p_init=p_init,
            gamma=gamma,
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
        if demand_params is not None:
            state = sample_state(demand_params, rng)
            current_d_a = state.d_a
        else:
            state = None

        # Two-point gradient estimate (Kiefer-Wolfowitz)
        p_plus = np.clip(p + delta_t, p_bounds[0], p_bounds[1])
        p_minus = np.clip(p - delta_t, p_bounds[0], p_bounds[1])

        # Run experiments at p+ and p-
        # We'll store the experiment at the current p for this timepoint
        mu_eq = find_equilibrium_supply_mu(
            p=p,
            d_a=current_d_a,
            choice=supplier_params.choice,
            private_features=supplier_params.private_features,
            allocation=allocation
        )
        q_eq = allocation(current_d_a / mu_eq) if mu_eq > 0 else 0.0

        timepoint = run_global_experiment(
            n=n,
            p=p,
            gamma=gamma,
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
            n, p_plus, current_d_a, gamma, allocation, supplier_params,
            demand_params, state, rng
        )
        U_minus = _compute_realized_utility(
            n, p_minus, current_d_a, gamma, allocation, supplier_params,
            demand_params, state, rng
        )

        # Finite difference gradient estimate
        grad_est = (U_plus - U_minus) / (2 * delta_t)

        # Gradient ascent step
        p = p + eta_t * grad_est
        p = np.clip(p, p_bounds[0], p_bounds[1])

    # Build results
    payments = [tp.p for tp in timepoints]
    total_utility = sum(tp.U for tp in timepoints)
    mean_utility = total_utility / T if T > 0 else 0.0

    results = ExperimentResults(
        final_payment=p,
        weighted_average_payment=None,  # Not used for global
        average_payment=float(np.mean(payments)),
        timepoints=timepoints,
        total_utility=total_utility,
        mean_utility=mean_utility
    )

    return Experiment(params=params, results=results)


def _compute_realized_utility(
    n: int,
    p: float,
    d_a: float,
    gamma: float,
    allocation: AllocationFunction,
    supplier_params: SupplierParameters,
    demand_params: Optional[DemandParameters],
    state: Optional[GlobalState],
    rng: np.random.Generator
) -> float:
    """Helper to compute realized utility for a single experiment."""
    from .find_equilibrium import find_equilibrium_supply_mu
    
    mu_eq = find_equilibrium_supply_mu(
        p=p, d_a=d_a,
        choice=supplier_params.choice,
        private_features=supplier_params.private_features,
        allocation=allocation
    )
    q_eq = allocation(d_a / mu_eq) if mu_eq > 0 else 0.0
    
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
    if T > 0:
        S = T * allocation(D / T)
    else:
        S = 0.0
    
    return gamma * S - p * S