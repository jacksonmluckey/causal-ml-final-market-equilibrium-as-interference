r"""
Core experiment setup and utilities.

This module provides shared utilities for all experimentation approaches
(local, Kiefer-Wolfowitz global, and baseline global):

- ExperimentParams: Configuration dataclass
- Helper functions for common operations (RNG setup, parameter extraction, etc.)
- Supporting dataclasses for type safety
"""

import numpy as np
from typing import Optional, Tuple, List, Union
from dataclasses import dataclass

from .demand import DemandParameters, GlobalState, sample_state, sample_demand
from .allocation import AllocationFunction
from .supplier import SupplierParameters, sample_supplier_activations
from .revenue import RevenueFunction
from .find_equilibrium import find_equilibrium_supply_mu


# Forward reference for type hints (experiment_results imports from us)
# We'll import at runtime to avoid circular dependency
TimePointData = None
ExperimentResults = None
Experiment = None


# =============================================================================
# EXPERIMENT PARAMETERS
# =============================================================================

@dataclass
class ExperimentParams:
    r"""
    Parameters for running an experiment.

    Attributes
    ----------
    T : int
        Number of time periods
    n : int
        Number of suppliers per period
    p_init : float
        Initial payment level
    revenue_fn : RevenueFunction
        Platform revenue function
    p_bounds : Tuple[float, float]
        Payment bounds [c_-, c_+]
    allocation : AllocationFunction
        The allocation function $\omega$
    supplier_params : SupplierParameters
        Supplier choice model parameters
    demand : Union[float, DemandParameters]
        Either fixed d_a (float) or full demand model (DemandParameters)
    eta : float
        Step size for optimization algorithm
    experiment_type : str
        Either "local" or "global"
    zeta : Optional[float]
        Base perturbation magnitude (local only)
    alpha : Optional[float]
        Zeta decay exponent for scaling $\zeta_n = \zeta \cdot n^{-\alpha}$ (local only)
    delta : Optional[float]
        Finite difference step size (global only)
    rng_seed : Optional[int]
        Random seed for reproducibility
    store_detailed_data : bool
        Whether to store individual-level data (Z_i, ε_i arrays)
    """
    T: int
    n: int
    p_init: float
    revenue_fn: RevenueFunction
    p_bounds: Tuple[float, float]
    allocation: AllocationFunction
    supplier_params: SupplierParameters
    demand: Union[float, DemandParameters]
    eta: float
    experiment_type: str
    zeta: Optional[float] = None
    alpha: Optional[float] = 0.3
    delta: Optional[float] = None
    rng_seed: Optional[int] = None
    store_detailed_data: bool = False

    @property
    def d_a(self) -> Optional[float]:
        r"""Get d_a if using fixed demand, else None"""
        return self.demand if isinstance(self.demand, float) else None

    @property
    def demand_params(self) -> Optional[DemandParameters]:
        r"""Get DemandParameters if using full model, else None"""
        return self.demand if isinstance(self.demand, DemandParameters) else None


# =============================================================================
# SUPPORTING DATACLASSES
# =============================================================================

@dataclass
class DemandConfig:
    """
    Extracted demand configuration from ExperimentParams.

    Attributes
    ----------
    d_a : Optional[float]
        Fixed expected demand per supplier (if using simple model)
    demand_params : Optional[DemandParameters]
        Full demand model with multiple states (if using stochastic model)
    """
    d_a: Optional[float]
    demand_params: Optional[DemandParameters]


@dataclass
class CurrentDemandState:
    """
    Current demand state for a timepoint.

    Attributes
    ----------
    state : Optional[GlobalState]
        Sampled global state (if using stochastic demand), None otherwise
    d_a : float
        Expected demand per supplier for this period
    """
    state: Optional[GlobalState]
    d_a: float


@dataclass
class EquilibriumAllocation:
    """
    Equilibrium supply and allocation quantities.

    Attributes
    ----------
    mu_eq : float
        Equilibrium supply fraction (fraction of suppliers who activate)
    q_eq : float
        Expected allocation per active supplier
    """
    mu_eq: float
    q_eq: float


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def setup_rng(
    rng: Optional[np.random.Generator] = None,
    rng_seed: Optional[int] = None
) -> np.random.Generator:
    """
    Initialize random number generator.

    If an existing RNG is provided, returns it as-is.
    Otherwise creates a new RNG, optionally with the given seed.

    Parameters
    ----------
    rng : Optional[np.random.Generator]
        Existing RNG to use (returned as-is if provided)
    rng_seed : Optional[int]
        Seed for creating new RNG (ignored if rng provided)

    Returns
    -------
    np.random.Generator
        Configured random number generator
    """
    if rng is None:
        if rng_seed is not None:
            rng = np.random.default_rng(rng_seed)
        else:
            rng = np.random.default_rng()
    return rng


def extract_demand_from_params(
    params: ExperimentParams
) -> DemandConfig:
    """
    Extract demand configuration from ExperimentParams.

    Handles the union type of params.demand which can be either
    a float (d_a) or DemandParameters.

    Parameters
    ----------
    params : ExperimentParams
        Experiment parameters

    Returns
    -------
    DemandConfig
        Extracted demand configuration with d_a and demand_params
    """
    if isinstance(params.demand, DemandParameters):
        return DemandConfig(d_a=None, demand_params=params.demand)
    else:
        return DemandConfig(d_a=params.demand, demand_params=None)


def sample_current_state(
    demand_params: Optional[DemandParameters],
    d_a: Optional[float],
    rng: np.random.Generator
) -> CurrentDemandState:
    """
    Sample or retrieve current demand state.

    If demand_params provided, samples a new stochastic state.
    Otherwise uses the fixed d_a value.

    Parameters
    ----------
    demand_params : Optional[DemandParameters]
        Demand model parameters (if using stochastic demand)
    d_a : Optional[float]
        Fixed expected demand per supplier (if not using demand_params)
    rng : np.random.Generator
        Random number generator

    Returns
    -------
    CurrentDemandState
        Sampled or fixed demand state

    Raises
    ------
    ValueError
        If neither demand_params nor d_a is provided
    """
    if demand_params is not None:
        state = sample_state(demand_params, rng)
        return CurrentDemandState(state=state, d_a=state.d_a)
    elif d_a is not None:
        return CurrentDemandState(state=None, d_a=d_a)
    else:
        raise ValueError("Must provide either demand_params or d_a")


def compute_equilibrium_allocation(
    p: float,
    d_a: float,
    supplier_params: SupplierParameters,
    allocation: AllocationFunction
) -> EquilibriumAllocation:
    """
    Compute equilibrium supply and allocation at payment p.

    Solves for equilibrium supply μ (fraction of suppliers who activate)
    and computes expected allocation q = ω(d_a/μ) per active supplier.

    Parameters
    ----------
    p : float
        Payment level
    d_a : float
        Expected demand per supplier
    supplier_params : SupplierParameters
        Supplier choice model parameters
    allocation : AllocationFunction
        Allocation function ω

    Returns
    -------
    EquilibriumAllocation
        Equilibrium quantities (mu_eq, q_eq)
    """
    mu_eq = find_equilibrium_supply_mu(
        p=p,
        d_a=d_a,
        choice=supplier_params.choice,
        private_features=supplier_params.private_features,
        allocation=allocation
    )
    q_eq = allocation(d_a / mu_eq) if mu_eq > 0 else 0.0

    return EquilibriumAllocation(mu_eq=mu_eq, q_eq=q_eq)


def compute_weighted_average_payment(
    timepoints: List
) -> float:
    """
    Compute weighted average payment per Corollary 8.

    Formula: p̄_T = (2/T(T+1)) Σ_{t=1}^T t·p_t

    This gives more weight to later payments as the algorithm converges,
    providing a better estimate of the optimal payment than a simple average.

    Parameters
    ----------
    timepoints : List[TimePointData]
        Data from all timepoints (requires .p attribute)

    Returns
    -------
    float
        Weighted average payment
    """
    T = len(timepoints)
    if T == 0:
        return 0.0

    payments = np.array([tp.p for tp in timepoints])
    weights = np.arange(1, T + 1)
    weight_sum = T * (T + 1) / 2

    weighted_avg = float(np.sum(weights * payments) / weight_sum)
    return weighted_avg


def build_experiment_results(
    timepoints: List,
    final_payment: float,
    weighted_average_payment: Optional[float] = None
):
    """
    Build ExperimentResults from timepoint data.

    Computes aggregate metrics from individual timepoints:
    - Total and mean utility
    - Average payment
    - Aggregates timepoint data

    Parameters
    ----------
    timepoints : List[TimePointData]
        Data from all experimental timepoints
    final_payment : float
        Final payment level
    weighted_average_payment : Optional[float]
        Weighted average payment (local experiments only, per Corollary 8)

    Returns
    -------
    ExperimentResults
        Aggregated experiment results
    """
    # Import here to avoid circular dependency
    from .experiment_results import ExperimentResults

    T = len(timepoints)
    payments = [tp.p for tp in timepoints]
    total_utility = sum(tp.U for tp in timepoints)
    mean_utility = total_utility / T if T > 0 else 0.0

    results = ExperimentResults(
        final_payment=final_payment,
        weighted_average_payment=weighted_average_payment,
        average_payment=float(np.mean(payments)) if payments else 0.0,
        timepoints=timepoints,
        total_utility=total_utility,
        mean_utility=mean_utility
    )

    return results


def run_global_one_timepoint(
    n: int,
    p: float,
    revenue_fn: RevenueFunction,
    allocation: AllocationFunction,
    supplier_params: SupplierParameters,
    t: int = 0,
    d_a: Optional[float] = None,
    demand_params: Optional[DemandParameters] = None,
    state: Optional[GlobalState] = None,
    rng: Optional[np.random.Generator] = None
):
    r"""
    Run one period of global experimentation.

    All suppliers receive the SAME payment p (no perturbations).
    This is used by both Kiefer-Wolfowitz and baseline global experimentation.

    Parameters
    ----------
    n : int
        Number of suppliers
    p : float
        Payment level for all suppliers
    revenue_fn : RevenueFunction
        Platform revenue function
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
    # Import here to avoid circular dependency
    from .experiment_results import TimePointData

    if rng is None:
        rng = np.random.default_rng()

    # Compute equilibrium allocation
    current_d_a = state.d_a if state else d_a
    eq = compute_equilibrium_allocation(p, current_d_a, supplier_params, allocation)
    mu_eq = eq.mu_eq
    q_eq = eq.q_eq

    # All suppliers get same payment
    payments = np.full(n, p)

    # Sample activations
    internal_seed = int(rng.integers(0, 2**31))
    Z = sample_supplier_activations(
        n=n, payments=payments, expected_allocation=q_eq,
        params=supplier_params, seed=internal_seed
    )
    T_active = int(Z.sum())

    # Sample demand
    if demand_params is not None and state is not None:
        D = sample_demand(demand_params, state, n, rng)
    else:
        D = int(round(n * current_d_a))

    # Compute realized utility
    if T_active > 0:
        x = D / T_active  # Demand per active supplier
        actual_q = allocation(x)
        S = T_active * actual_q
        total_revenue = revenue_fn.r(x) * T_active
        U = total_revenue - p * S
    else:
        S = 0.0
        U = 0.0

    return TimePointData(
        t=t,
        p=p,
        D=D,
        T=T_active,
        S=S,
        U=U,
        state=state,
        gradient_estimate=None,
        delta_hat=None,
        upsilon_hat=None,
        zeta=None,
        epsilon=None,
        Z=None
    )
