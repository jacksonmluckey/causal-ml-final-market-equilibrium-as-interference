"""
Global Experimentation (Bandit Baseline)

Section 4.4 of Wager & Xu (2021): Comparison with global experimentation.

In global experimentation, each period t we choose a SINGLE payment p_t
for all suppliers and observe aggregate utility U_t. This is a standard
continuous-armed bandit problem.

Key result (Shamir 2013): Even with strongly concave utility, no algorithm
can achieve expected regret growing slower than √T. This gives 1/√T rate
of decay in errors, compared to 1/T for local experimentation.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Tuple

from .allocation import AllocationFunction
from .supplier import SupplierParameters, sample_supplier_activations
from .demand import DemandParameters, GlobalState, sample_state, sample_demand


@dataclass
class GlobalExperimentData:
    """Data from one period of global experimentation."""
    n: int
    p: float
    D: int          # Realized demand
    T: int          # Active suppliers
    S: float        # Total demand served
    U: float        # Realized utility (revenue - payments)


@dataclass 
class GlobalLearningResult:
    """Results from global experimentation algorithm."""
    final_payment: float
    average_payment: float
    payment_history: List[float]
    utility_history: List[float]
    state_history: Optional[List[GlobalState]] = None


def run_global_experiment(
    n: int,
    p: float,
    gamma: float,
    allocation: AllocationFunction,
    supplier_params: SupplierParameters,
    d_a: Optional[float] = None,
    demand_params: Optional[DemandParameters] = None,
    state: Optional[GlobalState] = None,
    rng: Optional[np.random.Generator] = None
) -> GlobalExperimentData:
    """
    Run one period of global experimentation.
    
    All suppliers receive the SAME payment p (no perturbations).
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
    
    return GlobalExperimentData(n=n, p=p, D=D, T=T, S=S, U=U)


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
) -> GlobalLearningResult:
    """
    Run global experimentation using finite-difference gradient estimates.
    
    Uses Kiefer-Wolfowitz style stochastic approximation:
        p_{t+1} = p_t + η_t * (U(p_t + δ) - U(p_t - δ)) / (2δ)
    
    With step sizes η_t = η/t^(2/3) and perturbation δ_t = δ/t^(1/3),
    this achieves O(T^(-1/3)) convergence - worse than local's O(T^(-1)).
    
    Parameters
    ----------
    T : int
        Number of periods (uses 2 experiments per gradient estimate)
    delta : float
        Finite difference step size
    """
    if rng is None:
        rng = np.random.default_rng()
    
    from .find_equilibrium import find_equilibrium_supply_mu
    
    p = np.clip(p_init, p_bounds[0], p_bounds[1])
    payment_history = [p]
    utility_history = []
    state_history = [] if demand_params else None
    current_d_a = d_a
    
    for t in range(1, T + 1):
        # Decreasing step sizes for convergence
        eta_t = eta / (t ** (2/3))
        delta_t = delta / (t ** (1/3))
        
        # Sample state for this period
        if demand_params is not None:
            state = sample_state(demand_params, rng)
            current_d_a = state.d_a
            if state_history is not None:
                state_history.append(state)
        else:
            state = None
        
        # Two-point gradient estimate (Kiefer-Wolfowitz)
        p_plus = np.clip(p + delta_t, p_bounds[0], p_bounds[1])
        p_minus = np.clip(p - delta_t, p_bounds[0], p_bounds[1])
        
        # Compute utilities directly (avoid repeated equilibrium solves)
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
        
        payment_history.append(p)
        utility_history.append((U_plus + U_minus) / 2)
    
    return GlobalLearningResult(
        final_payment=p,
        average_payment=np.mean(payment_history[1:]),
        payment_history=payment_history,
        utility_history=utility_history,
        state_history=state_history
    )


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