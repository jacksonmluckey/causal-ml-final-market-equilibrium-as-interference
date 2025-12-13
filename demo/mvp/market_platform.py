"""
Platform Utility Model for Stochastic Market

This module implements the platform's objective function.

Key equations:
- Utility (3.8): U = R(D, T) - Σ P_i Z_i S_i
- Revenue (3.9): R(d, t) = r(d/t) * t
- Normalized utility (3.10): u_a(p) = (1/n) * E[U | A=a]

The platform wants to maximize expected utility by choosing the right payment p.
"""

import numpy as np
from typing import Callable, Optional
from dataclasses import dataclass

# Import our other modules
from .allocation import (
    AllocationFunction,
    compute_omega,
    compute_omega_derivative,
    compute_finite_allocation,
    compute_expected_allocation,
)
from .supplier import (
    SupplierParameters,
    compute_activation_probability,
    sample_supplier_activations,
)


@dataclass
class RevenueFunction:
    """
    Platform revenue function.

    R(d, t) = r(d/t) * t

    where r(x) is revenue per active supplier when demand ratio is x.

    Parameters
    ----------
    r : Callable[[float], float]
        Revenue per supplier when demand ratio is x = d/t
    r_prime : Optional[Callable[[float], float]]
        Derivative dr/dx. If None, computed numerically when needed.
    name : str
        Descriptive name
    """
    r: Callable[[float], float]
    r_prime: Optional[Callable[[float], float]] = None
    name: str = "Generic"


def create_linear_revenue(gamma: float, allocation: AllocationFunction) -> RevenueFunction:
    """
    Create linear revenue function.

    Linear revenue: platform gets γ per unit of demand served.

    R(D, T) = γ * T * Ω(D, T) = γ * T * ω(D/T)

    So r(x) = γ * ω(x)

    Parameters
    ----------
    gamma : float
        Payment per unit of demand served
    allocation : AllocationFunction
        Allocation function ω(x)

    Returns
    -------
    RevenueFunction
        The linear revenue function
    """
    def r(x: float) -> float:
        """r(x) = γ * ω(x)"""
        return gamma * compute_omega(allocation, x)

    def r_prime(x: float) -> float:
        """dr/dx = γ * ω'(x)"""
        return gamma * compute_omega_derivative(allocation, x)

    return RevenueFunction(
        r=r,
        r_prime=r_prime,
        name=f"Linear (γ={gamma})"
    )


def compute_total_revenue(
    revenue_fn: RevenueFunction,
    d: float,
    t: float
) -> float:
    """
    Compute total revenue R(d, t) = r(d/t) * t.

    Args:
        revenue_fn: Revenue function
        d: Total demand
        t: Number of active suppliers

    Returns:
        Total revenue
    """
    if t <= 0:
        return 0.0
    return revenue_fn.r(d / t) * t


def compute_revenue_per_supplier(
    revenue_fn: RevenueFunction,
    x: float
) -> float:
    """
    Compute revenue per supplier r(x) when demand ratio is x = d/t.

    Args:
        revenue_fn: Revenue function
        x: Demand per active supplier ratio

    Returns:
        Revenue per supplier
    """
    return revenue_fn.r(x)


def compute_revenue_derivative(
    revenue_fn: RevenueFunction,
    x: float
) -> float:
    """
    Compute derivative dr/dx.

    Args:
        revenue_fn: Revenue function
        x: Demand per active supplier ratio

    Returns:
        dr/dx
    """
    if revenue_fn.r_prime is not None:
        return revenue_fn.r_prime(x)
    else:
        from .utils import numerical_derivative
        return numerical_derivative(revenue_fn.r, x)


# =============================================================================
# PLATFORM UTILITY COMPUTATION
# =============================================================================

def compute_platform_utility(
    revenue_fn: RevenueFunction,
    allocation: AllocationFunction,
    d_a: float,
    mu: float,
    p: float
) -> float:
    """
    Compute mean-field utility given activation rate μ.

    u_a(p) = (r(d_a/μ) - p*ω(d_a/μ)) * μ

    Args:
        revenue_fn: Platform revenue function
        allocation: Allocation function
        d_a: Expected demand per supplier
        mu: Fraction of suppliers who are active
        p: Payment per unit of demand served

    Returns:
        Normalized utility (utility per supplier)
    """
    if mu <= 0:
        return 0.0

    x = d_a / mu  # Demand per active supplier

    revenue_per_supplier = revenue_fn.r(x)
    payment_per_supplier = p * compute_omega(allocation, x)

    profit_per_active_supplier = revenue_per_supplier - payment_per_supplier

    return profit_per_active_supplier * mu


def compute_platform_utility_derivative(
    revenue_fn: RevenueFunction,
    allocation: AllocationFunction,
    d_a: float,
    mu: float,
    mu_prime: float,
    p: float
) -> float:
    """
    Compute derivative of utility with respect to payment p.

    du/dp = μ'(p) * [r(d_a/μ) - p*ω(d_a/μ) - (r'(d_a/μ) - p*ω'(d_a/μ)) * d_a/μ]
            - ω(d_a/μ) * μ

    Args:
        revenue_fn: Platform revenue function
        allocation: Allocation function
        d_a: Expected demand per supplier
        mu: Activation rate μ_a(p)
        mu_prime: Derivative dμ/dp
        p: Payment

    Returns:
        du_a/dp
    """
    if mu <= 0:
        return 0.0

    x = d_a / mu

    r_val = revenue_fn.r(x)
    r_deriv = compute_revenue_derivative(revenue_fn, x)
    omega_val = compute_omega(allocation, x)
    omega_deriv = compute_omega_derivative(allocation, x)

    # Term from change in μ
    bracket_term = (r_val - p * omega_val -
                   (r_deriv - p * omega_deriv) * x)
    mu_contribution = mu_prime * bracket_term

    # Direct term from change in p
    direct_term = -omega_val * mu

    return mu_contribution + direct_term


# =============================================================================
# EQUILIBRIUM COMPUTATION
# =============================================================================

def find_equilibrium_mu(
    allocation: AllocationFunction,
    supplier_params: SupplierParameters,
    d_a: float,
    p: float,
    tol: float = 1e-6,
    max_iter: int = 100
) -> float:
    """
    Find equilibrium activation rate μ_a(p).

    Solves the fixed-point equation:
    μ = E[f_B(p * ω(d_a/μ))]

    Args:
        allocation: Allocation function
        supplier_params: Supplier population parameters
        d_a: Expected demand per supplier
        p: Payment level
        tol: Convergence tolerance
        max_iter: Maximum iterations

    Returns:
        Equilibrium activation rate μ
    """
    # Start with initial guess
    mu = 0.5

    for i in range(max_iter):
        # Compute expected allocation given current μ
        q = compute_expected_allocation(allocation, mu, d_a)

        # Compute expected revenue
        expected_revenue = p * q

        # Compute new activation rate
        mu_new = compute_activation_probability(expected_revenue, supplier_params)

        # Check convergence
        if abs(mu_new - mu) < tol:
            return mu_new

        # Update with some damping for stability
        mu = 0.7 * mu_new + 0.3 * mu

    return mu


def compute_mean_field_utility(
    revenue_fn: RevenueFunction,
    allocation: AllocationFunction,
    supplier_params: SupplierParameters,
    d_a: float,
    p: float
) -> float:
    """
    Compute mean-field utility at payment level p.

    u_a(p) = (r(d_a/μ) - p*ω(d_a/μ)) * μ

    where μ = μ_a(p) is the equilibrium activation rate.

    Args:
        revenue_fn: Revenue function
        allocation: Allocation function
        supplier_params: Supplier population parameters
        d_a: Expected demand per supplier
        p: Payment level

    Returns:
        Mean-field utility
    """
    mu = find_equilibrium_mu(allocation, supplier_params, d_a, p)
    return compute_platform_utility(revenue_fn, allocation, d_a, mu, p)


# =============================================================================
# MARKET SIMULATION
# =============================================================================

@dataclass
class MarketOutcome:
    """Results from simulating one period of the market."""
    demand: int
    num_active: int
    total_served: float
    revenue: float
    payments: float
    utility: float
    payment_level: float


@dataclass
class MarketParameters:
    """
    Parameters defining the complete market model.

    Parameters
    ----------
    allocation : AllocationFunction
        Allocation function
    supplier_params : SupplierParameters
        Supplier population parameters
    gamma : float
        Platform revenue per unit of demand served
    """
    allocation: AllocationFunction
    supplier_params: SupplierParameters
    gamma: float


def simulate_market_period(
    params: MarketParameters,
    n: int,
    d_a: float,
    p: float,
    zeta: float = 0.0,
    rng: Optional[np.random.Generator] = None
) -> MarketOutcome:
    """
    Simulate one period of the market.

    Args:
        params: Market parameters
        n: Number of potential suppliers
        d_a: Expected demand per supplier (or actual demand/n)
        p: Base payment level
        zeta: Payment perturbation magnitude (for local experimentation)
        rng: Random number generator

    Returns:
        MarketOutcome with simulation results
    """
    if rng is None:
        rng = np.random.default_rng()

    # Generate demand
    D = int(round(n * d_a))  # For simplicity, use deterministic demand

    # Generate payments (with possible perturbation)
    if zeta > 0:
        epsilon = rng.choice([-1, 1], size=n)
        payments = p + zeta * epsilon
    else:
        payments = np.full(n, p)
        epsilon = None

    # First, compute equilibrium expected allocation
    # This requires solving a fixed-point equation
    mu_eq = find_equilibrium_mu(
        params.allocation,
        params.supplier_params,
        d_a,
        p
    )
    q_eq = compute_expected_allocation(params.allocation, mu_eq, d_a)

    # Suppliers make activation decisions
    Z = sample_supplier_activations(
        n,
        payments,
        q_eq,
        params.supplier_params
    )
    T = Z.sum()  # Number of active suppliers

    # Compute actual allocations
    if T > 0:
        # In practice, allocations would be random
        # For simplicity, use expected allocation
        actual_q = compute_finite_allocation(params.allocation, D, T)
        S = np.where(Z == 1, actual_q, 0.0)
    else:
        S = np.zeros(n)

    # Compute outcomes
    total_served = S.sum() if T > 0 else 0.0
    revenue = params.gamma * total_served
    total_payments = (payments * Z * S).sum()
    utility = revenue - total_payments

    return MarketOutcome(
        demand=D,
        num_active=T,
        total_served=total_served,
        revenue=revenue,
        payments=total_payments,
        utility=utility,
        payment_level=p
    )
