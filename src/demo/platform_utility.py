"""
Platform Utility Computation

This module implements the platform's utility functions.

Key equations:
- Utility (3.8): U = R(D, T) - Σ P_i Z_i S_i
- Normalized utility (3.10): u_a(p) = (1/n) * E[U | A=a]

The platform wants to maximize expected utility by choosing the right payment p.
"""

from .allocation import (
    AllocationFunction,
    compute_omega,
    compute_omega_derivative,
    compute_expected_allocation,
)
from .supplier import SupplierParameters
from .market import MarketParameters
from .find_equilibrium import find_equilibrium_supply_mu
from .revenue import (
    RevenueFunction,
    compute_revenue_derivative,
)


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
    # Create MeanFieldMarketParameters for find_equilibrium_supply_mu
    # Extract gamma from revenue function if it's a linear revenue function
    # For now, we'll use a default value or extract from the revenue function
    gamma = 100.0  # Default, should ideally be passed or extracted from revenue_fn

    mean_field_params = MarketParameters(
        allocation=allocation,
        choice=supplier_params.choice,
        private_features=supplier_params.private_features,
        d_a=d_a,
        gamma=gamma,
        n_monte_carlo=supplier_params.n_monte_carlo
    )

    mu = find_equilibrium_supply_mu(p, mean_field_params)
    return compute_platform_utility(revenue_fn, allocation, d_a, mu, p)
