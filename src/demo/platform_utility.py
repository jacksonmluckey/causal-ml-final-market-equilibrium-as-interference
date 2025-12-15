"""
Platform Utility Computation

This module implements the platform's utility functions.

Key equations:
- Utility (3.8): $U = R(D, T) - \sum P_i Z_i S_i$
- Normalized utility (3.10): $u_a(p) = (1/n) \cdot E[U | A=a]$

The platform wants to maximize expected utility by choosing the right payment p.
"""

from .allocation import AllocationFunction
from .revenue import RevenueFunction


def compute_platform_utility(
    revenue_fn: RevenueFunction,
    allocation: AllocationFunction,
    d_a: float,
    mu: float,
    p: float
) -> float:
    """
    Compute mean-field utility given activation rate $\mu$.

    $u_a(p) = (r(d_a/\mu) - p \cdot \omega(d_a/\mu)) \cdot \mu$

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
    payment_per_supplier = p * allocation(x)

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

    $du/dp = \mu'(p) \cdot [r(d_a/\mu) - p \cdot \omega(d_a/\mu) - (r'(d_a/\mu) - p \cdot \omega'(d_a/\mu)) \cdot d_a/\mu] - \omega(d_a/\mu) \cdot \mu$

    Args:
        revenue_fn: Platform revenue function
        allocation: Allocation function
        d_a: Expected demand per supplier
        mu: Activation rate $\mu_a(p)$
        mu_prime: Derivative $d\mu/dp$
        p: Payment

    Returns:
        $du_a/dp$
    """
    if mu <= 0:
        return 0.0

    x = d_a / mu

    r_val = revenue_fn.r(x)
    r_deriv = revenue_fn.r_prime(x)
    omega_val = allocation(x)
    omega_deriv = allocation.derivative(x)

    # Term from change in μ
    bracket_term = (r_val - p * omega_val -
                   (r_deriv - p * omega_deriv) * x)
    mu_contribution = mu_prime * bracket_term

    # Direct term from change in p
    direct_term = -omega_val * mu

    return mu_contribution + direct_term
