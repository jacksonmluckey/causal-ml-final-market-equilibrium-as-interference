"""
Revenue Functions for Stochastic Market

This module implements the platform's revenue function.

Key equations:
- Revenue (3.9): R(d, t) = r(d/t) * t

where r(x) is revenue per active supplier when demand ratio is x = d/t.
"""

import numpy as np
from typing import Callable, Optional
from dataclasses import dataclass

from .allocation import (
    AllocationFunction,
    compute_omega,
    compute_omega_derivative,
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
