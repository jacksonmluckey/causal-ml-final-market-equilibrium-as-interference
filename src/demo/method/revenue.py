r"""
Revenue Functions for Stochastic Market

This module implements the platform's revenue function.

Key equations:
- Revenue (3.9): $R(d, t) = r(d/t) \cdot t$

where $r(x)$ is revenue per active supplier when demand ratio is $x = d/t$.
"""

from typing import Callable, Optional
from dataclasses import dataclass

from .allocation import (
    AllocationFunction,
    # compute_omega,
    # compute_omega_derivative,
)


@dataclass
class RevenueFunction:
    r"""
    Platform revenue function.

    $R(d, t) = r(d/t) \cdot t$

    where $r(x)$ is revenue per active supplier when demand ratio is x.

    Parameters
    ----------
    r : Callable[[float], float]
        Revenue per supplier when demand ratio is $x = d/t$
    r_prime : Optional[Callable[[float], float]]
        Derivative $dr/dx$. If None, computed numerically when needed.
    name : str
        Descriptive name
    """

    r: Callable[[float], float]
    r_prime: Optional[Callable[[float], float]] = None
    name: str = "Generic"


def create_linear_revenue(
    gamma: float, allocation: AllocationFunction
) -> RevenueFunction:
    r"""
    Create linear revenue function.

    Linear revenue: platform gets $\gamma$ per unit of demand served.

    $R(D, T) = \gamma \cdot T \cdot \Omega(D, T) = \gamma \cdot T \cdot \omega(D/T)$

    So $r(x) = \gamma \cdot \omega(x)$

    Parameters
    ----------
    gamma : float
        Payment per unit of demand served
    allocation : AllocationFunction
        Allocation function $\omega(x)$

    Returns
    -------
    RevenueFunction
        The linear revenue function
    """

    def r(x: float) -> float:
        r"""$r(x) = \gamma \cdot \omega(x)$"""
        return gamma * allocation(x)

    def r_prime(x: float) -> float:
        r"""$dr/dx = \gamma \cdot \omega'(x)$"""
        return gamma * allocation.derivative(x)

    return RevenueFunction(r=r, r_prime=r_prime, name=f"Linear ($\gamma$={gamma})")
