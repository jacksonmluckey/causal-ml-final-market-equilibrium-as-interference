r"""
Allocation Function for Stochastic Market

This module implements the regular allocation function $\omega(x)$ and the
finite-market allocation $\Omega(d, t)$.

A regular allocation function (Definition 5) must satisfy:
1. Smooth, concave, non-decreasing
2. $\lim_{x \to 0} \omega(x) = 0$ (no demand $\to$ no allocation)
3. $\lim_{x \to \infty} \omega(x) \leq 1$ (suppliers have bounded capacity)
4. $\lim_{x \to 0} \omega'(x) \leq 1$ (allocation rate bounded by demand)

Key relationship (Assumption 1):
$\Omega(d, t) = \omega(d/t) + l(d, t)$

where $l(d, t)$ is an error term that vanishes as $d, t \to \infty$
"""

import numpy as np
from typing import Callable, Optional
from dataclasses import dataclass

from .utils import numerical_derivative


@dataclass
class AllocationFunction:
    r"""
    Regular allocation function $\omega(x)$ as defined in Definition 5.

    The allocation function captures how much demand each active
    supplier serves when the demand-to-supply ratio is $x = d/t$.

    Parameters
    ----------
    omega : Callable[[float], float]
        The allocation function $\omega(x)$ mapping demand-to-supply ratio to
        expected demand served per active supplier
    omega_prime : Optional[Callable[[float], float]]
        The derivative $\omega'(x)$. If None, must be computed numerically when needed.
    name : str
        Descriptive name for the allocation function
    """
    omega: Callable[[float], float]
    omega_prime: Optional[Callable[[float], float]] = None
    name: str = "Generic"

    def __call__(self, x: float) -> float:
        r"""Evaluate $\omega(x)$."""
        return self.omega(x)

    def derivative(self, x: float) -> float:
        r"""Evaluate $\omega'(x)$."""
        if self.omega_prime is not None:
            return self.omega_prime(x)
        return numerical_derivative(self.omega, x)


def create_queue_allocation(L: int = 8) -> AllocationFunction:
    r"""
    Create allocation function from Example 6: Parallel Finite-Capacity Queues.

    Each active supplier operates as an M/M/1 queue with capacity L.
    The allocation function is:

    $\omega(x) = (x - x^L) / (1 - x^L)$ for $x \neq 1$
    $\omega(1) = 1 - 1/L$

    Properties:
    - As $L \to \infty$, $\omega(x) \to \min(x, 1)$ (suppliers can serve all demand up to capacity)
    - For finite L, some demand is dropped when queues are full

    Parameters
    ----------
    L : int
        Queue capacity (must be $\geq 2$)

    Returns
    -------
    AllocationFunction
        The queue allocation function
    """
    if L < 2:
        raise ValueError("Queue capacity L must be at least 2")

    def omega(x: float) -> float:
        r"""
        Allocation function for M/M/1 queue with capacity L.

        $\omega(x) = (x - x^L) / (1 - x^L)$
        """
        if x <= 0:
            return 0.0
        if x > 10:  # For very large x, $\omega(x) \to 1$
            return 1.0
        # Handle $x \approx 1$ separately to avoid numerical issues
        if abs(x - 1) < 1e-10:
            return 1 - 1/L
        x_L = x ** L
        return (x - x_L) / (1 - x_L)

    def omega_prime(x: float) -> float:
        r"""
        Derivative of $\omega(x)$.

        Using quotient rule on $\omega(x) = (x - x^L) / (1 - x^L)$
        """
        if x <= 0:
            return 1.0  # $\lim_{x \to 0} \omega'(x) = 1$
        if x > 10:
            return 0.0  # For large x, $\omega(x) \approx 1$, so derivative $\approx 0$
        if abs(x - 1) < 1e-10:
            # Use L'Hopital's rule or series expansion near $x=1$
            # $\omega'(1) = (L-1)/(2L)$
            return (L - 1) / (2 * L)
        x_L = x ** L
        x_Lm1 = x ** (L - 1)
        numerator = (1 - L * x_Lm1) * (1 - x_L) - (x - x_L) * (-L * x_Lm1)
        denominator = (1 - x_L) ** 2
        return numerator / denominator

    return AllocationFunction(
        omega=omega,
        omega_prime=omega_prime,
        name=f"M/M/1 Queue (L={L})"
    )


def create_linear_allocation() -> AllocationFunction:
    r"""
    Create simple linear allocation: $\omega(x) = \min(x, 1)$ (limiting case as $L \to \infty$).

    Each supplier serves all their demand up to capacity 1.

    Returns
    -------
    AllocationFunction
        The linear allocation function
    """
    def omega(x: float) -> float:
        return min(max(x, 0), 1.0)

    def omega_prime(x: float) -> float:
        if 0 < x < 1:
            return 1.0 
        else:
            return 0.0

    return AllocationFunction(
        omega=omega,
        omega_prime=omega_prime,
        name="Linear"
    )


def create_smooth_linear_allocation() -> AllocationFunction:
    r"""
    Create smooth approximation to linear allocation.

    $\omega(x) = 1 - \exp(-x)$

    This satisfies all regularity conditions and approximates $\min(x, 1)$.

    Returns
    -------
    AllocationFunction
        The smooth linear allocation function
    """
    def omega(x: float) -> float:
        if x <= 0:
            return 0.0
        else:
            return 1 - np.exp(-x)

    def omega_prime(x: float) -> float:
        if x <= 0:
            return 1.0
        else:
            return np.exp(-x)

    return AllocationFunction(
        omega=omega,
        omega_prime=omega_prime,
        name="Smooth Linear"
    )


def create_simple_allocation():
    r"""
    Create a simple concave allocation function for testing.

    Uses $\omega(x) = x / (1 + x)$, which satisfies all properties in Definition 5:
        - Smooth, concave, non-decreasing
        - $\omega(0) = 0$, $\omega(\infty) = 1$
        - $\omega'(0) = 1$
    """

    def omega(x: float) -> float:
        return x / (1.0 + x)

    def omega_prime(x: float) -> float:
        return 1.0 / (1.0 + x) ** 2

    return AllocationFunction(
        omega=omega,
        omega_prime=omega_prime,
        name="Simple Concave"
    )


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def compute_expected_allocation(allocation: AllocationFunction, mu: float, d_a: float) -> float:
    r"""
    Compute expected allocation given fraction of active suppliers.

    $q_a(\mu) = \omega(d_a / \mu)$

    This is the expected demand per active supplier when fraction $\mu$
    of suppliers are active and expected demand per supplier is $d_a$.

    Args:
        allocation: The allocation function
        mu: Fraction of suppliers who are active (in [0, 1])
        d_a: Expected demand per supplier for state a

    Returns:
        Expected allocation per active supplier
    """
    if mu <= 0:
        return 0.0
    else:
        return allocation(d_a / mu)


def compute_expected_allocation_derivative(
    allocation: AllocationFunction,
    mu: float,
    d_a: float
) -> float:
    r"""
    Compute derivative of q with respect to $\mu$.

    $\frac{d}{d\mu} q_a(\mu) = -\omega'(d_a/\mu) \cdot d_a / \mu^2$

    This is negative: more suppliers $\to$ less demand per supplier.

    Args:
        allocation: The allocation function
        mu: Fraction of active suppliers
        d_a: Expected demand per supplier

    Returns:
        $dq_a/d\mu$
    """
    if mu <= 0:
        return 0.0
    else:
        x = d_a / mu
        return -allocation.derivative(x) * d_a / (mu ** 2)


# TODO figure out if this is right
def compute_total_demand_served(
    allocation: AllocationFunction,
    d: float,
    t: float
) -> float:
    r"""
    Compute total demand served across all active suppliers.

    Total served = $t \cdot \omega(d/t)$

    This is bounded by d (can't serve more demand than exists).

    Args:
        d: Total demand
        t: Number of active suppliers
        allocation: The allocation function

    Returns:
        Total demand served
    """
    if t <= 0:
        return 0.0
    else:
        return t * allocation(d / t)
