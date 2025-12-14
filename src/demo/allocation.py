"""
Allocation Function for Stochastic Market

This module implements the regular allocation function ω(x) and the
finite-market allocation Ω(d, t).

A regular allocation function (Definition 5) must satisfy:
1. Smooth, concave, non-decreasing
2. lim_{x→0} ω(x) = 0 (no demand → no allocation)
3. lim_{x→∞} ω(x) ≤ 1 (suppliers have bounded capacity)
4. lim_{x→0} ω'(x) ≤ 1 (allocation rate bounded by demand)

Key relationship (Assumption 1):
Ω(d, t) = ω(d/t) + l(d, t)

where l(d, t) is an error term that vanishes as d, t → ∞
"""

import numpy as np
from typing import Callable, Optional
from dataclasses import dataclass

from .utils import numerical_derivative


@dataclass
class AllocationFunction:
    """
    Regular allocation function ω(x) as defined in Definition 5.

    The allocation function captures how much demand each active
    supplier serves when the demand-to-supply ratio is x = d/t.

    Parameters
    ----------
    omega : Callable[[float], float]
        The allocation function ω(x) mapping demand-to-supply ratio to
        expected demand served per active supplier
    omega_prime : Optional[Callable[[float], float]]
        The derivative ω'(x). If None, must be computed numerically when needed.
    name : str
        Descriptive name for the allocation function
    """
    omega: Callable[[float], float]
    omega_prime: Optional[Callable[[float], float]] = None
    name: str = "Generic"

    def __call__(self, x: float) -> float:
        """Evaluate ω(x)."""
        return self.omega(x)
    
    def derivative(self, x: float) -> float:
        """Evaluate ω'(x)."""
        if self.omega_prime is not None:
            return self.omega_prime(x)
        return numerical_derivative(self.omega, x)


def create_queue_allocation(L: int = 8) -> AllocationFunction:
    """
    Create allocation function from Example 6: Parallel Finite-Capacity Queues.

    Each active supplier operates as an M/M/1 queue with capacity L.
    The allocation function is:

    ω(x) = (x - x^L) / (1 - x^L)  for x ≠ 1
    ω(1) = 1 - 1/L

    Properties:
    - As L → ∞, ω(x) → min(x, 1) (suppliers can serve all demand up to capacity)
    - For finite L, some demand is dropped when queues are full

    Parameters
    ----------
    L : int
        Queue capacity (must be ≥ 2)

    Returns
    -------
    AllocationFunction
        The queue allocation function
    """
    if L < 2:
        raise ValueError("Queue capacity L must be at least 2")

    def omega(x: float) -> float:
        """
        Allocation function for M/M/1 queue with capacity L.

        ω(x) = (x - x^L) / (1 - x^L)
        """
        if x <= 0:
            return 0.0
        if x > 10:  # For very large x, ω(x) → 1
            return 1.0
        # Handle x ≈ 1 separately to avoid numerical issues
        if abs(x - 1) < 1e-10:
            return 1 - 1/L
        x_L = x ** L
        return (x - x_L) / (1 - x_L)

    def omega_prime(x: float) -> float:
        """
        Derivative of ω(x).

        Using quotient rule on ω(x) = (x - x^L) / (1 - x^L)
        """
        if x <= 0:
            return 1.0  # lim_{x→0} ω'(x) = 1
        if x > 10:
            return 0.0  # For large x, ω(x) ≈ 1, so derivative ≈ 0
        if abs(x - 1) < 1e-10:
            # Use L'Hopital's rule or series expansion near x=1
            # ω'(1) = (L-1)/(2L)
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
    """
    Create simple linear allocation: ω(x) = min(x, 1) (limiting case as L → ∞).

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
    """
    Create smooth approximation to linear allocation.

    ω(x) = 1 - exp(-x)

    This satisfies all regularity conditions and approximates min(x, 1).

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
    """
    Create a simple concave allocation function for testing.

    Uses ω(x) = x / (1 + x), which satisfies all properties in Definition 5:
        - Smooth, concave, non-decreasing
        - ω(0) = 0, ω(∞) = 1
        - ω'(0) = 1
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

def compute_omega(allocation: AllocationFunction, x: float) -> float:
    """
    Evaluate regular allocation function ω(x).

    Args:
        allocation: The allocation function
        x: Demand per active supplier ratio (d/t)

    Returns:
        Expected demand served per active supplier
    """
    return allocation(x)


def compute_omega_prime(allocation: AllocationFunction, x: float) -> float:
    """
    Evaluate derivative ω'(x) of the allocation function.

    Args:
        allocation: The allocation function
        x: Demand per active supplier ratio

    Returns:
        dω/dx evaluated at x
    """
    return allocation.derivative(x)


def compute_finite_allocation(allocation: AllocationFunction, d: float, t: float) -> float:
    """
    Compute finite-market allocation Ω(d, t).

    For large d and t, this converges to ω(d/t).
    Default implementation uses the mean-field approximation.

    Args:
        allocation: The allocation function
        d: Total demand
        t: Number of active suppliers

    Returns:
        Expected demand per active supplier
    """
    if t <= 0:
        return 0.0
    else:
        return allocation(d / t)


def compute_expected_allocation(allocation: AllocationFunction, mu: float, d_a: float) -> float:
    """
    Compute expected allocation given fraction of active suppliers.

    q_a(μ) = ω(d_a / μ)

    This is the expected demand per active supplier when fraction μ
    of suppliers are active and expected demand per supplier is d_a.

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
    """
    Compute derivative of q with respect to μ.

    d/dμ q_a(μ) = -ω'(d_a/μ) * d_a / μ²

    This is negative: more suppliers → less demand per supplier.

    Args:
        allocation: The allocation function
        mu: Fraction of active suppliers
        d_a: Expected demand per supplier

    Returns:
        dq_a/dμ
    """
    if mu <= 0:
        return 0.0
    else:
        x = d_a / mu
        return -allocation.derivative(x) * d_a / (mu ** 2)


def compute_total_demand_served(
    d: float,
    t: float,
    allocation: AllocationFunction
) -> float:
    """
    Compute total demand served across all active suppliers.

    Total served = t * ω(d/t)

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


def compute_utilization(
    d: float,
    t: float,
    allocation: AllocationFunction
) -> float:
    """
    Compute supplier utilization (fraction of capacity used).

    Utilization = ω(d/t)

    Returns value in [0, 1].

    Args:
        d: Total demand
        t: Number of active suppliers
        allocation: The allocation function

    Returns:
        Utilization fraction
    """
    if t <= 0:
        return 0.0
    else:
        return allocation(d / t)
