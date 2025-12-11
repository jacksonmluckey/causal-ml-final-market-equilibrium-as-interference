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
from scipy.stats import binom


class AllocationFunction:
    """
    Base class for allocation functions.
    
    The allocation function ω(x) captures how much demand each active 
    supplier serves when the demand-to-supply ratio is x = d/t.
    """
    
    def omega(self, x: float) -> float:
        """
        Regular allocation function ω(x).
        
        Args:
            x: Demand per active supplier ratio (d/t)
            
        Returns:
            Expected demand served per active supplier
        """
        raise NotImplementedError
    
    def omega_derivative(self, x: float) -> float:
        """
        Derivative ω'(x) of the allocation function.
        
        Args:
            x: Demand per active supplier ratio
            
        Returns:
            dω/dx evaluated at x
        """
        raise NotImplementedError
    
    def Omega(self, d: float, t: float) -> float:
        """
        Finite-market allocation Ω(d, t).
        
        For large d and t, this converges to ω(d/t).
        Default implementation uses the mean-field approximation.
        
        Args:
            d: Total demand
            t: Number of active suppliers
            
        Returns:
            Expected demand per active supplier
        """
        if t <= 0:
            return 0.0
        return self.omega(d / t)
    
    def q(self, mu: float, d_a: float) -> float:
        """
        Expected allocation given fraction of active suppliers.
        
        q_a(μ) = ω(d_a / μ)
        
        This is the expected demand per active supplier when fraction μ
        of suppliers are active and expected demand per supplier is d_a.
        
        Args:
            mu: Fraction of suppliers who are active (in [0, 1])
            d_a: Expected demand per supplier for state a
            
        Returns:
            Expected allocation per active supplier
        """
        if mu <= 0:
            return 0.0
        return self.omega(d_a / mu)
    
    def q_derivative(self, mu: float, d_a: float) -> float:
        """
        Derivative of q with respect to μ.
        
        d/dμ q_a(μ) = -ω'(d_a/μ) * d_a / μ²
        
        This is negative: more suppliers → less demand per supplier.
        
        Args:
            mu: Fraction of active suppliers
            d_a: Expected demand per supplier
            
        Returns:
            dq_a/dμ
        """
        if mu <= 0:
            return 0.0
        x = d_a / mu
        return -self.omega_derivative(x) * d_a / (mu ** 2)


class QueueAllocation(AllocationFunction):
    """
    Allocation function from Example 6: Parallel Finite-Capacity Queues.
    
    Each active supplier operates as an M/M/1 queue with capacity L.
    The allocation function is:
    
    ω(x) = (x - x^L) / (1 - x^L)  for x ≠ 1
    ω(1) = 1 - 1/L
    
    Properties:
    - As L → ∞, ω(x) → min(x, 1) (suppliers can serve all demand up to capacity)
    - For finite L, some demand is dropped when queues are full
    """
    
    def __init__(self, L: int = 8):
        """
        Args:
            L: Queue capacity (must be ≥ 2)
        """
        if L < 2:
            raise ValueError("Queue capacity L must be at least 2")
        self.L = L
        
    def omega(self, x: float) -> float:
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
            return 1 - 1/self.L
            
        x_L = x ** self.L
        return (x - x_L) / (1 - x_L)
    
    def omega_derivative(self, x: float) -> float:
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
            return (self.L - 1) / (2 * self.L)
            
        L = self.L
        x_L = x ** L
        x_Lm1 = x ** (L - 1)
        
        # Numerator derivative: 1 - L*x^{L-1}
        # Denominator: 1 - x^L
        # Denominator derivative: -L*x^{L-1}
        
        numerator = (1 - L * x_Lm1) * (1 - x_L) - (x - x_L) * (-L * x_Lm1)
        denominator = (1 - x_L) ** 2
        
        return numerator / denominator


class LinearAllocation(AllocationFunction):
    """
    Simple linear allocation (limiting case as L → ∞).
    
    ω(x) = min(x, 1)
    
    Each supplier serves all their demand up to capacity 1.
    """
    
    def omega(self, x: float) -> float:
        return min(max(x, 0), 1.0)
    
    def omega_derivative(self, x: float) -> float:
        if x <= 0 or x >= 1:
            return 0.0
        return 1.0


class SmoothLinearAllocation(AllocationFunction):
    """
    Smooth approximation to linear allocation.
    
    ω(x) = 1 - exp(-x)
    
    This satisfies all regularity conditions and approximates min(x, 1).
    """
    
    def omega(self, x: float) -> float:
        if x <= 0:
            return 0.0
        return 1 - np.exp(-x)
    
    def omega_derivative(self, x: float) -> float:
        if x <= 0:
            return 1.0
        return np.exp(-x)


# Utility functions for working with allocations

def compute_total_demand_served(d: float, t: float, allocation: AllocationFunction) -> float:
    """
    Compute total demand served across all active suppliers.
    
    Total served = t * ω(d/t)
    
    This is bounded by d (can't serve more demand than exists).
    """
    if t <= 0:
        return 0.0
    return t * allocation.omega(d / t)


def compute_utilization(d: float, t: float, allocation: AllocationFunction) -> float:
    """
    Compute supplier utilization (fraction of capacity used).
    
    Utilization = ω(d/t)
    
    Returns value in [0, 1].
    """
    if t <= 0:
        return 0.0
    return allocation.omega(d / t)