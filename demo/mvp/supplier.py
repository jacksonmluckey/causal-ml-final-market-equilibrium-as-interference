"""
Supplier Choice Model for Stochastic Market

This module implements the supplier decision-making process.

Key equation (3.6):
μ_a(π) = E[f_{B_i}(P_i * E[Ω(D,T)|A=a]) | A=a]

Where:
- P_i is the payment offered to supplier i
- B_i is supplier i's private feature (e.g., outside option/cost)
- f_b(x) is the choice function: probability of becoming active given expected revenue x

The key insight is that suppliers use STATIONARY reasoning:
- They consider the average market equilibrium, not their own effect on it
- This is justified in large markets where individual impact is negligible
"""

import numpy as np
from typing import Callable, Optional, Tuple
from dataclasses import dataclass
from scipy.optimize import brentq
from scipy.stats import norm, expon


@dataclass
class SupplierType:
    """
    Represents heterogeneity across suppliers.
    
    B_i is drawn from a distribution - this class represents one realization.
    """
    b: float  # The private feature value (e.g., cost threshold)


class ChoiceFunction:
    """
    Base class for supplier choice functions.
    
    f_b(x) maps expected revenue x to probability of becoming active,
    given private feature b.
    
    Requirements (Assumption 2):
    - Takes values in [0, 1]
    - Monotonically non-decreasing in x
    - Twice differentiable with bounded second derivative
    """
    
    def f(self, x: float, b: float) -> float:
        """
        Choice probability given expected revenue x and private feature b.
        
        Args:
            x: Expected revenue (= payment * expected allocation)
            b: Private feature (e.g., cost threshold)
            
        Returns:
            Probability in [0, 1] of becoming active
        """
        raise NotImplementedError
    
    def f_derivative(self, x: float, b: float) -> float:
        """
        Derivative df/dx of choice function with respect to expected revenue.
        
        This measures how sensitive the supplier's decision is to revenue changes.
        """
        raise NotImplementedError


class LogisticChoice(ChoiceFunction):
    """
    Logistic choice function (Example 7 in the paper).
    
    P[Z_i = 1 | P_i, π, A] = 1 / (1 + exp(-α(P_i * E[Ω] - B_i)))
    
    Properties:
    - α controls sensitivity: larger α → more deterministic decisions
    - B_i is the break-even cost threshold
    - Supplier activates if expected revenue exceeds their cost threshold
    
    As α → ∞, this becomes a step function: activate iff revenue > cost
    """
    
    def __init__(self, alpha: float = 1.0):
        """
        Args:
            alpha: Sensitivity parameter (α > 0)
        """
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        self.alpha = alpha
        
    def f(self, x: float, b: float) -> float:
        """
        Logistic choice probability.
        
        f_b(x) = 1 / (1 + exp(-α(x - b)))
        
        Args:
            x: Expected revenue
            b: Break-even cost threshold
        """
        z = self.alpha * (x - b)
        # Clip for numerical stability
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def f_derivative(self, x: float, b: float) -> float:
        """
        Derivative of logistic function.
        
        df/dx = α * f(x,b) * (1 - f(x,b))
        """
        f_val = self.f(x, b)
        return self.alpha * f_val * (1 - f_val)


class CostDistribution:
    """
    Distribution of supplier private features B_i.
    
    The paper assumes B_i are i.i.d. from some distribution over B.
    """
    
    def sample(self, n: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Sample n private feature values."""
        raise NotImplementedError
    
    def expected_choice_probability(self, x: float, choice_fn: ChoiceFunction) -> float:
        """
        Compute E[f_B(x)] = ∫ f_b(x) dF(b)
        
        This is the average probability of activation across all supplier types
        when expected revenue is x.
        """
        raise NotImplementedError
    
    def expected_choice_derivative(self, x: float, choice_fn: ChoiceFunction) -> float:
        """
        Compute E[f'_B(x)] = ∫ f'_b(x) dF(b)
        
        Average sensitivity of activation to revenue changes.
        """
        raise NotImplementedError


class LogNormalCosts(CostDistribution):
    """
    Log-normal distribution for supplier costs.
    
    log(B_i) ~ N(μ, σ²)
    
    This matches the paper's example where log(B_i/20) ~ N(0, 1).
    """
    
    def __init__(self, log_mean: float = 0.0, log_std: float = 1.0, scale: float = 20.0):
        """
        Args:
            log_mean: Mean of log(B/scale)
            log_std: Std of log(B/scale)  
            scale: Scaling factor (B = scale * exp(Normal))
        """
        self.log_mean = log_mean
        self.log_std = log_std
        self.scale = scale
        
    def sample(self, n: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Sample n cost thresholds."""
        if rng is None:
            rng = np.random.default_rng()
        log_values = rng.normal(self.log_mean, self.log_std, size=n)
        return self.scale * np.exp(log_values)
    
    def expected_choice_probability(self, x: float, choice_fn: ChoiceFunction,
                                     n_samples: int = 10000) -> float:
        """Monte Carlo estimate of E[f_B(x)]."""
        rng = np.random.default_rng(42)  # Fixed seed for consistency
        b_samples = self.sample(n_samples, rng)
        return np.mean([choice_fn.f(x, b) for b in b_samples])
    
    def expected_choice_derivative(self, x: float, choice_fn: ChoiceFunction,
                                    n_samples: int = 10000) -> float:
        """Monte Carlo estimate of E[f'_B(x)]."""
        rng = np.random.default_rng(42)
        b_samples = self.sample(n_samples, rng)
        return np.mean([choice_fn.f_derivative(x, b) for b in b_samples])


class UniformCosts(CostDistribution):
    """
    Uniform distribution for supplier costs.
    
    B_i ~ Uniform(low, high)
    """
    
    def __init__(self, low: float = 5.0, high: float = 50.0):
        self.low = low
        self.high = high
        
    def sample(self, n: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()
        return rng.uniform(self.low, self.high, size=n)
    
    def expected_choice_probability(self, x: float, choice_fn: ChoiceFunction,
                                     n_samples: int = 10000) -> float:
        rng = np.random.default_rng(42)
        b_samples = self.sample(n_samples, rng)
        return np.mean([choice_fn.f(x, b) for b in b_samples])
    
    def expected_choice_derivative(self, x: float, choice_fn: ChoiceFunction,
                                    n_samples: int = 10000) -> float:
        rng = np.random.default_rng(42)
        b_samples = self.sample(n_samples, rng)
        return np.mean([choice_fn.f_derivative(x, b) for b in b_samples])


class SupplierPopulation:
    """
    Models a population of heterogeneous suppliers.
    
    Combines:
    - Choice function f_b(x)
    - Cost distribution for B_i
    
    Main method: compute equilibrium activation probability μ_a(p)
    """
    
    def __init__(self, choice_fn: ChoiceFunction, cost_dist: CostDistribution):
        self.choice_fn = choice_fn
        self.cost_dist = cost_dist
        
    def activation_probability(self, expected_revenue: float) -> float:
        """
        Compute μ = E[f_B(expected_revenue)].
        
        This is the fraction of suppliers who become active when 
        expected revenue is `expected_revenue`.
        """
        return self.cost_dist.expected_choice_probability(
            expected_revenue, self.choice_fn
        )
    
    def activation_sensitivity(self, expected_revenue: float) -> float:
        """
        Compute E[f'_B(expected_revenue)].
        
        This measures how sensitive activation is to revenue changes.
        Used in computing the marginal response function.
        """
        return self.cost_dist.expected_choice_derivative(
            expected_revenue, self.choice_fn
        )
    
    def sample_suppliers(self, n: int, payments: np.ndarray, 
                         expected_allocation: float,
                         rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """
        Sample activation decisions for n suppliers.
        
        Args:
            n: Number of suppliers
            payments: Payment offered to each supplier (array of length n)
            expected_allocation: Expected demand per active supplier E[Ω]
            rng: Random number generator
            
        Returns:
            Z: Array of activation decisions (0 or 1) for each supplier
        """
        if rng is None:
            rng = np.random.default_rng()
            
        # Sample private costs
        costs = self.cost_dist.sample(n, rng)
        
        # Compute expected revenue for each supplier
        expected_revenues = payments * expected_allocation
        
        # Compute activation probabilities
        probs = np.array([
            self.choice_fn.f(rev, cost) 
            for rev, cost in zip(expected_revenues, costs)
        ])
        
        # Sample activation decisions
        Z = (rng.random(n) < probs).astype(int)
        
        return Z