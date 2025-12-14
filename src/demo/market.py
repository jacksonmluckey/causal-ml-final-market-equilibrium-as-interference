"""
Market Parameter Definitions

This module defines the parameter dataclasses used throughout the market model.
Centralized location for all market configuration structures.
"""

from dataclasses import dataclass
from .allocation import (
    AllocationFunction,
    create_queue_allocation
)
from .supplier import (
    ChoiceFunction,
    PrivateFeatureDistribution,
    create_logistic_choice,
    create_lognormal_costs,
)


@dataclass
class MarketParameters:
    """
    Parameters defining the marketplace model.

    Used in equilibrium computation, marginal response analysis, and market simulation.

    Parameters
    ----------
    allocation : AllocationFunction
        The allocation function ω(·) from Definition 5
    choice : ChoiceFunction
        The choice function f_b(·) from Assumption 2
    private_features : PrivateFeatureDistribution
        Distribution of B_i (outside options)
    d_a : float
        Expected demand per supplier (scaled), E[D/n | A=a]
    gamma : float
        Revenue per unit of demand served (for linear revenue function)
    n_monte_carlo : int
        Number of Monte Carlo samples for computing expectations
    """
    allocation: AllocationFunction
    choice: ChoiceFunction
    private_features: PrivateFeatureDistribution
    d_a: float = 0.4
    gamma: float = 100.0
    n_monte_carlo: int = 10000


def create_default_market_params() -> MarketParameters:
    """
    Create default market parameters matching the paper's simulations.
    
    From Figure 2's caption:
        - E[D/n | A] = 0.4
        - Logistic choice with α = 1
        - log(B_i/20) ~ N(0,1)
        - M/M/1 queues with L = 8
        - γ = 100
    """
    return MarketParameters(
        allocation=create_queue_allocation(L=8),
        choice=create_logistic_choice(alpha=1.0),
        private_features=create_lognormal_costs(
            log_mean=0.0,
            log_std=1.0,
            scale=20.0
        ),
        d_a=0.4,
        gamma=100.0,
        n_monte_carlo=10000
    )