"""
Equilibrium Finding Utilities

This module implements Section 3.1 (Mean-Field Asymptotics) of Wager & Xu (2021)
"Experimenting in Equilibrium".

It contains functions for finding and computing mean-field equilibrium in marketplace
models, including:
- Equilibrium supply computation (Definition 8, Lemma 1)
- Convergence results (Lemma 2)
- Utility computation (Equation 3.15)
"""

from dataclasses import dataclass
from typing import Tuple
import numpy as np
from scipy.optimize import brentq

from .supplier import (
    ChoiceFunction,
    PrivateFeatureDistribution,
    compute_expected_choice_probability
)
from .allocation import (
    AllocationFunction
)


@dataclass
class MeanFieldEquilibrium:
    """
    Mean-field equilibrium of the marketplace.

    This implements the limiting equilibrium from Lemma 2, where
    the number of suppliers n → ∞.

    Attributes
    ----------
    mu : float
        Equilibrium fraction of active suppliers $\mu_a(p)$
        Solves: $\mu = E[f_{B_1}(p \cdot \omega(d_a/\mu)) | A=a]$  (Equation 3.13)
    q : float
        Expected allocation per active supplier $q_a(\mu_a(p)) = \omega(d_a/\mu)$
    u : float
        Platform's expected utility per supplier $u_a(p)$
        From Equation 3.15: $u_a(p) = (r(d_a/\mu) - p \cdot \omega(d_a/\mu)) \cdot \mu$
    demand_supply_ratio : float
        The ratio $x = d_a / \mu$ used in the allocation function
    """
    p: float
    d_a: float
    mu: float                   # Equilibrium supply fraction
    q: float                    # Allocation per active supplier
    #u: float                    # Platform utility
    demand_supply_ratio: float  # x = d_a / μ


def find_equilibrium_supply_mu(
    p: float,
    d_a: float,
    choice: ChoiceFunction,
    private_features: PrivateFeatureDistribution,
    allocation: AllocationFunction,
    mu_bounds: Tuple[float, float] = (1e-6, 1.0 - 1e-6),
    tol: float = 1e-8,
    n_samples: int = 10000
) -> float:
    """
    Solve for equilibrium supply fraction $\mu_a(p)$.

    From Lemma 2, $\mu_a(p)$ is the unique solution to the fixed-point equation:
        $\mu = E[f_{B_1}(p \cdot \omega(d_a/\mu)) | A=a]$

    This is Equation 3.13 in the mean-field limit.

    Implementation uses Brent's method to find the root of:
        $g(\mu) = \mu - E[f_{B_1}(p \cdot \omega(d_a/\mu))] = 0$

    Parameters
    ----------
    p : float
        Payment per unit of demand served
    d_a: float
    choice: ChoiceFunction
    private_features: PrivateFeatureDistribution
    allocation: AllocationFunction
    mu_bounds : Tuple[float, float]
        Bounds for $\mu$ search (default: (1e-6, 1-1e-6))
    tol : float
        Tolerance for root finding
    m_samples: int

    Returns
    -------
    float
        Equilibrium supply fraction $\mu_a(p)$

    Notes
    -----
    Existence and uniqueness guaranteed by Lemma 1:
    - As $\mu$ increases, allocation q decreases (more competition)
    - This makes joining less attractive, reducing $E[f(...)]$
    - The negative feedback ensures a unique fixed point
    """
    def fixed_point_residual(mu: float) -> float:
        """$g(\mu) = \mu - E[f_{B_1}(p \cdot \omega(d_a/\mu))]$"""
        x = d_a / mu  # demand-to-supply ratio
        q = allocation(x)  # allocation per supplier
        expected_revenue = p * q
        expected_choice = compute_expected_choice_probability(
            expected_revenue,
            choice,
            private_features,
            n_samples
        )
        return mu - expected_choice

    # Brent's method to find root
    try:
        mu_star = brentq(fixed_point_residual, mu_bounds[0], mu_bounds[1], xtol=tol)
    except ValueError:
        # If bounds don't bracket root, try bisection with expanded search
        # This can happen for extreme parameter values
        mu_star = 0.5  # fallback
        for _ in range(100):
            residual = fixed_point_residual(mu_star)
            if abs(residual) < tol:
                break
            if residual > 0:
                mu_star *= 0.9
            else:
                mu_star = 0.5 * (mu_star + 1.0)

    return mu_star


def compute_mean_field_equilibrium(
    p: float,
    d_a: float,
    gamma: float,
    choice: ChoiceFunction,
    private_features: PrivateFeatureDistribution,
    allocation: AllocationFunction
) -> MeanFieldEquilibrium:
    """
    Compute the mean-field equilibrium for payment p.

    This implements Lemma 2, computing all key equilibrium quantities:

    1. $\mu_a(p)$: Equilibrium supply fraction (Equation 3.13)
    2. $q_a(\mu_a(p)) = \omega(d_a/\mu)$: Allocation per supplier (Equation 3.14)
    3. $u_a(p)$: Platform utility (Equation 3.15)

    For the linear revenue function $r(x) = \gamma \cdot \omega(x)$ from Lemma 3,
    the utility simplifies to:
        $u_a(p) = (\gamma - p) \cdot \omega(d_a/\mu) \cdot \mu$

    Parameters
    ----------
    p : float
        Payment per unit of demand served

    Returns
    -------
    MeanFieldEquilibrium
        The equilibrium quantities
    """
    # Step 1: Solve for equilibrium supply (Equation 3.13)
    mu = find_equilibrium_supply_mu(
        p,
        d_a,
        choice,
        private_features,
        allocation
    )

    # Step 2: Compute allocation (Equation 3.14)
    x = d_a / mu  # demand-to-supply ratio
    q = allocation(x)

    # Step 3: Compute utility (Equation 3.15)
    # Using linear revenue: $r(x) = \gamma \cdot \omega(x)$
    # $u_a(p) = (r(d_a/\mu) - p \cdot \omega(d_a/\mu)) \cdot \mu = (\gamma - p) \cdot q \cdot \mu$
    # TODO switch to using the utility in platform_utility
    #u = (gamma - p) * q * mu

    return MeanFieldEquilibrium(
        p=p,
        d_a=d_a,
        mu=mu,
        q=q,
        #u=u,
        demand_supply_ratio=x
    )
