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

from .mean_field import MarketParameters
from .supplier import ChoiceFunction, PrivateFeatureDistribution


@dataclass
class MeanFieldEquilibrium:
    """
    Mean-field equilibrium of the marketplace.

    This implements the limiting equilibrium from Lemma 2, where
    the number of suppliers n → ∞.

    Attributes
    ----------
    mu : float
        Equilibrium fraction of active suppliers μ_a(p)
        Solves: μ = E[f_{B_1}(p · ω(d_a/μ)) | A=a]  (Equation 3.13)
    q : float
        Expected allocation per active supplier q_a(μ_a(p)) = ω(d_a/μ)
    u : float
        Platform's expected utility per supplier u_a(p)
        From Equation 3.15: u_a(p) = (r(d_a/μ) - p·ω(d_a/μ)) · μ
    demand_supply_ratio : float
        The ratio x = d_a / μ used in the allocation function
    """
    mu: float                   # Equilibrium supply fraction
    q: float                    # Allocation per active supplier
    u: float                    # Platform utility
    demand_supply_ratio: float  # x = d_a / μ


def compute_expected_choice(
    revenue: float,
    choice: ChoiceFunction,
    private_features: PrivateFeatureDistribution,
    n_samples: int = 10000
) -> float:
    """
    Compute E[f_{B_1}(revenue) | A=a] via Monte Carlo.

    This is the expected probability of becoming active given
    expected revenue, averaging over the private feature distribution.

    Parameters
    ----------
    revenue : float
        Expected revenue (= p · q)
    choice : ChoiceFunction
        The choice function f_b(·)
    private_features : PrivateFeatureDistribution
        Distribution of B_i
    n_samples : int
        Number of Monte Carlo samples

    Returns
    -------
    float
        E[f_{B_1}(revenue)]
    """
    b_samples = private_features.sample(n_samples)
    probs = np.array([choice(revenue, b) for b in b_samples])
    return np.mean(probs)


def solve_equilibrium_supply(
    p: float,
    params: MarketParameters,
    mu_bounds: Tuple[float, float] = (1e-6, 1.0 - 1e-6),
    tol: float = 1e-8
) -> float:
    """
    Solve for equilibrium supply fraction μ_a(p).

    From Lemma 2, μ_a(p) is the unique solution to the fixed-point equation:
        μ = E[f_{B_1}(p · ω(d_a/μ)) | A=a]

    This is Equation 3.13 in the mean-field limit.

    Implementation uses Brent's method to find the root of:
        g(μ) = μ - E[f_{B_1}(p · ω(d_a/μ))] = 0

    Parameters
    ----------
    p : float
        Payment per unit of demand served
    params : MarketParameters
        Model parameters
    mu_bounds : Tuple[float, float]
        Bounds for μ search (default: (1e-6, 1-1e-6))
    tol : float
        Tolerance for root finding

    Returns
    -------
    float
        Equilibrium supply fraction μ_a(p)

    Notes
    -----
    Existence and uniqueness guaranteed by Lemma 1:
    - As μ increases, allocation q decreases (more competition)
    - This makes joining less attractive, reducing E[f(...)]
    - The negative feedback ensures a unique fixed point
    """
    def fixed_point_residual(mu: float) -> float:
        """g(μ) = μ - E[f_{B_1}(p · ω(d_a/μ))]"""
        x = params.d_a / mu  # demand-to-supply ratio
        q = params.allocation(x)  # allocation per supplier
        expected_revenue = p * q
        expected_choice = compute_expected_choice(
            expected_revenue,
            params.choice,
            params.private_features,
            params.n_monte_carlo
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
    params: MarketParameters
) -> MeanFieldEquilibrium:
    """
    Compute the mean-field equilibrium for payment p.

    This implements Lemma 2, computing all key equilibrium quantities:

    1. μ_a(p): Equilibrium supply fraction (Equation 3.13)
    2. q_a(μ_a(p)) = ω(d_a/μ): Allocation per supplier (Equation 3.14)
    3. u_a(p): Platform utility (Equation 3.15)

    For the linear revenue function r(x) = γ·ω(x) from Lemma 3,
    the utility simplifies to:
        u_a(p) = (γ - p) · ω(d_a/μ) · μ

    Parameters
    ----------
    p : float
        Payment per unit of demand served
    params : MarketParameters
        Model parameters

    Returns
    -------
    MeanFieldEquilibrium
        The equilibrium quantities
    """
    # Step 1: Solve for equilibrium supply (Equation 3.13)
    mu = solve_equilibrium_supply(p, params)

    # Step 2: Compute allocation (Equation 3.14)
    x = params.d_a / mu  # demand-to-supply ratio
    q = params.allocation(x)

    # Step 3: Compute utility (Equation 3.15)
    # Using linear revenue: r(x) = γ · ω(x)
    # u_a(p) = (r(d_a/μ) - p·ω(d_a/μ)) · μ = (γ - p) · q · μ
    u = (params.gamma - p) * q * mu

    return MeanFieldEquilibrium(
        mu=mu,
        q=q,
        u=u,
        demand_supply_ratio=x
    )
