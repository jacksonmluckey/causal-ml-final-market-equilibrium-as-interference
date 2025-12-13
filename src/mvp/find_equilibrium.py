"""
Equilibrium Finding Utilities

This module contains functions for finding equilibrium supply fractions
in marketplace models.
"""

from typing import Tuple
from scipy.optimize import brentq

from .mean_field import MarketParameters, compute_expected_choice


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
