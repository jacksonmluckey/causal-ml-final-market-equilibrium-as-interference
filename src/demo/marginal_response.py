"""
The Marginal Response Function

This module implements Section 3.2 of Wager & Xu (2021)
"Experimenting in Equilibrium".

Section 3.2: The Marginal Response Function
    - Marginal response function (Definition 9)
    - Relationship between Δ and μ' (Lemma 4)
    - Interference factor decomposition (Equation 3.21)
    - Utility gradient computation (Proposition 12)

References:
    Wager, S. & Xu, K. (2021). "Experimenting in Equilibrium"
"""

from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING
import numpy as np

from .allocation import AllocationFunction
from .supplier import ChoiceFunction, PrivateFeatureDistribution
from .find_equilibrium import compute_mean_field_equilibrium
from .supplier import compute_expected_choice_derivative

if TYPE_CHECKING:
    from .find_equilibrium import MeanFieldEquilibrium


# =============================================================================
# SECTION 3.2: THE MARGINAL RESPONSE FUNCTION
# =============================================================================

@dataclass
class MarginalResponseAnalysis:
    """
    Analysis of the marginal response function and interference effects.

    From Section 3.2, this captures:
    - The marginal response Δ_a(p) (Definition 9)
    - The actual supply gradient μ'_a(p) (Lemma 4)
    - The interference factor decomposition (Equation 3.21)

    Attributes
    ----------
    delta : float
        Marginal response function Δ_a(p) from Definition 9:
        Δ_a(p) = ω(d_a/μ) · E[f'_{B_1}(p·ω(d_a/μ))]
    mu_prime : float
        Actual supply gradient dμ_a(p)/dp from Lemma 4 (Equation 3.20)
    interference_factor : float
        The factor 1 + R_a(p) that attenuates the marginal response
    sigma_delta : float
        Scaled marginal sensitivity Σ^Δ_a(p) from Equation 3.21
    sigma_omega : float
        Scaled matching elasticity Σ^Ω_a(p) from Equation 3.21
    """
    delta: float              # Marginal response Δ_a(p)
    mu_prime: float          # Actual gradient dμ/dp
    interference_factor: float  # 1 + R_a(p)
    sigma_delta: float       # Scaled marginal sensitivity
    sigma_omega: float       # Scaled matching elasticity


def compute_marginal_response(
    p: float,
    equilibrium: MeanFieldEquilibrium,
    choice: ChoiceFunction,
    private_features: PrivateFeatureDistribution,
    n_monte_carlo: int = 10000
) -> float:
    """
    Compute the marginal response function Δ_a(p).

    From Definition 9 (Equation 3.17 and its limit 3.19):
        Δ_a(p) = ω(d_a/μ_a(p)) · E[f'_{B_1}(p·ω(d_a/μ_a(p))) | A=a]

    Intuition:
        - This measures how a SINGLE supplier's activation probability
          changes when their payment changes, HOLDING EQUILIBRIUM FIXED
        - It captures the "direct effect" ignoring feedback
        - q = ω(d_a/μ) is the allocation rate
        - E[f'_B(p·q)] is the expected sensitivity of choice to revenue

    Parameters
    ----------
    p : float
        Payment per unit of demand served
    equilibrium : MeanFieldEquilibrium
        Pre-computed equilibrium for payment p
    params : MarketParameters
        Model parameters

    Returns
    -------
    float
        Marginal response Δ_a(p)
    """
    q = equilibrium.q  # ω(d_a/μ)
    expected_revenue = p * q

    # E[f'_{B_1}(p·q)]
    expected_choice_deriv = compute_expected_choice_derivative(
        expected_revenue,
        choice,
        private_features,
        n_monte_carlo
    )

    # Δ = q · E[f'_B(p·q)]
    delta = q * expected_choice_deriv

    return delta


def compute_supply_gradient(
    p: float,
    equilibrium: MeanFieldEquilibrium,
    delta: float,
    allocation: AllocationFunction
) -> float:
    """
    Compute the actual supply gradient dμ_a(p)/dp using Lemma 4.

    From Equation 3.20:
        μ'_a(p) = Δ_a(p) / (1 + (p·d_a·Δ_a(p)·ω'(d_a/μ)) / (μ²·ω(d_a/μ)))

    The denominator captures the interference attenuation:
        - If Δ = 0: suppliers don't respond, so μ' = 0
        - If ω' = 0: allocation doesn't depend on supply, no interference
        - Otherwise: interference REDUCES the actual effect

    Note: The paper uses the form:
        μ' = Δ / (1 + R)
    where R = Σ^Δ · Σ^Ω (see Equation 3.21)

    Parameters
    ----------
    p : float
        Payment
    equilibrium : MeanFieldEquilibrium
        Pre-computed equilibrium
    delta : float
        Marginal response Δ_a(p)

    Returns
    -------
    float
        Supply gradient dμ_a(p)/dp
    """
    mu = equilibrium.mu
    q = equilibrium.q  # ω(d_a/μ)
    x = equilibrium.demand_supply_ratio  # d_a/μ

    # ω'(d_a/μ)
    omega_prime = allocation.derivative(x)

    # From Equation 3.20:
    # μ' = Δ / (1 + p·d_a·Δ·ω'(x) / (μ²·ω(x)))
    # The denominator is: 1 + interference_term

    if abs(q) < 1e-10:  # Avoid division by zero
        return delta

    interference_term = (p * equilibrium.d_a * delta * omega_prime) / (mu**2 * q)

    # Note: omega_prime is typically positive for our allocation functions,
    # and delta is positive, so interference_term > 0
    # This means μ' < Δ: the actual effect is attenuated by interference

    mu_prime = delta / (1.0 + interference_term)

    return mu_prime


def analyze_marginal_response(
    p: float,
    equilibrium: MeanFieldEquilibrium,
    choice: ChoiceFunction,
    private_features: PrivateFeatureDistribution,
    allocation: AllocationFunction,
    n_monte_carlo: int = 10000
) -> MarginalResponseAnalysis:
    """
    Comprehensive analysis of the marginal response function.

    This implements Section 3.2, computing:
    1. Marginal response Δ_a(p) (Definition 9)
    2. Actual supply gradient μ'_a(p) (Lemma 4)
    3. Interference factor decomposition (Equation 3.21)

    The interference factor 1 + R_a(p) can be decomposed as:
        R_a(p) = Σ^Δ_a(p) · Σ^Ω_a(p)

    where:
        Σ^Δ_a(p) = p·Δ_a(p)/μ_a(p)  [scaled marginal sensitivity]
        Σ^Ω_a(p) = (d_a/μ)·ω'(d_a/μ)/ω(d_a/μ)  [scaled matching elasticity]

    From the paper's discussion after Equation 3.21:
        - Interference is negligible when Σ^Δ is small (suppliers unresponsive)
        - Interference is negligible when Σ^Ω is small (demand >> supply)

    Parameters
    ----------
    p : float
        Payment per unit of demand served
    equilibrium : Optional[MeanFieldEquilibrium]
        Pre-computed equilibrium (computed if not provided)

    Returns
    -------
    MarginalResponseAnalysis
        Complete analysis of marginal response and interference
    """

    # Marginal response (Definition 9)
    delta = compute_marginal_response(
        p,
        equilibrium,
        choice,
        private_features,
        n_monte_carlo
    )

    # Decomposition of interference factor (Equation 3.21)
    mu = equilibrium.mu
    q = equilibrium.q
    x = equilibrium.demand_supply_ratio
    omega_prime = allocation.derivative(x)

    # Scaled marginal sensitivity: Σ^Δ = p·Δ/μ
    sigma_delta = (p * delta) / mu if mu > 1e-10 else 0.0

    # Scaled matching elasticity: Σ^Ω = x·ω'(x)/ω(x) where x = d_a/μ
    sigma_omega = (x * omega_prime) / q if q > 1e-10 else 0.0

    # Interference factor: 1 + R = 1 + Σ^Δ · Σ^Ω
    R = sigma_delta * sigma_omega
    interference_factor = 1.0 + R

    # Actual supply gradient (Lemma 4)
    mu_prime = delta / interference_factor

    return MarginalResponseAnalysis(
        delta=delta,
        mu_prime=mu_prime,
        interference_factor=interference_factor,
        sigma_delta=sigma_delta,
        sigma_omega=sigma_omega
    )


# def compute_utility_gradient(
#     p: float,
#     params: MarketParameters,
#     equilibrium: Optional["MeanFieldEquilibrium"] = None,
#     marginal_analysis: Optional[MarginalResponseAnalysis] = None
# ) -> float:
#     """
#     Compute the utility gradient du_a(p)/dp.

#     From Proposition 12 in the appendix, for linear revenue r(x) = γ·ω(x):
#         u'_a(p) = μ'_a(p) · [r(x) - p·ω(x) - (r'(x) - p·ω'(x))·x] - ω(x)·μ_a(p)

#     where x = d_a/μ_a(p).

#     For r(x) = γ·ω(x), this simplifies to:
#         u'_a(p) = μ'_a(p) · [(γ-p)·ω(x) - (γ-p)·ω'(x)·x] - ω(x)·μ_a(p)
#                 = μ'_a(p) · (γ-p) · [ω(x) - ω'(x)·x] - ω(x)·μ_a(p)

#     Parameters
#     ----------
#     p : float
#         Payment
#     params : MarketParameters
#         Model parameters
#     equilibrium : Optional[MeanFieldEquilibrium]
#         Pre-computed equilibrium
#     marginal_analysis : Optional[MarginalResponseAnalysis]
#         Pre-computed marginal response analysis

#     Returns
#     -------
#     float
#         Utility gradient du_a(p)/dp
#     """
#     from .find_equilibrium import compute_mean_field_equilibrium

#     if equilibrium is None:
#         equilibrium = compute_mean_field_equilibrium(p, params)

#     if marginal_analysis is None:
#         marginal_analysis = analyze_marginal_response(p, params, equilibrium)

#     mu = equilibrium.mu
#     q = equilibrium.q  # ω(x)
#     x = equilibrium.demand_supply_ratio
#     omega_prime = params.allocation.derivative(x)
#     mu_prime = marginal_analysis.mu_prime

#     # From Proposition 12 with r(x) = γ·ω(x):
#     # Term 1: μ' · (γ-p) · [ω(x) - ω'(x)·x]
#     # Term 2: -ω(x) · μ

#     bracket_term = q - omega_prime * x  # ω(x) - ω'(x)·x

#     u_prime = mu_prime * (params.gamma - p) * bracket_term - q * mu

#     return u_prime


# # =============================================================================
# # CONVENIENCE FUNCTIONS FOR ANALYSIS
# # =============================================================================

# def analyze_payment_range(
#     params: MarketParameters,
#     p_min: float = 5.0,
#     p_max: float = 60.0,
#     n_points: int = 20
# ) -> dict:
#     """
#     Analyze equilibrium and marginal response across a range of payments.

#     Useful for understanding how the market behaves and for finding
#     the optimal payment.

#     Parameters
#     ----------
#     params : MarketParameters
#         Model parameters
#     p_min, p_max : float
#         Payment range to analyze
#     n_points : int
#         Number of points to evaluate

#     Returns
#     -------
#     dict
#         Dictionary containing arrays of:
#         - payments
#         - equilibrium supply (mu)
#         - allocation rates (q)
#         - utilities (u)
#         - marginal responses (delta)
#         - supply gradients (mu_prime)
#         - utility gradients (u_prime)
#         - interference factors
#     """
#     from .find_equilibrium import compute_mean_field_equilibrium

#     payments = np.linspace(p_min, p_max, n_points)

#     results = {
#         'payments': payments,
#         'mu': np.zeros(n_points),
#         'q': np.zeros(n_points),
#         'u': np.zeros(n_points),
#         'delta': np.zeros(n_points),
#         'mu_prime': np.zeros(n_points),
#         'u_prime': np.zeros(n_points),
#         'interference_factor': np.zeros(n_points)
#     }

#     for i, p in enumerate(payments):
#         eq = compute_mean_field_equilibrium(p, params)
#         analysis = analyze_marginal_response(p, params, eq)
#         u_prime = compute_utility_gradient(p, params, eq, analysis)

#         results['mu'][i] = eq.mu
#         results['q'][i] = eq.q
#         results['u'][i] = eq.u
#         results['delta'][i] = analysis.delta
#         results['mu_prime'][i] = analysis.mu_prime
#         results['u_prime'][i] = u_prime
#         results['interference_factor'][i] = analysis.interference_factor

#     return results
