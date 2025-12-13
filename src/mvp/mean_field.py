"""
Mean-Field Asymptotics and Marginal Response Function

This module implements Sections 3.1 and 3.2 of Wager & Xu (2021) 
"Experimenting in Equilibrium".

Section 3.1: Mean-Field Asymptotics
    - Equilibrium supply computation (Definition 8, Lemma 1)
    - Convergence results (Lemma 2)
    - Utility computation (Equation 3.15)

Section 3.2: The Marginal Response Function
    - Marginal response function (Definition 9)
    - Relationship between Δ and μ' (Lemma 4)
    - Interference factor decomposition (Equation 3.21)

References:
    Wager, S. & Xu, K. (2021). "Experimenting in Equilibrium"
"""

from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple
import numpy as np
from scipy.optimize import brentq


def numerical_derivative(f: Callable[[float], float], x: float, dx: float = 1e-6) -> float:
    """Compute numerical derivative using central difference."""
    return (f(x + dx) - f(x - dx)) / (2 * dx)


# =============================================================================
# ALLOCATION FUNCTIONS (from Section 3, Definition 5)
# =============================================================================

@dataclass
class AllocationFunction:
    """
    Regular allocation function ω(x) as defined in Definition 5.
    
    A regular allocation function maps the demand-to-supply ratio x = d/μ
    to the expected allocation per active supplier.
    
    Properties (Definition 5):
        1. ω(·) is smooth, concave, and non-decreasing
        2. lim_{x→0} ω(x) = 0 and lim_{x→∞} ω(x) ≤ 1
        3. lim_{x→0} ω'(x) ≤ 1
    
    Parameters
    ----------
    omega : Callable[[float], float]
        The allocation function ω(x)
    omega_prime : Optional[Callable[[float], float]]
        The derivative ω'(x). If None, computed numerically.
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
        """
        Evaluate ω'(x).
        
        From Equation 3.16, this is used to compute how allocation
        changes with supply:
            (q_a)'(μ) = -ω'(d_a/μ) · d_a/μ²
        """
        if self.omega_prime is not None:
            return self.omega_prime(x)
        else:
            # Numerical derivative
            return numerical_derivative(self.omega, x)


def create_queue_allocation(L: int = 8) -> AllocationFunction:
    """
    Create the M/M/1 queue allocation function from Example 6.
    
    This models parallel finite-capacity queues where each active supplier
    operates as a single-server M/M/1 queue with capacity L.
    
    From Equation 3.5:
        ω(x) = (x - x^L) / (1 - x^L)  if x ≠ 1
        ω(1) = 1 - 1/L                 if x = 1
    
    Parameters
    ----------
    L : int
        Queue capacity (L ≥ 2)
    
    Returns
    -------
    AllocationFunction
        The allocation function for M/M/1 queues
    """
    def omega(x: float) -> float:
        """Allocation function ω(x) from Equation 3.5."""
        if x < 1e-10:
            return x  # Near x=0, ω(x) ≈ x
        elif abs(x - 1.0) < 1e-10:
            return 1.0 - 1.0 / L
        else:
            # Numerically stable computation
            x_L = x ** L
            return (x - x_L) / (1.0 - x_L)
    
    def omega_prime(x: float) -> float:
        """Derivative ω'(x) computed analytically."""
        if x < 1e-10:
            return 1.0  # Near x=0, ω'(x) ≈ 1
        elif abs(x - 1.0) < 1e-10:
            # L'Hôpital's rule at x=1
            return (L - 1) / (2 * L)
        else:
            x_L = x ** L
            x_Lm1 = x ** (L - 1)
            numerator = (1.0 - L * x_Lm1) * (1.0 - x_L) + (x - x_L) * L * x_Lm1
            denominator = (1.0 - x_L) ** 2
            return numerator / denominator
    
    return AllocationFunction(
        omega=omega,
        omega_prime=omega_prime,
        name=f"M/M/1 Queue (L={L})"
    )


def create_simple_allocation() -> AllocationFunction:
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
# CHOICE FUNCTIONS (from Section 3, Assumption 2)
# =============================================================================

@dataclass
class ChoiceFunction:
    """
    Supplier choice function f_b(x) as described in Assumption 2.
    
    The choice function maps expected revenue x to the probability of
    becoming active, given private feature b (e.g., outside option).
    
    Properties (Assumption 2):
        - Takes values in [0, 1]
        - Monotonically non-decreasing
        - Twice differentiable with bounded second derivative
    
    Parameters
    ----------
    f : Callable[[float, float], float]
        f(x, b) returns P(Z=1 | expected_revenue=x, private_feature=b)
    f_prime : Optional[Callable[[float, float], float]]
        Derivative ∂f/∂x(x, b)
    name : str
        Descriptive name
    """
    f: Callable[[float, float], float]
    f_prime: Optional[Callable[[float, float], float]] = None
    name: str = "Generic"
    
    def __call__(self, x: float, b: float) -> float:
        """Evaluate f_b(x) = P(active | expected_revenue=x, private_feature=b)."""
        return self.f(x, b)
    
    def derivative(self, x: float, b: float) -> float:
        """
        Evaluate f'_b(x) = ∂f_b/∂x.
        
        This appears in the marginal response function (Definition 9):
            Δ_a(p) = q_a(μ_a(p)) · E[f'_{B_1}(p · q_a(μ_a(p))) | A=a]
        """
        if self.f_prime is not None:
            return self.f_prime(x, b)
        else:
            return numerical_derivative(lambda xx: self.f(xx, b), x)


def create_logistic_choice(alpha: float = 1.0) -> ChoiceFunction:
    """
    Create logistic choice function from Example 7.
    
    From Equation 3.7:
        P(Z_i = 1 | P_i, π, A) = 1 / (1 + exp(-α(P_i · E[Ω(D,T)|A] - B_i)))
    
    Here, x = expected_revenue and b = B_i (break-even cost threshold).
    
    Parameters
    ----------
    alpha : float
        Sensitivity parameter. As α → ∞, decision becomes deterministic:
        active iff expected_revenue > break_even_cost
    
    Returns
    -------
    ChoiceFunction
        The logistic choice function
    """
    def f(x: float, b: float) -> float:
        """Logistic choice probability."""
        z = alpha * (x - b)
        # Numerically stable sigmoid
        if z >= 0:
            return 1.0 / (1.0 + np.exp(-z))
        else:
            exp_z = np.exp(z)
            return exp_z / (1.0 + exp_z)
    
    def f_prime(x: float, b: float) -> float:
        """Derivative of logistic choice function."""
        prob = f(x, b)
        return alpha * prob * (1.0 - prob)
    
    return ChoiceFunction(
        f=f,
        f_prime=f_prime,
        name=f"Logistic (α={alpha})"
    )


# =============================================================================
# PRIVATE FEATURE DISTRIBUTION
# =============================================================================

@dataclass
class PrivateFeatureDistribution:
    """
    Distribution of private features B_i.
    
    The private features capture heterogeneity across suppliers, such as
    their outside options or break-even cost thresholds.
    
    Parameters
    ----------
    sample : Callable[[int], np.ndarray]
        Function to sample n i.i.d. draws from the distribution
    name : str
        Descriptive name
    """
    sample: Callable[[int], np.ndarray]
    name: str = "Generic"


def create_lognormal_outside_option(
    log_mean: float = 0.0,
    log_std: float = 1.0,
    scale: float = 20.0
) -> PrivateFeatureDistribution:
    """
    Create lognormal distribution for outside options.
    
    From the paper's simulation (Figure 2 caption):
        log(B_i / 20) ~ N(0, 1)
    
    So B_i = 20 * exp(N(0,1)), which is lognormal.
    
    Parameters
    ----------
    log_mean : float
        Mean of log(B_i / scale)
    log_std : float
        Standard deviation of log(B_i / scale)
    scale : float
        Scale factor (20 in the paper)
    """
    def sample(n: int) -> np.ndarray:
        return scale * np.exp(np.random.normal(log_mean, log_std, n))
    
    return PrivateFeatureDistribution(
        sample=sample,
        name=f"LogNormal(μ={log_mean}, σ={log_std}, scale={scale})"
    )


# =============================================================================
# SECTION 3.1: MEAN-FIELD ASYMPTOTICS
# =============================================================================

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


@dataclass
class MarketParameters:
    """
    Parameters defining the marketplace model.
    
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


def compute_expected_choice_derivative(
    revenue: float,
    choice: ChoiceFunction,
    private_features: PrivateFeatureDistribution,
    n_samples: int = 10000
) -> float:
    """
    Compute E[f'_{B_1}(revenue) | A=a] via Monte Carlo.
    
    This appears in the marginal response function (Definition 9).
    
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
        E[f'_{B_1}(revenue)]
    """
    b_samples = private_features.sample(n_samples)
    derivs = np.array([choice.derivative(revenue, b) for b in b_samples])
    return np.mean(derivs)


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
    from .find_equilibrium import solve_equilibrium_supply

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
    params: MarketParameters
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
        params.choice,
        params.private_features,
        params.n_monte_carlo
    )
    
    # Δ = q · E[f'_B(p·q)]
    delta = q * expected_choice_deriv
    
    return delta


def compute_supply_gradient(
    p: float,
    equilibrium: MeanFieldEquilibrium,
    delta: float,
    params: MarketParameters
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
    params : MarketParameters
        Model parameters
    
    Returns
    -------
    float
        Supply gradient dμ_a(p)/dp
    """
    mu = equilibrium.mu
    q = equilibrium.q  # ω(d_a/μ)
    x = equilibrium.demand_supply_ratio  # d_a/μ
    
    # ω'(d_a/μ)
    omega_prime = params.allocation.derivative(x)
    
    # From Equation 3.20:
    # μ' = Δ / (1 + p·d_a·Δ·ω'(x) / (μ²·ω(x)))
    # The denominator is: 1 + interference_term
    
    if abs(q) < 1e-10:  # Avoid division by zero
        return delta
    
    interference_term = (p * params.d_a * delta * omega_prime) / (mu**2 * q)
    
    # Note: omega_prime is typically positive for our allocation functions,
    # and delta is positive, so interference_term > 0
    # This means μ' < Δ: the actual effect is attenuated by interference
    
    mu_prime = delta / (1.0 + interference_term)
    
    return mu_prime


def analyze_marginal_response(
    p: float,
    params: MarketParameters,
    equilibrium: Optional[MeanFieldEquilibrium] = None
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
    params : MarketParameters
        Model parameters
    equilibrium : Optional[MeanFieldEquilibrium]
        Pre-computed equilibrium (computed if not provided)
    
    Returns
    -------
    MarginalResponseAnalysis
        Complete analysis of marginal response and interference
    """
    # Compute equilibrium if not provided
    if equilibrium is None:
        equilibrium = compute_mean_field_equilibrium(p, params)
    
    # Marginal response (Definition 9)
    delta = compute_marginal_response(p, equilibrium, params)
    
    # Decomposition of interference factor (Equation 3.21)
    mu = equilibrium.mu
    q = equilibrium.q
    x = equilibrium.demand_supply_ratio
    omega_prime = params.allocation.derivative(x)
    
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


def compute_utility_gradient(
    p: float,
    params: MarketParameters,
    equilibrium: Optional[MeanFieldEquilibrium] = None,
    marginal_analysis: Optional[MarginalResponseAnalysis] = None
) -> float:
    """
    Compute the utility gradient du_a(p)/dp.
    
    From Proposition 12 in the appendix, for linear revenue r(x) = γ·ω(x):
        u'_a(p) = μ'_a(p) · [r(x) - p·ω(x) - (r'(x) - p·ω'(x))·x] - ω(x)·μ_a(p)
    
    where x = d_a/μ_a(p).
    
    For r(x) = γ·ω(x), this simplifies to:
        u'_a(p) = μ'_a(p) · [(γ-p)·ω(x) - (γ-p)·ω'(x)·x] - ω(x)·μ_a(p)
                = μ'_a(p) · (γ-p) · [ω(x) - ω'(x)·x] - ω(x)·μ_a(p)
    
    Parameters
    ----------
    p : float
        Payment
    params : MarketParameters
        Model parameters
    equilibrium : Optional[MeanFieldEquilibrium]
        Pre-computed equilibrium
    marginal_analysis : Optional[MarginalResponseAnalysis]
        Pre-computed marginal response analysis
    
    Returns
    -------
    float
        Utility gradient du_a(p)/dp
    """
    if equilibrium is None:
        equilibrium = compute_mean_field_equilibrium(p, params)
    
    if marginal_analysis is None:
        marginal_analysis = analyze_marginal_response(p, params, equilibrium)
    
    mu = equilibrium.mu
    q = equilibrium.q  # ω(x)
    x = equilibrium.demand_supply_ratio
    omega_prime = params.allocation.derivative(x)
    mu_prime = marginal_analysis.mu_prime
    
    # From Proposition 12 with r(x) = γ·ω(x):
    # Term 1: μ' · (γ-p) · [ω(x) - ω'(x)·x]
    # Term 2: -ω(x) · μ
    
    bracket_term = q - omega_prime * x  # ω(x) - ω'(x)·x
    
    u_prime = mu_prime * (params.gamma - p) * bracket_term - q * mu
    
    return u_prime


# =============================================================================
# CONVENIENCE FUNCTIONS FOR ANALYSIS
# =============================================================================

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
        private_features=create_lognormal_outside_option(
            log_mean=0.0,
            log_std=1.0,
            scale=20.0
        ),
        d_a=0.4,
        gamma=100.0,
        n_monte_carlo=10000
    )


def analyze_payment_range(
    params: MarketParameters,
    p_min: float = 5.0,
    p_max: float = 60.0,
    n_points: int = 20
) -> dict:
    """
    Analyze equilibrium and marginal response across a range of payments.
    
    Useful for understanding how the market behaves and for finding
    the optimal payment.
    
    Parameters
    ----------
    params : MarketParameters
        Model parameters
    p_min, p_max : float
        Payment range to analyze
    n_points : int
        Number of points to evaluate
    
    Returns
    -------
    dict
        Dictionary containing arrays of:
        - payments
        - equilibrium supply (mu)
        - allocation rates (q)
        - utilities (u)
        - marginal responses (delta)
        - supply gradients (mu_prime)
        - utility gradients (u_prime)
        - interference factors
    """
    payments = np.linspace(p_min, p_max, n_points)
    
    results = {
        'payments': payments,
        'mu': np.zeros(n_points),
        'q': np.zeros(n_points),
        'u': np.zeros(n_points),
        'delta': np.zeros(n_points),
        'mu_prime': np.zeros(n_points),
        'u_prime': np.zeros(n_points),
        'interference_factor': np.zeros(n_points)
    }
    
    for i, p in enumerate(payments):
        eq = compute_mean_field_equilibrium(p, params)
        analysis = analyze_marginal_response(p, params, eq)
        u_prime = compute_utility_gradient(p, params, eq, analysis)
        
        results['mu'][i] = eq.mu
        results['q'][i] = eq.q
        results['u'][i] = eq.u
        results['delta'][i] = analysis.delta
        results['mu_prime'][i] = analysis.mu_prime
        results['u_prime'][i] = u_prime
        results['interference_factor'][i] = analysis.interference_factor
    
    return results


if __name__ == "__main__":
    # Quick demonstration
    print("=" * 60)
    print("Mean-Field Equilibrium Analysis")
    print("Sections 3.1 and 3.2 of Wager & Xu (2021)")
    print("=" * 60)
    
    # Create default parameters
    params = create_default_market_params()
    print(f"\nModel Parameters:")
    print(f"  Allocation: {params.allocation.name}")
    print(f"  Choice: {params.choice.name}")
    print(f"  Demand (d_a): {params.d_a}")
    print(f"  Revenue (γ): {params.gamma}")
    
    # Analyze a specific payment
    p = 30.0
    print(f"\n--- Analysis for p = {p} ---")
    
    eq = compute_mean_field_equilibrium(p, params)
    print(f"\nMean-Field Equilibrium (Lemma 2):")
    print(f"  μ_a(p) = {eq.mu:.4f}  (supply fraction)")
    print(f"  q_a(μ) = {eq.q:.4f}  (allocation rate)")
    print(f"  u_a(p) = {eq.u:.4f}  (utility)")
    print(f"  d_a/μ  = {eq.demand_supply_ratio:.4f}  (demand/supply ratio)")
    
    analysis = analyze_marginal_response(p, params, eq)
    print(f"\nMarginal Response Analysis (Section 3.2):")
    print(f"  Δ_a(p)   = {analysis.delta:.6f}  (marginal response)")
    print(f"  μ'_a(p)  = {analysis.mu_prime:.6f}  (actual gradient)")
    print(f"  1+R_a(p) = {analysis.interference_factor:.4f}  (interference factor)")
    print(f"  Σ^Δ_a(p) = {analysis.sigma_delta:.4f}  (scaled marginal sensitivity)")
    print(f"  Σ^Ω_a(p) = {analysis.sigma_omega:.4f}  (scaled matching elasticity)")
    
    u_prime = compute_utility_gradient(p, params, eq, analysis)
    print(f"\nUtility Gradient:")
    print(f"  u'_a(p) = {u_prime:.6f}")
    
    print("\n" + "=" * 60)