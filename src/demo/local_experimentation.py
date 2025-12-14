"""
Learning via Local Experimentation

This module implements Section 4 of Wager & Xu (2021)
"Experimenting in Equilibrium".

Section 4 covers:
- 4.1: Estimating Utility Gradients (Theorem 6)
- 4.2: A First-Order Algorithm (Theorem 7, Corollary 8)
- 4.3: The Cost of Experimentation (Theorem 9)
- 4.4: Comparison with Global Experimentation

The key insight is that local experimentation with symmetric payment
perturbations P_i = p + ζε_i allows us to estimate utility gradients
without disturbing the market equilibrium, enabling gradient-based
optimization with vanishingly small perturbations as n → ∞.

References:
    Wager, S. & Xu, K. (2021). "Experimenting in Equilibrium"
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List, Callable, Union

from .allocation import AllocationFunction
from .supplier import (
    ChoiceFunction, 
    PrivateFeatureDistribution,
    SupplierParameters,
    sample_supplier_activations,
    compute_expected_choice_probability
)
from .demand import (
    DemandParameters,
    GlobalState,
    sample_state,
    sample_demand
)


# =============================================================================
# SECTION 4.1: ESTIMATING UTILITY GRADIENTS
# =============================================================================

@dataclass
class LocalExperimentData:
    """
    Data from one period of local experimentation.
    
    In each period, we:
    1. Apply payment perturbations P_i = p + ζε_i
    2. Observe supplier activations Z_i
    3. Observe demand D and total active suppliers T
    
    Attributes
    ----------
    n : int
        Number of potential suppliers
    p : float
        Base payment level
    zeta : float
        Perturbation magnitude
    epsilon : np.ndarray
        Payment perturbations ε_i ∈ {-1, +1}
    Z : np.ndarray
        Supplier activation decisions Z_i ∈ {0, 1}
    D : int
        Total demand
    T : int
        Number of active suppliers (sum of Z)
    """
    n: int
    p: float
    zeta: float
    epsilon: np.ndarray  # Shape (n,), values in {-1, +1}
    Z: np.ndarray        # Shape (n,), values in {0, 1}
    D: int
    T: int
    
    @property
    def D_bar(self) -> float:
        """Scaled demand D̄ = D/n"""
        return self.D / self.n
    
    @property
    def Z_bar(self) -> float:
        """Scaled active supply Z̄ = T/n"""
        return self.T / self.n


def generate_payment_perturbations(
    n: int,
    rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """
    Generate symmetric payment perturbations ε_i ∈ {-1, +1}.
    
    From equation (2.1)/(3.11):
        P_i = p + ζε_i, where ε_i ∼ {±1} uniformly
    
    Parameters
    ----------
    n : int
        Number of suppliers
    rng : Optional[np.random.Generator]
        Random number generator
        
    Returns
    -------
    np.ndarray
        Array of ε_i values, each ±1
    """
    if rng is None:
        rng = np.random.default_rng()
    return rng.choice([-1, 1], size=n)


def run_local_experiment(
    n: int,
    p: float,
    zeta: float,
    expected_allocation: float,
    supplier_params: SupplierParameters,
    D: Optional[int] = None,
    d_a: Optional[float] = None,
    demand_params: Optional[DemandParameters] = None,
    state: Optional[GlobalState] = None,
    rng: Optional[np.random.Generator] = None
) -> LocalExperimentData:
    """
    Run one period of local experimentation.
    
    This implements the data collection step of Section 4.1:
    1. Generate ε_i ∼ {±1} uniformly for each supplier
    2. Set payments P_i = p + ζε_i
    3. Suppliers decide to become active based on expected revenue
    4. Observe Z_i, D, T
    
    Parameters
    ----------
    n : int
        Number of potential suppliers
    p : float
        Base payment level
    zeta : float
        Perturbation magnitude (should scale as n^(-α) for 0 < α < 0.5)
    expected_allocation : float
        Expected allocation q per active supplier at equilibrium
    supplier_params : SupplierParameters
        Supplier choice parameters
    D : Optional[int]
        Realized demand (if already sampled externally)
    d_a : Optional[float]
        Expected demand per supplier (used if D not provided)
    demand_params : Optional[DemandParameters]
        Demand model parameters (used to sample D if not provided directly)
    state : Optional[GlobalState]
        Current global state (used with demand_params to sample D)
    rng : Optional[np.random.Generator]
        Random number generator
        
    Returns
    -------
    LocalExperimentData
        Data from this experimental period
        
    Notes
    -----
    Demand D is determined by (in order of priority):
    1. If D is provided directly, use it
    2. If demand_params and state provided, sample from demand model
    3. If d_a provided, use deterministic D = round(n * d_a)
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # Step 1: Generate perturbations ε_i ∈ {-1, +1}
    epsilon = rng.choice([-1, 1], size=n)
    
    # Step 2: Set payments
    payments = p + zeta * epsilon
    
    # Step 3: Suppliers make activation decisions
    # Note: sample_supplier_activations uses np.random internally, 
    # so we set the seed from our rng for consistency
    internal_seed = int(rng.integers(0, 2**31))
    Z = sample_supplier_activations(
        n=n,
        payments=payments,
        expected_allocation=expected_allocation,
        params=supplier_params,
        seed=internal_seed
    )
    
    # Step 4: Determine demand D
    if D is not None:
        # Use provided demand
        pass
    elif demand_params is not None and state is not None:
        # Sample from demand model
        D = sample_demand(demand_params, state, n, rng)
    elif d_a is not None:
        # Use deterministic approximation
        D = int(round(n * d_a))
    else:
        raise ValueError("Must provide D, (demand_params + state), or d_a")
    
    T = int(Z.sum())
    
    return LocalExperimentData(
        n=n,
        p=p,
        zeta=zeta,
        epsilon=epsilon,
        Z=Z,
        D=D,
        T=T
    )


def estimate_delta_hat(data: LocalExperimentData) -> float:
    """
    Estimate the marginal response function Δ̂ from local experiment data.
    
    From equation (4.1):
        Δ̂ = ζ⁻¹ * Cov(Z_i, ε_i) / Var(ε_i)
        
    This is the scaled regression coefficient of Z_i on ε_i.
    
    Since Var(ε_i) = 1 (ε_i ∈ {±1} uniformly), this simplifies to:
        Δ̂ = ζ⁻¹ * Cov(Z_i, ε_i)
    
    Parameters
    ----------
    data : LocalExperimentData
        Data from local experiment
        
    Returns
    -------
    float
        Estimated marginal response Δ̂
    """
    # Compute sample covariance
    Z_centered = data.Z - np.mean(data.Z)
    epsilon_centered = data.epsilon - np.mean(data.epsilon)
    
    # Cov(Z, ε) = (1/n) * Σ(Z_i - Z̄)(ε_i - ε̄)
    cov_Z_epsilon = np.mean(Z_centered * epsilon_centered)
    
    # Var(ε) = (1/n) * Σ(ε_i - ε̄)²
    var_epsilon = np.mean(epsilon_centered ** 2)
    
    # Δ̂ = ζ⁻¹ * Cov(Z, ε) / Var(ε)
    if var_epsilon < 1e-10:
        return 0.0
    
    delta_hat = cov_Z_epsilon / (data.zeta * var_epsilon)
    
    return delta_hat


def estimate_upsilon_hat(
    delta_hat: float,
    D_bar: float,
    Z_bar: float,
    p: float,
    allocation: AllocationFunction
) -> float:
    """
    Estimate the supply gradient Υ̂ ≈ μ'(p).
    
    From equation (4.2):
        Υ̂ = Δ̂ / (1 + p·D̄·Δ̂·ω'(D̄/Z̄) / (Z̄²·ω(D̄/Z̄)))
    
    This transforms the marginal response Δ̂ into an estimate of μ'(p)
    by accounting for the interference attenuation factor.
    
    Parameters
    ----------
    delta_hat : float
        Estimated marginal response from estimate_delta_hat
    D_bar : float
        Scaled demand D̄ = D/n
    Z_bar : float
        Scaled active supply Z̄ = T/n
    p : float
        Payment level
    allocation : AllocationFunction
        The allocation function ω
        
    Returns
    -------
    float
        Estimated supply gradient Υ̂ ≈ dμ/dp
    """
    if Z_bar < 1e-10:
        return delta_hat
    
    x = D_bar / Z_bar  # Demand per active supplier
    omega = allocation(x)
    omega_prime = allocation.derivative(x)
    
    if abs(omega) < 1e-10:
        return delta_hat
    
    # Interference term from denominator of (4.2)
    interference_term = (p * D_bar * delta_hat * omega_prime) / (Z_bar**2 * omega)
    
    upsilon_hat = delta_hat / (1.0 + interference_term)
    
    return upsilon_hat


def estimate_gamma_hat(
    upsilon_hat: float,
    D_bar: float,
    Z_bar: float,
    p: float,
    allocation: AllocationFunction,
    revenue_fn: Callable[[float], float],
    revenue_fn_prime: Callable[[float], float]
) -> float:
    """
    Estimate the utility gradient Γ̂ ≈ du(p)/dp.
    
    From equation (4.3):
        Γ̂ = Υ̂ · [r(D̄/Z̄) - p·ω(D̄/Z̄) - (r'(D̄/Z̄) - p·ω'(D̄/Z̄))·D̄/Z̄] - ω(D̄/Z̄)·Z̄
    
    This is the complete gradient estimator from Theorem 6.
    
    Parameters
    ----------
    upsilon_hat : float
        Estimated supply gradient from estimate_upsilon_hat
    D_bar : float
        Scaled demand D̄ = D/n
    Z_bar : float
        Scaled active supply Z̄ = T/n
    p : float
        Payment level
    allocation : AllocationFunction
        The allocation function ω
    revenue_fn : Callable[[float], float]
        Platform revenue function r(x)
    revenue_fn_prime : Callable[[float], float]
        Derivative r'(x)
        
    Returns
    -------
    float
        Estimated utility gradient Γ̂ ≈ du/dp
    """
    if Z_bar < 1e-10:
        return 0.0
    
    x = D_bar / Z_bar
    
    r_x = revenue_fn(x)
    r_prime_x = revenue_fn_prime(x)
    omega_x = allocation(x)
    omega_prime_x = allocation.derivative(x)
    
    # Bracket term in (4.3)
    bracket = r_x - p * omega_x - (r_prime_x - p * omega_prime_x) * x
    
    # Full gradient estimate
    gamma_hat = upsilon_hat * bracket - omega_x * Z_bar
    
    return gamma_hat


@dataclass
class GradientEstimate:
    """
    Complete gradient estimate from one period of local experimentation.
    
    Attributes
    ----------
    delta_hat : float
        Estimated marginal response Δ̂ (equation 4.1)
    upsilon_hat : float
        Estimated supply gradient Υ̂ ≈ μ' (equation 4.2)
    gamma_hat : float
        Estimated utility gradient Γ̂ ≈ du/dp (equation 4.3)
    D_bar : float
        Observed scaled demand
    Z_bar : float
        Observed scaled active supply
    """
    delta_hat: float
    upsilon_hat: float
    gamma_hat: float
    D_bar: float
    Z_bar: float


def estimate_utility_gradient(
    data: LocalExperimentData,
    allocation: AllocationFunction,
    gamma: float
) -> GradientEstimate:
    """
    Complete gradient estimation from local experiment data.
    
    This implements the full estimation procedure from Section 4.1
    (Theorem 6), combining equations (4.1), (4.2), and (4.3).
    
    For linear revenue r(x) = γ·ω(x), we have r'(x) = γ·ω'(x).
    
    Parameters
    ----------
    data : LocalExperimentData
        Data from one period of local experimentation
    allocation : AllocationFunction
        The allocation function ω
    gamma : float
        Platform revenue per unit of demand served
        
    Returns
    -------
    GradientEstimate
        Complete gradient estimates
    """
    # Step 1: Estimate Δ̂ (equation 4.1)
    delta_hat = estimate_delta_hat(data)
    
    # Step 2: Estimate Υ̂ (equation 4.2)
    upsilon_hat = estimate_upsilon_hat(
        delta_hat=delta_hat,
        D_bar=data.D_bar,
        Z_bar=data.Z_bar,
        p=data.p,
        allocation=allocation
    )
    
    # Step 3: Estimate Γ̂ (equation 4.3)
    # For linear revenue: r(x) = γ·ω(x), r'(x) = γ·ω'(x)
    def revenue_fn(x: float) -> float:
        return gamma * allocation(x)
    
    def revenue_fn_prime(x: float) -> float:
        return gamma * allocation.derivative(x)
    
    gamma_hat = estimate_gamma_hat(
        upsilon_hat=upsilon_hat,
        D_bar=data.D_bar,
        Z_bar=data.Z_bar,
        p=data.p,
        allocation=allocation,
        revenue_fn=revenue_fn,
        revenue_fn_prime=revenue_fn_prime
    )
    
    return GradientEstimate(
        delta_hat=delta_hat,
        upsilon_hat=upsilon_hat,
        gamma_hat=gamma_hat,
        D_bar=data.D_bar,
        Z_bar=data.Z_bar
    )


# =============================================================================
# SECTION 4.2: A FIRST-ORDER ALGORITHM
# =============================================================================

@dataclass
class OptimizationState:
    """
    State of the first-order optimization algorithm.
    
    Attributes
    ----------
    t : int
        Current time period (1-indexed)
    p : float
        Current payment level
    theta : float
        Accumulated weighted gradient sum
    gradient_history : List[float]
        History of gradient estimates Γ̂_t
    payment_history : List[float]
        History of payment levels p_t
    """
    t: int
    p: float
    theta: float
    gradient_history: List[float]
    payment_history: List[float]


def initialize_optimizer(
    p_init: float,
    p_bounds: Tuple[float, float] = (0.0, float('inf'))
) -> OptimizationState:
    """
    Initialize the first-order optimization algorithm.
    
    Parameters
    ----------
    p_init : float
        Initial payment level p_1
    p_bounds : Tuple[float, float]
        Payment bounds [c_-, c_+] (interval I)
        
    Returns
    -------
    OptimizationState
        Initial optimization state
    """
    return OptimizationState(
        t=1,
        p=np.clip(p_init, p_bounds[0], p_bounds[1]),
        theta=0.0,
        gradient_history=[],
        payment_history=[p_init]
    )


def mirror_descent_update(
    state: OptimizationState,
    gradient: float,
    eta: float,
    p_bounds: Tuple[float, float] = (0.0, float('inf'))
) -> OptimizationState:
    """
    Perform one step of the mirror descent update.
    
    From equation (4.5):
        p_{t+1} = argmin_p { (1/2η) Σ_{s=1}^t s(p - p_s)² - θ_t·p : p ∈ I }
        where θ_t = Σ_{s=1}^t s·Γ̂_s
    
    For the unconstrained case (I = ℝ), this reduces to:
        p_{t+1} = p_t + (2η·Γ̂_t) / (t+1)
    
    For the constrained case, we project onto the interval.
    
    Parameters
    ----------
    state : OptimizationState
        Current optimization state
    gradient : float
        Gradient estimate Γ̂_t from current period
    eta : float
        Step size η (should satisfy η > σ⁻¹ where σ is strong concavity)
    p_bounds : Tuple[float, float]
        Payment bounds [c_-, c_+]
        
    Returns
    -------
    OptimizationState
        Updated optimization state
    """
    t = state.t
    c_minus, c_plus = p_bounds
    
    # Update θ_t = Σ_{s=1}^t s·Γ̂_s
    theta_new = state.theta + t * gradient
    
    # For the mirror descent update with quadratic regularizer,
    # the solution to the optimization problem in (4.5) is:
    # 
    # Without constraint: weighted average of past payments plus gradient term
    # p_{t+1} = (Σ_{s=1}^t s·p_s + η·θ_t) / (Σ_{s=1}^t s)
    #         = (Σ_{s=1}^t s·p_s) / (t(t+1)/2) + η·θ_t / (t(t+1)/2)
    #
    # Simplified form (equivalent to basic gradient descent):
    # p_{t+1} = p_t + 2η·Γ̂_t / (t+1)
    
    p_new = state.p + (2 * eta * gradient) / (t + 1)
    
    # Project onto interval I = [c_-, c_+]
    p_new = np.clip(p_new, c_minus, c_plus)
    
    # Update histories
    gradient_history = state.gradient_history + [gradient]
    payment_history = state.payment_history + [p_new]
    
    return OptimizationState(
        t=t + 1,
        p=p_new,
        theta=theta_new,
        gradient_history=gradient_history,
        payment_history=payment_history
    )


@dataclass
class LearningResult:
    """
    Results from running the learning algorithm.
    
    Attributes
    ----------
    final_payment : float
        Final payment level after T periods
    weighted_average_payment : float
        Weighted average p̄_T = (2/(T(T+1))) Σ_{t=1}^T t·p_t (Corollary 8)
    payment_history : List[float]
        Sequence of payments p_1, ..., p_T
    gradient_history : List[float]
        Sequence of gradient estimates Γ̂_1, ..., Γ̂_T
    utility_history : List[float]
        Estimated utilities (if tracked)
    state_history : Optional[List[GlobalState]]
        Sequence of global states A_1, ..., A_T (if using DemandParameters)
    """
    final_payment: float
    weighted_average_payment: float
    payment_history: List[float]
    gradient_history: List[float]
    utility_history: List[float]
    state_history: Optional[List[GlobalState]] = None


def run_learning_algorithm(
    T: int,
    n: int,
    p_init: float,
    eta: float,
    zeta: float,
    gamma: float,
    allocation: AllocationFunction,
    supplier_params: SupplierParameters,
    d_a: Optional[float] = None,
    demand_params: Optional[DemandParameters] = None,
    p_bounds: Tuple[float, float] = (0.0, float('inf')),
    rng: Optional[np.random.Generator] = None,
    verbose: bool = False
) -> LearningResult:
    """
    Run the complete learning algorithm from Section 4.2.
    
    This implements the first-order optimization algorithm:
    
    For t = 1, ..., T:
        1. Sample global state A_t (if using demand_params)
        2. Deploy payment perturbations around p_t
        3. Estimate gradient Γ̂_t via local experimentation
        4. Update p_{t+1} via mirror descent (equation 4.5)
    
    Parameters
    ----------
    T : int
        Number of time periods (days)
    n : int
        Number of suppliers per period
    p_init : float
        Initial payment p_1
    eta : float
        Step size η (should satisfy η > σ⁻¹)
    zeta : float
        Perturbation magnitude ζ (should scale as n^(-α) for 0 < α < 0.5)
    gamma : float
        Platform revenue per unit served
    allocation : AllocationFunction
        The allocation function ω
    supplier_params : SupplierParameters
        Supplier behavior parameters
    d_a : Optional[float]
        Fixed expected demand per supplier (simple case)
    demand_params : Optional[DemandParameters]
        Demand model with multiple states (full model case)
        If provided, state is sampled each period per the paper
    p_bounds : Tuple[float, float]
        Payment bounds [c_-, c_+]
    rng : Optional[np.random.Generator]
        Random number generator for reproducibility
    verbose : bool
        Whether to print progress
        
    Returns
    -------
    LearningResult
        Learning outcomes including payment and state trajectories
        
    Notes
    -----
    Either d_a or demand_params must be provided. If demand_params is given,
    the global state A_t is sampled each period according to the paper's model.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    if d_a is None and demand_params is None:
        raise ValueError("Must provide either d_a or demand_params")
    
    # Initialize optimizer
    opt_state = initialize_optimizer(p_init, p_bounds)
    utility_history = []
    state_history = [] if demand_params is not None else None
    
    # Import here to avoid circular imports
    from .find_equilibrium import find_equilibrium_supply_mu
    
    for t in range(1, T + 1):
        if verbose and t % 20 == 0:
            print(f"  Period {t}/{T}: p = {opt_state.p:.4f}")
        
        # Sample global state A_t for this period
        if demand_params is not None:
            current_state = sample_state(demand_params, rng)
            current_d_a = current_state.d_a
            state_history.append(current_state)
        else:
            current_state = None
            current_d_a = d_a
        
        # Compute expected allocation at current payment for this state
        mu_eq = find_equilibrium_supply_mu(
            p=opt_state.p,
            d_a=current_d_a,
            choice=supplier_params.choice,
            private_features=supplier_params.private_features,
            allocation=allocation
        )
        q_eq = allocation(current_d_a / mu_eq) if mu_eq > 0 else 0.0
        
        # Run local experiment for this period
        data = run_local_experiment(
            n=n,
            p=opt_state.p,
            zeta=zeta,
            expected_allocation=q_eq,
            supplier_params=supplier_params,
            d_a=current_d_a,
            demand_params=demand_params,
            state=current_state,
            rng=rng
        )
        
        # Estimate gradient
        grad_est = estimate_utility_gradient(data, allocation, gamma)
        
        # Track estimated utility
        x = data.D_bar / data.Z_bar if data.Z_bar > 0 else 0
        est_utility = (gamma - opt_state.p) * allocation(x) * data.Z_bar if data.Z_bar > 0 else 0
        utility_history.append(est_utility)
        
        # Update payment via mirror descent
        opt_state = mirror_descent_update(
            state=opt_state,
            gradient=grad_est.gamma_hat,
            eta=eta,
            p_bounds=p_bounds
        )
    
    # Compute weighted average (Corollary 8)
    weights = np.arange(1, T + 1)
    weight_sum = T * (T + 1) / 2
    weighted_avg = np.sum(weights * np.array(opt_state.payment_history[:-1])) / weight_sum
    
    return LearningResult(
        final_payment=opt_state.p,
        weighted_average_payment=weighted_avg,
        payment_history=opt_state.payment_history,
        gradient_history=opt_state.gradient_history,
        utility_history=utility_history,
        state_history=state_history
    )


# =============================================================================
# SECTION 4.3: COST OF EXPERIMENTATION
# =============================================================================

def compute_experimentation_cost(
    p: float,
    zeta: float,
    d_a: float,
    mu: float,
    allocation: AllocationFunction
) -> float:
    """
    Compute the cost of experimentation (Theorem 9).
    
    The excess cost from randomization is O(ζ²) as shown in Theorem 9.
    This is because:
    - Symmetric perturbations don't shift equilibrium to first order
    - Suppliers with higher payments are more likely to activate
    - This increases average payment without increasing demand served
    
    Parameters
    ----------
    p : float
        Base payment level
    zeta : float
        Perturbation magnitude
    d_a : float
        Expected demand per supplier
    mu : float
        Equilibrium supply fraction
    allocation : AllocationFunction
        The allocation function
        
    Returns
    -------
    float
        Approximate cost of experimentation (O(ζ²) term)
    """
    # The cost is approximately proportional to ζ² times a constant
    # depending on the curvature of the choice function
    # (Theorem 9: u(p) - u(p, ζ) ≤ C·ζ²)
    
    # A rough bound based on the interference structure
    q = allocation(d_a / mu) if mu > 0 else 0
    
    # The cost is dominated by correlation between higher payments
    # and higher activation probabilities
    cost_bound = 0.5 * zeta**2 * q * mu
    
    return cost_bound


# =============================================================================
# UTILITY FUNCTIONS FOR ANALYSIS
# =============================================================================

def compute_optimal_zeta(n: int, alpha: float = 0.3) -> float:
    """
    Compute optimal perturbation magnitude ζ_n.
    
    From Theorem 6, perturbations should scale as ζ_n = ζ·n^(-α)
    for 0 < α < 0.5 to ensure consistency.
    
    Parameters
    ----------
    n : int
        Market size
    alpha : float
        Decay exponent (0 < α < 0.5)
        
    Returns
    -------
    float
        Optimal perturbation magnitude
    """
    base_zeta = 0.5  # Base perturbation (can be tuned)
    return base_zeta * (n ** (-alpha))


def analyze_convergence(
    result: LearningResult,
    p_optimal: float
) -> dict:
    """
    Analyze convergence of the learning algorithm.
    
    Parameters
    ----------
    result : LearningResult
        Output from run_learning_algorithm
    p_optimal : float
        True optimal payment (for comparison)
        
    Returns
    -------
    dict
        Convergence statistics
    """
    payments = np.array(result.payment_history)
    T = len(payments) - 1
    
    # Compute regret trajectory
    regret = (p_optimal - payments[:-1])**2
    
    # Compute weighted regret (as in Theorem 7)
    weights = np.arange(1, T + 1)
    weighted_regret = np.sum(weights * regret) / (T * (T + 1) / 2)
    
    return {
        'final_payment': result.final_payment,
        'weighted_average': result.weighted_average_payment,
        'optimal_payment': p_optimal,
        'final_error': abs(result.final_payment - p_optimal),
        'weighted_error': abs(result.weighted_average_payment - p_optimal),
        'mean_squared_error': np.mean(regret),
        'weighted_regret': weighted_regret
    }