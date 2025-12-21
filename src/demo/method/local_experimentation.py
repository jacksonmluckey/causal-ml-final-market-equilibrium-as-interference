r"""
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
from typing import Optional, Tuple, List, Callable, overload

from .allocation import AllocationFunction
from .revenue import RevenueFunction
from .platform_utility import compute_realized_utility
from .supplier import (
    SupplierParameters,
    sample_supplier_activations,
)
from .demand import DemandParameters, GlobalState, sample_demand
from .experiment_results import TimePointData, Experiment
from .experiment import (
    ExperimentParams,
    setup_rng,
    extract_demand_from_params,
    sample_current_state,
    compute_equilibrium_allocation,
    compute_weighted_average_payment,
    build_experiment_results,
)


# =============================================================================
# SECTION 4.1: ESTIMATING UTILITY GRADIENTS
# =============================================================================
# Note: LocalExperimentData has been replaced by TimePointData from experiment_results.py


def generate_payment_perturbations(
    n: int, rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    r"""
    Generate symmetric payment perturbations $\varepsilon_i \in \{-1, +1\}$.

    From equation (2.1)/(3.11):
        $P_i = p + \zeta \varepsilon_i$, where $\varepsilon_i \sim \{±1\}$ uniformly

    Parameters
    ----------
    n : int
        Number of suppliers
    rng : Optional[np.random.Generator]
        Random number generator

    Returns
    -------
    np.ndarray
        Array of $\varepsilon_i$ values, each $\pm 1$
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
    revenue_fn: RevenueFunction,
    allocation: AllocationFunction,
    t: int = 0,
    D: Optional[int] = None,
    d_a: Optional[float] = None,
    demand_params: Optional[DemandParameters] = None,
    state: Optional[GlobalState] = None,
    store_detailed_data: bool = False,
    rng: Optional[np.random.Generator] = None,
) -> TimePointData:
    r"""
    Run one period of local experimentation.

    This implements the data collection step of Section 4.1:
    1. Generate $\varepsilon_i \sim \{±1\}$ uniformly for each supplier
    2. Set payments $P_i = p + \zeta \varepsilon_i$
    3. Suppliers decide to become active based on expected revenue
    4. Observe $Z_i$, D, T
    5. Compute S (demand served) and U (utility)

    Parameters
    ----------
    n : int
        Number of potential suppliers
    p : float
        Base payment level
    zeta : float
        Perturbation magnitude (should scale as $n^{-\alpha}$ for $0 < \alpha < 0.5$)
    expected_allocation : float
        Expected allocation q per active supplier at equilibrium
    supplier_params : SupplierParameters
        Supplier choice parameters
    revenue_fn : RevenueFunction
        Platform revenue function
    allocation : AllocationFunction
        The allocation function ω
    t : int
        Time period (1-indexed)
    D : Optional[int]
        Realized demand (if already sampled externally)
    d_a : Optional[float]
        Expected demand per supplier (used if D not provided)
    demand_params : Optional[DemandParameters]
        Demand model parameters (used to sample D if not provided directly)
    state : Optional[GlobalState]
        Current global state (used with demand_params to sample D)
    store_detailed_data : bool
        Whether to store individual-level arrays (Z, epsilon)
    rng : Optional[np.random.Generator]
        Random number generator

    Returns
    -------
    TimePointData
        Data from this experimental period (gradient estimates set to None,
        to be filled in by calling code)

    Notes
    -----
    Demand D is determined by (in order of priority):
    1. If D is provided directly, use it
    2. If demand_params and state provided, sample from demand model
    3. If d_a provided, use deterministic D = round(n * d_a)
    """
    if rng is None:
        rng = np.random.default_rng()

    # Step 1: Generate perturbations $\varepsilon_i \in \{-1, +1\}$
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
        seed=internal_seed,
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

    # Step 5: Compute demand served and utility
    if T > 0:
        x = D / T  # Demand per active supplier
        S = T * allocation(x)
        U = compute_realized_utility(D, T, p, revenue_fn, allocation, n)
    else:
        S = 0.0
        U = 0.0

    return TimePointData(
        t=t,
        p=p,
        D=D,
        T=T,
        S=S,
        U=U,
        state=state,
        gradient_estimate=None,  # To be filled in by caller
        delta_hat=None,
        upsilon_hat=None,
        zeta=zeta,
        epsilon=epsilon if store_detailed_data else None,
        Z=Z if store_detailed_data else None,
    )


def estimate_delta_hat(data: TimePointData, n: int) -> float:
    r"""
    Estimate the marginal response function $\hat{\Delta}$ from local experiment data.

    From equation (4.1):
        $\hat{\Delta} = \zeta^{-1} \cdot \text{Cov}(Z_i, \varepsilon_i) / \text{Var}(\varepsilon_i)$

    This is the scaled regression coefficient of $Z_i$ on $\varepsilon_i$.

    Since $\text{Var}(\varepsilon_i) = 1$ ($\varepsilon_i \in \{±1\}$ uniformly), this simplifies to:
        $\hat{\Delta} = \zeta^{-1} \cdot \text{Cov}(Z_i, \varepsilon_i)$

    Parameters
    ----------
    data : TimePointData
        Data from local experiment (must have Z and epsilon arrays)
    n : int
        Number of suppliers

    Returns
    -------
    float
        Estimated marginal response $\hat{\Delta}$
    """
    if data.Z is None or data.epsilon is None:
        raise ValueError("Cannot estimate delta_hat without detailed data (Z, epsilon)")

    # Compute sample covariance
    Z_centered = data.Z - np.mean(data.Z)
    epsilon_centered = data.epsilon - np.mean(data.epsilon)

    # $\text{Cov}(Z, \varepsilon) = (1/n) \cdot \sum(Z_i - \bar{Z})(\varepsilon_i - \bar{\varepsilon})$
    cov_Z_epsilon = np.mean(Z_centered * epsilon_centered)

    # $\text{Var}(\varepsilon) = (1/n) \cdot \sum(\varepsilon_i - \bar{\varepsilon})^2$
    var_epsilon = np.mean(epsilon_centered**2)

    # $\hat{\Delta} = \zeta^{-1} \cdot \text{Cov}(Z, \varepsilon) / \text{Var}(\varepsilon)$
    if var_epsilon < 1e-10:
        return 0.0

    delta_hat = cov_Z_epsilon / (data.zeta * var_epsilon)

    return delta_hat


def estimate_upsilon_hat(
    delta_hat: float, data: TimePointData, n: int, allocation: AllocationFunction
) -> float:
    r"""
    Estimate the supply gradient $\hat{\Upsilon} \approx \mu'(p)$.

    From equation (4.2):
        $\hat{\Upsilon} = \hat{\Delta} / (1 + p \cdot \bar{D} \cdot \hat{\Delta} \cdot \omega'(\bar{D}/\bar{Z}) / (\bar{Z}^2 \cdot \omega(\bar{D}/\bar{Z})))$

    This transforms the marginal response $\hat{\Delta}$ into an estimate of $\mu'(p)$
    by accounting for the interference attenuation factor.

    Parameters
    ----------
    delta_hat : float
        Estimated marginal response from estimate_delta_hat
    data : TimePointData
        Data from local experiment
    n : int
        Number of suppliers
    allocation : AllocationFunction
        The allocation function $\omega$

    Returns
    -------
    float
        Estimated supply gradient $\hat{\Upsilon} \approx d\mu/dp$
    """
    D_bar = data.D / n
    Z_bar = data.T / n

    if Z_bar < 1e-10:
        return delta_hat

    x = D_bar / Z_bar  # Demand per active supplier
    omega = allocation(x)
    omega_prime = allocation.derivative(x)

    if abs(omega) < 1e-10:
        return delta_hat

    # Interference term from denominator of (4.2)
    interference_term = (data.p * D_bar * delta_hat * omega_prime) / (Z_bar**2 * omega)

    upsilon_hat = delta_hat / (1.0 + interference_term)

    return upsilon_hat


def estimate_gamma_hat(
    upsilon_hat: float,
    data: TimePointData,
    n: int,
    allocation: AllocationFunction,
    revenue_fn: Callable[[float], float],
    revenue_fn_prime: Callable[[float], float],
) -> float:
    r"""
    Estimate the utility gradient $\hat{\Gamma} \approx du(p)/dp$.

    From equation (4.3):
        $\hat{\Gamma} = \hat{\Upsilon} \cdot [r(\bar{D}/\bar{Z}) - p \cdot \omega(\bar{D}/\bar{Z}) - (r'(\bar{D}/\bar{Z}) - p \cdot \omega'(\bar{D}/\bar{Z})) \cdot \bar{D}/\bar{Z}] - \omega(\bar{D}/\bar{Z}) \cdot \bar{Z}$

    This is the complete gradient estimator from Theorem 6.

    Parameters
    ----------
    upsilon_hat : float
        Estimated supply gradient from estimate_upsilon_hat
    data : TimePointData
        Data from local experiment
    n : int
        Number of suppliers
    allocation : AllocationFunction
        The allocation function $\omega$
    revenue_fn : Callable[[float], float]
        Platform revenue function $r(x)$
    revenue_fn_prime : Callable[[float], float]
        Derivative $r'(x)$

    Returns
    -------
    float
        Estimated utility gradient $\hat{\Gamma} \approx du/dp$
    """
    D_bar = data.D / n
    Z_bar = data.T / n

    if Z_bar < 1e-10:
        return 0.0

    x = D_bar / Z_bar

    r_x = revenue_fn(x)
    r_prime_x = revenue_fn_prime(x)
    omega_x = allocation(x)
    omega_prime_x = allocation.derivative(x)

    # Bracket term in (4.3)
    bracket = r_x - data.p * omega_x - (r_prime_x - data.p * omega_prime_x) * x

    # Full gradient estimate
    gamma_hat = upsilon_hat * bracket - omega_x * Z_bar

    return gamma_hat


@dataclass
class GradientEstimate:
    r"""
    Complete gradient estimate from one period of local experimentation.

    Attributes
    ----------
    delta_hat : float
        Estimated marginal response $\hat{\Delta}$ (equation 4.1)
    upsilon_hat : float
        Estimated supply gradient $\hat{\Upsilon} \approx \mu'$ (equation 4.2)
    gamma_hat : float
        Estimated utility gradient $\hat{\Gamma} \approx du/dp$ (equation 4.3)
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
    data: TimePointData,
    n: int,
    allocation: AllocationFunction,
    revenue_fn: RevenueFunction,
) -> GradientEstimate:
    r"""
    Complete gradient estimation from local experiment data.

    This implements the full estimation procedure from Section 4.1
    (Theorem 6), combining equations (4.1), (4.2), and (4.3).

    Parameters
    ----------
    data : TimePointData
        Data from one period of local experimentation
    n : int
        Number of suppliers
    allocation : AllocationFunction
        The allocation function ω
    revenue_fn : RevenueFunction
        Platform revenue function

    Returns
    -------
    GradientEstimate
        Complete gradient estimates
    """
    # Step 1: Estimate $\hat{\Delta}$ (equation 4.1)
    delta_hat = estimate_delta_hat(data, n)

    # Step 2: Estimate $\hat{\Upsilon}$ (equation 4.2)
    upsilon_hat = estimate_upsilon_hat(
        delta_hat=delta_hat, data=data, n=n, allocation=allocation
    )

    # Step 3: Estimate $\hat{\Gamma}$ (equation 4.3)
    gamma_hat = estimate_gamma_hat(
        upsilon_hat=upsilon_hat,
        data=data,
        n=n,
        allocation=allocation,
        revenue_fn=revenue_fn.r,
        revenue_fn_prime=revenue_fn.r_prime,
    )

    D_bar = data.D / n
    Z_bar = data.T / n

    return GradientEstimate(
        delta_hat=delta_hat,
        upsilon_hat=upsilon_hat,
        gamma_hat=gamma_hat,
        D_bar=D_bar,
        Z_bar=Z_bar,
    )


# =============================================================================
# SECTION 4.2: A FIRST-ORDER ALGORITHM
# =============================================================================


@dataclass
class OptimizationState:
    r"""
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
        History of gradient estimates $\hat{\Gamma}_t$
    payment_history : List[float]
        History of payment levels $p_t$
    """

    t: int
    p: float
    theta: float
    gradient_history: List[float]
    payment_history: List[float]


def initialize_optimizer(
    p_init: float, p_bounds: Tuple[float, float] = (0.0, float("inf"))
) -> OptimizationState:
    r"""
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
    p_bounds = (float(p_bounds[0]), float(p_bounds[1]))

    return OptimizationState(
        t=1,
        p=np.clip(p_init, p_bounds[0], p_bounds[1]),
        theta=0.0,
        gradient_history=[],
        payment_history=[p_init],
    )


def mirror_descent_update(
    state: OptimizationState,
    gradient: float,
    eta: float,
    p_bounds: Tuple[float, float] = (0.0, float("inf")),
) -> OptimizationState:
    r"""
    Perform one step of the mirror descent update.

    From equation (4.5):
        $p_{t+1} = \arg\min_p \{ (1/2\eta) \sum_{s=1}^t s(p - p_s)^2 - \theta_t \cdot p : p \in I \}$
        where $\theta_t = \sum_{s=1}^t s \cdot \hat{\Gamma}_s$

    For the unconstrained case ($I = \mathbb{R}$), this reduces to:
        $p_{t+1} = p_t + (2\eta \cdot \hat{\Gamma}_t) / (t+1)$

    For the constrained case, we project onto the interval.

    Parameters
    ----------
    state : OptimizationState
        Current optimization state
    gradient : float
        Gradient estimate $\hat{\Gamma}_t$ from current period
    eta : float
        Step size $\eta$ (should satisfy $\eta > \sigma^{-1}$ where $\sigma$ is strong concavity)
    p_bounds : Tuple[float, float]
        Payment bounds [c_-, c_+]

    Returns
    -------
    OptimizationState
        Updated optimization state
    """
    p_bounds = (float(p_bounds[0]), float(p_bounds[1]))

    t = state.t
    c_minus, c_plus = p_bounds

    # Update $\theta_t = \sum_{s=1}^t s \cdot \hat{\Gamma}_s$
    theta_new = state.theta + t * gradient

    # For the mirror descent update with quadratic regularizer,
    # the solution to the optimization problem in (4.5) is:
    #
    # Without constraint: weighted average of past payments plus gradient term
    # $p_{t+1} = (\sum_{s=1}^t s \cdot p_s + \eta \cdot \theta_t) / (\sum_{s=1}^t s)$
    #         = $(\sum_{s=1}^t s \cdot p_s) / (t(t+1)/2) + \eta \cdot \theta_t / (t(t+1)/2)$
    #
    # Simplified form (equivalent to basic gradient descent):
    # $p_{t+1} = p_t + 2\eta \cdot \hat{\Gamma}_t / (t+1)$

    p_new = state.p + (2 * eta * gradient) / (t + 1)

    # Project onto interval $I = [c_-, c_+]$
    p_new = np.clip(p_new, c_minus, c_plus)

    # Update histories
    gradient_history = state.gradient_history + [gradient]
    payment_history = state.payment_history + [p_new]

    return OptimizationState(
        t=t + 1,
        p=p_new,
        theta=theta_new,
        gradient_history=gradient_history,
        payment_history=payment_history,
    )


@overload
def run_learning_algorithm(
    *,
    params: ExperimentParams,
    rng: Optional[np.random.Generator] = None,
    verbose: bool = False,
) -> Experiment: ...


@overload
def run_learning_algorithm(
    T: int,
    n: int,
    p_init: float,
    eta: float,
    zeta: float,
    revenue_fn: RevenueFunction,
    allocation: AllocationFunction,
    supplier_params: SupplierParameters,
    d_a: Optional[float] = None,
    demand_params: Optional[DemandParameters] = None,
    p_bounds: Tuple[float, float] = (0.0, float("inf")),
    alpha: float = 0.3,
    store_detailed_data: bool = False,
    rng: Optional[np.random.Generator] = None,
    rng_seed: Optional[int] = None,
    verbose: bool = False,
    *,
    params: None = None,
) -> Experiment: ...


def run_learning_algorithm(
    T: Optional[int] = None,
    n: Optional[int] = None,
    p_init: Optional[float] = None,
    eta: Optional[float] = None,
    zeta: Optional[float] = None,
    revenue_fn: Optional[RevenueFunction] = None,
    allocation: Optional[AllocationFunction] = None,
    supplier_params: Optional[SupplierParameters] = None,
    d_a: Optional[float] = None,
    demand_params: Optional[DemandParameters] = None,
    p_bounds: Optional[Tuple[float, float]] = None,
    alpha: float = 0.3,
    store_detailed_data: bool = False,
    rng: Optional[np.random.Generator] = None,
    rng_seed: Optional[int] = None,
    verbose: bool = False,
    *,
    params: Optional[ExperimentParams] = None,
) -> Experiment:
    r"""
    Run the complete learning algorithm from Section 4.2.

    This implements the first-order optimization algorithm:

    For t = 1, ..., T:
        1. Sample global state $A_t$ (if using demand_params)
        2. Deploy payment perturbations around $p_t$
        3. Estimate gradient $\hat{\Gamma}_t$ via local experimentation
        4. Update $p_{t+1}$ via mirror descent (equation 4.5)

    Parameters
    ----------
    T : int
        Number of time periods (days)
    n : int
        Number of suppliers per period
    p_init : float
        Initial payment p_1
    eta : float
        Step size $\eta$ (should satisfy $\eta > \sigma^{-1}$)
    zeta : float
        Perturbation magnitude $\zeta$ (should scale as $n^{-\alpha}$ for $0 < \alpha < 0.5$)
    revenue_fn : RevenueFunction
        Platform revenue function
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
    alpha : float
        Zeta decay exponent (stored in params, not actively used here)
    store_detailed_data : bool
        Whether to store individual-level data (Z_i, ε_i arrays)
    rng : Optional[np.random.Generator]
        Random number generator for reproducibility
    rng_seed : Optional[int]
        Random seed (if rng not provided)
    verbose : bool
        Whether to print progress

    Returns
    -------
    Experiment
        Complete experiment including parameters, timepoint data, and results

    Notes
    -----
    Either d_a or demand_params must be provided. If demand_params is given,
    the global state A_t is sampled each period according to the paper's model.

    Can be called in two ways:
    1. With ExperimentParams: run_learning_algorithm(params=exp_params)
    2. With individual parameters: run_learning_algorithm(T=100, n=1000, ...)
    """
    # Extract parameters from ExperimentParams if provided
    if params is not None:
        T = params.T
        n = params.n
        p_init = params.p_init
        revenue_fn = params.revenue_fn
        allocation = params.allocation
        supplier_params = params.supplier_params
        p_bounds = params.p_bounds
        eta = params.eta
        zeta = params.zeta
        alpha = params.alpha if params.alpha is not None else 0.3
        store_detailed_data = params.store_detailed_data
        rng_seed = params.rng_seed

        # Extract demand parameters
        demand_config = extract_demand_from_params(params)
        demand_params = demand_config.demand_params
        d_a = demand_config.d_a
    else:
        # Validate that required parameters are provided
        if any(
            x is None
            for x in [T, n, p_init, eta, zeta, revenue_fn, allocation, supplier_params]
        ):
            raise ValueError(
                "When params is not provided, all of T, n, p_init, eta, zeta, "
                "revenue_fn, allocation, and supplier_params must be provided"
            )
        if p_bounds is None:
            p_bounds = (0.0, float("inf"))

    # Setup RNG
    rng = setup_rng(rng, rng_seed)

    if d_a is None and demand_params is None:
        raise ValueError("Must provide either d_a or demand_params")

    # Create experiment parameters (only if not already provided)
    if params is None:
        params = ExperimentParams(
            T=T,
            n=n,
            p_init=p_init,
            revenue_fn=revenue_fn,
            p_bounds=p_bounds,
            allocation=allocation,
            supplier_params=supplier_params,
            demand=demand_params if demand_params is not None else d_a,
            eta=eta,
            experiment_type="local",
            zeta=zeta,
            alpha=alpha,
            delta=None,
            rng_seed=rng_seed,
            store_detailed_data=store_detailed_data,
        )

    # Initialize optimizer
    opt_state = initialize_optimizer(p_init, p_bounds)
    timepoints: List[TimePointData] = []

    # Import here to avoid circular imports

    for t in range(1, T + 1):
        if verbose and t % 20 == 0:
            print(f"  Period {t}/{T}: p = {opt_state.p:.4f}")

        # Sample global state A_t for this period
        current = sample_current_state(demand_params, d_a, rng)
        current_state = current.state
        current_d_a = current.d_a

        # Compute expected allocation at current payment for this state
        eq = compute_equilibrium_allocation(
            p=opt_state.p,
            d_a=current_d_a,
            supplier_params=supplier_params,
            allocation=allocation,
        )
        q_eq = eq.q_eq

        # Run local experiment for this period
        timepoint = run_local_experiment(
            n=n,
            p=opt_state.p,
            zeta=zeta,
            expected_allocation=q_eq,
            supplier_params=supplier_params,
            revenue_fn=revenue_fn,
            allocation=allocation,
            t=t,
            d_a=current_d_a,
            demand_params=demand_params,
            state=current_state,
            store_detailed_data=True,
            rng=rng,
        )

        # Estimate gradients
        grad_est = estimate_utility_gradient(timepoint, n, allocation, revenue_fn)

        # Update timepoint with gradient estimates
        timepoint.gradient_estimate = grad_est.gamma_hat
        timepoint.delta_hat = grad_est.delta_hat
        timepoint.upsilon_hat = grad_est.upsilon_hat

        timepoints.append(timepoint)

        # Update payment via mirror descent
        opt_state = mirror_descent_update(
            state=opt_state, gradient=grad_est.gamma_hat, eta=eta, p_bounds=p_bounds
        )

    # Compute weighted average (Corollary 8)
    weighted_avg = compute_weighted_average_payment(timepoints)

    # Build results
    results = build_experiment_results(
        timepoints=timepoints,
        final_payment=opt_state.p,
        weighted_average_payment=weighted_avg,
    )

    return Experiment(params=params, results=results)


# =============================================================================
# SECTION 4.3: COST OF EXPERIMENTATION
# =============================================================================


def compute_experimentation_cost(
    p: float, zeta: float, d_a: float, mu: float, allocation: AllocationFunction
) -> float:
    r"""
    Compute the cost of experimentation (Theorem 9).

    The excess cost from randomization is $O(\zeta^2)$ as shown in Theorem 9.
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
        Approximate cost of experimentation ($O(\zeta^2)$ term)
    """
    # The cost is approximately proportional to $\zeta^2$ times a constant
    # depending on the curvature of the choice function
    # (Theorem 9: $u(p) - u(p, \zeta) \leq C \cdot \zeta^2$)

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
    r"""
    Compute optimal perturbation magnitude $\zeta_n$.

    From Theorem 6, perturbations should scale as $\zeta_n = \zeta \cdot n^{-\alpha}$
    for $0 < \alpha < 0.5$ to ensure consistency.

    Parameters
    ----------
    n : int
        Market size
    alpha : float
        Decay exponent ($0 < \alpha < 0.5$)

    Returns
    -------
    float
        Optimal perturbation magnitude
    """
    base_zeta = 0.5  # Base perturbation (can be tuned)
    return base_zeta * (n ** (-alpha))


def analyze_convergence(result: Experiment, p_optimal: float) -> dict:
    r"""
    Analyze convergence of the learning algorithm.

    Parameters
    ----------
    result : Experiment
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
    regret = (p_optimal - payments[:-1]) ** 2

    # Compute weighted regret (as in Theorem 7)
    weights = np.arange(1, T + 1)
    weighted_regret = np.sum(weights * regret) / (T * (T + 1) / 2)

    return {
        "final_payment": result.final_payment,
        "weighted_average": result.weighted_average_payment,
        "optimal_payment": p_optimal,
        "final_error": abs(result.final_payment - p_optimal),
        "weighted_error": abs(result.weighted_average_payment - p_optimal),
        "mean_squared_error": np.mean(regret),
        "weighted_regret": weighted_regret,
    }
