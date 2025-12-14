"""
Finite-n Equilibrium Implementation for Wager & Xu (2021)

This prototype implements the pre-limit (finite n) system from Section 3,
before taking n → ∞.
"""

import numpy as np
from typing import Callable, Tuple, Optional
from dataclasses import dataclass
from scipy.optimize import brentq
from scipy.stats import binom


@dataclass
class FiniteSystemParams:
    """Parameters for the finite-n system."""
    n: int                              # Number of potential suppliers
    omega: Callable[[float], float]     # Limiting allocation function ω(x)
    omega_prime: Callable[[float], float]  # Derivative ω'(x)
    choice_function: Callable[[float, float], float]  # f_b(x)
    choice_derivative: Callable[[float, float], float]  # f'_b(x)
    sample_private_features: Callable[[int], np.ndarray]  # Sample B_i
    d_a: float                          # Expected demand per supplier E[D/n|A=a]
    demand_variance: float              # Var(D/n | A=a) for finite n
    gamma: float                        # Revenue per unit served
    n_mc_samples: int = 10000           # Monte Carlo samples


def finite_allocation(d: int, t: int, L: int = 8) -> float:
    """
    Compute finite-n allocation Ω(d, t) for M/M/1 queues.
    
    From Example 6: Each active supplier operates as M/M/1 queue with capacity L.
    
    Parameters
    ----------
    d : int
        Total demand
    t : int
        Number of active suppliers
    L : int
        Queue capacity
        
    Returns
    -------
    float
        Expected demand served per active supplier
    """
    if t <= 0:
        return 0.0
    
    x = d / t  # Demand ratio
    
    if x < 1e-10:
        return x
    elif abs(x - 1.0) < 1e-10:
        return 1.0 - 1.0 / L
    else:
        x_L = x ** L
        return (x - x_L) / (1.0 - x_L)


def sample_demand(n: int, d_a: float, var_factor: float = 0.1) -> int:
    """
    Sample demand D for finite system.
    
    Uses a scaled Binomial or Poisson approximation.
    For simplicity, using Poisson(n * d_a) which satisfies
    the concentration requirements (3.1)-(3.2).
    
    Parameters
    ----------
    n : int
        Number of suppliers
    d_a : float
        Expected demand per supplier
    var_factor : float
        Controls variance (not used for Poisson)
        
    Returns
    -------
    int
        Sampled demand
    """
    return np.random.poisson(n * d_a)


def compute_finite_expected_allocation(
    mu: float,
    d: int,
    n: int,
    omega: Callable[[float], float],
    n_samples: int = 10000
) -> float:
    """
    Compute E[Ω(d, T) | T ~ Binomial(n, μ)] for finite n.
    
    From Equation 3.12:
        q_a^(n)(μ) = E[Ω(D, X) | A=a]  where X ~ Binomial(n, μ)
    
    For fixed d, this computes E[Ω(d, T)] over the binomial distribution.
    
    Parameters
    ----------
    mu : float
        Activation probability
    d : int
        Total demand
    n : int
        Number of potential suppliers
    omega : Callable
        Limiting allocation function ω(x)
    n_samples : int
        Monte Carlo samples
        
    Returns
    -------
    float
        Expected allocation per active supplier
    """
    # Monte Carlo over binomial samples
    t_samples = np.random.binomial(n, mu, n_samples)
    allocations = []
    
    for t in t_samples:
        if t > 0:
            allocations.append(omega(d / t))
        else:
            allocations.append(0.0)
    
    return np.mean(allocations)


def solve_finite_equilibrium_mu(
    p: float,
    params: FiniteSystemParams,
    d: Optional[int] = None,
    tol: float = 1e-6
) -> float:
    """
    Solve for equilibrium μ in finite-n system.
    
    From the proof of Lemma 1 (Appendix A.1, Equation A.2):
        μ = E[f_{B_1}(P_1 · E[Ω(D, X) | A=a])]  where X ~ Binomial(n, μ)
    
    This is the finite-n analogue of the mean-field fixed-point equation.
    
    Parameters
    ----------
    p : float
        Payment per unit served
    params : FiniteSystemParams
        System parameters
    d : int, optional
        Fixed demand (if None, uses E[D] = n * d_a)
    tol : float
        Convergence tolerance
        
    Returns
    -------
    float
        Equilibrium activation probability μ
    """
    if d is None:
        d = int(round(params.n * params.d_a))
    
    def fixed_point_residual(mu: float) -> float:
        """Residual g(μ) = μ - E[f_B(p · E[Ω(d, T)])]"""
        # Compute expected allocation given μ
        q = compute_finite_expected_allocation(
            mu, d, params.n, params.omega, params.n_mc_samples
        )
        
        # Expected revenue for a single supplier
        expected_revenue = p * q
        
        # Compute E[f_B(expected_revenue)]
        b_samples = params.sample_private_features(params.n_mc_samples)
        probs = np.array([
            params.choice_function(expected_revenue, b) 
            for b in b_samples
        ])
        
        return mu - np.mean(probs)
    
    # Use Brent's method
    try:
        mu_star = brentq(fixed_point_residual, 1e-8, 1.0 - 1e-8, xtol=tol)
    except ValueError:
        # Fallback to fixed-point iteration
        mu = 0.5
        for _ in range(100):
            q = compute_finite_expected_allocation(
                mu, d, params.n, params.omega, params.n_mc_samples
            )
            expected_revenue = p * q
            b_samples = params.sample_private_features(params.n_mc_samples)
            probs = np.array([
                params.choice_function(expected_revenue, b) 
                for b in b_samples
            ])
            mu_new = np.mean(probs)
            if abs(mu_new - mu) < tol:
                return mu_new
            mu = 0.7 * mu_new + 0.3 * mu
        mu_star = mu
    
    return mu_star


@dataclass
class FiniteEquilibriumResult:
    """Result of finite-n equilibrium computation."""
    mu: float                  # Equilibrium activation probability
    q: float                   # Expected allocation per active supplier
    expected_active: float     # E[T] = n * μ
    expected_utility: float    # E[U] / n
    demand: int               # The demand value used
    n: int                     # System size


def compute_finite_equilibrium(
    p: float,
    params: FiniteSystemParams,
    d: Optional[int] = None
) -> FiniteEquilibriumResult:
    """
    Compute full finite-n equilibrium.
    
    This implements the pre-limit model from Section 3.
    
    Parameters
    ----------
    p : float
        Payment
    params : FiniteSystemParams
        System parameters
    d : int, optional
        Fixed demand (if None, uses E[D])
        
    Returns
    -------
    FiniteEquilibriumResult
        Equilibrium quantities
    """
    if d is None:
        d = int(round(params.n * params.d_a))
    
    # Solve for equilibrium μ
    mu = solve_finite_equilibrium_mu(p, params, d)
    
    # Compute expected allocation
    q = compute_finite_expected_allocation(
        mu, d, params.n, params.omega, params.n_mc_samples
    )
    
    # Expected number of active suppliers
    expected_active = params.n * mu
    
    # Expected utility per supplier (Equation 3.10 with 3.8)
    # u = (γ - p) * q * μ  (for linear revenue)
    expected_utility = (params.gamma - p) * q * mu
    
    return FiniteEquilibriumResult(
        mu=mu,
        q=q,
        expected_active=expected_active,
        expected_utility=expected_utility,
        demand=d,
        n=params.n
    )


def simulate_finite_market(
    p: float,
    params: FiniteSystemParams,
    zeta: float = 0.0,
    rng: Optional[np.random.Generator] = None
) -> dict:
    """
    Simulate a single period of the finite-n market.
    
    This implements the full stochastic simulation from Section 3,
    including random demand, random activations, and random allocations.
    
    Parameters
    ----------
    p : float
        Base payment
    params : FiniteSystemParams
        System parameters
    zeta : float
        Perturbation magnitude for local experimentation (Equation 2.1)
    rng : Generator, optional
        Random number generator
        
    Returns
    -------
    dict
        Simulation results
    """
    if rng is None:
        rng = np.random.default_rng()
    
    n = params.n
    
    # Step 1: Sample demand
    D = sample_demand(n, params.d_a)
    
    # Step 2: Generate payments (with possible perturbation)
    if zeta > 0:
        epsilon = rng.choice([-1, 1], size=n)
        payments = p + zeta * epsilon
    else:
        payments = np.full(n, p)
        epsilon = np.zeros(n)
    
    # Step 3: Compute equilibrium expected allocation
    # (suppliers base decisions on equilibrium expectations)
    mu_eq = solve_finite_equilibrium_mu(p, params, D)
    q_eq = compute_finite_expected_allocation(
        mu_eq, D, n, params.omega, params.n_mc_samples
    )
    
    # Step 4: Suppliers make activation decisions
    b_samples = params.sample_private_features(n)
    Z = np.zeros(n, dtype=int)
    for i in range(n):
        expected_revenue = payments[i] * q_eq
        prob_active = params.choice_function(expected_revenue, b_samples[i])
        Z[i] = rng.random() < prob_active
    
    T = Z.sum()  # Number of active suppliers
    
    # Step 5: Allocate demand to active suppliers
    if T > 0:
        # Each active supplier gets allocation based on Ω(D, T)
        omega_val = params.omega(D / T)
        S = np.where(Z == 1, omega_val, 0.0)
    else:
        S = np.zeros(n)
    
    # Step 6: Compute outcomes
    total_served = S.sum()
    revenue = params.gamma * total_served
    total_payments = (payments * Z * S).sum()
    utility = revenue - total_payments
    
    return {
        'demand': D,
        'num_active': T,
        'activations': Z,
        'allocations': S,
        'payments': payments,
        'epsilon': epsilon,
        'total_served': total_served,
        'revenue': revenue,
        'total_payments': total_payments,
        'utility': utility,
        'utility_per_supplier': utility / n,
        'payment_level': p,
        'zeta': zeta
    }


def estimate_gradient_via_local_experimentation(
    p: float,
    params: FiniteSystemParams,
    zeta: float,
    n_periods: int = 100,
    rng: Optional[np.random.Generator] = None
) -> dict:
    """
    Estimate utility gradient via local experimentation.
    
    This implements the gradient estimation from Theorem 6 (Equations 4.1-4.3):
    
    1. Run market with perturbations P_i = p + ζε_i
    2. Estimate Δ̂ via regression of Z_i on ε_i
    3. Transform Δ̂ to Γ̂ using the equilibrium model
    
    Parameters
    ----------
    p : float
        Base payment
    params : FiniteSystemParams
        System parameters
    zeta : float
        Perturbation magnitude
    n_periods : int
        Number of periods to simulate
    rng : Generator, optional
        Random number generator
        
    Returns
    -------
    dict
        Gradient estimates and diagnostics
    """
    if rng is None:
        rng = np.random.default_rng()
    
    all_Z = []
    all_epsilon = []
    all_D = []
    all_T = []
    
    for _ in range(n_periods):
        result = simulate_finite_market(p, params, zeta, rng)
        all_Z.append(result['activations'])
        all_epsilon.append(result['epsilon'])
        all_D.append(result['demand'])
        all_T.append(result['num_active'])
    
    # Pool all data
    Z = np.concatenate(all_Z)
    epsilon = np.concatenate(all_epsilon)
    
    # Average statistics
    D_bar = np.mean(all_D) / params.n  # Scaled demand
    Z_bar = np.mean(Z)  # Fraction active
    
    # Equation 4.1: Regression coefficient
    # Δ̂ = (1/ζ) * Cov(Z, ε) / Var(ε)
    cov_Z_eps = np.cov(Z, epsilon)[0, 1]
    var_eps = np.var(epsilon)
    
    if abs(var_eps) < 1e-10:
        delta_hat = 0.0
    else:
        delta_hat = cov_Z_eps / (zeta * var_eps)
    
    # Equation 4.2: Transform to actual gradient
    # Υ̂ = Δ̂ / (1 + (p·D̄·Δ̂·ω'(D̄/Z̄)) / (Z̄²·ω(D̄/Z̄)))
    if abs(Z_bar) < 1e-10:
        upsilon_hat = delta_hat
    else:
        x = D_bar / Z_bar
        omega_val = params.omega(x)
        omega_prime_val = params.omega_prime(x)
        
        if abs(omega_val) < 1e-10:
            upsilon_hat = delta_hat
        else:
            interference_term = (p * D_bar * delta_hat * omega_prime_val) / (Z_bar**2 * omega_val)
            upsilon_hat = delta_hat / (1.0 + interference_term)
    
    # Equation 4.3: Compute Γ̂ (utility gradient estimate)
    if abs(Z_bar) < 1e-10:
        gamma_hat = 0.0
    else:
        x = D_bar / Z_bar
        omega_val = params.omega(x)
        omega_prime_val = params.omega_prime(x)
        
        # r(x) = γ·ω(x) for linear revenue
        r_val = params.gamma * omega_val
        r_prime_val = params.gamma * omega_prime_val
        
        bracket_term = (r_val - p * omega_val - 
                       (r_prime_val - p * omega_prime_val) * x)
        
        gamma_hat = upsilon_hat * bracket_term - omega_val * Z_bar
    
    return {
        'delta_hat': delta_hat,
        'upsilon_hat': upsilon_hat,
        'gamma_hat': gamma_hat,
        'D_bar': D_bar,
        'Z_bar': Z_bar,
        'n_periods': n_periods,
        'n_observations': len(Z),
        'zeta': zeta
    }


# =============================================================================
# COMPARISON: MEAN-FIELD VS FINITE-N
# =============================================================================

def compare_finite_to_mean_field(
    p: float,
    params: FiniteSystemParams,
    mean_field_params: dict,
    n_values: list = [100, 500, 1000, 5000, 10000]
) -> dict:
    """
    Compare finite-n equilibrium to mean-field limit.
    
    Verifies Lemma 2: As n → ∞, finite-n quantities converge to mean-field.
    
    Parameters
    ----------
    p : float
        Payment
    params : FiniteSystemParams
        Base parameters (n will be varied)
    mean_field_params : dict
        Pre-computed mean-field equilibrium
    n_values : list
        System sizes to test
        
    Returns
    -------
    dict
        Convergence analysis
    """
    results = {
        'n': [],
        'mu_finite': [],
        'q_finite': [],
        'u_finite': [],
        'mu_error': [],
        'q_error': [],
        'u_error': []
    }
    
    mu_mf = mean_field_params['mu']
    q_mf = mean_field_params['q']
    u_mf = mean_field_params['u']
    
    for n in n_values:
        # Create params with new n
        params_n = FiniteSystemParams(
            n=n,
            omega=params.omega,
            omega_prime=params.omega_prime,
            choice_function=params.choice_function,
            choice_derivative=params.choice_derivative,
            sample_private_features=params.sample_private_features,
            d_a=params.d_a,
            demand_variance=params.demand_variance,
            gamma=params.gamma,
            n_mc_samples=params.n_mc_samples
        )
        
        # Compute finite equilibrium
        eq = compute_finite_equilibrium(p, params_n)
        
        results['n'].append(n)
        results['mu_finite'].append(eq.mu)
        results['q_finite'].append(eq.q)
        results['u_finite'].append(eq.expected_utility)
        results['mu_error'].append(abs(eq.mu - mu_mf))
        results['q_error'].append(abs(eq.q - q_mf))
        results['u_error'].append(abs(eq.expected_utility - u_mf))
    
    return results