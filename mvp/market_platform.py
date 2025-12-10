"""
Platform Utility Model for Stochastic Market

This module implements the platform's objective function.

Key equations:
- Utility (3.8): U = R(D, T) - Σ P_i Z_i S_i
- Revenue (3.9): R(d, t) = r(d/t) * t
- Normalized utility (3.10): u_a(p) = (1/n) * E[U | A=a]

The platform wants to maximize expected utility by choosing the right payment p.
"""

import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass

# Import our other modules
from .allocation import AllocationFunction, QueueAllocation
from .supplier import SupplierPopulation, LogisticChoice, LogNormalCosts


class RevenueFunction:
    """
    Base class for platform revenue functions.
    
    R(d, t) = r(d/t) * t
    
    where r(x) is revenue per active supplier when demand ratio is x.
    """
    
    def r(self, x: float) -> float:
        """Revenue per supplier when demand ratio is x = d/t."""
        raise NotImplementedError
    
    def r_derivative(self, x: float) -> float:
        """Derivative dr/dx."""
        raise NotImplementedError
    
    def R(self, d: float, t: float) -> float:
        """Total revenue R(d, t) = r(d/t) * t."""
        if t <= 0:
            return 0.0
        return self.r(d / t) * t


class LinearRevenue(RevenueFunction):
    """
    Linear revenue: platform gets γ per unit of demand served.
    
    R(D, T) = γ * T * Ω(D, T) = γ * T * ω(D/T)
    
    So r(x) = γ * ω(x)
    """
    
    def __init__(self, gamma: float, allocation: AllocationFunction):
        """
        Args:
            gamma: Payment per unit of demand served
            allocation: Allocation function ω(x)
        """
        self.gamma = gamma
        self.allocation = allocation
        
    def r(self, x: float) -> float:
        """r(x) = γ * ω(x)"""
        return self.gamma * self.allocation.omega(x)
    
    def r_derivative(self, x: float) -> float:
        """dr/dx = γ * ω'(x)"""
        return self.gamma * self.allocation.omega_derivative(x)


class PlatformUtility:
    """
    Computes platform utility in the stochastic market model.
    
    Utility = Revenue - Payments
    U = R(D, T) - Σ P_i Z_i S_i
    
    In the mean-field limit:
    u_a(p) = (r(d_a/μ_a(p)) - p*ω(d_a/μ_a(p))) * μ_a(p)
    """
    
    def __init__(self, 
                 revenue_fn: RevenueFunction,
                 allocation: AllocationFunction,
                 population: SupplierPopulation):
        """
        Args:
            revenue_fn: Platform revenue function R(d, t)
            allocation: Allocation function Ω(d, t)
            population: Supplier population model
        """
        self.revenue_fn = revenue_fn
        self.allocation = allocation
        self.population = population
        
    def compute_utility(self, d_a: float, mu: float, p: float) -> float:
        """
        Compute mean-field utility given activation rate μ.
        
        u_a(p) = (r(d_a/μ) - p*ω(d_a/μ)) * μ
        
        Args:
            d_a: Expected demand per supplier
            mu: Fraction of suppliers who are active
            p: Payment per unit of demand served
            
        Returns:
            Normalized utility (utility per supplier)
        """
        if mu <= 0:
            return 0.0
            
        x = d_a / mu  # Demand per active supplier
        
        revenue_per_supplier = self.revenue_fn.r(x)
        payment_per_supplier = p * self.allocation.omega(x)
        
        profit_per_active_supplier = revenue_per_supplier - payment_per_supplier
        
        return profit_per_active_supplier * mu
    
    def compute_utility_derivative(self, d_a: float, mu: float, mu_prime: float, 
                                   p: float) -> float:
        """
        Compute derivative of utility with respect to payment p.
        
        du/dp = μ'(p) * [r(d_a/μ) - p*ω(d_a/μ) - (r'(d_a/μ) - p*ω'(d_a/μ)) * d_a/μ]
                - ω(d_a/μ) * μ
        
        Args:
            d_a: Expected demand per supplier
            mu: Activation rate μ_a(p)
            mu_prime: Derivative dμ/dp
            p: Payment
            
        Returns:
            du_a/dp
        """
        if mu <= 0:
            return 0.0
            
        x = d_a / mu
        
        r_val = self.revenue_fn.r(x)
        r_deriv = self.revenue_fn.r_derivative(x)
        omega_val = self.allocation.omega(x)
        omega_deriv = self.allocation.omega_derivative(x)
        
        # Term from change in μ
        bracket_term = (r_val - p * omega_val - 
                       (r_deriv - p * omega_deriv) * x)
        mu_contribution = mu_prime * bracket_term
        
        # Direct term from change in p
        direct_term = -omega_val * mu
        
        return mu_contribution + direct_term


@dataclass
class MarketOutcome:
    """Results from simulating one period of the market."""
    demand: int
    num_active: int
    total_served: float
    revenue: float
    payments: float
    utility: float
    payment_level: float
    

class Market:
    """
    Complete market simulation combining all components.
    
    This class ties together:
    - Demand generation
    - Supplier activation decisions
    - Demand allocation
    - Platform utility computation
    """
    
    def __init__(self,
                 allocation: AllocationFunction,
                 population: SupplierPopulation,
                 gamma: float):
        """
        Args:
            allocation: Allocation function
            population: Supplier population
            gamma: Platform revenue per unit of demand served
        """
        self.allocation = allocation
        self.population = population
        self.gamma = gamma
        self.revenue_fn = LinearRevenue(gamma, allocation)
        self.utility_computer = PlatformUtility(
            self.revenue_fn, allocation, population
        )
        
    def simulate_period(self, n: int, d_a: float, p: float, zeta: float = 0.0,
                        rng: Optional[np.random.Generator] = None) -> MarketOutcome:
        """
        Simulate one period of the market.
        
        Args:
            n: Number of potential suppliers
            d_a: Expected demand per supplier (or actual demand/n)
            p: Base payment level
            zeta: Payment perturbation magnitude (for local experimentation)
            rng: Random number generator
            
        Returns:
            MarketOutcome with simulation results
        """
        if rng is None:
            rng = np.random.default_rng()
            
        # Generate demand
        D = int(round(n * d_a))  # For simplicity, use deterministic demand
        
        # Generate payments (with possible perturbation)
        if zeta > 0:
            epsilon = rng.choice([-1, 1], size=n)
            payments = p + zeta * epsilon
        else:
            payments = np.full(n, p)
            epsilon = None
            
        # First, compute equilibrium expected allocation
        # This requires solving a fixed-point equation
        mu_eq = self.find_equilibrium_mu(d_a, p)
        q_eq = self.allocation.q(mu_eq, d_a)  # Expected allocation per active
        
        # Suppliers make activation decisions
        Z = self.population.sample_suppliers(n, payments, q_eq, rng)
        T = Z.sum()  # Number of active suppliers
        
        # Compute actual allocations
        if T > 0:
            # In practice, allocations would be random
            # For simplicity, use expected allocation
            actual_q = self.allocation.Omega(D, T)
            S = np.where(Z == 1, actual_q, 0.0)
        else:
            S = np.zeros(n)
            
        # Compute outcomes
        total_served = S.sum() if T > 0 else 0.0
        revenue = self.gamma * total_served
        total_payments = (payments * Z * S).sum()
        utility = revenue - total_payments
        
        return MarketOutcome(
            demand=D,
            num_active=T,
            total_served=total_served,
            revenue=revenue,
            payments=total_payments,
            utility=utility,
            payment_level=p
        )
    
    def find_equilibrium_mu(self, d_a: float, p: float, 
                            tol: float = 1e-6, max_iter: int = 100) -> float:
        """
        Find equilibrium activation rate μ_a(p).
        
        Solves the fixed-point equation:
        μ = E[f_B(p * ω(d_a/μ))]
        
        Args:
            d_a: Expected demand per supplier
            p: Payment level
            tol: Convergence tolerance
            max_iter: Maximum iterations
            
        Returns:
            Equilibrium activation rate μ
        """
        # Start with initial guess
        mu = 0.5
        
        for i in range(max_iter):
            # Compute expected allocation given current μ
            q = self.allocation.q(mu, d_a)
            
            # Compute expected revenue
            expected_revenue = p * q
            
            # Compute new activation rate
            mu_new = self.population.activation_probability(expected_revenue)
            
            # Check convergence
            if abs(mu_new - mu) < tol:
                return mu_new
                
            # Update with some damping for stability
            mu = 0.7 * mu_new + 0.3 * mu
            
        return mu
    
    def mean_field_utility(self, d_a: float, p: float) -> float:
        """
        Compute mean-field utility at payment level p.
        
        u_a(p) = (r(d_a/μ) - p*ω(d_a/μ)) * μ
        
        where μ = μ_a(p) is the equilibrium activation rate.
        """
        mu = self.find_equilibrium_mu(d_a, p)
        return self.utility_computer.compute_utility(d_a, mu, p)


# Example usage and testing
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Set up market components
    allocation = QueueAllocation(L=8)
    choice_fn = LogisticChoice(alpha=1.0)
    cost_dist = LogNormalCosts(log_mean=0.0, log_std=1.0, scale=20.0)
    population = SupplierPopulation(choice_fn, cost_dist)
    
    gamma = 100  # Platform gets 100 per unit served
    
    market = Market(allocation, population, gamma)
    
    # Analyze utility as function of payment
    d_a = 0.4  # Expected demand per supplier
    
    payments = np.linspace(5, 60, 50)
    utilities = []
    mus = []
    
    for p in payments:
        mu = market.find_equilibrium_mu(d_a, p)
        u = market.mean_field_utility(d_a, p)
        utilities.append(u)
        mus.append(mu)
    
    # Find optimal payment
    optimal_idx = np.argmax(utilities)
    optimal_p = payments[optimal_idx]
    optimal_u = utilities[optimal_idx]
    
    # Plot results
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Utility vs payment
    axes[0].plot(payments, utilities, 'b-', linewidth=2)
    axes[0].axvline(optimal_p, color='r', linestyle='--', label=f'Optimal p={optimal_p:.1f}')
    axes[0].set_xlabel('Payment p')
    axes[0].set_ylabel('Mean-Field Utility u(p)')
    axes[0].set_title('Platform Utility vs Payment')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Activation rate vs payment
    axes[1].plot(payments, mus, 'g-', linewidth=2)
    axes[1].axvline(optimal_p, color='r', linestyle='--', label=f'Optimal p={optimal_p:.1f}')
    axes[1].set_xlabel('Payment p')
    axes[1].set_ylabel('Activation Rate μ(p)')
    axes[1].set_title('Supplier Activation vs Payment')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/home/claude/market_model/utility_analysis.png', dpi=150)
    plt.close()
    
    print(f"Market Analysis (d_a = {d_a}, γ = {gamma}):")
    print(f"  Optimal payment: p* = {optimal_p:.2f}")
    print(f"  Maximum utility: u* = {optimal_u:.4f}")
    print(f"  Activation rate at optimum: μ* = {mus[optimal_idx]:.4f}")
    
    # Simulate a few periods
    print("\nSimulation (n=1000):")
    rng = np.random.default_rng(42)
    for p in [10, optimal_p, 50]:
        outcome = market.simulate_period(1000, d_a, p, rng=rng)
        print(f"  p={p:.1f}: active={outcome.num_active}, served={outcome.total_served:.1f}, "
              f"utility={outcome.utility:.1f}")