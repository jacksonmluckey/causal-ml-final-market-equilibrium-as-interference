"""
Supplier Choice Model for Stochastic Market

This module implements the supplier decision-making process.

Key equation (3.6):
$\mu_a(\pi) = E[f_{B_i}(P_i \cdot E[\Omega(D,T)|A=a]) | A=a]$

Where:
- $P_i$ is the payment offered to supplier i
- $B_i$ is supplier i's private feature (e.g., outside option/cost)
- $f_b(x)$ is the choice function: probability of becoming active given expected revenue x

The key insight is that suppliers use STATIONARY reasoning:
- They consider the average market equilibrium, not their own effect on it
- This is justified in large markets where individual impact is negligible
"""

import numpy as np
from typing import Callable, Optional
from dataclasses import dataclass
from .utils import numerical_derivative


@dataclass
class ChoiceFunction:
    """
    Supplier choice function f_b(x) as described in Section 3 Assumption 2.
    
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
        """
        Calculate choice probability given expected revenue x and private feature by
        evaluating f_b(x) = P(active | expected_revenue=x, private_feature=b).
        
        Args:
            x: Expected revenue (= payment * expected allocation)
            b: Private feature (e.g., cost threshold)
            
        Returns:
            Probability in [0, 1] of becoming active
        """
        return self.f(x, b)
    
    def derivative(self, x: float, b: float) -> float:
        """
        Takes the derivative df/dx of choice function with respect to expected revenue.
        Evaluates $f'_b(x) = \partial f_b/\partial x$.

        This measures how sensitive the supplier's decision is to revenue changes.

        This appears in the marginal response function (Definition 9):
            $\Delta_a(p) = q_a(\mu_a(p)) \cdot E[f'_{B_1}(p \cdot q_a(\mu_a(p))) | A=a]$
        """
        if self.f_prime is not None:
            return self.f_prime(x, b)
        else:
            return numerical_derivative(lambda xx: self.f(xx, b), x)


def create_logistic_choice(alpha: float = 1.0) -> ChoiceFunction:
    """
    Create logistic choice function from Example 7.

    $P(Z_i = 1 | P_i, \pi, A) = 1 / (1 + \exp(-\alpha(P_i \cdot E[\Omega] - B_i)))$

    Properties:
    - $\alpha$ controls sensitivity: larger $\alpha \to$ more deterministic decisions
    - $B_i$ is the break-even cost threshold
    - Supplier activates if expected revenue exceeds their cost threshold

    As $\alpha \to \infty$, this becomes a step function: activate iff revenue > cost

    Parameters
    ----------
    alpha : float
        Sensitivity parameter. As $\alpha \to \infty$, decision becomes deterministic.

    Returns
    -------
    ChoiceFunction
        The logistic choice function
    """
    def f(x: float, b: float) -> float:
        """
        Logistic choice probability.
        $f_b(x) = 1 / (1 + \exp(-\alpha(x - b)))$

        Args:
            x: Expected revenue
            b: Break-even cost threshold
        """
        z = alpha * (x - b)
        # Numerically stable sigmoid
        z = np.clip(z, -500, 500)
        if z >= 0:
            return 1.0 / (1.0 + np.exp(-z))
        else:
            exp_z = np.exp(z)
            return exp_z / (1.0 + exp_z)

    def f_prime(x: float, b: float) -> float:
        """
        Derivative of logistic choice function.
        $df/dx = \alpha \cdot f(x,b) \cdot (1 - f(x,b))$
        """
        prob = f(x, b)
        return alpha * prob * (1.0 - prob)

    return ChoiceFunction(
        f=f,
        f_prime=f_prime,
        name=f"Logistic ($\\alpha$={alpha})"
    )


@dataclass
class PrivateFeatureDistribution:
    """
    Distribution of private features B_i.

    Captures heterogeneity across suppliers (e.g., outside options,
    break-even cost thresholds).

    Paper assumes $B_i$ is i.i.d. from some B distribution.

    Parameters
    ----------
    sample : Callable[[int], np.ndarray]
        Function to sample n i.i.d. draws from the distribution
    name : str
        Descriptive name
    """
    sample: Callable[[int], np.ndarray]
    name: str = "Generic"


def create_lognormal_costs(
    log_mean: float = 0.0,
    log_std: float = 1.0,
    scale: float = 20.0
) -> PrivateFeatureDistribution:
    """
    Create lognormal distribution for supplier costs.

    $\log(B_i / \text{scale}) \sim N(\text{log\_mean}, \text{log\_std}^2)$

    Parameters
    ----------
    log_mean : float
        Mean of $\log(B_i / \text{scale})$
    log_std : float
        Standard deviation of $\log(B_i / \text{scale})$
    scale : float
        Scale factor (20 in the paper's example)

    Returns
    -------
    PrivateFeatureDistribution
        The lognormal cost distribution
    """
    def sample(n: int) -> np.ndarray:
        return scale * np.exp(np.random.normal(log_mean, log_std, n))

    return PrivateFeatureDistribution(
        sample=sample,
        name=f"LogNormal($\\mu$={log_mean}, $\\sigma$={log_std}, scale={scale})"
    )


def create_uniform_costs(low: float = 5.0, high: float = 50.0) -> PrivateFeatureDistribution:
    """
    Create uniform distribution for supplier costs.

    $B_i \sim \text{Uniform}(\text{low}, \text{high})$
    """
    def sample(n: int) -> np.ndarray:
        return np.random.uniform(low, high, n)

    return PrivateFeatureDistribution(
        sample=sample,
        name=f"Uniform({low}, {high})"
    )


def compute_expected_choice_probability(
    revenue: float,
    choice: ChoiceFunction,
    private_features: PrivateFeatureDistribution,
    n_samples: int = 10000
) -> float:
    """
    Compute $E[f_B(x)]$ via Monte Carlo.

    This is the average probability of activation across all supplier types
    when expected revenue is x.

    Parameters
    ----------
    revenue : float
        Expected revenue (x)
    choice : ChoiceFunction
        The choice function $f_b(\cdot)$
    private_features : PrivateFeatureDistribution
        Distribution of $B_i$
    n_samples : int
        Number of Monte Carlo samples

    Returns
    -------
    float
        $E[f_B(x)]$
    """
    b_samples = private_features.sample(n_samples)
    probs = np.array([choice(revenue, b) for b in b_samples])
    return np.mean(probs)


def compute_expected_choice_derivative(
    x: float,
    choice: ChoiceFunction,
    private_features: PrivateFeatureDistribution,
    n_samples: int = 10000
) -> float:
    """
    Compute $E[f'_B(x)]$ via Monte Carlo.

    Average sensitivity of activation to revenue changes.

    Parameters
    ----------
    x : float
        Expected revenue
    choice : ChoiceFunction
        The choice function $f_b(\cdot)$
    private_features : PrivateFeatureDistribution
        Distribution of $B_i$
    n_samples : int
        Number of Monte Carlo samples

    Returns
    -------
    float
        $E[f'_B(x)]$
    """
    b_samples = private_features.sample(n_samples)
    derivs = np.array([choice.derivative(x, b) for b in b_samples])
    return np.mean(derivs)


@dataclass
class SupplierParameters:
    """
    Parameters defining the supplier population.

    Parameters
    ----------
    choice : ChoiceFunction
        The choice function $f_b(\cdot)$
    private_features : PrivateFeatureDistribution
        Distribution of $B_i$ (outside options/costs)
    """
    choice: ChoiceFunction
    private_features: PrivateFeatureDistribution


def compute_activation_probability(
    supplier_params: SupplierParameters,
    expected_revenue: float,
    n_monte_carlo: int
) -> float:
    """
    Compute $\mu = E[f_B(\text{expected\_revenue})]$.

    This is the fraction of suppliers who become active when
    expected revenue is `expected_revenue`.

    Parameters
    ----------
    params : SupplierParameters
        Supplier population parameters
    expected_revenue : float
        Expected revenue (payment $\times$ expected allocation)
    n_monte_carlo: int

    Returns
    -------
    float
        Activation probability $\mu$
    """
    return compute_expected_choice_probability(
        expected_revenue,
        supplier_params.choice,
        supplier_params.private_features,
        n_monte_carlo
    )


def compute_activation_sensitivity(
    supplier_params: SupplierParameters,
    expected_revenue: float,
    n_monte_carlo: int
) -> float:
    """
    Compute $E[f'_B(\text{expected\_revenue})]$.

    Measures how sensitive activation is to revenue changes.
    Used in computing the marginal response function.

    Parameters
    ----------
    expected_revenue : float
        Expected revenue
    params : SupplierParameters
        Supplier population parameters
    n_monte_carlo: int

    Returns
    -------
    float
        $E[f'_B(\text{expected\_revenue})]$
    """
    return compute_expected_choice_derivative(
        expected_revenue,
        supplier_params.choice,
        supplier_params.private_features,
        n_monte_carlo
    )


def sample_supplier_activations(
    n: int,
    payments: np.ndarray,
    expected_allocation: float,
    params: SupplierParameters,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Sample activation decisions for n suppliers.

    Args
    ----
    n : int
        Number of suppliers
    payments : np.ndarray
        Payment offered to each supplier (length n)
    expected_allocation : float
        Expected demand per active supplier $E[\Omega]$
    params : SupplierParameters
        Supplier population parameters
    seed : Optional[int]
        Random seed for reproducibility

    Returns
    -------
    np.ndarray
        Z: Array of activation decisions (0 or 1) for each supplier
    """
    if seed is not None:
        np.random.seed(seed)

    # Sample private costs
    costs = params.private_features.sample(n)

    # Compute expected revenue for each supplier
    expected_revenues = payments * expected_allocation

    # Compute activation probabilities
    probs = np.array([
        params.choice(rev, cost)
        for rev, cost in zip(expected_revenues, costs)
    ])

    # Sample activation decisions
    Z = (np.random.random(n) < probs).astype(int)

    return Z