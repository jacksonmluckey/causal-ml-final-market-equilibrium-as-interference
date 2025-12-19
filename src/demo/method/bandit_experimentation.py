r"""
Shared Utilities for Bandit-Based Global Experimentation

This module contains common utilities used by different bandit strategies:
- Baseline (two-phase explore/exploit)
- Epsilon-greedy (per-timestep explore/exploit)
- Potentially in the future: UCB, Thompson sampling

These utilities help manage exploration/exploitation tradeoffs and
learn optimal payments from observed utility data.
"""

import numpy as np
from scipy.interpolate import UnivariateSpline
from typing import Tuple, Optional, List


def fit_utility_spline(
    payments: np.ndarray,
    utilities: np.ndarray,
    p_bounds: Tuple[float, float],
    smoothing: Optional[float] = None
) -> Tuple[UnivariateSpline, float]:
    r"""
    Fit a smooth spline to utility observations and find the maximizer.

    Parameters
    ----------
    payments : np.ndarray
        Array of payment values from exploration
    utilities : np.ndarray
        Array of corresponding utilities
    p_bounds : Tuple[float, float]
        Payment bounds [p_min, p_max]
    smoothing : Optional[float]
        Smoothing parameter for spline (if None, uses default)

    Returns
    -------
    spline : UnivariateSpline
        Fitted spline function
    p_optimal : float
        Learned optimal payment (maximizer of spline)
    """
    # Fit univariate spline
    if smoothing is None:
        # Use default smoothing based on number of points
        smoothing = len(payments) * 0.1

    # Sort by payment for spline fitting
    sort_idx = np.argsort(payments)
    p_sorted = payments[sort_idx]
    u_sorted = utilities[sort_idx]

    # Fit spline (k=3 for cubic spline if we have enough points)
    k = min(3, len(payments) - 1)
    spline = UnivariateSpline(p_sorted, u_sorted, k=k, s=smoothing)

    # Find maximizer by evaluating on a fine grid
    p_min, p_max = p_bounds
    p_grid = np.linspace(p_min, p_max, 1000)
    u_grid = spline(p_grid)

    # Find the payment with highest utility
    best_idx = np.argmax(u_grid)
    p_optimal = p_grid[best_idx]

    return spline, float(p_optimal)


def find_best_payment_from_history(
    payment_history: List[float],
    utility_history: List[float]
) -> float:
    """
    Find the payment with highest observed utility.

    Parameters
    ----------
    payment_history : List[float]
        List of payments tried
    utility_history : List[float]
        List of observed utilities

    Returns
    -------
    float
        Payment with highest utility
    """
    if len(utility_history) == 0:
        raise ValueError("No history available")

    best_idx = np.argmax(utility_history)
    return payment_history[best_idx]


def sample_exploration_payment(
    p_best: float,
    p_bounds: Tuple[float, float],
    strategy: str,
    step_size_pct: float,
    rng: np.random.Generator
) -> float:
    """
    Sample a new payment for exploration.

    Parameters
    ----------
    p_best : float
        Current best payment
    p_bounds : Tuple[float, float]
        Payment bounds [p_min, p_max]
    strategy : str
        Exploration strategy: "adaptive_step" or "uniform"
    step_size_pct : float
        Step size as percentage of p_best (for adaptive_step)
    rng : np.random.Generator
        Random number generator

    Returns
    -------
    float
        Sampled payment for exploration
    """
    p_min, p_max = p_bounds

    if strategy == "uniform":
        # Sample uniformly from full range
        return rng.uniform(p_min, p_max)

    elif strategy == "adaptive_step":
        # Step ±(step_size_pct * p_best) from current best
        step_size = step_size_pct * p_best

        # Randomly choose direction: +1 (larger) or -1 (smaller)
        direction = rng.choice([-1, 1])

        # Compute new payment
        p_explore = p_best + direction * step_size

        # Clip to bounds
        p_explore = np.clip(p_explore, p_min, p_max)

        return float(p_explore)

    else:
        raise ValueError(f"Unknown exploration strategy: {strategy}")


def compute_epsilon(
    epsilon: float,
    epsilon_decay: Optional[str],
    decay_rate: Optional[float],
    t: int,
    T: int
) -> float:
    """
    Compute epsilon at timestep t based on decay schedule.

    Parameters
    ----------
    epsilon : float
        Base exploration rate
    epsilon_decay : Optional[str]
        Decay type: None, "linear", or "exponential"
    decay_rate : Optional[float]
        Decay rate for exponential schedule
    t : int
        Current timestep (1-indexed)
    T : int
        Total timesteps

    Returns
    -------
    float
        Epsilon value for timestep t
    """
    if epsilon_decay is None:
        return epsilon
    elif epsilon_decay == "linear":
        # Linear decay: ε_t = ε * (1 - t/T)
        # Ensures ε_T = 0
        return epsilon * (1.0 - t / T)
    elif epsilon_decay == "exponential":
        # Exponential decay: ε_t = ε * exp(-decay_rate * t)
        if decay_rate is None:
            raise ValueError("decay_rate required for exponential decay")
        return epsilon * np.exp(-decay_rate * t)
    else:
        raise ValueError(f"Unknown epsilon_decay: {epsilon_decay}")
