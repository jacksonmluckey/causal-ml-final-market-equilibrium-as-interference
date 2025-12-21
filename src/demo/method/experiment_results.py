r"""
Unified data structures for local and global experimentation.

This module provides a single set of dataclasses for storing experiment
parameters, timepoint data, and results for both local and global learning
algorithms.
"""

from typing import List, Optional
from dataclasses import dataclass
import numpy as np
import polars as pl

from .demand import GlobalState
from .experiment import ExperimentParams


@dataclass
class TimePointData:
    r"""
    Data from a single time period.

    Works for both local and global experimentation. Local-specific fields
    (gradient estimates, perturbations) are None for global experiments.

    Attributes
    ----------
    t : int
        Time period (1-indexed)
    p : float
        Payment level used this period
    D : int
        Realized demand
    T : int
        Number of active suppliers
    S : float
        Total demand served
    U : float
        Realized utility (revenue - cost)
    state : Optional[GlobalState]
        Global state for this period (if using DemandParameters)
    gradient_estimate : Optional[float]
        Estimated utility gradient $\hat{\Gamma}_t$ (local only)
    delta_hat : Optional[float]
        Estimated marginal response $\hat{\Delta}$ (local only)
    upsilon_hat : Optional[float]
        Estimated supply gradient $\hat{\Upsilon}$ (local only)
    zeta : Optional[float]
        Perturbation magnitude $\zeta$ (local only)
    epsilon : Optional[np.ndarray]
        Payment perturbations $\varepsilon_i$ (local only, if store_detailed_data=True)
    Z : Optional[np.ndarray]
        Individual supplier activations (local only, if store_detailed_data=True)
    """

    t: int
    p: float
    D: int
    T: int
    S: float
    U: float
    state: Optional[GlobalState] = None
    gradient_estimate: Optional[float] = None
    delta_hat: Optional[float] = None
    upsilon_hat: Optional[float] = None
    zeta: Optional[float] = None
    epsilon: Optional[np.ndarray] = None
    Z: Optional[np.ndarray] = None

    @property
    def D_bar(self) -> float:
        r"""Scaled demand $\bar{D} = D/n$ (requires n from context)"""
        # This will be computed in DataFrame conversion
        raise NotImplementedError("Use n from experiment params")

    @property
    def Z_bar(self) -> float:
        r"""Scaled active supply $\bar{Z} = T/n$ (requires n from context)"""
        raise NotImplementedError("Use n from experiment params")


@dataclass
class ExperimentResults:
    r"""
    Results from running an experiment.

    Attributes
    ----------
    final_payment : float
        Final payment level after T periods
    weighted_average_payment : Optional[float]
        Weighted average $\bar{p}_T = \frac{2}{T(T+1)} \sum_{t=1}^T t \cdot p_t$ (local only, Corollary 8)
    average_payment : float
        Simple average of payment levels
    timepoints : List[TimePointData]
        Data from each time period
    total_utility : float
        Sum of utilities across all periods
    mean_utility : float
        Average utility per period
    cumulative_regret : Optional[List[float]]
        Cumulative regret over time (if computed)
    convergence_metrics : Optional[dict]
        Convergence analysis results (if computed)
    """

    final_payment: float
    weighted_average_payment: Optional[float]
    average_payment: float
    timepoints: List[TimePointData]
    total_utility: float
    mean_utility: float
    cumulative_regret: Optional[List[float]] = None
    convergence_metrics: Optional[dict] = None


@dataclass
class Experiment:
    r"""
    Complete experiment with parameters and results.

    Attributes
    ----------
    params : ExperimentParams
        Experiment configuration
    results : ExperimentResults
        Experiment outcomes
    """

    params: ExperimentParams
    results: ExperimentResults

    def to_polars(self) -> pl.DataFrame:
        r"""
        Convert experiment to Polars DataFrame.

        Returns
        -------
        pl.DataFrame
            DataFrame with one row per timepoint, including:
            - t: time period
            - p: payment
            - D, T, S, U: demand, active suppliers, served, utility
            - D_bar, Z_bar: scaled versions
            - gradient_estimate, delta_hat, upsilon_hat (local only)
            - state columns (if using DemandParameters)
        """
        return experiment_to_dataframe(self)

    def compute_regret(self, p_optimal: float) -> List[float]:
        r"""
        Compute cumulative regret against optimal payment.

        Parameters
        ----------
        p_optimal : float
            True optimal payment level

        Returns
        -------
        List[float]
            Cumulative regret at each timepoint
        """
        regret = compute_cumulative_regret(self, p_optimal)
        self.results.cumulative_regret = regret
        return regret

    def convergence_analysis(self, p_optimal: float) -> dict:
        r"""
        Analyze convergence properties.

        Parameters
        ----------
        p_optimal : float
            True optimal payment level

        Returns
        -------
        dict
            Convergence metrics including final error, weighted error,
            mean squared error, and weighted regret
        """
        metrics = analyze_convergence(self, p_optimal)
        self.results.convergence_metrics = metrics
        return metrics


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def experiment_to_dataframe(experiment: Experiment) -> pl.DataFrame:
    r"""
    Convert Experiment to Polars DataFrame.

    Parameters
    ----------
    experiment : Experiment
        The experiment to convert

    Returns
    -------
    pl.DataFrame
        DataFrame with timepoint data and computed columns
    """
    timepoints = experiment.results.timepoints
    n = experiment.params.n

    # Extract data from timepoints
    data = {
        "t": [tp.t for tp in timepoints],
        "p": [tp.p for tp in timepoints],
        "D": [tp.D for tp in timepoints],
        "T": [tp.T for tp in timepoints],
        "S": [tp.S for tp in timepoints],
        "U": [tp.U for tp in timepoints],
    }

    # Add scaled versions
    data["D_bar"] = [tp.D / n for tp in timepoints]
    data["Z_bar"] = [tp.T / n for tp in timepoints]

    # Add local-specific columns if present
    if experiment.params.experiment_type == "local":
        data["gradient_estimate"] = [tp.gradient_estimate for tp in timepoints]
        data["delta_hat"] = [tp.delta_hat for tp in timepoints]
        data["upsilon_hat"] = [tp.upsilon_hat for tp in timepoints]
        data["zeta"] = [tp.zeta for tp in timepoints]

    # Add state columns if using DemandParameters
    if experiment.params.demand_params is not None:
        data["state_d_a"] = [tp.state.d_a if tp.state else None for tp in timepoints]
        data["state_name"] = [tp.state.name if tp.state else None for tp in timepoints]

    # Add experiment metadata as constants
    data["experiment_type"] = experiment.params.experiment_type
    data["n"] = n

    df = pl.DataFrame(data, strict=False)

    # Add cumulative regret if computed
    if experiment.results.cumulative_regret is not None:
        df = df.with_columns(
            pl.Series("cumulative_regret", experiment.results.cumulative_regret)
        )

    return df


def compare_experiments(
    local: Experiment, global_: Experiment, p_optimal: Optional[float] = None
) -> pl.DataFrame:
    r"""
    Create comparison DataFrame for local vs global experiments.

    Parameters
    ----------
    local : Experiment
        Local experimentation results
    global_ : Experiment
        Global experimentation results
    p_optimal : Optional[float]
        True optimal payment (for computing regret)

    Returns
    -------
    pl.DataFrame
        Combined DataFrame with 'method' column distinguishing experiments
    """
    df_local = local.to_polars().with_columns(pl.lit("local").alias("method"))
    df_global = global_.to_polars().with_columns(pl.lit("global").alias("method"))

    # Concatenate with diagonal strategy to handle different columns
    df = pl.concat([df_local, df_global], how="diagonal")

    # Add regret if optimal payment provided
    if p_optimal is not None:
        df = df.with_columns(((pl.col("p") - p_optimal) ** 2).alias("squared_error"))

    return df


def compute_cumulative_regret(experiment: Experiment, p_optimal: float) -> List[float]:
    r"""
    Compute cumulative regret over time.

    Regret at time t is defined as the cumulative squared error:
        $R_t = \sum_{s=1}^t (p_s - p^*)^2$

    Parameters
    ----------
    experiment : Experiment
        The experiment
    p_optimal : float
        True optimal payment

    Returns
    -------
    List[float]
        Cumulative regret at each timepoint
    """
    payments = [tp.p for tp in experiment.results.timepoints]
    squared_errors = [(p - p_optimal) ** 2 for p in payments]
    cumulative_regret = np.cumsum(squared_errors).tolist()
    return cumulative_regret


def analyze_convergence(experiment: Experiment, p_optimal: float) -> dict:
    r"""
    Analyze convergence properties of the learning algorithm.

    Parameters
    ----------
    experiment : Experiment
        The experiment results
    p_optimal : float
        True optimal payment level

    Returns
    -------
    dict
        Convergence metrics:
        - final_payment: Last payment level
        - weighted_average: Weighted average (local only)
        - optimal_payment: Target payment
        - final_error: $|p_T - p^*|$
        - weighted_error: $|\bar{p}_T - p^*|$ (local only)
        - mean_squared_error: Mean of $(p_t - p^*)^2$
        - weighted_regret: Weighted sum of squared errors
    """
    payments = np.array([tp.p for tp in experiment.results.timepoints])
    T = len(payments)

    # Compute errors
    errors = payments - p_optimal
    squared_errors = errors**2

    # Weighted metrics (for local experiments using Theorem 7)
    weights = np.arange(1, T + 1)
    weight_sum = T * (T + 1) / 2
    weighted_regret = np.sum(weights * squared_errors) / weight_sum

    metrics = {
        "final_payment": experiment.results.final_payment,
        "optimal_payment": p_optimal,
        "final_error": abs(experiment.results.final_payment - p_optimal),
        "mean_squared_error": float(np.mean(squared_errors)),
        "weighted_regret": float(weighted_regret),
    }

    # Add weighted average error for local experiments
    if experiment.results.weighted_average_payment is not None:
        metrics["weighted_average"] = experiment.results.weighted_average_payment
        metrics["weighted_error"] = abs(
            experiment.results.weighted_average_payment - p_optimal
        )

    return metrics
