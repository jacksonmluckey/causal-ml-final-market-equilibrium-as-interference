from typing import List, Optional, Union, Tuple
from dataclasses import dataclass
from .demand import DemandParameters, GlobalState
from .allocation import AllocationFunction
from .supplier import SupplierParameters
import numpy as np

# TODO figure out how to combine these
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

@dataclass
class GlobalExperimentData:
    """Data from one period of global experimentation."""
    n: int
    p: float
    D: int          # Realized demand
    T: int          # Active suppliers
    S: float        # Total demand served
    U: float        # Realized utility (revenue - payments)

@dataclass
class ExperimentResults:
    """Results from running an experiment"""
    final_payment: float
    total_demand,
    total_demand_served,
    total_active,
    payment: List[float]
    utility: List[float]
    state: Optional[List[GlobalState]] = None
    cumulative_regret: Optional[List[float]] = None
    weighted_average_payment: Optional[float] = None # Weighted average p̄_T = (2/(T(T+1))) Σ_{t=1}^T t·p_t (Corollary 8)


@dataclass
class ExperimentParams:
    T: int
    n: int
    p_init: float
    gamma: float
    allocation: AllocationFunction
    supplier_params: SupplierParameters
    demand: Union[float, DemandParameters]
    p_bounds: Tuple[float, float]
    rng: Optional[np.random.Generator] = None


@dataclass
class Experiment:
    params: ExperimentParams
    results: ExperimentResults
    #local_learning
    #global_learning