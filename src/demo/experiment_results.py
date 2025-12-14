from typing import List, Optional, Union, Tuple
from dataclasses import dataclass
from .demand import DemandParameters, GlobalState
from .allocation import AllocationFunction
from .supplier import SupplierParameters
import numpy as np


@dataclass
class ExperimentResults:
    """Results from running an experiment"""
    final_payment: float
    payment_history: List[float]
    utility_history: List[float]
    state_history: Optional[List[GlobalState]] = None
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