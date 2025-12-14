from dataclasses import dataclass
import numpy as np
from typing import Optional
from .supplier import SupplierParameters
from .allocation import AllocationFunction
from .demand import DemandParameters


@dataclass
class SimulationParameters:
    mean_field: bool = True
    n_monte_carlo: int = 10000
    rng: Optional[np.random.Generator] = None


@dataclass
class Simulation:
    simulation_params: SimulationParameters
    supplier_params: SupplierParameters
    demand_params: DemandParameters
    allocation_func: AllocationFunction
    gamma: float