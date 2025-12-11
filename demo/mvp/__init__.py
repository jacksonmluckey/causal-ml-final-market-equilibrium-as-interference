"""
Stochastic Market Model

Implementation of the model from:
Wager & Xu (2021) "Experimenting in Equilibrium"

This package implements a centralized marketplace model where:
- A platform sets payments to attract suppliers
- Suppliers decide whether to become active based on expected revenue
- Demand is randomly allocated among active suppliers
- The platform aims to maximize utility (revenue - payments)

Main components:
- demand: Demand generation given global state
- allocation: Regular allocation functions ω(x)
- supplier: Supplier choice behavior
- platform: Platform utility and market simulation
"""

from .utils import numerical_derivative

from .demand import DemandModel, GlobalState

from .allocation import (
    AllocationFunction, 
    QueueAllocation, 
    LinearAllocation, 
    SmoothLinearAllocation
)
#from .supplier import (
#    ChoiceFunction,
#    LogisticChoice,
#    CostDistribution,
#    LogNormalCosts,
#    UniformCosts,
#    SupplierPopulation
#)
from .market_platform import (
    RevenueFunction,
    LinearRevenue,
    PlatformUtility,
    Market,
    MarketOutcome
)

__all__ = [
    #'numerical_derivative',
    'DemandModel', 'GlobalState',
    'AllocationFunction', 'QueueAllocation', 'LinearAllocation', 'SmoothLinearAllocation',
    #'ChoiceFunction', 'LogisticChoice',
    #'CostDistribution', 'LogNormalCosts', 'UniformCosts', 'SupplierPopulation',
    'RevenueFunction', 'LinearRevenue', 'PlatformUtility', 'Market', 'MarketOutcome'
]