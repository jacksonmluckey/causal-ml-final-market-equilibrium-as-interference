"""
Market Platform Simulation

This module implements market simulation functionality.
"""

import numpy as np
from typing import Optional
from dataclasses import dataclass

from .allocation import (
    compute_finite_allocation,
    compute_expected_allocation,
)
from .supplier import (
    SupplierParameters,
    sample_supplier_activations,
)
from .market import MarketParameters
from .find_equilibrium import find_equilibrium_supply_mu


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


def simulate_market_period(
    params: MarketParameters,
    n: int,
    p: float,
    zeta: float = 0.0,
    rng: Optional[np.random.Generator] = None
) -> MarketOutcome:
    """
    Simulate one period of the market.

    Args:
        params: Market parameters
        n: Number of potential suppliers
        p: Base payment level
        zeta: Payment perturbation magnitude (for local experimentation)
        rng: Random number generator

    Returns:
        MarketOutcome with simulation results
    """
    if rng is None:
        rng = np.random.default_rng()

    # Generate demand
    D = int(round(n * params.d_a))  # For simplicity, use deterministic demand

    # Generate payments (with possible perturbation)
    if zeta > 0:
        epsilon = rng.choice([-1, 1], size=n)
        payments = p + zeta * epsilon
    else:
        payments = np.full(n, p)
        epsilon = None

    # Compute equilibrium expected allocation
    mu_eq = find_equilibrium_supply_mu(p, params)
    q_eq = compute_expected_allocation(params.allocation, mu_eq, params.d_a)

    # Create SupplierParameters for sampling activations
    supplier_params = SupplierParameters(
        choice=params.choice,
        private_features=params.private_features,
        n_monte_carlo=params.n_monte_carlo
    )

    # Suppliers make activation decisions
    Z = sample_supplier_activations(
        n,
        payments,
        q_eq,
        supplier_params
    )
    T = Z.sum()  # Number of active suppliers

    # Compute actual allocations
    if T > 0:
        # In practice, allocations would be random
        # For simplicity, use expected allocation
        actual_q = compute_finite_allocation(params.allocation, D, T)
        S = np.where(Z == 1, actual_q, 0.0)
    else:
        S = np.zeros(n)

    # Compute outcomes
    total_served = S.sum() if T > 0 else 0.0
    revenue = params.gamma * total_served
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
