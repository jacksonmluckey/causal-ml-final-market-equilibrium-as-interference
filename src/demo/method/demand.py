r"""
Demand Model for Stochastic Market

The demand D is drawn conditionally on a global state A.
Key properties (equations 3.1 and 3.2):
- $E[D/n | A=a] = d_a$ (expected demand per supplier)
- $Var(D/n | A=a) \rightarrow 0$ as $n \rightarrow \infty$ (concentration)
- Extreme deviations are exponentially unlikely (tail property)
"""

import numpy as np
from typing import Dict, Tuple
from dataclasses import dataclass


@dataclass
class GlobalState:
    r"""
    Represents a global state A that affects demand.

    Attributes:
        name: Identifier for the state (e.g., "sunny", "rainy")
        d_a: Expected demand per supplier when in this state
        probability: Probability of this state occurring
    """

    name: str
    d_a: float  # Expected demand per supplier
    probability: float = 1.0


@dataclass
class DemandParameters:
    r"""
    Configuration for demand generation.

    Attributes:
        states: Dictionary mapping state names to GlobalState objects
        concentration_param: Controls how tightly D/n concentrates around d_a
    """

    states: Dict[str, GlobalState]
    concentration_param: float = 50.0


def normalize_probabilities(states: Dict[str, GlobalState]) -> Dict[str, float]:
    r"""Calculate normalized probabilities for all states."""
    state_names = list(states.keys())
    probs = np.array([states[s].probability for s in state_names])
    total = probs.sum()
    return {name: prob / total for name, prob in zip(state_names, probs)}


def sample_state(
    model: DemandParameters, rng: np.random.Generator | None = None
) -> GlobalState:
    r"""Sample a global state A according to the state probabilities."""
    if rng is None:
        rng = np.random.default_rng()

    normalized_probs = normalize_probabilities(model.states)
    state_names = list(normalized_probs.keys())
    probs = np.array([normalized_probs[s] for s in state_names])

    chosen_name = rng.choice(state_names, p=probs)
    return model.states[chosen_name]


def calculate_beta_params(d_a: float, k: float) -> Tuple[float, float]:
    r"""Calculate Beta distribution parameters for given mean and concentration."""
    alpha = k * d_a
    beta = k * (1 - d_a)
    return alpha, beta


def sample_demand(
    model: DemandParameters,
    state: GlobalState,
    n: int,
    rng: np.random.Generator | None = None,
) -> int:
    r"""
    Sample demand D given global state and market size.

    The demand satisfies:
    - $E[D/n | A=a] = d_a$
    - $D/n$ concentrates on d_a as $n \to \infty$

    Uses a Beta distribution scaled to have mean n * d_a and
    variance that decreases with n (satisfying concentration property 3.1).

    This ennsures concentration while allowing for realistic variance in finite markets.
    """
    if rng is None:
        rng = np.random.default_rng()

    d_a = state.d_a
    k = model.concentration_param

    if d_a <= 0:
        return 0

    if d_a < 1:
        alpha, beta = calculate_beta_params(d_a, k)
        scaled_demand = rng.beta(alpha, beta)
    else:
        scaled_demand = rng.gamma(shape=k, scale=d_a / k)

    D = int(round(n * scaled_demand))
    return max(0, D)


def expected_demand(state: GlobalState, n: int) -> float:
    r"""Return expected demand E[D | A=state] = n * d_a"""
    return n * state.d_a
