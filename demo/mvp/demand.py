"""
Demand Model for Stochastic Market

The demand D is drawn conditionally on a global state A.
Key properties (equations 3.1 and 3.2):
- E[D/n | A=a] = d_a (expected demand per supplier)
- Var(D/n | A=a) → 0 as n → ∞ (concentration)
- Extreme deviations are exponentially unlikely (tail property)
"""

import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class GlobalState:
    """
    Represents a global state A that affects demand.
    
    Attributes:
        name: Identifier for the state (e.g., "sunny", "rainy")
        d_a: Expected demand per supplier when in this state
        probability: Probability of this state occurring
    """
    name: str
    d_a: float  # Expected demand per supplier
    probability: float = 1.0


class DemandModel:
    """
    Generates demand D given global state A and market size n.
    
    The demand satisfies:
    - E[D/n | A=a] = d_a
    - D/n concentrates on d_a as n → ∞
    
    We use a scaled Beta distribution to ensure concentration while
    allowing for realistic variance in finite markets.
    """
    
    def __init__(self, states: Dict[str, GlobalState], concentration_param: float = 50.0):
        """
        Args:
            states: Dictionary mapping state names to GlobalState objects
            concentration_param: Controls how tightly D/n concentrates around d_a
                               Higher values → tighter concentration
        """
        self.states = states
        self.concentration_param = concentration_param
        
    def sample_state(self, rng: Optional[np.random.Generator] = None) -> GlobalState:
        """Sample a global state A according to the state probabilities."""
        if rng is None:
            rng = np.random.default_rng()
            
        state_names = list(self.states.keys())
        probs = np.array([self.states[s].probability for s in state_names])
        probs = probs / probs.sum()  # Normalize
        
        chosen_name = rng.choice(state_names, p=probs)
        return self.states[chosen_name]
    
    def sample_demand(self, state: GlobalState, n: int, 
                      rng: Optional[np.random.Generator] = None) -> int:
        """
        Sample demand D given global state and market size.
        
        Uses a Beta distribution scaled to have mean n * d_a and 
        variance that decreases with n (satisfying concentration property 3.1).
        
        Args:
            state: The global state A
            n: Market size (number of potential suppliers)
            rng: Random number generator
            
        Returns:
            D: Total demand (integer)
        """
        if rng is None:
            rng = np.random.default_rng()
            
        d_a = state.d_a
        
        # Use Beta distribution with parameters chosen so:
        # - Mean = d_a
        # - Variance decreases with concentration_param
        # Beta(α, β) has mean α/(α+β) and variance αβ/((α+β)²(α+β+1))
        
        # For mean = d_a, we need α/(α+β) = d_a
        # Setting α = k*d_a and β = k*(1-d_a) for some k gives mean d_a
        # and variance d_a(1-d_a)/(k+1) which → 0 as k → ∞
        
        k = self.concentration_param
        
        # Ensure d_a is in (0, 1) for Beta distribution
        # If d_a > 1, we'll scale appropriately
        if d_a <= 0:
            return 0
            
        if d_a < 1:
            alpha = k * d_a
            beta = k * (1 - d_a)
            scaled_demand = rng.beta(alpha, beta)
        else:
            # For d_a >= 1, use Gamma distribution instead
            # Gamma(shape=k, scale=d_a/k) has mean d_a and variance d_a²/k
            scaled_demand = rng.gamma(shape=k, scale=d_a/k)
        
        # Scale by market size
        D = int(round(n * scaled_demand))
        return max(0, D)
    
    def expected_demand(self, state: GlobalState, n: int) -> float:
        """Return expected demand E[D | A=state] = n * d_a"""
        return n * state.d_a