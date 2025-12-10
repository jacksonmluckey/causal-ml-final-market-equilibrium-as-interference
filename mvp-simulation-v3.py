"""
experimenting_equilibrium.py
============================

Python implementation of the core estimators and optimisation routine from  
“Experimenting in Equilibrium: The Surprising Success of A/B Testing
in Stochastic Systems with Interference”  
(*Wager & Xu, 2021*).

The code is organised as a small, *composable* toolkit:

1. Symmetric (mean–zero) payment perturbations — Eq (2.1)/(3.11)  
2. Estimation of the **marginal response function** Δ  — Eq (4.1)  
3. Translation of Δ to an estimate of the **utility gradient** Γ — Eq (4.2–4.3)  
4. First-order optimisation (mirror-descent variant) — Eq (4.5)  
5. Lightweight simulation helpers (optional) to sanity-check the pipeline  
   on synthetic mean-field markets

All numerical work is vectorised with NumPy and each public function contains a
detailed docstring that cites the exact part of the paper it realises.

Notation mirrors the paper:
- *n*      … number of suppliers
- *ζ* (zeta) … perturbation magnitude (small, e.g. 0.01)
- *ε* (eps) … Rademacher ±1 noise
- *p*      … baseline payment
- *Zᵢ*     … indicator that supplier *i* is active
- *D*      … total demand (exogenous)
- *ω(·)*   … allocation/match-rate function (mean-field)
"""
from __future__ import annotations

import numpy as np
from typing import Callable, Tuple, Sequence

# ---------------------------------------------------------------------
# 1.  Unobtrusive randomisation: symmetric perturbations (Eq 2.1 / 3.11)
# ---------------------------------------------------------------------
def symmetric_perturbations(
    n: int,
    p: float,
    zeta: float,
    rng: np.random.Generator | None = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate ζ-perturbed payments   Pᵢ = p + ζ εᵢ   with   εᵢ ∈ {−1, +1} iid
    (Eq (2.1) and formalised in Eq (3.11)).

    Parameters
    ----------
    n     : int
        Number of suppliers.
    p     : float
        Baseline payment.
    zeta  : float
        Perturbation magnitude (small, >0).
    rng   : np.random.Generator | None
        Random generator (optional, for reproducibility).

    Returns
    -------
    P : (n,) ndarray
        Perturbed payments.
    eps : (n,) ndarray
        The Rademacher vector ε used (mean ≈ 0).
    """
    if rng is None:
        rng = np.random.default_rng()
    eps = rng.integers(0, 2, size=n, dtype=np.int8) * 2 - 1  # ±1
    P = p + zeta * eps
    return P.astype(np.float64), eps


# ----------------------------------------------------------
# 2.  Estimate marginal response  Δ  (reduced-form, Eq 4.1)
# ----------------------------------------------------------
def estimate_delta(
    Z: np.ndarray,
    eps: np.ndarray,
    zeta: float,
    centered: bool = True,
) -> float:
    r"""
    Compute   $\hat{\Delta}$   — the regression coefficient of  Zᵢ  on  ζ εᵢ  
    as in Eq (4.1):

        \hat{Δ} = ζ⁻¹ ⋅ Cov(Z, ε) / Var(ε)

    If *centered* is True (default) we subtract sample means
    (the paper subtracts \bar{ε}, but for ε ∈ {−1,+1} mean≈0 anyway).

    Parameters
    ----------
    Z    : (n,) ndarray
        Binary (or fractional) supplier activity indicators.
    eps  : (n,) ndarray
        Rademacher perturbations (±1).
    zeta : float
        Perturbation magnitude used when generating payments.
    centered : bool
        Whether to demean Z and ε before regression.

    Returns
    -------
    delta_hat : float
        Estimate of marginal response Δ̂.
    """
    if Z.shape != eps.shape:
        raise ValueError("Shapes of Z and eps must match.")
    if centered:
        Zc = Z - Z.mean()
        eps_c = eps - eps.mean()
    else:
        Zc, eps_c = Z, eps
    cov = np.mean(Zc * eps_c)
    var = np.mean(eps_c ** 2)
    return cov / (var * zeta)


# ------------------------------------------------------------------
# 3.  From Δ̂ to utility-gradient Γ̂   (Eq 4.2–4.3, mean-field theory)
# ------------------------------------------------------------------
def estimate_gradient(
    sD: float,
    sZ: float,
    delta_hat: float,
    omega: Callable[[float], float],
    domega: Callable[[float], float],
) -> float:
    """
    Implements Eq (4.2–4.3).  Translates the marginal response Δ̂,
    observed demand rate sD (=D/n) and active-supply rate sZ (=T/n)
    into an estimate of the *true* utility gradient Γ̂ that accounts
    for equilibrium feedback.

    Paper variables → function args
    --------------------------------
    sD          = 𝑠_D  in Eq (4.2–4.3)
    sZ          = 𝑠_Z
    delta_hat   = \hat{Δ}
    ω(·)        = ω
    ω′(·)       = ω₁

    Parameters
    ----------
    sD, sZ : float
        Scaled demand and supply (per supplier).
    delta_hat : float
        Marginal response estimator from `estimate_delta`.
    omega, domega : Callable
        Mean-field matching function ω and its derivative ω′.

    Returns
    -------
    gamma_hat : float
        Estimated gradient Γ̂_t (Eq 4.3) used for first-order updates.
    """
    # Eq 4.2 -- Υ̂
    denom = 1.0 + (sD * delta_hat * domega(sD / sZ)) / (sZ ** 2 * omega(sD / sZ))
    upsilon_hat = delta_hat / denom

    # Eq 4.3 -- Γ̂
    ratio = sD / sZ
    gamma_hat = upsilon_hat * (
        (omega(ratio) - sZ)  # r(·) − ω(·) collapses to same term under mean-field
        - (domega(ratio) * ratio - domega(ratio)) * ratio
        - omega(ratio) * sZ
    )
    return gamma_hat


# -------------------------------------------------------------------------
# 4.  Mirror-descent update with noisy gradients (optimiser, Eq 4.5)
# -------------------------------------------------------------------------
def mirror_descent_step(
    p_hist: Sequence[float],
    grad_hist: Sequence[float],
    eta: float,
    interval: Tuple[float, float],
) -> float:
    """
    One mirror-descent payment update enforcing  p ∈ I = [c⁻, c⁺]
    (Eq (4.5) — the argmin closed form of the quadratic + indicator).

    The cumulative gradient θ_t = Σ_{s≤t} Γ̂_s is implicit in grad_hist.

    Parameters
    ----------
    p_hist    : list/array of previous payments  [p₁, …, p_t]
    grad_hist : list/array of past gradients     [Γ̂₁, …, Γ̂_t]
    eta       : float  (learning-rate as in paper)
    interval  : (c_minus, c_plus)
        Feasible payment support.

    Returns
    -------
    p_next : float
        Next-period payment p_{t+1}.
    """
    c_minus, c_plus = interval
    t = len(grad_hist)
    if t == 0:
        raise ValueError("grad_hist must contain at least one element.")
    theta_t = np.sum(grad_hist)
    # Eq 4.5 minimises (1/2η) Σ (p - p_s)^2  - θ_t p  s.t. p ∈ I
    # This is equivalent to projecting   μ = (η θ_t + Σ p_s) / (t + η⁻¹)
    # onto the interval.
    mu = (eta * theta_t + np.sum(p_hist)) / (t + eta**-1)
    return float(np.clip(mu, c_minus, c_plus))


# ------------------------------------------------------------
# 5. Optional helpers for synthetic sanity-check / debugging
# ------------------------------------------------------------
def mean_field_omega(load: float, k: float = 1.0) -> float:
    """
    Simple M/M/∞-style matching rate:   ω(λ) = 1 / (1 + k λ)

    Useful as a placeholder for unknown allocation rules.
    """
    return 1.0 / (1.0 + k * load)


def d_mean_field_omega(load: float, k: float = 1.0) -> float:
    """Derivative of `mean_field_omega` wrt load."""
    return -k / (1.0 + k * load) ** 2


def check_rademacher(eps: np.ndarray, atol: float = 1e-2) -> None:
    """
    Verify that ε is near mean-zero & takes only ±1 values.
    """
    unique = np.unique(eps)
    assert set(unique).issubset({-1, 1}), "ε must be ±1 only."
    assert abs(eps.mean()) < atol, "ε not mean-zero (increase n?)."


# ---------------------------------------------------------------------
# Example: end-to-end one-period estimation (use small n for demo only)
# ---------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(42)

    # --- synthetic market parameters ---------------------------------
    n = 100000
    p_baseline = 1.0
    zeta = 0.02
    demand_rate = 0.8              # exogenous D / n
    true_participation_prob = 0.6  # so µ = 0.6

    # --- 1) randomise payments ---------------------------------------
    P, eps = symmetric_perturbations(n, p_baseline, zeta, rng)
    check_rademacher(eps)

    # --- 2) simulate supplier behaviour ------------------------------
    # Toy choice model: Zᵢ ~ Bernoulli(µ + β ζ εᵢ), here β = 0.05
    beta = 0.05
    Z = rng.binomial(1, true_participation_prob + beta * zeta * eps)

    # --- 3) estimate Δ and Γ -----------------------------------------
    delta_hat = estimate_delta(Z, eps, zeta)
    sZ = Z.mean()
    sD = demand_rate
    gamma_hat = estimate_gradient(
        sD, sZ, delta_hat,
        omega=lambda l: mean_field_omega(l, k=1.5),
        domega=lambda l: d_mean_field_omega(l, k=1.5),
    )

    print(f"Δ̂  = {delta_hat:.4f}")
    print(f"Γ̂  = {gamma_hat:.4f}")

    # --- 4) perform payment update -----------------------------------
    p_hist = [p_baseline]
    grad_hist = [gamma_hat]
    p_next = mirror_descent_step(p_hist, grad_hist, eta=0.1,
                                 interval=(0.5, 1.5))
    print(f"Next payment p₂ = {p_next:.4f}")