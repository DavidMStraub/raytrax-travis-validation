"""Generic geometry and profile sampling for ECRH ray tracing validation scans.

All W7-X-specific bounds live in ``w7x_setup``.  This module only contains
the geometry-independent sampling machinery; callers pass bounds as arguments.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from profiles import cylindrical_to_cartesian
from step1_trajectory import ScenarioParams

# Profile parameter ranges (shared defaults; callers may override)
NE_CENTRAL_RANGE = (0.05, 0.8)   # 10²⁰ m⁻³  (avoid near-zero and very high densities)
TE_CENTRAL_RANGE = (1.0, 10.0)   # keV
A_RANGE          = (0.0, 0.05)   # small edge pedestal only
P_RANGE          = (1.5, 4.0)    # physically reasonable profile widths
Q_RANGE          = (1.5, 4.0)    # q>=1.5 ensures finite slope at LCFS (avoids divergence)


def _eval_rho(rho_interp, R: float, phi_rad: float, Z: float, nfp: int = 5) -> float:
    """Evaluate ρ at (R, φ, Z), applying stellarator symmetry fold."""
    period     = 2 * np.pi / nfp
    half       = period / 2
    phi_mod    = phi_rad % period
    in_second  = phi_mod > half
    phi_mapped = period - phi_mod if in_second else phi_mod
    z_query    = -Z if in_second else Z
    return float(rho_interp(R, phi_mapped, z_query))


def compute_frequency_range(
    B_interp,
    rho_interp,
    r_range: tuple[float, float] = (5.0, 6.2),
    z_max: float = 0.5,
    nfp: int = 5,
    margin_low: float = 0.10,
    margin_high: float = 0.85,
) -> tuple[float, float]:
    """Scan B on a coarse grid and return (f_min_ghz, f_max_ghz) safely below the fundamental EC resonance.

    Args:
        B_interp: Magnetic field vector interpolator.
        rho_interp: ρ interpolator.
        r_range: (R_min, R_max) scan range [m].
        z_max: Maximum |Z| for scan [m].
        nfp: Number of field periods (for symmetry fold).
        margin_low: Fraction of f_ce_min for the lower bound.
        margin_high: Fraction of f_ce_min for the upper bound.

    Returns:
        ``(f_min_ghz, f_max_ghz)`` sub-O1 frequency range.
    """
    import jax.numpy as jnp

    half_period = np.pi / nfp
    b_min = np.inf
    for R in np.linspace(r_range[0], r_range[1], 15):
        for phi_mapped in np.linspace(0, half_period, 8):
            for Z in np.linspace(-z_max, z_max, 10):
                rho = float(rho_interp(R, phi_mapped, Z))
                if 0.0 < rho < 1.0:
                    B = float(jnp.linalg.norm(B_interp(R, phi_mapped, Z)))
                    b_min = min(b_min, B)

    f_ce_min_ghz = 28.0 * b_min
    return margin_low * f_ce_min_ghz, margin_high * f_ce_min_ghz


def sample_antenna(
    rho_interp,
    rng: np.random.Generator,
    r_min: float,
    r_max: float,
    z_max: float,
    nfp: int = 5,
    max_tries: int = 500,
) -> tuple[float, float, float]:
    """Sample (R, φ_deg, Z) in the launcher shell, outside the plasma (ρ > 1.05)."""
    for _ in range(max_tries):
        R       = rng.uniform(r_min, r_max)
        phi_deg = rng.uniform(0.0, 360.0)
        Z       = rng.uniform(-z_max, z_max)
        if _eval_rho(rho_interp, R, np.deg2rad(phi_deg), Z, nfp) > 1.05:
            return (R, phi_deg, Z)
    raise RuntimeError("Could not sample an outside-plasma antenna position after many tries.")


def sample_target(
    rho_interp,
    rng: np.random.Generator,
    r_min: float,
    r_max: float,
    z_max: float,
    rho_max: float,
    nfp: int = 5,
    max_tries: int = 500,
) -> np.ndarray:
    """Sample a Cartesian target point inside the plasma (ρ < rho_max)."""
    for _ in range(max_tries):
        R       = rng.uniform(r_min, r_max)
        phi_deg = rng.uniform(0.0, 360.0)
        Z       = rng.uniform(-z_max, z_max)
        if _eval_rho(rho_interp, R, np.deg2rad(phi_deg), Z, nfp) < rho_max:
            phi_rad = np.deg2rad(phi_deg)
            return np.array([R * np.cos(phi_rad), R * np.sin(phi_rad), Z])
    raise RuntimeError("Could not sample an inside-plasma target point after many tries.")


def sample_profiles(
    rng: np.random.Generator,
    ne_central_range: tuple[float, float] = NE_CENTRAL_RANGE,
    te_central_range: tuple[float, float] = TE_CENTRAL_RANGE,
    a_range:          tuple[float, float] = A_RANGE,
    p_range:          tuple[float, float] = P_RANGE,
    q_range:          tuple[float, float] = Q_RANGE,
) -> tuple:
    """Sample random (ne_central, ne_parm, te_central, te_parm)."""
    ne_central = rng.uniform(*ne_central_range)
    te_central = rng.uniform(*te_central_range)
    ne_parm = (rng.uniform(*a_range), rng.uniform(*p_range), rng.uniform(*q_range), 0.0, 0.0)
    te_parm = (rng.uniform(*a_range), rng.uniform(*p_range), rng.uniform(*q_range), 0.0, 0.0)
    return ne_central, ne_parm, te_central, te_parm


def sample_scenario(
    rho_interp,
    f_min_ghz: float,
    f_max_ghz: float,
    rng: np.random.Generator,
    *,
    antenna_r_min: float,
    antenna_r_max: float,
    antenna_z_max: float,
    target_r_min:  float,
    target_r_max:  float,
    target_z_max:  float,
    target_rho_max: float,
    ne_central_range: tuple[float, float] = NE_CENTRAL_RANGE,
    te_central_range: tuple[float, float] = TE_CENTRAL_RANGE,
    a_range:          tuple[float, float] = A_RANGE,
    p_range:          tuple[float, float] = P_RANGE,
    q_range:          tuple[float, float] = Q_RANGE,
    nfp:       int   = 5,
    b0_target: float = 3.0,
    power_mw:  float = 1.0,
    max_tries: int   = 500,
) -> ScenarioParams:
    """Sample one random ScenarioParams."""
    antenna_cyl = sample_antenna(rho_interp, rng, antenna_r_min, antenna_r_max, antenna_z_max, nfp)
    target_cart = sample_target(rho_interp, rng, target_r_min, target_r_max, target_z_max,
                                target_rho_max, nfp)

    antenna_cart = np.array(cylindrical_to_cartesian(*antenna_cyl))
    direction    = target_cart - antenna_cart
    direction   /= np.linalg.norm(direction)

    frequency_ghz = float(rng.uniform(f_min_ghz, f_max_ghz))
    f_ce0 = 28.0 * b0_target  # electron cyclotron frequency on axis [GHz]

    # Rejection-sample (mode, profiles) until no cutoff layer exists in the plasma.
    #
    # O-mode cutoff:  f < f_pe  →  ne_central > (f / 89.8)²  [1e20 m⁻³]
    # X-mode L-cutoff: f < f_L = ½(−f_ce + √(f_ce² + 4·f_pe²))
    #   evaluated at (B₀, ne_central) as a conservative worst-case.
    #   Both depend only on the sampled ne_central and the fixed f, B₀.
    for _ in range(max_tries):
        mode: Literal["O", "X"] = "O" if rng.random() < 0.5 else "X"
        ne_central, ne_parm, te_central, te_parm = sample_profiles(
            rng, ne_central_range, te_central_range, a_range, p_range, q_range
        )
        f_pe = 8.98 * np.sqrt(ne_central * 100)
        if mode == "O" and frequency_ghz < 1.1 * f_pe:   # 10 % margin
            continue
        if mode == "X":
            f_L = 0.5 * (-f_ce0 + np.sqrt(f_ce0**2 + 4 * f_pe**2))
            if frequency_ghz < 1.1 * f_L:                # 10 % margin
                continue
        break
    else:
        raise RuntimeError(
            f"Could not sample a non-cutoff (mode, profile) pair after {max_tries} tries "
            f"(f={frequency_ghz:.1f} GHz, n_crit={n_crit_1e20:.3f}×10²⁰ m⁻³)."
        )

    return ScenarioParams(
        antenna_cyl=tuple(antenna_cyl),
        direction_cart=tuple(float(x) for x in direction),
        target_cart=tuple(float(x) for x in target_cart),
        mode=mode,
        b0_target=b0_target,
        frequency_ghz=frequency_ghz,
        power_mw=power_mw,
        ne_central=float(ne_central),
        ne_parm=tuple(float(x) for x in ne_parm),
        te_central=float(te_central),
        te_parm=tuple(float(x) for x in te_parm),
    )
