"""Step 2: Absorption validation — α, τ, deposited power.

Scenarios
─────────
  X1 / O1  — frequency near the fundamental EC resonance (f ≈ f_ce inside
               the plasma).  The 2nd-harmonic resonant B = f/(2·28) lies well
               below B_min inside the plasma → no 2nd-harmonic absorption.

  X2 / O2  — frequency near the 2nd harmonic (f ≈ 2·f_ce inside the plasma).
               The 1st-harmonic resonant B = f/28 lies well above B_max inside
               the plasma → no fundamental absorption.

Both codes use the cold-plasma Hamiltonian for ray tracing; absorption is
always computed relativistically in both codes.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from travis import TravisECRHInput, TravisECRHOutput, run_travis, resolve_travis_exe
from profiles import build_profiles, cylindrical_to_cartesian, interpolate_to_arc
from sampling import sample_antenna, sample_target, sample_profiles, _eval_rho

from raytrax.api import trace
from raytrax.equilibrium.interpolate import (
    MagneticConfiguration,
    build_magnetic_field_interpolator,
    build_rho_interpolator,
    build_electron_density_profile_interpolator,
    build_electron_temperature_profile_interpolator,
)
from raytrax.physics.absorption import absorption_coefficient as rx_absorption_coefficient
from raytrax.tracer.buffers import Interpolators
from raytrax.tracer.solver import _eval_magnetic_field, _eval_rho as _eval_rho_rx
from raytrax.types import Beam, RadialProfiles, TraceResult, TracerSettings


# ── Thresholds ───────────────────────────────────────────────────────────────

THRESH_ALPHA_RMS_PCT        = 15.0   # RMS of (α_rx − α_tr) / max(α_tr)  [%]
THRESH_TAU_FINAL_PCT        = 10.0   # |τ_final_rx − τ_final_tr| / τ_final_tr  [%]
THRESH_PABS_PCT             = 10.0   # |P_abs_rx − P_abs_tr| / P_abs_tr  [%]
THRESH_DEPO_RHO_DIFF        = 0.05   # |⟨ρ⟩_rx − ⟨ρ⟩_tr|  (dimensionless)
THRESH_NE_AT_XYZ_PCT        = 3.0    # nₑ relative error at TRAVIS xyz  [%]
THRESH_TE_AT_XYZ_PCT        = 3.0    # Tₑ relative error at TRAVIS xyz  [%]
THRESH_RHO_AT_XYZ_RMS       = 0.02   # ρ difference at TRAVIS xyz
THRESH_ALPHA_AT_XYZ_PCT     = 15.0   # α RMS (rx interpolators, TRAVIS N) at TRAVIS xyz  [%]
THRESH_ALPHA_TR_INPUTS_PCT  = 5.0    # α RMS (all TRAVIS inputs → raytrax formula)  [%]


# ── Scenario parameters ───────────────────────────────────────────────────────

@dataclass
class AbsorptionScenarioParams:
    """All per-run parameters for a single absorption validation scenario."""

    antenna_cyl:    tuple[float, float, float]
    """Antenna position (R [m], φ [deg], Z [m]) in cylindrical coordinates."""

    direction_cart: tuple[float, float, float]
    """Initial ray direction unit vector in Cartesian coordinates."""

    target_cart:    tuple[float, float, float]
    """Cartesian target point used by TRAVIS to define the beam direction."""

    harmonic:       int   = 2
    """EC harmonic to target (1 = fundamental, 2 = second harmonic)."""

    mode:           Literal["O", "X"] = "X"
    """Polarisation mode ('O' or 'X')."""

    b0_target:      float = 2.52
    """On-axis B field [T] after scaling (sets the resonance frequency scale)."""

    frequency_ghz:  float = 140.0
    """Wave frequency [GHz]."""

    power_mw:       float = 1.0
    """Input power [MW]."""

    ne_central:     float = 0.75
    """Central electron density [10²⁰ m⁻³]."""

    ne_parm:        tuple[float, float, float, float, float] = (0.05, 2.0, 1.5, 0.0, 0.0)
    """TRAVIS Ne-parm (a, p, q, h, w)."""

    te_central:     float = 5.0
    """Central electron temperature [keV]."""

    te_parm:        tuple[float, float, float, float, float] = (0.05, 2.0, 1.5, 0.0, 0.0)
    """TRAVIS Te-parm (a, p, q, h, w)."""


# ── Comparison result ─────────────────────────────────────────────────────────

@dataclass
class AbsorptionComparison:
    """Point-by-point and integrated absorption comparison between raytrax and TRAVIS."""

    # ── geometry
    s_common: np.ndarray
    """Common arc-length grid [m] (TRAVIS inside-plasma points)."""

    rho_tr: np.ndarray
    """TRAVIS ρ on the common grid."""

    # ── local absorption coefficient
    alpha_rx: np.ndarray
    """raytrax α [m⁻¹] on the common grid."""

    alpha_tr: np.ndarray
    """TRAVIS α [m⁻¹] on the common grid."""

    alpha_diff_rel: np.ndarray
    """(α_rx − α_tr) / max(α_tr)  — normalised to avoid edge blow-up."""

    # ── optical depth profile
    tau_rx: np.ndarray
    """raytrax τ(s) on the common grid."""

    tau_tr: np.ndarray
    """TRAVIS τ(s) on the common grid."""

    tau_diff: np.ndarray
    """τ_rx − τ_tr at each common arc-length point."""

    # ── linear power density
    lin_pwr_rx: np.ndarray
    """raytrax dP/ds [W/m] on the common grid."""

    lin_pwr_tr: np.ndarray
    """TRAVIS dP/ds [W/m] on the common grid."""

    # ── radial deposition profile (raytrax native ρ-grid)
    rho_profile_rx: np.ndarray
    """raytrax ρ-grid for radial deposition."""

    power_density_rx: np.ndarray
    """raytrax power density [W/m³] on the ρ-grid."""

    rho_profile_tr: np.ndarray
    """TRAVIS ρ-grid for radial deposition."""

    power_density_tr: np.ndarray
    """TRAVIS power density [W/m³] on the ρ-grid."""

    # ── scalar summary
    tau_final_rx:       float
    """Final optical depth from raytrax."""

    tau_final_tr:       float
    """Final optical depth from TRAVIS (last point of beamtrace_1)."""

    tau_final_rel_err:  float
    """(τ_final_rx − τ_final_tr) / max(τ_final_tr, 1e-6)."""

    total_power_rx_mw:  float
    """Total absorbed power from raytrax [MW]."""

    total_power_tr_mw:  float
    """Total absorbed power from TRAVIS [MW]."""

    total_power_rel_err: float
    """(P_abs_rx − P_abs_tr) / max(P_abs_tr, 1e-6)."""

    depo_rho_mean_rx:   float
    """Flux-surface-volume-weighted mean deposition ρ from raytrax."""

    depo_rho_mean_tr:   float
    """Flux-surface-volume-weighted mean deposition ρ from TRAVIS (centroid of Pabs profile)."""

    depo_rho_diff:      float
    """⟨ρ⟩_rx − ⟨ρ⟩_tr."""

    alpha_rms_pct:      float
    """RMS of α_diff_rel in percent."""

    pos_rms_mm:         float = 0.0
    """RMS distance between raytrax and TRAVIS ray positions on the common
    arc-length grid [mm].  Same definition as in step 1."""

    # Uses raytrax interpolators evaluated at each TRAVIS (x,y,z) position,
    # bypassing arc-length interpolation entirely.

    ne_rx_at_tr_xyz_diff_rel: np.ndarray | None = field(default=None)
    """(nₑ_rx − nₑ_tr) / max(nₑ_tr) at each TRAVIS xyz position."""

    te_rx_at_tr_xyz_diff_rel: np.ndarray | None = field(default=None)
    """(Tₑ_rx − Tₑ_tr) / max(Tₑ_tr) at each TRAVIS xyz position."""

    rho_rx_at_tr_xyz_diff: np.ndarray | None = field(default=None)
    """ρ_rx − ρ_tr at each TRAVIS xyz position."""

    ne_at_tr_xyz_mean_err_pct:  float = 0.0
    te_at_tr_xyz_mean_err_pct:  float = 0.0
    rho_at_tr_xyz_rms:          float = 0.0

    # ── diagnostic: α computed at TRAVIS's xyz with raytrax interpolators ────
    # Same spatial points as TRAVIS; uses TRAVIS's N vector and raytrax's B/ne/Te.
    # Isolates trajectory-path error from formula error.

    alpha_rx_at_tr_xyz: np.ndarray | None = field(default=None)
    """raytrax α [m⁻¹] evaluated at TRAVIS's Cartesian positions (using TRAVIS N)."""

    alpha_rx_at_tr_xyz_diff_rel: np.ndarray | None = field(default=None)
    """(α_rx@xyz − α_tr) / max(α_tr)."""

    alpha_at_tr_xyz_rms_pct: float = 0.0
    """RMS of alpha_rx_at_tr_xyz_diff_rel in percent."""

    # ── diagnostic: α from all TRAVIS inputs fed into raytrax formula ────────
    # Uses ne_tr, Te_tr, B_tr (vector), N_tr directly from TRAVIS output.
    # Isolates any formula/implementation difference from trajectory/interpolation error.

    alpha_rx_from_tr_inputs: np.ndarray | None = field(default=None)
    """raytrax α [m⁻¹] computed with ne, Te, B, N all taken from TRAVIS output."""

    alpha_rx_from_tr_inputs_diff_rel: np.ndarray | None = field(default=None)
    """(α_rx_from_tr_inputs − α_tr) / max(α_tr)."""

    alpha_from_tr_inputs_rms_pct: float = 0.0
    """RMS of alpha_rx_from_tr_inputs_diff_rel in percent."""

    # ── metadata
    harmonic:           int   = 2
    mode:               str   = "X"
    frequency_ghz:      float = 140.0
    b0_target:          float = 2.52
    travis_wall_time_s: float = 0.0
    raytrax_wall_time_s: float = 0.0
    n_steps_rx:         int   = 0
    resonance_accessible: bool = True
    """True if the intended EC resonance layer (B = f / (harmonic × 28 GHz/T))
    lies within the B-field range along the ray path."""


# ── Frequency range helpers ───────────────────────────────────────────────────

def compute_resonance_frequency_range(
    B_interp,
    rho_interp,
    harmonic: int,
    r_range: tuple[float, float] = (5.0, 6.2),
    z_max: float = 0.5,
    nfp: int = 5,
    margin_low:  float = 0.92,
    margin_high: float = 1.03,
) -> tuple[float, float]:
    """Frequency range [f_lo, f_hi] that places the *harmonic*-th EC resonance inside the plasma.

    Scans B on a coarse grid, collects values at inside-plasma points
    (0 < ρ < 0.95), finds B_min and B_max, then returns

        f_lo = margin_low  × harmonic × 28 × B_min
        f_hi = margin_high × harmonic × 28 × B_max

    so a frequency sampled from ``[f_lo, f_hi]`` has its resonance layer
    somewhere inside the plasma.

    Args:
        B_interp: Magnetic field vector interpolator (takes R, φ_mapped, Z).
        rho_interp: ρ interpolator.
        harmonic: 1 for fundamental, 2 for second harmonic.
        r_range: (R_min, R_max) scan range [m].
        z_max: Maximum |Z| for scan [m].
        nfp: Number of field periods.
        margin_low: Safety margin below B_min (< 1 → extends resonance toward edge).
        margin_high: Safety margin above B_max (> 1 → extends resonance toward edge).

    Returns:
        ``(f_lo_ghz, f_hi_ghz)`` covering the harmonic resonance inside the plasma.
    """
    half_period = np.pi / nfp
    b_vals: list[float] = []
    for R in np.linspace(r_range[0], r_range[1], 15):
        for phi_mapped in np.linspace(0.0, half_period, 8):
            for Z in np.linspace(-z_max, z_max, 10):
                rho = float(rho_interp(R, phi_mapped, Z))
                if 0.0 < rho < 0.95:
                    B = float(jnp.linalg.norm(B_interp(R, phi_mapped, Z)))
                    b_vals.append(B)

    if not b_vals:
        raise RuntimeError("No inside-plasma points found during B scan.")

    b_min = float(np.min(b_vals))
    b_max = float(np.max(b_vals))
    f_lo = margin_low  * harmonic * 28.0 * b_min
    f_hi = margin_high * harmonic * 28.0 * b_max
    return f_lo, f_hi


# ── Scenario sampler ──────────────────────────────────────────────────────────

def _check_resonance_accessible(
    antenna_cart: np.ndarray,
    direction: np.ndarray,
    B_interp,
    rho_interp,
    harmonic: int,
    frequency_ghz: float,
    nfp: int,
    n_steps: int = 60,
    t_max: float = 6.0,
) -> bool:
    """Return True if the EC resonance layer B_res = f/(harmonic*28) is accessible.

    Walks along the ray from the antenna toward the target and checks whether the
    B-field magnitude at any inside-plasma point spans the resonance value
    B_res = frequency_ghz / (harmonic * 28.0) [T].

    Args:
        antenna_cart: Antenna position in Cartesian coordinates [m].
        direction: Unit direction vector (antenna → target).
        B_interp: Magnetic field interpolator (R, phi_mapped, Z) → B vector.
        rho_interp: Normalised-radius interpolator (R, phi_mapped, Z) → ρ.
        harmonic: EC harmonic number.
        frequency_ghz: Wave frequency [GHz].
        nfp: Number of field periods.
        n_steps: Number of ray samples.
        t_max: Maximum path length to probe [m].

    Returns:
        True if min(B_inside) ≤ B_res ≤ max(B_inside).
    """
    B_res = frequency_ghz / (harmonic * 28.0)  # resonance B-field [T]
    half_period = np.pi / nfp
    full_period = 2.0 * np.pi / nfp
    b_inside: list[float] = []
    for t in np.linspace(0.0, t_max, n_steps):
        xyz = antenna_cart + t * direction
        R = np.sqrt(xyz[0] ** 2 + xyz[1] ** 2)
        Z = xyz[2]
        phi = np.arctan2(xyz[1], xyz[0])
        # Fold into the fundamental domain [0, pi/nfp] using field-period + stellarator symmetry.
        phi_fold = phi % full_period
        phi_mapped = full_period - phi_fold if phi_fold > half_period else phi_fold
        rho = float(rho_interp(R, phi_mapped, Z))
        if 0.0 < rho < 1.0:
            B = float(jnp.linalg.norm(B_interp(R, phi_mapped, Z)))
            b_inside.append(B)
    if not b_inside:
        return False
    return float(np.min(b_inside)) <= B_res <= float(np.max(b_inside))


def sample_absorption_scenario(
    harmonic: int,
    mode: Literal["O", "X"],
    rho_interp,
    B_interp,
    rng: np.random.Generator,
    *,
    f_lo_ghz: float,
    f_hi_ghz: float,
    antenna_r_min: float,
    antenna_r_max: float,
    antenna_z_max: float,
    target_r_min:  float,
    target_r_max:  float,
    target_z_max:  float,
    target_rho_max: float,
    nfp:       int   = 5,
    b0_target: float = 2.52,
    power_mw:  float = 1.0,
    max_tries: int   = 500,
) -> AbsorptionScenarioParams:
    """Sample one random :class:`AbsorptionScenarioParams` for the given harmonic and mode.

    The frequency is drawn uniformly from ``[f_lo_ghz, f_hi_ghz]`` (which
    should already span the harmonic resonance inside the plasma via
    :func:`compute_resonance_frequency_range`).  Profiles are rejection-sampled
    to ensure no O-mode or X-mode cutoff layer at the on-axis field.

    Args:
        harmonic: 1 (fundamental) or 2 (second harmonic).
        mode: 'O' or 'X' polarisation.
        rho_interp: ρ interpolator.
        B_interp: Magnetic field interpolator.
        rng: NumPy random generator.
        f_lo_ghz: Lower frequency bound [GHz].
        f_hi_ghz: Upper frequency bound [GHz].
        ...
    """
    f_ce0 = 28.0 * b0_target  # on-axis cyclotron frequency [GHz]

    for _ in range(max_tries):
        antenna_cyl = sample_antenna(
            rho_interp, rng, antenna_r_min, antenna_r_max, antenna_z_max, nfp
        )
        target_cart = sample_target(
            rho_interp, rng, target_r_min, target_r_max, target_z_max, target_rho_max, nfp
        )
        antenna_cart = np.array(cylindrical_to_cartesian(*antenna_cyl))
        direction    = target_cart - antenna_cart
        direction   /= np.linalg.norm(direction)

        frequency_ghz = float(rng.uniform(f_lo_ghz, f_hi_ghz))

        # Reject if the intended resonance layer doesn't lie along this ray.
        if not _check_resonance_accessible(
            antenna_cart, direction, B_interp, rho_interp, harmonic, frequency_ghz, nfp
        ):
            continue

        ne_central, ne_parm, te_central, te_parm = sample_profiles(rng)
        f_pe = 8.98 * np.sqrt(ne_central * 100.0)
        if mode == "O" and frequency_ghz < 1.1 * f_pe:
            continue
        if mode == "X":
            f_L = 0.5 * (-f_ce0 + np.sqrt(f_ce0**2 + 4.0 * f_pe**2))
            if frequency_ghz < 1.1 * f_L:
                continue
        break
    else:
        raise RuntimeError(
            f"Could not sample a valid scenario after {max_tries} tries "
            f"(harmonic={harmonic}, mode={mode})."
        )

    return AbsorptionScenarioParams(
        antenna_cyl=tuple(float(x) for x in antenna_cyl),
        direction_cart=tuple(float(x) for x in direction),
        target_cart=tuple(float(x) for x in target_cart),
        harmonic=harmonic,
        mode=mode,
        b0_target=b0_target,
        frequency_ghz=frequency_ghz,
        power_mw=power_mw,
        ne_central=float(ne_central),
        ne_parm=tuple(float(x) for x in ne_parm),
        te_central=float(te_central),
        te_parm=tuple(float(x) for x in te_parm),
    )


# ── Point-evaluation helpers ──────────────────────────────────────────────────

def _eval_rx_at_xyz(
    xyz: np.ndarray,
    interpolators: Interpolators,
    nfp: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate B (Cartesian), ρ, nₑ, Tₑ at given Cartesian positions using raytrax.

    Returns:
        B_xyz : (N, 3) magnetic field vectors [T]
        rho   : (N,)   normalised effective radius
        ne    : (N,)   electron density [10²⁰ m⁻³]
        te    : (N,)   electron temperature [keV]
    """
    pos   = jnp.array(xyz)
    B_xyz = np.array(jax.vmap(lambda p: _eval_magnetic_field(p, interpolators, nfp))(pos))
    rho   = np.array(jax.vmap(lambda p: _eval_rho_rx(p, interpolators, nfp))(pos))
    ne    = np.array(jax.vmap(interpolators.electron_density)(jnp.array(rho)))
    te    = np.array(jax.vmap(interpolators.electron_temperature)(jnp.array(rho)))
    return B_xyz, rho, ne, te


def _eval_alpha_at_tr_xyz(
    tr_xyz: np.ndarray,
    tr_N: np.ndarray,
    interpolators: Interpolators,
    nfp: int,
    frequency_hz: float,
    mode: Literal["X", "O"],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute raytrax α at TRAVIS's Cartesian positions using TRAVIS's N vector.

    Uses raytrax's B/ne/Te interpolators to evaluate plasma quantities at
    TRAVIS's spatial positions, then feeds those into raytrax's absorption
    formula together with TRAVIS's refractive index.  This separates trajectory
    error (different paths) from formula error (same point, different α).

    Returns:
        alpha : (N,) raytrax α [m⁻¹]
        B_xyz : (N, 3) raytrax B vectors at TRAVIS xyz [T]
        rho   : (N,)  raytrax ρ at TRAVIS xyz
        ne    : (N,)  raytrax nₑ [10²⁰ m⁻³] at TRAVIS xyz
        te    : (N,)  raytrax Tₑ [keV] at TRAVIS xyz
    """
    B_xyz, rho, ne, te = _eval_rx_at_xyz(tr_xyz, interpolators, nfp)

    B_jax  = jnp.array(B_xyz)
    N_jax  = jnp.array(tr_N)
    ne_jax = jnp.array(ne)
    te_jax = jnp.array(te)
    freq   = jnp.array(frequency_hz)

    alpha = np.array(jax.vmap(
        lambda B, N, ne_val, te_val: rx_absorption_coefficient(
            refractive_index=N,
            magnetic_field=B,
            electron_density_1e20_per_m3=ne_val,
            electron_temperature_keV=te_val,
            frequency=freq,
            mode=mode,
        )
    )(B_jax, N_jax, ne_jax, te_jax))

    return alpha, B_xyz, rho, ne, te


def _eval_alpha_from_tr_inputs(
    tr_B_cart: np.ndarray,
    tr_N: np.ndarray,
    tr_ne: np.ndarray,
    tr_te: np.ndarray,
    frequency_hz: float,
    mode: Literal["X", "O"],
) -> np.ndarray:
    """Compute raytrax α using *all* inputs taken directly from TRAVIS output.

    Feeds ne_tr, Te_tr, B_tr (full vector), N_tr into raytrax's absorption
    formula.  Any remaining difference from TRAVIS's α is a pure formula /
    implementation discrepancy.

    Returns:
        alpha : (N,) raytrax α [m⁻¹]
    """
    B_jax  = jnp.array(tr_B_cart)
    N_jax  = jnp.array(tr_N)
    ne_jax = jnp.array(tr_ne)
    te_jax = jnp.array(tr_te)
    freq   = jnp.array(frequency_hz)

    return np.array(jax.vmap(
        lambda B, N, ne_val, te_val: rx_absorption_coefficient(
            refractive_index=N,
            magnetic_field=B,
            electron_density_1e20_per_m3=ne_val,
            electron_temperature_keV=te_val,
            frequency=freq,
            mode=mode,
        )
    )(B_jax, N_jax, ne_jax, te_jax))


# ── Comparison ────────────────────────────────────────────────────────────────

def _depo_centroid(rho: np.ndarray, pwr: np.ndarray) -> float:
    """Flux-surface-volume-weighted mean ρ of the deposition profile."""
    total = float(np.sum(pwr))
    if total < 1e-30:
        return float(np.nan)
    return float(np.sum(rho * pwr) / total)


def compare_absorption(
    rx: TraceResult,
    tr: TravisECRHOutput,
    interpolators: Interpolators | None = None,
    nfp: int = 5,
    frequency_hz: float | None = None,
    mode: Literal["X", "O"] = "X",
    skip_expensive_diagnostics: bool = True,  # Default: skip 13s/sample overhead
) -> AbsorptionComparison:
    """Interpolate both results onto a common inside-plasma arc-length grid and compare.

    When *interpolators* is provided three additional diagnostics are computed:

    1. **ne / Te / ρ at TRAVIS xyz** — raytrax plasma quantities evaluated at
       TRAVIS's Cartesian positions (no arc-length interpolation).
    2. **α at TRAVIS xyz** — raytrax's absorption formula evaluated at TRAVIS's
       positions using raytrax's B/ne/Te and TRAVIS's refractive index N.
       Separates trajectory error from formula error.
    3. **α from all TRAVIS inputs** — raytrax's formula fed with ne_tr, Te_tr,
       B_tr, N_tr.  Any remaining difference is a pure formula discrepancy.

    Args:
        rx: raytrax :class:`TraceResult`.
        tr: Parsed TRAVIS output.
        interpolators: Optional raytrax :class:`Interpolators` for diagnostics 1–3.
        nfp: Number of field periods (for stellarator symmetry fold).
        frequency_hz: Wave frequency in Hz (required for diagnostics 2–3).
        mode: Polarisation mode (required for diagnostics 2–3).

    Returns:
        :class:`AbsorptionComparison` with all metrics.
    """
    s_rx = np.asarray(rx.beam_profile.arc_length)
    s_tr = np.asarray(tr.arc_length_m)

    rho_tr_all = np.asarray(tr.rho)
    s_lo = max(s_rx[0],  s_tr[0])
    s_hi = min(s_rx[-1], s_tr[-1])
    mask_tr = (s_tr >= s_lo) & (s_tr <= s_hi) & (rho_tr_all < 1.0)
    s_common = s_tr[mask_tr]

    if len(s_common) < 2:
        raise ValueError(
            f"Arc-length overlap too small: rx=[{s_rx[0]:.3f},{s_rx[-1]:.3f}] "
            f"tr=[{s_tr[0]:.3f},{s_tr[-1]:.3f}]"
        )

    rho_tr_c = rho_tr_all[mask_tr]

    # ── absorption coefficient ────────────────────────────────────────────────
    alpha_rx_all = np.asarray(rx.beam_profile.absorption_coefficient)
    alpha_rx_c   = interpolate_to_arc(s_rx, alpha_rx_all, s_common)
    alpha_tr_c   = np.asarray(tr.absorption_m_inv)[mask_tr]

    # ── ray position RMS ─────────────────────────────────────────────────────
    pos_rx_all = np.asarray(rx.beam_profile.position)           # (npoints, 3)
    pos_tr_c   = np.asarray(tr.position_m)[mask_tr]             # (N, 3)
    pos_rx_c   = np.column_stack([
        interpolate_to_arc(s_rx, pos_rx_all[:, k], s_common) for k in range(3)
    ])                                                           # (N, 3)
    pos_dist_mm = np.linalg.norm((pos_rx_c - pos_tr_c) * 1e3, axis=1)
    _pos_rms_mm = float(np.sqrt(np.mean(pos_dist_mm**2)))

    alpha_ref     = float(np.max(np.abs(alpha_tr_c))) or 1.0
    alpha_diff_rel = (alpha_rx_c - alpha_tr_c) / alpha_ref

    # ── optical depth ─────────────────────────────────────────────────────────
    tau_rx_all = np.asarray(rx.beam_profile.optical_depth)
    tau_rx_c   = interpolate_to_arc(s_rx, tau_rx_all, s_common)
    tau_tr_c   = np.asarray(tr.optical_depth)[mask_tr]
    tau_diff   = tau_rx_c - tau_tr_c

    # ── linear power density ──────────────────────────────────────────────────
    lpwr_rx_all = np.asarray(rx.beam_profile.linear_power_density)
    lpwr_rx_c   = interpolate_to_arc(s_rx, lpwr_rx_all, s_common)
    lpwr_tr_c   = np.asarray(tr.linear_power_density_w_per_m)[mask_tr]

    # ── scalars ───────────────────────────────────────────────────────────────
    tau_final_rx = float(rx.optical_depth)
    tau_final_tr = float(tr.optical_depth[-1])
    tau_ref      = max(tau_final_tr, 1e-6)
    tau_final_rel_err = (tau_final_rx - tau_final_tr) / tau_ref

    total_power_rx_mw = float(rx.absorbed_power) / 1e6
    total_power_tr_mw = float(tr.total_absorbed_power_mw)
    pwr_ref           = max(total_power_tr_mw, 1e-9)
    total_power_rel_err = (total_power_rx_mw - total_power_tr_mw) / pwr_ref

    # ── deposition centroid ───────────────────────────────────────────────────
    rho_prof_rx   = np.asarray(rx.radial_profile.rho)
    pwr_dens_rx   = np.asarray(rx.radial_profile.volumetric_power_density)
    depo_rho_rx   = float(rx.deposition_rho_mean)

    rho_prof_tr   = np.asarray(tr.rho_profile)
    pwr_dens_tr   = np.asarray(tr.power_density_w_per_m3)
    depo_rho_tr   = _depo_centroid(rho_prof_tr, pwr_dens_tr)

    # ── diagnostic 1–3: at TRAVIS xyz ────────────────────────────────────────
    ne_xyz_diff_rel  = None
    te_xyz_diff_rel  = None
    rho_xyz_diff     = None
    ne_xyz_pct       = 0.0
    te_xyz_pct       = 0.0
    rho_xyz_rms      = 0.0

    alpha_at_xyz      = None
    alpha_at_xyz_drel = None
    alpha_at_xyz_pct  = 0.0

    alpha_tr_inputs      = None
    alpha_tr_inputs_drel = None
    alpha_tr_inputs_pct  = 0.0

    if interpolators is not None:
        tr_xyz_c = np.asarray(tr.position_m)[mask_tr]          # (N, 3) Cartesian
        tr_N_c   = np.asarray(tr.refractive_index)[mask_tr]    # (N, 3)
        tr_B_c   = np.asarray(tr.magnetic_field_cart)[mask_tr] # (N, 3)
        tr_ne_c  = np.asarray(tr.electron_density_1e20)[mask_tr]
        tr_te_c  = np.asarray(tr.electron_temperature_keV)[mask_tr]

        # Diagnostic 1: ne / Te / ρ at TRAVIS xyz
        _, rho_rx_xyz, ne_rx_xyz, te_rx_xyz = _eval_rx_at_xyz(tr_xyz_c, interpolators, nfp)
        ne_ref = max(float(np.max(tr_ne_c)), 1e-9)
        te_ref = max(float(np.max(tr_te_c)), 1e-9)
        ne_xyz_diff_rel = (ne_rx_xyz - tr_ne_c) / ne_ref
        te_xyz_diff_rel = (te_rx_xyz - tr_te_c) / te_ref
        rho_xyz_diff    = rho_rx_xyz - np.asarray(tr.rho)[mask_tr]
        ne_xyz_pct  = float(np.mean(np.abs(ne_xyz_diff_rel)) * 100.0)
        te_xyz_pct  = float(np.mean(np.abs(te_xyz_diff_rel)) * 100.0)
        rho_xyz_rms = float(np.sqrt(np.mean(rho_xyz_diff**2)))

        if frequency_hz is not None and not skip_expensive_diagnostics:
            # Diagnostic 2: α at TRAVIS xyz (rx interpolators + TRAVIS N)
            # NOTE: This is VERY slow (~13s per sample) - skip for large scans!
            alpha_at_xyz, *_ = _eval_alpha_at_tr_xyz(
                tr_xyz_c, tr_N_c, interpolators, nfp, frequency_hz, mode
            )
            alpha_at_xyz_drel = (alpha_at_xyz - alpha_tr_c) / alpha_ref
            alpha_at_xyz_pct  = float(np.sqrt(np.mean(alpha_at_xyz_drel**2)) * 100.0)

            # Diagnostic 3: α from all TRAVIS inputs
            alpha_tr_inputs = _eval_alpha_from_tr_inputs(
                tr_B_c, tr_N_c, tr_ne_c, tr_te_c, frequency_hz, mode
            )
            alpha_tr_inputs_drel = (alpha_tr_inputs - alpha_tr_c) / alpha_ref
            alpha_tr_inputs_pct  = float(np.sqrt(np.mean(alpha_tr_inputs_drel**2)) * 100.0)

    return AbsorptionComparison(
        s_common=s_common,
        rho_tr=rho_tr_c,
        alpha_rx=alpha_rx_c,
        alpha_tr=alpha_tr_c,
        alpha_diff_rel=alpha_diff_rel,
        tau_rx=tau_rx_c,
        tau_tr=tau_tr_c,
        tau_diff=tau_diff,
        lin_pwr_rx=lpwr_rx_c,
        lin_pwr_tr=lpwr_tr_c,
        rho_profile_rx=rho_prof_rx,
        power_density_rx=pwr_dens_rx,
        rho_profile_tr=rho_prof_tr,
        power_density_tr=pwr_dens_tr,
        tau_final_rx=tau_final_rx,
        tau_final_tr=tau_final_tr,
        tau_final_rel_err=tau_final_rel_err,
        total_power_rx_mw=total_power_rx_mw,
        total_power_tr_mw=total_power_tr_mw,
        total_power_rel_err=total_power_rel_err,
        depo_rho_mean_rx=depo_rho_rx,
        depo_rho_mean_tr=depo_rho_tr,
        depo_rho_diff=depo_rho_rx - depo_rho_tr,
        alpha_rms_pct=float(np.sqrt(np.mean(alpha_diff_rel**2)) * 100.0),
        pos_rms_mm=_pos_rms_mm,
        # diagnostics
        ne_rx_at_tr_xyz_diff_rel=ne_xyz_diff_rel,
        te_rx_at_tr_xyz_diff_rel=te_xyz_diff_rel,
        rho_rx_at_tr_xyz_diff=rho_xyz_diff,
        ne_at_tr_xyz_mean_err_pct=ne_xyz_pct,
        te_at_tr_xyz_mean_err_pct=te_xyz_pct,
        rho_at_tr_xyz_rms=rho_xyz_rms,
        alpha_rx_at_tr_xyz=alpha_at_xyz,
        alpha_rx_at_tr_xyz_diff_rel=alpha_at_xyz_drel,
        alpha_at_tr_xyz_rms_pct=alpha_at_xyz_pct,
        alpha_rx_from_tr_inputs=alpha_tr_inputs,
        alpha_rx_from_tr_inputs_diff_rel=alpha_tr_inputs_drel,
        alpha_from_tr_inputs_rms_pct=alpha_tr_inputs_pct,
    )


# ── Reporting ─────────────────────────────────────────────────────────────────

def _pass_fail(value: float, threshold: float) -> str:
    return "PASS" if abs(value) <= threshold else "FAIL"


def print_absorption_report(cmp: AbsorptionComparison) -> None:
    label = f"{cmp.mode}{cmp.harmonic}"
    print(f"\n── Absorption comparison  [{label}  f={cmp.frequency_ghz:.1f} GHz  "
          f"B₀={cmp.b0_target:.2f} T] ─────────────────────────────")
    print(f"  Arc-length range   : {cmp.s_common[0]:.3f} – {cmp.s_common[-1]:.3f} m "
          f"({len(cmp.s_common)} points)")
    print()

    def row(label: str, value: float, unit: str, threshold: float | None = None) -> None:
        thresh_str = f"  (threshold {threshold:.3g} {unit})" if threshold is not None else ""
        pf = f"  [{_pass_fail(abs(value), threshold)}]" if threshold is not None else ""
        print(f"  {label:<32} {value:>+10.3f} {unit}{thresh_str}{pf}")

    def row_pair(label: str, rx: float, tr: float, unit: str) -> None:
        print(f"  {label:<32} rx={rx:>8.4f}  tr={tr:>8.4f} {unit}")

    row_pair("τ_final",         cmp.tau_final_rx, cmp.tau_final_tr, "")
    row("  → relative error",   cmp.tau_final_rel_err * 100, "%", THRESH_TAU_FINAL_PCT)
    print()
    row_pair("P_abs",           cmp.total_power_rx_mw, cmp.total_power_tr_mw, "MW")
    row("  → relative error",   cmp.total_power_rel_err * 100, "%", THRESH_PABS_PCT)
    print()
    row_pair("⟨ρ⟩ deposition",  cmp.depo_rho_mean_rx, cmp.depo_rho_mean_tr, "")
    row("  → diff",             cmp.depo_rho_diff, "", THRESH_DEPO_RHO_DIFF)
    print()
    row("α RMS relative error", cmp.alpha_rms_pct, "%", THRESH_ALPHA_RMS_PCT)

    if cmp.ne_rx_at_tr_xyz_diff_rel is not None:
        print()
        print("  -- at TRAVIS xyz (same spatial point, no arc-length interpolation) --")
        row("nₑ mean error @xyz",    cmp.ne_at_tr_xyz_mean_err_pct,  "%", THRESH_NE_AT_XYZ_PCT)
        row("Tₑ mean error @xyz",    cmp.te_at_tr_xyz_mean_err_pct,  "%", THRESH_TE_AT_XYZ_PCT)
        row("ρ RMS @xyz",            cmp.rho_at_tr_xyz_rms,          "",  THRESH_RHO_AT_XYZ_RMS)

    if cmp.alpha_rx_at_tr_xyz_diff_rel is not None:
        print()
        print("  -- α at TRAVIS xyz (rx B/ne/Te, TRAVIS N) --")
        row("α RMS @xyz",            cmp.alpha_at_tr_xyz_rms_pct,    "%", THRESH_ALPHA_AT_XYZ_PCT)

    if cmp.alpha_rx_from_tr_inputs_diff_rel is not None:
        print()
        print("  -- α from all TRAVIS inputs (ne_tr, Te_tr, B_tr, N_tr → rx formula) --")
        row("α RMS from TRAVIS inputs", cmp.alpha_from_tr_inputs_rms_pct, "%", THRESH_ALPHA_TR_INPUTS_PCT)

    print()
    print(f"  {'TRAVIS wall time':<32} {cmp.travis_wall_time_s:>8.1f} s")
    print(f"  {'raytrax wall time':<32} {cmp.raytrax_wall_time_s:>8.1f} s")
    print(f"  {'ODE steps (raytrax)':<32} {cmp.n_steps_rx:>8d}")


# ── Scenario runner ───────────────────────────────────────────────────────────

def run_absorption_scenario(
    params: AbsorptionScenarioParams,
    eq: MagneticConfiguration,
    wout_nc: Path,
    travis_exe: Path,
    output_dir: Path,
    base_interpolators: Interpolators | None = None,
    mesh_cache_dir: Path | None = None,
    verbose: bool = True,
    max_step_size: float = 0.05,
    rtol: float = 1e-4,
    atol: float = 1e-6,
    travis_hgrid: float = 0.022,
    travis_dphi: float = 2.0,
    travis_rk_accuracy: float = 1e-5,
    travis_max_rk_stepsize: float = 10.0,
    travis_resonance_umax: float = 7.0,
    travis_resonance_grid_points: int = 700,
) -> AbsorptionComparison:
    """Run one TRAVIS + raytrax absorption scenario and return the comparison.

    Args:
        params: Scenario parameters (antenna, direction, frequency, profiles …).
        eq: raytrax :class:`MagneticConfiguration` (already scaled to ``params.b0_target``).
        wout_nc: Path to the VMEC wout NetCDF file for TRAVIS.
        travis_exe: Path to the TRAVIS executable.
        output_dir: Base output directory (sub-directories are created per run).
        mesh_cache_dir: Optional directory for caching the TRAVIS mesh.
        verbose: Print progress messages.
        max_step_size: raytrax ODE max step size [m].
        rtol / atol: raytrax ODE tolerances.
        travis_hgrid / travis_dphi: TRAVIS mesh step sizes.
        travis_rk_accuracy / travis_max_rk_stepsize: TRAVIS integrator settings.
travis_resonance_umax / travis_resonance_grid_points: Upper velocity limit
                (u/u_th, umaxgrid) and u_par grid resolution (nugrid) for the resonance
                line integral. Use 35/1500 for O2 mode, default 7/700 otherwise.

    Returns:
        :class:`AbsorptionComparison` populated with all metrics.
    """
    t_total_start = time.perf_counter()
    
    t0 = time.perf_counter()
    rho_np, ne_np, te_np = build_profiles(
        params.ne_central, params.ne_parm, params.te_central, params.te_parm
    )
    profiles = RadialProfiles(
        rho=jnp.array(rho_np),
        electron_density=jnp.array(ne_np),
        electron_temperature=jnp.array(te_np),
    )
    t_profiles = time.perf_counter() - t0

    if verbose:
        label = f"{params.mode}{params.harmonic}"
        print(f"Running TRAVIS  [{label}  f={params.frequency_ghz:.1f} GHz  "
              f"B₀={params.b0_target:.2f} T]  (profiles: {t_profiles:.3f}s) …")
    t0_travis = time.perf_counter()
    travis_params = TravisECRHInput(
        antenna_position_cyl=jnp.array(params.antenna_cyl),
        target_position=jnp.array(params.target_cart),
        frequency_ghz=params.frequency_ghz,
        power_mw=params.power_mw,
        equilibrium_file=str(wout_nc),
        target_coords_type="cart",
        mode=params.mode,
        rho_grid=jnp.array(rho_np),
        electron_density_1e20=jnp.array(ne_np),
        electron_temperature_keV=jnp.array(te_np),
        b0_normalization=params.b0_target,
        dielectric_tracing="cold",
        hamiltonian="West",
        ne_parm=params.ne_parm,
        te_parm=params.te_parm,
        hgrid=travis_hgrid,
        dphi=travis_dphi,
        rk_accuracy=travis_rk_accuracy,
        max_rk_stepsize_wavelengths=travis_max_rk_stepsize,
        resonance_umax=travis_resonance_umax,
        resonance_grid_points=travis_resonance_grid_points,
    )
    travis = run_travis(
        travis_exe, travis_params,
        output_dir=output_dir / "travis_run",
        mesh_cache_dir=mesh_cache_dir,
    )
    travis_time = time.perf_counter() - t0_travis
    if verbose:
        print(f"  {len(travis.arc_length_m)} points, "
              f"s=[{travis.arc_length_m[0]:.3f}, {travis.arc_length_m[-1]:.3f}] m  "
              f"τ={travis.optical_depth[-1]:.4f}  ({travis_time:.1f} s)")
        print("Running raytrax …")

    antenna_cart = cylindrical_to_cartesian(*params.antenna_cyl)
    beam = Beam(
        position=jnp.array(antenna_cart),
        direction=jnp.array(params.direction_cart),
        frequency=jnp.array(params.frequency_ghz * 1e9),
        mode=params.mode,
        power=params.power_mw * 1e6,
    )
    t0_rx = time.perf_counter()
    rx = trace(eq, profiles, beam, settings=TracerSettings(
        max_step_size=max_step_size,
        relative_tolerance=rtol,
        absolute_tolerance=atol,
    ))
    jax.block_until_ready(rx.beam_profile.arc_length)
    raytrax_time = time.perf_counter() - t0_rx
    s_rx = np.asarray(rx.beam_profile.arc_length)
    if verbose:
        print(f"  {len(s_rx)} points, s=[{s_rx[0]:.3f}, {s_rx[-1]:.3f}] m  "
              f"τ={float(rx.optical_depth):.4f}  ({raytrax_time:.1f} s)")

    t0_interp = time.perf_counter()
    if base_interpolators is not None:
        # Reuse B and rho interpolators, only rebuild profile interpolators
        interpolators = Interpolators(
            magnetic_field=base_interpolators.magnetic_field,
            rho=base_interpolators.rho,
            electron_density=build_electron_density_profile_interpolator(profiles),
            electron_temperature=build_electron_temperature_profile_interpolator(profiles),
            is_axisymmetric=eq.is_axisymmetric,
        )
    else:
        # Build all interpolators (slower path for standalone use)
        interpolators = Interpolators(
            magnetic_field=build_magnetic_field_interpolator(eq),
            rho=build_rho_interpolator(eq),
            electron_density=build_electron_density_profile_interpolator(profiles),
            electron_temperature=build_electron_temperature_profile_interpolator(profiles),
            is_axisymmetric=eq.is_axisymmetric,
        )
    t_interp = time.perf_counter() - t0_interp
    if verbose:
        print(f"  Interpolators: {t_interp:.3f} s")
    
    t0_cmp = time.perf_counter()
    cmp = compare_absorption(
        rx, travis,
        interpolators=interpolators,
        nfp=eq.nfp,
        frequency_hz=params.frequency_ghz * 1e9,
        mode=params.mode,
    )
    t_cmp = time.perf_counter() - t0_cmp
    if verbose:
        print(f"  Comparison: {t_cmp:.3f} s")
    cmp.harmonic           = params.harmonic
    cmp.mode               = params.mode
    cmp.frequency_ghz      = params.frequency_ghz
    cmp.b0_target          = params.b0_target
    cmp.travis_wall_time_s = travis_time
    cmp.raytrax_wall_time_s = raytrax_time
    cmp.n_steps_rx         = len(s_rx)
    # Check whether the TRAVIS absorption peak corresponds to the intended harmonic.
    # f / f_ce = harmonic ± 0.2 at the peak absorption point is the criterion.
    if cmp.tau_final_tr > 0.01 and len(travis.absorption_m_inv) > 0:
        peak_idx = int(np.argmax(np.asarray(travis.absorption_m_inv)))
        B_peak = float(travis.magnetic_field_magnitude_T[peak_idx])
        if B_peak > 0.0:
            f_over_fce = params.frequency_ghz / (28.0 * B_peak)
            cmp.resonance_accessible = bool(abs(f_over_fce - params.harmonic) < 0.2)
        else:
            cmp.resonance_accessible = False
    else:
        cmp.resonance_accessible = True  # no significant absorption — don't penalise

    t_total = time.perf_counter() - t_total_start
    # Always print timing to diagnose slowness
    print(f"  [TIMING] Total: {t_total:.1f}s = TRAVIS: {travis_time:.1f}s + raytrax: {raytrax_time:.1f}s + "
          f"interp: {t_interp:.1f}s + cmp: {t_cmp:.1f}s + profiles: {t_profiles:.1f}s + "
          f"other: {t_total - travis_time - raytrax_time - t_interp - t_cmp - t_profiles:.1f}s")
    if verbose:
        print_absorption_report(cmp)

    return cmp
