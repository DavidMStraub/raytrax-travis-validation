"""Step 1: Trajectory validation — beam path, ρ, |B|, B̂, nₑ, Tₑ.

Both codes trace with the cold-plasma Hamiltonian (absorption is still
computed relativistically).  B₀ = 3 T places the second harmonic at ~168 GHz,
well off the 140 GHz beam, so absorption is negligible and only geometric
errors are tested here.
"""

from __future__ import annotations

from typing import Literal

import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from travis import TravisECRHInput, TravisECRHOutput, run_travis
from profiles import build_profiles, cylindrical_to_cartesian, interpolate_to_arc

from raytrax.api import trace
from raytrax.equilibrium.interpolate import (
    MagneticConfiguration,
    build_magnetic_field_interpolator,
    build_rho_interpolator,
    build_electron_density_profile_interpolator,
    build_electron_temperature_profile_interpolator,
)
from raytrax.tracer.buffers import Interpolators
from raytrax.tracer.solver import _eval_magnetic_field, _eval_rho
from raytrax.types import Beam, RadialProfiles, TraceResult, TracerSettings


def compute_direction_sensitivity(
    eq: MagneticConfiguration,
    profiles: RadialProfiles,
    beam: Beam,
    settings: TracerSettings,
    eps: float = 1e-3,
) -> float:
    """Finite-difference sensitivity of mean trajectory position to initial direction [m/rad].

    Perturbs the initial direction by ±eps radians in a perpendicular direction,
    traces both rays, and returns the norm of the mean-position difference divided
    by 2*eps.  Large values mean the trajectory shifts strongly under small direction
    changes — comparison with TRAVIS is unreliable for those cases.
    """
    d = jnp.asarray(beam.direction)

    # Build a unit vector perpendicular to d via Gram-Schmidt.
    tmp = jnp.where(jnp.abs(d[0]) < 0.9, jnp.array([1.0, 0.0, 0.0]), jnp.array([0.0, 1.0, 0.0]))
    perp = tmp - jnp.dot(tmp, d) * d
    perp = perp / jnp.linalg.norm(perp)

    def _exit_pos(direction: jax.Array) -> jax.Array:
        direction = direction / jnp.linalg.norm(direction)
        r = trace(eq, profiles, replace(beam, direction=direction), settings=settings)
        return jnp.asarray(r.beam_profile.position[-1])

    pos_plus  = _exit_pos(d + eps * perp)
    pos_minus = _exit_pos(d - eps * perp)
    return float(jnp.linalg.norm(pos_plus - pos_minus) / (2.0 * eps))

# Defaults for ScenarioParams
_B0_TARGET    = 3.0
_FREQUENCY_GHZ = 140.0
_POWER_MW     = 1.0
_NE_CENTRAL   = 0.75
_NE_PARM      = (0.05, 2, 1.5, 0, 0)
_TE_CENTRAL   = 5.0
_TE_PARM      = (0.05, 2, 1.5, 0, 0)

# Thresholds for pass/fail reporting
THRESH_POS_RMS_MM = 5.0
THRESH_RHO_RMS    = 0.02
THRESH_B_MAG_PCT  = 2.0
THRESH_NE_PCT     = 3.0
THRESH_TE_PCT     = 3.0
THRESH_B_DIR_DEG  = 1.0


@dataclass
class ScenarioParams:
    """All per-run parameters for a single trajectory validation scenario."""
    antenna_cyl:    tuple[float, float, float]
    direction_cart: tuple[float, float, float]   # unit vector
    target_cart:    tuple[float, float, float]   # Cartesian target point for TRAVIS
    mode:           Literal["O", "X"] = "O"
    b0_target:      float = _B0_TARGET
    frequency_ghz:  float = _FREQUENCY_GHZ
    power_mw:       float = _POWER_MW
    ne_central:     float = _NE_CENTRAL
    ne_parm:        tuple[float, float, float, float, float] = _NE_PARM
    te_central:     float = _TE_CENTRAL
    te_parm:        tuple[float, float, float, float, float] = _TE_PARM


@dataclass
class TrajectoryComparison:
    """Point-by-point comparison between raytrax and TRAVIS trajectories."""

    s_common: np.ndarray
    """Common arc-length grid used for all comparisons [m]."""

    pos_dist_mm: np.ndarray
    """Euclidean distance between positions at each arc-length point [mm]."""

    rho_diff: np.ndarray
    """ρ_raytrax − ρ_TRAVIS at each point."""

    B_mag_rx: np.ndarray
    """raytrax |B| [T]."""

    B_mag_tr: np.ndarray
    """TRAVIS |B| [T] (interpolated onto common grid)."""

    B_mag_rel_err: np.ndarray
    """(|B|_rx − |B|_tr) / |B|_tr at each point."""

    B_dir_angle_deg: np.ndarray
    """Angle [deg] between raytrax B̂ and TRAVIS B̂ at common arc-length points."""

    rho_rx: np.ndarray
    """raytrax ρ on the common grid."""

    rho_tr: np.ndarray
    """TRAVIS ρ on the common grid."""

    ne_rx: np.ndarray
    """raytrax nₑ [10²⁰ m⁻³] on the common grid."""

    ne_tr: np.ndarray
    """TRAVIS nₑ [10²⁰ m⁻³] on the common grid (direct TRAVIS output)."""

    te_rx: np.ndarray
    """raytrax Tₑ [keV] on the common grid."""

    te_tr: np.ndarray
    """TRAVIS Tₑ [keV] on the common grid (direct TRAVIS output)."""

    ne_diff_rel: np.ndarray
    """(nₑ_rx − nₑ_tr) / max(nₑ_tr)  — normalised by peak to avoid edge blow-up."""

    te_diff_rel: np.ndarray
    """(Tₑ_rx − Tₑ_tr) / max(Tₑ_tr)  — normalised by peak to avoid edge blow-up."""

    curvature_rx: np.ndarray
    """Local ray curvature |d²r/ds²| for raytrax [m⁻¹] on the common grid."""

    curvature_tr: np.ndarray
    """Local ray curvature |d²r/ds²| for TRAVIS [m⁻¹] on the common grid."""

    min_rho_tr:        float
    """Minimum ρ reached along the TRAVIS trajectory."""
    min_rho_rx:        float
    """Minimum ρ reached along the raytrax trajectory."""
    max_ne_tr:         float
    """Maximum electron density along TRAVIS trajectory [10²⁰ m⁻³]."""
    max_te_tr:         float
    """Maximum electron temperature along TRAVIS trajectory [keV]."""
    max_ne_rx:         float
    """Maximum electron density along raytrax trajectory [10²⁰ m⁻³]."""
    max_te_rx:         float
    """Maximum electron temperature along raytrax trajectory [keV]."""
    arc_in_plasma_tr_m: float
    """Total arc length inside the plasma (ρ < 1) for TRAVIS [m]."""
    arc_in_plasma_rx_m: float
    """Total arc length inside the plasma (ρ < 1) for raytrax [m]."""
    max_curvature_tr_m_inv: float
    """Maximum local ray curvature |d²r/ds²| inside the plasma, TRAVIS [m⁻¹]."""
    max_curvature_rx_m_inv: float
    """Maximum local ray curvature |d²r/ds²| inside the plasma, raytrax [m⁻¹]."""
    Y_param:           float
    """f_ce / f — ratio of cyclotron to wave frequency; high Y → strong B sensitivity."""
    pos_rms_mm:        float
    rho_rms:           float
    B_mag_mean_err_pct: float
    B_dir_rms_deg:     float
    ne_mean_err_pct:   float
    te_mean_err_pct:   float
    deflection_rx_deg:   float
    """Total ray deflection angle [deg] for raytrax over the common arc."""
    deflection_tr_deg:   float
    """Total ray deflection angle [deg] for TRAVIS over the common arc."""
    deflection_diff_deg: float
    """deflection_rx_deg − deflection_tr_deg [deg]."""
    direction_sensitivity: float = 0.0
    """Frobenius norm of ∂(final position)/∂(initial direction) [m].
    Large value → trajectory is sensitive → comparison with TRAVIS is unreliable."""
    travis_wall_time_s:  float = 0.0
    raytrax_wall_time_s: float = 0.0
    n_steps_rx:          int   = 0
    """Number of accepted ODE steps in the raytrax solve (= len(arc_length) after trimming)."""

    # ── "At TRAVIS xyz" metrics ──────────────────────────────────────────────
    # Raytrax quantities re-evaluated at the TRAVIS Cartesian positions
    # (same spatial point, no arc-length interpolation artifact).
    # Only populated when interpolators are passed to compare_trajectories.
    B_mag_at_tr_xyz_rel_err: np.ndarray | None = field(default=None)
    """(|B|_rx − |B|_tr) / |B|_tr evaluated at each TRAVIS xyz position."""
    B_dir_at_tr_xyz_deg: np.ndarray | None = field(default=None)
    """Angle [deg] between B̂_rx and B̂_tr at each TRAVIS xyz position."""
    ne_at_tr_xyz_diff_rel: np.ndarray | None = field(default=None)
    """(nₑ_rx − nₑ_tr) / max(nₑ_tr) at each TRAVIS xyz position."""
    te_at_tr_xyz_diff_rel: np.ndarray | None = field(default=None)
    """(Tₑ_rx − Tₑ_tr) / max(Tₑ_tr) at each TRAVIS xyz position."""
    rho_at_tr_xyz_diff: np.ndarray | None = field(default=None)
    """ρ_rx − ρ_tr at each TRAVIS xyz position."""

    B_mag_at_tr_xyz_mean_err_pct: float = 0.0
    B_dir_at_tr_xyz_rms_deg:      float = 0.0
    ne_at_tr_xyz_mean_err_pct:    float = 0.0
    te_at_tr_xyz_mean_err_pct:    float = 0.0
    rho_at_tr_xyz_rms:            float = 0.0


def _ray_curvature(pos: np.ndarray, s: np.ndarray, ds_min: float = 1e-3) -> np.ndarray:
    """Local curvature |d²r/ds²| [m⁻¹] along a ray trajectory.

    Duplicate or near-duplicate points (ds < ds_min) are skipped before
    computing finite-difference tangents, then results are mapped back to the
    original grid.  This prevents TRAVIS boundary sub-steps from inflating the
    curvature estimate.
    """
    ds_raw = np.diff(s)
    keep = ds_raw >= ds_min
    if keep.sum() < 2:
        return np.zeros(len(s))

    # Build a deduplicated view.
    idx = np.concatenate([[0], np.where(keep)[0] + 1])
    pos_d = pos[idx]
    s_d   = s[idx]
    ds_d  = np.diff(s_d)

    tangent = np.diff(pos_d, axis=0) / ds_d[:, None]
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True).clip(1e-12)
    ds_mid = 0.5 * (ds_d[:-1] + ds_d[1:])
    kappa_d = np.linalg.norm(np.diff(tangent, axis=0), axis=1) / ds_mid.clip(1e-12)

    # Nearest-neighbour map back to original grid using deduplicated s values.
    s_mid_d = s_d[1:-1]
    out = np.interp(s, s_mid_d, kappa_d, left=kappa_d[0], right=kappa_d[-1])
    return out


def _eval_rx_at_xyz(
    xyz: np.ndarray,
    interpolators: Interpolators,
    nfp: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate B (Cartesian), rho, ne, Te at given Cartesian positions using raytrax.

    Returns:
        B_xyz : (N, 3) magnetic field vectors [T]
        rho   : (N,)   normalised effective radius
        ne    : (N,)   electron density [10²⁰ m⁻³]
        te    : (N,)   electron temperature [keV]
    """
    pos = jnp.array(xyz)
    B_xyz = np.array(jax.vmap(lambda p: _eval_magnetic_field(p, interpolators, nfp))(pos))
    rho   = np.array(jax.vmap(lambda p: _eval_rho(p, interpolators, nfp))(pos))
    ne    = np.array(jax.vmap(interpolators.electron_density)(jnp.array(rho)))
    te    = np.array(jax.vmap(interpolators.electron_temperature)(jnp.array(rho)))
    return B_xyz, rho, ne, te


def compare_trajectories(
    rx: TraceResult,
    tr: TravisECRHOutput,
    frequency_ghz: float = 140.0,
    b0_target: float = 3.0,
    interpolators: Interpolators | None = None,
    nfp: int = 5,
) -> TrajectoryComparison:
    """Interpolate both trajectories onto a common arc-length grid and compare.

    The common grid spans the overlap of both arc-length ranges and uses
    TRAVIS's arc-length points as the reference (denser output in the plasma).
    Only inside-plasma points (ρ < 1) are included in the comparison metrics.

    Args:
        rx: raytrax :class:`TraceResult`.
        tr: Parsed TRAVIS output.
        frequency_ghz: Wave frequency [GHz], used to compute Y = f_ce/f.
        b0_target: On-axis B field [T], used to compute f_ce = 28 * B0 [GHz].

    Returns:
        :class:`TrajectoryComparison` with all metrics.
    """
    s_rx = np.asarray(rx.beam_profile.arc_length)
    s_tr = np.asarray(tr.arc_length_m)

    # Inside-plasma overlap only.
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

    pos_rx_common = interpolate_to_arc(s_rx, np.asarray(rx.beam_profile.position), s_common)
    pos_tr_common = np.asarray(tr.position_m)[mask_tr]
    pos_dist_mm   = np.linalg.norm(pos_rx_common - pos_tr_common, axis=1) * 1e3

    rho_rx_common = interpolate_to_arc(s_rx, np.asarray(rx.beam_profile.normalized_effective_radius), s_common)
    rho_tr_common = rho_tr_all[mask_tr]
    rho_diff      = rho_rx_common - rho_tr_common

    B_rx_vecs  = np.asarray(rx.beam_profile.magnetic_field)
    B_rx_mag   = np.linalg.norm(B_rx_vecs, axis=1)
    B_rx_mag_c = interpolate_to_arc(s_rx, B_rx_mag, s_common)
    B_tr_mag_c = np.asarray(tr.magnetic_field_magnitude_T)[mask_tr]
    B_mag_rel  = (B_rx_mag_c - B_tr_mag_c) / np.clip(B_tr_mag_c, 1e-6, None)

    B_rx_vecs_c = interpolate_to_arc(s_rx, B_rx_vecs, s_common)
    B_tr_vecs_c = np.asarray(tr.magnetic_field_cart)[mask_tr]

    B_rx_mag_c2d = np.linalg.norm(B_rx_vecs_c, axis=1, keepdims=True)
    B_tr_mag_c2d = np.linalg.norm(B_tr_vecs_c, axis=1, keepdims=True)
    B_rx_hat = B_rx_vecs_c / np.clip(B_rx_mag_c2d, 1e-9, None)
    B_tr_hat = B_tr_vecs_c / np.clip(B_tr_mag_c2d, 1e-9, None)

    cos_angle     = np.clip(np.sum(B_rx_hat * B_tr_hat, axis=1), -1.0, 1.0)
    B_dir_ang_deg = np.degrees(np.arccos(cos_angle))

    ne_rx_all = np.asarray(rx.beam_profile.electron_density)
    te_rx_all = np.asarray(rx.beam_profile.electron_temperature)
    ne_rx_c = interpolate_to_arc(s_rx, ne_rx_all, s_common)
    te_rx_c = interpolate_to_arc(s_rx, te_rx_all, s_common)

    # Use TRAVIS output directly — not a re-evaluation of the profile formula.
    ne_tr_c = np.asarray(tr.electron_density_1e20)[mask_tr]
    te_tr_c = np.asarray(tr.electron_temperature_keV)[mask_tr]
    # Normalise by the peak (central) value so near-edge zeros don't inflate errors.
    ne_ref = np.clip(np.max(ne_tr_c), 1e-9, None)
    te_ref = np.clip(np.max(te_tr_c), 1e-9, None)
    ne_diff_rel = (ne_rx_c - ne_tr_c) / ne_ref
    te_diff_rel = (te_rx_c - te_tr_c) / te_ref

    kappa_rx = _ray_curvature(np.asarray(rx.beam_profile.position), s_rx)
    kappa_tr = _ray_curvature(np.asarray(tr.position_m), np.asarray(tr.arc_length_m))
    kappa_rx_c = interpolate_to_arc(s_rx, kappa_rx, s_common)
    kappa_tr_c = kappa_tr[mask_tr]

    def _deflection_deg(pos: np.ndarray) -> float:
        """Angle [deg] between entry and exit direction vectors."""
        d_in  = pos[1]  - pos[0]
        d_out = pos[-1] - pos[-2]
        d_in  /= np.linalg.norm(d_in)  + 1e-12
        d_out /= np.linalg.norm(d_out) + 1e-12
        return float(np.degrees(np.arccos(np.clip(np.dot(d_in, d_out), -1.0, 1.0))))

    defl_rx = _deflection_deg(pos_rx_common)
    defl_tr = _deflection_deg(pos_tr_common)

    # Whole-trajectory curvature and arc-in-plasma (not limited to common grid).
    rho_rx_all = np.asarray(rx.beam_profile.normalized_effective_radius)
    s_rx_inside = s_rx[rho_rx_all < 1.0]
    arc_in_plasma_rx = float(s_rx_inside[-1] - s_rx_inside[0]) if len(s_rx_inside) >= 2 else 0.0

    s_tr_inside = s_tr[rho_tr_all < 1.0]
    arc_in_plasma_tr = float(s_tr_inside[-1] - s_tr_inside[0]) if len(s_tr_inside) >= 2 else 0.0

    kappa_rx_inside = kappa_rx[rho_rx_all < 1.0]
    kappa_tr_inside = kappa_tr[rho_tr_all < 1.0]
    max_kappa_rx = float(kappa_rx_inside.max()) if len(kappa_rx_inside) else 0.0
    max_kappa_tr = float(kappa_tr_inside.max()) if len(kappa_tr_inside) else 0.0

    Y_param = 28.0 * b0_target / frequency_ghz  # f_ce / f

    # ── "At TRAVIS xyz" metrics ──────────────────────────────────────────────
    at_xyz_B_mag_rel   = None
    at_xyz_B_dir_deg   = None
    at_xyz_ne_diff_rel = None
    at_xyz_te_diff_rel = None
    at_xyz_rho_diff    = None
    at_xyz_B_mag_mean_pct = 0.0
    at_xyz_B_dir_rms_deg  = 0.0
    at_xyz_ne_mean_pct    = 0.0
    at_xyz_te_mean_pct    = 0.0
    at_xyz_rho_rms        = 0.0

    if interpolators is not None:
        B_rx_at_tr, rho_rx_at_tr, ne_rx_at_tr, te_rx_at_tr = _eval_rx_at_xyz(
            pos_tr_common, interpolators, nfp
        )
        B_rx_at_tr_mag = np.linalg.norm(B_rx_at_tr, axis=1)
        at_xyz_B_mag_rel = (B_rx_at_tr_mag - B_tr_mag_c) / np.clip(B_tr_mag_c, 1e-6, None)

        B_rx_at_tr_hat = B_rx_at_tr / np.linalg.norm(B_rx_at_tr, axis=1, keepdims=True).clip(1e-9)
        cos_at_tr = np.clip(np.sum(B_rx_at_tr_hat * B_tr_hat, axis=1), -1.0, 1.0)
        at_xyz_B_dir_deg = np.degrees(np.arccos(cos_at_tr))

        at_xyz_ne_diff_rel = (ne_rx_at_tr - ne_tr_c) / ne_ref
        at_xyz_te_diff_rel = (te_rx_at_tr - te_tr_c) / te_ref
        at_xyz_rho_diff    = rho_rx_at_tr - rho_tr_common

        at_xyz_B_mag_mean_pct = float(np.mean(np.abs(at_xyz_B_mag_rel)) * 100)
        at_xyz_B_dir_rms_deg  = float(np.sqrt(np.mean(at_xyz_B_dir_deg**2)))
        at_xyz_ne_mean_pct    = float(np.mean(np.abs(at_xyz_ne_diff_rel)) * 100)
        at_xyz_te_mean_pct    = float(np.mean(np.abs(at_xyz_te_diff_rel)) * 100)
        at_xyz_rho_rms        = float(np.sqrt(np.mean(at_xyz_rho_diff**2)))

    return TrajectoryComparison(
        s_common=s_common,
        pos_dist_mm=pos_dist_mm,
        rho_diff=rho_diff,
        rho_rx=rho_rx_common,
        rho_tr=rho_tr_common,
        B_mag_rx=B_rx_mag_c,
        B_mag_tr=B_tr_mag_c,
        B_mag_rel_err=B_mag_rel,
        B_dir_angle_deg=B_dir_ang_deg,
        ne_rx=ne_rx_c,
        ne_tr=ne_tr_c,
        te_rx=te_rx_c,
        te_tr=te_tr_c,
        ne_diff_rel=ne_diff_rel,
        te_diff_rel=te_diff_rel,
        curvature_rx=kappa_rx_c,
        curvature_tr=kappa_tr_c,
        min_rho_tr=float(rho_tr_all.min()),
        min_rho_rx=float(rho_rx_all.min()),
        max_ne_tr=float(ne_tr_c.max()),
        max_te_tr=float(te_tr_c.max()),
        max_ne_rx=float(ne_rx_c.max()),
        max_te_rx=float(te_rx_c.max()),
        arc_in_plasma_tr_m=arc_in_plasma_tr,
        arc_in_plasma_rx_m=arc_in_plasma_rx,
        max_curvature_tr_m_inv=max_kappa_tr,
        max_curvature_rx_m_inv=max_kappa_rx,
        Y_param=Y_param,
        pos_rms_mm=float(np.sqrt(np.mean(pos_dist_mm**2))),
        rho_rms=float(np.sqrt(np.mean(rho_diff**2))),
        B_mag_mean_err_pct=float(np.mean(np.abs(B_mag_rel)) * 100),
        B_dir_rms_deg=float(np.sqrt(np.mean(B_dir_ang_deg**2))),
        ne_mean_err_pct=float(np.mean(np.abs(ne_diff_rel)) * 100),
        te_mean_err_pct=float(np.mean(np.abs(te_diff_rel)) * 100),
        deflection_rx_deg=defl_rx,
        deflection_tr_deg=defl_tr,
        deflection_diff_deg=defl_rx - defl_tr,
        B_mag_at_tr_xyz_rel_err=at_xyz_B_mag_rel,
        B_dir_at_tr_xyz_deg=at_xyz_B_dir_deg,
        ne_at_tr_xyz_diff_rel=at_xyz_ne_diff_rel,
        te_at_tr_xyz_diff_rel=at_xyz_te_diff_rel,
        rho_at_tr_xyz_diff=at_xyz_rho_diff,
        B_mag_at_tr_xyz_mean_err_pct=at_xyz_B_mag_mean_pct,
        B_dir_at_tr_xyz_rms_deg=at_xyz_B_dir_rms_deg,
        ne_at_tr_xyz_mean_err_pct=at_xyz_ne_mean_pct,
        te_at_tr_xyz_mean_err_pct=at_xyz_te_mean_pct,
        rho_at_tr_xyz_rms=at_xyz_rho_rms,
    )


def _pass_fail(value: float, threshold: float) -> str:
    return "PASS" if value <= threshold else "FAIL"


def print_trajectory_report(cmp: TrajectoryComparison) -> None:
    print("\n── Trajectory comparison ─────────────────────────────────────────────")
    print(f"  Arc-length range   : {cmp.s_common[0]:.3f} – {cmp.s_common[-1]:.3f} m "
          f"({len(cmp.s_common)} points)")
    print()

    def row(label, value, unit, threshold=None):
        thresh_str = f"  (threshold {threshold:.1f} {unit})" if threshold else ""
        pf = f"  [{_pass_fail(value, threshold)}]" if threshold else ""
        print(f"  {label:<28} {value:>8.3f} {unit}{thresh_str}{pf}")

    row("Position RMS",           cmp.pos_rms_mm,         "mm",  THRESH_POS_RMS_MM)
    row("ρ RMS",                  cmp.rho_rms,             "",    THRESH_RHO_RMS)
    print(f"  {'Max nₑ (TR/RX)':<28} {cmp.max_ne_tr:>8.2f} / {cmp.max_ne_rx:.2f} ×10²⁰m⁻³")
    print(f"  {'Max Tₑ (TR/RX)':<28} {cmp.max_te_tr:>8.2f} / {cmp.max_te_rx:.2f} keV")
    row("|B| mean error",         cmp.B_mag_mean_err_pct,  "%",   THRESH_B_MAG_PCT)
    row("B̂ RMS angle (rx vs tr)", cmp.B_dir_rms_deg,      "deg", THRESH_B_DIR_DEG)
    row("nₑ mean error",          cmp.ne_mean_err_pct,     "%",   THRESH_NE_PCT)
    row("Tₑ mean error",          cmp.te_mean_err_pct,     "%",   THRESH_TE_PCT)
    if cmp.B_mag_at_tr_xyz_mean_err_pct or cmp.B_dir_at_tr_xyz_rms_deg:
        print()
        print("  -- at TRAVIS xyz (same spatial point) --")
        row("|B| mean error @xyz",    cmp.B_mag_at_tr_xyz_mean_err_pct, "%",   THRESH_B_MAG_PCT)
        row("B̂ RMS angle @xyz",       cmp.B_dir_at_tr_xyz_rms_deg,     "deg", THRESH_B_DIR_DEG)
        row("nₑ mean error @xyz",     cmp.ne_at_tr_xyz_mean_err_pct,   "%",   THRESH_NE_PCT)
        row("Tₑ mean error @xyz",     cmp.te_at_tr_xyz_mean_err_pct,   "%",   THRESH_TE_PCT)
        row("ρ RMS @xyz",              cmp.rho_at_tr_xyz_rms,            "",    THRESH_RHO_RMS)
    row("Direction sensitivity",  cmp.direction_sensitivity, "m")
    print(f"  {'ODE steps (raytrax)':<28} {cmp.n_steps_rx:>8d}")
    print()
    print(f"  {'Deflection':28}  {'raytrax':>8}    {'TRAVIS':>8}    {'diff':>8}")
    print(f"  {'':28}  {cmp.deflection_rx_deg:>8.3f} deg {cmp.deflection_tr_deg:>8.3f} deg "
          f"{cmp.deflection_diff_deg:>+8.3f} deg")


def print_first_points(
    travis: TravisECRHOutput,
    rx: TraceResult,
    n: int = 5,
) -> None:
    """Print antenna point and first *n* inside-plasma points, side by side."""
    s_tr   = np.asarray(travis.arc_length_m)
    rho_tr = np.asarray(travis.rho)
    s_rx   = np.asarray(rx.beam_profile.arc_length)
    rho_rx = np.asarray(rx.beam_profile.normalized_effective_radius)

    inside_tr = np.where(rho_tr < 1.0)[0]
    inside_rx = np.where(rho_rx < 1.0)[0]

    print("\n── Entry points ──────────────────────────────────────────────────────")
    print(f"  {'':10}  {'TRAVIS':>17}  {'raytrax':>17}")
    print(f"  {'':10}  {'s [m]':>8}  {'ρ':>7}  {'s [m]':>8}  {'ρ':>7}")

    def tr_row(i):
        return f"{s_tr[i]:>8.4f}  {rho_tr[i]:>7.3f}"

    def rx_row(i):
        return f"{s_rx[i]:>8.4f}  {rho_rx[i]:>7.3f}"

    blank = f"{'':>8}  {'':>7}"

    print(f"  {'antenna':<10}  {tr_row(0)}  {rx_row(0)}")

    skipped_tr = inside_tr[0] - 1 if len(inside_tr) else 0
    skipped_rx = inside_rx[0] - 1 if len(inside_rx) else 0
    if skipped_tr > 0 or skipped_rx > 0:
        skip_tr = f"({skipped_tr} skipped)" if skipped_tr > 0 else ""
        skip_rx = f"({skipped_rx} skipped)" if skipped_rx > 0 else ""
        print(f"  {'...':10}  {skip_tr:>17}  {skip_rx:>17}")

    for k in range(n):
        label = f"plasma+{k}"
        t = tr_row(inside_tr[k]) if k < len(inside_tr) else blank
        r = rx_row(inside_rx[k]) if k < len(inside_rx) else blank
        print(f"  {label:<10}  {t}  {r}")

    remaining = max(len(inside_tr) - n, len(inside_rx) - n)
    if remaining > 0:
        print(f"  {'...':10}  ({remaining} more inside-plasma points)")


def run_scenario(
    params: ScenarioParams,
    eq: MagneticConfiguration,
    wout_nc: Path,
    travis_exe: Path,
    output_dir: Path,
    mesh_cache_dir: Path | None = None,
    verbose: bool = True,
    max_step_size: float = 0.05,
    rtol: float = 1e-4,
    atol: float = 1e-6,
    travis_hgrid: float = 0.022,
    travis_dphi: float = 2.0,
    travis_rk_accuracy: float = 1e-5,
    travis_max_rk_stepsize: float = 10.0,
    travis_input_format: str = "legacy",
    b_interp=None,
    rho_interp=None,
) -> TrajectoryComparison:
    """Run one TRAVIS + raytrax scenario and return the comparison."""

    rho_np, ne_np, te_np = build_profiles(
        params.ne_central, params.ne_parm, params.te_central, params.te_parm
    )
    profiles = RadialProfiles(
        rho=jnp.array(rho_np),
        electron_density=jnp.array(ne_np),
        electron_temperature=jnp.array(te_np),
    )

    if verbose:
        print("Running TRAVIS …")
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
        input_format=travis_input_format,
    )
    travis = run_travis(travis_exe, travis_params, output_dir=output_dir / "travis_run",
                        mesh_cache_dir=mesh_cache_dir)
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
    jax.block_until_ready(rx.beam_profile.arc_length)  # block until XLA computation is done
    raytrax_time = time.perf_counter() - t0_rx
    s_rx = np.asarray(rx.beam_profile.arc_length)
    if verbose:
        print(f"  {len(s_rx)} points, s=[{s_rx[0]:.3f}, {s_rx[-1]:.3f}] m  ({raytrax_time:.1f} s)")

    if verbose:
        print("Computing direction sensitivity …")
    interpolators = Interpolators(
        magnetic_field=b_interp if b_interp is not None else build_magnetic_field_interpolator(eq),
        rho=rho_interp if rho_interp is not None else build_rho_interpolator(eq),
        electron_density=build_electron_density_profile_interpolator(profiles),
        electron_temperature=build_electron_temperature_profile_interpolator(profiles),
        is_axisymmetric=eq.is_axisymmetric,
    )
    cmp = compare_trajectories(
        rx, travis,
        frequency_ghz=params.frequency_ghz,
        b0_target=params.b0_target,
        interpolators=interpolators,
        nfp=eq.nfp,
    )
    cmp.direction_sensitivity = compute_direction_sensitivity(
        eq, profiles, beam,
        settings=TracerSettings(max_step_size=max_step_size, relative_tolerance=rtol, absolute_tolerance=atol),
    )
    cmp.travis_wall_time_s = travis_time
    cmp.raytrax_wall_time_s = raytrax_time
    cmp.n_steps_rx = len(s_rx)

    if verbose:
        print_first_points(travis, rx)
        print_trajectory_report(cmp)
    return cmp
