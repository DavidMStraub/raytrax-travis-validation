#!/usr/bin/env python3
r"""Compute the divergence of the interpolated B field for W7-X.

Evaluates ∇·B analytically using the interpax interpolator's built-in
derivative support (dx/dy/dz kwargs) — the same gradients that enter
jax.grad when differentiating the ray-tracing Hamiltonian.

Study:
  Four cylindrical grid resolutions × two interpax methods ('linear' / 'cubic2')
  are compared on a fixed physical evaluation grid.  Only the plasma interior
  (rho < 1) is assessed.

Divergence in cylindrical coordinates:

    ∇·B = (1/R) ∂(R B_R)/∂R  +  (1/R) ∂B_φ/∂φ  +  ∂B_Z/∂Z

Usage:
    python scripts/investigate_div_B.py
"""
import os
import time
import warnings

os.environ["JAX_ENABLE_X64"] = "1"

import numpy as np
import jax.numpy as jnp
import interpax

from raytrax.examples.w7x import get_w7x_equilibrium
from raytrax.equilibrium.interpolate import (
    CylindricalGridResolution,
    MagneticConfiguration,
    VmecGridResolution,
    build_rho_interpolator,
)


# ---------------------------------------------------------------------------
# Fixed physical evaluation grid — same for all stored-grid configurations so
# comparisons are apples-to-apples.  Coordinates are slightly inside the
# equilibrium extent so no stored-grid config needs to extrapolate.
# ---------------------------------------------------------------------------
_N_EVAL   = (80, 40, 90)          # (n_R, n_phi, n_Z)
_R_EVAL   = (4.60, 6.20)          # metres
_PHI_EVAL = (0.02, 0.61)          # radians  (fundamental domain 0 … π/nfp ≈ 0.628)
_Z_EVAL   = (-0.90, 0.93)         # metres

GRID_CONFIGS = [
    (
        "coarse  nR=50  nPhi=25 nZ=50",
        VmecGridResolution(
            CylindricalGridResolution(n_r=50,  n_z=50,  n_phi=25),
            n_rho=20, n_theta=22,
        ),
    ),
    (
        "default nR=95  nPhi=50 nZ=105",
        VmecGridResolution(
            CylindricalGridResolution(n_r=95,  n_z=105, n_phi=50),
            n_rho=40, n_theta=45,
        ),
    ),
    (
        "fine    nR=150 nPhi=80 nZ=170",
        VmecGridResolution(
            CylindricalGridResolution(n_r=150, n_z=170, n_phi=80),
            n_rho=60, n_theta=70,
        ),
    ),
    (
        "v.fine  nR=200 nPhi=100 nZ=220",
        VmecGridResolution(
            CylindricalGridResolution(n_r=200, n_z=220, n_phi=100),
            n_rho=80, n_theta=90,
        ),
    ),
]


def make_eval_grid() -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return the fixed (R, phi, Z) evaluation cloud as three flat JAX arrays."""
    n_er, n_ep, n_ez = _N_EVAL
    Rg, pg, Zg = np.meshgrid(
        np.linspace(*_R_EVAL,   n_er),
        np.linspace(*_PHI_EVAL, n_ep),
        np.linspace(*_Z_EVAL,   n_ez),
        indexing="ij",
    )
    return jnp.array(Rg.ravel()), jnp.array(pg.ravel()), jnp.array(Zg.ravel())


def analytic_div_B(
    interp: interpax.Interpolator3D,
    R_f: jnp.ndarray,
    phi_f: jnp.ndarray,
    Z_f: jnp.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact ∇·B at each point via the interpax derivative interface.

    Returns
    -------
    div_B : ndarray (N,) in T/m
    absB  : ndarray (N,) in T
    """
    B      = np.array(interp(R_f, phi_f, Z_f))
    dB_dR  = np.array(interp(R_f, phi_f, Z_f, dx=1))
    dB_dp  = np.array(interp(R_f, phi_f, Z_f, dy=1))
    dB_dZ  = np.array(interp(R_f, phi_f, Z_f, dz=1))
    Rn = np.array(R_f)
    div_B = dB_dR[:, 0] + B[:, 0] / Rn + dB_dp[:, 1] / Rn + dB_dZ[:, 2]
    return div_B, np.linalg.norm(B, axis=-1)


def report(
    div_B: np.ndarray,
    absB: np.ndarray,
    rho: np.ndarray,
    label: str,
) -> None:
    """Print |∇·B|/|B| statistics for plasma interior (rho < 1), by zone."""
    def zone_stats(mask):
        m = mask & (absB > 0.05)
        if not m.any():
            return None
        rel = np.abs(div_B[m]) / absB[m]
        return len(rel), rel.mean(), float(np.median(rel)), float(np.percentile(rel, 95))

    all_stats  = zone_stats(rho < 1.0)
    core_stats = zone_stats(rho < 0.5)
    out_stats  = zone_stats((rho >= 0.5) & (rho < 1.0))

    if all_stats is None:
        print(f"  {label}: no valid interior points")
        return

    n, mean, p50, p95 = all_stats
    print(f"  {label}")
    print(f"    all rho<1   (n={n:>6d}): mean={mean:.3e}  p50={p50:.3e}  p95={p95:.3e}")
    if core_stats:
        print(f"    core rho<0.5(n={core_stats[0]:>6d}): mean={core_stats[1]:.3e}")
    if out_stats:
        print(f"    outer 0.5-1 (n={out_stats[0]:>6d}): mean={out_stats[1]:.3e}")


def main() -> None:
    print("Loading W7-X equilibrium …")
    eq = get_w7x_equilibrium()
    print(f"  nfp={eq.nfp}  ns={eq.ns}  lasym={eq.lasym}")

    R_f, phi_f, Z_f = make_eval_grid()
    n_er, n_ep, n_ez = _N_EVAL
    print(f"\nEvaluation grid: {n_er}×{n_ep}×{n_ez} = {R_f.size} points")
    print(f"  R ∈ {_R_EVAL} m")
    print(f"  phi ∈ {_PHI_EVAL} rad  (full fundamental domain: 0 … {np.pi/eq.nfp:.4f})")
    print(f"  Z ∈ {_Z_EVAL} m")
    print("\nMetric: |∇·B|/|B| [m⁻¹]  (physical scale |∇B|/|B| ≈ 2–5 m⁻¹ for W7-X)")

    for label, grid in GRID_CONFIGS:
        print(f"\n{'─'*68}")
        print(f"Grid: {label}")

        t0 = time.perf_counter()
        mc = MagneticConfiguration.from_vmec_wout(eq, grid=grid)
        t_build = time.perf_counter() - t0

        rphiz  = np.array(mc.rphiz)
        R_s    = rphiz[:, 0, 0, 0]
        phi_s  = rphiz[0, :, 0, 1]
        Z_s    = rphiz[0, 0, :, 2]
        dR_mm  = 1000 * (R_s[1]   - R_s[0])
        dZ_mm  = 1000 * (Z_s[1]   - Z_s[0])
        dphi_mrad = 1000 * (phi_s[1] - phi_s[0])
        print(f"  Build: {t_build:.0f}s  |  ΔR={dR_mm:.1f}mm  Δφ={dphi_mrad:.1f}mrad  ΔZ={dZ_mm:.1f}mm")

        B_stored   = jnp.nan_to_num(mc.magnetic_field, nan=0.0)
        rho_interp = build_rho_interpolator(mc)
        rho_f      = np.array(rho_interp(R_f, phi_f, Z_f))

        for method in ("linear", "cubic2"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                interp = interpax.Interpolator3D(
                    x=jnp.array(R_s),
                    y=jnp.array(phi_s),
                    z=jnp.array(Z_s),
                    f=B_stored,
                    method=method,
                    extrap=0.0,
                )
            t1 = time.perf_counter()
            div_B, absB = analytic_div_B(interp, R_f, phi_f, Z_f)
            t_eval = time.perf_counter() - t1
            report(div_B, absB, rho_f, f"method={method}  (eval {t_eval:.0f}s)")


if __name__ == "__main__":
    main()
