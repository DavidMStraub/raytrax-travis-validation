"""Random parameter scan for W7-X ECRH validation.

Runs the requested validation step over many randomly sampled scenarios
and logs results to WandB (or prints them in dry-run mode).

Usage:
    python scan.py --step 1 --n-samples 100 --dry-run
    python scan.py --step 1 --n-samples 1000 --wandb-project raytrax-validation
    python scan.py --step 2 --n-samples 24 --dry-run
    python scan.py --step 2 --n-samples 500 --wandb-project raytrax-validation --wandb-group absorption
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path
from typing import Literal, cast

import jax
import jax.numpy as jnp
import numpy as np
jax.config.update("jax_enable_x64", True)
from tqdm import tqdm

from profiles import build_profiles, cylindrical_to_cartesian
from sampling import compute_frequency_range, sample_scenario
from step1_trajectory import run_scenario
from step2_absorption import (
    compute_resonance_frequency_range,
    run_absorption_scenario,
    sample_absorption_scenario,
)
from travis import resolve_travis_exe
from wandb_log import init_scan, log_sample, finish_scan
from w7x_setup import (
    build_scaled_equilibrium,
    get_w7x_equilibrium,
    B0_TARGET,
    ANTENNA_R_MIN, ANTENNA_R_MAX, ANTENNA_Z_MAX,
    TARGET_R_MIN, TARGET_R_MAX, TARGET_Z_MAX, TARGET_RHO_MAX,
)

# On-axis B₀ for step 2: places 2nd harmonic at 140 GHz (standard W7-X ECRH).
B0_TARGET_STEP2 = 2.5

from raytrax.api import trace
from raytrax.equilibrium.interpolate import (
    CylindricalGridResolution,
    VmecGridResolution,
    build_magnetic_field_interpolator,
    build_rho_interpolator,
)
from raytrax.types import Beam, RadialProfiles, TracerSettings

# Sentinel to read grid defaults directly from raytrax – stays in sync automatically.
_DG = VmecGridResolution()


# ── Shared helpers ────────────────────────────────────────────────────────────

def _build_grid(args: argparse.Namespace) -> VmecGridResolution:
    return VmecGridResolution(
        n_rho=args.grid_n_rho,
        n_theta=args.grid_n_theta,
        cylindrical=CylindricalGridResolution(
            n_r=args.grid_n_r,
            n_z=args.grid_n_z,
            n_phi=args.grid_n_phi,
        ),
    )


def _ensure_wout_nc(args: argparse.Namespace, output_dir: Path, wout) -> Path:
    if args.equilibrium is not None:
        return args.equilibrium
    wout_nc = output_dir / "w7x.nc"
    if not wout_nc.exists():
        print(f"  Exporting equilibrium \u2192 {wout_nc} \u2026")
        wout.save(wout_nc)
    return wout_nc


def _warmup_jit(eq, params, max_step_size: float, rtol: float = 1e-4, atol: float = 1e-6) -> None:
    """Trigger JAX JIT compilation with a single sample scenario."""
    profiles_np = build_profiles(
        params.ne_central, params.ne_parm,
        params.te_central, params.te_parm,
    )
    rx_profiles = RadialProfiles(
        rho=jnp.array(profiles_np[0]),
        electron_density=jnp.array(profiles_np[1]),
        electron_temperature=jnp.array(profiles_np[2]),
    )
    beam = Beam(
        position=jnp.array(cylindrical_to_cartesian(*params.antenna_cyl)),
        direction=jnp.array(params.direction_cart),
        frequency=jnp.array(params.frequency_ghz * 1e9),
        mode=params.mode,
        power=params.power_mw * 1e6,
    )
    np.asarray(trace(eq, rx_profiles, beam,
                     settings=TracerSettings(max_step_size=max_step_size,
                                             relative_tolerance=rtol,
                                             absolute_tolerance=atol)).beam_profile.arc_length)
    print("  JIT compile done.")


def _scan_step1(args: argparse.Namespace) -> None:
    travis_exe = resolve_travis_exe(args.travis_exe)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    grid = _build_grid(args)

    print("Loading W7-X equilibrium …")
    wout = get_w7x_equilibrium()
    eq   = build_scaled_equilibrium(wout, B0_TARGET, grid=grid)

    rho_interp = build_rho_interpolator(eq)
    B_interp   = build_magnetic_field_interpolator(eq)

    print("Computing safe frequency range …")
    f_min_ghz, f_max_ghz = compute_frequency_range(B_interp, rho_interp)
    print(f"  frequency range: {f_min_ghz:.1f} – {f_max_ghz:.1f} GHz  (sub-O1)")

    wout_nc = _ensure_wout_nc(args, output_dir, wout)

    scan_config = {
        "n_samples":             args.n_samples,
        "seed":                  args.seed,
        "raytrax_max_step_size": args.max_step_size,
        "raytrax_rtol":          args.raytrax_rtol,
        "raytrax_atol":          args.raytrax_atol,
        "eq_grid_n_r":           args.grid_n_r,
        "eq_grid_n_z":           args.grid_n_z,
        "eq_grid_n_phi":         args.grid_n_phi,
        "eq_grid_n_rho":         args.grid_n_rho,
        "eq_grid_n_theta":       args.grid_n_theta,
        "travis_hgrid":          args.travis_hgrid,
        "travis_dphi":           args.travis_dphi,
        "travis_rk_accuracy":    args.travis_rk_accuracy,
        "travis_max_rk_stepsize": args.travis_max_rk_stepsize,
        "travis_resonance_umax": args.travis_resonance_umax,
        "travis_resonance_grid_points": args.travis_resonance_grid_points,
        "f_min_ghz":             f_min_ghz,
        "f_max_ghz":             f_max_ghz,
    }
    init_scan(project=args.wandb_project, config=scan_config,
              group=args.wandb_group, name=args.wandb_name, dry_run=args.dry_run)

    mesh_cache_dir = args.mesh_cache_dir

    print("Warming up raytrax JIT …")
    _warmup_params = sample_scenario(
        rho_interp, f_min_ghz, f_max_ghz, np.random.default_rng(0),
        antenna_r_min=ANTENNA_R_MIN, antenna_r_max=ANTENNA_R_MAX, antenna_z_max=ANTENNA_Z_MAX,
        target_r_min=TARGET_R_MIN, target_r_max=TARGET_R_MAX, target_z_max=TARGET_Z_MAX,
        target_rho_max=TARGET_RHO_MAX, b0_target=B0_TARGET,
    )
    _warmup_jit(eq, _warmup_params, args.max_step_size, args.raytrax_rtol, args.raytrax_atol)

    rng = np.random.default_rng(args.seed)
    n_ok, n_fail = 0, 0
    bar = tqdm(range(args.n_samples), unit="run", dynamic_ncols=True)
    for i in bar:
        params = sample_scenario(
            rho_interp, f_min_ghz, f_max_ghz, rng,
            antenna_r_min=ANTENNA_R_MIN,
            antenna_r_max=ANTENNA_R_MAX,
            antenna_z_max=ANTENNA_Z_MAX,
            target_r_min=TARGET_R_MIN,
            target_r_max=TARGET_R_MAX,
            target_z_max=TARGET_Z_MAX,
            target_rho_max=TARGET_RHO_MAX,
            b0_target=B0_TARGET,
        )
        run_dir = output_dir / f"run_{i:05d}"
        run_dir.mkdir(exist_ok=True)

        bar.set_postfix(f=f"{params.frequency_ghz:.1f}GHz", mode=params.mode,
                        ne=f"{params.ne_central:.2f}", ok=n_ok, fail=n_fail)
        try:
            cmp = run_scenario(params, eq, wout_nc, travis_exe, run_dir,
                               mesh_cache_dir=mesh_cache_dir, verbose=False,
                               max_step_size=args.max_step_size,
                               rtol=args.raytrax_rtol,
                               atol=args.raytrax_atol,
                               travis_hgrid=args.travis_hgrid,
                               travis_dphi=args.travis_dphi,
                               travis_rk_accuracy=args.travis_rk_accuracy,
                               travis_max_rk_stepsize=args.travis_max_rk_stepsize)
            log_sample(params, cmp, step=i, dry_run=args.dry_run)
            n_ok += 1
        except Exception as exc:
            tqdm.write(f"  run_{i:05d} FAILED: {exc}")
            log_sample(params, None, step=i, error=str(exc), dry_run=args.dry_run)
            n_fail += 1
        
        # Clear JAX cache and run garbage collection every 100 samples to prevent OOM
        if (i + 1) % 100 == 0:
            jax.clear_caches()
            gc.collect()

    tqdm.write(f"\nDone: {n_ok} succeeded, {n_fail} failed.")
    finish_scan(dry_run=args.dry_run)


def _scan_step2(args: argparse.Namespace) -> None:
    travis_exe = resolve_travis_exe(args.travis_exe)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    grid = _build_grid(args)

    print("Loading W7-X equilibrium …")
    wout = get_w7x_equilibrium()
    eq   = build_scaled_equilibrium(wout, B0_TARGET_STEP2, grid=grid)

    print("Building interpolators (once for entire scan) …")
    rho_interp = build_rho_interpolator(eq)
    B_interp   = build_magnetic_field_interpolator(eq)
    
    # Build profile interpolators (will be recreated per sample with different profiles)
    # But the expensive B and rho interpolators are built once here
    from raytrax.tracer.buffers import Interpolators
    from raytrax.equilibrium.interpolate import (
        build_electron_density_profile_interpolator,
        build_electron_temperature_profile_interpolator,
    )
    # Create a dummy profile for JIT warmup
    _dummy_profiles = RadialProfiles(
        rho=jnp.linspace(0, 1, 501),
        electron_density=jnp.ones(501) * 0.5,
        electron_temperature=jnp.ones(501) * 2.0,
    )
    base_interpolators = Interpolators(
        magnetic_field=B_interp,
        rho=rho_interp,
        electron_density=build_electron_density_profile_interpolator(_dummy_profiles),
        electron_temperature=build_electron_temperature_profile_interpolator(_dummy_profiles),
        is_axisymmetric=eq.is_axisymmetric,
    )

    print("Computing harmonic frequency ranges …")
    f_lo_h1, f_hi_h1 = compute_resonance_frequency_range(B_interp, rho_interp, harmonic=1)
    f_lo_h2, f_hi_h2 = compute_resonance_frequency_range(B_interp, rho_interp, harmonic=2)
    print(f"  Harmonic 1 (fundamental) : {f_lo_h1:.1f} – {f_hi_h1:.1f} GHz")
    print(f"  Harmonic 2 (2nd)         : {f_lo_h2:.1f} – {f_hi_h2:.1f} GHz")

    wout_nc = _ensure_wout_nc(args, output_dir, wout)

    scan_config = {
        "n_samples":             args.n_samples,
        "seed":                  args.seed,
        "b0_target":             B0_TARGET_STEP2,
        "raytrax_max_step_size": args.max_step_size,
        "raytrax_rtol":          args.raytrax_rtol,
        "raytrax_atol":          args.raytrax_atol,
        "eq_grid_n_r":           args.grid_n_r,
        "eq_grid_n_z":           args.grid_n_z,
        "eq_grid_n_phi":         args.grid_n_phi,
        "eq_grid_n_rho":         args.grid_n_rho,
        "eq_grid_n_theta":       args.grid_n_theta,
        "travis_hgrid":          args.travis_hgrid,
        "travis_dphi":           args.travis_dphi,
        "travis_rk_accuracy":    args.travis_rk_accuracy,
        "travis_max_rk_stepsize": args.travis_max_rk_stepsize,
        "travis_resonance_umax": args.travis_resonance_umax,
        "travis_resonance_grid_points": args.travis_resonance_grid_points,
        "f_lo_h1_ghz":           f_lo_h1,
        "f_hi_h1_ghz":           f_hi_h1,
        "f_lo_h2_ghz":           f_lo_h2,
        "f_hi_h2_ghz":           f_hi_h2,
    }
    init_scan(project=args.wandb_project, config=scan_config,
              group=args.wandb_group, name=args.wandb_name, dry_run=args.dry_run)

    mesh_cache_dir = args.mesh_cache_dir

    # Cycle through all four (harmonic, mode) combinations so each is
    # represented roughly equally across the scan.
    _combos = [(1, "X"), (1, "O"), (2, "X"), (2, "O")]
    _f_ranges = {1: (f_lo_h1, f_hi_h1), 2: (f_lo_h2, f_hi_h2)}

    sample_kwargs = dict(
        antenna_r_min=ANTENNA_R_MIN,
        antenna_r_max=ANTENNA_R_MAX,
        antenna_z_max=ANTENNA_Z_MAX,
        target_r_min=TARGET_R_MIN,
        target_r_max=TARGET_R_MAX,
        target_z_max=TARGET_Z_MAX,
        target_rho_max=TARGET_RHO_MAX,
        b0_target=B0_TARGET_STEP2,
    )

    # Warm up raytrax JIT before starting the clock
    print("Warming up raytrax JIT …")
    _warmup_params = sample_absorption_scenario(
        2, "X", rho_interp, B_interp, np.random.default_rng(0),
        f_lo_ghz=f_lo_h2, f_hi_ghz=f_hi_h2,
        **sample_kwargs,
    )
    _warmup_jit(eq, _warmup_params, args.max_step_size, args.raytrax_rtol, args.raytrax_atol)

    rng = np.random.default_rng(args.seed)
    n_ok, n_fail = 0, 0
    bar = tqdm(range(args.n_samples), unit="run", dynamic_ncols=True)
    for i in bar:
        harmonic, mode = _combos[i % len(_combos)]
        f_lo, f_hi = _f_ranges[harmonic]
        params = sample_absorption_scenario(
            harmonic, cast(Literal["O", "X"], mode),
            rho_interp, B_interp, rng,
            f_lo_ghz=f_lo, f_hi_ghz=f_hi,
            **sample_kwargs,
        )
        run_dir = output_dir / f"run_{i:05d}"
        run_dir.mkdir(exist_ok=True)

        bar.set_postfix(
            h=harmonic, mode=mode,
            f=f"{params.frequency_ghz:.1f}GHz",
            ok=n_ok, fail=n_fail,
        )
        try:
            cmp = run_absorption_scenario(
                params, eq, wout_nc, travis_exe, run_dir,
                base_interpolators=base_interpolators,
                mesh_cache_dir=mesh_cache_dir,
                verbose=False,
                max_step_size=args.max_step_size,
                rtol=args.raytrax_rtol,
                atol=args.raytrax_atol,
                travis_hgrid=args.travis_hgrid,
                travis_dphi=args.travis_dphi,
                travis_rk_accuracy=args.travis_rk_accuracy,
                travis_max_rk_stepsize=args.travis_max_rk_stepsize,
                travis_resonance_umax=args.travis_resonance_umax,
                travis_resonance_grid_points=args.travis_resonance_grid_points,
            )
            log_sample(params, cmp, step=i, dry_run=args.dry_run)
            n_ok += 1
        except Exception as exc:
            tqdm.write(f"  run_{i:05d} FAILED: {exc}")
            log_sample(params, None, step=i, error=str(exc), dry_run=args.dry_run)
            n_fail += 1
        
        # Clear JAX cache and run garbage collection every 100 samples to prevent OOM
        if (i + 1) % 100 == 0:
            jax.clear_caches()
            gc.collect()

    tqdm.write(f"\nDone: {n_ok} succeeded, {n_fail} failed.")
    finish_scan(dry_run=args.dry_run)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--step", type=int, default=1, choices=[1, 2],
                        help="Validation step to run (1=trajectory, 2=absorption).")
    parser.add_argument("--n-samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--travis-exe", type=Path, default=None)
    parser.add_argument("--equilibrium", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("results/scan"))
    parser.add_argument("--wandb-project", type=str, default="raytrax-validation")
    parser.add_argument("--wandb-group", type=str, default=None,
                        help="WandB group (defaults to 'trajectory' for step 1, 'absorption' for step 2).")
    parser.add_argument("--wandb-name", type=str, default=None,
                        help="WandB run display name.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--mesh-cache-dir", type=Path, default=None,
                        help="Directory for TRAVIS mesh cache.")
    # raytrax resolution
    parser.add_argument("--max-step-size", type=float, default=0.05,
                        help="raytrax ODE max step size in metres (default 0.05).")
    parser.add_argument("--raytrax-rtol", type=float, default=1e-4,
                        help="raytrax ODE relative tolerance (default 1e-4).")
    parser.add_argument("--raytrax-atol", type=float, default=1e-6,
                        help="raytrax ODE absolute tolerance (default 1e-6).")
    parser.add_argument("--grid-n-r",     type=int, default=_DG.cylindrical.n_r,    help="Equilibrium grid n_r.")
    parser.add_argument("--grid-n-z",     type=int, default=_DG.cylindrical.n_z,    help="Equilibrium grid n_z.")
    parser.add_argument("--grid-n-phi",   type=int, default=_DG.cylindrical.n_phi,  help="Equilibrium grid n_phi.")
    parser.add_argument("--grid-n-rho",   type=int, default=_DG.n_rho,              help="Equilibrium grid n_rho.")
    parser.add_argument("--grid-n-theta", type=int, default=_DG.n_theta,            help="Equilibrium grid n_theta.")
    # TRAVIS mesh resolution
    parser.add_argument("--travis-hgrid", type=float, default=0.022,
                        help="TRAVIS poloidal mesh step [m] (default 0.022 ≈ rminor/25, min ~0.002).")
    parser.add_argument("--travis-dphi", type=float, default=2.0,
                        help="TRAVIS toroidal mesh step [deg] (default 2°, min ~0.2°).")
    parser.add_argument("--travis-rk-accuracy", type=float, default=1e-5,
                        help="TRAVIS RK integrator accuracy (default 1e-5).")
    parser.add_argument("--travis-max-rk-stepsize", type=float, default=10.0,
                        help="TRAVIS max RK step size in wavelengths (default 10).")
    parser.add_argument("--travis-resonance-umax", type=float, default=7.0,
                        help="TRAVIS upper velocity for resonance integral in u/u_th (umaxgrid, default 7; use 35 for O2 mode).")
    parser.add_argument("--travis-resonance-grid-points", type=int, default=700,
                        help="TRAVIS u_par grid points for resonance integral (nugrid, default 700; use 1500 for O2 mode).")
    args = parser.parse_args()

    if args.wandb_group is None:
        args.wandb_group = "trajectory" if args.step == 1 else "absorption"

    if args.step == 1:
        _scan_step1(args)
    elif args.step == 2:
        _scan_step2(args)


if __name__ == "__main__":
    main()
