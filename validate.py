"""Single-point trajectory validation for W7-X (Step 1).

Usage:
    python validate.py
    python validate.py --travis-exe /path/to/travis-nc
    python validate.py --output-dir results/my_run
"""

import argparse
from pathlib import Path

from raytrax.equilibrium.interpolate import CylindricalGridResolution, VmecGridResolution

# Sentinel to read grid defaults directly from raytrax – stays in sync automatically.
_DG = VmecGridResolution()

from travis import resolve_travis_exe
from w7x_setup import build_scaled_equilibrium, default_scenario_params, get_w7x_equilibrium, B0_TARGET
from step1_trajectory import run_scenario


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--travis-exe", type=Path, default=None)
    parser.add_argument("--equilibrium", type=Path, default=None,
                        help="VMEC wout NetCDF for TRAVIS; exported automatically if omitted.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/trajectory"))
    # raytrax resolution
    parser.add_argument("--max-step-size", type=float, default=0.05,
                        help="raytrax ODE max step size in metres (default 0.05).")
    parser.add_argument("--grid-n-r",     type=int, default=_DG.cylindrical.n_r,    help="Equilibrium grid n_r.")
    parser.add_argument("--grid-n-z",     type=int, default=_DG.cylindrical.n_z,    help="Equilibrium grid n_z.")
    parser.add_argument("--grid-n-phi",   type=int, default=_DG.cylindrical.n_phi,  help="Equilibrium grid n_phi.")
    parser.add_argument("--grid-n-rho",   type=int, default=_DG.n_rho,              help="Equilibrium grid n_rho.")
    parser.add_argument("--grid-n-theta", type=int, default=_DG.n_theta,            help="Equilibrium grid n_theta.")
    args = parser.parse_args()

    travis_exe = resolve_travis_exe(args.travis_exe)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    grid = VmecGridResolution(
        n_rho=args.grid_n_rho,
        n_theta=args.grid_n_theta,
        cylindrical=CylindricalGridResolution(
            n_r=args.grid_n_r,
            n_z=args.grid_n_z,
            n_phi=args.grid_n_phi,
        ),
    )

    print("Loading W7-X equilibrium …")
    wout = get_w7x_equilibrium()
    eq   = build_scaled_equilibrium(wout, B0_TARGET, grid=grid)

    wout_nc = args.equilibrium
    if wout_nc is None:
        wout_nc = output_dir / "w7x.nc"
        if not wout_nc.exists():
            print(f"  Exporting equilibrium → {wout_nc} …")
            wout.save(wout_nc)

    run_scenario(default_scenario_params(), eq, wout_nc, travis_exe, output_dir,
                 max_step_size=args.max_step_size)


if __name__ == "__main__":
    main()
