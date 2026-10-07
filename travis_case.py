"""Write (and optionally run) the TRAVIS input of individual step-2 scan samples.

Regenerates the scenarios of `scan.py --step 2` with the same seed, so sample *i*
here is `run_<i>` of the scan.

Usage:
    python travis_case.py 1 180 877 --output-dir cases
    python travis_case.py 1 --travis-input-format v13.3.7 --run --travis-exe /path/to/travis-nc
"""

from __future__ import annotations

import argparse
from itertools import islice
from pathlib import Path

import numpy as np

from scan import B0_TARGET_STEP2, step2_scenarios
from step2_absorption import build_travis_input
from travis import _write_travis_input_files, resolve_travis_exe, run_travis
from w7x_setup import build_scaled_equilibrium, get_w7x_equilibrium

from raytrax.equilibrium.interpolate import build_magnetic_field_interpolator, build_rho_interpolator


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("indices", type=int, nargs="+", help="Sample indices (N in run_N of the scan).")
    parser.add_argument("--seed", type=int, default=1, help="Scan seed (1 for the paper scans).")
    parser.add_argument("--travis-input-format", choices=["legacy", "v13.3.7"], default="legacy",
                        help="'legacy' for TRAVIS 13.3.1, 'v13.3.7' for 13.3.7 and newer.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/travis_cases"))
    parser.add_argument("--run", action="store_true", help="Also run TRAVIS and print the final optical depth.")
    parser.add_argument("--travis-exe", type=Path, default=None)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    wout = get_w7x_equilibrium()
    wout_nc = args.output_dir / "w7x.nc"
    if not wout_nc.exists():
        wout.save(wout_nc)
    eq = build_scaled_equilibrium(wout, B0_TARGET_STEP2)
    scenarios = step2_scenarios(build_rho_interpolator(eq), build_magnetic_field_interpolator(eq), args.seed)
    samples = list(islice(scenarios, max(args.indices) + 1))

    for i in args.indices:
        params = samples[i]
        travis_params = build_travis_input(params, wout_nc, travis_input_format=args.travis_input_format)
        run_dir = args.output_dir / f"run_{i:05d}"
        label = f"run_{i:05d}  {params.mode}{params.harmonic}  {params.frequency_ghz:.3f} GHz"
        if args.run:
            out = run_travis(resolve_travis_exe(args.travis_exe), travis_params, output_dir=run_dir)
            tau = float(out.optical_depth[-1])
            print(f"{label}  tau = {tau:.4g}  absorbed = {1 - np.exp(-tau):.3f}  ({run_dir})")
        else:
            run_dir.mkdir(exist_ok=True)
            _write_travis_input_files(travis_params, run_dir, use_mesh=True, save_mesh=False)
            print(f"{label}  -> {run_dir / 'travis_input.data'}")


if __name__ == "__main__":
    main()
