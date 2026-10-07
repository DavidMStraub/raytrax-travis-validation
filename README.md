# raytrax-travis-validation

Compares the [Raytrax](https://github.com/proximafusion/raytrax) JAX ray tracer against the TRAVIS Fortran ECRH code on W7-X stellarator scenarios. Accompanies the Raytrax paper.

## Overview

The validation is organised in two independent steps:

| Step | Script | What is tested |
|------|--------|----------------|
| **1 — Trajectory** | `scan.py --step 1` | Ray path (position, ρ, \|B\|, B̂, nₑ, Tₑ) |
| **2 — Absorption** | `scan.py --step 2` | Absorption coefficient α, optical depth τ, absorbed power, deposition centroid ⟨ρ⟩ |

`scan.py` runs randomised scans with WandB logging. `validate.py` runs a single step-1 reference scenario.
Both steps use the W7-X VMEC equilibrium shipped with `raytrax`, scaled to a chosen on-axis field B₀. Both codes trace with the cold-plasma Hamiltonian.

For each sample a single-ray TRAVIS run is compared with Raytrax. The TRAVIS optical depth is the `tau` column of `beamtrace_1`; the TRAVIS absorbed power is P_in (1 − exp(−τ)). Deposition centroids are the power-weighted mean ρ along the ray, computed in the same way for both codes.

## Prerequisites

```bash
pip install -r requirements.txt   # installs raytrax from GitHub, pinned to a specific commit
```

A TRAVIS executable with NetCDF support (`travis-nc`) is required. Set it in a `.env` file in this directory,

```
TRAVIS_EXE=/path/to/travis-nc
```

or pass `--travis-exe`. The validation for the paper used **TRAVIS release 13.3.1**, which reads the legacy input-file layout (the default). For TRAVIS 13.3.7 and newer, whose input file uses a different layout for the dielectric-tensor and pass-count lines, pass `--travis-input-format v13.3.7`.

Scans log to [Weights & Biases](https://wandb.ai) (`wandb login` once; project set with `--wandb-project`, default `raytrax-validation`). Use `--dry-run` to print metrics locally instead.

## Reproducing the paper results

1. Build TRAVIS 13.3.1 with NetCDF support and set `TRAVIS_EXE` to the resulting `travis-nc`.
2. Run the two default scans (1000 samples, seed 1). Step 1 and step 2 use different B₀, so they need separate mesh caches:

   ```bash
   python scan.py --step 1 --seed 1 --n-samples 1000 --wandb-name step1_default \
       --mesh-cache-dir results/mesh_cache_step1 --output-dir results/scan_step1_default
   python scan.py --step 2 --seed 1 --n-samples 1000 --wandb-name step2_default \
       --mesh-cache-dir results/mesh_cache_step2 --output-dir results/scan_step2_default
   ```

3. Create the figures and print the quoted statistics:

   ```bash
   python paper_figures.py
   ```

   This reads the most recent WandB runs named `step1_default` and `step2_default` and writes `fig1_trajectory.pdf` and `fig2_absorption.pdf` to `figures/`. Use `--step1`/`--step2` to select runs by ID or name, `--wandb-project` to select the project, and `--output-dir` to change the target directory.

The scripts `run_step1_grid_scans.sh`, `run_step1_solver_scans.sh` and `run_step2_scans.sh` run the default scans together with the fidelity variants (finer Raytrax grid, finer TRAVIS mesh and solver settings, tighter Raytrax ODE tolerances). `compare_step1_runs.py` and `hist_step2_runs.py` compare these runs from WandB.

## Step 1 — Trajectory validation

B₀ = 3 T, so the second-harmonic resonance lies at about 168 GHz. Random frequencies are chosen well below the fundamental resonance, so absorption is negligible and the comparison isolates geometric errors.

### Single reference scenario

```bash
python validate.py
python validate.py --travis-exe /path/to/travis-nc --output-dir results/my_run
```

### Random scan

Random antenna positions, directions, frequencies and plasma profiles:

```bash
python scan.py --step 1 --n-samples 12 --dry-run --seed 0
python scan.py --step 1 --n-samples 1000 --seed 1 --wandb-project raytrax-validation
```

### Metrics

On the common arc-length grid (the threshold is the pass/fail level in the printed report):

| Metric | Threshold | Description |
|--------|-----------|-------------|
| `pos_rms_mm` | 5 mm | RMS position distance inside the plasma |
| `rho_rms` | 0.02 | RMS ρ difference |
| `B_mag_mean_err_pct` | 2 % | Mean \|B\| relative error |
| `B_dir_rms_deg` | 1 ° | RMS angle between B̂ vectors |
| `ne_mean_err_pct`, `te_mean_err_pct` | 3 % | Mean error, normalised to the TRAVIS maximum |

Evaluated at the TRAVIS positions (no interpolation along the ray), with the same thresholds: `B_mag_at_tr_xyz_mean_err_pct`, `B_dir_at_tr_xyz_rms_deg`, `ne_at_tr_xyz_mean_err_pct`, `te_at_tr_xyz_mean_err_pct`, `rho_at_tr_xyz_rms`.

Further logged quantities: ray extrema (`min_rho_*`, `max_ne_*`, `max_te_*`), `arc_in_plasma_*_m`, `max_curvature_*_m_inv`, `deflection_*_deg`, `Y_param` (f_ce / f), `direction_sensitivity`, wall times and `n_steps_rx`.

## Step 2 — Absorption validation

B₀ is scaled to 2.5 T so that the second harmonic lies at 140 GHz. Random antenna positions, directions, frequencies and profiles are sampled for X and O mode at the fundamental and second harmonic, cycling through X1, O1, X2, O2. The frequency ranges follow from the B field inside the plasma (about 53–83 GHz for the first and 106–166 GHz for the second harmonic), so that only one harmonic resonates in the plasma.

```bash
python scan.py --step 2 --n-samples 12 --dry-run --seed 0
python scan.py --step 2 --n-samples 1000 --seed 1 --wandb-project raytrax-validation
```

| Metric | Threshold | Description |
|--------|-----------|-------------|
| `tau_final_rel_err` | 10 % | (τ_rx − τ_tr) / τ_tr, final optical depth |
| `total_power_rel_err` | 10 % | Relative error of the total absorbed power |
| `depo_rho_diff` | 0.05 | ⟨ρ⟩_rx − ⟨ρ⟩_tr |
| `alpha_rms_pct` | 15 % | RMS of (α_rx − α_tr) / max(α_tr) |
| `ne_at_tr_xyz_mean_err_pct`, `te_at_tr_xyz_mean_err_pct` | 3 % | Profile agreement at TRAVIS positions |
| `rho_at_tr_xyz_rms` | 0.02 | ρ difference at TRAVIS positions |

Absolute values: `tau_final_rx`/`tau_final_tr`, `total_power_rx_mw`/`total_power_tr_mw`, `depo_rho_mean_rx`/`depo_rho_mean_tr` (NaN if nothing is absorbed), `resonance_accessible` (the TRAVIS absorption peak lies at the intended harmonic), `pos_rms_mm`, wall times and `n_steps_rx`.

## Command-line options of `scan.py`

| Flag | Default | Description |
|------|---------|-------------|
| `--step` | 1 | Validation step (1 or 2) |
| `--n-samples` | 10 | Number of random scenarios |
| `--seed` | 0 | RNG seed |
| `--travis-exe` | `TRAVIS_EXE` / `PATH` | Path to `travis-nc` |
| `--travis-input-format` | `legacy` | TRAVIS input layout: `legacy` (13.3.1) or `v13.3.7` |
| `--equilibrium` | — | VMEC file for TRAVIS; the built-in W7-X is exported if omitted |
| `--output-dir` | `results/scan` | Per-run TRAVIS output |
| `--mesh-cache-dir` | — | Cache for the TRAVIS mesh (one directory per B₀) |
| `--dry-run` | off | Print metrics instead of logging to WandB |
| `--wandb-project` | `raytrax-validation` | WandB project |
| `--wandb-group` | `trajectory` / `absorption` | WandB group |
| `--wandb-name` | — | WandB run name |
| `--max-step-size` | 0.05 | Raytrax ODE maximum step [m] |
| `--raytrax-rtol`, `--raytrax-atol` | 1e-4, 1e-6 | Raytrax ODE tolerances |
| `--grid-n-r`, `--grid-n-z`, `--grid-n-phi` | 95, 105, 50 | Raytrax cylindrical grid |
| `--grid-n-rho`, `--grid-n-theta` | 40, 45 | Raytrax VMEC ρ and θ grid |
| `--travis-hgrid`, `--travis-dphi` | 0.022 m, 2° | TRAVIS mesh steps |
| `--travis-rk-accuracy` | 1e-5 | TRAVIS RK accuracy |
| `--travis-max-rk-stepsize` | 10 | TRAVIS maximum RK step [wavelengths] |
| `--travis-resonance-umax`, `--travis-resonance-grid-points` | 7, 700 | TRAVIS resonance-integral velocity limit and grid points |

The settings are stored in the WandB run config.
