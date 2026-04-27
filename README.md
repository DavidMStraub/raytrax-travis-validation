# raytrax-travis-validation

Compares the [Raytrax](https://github.com/proximafusion/raytrax) JAX ray tracer against the TRAVIS Fortran ECRH code on W7-X stellarator scenarios.

## Overview

The validation is organised in two independent steps:

| Step | Script | What is tested |
|------|--------|----------------|
| **1 — Trajectory** | `scan.py --step 1` | Ray path (position, ρ, \|B\|, B̂, nₑ, Tₑ) |
| **2 — Absorption** | `scan.py --step 2` | Absorption coefficient α, optical depth τ, deposited power P_abs, deposition centroid ⟨ρ⟩ |

Both steps use **`scan.py`** for large-scale statistical validation with WandB logging. Single-run alternatives (`validate.py` for step 1, `step2_absorption.py` for step 2) are available for quick testing.

Both steps run on W7-X geometry loaded via the `raytrax` built-in VMEC equilibrium, scaled to a chosen on-axis field B₀.


## Prerequisites

```bash
# Install all dependencies including raytrax
pip install -r requirements.txt

# TRAVIS executable — set one of:
export TRAVIS_EXE=/path/to/travis-nc   # via .env or shell
# or pass --travis-exe on the command line
```

Create a `.env` file in this directory with:
```
TRAVIS_EXE=/path/to/travis-nc
```

For large-scale scans, results are logged to [Weights & Biases (WandB)](https://wandb.ai). Create a free account, then authenticate once:

```bash
wandb login
```

All `scan.py` runs that omit `--dry-run` will log to your WandB workspace under the project specified by `--wandb-project` (default: `raytrax-validation`). Use `--dry-run` to print metrics locally without a WandB account.

---

## Step 1 — Trajectory validation

Both codes trace with the **cold-plasma Hamiltonian**.  B₀ = 3 T is used so the 2nd-harmonic resonance sits at ~168 GHz, far above the 140 GHz beam, keeping absorption negligible and isolating purely geometric errors.

### Single reference scenario

Runs the W7-X standard aiming angles and prints a side-by-side comparison:

```bash
python validate.py
python validate.py --travis-exe /path/to/travis-nc
python validate.py --output-dir results/my_run
```

### Random scan

Samples random antenna positions, directions, frequencies (all well below the fundamental EC resonance so absorption is negligible), and plasma profiles, then logs metrics to WandB or prints them locally:

```bash
# Quick sanity check (no WandB)
python scan.py --n-samples 12 --dry-run --seed 0

# Full scan to WandB
python scan.py --n-samples 500 --wandb-project raytrax-validation --wandb-group trajectory

# Resume with a different seed
python scan.py --n-samples 500 --seed 1 --wandb-project raytrax-validation
```

#### Key options

| Flag | Default | Description |
|------|---------|-------------|
| `--n-samples` | 10 | Number of random scenarios |
| `--seed` | 0 | RNG seed for reproducibility |
| `--output-dir` | `results/scan` | Directory for per-run TRAVIS output |
| `--mesh-cache-dir` | — | Cache dir for the TRAVIS VMEC mesh (strongly recommended — saves ~5 s/run on first build); use e.g. `results/mesh_cache` |
| `--dry-run` | off | Print metrics to stdout instead of logging to WandB |
| `--travis-exe` | — | Path to `travis-nc` (overrides `TRAVIS_EXE`) |
| `--equilibrium` | — | Path to VMEC equilibrium file (uses built-in W7-X if omitted) |

#### Grid resolution

All resolution parameters are accepted by both `scan.py` and `validate.py`:

```bash
python scan.py --n-samples 100 \
  --max-step-size 0.02   \   # Raytrax ODE max step [m]  (default 0.05)
  --grid-n-r      60     \   # cylindrical grid R points (default 45)
  --grid-n-z      70     \   # cylindrical grid Z points (default 55)
  --grid-n-phi    60     \   # cylindrical grid φ points (default 50)
  --grid-n-rho    50     \   # VMEC ρ grid points        (default 40)
  --grid-n-theta  55         # VMEC θ grid points        (default 45)
```

Resolution settings are stored in every WandB run config so scans at different resolutions can be compared directly.

#### Trajectory metrics (logged to WandB)

**Scenario config**: `frequency_ghz`, `mode` (O/X), `ne_central`, `ne_parm`, `te_central`, `te_parm`, `antenna_cyl`, `direction_cart`, `b0_target`, solver grid settings.

**Comparison metrics** (on common arc-length grid):

| Metric | Threshold | Description |
|--------|-----------|-------------|
| `pos_rms_mm` | 5 mm | RMS position distance on common inside-plasma arc |
| `rho_rms` | 0.02 | RMS ρ difference |
| `B_mag_mean_err_pct` | 2 % | Mean \|B\| relative error |
| `B_dir_rms_deg` | 1 ° | RMS angle between B̂ vectors |
| `ne_mean_err_pct` | 3 % | Mean \|nₑ_rx − nₑ_tr\| / max(nₑ_tr) — peak-normalised |
| `te_mean_err_pct` | 3 % | Mean \|Tₑ_rx − Tₑ_tr\| / max(Tₑ_tr) — peak-normalised |

**At TRAVIS xyz** (same spatial points, no interpolation):

| Metric | Threshold | Description |
|--------|-----------|-------------|
| `B_mag_at_tr_xyz_mean_err_pct` | 2 % | Mean \|B\| error at TRAVIS positions |
| `B_dir_at_tr_xyz_rms_deg` | 1 ° | RMS B̂ angle at TRAVIS positions |
| `ne_at_tr_xyz_mean_err_pct` | 3 % | Mean nₑ error at TRAVIS positions |
| `te_at_tr_xyz_mean_err_pct` | 3 % | Mean Tₑ error at TRAVIS positions |
| `rho_at_tr_xyz_rms` | 0.02 | RMS ρ difference at TRAVIS positions |

**Other metrics**:

| Metric | Description |
|--------|-------------|
| `max_ne_tr/rx` | Maximum nₑ [10²⁰ m⁻³] along trajectory — low values indicate ray barely enters plasma |
| `max_te_tr/rx` | Maximum Tₑ [keV] along trajectory — useful to filter edge-grazing cases |
| `deflection_rx/tr/diff_deg` | Entry→exit deflection angle per code and their difference |
| `min_rho_tr/rx` | Minimum ρ reached (core penetration depth) |
| `arc_in_plasma_tr/rx_m` | Arc length inside plasma [m] — large values flag trapped/near-cutoff rays |
| `max_curvature_tr/rx_m_inv` | Peak \|d²r/ds²\| [m⁻¹] — high values flag tightly curved trajectories |
| `Y_param` | f_ce / f — context for interpreting trajectory sensitivity |
| `direction_sensitivity` | Frobenius norm of ∂(final position)/∂(initial direction) [m] — large values indicate numerically sensitive trajectories |
| `travis_wall_time_s`, `raytrax_wall_time_s` | Per-run wall times |
| `n_steps_rx` | Number of ODE steps taken by Raytrax |

---

## Step 2 — Absorption validation

Tests the relativistic absorption calculation by randomly sampling antenna positions, directions, frequencies, and plasma profiles across both polarisations and both the **fundamental** and **2nd-harmonic** EC resonances.

B₀ is scaled to **2.5 T** so the 2nd harmonic falls at 140 GHz (standard W7-X ECRH heating frequency).  Frequency ranges are derived automatically from the actual B-field distribution inside the plasma:

| Harmonic | Typical f range | Why only one harmonic is active |
|----------|-----------------|----------------------------------|
| **X1, O1** (1st) | ~53–82 GHz | Resonant B for 2nd harmonic = f/56 < B_min → outside plasma |
| **X2, O2** (2nd) | ~105–164 GHz | Resonant B for 1st harmonic = f/28 > B_max → outside plasma |

The scan cycles through (X1, O1, X2, O2) to ensure balanced coverage of all mode/harmonic combinations.

### Random scan

```bash
# Quick sanity check (no WandB)
python scan.py --step 2 --n-samples 12 --dry-run --seed 0

# Full scan to WandB
python scan.py --step 2 --n-samples 500 --wandb-project raytrax-validation --wandb-group absorption

# Resume with different seed
python scan.py --step 2 --n-samples 500 --seed 1 --wandb-project raytrax-validation
```

### Single reference run

For quick testing of the four baseline scenarios (X1, O1, X2, O2) with fixed geometry:

```bash
python step2_absorption.py
python step2_absorption.py --seed 7
```

#### Key options

| Flag | Default | Description |
|------|---------|-------------|
| `--step` | 1 | Validation step (1 or 2) |
| `--n-samples` | 10 | Number of random scenarios |
| `--seed` | 0 | RNG seed for reproducibility |
| `--output-dir` | `results/scan` | Directory for per-run TRAVIS output |
| `--mesh-cache-dir` | — | Cache dir for TRAVIS VMEC mesh (recommended) |
| `--dry-run` | off | Print metrics to stdout instead of logging to WandB |
| `--wandb-project` | `raytrax-validation` | WandB project name |
| `--wandb-group` | `trajectory` / `absorption` | WandB group name (defaults by step) |
| `--wandb-name` | — | WandB run display name |
| `--travis-exe` | — | Path to `travis-nc` (overrides `TRAVIS_EXE`) |
| `--equilibrium` | — | Path to VMEC equilibrium file (uses built-in W7-X if omitted) |
| `--max-step-size` | 0.05 | Raytrax ODE max step size [m] |
| `--raytrax-rtol` | 1e-4 | Raytrax ODE relative tolerance |
| `--raytrax-atol` | 1e-6 | Raytrax ODE absolute tolerance |
| `--grid-n-*` | (same as Step 1) | Equilibrium interpolation grid resolution |

#### Absorption metrics (logged to WandB)

**Scenario config**: `harmonic`, `mode`, `frequency_ghz`, `antenna_cyl`, `direction_cart`, `ne_central`, `ne_parm`, `te_central`, `te_parm`, solver grid settings.

**Comparison metrics**:

| Metric | Threshold | Description |
|--------|-----------|-------------|
| `tau_final_rel_err` | 10 % | \|τ_rx − τ_tr\| / τ_tr — final optical depth relative error |
| `total_power_rel_err` | 10 % | \|P_abs_rx − P_abs_tr\| / P_abs_tr — total absorbed power relative error |
| `depo_rho_diff` | 0.05 | ⟨ρ⟩_rx − ⟨ρ⟩_tr — deposition centroid shift |
| `alpha_rms_pct` | 15 % | RMS of (α_rx − α_tr) / max(α_tr) in percent |
| `alpha_from_tr_inputs_rms_pct` | 5 % | Pure formula comparison using all TRAVIS inputs |
| `ne_at_tr_xyz_mean_err_pct` | 3 % | Electron density agreement at TRAVIS positions |
| `te_at_tr_xyz_mean_err_pct` | 3 % | Electron temperature agreement at TRAVIS positions |
| `rho_at_tr_xyz_rms` | 0.02 | ρ difference at TRAVIS positions |
| `alpha_at_tr_xyz_rms_pct` | 15 % | α agreement at TRAVIS positions |

**Absolute values** (also logged):

| Metric | Description |
|--------|-------------|
| `tau_final_rx`, `tau_final_tr` | Final optical depth from Raytrax and TRAVIS |
| `total_power_rx_mw`, `total_power_tr_mw` | Total absorbed power [MW] from Raytrax and TRAVIS |
| `depo_rho_mean_rx`, `depo_rho_mean_tr` | Flux-surface-weighted deposition centroid ⟨ρ⟩ |
| `travis_wall_time_s`, `raytrax_wall_time_s` | Per-run wall times [s] |
| `n_steps_rx` | Number of ODE steps taken by Raytrax |

