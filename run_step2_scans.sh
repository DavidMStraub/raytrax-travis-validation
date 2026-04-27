#!/usr/bin/env bash
# Run Step 2 (absorption) scans with identical sampling (seed=1, 5000 samples),
# cycling equally through X1/O1/X2/O2 (1250 each).
#
# Scans:
#   1. default   — raytrax defaults (rtol=1e-4, atol=1e-6, max_step=0.05 m)
#   2. hifi_ode  — tighter ODE tolerances + smaller max step (rtol=1e-6, atol=1e-8, max_step=0.01 m)
#                  Tests whether N_∥ integrator error is the scatter bottleneck.

set -euo pipefail

COMMON="--step 2 --seed 1 --n-samples 5000 --wandb-project raytrax-validation"

echo "============================================================"
echo " Scan 1/2: default settings"
echo "============================================================"
python scan.py $COMMON \
    --wandb-name "step2_default" \
    --output-dir results/scan_step2_default

echo "============================================================"
echo " Scan 2/2: high-fidelity ODE (rtol=1e-6, atol=1e-8, max_step=0.01)"
echo "============================================================"
python scan.py $COMMON \
    --wandb-name "step2_hifi_ode" \
    --output-dir results/scan_step2_hifi_ode \
    --raytrax-rtol 1e-6 \
    --raytrax-atol 1e-8 \
    --max-step-size 0.01

echo "============================================================"
echo " Scans complete."
echo "============================================================"
