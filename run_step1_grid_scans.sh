#!/usr/bin/env bash
# Run three Step 1 (trajectory) scans focused on grid/interpolation fidelity.
# All scans use identical sampling (seed=1, 1000 samples) for direct comparison.
#
# Scans:
#   1. default          — all settings at their defaults
#   2. hifi_rx_interp   — denser raytrax cylindrical interpolation grid only
#   3. hifi_travis_mesh — finer TRAVIS poloidal/toroidal mesh only

set -euo pipefail

COMMON="--step 1 --seed 1 --n-samples 1000 --wandb-project raytrax-validation"

echo "============================================================"
echo " Scan 1/3: default settings"
echo "============================================================"
python scan.py $COMMON \
    --wandb-name "step1_default" \
    --output-dir results/scan_step1_default

echo "============================================================"
echo " Scan 2/3: high-fidelity raytrax interpolation grid"
echo "============================================================"
python scan.py $COMMON \
    --wandb-name "step1_hifi_rx_interp" \
    --output-dir results/scan_step1_hifi_rx_interp \
    --grid-n-r    190 \
    --grid-n-z    210 \
    --grid-n-phi  100 \
    --grid-n-rho   80 \
    --grid-n-theta  90

echo "============================================================"
echo " Scan 3/3: high-fidelity TRAVIS mesh resolution"
echo "============================================================"
python scan.py $COMMON \
    --wandb-name "step1_hifi_travis_mesh" \
    --output-dir results/scan_step1_hifi_travis_mesh \
    --travis-hgrid 0.005 \
    --travis-dphi 0.5

echo "============================================================"
echo " All three grid-fidelity scans complete."
echo "============================================================"
