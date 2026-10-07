#!/usr/bin/env bash
# Run two Step 1 (trajectory) scans probing solver fidelity only.
# Run run_step1_grid_scans.sh first to obtain the grid-fidelity baselines.
#
# Scans:
#   1. hifi_rx_solver     — finer raytrax ODE step size and tighter tolerances only
#   2. hifi_travis_solver — tighter TRAVIS RK accuracy and step size only

set -euo pipefail

COMMON="--step 1 --seed 1 --n-samples 1000 --wandb-project raytrax-validation"

echo "============================================================"
echo " Scan 1/2: high-fidelity raytrax ODE solver"
echo "============================================================"
python scan.py $COMMON \
    --wandb-name "step1_hifi_rx_solver" \
    --output-dir results/scan_step1_hifi_rx_solver \
    --max-step-size 0.01 \
    --raytrax-rtol 1e-6 \
    --raytrax-atol 1e-8

echo "============================================================"
echo " Scan 2/2: high-fidelity TRAVIS RK solver"
echo "============================================================"
python scan.py $COMMON \
    --wandb-name "step1_hifi_travis_solver" \
    --output-dir results/scan_step1_hifi_travis_solver \
    --travis-rk-accuracy 1e-7 \
    --travis-max-rk-stepsize 2.0

echo "============================================================"
echo " Both solver-fidelity scans complete."
echo "============================================================"
