"""Step-2 scan histogram comparison (split by mode/harmonic and run).

Creates overlaid histograms per combo (O1, O2, X2, X1) for both:
  1) final optical depth relative error
  2) absorption-fraction relative error

For each metric, two figures are generated:
  • signed error histogram
  • absolute error histogram

Usage:
    python hist_step2_runs.py
    python hist_step2_runs.py --wandb-project raytrax-validation --output-dir results/step2_comparison
    python hist_step2_runs.py --latest-only        # only the most recent step2_default run
    python hist_step2_runs.py --compare-default    # latest vs. previous step2_default run
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb

# Each entry: (label, wandb_display_name, style_kwargs, run_index)
# run_index=0 → most recent, 1 → second most recent, etc.
RUNS = [
    ("default",  "step2_default",  dict(color="#0072B2", lw=1.8, ls="-"),  0),
    ("hifi ODE", "step2_hifi_ode", dict(color="#D55E00", lw=1.8, ls="--"), 0),
]

COMBO_ORDER = ["O1", "O2", "X2", "X1"]

# (column key, label, x-axis half-range clip, guard_col, guard_min)
# guard_col/guard_min: skip rows where guard_col < guard_min (None = no filter).
METRICS: list[tuple[str, str, float, str | None, float | None]] = [
    ("tau_final_rel_err",      "final optical depth rel. error",      0.50, "tau_final_tr", 0.1),
    ("abs_fraction_diff_ppt",  "absorption fraction abs. error [pp]", 20.0, None,           None),
]

# Exclude samples where the two ray trajectories deviate by more than this.
FILTER_POS_RMS_MM_MAX: float = 20.0

_COLS = [
    "success",
    "mode",
    "harmonic",
    "power_mw",
    "tau_final_rel_err",
    "total_power_rx_mw",
    "total_power_tr_mw",
    "pos_rms_mm",
]


def fetch_run_history(project: str, run_name: str, index: int = 0) -> tuple[pd.DataFrame, str]:
    """Return (history_df, run_id) for the `index`-th most recent run with the given name."""
    api = wandb.Api()
    runs = api.runs(project, filters={"display_name": run_name}, order="-created_at")
    matched = [r for r in runs]
    if not matched:
        raise RuntimeError(f"No WandB run named '{run_name}' in project '{project}'.")
    if index >= len(matched):
        raise RuntimeError(
            f"Requested run index {index} for '{run_name}', but only {len(matched)} exist."
        )
    run = matched[index]
    return run.history(keys=_COLS, samples=20_000, pandas=True), run.id


def load_all(project: str, runs: list | None = None) -> dict[str, pd.DataFrame]:
    data: dict[str, pd.DataFrame] = {}
    for entry in (runs or RUNS):
        label, run_name, _style, run_index = entry
        print(f"  Fetching '{run_name}' [index={run_index}] …", end=" ", flush=True)
        try:
            df, run_id = fetch_run_history(project, run_name, index=run_index)
            df = df[df["success"] == True].copy()  # noqa: E712
            n_before = len(df)
            if "pos_rms_mm" in df.columns:
                df = df[df["pos_rms_mm"].fillna(0.0) <= FILTER_POS_RMS_MM_MAX]
            n_after = len(df)
            n_filtered = n_before - n_after
            denom = np.maximum(df["power_mw"].astype(float).to_numpy(), 1e-9)
            frac_rx = df["total_power_rx_mw"].astype(float).to_numpy() / denom
            frac_tr = df["total_power_tr_mw"].astype(float).to_numpy() / denom
            df["abs_fraction_diff_ppt"] = (frac_rx - frac_tr) * 100.0
            df["combo"] = df["mode"].astype(str) + df["harmonic"].astype(int).astype(str)
            data[label] = df
            print(f"{n_after} samples ({n_filtered} filtered)  run_id={run_id}")
        except RuntimeError as e:
            print(f"SKIPPED ({e})")
    return data


def _panel_xlim(vals: np.ndarray, absolute: bool, clip: float) -> tuple[float, float]:
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return (0.0, clip) if absolute else (-clip, clip)

    if absolute:
        hi = min(float(np.percentile(vals, 99)), clip)
        return (0.0, max(hi, 1e-3))

    lo, hi = np.percentile(vals, [1, 99])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        m = float(np.max(np.abs(vals))) if vals.size else clip
        return (-min(m, clip), min(m, clip))
    m = min(max(abs(float(lo)), abs(float(hi))), clip)
    return (-m, m)


def print_summary(data: dict[str, pd.DataFrame], runs: list) -> None:
    """Print median ± IQR/2 per run × combo × metric."""
    print("\n── Summary statistics (median  ±  IQR/2) ──")
    for key, mlabel, _clip, guard_col, guard_min in METRICS:
        print(f"\n  {mlabel}")
        header = f"  {'combo':>4}  " + "  ".join(f"{e[0]:>14}" for e in runs)
        print(header)
        for combo in COMBO_ORDER:
            row = f"  {combo:>4}  "
            for entry in runs:
                label = entry[0]
                if label not in data:
                    row += f"{'N/A':>14}  "
                    continue
                sub = data[label][data[label]["combo"] == combo]
                if guard_col is not None and guard_col in sub.columns:
                    sub = sub[sub[guard_col].fillna(0.0) >= guard_min]
                vals = sub[key].dropna().to_numpy(dtype=float)
                vals = vals[np.isfinite(vals)]
                if vals.size == 0:
                    row += f"{'no data':>14}  "
                else:
                    med = float(np.median(vals))
                    q25, q75 = np.percentile(vals, [25, 75])
                    half_iqr = (q75 - q25) / 2.0
                    row += f"  {med:+.4f}±{half_iqr:.4f}"
            print(row)
    print()


def plot_histograms(data: dict[str, pd.DataFrame], metric_key: str, metric_label: str,
                    clip: float, guard_col: str | None, guard_min: float | None,
                    absolute: bool, output_path: Path, runs: list | None = None) -> None:
    active_runs = runs or RUNS
    fig, axes = plt.subplots(1, len(COMBO_ORDER), figsize=(17, 4), constrained_layout=True)
    if len(COMBO_ORDER) == 1:
        axes = [axes]

    for i, combo in enumerate(COMBO_ORDER):
        ax = axes[i]

        all_vals = []
        series_by_run: list[tuple[str, np.ndarray, dict]] = []
        for entry in active_runs:
            label, _run_name, style, _idx = entry
            if label not in data:
                continue
            sub = data[label][data[label]["combo"] == combo]
            if guard_col is not None and guard_col in sub.columns:
                sub = sub[sub[guard_col].fillna(0.0) >= guard_min]
            vals = sub[metric_key].dropna().to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if absolute:
                vals = np.abs(vals)
            if vals.size > 0:
                all_vals.append(vals)
            series_by_run.append((label, vals, style))

        if not all_vals:
            ax.set_title(f"{combo} (no data)")
            ax.grid(alpha=0.25)
            continue

        merged = np.concatenate(all_vals)
        xlo, xhi = _panel_xlim(merged, absolute=absolute, clip=clip)
        bins = np.linspace(xlo, xhi, 40)

        show_legend = sum(1 for _, v, _ in series_by_run if v.size > 0) > 1
        for label, vals, style in series_by_run:
            if vals.size == 0:
                continue
            ax.hist(vals, bins=bins, density=True, histtype="step",
                    label=label if show_legend else None, **style)

        if not absolute:
            ax.axvline(0.0, color="k", lw=1.0, alpha=0.6)

        ax.set_xlim(xlo, xhi)
        ax.set_title(combo)
        ax.grid(alpha=0.25)
        if i == 0:
            ax.set_ylabel("density")
            if show_legend:
                ax.legend(frameon=False, fontsize=9)

    suffix = "|err|" if absolute else "signed err"
    fig.suptitle(f"Step-2: {metric_label} ({suffix}), split by mode/harmonic", fontsize=12)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    print(f"  Saved {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wandb-project", type=str, default="raytrax-validation")
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    parser.add_argument("--latest-only", action="store_true",
                        help="Plot only the most recent step2_default run (no comparison).")
    parser.add_argument("--compare-default", action="store_true",
                        help="Compare latest vs. previous step2_default run.")
    parser.add_argument("--prev-index", type=int, default=1,
                        help="Which previous step2_default run to use with --compare-default (default=1).")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.compare_default:
        runs = [
            ("latest",   "step2_default", dict(color="#0072B2", lw=1.8, ls="-"),  0),
            ("previous", "step2_default", dict(color="#D55E00", lw=1.8, ls="--"), args.prev_index),
        ]
    elif args.latest_only:
        runs = [RUNS[0]]
    else:
        runs = RUNS

    print("Loading Step-2 run histories …")
    data = load_all(args.wandb_project, runs)

    print_summary(data, runs)

    for key, label, clip, guard_col, guard_min in METRICS:
        out_signed = args.output_dir / f"step2_{key}_hist_signed.png"
        out_abs = args.output_dir / f"step2_{key}_hist_abs.png"
        plot_histograms(data, key, label, clip, guard_col, guard_min, absolute=False,
                        output_path=out_signed, runs=runs)
        plot_histograms(data, key, label, clip, guard_col, guard_min, absolute=True,
                        output_path=out_abs, runs=runs)


if __name__ == "__main__":
    main()
