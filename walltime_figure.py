"""Scatter plot of Raytrax vs TRAVIS wall time per sample, from a step 2 WandB run.

Produces
  fig3_walltime.pdf  wall time of Raytrax vs TRAVIS for each successful sample,
                     coloured by mode/harmonic, on log axes
and prints the median and percentiles of both wall times and of their ratio.

The Raytrax wall time is one warm trace inside the running Python process. The TRAVIS
wall time covers the whole travis-nc subprocess, including start-up and file I/O. The
first sample of each mode is dropped because its Raytrax time includes JIT compilation.

Usage:
    python walltime_figure.py --step2 1h14qs3y --output-dir figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import wandb
from paper_figures import COMBOS, _set_style, get_run


def load_walltimes(run) -> pd.DataFrame:
    """Successful samples with both wall times, excluding the compiling first sample per mode."""
    cols = ["success", "mode", "harmonic", "raytrax_wall_time_s", "travis_wall_time_s"]
    df = run.history(keys=cols, samples=20_000, pandas=True).sort_values("_step")
    first_per_mode = df.groupby("mode")["_step"].transform("min") == df["_step"]
    print(
        f"dropped first sample per mode (JIT compilation): steps {df.loc[first_per_mode, '_step'].tolist()}"
    )
    df = df[(df["success"] == True) & ~first_per_mode].copy()
    df["combo"] = df["mode"].astype(str) + df["harmonic"].astype(int).astype(str)
    df["t_rx"] = df["raytrax_wall_time_s"].astype(float)
    df["t_tr"] = df["travis_wall_time_s"].astype(float)
    return df.dropna(subset=["t_rx", "t_tr"])


def plot_fig3(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(3.4, 3.0), constrained_layout=True)
    lo = 0.5 * min(df["t_rx"].min(), df["t_tr"].min())
    hi = 2.0 * max(df["t_rx"].max(), df["t_tr"].max())
    for factor, ls in [(1, "-"), (0.1, "--")]:
        ax.plot(
            [lo, hi], [factor * lo, factor * hi], color="0.6", lw=0.8, ls=ls, zorder=0
        )
    ax.text(
        hi / 1.5,
        0.1 * hi / 1.5,
        r"$\times$0.1",
        color="0.5",
        fontsize=7,
        ha="right",
        va="bottom",
    )
    for combo, style in COMBOS.items():
        sub = df[df["combo"] == combo]
        ax.scatter(
            sub["t_tr"], sub["t_rx"], s=7, alpha=0.6, linewidths=0, label=combo, **style
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel("Wall time, Travis [s]")
    ax.set_ylabel("Wall time, Raytrax [s]")
    ax.legend(frameon=False, markerscale=1.8, loc="upper left")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def print_walltime_stats(df: pd.DataFrame) -> None:
    print(f"\n=== Wall time ({len(df)} samples) ===")
    ratio = df["t_rx"] / df["t_tr"]
    for label, v in [
        ("Raytrax [s]", df["t_rx"]),
        ("TRAVIS [s]", df["t_tr"]),
        ("Raytrax / TRAVIS", ratio),
    ]:
        print(
            f"  {label:18s} median {v.median():8.4f}   5th percentile {v.quantile(0.05):8.4f}"
            f"   95th percentile {v.quantile(0.95):8.4f}   max {v.max():8.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--wandb-project", default="raytrax-validation")
    parser.add_argument(
        "--step2", default="step2_default", help="Step 2 WandB run ID or display name."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _set_style()
    run = get_run(wandb.Api(), args.wandb_project, args.step2)
    print(f"Step 2 run: {run.name} ({run.id})")
    df = load_walltimes(run)
    plot_fig3(df, args.output_dir / "fig3_walltime.pdf")
    print_walltime_stats(df)


if __name__ == "__main__":
    main()
