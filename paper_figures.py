"""Figures and statistics for the paper, computed from logged WandB metrics.

Produces
  fig1_trajectory.pdf  step 1: histograms of position RMS, |B| mean error and
                       n_e mean error at the TRAVIS positions
  fig2_absorption.pdf  step 2: Raytrax vs TRAVIS (a) absorbed power fraction,
                       (b) deposition centroid <rho>, coloured by mode/harmonic
and prints the statistics quoted in the paper.

Step 2 uses the logged tau_final_tr (TRAVIS `tau`), tau_final_rx, resonance_accessible,
depo_rho_mean_rx and depo_rho_diff. Runs are given by WandB run ID or display name; by
default the most recent runs named `step1_default` and `step2_default` are used.

Usage:
    python paper_figures.py
    python paper_figures.py --step1 7lzxvs14 --step2 qy6y9c97 --output-dir figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import wandb

TAU_MIN = 0.05  # minimum TRAVIS optical depth for the deposition-centroid comparison
FULL = 0.95  # absorbed fraction counted as fully absorbed

COMBOS = {
    "X1": dict(color="#0072B2", marker="o"),
    "O1": dict(color="#E69F00", marker="s"),
    "X2": dict(color="#009E73", marker="^"),
    "O2": dict(color="#D55E00", marker="D"),
}

_BLUE = "#0072B2"

# (wandb column, x label, histogram x limit, number of bins)
FIG1_PANELS: list[tuple[str, str, float, int]] = [
    ("pos_rms_mm", "Position RMS [mm]", 20.0, 25),
    ("B_mag_at_tr_xyz_mean_err_pct", r"|$\bf{B}$| mean error [%]", 0.25, 25),
    ("ne_at_tr_xyz_mean_err_pct", r"$n_\mathrm{e}$ mean error [%]", 5.0, 25),
]

STEP1_STATS = [
    ("pos_rms_mm", "position RMS [mm]"),
    ("B_mag_at_tr_xyz_mean_err_pct", "|B| mean error [%]"),
    ("ne_at_tr_xyz_mean_err_pct", "n_e mean error [%]"),
    ("rho_at_tr_xyz_rms", "rho RMS"),
    ("B_dir_at_tr_xyz_rms_deg", "B direction RMS [deg]"),
]


# ── WandB access ──────────────────────────────────────────────────────────────


def get_run(api: wandb.Api, project: str, name_or_id: str):
    """Return the run with this ID, or the most recent run with this display name."""
    path = project if "/" in project else f"{api.default_entity}/{project}"
    try:
        return api.run(f"{path}/{name_or_id}")
    except wandb.errors.CommError:
        pass
    runs = list(api.runs(path, filters={"display_name": name_or_id}, order="-created_at"))
    if not runs:
        raise RuntimeError(f"No WandB run with ID or name '{name_or_id}' in '{path}'.")
    return runs[0]


def load_step1(run) -> tuple[pd.DataFrame, int]:
    cols = ["success"] + [c for c, _ in STEP1_STATS]
    df = run.history(keys=cols, samples=20_000, pandas=True)
    return df[df["success"] == True].copy(), int(run.config["n_samples"])  # noqa: E712


def load_step2(run) -> pd.DataFrame:
    """Successful samples whose TRAVIS resonance is at the intended harmonic."""
    cols = [
        "success",
        "mode",
        "harmonic",
        "resonance_accessible",
        "tau_final_tr",
        "tau_final_rx",
        "depo_rho_mean_rx",
        "depo_rho_diff",
    ]
    df = run.history(keys=cols, samples=20_000, pandas=True)
    df = df[df["success"] == True].copy()  # noqa: E712
    df = df[df["resonance_accessible"].fillna(True) == True].copy()  # noqa: E712
    df["combo"] = df["mode"].astype(str) + df["harmonic"].astype(int).astype(str)
    df["tau_tr"] = df["tau_final_tr"].astype(float)
    df["tau_rx"] = df["tau_final_rx"].astype(float)
    df["rho_rx"] = pd.to_numeric(df["depo_rho_mean_rx"], errors="coerce")
    df["rho_tr"] = df["rho_rx"] - pd.to_numeric(df["depo_rho_diff"], errors="coerce")
    df = df.dropna(subset=["tau_tr", "tau_rx"])
    df["frac_tr"] = 1.0 - np.exp(-df["tau_tr"].clip(lower=0))
    df["frac_rx"] = 1.0 - np.exp(-df["tau_rx"].clip(lower=0))
    return df


# ── Plotting ──────────────────────────────────────────────────────────────────


def _set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "text.usetex": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 150,
        }
    )


def plot_fig1(df: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, len(FIG1_PANELS), figsize=(6.75, 2.2), constrained_layout=True)
    for ax, (col, xlabel, x_clip, n_bins) in zip(axes, FIG1_PANELS):
        vals = df[col].dropna().to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        x_hi = min(float(np.percentile(vals, 99)), x_clip)
        bins = np.linspace(0.0, x_hi, n_bins + 1)
        ax.hist(vals, bins=bins, density=True, histtype="stepfilled", color=_BLUE, alpha=0.20, linewidth=0)
        ax.hist(vals, bins=bins, density=True, histtype="step", color=_BLUE, linewidth=1.5)
        ax.set_xlabel(xlabel)
        ax.set_xlim(0.0, x_hi)
        ax.yaxis.set_major_locator(mticker.MaxNLocator(4, integer=False))
    axes[0].set_ylabel("Probability density")
    for ax, letter in zip(axes, "abc"):
        ax.text(-0.18, 1.02, f"({letter})", transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_fig2(df: pd.DataFrame, out: Path) -> None:
    ab = df["tau_tr"] >= TAU_MIN
    fig, axes = plt.subplots(1, 2, figsize=(494.4 / 72, 3.0), constrained_layout=True)
    panels = [
        (axes[0], "frac", "Absorbed power fraction", df),
        (axes[1], "rho", r"Deposition centroid $\langle\rho\rangle$", df[ab]),
    ]
    for ax, key, label, data in panels:
        ax.plot([0, 1], [0, 1], color="0.6", lw=0.8, zorder=0)
        for combo, style in COMBOS.items():
            sub = data[data["combo"] == combo].dropna(subset=[f"{key}_tr", f"{key}_rx"])
            ax.scatter(
                sub[f"{key}_tr"],
                sub[f"{key}_rx"],
                s=7,
                alpha=0.6,
                linewidths=0,
                label=f"{combo} (n={len(sub)})" if key == "frac" else None,
                **style,
            )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_xlabel(f"{label}, Travis")
        ax.set_ylabel(f"{label}, Raytrax")
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([0, 0.5, 1])
    fig.legend(
        *axes[0].get_legend_handles_labels(),
        loc="outside lower center",
        ncol=4,
        frameon=False,
        markerscale=1.8,
    )
    for ax, letter in zip(axes, "ab"):
        ax.set_title(f"({letter})", loc="left", fontsize=9, fontweight="bold")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")
    print(f"  panel (a): n={len(df)}, panel (b): n={int(ab.sum())}")


# ── Statistics ────────────────────────────────────────────────────────────────


def print_step1_stats(df: pd.DataFrame, n_total: int) -> None:
    print("\n=== Step 1 ===")
    print(f"successful samples: {len(df)} / {n_total}  (success rate {len(df) / n_total:.1%})")
    for col, label in STEP1_STATS:
        v = df[col].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
        print(f"  {label:26s} median {v.median():9.4f}   95th percentile {v.quantile(0.95):9.4f}")


def print_step2_stats(df: pd.DataFrame) -> None:
    print("\n=== Step 2 ===")
    absorbing = df[df["tau_tr"] >= TAU_MIN]
    print(f"rays reaching the resonance: {len(df)}")
    print(f"retained (tau_TRAVIS >= {TAU_MIN}): {len(absorbing)}")
    header = (
        f"{'group':6s} {'n_reach':>7s} {'n_ret':>6s} {'tau rel med':>12s} {'P rel med':>10s} "
        f"{'drho med':>9s} {'drho p90':>9s} {'full both':>10s} {'excl rx<10%':>12s}"
    )
    print(header)
    for group, d in [("all", df)] + [(c, df[df["combo"] == c]) for c in COMBOS]:
        a = d[d["tau_tr"] >= TAU_MIN]
        tau_rel = ((a["tau_rx"] - a["tau_tr"]).abs() / a["tau_tr"]).median()
        p_rel = ((a["frac_rx"] - a["frac_tr"]).abs() / a["frac_tr"]).median()
        drho = (a["rho_rx"] - a["rho_tr"]).abs().dropna()
        full_both = a[(a["frac_rx"] >= FULL) & (a["frac_tr"] >= FULL)]
        excluded = d[d["tau_tr"] < TAU_MIN]
        low = (excluded["frac_rx"] < 0.1).mean() if len(excluded) else np.nan
        print(
            f"{group:6s} {len(d):7d} {len(a):6d} {tau_rel:12.4f} {p_rel:10.4f} "
            f"{drho.median():9.4f} {drho.quantile(0.9):9.4f} "
            f"{len(full_both) / max(len(a), 1):10.3f} {low:12.3f}"
        )
    print(f"(full both: fraction of retained rays with absorbed fraction >= {FULL} in both codes;")
    print(" excl rx<10%: fraction of excluded rays where Raytrax absorbs < 10 %)")


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--wandb-project", default="raytrax-validation")
    parser.add_argument("--step1", default="step1_default", help="Step 1 WandB run ID or display name.")
    parser.add_argument("--step2", default="step2_default", help="Step 2 WandB run ID or display name.")
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _set_style()
    api = wandb.Api()

    run1 = get_run(api, args.wandb_project, args.step1)
    print(f"Step 1 run: {run1.name} ({run1.id})")
    df1, n_total = load_step1(run1)
    plot_fig1(df1, args.output_dir / "fig1_trajectory.pdf")
    print_step1_stats(df1, n_total)

    run2 = get_run(api, args.wandb_project, args.step2)
    print(f"\nStep 2 run: {run2.name} ({run2.id})")
    df2 = load_step2(run2)
    plot_fig2(df2, args.output_dir / "fig2_absorption.pdf")
    print_step2_stats(df2)


if __name__ == "__main__":
    main()
