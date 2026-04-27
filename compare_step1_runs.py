"""Publication-ready comparison of the five Step-1 fidelity scans.

Fetches per-sample scalar metrics from WandB for the five named runs and
produces:
  • A summary table (median ± IQR / 95th-pct) printed to stdout and saved
    as a LaTeX table.
  • A figure with one histogram panel per metric, all five runs overlaid.

Metrics (all evaluated at TRAVIS's Cartesian positions, so arc-length
interpolation error is not a factor):
  • ρ RMS               rho_at_tr_xyz_rms
  • nₑ mean err [%]     ne_at_tr_xyz_mean_err_pct
  • Tₑ mean err [%]     te_at_tr_xyz_mean_err_pct
  • |B| mean err [%]    B_mag_at_tr_xyz_mean_err_pct
  • B̂ RMS angle [°]     B_dir_at_tr_xyz_rms_deg

Usage:
    python compare_step1_runs.py
    python compare_step1_runs.py --wandb-project raytrax-validation --output-dir results/comparison
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import wandb

# ── Run definitions ───────────────────────────────────────────────────────────

RUNS = [
    ("default",          "step1_default"),
    # ("hifi rx solver",   "step1_hifi_rx_solver"),   # outdated scan
    ("hifi rx interp",   "step1_hifi_rx_interp"),
    ("hifi travis mesh", "step1_hifi_travis_mesh"),
    # ("hifi travis RK",   "step1_hifi_travis_solver"),  # outdated scan
]

# (column key, display label, unit, pass/fail threshold, x_max clip, cell fmt)
METRICS: list[tuple[str, str, str, float | None, float | None, str]] = [
    ("rho_at_tr_xyz_rms",           r"$\rho$ RMS",                       "",           0.02, None, ".4f"),
    ("ne_at_tr_xyz_mean_err_pct",   r"$n_{\mathrm{e}}$ mean err",        r"\%",        3.0,  5.0,  ".2f"),
    ("te_at_tr_xyz_mean_err_pct",   r"$T_{\mathrm{e}}$ mean err",        r"\%",        3.0,  5.0,  ".2f"),
    ("B_mag_at_tr_xyz_mean_err_pct",r"$|\bm{B}|$ mean err",              r"\%",        2.0,  None, ".2f"),
    ("B_dir_at_tr_xyz_rms_deg",     r"$\hat{\bm{B}}$ RMS angle",         r"$^\circ$",  1.0,  None, ".2f"),
]

# Colours and line styles for the five runs (colour-blind friendly)
STYLES = [
    dict(color="#0072B2", lw=1.8, ls="-"),        # default           — blue solid
    dict(color="#E69F00", lw=1.8, ls="--"),        # hifi rx solver    — amber dashed
    dict(color="#009E73", lw=1.8, ls="--"),        # hifi rx interp    — teal dashed
    dict(color="#D55E00", lw=1.8, ls="-."),        # hifi travis mesh  — vermillion dash-dot
    dict(color="#CC79A7", lw=1.8, ls=(0,(5,1))),   # hifi travis RK    — mauve dense-dash
]

# Subset used in the at-xyz plots: solver settings can't affect interpolation
# quality at fixed spatial points, so only grid runs are shown.
_PLOT_LABELS = {"default", "hifi rx interp", "hifi travis mesh"}
PLOT_RUNS   = [(lbl, name) for lbl, name in RUNS   if lbl in _PLOT_LABELS]
PLOT_STYLES = [style       for (lbl, _), style in zip(RUNS, STYLES) if lbl in _PLOT_LABELS]

# Trajectory-accuracy metrics: position and deflection angle.
# Solver settings DO affect these, so all five runs are shown.
# deflection_diff_deg is signed (rx − tr): non-zero median reveals systematic bias.
# The 7th element marks signed metrics; their p95 is computed on |val| (accuracy bound).
TRAJ_METRICS: list[tuple[str, str, str, float | None, float | None, str, bool]] = [
    ("pos_rms_mm",          r"pos.\ RMS",                            "mm",         5.0,  None, ".2f", False),
    ("rho_rms",             r"$\rho$ RMS (arc)",                     "",           0.02, 0.05, ".4f", False),
    ("deflection_diff_deg", r"$\Delta\theta_{\mathrm{defl}}$",      r"$^\circ$",  None, None, ".3f", True),
]
TRAJ_PLOT_RUNS   = RUNS    # all five — solver matters for trajectory accuracy
TRAJ_PLOT_STYLES = STYLES


# ── Data loading ──────────────────────────────────────────────────────────────

def fetch_run_history(project: str, run_name: str) -> pd.DataFrame:
    """Return the full step history for a WandB run identified by display name."""
    api = wandb.Api()
    runs = api.runs(project, filters={"display_name": run_name}, order="-created_at")
    matched = [r for r in runs]
    if not matched:
        raise RuntimeError(f"No WandB run named '{run_name}' in project '{project}'.")
    if len(matched) > 1:
        print(f"  Warning: {len(matched)} runs named '{run_name}'; using the most recent.")
    run = matched[0]
    _scatter_cols = ["deflection_rx_deg", "deflection_tr_deg", "min_rho_tr", "Y_param"]
    cols = (["success", "raytrax_wall_time_s"]
            + [m[0] for m in METRICS]
            + [m[0] for m in TRAJ_METRICS]
            + _scatter_cols)
    df = run.history(keys=list(dict.fromkeys(cols)), samples=10_000, pandas=True)
    return df


def load_all(project: str) -> dict[str, pd.DataFrame]:
    """Load and filter histories for all five runs."""
    data: dict[str, pd.DataFrame] = {}
    for label, name in RUNS:
        print(f"  Fetching '{name}' …", end=" ", flush=True)
        try:
            df = fetch_run_history(project, name)
            df = df[df["success"] == True].copy()  # noqa: E712
            data[label] = df
            print(f"{len(df)} successful samples")
        except RuntimeError as e:
            print(f"SKIPPED ({e})")
    return data


# ── Summary statistics ────────────────────────────────────────────────────────

def summary_stats(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Compute median, IQR, and 95th-percentile for every metric and run.

    For signed metrics (7th tuple element True), p95 is computed on |val| so it
    represents the magnitude of the worst-case error rather than the one-sided tail.
    """
    rows = []
    for label, _ in RUNS:
        if label not in data:
            continue
        df = data[label]
        row: dict = {"run": label}
        all_metrics = [(key, *rest) for key, *rest in METRICS + [(k, *r) for k, *r in TRAJ_METRICS]]
        for entry in METRICS:
            key = entry[0]
            signed = False
            if key not in df:
                row[f"{key}_med"] = np.nan
                row[f"{key}_iqr"] = np.nan
                row[f"{key}_p95"] = np.nan
                continue
            vals = df[key].dropna().values
            row[f"{key}_med"] = np.median(vals)
            row[f"{key}_iqr"] = float(np.percentile(vals, 75) - np.percentile(vals, 25))
            row[f"{key}_p95"] = float(np.percentile(vals, 95))
        for entry in TRAJ_METRICS:
            key, _, _, _, _, _, signed = entry
            if key not in df:
                row[f"{key}_med"] = np.nan
                row[f"{key}_iqr"] = np.nan
                row[f"{key}_p95"] = np.nan
                continue
            vals = df[key].dropna().values
            row[f"{key}_med"] = np.median(vals)
            row[f"{key}_iqr"] = float(np.percentile(vals, 75) - np.percentile(vals, 25))
            # For signed metrics use p95 of |val|: captures both tails symmetrically.
            row[f"{key}_p95"] = float(np.percentile(np.abs(vals), 95))
        rows.append(row)
    return pd.DataFrame(rows).set_index("run")


# ── Console table ─────────────────────────────────────────────────────────────

def print_table(stats: pd.DataFrame) -> None:
    col_w = 22

    def _header_row(metric_list):
        parts = [f"{'Run':<20}"]
        for entry in metric_list:
            lbl, unit = entry[1], entry[2]
            unit_str = f" [{unit}]" if unit else ""
            parts.append(f"{lbl + unit_str:>{col_w}}")
        return "  ".join(parts)

    def _sub_row(metric_list):
        parts = [f"{'':20}"]
        for entry in metric_list:
            signed = entry[6] if len(entry) > 6 else False
            lbl = "med / p95|\u00b7|" if signed else "med / p95"
            parts.append(f"{lbl:>{col_w}}")
        return "  ".join(parts)

    def _data_rows(metric_list):
        for run, _ in RUNS:
            if run not in stats.index:
                continue
            row_parts = [f"{run:<20}"]
            for entry in metric_list:
                key, fmt = entry[0], entry[5]
                med = stats.loc[run, f"{key}_med"]
                p95 = stats.loc[run, f"{key}_p95"]
                cell = f"{med:{fmt}} (p95:{p95:{fmt}})"
                row_parts.append(f"{cell:>{col_w}}")
            print("  ".join(row_parts))

    sep = "-" * (20 + (col_w + 2) * max(len(METRICS), len(TRAJ_METRICS)))

    print("\n" + _header_row(METRICS))
    print("  " + _sub_row(METRICS))
    print("  " + sep)
    _data_rows(METRICS)

    print("\n  Trajectory accuracy:")
    print("  " + _header_row(TRAJ_METRICS))
    print("  " + _sub_row(TRAJ_METRICS))
    print("  " + sep)
    _data_rows(TRAJ_METRICS)
    print()


# ── LaTeX table ───────────────────────────────────────────────────────────────

def write_latex_table(stats: pd.DataFrame, path: Path) -> None:
    n_cols = max(len(METRICS), len(TRAJ_METRICS))
    col_spec = "l" + "r" * n_cols
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\caption{Step-1 trajectory fidelity comparison.}",
        r"\label{tab:step1_fidelity}",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
    ]

    def _header_cells(metric_list):
        return ["Run"] + [
            rf"{entry[1]}" + (rf" [{entry[2]}]" if entry[2] else "")
            for entry in metric_list
        ]

    def _sub_cells(metric_list):
        out = [""]
        for entry in metric_list:
            signed = entry[6] if len(entry) > 6 else False
            label = r"\multicolumn{1}{c}{med / p95$|\cdot|$}" if signed else r"\multicolumn{1}{c}{med / p95}"
            out.append(label)
        return out

    def _data_rows(metric_list):
        out = []
        for run, _ in RUNS:
            if run not in stats.index:
                continue
            row_latex = [run.replace("_", r"\_")]
            for entry in metric_list:
                key, fmt = entry[0], entry[5]
                med = stats.loc[run, f"{key}_med"]
                p95 = stats.loc[run, f"{key}_p95"]
                row_latex.append(rf"{med:{fmt}} / {p95:{fmt}}")
            out.append(" & ".join(row_latex) + r" \\")
        return out

    # ── Interpolation quality section ────────────────────────────────────────
    lines.append(r"\multicolumn{" + str(1 + n_cols) + r"}{l}{\textit{Interpolation quality at TRAVIS positions}} \\")
    lines.append(r"\midrule")
    lines.append(" & ".join(_header_cells(METRICS)) + r" \\")
    lines.append(" & ".join(_sub_cells(METRICS)) + r" \\")
    lines.append(r"\midrule")
    lines.extend(_data_rows(METRICS))

    # ── Trajectory accuracy section ──────────────────────────────────────────
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{" + str(1 + n_cols) + r"}{l}{\textit{Trajectory accuracy}} \\")
    lines.append(r"\midrule")
    lines.append(" & ".join(_header_cells(TRAJ_METRICS)) + r" \\")
    lines.append(" & ".join(_sub_cells(TRAJ_METRICS)) + r" \\")
    lines.append(r"\midrule")
    lines.extend(_data_rows(TRAJ_METRICS))

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    path.write_text("\n".join(lines) + "\n")
    print(f"  LaTeX table → {path}")


def write_markdown_table(stats: pd.DataFrame, path: Path) -> None:
    def _md_label(lbl: str, unit: str) -> str:
        text = lbl + (f" [{unit}]" if unit else "")
        return text.replace("|", r"\|")  # escape pipes so they don't break the table

    def _md_section(title: str, metric_list) -> list[str]:
        header = ["Run"] + [_md_label(lbl, unit) for _, lbl, unit, *_ in metric_list]
        sep    = ["-" * max(len(h), 14) for h in header]
        sub_cells = [""]
        for entry in metric_list:
            signed = entry[6] if len(entry) > 6 else False
            sub_cells.append("med / p95|·|" if signed else "med / p95")
        out = [
            f"### {title}",
            "| " + " | ".join(header) + " |",
            "| " + " | ".join(sep)   + " |",
            "| " + " | ".join(sub_cells) + " |",
        ]
        for run, _ in RUNS:
            if run not in stats.index:
                continue
            cells = [run]
            for entry in metric_list:
                key, fmt = entry[0], entry[5]
                med = stats.loc[run, f"{key}_med"]
                p95 = stats.loc[run, f"{key}_p95"]
                cells.append(f"{med:{fmt}} / {p95:{fmt}}")
            out.append("| " + " | ".join(cells) + " |")
        return out

    lines = (_md_section("Interpolation quality at TRAVIS positions", METRICS)
             + [""]
             + _md_section("Trajectory accuracy", TRAJ_METRICS))
    path.write_text("\n".join(lines) + "\n")
    print(f"  Markdown table → {path}")


# ── Histograms ────────────────────────────────────────────────────────────────

def plot_histograms(data: dict[str, pd.DataFrame], output_path: Path) -> None:
    plt.rcParams.update({
        "font.family":       "serif",
        "font.size":         9,
        "axes.titlesize":    9,
        "axes.labelsize":    9,
        "xtick.labelsize":   8,
        "ytick.labelsize":   8,
        "legend.fontsize":   8,
        "figure.dpi":        150,
        "text.usetex":       True,
        "text.latex.preamble": r"\usepackage{bm}",
    })

    n_metrics = len(METRICS)
    fig, axes = plt.subplots(1, n_metrics, figsize=(3.5 * n_metrics, 3.2))
    if n_metrics == 1:
        axes = [axes]

    for ax, (key, label_str, unit, threshold, x_max, _fmt) in zip(axes, METRICS):
        present = [(label, style) for (label, _), style in zip(PLOT_RUNS, PLOT_STYLES) if label in data]
        all_vals = np.concatenate([
            data[label][key].dropna().values
            for label, _ in present
            if key in data[label]
        ])
        lo = max(np.percentile(all_vals, 0.5), 0.0)
        hi = np.percentile(all_vals, 99.5)
        if x_max is not None:
            hi = min(hi, x_max)
        bins = np.linspace(lo, hi, 50)

        for (label, style) in present:
            df = data[label]
            if key not in df:
                continue
            vals = df[key].dropna().values
            counts, edges = np.histogram(vals, bins=bins)
            # Normalise to probability density
            widths = np.diff(edges)
            density = counts / (counts.sum() * widths)
            centers = 0.5 * (edges[:-1] + edges[1:])
            ax.step(centers, density, where="mid", label=label, **style)

        unit_str = f" [{unit}]" if unit else ""
        ax.set_xlabel(label_str + unit_str)
        ax.set_ylabel("Probability density")
        ax.set_xlim(lo, hi)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
        ax.spines[["top", "right"]].set_visible(False)

    # Single shared legend below the figure
    handles, labels = axes[0].get_legend_handles_labels()
    # Collect all labels from all axes (threshold line may appear multiple times)
    all_handles, all_labels = [], []
    seen = set()
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in seen:
                all_handles.append(h)
                all_labels.append(l)
                seen.add(l)

    fig.legend(
        all_handles, all_labels,
        loc="lower center", ncol=len(present) + 1,
        bbox_to_anchor=(0.5, -0.08),
        frameon=False,
    )
    fig.suptitle("Step-1 fidelity: interpolation metrics at TRAVIS positions", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    print(f"  Histogram figure → {output_path}")
    plt.close(fig)


def plot_trajectory_histograms(data: dict[str, pd.DataFrame], output_path: Path) -> None:
    """Histogram panels for trajectory-accuracy metrics (all five runs)."""
    n_metrics = len(TRAJ_METRICS)
    fig, axes = plt.subplots(1, n_metrics, figsize=(3.5 * n_metrics, 3.2))
    if n_metrics == 1:
        axes = [axes]

    for ax, entry in zip(axes, TRAJ_METRICS):
        key, label_str, unit, _thresh, x_max, _fmt = entry[0], entry[1], entry[2], entry[3], entry[4], entry[5]
        present = [
            (lbl, style)
            for (lbl, _), style in zip(TRAJ_PLOT_RUNS, TRAJ_PLOT_STYLES)
            if lbl in data and key in data[lbl]
        ]
        if not present:
            ax.set_visible(False)
            continue
        all_vals = np.concatenate([data[lbl][key].dropna().values for lbl, _ in present])
        lo = np.percentile(all_vals,  0.5)
        hi = np.percentile(all_vals, 99.5)
        if x_max is not None:
            hi = min(hi, x_max)
        bins = np.linspace(lo, hi, 50)

        for lbl, style in present:
            vals = data[lbl][key].dropna().values
            counts, edges = np.histogram(vals, bins=bins)
            widths  = np.diff(edges)
            density = counts / (counts.sum() * widths)
            centers = 0.5 * (edges[:-1] + edges[1:])
            ax.step(centers, density, where="mid", label=lbl, **style)

        unit_str = f" [{unit}]" if unit else ""
        ax.set_xlabel(label_str + unit_str)
        ax.set_ylabel("Probability density")
        ax.set_xlim(lo, hi)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
        ax.spines[["top", "right"]].set_visible(False)

    all_handles, all_labels = [], []
    seen: set = set()
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in seen:
                all_handles.append(h)
                all_labels.append(l)
                seen.add(l)
    fig.legend(all_handles, all_labels,
               loc="lower center", ncol=len(all_labels),
               bbox_to_anchor=(0.5, -0.08), frameon=False)
    fig.suptitle("Step-1 fidelity: trajectory accuracy metrics", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    print(f"  Trajectory histogram figure → {output_path}")
    plt.close(fig)


def plot_deflection_scatter(data: dict[str, pd.DataFrame], output_path: Path) -> None:
    """Scatter of TRAVIS vs raytrax deflection angle for the default run.

    Each point is one sampled scenario.  Color encodes min_rho_tr (depth of
    ray penetration), so deeply penetrating rays can be distinguished from
    edge-grazing ones.  Identity line and ±0.5° band guide the eye.
    """
    ref_label = "default"
    if ref_label not in data:
        print("  Deflection scatter: 'default' run not loaded, skipping.")
        return

    df = data[ref_label]
    needed = {"deflection_rx_deg", "deflection_tr_deg", "min_rho_tr"}
    missing = needed - set(df.columns)
    if missing:
        print(f"  Deflection scatter: missing columns {missing}, skipping.")
        return

    df_ok = df[list(needed)].dropna()
    x = df_ok["deflection_tr_deg"].values
    y = df_ok["deflection_rx_deg"].values
    c = df_ok["min_rho_tr"].values

    # Pearson r and RMSE
    r_val = float(np.corrcoef(x, y)[0, 1]) if len(x) > 1 else np.nan
    rmse  = float(np.sqrt(np.mean((y - x) ** 2)))

    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    lim_lo = min(x.min(), y.min()) * 0.97
    lim_hi = max(x.max(), y.max()) * 1.03

    # Identity band
    id_x = np.array([lim_lo, lim_hi])
    ax.fill_between(id_x, id_x - 0.5, id_x + 0.5, color="0.88", zorder=0,
                    label=r"$\pm 0.5^\circ$ band")
    ax.plot(id_x, id_x, color="0.55", lw=1.0, ls="--", zorder=1)

    sc = ax.scatter(x, y, c=c, cmap="viridis_r", s=4, alpha=0.5,
                    vmin=0.0, vmax=1.0, zorder=2, rasterized=True, linewidths=0)
    cb = fig.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label(r"$\rho_{\min}^{\mathrm{TRAVIS}}$")

    ax.set_xlim(lim_lo, lim_hi)
    ax.set_ylim(lim_lo, lim_hi)
    ax.set_xlabel(r"TRAVIS deflection $\theta_{\mathrm{defl}}$ [$^\circ$]")
    ax.set_ylabel(r"raytrax deflection $\theta_{\mathrm{defl}}$ [$^\circ$]")
    ax.set_title(r"Deflection angle comparison (\textit{default} run)")
    ax.text(0.05, 0.95,
            rf"$r = {r_val:.4f}$" + "\n" + rf"RMSE $= {rmse:.3f}^\circ$",
            transform=ax.transAxes, va="top", ha="left", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.8", alpha=0.9))
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    print(f"  Deflection scatter figure → {output_path}")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--wandb-project", default="raytrax-validation")
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading WandB run histories …")
    data = load_all(args.wandb_project)
    if not data:
        raise SystemExit("No data loaded — are the scan runs complete?")

    print("\nComputing summary statistics …")
    stats = summary_stats(data)

    print_table(stats)

    write_latex_table(stats, args.output_dir / "step1_fidelity_table.tex")
    write_markdown_table(stats, args.output_dir / "step1_fidelity_table.md")

    print("Generating interpolation-quality histograms …")
    plot_histograms(data, args.output_dir / "step1_fidelity_histograms.pdf")

    print("Generating trajectory-accuracy histograms …")
    plot_trajectory_histograms(data, args.output_dir / "step1_trajectory_histograms.pdf")

    print("Generating deflection scatter …")
    plot_deflection_scatter(data, args.output_dir / "step1_deflection_scatter.pdf")


if __name__ == "__main__":
    main()
