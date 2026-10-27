"""Convergence of the Raytrax resonance integral on the step-2 scenarios (no TRAVIS needed).

Produces
  fig3_integrand.pdf    one resonance integrand per mode and harmonic over the resonance
                        circle, with the integration window and the Gauss-Legendre nodes
  fig4_convergence.pdf  error of the absorption coefficient alpha and of the optical depth
                        tau versus (a) the number of Gauss-Legendre nodes and (b) the
                        window width K in e-foldings of the Maxwell-Juettner factor
and prints the error statistics.

The rays are the step-2 rays of `scan.py` (same sampling and seed), traced with Raytrax.
alpha is evaluated at the trajectory points and at points linearly interpolated between
them, since a narrow resonance can lie between accepted ODE steps. Each quadrature
setting is compared with a reference of 256 nodes on a K = 60 window. Errors are
|delta alpha| normalised to the peak alpha of the ray, and |delta tau| / tau with tau
integrated along the fixed trajectory.

Requires a Raytrax version whose absorption functions take `n_quadrature_nodes`.

Usage:
    python convergence.py
    python convergence.py --n-samples 400 --output-dir figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from paper_figures import COMBOS, TAU_MIN, _set_style  # noqa: E402
from profiles import build_profiles, cylindrical_to_cartesian  # noqa: E402
from scan import B0_TARGET_STEP2, step2_scenarios  # noqa: E402
from w7x_setup import build_scaled_equilibrium, get_w7x_equilibrium  # noqa: E402

from raytrax.api import trace  # noqa: E402
from raytrax.equilibrium.interpolate import (  # noqa: E402
    build_magnetic_field_interpolator,
    build_rho_interpolator,
)
from raytrax.math import bessel  # noqa: E402
from raytrax.physics import absorption, polarization, quantities  # noqa: E402
from raytrax.physics import dielectric_tensor as dielectric_tensor_module  # noqa: E402
from raytrax.types import Beam, RadialProfiles  # noqa: E402

DEFAULT_NODES = 24
DEFAULT_EFOLDINGS = 40.0
REFERENCE = (256, 60.0)  # (nodes, K)
NODE_SWEEP = [6, 8, 12, 16, 20, 24, 32, 48]
EFOLDING_SWEEP = [10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 50.0]
NODES_FOR_EFOLDING_SWEEP = 64
SUBDIVISIONS = 8  # evaluation points per accepted ODE step
MAX_HARMONIC = 2  # Beam default, as in the step-2 scan


# ── Rays ──────────────────────────────────────────────────────────────────────


def trace_rays(n_samples: int, seed: int) -> list[dict]:
    """Trace the first n_samples step-2 rays and return alpha inputs along each."""
    eq = build_scaled_equilibrium(get_w7x_equilibrium(), B0_TARGET_STEP2)
    scenarios = step2_scenarios(build_rho_interpolator(eq), build_magnetic_field_interpolator(eq), seed)
    rays = []
    for i in range(n_samples):
        params = next(scenarios)
        rho, ne, te = build_profiles(params.ne_central, params.ne_parm, params.te_central, params.te_parm)
        profiles = RadialProfiles(
            rho=jnp.array(rho), electron_density=jnp.array(ne), electron_temperature=jnp.array(te)
        )
        beam = Beam(
            position=jnp.array(cylindrical_to_cartesian(*params.antenna_cyl)),
            direction=jnp.array(params.direction_cart),
            frequency=jnp.array(params.frequency_ghz * 1e9),
            mode=params.mode,
            power=params.power_mw * 1e6,
        )
        try:
            bp = trace(eq, profiles, beam).beam_profile
        except Exception as exc:  # keep going; the scan logs such samples as failures
            print(f"  ray {i}: trace failed ({exc})")
            continue
        fields = {
            "s": np.asarray(bp.arc_length),
            "N": np.asarray(bp.refractive_index),
            "B": np.asarray(bp.magnetic_field),
            "ne": np.asarray(bp.electron_density),
            "te": np.asarray(bp.electron_temperature),
        }
        # Linear interpolation between accepted steps, so the evaluation is not limited to
        # the points the adaptive solver happened to accept.
        t = np.linspace(0.0, 1.0, SUBDIVISIONS, endpoint=False)
        dense = {}
        for key, v in fields.items():
            v2 = v.reshape(len(v), -1)
            seg = v2[:-1, None, :] + t[None, :, None] * (v2[1:] - v2[:-1])[:, None, :]
            dense[key] = np.concatenate([seg.reshape(-1, v2.shape[1]), v2[-1:]]).squeeze()
        dense.update(
            ray=i,
            combo=f"{params.mode}{params.harmonic}",
            mode=params.mode,
            harmonic=params.harmonic,
            f=params.frequency_ghz * 1e9,
        )
        rays.append(dense)
    return rays


# ── Absorption coefficient for a quadrature setting ──────────────────────────


def alpha_along_rays(rays: list[dict], n_nodes: int, efoldings: float) -> list[np.ndarray]:
    """alpha at every point of every ray with n_nodes Gauss-Legendre nodes on a K-e-folding window."""
    # The window width is a module constant; it is read when the function is traced.
    absorption._RESONANCE_EFOLDINGS = efoldings
    jax.clear_caches()
    out: list[np.ndarray | None] = [None] * len(rays)
    for mode in ("X", "O"):
        idx = [k for k, r in enumerate(rays) if r["mode"] == mode]
        if not idx:
            continue

        def point(p, mode=mode):
            N, B, ne, te, f = p
            return absorption.absorption_coefficient_conditional(
                refractive_index=N,
                magnetic_field=B,
                electron_density_1e20_per_m3=ne,
                electron_temperature_keV=te,
                frequency=f,
                mode=mode,
                max_harmonic=MAX_HARMONIC,
                n_quadrature_nodes=n_nodes,
            )

        args = tuple(
            jnp.asarray(np.concatenate([rays[k][key] if key != "f" else np.full(len(rays[k]["s"]), rays[k]["f"]) for k in idx]))
            for key in ("N", "B", "ne", "te", "f")
        )
        a = np.asarray(jax.jit(lambda *x, point=point: jax.lax.map(point, x))(*args))
        offsets = np.cumsum([0] + [len(rays[k]["s"]) for k in idx])
        for j, k in enumerate(idx):
            out[k] = a[offsets[j] : offsets[j + 1]]
    absorption._RESONANCE_EFOLDINGS = DEFAULT_EFOLDINGS
    return out  # type: ignore[return-value]


def ray_errors(rays, alpha, alpha_ref) -> list[dict]:
    """Per-ray alpha error (relative to the ray's peak alpha) and tau error on the fixed path.

    Only rays with tau >= TAU_MIN count, as in the paper figures: on rays that barely absorb,
    errors relative to the peak alpha say nothing about the absorbed power.
    """
    errs = []
    for r, a, ar in zip(rays, alpha, alpha_ref):
        tau_ref = np.trapezoid(ar, r["s"])
        if tau_ref < TAU_MIN:
            continue
        errs.append(
            {
                "combo": r["combo"],
                "alpha": float(np.max(np.abs(a - ar)) / np.max(np.abs(ar))),
                "tau": float(abs(np.trapezoid(a, r["s"]) - tau_ref) / tau_ref),
            }
        )
    return errs


# ── Integrand on the resonance circle ─────────────────────────────────────────


def integrand_on_circle(N, B, ne, te, f, mode, harmonic, n_nodes=DEFAULT_NODES, efoldings=DEFAULT_EFOLDINGS):
    """Resonance integrand over the full circle, the integration window and the node positions.

    Mirrors the setup in raytrax.physics.absorption.absorption_coefficient and
    compute_resonance_integral for one point.
    """
    fce = quantities.electron_cyclotron_frequency(magnetic_field_strength=jnp.linalg.norm(B))
    fpe = quantities.electron_plasma_frequency(electron_density_1e20_per_m3=ne)
    b = B / jnp.linalg.norm(B)
    n_para = jnp.dot(N, b)
    n_perp = jnp.linalg.norm(N - n_para * b)
    vth = quantities.normalized_electron_thermal_velocity(electron_temperature_keV=te)
    eps = dielectric_tensor_module.weakly_relativistic_dielectric_tensor(
        frequency=f, plasma_frequency=fpe, cyclotron_frequency=fce, thermal_velocity=vth,
        refractive_index_para=n_para, refractive_index_perp=n_perp,
        max_s=MAX_HARMONIC, max_k=MAX_HARMONIC,
    )
    pol = polarization.polarization(
        dielectric_tensor=eps, refractive_index_perp=n_perp, refractive_index_para=n_para,
        frequency=f, cyclotron_frequency=fce, mode=mode,
    )
    nY = harmonic * fce / f
    denom = 1 - n_para**2
    delta_u = jnp.sqrt(jnp.maximum(n_para**2 + nY**2 - 1, 0.0)) / denom
    u_res = nY * n_para / denom
    lo, hi = float(u_res - delta_u), float(u_res + delta_u)
    mu = 2 / vth**2
    support = efoldings / (mu * max(abs(float(n_para)), 1e-12))
    w_lo, w_hi = (max(lo, hi - support), hi) if n_para < 0 else (lo, min(hi, lo + support))
    K2 = bessel.kve_jax(2, mu)

    def g(u):
        return absorption.resonance_integrand(
            harmonic_index=harmonic, cyclotron_frequency=fce, frequency=f,
            refractive_index_para=n_para, refractive_index_perp=n_perp, parallel_momentum=u,
            thermal_velocity=vth, K2_scaled=K2, polarization_vector=pol,
        )

    u = np.linspace(lo, hi, 4001)
    x, _ = np.polynomial.legendre.leggauss(n_nodes)
    nodes = 0.5 * (w_hi - w_lo) * x + 0.5 * (w_hi + w_lo)
    return {
        "u": u, "I": np.asarray(jax.vmap(g)(jnp.asarray(u))),
        "nodes": nodes, "I_nodes": np.asarray(jax.vmap(g)(jnp.asarray(nodes))),
        "window": (w_lo, w_hi), "vth": float(vth), "n_para": float(n_para), "te": float(te),
    }


# ── Plotting ──────────────────────────────────────────────────────────────────


def plot_integrands(rays, alpha_ref, out: Path) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(494.4 / 72, 3.6), constrained_layout=True)
    for col, combo in enumerate(COMBOS):
        cand = [(k, int(np.argmax(alpha_ref[k]))) for k, r in enumerate(rays) if r["combo"] == combo]
        cand = [(k, j) for k, j in cand if alpha_ref[k][j] > 0 and np.trapezoid(alpha_ref[k], rays[k]["s"]) >= TAU_MIN]
        if not cand:
            continue
        k, j = max(cand, key=lambda kj: alpha_ref[kj[0]][kj[1]])  # most strongly absorbing point
        r = rays[k]
        d = integrand_on_circle(
            jnp.asarray(r["N"][j]), jnp.asarray(r["B"][j]), float(r["ne"][j]), float(r["te"][j]),
            r["f"], r["mode"], r["harmonic"],
        )
        scale = np.max(np.abs(d["I"]))
        style = COMBOS[combo]
        for row, log in enumerate((False, True)):
            ax = axes[row, col]
            y, yn = np.abs(d["I"]) / scale, np.abs(d["I_nodes"]) / scale
            ax.axvspan(*(np.array(d["window"]) / d["vth"]), color="0.9", lw=0)
            ax.plot(d["u"] / d["vth"], y, color=style["color"], lw=1.2)
            ax.plot(d["nodes"] / d["vth"], yn, "|", color="k", ms=5, mew=0.8)
            # The circle can span many v_th (N_par near 1); show the window with a margin.
            w_lo, w_hi = d["window"]
            margin = 0.3 * (w_hi - w_lo)
            ax.set_xlim(max(d["u"][0], w_lo - margin) / d["vth"], min(d["u"][-1], w_hi + margin) / d["vth"])
            if log:
                ax.set_yscale("log")
                ax.set_ylim(1e-20, 2)
            else:
                ax.set_ylim(0, 1.05)
            ax.set_xlabel(r"$u_\parallel / v_\mathrm{th}$" if log else "")
        axes[0, col].set_title(
            f"{combo}  ($T_\\mathrm{{e}}$={d['te']:.1f} keV, $N_\\parallel$={d['n_para']:.2f})", fontsize=8
        )
    axes[0, 0].set_ylabel("|integrand| (norm.)")
    axes[1, 0].set_ylabel("|integrand| (norm.)")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_convergence(node_errs: dict, k_errs: dict, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(494.4 / 72, 2.8), constrained_layout=True)
    for ax, errs, xs in ((axes[0], node_errs, NODE_SWEEP), (axes[1], k_errs, EFOLDING_SWEEP)):
        for combo, style in COMBOS.items():
            worst = [max((e["alpha"] for e in errs[x] if e["combo"] == combo), default=np.nan) for x in xs]
            median = [np.median([e["alpha"] for e in errs[x] if e["combo"] == combo]) for x in xs]
            ax.semilogy(xs, np.maximum(worst, 1e-17), "-", color=style["color"], marker=style["marker"], ms=3,
                        label=f"{combo} (max)")
            ax.semilogy(xs, np.maximum(median, 1e-17), ":", color=style["color"], lw=1)
        ax.grid(True, which="major", alpha=0.3)
        ax.set_ylabel(r"$|\Delta\alpha|$ / peak $\alpha$ of the ray")
    kk = np.linspace(min(EFOLDING_SWEEP), max(EFOLDING_SWEEP), 100)
    axes[1].semilogy(kk, kk**2 * np.exp(-kk), "k--", lw=0.8, label=r"$K^2 e^{-K}$")
    axes[0].axvline(DEFAULT_NODES, color="0.5", lw=0.8)
    axes[1].axvline(DEFAULT_EFOLDINGS, color="0.5", lw=0.8)
    axes[0].set_xlabel(f"Gauss-Legendre nodes (K = {DEFAULT_EFOLDINGS:g})")
    axes[1].set_xlabel(f"Window K [e-foldings] ({NODES_FOR_EFOLDING_SWEEP} nodes)")
    axes[1].legend(*axes[1].get_legend_handles_labels(), frameon=False, fontsize=7, ncol=1)
    for ax, letter in zip(axes, "ab"):
        ax.set_title(f"({letter})", loc="left", fontsize=9, fontweight="bold")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def print_stats(label: str, errs: dict, xs: list) -> None:
    print(f"\n=== {label} ===")
    print(f"{'':>6}" + "".join(f"{c + ' max α':>12}{c + ' max τ':>12}" for c in COMBOS))
    for x in xs:
        row = f"{x:>6g}"
        for combo in COMBOS:
            ea = [e["alpha"] for e in errs[x] if e["combo"] == combo]
            et = [e["tau"] for e in errs[x] if e["combo"] == combo and np.isfinite(e["tau"])]
            row += f"{max(ea, default=np.nan):12.2e}{max(et, default=np.nan):12.2e}"
        print(row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _set_style()

    print(f"Tracing {args.n_samples} step-2 rays …")
    rays = trace_rays(args.n_samples, args.seed)
    n_points = sum(len(r["s"]) for r in rays)
    print(f"{len(rays)} rays, {n_points} evaluation points ({SUBDIVISIONS} per ODE step)")

    alpha_ref = alpha_along_rays(rays, *REFERENCE)
    check = alpha_along_rays(rays, 384, 70.0)
    print("reference vs (384 nodes, K=70): worst |Δα| / peak α =",
          f"{max(e['alpha'] for e in ray_errors(rays, check, alpha_ref)):.1e}")

    node_errs = {n: ray_errors(rays, alpha_along_rays(rays, n, DEFAULT_EFOLDINGS), alpha_ref) for n in NODE_SWEEP}
    k_errs = {
        k: ray_errors(rays, alpha_along_rays(rays, NODES_FOR_EFOLDING_SWEEP, k), alpha_ref) for k in EFOLDING_SWEEP
    }
    print(f"rays with tau >= {TAU_MIN} per mode: " + ", ".join(
        f"{c} {sum(e['combo'] == c for e in node_errs[DEFAULT_NODES])}" for c in COMBOS))
    print_stats(f"Error vs Gauss-Legendre nodes (K = {DEFAULT_EFOLDINGS:g})", node_errs, NODE_SWEEP)
    print_stats(f"Error vs window K ({NODES_FOR_EFOLDING_SWEEP} nodes)", k_errs, EFOLDING_SWEEP)

    plot_integrands(rays, alpha_ref, args.output_dir / "fig3_integrand.pdf")
    plot_convergence(node_errs, k_errs, args.output_dir / "fig4_convergence.pdf")


if __name__ == "__main__":
    main()
