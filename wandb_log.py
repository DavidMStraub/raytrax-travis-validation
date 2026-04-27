"""WandB logging helpers for validation scans.

One WandB run per scan.  Scan-level settings (grid resolution, solver config)
go into the run config.  Each sampled scenario is logged as a step so all
scalar metrics are accessible via ``run.history()`` as a DataFrame.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import wandb


def _scalar_fields(obj) -> dict:
    """Extract loggable scalar fields from a dataclass instance."""
    out = {}
    for f in dataclasses.fields(obj):
        v = getattr(obj, f.name)
        if isinstance(v, (int, float, bool, str)):
            out[f.name] = v
        elif isinstance(v, tuple) and all(isinstance(x, (int, float, bool, str)) for x in v):
            out[f.name] = list(v)
    return out


def init_scan(
    project: str,
    config: dict,
    group: str | None = None,
    name: str | None = None,
    dry_run: bool = False,
) -> None:
    """Initialize a single WandB run for a full parameter scan.

    Call once before the scan loop.  Scan-level settings (grid resolution,
    solver parameters, seed) go into *config* and are shared across all steps.
    """
    if dry_run:
        print(f"[dry-run] init scan: project={project}  group={group}  name={name}")
        print(f"  config: {config}")
        return
    wandb.init(project=project, config=config, group=group, name=name)


def log_sample(
    scenario,
    metrics,
    step: int,
    error: str | None = None,
    dry_run: bool = False,
) -> None:
    """Log one scenario as a step.

    Per-scenario parameters and comparison metrics are merged into a single
    flat dict and logged at *step*.  Failed runs are logged with
    ``success=False`` so they appear in the history and can be filtered out.
    """
    row: dict[str, Any] = {"success": error is None}
    row.update(_scalar_fields(scenario))
    if metrics is not None:
        row.update(_scalar_fields(metrics))
    if dry_run:
        print(f"[dry-run] step={step}  {row}")
        return
    wandb.log(row, step=step)


def finish_scan(dry_run: bool = False) -> None:
    """Finish the current scan run."""
    if dry_run:
        print("[dry-run] scan finished")
        return
    wandb.finish()
