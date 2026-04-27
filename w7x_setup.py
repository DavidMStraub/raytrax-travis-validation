"""W7-X equilibrium loading, scaling, and default scenario factory."""

from __future__ import annotations

from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from raytrax.equilibrium.interpolate import (
    MagneticConfiguration,
    VmecGridResolution,
    build_magnetic_field_interpolator,
    build_rho_interpolator,
)
from raytrax.examples.w7x import get_w7x_equilibrium, w7x_aiming_angles_to_direction

from profiles import cylindrical_to_cartesian
from step1_trajectory import ScenarioParams

# Re-export for convenience
__all__ = [
    "get_w7x_equilibrium",
    "build_scaled_equilibrium",
    "default_scenario_params",
    "ANTENNA_R_MIN", "ANTENNA_R_MAX", "ANTENNA_Z_MAX",
    "TARGET_R_MIN", "TARGET_R_MAX", "TARGET_Z_MAX", "TARGET_RHO_MAX",
]

# Antenna shell: outside the plasma, within realistic launcher range
ANTENNA_R_MIN = 6.25   # m  (just outside outer LCFS)
ANTENNA_R_MAX = 7.00   # m
ANTENNA_Z_MAX = 0.60   # m

# Target sampling box: inside the plasma
TARGET_R_MIN    = 4.80   # m
TARGET_R_MAX    = 6.10   # m
TARGET_Z_MAX    = 0.50   # m
TARGET_RHO_MAX  = 0.85   # only aim at well-inside-plasma points

# Fixed field for step-1 validation
B0_TARGET = 3.0   # T

# Default single-point launch parameters (W7-X standard)
_ANTENNA_CYL = (6.50866, -6.56378, -0.38)
_THETA_POL   = 15.7
_THETA_TOR   = 19.7001


def build_scaled_equilibrium(
    wout,
    b0_target: float,
    grid: VmecGridResolution | None = None,
) -> MagneticConfiguration:
    """Build a MagneticConfiguration scaled to the given on-axis field.

    Args:
        wout: VMEC wout equilibrium object.
        b0_target: Desired on-axis field strength [T].
        grid: Optional :class:`VmecGridResolution` controlling the cylindrical
            interpolation grid.  Defaults to the raytrax built-in defaults
            (n_r=45, n_z=55, n_phi=50, n_rho=40, n_theta=45).
    """
    eq_unscaled    = MagneticConfiguration.from_vmec_wout(wout, grid=grid)
    B_interp_tmp   = build_magnetic_field_interpolator(eq_unscaled)
    rho_interp_tmp = build_rho_interpolator(eq_unscaled)

    best_rho, b0_native = 999.0, 0.0
    for R in np.arange(5.0, 6.5, 0.002):
        rv = float(rho_interp_tmp(R, 0.0, 0.0))
        if rv < best_rho:
            best_rho = rv
            b0_native = float(jnp.linalg.norm(B_interp_tmp(R, 0.0, 0.0)))

    b_scale = b0_target / b0_native
    print(f"  B0 native={b0_native:.4f} T  target={b0_target:.4f} T  scale={b_scale:.6f}")
    return MagneticConfiguration.from_vmec_wout(wout, magnetic_field_scale=b_scale, grid=grid)


def default_scenario_params() -> ScenarioParams:
    """Single-point W7-X scenario using the standard aiming angles."""
    antenna_cart = cylindrical_to_cartesian(*_ANTENNA_CYL)
    direction    = w7x_aiming_angles_to_direction(_THETA_POL, _THETA_TOR, _ANTENNA_CYL[1])
    target_cart  = tuple(float(x) for x in np.array(antenna_cart) + 10.0 * np.array(direction))
    return ScenarioParams(
        antenna_cyl=_ANTENNA_CYL,
        direction_cart=tuple(float(x) for x in direction),
        target_cart=target_cart,
    )
