"""Plasma profile utilities and coordinate helpers."""

import numpy as np
from scipy.interpolate import interp1d


def cylindrical_to_cartesian(r_m: float, phi_deg: float, z_m: float) -> tuple[float, float, float]:
    """Convert cylindrical coordinates to Cartesian.

    Args:
        r_m: Major radius in metres.
        phi_deg: Toroidal angle in degrees.
        z_m: Vertical position in metres.

    Returns:
        Cartesian position ``(x, y, z)`` in metres.
    """
    phi_rad = np.deg2rad(phi_deg)
    return (float(r_m * np.cos(phi_rad)), float(r_m * np.sin(phi_rad)), float(z_m))


def travis_profile(
    rho: np.ndarray,
    central_value: float,
    a: float,
    p: float,
    q: float,
    h: float,
    w: float,
) -> np.ndarray:
    """Evaluate the TRAVIS analytic plasma profile.

    The profile formula is::

        y(rho) = a_adj + (1 - a_adj) * (1 - rho^p)^q + h * (1 - exp(-rho^2 / w^2))

    where ``a_adj = a - h`` when ``|h| > 1e-6`` (the pedestal offset is absorbed
    into the core level so the edge value equals *a*).

    Args:
        rho: Normalised flux-coordinate grid, values in ``[0, 1]``.
        central_value: Profile value at ``rho = 0``.
        a: Edge (pedestal) level as a fraction of *central_value*.
        p: Exponent controlling the profile width.
        q: Exponent controlling the profile peaking.
        h: Hollow-core amplitude (negative for a hollow profile).
        w: Width of the hollow-core Gaussian.

    Returns:
        Profile values on *rho* with the same shape, scaled by *central_value*.
    """
    rho = np.asarray(rho)
    a_adj = a - h if abs(h) > 1e-6 else a
    h_ = h if abs(h) > 1e-6 else 0.0

    if abs(h_) > 1e-6 and abs(w) > 1e-3:
        hole = h_ * (1.0 - np.exp(-np.clip(rho**2 / w**2, 0.0, 40.0)))
    else:
        hole = 0.0

    if q < 1e-6 and abs(h_) < 1e-6:
        y = np.ones_like(rho)
    else:
        y = a_adj + (1.0 - a_adj) * (1.0 - np.clip(rho, 0.0, 1.0) ** p) ** q + hole

    return central_value * y


def build_profiles(
    ne_central: float,
    ne_parm: tuple[float, float, float, float, float],
    te_central: float,
    te_parm: tuple[float, float, float, float, float],
    n_rho: int = 501,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build electron density and temperature profile arrays using the TRAVIS analytic formula.

    Args:
        ne_central: Central electron density in units of 10^20 m^-3.
        ne_parm: TRAVIS ``Ne-parm`` parameters ``(a, p, q, h, w)``.
        te_central: Central electron temperature in keV.
        te_parm: TRAVIS ``Te-parm`` parameters ``(a, p, q, h, w)``.
        n_rho: Number of points on the ``[0, 1]`` rho grid.

    Returns:
        Tuple ``(rho, ne_1e20, te_keV)`` of arrays with *n_rho* elements each.
    """
    rho = np.linspace(0.0, 1.0, n_rho)
    ne = travis_profile(rho, ne_central, *ne_parm)
    te = travis_profile(rho, te_central, *te_parm)
    return rho, ne, te


def interpolate_to_arc(
    s_source: np.ndarray,
    values: np.ndarray,
    s_target: np.ndarray,
) -> np.ndarray:
    """Interpolate scalar or vector values from one arc-length grid onto another.

    Uses linear interpolation and clamps to the boundary values outside the
    source range (no extrapolation artefacts at the edges).

    Args:
        s_source: Arc-length coordinates of the source data, shape ``(N,)``.
            Must be strictly increasing.
        values: Values to interpolate, shape ``(N,)`` or ``(N, D)``.
        s_target: Target arc-length coordinates, shape ``(M,)``.

    Returns:
        Interpolated values at *s_target*, shape ``(M,)`` or ``(M, D)``.
    """
    kind = "cubic" if len(s_source) >= 4 else "linear"
    return interp1d(
        s_source,
        values,
        axis=0,
        kind=kind,
        bounds_error=False,
        fill_value=(values[0], values[-1]),  # type: ignore[arg-type]
    )(s_target)
