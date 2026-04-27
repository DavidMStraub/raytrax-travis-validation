"""Functional wrapper for TRAVIS ECRH code with dataclass containers."""

from __future__ import annotations

from typing import Literal
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
from jax import Array
from jaxtyping import Float


@dataclass
class TravisECRHInput:
    """Input parameters for TRAVIS ECRH calculation."""

    antenna_position_cyl: Float[Array, "3"]
    """Antenna position (R_m, phi_deg, Z_m) in cylindrical coordinates."""

    target_position: Float[Array, "3"]
    """Target position in coordinates specified by target_coords_type."""

    frequency_ghz: float
    """Beam frequency in GHz."""

    power_mw: float
    """Input power in MW."""

    equilibrium_file: str
    """Path to VMEC wout equilibrium file."""

    target_coords_type: str = "cyl"
    """Target coordinate system: 'cart' (X,Y,Z), 'cyl' (R,phi,Z), or 'W7X-angles' (theta_pol,theta_tor,0)."""

    mode: Literal["O", "X"] = "O"
    """Heating scenario: 'O' or 'X' mode."""

    rho_grid: Float[Array, " n"] | None = None
    """Normalized flux coordinate grid for plasma profiles."""

    electron_density_1e20: Float[Array, " n"] | None = None
    """Electron density in 10^20 m^-3."""

    electron_temperature_keV: Float[Array, " n"] | None = None
    """Electron temperature in keV."""

    max_length_m: float = 30.0
    """Maximum ray path length in meters."""

    max_steps: int = 5000
    """Maximum number of ray tracing steps."""

    dielectric_tracing: str = "cold"
    """Dielectric tensor model for ray tracing: 'cold' or 'weakly_relativistic'."""

    hamiltonian: str = "West"
    """Hamiltonian formulation: 'West' or 'Tokman'."""

    b0_normalization: float | None = None
    """If set, normalize B field to this value [T] on the magnetic axis at phi=0.
    Corresponds to TRAVIS's 'B0_normalization_type at angle on magn.axis'."""

    ne_parm: tuple[float, float, float, float, float] = (0.0, 1.0, 2.0, 0.0, 0.0)
    """TRAVIS Ne-parm parameters (a, p, q, h, w) for analytic profile."""

    te_parm: tuple[float, float, float, float, float] = (0.0, 1.0, 2.0, 0.0, 0.0)
    """TRAVIS Te-parm parameters (a, p, q, h, w) for analytic profile."""

    hgrid: float = 0.022
    """Cylindrical mesh step size [m] (hgrid): sets dR = dZ = hgrid for the 3D
    lookup mesh. Default 0.022 ≈ a_minor/25. TRAVIS clamps to [a_minor/250, a_minor/25]."""

    dphi: float = 2.0
    """Toroidal mesh step size [deg] (dphi) for the 3D lookup mesh. Default 2° = 10°/N_periods.
    TRAVIS clamps to [1°/N_periods, 10°/N_periods]."""

    rk_accuracy: float = 1e-5
    """RK integrator accuracy (RK_accuracy). Default 1e-5."""

    max_rk_stepsize_wavelengths: float = 10.0
    """Maximum RK step size in wavelengths (max_RK_stepsize_[wave_length]). Default 10."""

    resonance_umax: float = 7.0
    """Upper velocity limit for the resonance integral in units of u/u_th (umaxgrid in TRAVIS,
    second value in Max_power_of_larmor_expansion_and_grid_parms). Default 7; use 35 for O2
    mode to capture the relativistic high-velocity tail."""

    resonance_grid_points: int = 700
    """Number of u_par grid points for the resonance line integral (nugrid in TRAVIS,
    third value in Max_power_of_larmor_expansion_and_grid_parms). Default 700; use 1500
    for O2 mode for finer velocity-space resolution."""


@dataclass
class TravisECRHOutput:
    """Output from TRAVIS ECRH calculation."""

    position_m: Float[Array, "n_points 3"]
    """Ray trajectory positions in Cartesian coordinates (X, Y, Z) in meters."""

    refractive_index: Float[Array, "n_points 3"]
    """Refractive index vector components (Nx, Ny, Nz) along trajectory."""

    arc_length_m: Float[Array, " n_points"]
    """Arc length along ray in meters."""

    optical_depth: Float[Array, " n_points"]
    """Optical depth (tau) along trajectory."""

    rho: Float[Array, " n_points"]
    """Normalized flux coordinate along trajectory."""

    absorption_m_inv: Float[Array, " n_points"]
    """Absorption coefficient in m^-1."""

    linear_power_density_w_per_m: Float[Array, " n_points"]
    """Power deposition per unit length in W/m."""

    electron_density_1e20: Float[Array, " n_points"]
    """Electron density in 10^20 m^-3 along trajectory."""

    electron_temperature_keV: Float[Array, " n_points"]
    """Electron temperature in keV along trajectory."""

    magnetic_field_magnitude_T: Float[Array, " n_points"]
    """Magnetic field magnitude in Tesla along trajectory."""

    magnetic_field_cart: Float[Array, "n_points 3"]
    """Magnetic field vector (Bx, By, Bz) in Cartesian coordinates [T]."""

    rho_profile: Float[Array, " n_rho"]
    """Radial grid for integrated power deposition profile."""

    power_density_w_per_m3: Float[Array, " n_rho"]
    """Power density profile in W/m^3."""

    total_absorbed_power_mw: float
    """Total absorbed power in MW."""

    success: bool
    """Whether TRAVIS execution completed successfully."""


_MESH_FILE = "mesh.bin4"
_MESH_HASH_FILE = "equilibrium.md5"


def _equilibrium_hash(params: TravisECRHInput) -> str:
    h = hashlib.md5()
    h.update(Path(params.equilibrium_file).read_bytes())
    h.update(f"hgrid={params.hgrid}|dphi={params.dphi}".encode())
    return h.hexdigest()


def _mesh_cache_valid(mesh_cache_dir: Path, params: TravisECRHInput) -> bool:
    hash_file = mesh_cache_dir / _MESH_HASH_FILE
    mesh_file = mesh_cache_dir / _MESH_FILE
    if not mesh_file.exists() or not hash_file.exists():
        return False
    return hash_file.read_text().strip() == _equilibrium_hash(params)


def run_travis(
    travis_executable: Path,
    params: TravisECRHInput,
    output_dir: Path | None = None,
    mesh_cache_dir: Path | None = None,
) -> TravisECRHOutput:
    """Run TRAVIS with given parameters and return results."""
    if output_dir is None:
        with tempfile.TemporaryDirectory() as tmpdir:
            return _run_travis_internal(travis_executable, params, Path(tmpdir), mesh_cache_dir)
    else:
        return _run_travis_internal(travis_executable, params, output_dir, mesh_cache_dir)


def _run_travis_internal(
    travis_executable: Path,
    params: TravisECRHInput,
    output_dir: Path,
    mesh_cache_dir: Path | None,
) -> TravisECRHOutput:
    """Execute TRAVIS workflow: write inputs, run, parse outputs."""
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    use_cached = mesh_cache_dir is not None and _mesh_cache_valid(mesh_cache_dir, params)
    if use_cached:
        assert mesh_cache_dir is not None
        shutil.copy2(mesh_cache_dir / _MESH_FILE, output_dir / _MESH_FILE)

    if use_cached:
        # mesh.bin4 contains both equilibrium data and mesh — use it directly so
        # TRAVIS loads the pre-built mesh without regenerating it.
        _write_travis_input_files(
            params, output_dir,
            use_mesh=True, save_mesh=False,
            equilibrium_override=output_dir / _MESH_FILE,
        )
    else:
        # Build mesh from VMEC with the requested hgrid/dphi resolution.
        # Save it so it can be cached for future runs.
        _write_travis_input_files(
            params, output_dir,
            use_mesh=True, save_mesh=(mesh_cache_dir is not None),
        )
    _execute_travis(travis_executable, output_dir)

    if not use_cached and mesh_cache_dir is not None:
        mesh_file = output_dir / _MESH_FILE
        if mesh_file.exists():
            mesh_cache_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(mesh_file, mesh_cache_dir / _MESH_FILE)
            (mesh_cache_dir / _MESH_HASH_FILE).write_text(_equilibrium_hash(params))

    return _parse_travis_output(output_dir)


def _write_travis_input_files(
    params: TravisECRHInput,
    output_dir: Path,
    use_mesh: bool = True,
    save_mesh: bool = False,
    equilibrium_override: Path | None = None,
) -> None:
    """Write TRAVIS input files in the .data format."""
    input_file = output_dir / "travis_input.data"

    if not params.equilibrium_file:
        raise ValueError("Equilibrium file is required")

    # When a cached mesh.bin4 is available, use it directly as the Magnetic_Configuration.
    # mesh.bin4 stores both the Fourier harmonics and the pre-built 3D lookup mesh,
    # so TRAVIS will skip mesh generation entirely (mcIsMeshOK returns true).
    # Otherwise use the VMEC netCDF file with absolute path.
    if equilibrium_override is not None:
        equilibrium_str = str(equilibrium_override.resolve())
    else:
        equilibrium_str = str(Path(params.equilibrium_file).resolve())

    # Always use analytic plasma profiles
    ne_central = (
        params.electron_density_1e20[0]
        if params.electron_density_1e20 is not None
        else 1.0
    )
    te_central = (
        params.electron_temperature_keV[0]
        if params.electron_temperature_keV is not None
        else 3.0
    )

    ne_parm_str = " ".join(str(x) for x in params.ne_parm)
    te_parm_str = " ".join(str(x) for x in params.te_parm)

    plasma_section = f"""File_with_plasma_profiles nofile analytic
Central_Ne_[1e20/m^3] {ne_central}
Ne-parm {ne_parm_str}
Central_Te_[keV] {te_central}
Te-parm {te_parm_str}"""

    # Coordinate system mapping
    coord_type_map = {
        "cart": "cartesian",
        "cyl": "cylindrical",
        "W7X-angles": "W7X aiming angles",
        "ITER-angles": "ITER aiming angles",
    }
    target_coords_str = coord_type_map.get(
        params.target_coords_type, params.target_coords_type
    )

    input_content = f"""***This_is_input_file_for_TRAVIS_ECRH_code***
TempDirectory ./
Vessel_Configuration nofile
Mirror_Configuration nofile
Magnetic_Configuration {equilibrium_str}
Use_EFIT_file_directly 0
B0_normalization_type {"at angle on magn.axis" if params.b0_normalization is not None else "do not scale"}
B0_normalization_value_and_B_direction {params.b0_normalization if params.b0_normalization is not None else 1.0} 0
Angle_for_B0_[degree] 0
Flux_surface_label toroidal_rho
Use_and_save_mesh {1 if use_mesh else 0} {1 if save_mesh else 0}
Stellarator_symmetry 1
Accuracy_[m] 0.001
gridStep_[m] {params.hgrid}
gridStep_[degree] {params.dphi}
Bmn_truncation_level 2e-05
Plasma_size(rmax/a)_and_edge_width(dr/a) 1 0
{plasma_section}
Central_Zeff 1.5
Zeff-parm 0 1 1 0 0
Type_of_distribution_function Maxwell
File_with_distribution_function nofile
Adjoint_approach_model lmfp (with trapped particles)
Adjoint_approach_DKES_data nofile
Adjoint_approach_collision_operator momentum_conservation
#_of_points_in_deposition_profile 100
***Beams_data_below_this_line***
Number_of_beams 1
******Single ray*****
Beam_id 1
Beam_name single_ray
Heating_Scenario {params.mode}
Frequency_[GHz] {params.frequency_ghz}
Input_power_[MW] {params.power_mw}
Antenna_position {params.antenna_position_cyl[0]} {params.antenna_position_cyl[1]} {params.antenna_position_cyl[2]} 0
antenna_position_in_cartesian_coordinates 0
Target_position {params.target_position[0]} {params.target_position[1]} {params.target_position[2]}
Target_coordinates {target_coords_str}
Beam_radii_[m]_and_astigmatism_axis[deg] 0.02 0.02 0
Beam_focal_lengths_[m]_and_QOemul_flag 10 10 0
Number_of_concentric_circles_about_the_central_ray 0
Number_of_rays_in_each_circle 0
Stop_tracing_if_no_more_power 1
max_path_of_beam_[m] {params.max_length_m}
max_RK_iterations {params.max_steps}
min_RK_stepsize_[wave_length] 1e-05
max_RK_stepsize_[wave_length] {params.max_rk_stepsize_wavelengths}
RK_accuracy {params.rk_accuracy}
Dielectric_tensor_summation_limit_[0_for_auto] 0
Max_power_of_larmor_expansion_and_grid_parms 1 {params.resonance_umax} {params.resonance_grid_points}
Dielectric_tensor_model_for_tracing {params.dielectric_tracing}
Hamiltonian_for_tracing {params.hamiltonian}
Number_of_passes_and_reflection_coefficients 1 1 1
"""

    with open(input_file, "w") as f:
        f.write(input_content)


def _execute_travis(travis_executable: Path, output_dir: Path) -> None:
    """Execute TRAVIS."""
    result = subprocess.run(
        [str(travis_executable.resolve()), "travis_input.data"],
        cwd=output_dir,
        capture_output=True,
        text=True,
        timeout=300,
    )

    status_file = output_dir / "run.status"
    if status_file.exists():
        with open(status_file, "r") as f:
            status_line = f.readline().strip()
            exit_code = int(status_line.split()[0])
            if exit_code != 0:
                raise RuntimeError(
                    f"TRAVIS execution failed with exit code {exit_code}"
                )
    elif result.returncode != 0:
        raise RuntimeError(f"TRAVIS execution failed: {result.stderr}")


def _parse_travis_output(output_dir: Path) -> TravisECRHOutput:
    """Parse TRAVIS output files."""
    beamtrace_file = output_dir / "beamtrace_1"
    if not beamtrace_file.exists():
        raise RuntimeError("TRAVIS output file beamtrace_1 not found")

    traj_data = _parse_beamtrace(beamtrace_file)

    profile_file = output_dir / "Pabs_Icd_profiles_1"
    if not profile_file.exists():
        raise RuntimeError("TRAVIS output file Pabs_Icd_profiles_1 not found")

    profile_data = _parse_radial_profile(profile_file)

    return TravisECRHOutput(
        position_m=traj_data["position_m"],
        refractive_index=traj_data["refractive_index"],
        arc_length_m=traj_data["arc_length_m"],
        optical_depth=traj_data["optical_depth"],
        rho=traj_data["rho"],
        absorption_m_inv=traj_data["absorption_m_inv"],
        linear_power_density_w_per_m=traj_data["linear_power_density_w_per_m"],
        electron_density_1e20=traj_data["electron_density_1e20"],
        electron_temperature_keV=traj_data["electron_temperature_keV"],
        magnetic_field_magnitude_T=traj_data["magnetic_field_magnitude_T"],
        magnetic_field_cart=traj_data["magnetic_field_cart"],
        rho_profile=profile_data["rho"],
        power_density_w_per_m3=profile_data["power_density_w_per_m3"],
        total_absorbed_power_mw=profile_data["total_absorbed_power_mw"],
        success=True,
    )


def _parse_beamtrace(filepath: Path) -> dict:
    """Parse TRAVIS beamtrace output file.

    Columns: Nray, path, X, Y, Z, Nx, Ny, Nz, rho, ne, Te, |B|,
             Nper, Npar, Nperc, Nparc, damp0, damp, tau0, tau, ...
    """
    data = []
    with open(filepath, "r") as f:
        f.readline()

        for line in f:
            values = line.split()
            if len(values) > 50:
                data.append([float(v) for v in values])

    data = np.array(data)

    return {
        "position_m": data[:, 2:5],
        "refractive_index": data[:, 5:8],
        "arc_length_m": data[:, 1],
        "rho": data[:, 8],
        "optical_depth": data[:, 18],
        "absorption_m_inv": data[:, 17],  # damp (col 17); damp0 (col 16) is always zero
        "linear_power_density_w_per_m": (data[:, 23] + data[:, 24]) * 1e6,  # MW/m → W/m
        "electron_density_1e20": data[:, 9] / 1e20,  # Column 10: ne in m^-3
        "electron_temperature_keV": data[:, 10],  # Column 11: Te in keV
        "magnetic_field_magnitude_T": data[:, 11],
        "magnetic_field_cart": data[:, 48:51],  # Bx, By, Bz [T] Cartesian
    }


def _parse_radial_profile(filepath: Path) -> dict:
    """Parse TRAVIS radial profile output file.

    Columns: reff/a, dP_p/dV, dP_t/dV, P_p, P_t, dP/dV, Pabs, ...
    """
    data = []
    with open(filepath, "r") as f:
        f.readline()
        f.readline()

        for line in f:
            values = line.split()
            if len(values) >= 6:
                data.append([float(v) for v in values])

    data = np.array(data)

    return {
        "rho": data[:, 0],
        "power_density_w_per_m3": data[:, 5] * 1e6,
        "total_absorbed_power_mw": data[-1, 6] if len(data) > 0 else 0.0,
    }


def resolve_travis_exe(cli_arg: Path | None) -> Path:
    """Resolve TRAVIS executable: CLI arg → TRAVIS_EXE env var → PATH."""
    if cli_arg is not None:
        return cli_arg
    env_exe = os.environ.get("TRAVIS_EXE")
    if env_exe:
        return Path(env_exe)
    found = shutil.which("travis-nc")
    if found:
        return Path(found)
    print("ERROR: travis-nc not found. Set TRAVIS_EXE or pass --travis-exe.")
    sys.exit(1)
