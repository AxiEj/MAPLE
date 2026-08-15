#!/usr/bin/env python3
"""Generate a matched QM/PCMSolver electrostatic energy decomposition.

The runner starts from a frozen gas-phase PySCF checkpoint, self-consistently
polarizes that same electronic method with the exact frozen PCMSolver cavity
and response operator, then evaluates the converged PCM density once in the
unchanged vacuum functional.  It records

    distortion = E_vac[gamma_pcm] - E_vac[gamma_vac]
    continuum  = E_pcm[gamma_pcm] - E_vac[gamma_pcm]
    total      = E_pcm[gamma_pcm] - E_vac[gamma_vac].

No MACE source, operational ledger prediction, CDS, standard-state term, or
experimental solvation label enters this reference calculation.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import time

import numpy as np

from maple.solvation.api.units import HARTREE_TO_EV
from maple.solvation.reference import (
    AOInverseDistanceIntegralCache,
    PCMSolverSCFSolvent,
    array_sha256,
    attach_pcmsolver_to_scf,
    closed_shell_density_from_orbitals,
    response_symmetry_defect,
    solvent_energy_directional_derivative_error,
)
from maple.solvation.release import (
    MatchedElectrostaticReferenceCase,
    MatchedElectrostaticReferencePanel,
    RepositorySnapshot,
    canonical_json_sha256,
    collect_loaded_repository_sources,
    committed_source_hashes,
    sha256_file,
    write_external_json_artifact,
)

SCHEMA_VERSION = "route2-matched-qm-pcmsolver-decomposition-run-v3"
PREREGISTRATION_RELATIVE_PATH = (
    "docs/implicit-solvation/benchmarks/"
    "route2-matched-qm-pcmsolver-four-prereg-v3.json"
)
INHERITED_PREREGISTRATION_RELATIVE_PATH = (
    "docs/implicit-solvation/benchmarks/"
    "route2-gto-pcm-energy-projection-four-prereg-v1.json"
)
PANEL_RELATIVE_ROOT = ".omx/benchmarks/route2-gto-pcm-energy-projection-four-v1"
CUTOFF_DIRECTORY = "cutoff-1e-12"
REFERENCE_METHOD_ID = "pyscf-2.13.1-wb97m-v-def2-tzvpd-df-rks-v1"
CONTINUUM_EQUATION_ID = "pcmsolver-bbd992d54ebe-iefpcm-matrixsymm-electrostatic-v1"
CAVITY_PROFILE_ID = "smd-intrinsic-gepol-probe0-area0.28a2-v1"
NO_CAPABILITIES = {tier: False for tier in ("E", "F", "H", "V", "M")}
REQUIRED_SOURCE_PATHS = (
    "maple/function/calculator/extra_correction/implicit/pcmsolver.py",
    "maple/solvation/reference/pyscf_pcmsolver.py",
    "maple/solvation/reference/replay.py",
    "maple/solvation/release/electrostatic_decomposition.py",
    "maple/solvation/release/evidence.py",
    "tools/route2_release/audit_matched_qm_pcmsolver_replay.py",
    "tools/route2_release/run_matched_qm_pcmsolver_decomposition.py",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--pcmsolver-library", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--preregistration",
        type=Path,
        default=None,
        help="Defaults to the tracked frozen preregistration.",
    )
    parser.add_argument(
        "--case-id",
        action="append",
        default=[],
        help="Run an exact preregistered subset; omitted means the full panel.",
    )
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--max-memory-mb", type=int, default=8000)
    parser.add_argument("--integral-batch-size", type=int, default=16)
    parser.add_argument("--verbose", type=int, choices=(0, 3, 4), default=0)
    return parser.parse_args()


def _load_json(path: Path, *, name: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot load {name} at {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"{name} must contain one JSON object.")
    return payload


def _file_record(path: Path, *, role: str) -> dict[str, object]:
    resolved = path.expanduser().resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"{role} must be a regular file: {resolved}")
    return {
        "role": role,
        "resolved_path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def _rebased_asset_path(asset_root: Path, raw_path: object) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        raise RuntimeError("frozen asset path must be a non-empty string.")
    path = Path(raw_path)
    if path.is_file():
        return path.resolve()
    try:
        marker = path.parts.index(".omx")
    except ValueError as exc:
        raise RuntimeError(f"cannot rebase frozen asset path {raw_path!r}.") from exc
    rebased = asset_root.joinpath(*path.parts[marker:]).resolve()
    if not rebased.is_file():
        raise FileNotFoundError(rebased)
    return rebased


def _validated_file(
    asset_root: Path,
    record: object,
    *,
    role: str,
) -> Path:
    if not isinstance(record, dict):
        raise RuntimeError(f"{role} record is missing.")
    path = _rebased_asset_path(asset_root, record.get("path"))
    expected = record.get("sha256")
    if not isinstance(expected, str) or sha256_file(path) != expected:
        raise RuntimeError(f"{role} SHA256 does not match its frozen record.")
    return path


def _case_asset_record(
    *,
    preregistration: dict[str, object],
    case_id: str,
) -> dict[str, object]:
    cases = preregistration.get("cases")
    if not isinstance(cases, list):
        raise RuntimeError("preregistration omits case records.")
    matching = [
        case
        for case in cases
        if isinstance(case, dict) and case.get("case_id") == case_id
    ]
    if len(matching) != 1:
        raise RuntimeError(f"preregistration has no unique case {case_id!r}.")
    return matching[0]


def _validate_preregistration(
    *,
    preregistration_path: Path,
    inherited_path: Path,
    snapshot: RepositorySnapshot,
    asset_root: Path,
    pcmsolver_library: Path,
    pyscf_module,
    threads: int,
    max_memory_mb: int,
    batch_size: int,
) -> dict[str, object]:
    expected_path = snapshot.root / PREREGISTRATION_RELATIVE_PATH
    if preregistration_path != expected_path.resolve(strict=True):
        raise RuntimeError("runner accepts only its tracked preregistration.")
    preregistration = _load_json(preregistration_path, name="preregistration")
    if (
        preregistration.get("protocol_id")
        != "route2-matched-qm-pcmsolver-four-prereg-v3"
        or preregistration.get("status") != "frozen-before-execution"
    ):
        raise RuntimeError("matched decomposition preregistration identity is invalid.")
    inherited = preregistration.get("inherited_asset_contract")
    if not isinstance(inherited, dict) or (
        inherited.get("path") != INHERITED_PREREGISTRATION_RELATIVE_PATH
        or inherited.get("sha256") != sha256_file(inherited_path)
    ):
        raise RuntimeError("inherited four-case asset contract changed.")
    execution = preregistration.get("execution_contract")
    if not isinstance(execution, dict):
        raise RuntimeError("preregistration omits its execution contract.")
    expected_sources = execution.get("source_sha256")
    loaded_sources = collect_loaded_repository_sources(
        snapshot.root, required_paths=REQUIRED_SOURCE_PATHS
    )
    sources = committed_source_hashes(snapshot, loaded_sources)
    if expected_sources != sources:
        raise RuntimeError("reference runner sources changed after preregistration.")
    runtime = execution.get("runtime")
    if not isinstance(runtime, dict):
        raise RuntimeError("preregistration omits runtime identity.")
    executable = Path(sys.executable).resolve(strict=True)
    if (
        runtime.get("python_resolved_sha256") != sha256_file(executable)
        or runtime.get("pyscf_version") != str(pyscf_module.__version__)
        or runtime.get("pyscf_init_sha256")
        != sha256_file(Path(pyscf_module.__file__).resolve(strict=True))
        or runtime.get("numpy_version") != str(np.__version__)
        or runtime.get("pcmsolver_library_sha256") != sha256_file(pcmsolver_library)
        or runtime.get("threads") != threads
        or runtime.get("max_memory_mb") != max_memory_mb
        or runtime.get("integral_batch_size") != batch_size
    ):
        raise RuntimeError(
            "matched decomposition runtime changed after preregistration."
        )
    if str(pyscf_module.__version__) != "2.13.1":
        raise RuntimeError("matched decomposition requires PySCF 2.13.1.")
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        if os.environ.get(variable) != str(threads):
            raise RuntimeError(f"{variable} must equal the preregistered thread count.")
    for case in preregistration.get("cases", []):
        if not isinstance(case, dict):
            raise RuntimeError("preregistration case must be an object.")
        for key, role in (
            ("mol2", "MOL2"),
            ("projection_result", "projection result"),
            ("gas_checkpoint", "gas checkpoint"),
            ("gas_ledger", "gas ledger"),
            ("pcm_input", "parsed PCM input"),
            ("frozen_surface", "frozen PCMSolver surface"),
            ("frozen_gas_mep", "frozen gas MEP"),
        ):
            _validated_file(
                asset_root, case.get(key), role=f"{case.get('case_id')} {role}"
            )
    return preregistration


def _make_rks(dft_module, molecule, *, protocol: dict[str, object], verbose: int):
    mean_field = dft_module.RKS(
        molecule, xc=str(protocol["pyscf_xc_token"])
    ).density_fit()
    mean_field.grids.level = int(protocol["semilocal_grid_level"])
    elements = sorted({molecule.atom_symbol(index) for index in range(molecule.natm)})
    nlc_grid = tuple(protocol["nonlocal_atom_grid"])
    mean_field.nlcgrids.atom_grid = {
        symbol: (int(nlc_grid[0]), int(nlc_grid[1])) for symbol in elements
    }
    mean_field.nlcgrids.prune = dft_module.gen_grid.sg1_prune
    mean_field.conv_tol = float(protocol["scf_energy_tolerance_hartree"])
    mean_field.conv_tol_grad = float(protocol["scf_gradient_tolerance"])
    mean_field.max_cycle = int(protocol["maximum_scf_cycles"])
    mean_field.max_memory = int(protocol["maximum_memory_mb"])
    mean_field.verbose = verbose
    return mean_field


def _orbital_gradient_inf(mean_field) -> float:
    gradient = mean_field.get_grad(
        mean_field.mo_coeff,
        mean_field.mo_occ,
        mean_field.get_fock(),
    )
    return float(np.max(np.abs(np.asarray(gradient, dtype=np.float64))))


def _electrostatic_multipoles(molecule, density: np.ndarray) -> dict[str, object]:
    charges = np.asarray(molecule.atom_charges(), dtype=np.float64)
    coordinates = np.asarray(molecule.atom_coords(), dtype=np.float64)
    origin = np.einsum("a,ax->x", charges, coordinates) / float(np.sum(charges))
    relative = coordinates - origin[None, :]
    overlap = molecule.intor_symmetric("int1e_ovlp")
    electron_count = float(np.einsum("ij,ji->", density, overlap))
    total_charge = float(np.sum(charges) - electron_count)
    first = np.asarray(molecule.intor_symmetric("int1e_r", comp=3), dtype=float)
    electronic_first = np.einsum("xij,ji->x", first, density)
    electronic_first_relative = electronic_first - origin * electron_count
    dipole = np.einsum("a,ax->x", charges, relative) - electronic_first_relative
    second = np.asarray(
        molecule.intor_symmetric("int1e_rr", comp=9), dtype=float
    ).reshape(3, 3, molecule.nao_nr(), molecule.nao_nr())
    electronic_second = np.einsum("xyij,ji->xy", second, density)
    electronic_second_relative = (
        electronic_second
        - np.outer(origin, electronic_first)
        - np.outer(electronic_first, origin)
        + electron_count * np.outer(origin, origin)
    )
    nuclear_second = np.einsum("a,ax,ay->xy", charges, relative, relative)
    charge_second = nuclear_second - electronic_second_relative
    quadrupole = 3.0 * charge_second - np.trace(charge_second) * np.eye(3)
    quadrupole = 0.5 * (quadrupole + quadrupole.T)
    return {
        "origin_bohr": origin.tolist(),
        "total_charge_e": total_charge,
        "molecular_dipole_e_bohr": dipole.tolist(),
        "traceless_quadrupole_e_bohr2": quadrupole.tolist(),
        "traceless_quadrupole_trace_abs_e_bohr2": abs(float(np.trace(quadrupole))),
    }


def _runtime_manifest(pyscf_module, lib_module, *, threads: int) -> dict[str, object]:
    try:
        import importlib.metadata

        libxc_version = importlib.metadata.version("pyscf")
    except Exception:
        libxc_version = None
    return {
        "python": platform.python_version(),
        "python_executable_sha256": sha256_file(Path(sys.executable).resolve()),
        "numpy": np.__version__,
        "pyscf": str(pyscf_module.__version__),
        "pyscf_init_sha256": sha256_file(Path(pyscf_module.__file__).resolve()),
        "pyscf_distribution_version": libxc_version,
        "threads": int(lib_module.num_threads()),
        "platform": platform.platform(),
        "machine": platform.machine(),
    }


def _run_case(
    *,
    case_id: str,
    case_asset: dict[str, object],
    preregistration: dict[str, object],
    preregistration_sha256: str,
    asset_root: Path,
    pcmsolver_library: Path,
    output_dir: Path,
    dft_module,
    lib_module,
    PCMSolverSession,
    runtime_manifest: dict[str, object],
    source_hashes: dict[str, str],
    max_memory_mb: int,
    batch_size: int,
    verbose: int,
) -> tuple[MatchedElectrostaticReferenceCase, dict[str, object]]:
    case_started = time.perf_counter()
    projection_result = _validated_file(
        asset_root, case_asset["projection_result"], role=f"{case_id} projection result"
    )
    projection = _load_json(projection_result, name=f"{case_id} projection result")
    inputs = projection.get("inputs")
    if not isinstance(inputs, dict) or inputs.get("compound_id") != case_id:
        raise RuntimeError(f"{case_id} projection input identity is invalid.")
    gas_checkpoint = _validated_file(
        asset_root, case_asset["gas_checkpoint"], role=f"{case_id} gas checkpoint"
    )
    gas_ledger = _validated_file(
        asset_root, case_asset["gas_ledger"], role=f"{case_id} gas ledger"
    )
    pcm_input = _validated_file(
        asset_root, case_asset["pcm_input"], role=f"{case_id} PCM input"
    )
    frozen_surface = _validated_file(
        asset_root, case_asset["frozen_surface"], role=f"{case_id} frozen surface"
    )
    frozen_gas_mep = _validated_file(
        asset_root, case_asset["frozen_gas_mep"], role=f"{case_id} frozen gas MEP"
    )
    mol2 = _validated_file(asset_root, case_asset["mol2"], role=f"{case_id} MOL2")
    gas_record = _load_json(gas_ledger, name=f"{case_id} gas ledger")
    protocol = preregistration["reference_protocol"]
    gates = preregistration["numerical_gates"]
    if not isinstance(protocol, dict) or not isinstance(gates, dict):
        raise RuntimeError("preregistration reference protocol/gates are invalid.")

    molecule = lib_module.chkfile.load_mol(str(gas_checkpoint))
    if molecule.charge != int(protocol["charge"]) or molecule.spin != int(
        protocol["spin"]
    ):
        raise RuntimeError(f"{case_id} gas checkpoint charge/spin changed.")
    if str(molecule.basis).lower() != str(protocol["basis"]).lower():
        raise RuntimeError(f"{case_id} gas checkpoint basis changed.")
    mo_coeff = lib_module.chkfile.load(str(gas_checkpoint), "scf/mo_coeff")
    mo_occ = lib_module.chkfile.load(str(gas_checkpoint), "scf/mo_occ")
    overlap = molecule.intor_symmetric("int1e_ovlp")
    vacuum_density, vacuum_density_record = closed_shell_density_from_orbitals(
        mo_coeff, mo_occ, overlap
    )
    checkpoint_energy = float(lib_module.chkfile.load(str(gas_checkpoint), "scf/e_tot"))
    configured = _make_rks(dft_module, molecule, protocol=protocol, verbose=verbose)
    replay_energy = float(configured.energy_tot(vacuum_density))
    gas_replay_error = abs(replay_energy - checkpoint_energy)
    if gas_replay_error > float(gates["gas_checkpoint_replay_abs_hartree_max"]):
        raise RuntimeError(f"{case_id} gas checkpoint functional replay failed.")

    case_dir = output_dir / case_id
    case_dir.mkdir(parents=False, exist_ok=False)
    pcm_checkpoint = case_dir / "pcm.chk"
    surface_output = case_dir / "surface.npz"
    state_output = case_dir / "pcm-state.npz"
    cache_path = case_dir / ".ao-rinv-cache.npy"
    previous_cwd = Path.cwd()
    cycle_count = 0

    def count_cycle(_environment):
        nonlocal cycle_count
        cycle_count += 1

    try:
        with tempfile.TemporaryDirectory(prefix=f"route2-qm-pcm-{case_id}-") as work:
            os.chdir(work)
            with PCMSolverSession(
                np.asarray(molecule.atom_charges(), dtype=float),
                np.asarray(molecule.atom_coords(), dtype=float),
                pcm_input,
                library_path=pcmsolver_library,
            ) as session:
                if not session.response_operator_is_symmetric:
                    raise RuntimeError(f"{case_id} PCM input lacks MATRIXSYMM=True.")
                points = np.asarray(session.cavity_centers_bohr, dtype=np.float64)
                areas = np.asarray(session.cavity_areas_bohr2, dtype=np.float64)
                with np.load(frozen_surface) as frozen:
                    expected_points = np.asarray(
                        frozen["surface_points_bohr"], dtype=np.float64
                    )
                legacy_surface_drift = float(np.max(np.abs(points - expected_points)))
                if legacy_surface_drift > float(
                    gates["legacy_surface_diagnostic_abs_bohr_max"]
                ):
                    raise RuntimeError(
                        f"{case_id} legacy surface continuity diagnostic failed."
                    )
                symmetry = response_symmetry_defect(session, surface_size=len(points))
                if symmetry["relative_defect"] > float(
                    gates["response_symmetry_relative_max"]
                ):
                    raise RuntimeError(f"{case_id} PCMSolver symmetry gate failed.")
                cache_started = time.perf_counter()
                cache = AOInverseDistanceIntegralCache(
                    molecule,
                    points,
                    path=cache_path,
                    batch_size=batch_size,
                )
                cache_elapsed = time.perf_counter() - cache_started
                solvent = PCMSolverSCFSolvent(molecule, session, cache)
                pcm_mean_field = attach_pcmsolver_to_scf(configured, solvent)
                pcm_mean_field.chkfile = str(pcm_checkpoint)
                pcm_mean_field.callback = count_cycle
                scf_started = time.perf_counter()
                pcm_energy = float(pcm_mean_field.kernel(dm0=vacuum_density))
                scf_elapsed = time.perf_counter() - scf_started
                if not pcm_mean_field.converged:
                    raise RuntimeError(f"{case_id} QM/PCMSolver SCF did not converge.")
                pcm_density = np.asarray(pcm_mean_field.make_rdm1(), dtype=np.float64)
                pcm_density = np.ascontiguousarray(0.5 * (pcm_density + pcm_density.T))
                electron_count = float(np.einsum("ij,ji->", pcm_density, overlap))
                expected_electrons = float(vacuum_density_record["electron_count_e"])
                electron_error = abs(electron_count - expected_electrons)
                if electron_error > float(gates["electron_count_abs_e_max"]):
                    raise RuntimeError(f"{case_id} PCM density electron count failed.")
                orbital_gradient = _orbital_gradient_inf(pcm_mean_field)
                if orbital_gradient > float(gates["orbital_gradient_inf_max"]):
                    raise RuntimeError(f"{case_id} PCM orbital gradient failed.")
                vacuum_view = pcm_mean_field.undo_solvent()
                polarized_vacuum_energy = float(vacuum_view.energy_tot(dm=pcm_density))
                response = solvent.evaluate_density(pcm_density)
                pcm_replay_energy = float(pcm_mean_field.energy_tot(dm=pcm_density))
                if response.half_coupling_residual_hartree > float(
                    gates["half_coupling_abs_hartree_max"]
                ):
                    raise RuntimeError(f"{case_id} half-coupling identity failed.")
                density_direction = pcm_density - vacuum_density
                if float(np.linalg.norm(density_direction)) <= 1.0e-12:
                    raise RuntimeError(f"{case_id} PCM density did not change.")
                derivative = solvent_energy_directional_derivative_error(
                    solvent,
                    density=pcm_density,
                    direction=density_direction,
                    step=float(gates["solvent_energy_fd_step"]),
                )
                if derivative["absolute_error_hartree"] > float(
                    gates["solvent_energy_fd_abs_hartree_max"]
                ):
                    raise RuntimeError(
                        f"{case_id} solvent energy/Fock derivative failed."
                    )
                response = solvent.evaluate_density(pcm_density)
                decomposition_closure = abs(
                    (pcm_energy - checkpoint_energy)
                    - (polarized_vacuum_energy - checkpoint_energy)
                    - response.polarization_energy_hartree
                )
                replay_closure = abs(
                    pcm_replay_energy
                    - polarized_vacuum_energy
                    - response.polarization_energy_hartree
                )
                if max(decomposition_closure, replay_closure) > float(
                    gates["decomposition_closure_abs_hartree_max"]
                ):
                    raise RuntimeError(f"{case_id} energy decomposition did not close.")
                frozen_gas = np.load(frozen_gas_mep)
                frozen_numbers = np.asarray(frozen_gas["atomic_numbers"], dtype=int)
                frozen_positions = np.asarray(
                    frozen_gas["atom_positions_angstrom"], dtype=float
                )
                checkpoint_positions = np.asarray(molecule.atom_coords(), dtype=float)
                checkpoint_positions *= float(protocol["bohr_to_angstrom"])
                if not np.array_equal(
                    frozen_numbers, np.asarray(molecule.atom_charges(), dtype=int)
                ) or float(
                    np.max(np.abs(frozen_positions - checkpoint_positions))
                ) > float(
                    gates["geometry_replay_abs_angstrom_max"]
                ):
                    raise RuntimeError(f"{case_id} frozen gas MEP geometry changed.")

                np.savez(
                    surface_output,
                    surface_points_bohr=points,
                    surface_areas_bohr2=areas,
                    surface_points_sha256=np.asarray(array_sha256(points)),
                    surface_areas_sha256=np.asarray(array_sha256(areas)),
                )
                multipoles = _electrostatic_multipoles(molecule, pcm_density)
                np.savez(
                    state_output,
                    atomic_numbers=np.asarray(molecule.atom_charges(), dtype=int),
                    atom_positions_bohr=np.asarray(molecule.atom_coords(), dtype=float),
                    total_surface_mep_hartree_per_e=(
                        response.total_surface_mep_hartree_per_e
                    ),
                    apparent_surface_charge_e=response.apparent_surface_charge_e,
                    ao_density_matrix=pcm_density,
                    polarization_energy_hartree=np.asarray(
                        response.polarization_energy_hartree
                    ),
                    density_sha256=np.asarray(array_sha256(pcm_density)),
                    boundary_rhs_sha256=np.asarray(
                        array_sha256(response.total_surface_mep_hartree_per_e)
                    ),
                    multipoles_json=np.asarray(
                        json.dumps(multipoles, sort_keys=True, separators=(",", ":"))
                    ),
                )
                cache.close()
    finally:
        os.chdir(previous_cwd)
        if cache_path.exists():
            cache_path.unlink()

    pcm_coeff = lib_module.chkfile.load(str(pcm_checkpoint), "scf/mo_coeff")
    pcm_occ = lib_module.chkfile.load(str(pcm_checkpoint), "scf/mo_occ")
    checkpoint_pcm_density, pcm_checkpoint_density_record = (
        closed_shell_density_from_orbitals(pcm_coeff, pcm_occ, overlap)
    )
    pcm_checkpoint_binding = float(np.max(np.abs(checkpoint_pcm_density - pcm_density)))
    if pcm_checkpoint_binding > float(gates["checkpoint_density_binding_inf_max"]):
        raise RuntimeError(f"{case_id} PCM checkpoint density binding failed.")

    geometry_payload = {
        "atomic_numbers": np.asarray(molecule.atom_charges(), dtype=int).tolist(),
        "positions_bohr": np.asarray(molecule.atom_coords(), dtype=float).tolist(),
        "charge": int(molecule.charge),
        "spin": int(molecule.spin),
    }
    geometry_sha = canonical_json_sha256(geometry_payload)
    topology_sha = canonical_json_sha256(
        {
            "points_sha256": array_sha256(points),
            "areas_sha256": array_sha256(areas),
            "surface_count": len(points),
        }
    )
    continuum_protocol_sha = canonical_json_sha256(
        preregistration["continuum_protocol"]
    )
    continuum_provenance_sha = canonical_json_sha256(
        {
            "library_sha256": sha256_file(pcmsolver_library),
            "source_sha256": {
                path: digest
                for path, digest in source_hashes.items()
                if path.endswith("pcmsolver.py")
            },
            "equation_id": CONTINUUM_EQUATION_ID,
        }
    )
    continuum_configuration_sha = canonical_json_sha256(
        {
            "pcm_input_sha256": sha256_file(pcm_input),
            "topology_sha256": topology_sha,
            "geometry_sha256": geometry_sha,
            "dielectric": float(preregistration["continuum_protocol"]["dielectric"]),
        }
    )
    reference_protocol_sha = canonical_json_sha256(
        preregistration["reference_protocol"]
    )
    runtime_sha = canonical_json_sha256(runtime_manifest)
    raw_reference_identity = {
        "gas_checkpoint_sha256": sha256_file(gas_checkpoint),
        "pcm_density_sha256": array_sha256(pcm_density),
        "surface_points_sha256": array_sha256(points),
        "surface_areas_sha256": array_sha256(areas),
        "boundary_rhs_sha256": array_sha256(response.total_surface_mep_hartree_per_e),
        "asc_sha256": array_sha256(response.apparent_surface_charge_e),
        "vacuum_ground_state_energy_hartree": checkpoint_energy,
        "polarized_density_vacuum_energy_hartree": polarized_vacuum_energy,
        "pcm_electrostatic_total_energy_hartree": pcm_energy,
        "reference_protocol_sha256": reference_protocol_sha,
        "continuum_configuration_sha256": continuum_configuration_sha,
    }
    reference_artifact_sha = canonical_json_sha256(raw_reference_identity)
    reference_case = MatchedElectrostaticReferenceCase(
        case_id=case_id,
        molecule_group_sha256=sha256_file(mol2),
        geometry_sha256=geometry_sha,
        reference_artifact_sha256=reference_artifact_sha,
        reference_method_id=REFERENCE_METHOD_ID,
        reference_protocol_sha256=reference_protocol_sha,
        qm_runtime_manifest_sha256=runtime_sha,
        total_charge=int(molecule.charge),
        spin_multiplicity=int(molecule.spin) + 1,
        vacuum_density_sha256=array_sha256(vacuum_density),
        pcm_density_sha256=array_sha256(pcm_density),
        reference_boundary_rhs_sha256=array_sha256(
            response.total_surface_mep_hartree_per_e
        ),
        continuum_equation_id=CONTINUUM_EQUATION_ID,
        continuum_protocol_sha256=continuum_protocol_sha,
        continuum_configuration_sha256=continuum_configuration_sha,
        continuum_provenance_sha256=continuum_provenance_sha,
        topology_sha256=topology_sha,
        cavity_profile_id=CAVITY_PROFILE_ID,
        dielectric=float(preregistration["continuum_protocol"]["dielectric"]),
        vacuum_ground_state_energy_eV=checkpoint_energy * HARTREE_TO_EV,
        polarized_density_vacuum_energy_eV=(polarized_vacuum_energy * HARTREE_TO_EV),
        pcm_electrostatic_total_energy_eV=pcm_energy * HARTREE_TO_EV,
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "pass-matched-electrostatic-reference-case",
        "case_id": case_id,
        "capabilities": NO_CAPABILITIES,
        "claim_boundary": (
            "Fixed-geometry electrostatic-only QM/PCMSolver component reference. "
            "No ML source, operational ledger admission, CDS, standard-state, "
            "experimental solvation, force, Hessian, frequency, or MD claim."
        ),
        "preregistration": {
            "path": PREREGISTRATION_RELATIVE_PATH,
            "sha256": preregistration_sha256,
        },
        "inputs": {
            "mol2": _file_record(mol2, role="frozen molecular structure"),
            "projection_result": _file_record(
                projection_result, role="frozen projection result"
            ),
            "gas_checkpoint": _file_record(
                gas_checkpoint, role="frozen gas PySCF checkpoint"
            ),
            "gas_ledger": _file_record(gas_ledger, role="frozen gas ledger"),
            "pcm_input": _file_record(pcm_input, role="frozen parsed PCMSolver input"),
            "frozen_surface": _file_record(
                frozen_surface,
                role="legacy ASE-coordinate source-gate surface diagnostic",
            ),
            "frozen_gas_mep": _file_record(
                frozen_gas_mep, role="frozen gas-density surface MEP"
            ),
            "gas_ledger_status": gas_record.get("status"),
        },
        "outputs": {
            "pcm_checkpoint": _file_record(
                pcm_checkpoint, role="self-consistent QM/PCMSolver checkpoint"
            ),
            "surface": _file_record(surface_output, role="replayed PCMSolver surface"),
            "pcm_state": _file_record(state_output, role="PCM-density boundary state"),
        },
        "reference_case": reference_case.as_dict(),
        "energies_hartree": {
            "vacuum_ground_state": checkpoint_energy,
            "vacuum_functional_replay": replay_energy,
            "polarized_density_in_vacuum_functional": polarized_vacuum_energy,
            "pcm_electrostatic_total": pcm_energy,
            "pcm_electrostatic_total_replay": pcm_replay_energy,
            "solute_distortion": polarized_vacuum_energy - checkpoint_energy,
            "continuum_stabilization": response.polarization_energy_hartree,
            "total_electrostatic_solvation": pcm_energy - checkpoint_energy,
        },
        "diagnostics": {
            "gas_checkpoint_replay_abs_hartree": gas_replay_error,
            "legacy_surface_diagnostic_abs_bohr": legacy_surface_drift,
            "response_symmetry": symmetry,
            "half_coupling_abs_hartree": response.half_coupling_residual_hartree,
            "solvent_energy_directional_derivative": derivative,
            "decomposition_closure_abs_hartree": decomposition_closure,
            "pcm_energy_replay_closure_abs_hartree": replay_closure,
            "pcm_orbital_gradient_inf": orbital_gradient,
            "pcm_density_electron_count_e": electron_count,
            "pcm_density_electron_count_abs_error_e": electron_error,
            "pcm_checkpoint_density_binding_inf": pcm_checkpoint_binding,
            "vacuum_density": vacuum_density_record,
            "pcm_checkpoint_density": pcm_checkpoint_density_record,
            "pcm_density_change_frobenius": float(
                np.linalg.norm(pcm_density - vacuum_density)
            ),
            "pcm_multipoles": multipoles,
            "surface_count": len(points),
            "ao_count": int(molecule.nao_nr()),
            "integral_cache": {
                "storage": "temporary-npy-memmap",
                "bytes": int(len(points) * molecule.nao_nr() ** 2 * 8),
                "batch_size": batch_size,
                "build_elapsed_seconds": cache_elapsed,
            },
            "scf_cycles": cycle_count,
            "scf_elapsed_seconds": scf_elapsed,
            "case_elapsed_seconds": time.perf_counter() - case_started,
        },
        "raw_reference_identity": raw_reference_identity,
    }
    case_json = case_dir / "case.json"
    file_record = write_external_json_artifact(
        RepositorySnapshot.capture(Path(__file__).resolve().parents[2]),
        case_json,
        payload,
    )
    return reference_case, {
        "case_id": case_id,
        "case_artifact": file_record,
        "reference_case": reference_case.as_dict(),
    }


def main() -> int:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    snapshot = RepositorySnapshot.capture(repo_root)
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    try:
        output_dir.relative_to(snapshot.root)
    except ValueError:
        pass
    else:
        raise RuntimeError("reference output directory must be outside the checkout.")
    asset_root = args.asset_root.expanduser().resolve(strict=True)
    pcmsolver_library = args.pcmsolver_library.expanduser().resolve(strict=True)
    preregistration_path = (
        snapshot.root / PREREGISTRATION_RELATIVE_PATH
        if args.preregistration is None
        else args.preregistration.expanduser()
    ).resolve(strict=True)
    inherited_path = (snapshot.root / INHERITED_PREREGISTRATION_RELATIVE_PATH).resolve(
        strict=True
    )

    try:
        import pyscf
        from pyscf import dft, lib
        from maple.function.calculator.extra_correction.implicit.pcmsolver import (
            PCMSolverSession,
        )
    except ImportError as exc:
        raise RuntimeError(
            "matched reference capture requires PySCF 2.13.1 and PCMSolver binding."
        ) from exc

    preregistration = _validate_preregistration(
        preregistration_path=preregistration_path,
        inherited_path=inherited_path,
        snapshot=snapshot,
        asset_root=asset_root,
        pcmsolver_library=pcmsolver_library,
        pyscf_module=pyscf,
        threads=args.threads,
        max_memory_mb=args.max_memory_mb,
        batch_size=args.integral_batch_size,
    )
    case_ids = tuple(preregistration["case_ids"])
    requested = tuple(args.case_id) if args.case_id else case_ids
    if len(requested) != len(set(requested)) or not set(requested).issubset(case_ids):
        raise RuntimeError("requested cases must be a unique preregistered subset.")
    output_dir.mkdir(parents=True, exist_ok=False)
    lib.num_threads(args.threads)
    runtime_manifest = _runtime_manifest(pyscf, lib, threads=args.threads)
    source_paths = collect_loaded_repository_sources(
        snapshot.root, required_paths=REQUIRED_SOURCE_PATHS
    )
    source_hashes = committed_source_hashes(snapshot, source_paths)
    started = time.perf_counter()
    cases = []
    case_records = []
    for case_id in requested:
        reference_case, record = _run_case(
            case_id=case_id,
            case_asset=_case_asset_record(
                preregistration=preregistration, case_id=case_id
            ),
            preregistration=preregistration,
            preregistration_sha256=sha256_file(preregistration_path),
            asset_root=asset_root,
            pcmsolver_library=pcmsolver_library,
            output_dir=output_dir,
            dft_module=dft,
            lib_module=lib,
            PCMSolverSession=PCMSolverSession,
            runtime_manifest=runtime_manifest,
            source_hashes=source_hashes,
            max_memory_mb=args.max_memory_mb,
            batch_size=args.integral_batch_size,
            verbose=args.verbose,
        )
        cases.append(reference_case)
        case_records.append(record)
    panel = None
    if set(requested) == set(case_ids) and len(requested) == len(case_ids):
        panel = MatchedElectrostaticReferencePanel(
            panel_id="route2-matched-qm-pcmsolver-electrostatic-four-v1",
            preregistration_sha256=sha256_file(preregistration_path),
            expected_case_ids=tuple(sorted(case_ids)),
            cases=tuple(cases),
        )
    scientific_payload = {
        "requested_case_ids": sorted(requested),
        "reference_cases": [
            case.as_dict() for case in sorted(cases, key=lambda value: value.case_id)
        ],
        "reference_panel": None if panel is None else panel.as_dict(),
    }
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": (
            "pass-complete-matched-electrostatic-reference-panel"
            if panel is not None
            else "pass-partial-reference-capture-not-panel-admission"
        ),
        "capabilities": NO_CAPABILITIES,
        "claim_boundary": (
            "Source-independent matched electrostatic component reference only. "
            "A subset run is an implementation canary, not panel evidence. No "
            "operational ledger or E/F/H/V/M capability is admitted."
        ),
        "repository": snapshot.as_dict(),
        "preregistration": {
            "path": PREREGISTRATION_RELATIVE_PATH,
            "sha256": sha256_file(preregistration_path),
        },
        "runtime": runtime_manifest,
        "pcmsolver_library": _file_record(
            pcmsolver_library, role="pinned PCMSolver shared library"
        ),
        "source_sha256": source_hashes,
        "requested_case_ids": list(requested),
        "full_preregistered_case_ids": list(case_ids),
        "panel_complete": panel is not None,
        "case_records": case_records,
        "scientific_payload": scientific_payload,
        "scientific_payload_sha256": canonical_json_sha256(scientific_payload),
        "elapsed_seconds": time.perf_counter() - started,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    snapshot.assert_unchanged()
    write_external_json_artifact(snapshot, output_dir / "result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
