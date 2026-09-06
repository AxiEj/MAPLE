#!/usr/bin/env python3
"""Compare current hybrid boundary potentials with matched finite-field QM.

This is a one-case causal mechanism audit, not a solvation benchmark.  It uses
the exact frozen acetone PCMSolver cavity twice:

* vacuum QM MEP versus the MACE-POLAR zero-field point-q/p permanent source;
* finite-field QM induced MEP versus the MACE-MDP-alpha canonical ADT response
  for three Cartesian native-gradient directions.

No charge fit, source scale, projector, experimental solvation target, CDS
term, or capability flag is read or produced.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import tempfile
import time

import numpy as np

SOURCE_ROOT = Path(__file__).resolve().parents[2]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from maple.solvation.release.qm_boundary_response import (  # noqa: E402
    central_difference,
    compare_boundary_response,
    reference_uncertainty,
)

EXPECTED_ARTIFACT = "route2-mace-mdp-polar-matched-qm-boundary-response-prereg-v1"
RESULT_ARTIFACT = "route2-mace-mdp-polar-matched-qm-boundary-response-v1"
NO_CAPABILITIES = {tier: False for tier in ("E", "F", "H", "V", "M")}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(value: object) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(repr(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_json(path: Path, *, name: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot load {name} at {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"{name} must contain one JSON object.")
    return payload


def _validated_file(record: object, *, name: str) -> Path:
    if not isinstance(record, dict):
        raise RuntimeError(f"{name} file record is missing.")
    path_raw = record.get("path")
    expected = record.get("sha256")
    expected_bytes = record.get("bytes")
    if not isinstance(path_raw, str) or not isinstance(expected, str):
        raise RuntimeError(f"{name} file record is invalid.")
    path = Path(path_raw).expanduser().resolve(strict=True)
    if (
        not path.is_file()
        or path.stat().st_size != expected_bytes
        or _sha256_file(path) != expected
    ):
        raise RuntimeError(f"{name} file identity changed after preregistration.")
    return path


def _runtime_identity() -> dict[str, object]:
    import pyscf
    import scipy
    import torch

    executable = Path(sys.executable).resolve(strict=True)
    return {
        "python_version": platform.python_version(),
        "python_executable": {
            "path": str(executable),
            "bytes": executable.stat().st_size,
            "sha256": _sha256_file(executable),
        },
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "pyscf_version": pyscf.__version__,
        "pyscf_init": {
            "path": str(Path(pyscf.__file__).resolve(strict=True)),
            "bytes": Path(pyscf.__file__).resolve(strict=True).stat().st_size,
            "sha256": _sha256_file(Path(pyscf.__file__).resolve(strict=True)),
        },
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available_at_lock": bool(torch.cuda.is_available()),
    }


def _validate_preregistration(path: Path) -> dict[str, object]:
    prereg = _load_json(path, name="preregistration")
    if (
        prereg.get("artifact") != EXPECTED_ARTIFACT
        or prereg.get("schema_version") != 1
        or prereg.get("status")
        != "frozen-before-first-qm-induced-boundary-response-execution"
    ):
        raise RuntimeError("preregistration identity is invalid.")
    expected_hash = prereg.get("preregistration_sha256")
    unsigned = dict(prereg)
    unsigned.pop("preregistration_sha256", None)
    if expected_hash != _canonical_sha256(unsigned):
        raise RuntimeError("preregistration self hash is invalid.")
    if prereg.get("source_root") != str(SOURCE_ROOT):
        raise RuntimeError("preregistration belongs to a different source root.")
    sources = prereg.get("source_files_sha256")
    if not isinstance(sources, dict) or not sources:
        raise RuntimeError("preregistration omits source hashes.")
    for relative, expected in sources.items():
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise RuntimeError("preregistered source hash record is invalid.")
        source = SOURCE_ROOT / relative
        if not source.is_file() or _sha256_file(source) != expected:
            raise RuntimeError(f"source changed after preregistration: {relative}")
    runtime = prereg.get("runtime")
    if not isinstance(runtime, dict):
        raise RuntimeError("preregistration omits runtime identity.")
    current = _runtime_identity()
    for key in (
        "python_version",
        "python_executable",
        "numpy_version",
        "scipy_version",
        "pyscf_version",
        "pyscf_init",
        "torch_version",
        "torch_cuda_version",
        "cuda_available_at_lock",
    ):
        if runtime.get(key) != current[key]:
            raise RuntimeError(f"runtime changed after preregistration: {key}")
    if runtime.get("pyscf_version") != "2.13.1":
        raise RuntimeError("the audit requires PySCF 2.13.1.")
    threads = runtime.get("threads")
    if not isinstance(threads, int) or threads < 1:
        raise RuntimeError("preregistered thread count is invalid.")
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        if os.environ.get(variable) != str(threads):
            raise RuntimeError(f"{variable} must equal {threads}.")
    pcmsolver = _validated_file(runtime.get("pcmsolver_library"), name="PCMSolver")
    if pcmsolver != Path(str(runtime["pcmsolver_library"]["path"])):
        raise RuntimeError("PCMSolver path normalization changed.")
    inputs = prereg.get("inputs")
    if not isinstance(inputs, dict):
        raise RuntimeError("preregistration omits inputs.")
    _validated_file(inputs.get("mace_mdp_checkpoint"), name="MACE-MDP checkpoint")
    _validated_file(inputs.get("mace_polar_checkpoint"), name="MACE-POLAR checkpoint")
    case = inputs.get("case")
    if not isinstance(case, dict) or case.get("case_id") != "mobley_3867265":
        raise RuntimeError("preregistered case identity is invalid.")
    files = case.get("files")
    if not isinstance(files, dict):
        raise RuntimeError("preregistered case files are missing.")
    expected_names = {
        "parent_preregistration",
        "projection_result",
        "mol2",
        "qm_checkpoint",
        "qm_ledger",
        "pcm_input",
        "frozen_surface",
        "frozen_qm_surface_mep",
    }
    if set(files) != expected_names:
        raise RuntimeError("preregistered case file contract changed.")
    for name, record in files.items():
        _validated_file(record, name=name)
    decision = prereg.get("decision_contract")
    if not isinstance(decision, dict) or not all(
        decision.get(name) is True
        for name in (
            "not_a_fit",
            "no_projector_selected",
            "no_source_rescaling",
            "no_solvation_target_read",
            "no_capability_admission",
        )
    ):
        raise RuntimeError("preregistered no-fit claim boundary changed.")
    return prereg


def _make_rks(dft, molecule, *, protocol: dict[str, object], verbose: int = 0):
    mean_field = dft.RKS(molecule, xc="wb97m-v").density_fit()
    mean_field.grids.level = int(protocol["semilocal_grid_level"])
    elements = sorted({molecule.atom_symbol(index) for index in range(molecule.natm)})
    nlc = tuple(protocol["nonlocal_atom_grid"])
    mean_field.nlcgrids.atom_grid = {
        symbol: (int(nlc[0]), int(nlc[1])) for symbol in elements
    }
    mean_field.nlcgrids.prune = dft.gen_grid.sg1_prune
    mean_field.conv_tol = float(protocol["scf_energy_tolerance_hartree"])
    mean_field.conv_tol_grad = float(protocol["scf_gradient_tolerance"])
    mean_field.max_cycle = int(protocol["maximum_scf_cycles"])
    mean_field.verbose = verbose
    return mean_field


def _closed_shell_density(lib, checkpoint: Path, molecule) -> np.ndarray:
    coefficients = np.asarray(lib.chkfile.load(str(checkpoint), "scf/mo_coeff"))
    occupations = np.asarray(lib.chkfile.load(str(checkpoint), "scf/mo_occ"))
    if coefficients.ndim != 2 or occupations.ndim != 1:
        raise RuntimeError("checkpoint orbitals have invalid shape.")
    density = np.einsum(
        "pi,i,qi->pq",
        coefficients,
        occupations,
        coefficients,
        optimize=True,
    )
    density = np.ascontiguousarray(0.5 * (density + density.T), dtype=np.float64)
    overlap = molecule.intor_symmetric("int1e_ovlp")
    electrons = float(np.einsum("ij,ji->", density, overlap))
    if abs(electrons - float(np.sum(occupations))) > 1.0e-8:
        raise RuntimeError("checkpoint density electron count is inconsistent.")
    return density


def _control_molecule(gto, primary, basis: str):
    atoms = [
        (primary.atom_symbol(index), tuple(primary.atom_coord(index)))
        for index in range(primary.natm)
    ]
    molecule = gto.Mole()
    molecule.atom = atoms
    molecule.unit = "Bohr"
    molecule.basis = basis
    molecule.charge = int(primary.charge)
    molecule.spin = int(primary.spin)
    molecule.verbose = 0
    molecule.build()
    return molecule


def _field_density(
    *,
    dft,
    molecule,
    protocol: dict[str, object],
    zero_density: np.ndarray,
    axis: int,
    gradient_volt_per_angstrom: float,
) -> tuple[np.ndarray, float, int]:
    mean_field = _make_rks(dft, molecule, protocol=protocol)
    base_hcore = np.asarray(mean_field.get_hcore(), dtype=np.float64)
    position_integrals = np.asarray(
        molecule.intor_symmetric("int1e_r", comp=3), dtype=np.float64
    )
    conversion = float(protocol["atomic_unit_electric_field_volt_per_angstrom"])
    gradient_au = float(gradient_volt_per_angstrom) / conversion
    perturbation = -gradient_au * position_integrals[axis]

    def get_hcore(_molecule=None):
        return base_hcore + perturbation

    mean_field.get_hcore = get_hcore
    cycles = 0

    def count_cycle(_environment):
        nonlocal cycles
        cycles += 1

    mean_field.callback = count_cycle
    energy = float(mean_field.kernel(dm0=zero_density))
    if not mean_field.converged:
        raise RuntimeError(
            f"finite-field SCF did not converge for axis {axis}, "
            f"gradient {gradient_volt_per_angstrom}."
        )
    density = np.asarray(mean_field.make_rdm1(), dtype=np.float64)
    density = np.ascontiguousarray(0.5 * (density + density.T))
    return density, energy, cycles


def _finite_field_axis(
    *,
    dft,
    molecule,
    protocol: dict[str, object],
    zero_density: np.ndarray,
    axis: int,
    step: float,
    integral_cache,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    positive, energy_plus, cycles_plus = _field_density(
        dft=dft,
        molecule=molecule,
        protocol=protocol,
        zero_density=zero_density,
        axis=axis,
        gradient_volt_per_angstrom=step,
    )
    negative, energy_minus, cycles_minus = _field_density(
        dft=dft,
        molecule=molecule,
        protocol=protocol,
        zero_density=zero_density,
        axis=axis,
        gradient_volt_per_angstrom=-step,
    )
    density_direction = central_difference(positive, negative, step=step)
    electronic = integral_cache.electronic_surface_potential(density_direction)
    surface_response = -np.asarray(electronic, dtype=np.float64)
    position_integrals = np.asarray(
        molecule.intor_symmetric("int1e_r", comp=3), dtype=np.float64
    )
    dipole_response = -np.einsum("xij,ji->x", position_integrals, density_direction)
    from ase.units import Bohr

    dipole_response = np.asarray(dipole_response * Bohr, dtype=np.float64)
    overlap = molecule.intor_symmetric("int1e_ovlp")
    electron_derivative = float(np.einsum("ij,ji->", density_direction, overlap))
    record = {
        "step_volt_per_angstrom": step,
        "energy_plus_hartree": energy_plus,
        "energy_minus_hartree": energy_minus,
        "cycles_plus": cycles_plus,
        "cycles_minus": cycles_minus,
        "density_direction_sha256": _array_sha256(density_direction),
        "electron_count_derivative_e_per_volt_per_angstrom": electron_derivative,
    }
    return surface_response, dipole_response, record


def _build_model(prereg: dict[str, object], atoms):
    import torch

    from maple.solvation.api.profiles import (
        MACE_POLAR_ANALYTIC_GAUSSIAN_MULTIPOLE_EVALUATOR_ID,
    )
    from maple.solvation.models import (
        MACEPolarOriginalSourceNativeFieldAdapter,
        build_mace_mdp_moment_adapter,
        build_mdp_polar_role_separated_adt_response,
        build_official_mace_polar_1_m_radial_gto_adapter,
    )

    runtime = prereg["runtime"]
    inputs = prereg["inputs"]
    mdp_checkpoint = _validated_file(
        inputs["mace_mdp_checkpoint"], name="MACE-MDP checkpoint"
    )
    polar_checkpoint = _validated_file(
        inputs["mace_polar_checkpoint"], name="MACE-POLAR checkpoint"
    )
    device = str(runtime["polar_device"])
    torch.manual_seed(20260820)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(20260820)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("preregistered CUDA MACE-POLAR runtime is unavailable.")
    mdp = build_mace_mdp_moment_adapter(
        checkpoint_path=mdp_checkpoint,
        device="cpu",
    )
    radial = build_official_mace_polar_1_m_radial_gto_adapter(
        checkpoint_path=polar_checkpoint,
        device=device,
        long_range_evaluator_profile=(
            MACE_POLAR_ANALYTIC_GAUSSIAN_MULTIPOLE_EVALUATOR_ID
        ),
    )
    base = MACEPolarOriginalSourceNativeFieldAdapter(radial)
    response = build_mdp_polar_role_separated_adt_response(
        mdp=mdp,
        base=base,
        source_root=SOURCE_ROOT,
    )
    return response, response.chart_for_geometry(atoms)


def _run(prereg: dict[str, object], output_dir: Path) -> dict[str, object]:
    from pyscf import dft, gto, lib
    from ase import Atoms
    from ase.units import Bohr

    from maple.function.calculator.extra_correction.implicit.gto_density import (
        point_multipole_potential,
    )
    from maple.function.calculator.extra_correction.implicit.pcmsolver import (
        PCMSolverSession,
    )
    from maple.solvation.reference import AOInverseDistanceIntegralCache

    started = time.perf_counter()
    runtime = prereg["runtime"]
    lib.num_threads(int(runtime["threads"]))
    protocol = prereg["reference_protocol"]
    case_files = prereg["inputs"]["case"]["files"]
    checkpoint = _validated_file(case_files["qm_checkpoint"], name="QM checkpoint")
    pcm_input = _validated_file(case_files["pcm_input"], name="PCM input")
    surface_path = _validated_file(case_files["frozen_surface"], name="frozen surface")
    qm_mep_path = _validated_file(
        case_files["frozen_qm_surface_mep"], name="frozen QM surface MEP"
    )
    pcmsolver_library = _validated_file(runtime["pcmsolver_library"], name="PCMSolver")
    primary = lib.chkfile.load_mol(str(checkpoint))
    if str(primary.basis).lower() != str(protocol["primary_basis"]).lower():
        raise RuntimeError("primary checkpoint basis changed.")
    vacuum_density = _closed_shell_density(lib, checkpoint, primary)
    replay = _make_rks(dft, primary, protocol=protocol)
    checkpoint_energy = float(lib.chkfile.load(str(checkpoint), "scf/e_tot"))
    replay_energy = float(replay.energy_tot(vacuum_density))
    if abs(checkpoint_energy - replay_energy) > 2.0e-9:
        raise RuntimeError("primary checkpoint functional replay failed.")
    control = _control_molecule(gto, primary, str(protocol["control_basis"]))
    control_zero = _make_rks(dft, control, protocol=protocol)
    control_energy = float(control_zero.kernel())
    if not control_zero.converged:
        raise RuntimeError("control-basis zero-field SCF did not converge.")
    control_density = np.asarray(control_zero.make_rdm1(), dtype=np.float64)
    control_density = np.ascontiguousarray(0.5 * (control_density + control_density.T))

    atoms = Atoms(
        numbers=np.asarray(primary.atom_charges(), dtype=np.int64),
        positions=np.asarray(primary.atom_coords(), dtype=np.float64) * Bohr,
        info={"charge": int(primary.charge), "multiplicity": int(primary.spin + 1)},
    )
    with np.load(qm_mep_path) as frozen_qm:
        qm_vacuum_mep = np.asarray(
            frozen_qm["surface_potential_hartree_per_e"], dtype=np.float64
        )
        frozen_positions = np.asarray(
            frozen_qm["atom_positions_angstrom"], dtype=np.float64
        )
        frozen_numbers = np.asarray(frozen_qm["atomic_numbers"], dtype=np.int64)
    if (
        not np.array_equal(frozen_numbers, atoms.numbers)
        or np.max(np.abs(frozen_positions - atoms.positions)) > 2.0e-10
    ):
        raise RuntimeError("frozen QM MEP geometry differs from the checkpoint.")
    with np.load(surface_path) as frozen_surface:
        expected_points = np.asarray(
            frozen_surface["surface_points_bohr"], dtype=np.float64
        )

    response_model, chart = _build_model(prereg, atoms)
    if chart.atom_count != len(atoms):
        raise RuntimeError("hybrid chart atom count changed.")

    work = output_dir / "work"
    work.mkdir()
    previous = Path.cwd()
    arrays: dict[str, np.ndarray] = {}
    try:
        with tempfile.TemporaryDirectory(prefix="route2-qm-boundary-") as temp:
            os.chdir(temp)
            with PCMSolverSession(
                np.asarray(primary.atom_charges(), dtype=np.float64),
                np.asarray(primary.atom_coords(), dtype=np.float64),
                pcm_input,
                library_path=pcmsolver_library,
            ) as session:
                points = np.asarray(session.cavity_centers_bohr, dtype=np.float64)
                areas = np.asarray(session.cavity_areas_bohr2, dtype=np.float64)
                if (
                    points.shape != expected_points.shape
                    or np.max(np.abs(points - expected_points)) > 1.0e-12
                ):
                    raise RuntimeError("PCMSolver cavity replay drifted.")
                response_counter = 0

                def apply_response(value: np.ndarray) -> np.ndarray:
                    nonlocal response_counter
                    response_counter += 1
                    return np.asarray(
                        session.compute_asc(
                            np.asarray(value, dtype=np.float64),
                            mep_label=f"MAPLE_QM_BOUNDARY_MEP_{response_counter}",
                            asc_label=f"MAPLE_QM_BOUNDARY_ASC_{response_counter}",
                        ),
                        dtype=np.float64,
                    )

                primary_cache = AOInverseDistanceIntegralCache(
                    primary,
                    points,
                    path=work / "primary-rinv.npy",
                    batch_size=16,
                )
                control_cache = AOInverseDistanceIntegralCache(
                    control,
                    points,
                    path=work / "control-rinv.npy",
                    batch_size=16,
                )
                permanent_mep = point_multipole_potential(
                    points,
                    atoms.positions,
                    chart.polar_zero_source4,
                )
                permanent_metrics = compare_boundary_response(
                    permanent_mep,
                    qm_vacuum_mep,
                    areas=areas,
                    apply_response=apply_response,
                ).as_dict()

                coarse, fine = [
                    float(value)
                    for value in protocol["field_gradient_steps_volt_per_angstrom"]
                ]
                if not coarse > fine > 0.0:
                    raise RuntimeError(
                        "finite-field steps are not strictly decreasing."
                    )
                axis_names = tuple(protocol["axes"])
                axes: list[dict[str, object]] = []
                primary_coarse = []
                primary_fine = []
                control_fine = []
                hybrid_adt = []
                primary_dipoles = []
                control_dipoles = []
                hybrid_dipoles = []
                for axis, axis_name in enumerate(axis_names):
                    coarse_mep, coarse_dipole, coarse_record = _finite_field_axis(
                        dft=dft,
                        molecule=primary,
                        protocol=protocol,
                        zero_density=vacuum_density,
                        axis=axis,
                        step=coarse,
                        integral_cache=primary_cache,
                    )
                    fine_mep, fine_dipole, fine_record = _finite_field_axis(
                        dft=dft,
                        molecule=primary,
                        protocol=protocol,
                        zero_density=vacuum_density,
                        axis=axis,
                        step=fine,
                        integral_cache=primary_cache,
                    )
                    control_mep, control_dipole, control_record = _finite_field_axis(
                        dft=dft,
                        molecule=control,
                        protocol=protocol,
                        zero_density=control_density,
                        axis=axis,
                        step=fine,
                        integral_cache=control_cache,
                    )
                    adt_dipoles = np.asarray(
                        chart.adt_atomic_dipole_jacobian_eangstrom[:, :, axis],
                        dtype=np.float64,
                    )
                    adt_mep = chart.adt_lift.tangent_potential(
                        points_bohr=points,
                        centers_bohr=atoms.positions / Bohr,
                        atomic_dipoles_ebohr=adt_dipoles / Bohr,
                    )
                    metrics = compare_boundary_response(
                        adt_mep,
                        fine_mep,
                        areas=areas,
                        apply_response=apply_response,
                    )
                    uncertainty = reference_uncertainty(
                        primary_fine=fine_mep,
                        primary_coarse=coarse_mep,
                        control_fine=control_mep,
                        apply_response=apply_response,
                    )
                    uncertainty_ratio = metrics.error_response_norm / max(
                        uncertainty.envelope_response_norm,
                        np.finfo(np.float64).tiny,
                    )
                    hybrid_dipole = np.sum(adt_dipoles, axis=0)
                    axes.append(
                        {
                            "axis": axis_name,
                            "primary_coarse": coarse_record,
                            "primary_fine": fine_record,
                            "control_fine": control_record,
                            "comparison": metrics.as_dict(),
                            "reference_uncertainty": uncertainty.as_dict(),
                            "hybrid_error_to_reference_uncertainty_ratio": (
                                uncertainty_ratio
                            ),
                            "resolved_beyond_reference_uncertainty": bool(
                                uncertainty_ratio > 5.0
                            ),
                            "qm_primary_dipole_derivative_eangstrom_per_volt_per_angstrom": (
                                fine_dipole.tolist()
                            ),
                            "qm_control_dipole_derivative_eangstrom_per_volt_per_angstrom": (
                                control_dipole.tolist()
                            ),
                            "hybrid_adt_dipole_derivative_eangstrom_per_volt_per_angstrom": (
                                hybrid_dipole.tolist()
                            ),
                            "hybrid_adt_dipole_relative_l2_error": float(
                                np.linalg.norm(hybrid_dipole - fine_dipole)
                                / max(
                                    np.linalg.norm(fine_dipole),
                                    np.finfo(np.float64).tiny,
                                )
                            ),
                        }
                    )
                    primary_coarse.append(coarse_mep)
                    primary_fine.append(fine_mep)
                    control_fine.append(control_mep)
                    hybrid_adt.append(adt_mep)
                    primary_dipoles.append(fine_dipole)
                    control_dipoles.append(control_dipole)
                    hybrid_dipoles.append(hybrid_dipole)

                arrays = {
                    "surface_points_bohr": points,
                    "surface_areas_bohr2": areas,
                    "qm_vacuum_surface_mep_hartree_per_e": qm_vacuum_mep,
                    "hybrid_permanent_surface_mep_hartree_per_e": permanent_mep,
                    "qm_primary_coarse_induced_surface_mep_per_volt_per_angstrom": np.stack(
                        primary_coarse
                    ),
                    "qm_primary_fine_induced_surface_mep_per_volt_per_angstrom": np.stack(
                        primary_fine
                    ),
                    "qm_control_fine_induced_surface_mep_per_volt_per_angstrom": np.stack(
                        control_fine
                    ),
                    "hybrid_adt_induced_surface_mep_per_volt_per_angstrom": np.stack(
                        hybrid_adt
                    ),
                    "qm_primary_dipole_derivative_eangstrom_per_volt_per_angstrom": np.stack(
                        primary_dipoles
                    ),
                    "qm_control_dipole_derivative_eangstrom_per_volt_per_angstrom": np.stack(
                        control_dipoles
                    ),
                    "hybrid_adt_dipole_derivative_eangstrom_per_volt_per_angstrom": np.stack(
                        hybrid_dipoles
                    ),
                }
                result = {
                    "artifact": RESULT_ARTIFACT,
                    "schema_version": 1,
                    "status": "pass",
                    "scientific_status": "mechanism audit; no capability admission",
                    "preregistration_file_sha256": _sha256_file(Path(prereg["_path"])),
                    "preregistration_artifact_sha256": prereg["preregistration_sha256"],
                    "case_id": "mobley_3867265",
                    "case_name": "acetone",
                    "checkpoint_replay_absolute_error_hartree": abs(
                        checkpoint_energy - replay_energy
                    ),
                    "control_zero_energy_hartree": control_energy,
                    "surface_point_count": len(points),
                    "surface_replay_maximum_absolute_error_bohr": float(
                        np.max(np.abs(points - expected_points))
                    ),
                    "model": {
                        "response_configuration_sha256": (
                            response_model.configuration_sha256()
                        ),
                        "chart_sha256": chart.state_sha256,
                        "mace_mdp_molecular_polarizability_eangstrom2_per_volt": (
                            chart.mdp_molecular_polarizability_eangstrom2_per_volt.tolist()
                        ),
                        "polar_zero_total_charge_e": float(
                            np.sum(chart.polar_zero_source4[:, 0])
                        ),
                    },
                    "permanent_boundary_potential": permanent_metrics,
                    "uniform_response_axes": axes,
                    "aggregate": {
                        "maximum_boundary_response_relative_error": max(
                            float(item["comparison"]["relative_response_error"])
                            for item in axes
                        ),
                        "maximum_area_weighted_relative_error": max(
                            float(item["comparison"]["area_weighted_relative_error"])
                            for item in axes
                        ),
                        "minimum_response_metric_correlation": min(
                            float(item["comparison"]["correlation"]) for item in axes
                        ),
                        "minimum_error_to_uncertainty_ratio": min(
                            float(item["hybrid_error_to_reference_uncertainty_ratio"])
                            for item in axes
                        ),
                        "all_axes_resolved_beyond_reference_uncertainty": all(
                            bool(item["resolved_beyond_reference_uncertainty"])
                            for item in axes
                        ),
                    },
                    "runtime": {
                        **_runtime_identity(),
                        "threads": int(runtime["threads"]),
                        "polar_device": runtime["polar_device"],
                    },
                    "capabilities": NO_CAPABILITIES,
                    "claim_boundary": prereg["claim_boundary"],
                    "wall_seconds": time.perf_counter() - started,
                }
    finally:
        os.chdir(previous)

    for cache in (work / "primary-rinv.npy", work / "control-rinv.npy"):
        cache.unlink(missing_ok=True)
    work.rmdir()
    arrays_path = output_dir / "arrays.npz"
    np.savez_compressed(arrays_path, **arrays)
    result["arrays"] = {
        "path": arrays_path.name,
        "sha256": _sha256_file(arrays_path),
        "keys": sorted(arrays),
        "array_sha256": {name: _array_sha256(value) for name, value in arrays.items()},
    }
    result["result_sha256"] = _canonical_sha256(result)
    return result


def main() -> None:
    args = _parse_args()
    prereg_path = args.preregistration.expanduser().resolve(strict=True)
    prereg = _validate_preregistration(prereg_path)
    prereg["_path"] = str(prereg_path)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    try:
        result = _run(prereg, output_dir)
        result_path = output_dir / "result.json"
        with result_path.open("x", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        print(json.dumps({"output": str(result_path), "status": result["status"]}))
    except BaseException:
        shutil.rmtree(output_dir, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
