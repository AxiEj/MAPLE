#!/usr/bin/env python3
"""Preregister and run a standard auxiliary-density basis ladder.

The already-observed exterior-MEP v1 gate remains failed.  This independent
v2 gate retains all twelve cases and measures the same QM and fitted-density
MEPs through one symmetric, content-addressed PCMSolver operator.  The primary
quantity is the reciprocal reaction-metric upper bound on the fixed-source
polarization-energy error; no experimental solvation label is read.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import time
from typing import Any, Mapping

from ase import Atoms
from ase.units import Bohr
import numpy as np


ARTIFACT = "route2-auxiliary-density-standard-basis-ladder-v1"
PREREG_ARTIFACT = f"{ARTIFACT}-preregistration"
RESULT_ARTIFACT = f"{ARTIFACT}-result"
PARENT_ARTIFACT = (
    "route2-v0-freesolv12-zero-field-mace-static-surface-mep-execution-edfae78e"
)
PARENT_STATUS = "complete-source-diagnostic-no-acceptance-threshold"
EXPECTED_CASE_COUNT = 12
KCAL_PER_HARTREE = 627.5094740631

# One quarter of the one-kcal/mol mean end-to-end target is allocated to the
# static density representation.  A separate 0.50-kcal/mol per-case tail cap
# prevents a low mean from hiding a representation disaster.  Both values are
# locked before this gate is run.
MEAN_ENERGY_BOUND_MAX_KCAL_PER_MOL = 0.25
PER_CASE_ENERGY_BOUND_MAX_KCAL_PER_MOL = 0.50
CONSTRAINT_MAX_ABSOLUTE_ERROR = 1.0e-8
RECIPROCITY_RELATIVE_TOLERANCE = 2.0e-10
CONSTRAINT_RANK_RELATIVE_TOLERANCE = 1.0e-12
METRIC_CONDITION_NUMBER_MAXIMUM = 1.0e10
CANDIDATE_ORDER = ("etb-beta-2p0", "autoaux", "etb-beta-1p5")

SOURCE_ROOT = Path(__file__).resolve().parents[2]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))
SELF_REPO_PATH = (
    "tools/route2_release/run_auxiliary_density_basis_ladder.py"
)
SOURCE_PATHS = (
    SELF_REPO_PATH,
    "maple/solvation/reference/auxiliary_density.py",
    "maple/solvation/reference/auxiliary_density_basis.py",
    "maple/solvation/reference/pyscf_pcmsolver.py",
    "maple/function/calculator/extra_correction/implicit/continuum_response.py",
    "maple/function/calculator/extra_correction/implicit/pcmsolver.py",
    "maple/function/calculator/extra_correction/implicit/route2_pcmsolver_cavity.py",
    "maple/function/calculator/extra_correction/implicit/smd.py",
    "maple/function/calculator/extra_correction/implicit/smd_cds.py",
    "maple/function/route2_smd_profiles.py",
    "maple/function/route2_solvents.py",
)
PRIOR_MAKE_AUXBASIS_RESULT = SOURCE_ROOT / (
    "docs/route2/evidence/auxiliary-density-pcmsolver-energy-gate-20260824/"
    "result.json"
)
PRO_ADVISOR_ANSWER = SOURCE_ROOT / (
    "docs/route2/evidence/auxiliary-density-basis-ladder-pro-20260824/answer.md"
)
CONFIRMATION_SELECTION = SOURCE_ROOT / (
    "docs/route2/preregistrations/"
    "auxiliary-density-confirmation-selection-v1.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _load_object(path: Path, *, name: str) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain one JSON object.")
    return value


def _file_record(path: Path) -> dict[str, object]:
    resolved = path.expanduser().resolve(strict=True)
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def _write_exclusive(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        handle.write(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
    path.chmod(0o444)


def _write_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def _case_files(static_root: Path, compound_id: str) -> dict[str, Path]:
    root = static_root / "records" / compound_id / "qm-attempt-001"
    return {
        "checkpoint": root / "gas/gas.chk",
        "checkpoint_ledger": root / "gas/gas.json",
        "qm_mep": root / "qm-static-mep.npz",
    }


def _basis_for(molecule: Any, candidate_id: str) -> Mapping[str, object]:
    from pyscf import df

    if candidate_id == "etb-beta-2p0":
        return df.aug_etb(molecule, beta=2.0)
    if candidate_id == "autoaux":
        return df.autoaux(molecule)
    if candidate_id == "etb-beta-1p5":
        return df.aug_etb(molecule, beta=1.5)
    raise ValueError(f"unknown auxiliary-basis candidate {candidate_id!r}.")


def _basis_record(molecule: Any, candidate_id: str) -> dict[str, object]:
    from pyscf import df

    basis = _basis_for(molecule, candidate_id)
    serialized = json.dumps(
        basis,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    auxiliary = df.addons.make_auxmol(molecule, auxbasis=dict(basis))
    return {
        "candidate_id": candidate_id,
        "basis_sha256": hashlib.sha256(serialized).hexdigest(),
        "auxiliary_dimension": int(auxiliary.nao_nr()),
    }


def _load_pcmsolver_parser_from_paths(
    *, library: Path, python_path: Path
):
    previous_library = os.environ.get("PCMSOLVER_LIBRARY")
    previous_python = os.environ.get("PCMSOLVER_PYTHON_PATH")
    os.environ["PCMSOLVER_LIBRARY"] = str(library)
    os.environ["PCMSOLVER_PYTHON_PATH"] = str(python_path)
    try:
        from maple.function.calculator.extra_correction.implicit.smd import (
            _load_pcmsolver_parser,
        )

        return _load_pcmsolver_parser()
    finally:
        if previous_library is None:
            os.environ.pop("PCMSOLVER_LIBRARY", None)
        else:
            os.environ["PCMSOLVER_LIBRARY"] = previous_library
        if previous_python is None:
            os.environ.pop("PCMSOLVER_PYTHON_PATH", None)
        else:
            os.environ["PCMSOLVER_PYTHON_PATH"] = previous_python


def create_preregistration(args: argparse.Namespace) -> dict[str, Any]:
    static_root = args.static_root.expanduser().resolve(strict=True)
    parent_path = args.parent_execution.expanduser().resolve(strict=True)
    library = args.pcmsolver_library.expanduser().resolve(strict=True)
    python_path = args.pcmsolver_python_path.expanduser().resolve(strict=True)
    parser_init = (python_path / "pcmsolver/__init__.py").resolve(strict=True)
    output = args.output.expanduser().resolve()
    result_path = args.result.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    if result_path.exists():
        raise ValueError("result path must not exist before preregistration.")
    parent = _load_object(parent_path, name="parent execution")
    confirmation = _load_object(
        CONFIRMATION_SELECTION.resolve(strict=True),
        name="confirmation selection",
    )
    if (
        confirmation.get("artifact")
        != "route2-auxiliary-density-confirmation-selection-v1"
        or confirmation.get("status")
        != "locked-before-any-standard-basis-ladder-result"
        or len(confirmation.get("records", ())) != EXPECTED_CASE_COUNT
    ):
        raise ValueError("target-free confirmation selection is incompatible.")
    records = parent.get("records")
    if (
        parent.get("artifact") != PARENT_ARTIFACT
        or parent.get("status") != PARENT_STATUS
        or not isinstance(records, list)
        or len(records) != EXPECTED_CASE_COUNT
    ):
        raise ValueError("parent twelve-case source execution is incompatible.")
    from pyscf import lib

    cases = []
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("parent source record is malformed.")
        compound_id = str(record["compound_id"])
        files = _case_files(static_root, compound_id)
        molecule = lib.chkfile.load_mol(str(files["checkpoint"]))
        molecule.verbose = 0
        cases.append(
            {
                "compound_id": compound_id,
                "name": str(record["name"]),
                "chemical_class": str(record["chemical_class"]),
                "functional_group": record.get("functional_group"),
                "files": {
                    key: _file_record(path)
                    for key, path in files.items()
                },
                "candidate_bases": [
                    _basis_record(molecule, candidate_id)
                    for candidate_id in CANDIDATE_ORDER
                ],
            }
        )
    import pyscf

    from maple.function.calculator.extra_correction.implicit.route2_pcmsolver_cavity import (
        PCM_INTRINSIC_MIN_RADIUS_ANGSTROM,
        PCM_INTRINSIC_TESSERA_AREA_ANGSTROM2,
        PCM_SMD_WATER_DIELECTRIC,
        PCM_SMD_WATER_OPTICAL_DIELECTRIC,
    )

    payload: dict[str, Any] = {
        "artifact": PREREG_ARTIFACT,
        "schema_version": 1,
        "status": "locked-before-first-standard-basis-ladder-evaluation",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(SOURCE_ROOT),
        "source_files_sha256": {
            relative: _sha256(SOURCE_ROOT / relative) for relative in SOURCE_PATHS
        },
        "parent_execution": _file_record(parent_path),
        "prior_evidence": {
            "terminated_make_auxbasis_result": _file_record(
                PRIOR_MAKE_AUXBASIS_RESULT
            ),
            "basis_ladder_pro_advice": _file_record(PRO_ADVISOR_ANSWER),
            "sealed_confirmation_selection": _file_record(
                CONFIRMATION_SELECTION
            ),
            "advice_is_non_authoritative": True,
        },
        "static_root": str(static_root),
        "cases": cases,
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pyscf": pyscf.__version__,
            "pyscf_init": _file_record(Path(pyscf.__file__).resolve(strict=True)),
            "pcmsolver_library": _file_record(library),
            "pcmsolver_python_path": str(python_path),
            "pcmsolver_parser_init": _file_record(parser_init),
        },
        "density_method": {
            "orbital_basis": "checkpoint-defined def2-TZVPD",
            "candidate_order": list(CANDIDATE_ORDER),
            "candidate_definitions": {
                "etb-beta-2p0": "pyscf.df.aug_etb(molecule,beta=2.0)",
                "autoaux": "pyscf.df.autoaux(molecule)",
                "etb-beta-1p5": "pyscf.df.aug_etb(molecule,beta=1.5)",
            },
            "selection_rule": (
                "evaluate candidates sequentially and retain the first full "
                "twelve-case pass; do not evaluate later candidates after a pass"
            ),
            "density_projection": "Coulomb metric",
            "normalized_metric": "Jhat=D^-1/2 J D^-1/2; D=diag(J)",
            "constraint_nullspace": "Z spans ker(C D^-1/2)",
            "constraint_rank_relative_tolerance": (
                CONSTRAINT_RANK_RELATIVE_TOLERANCE
            ),
            "constraint_restricted_condition_number_maximum": (
                METRIC_CONDITION_NUMBER_MAXIMUM
            ),
            "condition_policy": (
                "fail cutoff only; no eigenmode truncation or pseudoinverse"
            ),
            "moment_grid_level": 4,
            "hard_constraints": "electron count and electronic first moment",
        },
        "continuum_oracle": {
            "backend": "pinned symmetric PCMSolver external-MEP response",
            "model": "IEFPCM",
            "cavity": "intrinsic SMD water radii; zero probe; no added spheres",
            "tessera_area_angstrom2": PCM_INTRINSIC_TESSERA_AREA_ANGSTROM2,
            "minimum_added_sphere_radius_angstrom": (
                PCM_INTRINSIC_MIN_RADIUS_ANGSTROM
            ),
            "static_dielectric": PCM_SMD_WATER_DIELECTRIC,
            "optical_dielectric": PCM_SMD_WATER_OPTICAL_DIELECTRIC,
            "metric": "sqrt(-v.T R v) with reciprocal R",
            "bound": "x*d + 0.5*d^2",
        },
        "error_budget": {
            "end_to_end_goal_kcal_per_mol": 1.0,
            "allocation_policy": (
                "additive mean budget with an independent per-case tail cap"
            ),
            "density_representation_mean_fraction": 0.25,
            "mean_energy_upper_bound_max_kcal_per_mol": (
                MEAN_ENERGY_BOUND_MAX_KCAL_PER_MOL
            ),
            "per_case_energy_upper_bound_max_kcal_per_mol": (
                PER_CASE_ENERGY_BOUND_MAX_KCAL_PER_MOL
            ),
            "actual_energy_error_role": (
                "reported diagnostic only; not primary because cross/self terms "
                "can cancel"
            ),
        },
        "gates": {
            "mean_reaction_metric_upper_bound_max_kcal_per_mol": (
                MEAN_ENERGY_BOUND_MAX_KCAL_PER_MOL
            ),
            "maximum_reaction_metric_upper_bound_max_kcal_per_mol": (
                PER_CASE_ENERGY_BOUND_MAX_KCAL_PER_MOL
            ),
            "every_constraint_max_absolute_error": (
                CONSTRAINT_MAX_ABSOLUTE_ERROR
            ),
            "every_reciprocity_relative_defect_max": (
                RECIPROCITY_RELATIVE_TOLERANCE
            ),
        },
        "case_policy": {
            "all_twelve_retained": True,
            "methane_or_symmetry_control_excluded": False,
            "v1_result_reinterpreted": False,
            "same_panel_role": "representation-development-only",
            "fresh_confirmation_required_before_training_admission": True,
        },
        "selection_history": {
            "make_auxbasis_twelve_case_energy_gate_opened_and_failed": True,
            "one_acetone_autoaux_exterior_mep_probe_previously_opened": True,
            "etb_candidate_energy_results_opened_before_lock": False,
            "autoaux_twelve_case_energy_results_opened_before_lock": False,
            "experimental_solvation_target_used_for_candidate_order": False,
        },
        "claim_boundary": {
            "experimental_solvation_target_read": False,
            "fit_or_training_performed": False,
            "capability_admitted": False,
            "allowed_decision": (
                "select the first passing standard basis or reject the ladder"
            ),
        },
        "result_path": str(result_path),
    }
    payload["preregistration_sha256"] = _canonical_sha256(payload)
    _write_exclusive(output, payload)
    return payload


def _validate_preregistration(path: Path) -> dict[str, Any]:
    payload = _load_object(path, name="preregistration")
    unsigned = dict(payload)
    observed = unsigned.pop("preregistration_sha256", None)
    if (
        payload.get("artifact") != PREREG_ARTIFACT
        or payload.get("schema_version") != 1
        or payload.get("status")
        != "locked-before-first-standard-basis-ladder-evaluation"
        or observed != _canonical_sha256(unsigned)
    ):
        raise ValueError("auxiliary-density PCM preregistration identity drifted.")
    if payload.get("source_root") != str(SOURCE_ROOT):
        raise ValueError("preregistration source root drifted.")
    for relative, expected in payload["source_files_sha256"].items():
        if _sha256(SOURCE_ROOT / relative) != expected:
            raise ValueError(f"preregistered source {relative!r} drifted.")
    cases = payload.get("cases")
    if not isinstance(cases, list) or len(cases) != EXPECTED_CASE_COUNT:
        raise ValueError("preregistered case panel drifted.")
    for case in cases:
        for name, record in case["files"].items():
            path_value = Path(str(record["path"])).resolve(strict=True)
            if _sha256(path_value) != record["sha256"]:
                raise ValueError(f"case file {name!r} drifted.")
        candidate_bases = case.get("candidate_bases")
        if (
            not isinstance(candidate_bases, list)
            or tuple(item.get("candidate_id") for item in candidate_bases)
            != CANDIDATE_ORDER
        ):
            raise ValueError("preregistered candidate-basis order drifted.")
    if tuple(payload["density_method"]["candidate_order"]) != CANDIDATE_ORDER:
        raise ValueError("basis-ladder candidate order drifted.")
    if payload["case_policy"] != {
        "all_twelve_retained": True,
        "methane_or_symmetry_control_excluded": False,
        "v1_result_reinterpreted": False,
        "same_panel_role": "representation-development-only",
        "fresh_confirmation_required_before_training_admission": True,
    }:
        raise ValueError("case policy drifted.")
    return payload


def _parsed_pcm_input(
    directory: Path,
    *,
    atom_count: int,
    radii_angstrom: np.ndarray,
    parser,
) -> Path:
    from maple.function.calculator.extra_correction.implicit.route2_pcmsolver_cavity import (
        PCM_INTRINSIC_MIN_RADIUS_ANGSTROM,
        PCM_INTRINSIC_TESSERA_AREA_ANGSTROM2,
        _intrinsic_pcm_parser_surrogate,
        _patch_intrinsic_pcm_machine_input,
        _pcm_input_text,
    )

    raw = directory / "aux-density-intrinsic-smd.pcm"
    raw_text = _pcm_input_text(
        atom_count,
        radii_angstrom,
        tessera_area_angstrom2=PCM_INTRINSIC_TESSERA_AREA_ANGSTROM2,
        minimum_added_sphere_radius_angstrom=PCM_INTRINSIC_MIN_RADIUS_ANGSTROM,
        dielectric_policy="explicit-smd-water-78.355-v1",
    )
    raw.write_text(raw_text)
    surrogate = directory / "aux-density-intrinsic-smd.parser-surrogate.pcm"
    surrogate.write_text(_intrinsic_pcm_parser_surrogate(raw_text))
    parser(str(surrogate), write_out=True)
    parser_output = surrogate.with_name("@" + surrogate.name)
    if not parser_output.is_file():
        raise RuntimeError("PCMSolver parser did not produce a machine input.")
    patched, _semantics = _patch_intrinsic_pcm_machine_input(
        parser_output.read_text()
    )
    result = directory / "@aux-density-intrinsic-smd.pcm"
    result.write_text(patched)
    return result


def _project_case(
    case: Mapping[str, Any],
    *,
    candidate_id: str,
    library: Path,
    parser,
) -> dict[str, Any]:
    from pyscf import gto, lib

    from maple.function.calculator.extra_correction.implicit.continuum_response import (
        PCMSolverExternalMEPCavityResponse,
    )
    from maple.function.calculator.extra_correction.implicit.pcmsolver import (
        PCMSolverSession,
    )
    from maple.function.calculator.extra_correction.implicit.smd_cds import (
        smd_coulomb_radii,
    )
    from maple.solvation.reference.pyscf_pcmsolver import (
        AOInverseDistanceIntegralCache,
        closed_shell_density_from_orbitals,
    )
    from maple.solvation.reference.auxiliary_density_basis import (
        coulomb_fit_auxiliary_density_in_basis,
    )

    started = time.perf_counter()
    checkpoint = Path(str(case["files"]["checkpoint"]["path"]))
    molecule = lib.chkfile.load_mol(str(checkpoint))
    molecule.verbose = 0
    overlap = np.asarray(molecule.intor_symmetric("int1e_ovlp"), dtype=np.float64)
    density, density_record = closed_shell_density_from_orbitals(
        lib.chkfile.load(str(checkpoint), "scf/mo_coeff"),
        lib.chkfile.load(str(checkpoint), "scf/mo_occ"),
        overlap,
    )
    basis = _basis_for(molecule, candidate_id)
    basis_sha256 = hashlib.sha256(
        json.dumps(
            basis,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()
    expected_basis = next(
        item
        for item in case["candidate_bases"]
        if item["candidate_id"] == candidate_id
    )
    if basis_sha256 != expected_basis["basis_sha256"]:
        raise RuntimeError("preregistered auxiliary basis drifted.")
    auxiliary, projection = coulomb_fit_auxiliary_density_in_basis(
        molecule,
        density,
        auxiliary_basis=basis,
        moment_grid_level=4,
        constraint_rank_relative_tolerance=(
            CONSTRAINT_RANK_RELATIVE_TOLERANCE
        ),
    )
    if int(auxiliary.nao_nr()) != expected_basis["auxiliary_dimension"]:
        raise RuntimeError("preregistered auxiliary dimension drifted.")
    with np.load(str(case["files"]["qm_mep"]["path"])) as state:
        positions_angstrom = np.asarray(
            state["atom_positions_angstrom"], dtype=np.float64
        )
        atomic_numbers = np.asarray(state["atomic_numbers"], dtype=np.int64)
    atoms = Atoms(numbers=atomic_numbers, positions=positions_angstrom)
    geometry_error = float(
        np.max(
            np.abs(np.asarray(molecule.atom_coords()) * Bohr - positions_angstrom)
        )
    )
    if not np.array_equal(
        np.asarray(molecule.atom_charges(), dtype=np.int64),
        atomic_numbers,
    ):
        raise RuntimeError("checkpoint and frozen atom identities differ.")
    if geometry_error > 1.0e-8:
        raise RuntimeError("checkpoint and frozen geometry differ.")
    radii = smd_coulomb_radii(atoms.get_chemical_symbols(), solvent="water")

    with tempfile.TemporaryDirectory(prefix="maple-aux-density-pcm-") as temporary:
        directory = Path(temporary)
        parsed = _parsed_pcm_input(
            directory,
            atom_count=len(atoms),
            radii_angstrom=radii,
            parser=parser,
        )
        previous = Path.cwd()
        try:
            os.chdir(directory)
            with PCMSolverSession(
                np.asarray(atomic_numbers, dtype=np.float64),
                positions_angstrom / Bohr,
                parsed,
                library_path=library,
            ) as session:
                response = PCMSolverExternalMEPCavityResponse(
                    session,
                    cavity_radii_angstrom=radii,
                )
                points = response.surface_points_bohr
                cache = AOInverseDistanceIntegralCache(molecule, points)
                electronic = cache.electronic_surface_potential(density)
                cache.close()
                distances = np.linalg.norm(
                    points[:, None, :] - np.asarray(molecule.atom_coords())[None, :, :],
                    axis=2,
                )
                nuclear = (1.0 / distances) @ np.asarray(
                    molecule.atom_charges(), dtype=np.float64
                )
                reference = nuclear - electronic
                cross = gto.mole.intor_cross(
                    "int2c2e",
                    auxiliary,
                    gto.fakemol_for_charges(points),
                )
                predicted = nuclear - projection.constrained_coefficients @ cross
                error = predicted - reference
                reference_state = response.solve(reference)
                predicted_state = response.solve(predicted)
                reference_charge = reference_state.energy_conjugate_surface_charge_e
                predicted_charge = predicted_state.energy_conjugate_surface_charge_e
                error_charge = response.apply_energy_conjugate(error)
                areas = response.surface_areas_bohr2
                configuration_sha256 = response.configuration_sha256()
                cavity_sha256 = response.cavity_configuration_sha256()
                parsed_record = {
                    "logical_name": parsed.name,
                    "bytes": parsed.stat().st_size,
                    "sha256": _sha256(parsed),
                }
        finally:
            os.chdir(previous)

    reference_energy = reference_state.polarization_energy_hartree
    predicted_energy = predicted_state.polarization_energy_hartree
    reference_half_work = 0.5 * float(np.vdot(reference, reference_charge))
    predicted_half_work = 0.5 * float(np.vdot(predicted, predicted_charge))
    half_work_residual = max(
        abs(reference_energy - reference_half_work),
        abs(predicted_energy - predicted_half_work),
    )
    if half_work_residual > 2.0e-12:
        raise RuntimeError("PCMSolver energy and half-work identity differ.")
    reference_quadratic = -float(np.vdot(reference, reference_charge))
    error_quadratic = -float(np.vdot(error, error_charge))
    tolerance = 2.0e-12 * max(
        1.0,
        abs(reference_quadratic),
        abs(error_quadratic),
    )
    if reference_quadratic < -tolerance or error_quadratic < -tolerance:
        raise RuntimeError("PCMSolver response is not negative semidefinite.")
    reference_norm = float(np.sqrt(max(0.0, reference_quadratic)))
    error_norm = float(np.sqrt(max(0.0, error_quadratic)))
    upper_bound = reference_norm * error_norm + 0.5 * error_norm * error_norm
    actual_error = abs(predicted_energy - reference_energy)
    if actual_error > upper_bound + 2.0e-10 / KCAL_PER_HARTREE:
        raise RuntimeError("reaction-metric bound does not bound the energy error.")
    cross_forward = float(np.vdot(reference, predicted_charge))
    cross_reverse = float(np.vdot(predicted, reference_charge))
    reciprocity_defect = abs(cross_forward - cross_reverse) / max(
        1.0e-15,
        abs(cross_forward),
        abs(cross_reverse),
    )
    area_relative = float(
        np.sqrt(np.sum(areas * error**2) / np.sum(areas * reference**2))
    )
    return {
        "candidate_id": candidate_id,
        "basis_sha256": basis_sha256,
        "compound_id": case["compound_id"],
        "name": case["name"],
        "chemical_class": case["chemical_class"],
        "functional_group": case["functional_group"],
        "atom_count": len(atoms),
        "geometry_max_absolute_error_angstrom": geometry_error,
        "auxiliary_dimension": int(auxiliary.nao_nr()),
        "constraint_rank": projection.constraint_rank,
        "normalized_restricted_minimum_eigenvalue": (
            projection.normalized_restricted_minimum_eigenvalue
        ),
        "normalized_restricted_maximum_eigenvalue": (
            projection.normalized_restricted_maximum_eigenvalue
        ),
        "metric_condition_number": projection.metric_condition_number,
        "constrained_moment_max_absolute_error": (
            projection.constrained_moment_max_absolute_error
        ),
        "projected_stationarity_max_absolute_error": (
            projection.projected_stationarity_max_absolute_error
        ),
        "density_record": density_record,
        "pcmsolver_configuration_sha256": configuration_sha256,
        "pcmsolver_cavity_sha256": cavity_sha256,
        "parsed_input": parsed_record,
        "cavity_point_count": len(reference),
        "surface_mep_area_weighted_relative_l2": area_relative,
        "reference_polarization_energy_kcal_per_mol": (
            reference_energy * KCAL_PER_HARTREE
        ),
        "predicted_polarization_energy_kcal_per_mol": (
            predicted_energy * KCAL_PER_HARTREE
        ),
        "actual_energy_error_kcal_per_mol": actual_error * KCAL_PER_HARTREE,
        "reference_reaction_norm_sqrt_hartree": reference_norm,
        "error_reaction_norm_sqrt_hartree": error_norm,
        "energy_error_upper_bound_kcal_per_mol": (
            upper_bound * KCAL_PER_HARTREE
        ),
        "reciprocity_relative_defect": reciprocity_defect,
        "half_work_max_absolute_residual_hartree": half_work_residual,
        "case_passed": bool(
            upper_bound * KCAL_PER_HARTREE
            <= PER_CASE_ENERGY_BOUND_MAX_KCAL_PER_MOL
            and projection.constrained_moment_max_absolute_error
            <= CONSTRAINT_MAX_ABSOLUTE_ERROR
            and projection.metric_condition_number
            <= METRIC_CONDITION_NUMBER_MAXIMUM
            and reciprocity_defect <= RECIPROCITY_RELATIVE_TOLERANCE
        ),
        "wall_seconds": time.perf_counter() - started,
    }


def _candidate_summary(
    candidate_id: str,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    actual = np.asarray(
        [record["actual_energy_error_kcal_per_mol"] for record in records],
        dtype=np.float64,
    )
    bounds = np.asarray(
        [record["energy_error_upper_bound_kcal_per_mol"] for record in records],
        dtype=np.float64,
    )
    constraint_errors = np.asarray(
        [record["constrained_moment_max_absolute_error"] for record in records],
        dtype=np.float64,
    )
    reciprocity = np.asarray(
        [record["reciprocity_relative_defect"] for record in records],
        dtype=np.float64,
    )
    conditions = np.asarray(
        [record["metric_condition_number"] for record in records],
        dtype=np.float64,
    )
    gates = {
        "mean_bound": bool(
            np.mean(bounds) <= MEAN_ENERGY_BOUND_MAX_KCAL_PER_MOL
        ),
        "maximum_bound": bool(
            np.max(bounds) <= PER_CASE_ENERGY_BOUND_MAX_KCAL_PER_MOL
        ),
        "constraints": bool(
            np.max(constraint_errors) <= CONSTRAINT_MAX_ABSOLUTE_ERROR
        ),
        "reciprocity": bool(
            np.max(reciprocity) <= RECIPROCITY_RELATIVE_TOLERANCE
        ),
        "conditioning": bool(
            np.max(conditions) <= METRIC_CONDITION_NUMBER_MAXIMUM
        ),
        "all_cases": all(bool(record["case_passed"]) for record in records),
    }
    return {
        "candidate_id": candidate_id,
        "status": "pass" if all(gates.values()) else "fail",
        "record_count": len(records),
        "records": records,
        "aggregate": {
            "total_auxiliary_dimension": int(
                sum(record["auxiliary_dimension"] for record in records)
            ),
            "maximum_auxiliary_dimension": int(
                max(record["auxiliary_dimension"] for record in records)
            ),
            "mean_actual_energy_error_kcal_per_mol": float(np.mean(actual)),
            "maximum_actual_energy_error_kcal_per_mol": float(np.max(actual)),
            "mean_energy_error_upper_bound_kcal_per_mol": float(np.mean(bounds)),
            "maximum_energy_error_upper_bound_kcal_per_mol": float(np.max(bounds)),
            "maximum_constraint_absolute_error": float(np.max(constraint_errors)),
            "maximum_reciprocity_relative_defect": float(np.max(reciprocity)),
            "maximum_metric_condition_number": float(np.max(conditions)),
        },
        "gates": gates,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    preregistration_path = args.preregistration.expanduser().resolve(strict=True)
    payload = _validate_preregistration(preregistration_path)
    result_path = Path(str(payload["result_path"])).resolve()
    if result_path.exists():
        raise FileExistsError(result_path)
    runtime = payload["runtime"]
    library = Path(str(runtime["pcmsolver_library"]["path"])).resolve(strict=True)
    python_path = Path(str(runtime["pcmsolver_python_path"])).resolve(strict=True)
    parser = _load_pcmsolver_parser_from_paths(
        library=library,
        python_path=python_path,
    )
    summaries = []
    selected_candidate_id = None
    for candidate_id in CANDIDATE_ORDER:
        records = []
        for case in payload["cases"]:
            record = _project_case(
                case,
                candidate_id=candidate_id,
                library=library,
                parser=parser,
            )
            records.append(record)
            print(
                f"{candidate_id} {record['compound_id']} bound_kcal="
                f"{record['energy_error_upper_bound_kcal_per_mol']:.8f} "
                f"actual_kcal={record['actual_energy_error_kcal_per_mol']:.8f}",
                flush=True,
            )
        summary = _candidate_summary(candidate_id, records)
        summaries.append(summary)
        print(
            f"{candidate_id} status={summary['status']} "
            f"mean_bound={summary['aggregate']['mean_energy_error_upper_bound_kcal_per_mol']:.8f} "
            f"max_bound={summary['aggregate']['maximum_energy_error_upper_bound_kcal_per_mol']:.8f}",
            flush=True,
        )
        if summary["status"] == "pass":
            selected_candidate_id = candidate_id
            break

    result: dict[str, Any] = {
        "artifact": RESULT_ARTIFACT,
        "schema_version": 1,
        "status": "pass" if selected_candidate_id is not None else "fail",
        "preregistration_file_sha256": _sha256(preregistration_path),
        "preregistration_sha256": payload["preregistration_sha256"],
        "candidate_order": list(CANDIDATE_ORDER),
        "evaluated_candidate_count": len(summaries),
        "evaluated_candidates": summaries,
        "selected_candidate_id": selected_candidate_id,
        "selection_rule": payload["density_method"]["selection_rule"],
        "claim_boundary": payload["claim_boundary"],
    }
    result["result_sha256"] = _canonical_sha256(result)
    _write_atomic(result_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("preregister")
    create.add_argument("--static-root", type=Path, required=True)
    create.add_argument("--parent-execution", type=Path, required=True)
    create.add_argument("--pcmsolver-library", type=Path, required=True)
    create.add_argument("--pcmsolver-python-path", type=Path, required=True)
    create.add_argument("--result", type=Path, required=True)
    create.add_argument("--output", type=Path, required=True)
    execute = subparsers.add_parser("run")
    execute.add_argument("--preregistration", type=Path, required=True)
    args = parser.parse_args()
    value = (
        create_preregistration(args)
        if args.command == "preregister"
        else run(args)
    )
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
