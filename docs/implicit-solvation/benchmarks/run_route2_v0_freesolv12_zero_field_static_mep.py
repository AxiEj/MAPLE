#!/usr/bin/env python3
"""Diagnose the zero-field MACE-POLAR permanent source on twelve QM MEPs.

The runner intentionally consumes a no-experimental-label identity manifest.
For each hash-locked geometry, it creates a geometry-only exterior SMD shell,
calculates a gas-phase omegaB97M-V/def2-TZVPD static MEP, and compares it with
the *zero-field* MACE-POLAR point-monopole/dipole source.  It does not invoke a
continuum, evaluate a solvation free energy, or decide a model from error to
experiment.  In particular, it cannot make the raw MACE source a stationary
V0 permanent reference.

Run only from a clean committed checkout with an interpreter containing the
pinned MACE-POLAR and PySCF runtimes, for example::

    PYTHONPATH=$PWD /path/to/python \\
      docs/implicit-solvation/benchmarks/run_route2_v0_freesolv12_zero_field_static_mep.py \\
      --mol2-root .omx/benchmarks/route2-macepolar-smd-smoke/dataset \\
      --work-dir .omx/benchmarks/route2-v0-freesolv12-zero-field-static-mep-<sha> \\
      --qm-python /path/to/python
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Sequence

import numpy as np
from ase import Atoms


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
for _path in (REPO_ROOT, SCRIPT_DIR):
    _value = str(_path)
    if _value not in sys.path:
        sys.path.insert(0, _value)

from benchmark_core import canonical_json_bytes, sha256_file, write_json_atomic  # noqa: E402
from maple.function.calculator.set_calculator import SetCalculator  # noqa: E402
from maple.function.read.command_control import CommandControl  # noqa: E402
from maple.function.calculator.extra_correction.implicit.gto_density import (  # noqa: E402
    cartesian_multipoles,
    point_multipole_potential,
)
from maple.function.calculator.extra_correction.implicit.route2_static_surface_mep import (  # noqa: E402
    build_route2_smd_exterior_probe_surface,
    route2_weighted_surface_mep_discrepancy,
)
from maple.function.calculator.extra_correction.implicit.smd_cds import (  # noqa: E402
    route2_coulomb_radii,
)
from maple.function.read.filereader.mol2_reader import MOL2Reader  # noqa: E402
from maple.function.route2_smd_profiles import (  # noqa: E402
    DDPCM_MULTISOLVENT_SMD_DIRECT_PCM_V2_PROFILE,
)


ARTIFACT = "route2-v0-freesolv12-zero-field-mace-static-surface-mep-v1"
MANIFEST_PATH = (
    SCRIPT_DIR
    / "route2-v0-freesolv12-zero-field-mace-static-surface-mep-prereg-v1.json"
)
QM_CHECKPOINT_RUNNER = SCRIPT_DIR / "run_route2_qm_gas_checkpoint.py"
QM_MEP_RUNNER = SCRIPT_DIR / "route2_qm_surface_mep.py"
WATER_SOLVENT = "water"
CLEARANCE_ANGSTROM = 1.0


_REQUIRED_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "protocol_id",
        "status",
        "claim_boundary",
        "parent_identity",
        "selection_policy",
        "source_candidate",
        "qm_reference",
        "probe_surface",
        "required_metrics",
        "locked_records",
        "locked_record_identity_sha256",
    }
)
_REQUIRED_RECORD_KEYS = frozenset(
    {
        "ordinal",
        "compound_id",
        "name",
        "chemical_class",
        "functional_group",
        "mol2_archive_member",
        "mol2_sha256",
        "natoms",
        "source_database_record_sha256",
    }
)
_FORBIDDEN_LABEL_KEYS = frozenset(
    {
        "experimental_kcal_mol",
        "experimental_uncertainty_kcal_mol",
        "experimental_reference",
        "smiles",
    }
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mol2-root", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--qm-python", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument(
        "--retry-failures",
        action="store_true",
        help="Retry only persisted computational failures under the identical run lock.",
    )
    return parser.parse_args(argv)


def _require_private_path(path: Path, *, label: str) -> Path:
    resolved = path.resolve()
    private_root = (REPO_ROOT / ".omx").resolve()
    try:
        resolved.relative_to(private_root)
    except ValueError as exc:
        raise ValueError(f"{label} must be located below {private_root}.") from exc
    return resolved


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(REPO_ROOT), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _require_clean_source() -> str:
    if _git("status", "--porcelain"):
        raise RuntimeError(
            "The zero-field static-MEP oracle requires a clean committed checkout; "
            "commit source changes before creating evidence."
        )
    return _git("rev-parse", "HEAD")


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def load_static_mep_manifest(path: Path) -> dict[str, object]:
    """Load the label-free source identity panel and reject hidden labels."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) != _REQUIRED_MANIFEST_KEYS:
        raise ValueError("Static-MEP manifest does not have the exact v1 schema.")
    if payload["schema_version"] != 1 or payload["status"] != "preregistered-not-executed":
        raise ValueError("Static-MEP manifest version/status is invalid.")
    records = payload["locked_records"]
    if not isinstance(records, list) or len(records) != 12:
        raise ValueError("Static-MEP source panel must contain all twelve locked geometries.")
    observed_ids: list[str] = []
    for ordinal, record in enumerate(records, start=1):
        if not isinstance(record, dict) or set(record) != _REQUIRED_RECORD_KEYS:
            raise ValueError("Static-MEP record does not have the exact no-label schema.")
        if _FORBIDDEN_LABEL_KEYS.intersection(record):
            raise ValueError("Static-MEP manifest must not carry experimental or SMILES fields.")
        if record["ordinal"] != ordinal:
            raise ValueError("Static-MEP records must be in immutable ordinal order.")
        compound_id = record["compound_id"]
        if not isinstance(compound_id, str) or not compound_id:
            raise ValueError("Static-MEP record compound IDs must be nonempty strings.")
        observed_ids.append(compound_id)
        if int(record["natoms"]) <= 0:
            raise ValueError("Static-MEP record atom counts must be positive.")
        digest = record["mol2_sha256"]
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError("Static-MEP record MOL2 digests must be SHA-256 values.")
    if len(set(observed_ids)) != len(observed_ids):
        raise ValueError("Static-MEP record IDs must be unique.")
    if payload["locked_record_identity_sha256"] != _canonical_hash(records):
        raise ValueError("Static-MEP record identity digest does not match its records.")
    return payload


def _resolve_mol2_path(mol2_root: Path, record: dict[str, object]) -> Path:
    member = str(record["mol2_archive_member"])
    candidates = (mol2_root / member, mol2_root / Path(member).name)
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        raise FileNotFoundError(f"Locked MOL2 member is absent: {member}")
    expected_digest = str(record["mol2_sha256"])
    if sha256_file(path) != expected_digest:
        raise ValueError(f"MOL2 identity drifted for {record['compound_id']}.")
    return path


def _atoms(record: dict[str, object], mol2_root: Path) -> Atoms:
    atoms = MOL2Reader(str(_resolve_mol2_path(mol2_root, record)), charge=0, mult=1)
    if len(atoms) != int(record["natoms"]):
        raise RuntimeError(f"Locked atom count drifted for {record['compound_id']}.")
    atoms.info.update(charge=0, mult=1)
    return atoms


def _load_route2_float64_source_calculator(first_atoms: Atoms, work_dir: Path):
    """Load the registered float64 Route-2 MACE state without a solvent solve.

    ``MACEPolCalculator`` reserves float64 for its Route-2 factory path so
    that the ordinary float32 gas-only calculator remains unchanged.  This
    source oracle reuses that existing factory only to load the same official
    zero-field MACE state at Route-2 precision.  It calls ``polar_state``
    directly below and never invokes the attached correction's evaluate/SCF
    path or obtains a combined energy.
    """

    settings = CommandControl.from_settings(
        [
            "#model=macepol-m",
            "#sp",
            (
                "#solv(implicit=water,method=smd,provider=pyddx,"
                f"profile={DDPCM_MULTISOLVENT_SMD_DIRECT_PCM_V2_PROFILE},"
                "response=scf,standard_state=1m,experimental=true)"
            ),
        ]
    ).as_dict()
    calculator = SetCalculator(
        "cpu",
        settings["model"],
        str(work_dir / "mace-zero-field-source-load.out"),
        atoms=first_atoms,
        d4=False,
        implicit=settings["solv"]["method"],
        solvent=settings["solv"]["implicit"],
        model_options=settings.get("model_options"),
        solvation_options=settings["solv"],
        charge_options=settings.get("charge") or {},
    ).set_calculator()
    if str(getattr(calculator, "dtype", "")) != "torch.float64":
        raise RuntimeError(
            "Static-MEP source oracle requires Route-2 float64 MACE-POLAR."
        )
    return calculator


def _source_hashes(manifest_path: Path) -> dict[str, str]:
    paths = {
        "static_mep_runner": Path(__file__),
        "static_mep_probe_module": (
            REPO_ROOT
            / "maple/function/calculator/extra_correction/implicit/route2_static_surface_mep.py"
        ),
        "mace_adapter": REPO_ROOT / "maple/function/calculator/mace/_macepol_calculator.py",
        "point_multipole_kernel": (
            REPO_ROOT / "maple/function/calculator/extra_correction/implicit/gto_density.py"
        ),
        "smd_radius_policy": (
            REPO_ROOT / "maple/function/calculator/extra_correction/implicit/smd_cds.py"
        ),
        "route2_profiles": REPO_ROOT / "maple/function/route2_smd_profiles.py",
        "qm_checkpoint_runner": QM_CHECKPOINT_RUNNER,
        "qm_surface_mep_runner": QM_MEP_RUNNER,
        "manifest": manifest_path,
    }
    return {name: sha256_file(path) for name, path in paths.items()}


def _run_lock(
    *,
    manifest_path: Path,
    manifest: dict[str, object],
    execution_git_head: str,
    qm_python: Path,
) -> dict[str, object]:
    records = manifest["locked_records"]
    assert isinstance(records, list)
    return {
        "artifact": ARTIFACT,
        "schema_version": 1,
        "execution_git_head": execution_git_head,
        "manifest_path": str(manifest_path.relative_to(REPO_ROOT)),
        "manifest_sha256": sha256_file(manifest_path),
        "record_identity_sha256": manifest["locked_record_identity_sha256"],
        "compound_ids": [record["compound_id"] for record in records],
        "water_profile_for_probe_radii": DDPCM_MULTISOLVENT_SMD_DIRECT_PCM_V2_PROFILE,
        "probe_clearance_angstrom": CLEARANCE_ANGSTROM,
        "qm_python": str(qm_python),
        "source_files_sha256": _source_hashes(manifest_path),
        "experimental_solvation_labels_read": False,
    }


def _establish_or_validate_lock(work_dir: Path, lock: dict[str, object]) -> None:
    path = work_dir / "run-lock.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != lock:
            raise RuntimeError("Existing source-oracle work directory has a different immutable lock.")
        return
    write_json_atomic(path, lock)


def _record_needs_run(path: Path, *, retry_failures: bool) -> bool:
    if not path.exists():
        return True
    record = json.loads(path.read_text(encoding="utf-8"))
    status = record.get("status")
    if status == "success":
        return False
    if status == "failure":
        return retry_failures
    raise RuntimeError(f"Unexpected persisted source-oracle status: {status!r}.")


def _attempt_directory(record_dir: Path) -> Path:
    for index in range(1, 1000):
        candidate = record_dir / f"qm-attempt-{index:03d}"
        if not candidate.exists():
            return candidate
    raise RuntimeError("Exhausted source-oracle QM attempt directory names.")


def _run_qm_reference(
    *,
    qm_python: Path,
    mol2_path: Path,
    record: dict[str, object],
    surface_points_bohr: np.ndarray,
    record_dir: Path,
) -> tuple[Path, Path]:
    """Create one checkpoint and its static MEP on the already frozen points."""

    qm_dir = _attempt_directory(record_dir)
    surface_path = qm_dir / "surface.npz"
    qm_dir.mkdir(parents=True)
    np.savez(
        surface_path,
        surface_points_bohr=np.asarray(surface_points_bohr, dtype=float),
    )
    checkpoint_command = [
        str(qm_python),
        str(QM_CHECKPOINT_RUNNER),
        "--mol2", str(mol2_path),
        "--compound-id", str(record["compound_id"]),
        "--molecule-name", str(record["name"]),
        "--output-dir", str(qm_dir / "gas"),
    ]
    checkpoint = subprocess.run(
        checkpoint_command,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )
    (qm_dir / "gas.stdout.txt").write_text(checkpoint.stdout, encoding="utf-8")
    (qm_dir / "gas.stderr.txt").write_text(checkpoint.stderr, encoding="utf-8")
    if checkpoint.returncode:
        raise RuntimeError(
            f"QM gas checkpoint failed for {record['compound_id']}; see {qm_dir}."
        )
    mep_path = qm_dir / "qm-static-mep.npz"
    mep_command = [
        str(qm_python),
        str(QM_MEP_RUNNER),
        "--checkpoint", str(qm_dir / "gas" / "gas.chk"),
        "--surface", str(surface_path),
        "--output", str(mep_path),
    ]
    mep = subprocess.run(
        mep_command,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )
    (qm_dir / "mep.stdout.txt").write_text(mep.stdout, encoding="utf-8")
    (qm_dir / "mep.stderr.txt").write_text(mep.stderr, encoding="utf-8")
    if mep.returncode:
        raise RuntimeError(
            f"QM static MEP failed for {record['compound_id']}; see {qm_dir}."
        )
    return qm_dir / "gas" / "gas.json", mep_path


def _relative_l2(candidate: np.ndarray, reference: np.ndarray) -> float:
    denominator = max(float(np.linalg.norm(reference)), 1.0e-30)
    return float(np.linalg.norm(candidate - reference) / denominator)


def _evaluate_record(
    *,
    calculator,
    qm_python: Path,
    record: dict[str, object],
    mol2_root: Path,
    record_dir: Path,
) -> dict[str, object]:
    atoms = _atoms(record, mol2_root)
    symbols = atoms.get_chemical_symbols()
    radii = route2_coulomb_radii(
        symbols,
        solvent=WATER_SOLVENT,
        profile=DDPCM_MULTISOLVENT_SMD_DIRECT_PCM_V2_PROFILE,
    )
    surface = build_route2_smd_exterior_probe_surface(
        np.asarray(atoms.get_positions(), dtype=float),
        radii,
        clearance_angstrom=CLEARANCE_ANGSTROM,
    )
    mol2_path = _resolve_mol2_path(mol2_root, record)
    gas_json_path, qm_mep_path = _run_qm_reference(
        qm_python=qm_python,
        mol2_path=mol2_path,
        record=record,
        surface_points_bohr=surface.surface_points_bohr,
        record_dir=record_dir,
    )
    gas = json.loads(gas_json_path.read_text(encoding="utf-8"))
    with np.load(qm_mep_path) as qm:
        qm_potential = np.asarray(qm["surface_potential_hartree_per_e"], dtype=float)
        qm_charge = float(np.asarray(qm["total_charge_e"]))
        qm_dipole = np.asarray(qm["molecular_dipole_e_angstrom"], dtype=float)
    if qm_potential.shape != (surface.retained_point_count,):
        raise RuntimeError("QM MEP length does not match the frozen probe surface.")

    mace_state, _ = calculator.polar_state(atoms)
    mace_density = np.asarray(mace_state.density_coefficients, dtype=float)
    mace_potential = point_multipole_potential(
        surface.surface_points_bohr,
        np.asarray(atoms.get_positions(), dtype=float),
        mace_density,
    )
    mace_charges, _ = cartesian_multipoles(mace_density)
    metrics = route2_weighted_surface_mep_discrepancy(
        mace_potential,
        qm_potential,
        surface.quadrature_weights,
    )
    return {
        "compound_id": record["compound_id"],
        "name": record["name"],
        "chemical_class": record["chemical_class"],
        "functional_group": record["functional_group"],
        "atom_count": len(atoms),
        "mol2_sha256": record["mol2_sha256"],
        "surface": {
            "candidate_point_count": surface.candidate_count,
            "retained_exterior_point_count": surface.retained_point_count,
            "clearance_angstrom": surface.clearance_angstrom,
            "smd_coulomb_radii_angstrom": np.asarray(radii, dtype=float).tolist(),
            "surface_points_sha256": hashlib.sha256(
                np.ascontiguousarray(surface.surface_points_bohr).view(np.uint8)
            ).hexdigest(),
        },
        "qm_reference": {
            "gas_checkpoint_json_sha256": sha256_file(gas_json_path),
            "static_mep_npz_sha256": sha256_file(qm_mep_path),
            "method": gas["system"]["method"],
            "basis": gas["system"]["basis"],
            "pyscf_version": gas["runtime"]["pyscf"],
            "total_charge_e": qm_charge,
            "molecular_dipole_e_angstrom": qm_dipole.tolist(),
        },
        "mace_zero_field_source": {
            "density_coefficients_sha256": hashlib.sha256(
                np.ascontiguousarray(mace_density).view(np.uint8)
            ).hexdigest(),
            "total_charge_e": float(np.sum(mace_charges)),
            "molecular_dipole_e_angstrom": np.asarray(
                mace_state.dipole_e_angstrom, dtype=float
            ).tolist(),
        },
        "static_surface_mep_metrics": metrics,
        "mace_minus_qm_dipole_relative_l2": _relative_l2(
            np.asarray(mace_state.dipole_e_angstrom, dtype=float), qm_dipole
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    manifest_path = args.manifest.resolve()
    manifest = load_static_mep_manifest(manifest_path)
    work_dir = _require_private_path(args.work_dir, label="Source-oracle work directory")
    qm_python = args.qm_python.resolve()
    if not qm_python.is_file():
        raise FileNotFoundError(f"QM interpreter does not exist: {qm_python}")
    mol2_root = args.mol2_root.resolve()
    records = manifest["locked_records"]
    assert isinstance(records, list)
    for record in records:
        assert isinstance(record, dict)
        _resolve_mol2_path(mol2_root, record)

    execution_head = _require_clean_source()
    work_dir.mkdir(parents=True, exist_ok=True)
    lock = _run_lock(
        manifest_path=manifest_path,
        manifest=manifest,
        execution_git_head=execution_head,
        qm_python=qm_python,
    )
    _establish_or_validate_lock(work_dir, lock)
    records_dir = work_dir / "records"
    records_dir.mkdir(exist_ok=True)
    pending = [
        record
        for record in records
        if _record_needs_run(
            records_dir / f"{record['compound_id']}.json",
            retry_failures=bool(args.retry_failures),
        )
    ]

    calculator = None
    if pending:
        first_record = pending[0]
        assert isinstance(first_record, dict)
        calculator = _load_route2_float64_source_calculator(
            _atoms(first_record, mol2_root), work_dir
        )

    for ordinal, record in enumerate(records, start=1):
        assert isinstance(record, dict)
        record_path = records_dir / f"{record['compound_id']}.json"
        if not _record_needs_run(record_path, retry_failures=bool(args.retry_failures)):
            continue
        if calculator is None:
            raise AssertionError("A MACE-POLAR calculator is required for a pending record.")
        record_dir = records_dir / str(record["compound_id"])
        try:
            result = _evaluate_record(
                calculator=calculator,
                qm_python=qm_python,
                record=record,
                mol2_root=mol2_root,
                record_dir=record_dir,
            )
            result["status"] = "success"
            print(
                f"{ordinal:02d}/12 {str(record['name']):<24.24} "
                f"relL2={result['static_surface_mep_metrics']['weighted_relative_l2']:.6f}",
                flush=True,
            )
        except Exception as exc:
            result = {
                "compound_id": record["compound_id"],
                "name": record["name"],
                "status": "failure",
                "exception_class": type(exc).__name__,
                "reason": str(exc),
            }
            print(f"{ordinal:02d}/12 {record['name']} FAILURE: {exc}", flush=True)
        write_json_atomic(record_path, result)

    completed = [
        json.loads((records_dir / f"{record['compound_id']}.json").read_text(encoding="utf-8"))
        for record in records
    ]
    failures = [record for record in completed if record.get("status") != "success"]
    success = [record for record in completed if record.get("status") == "success"]
    metrics: dict[str, float | int] | None
    if failures:
        status = "incomplete-computational-failure"
        metrics = None
        exit_code = 2
    else:
        values = np.asarray(
            [
                row["static_surface_mep_metrics"]["weighted_relative_l2"]
                for row in success
            ],
            dtype=float,
        )
        metrics = {
            "record_count": int(values.size),
            "mean_weighted_relative_l2": float(np.mean(values)),
            "maximum_weighted_relative_l2": float(np.max(values)),
            "minimum_weighted_relative_l2": float(np.min(values)),
        }
        status = "complete-source-diagnostic-no-acceptance-threshold"
        exit_code = 0
    payload = {
        "artifact": ARTIFACT,
        "schema_version": 1,
        "status": status,
        "claim_boundary": manifest["claim_boundary"],
        "run_lock": lock,
        "records_required": len(records),
        "success_count": len(success),
        "failure_count": len(failures),
        "failures": failures,
        "aggregate_metrics": metrics,
        "records": completed,
        "runtime": (
            None
            if calculator is None
            else {
                "mace_dtype": str(calculator.dtype),
                "mace_polar_checkpoint": dict(
                    calculator.mace_polar_checkpoint_provenance
                ),
                "solvent_correction_evaluate_called": False,
            }
        ),
        "disposition": {
            "experimental_solvation_labels_read": False,
            "continuum_or_solvation_energy_invoked": False,
            "force_or_pes_claim": False,
            "v0_permanent_reference_admitted": False,
            "interpretation": (
                "This is a source-representation diagnostic only. Its result must not "
                "be used to choose a continuum equation, radius, scale, or V0 permanent "
                "reference; the next independent nonuniform-response and stationary-source "
                "gates remain required."
            ),
        },
    }
    write_json_atomic(work_dir / "summary.json", payload)
    print(f"Static source-MEP status={status}; wrote {work_dir / 'summary.json'}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
