#!/usr/bin/env python3
"""Generate and verify independent Amber-GB and APBS parity artifacts."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from benchmark_core import (  # noqa: E402
    load_json,
    load_protocol,
    sha256_file,
    write_json_atomic,
)
from maple.function.calculator.extra_correction.implicit.apbs_pb import (  # noqa: E402
    APBSLPB,
    _apbs_version,
    parse_apbs_print_energy,
)
from maple.function.calculator.extra_correction.implicit.openmm_gb import (  # noqa: E402
    KJ_PER_MOL_PER_HARTREE,
    OpenMMGB,
)
from maple.function.read.filereader.mol2_reader import MOL2Reader  # noqa: E402


KCAL_PER_HARTREE = KJ_PER_MOL_PER_HARTREE / 4.184


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _protocol_path(protocol: dict[str, Any], key: str) -> Path:
    configured = Path(protocol["provider_parity"][key])
    return configured if configured.is_absolute() else REPOSITORY_ROOT / configured


def _load_reference_manifest(path: Path, expected_provider: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Independent {expected_provider} reference manifest is absent: {path}. "
            "The parity gate stays closed until Stage 2 creates and reviews this corpus."
        )
    manifest = load_json(path)
    if manifest.get("schema_version") != 1:
        raise ValueError(f"Unsupported {expected_provider} reference manifest schema.")
    if str(manifest.get("provider", "")).lower() != expected_provider.lower():
        raise ValueError(f"Reference manifest is not declared as provider={expected_provider}.")
    if not isinstance(manifest.get("cases"), list) or not manifest["cases"]:
        raise ValueError(f"{expected_provider} reference manifest has no cases.")
    return manifest


def _case_mol2(manifest_path: Path, case: dict[str, Any]) -> Path:
    path = Path(case["mol2"])
    path = path if path.is_absolute() else manifest_path.parent / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Parity case MOL2 is absent: {path}")
    expected = str(case.get("mol2_sha256", ""))
    observed = sha256_file(path)
    if not expected or observed != expected:
        raise ValueError(
            f"Parity MOL2 hash mismatch for {case.get('case_id')}: expected {expected}, "
            f"observed {observed}."
        )
    return path


def _case_pqr(manifest_path: Path, case: dict[str, Any]) -> Path:
    path = Path(case["pqr"])
    path = path if path.is_absolute() else manifest_path.parent / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Parity case PQR is absent: {path}")
    expected = str(case.get("pqr_sha256", ""))
    observed = sha256_file(path)
    if not expected or observed != expected:
        raise ValueError(
            f"Parity PQR hash mismatch for {case.get('case_id')}: expected {expected}, "
            f"observed {observed}."
        )
    return path


def _render_official_born_input(pqr_name: str, grid_points: int, grid_spacing: float) -> str:
    """Render the APBS documentation's canonical Born-ion calculation."""
    def section(name: str, solvent_dielectric: float) -> str:
        return f"""\
elec name {name}
  mg-manual
  dime {grid_points} {grid_points} {grid_points}
  nlev 4
  grid {grid_spacing:g} {grid_spacing:g} {grid_spacing:g}
  gcent mol 1
  mol 1
  lpbe
  bcfl mdh
  pdie 1.0
  sdie {solvent_dielectric:g}
  chgm spl2
  srfm mol
  srad 1.4
  swin 0.3
  sdens 10.0
  temp 298.15
  calcenergy total
  calcforce no
end"""
    return f"""\
read
  mol pqr {pqr_name}
end
{section("solv", 78.54)}
{section("ref", 1.0)}
print energy solv - ref end
quit
"""


def _run_official_born(
    *,
    pqr_path: Path,
    executable: str,
    grid_points: int,
    grid_spacing: float,
    audit_dir: Path,
    timeout: float,
) -> tuple[float, dict[str, Any]]:
    resolved = shutil.which(executable)
    if resolved is None:
        raise ImportError(f"The APBS executable '{executable}' was not found on PATH.")
    audit_dir.mkdir(parents=True, exist_ok=True)
    copied_pqr = audit_dir / pqr_path.name
    shutil.copy2(pqr_path, copied_pqr)
    input_path = audit_dir / "apbs.in"
    input_path.write_text(
        _render_official_born_input(copied_pqr.name, grid_points, grid_spacing),
        encoding="utf-8",
    )
    completed = subprocess.run(
        [resolved, input_path.name],
        cwd=audit_dir,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    (audit_dir / "apbs.stdout.log").write_text(completed.stdout, encoding="utf-8")
    (audit_dir / "apbs.stderr.log").write_text(completed.stderr, encoding="utf-8")
    write_json_atomic(
        audit_dir / "apbs.command.json",
        {
            "command": [resolved, input_path.name],
            "returncode": completed.returncode,
            "executable_sha256": sha256_file(Path(resolved)),
        },
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"APBS exited with code {completed.returncode}; inspect {audit_dir}."
        )
    energy_kj_mol = parse_apbs_print_energy(completed.stdout, "ELEC")
    provenance = {
        "provider": "apbs",
        "provider_version": _apbs_version(resolved, timeout, cwd=audit_dir),
        "executable": resolved,
        "executable_sha256": sha256_file(Path(resolved)),
        "control": "official-born-ion",
        "source": "APBS Solvation energies documentation, Born ion example",
        "grid_points": grid_points,
        "grid_spacing_angstrom": grid_spacing,
        "solute_dielectric": 1.0,
        "solvent_dielectric": 78.54,
    }
    return energy_kj_mol, provenance


def amber_gb(args: argparse.Namespace) -> None:
    protocol, fingerprint = load_protocol(args.protocol)
    manifest_path = (
        Path(args.reference_manifest).resolve()
        if args.reference_manifest
        else _protocol_path(protocol, "amber_gb_reference_manifest")
    )
    manifest = _load_reference_manifest(manifest_path, "amber")
    required_amber = str(protocol["providers"]["ambertools"]["required_version"])
    if str(manifest.get("provider_version")) != required_amber:
        raise ValueError(
            f"Protocol requires Amber reference {required_amber}, "
            f"manifest declares {manifest.get('provider_version')}."
        )
    openmm_version = importlib.metadata.version("openmm")
    required_openmm = str(protocol["providers"]["openmm"]["required_version"])
    if openmm_version != required_openmm:
        raise ValueError(
            f"Protocol requires OpenMM {required_openmm}, observed {openmm_version}."
        )
    required_models = list(protocol["methods"]["gb_models"])
    required_components = list(protocol["provider_parity"]["amber_required_components"])
    output_dir = Path(args.output_dir).resolve()
    records: list[dict[str, Any]] = []
    for case in manifest["cases"]:
        case_id = str(case.get("case_id", ""))
        if not case_id:
            raise ValueError("Amber reference case is missing case_id.")
        mol2_path = _case_mol2(manifest_path, case)
        atoms = MOL2Reader(
            str(mol2_path), charge=int(case.get("charge", 0)), mult=int(case.get("multiplicity", 1))
        )
        charges = np.asarray(case.get("charges_e"), dtype=np.float64)
        if charges.shape != (len(atoms),) or not np.isfinite(charges).all():
            raise ValueError(f"Amber reference case {case_id} has invalid charges_e.")
        references = case.get("models")
        if not isinstance(references, dict) or set(references) != set(required_models):
            raise ValueError(
                f"Amber reference case {case_id} must contain exactly: {', '.join(required_models)}."
            )
        for model in required_models:
            reference = references[model]
            if set(reference) < set(required_components):
                raise ValueError(
                    f"Amber reference {case_id}/{model} lacks polar/nonpolar/total components."
                )
            base = {
                "case_id": case_id,
                "model": model,
                "profile": protocol["providers"]["openmm"]["models"][model]["profile"],
                "mol2_sha256": case["mol2_sha256"],
                "reference_provider": manifest["provider"],
                "reference_provider_version": manifest.get("provider_version"),
                "reference_kcal_mol": {
                    component: float(reference[component]) for component in required_components
                },
            }
            try:
                polar_provider = OpenMMGB(
                    atoms,
                    charges,
                    model=model,
                    nonpolar="none",
                    platform=protocol["methods"]["openmm_platform"],
                )
                complete_provider = OpenMMGB(
                    atoms,
                    charges,
                    model=model,
                    nonpolar=protocol["provider_parity"]["amber_complete_parity_nonpolar"],
                    platform=protocol["methods"]["openmm_platform"],
                )
                polar_result = polar_provider.evaluate(atoms, need_forces=True)
                complete_result = complete_provider.evaluate(atoms, need_forces=True)
                actual = {
                    "polar": polar_result.energy_hartree * KCAL_PER_HARTREE,
                    "nonpolar_lcpo": complete_result.components_hartree["nonpolar"] * KCAL_PER_HARTREE,
                    "total_lcpo": complete_result.energy_hartree * KCAL_PER_HARTREE,
                }
                force_actual = {
                    "polar": polar_result.forces_hartree_per_angstrom * KCAL_PER_HARTREE,
                    "total_lcpo": (
                        complete_result.forces_hartree_per_angstrom * KCAL_PER_HARTREE
                    ),
                }
                force_reference = {
                    "polar": np.asarray(
                        reference["polar_force_kcal_mol_angstrom"], dtype=np.float64
                    ),
                    "total_lcpo": np.asarray(
                        reference["total_lcpo_force_kcal_mol_angstrom"], dtype=np.float64
                    ),
                }
                force_difference = {
                    component: force_actual[component] - force_reference[component]
                    for component in protocol["provider_parity"][
                        "amber_required_force_components"
                    ]
                }
                base.update(
                    status="success",
                    maple_kcal_mol=actual,
                    signed_difference_kcal_mol={
                        component: actual[component] - float(reference[component])
                        for component in required_components
                    },
                    reference_force_kcal_mol_angstrom={
                        component: force_reference[component].tolist()
                        for component in force_reference
                    },
                    maple_force_kcal_mol_angstrom={
                        component: force_actual[component].tolist()
                        for component in force_actual
                    },
                    force_difference_kcal_mol_angstrom={
                        component: force_difference[component].tolist()
                        for component in force_difference
                    },
                    force_difference_metrics={
                        component: {
                            "max_abs": float(np.max(np.abs(force_difference[component]))),
                            "rms": float(np.sqrt(np.mean(force_difference[component] ** 2))),
                        }
                        for component in force_difference
                    },
                    maple_provenance={
                        "polar": polar_result.provenance,
                        "complete_lcpo": complete_result.provenance,
                    },
                )
            except Exception as exc:
                base.update(
                    status="failure",
                    failure={"exception_class": type(exc).__name__, "reason": str(exc)},
                )
            records.append(base)
    artifact = {
        "schema_version": 1,
        "artifact_type": "amber-gb-parity",
        "protocol_id": protocol["protocol_id"],
        "protocol_fingerprint": fingerprint,
        "reference_manifest": str(manifest_path),
        "reference_manifest_sha256": sha256_file(manifest_path),
        "provider_versions": {"amber": required_amber, "openmm": openmm_version},
        "created_at_utc": utc_now(),
        "records": records,
        "tolerance_status": "unverified-until-reviewed-tolerances-are-frozen",
    }
    write_json_atomic(output_dir / "results.json", artifact)
    print(f"Wrote {len(records)} Amber/OpenMM parity records to {output_dir / 'results.json'}.")


def apbs_grid(args: argparse.Namespace) -> None:
    protocol, fingerprint = load_protocol(args.protocol)
    manifest_path = (
        Path(args.reference_manifest).resolve()
        if args.reference_manifest
        else _protocol_path(protocol, "apbs_reference_manifest")
    )
    manifest = _load_reference_manifest(manifest_path, "apbs")
    resolved_apbs = shutil.which(args.apbs)
    if resolved_apbs is None:
        raise ImportError(f"The APBS executable '{args.apbs}' was not found on PATH.")
    observed_apbs = _apbs_version(resolved_apbs, args.timeout)
    required_apbs = str(protocol["providers"]["apbs"]["required_version"])
    if observed_apbs != required_apbs:
        raise ValueError(
            f"Protocol requires APBS {required_apbs}, observed {observed_apbs}."
        )
    if str(manifest.get("provider_version")) != required_apbs:
        raise ValueError(
            f"APBS reference manifest declares {manifest.get('provider_version')}; "
            f"protocol requires {required_apbs}."
        )
    output_dir = Path(args.output_dir).resolve()
    records: list[dict[str, Any]] = []
    for case in manifest["cases"]:
        case_id = str(case.get("case_id", ""))
        control_kind = str(case.get("control_kind", ""))
        if control_kind not in set(protocol["provider_parity"]["apbs_required_controls"]):
            raise ValueError(f"APBS case {case_id!r} has an undeclared control_kind.")
        if not case_id:
            raise ValueError("APBS reference case is missing case_id.")
        atoms = None
        charges = None
        mol2_path = None
        pqr_path = None
        if control_kind == "official-born-ion":
            pqr_path = _case_pqr(manifest_path, case)
        else:
            mol2_path = _case_mol2(manifest_path, case)
            atoms = MOL2Reader(
                str(mol2_path),
                charge=int(case.get("charge", 0)),
                mult=int(case.get("multiplicity", 1)),
            )
            charges = np.asarray(case.get("charges_e"), dtype=np.float64)
            if charges.shape != (len(atoms),) or not np.isfinite(charges).all():
                raise ValueError(f"APBS reference case {case_id} has invalid charges_e.")
        grids = case.get("grids")
        if not isinstance(grids, list) or not grids:
            raise ValueError(f"APBS reference case {case_id} has no grid sweep.")
        for grid in grids:
            grid_points = int(grid["grid_points"])
            grid_spacing = float(grid["grid_spacing_angstrom"])
            audit_dir = output_dir / "audit" / case_id / f"n{grid_points}-h{grid_spacing:g}"
            record = {
                "case_id": case_id,
                "control_kind": control_kind,
                "grid_points": grid_points,
                "grid_spacing_angstrom": grid_spacing,
                "expected_kcal_mol": case.get("expected_kcal_mol"),
            }
            if mol2_path is not None:
                record["mol2_sha256"] = case["mol2_sha256"]
            if pqr_path is not None:
                record["pqr_sha256"] = case["pqr_sha256"]
            try:
                if control_kind == "official-born-ion":
                    polar_kj_mol, provenance = _run_official_born(
                        pqr_path=pqr_path,
                        executable=resolved_apbs,
                        grid_points=grid_points,
                        grid_spacing=grid_spacing,
                        audit_dir=audit_dir,
                        timeout=args.timeout,
                    )
                    actual = {
                        "polar": polar_kj_mol / 4.184,
                        "nonpolar": 0.0,
                        "total": polar_kj_mol / 4.184,
                    }
                else:
                    provider = APBSLPB(
                        atoms,
                        charges,
                        executable=resolved_apbs,
                        grid_points=grid_points,
                        grid_spacing=grid_spacing,
                        timeout=args.timeout,
                        audit_dir=audit_dir,
                    )
                    result = provider.evaluate(atoms, need_forces=False)
                    actual = {
                        "polar": result.components_hartree["polar"] * KCAL_PER_HARTREE,
                        "nonpolar": result.components_hartree["nonpolar"] * KCAL_PER_HARTREE,
                        "total": result.energy_hartree * KCAL_PER_HARTREE,
                    }
                    provenance = result.provenance
                expected = case.get("expected_kcal_mol") or {}
                record.update(
                    status="success",
                    maple_kcal_mol=actual,
                    signed_difference_kcal_mol={
                        component: actual[component] - float(value)
                        for component, value in expected.items()
                    },
                    maple_provenance=provenance,
                )
            except Exception as exc:
                record.update(
                    status="failure",
                    failure={"exception_class": type(exc).__name__, "reason": str(exc)},
                )
            records.append(record)
    artifact = {
        "schema_version": 1,
        "artifact_type": "apbs-grid-parity",
        "protocol_id": protocol["protocol_id"],
        "protocol_fingerprint": fingerprint,
        "reference_manifest": str(manifest_path),
        "reference_manifest_sha256": sha256_file(manifest_path),
        "provider_version": observed_apbs,
        "provider_executable": resolved_apbs,
        "provider_executable_sha256": sha256_file(Path(resolved_apbs)),
        "created_at_utc": utc_now(),
        "records": records,
        "tolerance_status": "unverified-until-reviewed-tolerances-are-frozen",
    }
    write_json_atomic(output_dir / "results.json", artifact)
    print(f"Wrote {len(records)} APBS grid records to {output_dir / 'results.json'}.")


def _require_number(mapping: dict[str, Any], key: str) -> float:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise ValueError(f"Parity tolerance {key} must be a non-negative number.")
    return float(value)


def verify(args: argparse.Namespace) -> None:
    protocol, fingerprint = load_protocol(args.protocol)
    artifact_dir = Path(args.artifact_dir).resolve()
    tolerance_path = (
        Path(args.tolerances).resolve()
        if args.tolerances
        else _protocol_path(protocol, "tolerances_file")
    )
    if not tolerance_path.is_file():
        raise FileNotFoundError(
            f"Reviewed provider-parity tolerances are absent: {tolerance_path}. "
            "Do not guess numerical parity thresholds."
        )
    tolerances = load_json(tolerance_path)
    if tolerances.get("schema_version") != 1:
        raise ValueError("Unsupported provider-parity tolerance schema.")
    if tolerances.get("protocol_fingerprint") != fingerprint:
        raise ValueError("Provider-parity tolerances target a different protocol fingerprint.")
    if tolerances.get("review_status") != "human-reviewed-frozen":
        raise ValueError("Provider-parity tolerances are not marked human-reviewed-frozen.")

    amber_path = artifact_dir / "amber-gb-parity/results.json"
    apbs_path = artifact_dir / "apbs-grid/results.json"
    if not amber_path.is_file() or not apbs_path.is_file():
        raise FileNotFoundError(
            "Provider parity requires amber-gb-parity/results.json and apbs-grid/results.json."
        )
    amber = load_json(amber_path)
    apbs = load_json(apbs_path)
    for artifact, expected_type in (
        (amber, "amber-gb-parity"),
        (apbs, "apbs-grid-parity"),
    ):
        if (
            artifact.get("artifact_type") != expected_type
            or artifact.get("protocol_fingerprint") != fingerprint
        ):
            raise ValueError(f"Invalid or mismatched provider artifact: {expected_type}.")

    checks: list[dict[str, Any]] = []
    amber_tolerances = tolerances.get("amber_gb", {})
    required_models = set(protocol["methods"]["gb_models"])
    observed_models = {record.get("model") for record in amber["records"]}
    if observed_models != required_models:
        checks.append(
            {
                "check": "amber-model-completeness",
                "passed": False,
                "detail": f"expected={sorted(required_models)}, observed={sorted(observed_models)}",
            }
        )
    for record in amber["records"]:
        if record.get("status") != "success":
            checks.append(
                {"check": f"amber/{record.get('case_id')}/{record.get('model')}", "passed": False, "detail": record.get("failure")}
            )
            continue
        for component in protocol["provider_parity"]["amber_required_components"]:
            tolerance = _require_number(amber_tolerances, f"max_abs_{component}_kcal_mol")
            difference = abs(float(record["signed_difference_kcal_mol"][component]))
            checks.append(
                {
                    "check": f"amber/{record['case_id']}/{record['model']}/{component}",
                    "passed": difference <= tolerance,
                    "observed": difference,
                    "tolerance": tolerance,
                }
            )
        for component in protocol["provider_parity"]["amber_required_force_components"]:
            tolerance = _require_number(
                amber_tolerances,
                f"max_abs_{component}_force_kcal_mol_angstrom",
            )
            difference = float(record["force_difference_metrics"][component]["max_abs"])
            checks.append(
                {
                    "check": f"amber/{record['case_id']}/{record['model']}/{component}-force",
                    "passed": difference <= tolerance,
                    "observed": difference,
                    "tolerance": tolerance,
                }
            )

    apbs_tolerances = tolerances.get("apbs", {})
    official_tolerance = _require_number(
        apbs_tolerances, "max_abs_official_component_kcal_mol"
    )
    grid_tolerance = _require_number(
        apbs_tolerances, "max_successive_grid_total_difference_kcal_mol"
    )
    by_case: dict[str, list[dict[str, Any]]] = {}
    for record in apbs["records"]:
        by_case.setdefault(record["case_id"], []).append(record)
        if record.get("status") != "success":
            checks.append(
                {"check": f"apbs/{record.get('case_id')}/provider", "passed": False, "detail": record.get("failure")}
            )
            continue
        if record["control_kind"] == "official-born-ion":
            differences = record.get("signed_difference_kcal_mol", {})
            if not differences:
                checks.append(
                    {
                        "check": f"apbs/{record['case_id']}/official/expected-components",
                        "passed": False,
                        "detail": "official control has no signed comparison to its documented value",
                    }
                )
            for component, difference in differences.items():
                observed = abs(float(difference))
                checks.append(
                    {
                        "check": f"apbs/{record['case_id']}/official/{component}",
                        "passed": observed <= official_tolerance,
                        "observed": observed,
                        "tolerance": official_tolerance,
                    }
                )
    observed_controls = {
        record.get("control_kind") for record in apbs["records"] if record.get("status") == "success"
    }
    required_controls = set(protocol["provider_parity"]["apbs_required_controls"])
    if observed_controls != required_controls:
        checks.append(
            {
                "check": "apbs-control-completeness",
                "passed": False,
                "detail": f"expected={sorted(required_controls)}, observed={sorted(observed_controls)}",
            }
        )
    for case_id, records in by_case.items():
        successful = [record for record in records if record.get("status") == "success"]
        if not successful or successful[0]["control_kind"] != "neutral-grid-convergence":
            continue
        ordered = sorted(
            successful,
            key=lambda record: (
                float(record["grid_spacing_angstrom"]),
                -int(record["grid_points"]),
            ),
            reverse=True,
        )
        if len(ordered) < 2:
            checks.append(
                {"check": f"apbs/{case_id}/grid-count", "passed": False, "detail": "need at least two successful grids"}
            )
            continue
        difference = abs(
            float(ordered[-1]["maple_kcal_mol"]["total"])
            - float(ordered[-2]["maple_kcal_mol"]["total"])
        )
        checks.append(
            {
                "check": f"apbs/{case_id}/successive-finest-grid-total",
                "passed": difference <= grid_tolerance,
                "observed": difference,
                "tolerance": grid_tolerance,
            }
        )

    passed = bool(checks) and all(check["passed"] for check in checks)
    verification = {
        "schema_version": 1,
        "artifact_type": "provider-parity-verification",
        "protocol_id": protocol["protocol_id"],
        "protocol_fingerprint": fingerprint,
        "tolerances_sha256": sha256_file(tolerance_path),
        "amber_artifact_sha256": sha256_file(amber_path),
        "apbs_artifact_sha256": sha256_file(apbs_path),
        "passed": passed,
        "checks": checks,
    }
    output = artifact_dir / "provider-parity-verification.json"
    write_json_atomic(output, verification)
    if not passed:
        raise ValueError(f"Provider parity verification failed; inspect {output}.")
    print(f"Provider parity verification passed; evidence: {output}.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="phase", required=True)

    amber = subparsers.add_parser("amber-gb")
    amber.add_argument("--protocol", required=True)
    amber.add_argument("--output-dir", required=True)
    amber.add_argument("--reference-manifest")
    amber.set_defaults(handler=amber_gb)

    apbs = subparsers.add_parser("apbs-grid")
    apbs.add_argument("--protocol", required=True)
    apbs.add_argument("--output-dir", required=True)
    apbs.add_argument("--reference-manifest")
    apbs.add_argument("--apbs", default="apbs")
    apbs.add_argument("--timeout", type=float, default=3600.0)
    apbs.set_defaults(handler=apbs_grid)

    verification = subparsers.add_parser("verify")
    verification.add_argument("--protocol", required=True)
    verification.add_argument("--artifact-dir", required=True)
    verification.add_argument("--tolerances")
    verification.set_defaults(handler=verify)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        args.handler(args)
    except Exception as exc:
        parser.exit(2, f"error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
