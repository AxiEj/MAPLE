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

KJ_PER_KCAL = 4.184
KCAL_PER_HARTREE = KJ_PER_MOL_PER_HARTREE / KJ_PER_KCAL


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
        raise ValueError(
            f"Reference manifest is not declared as provider={expected_provider}."
        )
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


def _case_parity_targets(
    case: dict[str, Any],
    key: str,
    required: list[str],
) -> list[str]:
    values = case.get(key, required)
    if not isinstance(values, list) or not values or len(values) != len(set(values)):
        raise ValueError(
            f"Amber reference case {case.get('case_id')} has invalid {key}."
        )
    if not set(values) <= set(required):
        raise ValueError(
            f"Amber reference case {case.get('case_id')} declares unsupported {key}."
        )
    if values not in (required, ["polar"]):
        raise ValueError(
            f"Amber reference case {case.get('case_id')} must target either the "
            f"complete parity set or polar alone for {key}."
        )
    return values


def _render_official_born_input(
    pqr_name: str, grid_points: int, grid_spacing: float
) -> str:
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


def _matches_declared_model_unavailability_evidence(
    exception_class: Any,
    reason: Any,
) -> bool:
    return (
        exception_class == "NotImplementedError"
        and isinstance(reason, str)
        and all(
            marker in reason
            for marker in (
                "GBn2",
                "signed near-pair descreening branch",
                "sulfur",
                "negative screening radius",
            )
        )
    )


def _matches_declared_model_unavailability(exc: Exception) -> bool:
    return _matches_declared_model_unavailability_evidence(
        type(exc).__name__,
        str(exc),
    )


def _matches_declared_lcpo_unavailability_evidence(
    exception_class: Any,
    reason: Any,
) -> bool:
    return (
        exception_class == "ValueError"
        and isinstance(reason, str)
        and reason.startswith(
            "No LCPO parameters found for element with atomic number "
        )
    )


def _matches_declared_lcpo_unavailability(exc: Exception) -> bool:
    return _matches_declared_lcpo_unavailability_evidence(
        type(exc).__name__,
        str(exc),
    )


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
            str(mol2_path),
            charge=int(case.get("charge", 0)),
            mult=int(case.get("multiplicity", 1)),
        )
        charges = np.asarray(case.get("charges_e"), dtype=np.float64)
        if charges.shape != (len(atoms),) or not np.isfinite(charges).all():
            raise ValueError(f"Amber reference case {case_id} has invalid charges_e.")
        references = case.get("models")
        if not isinstance(references, dict) or set(references) != set(required_models):
            raise ValueError(
                f"Amber reference case {case_id} must contain exactly: {', '.join(required_models)}."
            )
        model_expectations = case.get("model_expectations", {})
        if not isinstance(model_expectations, dict) or (
            set(model_expectations) - set(required_models)
        ):
            raise ValueError(
                f"Amber reference case {case_id} has invalid model expectations."
            )
        parity_components = _case_parity_targets(
            case,
            "parity_components",
            required_components,
        )
        required_force_components = list(
            protocol["provider_parity"]["amber_required_force_components"]
        )
        parity_force_components = _case_parity_targets(
            case,
            "parity_force_components",
            required_force_components,
        )
        lcpo_expectation = case.get(
            "lcpo_expectation",
            {
                "parity_target": True,
                "openmm_supported": True,
                "reason": "complete Amber/OpenMM LCPO parity target",
            },
        )
        if not isinstance(lcpo_expectation, dict):
            raise ValueError(
                f"Amber reference case {case_id} has invalid lcpo_expectation."
            )
        if not isinstance(
            lcpo_expectation.get("parity_target"), bool
        ) or not isinstance(lcpo_expectation.get("openmm_supported"), bool):
            raise ValueError(
                f"Amber reference case {case_id} requires Boolean LCPO expectations."
            )
        lcpo_parity_target = lcpo_expectation["parity_target"]
        expected_lcpo_support = lcpo_expectation["openmm_supported"]
        if lcpo_parity_target != (parity_components == required_components):
            raise ValueError(
                f"Amber reference case {case_id} has inconsistent LCPO energy targets."
            )
        if lcpo_parity_target != (parity_force_components == required_force_components):
            raise ValueError(
                f"Amber reference case {case_id} has inconsistent LCPO force targets."
            )
        if not str(lcpo_expectation.get("reason", "")).strip():
            raise ValueError(
                f"Amber reference case {case_id} requires an LCPO expectation reason."
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
                    component: float(reference[component])
                    for component in required_components
                },
                "parity_components": parity_components,
                "parity_force_components": parity_force_components,
            }
            model_expectation = model_expectations.get(
                model,
                {
                    "openmm_supported": True,
                    "reason": "OpenMM model is expected to support this reference case.",
                },
            )
            if not isinstance(model_expectation, dict) or not isinstance(
                model_expectation.get("openmm_supported"), bool
            ):
                raise ValueError(
                    f"Amber reference case {case_id}/{model} has an invalid "
                    "OpenMM support expectation."
                )
            if not str(model_expectation.get("reason", "")).strip():
                raise ValueError(
                    f"Amber reference case {case_id}/{model} requires a model "
                    "support-expectation reason."
                )
            expected_model_support = model_expectation["openmm_supported"]
            model_observation = {
                "expected_openmm_supported": expected_model_support,
                "reason": str(model_expectation["reason"]),
            }
            base["model_observation"] = model_observation
            try:
                polar_provider = OpenMMGB(
                    atoms,
                    charges,
                    model=model,
                    nonpolar="none",
                    platform=protocol["methods"]["openmm_platform"],
                )
            except Exception as exc:
                failure_matches_expectation = (
                    not expected_model_support
                    and _matches_declared_model_unavailability(exc)
                )
                model_observation.update(
                    observed_openmm_supported=False,
                    support_matches_expectation=failure_matches_expectation,
                    failure={
                        "exception_class": type(exc).__name__,
                        "reason": str(exc),
                    },
                )
                if expected_model_support or not failure_matches_expectation:
                    base.update(
                        status="failure",
                        failure=model_observation["failure"],
                    )
                else:
                    base.update(status="expected-unavailable")
                records.append(base)
                continue
            model_observation.update(
                observed_openmm_supported=True,
                support_matches_expectation=expected_model_support,
            )
            if not expected_model_support:
                base.update(
                    status="failure",
                    failure={
                        "exception_class": "RuntimeError",
                        "reason": (
                            f"OpenMM model support for {case_id}/{model} did not "
                            "match the frozen case expectation."
                        ),
                    },
                )
                records.append(base)
                continue
            try:
                polar_result = polar_provider.evaluate(atoms, need_forces=True)
                polar_forces = polar_result.forces_hartree_per_angstrom
                if polar_forces is None:
                    raise RuntimeError(
                        f"OpenMM polar forces are absent for {case_id}/{model}."
                    )
                actual = {
                    "polar": polar_result.energy_hartree * KCAL_PER_HARTREE,
                }
                force_actual = {
                    "polar": polar_forces * KCAL_PER_HARTREE,
                }
                complete_result = None
                complete_failure = None
                try:
                    complete_provider = OpenMMGB(
                        atoms,
                        charges,
                        model=model,
                        nonpolar=protocol["provider_parity"][
                            "amber_complete_parity_nonpolar"
                        ],
                        platform=protocol["methods"]["openmm_platform"],
                    )
                    complete_result = complete_provider.evaluate(
                        atoms,
                        need_forces=True,
                    )
                    complete_forces = complete_result.forces_hartree_per_angstrom
                    if complete_forces is None:
                        raise RuntimeError(
                            f"OpenMM LCPO forces are absent for {case_id}/{model}."
                        )
                    actual.update(
                        nonpolar_lcpo=(
                            complete_result.components_hartree["nonpolar"]
                            * KCAL_PER_HARTREE
                        ),
                        total_lcpo=(complete_result.energy_hartree * KCAL_PER_HARTREE),
                    )
                    force_actual["total_lcpo"] = (
                        complete_forces * KCAL_PER_HARTREE
                    )
                except Exception as exc:
                    complete_failure = {
                        "exception_class": type(exc).__name__,
                        "reason": str(exc),
                    }
                    if (
                        not expected_lcpo_support
                        and not _matches_declared_lcpo_unavailability(exc)
                    ):
                        raise RuntimeError(
                            f"OpenMM LCPO failure for {case_id}/{model} does not "
                            "match the frozen missing-parameter boundary."
                        ) from exc
                observed_lcpo_support = complete_result is not None
                support_matches = observed_lcpo_support == expected_lcpo_support
                if not support_matches:
                    raise RuntimeError(
                        f"OpenMM LCPO support for {case_id}/{model} did not match "
                        "the frozen case expectation."
                    )
                if lcpo_parity_target and not observed_lcpo_support:
                    raise RuntimeError(
                        f"OpenMM LCPO parity target {case_id}/{model} is unavailable."
                    )
                force_reference = {
                    "polar": np.asarray(
                        reference["polar_force_kcal_mol_angstrom"], dtype=np.float64
                    ),
                    "total_lcpo": np.asarray(
                        reference["total_lcpo_force_kcal_mol_angstrom"],
                        dtype=np.float64,
                    ),
                }
                force_difference = {
                    component: force_actual[component] - force_reference[component]
                    for component in force_actual
                }
                signed_difference = {
                    component: actual[component] - float(reference[component])
                    for component in actual
                }
                force_difference_metrics = {
                    component: {
                        "max_abs": float(np.max(np.abs(force_difference[component]))),
                        "rms": float(
                            np.sqrt(np.mean(force_difference[component] ** 2))
                        ),
                    }
                    for component in force_difference
                }
                lcpo_observation = {
                    "parity_target": lcpo_parity_target,
                    "expected_openmm_supported": expected_lcpo_support,
                    "observed_openmm_supported": observed_lcpo_support,
                    "support_matches_expectation": support_matches,
                    "reason": str(lcpo_expectation["reason"]),
                }
                if observed_lcpo_support:
                    lcpo_observation.update(
                        signed_difference_kcal_mol={
                            component: signed_difference[component]
                            for component in ("nonpolar_lcpo", "total_lcpo")
                        },
                        force_difference_metrics={
                            "total_lcpo": force_difference_metrics["total_lcpo"]
                        },
                    )
                else:
                    lcpo_observation["failure"] = complete_failure
                base.update(
                    status="success",
                    maple_kcal_mol=actual,
                    signed_difference_kcal_mol=signed_difference,
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
                    force_difference_metrics=force_difference_metrics,
                    lcpo_observation=lcpo_observation,
                    maple_provenance={
                        "polar": polar_result.provenance,
                        **(
                            {"complete_lcpo": complete_result.provenance}
                            if complete_result is not None
                            else {}
                        ),
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
        "case_count": len(manifest["cases"]),
        "record_count": len(records),
        "records": records,
        "tolerance_status": "unverified-until-reviewed-tolerances-are-frozen",
    }
    write_json_atomic(output_dir / "results.json", artifact)
    print(
        f"Wrote {len(records)} Amber/OpenMM parity records to {output_dir / 'results.json'}."
    )


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
        if control_kind not in set(
            protocol["provider_parity"]["apbs_required_controls"]
        ):
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
                raise ValueError(
                    f"APBS reference case {case_id} has invalid charges_e."
                )
        grids = case.get("grids")
        if not isinstance(grids, list) or not grids:
            raise ValueError(f"APBS reference case {case_id} has no grid sweep.")
        for grid in grids:
            grid_points = int(grid["grid_points"])
            grid_spacing = float(grid["grid_spacing_angstrom"])
            audit_dir = (
                output_dir / "audit" / case_id / f"n{grid_points}-h{grid_spacing:g}"
            )
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
                    if pqr_path is None:
                        raise RuntimeError(
                            f"APBS official control {case_id} lacks a PQR path."
                        )
                    polar_kj_mol, provenance = _run_official_born(
                        pqr_path=pqr_path,
                        executable=resolved_apbs,
                        grid_points=grid_points,
                        grid_spacing=grid_spacing,
                        audit_dir=audit_dir,
                        timeout=args.timeout,
                    )
                    actual = {
                        "polar": polar_kj_mol / KJ_PER_KCAL,
                        "nonpolar": 0.0,
                        "total": polar_kj_mol / KJ_PER_KCAL,
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
                        "nonpolar": result.components_hartree["nonpolar"]
                        * KCAL_PER_HARTREE,
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


def _relative_to_repository(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def _maximum_energy_difference(
    records: list[dict[str, Any]],
    component: str,
) -> dict[str, Any]:
    candidates = [
        record
        for record in records
        if record.get("status") == "success"
        and component in record.get("parity_components", [])
    ]
    values = np.asarray(
        [
            float(record["signed_difference_kcal_mol"][component])
            for record in candidates
        ],
        dtype=np.float64,
    )
    if not len(values):
        raise ValueError(f"No Amber parity energy records target {component}.")
    index = int(np.argmax(np.abs(values)))
    record = candidates[index]
    return {
        "max_abs": float(abs(values[index])),
        "rms": float(np.sqrt(np.mean(values**2))),
        "case_id": record["case_id"],
        "model": record["model"],
    }


def _maximum_force_difference(
    records: list[dict[str, Any]],
    component: str,
) -> dict[str, Any]:
    candidates = [
        record
        for record in records
        if record.get("status") == "success"
        and component in record.get("parity_force_components", [])
    ]
    if not candidates:
        raise ValueError(f"No Amber parity force records target {component}.")
    record = max(
        candidates,
        key=lambda item: float(item["force_difference_metrics"][component]["max_abs"]),
    )
    metrics = record["force_difference_metrics"][component]
    return {
        "max_abs": float(metrics["max_abs"]),
        "record_rms": float(metrics["rms"]),
        "case_id": record["case_id"],
        "model": record["model"],
    }


def _lcpo_applicability_observations(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    by_case: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        observation = record.get("lcpo_observation", {})
        if observation.get("parity_target") is False:
            by_case.setdefault(record["case_id"], []).append(record)
    summary: dict[str, Any] = {}
    for case_id, case_records in sorted(by_case.items()):
        observations = [record["lcpo_observation"] for record in case_records]
        expected = {item["expected_openmm_supported"] for item in observations}
        observed = {item["observed_openmm_supported"] for item in observations}
        reasons = {item["reason"] for item in observations}
        if len(expected) != 1 or len(observed) != 1 or len(reasons) != 1:
            raise ValueError(
                f"Inconsistent LCPO applicability observations for {case_id}."
            )
        entry: dict[str, Any] = {
            "model_count": len(case_records),
            "expected_openmm_supported": expected.pop(),
            "observed_openmm_supported": observed.pop(),
            "all_models_match_expectation": all(
                item["support_matches_expectation"] for item in observations
            ),
            "reason": reasons.pop(),
        }
        if entry["observed_openmm_supported"]:
            entry["maximum_observed_difference"] = {
                component: max(
                    abs(float(record["signed_difference_kcal_mol"][component]))
                    for record in case_records
                )
                for component in ("nonpolar_lcpo", "total_lcpo")
            }
            entry["maximum_observed_total_lcpo_force_difference"] = max(
                float(record["force_difference_metrics"]["total_lcpo"]["max_abs"])
                for record in case_records
            )
        else:
            entry["failure_reasons"] = sorted(
                {
                    item["failure"]["reason"]
                    for item in observations
                    if item.get("failure")
                }
            )
        summary[case_id] = entry
    return summary


def _model_applicability_observations(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "case_id": record["case_id"],
            "model": record["model"],
            **record["model_observation"],
        }
        for record in records
        if record.get("model_observation", {}).get("expected_openmm_supported")
        is False
    ]


def _compact_amber_record(
    record: dict[str, Any],
    required_components: list[str],
    required_force_components: list[str],
) -> dict[str, Any]:
    compact = {
        "case_id": record["case_id"],
        "model": record["model"],
        "status": record["status"],
        "parity_components": record.get(
            "parity_components",
            required_components,
        ),
        "parity_force_components": record.get(
            "parity_force_components",
            required_force_components,
        ),
    }
    if record["status"] == "success":
        compact.update(
            signed_difference_kcal_mol=record["signed_difference_kcal_mol"],
            force_difference_metrics=record["force_difference_metrics"],
            lcpo_observation=record["lcpo_observation"],
            model_observation=record["model_observation"],
        )
    else:
        if record["status"] == "failure":
            compact["failure"] = record.get("failure")
        compact["model_observation"] = record.get("model_observation")
    return compact


def observations(args: argparse.Namespace) -> None:
    protocol, fingerprint = load_protocol(args.protocol)
    amber_path = Path(args.amber_artifact).resolve()
    apbs_path = Path(args.apbs_artifact).resolve()
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
            raise ValueError(
                f"Invalid or mismatched provider artifact: {expected_type}."
            )

    amber_records = list(amber["records"])
    required_components = list(protocol["provider_parity"]["amber_required_components"])
    required_force_components = list(
        protocol["provider_parity"]["amber_required_force_components"]
    )
    compact_records = [
        _compact_amber_record(
            record,
            required_components,
            required_force_components,
        )
        for record in amber_records
    ]
    records_by_case: dict[str, list[dict[str, Any]]] = {}
    for record in amber_records:
        records_by_case.setdefault(record["case_id"], []).append(record)
    full_lcpo_cases = {
        case_id
        for case_id, case_records in records_by_case.items()
        if all(
            record.get("status") == "success"
            and record.get("parity_components", required_components)
            == required_components
            for record in case_records
        )
    }
    polar_only_cases = {
        case_id
        for case_id, case_records in records_by_case.items()
        if any(record.get("status") == "success" for record in case_records)
        and all(
            record.get("parity_components", required_components) == ["polar"]
            for record in case_records
            if record.get("status") == "success"
        )
    }

    official_records = [
        record
        for record in apbs["records"]
        if record.get("control_kind") == "official-born-ion"
        and record.get("status") == "success"
    ]
    if len(official_records) != 1:
        raise ValueError("APBS observations require one successful official Born ion.")
    official = official_records[0]
    expected_polar = float(official["expected_kcal_mol"]["polar"])
    observed_polar = float(official["maple_kcal_mol"]["polar"])
    neutral_grid: dict[str, Any] = {}
    neutral_case_ids = sorted(
        {
            record["case_id"]
            for record in apbs["records"]
            if record.get("control_kind") == "neutral-grid-convergence"
        }
    )
    for case_id in neutral_case_ids:
        records = sorted(
            (
                record
                for record in apbs["records"]
                if record["case_id"] == case_id and record.get("status") == "success"
            ),
            key=lambda record: float(record["grid_spacing_angstrom"]),
            reverse=True,
        )
        if len(records) < 2:
            raise ValueError(
                f"APBS neutral grid case {case_id} has fewer than two records."
            )
        finest = sorted(
            records, key=lambda record: float(record["grid_spacing_angstrom"])
        )[:2]
        neutral_grid[case_id] = {
            "records": [
                {
                    "grid_points": int(record["grid_points"]),
                    "grid_spacing_angstrom": float(record["grid_spacing_angstrom"]),
                    "polar_kcal_mol": float(record["maple_kcal_mol"]["polar"]),
                    "nonpolar_kcal_mol": float(record["maple_kcal_mol"]["nonpolar"]),
                    "total_kcal_mol": float(record["maple_kcal_mol"]["total"]),
                }
                for record in records
            ],
            "finest_pair_spacing_angstrom": [
                float(record["grid_spacing_angstrom"]) for record in finest
            ],
            "successive_finest_total_difference_kcal_mol": abs(
                float(finest[0]["maple_kcal_mol"]["total"])
                - float(finest[1]["maple_kcal_mol"]["total"])
            ),
        }

    artifact = {
        "schema_version": 1,
        "artifact_type": "provider-parity-observations",
        "protocol_id": protocol["protocol_id"],
        "protocol_fingerprint": fingerprint,
        "evidence_status": "measured-awaiting-independent-tolerance-review",
        "providers": {
            "amber": amber["provider_versions"]["amber"],
            "openmm": amber["provider_versions"]["openmm"],
            "apbs": apbs["provider_version"],
        },
        "source_artifacts": {
            "amber_openmm_results": _relative_to_repository(amber_path),
            "amber_openmm_results_sha256": sha256_file(amber_path),
            "amber_reference_manifest_sha256": amber["reference_manifest_sha256"],
            "apbs_grid_results": _relative_to_repository(apbs_path),
            "apbs_grid_results_sha256": sha256_file(apbs_path),
            "apbs_reference_manifest_sha256": apbs["reference_manifest_sha256"],
            "apbs_executable_sha256": apbs["provider_executable_sha256"],
        },
        "amber_openmm": {
            "all_successful": all(
                record.get("status") in {"success", "expected-unavailable"}
                and record.get("model_observation", {}).get(
                    "support_matches_expectation",
                    record.get("status") == "success",
                )
                for record in amber_records
            ),
            "case_count": int(amber["case_count"]),
            "record_count": len(amber_records),
            "supported_record_count": sum(
                record.get("status") == "success" for record in amber_records
            ),
            "expected_unavailable_record_count": sum(
                record.get("status") == "expected-unavailable"
                for record in amber_records
            ),
            "full_lcpo_parity_case_count": len(full_lcpo_cases),
            "polar_only_case_count": len(polar_only_cases),
            "energy_difference_kcal_mol": {
                component: _maximum_energy_difference(
                    amber_records,
                    component,
                )
                for component in required_components
            },
            "force_difference_kcal_mol_angstrom": {
                component: _maximum_force_difference(
                    amber_records,
                    component,
                )
                for component in required_force_components
            },
            "lcpo_applicability_observations": _lcpo_applicability_observations(
                amber_records
            ),
            "model_applicability_observations": _model_applicability_observations(
                amber_records
            ),
            "records": compact_records,
        },
        "apbs": {
            "all_successful": all(
                record.get("status") == "success" for record in apbs["records"]
            ),
            "record_count": len(apbs["records"]),
            "official_born": {
                "grid_points": int(official["grid_points"]),
                "grid_spacing_angstrom": float(official["grid_spacing_angstrom"]),
                "documented_kj_mol": expected_polar * KJ_PER_KCAL,
                "observed_kj_mol": observed_polar * KJ_PER_KCAL,
                "absolute_difference_kcal_mol": abs(observed_polar - expected_polar),
            },
            "neutral_grid": neutral_grid,
        },
    }
    write_json_atomic(args.output, artifact)
    print(f"Wrote provider observations to {Path(args.output).resolve()}.")


def _require_number(mapping: dict[str, Any], key: str) -> float:
    value = mapping.get(key)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not np.isfinite(value)
        or value < 0
    ):
        raise ValueError(
            f"Parity tolerance {key} must be a finite non-negative number."
        )
    return float(value)


def _finite_float(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not np.isfinite(value)
    ):
        raise ValueError(f"{label} must be a finite number.")
    return float(value)


def _finite_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer.")
    return value


def _finite_mapping(
    value: Any,
    required_keys: set[str],
    label: str,
) -> dict[str, float]:
    if not isinstance(value, dict) or set(value) != required_keys:
        raise ValueError(
            f"{label} must contain exactly {sorted(required_keys)}."
        )
    return {
        key: _finite_float(value[key], f"{label}.{key}")
        for key in sorted(required_keys)
    }


def _finite_force_array(value: Any, atom_count: int, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (atom_count, 3) or not np.isfinite(array).all():
        raise ValueError(
            f"{label} must be a finite ({atom_count}, 3) force array."
        )
    return array


def _assert_close(
    actual: Any,
    expected: Any,
    label: str,
    *,
    atol: float = 1.0e-12,
) -> None:
    actual_array = np.asarray(actual, dtype=np.float64)
    expected_array = np.asarray(expected, dtype=np.float64)
    if (
        actual_array.shape != expected_array.shape
        or not np.isfinite(actual_array).all()
        or not np.isfinite(expected_array).all()
        or not np.allclose(actual_array, expected_array, rtol=0.0, atol=atol)
    ):
        raise ValueError(f"{label} does not match its independently recomputed value.")


def _contract_path(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must name a pinned artifact.")
    path = Path(value)
    return path.resolve() if path.is_absolute() else (REPOSITORY_ROOT / path).resolve()


def _require_artifact_hash(path: Path, expected: Any, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} is absent: {path}.")
    observed = sha256_file(path)
    if not isinstance(expected, str) or observed != expected:
        raise ValueError(
            f"{label} artifact hash mismatch: expected {expected}, observed {observed}."
        )


def _load_review_chain(
    *,
    protocol: dict[str, Any],
    fingerprint: str,
    tolerances: dict[str, Any],
    amber_path: Path,
    apbs_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    proposal_path = _contract_path(
        tolerances.get("reviewed_proposal_artifact"),
        "reviewed_proposal_artifact",
    )
    _require_artifact_hash(
        proposal_path,
        tolerances.get("reviewed_proposal_sha256"),
        "Reviewed proposal",
    )
    proposal = load_json(proposal_path)
    if proposal.get("review_status") != "proposed-awaiting-independent-review":
        raise ValueError("Reviewed proposal is not the immutable pre-freeze proposal.")
    for key, value in proposal.items():
        if key != "review_status" and tolerances.get(key) != value:
            raise ValueError(
                "Frozen provider-parity contract changed reviewed proposal "
                f"field {key}."
            )
    if (
        proposal.get("protocol_id") != protocol["protocol_id"]
        or proposal.get("protocol_fingerprint") != fingerprint
    ):
        raise ValueError("Reviewed proposal targets a different protocol.")

    observations_path = _contract_path(
        proposal.get("evidence_artifact"),
        "evidence_artifact",
    )
    _require_artifact_hash(
        observations_path,
        proposal.get("evidence_artifact_sha256"),
        "Reviewed observations",
    )
    observations = load_json(observations_path)
    if (
        observations.get("schema_version") != 1
        or observations.get("artifact_type") != "provider-parity-observations"
        or observations.get("protocol_id") != protocol["protocol_id"]
        or observations.get("protocol_fingerprint") != fingerprint
    ):
        raise ValueError("Reviewed provider-parity observations are invalid.")
    sources = observations.get("source_artifacts")
    if not isinstance(sources, dict):
        raise ValueError("Reviewed provider-parity observations lack source artifacts.")
    _require_artifact_hash(
        amber_path,
        sources.get("amber_openmm_results_sha256"),
        "Amber/OpenMM",
    )
    _require_artifact_hash(
        apbs_path,
        sources.get("apbs_grid_results_sha256"),
        "APBS",
    )

    amber_manifest_path = _protocol_path(
        protocol,
        "amber_gb_reference_manifest",
    )
    apbs_manifest_path = _protocol_path(
        protocol,
        "apbs_reference_manifest",
    )
    amber_manifest = _load_reference_manifest(amber_manifest_path, "amber")
    apbs_manifest = _load_reference_manifest(apbs_manifest_path, "apbs")
    amber_manifest_sha = sha256_file(amber_manifest_path)
    apbs_manifest_sha = sha256_file(apbs_manifest_path)
    if (
        sources.get("amber_reference_manifest_sha256") != amber_manifest_sha
        or sources.get("apbs_reference_manifest_sha256") != apbs_manifest_sha
    ):
        raise ValueError("Reviewed observations target different reference manifests.")

    amber = load_json(amber_path)
    apbs = load_json(apbs_path)
    if amber.get("reference_manifest_sha256") != amber_manifest_sha:
        raise ValueError("Amber artifact reference-manifest hash mismatch.")
    if apbs.get("reference_manifest_sha256") != apbs_manifest_sha:
        raise ValueError("APBS artifact reference-manifest hash mismatch.")
    return observations, amber_manifest, apbs_manifest


def _reviewed_amber_records(
    observations: dict[str, Any],
    expected_keys: set[tuple[str, str]],
) -> dict[tuple[str, str], dict[str, Any]]:
    reviewed = observations.get("amber_openmm", {}).get("records")
    if not isinstance(reviewed, list):
        raise ValueError("Reviewed observations lack Amber compact records.")
    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    for record in reviewed:
        if not isinstance(record, dict):
            raise ValueError("Reviewed Amber compact record is invalid.")
        key = (str(record.get("case_id")), str(record.get("model")))
        if key in indexed:
            raise ValueError(f"Duplicate reviewed Amber record: {key}.")
        indexed[key] = record
    if set(indexed) != expected_keys:
        raise ValueError("Reviewed Amber compact-record matrix is incomplete.")
    return indexed


def _validate_amber_artifact(
    *,
    protocol: dict[str, Any],
    amber: dict[str, Any],
    manifest: dict[str, Any],
    observations: dict[str, Any],
    tolerances: dict[str, Any],
) -> list[dict[str, Any]]:
    required_models = list(protocol["methods"]["gb_models"])
    required_components = list(
        protocol["provider_parity"]["amber_required_components"]
    )
    required_force_components = list(
        protocol["provider_parity"]["amber_required_force_components"]
    )
    cases: dict[str, dict[str, Any]] = {}
    for case in manifest["cases"]:
        case_id = str(case.get("case_id", ""))
        if not case_id or case_id in cases:
            raise ValueError(f"Amber manifest has invalid case_id {case_id!r}.")
        references = case.get("models")
        if not isinstance(references, dict) or set(references) != set(required_models):
            raise ValueError(
                f"Amber manifest case {case_id} lacks the exact required model set."
            )
        cases[case_id] = case
    expected_keys = {
        (case_id, model) for case_id in cases for model in required_models
    }
    records = amber.get("records")
    if not isinstance(records, list):
        raise ValueError("Amber artifact records must be a list.")
    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Amber artifact contains a non-object record.")
        key = (str(record.get("case_id")), str(record.get("model")))
        if key in indexed:
            raise ValueError(f"Duplicate Amber record: {key}.")
        indexed[key] = record
    if set(indexed) != expected_keys:
        missing = sorted(expected_keys - set(indexed))
        extra = sorted(set(indexed) - expected_keys)
        raise ValueError(
            "Amber artifact matrix differs from the manifest: "
            f"missing={missing}, extra={extra}."
        )
    if (
        amber.get("case_count") != len(cases)
        or amber.get("record_count") != len(expected_keys)
        or len(records) != len(expected_keys)
    ):
        raise ValueError(
            "Amber artifact self-reported counts differ from the manifest."
        )
    expected_versions = {
        "amber": str(protocol["providers"]["ambertools"]["required_version"]),
        "openmm": str(protocol["providers"]["openmm"]["required_version"]),
    }
    if amber.get("provider_versions") != expected_versions:
        raise ValueError("Amber artifact provider versions differ from the protocol.")

    reviewed = _reviewed_amber_records(observations, expected_keys)
    amber_tolerances = tolerances.get("amber_gb")
    if not isinstance(amber_tolerances, dict):
        raise ValueError("Frozen Amber parity tolerances are absent.")
    checks: list[dict[str, Any]] = [
        {
            "check": "amber-record-completeness",
            "passed": True,
            "detail": {
                "case_count": len(cases),
                "observed_case_count": len(cases),
                "record_count": len(records),
                "expected_record_count": len(expected_keys),
            },
        }
    ]

    for key in sorted(expected_keys):
        case_id, model = key
        record = indexed[key]
        case = cases[case_id]
        if (
            _compact_amber_record(
                record,
                required_components,
                required_force_components,
            )
            != reviewed[key]
        ):
            raise ValueError(
                f"Amber compact record {case_id}/{model} differs from reviewed "
                "observations."
            )
        expected_profile = protocol["providers"]["openmm"]["models"][model][
            "profile"
        ]
        if (
            record.get("profile") != expected_profile
            or record.get("mol2_sha256") != case.get("mol2_sha256")
            or record.get("reference_provider") != "amber"
            or str(record.get("reference_provider_version"))
            != str(protocol["providers"]["ambertools"]["required_version"])
        ):
            raise ValueError(f"Amber record identity mismatch for {case_id}/{model}.")
        parity_components = _case_parity_targets(
            case,
            "parity_components",
            required_components,
        )
        parity_force_components = _case_parity_targets(
            case,
            "parity_force_components",
            required_force_components,
        )
        if (
            record.get("parity_components") != parity_components
            or record.get("parity_force_components") != parity_force_components
        ):
            raise ValueError(f"Amber parity targets changed for {case_id}/{model}.")

        reference = case["models"][model]
        reference_energy = _finite_mapping(
            record.get("reference_kcal_mol"),
            set(required_components),
            f"amber/{case_id}/{model}/reference_kcal_mol",
        )
        for component in required_components:
            _assert_close(
                reference_energy[component],
                _finite_float(
                    reference.get(component),
                    f"manifest/{case_id}/{model}/{component}",
                ),
                f"amber/{case_id}/{model}/{component} reference energy",
            )
        _assert_close(
            reference_energy["total_lcpo"],
            reference_energy["polar"] + reference_energy["nonpolar_lcpo"],
            f"amber/{case_id}/{model} reference energy closure",
        )

        model_expectation = case.get("model_expectations", {}).get(
            model,
            {
                "openmm_supported": True,
                "reason": "OpenMM model is expected to support this reference case.",
            },
        )
        expected_model_support = model_expectation.get("openmm_supported")
        model_observation = record.get("model_observation")
        if not isinstance(expected_model_support, bool) or not isinstance(
            model_observation, dict
        ):
            raise ValueError(
                f"Amber model expectation is invalid for {case_id}/{model}."
            )
        expected_model_fields = {
            "expected_openmm_supported": expected_model_support,
            "observed_openmm_supported": expected_model_support,
            "support_matches_expectation": True,
            "reason": str(model_expectation.get("reason", "")),
        }
        if any(
            model_observation.get(field) != value
            for field, value in expected_model_fields.items()
        ):
            raise ValueError(
                f"Amber model support evidence is invalid for {case_id}/{model}."
            )
        checks.append(
            {
                "check": f"amber/{case_id}/{model}/model-support-expectation",
                "passed": True,
                "detail": model_observation,
            }
        )
        if not expected_model_support:
            failure = model_observation.get("failure")
            if (
                record.get("status") != "expected-unavailable"
                or not isinstance(failure, dict)
                or not _matches_declared_model_unavailability_evidence(
                    failure.get("exception_class"),
                    failure.get("reason"),
                )
            ):
                raise ValueError(
                    "Amber expected-unavailable evidence has an undeclared "
                    "failure signature for "
                    f"{case_id}/{model}."
                )
            continue
        if record.get("status") != "success":
            raise ValueError(
                f"Supported Amber record {case_id}/{model} is not successful."
            )

        lcpo_expectation = case.get(
            "lcpo_expectation",
            {
                "parity_target": True,
                "openmm_supported": True,
                "reason": "complete Amber/OpenMM LCPO parity target",
            },
        )
        lcpo_observation = record.get("lcpo_observation")
        if not isinstance(lcpo_observation, dict):
            raise ValueError(f"Amber LCPO observation is absent for {case_id}/{model}.")
        expected_lcpo_fields = {
            "parity_target": lcpo_expectation.get("parity_target"),
            "expected_openmm_supported": lcpo_expectation.get("openmm_supported"),
            "observed_openmm_supported": lcpo_expectation.get("openmm_supported"),
            "support_matches_expectation": True,
            "reason": str(lcpo_expectation.get("reason", "")),
        }
        if any(
            lcpo_observation.get(field) != value
            for field, value in expected_lcpo_fields.items()
        ):
            raise ValueError(
                f"Amber LCPO support evidence is invalid for {case_id}/{model}."
            )
        if lcpo_expectation.get("openmm_supported") is False:
            failure = lcpo_observation.get("failure")
            if (
                not isinstance(failure, dict)
                or not _matches_declared_lcpo_unavailability_evidence(
                    failure.get("exception_class"),
                    failure.get("reason"),
                )
            ):
                raise ValueError(
                    "Amber LCPO unavailability has an undeclared failure "
                    "signature for "
                    f"{case_id}/{model}."
                )
        checks.append(
            {
                "check": f"amber/{case_id}/{model}/lcpo-support-expectation",
                "passed": True,
                "detail": lcpo_observation,
            }
        )

        complete_available = lcpo_expectation.get("openmm_supported") is True
        actual_components = {"polar"}
        actual_force_components = {"polar"}
        if complete_available:
            actual_components.update({"nonpolar_lcpo", "total_lcpo"})
            actual_force_components.add("total_lcpo")
        actual_energy = _finite_mapping(
            record.get("maple_kcal_mol"),
            actual_components,
            f"amber/{case_id}/{model}/maple_kcal_mol",
        )
        stored_difference = _finite_mapping(
            record.get("signed_difference_kcal_mol"),
            actual_components,
            f"amber/{case_id}/{model}/signed_difference_kcal_mol",
        )
        recomputed_difference = {
            component: actual_energy[component] - reference_energy[component]
            for component in actual_components
        }
        _assert_close(
            [stored_difference[item] for item in sorted(actual_components)],
            [recomputed_difference[item] for item in sorted(actual_components)],
            f"amber/{case_id}/{model} signed energy differences",
        )
        if complete_available:
            _assert_close(
                actual_energy["total_lcpo"],
                actual_energy["polar"] + actual_energy["nonpolar_lcpo"],
                f"amber/{case_id}/{model} MAPLE energy closure",
            )

        atom_count = len(case.get("charges_e", []))
        if atom_count <= 0:
            raise ValueError(f"Amber manifest case {case_id} lacks charges_e.")
        reference_forces = record.get("reference_force_kcal_mol_angstrom")
        maple_forces = record.get("maple_force_kcal_mol_angstrom")
        stored_force_difference = record.get("force_difference_kcal_mol_angstrom")
        stored_force_metrics = record.get("force_difference_metrics")
        if (
            not isinstance(reference_forces, dict)
            or set(reference_forces) != set(required_force_components)
            or not isinstance(maple_forces, dict)
            or set(maple_forces) != actual_force_components
            or not isinstance(stored_force_difference, dict)
            or set(stored_force_difference) != actual_force_components
            or not isinstance(stored_force_metrics, dict)
            or set(stored_force_metrics) != actual_force_components
        ):
            raise ValueError(
                f"Amber force component schema is invalid for {case_id}/{model}."
            )
        reference_arrays: dict[str, np.ndarray] = {}
        for component in required_force_components:
            reference_arrays[component] = _finite_force_array(
                reference_forces[component],
                atom_count,
                f"amber/{case_id}/{model}/reference_force/{component}",
            )
            _assert_close(
                reference_arrays[component],
                _finite_force_array(
                    reference[f"{component}_force_kcal_mol_angstrom"],
                    atom_count,
                    f"manifest/{case_id}/{model}/{component}_force",
                ),
                f"amber/{case_id}/{model}/{component} reference force",
            )
        recomputed_force_metrics: dict[str, dict[str, float]] = {}
        for component in actual_force_components:
            maple_array = _finite_force_array(
                maple_forces[component],
                atom_count,
                f"amber/{case_id}/{model}/maple_force/{component}",
            )
            difference_array = maple_array - reference_arrays[component]
            _assert_close(
                _finite_force_array(
                    stored_force_difference[component],
                    atom_count,
                    f"amber/{case_id}/{model}/force_difference/{component}",
                ),
                difference_array,
                f"amber/{case_id}/{model}/{component} force difference",
            )
            metrics = stored_force_metrics[component]
            if not isinstance(metrics, dict) or set(metrics) != {"max_abs", "rms"}:
                raise ValueError(
                    "Amber force metrics are invalid for "
                    f"{case_id}/{model}/{component}."
                )
            recomputed_force_metrics[component] = {
                "max_abs": float(np.max(np.abs(difference_array))),
                "rms": float(np.sqrt(np.mean(difference_array**2))),
            }
            _assert_close(
                [
                    _finite_float(
                        metrics["max_abs"],
                        f"amber/{case_id}/{model}/{component}/max_abs",
                    ),
                    _finite_float(
                        metrics["rms"],
                        f"amber/{case_id}/{model}/{component}/rms",
                    ),
                ],
                [
                    recomputed_force_metrics[component]["max_abs"],
                    recomputed_force_metrics[component]["rms"],
                ],
                f"amber/{case_id}/{model}/{component} force metrics",
            )
        if complete_available:
            if (
                lcpo_observation.get("signed_difference_kcal_mol")
                != {
                    component: stored_difference[component]
                    for component in ("nonpolar_lcpo", "total_lcpo")
                }
                or lcpo_observation.get("force_difference_metrics")
                != {"total_lcpo": stored_force_metrics["total_lcpo"]}
            ):
                raise ValueError(
                    f"Amber nested LCPO evidence is inconsistent for {case_id}/{model}."
                )

        for component in parity_components:
            tolerance = _require_number(
                amber_tolerances,
                f"max_abs_{component}_kcal_mol",
            )
            difference = abs(recomputed_difference[component])
            checks.append(
                {
                    "check": f"amber/{case_id}/{model}/{component}",
                    "passed": difference <= tolerance,
                    "observed": difference,
                    "tolerance": tolerance,
                }
            )
        for component in parity_force_components:
            tolerance = _require_number(
                amber_tolerances,
                f"max_abs_{component}_force_kcal_mol_angstrom",
            )
            difference = recomputed_force_metrics[component]["max_abs"]
            checks.append(
                {
                    "check": f"amber/{case_id}/{model}/{component}-force",
                    "passed": difference <= tolerance,
                    "observed": difference,
                    "tolerance": tolerance,
                }
            )

    reviewed_summary = observations.get("amber_openmm")
    if not isinstance(reviewed_summary, dict):
        raise ValueError("Reviewed Amber observation summary is absent.")
    by_case: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_case.setdefault(record["case_id"], []).append(record)
    full_lcpo_cases = {
        case_id
        for case_id, case_records in by_case.items()
        if all(
            record.get("status") == "success"
            and record["parity_components"] == required_components
            for record in case_records
        )
    }
    polar_only_cases = {
        case_id
        for case_id, case_records in by_case.items()
        if any(record.get("status") == "success" for record in case_records)
        and all(
            record["parity_components"] == ["polar"]
            for record in case_records
            if record.get("status") == "success"
        )
    }
    expected_summary_fields = {
        "all_successful": all(
            record.get("status") in {"success", "expected-unavailable"}
            for record in records
        ),
        "case_count": len(cases),
        "record_count": len(records),
        "supported_record_count": sum(
            record.get("status") == "success" for record in records
        ),
        "expected_unavailable_record_count": sum(
            record.get("status") == "expected-unavailable" for record in records
        ),
        "full_lcpo_parity_case_count": len(full_lcpo_cases),
        "polar_only_case_count": len(polar_only_cases),
        "energy_difference_kcal_mol": {
            component: _maximum_energy_difference(records, component)
            for component in required_components
        },
        "force_difference_kcal_mol_angstrom": {
            component: _maximum_force_difference(records, component)
            for component in required_force_components
        },
        "lcpo_applicability_observations": _lcpo_applicability_observations(records),
        "model_applicability_observations": _model_applicability_observations(records),
    }
    for field, expected in expected_summary_fields.items():
        if reviewed_summary.get(field) != expected:
            raise ValueError(f"Reviewed Amber summary field {field} is inconsistent.")
    return checks


def _validate_apbs_artifact(
    *,
    protocol: dict[str, Any],
    apbs: dict[str, Any],
    manifest: dict[str, Any],
    observations: dict[str, Any],
    tolerances: dict[str, Any],
) -> list[dict[str, Any]]:
    cases: dict[str, dict[str, Any]] = {}
    expected_records: dict[tuple[str, int, float], dict[str, Any]] = {}
    for case in manifest["cases"]:
        case_id = str(case.get("case_id", ""))
        if not case_id or case_id in cases:
            raise ValueError(f"APBS manifest has invalid case_id {case_id!r}.")
        control_kind = case.get("control_kind")
        if control_kind not in set(
            protocol["provider_parity"]["apbs_required_controls"]
        ):
            raise ValueError(f"APBS manifest case {case_id} has invalid control kind.")
        grids = case.get("grids")
        if not isinstance(grids, list) or not grids:
            raise ValueError(f"APBS manifest case {case_id} has no grids.")
        cases[case_id] = case
        for grid in grids:
            key = (
                case_id,
                _finite_int(
                    grid.get("grid_points"),
                    f"apbs-manifest/{case_id}/grid_points",
                ),
                _finite_float(
                    grid.get("grid_spacing_angstrom"),
                    f"apbs-manifest/{case_id}/grid_spacing_angstrom",
                ),
            )
            if key in expected_records:
                raise ValueError(f"Duplicate APBS manifest grid: {key}.")
            expected_records[key] = case
    records = apbs.get("records")
    if not isinstance(records, list):
        raise ValueError("APBS artifact records must be a list.")
    indexed: dict[tuple[str, int, float], dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("APBS artifact contains a non-object record.")
        case_id = record.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError("APBS artifact record has an invalid case_id.")
        key = (
            case_id,
            _finite_int(
                record.get("grid_points"),
                f"apbs/{case_id}/grid_points",
            ),
            _finite_float(
                record.get("grid_spacing_angstrom"),
                f"apbs/{case_id}/grid_spacing_angstrom",
            ),
        )
        if key in indexed:
            raise ValueError(f"Duplicate APBS grid record: {key}.")
        indexed[key] = record
    if set(indexed) != set(expected_records):
        missing = sorted(set(expected_records) - set(indexed))
        extra = sorted(set(indexed) - set(expected_records))
        raise ValueError(
            "APBS artifact matrix differs from the manifest: "
            f"missing={missing}, extra={extra}."
        )
    if str(apbs.get("provider_version")) != str(
        protocol["providers"]["apbs"]["required_version"]
    ):
        raise ValueError("APBS artifact provider version differs from the protocol.")
    reviewed_sources = observations["source_artifacts"]
    if (
        apbs.get("provider_executable_sha256")
        != reviewed_sources.get("apbs_executable_sha256")
    ):
        raise ValueError("APBS executable hash differs from reviewed observations.")
    apbs_tolerances = tolerances.get("apbs")
    if not isinstance(apbs_tolerances, dict):
        raise ValueError("Frozen APBS parity tolerances are absent.")
    official_tolerance = _require_number(
        apbs_tolerances,
        "max_abs_official_component_kcal_mol",
    )
    grid_tolerance = _require_number(
        apbs_tolerances,
        "max_successive_grid_total_difference_kcal_mol",
    )
    checks: list[dict[str, Any]] = []
    by_case: dict[str, list[dict[str, Any]]] = {}
    official_records: list[dict[str, Any]] = []
    for key in sorted(expected_records):
        case_id, _grid_points, _spacing = key
        record = indexed[key]
        case = expected_records[key]
        by_case.setdefault(case_id, []).append(record)
        if (
            record.get("control_kind") != case["control_kind"]
            or record.get("status") != "success"
            or record.get("expected_kcal_mol") != case.get("expected_kcal_mol")
        ):
            raise ValueError(f"APBS record identity/status mismatch for {key}.")
        source_key = "pqr_sha256" if "pqr_sha256" in case else "mol2_sha256"
        if record.get(source_key) != case.get(source_key):
            raise ValueError(f"APBS source hash mismatch for {key}.")
        actual = _finite_mapping(
            record.get("maple_kcal_mol"),
            {"polar", "nonpolar", "total"},
            f"apbs/{case_id}/{key[1]}/maple_kcal_mol",
        )
        _assert_close(
            actual["total"],
            actual["polar"] + actual["nonpolar"],
            f"apbs/{case_id}/{key[1]} energy closure",
        )
        expected_values = case.get("expected_kcal_mol") or {}
        if not isinstance(expected_values, dict):
            raise ValueError(f"APBS expected values are invalid for {case_id}.")
        stored_difference = _finite_mapping(
            record.get("signed_difference_kcal_mol"),
            set(expected_values),
            f"apbs/{case_id}/{key[1]}/signed_difference_kcal_mol",
        )
        recomputed_difference = {
            component: actual[component]
            - _finite_float(
                expected_values[component],
                f"apbs-manifest/{case_id}/{component}",
            )
            for component in expected_values
        }
        _assert_close(
            [stored_difference[item] for item in sorted(stored_difference)],
            [recomputed_difference[item] for item in sorted(recomputed_difference)],
            f"apbs/{case_id}/{key[1]} signed energy differences",
        )
        if case["control_kind"] == "official-born-ion":
            official_records.append(record)
            if not recomputed_difference:
                raise ValueError("APBS official control has no documented comparison.")
            for component, difference in recomputed_difference.items():
                observed = abs(difference)
                checks.append(
                    {
                        "check": f"apbs/{case_id}/official/{component}",
                        "passed": observed <= official_tolerance,
                        "observed": observed,
                        "tolerance": official_tolerance,
                    }
                )

    if len(official_records) != 1:
        raise ValueError("APBS manifest must produce exactly one official control.")
    official = official_records[0]
    official_case = cases[official["case_id"]]
    expected_polar = _finite_float(
        official_case["expected_kcal_mol"]["polar"],
        "APBS official expected polar energy",
    )
    expected_kj_mol = _finite_mapping(
        official_case.get("expected_kj_mol"),
        set(official_case["expected_kcal_mol"]),
        "APBS official expected_kj_mol",
    )
    for component, expected_kcal_mol in official_case["expected_kcal_mol"].items():
        _assert_close(
            expected_kj_mol[component],
            _finite_float(
                expected_kcal_mol,
                f"APBS official expected_kcal_mol.{component}",
            )
            * KJ_PER_KCAL,
            f"APBS official {component} kcal/kJ contract",
        )
    observed_polar = _finite_float(
        official["maple_kcal_mol"]["polar"],
        "APBS official observed polar energy",
    )
    reviewed_apbs = observations.get("apbs")
    if not isinstance(reviewed_apbs, dict):
        raise ValueError("Reviewed APBS observation summary is absent.")
    expected_official = {
        "grid_points": int(official["grid_points"]),
        "grid_spacing_angstrom": float(official["grid_spacing_angstrom"]),
        "documented_kj_mol": expected_kj_mol["polar"],
        "observed_kj_mol": observed_polar * KJ_PER_KCAL,
        "absolute_difference_kcal_mol": abs(observed_polar - expected_polar),
    }
    if reviewed_apbs.get("official_born") != expected_official:
        raise ValueError(
            "Reviewed APBS kcal/kJ official-control conversion is inconsistent."
        )

    neutral_grid: dict[str, Any] = {}
    for case_id, case_records in sorted(by_case.items()):
        if cases[case_id]["control_kind"] != "neutral-grid-convergence":
            continue
        ordered = sorted(
            case_records,
            key=lambda record: float(record["grid_spacing_angstrom"]),
            reverse=True,
        )
        if len(ordered) < 2:
            raise ValueError(
                f"APBS neutral grid case {case_id} has fewer than two grids."
            )
        finest = sorted(
            ordered,
            key=lambda record: float(record["grid_spacing_angstrom"]),
        )[:2]
        difference = abs(
            float(finest[0]["maple_kcal_mol"]["total"])
            - float(finest[1]["maple_kcal_mol"]["total"])
        )
        neutral_grid[case_id] = {
            "records": [
                {
                    "grid_points": int(record["grid_points"]),
                    "grid_spacing_angstrom": float(record["grid_spacing_angstrom"]),
                    "polar_kcal_mol": float(record["maple_kcal_mol"]["polar"]),
                    "nonpolar_kcal_mol": float(record["maple_kcal_mol"]["nonpolar"]),
                    "total_kcal_mol": float(record["maple_kcal_mol"]["total"]),
                }
                for record in ordered
            ],
            "finest_pair_spacing_angstrom": [
                float(record["grid_spacing_angstrom"]) for record in finest
            ],
            "successive_finest_total_difference_kcal_mol": difference,
        }
        checks.append(
            {
                "check": f"apbs/{case_id}/successive-finest-grid-total",
                "passed": difference <= grid_tolerance,
                "observed": difference,
                "tolerance": grid_tolerance,
            }
        )
    if (
        reviewed_apbs.get("all_successful") is not True
        or reviewed_apbs.get("record_count") != len(records)
        or reviewed_apbs.get("neutral_grid") != neutral_grid
    ):
        raise ValueError("Reviewed APBS grid summary is inconsistent.")
    return checks


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
    if (
        tolerances.get("protocol_id") != protocol["protocol_id"]
        or tolerances.get("protocol_fingerprint") != fingerprint
    ):
        raise ValueError(
            "Provider-parity tolerances target a different protocol fingerprint."
        )
    if tolerances.get("review_status") != "independently-reviewed-frozen":
        raise ValueError(
            "Provider-parity tolerances are not marked independently-reviewed-frozen."
        )

    amber_path = artifact_dir / "amber-gb-parity/results.json"
    apbs_path = artifact_dir / "apbs-grid/results.json"
    if not amber_path.is_file() or not apbs_path.is_file():
        raise FileNotFoundError(
            "Provider parity requires amber-gb-parity/results.json and "
            "apbs-grid/results.json."
        )
    observations, amber_manifest, apbs_manifest = _load_review_chain(
        protocol=protocol,
        fingerprint=fingerprint,
        tolerances=tolerances,
        amber_path=amber_path,
        apbs_path=apbs_path,
    )
    amber = load_json(amber_path)
    apbs = load_json(apbs_path)
    for artifact, expected_type in (
        (amber, "amber-gb-parity"),
        (apbs, "apbs-grid-parity"),
    ):
        if (
            artifact.get("schema_version") != 1
            or artifact.get("artifact_type") != expected_type
            or artifact.get("protocol_id") != protocol["protocol_id"]
            or artifact.get("protocol_fingerprint") != fingerprint
        ):
            raise ValueError(
                f"Invalid or mismatched provider artifact: {expected_type}."
            )

    checks = _validate_amber_artifact(
        protocol=protocol,
        amber=amber,
        manifest=amber_manifest,
        observations=observations,
        tolerances=tolerances,
    )
    checks.extend(
        _validate_apbs_artifact(
            protocol=protocol,
            apbs=apbs,
            manifest=apbs_manifest,
            observations=observations,
            tolerances=tolerances,
        )
    )
    passed = bool(checks) and all(check["passed"] for check in checks)
    verification = {
        "schema_version": 2,
        "artifact_type": "provider-parity-verification",
        "protocol_id": protocol["protocol_id"],
        "protocol_fingerprint": fingerprint,
        "tolerances_sha256": sha256_file(tolerance_path),
        "reviewed_proposal_sha256": tolerances["reviewed_proposal_sha256"],
        "reviewed_observations_sha256": tolerances["evidence_artifact_sha256"],
        "amber_artifact_sha256": sha256_file(amber_path),
        "apbs_artifact_sha256": sha256_file(apbs_path),
        "amber_reference_manifest_sha256": amber[
            "reference_manifest_sha256"
        ],
        "apbs_reference_manifest_sha256": apbs["reference_manifest_sha256"],
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

    observed = subparsers.add_parser("observations")
    observed.add_argument("--protocol", required=True)
    observed.add_argument("--amber-artifact", required=True)
    observed.add_argument("--apbs-artifact", required=True)
    observed.add_argument("--output", required=True)
    observed.set_defaults(handler=observations)

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
