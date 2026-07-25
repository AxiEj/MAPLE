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
                model_observation.update(
                    observed_openmm_supported=False,
                    support_matches_expectation=not expected_model_support,
                    failure={
                        "exception_class": type(exc).__name__,
                        "reason": str(exc),
                    },
                )
                if expected_model_support:
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
                actual = {
                    "polar": polar_result.energy_hartree * KCAL_PER_HARTREE,
                }
                force_actual = {
                    "polar": polar_result.forces_hartree_per_angstrom
                    * KCAL_PER_HARTREE,
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
                    actual.update(
                        nonpolar_lcpo=(
                            complete_result.components_hartree["nonpolar"]
                            * KCAL_PER_HARTREE
                        ),
                        total_lcpo=(complete_result.energy_hartree * KCAL_PER_HARTREE),
                    )
                    force_actual["total_lcpo"] = (
                        complete_result.forces_hartree_per_angstrom * KCAL_PER_HARTREE
                    )
                except Exception as exc:
                    complete_failure = {
                        "exception_class": type(exc).__name__,
                        "reason": str(exc),
                    }
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
        {
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
            **(
                {
                    "signed_difference_kcal_mol": record["signed_difference_kcal_mol"],
                    "force_difference_metrics": record["force_difference_metrics"],
                    "lcpo_observation": record["lcpo_observation"],
                    "model_observation": record["model_observation"],
                }
                if record["status"] == "success"
                else {
                    **(
                        {"failure": record.get("failure")}
                        if record["status"] == "failure"
                        else {}
                    ),
                    "model_observation": record.get("model_observation"),
                }
            ),
        }
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
        "evidence_status": "measured-awaiting-human-tolerance-review",
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
                "documented_kj_mol": expected_polar * 4.184,
                "observed_kj_mol": observed_polar * 4.184,
                "absolute_difference_kcal_mol": abs(observed_polar - expected_polar),
            },
            "neutral_grid": neutral_grid,
        },
    }
    write_json_atomic(args.output, artifact)
    print(f"Wrote provider observations to {Path(args.output).resolve()}.")


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
        raise ValueError(
            "Provider-parity tolerances target a different protocol fingerprint."
        )
    if tolerances.get("review_status") != "human-reviewed-frozen":
        raise ValueError(
            "Provider-parity tolerances are not marked human-reviewed-frozen."
        )

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
            raise ValueError(
                f"Invalid or mismatched provider artifact: {expected_type}."
            )

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
    case_models: dict[str, set[str]] = {}
    for record in amber["records"]:
        case_models.setdefault(str(record.get("case_id")), set()).add(
            str(record.get("model"))
        )
    expected_record_count = int(amber.get("case_count", 0)) * len(required_models)
    checks.append(
        {
            "check": "amber-record-completeness",
            "passed": (
                int(amber.get("record_count", -1)) == len(amber["records"])
                and len(amber["records"]) == expected_record_count
                and len(case_models) == int(amber.get("case_count", 0))
                and all(models == required_models for models in case_models.values())
            ),
            "detail": {
                "case_count": amber.get("case_count"),
                "observed_case_count": len(case_models),
                "record_count": amber.get("record_count"),
                "expected_record_count": expected_record_count,
            },
        }
    )
    for record in amber["records"]:
        model_observation = record.get("model_observation")
        if model_observation:
            checks.append(
                {
                    "check": (
                        f"amber/{record['case_id']}/{record['model']}/"
                        "model-support-expectation"
                    ),
                    "passed": (
                        model_observation.get("support_matches_expectation") is True
                    ),
                    "detail": model_observation,
                }
            )
        if record.get("status") == "expected-unavailable":
            continue
        if record.get("status") != "success":
            checks.append(
                {
                    "check": f"amber/{record.get('case_id')}/{record.get('model')}",
                    "passed": False,
                    "detail": record.get("failure"),
                }
            )
            continue
        lcpo_observation = record.get("lcpo_observation", {})
        checks.append(
            {
                "check": (
                    f"amber/{record['case_id']}/{record['model']}/"
                    "lcpo-support-expectation"
                ),
                "passed": lcpo_observation.get("support_matches_expectation") is True,
                "detail": lcpo_observation,
            }
        )
        for component in record.get(
            "parity_components",
            protocol["provider_parity"]["amber_required_components"],
        ):
            tolerance = _require_number(
                amber_tolerances, f"max_abs_{component}_kcal_mol"
            )
            difference = abs(float(record["signed_difference_kcal_mol"][component]))
            checks.append(
                {
                    "check": f"amber/{record['case_id']}/{record['model']}/{component}",
                    "passed": difference <= tolerance,
                    "observed": difference,
                    "tolerance": tolerance,
                }
            )
        for component in record.get(
            "parity_force_components",
            protocol["provider_parity"]["amber_required_force_components"],
        ):
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
                {
                    "check": f"apbs/{record.get('case_id')}/provider",
                    "passed": False,
                    "detail": record.get("failure"),
                }
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
        record.get("control_kind")
        for record in apbs["records"]
        if record.get("status") == "success"
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
        if (
            not successful
            or successful[0]["control_kind"] != "neutral-grid-convergence"
        ):
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
                {
                    "check": f"apbs/{case_id}/grid-count",
                    "passed": False,
                    "detail": "need at least two successful grids",
                }
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
