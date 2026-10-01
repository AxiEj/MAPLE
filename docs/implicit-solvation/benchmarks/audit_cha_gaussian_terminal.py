#!/usr/bin/env python3
"""Independent terminal E/F audit for the bounded Gaussian-CHA campaign.

This is a post-campaign evidence tool.  It neither optimizes coordinates nor
changes the model.  Every output is create-only and bound to externally pinned
campaign and audit preregistration bytes.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable

import numpy as np
from ase import Atoms

ROOT = Path("/home/axie/MAPLE/MAPLE-implicitsolv-route1")
EXPECTED_WORKTREE = ROOT / ".omx/worktrees/cha-analytic-v1-20260930"
OUTPUT_ROOT = ROOT / ".omx/benchmarks/route1-cha-gaussian-opt-v1-20261001"
SPEC = OUTPUT_ROOT / "terminal-independent-spec-v1.json"
SPEC_SHA256 = "f5e75130ef3259ad269826558538b7016cce79a08f721e8f2cfb042fa0a8cf02"
SCRIPT = Path(__file__).resolve()
KCAL_PER_HARTREE = 627.5094740631


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        json_safe(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def read_externally_pinned_json(
    path: Path, expected_sha256: str, label: str
) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    digest = sha256_bytes(raw)
    if digest != expected_sha256:
        raise ValueError(f"{label} external SHA256 pin differs")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value, digest


def write_json_new_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_bytes(value) + b"\n"
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            temporary = handle.name
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)


def coordinate_sha256(positions: Any) -> str:
    array = np.asarray(positions, dtype=np.float64)
    if array.ndim != 2 or array.shape[1:] != (3,) or not np.isfinite(array).all():
        raise ValueError("coordinates must be finite float64 shape (N,3)")
    return sha256_bytes(array.astype("<f8", copy=False).tobytes(order="C"))


def oracle_coordinate_sha256(positions: Any) -> str:
    """Reproduce the independent oracle's labeled coordinate identity."""
    array = np.asarray(positions, dtype=np.float64)
    if array.ndim != 2 or array.shape[1:] != (3,) or not np.isfinite(array).all():
        raise ValueError("oracle coordinates must be finite float64 shape (N,3)")
    contiguous = np.ascontiguousarray(array, dtype="<f8")
    digest = hashlib.sha256(b"gaussian-cha-coordinates-v1")
    digest.update(str(contiguous.shape).encode("ascii"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def validate_reference_metadata_coordinates(
    metadata: dict[str, Any], positions: Any
) -> None:
    if metadata.get("coordinates_sha256") != oracle_coordinate_sha256(positions):
        raise ValueError("reference metadata coordinates differ")


def validate_reference_error_semantics(metadata: dict[str, Any]) -> None:
    diagnostics = metadata["nonpolar_diagnostics"]
    for key in (
        "cavity_mixed_vector_quad_error",
        "dispersion_mixed_vector_quad_error",
    ):
        values = np.asarray(diagnostics[key], dtype=np.float64)
        if not np.isfinite(values).all() or np.any(values < 0.0):
            raise ValueError(
                "reference quadrature errors must be finite and nonnegative"
            )
    for level in metadata["r6_levels"]:
        error = float(level["inverse_cube_quad_error_estimate_per_angstrom3"])
        if not math.isfinite(error) or error < 0.0:
            raise ValueError("reference R6 errors must be finite and nonnegative")


def validated_force_array(name: str, values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (3, 3) or not np.isfinite(array).all():
        raise ValueError(f"{name} forces must have finite shape (3, 3)")
    return array


def validate_combined_base_ledger(
    combined: dict[str, Any], expected_positions: np.ndarray, runner: Any
) -> None:
    observed = np.asarray(combined["base_positions_angstrom"], dtype=np.float64)
    if (
        not np.array_equal(observed, expected_positions)
        or coordinate_sha256(observed) != combined["base_coordinate_sha256"]
    ):
        raise ValueError("terminal combined base coordinates differ")
    ledger = runner._energy_ledger(
        {
            "combined_energy_hartree": combined["analytic_combined_energy_hartree"],
            "solvation": combined["analytic_solvation_ledger"],
        }
    )
    analytic_energy = float(combined["analytic_combined_energy_hartree"])
    raw_combined = float(
        combined["analytic_solvation_ledger"]["combined_energy_hartree"]
    )
    if (
        abs(analytic_energy - ledger["combined"]) > 1.0e-12
        or abs(analytic_energy - raw_combined) > 1.0e-12
    ):
        raise ValueError("terminal analytic combined energy differs from ledger")
    if abs(ledger["gas"] + ledger["solvent"] - ledger["combined"]) > 1.0e-10:
        raise ValueError("terminal gas plus solvent ledger differs from combined")
    if (
        abs(
            ledger["component_polar"]
            + ledger["component_cavity"]
            + ledger["component_dispersion"]
            - ledger["solvent"]
        )
        > 1.0e-10
    ):
        raise ValueError("terminal solvent component ledger differs from solvent total")


def assert_post_phase_snapshot_stable(
    preregistration: dict[str, Any],
    current_source: dict[str, str],
    current_inputs: dict[str, str],
) -> None:
    if current_source != preregistration.get("source_before"):
        raise ValueError("terminal audit source changed during phase")
    if current_inputs != preregistration.get("campaign_input_sha256"):
        raise ValueError("campaign inputs changed during terminal audit phase")


def _runner():
    return importlib.import_module(
        "docs.implicit-solvation.benchmarks.run_cha_gaussian_opt"
    )


def _oracle():
    return importlib.import_module(
        "docs.implicit-solvation.benchmarks.cha_gaussian_reference"
    )


def _load_spec() -> dict[str, Any]:
    spec, _ = read_externally_pinned_json(SPEC, SPEC_SHA256, "terminal audit spec")
    if spec.get("approved_plan_sha256") != (
        "74d4aac2a5275d38599985398e45bb1e8d79b0dfd1fc2e3ae5a33a9d8f401838"
    ):
        raise ValueError("terminal audit spec plan identity differs")
    return spec


def _source_snapshot(runner: Any) -> dict[str, str]:
    snapshot = dict(runner._source_snapshot())
    if not SCRIPT.is_relative_to(EXPECTED_WORKTREE):
        raise ValueError("terminal audit script resolved outside expected worktree")
    snapshot[str(SCRIPT.relative_to(EXPECTED_WORKTREE))] = sha256_bytes(
        SCRIPT.read_bytes()
    )
    return dict(sorted(snapshot.items()))


def _source_identity(snapshot: dict[str, str]) -> str:
    return sha256_bytes(canonical_bytes(snapshot))


def terminal_inventory_entry(
    row: dict[str, Any], row_file_sha256: str
) -> dict[str, Any]:
    retained = np.asarray(row["retained_positions_angstrom"], dtype=np.float64)
    fresh_record = row["fresh_final"]
    if fresh_record.get("tag") != "FRESH_FINAL_UNCACHED":
        raise ValueError("fresh-final tag differs")
    fresh = np.asarray(fresh_record["positions_angstrom"], dtype=np.float64)
    retained_hash = coordinate_sha256(retained)
    fresh_hash = coordinate_sha256(fresh)
    if retained_hash != row.get("retained_coordinate_sha256"):
        raise ValueError("retained coordinate hash differs")
    if fresh_hash != fresh_record.get("coordinate_sha256"):
        raise ValueError("fresh-final coordinate hash differs")
    if not np.array_equal(retained, fresh) or retained_hash != fresh_hash:
        raise ValueError("retained and fresh-final coordinates differ")
    frames = row.get("successful_force_frames", [])
    if frames:
        last = frames[-1]
        last_positions = np.asarray(last["positions_angstrom"], dtype=np.float64)
        if not np.array_equal(last_positions, retained) or coordinate_sha256(
            last_positions
        ) != last.get("coordinate_sha256"):
            raise ValueError(
                "last retained force frame differs from terminal coordinates"
            )
    return {
        "row_id": row["row_id"],
        "center_index": int(row["center_index"]),
        "center_scale": float(row["center_scale"]),
        "sigma_e": float(row["sigma_e"]),
        "opt_status": row["status"],
        "row_receipt_file_sha256": row_file_sha256,
        "row_receipt_content_sha256": row.get("content_sha256"),
        "terminal_positions_angstrom": retained.tolist(),
        "terminal_coordinate_sha256": retained_hash,
    }


class ReferenceGeometryCache:
    """Width-independent cache keyed by exact coordinates, parameters and order."""

    def __init__(
        self, oracle: Any, topology: dict[str, Any], epsabs: float, epsrel: float
    ):
        self.oracle = oracle
        self.topology = topology
        self.epsabs = float(epsabs)
        self.epsrel = float(epsrel)
        parameter_payload = {
            key: topology[key]
            for key in (
                "effective_charges_e",
                "cha_radii_angstrom",
                "lj_rmin_angstrom",
                "lj_epsilon_kcal_mol",
            )
        }
        self.parameter_sha256 = sha256_bytes(canonical_bytes(parameter_payload))
        self._values: dict[tuple[str, str, int, float, float], Any] = {}

    def get(self, positions: Any, order: int):
        array = np.asarray(positions, dtype=np.float64)
        key = (
            coordinate_sha256(array),
            self.parameter_sha256,
            int(order),
            self.epsabs,
            self.epsrel,
        )
        if key not in self._values:
            self._values[key] = self.oracle.prepare_gaussian_reference_geometry(
                array,
                self.topology["effective_charges_e"],
                self.topology["cha_radii_angstrom"],
                self.topology["lj_rmin_angstrom"],
                self.topology["lj_epsilon_kcal_mol"],
                azimuth_orders=(int(order),),
                epsabs=self.epsabs,
                epsrel=self.epsrel,
            )
        return self._values[key]


def derive_five_point_force(
    stencils: list[dict[str, Any]],
    expected_steps: Iterable[float],
    *,
    energy_key: str,
) -> dict[str, Any]:
    forces = []
    steps = [float(value) for value in expected_steps]
    if len(stencils) != len(steps):
        raise ValueError("finite-difference step inventory differs")
    for expected, stencil in zip(steps, stencils, strict=True):
        if float(stencil["step_angstrom"]) != expected:
            raise ValueError("finite-difference step differs")
        samples = stencil["samples"]
        if [float(sample["multiplier"]) for sample in samples] != [
            -2.0,
            -1.0,
            1.0,
            2.0,
        ]:
            raise ValueError("finite-difference multiplier inventory differs")
        energies = [float(sample[energy_key]) for sample in samples]
        derivative = (energies[0] - 8 * energies[1] + 8 * energies[2] - energies[3]) / (
            12 * expected
        )
        forces.append(-float(derivative))
    return {
        "forces": forces,
        "finest_step_disagreement": abs(forces[-1] - forces[-2]),
    }


def summarize_terminal_audit(
    rows: list[dict[str, Any]],
    expected_row_ids: list[str],
    *,
    campaign_validated: bool,
) -> dict[str, Any]:
    observed = [row.get("row_id") for row in rows]
    inventory_exact = observed == expected_row_ids and len(set(observed)) == 15
    opt_count = sum(row.get("opt_status") == "CONVERGED" for row in rows)
    solvent_count = sum(bool(row.get("solvent_passed")) for row in rows)
    combined_count = sum(bool(row.get("combined_passed")) for row in rows)
    passed = bool(
        campaign_validated
        and inventory_exact
        and len(rows) == 15
        and opt_count == solvent_count == combined_count == 15
    )
    return {
        "row_count": len(rows),
        "inventory_exact": inventory_exact,
        "campaign_validation_recomputed": bool(campaign_validated),
        "opt_converged_count": opt_count,
        "solvent_independent_pass_count": solvent_count,
        "combined_total_fd_pass_count": combined_count,
        "status": "VALIDATED" if passed else "GAUSSIAN_OPT_INCOMPLETE",
    }


def _validate_roster(rows: list[dict[str, Any]], spec: dict[str, Any]) -> None:
    inventory = spec["inventory"]
    if len(rows) != int(inventory["row_count"]):
        raise ValueError("campaign does not contain exactly 15 rows")
    expected = [
        (float(center), float(width))
        for center in inventory["center_scales"]
        for width in inventory["widths_e"]
    ]
    observed = [(float(row["center_scale"]), float(row["sigma_e"])) for row in rows]
    if observed != expected or len({row["row_id"] for row in rows}) != 15:
        raise ValueError("campaign row roster differs from terminal audit spec")


def preregister(campaign: Path, output: Path, campaign_prereg_sha256: str):
    spec = _load_spec()
    runner = _runner()
    _protocol, campaign_prereg, evidence = runner.load_preregistration(
        campaign, campaign_prereg_sha256
    )
    if campaign_prereg.get("protocol_template_sha256") != spec.get(
        "protocol_template_sha256"
    ):
        raise ValueError("campaign protocol identity differs from terminal audit spec")
    rows = campaign_prereg["rows"]
    _validate_roster(rows, spec)
    inventory = []
    for expected in rows:
        path = campaign / "rows" / f"{expected['row_id']}.json"
        row, row_digest = runner.read_hashed_json(path, "OPT row receipt")
        runner._verify_receipt_binding(
            row,
            evidence,
            receipt_kind="OPT_ROW",
            case_id=expected["row_id"],
            case_identity=expected,
        )
        inventory.append(terminal_inventory_entry(row, row_digest))
    campaign_validation, campaign_validation_sha256 = runner.read_hashed_json(
        campaign / "validation.json", "campaign validation"
    )
    runner.verify_seal(campaign_validation, "campaign validation")
    source = _source_snapshot(runner)
    record = runner.seal(
        {
            "schema_version": 1,
            "phase": "TERMINAL_AUDIT_PREREGISTERED",
            "spec_sha256": SPEC_SHA256,
            "campaign": str(campaign.resolve()),
            "campaign_preregistration_sha256": campaign_prereg_sha256,
            "campaign_source_identity_sha256": evidence["source_identity_sha256"],
            "campaign_input_sha256": evidence["input_sha256"],
            "campaign_validation_file_sha256": campaign_validation_sha256,
            "campaign_validation_content_sha256": campaign_validation["content_sha256"],
            "campaign_protocol_template_sha256": campaign_prereg[
                "protocol_template_sha256"
            ],
            "topology_content_sha256": evidence["topology"]["content_sha256"],
            "source_before": source,
            "source_identity_sha256": _source_identity(source),
            "rows": inventory,
            "claim_boundary": spec["claims"],
        }
    )
    write_json_new_atomic(output / "preregistration.json", record)
    return record


def _load_bound_context(
    campaign: Path,
    output: Path,
    campaign_prereg_sha256: str,
    audit_prereg_sha256: str,
):
    spec = _load_spec()
    runner = _runner()
    audit_prereg, observed = read_externally_pinned_json(
        output / "preregistration.json",
        audit_prereg_sha256,
        "terminal audit preregistration",
    )
    runner.verify_seal(audit_prereg, "terminal audit preregistration")
    if observed != audit_prereg_sha256:
        raise ValueError("terminal audit preregistration pin differs")
    if audit_prereg.get("campaign") != str(campaign.resolve()):
        raise ValueError("terminal audit campaign path differs")
    if audit_prereg.get("spec_sha256") != SPEC_SHA256:
        raise ValueError("terminal audit spec pin differs")
    if audit_prereg.get("campaign_preregistration_sha256") != campaign_prereg_sha256:
        raise ValueError("terminal audit campaign preregistration pin differs")
    protocol, campaign_prereg, evidence = runner.load_preregistration(
        campaign, campaign_prereg_sha256
    )
    if campaign_prereg.get("protocol_template_sha256") != spec.get(
        "protocol_template_sha256"
    ):
        raise ValueError("campaign protocol identity differs from terminal audit spec")
    if evidence["source_identity_sha256"] != audit_prereg.get(
        "campaign_source_identity_sha256"
    ):
        raise ValueError("campaign source identity differs after audit preregistration")
    if evidence["input_sha256"] != audit_prereg.get("campaign_input_sha256"):
        raise ValueError("campaign inputs differ after audit preregistration")
    campaign_validation, campaign_validation_sha256 = runner.read_hashed_json(
        campaign / "validation.json", "campaign validation"
    )
    runner.verify_seal(campaign_validation, "campaign validation")
    if campaign_validation_sha256 != audit_prereg.get(
        "campaign_validation_file_sha256"
    ) or campaign_validation.get("content_sha256") != audit_prereg.get(
        "campaign_validation_content_sha256"
    ):
        raise ValueError("campaign validation changed after audit preregistration")
    current_source = _source_snapshot(runner)
    if current_source != audit_prereg.get("source_before"):
        raise ValueError("terminal audit helper/runtime source changed")
    expected_by_id = {row["row_id"]: row for row in campaign_prereg["rows"]}
    for entry in audit_prereg["rows"]:
        expected = expected_by_id.get(entry["row_id"])
        if expected is None:
            raise ValueError("terminal audit row is absent from campaign roster")
        row_path = campaign / "rows" / f"{entry['row_id']}.json"
        row, digest = runner.read_hashed_json(row_path, "OPT row receipt")
        if digest != entry["row_receipt_file_sha256"]:
            raise ValueError("campaign row receipt changed after audit preregistration")
        runner._verify_receipt_binding(
            row,
            evidence,
            receipt_kind="OPT_ROW",
            case_id=expected["row_id"],
            case_identity=expected,
        )
        if terminal_inventory_entry(row, digest) != entry:
            raise ValueError("campaign terminal identity changed after preregistration")
    return spec, runner, protocol, campaign_prereg, evidence, audit_prereg


def _reference_record(runner: Any, result: Any, geometry: Any) -> dict[str, Any]:
    return {
        "energies_kcal_mol": dict(result.energies_kcal_mol),
        "reference_geometry_metadata": runner._reference_geometry_metadata(geometry),
    }


def _solvent_audit(
    entry: dict[str, Any], topology: dict[str, Any], spec: dict[str, Any], runner: Any
) -> dict[str, Any]:
    oracle = _oracle()
    correction_module = importlib.import_module(
        "maple.function.calculator.extra_correction.implicit.gaussian_cha_correction"
    )
    positions = np.asarray(entry["terminal_positions_angstrom"], dtype=np.float64)
    atoms = Atoms(numbers=topology["atomic_numbers"], positions=positions)
    correction = correction_module.GaussianChaCorrection(
        atoms,
        runner._typed_topology(topology),
        expected_topology_sha256=topology["content_sha256"],
        sigma_e=entry["sigma_e"],
        order=64,
    )
    production = correction.evaluate(atoms, need_forces=True)
    production_components = {
        key: float(value) * KCAL_PER_HARTREE
        for key, value in production.components_hartree.items()
    }
    production_components["total"] = float(production.energy_hartree) * KCAL_PER_HARTREE
    production_forces = (
        np.asarray(production.forces_hartree_per_angstrom, dtype=np.float64)
        * KCAL_PER_HARTREE
    )
    reference_spec = spec["reference"]
    cache = ReferenceGeometryCache(
        oracle, topology, reference_spec["epsabs"], reference_spec["epsrel"]
    )
    reference_by_order = {}
    for order in reference_spec["orders"]:
        geometry = cache.get(positions, int(order))
        result = oracle.gaussian_cha_from_reference_geometry(
            geometry, sigma_e=entry["sigma_e"]
        )
        reference_by_order[str(order)] = _reference_record(runner, result, geometry)
    comparisons = []
    for flat in range(positions.size):
        atom, axis = divmod(flat, 3)
        stencils = []
        for step in spec["terminal_fd"]["steps_angstrom"]:
            samples = []
            for multiplier in spec["terminal_fd"]["stencil_multipliers"]:
                displaced = positions.copy()
                displaced[atom, axis] += float(multiplier) * float(step)
                geometry = cache.get(displaced, 128)
                result = oracle.gaussian_cha_from_reference_geometry(
                    geometry, sigma_e=entry["sigma_e"]
                )
                samples.append(
                    {
                        "multiplier": float(multiplier),
                        "positions_angstrom": displaced.tolist(),
                        "coordinate_sha256": coordinate_sha256(displaced),
                        "reference_total_kcal_mol": result.total_kcal_mol,
                        "reference_geometry_metadata": runner._reference_geometry_metadata(
                            geometry
                        ),
                    }
                )
            stencils.append({"step_angstrom": float(step), "samples": samples})
        comparisons.append({"atom_index": atom, "axis": axis, "stencils": stencils})
    return {
        "production_components_kcal_mol": production_components,
        "production_forces_kcal_mol_per_angstrom": production_forces.tolist(),
        "reference_by_order": reference_by_order,
        "reference_fd_comparisons": comparisons,
    }


def _combined_audit(
    entry: dict[str, Any],
    topology: dict[str, Any],
    spec: dict[str, Any],
    runner: Any,
    log: Path,
) -> dict[str, Any]:
    log.parent.mkdir(parents=True, exist_ok=True)
    positions = np.asarray(entry["terminal_positions_angstrom"], dtype=np.float64)
    atoms = Atoms(numbers=topology["atomic_numbers"], positions=positions)
    atoms.calc = runner.build_composed_calculator(
        atoms, topology, entry["sigma_e"], log
    )
    base_energy, analytic = runner._evaluate_total(atoms)
    base_solvation = runner.json_safe(atoms.calc.results.get("solvation"))
    comparisons = []
    for flat in range(positions.size):
        atom, axis = divmod(flat, 3)
        stencils = []
        for step in spec["terminal_fd"]["steps_angstrom"]:
            samples = []
            for multiplier in spec["terminal_fd"]["stencil_multipliers"]:
                displaced = positions.copy()
                displaced[atom, axis] += float(multiplier) * float(step)
                atoms.set_positions(displaced)
                energy = float(atoms.get_potential_energy(force_consistent=True))
                samples.append(
                    {
                        "multiplier": float(multiplier),
                        "positions_angstrom": displaced.tolist(),
                        "coordinate_sha256": coordinate_sha256(displaced),
                        "combined_energy_hartree": energy,
                    }
                )
            stencils.append({"step_angstrom": float(step), "samples": samples})
        comparisons.append({"atom_index": atom, "axis": axis, "stencils": stencils})
    return {
        "tag": "FRESH_TERMINAL_COMBINED_FD_CALCULATOR",
        "base_positions_angstrom": positions.tolist(),
        "base_coordinate_sha256": coordinate_sha256(positions),
        "analytic_combined_energy_hartree": base_energy,
        "analytic_forces_hartree_per_angstrom": analytic.tolist(),
        "analytic_solvation_ledger": base_solvation,
        "combined_fd_comparisons": comparisons,
    }


def _validate_samples(
    comparisons: list[dict[str, Any]],
    positions: np.ndarray,
    steps: list[float],
    *,
    energy_key: str,
    runner: Any = None,
    reference_order: int | None = None,
    epsabs: float | None = None,
    epsrel: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if reference_order is not None and runner is None:
        raise ValueError("reference metadata validation requires the pinned runner")
    if len(comparisons) != positions.size:
        raise ValueError("terminal FD Cartesian inventory differs")
    finest = []
    disagreement = []
    for flat, comparison in enumerate(comparisons):
        atom, axis = divmod(flat, 3)
        if (comparison["atom_index"], comparison["axis"]) != (atom, axis):
            raise ValueError("terminal FD Cartesian order differs")
        for stencil in comparison["stencils"]:
            step = float(stencil["step_angstrom"])
            for sample in stencil["samples"]:
                expected = positions.copy()
                expected[atom, axis] += float(sample["multiplier"]) * step
                observed = np.asarray(sample["positions_angstrom"], dtype=np.float64)
                if not np.array_equal(observed, expected):
                    raise ValueError("terminal FD sample coordinates differ")
                if coordinate_sha256(observed) != sample["coordinate_sha256"]:
                    raise ValueError("terminal FD sample coordinate hash differs")
                if (
                    reference_order is not None
                    and not runner._validate_reference_geometry_metadata(
                        sample["reference_geometry_metadata"],
                        (reference_order,),
                        epsabs,
                        epsrel,
                    )
                ):
                    raise ValueError("terminal reference metadata differs")
                if reference_order is not None:
                    validate_reference_metadata_coordinates(
                        sample["reference_geometry_metadata"], observed
                    )
                    validate_reference_error_semantics(
                        sample["reference_geometry_metadata"]
                    )
        derived = derive_five_point_force(
            comparison["stencils"], steps, energy_key=energy_key
        )
        finest.append(derived["forces"][-1])
        disagreement.append(derived["finest_step_disagreement"])
    return np.asarray(finest), np.asarray(disagreement)


def validate_terminal_record(
    record: dict[str, Any], entry: dict[str, Any], spec: dict[str, Any], runner: Any
) -> dict[str, Any]:
    runner.verify_seal(record, "terminal audit row")
    if record.get("case_identity") != entry:
        raise ValueError("terminal audit row identity differs")
    runtime = record["runtime"]
    expected_environment = spec["prohibitions"]["thread_environment"]
    expected_thread_environment = {
        key: expected_environment[key]
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
    }
    if (
        runtime["torch_num_threads"] != 1
        or runtime["thread_environment"] != expected_thread_environment
        or runtime["torch_default_dtype"] != "torch.float32"
    ):
        raise ValueError("terminal audit runtime identity differs")
    positions = np.asarray(entry["terminal_positions_angstrom"], dtype=np.float64)
    solvent = record["solvent_independent"]
    production_components = solvent["production_components_kcal_mol"]
    if (
        set(production_components) != {"polar", "cavity", "dispersion", "total"}
        or not all(
            math.isfinite(float(value)) for value in production_components.values()
        )
        or abs(
            sum(
                float(production_components[key])
                for key in ("polar", "cavity", "dispersion")
            )
            - float(production_components["total"])
        )
        > 1.0e-9
    ):
        raise ValueError("terminal production solvent component ledger differs")
    energy_errors = []
    for order in spec["reference"]["orders"]:
        reference = solvent["reference_by_order"][str(order)]
        validate_reference_metadata_coordinates(
            reference["reference_geometry_metadata"], positions
        )
        if not runner._validate_reference_geometry_metadata(
            reference["reference_geometry_metadata"],
            (int(order),),
            spec["reference"]["epsabs"],
            spec["reference"]["epsrel"],
        ):
            raise ValueError("terminal base reference metadata differs")
        validate_reference_error_semantics(reference["reference_geometry_metadata"])
        for component in ("polar", "cavity", "dispersion", "total"):
            energy_errors.append(
                float(production_components[component])
                - float(reference["energies_kcal_mol"][component])
            )
    solvent_fd, solvent_disagreement = _validate_samples(
        solvent["reference_fd_comparisons"],
        positions,
        spec["terminal_fd"]["steps_angstrom"],
        energy_key="reference_total_kcal_mol",
        runner=runner,
        reference_order=128,
        epsabs=spec["reference"]["epsabs"],
        epsrel=spec["reference"]["epsrel"],
    )
    analytic_solvent = validated_force_array(
        "terminal solvent analytic",
        solvent["production_forces_kcal_mol_per_angstrom"],
    ).reshape(-1)
    solvent_errors = solvent_fd - analytic_solvent
    solvent_gates = spec["solvent_gates"]
    solvent_metrics = {
        "energy_max_abs_kcal_mol": float(np.max(np.abs(energy_errors))),
        "force_max_abs_kcal_mol_per_angstrom": float(np.max(np.abs(solvent_errors))),
        "force_rms_kcal_mol_per_angstrom": float(np.sqrt(np.mean(solvent_errors**2))),
        "finest_step_disagreement_max_kcal_mol_per_angstrom": float(
            np.max(solvent_disagreement)
        ),
    }
    solvent_passed = bool(
        solvent_metrics["energy_max_abs_kcal_mol"]
        <= solvent_gates["energy_max_abs_kcal_mol"]
        and solvent_metrics["force_max_abs_kcal_mol_per_angstrom"]
        <= solvent_gates["force_max_abs_kcal_mol_per_angstrom"]
        and solvent_metrics["force_rms_kcal_mol_per_angstrom"]
        <= solvent_gates["force_rms_kcal_mol_per_angstrom"]
        and solvent_metrics["finest_step_disagreement_max_kcal_mol_per_angstrom"]
        <= solvent_gates["finest_step_disagreement_max_kcal_mol_per_angstrom"]
    )
    combined = record["combined_total_fd"]
    if combined.get("tag") != "FRESH_TERMINAL_COMBINED_FD_CALCULATOR":
        raise ValueError("terminal combined calculator tag differs")
    validate_combined_base_ledger(combined, positions, runner)
    combined_fd, combined_disagreement = _validate_samples(
        combined["combined_fd_comparisons"],
        positions,
        spec["terminal_fd"]["steps_angstrom"],
        energy_key="combined_energy_hartree",
    )
    analytic_combined = validated_force_array(
        "terminal combined analytic",
        combined["analytic_forces_hartree_per_angstrom"],
    ).reshape(-1)
    combined_errors = combined_fd - analytic_combined
    combined_gates = spec["combined_gates"]
    combined_metrics = {
        "force_max_abs_hartree_per_angstrom": float(np.max(np.abs(combined_errors))),
        "force_rms_hartree_per_angstrom": float(np.sqrt(np.mean(combined_errors**2))),
        "finest_step_disagreement_max_hartree_per_angstrom": float(
            np.max(combined_disagreement)
        ),
    }
    combined_passed = bool(
        combined_metrics["force_max_abs_hartree_per_angstrom"]
        <= combined_gates["force_max_abs_hartree_per_angstrom"]
        and combined_metrics["force_rms_hartree_per_angstrom"]
        <= combined_gates["force_rms_hartree_per_angstrom"]
        and combined_metrics["finest_step_disagreement_max_hartree_per_angstrom"]
        <= combined_gates["finest_step_disagreement_max_hartree_per_angstrom"]
    )
    return {
        "row_id": entry["row_id"],
        "opt_status": entry["opt_status"],
        "solvent_metrics": solvent_metrics,
        "combined_metrics": combined_metrics,
        "solvent_passed": solvent_passed,
        "combined_passed": combined_passed,
    }


def _row_receipt(
    entry: dict[str, Any],
    audit_prereg_sha256: str,
    campaign_prereg_sha256: str,
    payload: dict[str, Any],
    runner: Any,
) -> dict[str, Any]:
    return runner.seal(
        {
            "schema_version": 1,
            "receipt_kind": "TERMINAL_INDEPENDENT_EF",
            "audit_preregistration_sha256": audit_prereg_sha256,
            "campaign_preregistration_sha256": campaign_prereg_sha256,
            "case_identity": entry,
            **payload,
        }
    )


def verify_audit_record_binding(
    record: dict[str, Any],
    entry: dict[str, Any],
    audit_prereg_sha256: str,
    campaign_prereg_sha256: str,
    runner: Any,
) -> None:
    runner.verify_seal(record, "terminal audit row")
    if record.get("case_identity") != entry:
        raise ValueError("terminal audit row identity differs")
    if record.get("audit_preregistration_sha256") != audit_prereg_sha256:
        raise ValueError("terminal row audit preregistration binding differs")
    if record.get("campaign_preregistration_sha256") != campaign_prereg_sha256:
        raise ValueError("terminal row campaign preregistration binding differs")


def derive_terminal_records(
    records: list[dict[str, Any] | None],
    entries: list[dict[str, Any]],
    spec: dict[str, Any],
    runner: Any,
) -> list[dict[str, Any]]:
    if len(records) != len(entries):
        raise ValueError("terminal record denominator differs from preregistration")
    derived = []
    for record, entry in zip(records, entries, strict=True):
        if record is not None and record.get("case_identity") != entry:
            raise ValueError("terminal audit row identity differs")
        try:
            if record is None:
                raise FileNotFoundError("terminal audit row receipt is missing")
            derived.append(validate_terminal_record(record, entry, spec, runner))
        except (KeyError, TypeError, ValueError, IndexError, RuntimeError) as exc:
            derived.append(
                {
                    "row_id": entry["row_id"],
                    "opt_status": entry.get("opt_status"),
                    "solvent_passed": False,
                    "combined_passed": False,
                    "error": type(exc).__name__,
                    "message": str(exc),
                }
            )
    return derived


def run(
    campaign: Path,
    output: Path,
    campaign_prereg_sha256: str,
    audit_prereg_sha256: str,
):
    spec, runner, _, _, evidence, audit_prereg = _load_bound_context(
        campaign, output, campaign_prereg_sha256, audit_prereg_sha256
    )
    topology = evidence["topology"]
    records = []
    for entry in audit_prereg["rows"]:
        path = output / "rows" / f"{entry['row_id']}.json"
        if path.exists():
            record, _ = runner.read_hashed_json(path, "terminal audit row")
            verify_audit_record_binding(
                record,
                entry,
                audit_prereg_sha256,
                campaign_prereg_sha256,
                runner,
            )
            records.append(record)
            continue
        try:
            payload = {
                "status": "COMPLETED",
                "runtime": runner._runtime_record(),
                "solvent_independent": _solvent_audit(entry, topology, spec, runner),
                "combined_total_fd": _combined_audit(
                    entry,
                    topology,
                    spec,
                    runner,
                    output / "logs" / f"{entry['row_id']}.log",
                ),
            }
        except Exception as exc:
            payload = {
                "status": "FAILED",
                "error": type(exc).__name__,
                "message": str(exc),
            }
        record = _row_receipt(
            entry,
            audit_prereg_sha256,
            campaign_prereg_sha256,
            payload,
            runner,
        )
        write_json_new_atomic(path, record)
        records.append(record)
    _, _, _, _, post_evidence, _ = _load_bound_context(
        campaign, output, campaign_prereg_sha256, audit_prereg_sha256
    )
    assert_post_phase_snapshot_stable(
        audit_prereg, _source_snapshot(runner), post_evidence["input_sha256"]
    )
    derived = derive_terminal_records(records, audit_prereg["rows"], spec, runner)
    summary = summarize_terminal_audit(
        derived,
        [entry["row_id"] for entry in audit_prereg["rows"]],
        campaign_validated=False,
    )
    row_phase_passed = bool(
        summary["inventory_exact"]
        and summary["opt_converged_count"] == 15
        and summary["solvent_independent_pass_count"] == 15
        and summary["combined_total_fd_pass_count"] == 15
    )
    summary["status"] = (
        "TERMINAL_AUDIT_PASSED" if row_phase_passed else "GAUSSIAN_OPT_INCOMPLETE"
    )
    sealed = runner.seal(
        {
            **summary,
            "audit_preregistration_sha256": audit_prereg_sha256,
            "campaign_preregistration_sha256": campaign_prereg_sha256,
            "rows": derived,
        }
    )
    write_json_new_atomic(output / "run-summary.json", sealed)
    return sealed


def _campaign_recomputed(runner: Any, campaign: Path, campaign_sha: str) -> bool:
    existing, _ = runner.read_hashed_json(
        campaign / "validation.json", "campaign validation"
    )
    runner.verify_seal(existing, "campaign validation")
    original_writer = runner.write_json_new_atomic
    runner.write_json_new_atomic = lambda path, value: None
    try:
        recomputed = runner.validate(campaign, campaign_sha)
    finally:
        runner.write_json_new_atomic = original_writer
    return bool(
        recomputed.get("status") == "VALIDATED"
        and runner.canonical_bytes(recomputed) == runner.canonical_bytes(existing)
    )


def validate(
    campaign: Path,
    output: Path,
    campaign_prereg_sha256: str,
    audit_prereg_sha256: str,
):
    spec, runner, _, _, _, audit_prereg = _load_bound_context(
        campaign, output, campaign_prereg_sha256, audit_prereg_sha256
    )
    records: list[dict[str, Any] | None] = []
    for entry in audit_prereg["rows"]:
        try:
            record, _ = runner.read_hashed_json(
                output / "rows" / f"{entry['row_id']}.json", "terminal audit row"
            )
        except (FileNotFoundError, json.JSONDecodeError, ValueError):
            records.append(None)
            continue
        if record.get("case_identity") != entry:
            raise ValueError("terminal audit row identity differs")
        verify_audit_record_binding(
            record,
            entry,
            audit_prereg_sha256,
            campaign_prereg_sha256,
            runner,
        )
        records.append(record)
    derived = derive_terminal_records(records, audit_prereg["rows"], spec, runner)
    campaign_validated = _campaign_recomputed(runner, campaign, campaign_prereg_sha256)
    _, _, _, _, post_evidence, _ = _load_bound_context(
        campaign, output, campaign_prereg_sha256, audit_prereg_sha256
    )
    assert_post_phase_snapshot_stable(
        audit_prereg, _source_snapshot(runner), post_evidence["input_sha256"]
    )
    summary = summarize_terminal_audit(
        derived,
        [entry["row_id"] for entry in audit_prereg["rows"]],
        campaign_validated=campaign_validated,
    )
    result = runner.seal(
        {
            **summary,
            "audit_preregistration_sha256": audit_prereg_sha256,
            "campaign_preregistration_sha256": campaign_prereg_sha256,
            "rows": derived,
            "claim_boundary": spec["claims"],
            "width_selection_performed": False,
        }
    )
    write_json_new_atomic(output / "validation.json", result)
    return result


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("preregister", "run", "validate"):
        command = sub.add_parser(name)
        command.add_argument("--campaign", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument(
            "--expected-campaign-preregistration-sha256", required=True
        )
        if name in ("run", "validate"):
            command.add_argument(
                "--expected-audit-preregistration-sha256", required=True
            )
    args = parser.parse_args(argv)
    if not _inside(args.campaign, OUTPUT_ROOT):
        parser.error("--campaign must be a descendant of the frozen evidence root")
    if not _inside(args.output, args.campaign):
        parser.error("--output must be a descendant of --campaign")
    if args.output.resolve() == args.campaign.resolve():
        parser.error("--output must be a proper descendant of --campaign")
    if args.command == "preregister":
        result = preregister(
            args.campaign,
            args.output,
            args.expected_campaign_preregistration_sha256,
        )
    elif args.command == "run":
        result = run(
            args.campaign,
            args.output,
            args.expected_campaign_preregistration_sha256,
            args.expected_audit_preregistration_sha256,
        )
    else:
        result = validate(
            args.campaign,
            args.output,
            args.expected_campaign_preregistration_sha256,
            args.expected_audit_preregistration_sha256,
        )
    print(json.dumps(json_safe(result), sort_keys=True))
    if args.command == "preregister":
        succeeded = result.get("phase") == "TERMINAL_AUDIT_PREREGISTERED"
    elif args.command == "run":
        succeeded = result.get("status") == "TERMINAL_AUDIT_PASSED"
    else:
        succeeded = result.get("status") == "VALIDATED"
    return 0 if succeeded else 1


if __name__ == "__main__":
    raise SystemExit(main())
