#!/usr/bin/env python3
"""Run one frozen FreeSolv-12 rho-DROP + Exact-GTO total-energy record."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = REPO_ROOT / "docs/implicit-solvation/benchmarks"
for search_path in (REPO_ROOT, BENCHMARK_DIR):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

import run_mnsol_macepolar_multisolvent_pilot as benchmark  # noqa: E402
from maple.function.calculator.extra_correction.implicit.route2_atomic_reference_density import (  # noqa: E402
    load_atomic_reference_density_asset,
)
from maple.function.calculator.extra_correction.implicit.route2_engine import (  # noqa: E402
    Route2ContinuumEngine,
    molecular_dipole_response_from_density_coefficients,
    project_density_total_charge,
)
from maple.function.calculator.extra_correction.implicit.route2_moist_drop import (  # noqa: E402
    MoistRuntimeProvenance,
)
from maple.function.calculator.extra_correction.implicit.route2_rhodrop_cpcm import (  # noqa: E402
    RhoDropCPCMReactionField,
)
from maple.function.calculator.extra_correction.implicit.route2_rhodrop_profile import (  # noqa: E402
    ROUTE2_RHODROP_CPCM_OPERATIONAL_V1,
)
from maple.function.calculator.extra_correction.implicit.smd_cds import (  # noqa: E402
    HARTREE_TO_KCAL_MOL,
    smd_water_cds,
)
from maple.function.calculator.set_calculator import SetCalculator  # noqa: E402
from maple.function.read.filereader.mol2_reader import MOL2Reader  # noqa: E402
from maple.function.route2_energy_ledger import (  # noqa: E402
    LEGACY_MACE_FIELD_ENERGY_PLUS_PCM_V1,
)
from maple.function.route2_smd_profiles import (  # noqa: E402
    DDPCM_MULTISOLVENT_SMD_PROFILE,
)


ARTIFACT_ID = "route2-rhodrop-exactgto-freesolv12-total-v1"
PROFILE_ID = "route2-research-macepolar-point-l1-rhodrop-cpcm-exactgto-smdcds-v1"
EXPECTED_CHECKPOINT_SHA256 = (
    "fab8b8713c832f31a2a853aaa22fd638be8a369cbf5095e6b3e982a18d10e93a"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _json_default(value: object):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def _runtime(extension: Path, shared_library: Path) -> MoistRuntimeProvenance:
    return MoistRuntimeProvenance(
        evidence_kind="real-pinned-build",
        python_extension_path=str(extension),
        python_extension_sha256=_sha256(extension),
        shared_library_path=str(shared_library),
        shared_library_sha256=_sha256(shared_library),
        build_toolchain=(
            "gfortran-11.4.0",
            "meson-1.11.2",
            "ninja-1.13.0",
            "openmp-enabled",
        ),
    )


def _calculator(atoms, *, device: str, work_dir: Path):
    parameters = benchmark._settings(
        "water",
        DDPCM_MULTISOLVENT_SMD_PROFILE,
        benchmark.SCF_RESPONSE_MODE,
    )
    return SetCalculator(
        device,
        parameters["model"],
        str(work_dir / "model-load.out"),
        atoms=atoms,
        d4=False,
        implicit=parameters["solv"]["method"],
        solvent="water",
        model_options=parameters.get("model_options"),
        solvation_options=parameters["solv"],
        charge_options={},
    ).set_calculator()


def _cds(atoms):
    return smd_water_cds(atoms.get_chemical_symbols(), atoms.get_positions())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--record-index", type=int, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--reference-density", type=Path, required=True)
    parser.add_argument("--moist-extension", type=Path, required=True)
    parser.add_argument("--moist-library", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    started = time.perf_counter()
    selection_path = args.selection.expanduser().resolve(strict=True)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    records = selection["locked_records"]
    index = int(args.record_index)
    if not 0 <= index < len(records):
        raise ValueError("--record-index is outside the frozen FreeSolv-12 panel.")
    selected = records[index]
    dataset_root = args.dataset_root.expanduser().resolve(strict=True)
    mol2 = dataset_root / str(selected["mol2_archive_member"])
    if _sha256(mol2) != selected["mol2_sha256"]:
        raise RuntimeError("Frozen FreeSolv MOL2 SHA256 drifted.")
    atoms = MOL2Reader(str(mol2), charge=0, mult=1)
    if len(atoms) != int(selected["natoms"]):
        raise RuntimeError("Frozen FreeSolv atom count drifted.")

    reference_table = args.reference_density.expanduser().resolve(strict=True)
    asset = load_atomic_reference_density_asset(
        table_path=reference_table,
        manifest_path=reference_table.with_suffix(".json"),
    )
    extension = args.moist_extension.expanduser().resolve(strict=True)
    shared_library = args.moist_library.expanduser().resolve(strict=True)
    runtime = _runtime(extension, shared_library)
    import moist

    runtime.verify_module(moist)
    work_dir = args.work_dir.expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=False)
    model_started = time.perf_counter()
    calculator = _calculator(atoms, device=args.device, work_dir=work_dir)
    model = calculator.route2_electronic_model_adapter()
    projector = model.preprojected_field_projector()
    model_load_seconds = time.perf_counter() - model_started
    checkpoint = dict(calculator.mace_polar_checkpoint_provenance)
    if checkpoint.get("sha256") != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("MACE-POLAR checkpoint identity drifted.")

    profile = ROUTE2_RHODROP_CPCM_OPERATIONAL_V1
    fast_cpcm_settings = replace(
        profile.make_cpcm_settings(),
        require_exact_cold_replay=False,
    )

    def provider(current_atoms, *, exact_replay: bool):
        return RhoDropCPCMReactionField(
            asset,
            np.asarray(current_atoms.numbers, dtype=int),
            np.asarray(current_atoms.get_positions(), dtype=float),
            expected_total_charge_e=0.0,
            runtime=runtime,
            n_iso_e_per_bohr3=profile.n_iso_e_per_bohr3,
            cpcm_settings=(
                profile.make_cpcm_settings() if exact_replay else fast_cpcm_settings
            ),
            drop_settings=profile.make_drop_settings(),
            sigma_angstrom=profile.sigma_angstrom,
            minimum_density_e_per_bohr3=profile.minimum_density_e_per_bohr3,
            electron_count_tolerance=profile.electron_count_tolerance,
            model_field_projector=projector,
            moist_module=moist,
        )

    def factory(current_atoms):
        return provider(current_atoms, exact_replay=False)

    engine = Route2ContinuumEngine(
        reaction_field_factory=factory,
        cds_evaluator=_cds,
        settings=profile.engine_settings(total_charge_e=0.0),
        plugin_continuum_binding={
            **profile.as_provenance(),
            "profile_id": PROFILE_ID,
            "model_drive": "checkpoint-native-exact-gto-reaction-field",
            "electrostatic_energy_ledger": (
                LEGACY_MACE_FIELD_ENERGY_PLUS_PCM_V1
            ),
            "cds": "stock-native-water-smd-cds",
            "include_cds": True,
            "electrostatics_only": False,
            "model_field_projection": projector.spec.provenance,
        },
        required_electrostatic_energy_ledger=(
            LEGACY_MACE_FIELD_ENERGY_PLUS_PCM_V1
        ),
    )
    gas_state, _ = model.evaluate_state(atoms, None, compute_forces=False)
    signature = _canonical_sha256(
        {
            "base": profile.provider_cache_signature(
                atoms,
                runtime=runtime,
                total_charge_e=0.0,
            ),
            "profile_id": PROFILE_ID,
            "projector": projector.spec.provenance,
            "ledger": LEGACY_MACE_FIELD_ENERGY_PLUS_PCM_V1,
        }
    )
    solve_started = time.perf_counter()
    coupled = engine.solve_coupled_state(
        atoms,
        model,
        gas_state,
        provider_cache_signature=signature,
        electrostatic_energy_ledger=LEGACY_MACE_FIELD_ENERGY_PLUS_PCM_V1,
    )
    accelerated_seconds = time.perf_counter() - solve_started

    replay_started = time.perf_counter()
    replay_provider = provider(atoms, exact_replay=True)
    replay_drive = replay_provider.apply_scf_drive(coupled.density_coefficients)
    replay_state = replay_provider.scf_snapshot(coupled.density_coefficients)
    replay_response, _ = model.evaluate_state(
        atoms,
        replay_drive,
        compute_forces=False,
    )
    raw_replay_density = np.asarray(replay_response.density_coefficients, dtype=float)
    replay_density = project_density_total_charge(
        raw_replay_density,
        total_charge_e=0.0,
    )
    root_density = np.asarray(coupled.density_coefficients, dtype=float)
    fixed_point_residual = replay_density - root_density
    monopole_residual = float(np.max(np.abs(fixed_point_residual[:, 0])))
    dipole_residual = float(np.max(np.abs(fixed_point_residual[:, 1:])))
    molecular_dipole_residual = molecular_dipole_response_from_density_coefficients(
        np.asarray(atoms.get_positions(), dtype=float),
        fixed_point_residual,
    )
    molecular_dipole_residual_l2 = float(np.linalg.norm(molecular_dipole_residual))
    raw_charge_delta = float(
        np.sum(raw_replay_density[:, 0]) - np.sum(root_density[:, 0])
    )
    root_charge = float(np.sum(root_density[:, 0]))
    tolerance = profile.engine_settings(total_charge_e=0.0)
    replay_gate = {
        "monopole_residual_e": monopole_residual,
        "dipole_component_residual_e_angstrom": dipole_residual,
        "molecular_dipole_residual_vector_e_angstrom": [
            float(value) for value in molecular_dipole_residual
        ],
        "molecular_dipole_residual_l2_e_angstrom": molecular_dipole_residual_l2,
        "raw_response_charge_delta_e": raw_charge_delta,
        "projected_total_charge_residual_e": abs(
            float(np.sum(replay_density[:, 0])) - root_charge
        ),
        "monopole_tolerance_e": tolerance.scf_density_tolerance,
        "dipole_component_tolerance_e_angstrom": (
            tolerance.scf_dipole_tolerance_e_angstrom
        ),
        "molecular_dipole_tolerance_e_angstrom": (
            tolerance.effective_molecular_dipole_tolerance_e_angstrom
        ),
        "raw_response_charge_tolerance_e": (
            tolerance.effective_raw_response_charge_tolerance_e
        ),
        "projected_charge_tolerance_e": (
            tolerance.effective_total_charge_residual_tolerance_e
        ),
    }
    replay_gate["passed"] = bool(
        monopole_residual <= tolerance.scf_density_tolerance
        and dipole_residual <= tolerance.scf_dipole_tolerance_e_angstrom
        and molecular_dipole_residual_l2
        <= tolerance.effective_molecular_dipole_tolerance_e_angstrom
        and abs(raw_charge_delta)
        <= tolerance.effective_raw_response_charge_tolerance_e
        and replay_gate["projected_total_charge_residual_e"]
        <= tolerance.effective_total_charge_residual_tolerance_e
    )
    if not replay_gate["passed"]:
        raise RuntimeError(
            "The accelerated rho-DROP root failed the exact cold-replay "
            f"fixed-point gate: {replay_gate!r}"
        )
    replay_seconds = time.perf_counter() - replay_started
    cds = _cds(atoms)
    components = engine.compose_energy_components(
        gas_energy_ev=float(gas_state.energy_ev),
        solvent_energy_ev=float(replay_response.energy_ev),
        polarization_energy_hartree=replay_state.polarization_energy_hartree,
        cds_energy_hartree=cds.energy_hartree,
        electrostatic_energy_ledger=LEGACY_MACE_FIELD_ENERGY_PLUS_PCM_V1,
    )
    predicted = components["delta_g_solv"] * HARTREE_TO_KCAL_MOL
    experimental = float(selected["experimental_kcal_mol"])
    record = {
        "schema_version": 1,
        "artifact_id": ARTIFACT_ID,
        "profile_id": PROFILE_ID,
        "status": "success",
        "claim_boundary": (
            "Known FreeSolv-12 development diagnostic for reconstructed-density "
            "rho-DROP/CPCM plus Exact-GTO receiver and stock water SMD-CDS; not "
            "a blind, force, variational, or production admission result."
        ),
        "record_index": index,
        "compound_id": selected["compound_id"],
        "name": selected["name"],
        "experimental_kcal_mol": experimental,
        "experimental_uncertainty_kcal_mol": float(
            selected["experimental_uncertainty_kcal_mol"]
        ),
        "predicted_kcal_mol": predicted,
        "signed_error_kcal_mol": predicted - experimental,
        "absolute_error_kcal_mol": abs(predicted - experimental),
        "components_hartree": components,
        "components_kcal_mol": {
            name: value * HARTREE_TO_KCAL_MOL
            for name, value in components.items()
        },
        "iterations": len(coupled.history),
        "scf_convergence": coupled.scf_convergence,
        "final_fixed_point_residual_max": float(
            np.max(np.abs(fixed_point_residual))
        ),
        "exact_cold_replay_gate": replay_gate,
        "provider_audit": replay_provider.audit_snapshot(),
        "model_field_projection": projector.spec.provenance,
        "checkpoint": checkpoint,
        "mace_source_total_charge_e": root_charge,
        "input": {
            "selection_sha256": _sha256(selection_path),
            "mol2_sha256": _sha256(mol2),
            "reference_density_table_sha256": _sha256(reference_table),
            "reference_density_manifest_sha256": _sha256(
                reference_table.with_suffix(".json")
            ),
            "moist_extension_sha256": _sha256(extension),
            "moist_library_sha256": _sha256(shared_library),
        },
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "moist": getattr(moist, "__version__", "unknown"),
            "moist_provenance": asdict(runtime),
            "device": args.device,
        },
        "timing_seconds": {
            "model_load": model_load_seconds,
            "accelerated_scf": accelerated_seconds,
            "exact_final_replay": replay_seconds,
            "total": time.perf_counter() - started,
        },
    }
    output = args.output.expanduser().resolve()
    _write(output, record)
    print(
        json.dumps(
            {
                "output": str(output),
                "compound_id": selected["compound_id"],
                "predicted_kcal_mol": predicted,
                "absolute_error_kcal_mol": abs(predicted - experimental),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
