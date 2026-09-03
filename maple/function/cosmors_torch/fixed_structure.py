"""Pure-Torch MACE-EF/COSMO-RS fixed-structure workflow and CLI."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from ase.io import read

from maple.function.calculator.extra_correction.implicit.mace_polar_ef import (
    MACEPolarEFConfig,
    MACEPolarEFEnergyModel,
)
from maple.function.calculator.extra_correction.implicit.mace_polar_ef_stationary import (
    MACEPolarEFSCFSettings,
)
from maple.function.mlip_cosmo_rs import open_cosmors_24a_cavity_radii

from .kse import (
    ActivationSolvationFreeEnergy,
    SolvationFreeEnergy,
    compute_relative_kinetic_solvent_effect,
)
from .mace_ef_segment_cosmo import (
    MACEPolarEFSegmentCOSMOConfig,
    MACEPolarEFSegmentCOSMOCoupling,
)
from .segment_cosmo import (
    HARTREE_EV,
    TorchSegmentCOSMO,
    TorchSegmentCOSMOConfig,
)
from .surface import build_sigma_profile, parse_orca_cosmo, read_sigma_profile
from .thermodynamics import open24a_solvation_free_energy

FIXED_STRUCTURE_PROVIDER_IDENTITY = (
    "mace-polar-ef-v2+torch-segment-cosmo-swig-v1+openCOSMO-RS-24a"
)


def _resolved_path(value: object, *, base: Path) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = base / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _integer(value: object, *, name: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    return value


def _positive_float(value: object, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return result


def _validate_fixed_structure_cutoff_margins(
    species_payloads: Sequence[dict[str, Any]],
    *,
    acknowledged: bool,
) -> list[str]:
    reduced = [
        str(item["name"])
        for item in species_payloads
        if _positive_float(
            item.get("cutoff_margin_angstrom", 0.05),
            name=f"{item['name']} cutoff_margin_angstrom",
        )
        < 0.05
    ]
    if reduced and not acknowledged:
        raise ValueError(
            "Reduced MACE cutoff topology margins are fixed-structure-only; "
            "set acknowledge_fixed_structure_cutoff_margin=true explicitly."
        )
    return reduced


def _load_species(
    payload: dict[str, Any],
    *,
    base: Path,
    checkpoint_path: Path,
    device: str,
    angular_degree: int,
    scf_payload: dict[str, Any],
):
    import torch

    name = str(payload["name"]).strip()
    if not name:
        raise ValueError("Every fixed-structure species needs a nonempty name.")
    path = _resolved_path(payload["xyz"], base=base)
    atoms = read(path, index=0)
    indices = payload.get("atom_indices")
    if indices is not None:
        selected = tuple(
            _integer(value, name="atom index", minimum=0) for value in indices
        )
        if not selected or len(set(selected)) != len(selected):
            raise ValueError("atom_indices must be a nonempty unique sequence.")
        if max(selected) >= len(atoms):
            raise ValueError("atom_indices select beyond the XYZ atom count.")
        atoms = atoms[list(selected)]
    atomic_numbers = tuple(int(value) for value in atoms.numbers)
    symbols = tuple(atoms.get_chemical_symbols())
    positions = np.asarray(atoms.positions, dtype=np.float64)
    charge = _integer(payload.get("charge", 0), name=f"{name} charge")
    multiplicity = _integer(
        payload.get("multiplicity", 1),
        name=f"{name} multiplicity",
        minimum=1,
    )
    ring_atom_count = _integer(
        payload.get("ring_atom_count", 0),
        name=f"{name} ring_atom_count",
        minimum=0,
    )
    cutoff_margin = _positive_float(
        payload.get("cutoff_margin_angstrom", 0.05),
        name=f"{name} cutoff_margin_angstrom",
    )
    electronic_config = MACEPolarEFConfig(
        checkpoint_path=str(checkpoint_path),
        atomic_numbers=atomic_numbers,
        total_charge=charge,
        spin_multiplicity=multiplicity,
        device=device,
        cutoff_margin_angstrom=cutoff_margin,
    )
    electronic = MACEPolarEFEnergyModel(electronic_config)
    gas = electronic.evaluate(
        positions,
        np.zeros((len(atomic_numbers), 4), dtype=float),
    )
    continuum = TorchSegmentCOSMO(
        TorchSegmentCOSMOConfig(
            atomic_numbers=atomic_numbers,
            radii_angstrom=tuple(open_cosmors_24a_cavity_radii(symbols)),
            angular_degree=angular_degree,
        )
    )
    scf = MACEPolarEFSCFSettings(
        electronic_passivity_policy="record-known-failure-diagnostic",
        **scf_payload,
    )
    coupling = MACEPolarEFSegmentCOSMOCoupling(
        MACEPolarEFSegmentCOSMOConfig(
            electronic=electronic_config,
            continuum=continuum,
            scf=scf,
        ),
        electronic_model=electronic,
    )
    coupled = coupling.evaluate(positions)
    surface = coupling.surface_from_result(positions, coupled, name=name)
    total_difference_hartree = (coupled.total_energy_ev - gas.energy_ev) / HARTREE_EV
    profile = replace(
        build_sigma_profile(surface),
        dielectric_energy_hartree=(
            surface.dielectric_energy_hartree.new_tensor(total_difference_hartree)
        ),
        dielectric_energy_role="total-solvated-minus-gas",
    )
    result = {
        "name": name,
        "xyz": str(path),
        "atom_indices": None if indices is None else list(indices),
        "symbols": list(symbols),
        "charge": charge,
        "multiplicity": multiplicity,
        "ring_atom_count": ring_atom_count,
        "cutoff_margin_angstrom": cutoff_margin,
        "gas_energy_ev": gas.energy_ev,
        "conductor_total_energy_ev": coupled.total_energy_ev,
        "conductor_total_minus_gas_ev": coupled.total_energy_ev - gas.energy_ev,
        "boundary_polarization_energy_ev": coupled.continuum_energy_ev,
        "scf_iterations": coupled.iterations,
        "maximum_source_residual": coupled.maximum_source_residual,
        "maximum_field_replay_difference": coupled.maximum_field_replay_difference,
        "surface_segment_count": int(surface.segment_areas_angstrom2.numel()),
        "surface_area_angstrom2": float(surface.cavity_area_angstrom2),
        "cavity_volume_angstrom3": float(surface.cavity_volume_angstrom3),
        "surface_screening_charge_e": float(surface.segment_screening_charge_e.sum()),
        "mace_source_charge_e": float(surface.molecular_charge_e),
        "electronic_passivity": dict(coupled.provenance["electronic_passivity"]),
        "continuum_configuration_sha256": continuum.config.configuration_sha256,
    }
    del coupling, electronic
    torch.cuda.empty_cache()
    return profile, ring_atom_count, result


def evaluate_fixed_structure_payload(
    payload: dict[str, Any],
    *,
    base_directory: str | Path = ".",
) -> dict[str, Any]:
    """Evaluate one reaction in multiple solvents without external executables."""

    base = Path(base_directory).expanduser().resolve()
    checkpoint = _resolved_path(payload["checkpoint_path"], base=base)
    device = str(payload.get("device", "cuda:0"))
    temperature = float(payload.get("temperature_k", 298.15))
    if not math.isclose(temperature, 298.15, rel_tol=0.0, abs_tol=1.0e-9):
        raise ValueError("The open24a fixed-structure workflow is locked to 298.15 K.")
    angular_degree = _integer(
        payload.get("angular_degree", 4),
        name="angular_degree",
        minimum=2,
    )
    scf_payload = dict(payload.get("scf", {}))
    if "electronic_passivity_policy" in scf_payload:
        raise ValueError(
            "electronic_passivity_policy is owned by this diagnostic workflow."
        )
    acknowledge_ions = payload.get("acknowledge_unvalidated_ions") is True

    transition_payload = dict(payload["transition_state"])
    reactant_payloads = [dict(item) for item in payload["reactants"]]
    if not reactant_payloads:
        raise ValueError("At least one reactant is required.")
    species_payloads = [transition_payload, *reactant_payloads]
    names = [str(item["name"]).strip() for item in species_payloads]
    if len(set(names)) != len(names):
        raise ValueError("Transition-state and reactant names must be unique.")
    cutoff_margin_acknowledged = (
        payload.get("acknowledge_fixed_structure_cutoff_margin") is True
    )
    reduced_cutoff_margins = _validate_fixed_structure_cutoff_margins(
        species_payloads,
        acknowledged=cutoff_margin_acknowledged,
    )

    profiles = {}
    ring_counts = {}
    species_records = {}
    for item in species_payloads:
        profile, rings, record = _load_species(
            item,
            base=base,
            checkpoint_path=checkpoint,
            device=device,
            angular_degree=angular_degree,
            scf_payload=scf_payload,
        )
        profiles[record["name"]] = profile
        ring_counts[record["name"]] = rings
        species_records[record["name"]] = record

    solvent_payloads = [dict(item) for item in payload["solvents"]]
    if not solvent_payloads:
        raise ValueError("At least one solvent profile is required.")
    solvent_names = [str(item["name"]).strip().lower() for item in solvent_payloads]
    if len(set(solvent_names)) != len(solvent_names):
        raise ValueError("Solvent names must be unique.")
    reference_solvent = str(payload["reference_solvent"]).strip().lower()
    if reference_solvent not in solvent_names:
        raise ValueError("reference_solvent is absent from solvents.")

    activations = {}
    solvent_records = {}
    for solvent_payload, solvent_name in zip(
        solvent_payloads, solvent_names, strict=True
    ):
        solvent_path = _resolved_path(solvent_payload["profile"], base=base)
        if solvent_path.suffix.lower() == ".json":
            solvent_profile = read_sigma_profile(solvent_path)
            profile_format = "maple-torch-cosmors-sigma-profile"
        else:
            solvent_profile = build_sigma_profile(parse_orca_cosmo(solvent_path))
            profile_format = "orca-cosmo-interoperability"
        volume = solvent_payload.get("liquid_molar_volume_cm3_mol")
        states = {}
        energy_records = {}
        for species_name in names:
            result = open24a_solvation_free_energy(
                profiles[species_name],
                solvent_profile,
                temperature_k=temperature,
                ring_atom_count=ring_counts[species_name],
                solvent_liquid_molar_volume_cm3_mol=volume,
                acknowledge_unvalidated_ions=acknowledge_ions,
            )
            state = SolvationFreeEnergy(
                species=species_name,
                solvent=solvent_name,
                delta_g_solvation_kcal_mol=float(
                    result.delta_g_solvation_kcal_mol.detach().cpu()
                ),
                temperature_k=temperature,
                standard_state="open24a-1atm-gas-to-pure-liquid",
                provider_identity=FIXED_STRUCTURE_PROVIDER_IDENTITY,
            )
            states[species_name] = state
            energy_records[species_name] = result.as_dict()
        activation = ActivationSolvationFreeEnergy.from_states(
            states[names[0]],
            tuple(states[name] for name in names[1:]),
        )
        activations[solvent_name] = activation
        solvent_records[solvent_name] = {
            "profile": str(solvent_path),
            "profile_sha256": hashlib.sha256(solvent_path.read_bytes()).hexdigest(),
            "profile_format": profile_format,
            "profile_source_identity": solvent_profile.source_identity,
            "delta_g_solvation": energy_records,
            "delta_g_activation_solvation_kcal_mol": (
                activation.delta_g_activation_solvation_kcal_mol
            ),
        }

    reference = activations[reference_solvent]
    kse = {}
    for solvent_name, activation in activations.items():
        kse[solvent_name] = compute_relative_kinetic_solvent_effect(
            activation,
            reference,
        ).as_dict()
    ionic_species = any(
        abs(float(profile.molecular_charge_e.detach())) > 2.0e-8
        for profile in profiles.values()
    )
    passivity_failures = sorted(
        name
        for name, record in species_records.items()
        if record["electronic_passivity"]["passivity_passed"] is not True
    )
    diagnostic_reasons = ["mace-ef-surface-outside-open24a-orca-fit-domain"]
    if ionic_species:
        diagnostic_reasons.append("open24a-ionic-parameterization-unvalidated")
    if passivity_failures:
        diagnostic_reasons.append("mace-ef-electronic-passivity-failed")
    return {
        "schema_version": 1,
        "workflow": "MACE-EF -> Torch segment COSMO -> Torch openCOSMO-RS 24a",
        "external_executable_invoked": False,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "provider_identity": FIXED_STRUCTURE_PROVIDER_IDENTITY,
        "temperature_k": temperature,
        "angular_degree": angular_degree,
        "reference_solvent": reference_solvent,
        "acknowledge_unvalidated_ions": acknowledge_ions,
        "acknowledge_fixed_structure_cutoff_margin": cutoff_margin_acknowledged,
        "reduced_cutoff_margin_species": reduced_cutoff_margins,
        "parameterization_scope": "neutral-molecules",
        "diagnostic_only": True,
        "diagnostic_reasons": diagnostic_reasons,
        "electronic_passivity_failures": passivity_failures,
        "species": species_records,
        "solvents": solvent_records,
        "relative_kinetic_solvent_effects": kse,
        "claim_boundary": (
            "The implementation is executable and equation-tested. MACE-EF "
            "segment surfaces are not the ORCA BP86/def2-TZVPD surfaces used "
            "to fit open24a, and open24a ions remain unvalidated diagnostics."
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run fixed-structure MACE-EF/Torch-COSMO-RS KSE calculations."
    )
    parser.add_argument("input", type=Path, help="JSON workflow input")
    parser.add_argument("output", type=Path, help="JSON result path")
    arguments = parser.parse_args(argv)
    input_path = arguments.input.expanduser().resolve()
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    result = evaluate_fixed_structure_payload(
        payload,
        base_directory=input_path.parent,
    )
    result["input_sha256"] = hashlib.sha256(input_path.read_bytes()).hexdigest()
    output_path = arguments.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "FIXED_STRUCTURE_PROVIDER_IDENTITY",
    "evaluate_fixed_structure_payload",
    "main",
]
