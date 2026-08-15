"""Preregistered AIMNet2 geometry-mediated PES-panel shard audit.

The frozen Route-2 PES asset contains twenty neutral molecules.  The local
AIMNet2 checkpoint contract exercised by this research lane declares only
H/C/N/O support, so this module explicitly restricts the panel to the first
seventeen compatible molecules and records the three S/Cl controls as excluded.

One shard contains exactly one molecule, all three frozen geometry variants,
all three frozen internal directions, and all three central-difference steps.
The summarizer recomputes every derivative from raw energies and verifies the
geometry, direction, topology, event-margin, reciprocity, replay, and stationary
continuum records.  It is an evidence helper only and never admits a public
energy, force, optimizer, Hessian, dynamics, or mutual-polarization capability.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math

import numpy as np

from maple.solvation.coupling.state_equation import geometry_sha256

from .geometry_mediated import (
    GEOMETRY_MEDIATED_REPLAY_ENERGY_TOLERANCE_EV,
    GEOMETRY_MEDIATED_REPLAY_GRADIENT_TOLERANCE_EV_PER_A,
    GEOMETRY_MEDIATED_REPLAY_SOURCE_TOLERANCE,
    summarize_geometry_mediated_directional_audit,
)
from .pes_panel import (
    PES_PANEL_DIRECTION_NAMES,
    PES_PANEL_DIRECTIONAL_STEPS_A,
    PES_PANEL_VARIANT_NAMES,
    PESPanelMolecule,
    load_pes_panel,
    panel_directions,
    panel_geometries,
)

AIMNET2_GEOMETRY_MEDIATED_PES_SHARD_CONTRACT_VERSION = (
    "route2-aimnet2-geometry-mediated-pes-shard-contract-v1"
)
AIMNET2_GEOMETRY_MEDIATED_PES_SHARD_SCHEMA_VERSION = (
    "route2-aimnet2-geometry-mediated-pes-shard-summary-v1"
)
AIMNET2_GEOMETRY_MEDIATED_PES_PANEL_CONTRACT_VERSION = (
    "route2-aimnet2-geometry-mediated-pes-panel-contract-v1"
)
AIMNET2_GEOMETRY_MEDIATED_PES_PANEL_SCHEMA_VERSION = (
    "route2-aimnet2-geometry-mediated-pes-panel-summary-v1"
)
AIMNET2_GEOMETRY_MEDIATED_SUPPORTED_ATOMIC_NUMBERS = (1, 6, 7, 8)
AIMNET2_GEOMETRY_MEDIATED_PES_MOLECULE_IDS = (
    "water",
    "methanol",
    "ethanol",
    "acetone",
    "acetonitrile",
    "benzene",
    "methane",
    "trans-butane",
    "dimethyl-ether",
    "formic-acid",
    "acetic-acid",
    "acetaldehyde",
    "acetamide",
    "ethylamine",
    "pyridine",
    "nitromethane",
    "hydrogen-peroxide",
)
AIMNET2_GEOMETRY_MEDIATED_PES_EXCLUDED_MOLECULES = {
    "thiophene": "contains sulfur outside the local H/C/N/O checkpoint contract",
    "methanethiol": "contains sulfur outside the local H/C/N/O checkpoint contract",
    "chloroform": "contains chlorine outside the local H/C/N/O checkpoint contract",
}
AIMNET2_GEOMETRY_MEDIATED_STATIONARITY_ABSOLUTE_TOLERANCE_EV_PER_E = 1.0e-10
AIMNET2_GEOMETRY_MEDIATED_STATIONARITY_RELATIVE_TOLERANCE = 1.0e-10
AIMNET2_GEOMETRY_MEDIATED_STATIONARITY_MAXIMUM_CONDITION_NUMBER = 1.0e12


def _finite_float(value: object, *, name: str, nonnegative: bool = False) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be numeric.")
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        qualifier = "finite and non-negative" if nonnegative else "finite"
        raise ValueError(f"{name} must be {qualifier}.")
    return result


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping.")
    return value


def _eligible_panel() -> tuple[PESPanelMolecule, ...]:
    panel = load_pes_panel()
    identifiers = tuple(molecule.molecule_id for molecule in panel)
    expected_all = AIMNET2_GEOMETRY_MEDIATED_PES_MOLECULE_IDS + tuple(
        AIMNET2_GEOMETRY_MEDIATED_PES_EXCLUDED_MOLECULES
    )
    if identifiers != expected_all:
        raise RuntimeError(
            "Frozen PES-panel ordering changed from the AIMNet2 shard contract."
        )
    supported = set(AIMNET2_GEOMETRY_MEDIATED_SUPPORTED_ATOMIC_NUMBERS)
    for molecule in panel[: len(AIMNET2_GEOMETRY_MEDIATED_PES_MOLECULE_IDS)]:
        if not set(int(value) for value in molecule.atoms.numbers) <= supported:
            raise RuntimeError(
                f"Eligible molecule {molecule.molecule_id!r} left the H/C/N/O domain."
            )
    for molecule in panel[len(AIMNET2_GEOMETRY_MEDIATED_PES_MOLECULE_IDS) :]:
        if set(int(value) for value in molecule.atoms.numbers) <= supported:
            raise RuntimeError(
                f"Excluded molecule {molecule.molecule_id!r} no longer needs exclusion."
            )
    return panel[: len(AIMNET2_GEOMETRY_MEDIATED_PES_MOLECULE_IDS)]


def aimnet2_geometry_mediated_pes_molecule(
    molecule_index: int,
) -> PESPanelMolecule:
    """Return one exact H/C/N/O molecule selected by stable shard index."""

    if isinstance(molecule_index, bool) or not isinstance(
        molecule_index, (int, np.integer)
    ):
        raise TypeError("molecule_index must be an integer.")
    panel = _eligible_panel()
    index = int(molecule_index)
    if not 0 <= index < len(panel):
        raise ValueError(
            f"molecule_index must be in [0, {len(panel) - 1}] for this contract."
        )
    return panel[index]


def _replay_summary(record: Mapping[str, object]) -> dict[str, object]:
    energy = _finite_float(
        record.get("energy_absolute_error_eV"),
        name="replay energy error",
        nonnegative=True,
    )
    source = _finite_float(
        record.get("source_difference_norm"),
        name="replay source error",
        nonnegative=True,
    )
    gradient = _finite_float(
        record.get("gradient_difference_norm_eV_per_A"),
        name="replay gradient error",
        nonnegative=True,
    )
    gate = (
        energy <= GEOMETRY_MEDIATED_REPLAY_ENERGY_TOLERANCE_EV
        and source <= GEOMETRY_MEDIATED_REPLAY_SOURCE_TOLERANCE
        and gradient <= GEOMETRY_MEDIATED_REPLAY_GRADIENT_TOLERANCE_EV_PER_A
    )
    if record.get("gate_passed") is not gate:
        raise ValueError("deterministic replay gate disagrees with raw errors.")
    return {
        "energy_absolute_error_eV": energy,
        "source_difference_norm": source,
        "gradient_difference_norm_eV_per_A": gradient,
        "gate_passed": gate,
    }


def _stationarity_summary(record: Mapping[str, object]) -> dict[str, object]:
    absolute = _finite_float(
        record.get("absolute_residual_eV_per_e"),
        name="stationarity absolute residual",
        nonnegative=True,
    )
    relative = _finite_float(
        record.get("relative_residual"),
        name="stationarity relative residual",
        nonnegative=True,
    )
    condition = _finite_float(
        record.get("surface_condition_number"),
        name="stationarity condition number",
        nonnegative=True,
    )
    dimension = record.get("state_dimension")
    if isinstance(dimension, bool) or not isinstance(dimension, (int, np.integer)):
        raise TypeError("stationarity state dimension must be an integer.")
    dimension = int(dimension)
    if dimension < 1:
        raise ValueError("stationarity state dimension must be positive.")
    thresholds = _mapping(record.get("thresholds"), name="stationarity thresholds")
    expected_thresholds = {
        "absolute_residual_eV_per_e": (
            AIMNET2_GEOMETRY_MEDIATED_STATIONARITY_ABSOLUTE_TOLERANCE_EV_PER_E
        ),
        "relative_residual": (
            AIMNET2_GEOMETRY_MEDIATED_STATIONARITY_RELATIVE_TOLERANCE
        ),
        "surface_condition_number": (
            AIMNET2_GEOMETRY_MEDIATED_STATIONARITY_MAXIMUM_CONDITION_NUMBER
        ),
    }
    if set(thresholds) != set(expected_thresholds) or any(
        _finite_float(thresholds[name], name=f"stationarity threshold {name}")
        != expected
        for name, expected in expected_thresholds.items()
    ):
        raise ValueError("stationarity thresholds changed from the shard contract.")
    gate = (
        absolute <= AIMNET2_GEOMETRY_MEDIATED_STATIONARITY_ABSOLUTE_TOLERANCE_EV_PER_E
        and relative <= AIMNET2_GEOMETRY_MEDIATED_STATIONARITY_RELATIVE_TOLERANCE
        and condition <= AIMNET2_GEOMETRY_MEDIATED_STATIONARITY_MAXIMUM_CONDITION_NUMBER
    )
    if record.get("gate_passed") is not gate:
        raise ValueError("stationarity gate disagrees with raw measurements.")
    if record.get("capability_admitted") is not False:
        raise ValueError("stationarity diagnostic must not admit a capability.")
    return {
        "absolute_residual_eV_per_e": absolute,
        "relative_residual": relative,
        "surface_condition_number": condition,
        "state_dimension": dimension,
        "gate_passed": gate,
        "capability_admitted": False,
    }


def _geometry_record(
    raw: Mapping[str, object],
    *,
    molecule: PESPanelMolecule,
    variant: str,
) -> dict[str, object]:
    geometries = panel_geometries(molecule)
    expected_atoms = geometries[variant]
    if (
        raw.get("molecule_id") != molecule.molecule_id
        or raw.get("source_record_id") != molecule.source_record_id
        or raw.get("chemical_formula") != molecule.chemical_formula
        or tuple(raw.get("scope_tags", ())) != molecule.scope_tags
        or raw.get("variant") != variant
        or raw.get("geometry_sha256") != geometry_sha256(expected_atoms)
    ):
        raise ValueError("PES shard geometry identity changed from the frozen asset.")
    numbers = np.asarray(raw.get("atomic_numbers"))
    positions = np.asarray(raw.get("positions_A"), dtype=float)
    if not np.array_equal(numbers, expected_atoms.numbers) or not np.array_equal(
        positions, expected_atoms.positions
    ):
        raise ValueError("PES shard atomic numbers or positions changed.")
    if raw.get("charge") != 0 or raw.get("multiplicity") != 1:
        raise ValueError("AIMNet2 PES shards are restricted to neutral singlets.")

    center = _mapping(raw.get("center"), name="geometry center")
    energy = _finite_float(center.get("total_energy_eV"), name="center energy")
    vacuum_energy = _finite_float(
        center.get("vacuum_energy_eV"), name="center vacuum energy"
    )
    continuum_energy = _finite_float(
        center.get("continuum_energy_eV"), name="center continuum energy"
    )
    if not math.isclose(
        energy,
        vacuum_energy + continuum_energy,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError("center energy ledger does not close.")
    gradient = np.asarray(center.get("total_gradient_eV_per_A"), dtype=float)
    if gradient.shape != (len(expected_atoms), 3) or not np.all(np.isfinite(gradient)):
        raise ValueError("center gradient must be finite with shape (N,3).")
    forces = np.asarray(center.get("forces_eV_per_A"), dtype=float)
    source = np.asarray(center.get("source"), dtype=float)
    reaction_field = np.asarray(center.get("reaction_field"), dtype=float)
    if (
        forces.shape != gradient.shape
        or source.shape != (len(expected_atoms), 4)
        or reaction_field.shape != source.shape
        or not np.all(np.isfinite(forces))
        or not np.all(np.isfinite(source))
        or not np.all(np.isfinite(reaction_field))
    ):
        raise ValueError("center force/source/reaction-field arrays are invalid.")
    if not np.array_equal(forces, -gradient):
        raise ValueError("center force is not the exact negative scalar gradient.")
    if not np.array_equal(
        source[:, 1:], np.zeros_like(source[:, 1:])
    ) or not math.isclose(
        float(np.sum(source[:, 0])), 0.0, rel_tol=0.0, abs_tol=1.0e-10
    ):
        raise ValueError("center point-l0 source violates the neutral-charge contract.")
    model_topology = _mapping(
        raw.get("center_model_topology"), name="center model topology"
    )
    continuum_topology = _mapping(
        raw.get("center_continuum_topology"), name="center continuum topology"
    )
    reciprocity = _mapping(
        raw.get("reciprocity_metric_charge_gauge"), name="reciprocity audit"
    )
    if reciprocity.get("gate_passed") is not True:
        reciprocity_gate = False
    else:
        reciprocity_gate = True
    replay = _replay_summary(
        _mapping(raw.get("deterministic_replay"), name="deterministic replay")
    )
    stationarity = _stationarity_summary(
        _mapping(raw.get("stationarity"), name="stationarity audit")
    )
    if stationarity["state_dimension"] != 4 * len(expected_atoms):
        raise ValueError("stationarity dimension changed from surface_lmax=1.")
    if continuum_topology.get("coefficient_count") != stationarity["state_dimension"]:
        raise ValueError("continuum topology and stationarity dimensions disagree.")

    raw_directions = _mapping(raw.get("directions"), name="direction records")
    if len(raw_directions) != len(PES_PANEL_DIRECTION_NAMES) or set(
        raw_directions
    ) != set(PES_PANEL_DIRECTION_NAMES):
        raise ValueError("direction record coverage changed from the frozen contract.")
    expected_directions = panel_directions(expected_atoms, molecule.molecule_id)
    audits: dict[str, object] = {}
    neighbor_margins = [
        _finite_float(
            model_topology.get("minimum_cutoff_margin_angstrom"),
            name="center neighbor cutoff margin",
            nonnegative=True,
        )
    ]
    continuum_margins = [
        _finite_float(
            continuum_topology.get("minimum_point_source_shell_margin_angstrom"),
            name="center continuum event margin",
            nonnegative=True,
        )
    ]
    for direction_name in PES_PANEL_DIRECTION_NAMES:
        direction_record = _mapping(
            raw_directions[direction_name], name=f"{direction_name} record"
        )
        direction = np.asarray(direction_record.get("direction"), dtype=float)
        if not np.array_equal(direction, expected_directions[direction_name]):
            raise ValueError(
                f"{direction_name} changed from the frozen panel direction."
            )
        samples = direction_record.get("samples")
        if not isinstance(samples, Sequence) or isinstance(samples, (str, bytes)):
            raise TypeError("directional samples must be a sequence of mappings.")
        normalized_samples = tuple(
            _mapping(sample, name=f"{direction_name} sample") for sample in samples
        )
        for sample in normalized_samples:
            for sign in ("plus", "minus"):
                sampled_model = _mapping(
                    sample.get(f"{sign}_model_topology"),
                    name=f"{direction_name} {sign} model topology",
                )
                sampled_continuum = _mapping(
                    sample.get(f"{sign}_continuum_topology"),
                    name=f"{direction_name} {sign} continuum topology",
                )
                neighbor_margins.append(
                    _finite_float(
                        sampled_model.get("minimum_cutoff_margin_angstrom"),
                        name="sampled neighbor cutoff margin",
                        nonnegative=True,
                    )
                )
                continuum_margins.append(
                    _finite_float(
                        sampled_continuum.get(
                            "minimum_point_source_shell_margin_angstrom"
                        ),
                        name="sampled continuum event margin",
                        nonnegative=True,
                    )
                )
        audits[direction_name] = summarize_geometry_mediated_directional_audit(
            analytic_gradient_eV_per_A=gradient,
            direction=direction,
            center_model_topology=model_topology,
            center_continuum_topology=continuum_topology,
            samples=normalized_samples,
            reciprocity_audit=reciprocity,
            expected_steps_A=PES_PANEL_DIRECTIONAL_STEPS_A,
        )

    directional_gate = all(
        bool(_mapping(audits[name], name="directional audit").get("gate_passed"))
        for name in PES_PANEL_DIRECTION_NAMES
    )
    gate = (
        replay["gate_passed"] is True
        and stationarity["gate_passed"] is True
        and reciprocity_gate
        and directional_gate
    )
    return {
        "molecule_id": molecule.molecule_id,
        "variant": variant,
        "geometry_sha256": geometry_sha256(expected_atoms),
        "vacuum_energy_eV": vacuum_energy,
        "continuum_energy_eV": continuum_energy,
        "total_energy_eV": energy,
        "deterministic_replay": replay,
        "stationarity": stationarity,
        "reciprocity_metric_charge_gauge_gate_passed": reciprocity_gate,
        "directional_force_fd": audits,
        "minimum_neighbor_cutoff_margin_A": min(neighbor_margins),
        "minimum_continuum_event_margin_A": min(continuum_margins),
        "gate_passed": gate,
    }


def summarize_aimnet2_geometry_mediated_pes_shard(
    *, molecule_index: int, records: Sequence[Mapping[str, object]]
) -> dict[str, object]:
    """Validate and summarize one exact H/C/N/O three-geometry shard."""

    molecule = aimnet2_geometry_mediated_pes_molecule(molecule_index)
    values = tuple(records)
    if len(values) != len(PES_PANEL_VARIANT_NAMES):
        raise ValueError(
            "AIMNet2 geometry-mediated PES shard requires exactly three variants."
        )
    by_variant = {str(record.get("variant")): record for record in values}
    if len(by_variant) != len(values) or tuple(by_variant) != PES_PANEL_VARIANT_NAMES:
        raise ValueError("PES shard variant coverage or order is invalid.")
    summaries = [
        _geometry_record(by_variant[variant], molecule=molecule, variant=variant)
        for variant in PES_PANEL_VARIANT_NAMES
    ]

    directional = [
        audit
        for summary in summaries
        for audit in _mapping(
            summary["directional_force_fd"], name="directional force summary"
        ).values()
    ]
    topology_gates = [
        _mapping(audit, name="directional audit")["topology"] for audit in directional
    ]
    maximum_condition = max(
        float(
            _mapping(summary["stationarity"], name="stationarity")[
                "surface_condition_number"
            ]
        )
        for summary in summaries
    )
    maximum_directional_error = max(
        float(record["absolute_error_eV_per_A"])
        for audit in directional
        for record in _mapping(audit, name="directional audit")["records"]
    )
    minimum_neighbor_margin = min(
        float(summary["minimum_neighbor_cutoff_margin_A"]) for summary in summaries
    )
    minimum_continuum_margin = min(
        float(summary["minimum_continuum_event_margin_A"]) for summary in summaries
    )
    gates = {
        "all_deterministic_replays": all(
            _mapping(summary["deterministic_replay"], name="replay").get("gate_passed")
            is True
            for summary in summaries
        ),
        "all_stationarity_audits": all(
            _mapping(summary["stationarity"], name="stationarity").get("gate_passed")
            is True
            for summary in summaries
        ),
        "all_reciprocity_metric_charge_gauge_audits": all(
            summary["reciprocity_metric_charge_gauge_gate_passed"] is True
            for summary in summaries
        ),
        "all_directional_force_fd": all(
            _mapping(audit, name="directional audit").get("gate_passed") is True
            for audit in directional
        ),
        "all_stencils_same_stratum": all(
            _mapping(topology, name="topology").get("all_stencils_same_stratum") is True
            for topology in topology_gates
        ),
        "all_neighbor_cutoff_guards": all(
            _mapping(topology, name="topology").get("all_neighbor_cutoff_guards_passed")
            is True
            for topology in topology_gates
        ),
        "all_continuum_event_guards": all(
            _mapping(topology, name="topology").get("all_continuum_event_guards_passed")
            is True
            for topology in topology_gates
        ),
    }
    return {
        "schema_version": AIMNET2_GEOMETRY_MEDIATED_PES_SHARD_SCHEMA_VERSION,
        "contract_version": AIMNET2_GEOMETRY_MEDIATED_PES_SHARD_CONTRACT_VERSION,
        "molecule_index": int(molecule_index),
        "molecule_id": molecule.molecule_id,
        "source_record_id": molecule.source_record_id,
        "chemical_formula": molecule.chemical_formula,
        "supported_atomic_numbers": list(
            AIMNET2_GEOMETRY_MEDIATED_SUPPORTED_ATOMIC_NUMBERS
        ),
        "excluded_molecules": dict(AIMNET2_GEOMETRY_MEDIATED_PES_EXCLUDED_MOLECULES),
        "variant_count": len(summaries),
        "directional_record_count": len(directional),
        "directional_sample_count": len(directional)
        * len(PES_PANEL_DIRECTIONAL_STEPS_A),
        "maximum_surface_condition_number": maximum_condition,
        "maximum_directional_absolute_error_eV_per_A": maximum_directional_error,
        "minimum_neighbor_cutoff_margin_A": minimum_neighbor_margin,
        "minimum_continuum_event_margin_A": minimum_continuum_margin,
        "geometry_records": summaries,
        "gates": gates,
        "diagnostic_gates_passed": all(gates.values()),
        "capabilities": {tier: False for tier in ("E", "F", "H", "V", "M")},
        "opt_admitted": False,
        "freq_ts_irc_admitted": False,
        "md_admitted": False,
    }


def summarize_aimnet2_geometry_mediated_pes_panel(
    shards: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Recompute and aggregate all seventeen exact H/C/N/O shards."""

    values = tuple(shards)
    expected_count = len(AIMNET2_GEOMETRY_MEDIATED_PES_MOLECULE_IDS)
    if len(values) != expected_count:
        raise ValueError(
            f"AIMNet2 geometry-mediated PES panel requires exactly {expected_count} "
            "shards."
        )
    summaries: list[dict[str, object]] = []
    for expected_index, shard in enumerate(values):
        raw_index = shard.get("molecule_index")
        if (
            isinstance(raw_index, bool)
            or not isinstance(raw_index, (int, np.integer))
            or int(raw_index) != expected_index
        ):
            raise ValueError("PES-panel shard indices changed from frozen order.")
        records = shard.get("records")
        if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
            raise TypeError("Every PES-panel shard requires raw geometry records.")
        normalized_records = tuple(
            _mapping(record, name="PES-panel geometry record") for record in records
        )
        summaries.append(
            summarize_aimnet2_geometry_mediated_pes_shard(
                molecule_index=expected_index,
                records=normalized_records,
            )
        )

    gate_names = tuple(_mapping(summaries[0]["gates"], name="shard gates"))
    if any(
        tuple(_mapping(summary["gates"], name="shard gates")) != gate_names
        for summary in summaries[1:]
    ):
        raise RuntimeError("PES-panel shard gate schemas disagree.")
    gates = {
        name: all(
            _mapping(summary["gates"], name="shard gates").get(name) is True
            for summary in summaries
        )
        for name in gate_names
    }
    failed_molecule_ids = [
        str(summary["molecule_id"])
        for summary in summaries
        if summary["diagnostic_gates_passed"] is not True
    ]
    return {
        "schema_version": AIMNET2_GEOMETRY_MEDIATED_PES_PANEL_SCHEMA_VERSION,
        "contract_version": AIMNET2_GEOMETRY_MEDIATED_PES_PANEL_CONTRACT_VERSION,
        "molecule_count": len(summaries),
        "molecule_ids": [str(summary["molecule_id"]) for summary in summaries],
        "geometry_count": sum(int(summary["variant_count"]) for summary in summaries),
        "directional_record_count": sum(
            int(summary["directional_record_count"]) for summary in summaries
        ),
        "directional_sample_count": sum(
            int(summary["directional_sample_count"]) for summary in summaries
        ),
        "maximum_surface_condition_number": max(
            float(summary["maximum_surface_condition_number"]) for summary in summaries
        ),
        "maximum_directional_absolute_error_eV_per_A": max(
            float(summary["maximum_directional_absolute_error_eV_per_A"])
            for summary in summaries
        ),
        "minimum_neighbor_cutoff_margin_A": min(
            float(summary["minimum_neighbor_cutoff_margin_A"]) for summary in summaries
        ),
        "minimum_continuum_event_margin_A": min(
            float(summary["minimum_continuum_event_margin_A"]) for summary in summaries
        ),
        "failed_molecule_ids": failed_molecule_ids,
        "shard_summaries": summaries,
        "gates": gates,
        "diagnostic_gates_passed": not failed_molecule_ids and all(gates.values()),
        "capabilities": {tier: False for tier in ("E", "F", "H", "V", "M")},
        "opt_admitted": False,
        "freq_ts_irc_admitted": False,
        "md_admitted": False,
    }


__all__ = [
    "AIMNET2_GEOMETRY_MEDIATED_PES_EXCLUDED_MOLECULES",
    "AIMNET2_GEOMETRY_MEDIATED_PES_MOLECULE_IDS",
    "AIMNET2_GEOMETRY_MEDIATED_PES_PANEL_CONTRACT_VERSION",
    "AIMNET2_GEOMETRY_MEDIATED_PES_PANEL_SCHEMA_VERSION",
    "AIMNET2_GEOMETRY_MEDIATED_PES_SHARD_CONTRACT_VERSION",
    "AIMNET2_GEOMETRY_MEDIATED_PES_SHARD_SCHEMA_VERSION",
    "AIMNET2_GEOMETRY_MEDIATED_SUPPORTED_ATOMIC_NUMBERS",
    "AIMNET2_GEOMETRY_MEDIATED_STATIONARITY_ABSOLUTE_TOLERANCE_EV_PER_E",
    "AIMNET2_GEOMETRY_MEDIATED_STATIONARITY_MAXIMUM_CONDITION_NUMBER",
    "AIMNET2_GEOMETRY_MEDIATED_STATIONARITY_RELATIVE_TOLERANCE",
    "aimnet2_geometry_mediated_pes_molecule",
    "summarize_aimnet2_geometry_mediated_pes_shard",
    "summarize_aimnet2_geometry_mediated_pes_panel",
]
