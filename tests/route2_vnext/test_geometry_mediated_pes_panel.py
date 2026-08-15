from __future__ import annotations

from copy import deepcopy
import hashlib
import json

import numpy as np
import pytest

from maple.solvation.coupling.state_equation import geometry_sha256
from maple.solvation.release.geometry_mediated_panel import (
    AIMNET2_GEOMETRY_MEDIATED_PES_EXCLUDED_MOLECULES,
    AIMNET2_GEOMETRY_MEDIATED_PES_MOLECULE_IDS,
    AIMNET2_GEOMETRY_MEDIATED_PES_PANEL_CONTRACT_VERSION,
    AIMNET2_GEOMETRY_MEDIATED_PES_SHARD_CONTRACT_VERSION,
    AIMNET2_GEOMETRY_MEDIATED_SUPPORTED_ATOMIC_NUMBERS,
    aimnet2_geometry_mediated_pes_molecule,
    summarize_aimnet2_geometry_mediated_pes_panel,
    summarize_aimnet2_geometry_mediated_pes_shard,
)
from maple.solvation.release.pes_panel import (
    PES_PANEL_DIRECTION_NAMES,
    PES_PANEL_DIRECTIONAL_STEPS_A,
    PES_PANEL_VARIANT_NAMES,
    load_pes_panel,
    panel_directions,
    panel_geometries,
)


def _topology(
    label: str,
    *,
    continuum: bool = False,
    margin: float = 1.0,
    sphere_margin: float = 1.0,
    count: int = 12,
):
    digest = hashlib.sha256(label.encode()).hexdigest()
    if continuum:
        return {
            "cavity_topology_sha256": digest,
            "coefficient_count": count,
            "minimum_point_source_shell_margin_angstrom": margin,
            "minimum_sphere_tangency_margin_angstrom": sphere_margin,
            "point_source_topology_sha256": hashlib.sha256(
                f"{label}:point".encode()
            ).hexdigest(),
            "sphere_pair_topology_sha256": hashlib.sha256(
                f"{label}:sphere".encode()
            ).hexdigest(),
        }
    return {
        "topology_sha256": digest,
        "minimum_cutoff_margin_angstrom": margin,
    }


def _stationarity(atom_count: int, *, gate: bool = True):
    return {
        "absolute_residual_eV_per_e": 1.0e-14 if gate else 1.0e-4,
        "relative_residual": 1.0e-15,
        "surface_condition_number": 100.0,
        "state_dimension": 4 * atom_count,
        "thresholds": {
            "absolute_residual_eV_per_e": 1.0e-10,
            "relative_residual": 1.0e-10,
            "surface_condition_number": 1.0e12,
        },
        "gate_passed": gate,
        "capability_admitted": False,
    }


def _records(molecule_index: int = 0):
    molecule = aimnet2_geometry_mediated_pes_molecule(molecule_index)
    result = []
    for variant, atoms in panel_geometries(molecule).items():
        gradient = np.arange(1.0, 3 * len(atoms) + 1.0).reshape(len(atoms), 3) / 10.0
        center_energy = -10.0 + 0.1 * len(result)
        model_topology = _topology(f"model:{molecule.molecule_id}:{variant}")
        continuum_topology = _topology(
            f"continuum:{molecule.molecule_id}:{variant}",
            continuum=True,
            count=4 * len(atoms),
        )
        directions = {}
        for name, direction in panel_directions(atoms, molecule.molecule_id).items():
            derivative = float(np.vdot(gradient, direction))
            samples = []
            for step in PES_PANEL_DIRECTIONAL_STEPS_A:
                cubic = 0.1 * step**3
                samples.append(
                    {
                        "step_A": step,
                        "plus_energy_eV": center_energy + derivative * step + cubic,
                        "minus_energy_eV": center_energy - derivative * step - cubic,
                        "plus_model_topology": deepcopy(model_topology),
                        "minus_model_topology": deepcopy(model_topology),
                        "plus_continuum_topology": deepcopy(continuum_topology),
                        "minus_continuum_topology": deepcopy(continuum_topology),
                    }
                )
            directions[name] = {
                "direction": direction.tolist(),
                "samples": samples,
            }
        result.append(
            {
                "molecule_id": molecule.molecule_id,
                "source_record_id": molecule.source_record_id,
                "chemical_formula": molecule.chemical_formula,
                "scope_tags": list(molecule.scope_tags),
                "variant": variant,
                "geometry_sha256": geometry_sha256(atoms),
                "atomic_numbers": atoms.numbers.tolist(),
                "positions_A": atoms.positions.tolist(),
                "charge": 0,
                "multiplicity": 1,
                "center": {
                    "vacuum_energy_eV": center_energy - 0.25,
                    "continuum_energy_eV": 0.25,
                    "total_energy_eV": center_energy,
                    "source": np.zeros((len(atoms), 4)).tolist(),
                    "reaction_field": np.zeros((len(atoms), 4)).tolist(),
                    "forces_eV_per_A": (-gradient).tolist(),
                    "total_gradient_eV_per_A": gradient.tolist(),
                },
                "center_model_topology": model_topology,
                "center_continuum_topology": continuum_topology,
                "deterministic_replay": {
                    "energy_absolute_error_eV": 0.0,
                    "source_difference_norm": 0.0,
                    "gradient_difference_norm_eV_per_A": 0.0,
                    "gate_passed": True,
                },
                "reciprocity_metric_charge_gauge": {"gate_passed": True},
                "stationarity": _stationarity(len(atoms)),
                "directions": directions,
            }
        )
    return result


def test_aimnet2_pes_shard_domain_is_explicit_hcno_subset_of_frozen_asset():
    assert AIMNET2_GEOMETRY_MEDIATED_PES_SHARD_CONTRACT_VERSION.endswith("-v2")
    panel = load_pes_panel()
    assert tuple(item.molecule_id for item in panel[:17]) == (
        AIMNET2_GEOMETRY_MEDIATED_PES_MOLECULE_IDS
    )
    assert tuple(item.molecule_id for item in panel[17:]) == tuple(
        AIMNET2_GEOMETRY_MEDIATED_PES_EXCLUDED_MOLECULES
    )
    supported = set(AIMNET2_GEOMETRY_MEDIATED_SUPPORTED_ATOMIC_NUMBERS)
    assert all(set(item.atoms.numbers) <= supported for item in panel[:17])
    assert all(not set(item.atoms.numbers) <= supported for item in panel[17:])
    assert aimnet2_geometry_mediated_pes_molecule(0).molecule_id == "water"
    assert aimnet2_geometry_mediated_pes_molecule(16).molecule_id == (
        "hydrogen-peroxide"
    )
    with pytest.raises(ValueError, match=r"\[0, 16\]"):
        aimnet2_geometry_mediated_pes_molecule(17)


def test_aimnet2_pes_shard_recomputes_exact_three_by_three_by_three_coverage():
    summary = summarize_aimnet2_geometry_mediated_pes_shard(
        molecule_index=0, records=_records()
    )
    assert summary["molecule_id"] == "water"
    assert summary["variant_count"] == len(PES_PANEL_VARIANT_NAMES) == 3
    assert summary["directional_record_count"] == len(PES_PANEL_DIRECTION_NAMES) * 3
    assert summary["directional_sample_count"] == 27
    assert summary["diagnostic_gates_passed"] is True
    assert summary["maximum_directional_absolute_error_eV_per_A"] < 1.0e-6
    assert summary["minimum_neighbor_cutoff_margin_A"] == pytest.approx(1.0)
    assert summary["minimum_continuum_event_margin_A"] == pytest.approx(1.0)
    assert summary["minimum_sphere_tangency_margin_A"] == pytest.approx(1.0)
    assert all(value is False for value in summary["capabilities"].values())
    assert summary["opt_admitted"] is False
    assert summary["freq_ts_irc_admitted"] is False
    assert summary["md_admitted"] is False

    for geometry in summary["geometry_records"]:
        assert tuple(geometry["directional_force_fd"]) == PES_PANEL_DIRECTION_NAMES
        assert all(
            audit["gate_passed"] is True
            for audit in geometry["directional_force_fd"].values()
        )

    serialized = json.loads(json.dumps(_records(), sort_keys=True))
    assert (
        summarize_aimnet2_geometry_mediated_pes_shard(
            molecule_index=0, records=serialized
        )["diagnostic_gates_passed"]
        is True
    )


def test_aimnet2_pes_shard_fails_closed_on_coverage_geometry_and_dishonest_gate():
    records = _records()
    with pytest.raises(ValueError, match="exactly three"):
        summarize_aimnet2_geometry_mediated_pes_shard(
            molecule_index=0, records=records[:-1]
        )

    changed_geometry = deepcopy(records)
    changed_geometry[0]["positions_A"][0][0] += 1.0e-6
    with pytest.raises(ValueError, match="atomic numbers or positions"):
        summarize_aimnet2_geometry_mediated_pes_shard(
            molecule_index=0, records=changed_geometry
        )

    dishonest_stationarity = deepcopy(records)
    dishonest_stationarity[0]["stationarity"]["absolute_residual_eV_per_e"] = 1.0
    with pytest.raises(ValueError, match="disagrees"):
        summarize_aimnet2_geometry_mediated_pes_shard(
            molecule_index=0, records=dishonest_stationarity
        )


def test_aimnet2_pes_shard_preserves_topology_and_event_margin_failure():
    records = _records()
    event = deepcopy(records)
    displaced = event[1]["directions"]["radial-internal"]["samples"][0]
    displaced["plus_continuum_topology"] = _topology(
        "continuum:event", continuum=True, margin=0.01
    )
    summary = summarize_aimnet2_geometry_mediated_pes_shard(
        molecule_index=0, records=event
    )
    assert summary["gates"]["all_stencils_same_stratum"] is False
    assert summary["gates"]["all_continuum_event_guards"] is False
    assert summary["minimum_continuum_event_margin_A"] == pytest.approx(0.01)
    assert summary["diagnostic_gates_passed"] is False
    assert all(value is False for value in summary["capabilities"].values())

    tangency = deepcopy(records)
    sphere_sample = tangency[0]["directions"]["seeded-internal"]["samples"][0]
    sphere_sample["minus_continuum_topology"] = _topology(
        "continuum:water:reference",
        continuum=True,
        sphere_margin=0.01,
    )
    sphere_summary = summarize_aimnet2_geometry_mediated_pes_shard(
        molecule_index=0, records=tangency
    )
    assert sphere_summary["gates"]["all_stencils_same_stratum"] is True
    assert sphere_summary["gates"]["all_sphere_tangency_guards"] is False
    assert sphere_summary["minimum_sphere_tangency_margin_A"] == pytest.approx(0.01)
    assert sphere_summary["diagnostic_gates_passed"] is False


def test_aimnet2_full_pes_panel_requires_all_seventeen_raw_shards():
    assert AIMNET2_GEOMETRY_MEDIATED_PES_PANEL_CONTRACT_VERSION.endswith("-v2")
    shards = [
        {"molecule_index": index, "records": _records(index)}
        for index in range(len(AIMNET2_GEOMETRY_MEDIATED_PES_MOLECULE_IDS))
    ]
    summary = summarize_aimnet2_geometry_mediated_pes_panel(shards)
    assert summary["molecule_count"] == 17
    assert summary["geometry_count"] == 51
    assert summary["directional_record_count"] == 153
    assert summary["directional_sample_count"] == 459
    assert summary["molecule_ids"] == list(AIMNET2_GEOMETRY_MEDIATED_PES_MOLECULE_IDS)
    assert summary["failed_molecule_ids"] == []
    assert summary["diagnostic_gates_passed"] is True
    assert all(value is False for value in summary["capabilities"].values())

    serialized = json.loads(json.dumps(shards, sort_keys=True))
    assert (
        summarize_aimnet2_geometry_mediated_pes_panel(serialized)[
            "diagnostic_gates_passed"
        ]
        is True
    )

    with pytest.raises(ValueError, match="exactly 17"):
        summarize_aimnet2_geometry_mediated_pes_panel(shards[:-1])
    reordered = deepcopy(shards)
    reordered[0], reordered[1] = reordered[1], reordered[0]
    with pytest.raises(ValueError, match="indices"):
        summarize_aimnet2_geometry_mediated_pes_panel(reordered)
