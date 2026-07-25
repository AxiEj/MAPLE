from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import numpy as np
from ase import Atoms
import pytest

BENCHMARK_DIR = (
    Path(__file__).resolve().parents[2] / "docs" / "implicit-solvation" / "benchmarks"
)
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

import run_mlip_chagb_metropolis as sampling


def test_protocol_is_label_blind_and_development_only():
    protocol, _ = sampling.load_sampling_protocol(
        BENCHMARK_DIR / "mlip_chagb_metropolis_protocol.json"
    )
    boundary = protocol["execution_boundary"]
    assert boundary["energy_phase_reads_experimental_labels"] is False
    assert boundary["no_experimental_fit_or_residual_model"] is True
    assert boundary["confirmation_remains_closed"] is True
    assert {case["flexibility_bin"] for case in protocol["cases"]} == {
        "rigid",
        "limited",
        "flexible",
    }


def test_rotatable_bonds_exclude_hydrogen_and_ring_bonds():
    chain = Atoms("CCC", positions=[[0, 0, 0], [1.5, 0, 0], [3.0, 0, 0]])
    chain.info["mol2"] = {
        "bonds": [[0, 1, "1"], [1, 2, "1"]],
    }
    assert sampling.rotatable_bond_sides(chain) == [
        (0, 1, (1, 2)),
        (1, 2, (2,)),
    ]

    ring = Atoms(
        "CCC",
        positions=[[0, 0, 0], [1.5, 0, 0], [0.75, 1.2, 0]],
    )
    ring.info["mol2"] = {
        "bonds": [[0, 1, "1"], [1, 2, "1"], [2, 0, "1"]],
    }
    assert sampling.rotatable_bond_sides(ring) == []

    hydrogen = Atoms("CH", positions=[[0, 0, 0], [1.1, 0, 0]])
    hydrogen.info["mol2"] = {"bonds": [[0, 1, "1"]]}
    assert sampling.rotatable_bond_sides(hydrogen) == []


def test_torsion_rotation_preserves_axis_and_internal_distances():
    positions = np.asarray(
        [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [2.5, 1.0, 0.0], [3.0, 1.0, 1.0]]
    )
    rotated = sampling.rotate_bond_side(
        positions,
        axis_start=0,
        axis_end=1,
        side=(1, 2, 3),
        angle_radians=math.pi / 3.0,
    )
    np.testing.assert_allclose(rotated[:2], positions[:2], atol=1.0e-12)
    for i, j in [(1, 2), (2, 3), (1, 3)]:
        np.testing.assert_allclose(
            np.linalg.norm(rotated[i] - rotated[j]),
            np.linalg.norm(positions[i] - positions[j]),
            atol=1.0e-12,
        )


def test_metropolis_rule_and_simpson_integral():
    assert sampling.metropolis_accept(delta_energy_ev=-1.0, beta_ev_inverse=2.0, u=0.9)
    assert sampling.metropolis_accept(delta_energy_ev=0.5, beta_ev_inverse=2.0, u=0.3)
    assert not sampling.metropolis_accept(
        delta_energy_ev=0.5, beta_ev_inverse=2.0, u=0.5
    )
    x = np.linspace(0.0, 1.0, 5)
    y = x**2
    assert sampling.composite_simpson(x, y) == pytest.approx(1.0 / 3.0)


def test_amber_inpcrd_writer(tmp_path):
    path = tmp_path / "coords.rst7"
    positions = np.asarray([[1.0, 2.0, 3.0], [-1.5, 0.25, 4.75]])
    sampling.write_amber_inpcrd(path, positions, title="probe")
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "probe"
    assert int(lines[1]) == 2
    values = np.fromstring(" ".join(lines[2:]), sep=" ")
    np.testing.assert_allclose(values, positions.reshape(-1), atol=5.0e-8)


def test_sampling_protocol_contains_no_experimental_fields():
    protocol = json.loads(
        (BENCHMARK_DIR / "mlip_chagb_metropolis_protocol.json").read_text(
            encoding="utf-8"
        )
    )
    payload = json.dumps(protocol).lower()
    assert "experimental_kcal_mol" not in payload
    assert "experimental_value" not in payload


def test_scientific_reproduction_projection_excludes_only_runtime_fields():
    record = {
        "compound_id": "probe",
        "sampling": {
            "free_energy_kcal_mol": -1.25,
            "target_provider_seconds": 3.0,
            "wall_seconds": 4.0,
        },
    }

    projected = sampling.scientific_record_projection(record)

    assert projected == {
        "compound_id": "probe",
        "sampling": {"free_energy_kcal_mol": -1.25},
    }
    assert record["sampling"]["target_provider_seconds"] == 3.0
