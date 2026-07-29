from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from ase.units import Bohr

from maple.function.calculator.extra_correction.implicit.route2_v0_promolecular_density import (
    V0_PROMOLECULAR_DENSITY_ARTIFACT,
    V0_PROMOLECULAR_DENSITY_CONSTRUCTION,
    load_route2_v0_promolecular_density_table,
)


ROOT = Path(__file__).resolve().parents[2]
TABLE = (
    ROOT
    / "docs/implicit-solvation/benchmarks/"
    "route2-v0-promolecular-atomic-hf-def2-tzvpd-v1.npz"
)
MANIFEST = TABLE.with_suffix(".json")


def _table():
    return load_route2_v0_promolecular_density_table(
        table_path=TABLE,
        manifest_path=MANIFEST,
    )


def test_promolecular_table_loads_the_frozen_reference_asset():
    table = _table()

    assert table.construction == V0_PROMOLECULAR_DENSITY_CONSTRUCTION
    assert table.supported_atomic_numbers == (1, 6, 7, 8, 16, 17)
    assert table.radial_grid_bohr.shape == (2049,)
    assert table.radial_grid_bohr[0] == 0.0
    assert np.all(np.diff(table.radial_grid_bohr) > 0.0)
    assert table.radial_grid_bohr.flags.writeable is False
    assert table.densities_by_atomic_number[8].flags.writeable is False


def test_promolecular_evaluation_reproduces_tabulated_atomic_density():
    table = _table()
    selected = np.array([0, 17, 314, 2048])
    points = np.zeros((len(selected), 3))
    points[:, 0] = table.radial_grid_bohr[selected]

    value = table.evaluate(
        points,
        atomic_numbers=np.array([8]),
        atom_positions_angstrom=np.zeros((1, 3)),
    )
    np.testing.assert_allclose(
        value,
        table.densities_by_atomic_number[8][selected],
        rtol=0.0,
        atol=0.0,
    )
    assert value.flags.writeable is False
    assert np.all(value >= 0.0)


def test_promolecular_density_is_translation_covariant_and_additive():
    table = _table()
    points = np.array([[0.0, 0.0, 0.0], [2.1, -1.2, 0.7], [-3.4, 0.2, 1.1]])
    positions = np.array([[0.0, 0.0, 0.0], [0.8, -0.4, 0.3]])
    numbers = np.array([6, 8])
    shift_angstrom = np.array([0.23, -0.11, 0.37])

    original = table.evaluate(points, numbers, positions)
    shifted = table.evaluate(
        points + shift_angstrom / Bohr,
        numbers,
        positions + shift_angstrom,
    )
    np.testing.assert_allclose(shifted, original, rtol=0.0, atol=2.0e-15)

    carbon = table.evaluate(points, np.array([6]), positions[:1])
    oxygen = table.evaluate(points, np.array([8]), positions[1:])
    np.testing.assert_allclose(original, carbon + oxygen, rtol=0.0, atol=2.0e-15)


def test_promolecular_table_fails_closed_for_unsupported_elements_and_tampering(tmp_path):
    table = _table()
    with pytest.raises(ValueError, match="no frozen reference"):
        table.evaluate(
            np.zeros((1, 3)),
            atomic_numbers=np.array([35]),
            atom_positions_angstrom=np.zeros((1, 3)),
        )

    tampered = tmp_path / "table.npz"
    tampered.write_bytes(TABLE.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="manifest hash"):
        load_route2_v0_promolecular_density_table(
            table_path=tampered,
            manifest_path=MANIFEST,
        )


def test_promolecular_manifest_identifies_the_expected_scientific_asset():
    import json

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["artifact"] == V0_PROMOLECULAR_DENSITY_ARTIFACT
    assert manifest["status"] == "pass"
    assert all(result["scf_converged"] for result in manifest["results"])
    assert max(
        result["electron_count_integral_absolute_error"]
        for result in manifest["results"]
    ) < 1.0e-8
    assert min(
        result["minimum_density_e_per_bohr3"] for result in manifest["results"]
    ) >= 0.0
