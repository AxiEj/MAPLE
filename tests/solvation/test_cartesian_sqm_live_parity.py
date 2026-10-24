"""Opt-in native AmberTools parity; no package installation or method fallback.

MAPLE_TEST_ANTECHAMBER=/absolute/bin/antechamber python -m pytest -q -s ...
Raw provider artifacts and exact comparisons are retained under pytest basetemp.
"""

from copy import deepcopy
import json
import os
from pathlib import Path

from ase.build import molecule
import numpy as np
import pytest

from maple.function.calculator.extra_correction.implicit.charges import prepare_charges
from maple.function.calculator.extra_correction.implicit.correction import (
    ImplicitSolvationCorrection,
)
from maple.function.read.filereader.mol2_reader import MOL2Reader


@pytest.fixture
def native_executable():
    executable = os.environ.get("MAPLE_TEST_ANTECHAMBER")
    if not executable:
        pytest.skip("Set MAPLE_TEST_ANTECHAMBER for explicit native AmberTools parity")
    assert Path(executable).is_file(), "explicit native parity executable is missing"
    return executable


def _cartesian(name, *, high_precision=False):
    atoms = molecule(name)
    atoms.positions = np.round(atoms.positions, 3)
    if high_precision:
        atoms.positions += (
            1 + np.arange(atoms.positions.size).reshape(atoms.positions.shape)
        ) * 0.0000123456789
    atoms.info.update(charge=0, mult=1)
    return atoms


def _sqm_input_positions(path):
    rows = Path(path).read_text().split("/\n")[1].strip().splitlines()
    return np.array([[float(value) for value in row.split()[2:]] for row in rows])


@pytest.mark.parametrize("name", ["H2O", "CH3OH", "C2H6"])
@pytest.mark.parametrize("method", ["am1bcc", "abcg2"])
def test_native_grid_chain_matches_unchanged_highlevel_protocol(
    tmp_path, native_executable, name, method
):
    atoms = _cartesian(name)
    options = {"source": "maple", "method": method, "executable": native_executable}
    result = prepare_charges(atoms, options, tmp_path / "direct")
    # Existing user-MOL2 provider path is the independent unchanged comparator.
    typed = MOL2Reader(result.provenance["provider_input_mol2"], charge=0, mult=1)
    standard = prepare_charges(typed, options, tmp_path / "standard")
    direct_output = MOL2Reader(result.provenance["output_mol2"], charge=0, mult=1)
    standard_output = MOL2Reader(standard.provenance["output_mol2"], charge=0, mult=1)
    for key in ("atom_ids", "atom_names", "atom_types", "bonds"):
        assert direct_output.info["mol2"][key] == standard_output.info["mol2"][key]
    np.testing.assert_array_equal(direct_output.numbers, standard_output.numbers)
    np.testing.assert_array_equal(
        direct_output.get_initial_charges(), standard_output.get_initial_charges()
    )
    np.testing.assert_array_equal(result.charges, standard.charges)
    sqm = result.provenance["cartesian_sqm"]
    np.testing.assert_array_equal(
        _sqm_input_positions(sqm["sqm_input"]), atoms.positions
    )
    assert (
        sqm["sqm_default_namelist"]
        == (tmp_path / "standard" / "sqm.in").read_text().split("/\n")[0] + "/\n"
    )
    assert not (Path(sqm["sqm_input"]).parent / "sqm.in").exists()
    record = {
        "molecule": name,
        "method": method,
        "status": "exact-parity",
        "raw_charge_max_error_e": 0.0,
        "normalized_charge_max_error_e": 0.0,
        "graph_types_names_order_equal": True,
        "default_namelist_equal": True,
        "audit": str(tmp_path),
    }
    (tmp_path / "parity.json").write_text(json.dumps(record, indent=2))
    print(json.dumps(record))


@pytest.mark.parametrize("name", ["H2O", "CH3OH", "C2H6"])
@pytest.mark.parametrize("method", ["am1bcc", "abcg2"])
def test_native_highprecision_preserves_source_not_optimized_geometry(
    tmp_path, native_executable, name, method
):
    atoms = _cartesian(name, high_precision=True)
    before, environment = deepcopy(atoms), dict(os.environ)
    result = prepare_charges(
        atoms,
        {"source": "maple", "method": method, "executable": native_executable},
        tmp_path / "direct",
    )
    sqm = result.provenance["cartesian_sqm"]
    assert _sqm_input_positions(sqm["sqm_input"]).tobytes() == atoms.positions.tobytes()
    assert result.prepared_atoms.positions.tobytes() == atoms.positions.tobytes()
    assert atoms.positions.tobytes() == before.positions.tobytes()
    assert atoms.info == before.info and set(atoms.arrays) == set(before.arrays)
    assert dict(os.environ) == environment
    assert (
        np.max(
            np.abs(
                np.asarray(sqm["optimized_geometry"]["positions_angstrom"])
                - atoms.positions
            )
        )
        > 5e-5
    )
    assert sqm["source_to_bcc_serialized_maximum_error_angstrom"] > 5e-5
    assert abs(float(result.charges.sum())) < 1e-14
    assert result.provenance["labeled_bond_graph_verified"]
    assert [
        row["provider_index"]
        for row in json.loads(Path(result.provenance["atom_mapping"]).read_text())
    ] == list(range(len(atoms)))
    assert not (Path(sqm["sqm_input"]).parent / "sqm.in").exists()
    record = {
        "molecule": name,
        "method": method,
        "status": "exact-source-roundtrip",
        "sum_e": float(result.charges.sum()),
        "caller_environment_unchanged": True,
        "sqm_input_bitwise_equal": True,
        "final_mol2_bitwise_equal": True,
        "bcc_serialization_error_A": sqm[
            "source_to_bcc_serialized_maximum_error_angstrom"
        ],
        "audit": str(tmp_path),
    }
    (tmp_path / "precision.json").write_text(json.dumps(record, indent=2))
    print(json.dumps(record))


def test_native_force_hessian_reuse_frozen_preparation(
    tmp_path, native_executable, monkeypatch
):
    atoms = _cartesian("H2O", high_precision=True)
    runtime = ImplicitSolvationCorrection(
        atoms,
        {"source": "maple", "executable": native_executable},
        {"method": "gb", "experimental": True},
        output=tmp_path / "reuse.out",
    )
    from maple.function.calculator.extra_correction.implicit import ambertools

    monkeypatch.setattr(
        ambertools.subprocess,
        "run",
        lambda *a, **k: pytest.fail("preparation reran during derivative evaluation"),
    )
    hashes = {path: path.read_bytes() for path in runtime.audit_dir.rglob("*sqm*.in")}
    charges = atoms.get_initial_charges().copy()
    for _ in range(2):
        force = runtime.evaluate(atoms, need_forces=True).forces_hartree_per_angstrom
        hessian = runtime.get_hessian(atoms)
        assert force is not None and np.isfinite(force).all()
        assert hessian.shape == (9, 9) and np.isfinite(hessian).all()
    assert hashes == {path: path.read_bytes() for path in hashes}
    np.testing.assert_array_equal(atoms.get_initial_charges(), charges)
