from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys

import numpy as np


SCRIPT = (
    Path(__file__).parents[2]
    / "tools/route2_release/audit_spice_mbis_source_labels.py"
)
SPEC = importlib.util.spec_from_file_location("spice_mbis_label_audit", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_synthetic_spice_source_labels_close_in_declared_units(tmp_path: Path) -> None:
    import h5py

    dataset = tmp_path / "source.h5"
    positions_angstrom = np.asarray(
        (((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),), dtype=np.float64
    )
    charges = np.asarray((((0.2,), (-0.2,)),), dtype=np.float64)
    dipoles_eangstrom = np.asarray(
        (((0.01, 0.02, 0.03), (-0.01, 0.01, 0.02)),), dtype=np.float64
    )
    molecular_eangstrom = np.sum(
        charges[..., 0, None] * positions_angstrom + dipoles_eangstrom, axis=1
    )
    with h5py.File(dataset, "w") as handle:
        group = handle.create_group("molecule")
        group["atomic_numbers"] = np.asarray((1, 8), dtype=np.int64)
        group["positions"] = positions_angstrom / 10.0
        group["mbis_charges"] = charges
        group["mbis_dipoles"] = dipoles_eangstrom / 10.0
        group["scf_dipole"] = molecular_eangstrom / 10.0
        group["total_charge"] = np.asarray((0.0,))

    result = MODULE.audit_dataset(dataset)

    assert result["eligible_configuration_count"] == 1
    assert result["charge_closure_max_abs_e"] < 1.0e-15
    assert result["dipole_component_closure_max_abs_eangstrom"] < 1.0e-15
    assert all(result["gates"].values())


def test_help_is_dependency_light() -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "MBIS charge/dipole units" in completed.stdout

