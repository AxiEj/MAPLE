from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

from maple.function.calculator.analytic_protocol import geometry_sha256
from maple.function.read.filereader.mol2_reader import MOL2Reader


ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "maple/function/calculator/model"
WATER_MOL2 = (
    ROOT / "tests/solvation/data/independent_obc2_analytic/water.mol2"
)

# Frozen by the execution leader before adapter qualification.  This is an
# analytic-operation guard, not a change to the legacy gas E/F graph builder.
CUTOFF_GUARD_POLICY = "neighbor-cutoff-guard-v1"
CUTOFF_GUARD_MARGIN_ANGSTROM = 0.003

AIMNET2_SHA256 = "85ba59d8c78eb4d3185f6b1614df79706427f7ca72f53f2d90e365a1723d953d"
MACEOFF23M_SHA256 = "ac172fdf9b5173fef4c64667739dbd06f230b0167b4ae2c67e8c08033254c9ee"
WATER_MOL2_SHA256 = "54a77f1a50e0f37782aba1244143b660a8dba554444f5a342d243eefa10f27a9"
WATER_GEOMETRY_SHA256 = (
    "8ce7924bdd0f3d41806588f9e0e6cd68eb38022543b38335356bcd04b3266d4d"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _water() -> Atoms:
    assert _sha256(WATER_MOL2) == WATER_MOL2_SHA256
    atoms = MOL2Reader(str(WATER_MOL2), charge=0, mult=1)
    assert geometry_sha256(atoms) == WATER_GEOMETRY_SHA256
    return atoms


def _unit_direction(size: int, seed: int) -> np.ndarray:
    direction = np.random.default_rng(seed).normal(size=size)
    return direction / np.linalg.norm(direction)


def _require_checkpoint(filename: str, expected_sha256: str) -> Path:
    path = MODEL_DIR / filename
    if not path.is_file():
        pytest.skip(f"Local checkpoint is unavailable: {path}")
    assert _sha256(path) == expected_sha256
    return path


def _assert_frozen_point_identity(point, *, model: str, dtype: str) -> None:
    assert point.cell.model == model
    assert point.cell.effective_dtype == dtype
    assert point.cell.d4 is False
    assert point.cell.source_energy_unit == "eV"
    assert point.charge == 0.0
    assert point.multiplicity == 1.0
    assert point.atom_topology_sha256 is None
    assert point.neighbor_graph_sha256
    assert len(point.neighbor_graph_sha256) == 64
    assert point.minimum_cutoff_gap_angstrom > CUTOFF_GUARD_MARGIN_ANGSTROM
    assert point.cell.cutoff_guard_policy == CUTOFF_GUARD_POLICY
    assert point.cell.cutoff_guard_margin_angstrom == pytest.approx(
        CUTOFF_GUARD_MARGIN_ANGSTROM,
        abs=0.0,
    )


def _identity_options(point) -> dict:
    return {
        key: json.loads(encoded_value)
        for key, encoded_value in point.cell.options
    }


def test_aimnet2_cpu_native_n32_same_scalar_hessian_and_direct_hvp(monkeypatch):
    torch = pytest.importorskip("torch")
    from maple.function.calculator.aimnet._aimnet2_calculator import (
        AIMNet2Calculator,
    )

    checkpoint = _require_checkpoint("aimnet2.pt", AIMNET2_SHA256)
    atoms = _water()
    calculator = AIMNet2Calculator(
        torch.device("cpu"),
        model="aimnet2",
        model_path=str(checkpoint),
        coulomb_method="simple",
        implicit="none",
    )
    ef_operation = calculator._analytic_energy_forces_operation(
        atoms,
        need_forces=True,
    )
    point = ef_operation.identity
    _assert_frozen_point_identity(point, model="aimnet2", dtype="torch.float32")
    assert point.cell.checkpoint_sha256 == AIMNET2_SHA256
    assert dict(point.active_cutoffs_angstrom) == {
        "short_range": 5.0,
        "long_range": 5.0,
    }
    assert _identity_options(point)["coulomb_method"] == "simple"

    calculator.calculate(atoms, properties=["energy", "forces"])
    public_energy = float(calculator.results["energy"])
    public_forces = np.asarray(calculator.results["forces"], dtype=np.float64)

    hessian_operation = calculator._analytic_hessian_operation(atoms)
    raw_hessian = np.asarray(
        hessian_operation.hessian_hartree_per_angstrom2,
        dtype=np.float64,
    )
    direction = _unit_direction(3 * len(atoms), seed=2026092701)
    hvp_operation = calculator._analytic_hvp_operation(
        atoms,
        direction,
    )
    assert ef_operation.identity == hessian_operation.identity == hvp_operation.identity
    assert ef_operation.operation == "energy_forces"
    assert hessian_operation.operation == "hessian"
    assert hvp_operation.operation == "hvp"
    np.testing.assert_array_equal(hvp_operation.direction, direction)
    direct_hvp = np.asarray(
        hvp_operation.hvp_hartree_per_angstrom2,
        dtype=np.float64,
    )
    analytic_forces = np.asarray(
        hvp_operation.forces_hartree_per_angstrom,
        dtype=np.float64,
    ).reshape(-1, 3)
    analytic_energy = float(np.asarray(hvp_operation.energy_hartree))

    assert abs(analytic_energy - public_energy) <= 2.0e-5
    force_error = analytic_forces - public_forces
    assert np.max(np.abs(force_error)) <= 2.0e-6
    assert np.sqrt(np.mean(force_error**2)) <= 5.0e-7
    np.testing.assert_allclose(
        direct_hvp,
        raw_hessian @ direction,
        atol=5.0e-6,
        rtol=5.0e-5,
    )

    # Backend calculate must hand its typed gas Hessian to the common finalizer;
    # re-entering public get_hessian would compose solvent twice when attached.
    monkeypatch.setattr(
        calculator,
        "get_hessian",
        lambda *_args, **_kwargs: pytest.fail(
            "calculate(hessian) re-entered public get_hessian"
        ),
    )
    calculator.calculate(atoms, properties=["energy", "forces", "hessian"])
    np.testing.assert_allclose(
        calculator.results["hessian"],
        raw_hessian,
        atol=0.0,
        rtol=0.0,
    )

    # The public direct-HVP route must never construct or multiply a dense H.
    monkeypatch.setattr(
        calculator,
        "_analytic_hessian",
        lambda _atoms: pytest.fail("direct HVP materialized the full Hessian"),
    )
    public_hvp, public_hvp_forces, public_hvp_energy = calculator.get_hvp(
        atoms,
        direction,
    )
    np.testing.assert_allclose(public_hvp, direct_hvp, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(
        np.asarray(public_hvp_forces).reshape(-1, 3),
        analytic_forces,
        atol=0.0,
        rtol=0.0,
    )
    assert float(np.asarray(public_hvp_energy)) == pytest.approx(
        analytic_energy,
        abs=0.0,
    )


def test_maceoff23m_cpu_native_d64_same_scalar_hessian_and_direct_hvp(monkeypatch):
    torch = pytest.importorskip("torch")
    from maple.function.calculator.mace._mace_calculator import MACECalculator

    checkpoint = _require_checkpoint("maceoff23m.pt", MACEOFF23M_SHA256)
    atoms = _water()
    calculator = MACECalculator(
        torch.device("cpu"),
        model="maceoff23m",
        model_path=str(checkpoint),
        implicit="none",
    )
    ef_operation = calculator._analytic_energy_forces_operation(
        atoms,
        need_forces=True,
    )
    point = ef_operation.identity
    _assert_frozen_point_identity(
        point,
        model="maceoff23m",
        dtype="torch.float64",
    )
    assert point.cell.checkpoint_sha256 == MACEOFF23M_SHA256
    assert dict(point.active_cutoffs_angstrom) == {"radius_graph": 5.0}

    calculator.calculate(atoms, properties=["energy", "forces"])
    public_energy = float(calculator.results["energy"])
    public_forces = np.asarray(calculator.results["forces"], dtype=np.float64)

    hessian_operation = calculator._analytic_hessian_operation(atoms)
    raw_hessian = np.asarray(
        hessian_operation.hessian_hartree_per_angstrom2,
        dtype=np.float64,
    )
    direction = _unit_direction(3 * len(atoms), seed=2026092702)
    hvp_operation = calculator._analytic_hvp_operation(
        atoms,
        direction,
    )
    assert ef_operation.identity == hessian_operation.identity == hvp_operation.identity
    assert ef_operation.operation == "energy_forces"
    assert hessian_operation.operation == "hessian"
    assert hvp_operation.operation == "hvp"
    np.testing.assert_array_equal(hvp_operation.direction, direction)
    direct_hvp = np.asarray(
        hvp_operation.hvp_hartree_per_angstrom2,
        dtype=np.float64,
    )
    analytic_forces = np.asarray(
        hvp_operation.forces_hartree_per_angstrom,
        dtype=np.float64,
    ).reshape(-1, 3)
    analytic_energy = float(np.asarray(hvp_operation.energy_hartree))

    assert abs(analytic_energy - public_energy) <= (
        2.0e-10 + 1.0e-9 * abs(public_energy)
    )
    force_error = analytic_forces - public_forces
    assert np.max(np.abs(force_error)) <= 2.0e-7
    assert np.sqrt(np.mean(force_error**2)) <= 5.0e-8
    np.testing.assert_allclose(
        direct_hvp,
        raw_hessian @ direction,
        atol=1.0e-9,
        rtol=1.0e-9,
    )

    monkeypatch.setattr(
        calculator,
        "get_hessian",
        lambda *_args, **_kwargs: pytest.fail(
            "calculate(hessian) re-entered public get_hessian"
        ),
    )
    calculator.calculate(atoms, properties=["energy", "forces", "hessian"])
    np.testing.assert_allclose(
        calculator.results["hessian"],
        raw_hessian,
        atol=0.0,
        rtol=0.0,
    )

    monkeypatch.setattr(
        calculator,
        "_analytic_hessian",
        lambda _atoms: pytest.fail("direct HVP materialized the full Hessian"),
    )
    public_hvp, public_hvp_forces, public_hvp_energy = calculator.get_hvp(
        atoms,
        direction,
    )
    np.testing.assert_allclose(public_hvp, direct_hvp, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(
        np.asarray(public_hvp_forces).reshape(-1, 3),
        analytic_forces,
        atol=0.0,
        rtol=0.0,
    )
    assert float(np.asarray(public_hvp_energy)) == pytest.approx(
        analytic_energy,
        abs=0.0,
    )


@pytest.mark.parametrize(
    ("backend", "filename", "expected_sha256"),
    [
        ("aimnet2", "aimnet2.pt", AIMNET2_SHA256),
        ("maceoff23m", "maceoff23m.pt", MACEOFF23M_SHA256),
    ],
)
def test_new_analytic_paths_reject_cutoff_adjacent_geometry_without_changing_ef(
    backend,
    filename,
    expected_sha256,
):
    torch = pytest.importorskip("torch")

    checkpoint = _require_checkpoint(filename, expected_sha256)
    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [4.998, 0.0, 0.0]])
    atoms.info.update(charge=0, mult=1)
    if backend == "aimnet2":
        from maple.function.calculator.aimnet._aimnet2_calculator import (
            AIMNet2Calculator,
        )

        calculator = AIMNet2Calculator(
            torch.device("cpu"),
            model="aimnet2",
            model_path=str(checkpoint),
            coulomb_method="simple",
            implicit="none",
        )
    else:
        from maple.function.calculator.mace._mace_calculator import MACECalculator

        calculator = MACECalculator(
            torch.device("cpu"),
            model="maceoff23m",
            model_path=str(checkpoint),
            implicit="none",
        )

    # Existing ordinary gas E/F behavior stays unchanged by the new derivative
    # guard; only analytic H/HVP must fail closed at the discontinuity.
    calculator.calculate(atoms, properties=["energy", "forces"])
    assert np.isfinite(calculator.results["energy"])
    assert np.isfinite(calculator.results["forces"]).all()

    with pytest.raises(ValueError, match="neighbor-cutoff-guard-v1|cutoff"):
        calculator._analytic_hessian_operation(atoms)
    with pytest.raises(ValueError, match="neighbor-cutoff-guard-v1|cutoff"):
        calculator._analytic_hvp_operation(
            atoms,
            _unit_direction(6, seed=2026092703),
        )


def test_unverified_mace_model_names_are_not_added_to_route1_wrappers():
    from maple.function.calculator.mace._mace_calculator import MACECalculator
    from maple.function.calculator.mace._mace_general_calculator import (
        MACEModelCalculator,
    )

    # Cached upstream `.model` files are separate unresolved format/provenance
    # cells.  They must not be admitted by widening the Route-1 `.pt` wrapper.
    assert MACECalculator.MODEL_NAMES == (
        "maceoff23s",
        "maceoff23m",
        "maceoff23l",
        "egret",
    )
    assert MACEModelCalculator.MODEL_NAMES == ("maceomol",)
