"""Discrete PBSA SAV cavity energy, not a force or complete CHA endpoint."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from maple.function.calculator.extra_correction.implicit.torch_pbsa_exact_cavity import (
    cavity_from_native_grid,
)

_ROOT = Path(__file__).resolve().parents[2]
_INPUTS = _ROOT / ".omx/benchmarks/route1-torch-full-cha/panel30/continuum_inputs"
_ORACLE = _ROOT / ".omx/benchmarks/route1-torch-amber-exact-sp/native-oracle30-v1"
_INPUT_MANIFEST_FILE_SHA256 = (
    "344711992c25d89f8413306208ce0a1dcaf9910f02042aa98200364361a9ae9b"
)
_TRANSLATIONS = (
    _ROOT
    / ".omx/benchmarks/route1-torch-amber-exact-sp/cavity-translation-oracle-v1.json"
)
_TRANSLATION_FILE_SHA256 = (
    "cb5901517b92929c3e969a774bbea2e501f0a4e42299406595dfefee0d4fef81"
)
_SHIFT_SCREEN = (
    _ROOT / ".omx/benchmarks/route1-torch-amber-exact-sp/cavity-shift-panel30-v1.json"
)
_SHIFT_SCREEN_FILE_SHA256 = (
    "55d5f399931a14a2933baf48acda61b4691ac341213aaa0c0fab3c8b863f02cf"
)
_NATIVE_STEP_INPUT = (
    _ROOT
    / ".omx/benchmarks/route1-numerical-repair-20260905/reference/nonpolar-parameters.json"
)


def test_one_site_integer_lattice_count_is_an_independent_reference():
    positions = torch.zeros((1, 3), dtype=torch.float64)
    rmin = torch.tensor([1.7], dtype=torch.float64)
    result = cavity_from_native_grid(positions, rmin)
    radius = 1.7 + 1.3
    expected_count = sum(
        (0.5 * x) ** 2 + (0.5 * y) ** 2 + (0.5 * z) ** 2 <= radius**2
        for x in range(-6, 7)
        for y in range(-6, 7)
        for z in range(-6, 7)
    )
    assert result.occupied_voxels == expected_count
    assert result.energy_kcal_mol.item() == pytest.approx(
        0.0378 * 0.5**3 * expected_count - 0.5692, abs=1e-13
    )
    assert result.supports_forces is False


def test_native_half_angstrom_translation_keeps_lattice_count():
    positions = torch.tensor(
        [[-0.22, 0.13, -0.06], [1.42, -0.83, 0.64]], dtype=torch.float64
    )
    rmin = torch.tensor([1.9069, 1.4593], dtype=torch.float64)
    baseline = cavity_from_native_grid(positions, rmin)
    shifted = cavity_from_native_grid(
        positions + torch.tensor([0.5, -1.0, 1.5], dtype=torch.float64), rmin
    )
    assert shifted.occupied_voxels == baseline.occupied_voxels
    assert shifted.energy_kcal_mol.item() == baseline.energy_kcal_mol.item()


def test_cavity_rejects_gradient_request_and_unbounded_grid():
    positions = torch.zeros((1, 3), dtype=torch.float64, requires_grad=True)
    with pytest.raises(ValueError, match="energy-only"):
        cavity_from_native_grid(positions, torch.tensor([1.7], dtype=torch.float64))
    with pytest.raises(RuntimeError, match="grid budget"):
        cavity_from_native_grid(
            positions.detach(),
            torch.tensor([100.0], dtype=torch.float64),
            max_grid_points=10_000,
        )


def test_cavity_requires_coordinates_after_native_seven_decimal_serialization():
    raw = torch.tensor([[0.123456789, 0.0, 0.0]], dtype=torch.float64)
    with pytest.raises(ValueError, match="serialized"):
        cavity_from_native_grid(raw, torch.tensor([1.7], dtype=torch.float64))
    serialized = torch.tensor([[0.1234568, 0.0, 0.0]], dtype=torch.float64)
    assert cavity_from_native_grid(serialized, torch.tensor([1.7], dtype=torch.float64))


@pytest.mark.parametrize(
    "coordinate,radius",
    [(0.0, 1e19), (0.0, 1e20), (0.0, 1e308)],
)
def test_cavity_rejects_unrepresentable_grid_before_integer_conversion(
    coordinate, radius
):
    with pytest.raises(RuntimeError, match="grid.*range"):
        cavity_from_native_grid(
            torch.full((1, 3), coordinate, dtype=torch.float64),
            torch.tensor([radius], dtype=torch.float64),
        )


def test_cavity_bounds_pair_work_not_only_grid_points():
    with pytest.raises(RuntimeError, match="pair.*budget"):
        cavity_from_native_grid(
            torch.zeros((2, 3), dtype=torch.float64),
            torch.full((2,), 1.7, dtype=torch.float64),
            max_pair_evaluations=1,
        )


def test_cavity_rejects_an_unaffordable_single_grid_point(monkeypatch):
    from maple.function.calculator.extra_correction.implicit import (
        torch_pbsa_exact_cavity as module,
    )

    monkeypatch.setattr(module, "_MAX_PAIR_TEMPORARY_BYTES", 32)
    with pytest.raises(RuntimeError, match="memory budget"):
        cavity_from_native_grid(
            torch.zeros((1, 3), dtype=torch.float64),
            torch.tensor([1.7], dtype=torch.float64),
        )


@pytest.mark.parametrize("coordinate", [10_000.0, -1_000.0, 1e10, 1e20, -1e20])
def test_cavity_rejects_coordinate_tokens_wider_than_native_f12_7(coordinate):
    with pytest.raises(ValueError, match="serialized"):
        cavity_from_native_grid(
            torch.tensor([[coordinate, 0.0, 0.0]], dtype=torch.float64),
            torch.tensor([1.7], dtype=torch.float64),
        )


@pytest.mark.parametrize("coordinate", [9999.0, -999.0])
def test_cavity_accepts_valid_native_field_width(coordinate):
    rmin = torch.tensor([1.7], dtype=torch.float64)
    shifted = cavity_from_native_grid(
        torch.tensor([[coordinate, 0.0, 0.0]], dtype=torch.float64), rmin
    )
    baseline = cavity_from_native_grid(torch.zeros((1, 3), dtype=torch.float64), rmin)
    assert shifted.occupied_voxels == baseline.occupied_voxels


def test_frozen_panel30_cavity_matches_full_precision_native_components():
    manifest_path = _INPUTS / "manifest.json"
    oracle_path = _ORACLE / "summary.json"
    if not manifest_path.is_file() and not oracle_path.is_file():
        pytest.skip("optional frozen panel and native oracle are not installed")
    assert manifest_path.is_file() and oracle_path.is_file()
    manifest_bytes = manifest_path.read_bytes()
    assert hashlib.sha256(manifest_bytes).hexdigest() == _INPUT_MANIFEST_FILE_SHA256
    manifest = json.loads(manifest_bytes)
    oracle = json.loads(oracle_path.read_bytes())
    assert manifest["case_count"] == oracle["success_count"] == 30
    assert [record["compound_id"] for record in manifest["records"]] == oracle[
        "ordered_ids"
    ]
    for entry in manifest["records"]:
        case = _INPUTS / entry["compound_id"]
        topology_path = case / "topology.json"
        coordinates_path = case / "coordinates.json"
        assert (
            hashlib.sha256(topology_path.read_bytes()).hexdigest()
            == entry["topology_file_sha256"]
        )
        assert (
            hashlib.sha256(coordinates_path.read_bytes()).hexdigest()
            == entry["coordinate_file_sha256"]
        )
        topology = json.loads(topology_path.read_bytes())
        coordinates = json.loads(coordinates_path.read_bytes())
        native_path = _ORACLE / "records" / entry["compound_id"] / "record.json"
        assert (
            hashlib.sha256(native_path.read_bytes()).hexdigest()
            == oracle["record_file_sha256"][entry["compound_id"]]
        )
        native = json.loads(native_path.read_bytes())
        result = cavity_from_native_grid(
            torch.tensor(coordinates["positions_angstrom"], dtype=torch.float64),
            torch.tensor(topology["lj_rmin_angstrom"], dtype=torch.float64),
        )
        assert (
            abs(result.energy_kcal_mol.item() - native["components_kcal_mol"]["cavity"])
            < 1e-10
        )


def test_native_coordinate_translation_cavity_replay():
    if not _TRANSLATIONS.is_file():
        pytest.skip("optional native coordinate-translation oracle is not installed")
    data = _TRANSLATIONS.read_bytes()
    assert hashlib.sha256(data).hexdigest() == _TRANSLATION_FILE_SHA256
    oracle = json.loads(data)
    assert oracle["translation_count"] == len(oracle["rows"]) == 21
    assert oracle["label_values_read"] is oracle["new_qm"] is False
    rmin = torch.tensor(oracle["lj_rmin_angstrom"], dtype=torch.float64)
    for row in oracle["rows"]:
        result = cavity_from_native_grid(
            torch.tensor(row["positions_angstrom"], dtype=torch.float64), rmin
        )
        assert result.occupied_voxels == row["occupied_voxels"]
        assert result.energy_kcal_mol.item() == row["native_cavity_kcal_mol"]


def test_panel30_shifted_coordinate_cavity_replay():
    if not _SHIFT_SCREEN.is_file():
        pytest.skip("optional native panel30 coordinate screen is not installed")
    data = _SHIFT_SCREEN.read_bytes()
    assert hashlib.sha256(data).hexdigest() == _SHIFT_SCREEN_FILE_SHA256
    screen = json.loads(data)
    assert screen["case_count"] == len(screen["rows"]) == 60
    assert screen["label_values_read"] is screen["new_qm"] is False
    radii_by_id = {}
    for row in screen["rows"]:
        compound_id = row["compound_id"]
        if compound_id not in radii_by_id:
            topology_path = _INPUTS / compound_id / "topology.json"
            radii_by_id[compound_id] = torch.tensor(
                json.loads(topology_path.read_bytes())["lj_rmin_angstrom"],
                dtype=torch.float64,
            )
        result = cavity_from_native_grid(
            torch.tensor(row["positions_angstrom"], dtype=torch.float64),
            radii_by_id[compound_id],
        )
        assert result.occupied_voxels == row["occupied_voxels"]
        assert result.energy_kcal_mol.item() == row["native_cavity_kcal_mol"]


def test_panel30_cavity_cuda_cpu_parity_when_available():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    manifest_path = _INPUTS / "manifest.json"
    if not manifest_path.is_file():
        pytest.skip("optional frozen panel is not installed")
    manifest = json.loads(manifest_path.read_bytes())
    for entry in manifest["records"]:
        case = _INPUTS / entry["compound_id"]
        topology = json.loads((case / "topology.json").read_bytes())
        coordinates = json.loads((case / "coordinates.json").read_bytes())
        positions = torch.tensor(coordinates["positions_angstrom"], dtype=torch.float64)
        rmin = torch.tensor(topology["lj_rmin_angstrom"], dtype=torch.float64)
        cpu = cavity_from_native_grid(positions, rmin)
        cuda = cavity_from_native_grid(positions.cuda(), rmin.cuda())
        assert cuda.occupied_voxels == cpu.occupied_voxels
        assert cuda.energy_kcal_mol.item() == cpu.energy_kcal_mol.item()


def test_native_methyl_hexanoate_cavity_step_is_not_smoothed():
    if not _NATIVE_STEP_INPUT.is_file():
        pytest.skip("optional native cavity-step input is not installed")
    inputs = json.loads(_NATIVE_STEP_INPUT.read_bytes())
    positions = torch.tensor(inputs["positions_angstrom"], dtype=torch.float64)
    rmin = torch.tensor(inputs["rmin_angstrom"], dtype=torch.float64)
    energies = {}
    for displacement in (-0.005, -0.001, 0.0, 0.001):
        moved = positions.clone()
        moved[0, 0] += displacement
        energies[displacement] = cavity_from_native_grid(
            moved, rmin
        ).energy_kcal_mol.item()
    assert energies[-0.005] == pytest.approx(20.9437, abs=5e-5)
    for displacement in (-0.001, 0.0, 0.001):
        assert energies[displacement] == pytest.approx(20.9485, abs=5e-5)
    assert energies[-0.001] - energies[-0.005] > 0.004
