"""Focused FP32 coverage for the public CHA polar and PBSA cavity primitives."""

from __future__ import annotations

from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
_PRIVATE_ENGINE_INSTALLED = (
    importlib.util.find_spec("maple_cha_exact_private") is not None
)

DATA = Path(__file__).parent / "data"

from maple.function.calculator.extra_correction.implicit.torch_chagb import (
    cha_polar_from_inverse_born,
)
from maple.function.calculator.extra_correction.implicit.torch_pbsa_exact_cavity import (
    cavity_from_native_grid,
)


def _cha_inputs(dtype):
    return (
        torch.tensor(
            [[0.0, 0.0, 0.0], [1.2, 0.3, 0.0], [0.2, 1.1, 0.1]],
            dtype=dtype,
        ),
        torch.tensor([0.35, -0.55, 0.2], dtype=dtype),
        torch.tensor([1.7, 1.5, 1.2], dtype=dtype),
        torch.tensor([0.40, 0.50, 0.48], dtype=dtype),
    )


def _cavity_inputs(dtype):
    return (
        torch.tensor(
            [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=dtype,
        ),
        torch.tensor([1.7, 1.5, 1.2], dtype=dtype),
    )


@contextmanager
def _record_floating_dtypes():
    from torch.utils._python_dispatch import TorchDispatchMode
    from torch.utils._pytree import tree_flatten

    seen = []

    class Recorder(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            for value in tree_flatten((args, kwargs))[0]:
                if isinstance(value, torch.Tensor) and value.is_floating_point():
                    seen.append((str(func), value.dtype))
            result = func(*args, **kwargs)
            for value in tree_flatten(result)[0]:
                if isinstance(value, torch.Tensor) and value.is_floating_point():
                    seen.append((str(func), value.dtype))
            return result

    with Recorder():
        yield seen


def test_fp32_cha_fields_and_cavity_energy_remain_float32_end_to_end():
    with _record_floating_dtypes() as seen:
        polar = cha_polar_from_inverse_born(*_cha_inputs(torch.float32))
        cavity = cavity_from_native_grid(*_cavity_inputs(torch.float32))

    tensor_fields = (
        polar.polar_kcal_mol,
        polar.self_kcal_mol,
        polar.pair_kcal_mol,
        polar.electrostatic_size_angstrom,
        polar.inverse_born_shift_per_angstrom,
        polar.shifted_inverse_born_per_angstrom,
        polar.born_radii_angstrom,
        polar.effective_charges_e,
        polar.cha_factors,
        cavity.energy_kcal_mol,
    )
    assert all(value.dtype == torch.float32 for value in tensor_fields)
    assert seen
    assert {dtype for _, dtype in seen} == {torch.float32}
    assert cavity.supports_forces is False


def test_fp32_components_match_fp64_compatibility_branch_within_engineering_budget():
    fp32_polar = cha_polar_from_inverse_born(*_cha_inputs(torch.float32))
    fp64_polar = cha_polar_from_inverse_born(*_cha_inputs(torch.float64))
    fp32_cavity = cavity_from_native_grid(*_cavity_inputs(torch.float32))
    fp64_cavity = cavity_from_native_grid(*_cavity_inputs(torch.float64))

    for fp32_value, fp64_value in (
        (fp32_polar.self_kcal_mol, fp64_polar.self_kcal_mol),
        (fp32_polar.pair_kcal_mol, fp64_polar.pair_kcal_mol),
        (fp32_polar.polar_kcal_mol, fp64_polar.polar_kcal_mol),
        (fp32_cavity.energy_kcal_mol, fp64_cavity.energy_kcal_mol),
    ):
        assert abs(float(fp32_value) - float(fp64_value)) <= 1.0e-3
    assert fp32_cavity.occupied_voxels == fp64_cavity.occupied_voxels
    assert fp32_cavity.evaluated_grid_points == fp64_cavity.evaluated_grid_points


@pytest.mark.parametrize("case_index", [0, 1, 2, 3, 4, 6, 7, 8])
def test_fp32_polar_matches_portable_native_oracle_within_engineering_budget(
    case_index,
):
    artifact = json.loads((DATA / "chagb_polar_oracle_v1.json").read_bytes())
    case = artifact["cases"][case_index]
    inputs = case["inputs"]
    expected = case["expected"]
    result = cha_polar_from_inverse_born(
        torch.tensor(inputs["positions_angstrom"], dtype=torch.float32),
        torch.tensor(inputs["charges_e"], dtype=torch.float32),
        torch.tensor(inputs["effective_cha_radii_angstrom"], dtype=torch.float32),
        torch.tensor(
            inputs["unshifted_inverse_born_per_angstrom"], dtype=torch.float32
        ),
    )
    for name in ("self_kcal_mol", "pair_kcal_mol", "polar_kcal_mol"):
        assert abs(float(getattr(result, name)) - expected[name]) <= 1.0e-3


def test_fp32_polar_preserves_the_exact_size_switch_without_smoothing():
    artifact = json.loads((DATA / "chagb_polar_oracle_v1.json").read_bytes())
    inputs = artifact["cases"][5]["inputs"]
    result = cha_polar_from_inverse_born(
        torch.tensor(inputs["positions_angstrom"], dtype=torch.float32),
        torch.tensor(inputs["charges_e"], dtype=torch.float32),
        torch.tensor(inputs["effective_cha_radii_angstrom"], dtype=torch.float32),
        torch.tensor(
            inputs["unshifted_inverse_born_per_angstrom"], dtype=torch.float32
        ),
    )
    assert result.electrostatic_size_angstrom.item() >= 10.0
    assert result.inverse_born_shift_per_angstrom.item() > 0.0


@pytest.mark.parametrize("case_index", range(6))
def test_fp32_cavity_matches_portable_native_oracle_within_engineering_budget(
    case_index,
):
    artifact = json.loads((DATA / "pbsa_cavity_oracle_v1.json").read_bytes())
    case = artifact["cases"][case_index]
    result = cavity_from_native_grid(
        torch.tensor(case["positions_angstrom"], dtype=torch.float32),
        torch.tensor(case["lj_rmin_angstrom"], dtype=torch.float32),
    )
    assert abs(float(result.energy_kcal_mol) - case["native_cavity_kcal_mol"]) <= 1.0e-3


def test_fp32_components_are_independent_of_process_default_dtype():
    original = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        polar = cha_polar_from_inverse_born(*_cha_inputs(torch.float32))
        cavity = cavity_from_native_grid(*_cavity_inputs(torch.float32))
    finally:
        torch.set_default_dtype(original)

    assert polar.polar_kcal_mol.dtype == torch.float32
    assert cavity.energy_kcal_mol.dtype == torch.float32
    assert torch.get_default_dtype() == original


@pytest.mark.parametrize("component", range(4))
def test_cha_rejects_mixed_floating_dtypes(component):
    values = list(_cha_inputs(torch.float32))
    values[component] = values[component].double()
    with pytest.raises(TypeError, match="same explicit dtype"):
        cha_polar_from_inverse_born(*values)


def test_cavity_rejects_mixed_floating_dtypes():
    positions, radii = _cavity_inputs(torch.float32)
    with pytest.raises(TypeError, match="same-device, same-dtype"):
        cavity_from_native_grid(positions, radii.double())


@pytest.mark.skipif(
    not torch.cuda.is_available() or not _PRIVATE_ENGINE_INSTALLED,
    reason="CUDA and the separately installed private engine are required",
)
def test_cuda_fp32_polar_uses_outer_product_reduction_without_matmul():
    from maple_cha_exact_private.torch_runtime import ReadbackBudget

    device = torch.device("cuda:0")
    seen = []
    from torch.utils._python_dispatch import TorchDispatchMode

    class Recorder(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            seen.append(str(func))
            return func(*args, **(kwargs or {}))

    with Recorder():
        result = cha_polar_from_inverse_born(
            *tuple(value.to(device) for value in _cha_inputs(torch.float32)),
            readback_budget=ReadbackBudget(),
        )
    assert result.polar_kcal_mol.device == device
    assert not any("aten.mm" in name or "aten.matmul" in name for name in seen)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not _PRIVATE_ENGINE_INSTALLED,
    reason="CUDA and the separately installed private engine are required",
)
def test_cuda_cavity_accumulates_device_integer_count_with_one_final_count_read():
    from maple_cha_exact_private.torch_runtime import ReadbackBudget

    device = torch.device("cuda:0")
    positions, radii = (value.to(device) for value in _cavity_inputs(torch.float32))
    budget = ReadbackBudget()
    result = cavity_from_native_grid(
        positions,
        radii,
        coordinates_validated_f12_7=True,
        grid_chunk_size=17,
        readback_budget=budget,
    )
    assert result.energy_kcal_mol.device == device
    assert result.occupied_voxels > 0
    count_reads = [
        record for record in budget.records if record.purpose == "final-occupied-voxel-count"
    ]
    assert len(count_reads) == 1
    assert count_reads[0].scalar_count == 1


@pytest.mark.skipif(
    not torch.cuda.is_available() or not _PRIVATE_ENGINE_INSTALLED,
    reason="CUDA and the separately installed private engine are required",
)
def test_cuda_dispersion_is_chunk_invariant_and_uses_bounded_readbacks():
    from maple_cha_exact_private.torch_runtime import (
        ExecutionPolicy,
        ReadbackBudget,
    )
    from maple_cha_exact_private.dispersion import dispersion_energy

    device = torch.device("cuda:0")
    positions, rmin = (value.to(device) for value in _cavity_inputs(torch.float32))
    epsilon = torch.tensor([0.12, 0.10, 0.08], dtype=torch.float32, device=device)
    energies = []
    for atom_chunk, surface_chunk in ((3, 1200), (1, 37)):
        policy = ExecutionPolicy(
            reserve_bytes=1,
            max_workspace_bytes=64 * 1024 * 1024,
            min_workspace_bytes=1,
            atom_chunk=atom_chunk,
            surface_chunk=surface_chunk,
        )
        budget = ReadbackBudget()
        energies.append(
            dispersion_energy(
                positions,
                rmin,
                epsilon,
                execution_policy=policy,
                readback_budget=budget,
                workspace_bytes=policy.max_workspace_bytes,
            )
        )
        assert [
            (record.stage, record.purpose, record.dtype, record.scalar_count, record.bytes)
            for record in budget.records
        ] == [
            ("dispersion", "aggregate-input-validation", "torch.int32", 1, 4),
            ("dispersion", "sphere-point-count", "torch.int64", 1, 8),
            ("dispersion", "aggregate-output-validation", "torch.int32", 1, 4),
        ]
        assert budget.bytes_used == 16
        assert all(
            record.source_device == "cuda:0"
            and record.transfer_kind == "device-to-host"
            for record in budget.records
        )
    assert torch.allclose(energies[0], energies[1], atol=1.0e-5, rtol=0.0)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not _PRIVATE_ENGINE_INSTALLED,
    reason="CUDA and the separately installed private engine are required",
)
def test_cuda_r6_edge_chunking_is_device_resident_and_chunk_invariant():
    from maple_cha_exact_private.torch_runtime import (
        ExecutionPolicy,
        ReadbackBudget,
    )
    from maple_cha_exact_private.r6_surface_accumulator import (
        AxisBoundaries,
        accumulate_r6_from_native_boundaries,
    )

    device = torch.device("cuda:0")
    atoms = torch.zeros((1, 3), dtype=torch.float32, device=device)
    radii = torch.ones(1, dtype=torch.float32, device=device)
    arcs = torch.empty((0, 3), dtype=torch.float32, device=device)
    origin = torch.tensor([-1.0, -1.0, -1.0], device=device)
    owner = torch.ones(2, dtype=torch.int64, device=device)
    fraction = torch.zeros(2, dtype=torch.float32, device=device)
    axes = tuple(
        AxisBoundaries(
            torch.tensor(indices, dtype=torch.int64, device=device), owner, fraction
        )
        for indices in (
            [[0, 1, 1], [2, 1, 1]],
            [[1, 0, 1], [1, 2, 1]],
            [[1, 1, 0], [1, 1, 2]],
        )
    )
    values = []
    for edge_chunk in (1, 8):
        policy = ExecutionPolicy(
            reserve_bytes=1,
            max_workspace_bytes=1024 * 1024,
            min_workspace_bytes=1,
            edge_chunk=edge_chunk,
        )
        budget = ReadbackBudget()
        result = accumulate_r6_from_native_boundaries(
            atoms,
            radii,
            arcs,
            origin,
            1.0,
            *axes,
            sasopt=False,
            probe_radius_angstrom=0.88,
            execution_policy=policy,
            readback_budget=budget,
            workspace_bytes=policy.max_workspace_bytes,
        )
        values.append(result.inverse_cube_per_angstrom3)
        assert result.boundary_count == 6
        assert all(record.stage == "R6" for record in budget.records)
    assert torch.equal(values[0], values[1])
