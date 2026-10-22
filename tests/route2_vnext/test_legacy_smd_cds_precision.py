from __future__ import annotations

import pytest
import torch
from ase.build import molecule

from maple.solvation.nonpolar.legacy_smd_cds import TorchLegacySMDCDS
from maple.solvation.surfaces.legacy_dareal import (
    LegacyDAREALConfig,
    LegacyDAREALTopologyError,
    legacy_dareal_areas_torch,
)


def _hco_geometry(dtype: torch.dtype, *, requires_grad: bool = False) -> torch.Tensor:
    atoms = molecule("CH3COCH3")
    positions = torch.tensor(atoms.positions, dtype=dtype)
    # The idealized ASE geometry has an exact four-sphere shared point.  Use the
    # same deterministic nonsymmetric displacement as the FP64 oracle tests.
    positions = positions + torch.arange(
        positions.numel(), dtype=dtype
    ).reshape_as(positions) * positions.new_tensor(0.0017)
    return positions.requires_grad_(requires_grad)


def test_fp32_hco_forward_gradient_and_hvp_preserve_dtype():
    atoms = molecule("CH3COCH3")
    positions = _hco_geometry(torch.float32, requires_grad=True)
    functional = TorchLegacySMDCDS(
        tuple(atoms.get_chemical_symbols()),
        "water",
        dtype=torch.float32,
    )

    state = functional.evaluate_torch(positions)
    gradient = torch.autograd.grad(state.energy_eV, positions, create_graph=True)[0]
    direction = torch.linspace(
        -0.5, 0.5, positions.numel(), dtype=torch.float32
    ).reshape_as(positions)
    hvp = torch.autograd.grad((gradient * direction).sum(), positions)[0]

    assert functional.dtype is torch.float32
    assert state.provenance["dtype"] == "torch.float32"
    assert state.diagnostics.dareal.guard_version == (
        "pyscf-2.13.1-dareal-fixed-topology-v1"
    )
    assert state.diagnostics.dareal.floating_dtype == "torch.float32"
    assert state.diagnostics.dareal.requested_relative_topology_tolerance is None
    assert state.diagnostics.dareal.effective_relative_topology_tolerance == (
        100.0 * torch.finfo(torch.float32).eps
    )
    assert state.diagnostics.dareal.tolerance_policy == (
        "max(legacy-epsi,100*machine-epsilon)"
    )
    for value in (
        state.energy_eV,
        state.energy_hartree,
        state.energy_kcal_mol,
        state.atom_areas_angstrom2,
        state.atom_tensions_cal_mol_angstrom2,
        gradient,
        hvp,
    ):
        assert value.dtype is torch.float32
        assert bool(torch.isfinite(value).all())


def test_fp32_tracks_fp64_on_same_regular_hco_geometry():
    atoms = molecule("CH3COCH3")
    symbols = tuple(atoms.get_chemical_symbols())
    position64 = _hco_geometry(torch.float64, requires_grad=True)
    position32 = position64.detach().to(torch.float32).requires_grad_(True)
    functional64 = TorchLegacySMDCDS(symbols, "ethanol")
    functional32 = TorchLegacySMDCDS(symbols, "ethanol", dtype=torch.float32)

    energy64 = functional64.energy_torch(position64)
    energy32 = functional32.energy_torch(position32)
    gradient64 = torch.autograd.grad(energy64, position64, create_graph=True)[0]
    gradient32 = torch.autograd.grad(energy32, position32, create_graph=True)[0]
    direction64 = torch.linspace(
        -0.5, 0.5, position64.numel(), dtype=torch.float64
    ).reshape_as(position64)
    direction32 = direction64.to(torch.float32)
    hvp64 = torch.autograd.grad((gradient64 * direction64).sum(), position64)[0]
    hvp32 = torch.autograd.grad((gradient32 * direction32).sum(), position32)[0]

    torch.testing.assert_close(energy32.double(), energy64, atol=2.0e-5, rtol=2.0e-5)
    torch.testing.assert_close(
        gradient32.double(), gradient64, atol=2.0e-4, rtol=2.0e-4
    )
    torch.testing.assert_close(hvp32.double(), hvp64, atol=2.0e-3, rtol=2.0e-3)
    assert functional32.configuration_sha256() != functional64.configuration_sha256()
    assert functional32.provenance_sha256 != functional64.provenance_sha256


def test_dareal_accepts_both_supported_dtypes_without_implicit_upcast():
    for dtype in (torch.float32, torch.float64):
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=dtype
        )
        radii = torch.ones(2, dtype=dtype)
        result = legacy_dareal_areas_torch(positions, radii)
        assert result.areas_angstrom2.dtype is dtype
        assert result.solid_angles.dtype is dtype
        assert result.provenance["dtype"] == str(dtype)
        expected_tolerance = (
            1.0e-11 if dtype is torch.float64 else 100.0 * torch.finfo(dtype).eps
        )
        assert result.diagnostics.effective_relative_topology_tolerance == (
            expected_tolerance
        )


@pytest.mark.parametrize("molecule_name", ["H2O", "CH4", "CH3OH", "CH3CH2OH"])
def test_fp32_regular_small_molecules_have_finite_energy_and_gradient(molecule_name):
    atoms = molecule(molecule_name)
    positions = torch.tensor(
        atoms.positions, dtype=torch.float32, requires_grad=True
    )
    functional = TorchLegacySMDCDS(
        tuple(atoms.get_chemical_symbols()), "water", dtype=torch.float32
    )
    energy = functional.energy_torch(positions)
    gradient = torch.autograd.grad(energy, positions)[0]
    assert energy.dtype is torch.float32
    assert gradient.dtype is torch.float32
    assert bool(torch.isfinite(energy))
    assert bool(torch.isfinite(gradient).all())


def test_explicit_topology_tolerance_is_honored_for_fp32():
    config = LegacyDAREALConfig(relative_topology_tolerance=2.5e-4)
    result = legacy_dareal_areas_torch(
        torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float32),
        torch.ones(2, dtype=torch.float32),
        config=config,
    )
    assert result.diagnostics.requested_relative_topology_tolerance == 2.5e-4
    assert result.diagnostics.effective_relative_topology_tolerance == 2.5e-4
    assert result.diagnostics.tolerance_policy == "explicit"


def test_fp32_exact_shared_point_still_fails_closed_without_radius_jitter():
    atoms = molecule("CH3COCH3")
    functional = TorchLegacySMDCDS(
        tuple(atoms.get_chemical_symbols()), "water", dtype=torch.float32
    )
    with pytest.raises(LegacyDAREALTopologyError):
        functional.energy_torch(torch.tensor(atoms.positions, dtype=torch.float32))


def test_dtype_contract_rejects_mismatch_and_unsupported_precision():
    symbols = tuple(molecule("H2O").get_chemical_symbols())
    functional = TorchLegacySMDCDS(symbols, "water", dtype=torch.float32)
    with pytest.raises(TypeError, match="match the configured dtype"):
        functional.energy_torch(
            torch.tensor(molecule("H2O").positions, dtype=torch.float64)
        )
    with pytest.raises(TypeError, match="supports only"):
        TorchLegacySMDCDS(symbols, "water", dtype=torch.float16)
    with pytest.raises(TypeError, match="share one floating dtype"):
        legacy_dareal_areas_torch(
            torch.zeros((1, 3), dtype=torch.float32),
            torch.ones(1, dtype=torch.float64),
        )
