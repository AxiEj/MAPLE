from __future__ import annotations

from dataclasses import asdict
import json

import numpy as np
import pytest
import torch
from ase.build import molecule
from ase.units import Hartree

import maple.solvation.nonpolar.legacy_smd_cds as legacy_cds_module
from maple.function.calculator.extra_correction.implicit.pyscf_smd_cds import (
    pyscf_smd_cds,
)
from maple.solvation.nonpolar.legacy_smd_cds import TorchLegacySMDCDS
from maple.solvation.surfaces.legacy_dareal import LegacyDAREALTopologyError


@pytest.mark.parametrize("molecule_name", ["H2O", "CH4", "CH3CH2OH", "CH3COCH3"])
@pytest.mark.parametrize("solvent", ["water", "ethanol", "hexane"])
def test_energy_and_gradient_match_pyscf_legacy(molecule_name, solvent):
    atoms = molecule(molecule_name)
    positions = atoms.positions.copy()
    # The idealized ASE acetone geometry has an exact four-sphere shared point;
    # use a deterministic nonsymmetric conformer for the regular parity case.
    if molecule_name == "CH3COCH3":
        positions += np.arange(positions.size).reshape(positions.shape) * 0.0017
    symbols = tuple(atoms.get_chemical_symbols())
    coordinates = torch.tensor(positions, dtype=torch.float64, requires_grad=True)
    functional = TorchLegacySMDCDS(symbols, solvent)
    state = functional.evaluate_torch(coordinates)
    gradient = torch.autograd.grad(state.energy_eV, coordinates, create_graph=True)[0]
    oracle = pyscf_smd_cds(symbols, positions, solvent=solvent)
    assert state.energy_eV.item() == pytest.approx(
        oracle.energy_hartree * Hartree, abs=1.0e-7
    )
    np.testing.assert_allclose(
        gradient.detach().cpu().numpy(),
        oracle.position_gradient_hartree_per_angstrom * Hartree,
        atol=1.0e-6,
        rtol=0.0,
    )
    assert state.provenance["upstream_sha256"] == (
        "f57b94c0eb6d5a29f1a1441294795321a5a39af74e0bec890628fac83027250b"
    )
    json.dumps(asdict(state.diagnostics), allow_nan=False)
    assert len(functional.configuration_sha256()) == 64
    assert len(functional.provenance_sha256) == 64


def test_hessian_matches_fixed_topology_gradient_difference():
    atoms = molecule("H2O")
    symbols = tuple(atoms.get_chemical_symbols())
    reference = torch.tensor(atoms.positions, dtype=torch.float64)
    functional = TorchLegacySMDCDS(symbols, "water")
    analytic = torch.autograd.functional.hessian(
        functional.energy_torch, reference
    ).reshape(9, 9)
    step = 2.0e-5
    columns = []
    for coordinate in range(reference.numel()):
        direction = torch.zeros_like(reference).reshape(-1)
        direction[coordinate] = step
        plus = (
            (reference.reshape(-1) + direction)
            .reshape_as(reference)
            .requires_grad_(True)
        )
        minus = (
            (reference.reshape(-1) - direction)
            .reshape_as(reference)
            .requires_grad_(True)
        )
        plus_gradient = torch.autograd.grad(functional.energy_torch(plus), plus)[0]
        minus_gradient = torch.autograd.grad(functional.energy_torch(minus), minus)[0]
        columns.append(((plus_gradient - minus_gradient) / (2.0 * step)).reshape(-1))
    audited = torch.stack(columns, dim=1)
    torch.testing.assert_close(analytic, audited, atol=1.0e-4, rtol=0.0)
    torch.testing.assert_close(analytic, analytic.T, atol=1.0e-10, rtol=0.0)


def test_exact_shared_point_fails_closed_instead_of_perturbing_radius():
    atoms = molecule("CH3COCH3")
    functional = TorchLegacySMDCDS(tuple(atoms.get_chemical_symbols()), "water")
    with pytest.raises(LegacyDAREALTopologyError, match="four-sphere shared-point"):
        functional.energy_torch(torch.tensor(atoms.positions, dtype=torch.float64))


def test_configuration_hash_binds_all_live_source_files(monkeypatch):
    symbols = tuple(molecule("H2O").get_chemical_symbols())
    baseline = TorchLegacySMDCDS(symbols, "water")
    assert {
        name for name, _ in json.loads(baseline.provenance["source_files_sha256"])
    } == {
        "maple.function.route2_solvents",
        "maple.solvation.nonpolar.legacy_smd_cds",
        "maple.solvation.surfaces.legacy_dareal",
    }
    baseline_hash = baseline.configuration_sha256()
    monkeypatch.setattr(
        legacy_cds_module,
        "source_files_sha256",
        lambda files: tuple((name, "0" * 64) for name in sorted(files)),
    )
    drifted = TorchLegacySMDCDS(symbols, "water")
    assert drifted.configuration_sha256() != baseline_hash
    assert drifted.provenance_sha256 != baseline.provenance_sha256


def test_scientific_coefficient_tables_are_immutable_and_hash_bound():
    symbols = tuple(molecule("H2O").get_chemical_symbols())
    functional = TorchLegacySMDCDS(symbols, "water")
    baseline_hash = functional.configuration_sha256()
    baseline_sigma = functional._sigma[6]
    assert functional._sigma is not legacy_cds_module._AQ_SIGMA

    with pytest.raises(TypeError):
        functional._sigma[6] = baseline_sigma + 1.0
    with pytest.raises(TypeError):
        legacy_cds_module._AQ_SIGMA[6] = baseline_sigma + 1.0

    assert functional._sigma[6] == baseline_sigma
    assert functional.configuration_sha256() == baseline_hash

    # Even deliberate private whole-table replacement cannot silently retain
    # the scientific identity used by the parent component guard.
    functional._sigma = {**functional._sigma, 6: baseline_sigma + 1.0}
    assert functional.configuration_sha256() != baseline_hash


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_matches_cpu():
    atoms = molecule("H2O")
    symbols = tuple(atoms.get_chemical_symbols())
    cpu_positions = torch.tensor(
        atoms.positions, dtype=torch.float64, requires_grad=True
    )
    cuda_positions = cpu_positions.detach().to("cuda").requires_grad_(True)
    cpu = TorchLegacySMDCDS(symbols, "ethanol", device="cpu").energy_torch(
        cpu_positions
    )
    cuda = TorchLegacySMDCDS(symbols, "ethanol", device="cuda").energy_torch(
        cuda_positions
    )
    cpu_gradient = torch.autograd.grad(cpu, cpu_positions)[0]
    cuda_gradient = torch.autograd.grad(cuda, cuda_positions)[0]
    torch.testing.assert_close(cpu, cuda.cpu(), atol=1.0e-10, rtol=0.0)
    torch.testing.assert_close(cpu_gradient, cuda_gradient.cpu(), atol=1.0e-9, rtol=0.0)
