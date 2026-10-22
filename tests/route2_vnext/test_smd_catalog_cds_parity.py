"""Published-parameter/CDS parity across all named SMD solvents, not experiment."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from ase.units import Hartree

from maple.function.route2_solvents import SMD_SOLVENTS
from maple.function.calculator.extra_correction.implicit.pyscf_smd_cds import (
    pyscf_smd_cds,
)
from maple.solvation.nonpolar.legacy_smd_cds import TorchLegacySMDCDS


@pytest.mark.parametrize("solvent", sorted(SMD_SOLVENTS))
def test_full_smd_catalog_cds_energy_and_analytic_gradient_match_upstream(solvent):
    pytest.importorskip("pyscf")
    positions = np.array([[0.0, 0.0, 0.0], [0.9572, 0.0, 0.0], [-0.239, 0.9266, 0.0]])
    symbols = ("O", "H", "H")
    coordinates = torch.tensor(positions, dtype=torch.float64, requires_grad=True)
    energy = TorchLegacySMDCDS(symbols, solvent, dtype=torch.float64).energy_torch(
        coordinates
    )
    gradient = torch.autograd.grad(energy, coordinates)[0]
    reference = pyscf_smd_cds(symbols, positions, solvent=solvent)
    # Same tolerances as the existing legacy CDS parity regression; no fitting.
    assert float(energy.detach()) == pytest.approx(
        reference.energy_hartree * Hartree, abs=1e-7
    )
    np.testing.assert_allclose(
        gradient.detach().numpy(),
        reference.position_gradient_hartree_per_angstrom * Hartree,
        atol=1e-6,
        rtol=0.0,
    )
