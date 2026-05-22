"""WS1 — shared periodic-cell geometry admission for every ensemble.

``validate_pbc_cell_geometry`` runs on the shared MD path (via
``validate_md_capabilities``) for NVE/NVT/NPT, not only NPT.  It requires a
full rank-3 cell for *any* periodic system because MAPLE reconstructs unwrapped
coordinates through the full 3x3 cell matrix; a rank-deficient cell would
silently drop a coordinate.  Two regression guards live here:

* no false-accept of a degenerate periodic cell (rank < 3 / zero volume), and
* no false-reject of a real slab — its finite vacuum lattice vector keeps the
  full cell rank 3, so ``rank == 3`` (not ``rank == sum(pbc)``) accepts it.
"""

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from maple.function.calculator._ase_unit_contract import MAPLE_ENERGY_UNIT, MAPLE_FORCE_UNIT
from maple.function.dispatcher.md.ensemble.nve import NVE
from maple.function.dispatcher.md.ensemble.nvt import NVT


class _PBCCalc(Calculator):
    """Minimal PBC-capable calculator with a small declared neighbor cutoff."""

    implemented_properties = ["energy", "forces"]

    def __init__(self):
        super().__init__()
        self.maple_model_name = "fake-pbc"
        self.maple_pbc_md_supported = True
        self.maple_stress_supported = False
        self.maple_energy_unit = MAPLE_ENERGY_UNIT
        self.maple_force_unit = MAPLE_FORCE_UNIT
        self.maple_neighbor_cutoff = 2.0  # < min-image radius of the test cells

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results["energy"] = 0.0
        self.results["forces"] = np.zeros((len(atoms), 3))


def _atoms(cell, pbc):
    atoms = Atoms("Ar2", positions=[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], cell=cell, pbc=pbc)
    atoms.calc = _PBCCalc()
    return atoms


def test_nve_rejects_rank2_partial_cell(tmp_path):
    # pbc on x,y but a zero z lattice vector -> full cell rank 2.  Even with the
    # experimental partial-PBC opt-in this must be rejected: the unwrap
    # reconstruction would silently drop the z coordinate.
    atoms = _atoms([[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 0.0]], [True, True, False])
    with pytest.raises(ValueError, match="rank-3"):
        NVE(
            output=str(tmp_path / "x.out"),
            atoms=atoms,
            paras={"steps": 0, "verbose": 0, "remove_com_every": 0, "allow_partial_pbc": True},
        )


def test_nve_rejects_linearly_dependent_partial_cell(tmp_path):
    # pbc on x,y; the third vector is a nonzero in-plane DUPLICATE of the first.
    # ASE's length-based Cell.rank counts this as 3 (all vectors nonzero), but the
    # matrix rank is 2, so the full-cell unwrap inverse is singular. It must be
    # rejected at admission with a clear error, not a downstream LinAlgError.
    atoms = _atoms([[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [5.0, 0.0, 0.0]], [True, True, False])
    with pytest.raises(ValueError, match="rank-3"):
        NVE(
            output=str(tmp_path / "x.out"),
            atoms=atoms,
            paras={"steps": 0, "verbose": 0, "remove_com_every": 0, "allow_partial_pbc": True},
        )


def test_nvt_rejects_degenerate_full_cell(tmp_path):
    # All-periodic but coplanar lattice vectors -> zero volume / rank 2.
    atoms = _atoms([[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [5.0, 0.0, 0.0]], [True, True, True])
    with pytest.raises(ValueError):
        NVT(
            output=str(tmp_path / "x.out"),
            atoms=atoms,
            paras={"steps": 0, "verbose": 0, "remove_com_every": 0},
        )


def test_nve_accepts_valid_slab(tmp_path):
    # Real slab: finite vacuum z vector -> full cell rank 3, active 2-D lattice
    # valid.  Accepted under the explicit partial-PBC opt-in.
    atoms = _atoms([[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 20.0]], [True, True, False])
    sim = NVE(
        output=str(tmp_path / "slab.out"),
        atoms=atoms,
        paras={"steps": 0, "verbose": 0, "remove_com_every": 0, "allow_partial_pbc": True},
    )
    assert sim is not None
