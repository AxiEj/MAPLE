"""Charged V4 admission keeps all non-charge domain and source checks."""

import numpy as np
import pytest
from ase import Atoms

from maple.function.calculator.extra_correction.implicit.route2_domain import (
    validate_route2_domain,
)


def ammonium(charge=1, mult=1):
    return Atoms(
        "NH4",
        positions=[
            [0, 0, 0],
            [0.6, 0.6, 0.6],
            [-0.6, -0.6, 0.6],
            [-0.6, 0.6, -0.6],
            [0.6, -0.6, -0.6],
        ],
        info={"charge": charge, "mult": mult},
    )


@pytest.mark.parametrize(
    "atoms",
    [
        ammonium(),
        Atoms(
            "OH", positions=[[0, 0, 0], [0.97, 0, 0]], info={"charge": -1, "mult": 1}
        ),
    ],
)
def test_explicit_charged_route_accepts_closed_shell_ions(atoms):
    validate_route2_domain(atoms, allow_charged=True)


def test_legacy_domain_keeps_neutral_restriction():
    with pytest.raises(ValueError, match="neutral"):
        validate_route2_domain(ammonium())


def test_ionic_parity_uses_electron_count_minus_charge():
    with pytest.raises(ValueError, match="parity"):
        validate_route2_domain(ammonium(charge=0), allow_charged=True)


@pytest.mark.parametrize("charge", [True, 0.5, np.nan, np.inf])
def test_invalid_charge_metadata_still_rejected(charge):
    with pytest.raises((TypeError, ValueError), match="integer|numeric|finite"):
        validate_route2_domain(ammonium(charge=charge), allow_charged=True)


def test_non_singlet_is_not_opened_by_charged_route():
    with pytest.raises(ValueError, match="closed-shell|multiplicity"):
        validate_route2_domain(ammonium(charge=0, mult=2), allow_charged=True)


def test_charged_tripos_markers_are_not_a_neutrality_backdoor():
    atoms = ammonium()
    atoms.info["mol2"] = {"atom_types": ["n.4", "h", "h", "h", "h"]}
    validate_route2_domain(atoms, allow_charged=True)
    atoms.pbc = True
    with pytest.raises(ValueError, match="non-periodic"):
        validate_route2_domain(atoms, allow_charged=True)


def test_disconnected_ions_are_still_rejected():
    atoms = ammonium()
    atoms.positions[-1] += 20
    with pytest.raises(ValueError, match="connected"):
        validate_route2_domain(atoms, allow_charged=True)


def test_ase_results_cache_is_invalidated_by_charge_only_change():
    from maple.function.calculator.route2._mace_polar_torch_calculator import (
        PureMACEPolarTorchCalculator,
    )
    from maple.function.route2_smd_profiles import route2_smd_profile_spec

    spec = route2_smd_profile_spec(
        "pure-macepolar-frozen-point-l1-ddpcm-smd-torch-cpu-v4"
    )

    class ChargePES:
        scalar_contract_id = spec.scalar_contract_id
        device = "cpu"
        calls = 0

        def configuration_sha256(self):
            return "f" * 64

        def get_potential_energy(self, atoms):
            self.calls += 1
            return float(atoms.info["charge"])

    atoms = ammonium()
    pes = ChargePES()
    atoms.calc = PureMACEPolarTorchCalculator._from_test_pes(
        atoms=atoms,
        solvent="water",
        pes=pes,
        profile_spec=spec,
    )
    assert atoms.get_potential_energy() == 1.0
    assert atoms.get_potential_energy() == 1.0
    assert pes.calls == 1
    # The geometry is unchanged; both odd charges retain even electron parity.
    atoms.info["charge"] = -1
    assert atoms.get_potential_energy() == -1.0
    assert pes.calls == 2


def test_v4_source_charge_matches_requested_charge_not_zero():
    torch = pytest.importorskip("torch")
    from maple.solvation.experimental.mace_polar_torch import MACEPolarTorchPES

    class Model:
        device = "cpu"
        dtype = torch.float32
        charge_error = 0.0

        def configuration_sha256(self):
            return "a" * 64

        def energy_source_torch(self, atoms, positions):
            x, y, z = positions.unbind(-1)
            q = x * 0 + (atoms.info["charge"] + self.charge_error) / len(atoms)
            source = torch.stack((q, y**2, z**2, x**2), dim=-1)
            return 0.07 * positions.square().sum(), source

    class Continuum:
        def configuration_sha256(self):
            return "b" * 64

        def energy_torch(self, positions, source):
            return 0.2 * source.square().sum() + 0.1 * positions.square().sum()

    class CDS:
        def configuration_sha256(self):
            return "c" * 64

        def energy_torch(self, positions):
            return 0.05 * positions.pow(3).sum()

    model = Model()
    pes = MACEPolarTorchPES._for_testing(
        symbols=("N", "H", "H", "H", "H"),
        model=model,
        continuum=Continuum(),
        solvent_term=CDS(),
        requested_profile="pure-macepolar-frozen-point-l1-ddpcm-smd-torch-cpu-v4",
    )
    atoms = ammonium()
    assert np.isfinite(pes.get_potential_energy(atoms))
    assert np.isfinite(pes.get_forces(atoms)).all()
    model.charge_error = 0.1
    with pytest.raises(RuntimeError, match="charge"):
        pes.get_potential_energy(atoms)
