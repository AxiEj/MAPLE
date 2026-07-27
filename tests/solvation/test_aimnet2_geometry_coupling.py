from __future__ import annotations

from types import SimpleNamespace

from ase import Atoms
from ase.units import Hartree
import numpy as np
import pytest

from maple.function.calculator.aimnet._aimnet2_calculator import (
    AIMNet2ChargePositionResponse,
    AIMNet2ChargeState,
)
from maple.function.calculator.extra_correction.implicit import (
    aimnet2_geometry_coupling as coupling,
)


_ALPHA = 0.07
_BETA = 0.03
_GAMMA = 0.02


class _FakeAIMNet2:
    @staticmethod
    def _charges(atoms):
        x = np.asarray(atoms.get_positions(), dtype=float)[:, 0]
        return np.asarray([-0.6, 0.3, 0.3]) + _ALPHA * (
            x - float(np.mean(x))
        )

    def charge_state(self, atoms):
        charges = self._charges(atoms)
        energy = 0.5 * float(np.vdot(atoms.positions, atoms.positions))
        return AIMNet2ChargeState(
            energy_ev=energy,
            raw_charges_e=charges,
            charges_e=charges,
            requested_total_charge_e=0.0,
            raw_charge_residual_e=float(np.sum(charges)),
            charge_projection_per_atom_e=0.0,
            model_name="fake-aimnet2",
        )

    def charge_position_response(self, atoms, charge_cotangent_ev_per_e):
        cotangent = np.asarray(charge_cotangent_ev_per_e, dtype=float)
        charge_vjp = np.zeros_like(atoms.positions)
        charge_vjp[:, 0] = _ALPHA * (
            cotangent - float(np.mean(cotangent))
        )
        return AIMNet2ChargePositionResponse(
            charge_state=self.charge_state(atoms),
            charge_cotangent_ev_per_e=cotangent,
            intrinsic_energy_gradient_ev_per_angstrom=atoms.positions,
            charge_position_vjp_ev_per_angstrom=charge_vjp,
        )


class _FakeReactionField:
    def __init__(self, positions_angstrom, _radii_angstrom, **kwargs):
        self.positions = np.asarray(positions_angstrom, dtype=float).copy()
        self.scale = 1.0 + _BETA * float(
            np.vdot(self.positions, self.positions)
        )
        self.runtime_provenance = {
            "backend": "fake-ddpcm",
            "lmax": kwargs["lmax"],
            "n_lebedev": kwargs["n_lebedev"],
        }

    def apply_scf(self, density):
        field = np.zeros_like(density)
        field[:, 0] = self.scale * density[:, 0]
        return field

    def scf_polarization_energy_hartree(self, density):
        return (
            0.5
            * self.scale
            * float(np.vdot(density[:, 0], density[:, 0]))
            / Hartree
        )

    def polarization_position_gradient_ev_per_angstrom(self, density):
        q2 = float(np.vdot(density[:, 0], density[:, 0]))
        return _BETA * q2 * self.positions


def _fake_cds(_symbols, positions_angstrom, *, solvent):
    assert solvent == "water"
    positions = np.asarray(positions_angstrom, dtype=float)
    energy_ev = _GAMMA * float(np.vdot(positions, positions))
    return SimpleNamespace(
        energy_hartree=energy_ev / Hartree,
        position_gradient_hartree_per_angstrom=(
            2.0 * _GAMMA * positions / Hartree
        ),
    )


def _water():
    return Atoms(
        "OHH",
        positions=[
            [0.0, 0.0, 0.1],
            [0.95, 0.0, -0.2],
            [-0.25, 0.91, 0.0],
        ],
        info={"charge": 0, "mult": 1},
    )


def _objective(monkeypatch):
    monkeypatch.setattr(
        coupling,
        "PyDDXPCMReactionFieldLinearMap",
        _FakeReactionField,
    )
    monkeypatch.setattr(coupling, "pyscf_smd_cds", _fake_cds)
    return coupling.AIMNet2GeometryCoupledObjective(
        _water(),
        _FakeAIMNet2(),
        solvent="water",
        lmax=11,
        n_lebedev=590,
    )


def test_geometry_coupled_gradient_matches_complete_scalar_finite_difference(
    monkeypatch,
):
    objective = _objective(monkeypatch)
    atoms = _water()
    state = objective.evaluate(atoms)
    step = 1.0e-5
    numerical = np.zeros_like(atoms.positions)
    for atom_index in range(len(atoms)):
        for axis in range(3):
            plus = atoms.copy()
            minus = atoms.copy()
            plus.positions[atom_index, axis] += step
            minus.positions[atom_index, axis] -= step
            numerical[atom_index, axis] = (
                objective.evaluate(plus).solution_energy_ev
                - objective.evaluate(minus).solution_energy_ev
            ) / (2.0 * step)

    np.testing.assert_allclose(
        state.total_gradient_ev_per_angstrom,
        numerical,
        rtol=2.0e-7,
        atol=2.0e-8,
    )
    assert state.continuum_provenance["lmax"] == 11
    assert state.continuum_provenance["n_lebedev"] == 590
    assert state.positions_angstrom.flags.writeable is False
    assert state.total_gradient_ev_per_angstrom.flags.writeable is False


def test_geometry_coupled_ase_bridge_returns_same_energy_and_force(monkeypatch):
    objective = _objective(monkeypatch)
    atoms = _water()
    direct = objective.evaluate(atoms)
    atoms.calc = coupling.AIMNet2GeometryCoupledASECalculator(objective)

    assert atoms.get_potential_energy() == direct.solution_energy_ev
    np.testing.assert_allclose(
        atoms.get_forces(),
        -direct.total_gradient_ev_per_angstrom,
    )
    assert atoms.calc.last_state is not None


@pytest.mark.parametrize(
    ("keyword", "value"),
    (
        ("lmax", 1.5),
        ("n_lebedev", True),
        ("n_proc", 0),
    ),
)
def test_geometry_coupled_objective_rejects_nonpositive_or_noninteger_grids(
    keyword,
    value,
):
    with pytest.raises(ValueError, match=keyword):
        coupling.AIMNet2GeometryCoupledObjective(
            _water(),
            _FakeAIMNet2(),
            solvent="water",
            **{keyword: value},
        )


def test_geometry_relaxed_solvation_ledger_uses_declared_gas_reference(
    monkeypatch,
):
    objective = _objective(monkeypatch)
    state = objective.evaluate(_water())
    reference = state.solute_energy_ev - 0.25

    expected = (
        (
            0.25
            + state.polarization_energy_hartree * Hartree
            + state.cds_energy_hartree * Hartree
        )
        / Hartree
        * coupling.HARTREE_TO_KCAL_MOL
    )
    assert state.solvation_energy_kcal_mol(
        gas_reference_energy_ev=reference
    ) == expected
