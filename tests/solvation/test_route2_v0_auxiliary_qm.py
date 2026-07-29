from __future__ import annotations

import numpy as np
import pytest
from ase.units import Bohr, Hartree

from maple.function.calculator.extra_correction.implicit.route2_v0_auxiliary_qm import (
    AuxiliaryQMProvenance,
    AuxiliaryQMStationaryState,
    Route2V0AuxiliaryQMElectrostaticDifference,
    V0_AUXILIARY_QM_ELECTROSTATIC_CONSTRUCTION,
)


def _provenance(*, geometry_sha256: str = "geometry-a") -> AuxiliaryQMProvenance:
    return AuxiliaryQMProvenance(
        implementation="PySCF 2.13.1",
        electronic_structure="omegaB97M-V",
        basis="def2-TZVPD",
        reference="RKS",
        charge=0,
        spin=0,
        geometry_sha256=geometry_sha256,
        numerical_settings_sha256="numerics-a",
        atom_count=2,
    )


def _state(
    phase: str,
    *,
    energy_hartree: float,
    gradient_hartree_per_bohr: np.ndarray,
    provenance: AuxiliaryQMProvenance | None = None,
) -> AuxiliaryQMStationaryState:
    return AuxiliaryQMStationaryState(
        phase=phase,
        provenance=_provenance() if provenance is None else provenance,
        total_stationary_energy_hartree=energy_hartree,
        nuclear_gradient_hartree_per_bohr=gradient_hartree_per_bohr,
        orbital_gradient_inf=1.0e-9,
        scf_converged=True,
    )


def test_auxiliary_qm_electrostatic_difference_is_the_exact_gas_subtraction_ledger():
    gas_gradient = np.array([[0.2, -0.3, 0.1], [-0.2, 0.3, -0.1]])
    solvated_gradient = np.array([[0.1, -0.2, 0.3], [-0.1, 0.2, -0.3]])
    mace_gradient = np.array([[1.2, 0.3, -0.4], [-1.2, -0.3, 0.4]])
    state = Route2V0AuxiliaryQMElectrostaticDifference(
        mace_gas_energy_ev=-4.0,
        mace_gas_gradient_ev_per_angstrom=mace_gradient,
        auxiliary_gas_state=_state(
            "gas",
            energy_hartree=-10.0,
            gradient_hartree_per_bohr=gas_gradient,
        ),
        auxiliary_solvated_state=_state(
            "solvated",
            energy_hartree=-10.25,
            gradient_hartree_per_bohr=solvated_gradient,
        ),
    )

    expected_auxiliary_gradient = solvated_gradient - gas_gradient
    assert state.construction == V0_AUXILIARY_QM_ELECTROSTATIC_CONSTRUCTION
    assert state.auxiliary_energy_difference_hartree == pytest.approx(-0.25)
    np.testing.assert_allclose(
        state.auxiliary_gradient_difference_hartree_per_bohr,
        expected_auxiliary_gradient,
    )
    assert state.composite_energy_ev == pytest.approx(-4.0 - 0.25 * Hartree)
    np.testing.assert_allclose(
        state.composite_gradient_ev_per_angstrom,
        mace_gradient + expected_auxiliary_gradient * Hartree / Bohr,
    )
    np.testing.assert_allclose(
        state.composite_forces_ev_per_angstrom,
        -state.composite_gradient_ev_per_angstrom,
    )
    assert not state.composite_gradient_ev_per_angstrom.flags.writeable
    assert not state.composite_forces_ev_per_angstrom.flags.writeable


def test_auxiliary_qm_electrostatic_difference_rejects_cross_method_or_geometry_subtraction():
    gas = _state(
        "gas",
        energy_hartree=-10.0,
        gradient_hartree_per_bohr=np.zeros((2, 3)),
    )
    solvated = _state(
        "solvated",
        energy_hartree=-10.1,
        gradient_hartree_per_bohr=np.zeros((2, 3)),
        provenance=_provenance(geometry_sha256="geometry-b"),
    )

    with pytest.raises(ValueError, match="identical method, geometry"):
        Route2V0AuxiliaryQMElectrostaticDifference(
            mace_gas_energy_ev=0.0,
            mace_gas_gradient_ev_per_angstrom=np.zeros((2, 3)),
            auxiliary_gas_state=gas,
            auxiliary_solvated_state=solvated,
        )


def test_auxiliary_qm_electrostatic_difference_rejects_nonstationary_or_wrong_phase_inputs():
    provenance = _provenance()
    with pytest.raises(ValueError, match="SCF-converged"):
        AuxiliaryQMStationaryState(
            phase="gas",
            provenance=provenance,
            total_stationary_energy_hartree=-10.0,
            nuclear_gradient_hartree_per_bohr=np.zeros((2, 3)),
            orbital_gradient_inf=1.0e-9,
            scf_converged=False,
        )

    gas = _state(
        "gas",
        energy_hartree=-10.0,
        gradient_hartree_per_bohr=np.zeros((2, 3)),
        provenance=provenance,
    )
    wrong_phase = _state(
        "gas",
        energy_hartree=-10.1,
        gradient_hartree_per_bohr=np.zeros((2, 3)),
        provenance=provenance,
    )
    with pytest.raises(ValueError, match="one gas and one solvated"):
        Route2V0AuxiliaryQMElectrostaticDifference(
            mace_gas_energy_ev=0.0,
            mace_gas_gradient_ev_per_angstrom=np.zeros((2, 3)),
            auxiliary_gas_state=gas,
            auxiliary_solvated_state=wrong_phase,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("phase", "unknown", "either 'gas' or 'solvated'"),
        ("orbital_gradient_inf", -1.0, "is negative"),
    ],
)
def test_auxiliary_qm_state_rejects_invalid_stationarity_metadata(
    field: str,
    value: str | float,
    message: str,
):
    kwargs: dict[str, object] = {
        "phase": "gas",
        "provenance": _provenance(),
        "total_stationary_energy_hartree": -10.0,
        "nuclear_gradient_hartree_per_bohr": np.zeros((2, 3)),
        "orbital_gradient_inf": 1.0e-9,
        "scf_converged": True,
    }
    kwargs[field] = value

    with pytest.raises(ValueError, match=message):
        AuxiliaryQMStationaryState(**kwargs)
