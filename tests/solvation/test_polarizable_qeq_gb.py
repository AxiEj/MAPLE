from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from maple.function.read.filereader.mol2_reader import MOL2Reader


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("openmm") is None,
    reason="OpenMM optional dependency is not installed",
)


def test_corrected_qeq_fixed_charges_run_through_openmm_gb(water_mol2, tmp_path):
    from maple.function.calculator.extra_correction.implicit.correction import (
        ImplicitSolvationCorrection,
    )

    atoms = MOL2Reader(str(water_mol2), charge=0, mult=1)
    correction = ImplicitSolvationCorrection(
        atoms,
        {"source": "maple", "method": "qeq-gto", "mode": "fixed", "geometry": "keep"},
        {"method": "gb", "model": "obc2", "nonpolar": "ace", "experimental": True},
        output=tmp_path / "job.out",
    )
    result = correction.evaluate(atoms, need_forces=True)
    assert np.isfinite(result.energy_hartree)
    assert np.isfinite(result.forces_hartree_per_angstrom).all()
    assert correction.charge_result.provenance["profile"] == "rappe-goddard-gto-full-h-scf"
    assert correction.charge_result.provenance["hydrogen_screening_exponent_update"] is True
    assert correction.charge_result.provenance["scientific_status"] == "experimental"
    assert correction.charge_result.provenance["accuracy_certified"] is False
    assert correction.charge_result.provenance["default_eligible"] is False
    assert correction.charge_result.provenance["selection_policy"] == "explicit-only-no-fallback"


def test_variational_cqeq_gb_force_matches_minimized_energy_finite_difference(
    water_mol2, tmp_path
):
    from maple.function.calculator.extra_correction.implicit.correction import (
        ImplicitSolvationCorrection,
    )

    atoms = MOL2Reader(str(water_mol2), charge=0, mult=1)
    correction = ImplicitSolvationCorrection(
        atoms,
        {
            "source": "maple",
            "method": "qeq-gto",
            "mode": "polarizable",
            "geometry": "keep",
        },
        {"method": "gb", "model": "obc2", "nonpolar": "ace", "experimental": True},
        output=tmp_path / "job.out",
    )
    result = correction.evaluate(atoms, need_forces=True)
    h = 1.0e-4
    plus = atoms.copy()
    minus = atoms.copy()
    plus.positions[1, 0] += h
    minus.positions[1, 0] -= h
    finite_difference = -(
        correction.evaluate(plus).energy_hartree
        - correction.evaluate(minus).energy_hartree
    ) / (2.0 * h)
    assert np.isclose(
        result.forces_hartree_per_angstrom[1, 0], finite_difference, atol=2.0e-6
    )
    assert "qeq_polarization" in result.components_hartree
    assert result.provenance["charge_method"] == "cqeq-gto"
    assert result.provenance["scientific_status"] == "experimental"
    assert result.provenance["accuracy_certified"] is False
    assert result.provenance["default_eligible"] is False
    assert result.provenance["selection_policy"] == "explicit-only-no-fallback"
    assert result.provenance["solvent_solver"]["kkt_residual_ev"] < 2.0e-6
