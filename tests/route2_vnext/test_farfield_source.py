from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from ase.units import Bohr

from maple.solvation.release.farfield_source import (
    FarFieldSourceThresholds,
    compare_far_field_source_to_reference,
)
from maple.solvation.release.source_mep import (
    SourceElectrostaticObservables,
    source_electrostatic_observables,
)


def _thresholds() -> FarFieldSourceThresholds:
    return FarFieldSourceThresholds(
        total_charge_absolute_tolerance_e=1.0e-8,
        dipole_absolute_tolerance_e_angstrom=0.05,
        dipole_relative_tolerance=0.1,
        quadrupole_absolute_tolerance_e_angstrom2=0.25,
        quadrupole_relative_tolerance=0.2,
        far_field_absolute_rms_tolerance_hartree_per_e=2.0e-5,
        far_field_relative_rms_tolerance=0.15,
    )


def _observables(
    *,
    quadrupole_scale: float = 1.0,
    mep: tuple[float, ...] = (1.0e-3, -1.0e-3, 3.0e-4, -3.0e-4),
) -> SourceElectrostaticObservables:
    quadrupole = np.diag([1.0, -0.4, -0.6]) * quadrupole_scale
    return SourceElectrostaticObservables(
        total_charge_e=0.0,
        molecular_dipole_e_angstrom=(0.8, -0.2, 0.1),
        traceless_quadrupole_e_angstrom2=tuple(
            tuple(float(value) for value in row) for row in quadrupole
        ),
        sampled_mep_hartree_per_e=mep,
        multipole_origin_angstrom=(0.2, -0.3, 0.1),
    )


def test_source_multipoles_are_translation_covariant_with_explicit_origin():
    positions = np.asarray([[0.0, 0.0, 0.0], [1.2, -0.3, 0.4]])
    source = np.asarray([[0.4, 0.1, -0.2, 0.3], [-0.4, -0.2, 0.05, 0.2]])
    origin = np.asarray([0.3, -0.1, 0.2])
    points_bohr = np.asarray([[12.0, -4.0, 3.0], [-9.0, 7.0, 5.0]])
    baseline = source_electrostatic_observables(
        positions_angstrom=positions,
        source4=source,
        evaluation_points_bohr=points_bohr,
        multipole_origin_angstrom=origin,
    )

    translation = np.asarray([2.1, -1.3, 0.7])
    translated = source_electrostatic_observables(
        positions_angstrom=positions + translation,
        source4=source,
        evaluation_points_bohr=points_bohr + translation / Bohr,
        multipole_origin_angstrom=origin + translation,
    )
    assert translated.total_charge_e == pytest.approx(baseline.total_charge_e)
    assert translated.molecular_dipole_e_angstrom == pytest.approx(
        baseline.molecular_dipole_e_angstrom, abs=2.0e-15
    )
    assert np.asarray(translated.traceless_quadrupole_e_angstrom2) == pytest.approx(
        np.asarray(baseline.traceless_quadrupole_e_angstrom2), abs=3.0e-15
    )
    assert translated.sampled_mep_hartree_per_e == pytest.approx(
        baseline.sampled_mep_hartree_per_e, abs=2.0e-15
    )


def test_far_field_gate_requires_every_multipole_and_shell():
    reference = _observables()
    exact = compare_far_field_source_to_reference(
        reference,
        reference,
        shell_radii_angstrom=(10.0, 14.0),
        shell_point_counts=(2, 2),
        angular_weights=(1.0, 1.0, 1.0, 1.0),
        thresholds=_thresholds(),
    )
    assert exact.case_passed is True
    assert exact.as_dict()["capability_admitted"] is False

    bad_quadrupole = compare_far_field_source_to_reference(
        _observables(quadrupole_scale=2.0),
        reference,
        shell_radii_angstrom=(10.0, 14.0),
        shell_point_counts=(2, 2),
        angular_weights=(1.0, 1.0, 1.0, 1.0),
        thresholds=_thresholds(),
    )
    assert bad_quadrupole.quadrupole_gate_passed is False
    assert bad_quadrupole.case_passed is False

    bad_shell = compare_far_field_source_to_reference(
        _observables(mep=(1.0e-3, -1.0e-3, 8.0e-4, -8.0e-4)),
        reference,
        shell_radii_angstrom=(10.0, 14.0),
        shell_point_counts=(2, 2),
        angular_weights=(1.0, 1.0, 1.0, 1.0),
        thresholds=_thresholds(),
    )
    assert bad_shell.shell_comparisons[0].passed is True
    assert bad_shell.shell_comparisons[1].passed is False
    assert bad_shell.far_field_mep_gate_passed is False
    assert bad_shell.case_passed is False


def test_far_field_gate_rejects_origin_and_shell_mismatch():
    reference = _observables()
    wrong_origin = SourceElectrostaticObservables(
        total_charge_e=reference.total_charge_e,
        molecular_dipole_e_angstrom=reference.molecular_dipole_e_angstrom,
        traceless_quadrupole_e_angstrom2=(reference.traceless_quadrupole_e_angstrom2),
        sampled_mep_hartree_per_e=reference.sampled_mep_hartree_per_e,
        multipole_origin_angstrom=(0.0, 0.0, 0.0),
    )
    with pytest.raises(ValueError, match="one origin"):
        compare_far_field_source_to_reference(
            wrong_origin,
            reference,
            shell_radii_angstrom=(10.0, 14.0),
            shell_point_counts=(2, 2),
            angular_weights=(1.0, 1.0, 1.0, 1.0),
            thresholds=_thresholds(),
        )
    with pytest.raises(ValueError, match="inconsistent"):
        compare_far_field_source_to_reference(
            reference,
            reference,
            shell_radii_angstrom=(10.0,),
            shell_point_counts=(3,),
            angular_weights=(1.0, 1.0, 1.0, 1.0),
            thresholds=_thresholds(),
        )


def test_source_observables_copy_mutable_inputs_and_gate_flags_cannot_be_forged():
    dipole = [0.8, -0.2, 0.1]
    quadrupole = [[1.0, 0.0, 0.0], [0.0, -0.4, 0.0], [0.0, 0.0, -0.6]]
    mep = [1.0e-3, -1.0e-3, 3.0e-4, -3.0e-4]
    origin = [0.2, -0.3, 0.1]
    reference = SourceElectrostaticObservables(
        total_charge_e=0.0,
        molecular_dipole_e_angstrom=dipole,
        traceless_quadrupole_e_angstrom2=quadrupole,
        sampled_mep_hartree_per_e=mep,
        multipole_origin_angstrom=origin,
    )
    dipole[0] = 99.0
    quadrupole[0][0] = 99.0
    mep[0] = 99.0
    origin[0] = 99.0
    assert reference.molecular_dipole_e_angstrom[0] == 0.8
    assert reference.traceless_quadrupole_e_angstrom2[0][0] == 1.0
    assert reference.sampled_mep_hartree_per_e[0] == 1.0e-3
    assert reference.multipole_origin_angstrom[0] == 0.2

    exact = compare_far_field_source_to_reference(
        reference,
        reference,
        shell_radii_angstrom=(10.0, 14.0),
        shell_point_counts=(2, 2),
        angular_weights=(1.0, 1.0, 1.0, 1.0),
        thresholds=_thresholds(),
    )
    with pytest.raises(ValueError, match="charge gate"):
        replace(exact, total_charge_allowed_error_e=0.0, charge_gate_passed=False)
