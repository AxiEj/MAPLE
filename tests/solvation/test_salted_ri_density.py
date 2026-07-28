from __future__ import annotations

import numpy as np
import pytest

from maple.function.calculator.extra_correction.implicit.salted_ri_density import (
    project_ri_coefficients_to_electron_count,
    pyscf_coefficients_to_salted_order,
    salted_coefficients_to_pyscf_order,
    salted_ri_surface_mep,
)


def test_salted_l1_order_round_trip_is_exact():
    coefficients = np.arange(18, dtype=float) + 0.25
    l1_slices = (slice(2, 5), slice(10, 13))

    salted = pyscf_coefficients_to_salted_order(
        coefficients,
        l1_slices=l1_slices,
    )
    restored = salted_coefficients_to_pyscf_order(
        salted,
        l1_slices=l1_slices,
    )

    np.testing.assert_array_equal(restored, coefficients)
    np.testing.assert_array_equal(salted[2:5], coefficients[[3, 4, 2]])
    np.testing.assert_array_equal(salted[10:13], coefficients[[11, 12, 10]])


def test_coulomb_metric_projection_is_minimum_norm_and_charge_conserving():
    coefficients = np.asarray([0.8, -0.2, 0.4])
    integrated_basis = np.asarray([1.0, 0.5, -0.25])
    coulomb_metric = np.asarray(
        [
            [2.0, 0.1, 0.0],
            [0.1, 1.5, 0.2],
            [0.0, 0.2, 1.2],
        ]
    )
    target = 1.25

    projection = project_ri_coefficients_to_electron_count(
        coefficients,
        integrated_basis=integrated_basis,
        coulomb_metric=coulomb_metric,
        target_electron_count_e=target,
    )

    assert projection.projected_electron_count_e == pytest.approx(
        target,
        abs=2.0e-15,
    )
    correction = projection.coefficients - coefficients
    lagrange_direction = np.linalg.solve(coulomb_metric, integrated_basis)
    np.testing.assert_allclose(
        correction / correction[0],
        lagrange_direction / lagrange_direction[0],
        rtol=2.0e-15,
        atol=2.0e-15,
    )
    assert projection.coefficients.flags.writeable is False


def test_salted_ri_surface_mep_matches_direct_pyscf_ri_contraction():
    pyscf = pytest.importorskip("pyscf")
    from pyscf import df, gto, scf
    from pyscf.gto import ft_ao

    molecule = gto.M(
        atom="H 0 0 -0.37; H 0 0 0.37",
        basis="def2-svp",
        unit="Angstrom",
        charge=0,
        spin=0,
        verbose=0,
    )
    mean_field = scf.RHF(molecule).run()
    density_matrix = mean_field.make_rdm1()
    auxiliary = df.addons.make_auxmol(molecule, "def2-svp-jkfit")
    eri3c = df.incore.aux_e2(molecule, auxiliary)
    rhs = np.einsum("ijp,ij->p", eri3c, density_matrix)
    coefficients_pyscf = np.linalg.solve(
        auxiliary.intor("int2c2e_sph"),
        rhs,
    )

    # Discover the same l=1 blocks used by the production adapter, then
    # simulate one SALTED prediction vector.
    l1_slices = []
    ao_locations = auxiliary.ao_loc_nr()
    for shell in range(auxiliary.nbas):
        if auxiliary.bas_angular(shell) != 1:
            continue
        start = int(ao_locations[shell])
        for contraction in range(auxiliary.bas_nctr(shell)):
            offset = start + 3 * contraction
            l1_slices.append(slice(offset, offset + 3))
    coefficients_salted = pyscf_coefficients_to_salted_order(
        coefficients_pyscf,
        l1_slices=tuple(l1_slices),
    )

    points_bohr = np.asarray(
        [[0.0, 0.0, 4.0], [2.3, -1.7, 0.5], [-2.1, 1.1, -0.8]]
    )
    result = salted_ri_surface_mep(
        symbols=("H", "H"),
        atom_positions_angstrom=np.asarray(
            [[0.0, 0.0, -0.37], [0.0, 0.0, 0.37]]
        ),
        surface_points_bohr=points_bohr,
        salted_coefficients=coefficients_salted,
        auxiliary_basis="def2-svp-jkfit",
        source_model="unit-test",
        declared_total_charge_e=0.0,
        charge_tolerance_e=2.0e-2,
    )

    fake_points = gto.fakemol_for_charges(points_bohr)
    kernel = gto.mole.intor_cross(
        auxiliary._add_suffix("int2c2e"),
        auxiliary,
        fake_points,
    )
    expected_electronic = coefficients_pyscf @ kernel
    coordinates_bohr = molecule.atom_coords(unit="Bohr")
    expected_nuclear = sum(
        charge / np.linalg.norm(points_bohr - center, axis=1)
        for charge, center in zip(
            molecule.atom_charges(),
            coordinates_bohr,
            strict=True,
        )
    )
    expected_electron_count = float(
        np.real(
            np.asarray(ft_ao.ft_ao(auxiliary, np.zeros((1, 3))))[0]
            @ coefficients_pyscf
        )
    )

    np.testing.assert_allclose(
        result.surface_potential_hartree_per_e,
        expected_nuclear - expected_electronic,
        rtol=2.0e-13,
        atol=2.0e-13,
    )
    assert result.electron_count_e == pytest.approx(
        expected_electron_count,
        abs=2.0e-11,
    )
    assert result.source.surface_points_bohr.flags.writeable is False
    assert result.source.source_model == "unit-test"
    assert result.pyscf_version == pyscf.__version__


def test_salted_ri_surface_mep_can_apply_explicit_charge_projection():
    pytest.importorskip("pyscf")
    from pyscf import df, gto, scf

    molecule = gto.M(
        atom="H 0 0 -0.37; H 0 0 0.37",
        basis="def2-svp",
        unit="Angstrom",
        charge=0,
        spin=0,
        verbose=0,
    )
    mean_field = scf.RHF(molecule).run()
    density_matrix = mean_field.make_rdm1()
    auxiliary = df.addons.make_auxmol(molecule, "def2-svp-jkfit")
    coefficients_pyscf = np.linalg.solve(
        auxiliary.intor("int2c2e_sph"),
        np.einsum(
            "ijp,ij->p",
            df.incore.aux_e2(molecule, auxiliary),
            density_matrix,
        ),
    )
    l1_slices = []
    ao_locations = auxiliary.ao_loc_nr()
    for shell in range(auxiliary.nbas):
        if auxiliary.bas_angular(shell) != 1:
            continue
        start = int(ao_locations[shell])
        for contraction in range(auxiliary.bas_nctr(shell)):
            offset = start + 3 * contraction
            l1_slices.append(slice(offset, offset + 3))
    coefficients_salted = pyscf_coefficients_to_salted_order(
        coefficients_pyscf,
        l1_slices=tuple(l1_slices),
    )

    result = salted_ri_surface_mep(
        symbols=("H", "H"),
        atom_positions_angstrom=np.asarray(
            [[0.0, 0.0, -0.37], [0.0, 0.0, 0.37]]
        ),
        surface_points_bohr=np.asarray(
            [[0.0, 0.0, 4.0], [2.3, -1.7, 0.5]]
        ),
        salted_coefficients=coefficients_salted,
        auxiliary_basis="def2-svp-jkfit",
        source_model="unit-test-ri-fit",
        declared_total_charge_e=0.0,
        charge_constraint="coulomb-metric",
    )

    assert abs(result.raw_electron_count_e - 2.0) > 1.0e-4
    assert result.electron_count_e == pytest.approx(2.0, abs=2.0e-12)
    assert result.observed_total_charge_e == pytest.approx(0.0, abs=2.0e-12)
    assert result.charge_constraint == "coulomb-metric"
    assert result.charge_correction_coulomb_norm > 0.0
    assert result.max_abs_coefficient_correction > 0.0
