from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

from maple.function.read.filereader.mol2_reader import MOL2Reader

pytest.importorskip("openmm", reason="OpenMM optional dependency is not installed")
torch = pytest.importorskip(
    "torch", reason="Torch optional dependency is not installed"
)

DATA_DIR = Path(__file__).parent / "data" / "amber_gb_reference" / "audit"
HESSIAN_ASYMMETRY_HARTREE_PER_ANGSTROM2 = 1.0e-10
HVP_MAX_HARTREE_PER_ANGSTROM2 = 1.0e-9
TRANSLATION_LEAKAGE_HARTREE_PER_ANGSTROM2 = 1.0e-8
FD_RMS_HARTREE_PER_ANGSTROM2 = 5.0e-6
FD_MAX_HARTREE_PER_ANGSTROM2 = 2.0e-5


def _atoms(case: str, water_mol2):
    path = (
        water_mol2
        if case == "water"
        else DATA_DIR / "methyl-hexanoate" / "normalized.mol2"
    )
    return MOL2Reader(str(path), charge=0, mult=1)


def _backends(atoms, nonpolar: str = "ace"):
    from maple.function.calculator.extra_correction.implicit.obc2_parameters import (
        build_obc2_parameters,
    )
    from maple.function.calculator.extra_correction.implicit.openmm_gb import OpenMMGB
    from maple.function.calculator.extra_correction.implicit.torch_obc2 import (
        TorchOBC2,
    )

    reference = OpenMMGB(
        atoms,
        atoms.get_initial_charges(),
        model="obc2",
        nonpolar=nonpolar,
        platform="Reference",
    )
    parameters = build_obc2_parameters(
        reference.charges,
        reference.radius_result,
        nonpolar=nonpolar,
    )
    candidate = TorchOBC2(parameters, device="cpu", dtype=torch.float64)
    return reference, candidate


def _openmm_force_fd_hessian(reference, atoms, step_angstrom: float) -> np.ndarray:
    size = 3 * len(atoms)
    hessian = np.empty((size, size), dtype=np.float64)
    for coordinate in range(size):
        atom, axis = divmod(coordinate, 3)
        plus = atoms.copy()
        minus = atoms.copy()
        plus.positions[atom, axis] += step_angstrom
        minus.positions[atom, axis] -= step_angstrom
        plus_force = reference.evaluate(
            plus, need_forces=True
        ).forces_hartree_per_angstrom.reshape(-1)
        minus_force = reference.evaluate(
            minus, need_forces=True
        ).forces_hartree_per_angstrom.reshape(-1)
        hessian[:, coordinate] = -(plus_force - minus_force) / (2.0 * step_angstrom)
    return hessian


def _synthetic_two_atom_backend(distance_angstrom: float):
    from maple.function.calculator.extra_correction.implicit.obc2_parameters import (
        build_obc2_parameters,
    )
    from maple.function.calculator.extra_correction.implicit.radii import RadiusResult
    from maple.function.calculator.extra_correction.implicit.torch_obc2 import TorchOBC2

    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [distance_angstrom, 0.0, 0.0]])
    provider_parameters = np.array([[0.15, 0.8], [0.15, 0.8]])
    radii = RadiusResult(
        profile="synthetic-obc2-test",
        radii_angstrom=provider_parameters[:, 0] * 10.0,
        provider_parameters=provider_parameters,
        provenance={"testing_only": True},
    )
    parameters = build_obc2_parameters(np.array([0.2, -0.2]), radii, nonpolar="ace")
    return atoms, TorchOBC2(parameters, device="cpu", dtype=torch.float64)


@pytest.mark.parametrize("case", ["water", "methyl-hexanoate"])
def test_hessian_is_symmetric(water_mol2, case):
    atoms = _atoms(case, water_mol2)
    _reference, candidate = _backends(atoms)

    hessian = candidate.hessian(atoms)

    assert hessian.shape == (3 * len(atoms), 3 * len(atoms))
    assert np.isfinite(hessian).all()
    assert np.max(np.abs(hessian - hessian.T)) <= (
        HESSIAN_ASYMMETRY_HARTREE_PER_ANGSTROM2
    )


@pytest.mark.parametrize("case", ["water", "methyl-hexanoate"])
def test_direct_hvp_matches_explicit_hessian_product(water_mol2, case):
    atoms = _atoms(case, water_mol2)
    _reference, candidate = _backends(atoms)
    rng = np.random.default_rng(20260920)
    direction = rng.normal(size=3 * len(atoms))
    direction /= np.linalg.norm(direction)

    expected = candidate.hessian(atoms) @ direction
    actual = candidate.hvp(atoms, direction)

    np.testing.assert_allclose(
        actual,
        expected,
        rtol=1.0e-9,
        atol=HVP_MAX_HARTREE_PER_ANGSTROM2,
    )


@pytest.mark.parametrize("case", ["water", "methyl-hexanoate"])
def test_directional_result_energy_force_and_hvp_share_one_scalar(water_mol2, case):
    atoms = _atoms(case, water_mol2)
    _reference, candidate = _backends(atoms)
    generator = np.random.default_rng(20260922)
    direction = generator.normal(size=3 * len(atoms))
    direction /= np.linalg.norm(direction)

    directional = candidate.directional_derivatives(atoms, direction)
    ordinary = candidate.evaluate(atoms, need_forces=True)

    assert directional.energy_hartree == pytest.approx(
        ordinary.energy_hartree, abs=1.0e-14
    )
    np.testing.assert_allclose(
        directional.forces_hartree_per_angstrom,
        ordinary.forces_hartree_per_angstrom,
        rtol=0.0,
        atol=1.0e-13,
    )
    np.testing.assert_allclose(
        directional.hvp_hartree_per_angstrom2,
        candidate.hessian(atoms) @ direction,
        rtol=1.0e-9,
        atol=HVP_MAX_HARTREE_PER_ANGSTROM2,
    )


def test_hvp_rejects_wrong_vector_shape(water_mol2):
    atoms = _atoms("water", water_mol2)
    _reference, candidate = _backends(atoms)

    with pytest.raises(ValueError, match="shape.*9|9.*shape"):
        candidate.hvp(atoms, np.zeros((3, 3), dtype=np.float64))


@pytest.mark.parametrize("case", ["water", "methyl-hexanoate"])
def test_hessian_has_no_translational_curvature(water_mol2, case):
    atoms = _atoms(case, water_mol2)
    _reference, candidate = _backends(atoms)
    hessian = candidate.hessian(atoms)

    for axis in range(3):
        translation = np.zeros(3 * len(atoms), dtype=np.float64)
        translation[axis::3] = 1.0
        leakage = hessian @ translation
        scale = max(
            TRANSLATION_LEAKAGE_HARTREE_PER_ANGSTROM2,
            1.0e-8 * np.linalg.norm(hessian),
        )
        assert np.linalg.norm(leakage) <= scale


def test_torch_hessian_matches_openmm_force_fd_multistep_plateau(water_mol2):
    atoms = _atoms("water", water_mol2)
    reference, candidate = _backends(atoms)
    analytic = candidate.hessian(atoms)
    errors = []

    for step in (0.001, 0.002, 0.005):
        finite_difference = _openmm_force_fd_hessian(reference, atoms, step)
        difference = analytic - finite_difference
        errors.append(
            (
                float(np.sqrt(np.mean(difference**2))),
                float(np.max(np.abs(difference))),
            )
        )

    # A centered force difference has O(h^2) truncation error, so a healthy
    # analytic Hessian should approach it monotonically as h decreases rather
    # than give three nearly identical errors.  Keep the absolute admission
    # limits on every step and require the expected convergence direction.
    assert all(rms <= FD_RMS_HARTREE_PER_ANGSTROM2 for rms, _ in errors)
    assert all(maximum <= FD_MAX_HARTREE_PER_ANGSTROM2 for _, maximum in errors)
    assert errors[0][0] < errors[1][0] < errors[2][0]


def test_kink_margin_constant_is_frozen():
    from maple.function.calculator.extra_correction.implicit.torch_obc2 import (
        KINK_MARGIN_NM,
    )

    assert KINK_MARGIN_NM == 1.0e-8


@pytest.mark.parametrize("method", ["hessian", "hvp"])
def test_second_derivatives_fail_closed_at_collision(water_mol2, method):
    atoms = _atoms("water", water_mol2)
    atoms.positions[1] = atoms.positions[0]
    _reference, candidate = _backends(atoms)

    with pytest.raises(ValueError, match="collision|branch|margin"):
        if method == "hessian":
            candidate.hessian(atoms)
        else:
            candidate.hvp(atoms, np.ones(3 * len(atoms), dtype=np.float64))


@pytest.mark.parametrize("surface_distance_angstrom", [0.282, 2.538])
def test_second_derivatives_fail_closed_at_effective_switching_surfaces(
    surface_distance_angstrom,
):
    atoms, candidate = _synthetic_two_atom_backend(surface_distance_angstrom)

    for displacement in (-5.0e-7, 5.0e-7):
        side = atoms.copy()
        side.positions[1, 0] += displacement
        result = candidate.evaluate(side, need_forces=True)
        assert np.isfinite(result.energy_hartree)
        assert np.isfinite(result.forces_hartree_per_angstrom).all()

    with pytest.raises(ValueError, match="branch|kink|margin"):
        candidate.hessian(atoms)


def test_inactive_absolute_value_kink_does_not_create_false_rejection():
    # r == sr gives abs(r-sr)==0, but max(or, abs(...)) still selects the
    # strictly positive offset radius.  That inactive abs branch is not an
    # effective energy switching surface and must not block second derivatives.
    atoms, candidate = _synthetic_two_atom_backend(1.128)

    hessian = candidate.hessian(atoms)

    assert np.isfinite(hessian).all()


def test_energy_evaluation_fails_closed_for_nonfinite_positions(water_mol2):
    atoms = _atoms("water", water_mol2)
    _reference, candidate = _backends(atoms)
    atoms.positions[0, 0] = np.nan

    with pytest.raises(ValueError, match="finite"):
        candidate.evaluate(atoms, need_forces=False)
