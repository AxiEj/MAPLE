from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.constraints import FixAtoms

from maple.function.dispatcher.ts.algorithm.PRFO import (
    PRFO,
    _mass_weighted_optimization_basis,
    prfo_step,
)
from maple.function.read.command_control import CommandControl


def test_parser_forwards_explicit_rigid_projection_to_prfo(tmp_path):
    options = CommandControl.from_settings(
        [
            "#model=ani2x(hessian=numerical)",
            "#ts(method=prfo,project_rigid_modes=true,max_iter=1)",
        ]
    ).as_dict()
    optimizer = PRFO(str(tmp_path / "parsed.out"), _nonlinear_water(), paras=options)
    assert options["project_rigid_modes"] is True
    assert optimizer.params.project_rigid_modes is True


def _nonlinear_water() -> Atoms:
    return Atoms(
        "OH2",
        positions=[
            [0.000000, 0.000000, 0.000000],
            [0.957200, 0.000000, 0.000000],
            [-0.239987, 0.927297, 0.000000],
        ],
    )


def _mass_weighted_rigid_columns(atoms: Atoms) -> np.ndarray:
    masses = np.asarray(atoms.get_masses(), dtype=np.float64)
    positions = np.asarray(atoms.get_positions(), dtype=np.float64)
    center = np.sum(masses[:, None] * positions, axis=0) / np.sum(masses)
    columns = np.zeros((3 * len(atoms), 6), dtype=np.float64)
    for index, (mass, position) in enumerate(zip(masses, positions)):
        root_mass = np.sqrt(mass)
        relative = position - center
        block = slice(3 * index, 3 * index + 3)
        columns[block, :3] = root_mass * np.eye(3)
        columns[block, 3:] = root_mass * np.asarray(
            [
                [0.0, -relative[2], relative[1]],
                [relative[2], 0.0, -relative[0]],
                [-relative[1], relative[0], 0.0],
            ]
        )
    return columns


def test_internal_basis_removes_six_nonlinear_rigid_modes():
    atoms = _nonlinear_water()

    basis, rigid_rank = _mass_weighted_optimization_basis(
        atoms,
        project_rigid_modes=True,
    )

    rigid = _mass_weighted_rigid_columns(atoms)
    assert rigid_rank == 6
    assert basis.shape == (9, 3)
    np.testing.assert_allclose(basis.T @ basis, np.eye(3), atol=2e-15)
    np.testing.assert_allclose(basis.T @ rigid, 0.0, atol=2e-14)


def test_internal_basis_preserves_four_vibrations_for_linear_triatomic():
    atoms = Atoms(
        "OCO",
        positions=[[-1.16, 0.0, 0.0], [0.0, 0.0, 0.0], [1.16, 0.0, 0.0]],
    )

    basis, rigid_rank = _mass_weighted_optimization_basis(
        atoms,
        project_rigid_modes=True,
    )

    assert rigid_rank == 5
    assert basis.shape == (9, 4)
    np.testing.assert_allclose(basis.T @ basis, np.eye(4), atol=2e-15)


def test_projection_is_explicit_and_fails_closed_outside_free_molecules():
    periodic = _nonlinear_water()
    periodic.set_cell([10.0, 10.0, 10.0])
    periodic.set_pbc(True)
    constrained = _nonlinear_water()
    constrained.set_constraint(FixAtoms(indices=[0]))

    for atoms in (periodic, constrained):
        basis, rigid_rank = _mass_weighted_optimization_basis(
            atoms,
            project_rigid_modes=False,
        )
        np.testing.assert_array_equal(basis, np.eye(9))
        assert rigid_rank == 0
        with pytest.raises(ValueError, match="isolated unconstrained"):
            _mass_weighted_optimization_basis(atoms, project_rigid_modes=True)


@pytest.mark.parametrize("invalid", ["mass", "position"])
def test_projection_rejects_nonfinite_geometry_data(invalid):
    atoms = _nonlinear_water()
    if invalid == "mass":
        masses = atoms.get_masses()
        masses[1] = np.nan
        atoms.arrays["masses"] = masses
        message = "finite positive atomic masses"
    else:
        positions = atoms.get_positions()
        positions[1, 2] = np.inf
        atoms.set_positions(positions)
        message = "finite positions"

    with pytest.raises(ValueError, match=message):
        _mass_weighted_optimization_basis(atoms, project_rigid_modes=True)


def test_reduced_prfo_step_ignores_rigid_hessian_and_gradient_contamination():
    atoms = _nonlinear_water()
    internal, rigid_rank = _mass_weighted_optimization_basis(
        atoms,
        project_rigid_modes=True,
    )
    rigid_raw = _mass_weighted_rigid_columns(atoms)
    rigid, _, _ = np.linalg.svd(rigid_raw, full_matrices=False)
    rigid = rigid[:, :rigid_rank]

    h_internal = np.diag([-2.0, 3.0, 5.0])
    g_internal = np.array([0.08, -0.03, 0.02])
    h_mw = internal @ h_internal @ internal.T
    h_mw += rigid @ np.diag([-100.0, 1e-9, 2e-9, 3e-9, 4e-9, 5e-9]) @ rigid.T
    g_mw = internal @ g_internal + 0.5 * rigid[:, 0]

    h_reduced = internal.T @ h_mw @ internal
    g_reduced = internal.T @ g_mw
    step_reduced = prfo_step(
        h_reduced,
        g_reduced,
        is_ts=True,
        trust_radius=0.1,
    )
    step_mw = internal @ step_reduced

    np.testing.assert_allclose(rigid.T @ step_mw, 0.0, atol=2e-15)
    assert np.linalg.norm(step_mw) <= 0.1 * (1.0 + 1e-12)
    # The physical negative mode, rather than the contaminated rigid mode, is followed.
    assert abs(float(internal[:, 0] @ step_mw)) > 1e-4


def test_mode_tracking_compares_ambient_modes_when_internal_basis_rotates(tmp_path):
    atoms = _nonlinear_water()
    first_basis, _ = _mass_weighted_optimization_basis(
        atoms,
        project_rigid_modes=True,
    )
    optimizer = PRFO(str(tmp_path / "track.out"), atoms)
    eigenvalues = np.array([-2.0, 3.0, 5.0])
    first_vectors = np.eye(3)
    first_gradient = np.array([0.2, 0.1, 0.05])

    first = optimizer.update_mode_tracking(
        eigenvalues,
        first_vectors,
        first_gradient,
        mode_basis_mw=first_basis,
    )
    assert first == 0

    angle = 0.41
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    second_basis = first_basis @ rotation
    # Present the same three ambient physical modes in a different order.
    ambient_modes = first_basis[:, [1, 2, 0]]
    second_vectors = second_basis.T @ ambient_modes

    second = optimizer.update_mode_tracking(
        np.array([3.0, 5.0, -2.0]),
        second_vectors,
        first_gradient,
        mode_basis_mw=second_basis,
    )
    assert second == 2
    np.testing.assert_allclose(
        abs(float(optimizer.tracked_mode_vec_mw @ first_basis[:, 0])),
        1.0,
        atol=2e-15,
    )


class _ContaminatedQuadratic(Calculator):
    def __init__(
        self,
        reference: np.ndarray,
        hessian_cartesian: np.ndarray,
        gradient_cartesian: np.ndarray,
    ):
        super().__init__()
        self.implemented_properties = ["energy", "free_energy", "forces"]
        self.reference = np.asarray(reference, dtype=np.float64).reshape(-1)
        self.hessian_cartesian = np.asarray(hessian_cartesian, dtype=np.float64)
        self.gradient_cartesian = np.asarray(gradient_cartesian, dtype=np.float64)

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        assert atoms is not None
        super().calculate(atoms, properties, system_changes)
        displacement = np.asarray(atoms.get_positions(), dtype=np.float64).reshape(-1)
        displacement -= self.reference
        gradient = self.gradient_cartesian + self.hessian_cartesian @ displacement
        energy = float(
            self.gradient_cartesian @ displacement
            + 0.5 * displacement @ self.hessian_cartesian @ displacement
        )
        self.results = {
            "energy": energy,
            "free_energy": energy,
            "forces": -gradient.reshape(-1, 3),
        }

    def get_hessian(self, atoms):
        return self.hessian_cartesian.copy()


def _run_contaminated_quadratic_prfo(tmp_path, *, project_rigid_modes: bool):
    atoms = _nonlinear_water()
    reference = atoms.get_positions().copy()
    internal, rigid_rank = _mass_weighted_optimization_basis(
        atoms,
        project_rigid_modes=True,
    )
    rigid_raw = _mass_weighted_rigid_columns(atoms)
    rigid, _, _ = np.linalg.svd(rigid_raw, full_matrices=False)
    rigid = rigid[:, :rigid_rank]
    h_mw = internal @ np.diag([-2.0, 3.0, 5.0]) @ internal.T
    h_mw += rigid @ np.diag([-100.0, 1.0, 1.5, 2.0, 2.5, 3.0]) @ rigid.T
    g_mw = 0.04 * internal[:, 0] + 0.25 * rigid[:, 0]
    expected_legacy_step = prfo_step(
        h_mw,
        g_mw,
        is_ts=True,
        trust_radius=0.02,
    )
    root_masses = np.sqrt(np.repeat(atoms.get_masses(), 3))
    h_cartesian = root_masses[:, None] * h_mw * root_masses[None, :]
    g_cartesian = root_masses * g_mw
    atoms.calc = _ContaminatedQuadratic(reference, h_cartesian, g_cartesian)

    PRFO(
        str(tmp_path / f"project-{project_rigid_modes}.out"),
        atoms,
        paras={
            "max_iter": 1,
            "trust_radius": 0.02,
            "trust_min": 0.02,
            "project_rigid_modes": project_rigid_modes,
        },
    ).run()
    step_cartesian = (atoms.get_positions() - reference).reshape(-1)
    step_mw = root_masses * step_cartesian
    return step_mw, internal, rigid, expected_legacy_step


def test_prfo_run_projects_contaminated_rigid_step_only_when_explicitly_enabled(
    tmp_path,
):
    projected_step, internal, rigid, _ = _run_contaminated_quadratic_prfo(
        tmp_path,
        project_rigid_modes=True,
    )
    legacy_step, _, _, expected_legacy_step = _run_contaminated_quadratic_prfo(
        tmp_path,
        project_rigid_modes=False,
    )

    # The enabled path reaches PRFO.run's reduced eigensystem and maps the step
    # back to ambient MW coordinates without translation/rotation drift.
    np.testing.assert_allclose(rigid.T @ projected_step, 0.0, atol=2e-15)
    assert abs(float(internal[:, 0] @ projected_step)) > 1e-4

    # Default/False is the historical full-space behavior: a lab-frame rigid
    # mode remains eligible and the most-negative contaminated mode is followed.
    np.testing.assert_allclose(legacy_step, expected_legacy_step, atol=2e-15)
    assert np.linalg.norm(rigid.T @ legacy_step) > 1e-4
