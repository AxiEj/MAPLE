from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import pytest
from ase import Atoms
from ase.units import Bohr, Hartree
from route2_v0_asset_fixture import write_route2_v0_test_manifest

from maple.function.calculator.calculator_base import ROUTE2_SMD_CALCULATOR_PROFILE
from maple.function.calculator.extra_correction.implicit.route2_v0_mace_cluster_external_potential import (
    evaluate_route2_v0_mace_cluster_molecular_external_potential,
)
from maple.function.calculator.extra_correction.implicit.route2_v0_mace_cluster_rism_bridge import (
    V0_MACE_CLUSTER_RISM_BRIDGE_CONSTRUCTION,
    V0_MACE_CLUSTER_RISM_CROSS_MODEL_REFERENCE,
    Route2V0MaceClusterRismMolecularHNCBridge,
    build_route2_v0_asset_bound_rism_kernel,
)
from maple.function.calculator.extra_correction.implicit.route2_v0_molecular_external_potential import (
    Route2V0MolecularConfigurations,
    Route2V0MolecularSolventReference,
)
from maple.function.calculator.extra_correction.implicit.route2_v0_molecular_ideal_gas import (
    RIGID_MOLECULAR_ORIENTATION_MEASURE,
    Route2V0MolecularConfigurationQuadrature,
)
from maple.function.calculator.extra_correction.implicit.route2_v0_molecular_thermodynamics import (
    V0_MOLECULAR_HNC_FIXED_SOLUTE_THERMODYNAMICS,
)
from maple.function.calculator.extra_correction.implicit.route2_v0_periodic_coulomb import (
    Route2V0PeriodicCoulombOperator,
)
from maple.function.calculator.extra_correction.implicit.route2_v0_rism_energy_conjugate import (
    Route2V0RismEnergyConjugateKernel,
)
from maple.function.calculator.extra_correction.implicit.route2_v0_rism_reciprocal import (
    Route2V0RismShortRangeReciprocalControl,
)
from maple.function.calculator.extra_correction.implicit.route2_v0_solvent_asset import (
    load_route2_v0_frozen_solvent_registry,
)
from maple.function.calculator.extra_correction.implicit.route2_v0_structured_solvent import (
    RegularCartesianGrid,
)
from maple.function.route2_smd_profiles import MACEPOL_MOLECULAR_REALSPACE_PROFILE


class _FakeZeroFieldMACE:
    """A translation-invariant zero-field scalar stand-in for bridge tests."""

    mace_polar_checkpoint_provenance: ClassVar[dict[str, object]] = {
        "identifier": "polar-1-m",
        "release_url": (
            "https://github.com/ACEsuit/mace-foundations/releases/download/"
            "mace_polar_1/MACE-POLAR-1-M.model"
        ),
        "sha256": "fab8b8713c832f31a2a853aaa22fd638be8a369cbf5095e6b3e982a18d10e93a",
        "size_bytes": 68_133_235,
    }
    mace_torch_version = "0.3.16"
    route2_smd_profile = ROUTE2_SMD_CALCULATOR_PROFILE
    long_range_evaluator_profile = MACEPOL_MOLECULAR_REALSPACE_PROFILE

    def polar_state(self, atoms: Atoms, *, compute_forces: bool):
        positions = atoms.get_positions()
        numbers = np.asarray(atoms.numbers, dtype=float)
        energy = float(np.sum(0.1 * numbers))
        for left in range(len(atoms)):
            for right in range(left + 1, len(atoms)):
                displacement = positions[left] - positions[right]
                energy += float(
                    0.02
                    * numbers[left]
                    * numbers[right]
                    * np.exp(-np.dot(displacement, displacement))
                )
        return (
            SimpleNamespace(
                energy_ev=energy,
                fixed_field_forces_ev_per_angstrom=(
                    np.zeros((len(atoms), 3)) if compute_forces else None
                ),
            ),
            {},
        )


def _asset(tmp_path: Path):
    manifest, _ = write_route2_v0_test_manifest(
        tmp_path,
        model_identifier="cSPCE",
    )
    return load_route2_v0_frozen_solvent_registry(manifest).asset_for("water")


def _bridge_inputs(
    tmp_path: Path,
    *,
    compute_forces: bool = False,
    solute_x_angstrom: float = -1.0,
):
    asset = _asset(tmp_path)
    grid = RegularCartesianGrid(
        origin_bohr=np.zeros(3),
        spacing_bohr=np.full(3, 8.0),
        shape=(2, 2, 2),
    )
    solvent = asset.molecular_reference.molecular_reference
    configurations = Route2V0MolecularConfigurations(
        translations_bohr=np.array([[8.0, 0.0, 0.0], [9.0, 0.0, 0.0]]),
        rotations=np.repeat(np.eye(3)[None], 2, axis=0),
    )
    external = evaluate_route2_v0_mace_cluster_molecular_external_potential(
        calculator=_FakeZeroFieldMACE(),
        integration_grid=grid,
        solute_atomic_numbers=np.array([1]),
        solute_atom_positions_angstrom=np.array([[solute_x_angstrom, 0.0, 0.0]]),
        solvent=solvent,
        configurations=configurations,
        compute_forces=compute_forces,
    )
    quadrature = Route2V0MolecularConfigurationQuadrature(
        configurations=configurations,
        phase_space_weights_bohr3=np.full(
            configurations.configuration_count,
            grid.point_count
            * grid.volume_element_bohr3
            * RIGID_MOLECULAR_ORIENTATION_MEASURE
            / configurations.configuration_count,
        ),
    )
    occupancy = np.empty((2, *grid.shape, configurations.configuration_count))
    occupancy[0] = 1.0 / grid.point_count
    occupancy[1] = 2.0 / grid.point_count
    return asset, grid, external, quadrature, occupancy


def _bridge(
    tmp_path: Path,
    *,
    compute_forces: bool = False,
    solute_x_angstrom: float = -1.0,
):
    asset, grid, external, quadrature, occupancy = _bridge_inputs(
        tmp_path,
        compute_forces=compute_forces,
        solute_x_angstrom=solute_x_angstrom,
    )
    return Route2V0MaceClusterRismMolecularHNCBridge(
        frozen_solvent_asset=asset,
        external_potential=external,
        rism_kernel=build_route2_v0_asset_bound_rism_kernel(
            frozen_solvent_asset=asset,
            grid=grid,
        ),
        quadrature=quadrature,
        site_occupancy_weights=occupancy,
    )


def test_asset_bound_bridge_uses_one_checkpoint_locked_mace_source_and_one_frozen_rism_kernel(
    tmp_path,
):
    asset, grid, external, quadrature, occupancy = _bridge_inputs(tmp_path)
    kernel = build_route2_v0_asset_bound_rism_kernel(
        frozen_solvent_asset=asset,
        grid=grid,
    )
    bridge = Route2V0MaceClusterRismMolecularHNCBridge(
        frozen_solvent_asset=asset,
        external_potential=external,
        rism_kernel=kernel,
        quadrature=quadrature,
        site_occupancy_weights=occupancy,
    )

    assert bridge.construction == V0_MACE_CLUSTER_RISM_BRIDGE_CONSTRUCTION
    assert bridge.cross_model_reference == V0_MACE_CLUSTER_RISM_CROSS_MODEL_REFERENCE
    assert bridge.source_provenance.checkpoint_identifier == "polar-1-m"
    np.testing.assert_array_equal(bridge.site_type_indices, np.array([0, 1, 1]))
    assert bridge.functional.projection.external_potential is external
    assert bridge.functional.projection.site_hnc_asset is kernel.site_hnc_asset

    density = bridge.projection.uniform_configuration_density_bohr3 * np.array(
        [1.05, 0.95]
    )
    direction = bridge.projection.uniform_configuration_density_bohr3 * np.array(
        [0.1, -0.1]
    )
    step = 1.0e-5
    finite_difference = (
        bridge.functional.grand_potential_hartree(density + step * direction)
        - bridge.functional.grand_potential_hartree(density - step * direction)
    ) / (2.0 * step)
    analytic = kernel.kbt_hartree * np.sum(
        quadrature.phase_space_weights_bohr3
        * bridge.functional.dimensionless_gradient(density)
        * direction
    )
    assert finite_difference == pytest.approx(analytic, rel=3.0e-9, abs=3.0e-15)


def test_asset_bound_bridge_hessian_certificate_requires_its_stationary_source_bound_state(
    tmp_path,
):
    bridge = _bridge(tmp_path / "certificate")
    state = bridge.functional.solve_picard(
        residual_tolerance=1.0e-12,
        picard_mixing=0.2,
        max_iterations=1000,
    )
    certificate = bridge.stationary_hessian_stability_certificate(
        state,
        residual_tolerance=1.0e-12,
        maximum_dimension=2,
    )

    assert certificate.classification == "positive-definite"
    assert certificate.is_positive_definite
    assert certificate.minimum_eigenvalue_bohr3 > (
        certificate.numerical_eigenvalue_tolerance_bohr3
    )
    assert (
        certificate.reciprocity_relative_frobenius_residual
        <= certificate.reciprocity_relative_tolerance
    )

    nonstationary = bridge.functional.stationary_state(
        bridge.projection.uniform_configuration_density_bohr3 * np.array([1.1, 0.9]),
        iterations=0,
    )
    with pytest.raises(ValueError, match="stationary molecular HNC"):
        bridge.stationary_hessian_stability_certificate(
            nonstationary,
            residual_tolerance=1.0e-12,
        )
    with pytest.raises(ValueError, match="independently verifies molecular HNC"):
        bridge.stationary_hessian_stability_certificate(
            replace(nonstationary, residual_inf=0.0),
            residual_tolerance=1.0e-12,
        )
    with pytest.raises(TypeError, match="finite positive real number"):
        bridge.stationary_hessian_stability_certificate(
            state,
            residual_tolerance=True,
        )
    with pytest.raises(ValueError, match="dense diagnostic limit"):
        bridge.stationary_hessian_stability_certificate(
            state,
            residual_tolerance=1.0e-12,
            maximum_dimension=1,
        )

    other = _bridge(tmp_path / "other")
    other_state = other.functional.solve_picard(
        residual_tolerance=1.0e-12,
        picard_mixing=0.2,
        max_iterations=1000,
    )
    with pytest.raises(ValueError, match="exact bridge projection"):
        bridge.stationary_hessian_stability_certificate(
            other_state,
            residual_tolerance=1.0e-12,
        )

    source_file = bridge.frozen_solvent_asset.files[0].path
    source_file.write_bytes(source_file.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        bridge.stationary_hessian_stability_certificate(
            state,
            residual_tolerance=1.0e-12,
        )


def test_asset_bound_bridge_stationary_mace_envelope_force_matches_minimized_scalar(
    tmp_path,
):
    bridge = _bridge(tmp_path / "center", compute_forces=True)
    state = bridge.functional.solve_picard(
        residual_tolerance=1.0e-12,
        picard_mixing=0.2,
        max_iterations=1000,
    )
    force = bridge.stationary_mace_solute_force_ev_per_angstrom(
        state,
        residual_tolerance=1.0e-12,
    )

    step_angstrom = 1.0e-5
    plus = _bridge(
        tmp_path / "plus",
        solute_x_angstrom=-1.0 + step_angstrom,
    )
    minus = _bridge(
        tmp_path / "minus",
        solute_x_angstrom=-1.0 - step_angstrom,
    )
    plus_state = plus.functional.solve_picard(
        residual_tolerance=1.0e-12,
        picard_mixing=0.2,
        max_iterations=1000,
    )
    minus_state = minus.functional.solve_picard(
        residual_tolerance=1.0e-12,
        picard_mixing=0.2,
        max_iterations=1000,
    )
    finite_difference_force = (
        -Hartree
        * (plus_state.grand_potential_hartree - minus_state.grand_potential_hartree)
        / (2.0 * step_angstrom)
    )
    assert force.shape == (1, 3)
    assert force[0, 0] == pytest.approx(
        finite_difference_force,
        rel=3.0e-8,
        abs=3.0e-10,
    )

    with pytest.raises(ValueError, match="stationary molecular HNC"):
        bridge.stationary_mace_solute_force_ev_per_angstrom(
            bridge.functional.stationary_state(
                bridge.projection.uniform_configuration_density_bohr3
                * np.array([1.1, 0.9]),
                iterations=0,
            ),
            residual_tolerance=1.0e-12,
        )


def test_asset_bound_bridge_exposes_only_same_functional_fixed_solute_thermodynamics(
    tmp_path,
):
    bridge = _bridge(tmp_path)
    state = bridge.functional.solve_picard(
        residual_tolerance=1.0e-12,
        picard_mixing=0.2,
        max_iterations=1000,
    )
    ledger = bridge.stationary_fixed_solute_thermodynamics(
        state,
        residual_tolerance=1.0e-12,
    )

    assert ledger.construction == V0_MOLECULAR_HNC_FIXED_SOLUTE_THERMODYNAMICS
    assert ledger.state is state
    assert ledger.bulk_functional_pressure_work_hartree == pytest.approx(
        ledger.bulk_functional_pressure_hartree_per_bohr3
        * ledger.partial_molar_volume_bohr3
    )
    assert ledger.fixed_solute_pressure_corrected_free_energy_hartree == pytest.approx(
        state.grand_potential_hartree - ledger.bulk_functional_pressure_work_hartree
    )

    bridge.frozen_solvent_asset.file_for("thermodynamic_output").path.write_text(
        "mutated\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="hash mismatch"):
        bridge.stationary_fixed_solute_thermodynamics(
            state,
            residual_tolerance=1.0e-12,
        )


def test_asset_bound_bridge_rejects_noncanonical_geometry_or_mutated_source(
    tmp_path,
):
    asset, grid, external, quadrature, occupancy = _bridge_inputs(tmp_path)
    kernel = build_route2_v0_asset_bound_rism_kernel(
        frozen_solvent_asset=asset,
        grid=grid,
    )
    common = {
        "frozen_solvent_asset": asset,
        "external_potential": external,
        "rism_kernel": kernel,
        "quadrature": quadrature,
        "site_occupancy_weights": occupancy,
    }
    noncanonical_solvent = Route2V0MolecularSolventReference(
        atomic_numbers=np.array([8, 1, 1]),
        site_charges_e=np.array([-0.8, 0.4, 0.4]),
        reference_positions_bohr=np.array(
            [[0.1, 0.0, 0.0], [1.5, 0.0, 0.0], [-0.5, 1.4, 0.0]]
        ),
        provenance_label=asset.molecular_reference.molecular_reference.provenance_label,
    )
    noncanonical_external = (
        evaluate_route2_v0_mace_cluster_molecular_external_potential(
            calculator=_FakeZeroFieldMACE(),
            integration_grid=grid,
            solute_atomic_numbers=np.array([1]),
            solute_atom_positions_angstrom=np.array([[-1.0, 0.0, 0.0]]),
            solvent=noncanonical_solvent,
            configurations=quadrature.configurations,
        )
    )
    with pytest.raises(ValueError, match="canonical molecular reference"):
        Route2V0MaceClusterRismMolecularHNCBridge(
            **{**common, "external_potential": noncanonical_external},
        )

    changed_tail_control = Route2V0RismShortRangeReciprocalControl.from_radial(
        radial=asset.bulk_direct_correlation.split_coulomb_long_range(),
        grid=grid,
        tail_start_angstrom=6.0,
        tail_tolerance_dimensionless=1.0e-8,
    )
    changed_tail_kernel = Route2V0RismEnergyConjugateKernel(
        short_range=changed_tail_control,
        periodic_coulomb=Route2V0PeriodicCoulombOperator(
            grid=grid,
            smear_bohr=(
                asset.bulk_direct_correlation.metadata.coulomb_smear_angstrom / Bohr
            ),
        ),
    )
    with pytest.raises(ValueError, match="does not derive"):
        Route2V0MaceClusterRismMolecularHNCBridge(
            **{**common, "rism_kernel": changed_tail_kernel},
        )

    asset.file_for("site_model").path.write_text("mutated\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        Route2V0MaceClusterRismMolecularHNCBridge(
            **common,
        )
