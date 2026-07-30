from __future__ import annotations

import json
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
    V0_MOLECULAR_HNC_REQUIRED_RISM_CLOSURE,
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
from maple.function.calculator.extra_correction.implicit.route2_v0_molecular_so3_quadrature import (
    build_route2_v0_cartesian_euler_product_quadrature,
)
from maple.function.calculator.extra_correction.implicit.route2_v0_molecular_thermodynamics import (
    V0_MOLECULAR_HNC_FIXED_SOLUTE_THERMODYNAMICS,
    molecular_hnc_bulk_functional_pressure_hartree_per_bohr3,
)
from maple.function.calculator.extra_correction.implicit.route2_v0_molecular_weighted_density_bridge import (
    Route2V0MolecularCenterProjection,
    Route2V0MolecularWeightedDensityBridgeAsset,
    Route2V0MolecularWeightedDensityBridgeFunctional,
    Route2V0PeriodicWeightedDensityKernel,
)
from maple.function.calculator.extra_correction.implicit.route2_v0_periodic_coulomb import (
    Route2V0PeriodicCoulombOperator,
)
from maple.function.calculator.extra_correction.implicit.route2_v0_pure_solvent_bridge_certificate import (
    load_route2_v0_pure_solvent_bridge_certificate,
    weighted_density_operator_sha256,
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
        closure="HNC",
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


def test_mace_rism_bridge_derives_its_standard_full_so3_cubic_bspline_map(
    tmp_path: Path,
):
    asset = _asset(tmp_path)
    grid = RegularCartesianGrid(
        origin_bohr=np.array([-4.0, -4.0, -4.0]),
        spacing_bohr=np.full(3, 8.0),
        shape=(2, 2, 2),
    )
    product = build_route2_v0_cartesian_euler_product_quadrature(
        grid=grid,
        polar_order=1,
    )
    external = evaluate_route2_v0_mace_cluster_molecular_external_potential(
        calculator=_FakeZeroFieldMACE(),
        integration_grid=grid,
        solute_atomic_numbers=np.array([1]),
        solute_atom_positions_angstrom=np.array([[-1.0, 0.0, 0.0]]),
        solvent=asset.molecular_reference.molecular_reference,
        configurations=product.configurations,
    )
    bridge = Route2V0MaceClusterRismMolecularHNCBridge.from_cartesian_euler_cubic_bspline_occupancy(
        frozen_solvent_asset=asset,
        external_potential=external,
        rism_kernel=build_route2_v0_asset_bound_rism_kernel(
            frozen_solvent_asset=asset,
            grid=grid,
        ),
        cartesian_euler_quadrature=product,
    )
    compact_bridge = Route2V0MaceClusterRismMolecularHNCBridge.from_cartesian_euler_cubic_bspline_matrix_free(
        frozen_solvent_asset=asset,
        external_potential=external,
        rism_kernel=bridge.rism_kernel,
        cartesian_euler_quadrature=product,
    )

    assert bridge.quadrature is product.quadrature
    assert bridge.is_matrix_free is False
    assert compact_bridge.is_matrix_free is True
    assert compact_bridge.projection.site_occupancy_weights is None
    assert bridge.projection.site_multiplicity.tolist() == [1, 2]
    np.testing.assert_array_equal(
        compact_bridge.projection.site_multiplicity,
        bridge.projection.site_multiplicity,
    )
    occupancy = bridge.projection.site_occupancy_weights.reshape(
        (2, grid.point_count, product.configurations.configuration_count)
    )
    np.testing.assert_allclose(
        np.sum(occupancy, axis=1),
        np.broadcast_to(
            np.array([[1.0], [2.0]]),
            (2, product.configurations.configuration_count),
        ),
        rtol=0.0,
        atol=2.0e-14,
    )
    uniform = np.full(
        product.configurations.configuration_count,
        bridge.projection.uniform_configuration_density_bohr3,
    )
    np.testing.assert_allclose(
        bridge.projection.project_configuration_density(uniform),
        np.broadcast_to(
            bridge.rism_kernel.site_hnc_asset.bulk_number_density_bohr3.reshape(
                (2, 1, 1, 1)
            ),
            (2, *grid.shape),
        ),
        rtol=1.0e-12,
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        compact_bridge.projection.project_configuration_density(uniform),
        bridge.projection.project_configuration_density(uniform),
        rtol=2.0e-14,
        atol=2.0e-14,
    )
    density = uniform * (
        1.0 + 0.02 * np.sin(np.arange(product.configurations.configuration_count))
    )
    direction = (
        uniform * 0.05 * np.cos(np.arange(product.configurations.configuration_count))
    )
    np.testing.assert_allclose(
        compact_bridge.functional.dimensionless_gradient(density),
        bridge.functional.dimensionless_gradient(density),
        rtol=3.0e-14,
        atol=3.0e-14,
    )
    np.testing.assert_allclose(
        compact_bridge.functional.dimensionless_hessian_matvec(density, direction),
        bridge.functional.dimensionless_hessian_matvec(density, direction),
        rtol=4.0e-14,
        atol=4.0e-14,
    )
    assert compact_bridge.functional.grand_potential_hartree(density) == pytest.approx(
        bridge.functional.grand_potential_hartree(density),
        rel=3.0e-14,
        abs=3.0e-16,
    )

    mismatched_product = build_route2_v0_cartesian_euler_product_quadrature(
        grid=grid,
        polar_order=2,
    )
    with pytest.raises(ValueError, match="configurations to equal"):
        Route2V0MaceClusterRismMolecularHNCBridge.from_cartesian_euler_cubic_bspline_occupancy(
            frozen_solvent_asset=asset,
            external_potential=external,
            rism_kernel=bridge.rism_kernel,
            cartesian_euler_quadrature=mismatched_product,
        )


def _source_bound_weighted_density_inputs(tmp_path: Path):
    """Build one source-bound synthetic control without claiming a real liquid."""

    asset, grid, external, quadrature, occupancy = _bridge_inputs(tmp_path)
    rism_kernel = build_route2_v0_asset_bound_rism_kernel(
        frozen_solvent_asset=asset,
        grid=grid,
    )
    hnc_bridge = Route2V0MaceClusterRismMolecularHNCBridge(
        frozen_solvent_asset=asset,
        external_potential=external,
        rism_kernel=rism_kernel,
        quadrature=quadrature,
        site_occupancy_weights=occupancy,
    )
    center = Route2V0MolecularCenterProjection(
        projection=hnc_bridge.projection,
        center_occupancy_weights=np.full(
            (*grid.shape, quadrature.configuration_count),
            1.0 / grid.point_count,
        ),
    )
    weighted_kernel = Route2V0PeriodicWeightedDensityKernel(
        grid=grid,
        kernel_bohr_minus3=np.full(
            grid.shape,
            1.0 / (grid.volume_element_bohr3 * grid.point_count),
        ),
    )
    hnc_pressure = molecular_hnc_bulk_functional_pressure_hartree_per_bohr3(
        hnc_bridge.functional
    )
    target_pressure = (
        asset.pressure_bar * 1.0e5 / (Hartree * 1.602176634e-19 / (Bohr * 1.0e-10) ** 3)
    )
    surface_tension_n_per_m = 0.05
    target_surface_tension = surface_tension_n_per_m / (
        Hartree * 1.602176634e-19 / (Bohr * 1.0e-10) ** 2
    )
    center_digest = weighted_density_operator_sha256(
        operator="molecular-centre-projection",
        construction=center.construction,
        grid=center.grid,
        values=center.center_occupancy_weights,
    )
    kernel_digest = weighted_density_operator_sha256(
        operator="weighted-density-kernel",
        construction=weighted_kernel.construction,
        grid=weighted_kernel.grid,
        values=weighted_kernel.kernel_bohr_minus3,
    )
    certificate_payload = {
        "construction": "route2-v0-pure-solvent-bridge-certificate-v1",
        "schema_version": 1,
        "admission": {
            "evidence_scope": "synthetic-control",
            "physical_liquid_admitted": False,
            "claim_boundary": (
                "Synthetic source-bound test control; no physical-liquid claim."
            ),
        },
        "liquid_source": {
            "solvent_id": asset.solvent_id,
            "model_identifier": asset.model_identifier,
            "closure": asset.closure,
            "temperature_kelvin": asset.temperature_kelvin,
            "pressure_bar": asset.pressure_bar,
            "molecular_bulk_number_density_bohr3": (
                center.molecular_bulk_number_density_bohr3
            ),
            "source_file_sha256": {file.role: file.sha256 for file in asset.files},
        },
        "operators": {
            "center_projection_sha256": center_digest,
            "weighted_density_kernel_sha256": kernel_digest,
        },
        "pure_liquid_pressure": {
            "target_hartree_per_bohr3": target_pressure,
        },
        "surface_tension": {
            "target_n_per_m": surface_tension_n_per_m,
            "target_hartree_per_bohr2": target_surface_tension,
            "independent_reference": "synthetic control; no physical claim",
        },
        "bridge": {
            "hnc_bulk_pressure_hartree_per_bohr3": hnc_pressure,
            "cubic_coefficient_hartree_bohr6": (
                (hnc_pressure - target_pressure)
                / center.molecular_bulk_number_density_bohr3**3
            ),
        },
        "planar_interface": {
            "quartic_bracket": {
                "low_hartree_bohr15": 0.0,
                "high_hartree_bohr15": 2.0e7,
                "root_hartree_bohr15": 1.0e7,
            },
            "surface_tension": {
                "low_hartree_per_bohr2": 0.9 * target_surface_tension,
                "high_hartree_per_bohr2": 1.1 * target_surface_tension,
                "root_hartree_per_bohr2": target_surface_tension,
                "root_tolerance_hartree_per_bohr2": 1.0e-12,
            },
            "stationarity": {
                "low_residual": 1.0e-12,
                "high_residual": 1.0e-12,
                "root_residual": 1.0e-12,
                "tolerance": 1.0e-10,
            },
            "transverse_area_bohr2": 64.0,
            "interface_count": 2,
            "grid_refinement": {
                "coarse_hartree_per_bohr2": target_surface_tension,
                "fine_hartree_per_bohr2": target_surface_tension,
                "tolerance_hartree_per_bohr2": 1.0e-12,
            },
            "same_scalar_all_terms_retained": True,
        },
        "no_target_policy": {
            "target_solvation_labels_used": False,
            "post_training": False,
            "fine_tuning": False,
            "experimental_solvation_fit": False,
            "map_or_uq_calibration": False,
            "error_driven_cavity_or_dispersion_adjustment": False,
            "excluded_target_label_sets": [
                "mnsol",
                "freesolv",
                "development",
                "confirmation",
                "blind",
            ],
        },
    }
    certificate_path = tmp_path / "pure-solvent-bridge-certificate.json"
    certificate_path.write_text(
        json.dumps(certificate_payload, indent=2),
        encoding="utf-8",
    )
    return (
        asset,
        rism_kernel,
        hnc_bridge.functional,
        center,
        weighted_kernel,
        certificate_path,
        certificate_payload,
    )


def _synthetic_homogeneous_phase_coexistence_evidence() -> dict[str, object]:
    """Return parser-only phase evidence for a nonphysical source-upgrade test.

    These values are intentionally confined to a temporary test certificate.
    They do not describe the fixture's liquid and cannot promote its explicitly
    nonphysical source asset; the test verifies that this second protection
    still rejects the attempted scope upgrade.
    """

    return {
        "construction": "route2-v0-molecular-homogeneous-phase-gate-v1",
        "same_scalar_all_terms_retained": True,
        "zero_external_potential": {
            "residual_hartree": 0.0,
            "tolerance_hartree": 1.0e-12,
        },
        "tolerances": {
            "stationarity": 1.0e-10,
            "gradient_uniformity": 1.0e-10,
            "curvature_hartree_per_bohr3": 1.0e-12,
        },
        "liquid": {
            "density_scale": 1.0,
            "grand_potential_density_hartree_per_bohr3": 0.0,
            "stationarity_residual": 0.0,
            "gradient_uniformity_residual": 0.0,
            "curvature_hartree_per_bohr3": 1.0e-6,
        },
        "gas": {
            "density_scale": 0.1,
            "grand_potential_density_hartree_per_bohr3": 0.0,
            "stationarity_residual": 0.0,
            "gradient_uniformity_residual": 0.0,
            "curvature_hartree_per_bohr3": 1.0e-6,
        },
        "coexistence": {
            "grand_potential_density_difference_hartree_per_bohr3": 0.0,
            "tolerance_hartree_per_bohr3": 1.0e-12,
        },
    }


def test_weighted_density_bridge_requires_a_live_source_bound_certificate(tmp_path):
    (
        asset,
        rism_kernel,
        hnc_functional,
        center,
        weighted_kernel,
        certificate_path,
        _,
    ) = _source_bound_weighted_density_inputs(tmp_path)
    certificate = load_route2_v0_pure_solvent_bridge_certificate(certificate_path)

    bridge_asset = Route2V0MolecularWeightedDensityBridgeAsset.from_source_bound_pure_solvent_certificate(
        hnc_functional=hnc_functional,
        frozen_solvent_asset=asset,
        rism_kernel=rism_kernel,
        center_projection=center,
        kernel=weighted_kernel,
        certificate=certificate,
    )
    functional = Route2V0MolecularWeightedDensityBridgeFunctional(
        hnc_functional=hnc_functional,
        bridge_asset=bridge_asset,
    )

    assert bridge_asset.pure_solvent_certificate_sha256 == certificate.content_sha256
    assert bridge_asset.is_source_bound_pure_solvent_asset is True
    bridge_asset.require_source_bound_pure_solvent_asset()
    assert certificate.is_physical_pure_liquid_admission is False
    assert certificate.has_homogeneous_phase_coexistence is False
    assert bridge_asset.is_physical_pure_solvent_asset is False
    with pytest.raises(ValueError, match="physical pure-liquid admission"):
        bridge_asset.require_physical_pure_solvent_asset()
    assert bridge_asset.quartic_coefficient_hartree_bohr15 == pytest.approx(
        certificate.quartic_coefficient_hartree_bohr15
    )
    assert bridge_asset.target_surface_tension_hartree_per_bohr2 == pytest.approx(
        certificate.target_surface_tension_hartree_per_bohr2
    )
    assert functional.bulk_functional_pressure_hartree_per_bohr3 == pytest.approx(
        certificate.target_bulk_pressure_hartree_per_bohr3,
        rel=2.0e-13,
        abs=2.0e-16,
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda payload: payload["liquid_source"]["source_file_sha256"].update(
                {"cvv": "0" * 64}
            ),
            "source hashes do not match",
        ),
        (
            lambda payload: payload["operators"].update(
                {"center_projection_sha256": "0" * 64}
            ),
            "molecular-centre projection does not match",
        ),
    ],
)
def test_weighted_density_certificate_rejects_mutated_source_or_operator(
    tmp_path,
    mutation,
    message,
):
    (
        asset,
        rism_kernel,
        hnc_functional,
        center,
        weighted_kernel,
        certificate_path,
        payload,
    ) = _source_bound_weighted_density_inputs(tmp_path)
    mutation(payload)
    certificate_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    certificate = load_route2_v0_pure_solvent_bridge_certificate(certificate_path)

    with pytest.raises(ValueError, match=message):
        Route2V0MolecularWeightedDensityBridgeAsset.from_source_bound_pure_solvent_certificate(
            hnc_functional=hnc_functional,
            frozen_solvent_asset=asset,
            rism_kernel=rism_kernel,
            center_projection=center,
            kernel=weighted_kernel,
            certificate=certificate,
        )


def test_weighted_density_certificate_rejects_solvation_label_policy(tmp_path):
    *_, certificate_path, payload = _source_bound_weighted_density_inputs(tmp_path)
    payload["no_target_policy"]["experimental_solvation_fit"] = True
    certificate_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with pytest.raises(ValueError, match="experimental_solvation_fit"):
        load_route2_v0_pure_solvent_bridge_certificate(certificate_path)


@pytest.mark.parametrize(
    ("scope", "physical", "message"),
    [
        ("synthetic-control", True, "scope and physical-liquid admission flag"),
        ("not-a-known-scope", False, "Unsupported pure-solvent bridge evidence"),
    ],
)
def test_weighted_density_certificate_rejects_ambiguous_evidence_admission(
    tmp_path,
    scope,
    physical,
    message,
):
    *_, certificate_path, payload = _source_bound_weighted_density_inputs(tmp_path)
    payload["admission"]["evidence_scope"] = scope
    payload["admission"]["physical_liquid_admitted"] = physical
    certificate_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_route2_v0_pure_solvent_bridge_certificate(certificate_path)


def test_v1_weighted_density_certificate_cannot_claim_physical_liquid_admission(
    tmp_path,
):
    *_, certificate_path, payload = _source_bound_weighted_density_inputs(tmp_path)
    payload["admission"] = {
        "evidence_scope": "physical-pure-liquid-admission",
        "physical_liquid_admitted": True,
        "claim_boundary": "Attempted upgrade of a nonphysical source asset.",
    }
    payload["homogeneous_phase_coexistence"] = (
        _synthetic_homogeneous_phase_coexistence_evidence()
    )
    certificate_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with pytest.raises(ValueError, match="certificate v1 cannot claim"):
        load_route2_v0_pure_solvent_bridge_certificate(certificate_path)


def test_physical_scope_certificate_requires_full_homogeneous_phase_evidence(tmp_path):
    *_, certificate_path, payload = _source_bound_weighted_density_inputs(tmp_path)
    payload["admission"] = {
        "evidence_scope": "physical-pure-liquid-admission",
        "physical_liquid_admitted": True,
        "claim_boundary": "Missing phase evidence must fail before source admission.",
    }
    certificate_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with pytest.raises(ValueError, match="requires homogeneous phase coexistence"):
        load_route2_v0_pure_solvent_bridge_certificate(certificate_path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda payload: payload["homogeneous_phase_coexistence"]["gas"].update(
                {"curvature_hartree_per_bohr3": 0.0}
            ),
            "gas phase is not a strict local minimum",
        ),
        (
            lambda payload: payload["homogeneous_phase_coexistence"][
                "coexistence"
            ].update({"grand_potential_density_difference_hartree_per_bohr3": 1.0e-5}),
            "coexistence gap must equal",
        ),
    ],
)
def test_physical_scope_certificate_rejects_invalid_homogeneous_phase_evidence(
    tmp_path,
    mutation,
    message,
):
    *_, certificate_path, payload = _source_bound_weighted_density_inputs(tmp_path)
    payload["admission"] = {
        "evidence_scope": "physical-pure-liquid-admission",
        "physical_liquid_admitted": True,
        "claim_boundary": "Temporary parser-only physical-scope validation control.",
    }
    payload["homogeneous_phase_coexistence"] = (
        _synthetic_homogeneous_phase_coexistence_evidence()
    )
    mutation(payload)
    certificate_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_route2_v0_pure_solvent_bridge_certificate(certificate_path)


def test_weighted_density_bridge_rechecks_certificate_content_before_admission(
    tmp_path,
):
    (
        asset,
        rism_kernel,
        hnc_functional,
        center,
        weighted_kernel,
        certificate_path,
        payload,
    ) = _source_bound_weighted_density_inputs(tmp_path)
    certificate = load_route2_v0_pure_solvent_bridge_certificate(certificate_path)
    bridge_asset = Route2V0MolecularWeightedDensityBridgeAsset.from_source_bound_pure_solvent_certificate(
        hnc_functional=hnc_functional,
        frozen_solvent_asset=asset,
        rism_kernel=rism_kernel,
        center_projection=center,
        kernel=weighted_kernel,
        certificate=certificate,
    )
    certificate_path.write_text(
        json.dumps(payload, sort_keys=True),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="content changed"):
        bridge_asset.require_source_bound_pure_solvent_asset()
    with pytest.raises(ValueError, match="content changed"):
        Route2V0MolecularWeightedDensityBridgeAsset.from_source_bound_pure_solvent_certificate(
            hnc_functional=hnc_functional,
            frozen_solvent_asset=asset,
            rism_kernel=rism_kernel,
            center_projection=center,
            kernel=weighted_kernel,
            certificate=certificate,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda payload: payload["bridge"].update(
                {
                    "cubic_coefficient_hartree_bohr6": (
                        payload["bridge"]["cubic_coefficient_hartree_bohr6"] * 1.01
                    )
                }
            ),
            "pressure identity",
        ),
        (
            lambda payload: payload["surface_tension"].update(
                {
                    "target_hartree_per_bohr2": (
                        payload["surface_tension"]["target_hartree_per_bohr2"] * 1.01
                    )
                }
            ),
            "exact SI conversion",
        ),
        (
            lambda payload: payload["planar_interface"]["surface_tension"].update(
                {
                    "root_hartree_per_bohr2": (
                        payload["planar_interface"]["surface_tension"][
                            "root_hartree_per_bohr2"
                        ]
                        * 1.2
                    )
                }
            ),
            "root misses",
        ),
    ],
)
def test_weighted_density_certificate_rejects_broken_pure_liquid_math(
    tmp_path,
    mutation,
    message,
):
    *_, certificate_path, payload = _source_bound_weighted_density_inputs(tmp_path)
    mutation(payload)
    certificate_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_route2_v0_pure_solvent_bridge_certificate(certificate_path)


def test_asset_bound_molecular_hnc_bridge_rejects_a_pse3_bulk_source(tmp_path):
    hnc_asset, grid, external, quadrature, occupancy = _bridge_inputs(tmp_path / "hnc")
    hnc_kernel = build_route2_v0_asset_bound_rism_kernel(
        frozen_solvent_asset=hnc_asset,
        grid=grid,
    )
    pse3_manifest, _ = write_route2_v0_test_manifest(
        tmp_path / "pse3",
        model_identifier="cSPCE-pse3",
        closure="PSE3",
    )
    pse3_asset = load_route2_v0_frozen_solvent_registry(pse3_manifest).asset_for(
        "water"
    )

    assert V0_MOLECULAR_HNC_REQUIRED_RISM_CLOSURE == "HNC"
    with pytest.raises(ValueError, match="requires an HNC bulk closure"):
        build_route2_v0_asset_bound_rism_kernel(
            frozen_solvent_asset=pse3_asset,
            grid=grid,
        )
    with pytest.raises(ValueError, match="requires an HNC bulk closure"):
        Route2V0MaceClusterRismMolecularHNCBridge(
            frozen_solvent_asset=pse3_asset,
            external_potential=external,
            rism_kernel=hnc_kernel,
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
