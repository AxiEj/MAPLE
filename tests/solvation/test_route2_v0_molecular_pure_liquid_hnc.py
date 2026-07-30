from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from route2_v0_asset_fixture import write_route2_v0_test_manifest

from maple.function.calculator.extra_correction.implicit.route2_v0_molecular_external_potential import (
    Route2V0MolecularSolventReference,
)
from maple.function.calculator.extra_correction.implicit.route2_v0_molecular_pure_liquid_external_potential import (
    V0_MOLECULAR_PURE_LIQUID_EXTERNAL_POTENTIAL_CONSTRUCTION,
    Route2V0MolecularPureLiquidExternalPotential,
    build_route2_v0_molecular_pure_liquid_external_potential,
)
from maple.function.calculator.extra_correction.implicit.route2_v0_molecular_pure_liquid_hnc import (
    V0_MOLECULAR_PURE_LIQUID_HNC_CONSTRUCTION,
    Route2V0MolecularPureLiquidHNCAssembly,
)
from maple.function.calculator.extra_correction.implicit.route2_v0_molecular_site_bspline import (
    build_route2_v0_cartesian_euler_cubic_bspline_site_deposition,
)
from maple.function.calculator.extra_correction.implicit.route2_v0_molecular_so3_quadrature import (
    Route2V0CartesianEulerProductQuadrature,
    build_route2_v0_cartesian_euler_product_quadrature,
)
from maple.function.calculator.extra_correction.implicit.route2_v0_solvent_asset import (
    load_route2_v0_frozen_solvent_registry,
)
from maple.function.calculator.extra_correction.implicit.route2_v0_structured_solvent import (
    RegularCartesianGrid,
)

ROOT = Path(__file__).resolve().parents[2]
CSPCE_HNC_CANDIDATE = (
    ROOT
    / "docs/implicit-solvation/benchmarks/route2-v0-solvent-assets/"
    "water-cspce-hnc/manifest.json"
)


def _asset(tmp_path: Path, *, closure: str = "HNC"):
    manifest, _ = write_route2_v0_test_manifest(tmp_path, closure=closure)
    return load_route2_v0_frozen_solvent_registry(manifest).asset_for("water")


def _grid() -> RegularCartesianGrid:
    return RegularCartesianGrid(
        origin_bohr=np.array([-4.0, -4.0, -4.0]),
        spacing_bohr=np.full(3, 8.0),
        shape=(2, 2, 2),
    )


def _product() -> Route2V0CartesianEulerProductQuadrature:
    return build_route2_v0_cartesian_euler_product_quadrature(
        grid=_grid(),
        polar_order=1,
    )


def test_pure_liquid_external_source_is_exact_zero_and_rejects_any_solute_term(
    tmp_path: Path,
):
    asset = _asset(tmp_path)
    product = _product()
    external = build_route2_v0_molecular_pure_liquid_external_potential(
        integration_grid=product.grid,
        solvent=asset.molecular_reference.molecular_reference,
        configurations=product.configurations,
    )

    assert (
        external.construction
        == V0_MOLECULAR_PURE_LIQUID_EXTERNAL_POTENTIAL_CONSTRUCTION
    )
    assert external.external_potential_hartree.flags.writeable is False
    np.testing.assert_array_equal(
        external.external_potential_hartree,
        np.zeros(product.configurations.configuration_count),
    )

    with pytest.raises(ValueError, match="exactly zero"):
        Route2V0MolecularPureLiquidExternalPotential(
            integration_grid=product.grid,
            solvent=asset.molecular_reference.molecular_reference,
            configurations=product.configurations,
            external_potential_hartree=np.full(
                product.configurations.configuration_count,
                np.finfo(float).tiny,
            ),
        )


def test_pure_liquid_hnc_factory_binds_one_hnc_asset_to_full_so3_zero_scalar(
    tmp_path: Path,
):
    asset = _asset(tmp_path)
    product = _product()
    assembly = (
        Route2V0MolecularPureLiquidHNCAssembly.from_cartesian_euler_cubic_bspline_matrix_free(
            frozen_solvent_asset=asset,
            cartesian_euler_quadrature=product,
        )
    )

    assert assembly.construction == V0_MOLECULAR_PURE_LIQUID_HNC_CONSTRUCTION
    assert assembly.projection.is_matrix_free
    assert assembly.functional.projection is assembly.projection
    assert assembly.rism_kernel.grid.shape == product.grid.shape
    np.testing.assert_array_equal(
        assembly.external_potential.external_potential_hartree,
        np.zeros(product.configurations.configuration_count),
    )
    np.testing.assert_array_equal(
        assembly.projection.site_multiplicity,
        assembly.rism_kernel.short_range.radial.metadata.site_multiplicity,
    )

    uniform = np.full(
        product.configurations.configuration_count,
        assembly.projection.uniform_configuration_density_bohr3,
    )
    expected_site_density = np.broadcast_to(
        assembly.rism_kernel.site_hnc_asset.bulk_number_density_bohr3.reshape(
            (assembly.rism_kernel.site_hnc_asset.site_count, 1, 1, 1)
        ),
        (
            assembly.rism_kernel.site_hnc_asset.site_count,
            *product.grid.shape,
        ),
    )
    np.testing.assert_allclose(
        assembly.projection.project_configuration_density(uniform),
        expected_site_density,
        rtol=1.0e-12,
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        assembly.functional.dimensionless_gradient(uniform),
        0.0,
        rtol=0.0,
        atol=3.0e-14,
    )
    _, _, external = assembly.functional.energy_components(uniform)
    assert external == pytest.approx(0.0, abs=1.0e-18)


def test_pure_liquid_hnc_assembly_rejects_non_hnc_assets_and_reference_relabeling(
    tmp_path: Path,
):
    pse3_asset = _asset(tmp_path / "pse3", closure="PSE3")
    product = _product()
    with pytest.raises(ValueError, match="requires an HNC frozen solvent asset"):
        Route2V0MolecularPureLiquidHNCAssembly.from_cartesian_euler_cubic_bspline_matrix_free(
            frozen_solvent_asset=pse3_asset,
            cartesian_euler_quadrature=product,
        )

    hnc_asset = _asset(tmp_path / "hnc")
    source_assembly = (
        Route2V0MolecularPureLiquidHNCAssembly.from_cartesian_euler_cubic_bspline_matrix_free(
            frozen_solvent_asset=hnc_asset,
            cartesian_euler_quadrature=product,
        )
    )
    canonical = hnc_asset.molecular_reference.molecular_reference
    relabelled = Route2V0MolecularSolventReference(
        atomic_numbers=canonical.atomic_numbers,
        site_charges_e=canonical.site_charges_e,
        reference_positions_bohr=canonical.reference_positions_bohr,
        provenance_label="same coordinates but an unbound source label",
    )
    external = build_route2_v0_molecular_pure_liquid_external_potential(
        integration_grid=product.grid,
        solvent=relabelled,
        configurations=product.configurations,
    )
    deposition = build_route2_v0_cartesian_euler_cubic_bspline_site_deposition(
        cartesian_euler_quadrature=product,
        solvent=canonical,
        solvent_site_type_indices=source_assembly.site_type_indices,
        site_count=source_assembly.rism_kernel.site_hnc_asset.site_count,
    )
    with pytest.raises(ValueError, match="geometry and charges must equal"):
        Route2V0MolecularPureLiquidHNCAssembly(
            frozen_solvent_asset=hnc_asset,
            external_potential=external,
            rism_kernel=source_assembly.rism_kernel,
            quadrature=product.quadrature,
            compact_site_deposition=deposition,
        )


def test_checked_in_hnc_candidate_enters_only_the_zero_external_source_control():
    asset = load_route2_v0_frozen_solvent_registry(CSPCE_HNC_CANDIDATE).asset_for(
        "water"
    )
    product = _product()
    assembly = (
        Route2V0MolecularPureLiquidHNCAssembly.from_cartesian_euler_cubic_bspline_matrix_free(
            frozen_solvent_asset=asset,
            cartesian_euler_quadrature=product,
        )
    )

    assert asset.generation_source.status == (
        "source-complete-candidate-not-production-or-accuracy-admitted"
    )
    assert assembly.rism_kernel.short_range.radial.metadata.temperature_kelvin == (
        asset.temperature_kelvin
    )
    assert assembly.functional.projection is assembly.projection
    assert np.max(np.abs(assembly.external_potential.external_potential_hartree)) == 0.0
