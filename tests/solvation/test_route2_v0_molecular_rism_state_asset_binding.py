from __future__ import annotations

from pathlib import Path

import pytest
from ase.units import Bohr, Hartree
from route2_v0_asset_fixture import write_route2_v0_test_manifest

from maple.function.calculator.extra_correction.implicit.route2_v0_bulk_liquid_state_source import (
    Route2V0BulkLiquidPropertySource,
)
from maple.function.calculator.extra_correction.implicit.route2_v0_molecular_rism_state_asset_binding import (
    V0_MOLECULAR_RISM_STATE_ASSET_BINDING_CONSTRUCTION,
    V0_MOLECULAR_RISM_STATE_ASSET_BINDING_STATUS,
    Route2V0MolecularRismStateAssetBinding,
)
from maple.function.calculator.extra_correction.implicit.route2_v0_molecular_rism_state_source import (
    V0_MOLECULAR_RISM_BULK_STATE_REQUIRED_PROPERTIES,
    Route2V0MolecularRismBulkStateSource,
    load_route2_v0_molecular_rism_bulk_state_source,
)
from maple.function.calculator.extra_correction.implicit.route2_v0_solvent_asset import (
    Route2V0FrozenSolventAsset,
    load_route2_v0_frozen_solvent_registry,
)

ROOT = Path(__file__).resolve().parents[2]
BENCHMARKS = ROOT / "docs/implicit-solvation/benchmarks"
CSPCE_HNC_CANDIDATE = (
    BENCHMARKS / "route2-v0-solvent-assets/water-cspce-hnc/manifest.json"
)
DCM_STATE_SOURCE = (
    BENCHMARKS
    / "route2-v0-molecular-rism-state-sources"
    / "dichloromethane-scm-adf-3drism-v1.json"
)

_ELEMENTARY_CHARGE_JOULE = 1.602176634e-19
_HARTREE_PER_BOHR2_TO_NEWTON_PER_METER = (
    Hartree * _ELEMENTARY_CHARGE_JOULE / (Bohr * 1.0e-10) ** 2
)
_HARTREE_PER_BOHR3_TO_PASCAL = (
    Hartree * _ELEMENTARY_CHARGE_JOULE / (Bohr * 1.0e-10) ** 3
)


def _asset(tmp_path: Path):
    manifest, _ = write_route2_v0_test_manifest(tmp_path, closure="HNC")
    return load_route2_v0_frozen_solvent_registry(manifest).asset_for("water")


def _property_source(name: str) -> Route2V0BulkLiquidPropertySource:
    return Route2V0BulkLiquidPropertySource(
        property_name=name,
        origin="independent_measurement",
        document_url="https://example.org/route2-v0-liquid-state",
        document_sha256="a" * 64,
        source_locator=f"Table 1, {name}",
        retrieved_utc="2026-07-30T00:00:00Z",
    )


def _state_source(
    asset: Route2V0FrozenSolventAsset,
    *,
    solvent_id: str | None = None,
    model_identifier: str | None = None,
    model_source_sha256: str | None = None,
    temperature_kelvin: float | None = None,
    pressure_bar: float | None = None,
    molecular_number_density_angstrom3: float | None = None,
    static_dielectric_constant: float | None = None,
) -> Route2V0MolecularRismBulkStateSource:
    rism = asset.molecular_source.rism1d_input
    return Route2V0MolecularRismBulkStateSource(
        solvent_id=asset.solvent_id if solvent_id is None else solvent_id,
        model_identifier=(
            asset.model_identifier if model_identifier is None else model_identifier
        ),
        model_source_sha256=(
            asset.file_for("site_model").sha256
            if model_source_sha256 is None
            else model_source_sha256
        ),
        temperature_kelvin=(
            rism.temperature_kelvin
            if temperature_kelvin is None
            else temperature_kelvin
        ),
        pressure_bar=asset.pressure_bar if pressure_bar is None else pressure_bar,
        molecular_number_density_angstrom3=(
            rism.molecular_number_density_angstrom3
            if molecular_number_density_angstrom3 is None
            else molecular_number_density_angstrom3
        ),
        static_dielectric_constant=(
            rism.dielectric_constant
            if static_dielectric_constant is None
            else static_dielectric_constant
        ),
        isothermal_compressibility_pa_inverse=4.0e-10,
        surface_tension_newton_per_meter=0.072,
        property_sources=tuple(
            _property_source(name)
            for name in sorted(V0_MOLECULAR_RISM_BULK_STATE_REQUIRED_PROPERTIES)
        ),
        claim_boundary=(
            "Synthetic source-bound state fixture; not a molecular liquid or "
            "solvation result."
        ),
        not_claimed=("Finite-k liquid physics remains a separate requirement.",),
    )


def test_state_asset_binding_requires_one_exact_source_bound_rism_state_and_units(
    tmp_path: Path,
):
    asset = _asset(tmp_path)
    source = _state_source(asset)
    binding = Route2V0MolecularRismStateAssetBinding(
        frozen_solvent_asset=asset,
        bulk_state_source=source,
    )

    assert binding.construction == V0_MOLECULAR_RISM_STATE_ASSET_BINDING_CONSTRUCTION
    assert binding.status == V0_MOLECULAR_RISM_STATE_ASSET_BINDING_STATUS
    assert binding.is_molecular_liquid_asset is False
    assert binding.molecular_bulk_number_density_bohr3 == pytest.approx(
        source.molecular_number_density_angstrom3 * Bohr**3
    )
    assert binding.target_bulk_pressure_hartree_per_bohr3 == pytest.approx(
        source.pressure_bar * 1.0e5 / _HARTREE_PER_BOHR3_TO_PASCAL
    )
    assert binding.target_surface_tension_hartree_per_bohr2 == pytest.approx(
        source.surface_tension_newton_per_meter
        / _HARTREE_PER_BOHR2_TO_NEWTON_PER_METER
    )
    assert binding.isothermal_compressibility_hartree_inverse_bohr3 == pytest.approx(
        source.isothermal_compressibility_pa_inverse
        * _HARTREE_PER_BOHR3_TO_PASCAL
    )
    assert binding.number_structure_factor_zero_mode == pytest.approx(
        source.number_structure_factor_zero_mode
    )
    binding.verify_integrity()


def test_state_asset_binding_rejects_identity_model_digest_and_state_mismatches(
    tmp_path: Path,
):
    asset = _asset(tmp_path)

    with pytest.raises(ValueError, match="solvent ID"):
        Route2V0MolecularRismStateAssetBinding(
            frozen_solvent_asset=asset,
            bulk_state_source=_state_source(asset, solvent_id="methanol"),
        )
    with pytest.raises(ValueError, match="model identifier"):
        Route2V0MolecularRismStateAssetBinding(
            frozen_solvent_asset=asset,
            bulk_state_source=_state_source(
                asset,
                model_identifier="different-model",
            ),
        )
    with pytest.raises(ValueError, match="model source digest"):
        Route2V0MolecularRismStateAssetBinding(
            frozen_solvent_asset=asset,
            bulk_state_source=_state_source(asset, model_source_sha256="b" * 64),
        )
    with pytest.raises(ValueError, match="pressure"):
        Route2V0MolecularRismStateAssetBinding(
            frozen_solvent_asset=asset,
            bulk_state_source=_state_source(asset, pressure_bar=1.1),
        )
    with pytest.raises(ValueError, match="temperature"):
        Route2V0MolecularRismStateAssetBinding(
            frozen_solvent_asset=asset,
            bulk_state_source=_state_source(asset, temperature_kelvin=299.0),
        )
    with pytest.raises(ValueError, match="density"):
        Route2V0MolecularRismStateAssetBinding(
            frozen_solvent_asset=asset,
            bulk_state_source=_state_source(
                asset,
                molecular_number_density_angstrom3=0.032,
            ),
        )
    with pytest.raises(ValueError, match="static dielectric"):
        Route2V0MolecularRismStateAssetBinding(
            frozen_solvent_asset=asset,
            bulk_state_source=_state_source(
                asset,
                static_dielectric_constant=70.0,
            ),
        )


def test_checked_in_dichloromethane_state_cannot_promote_water_asset(tmp_path: Path):
    asset = _asset(tmp_path)
    dcm = load_route2_v0_molecular_rism_bulk_state_source(DCM_STATE_SOURCE)

    with pytest.raises(ValueError, match="solvent ID"):
        Route2V0MolecularRismStateAssetBinding(
            frozen_solvent_asset=asset,
            bulk_state_source=dcm,
        )


def test_source_complete_cspce_candidate_still_has_no_matching_state_source():
    water_asset = load_route2_v0_frozen_solvent_registry(
        CSPCE_HNC_CANDIDATE
    ).asset_for("water")
    dcm = load_route2_v0_molecular_rism_bulk_state_source(DCM_STATE_SOURCE)

    with pytest.raises(ValueError, match="solvent ID"):
        Route2V0MolecularRismStateAssetBinding(
            frozen_solvent_asset=water_asset,
            bulk_state_source=dcm,
        )


def test_state_asset_binding_rechecks_the_frozen_asset_hashes(tmp_path: Path):
    asset = _asset(tmp_path)
    binding = Route2V0MolecularRismStateAssetBinding(
        frozen_solvent_asset=asset,
        bulk_state_source=_state_source(asset),
    )
    asset.file_for("site_model").path.write_text("mutated", encoding="utf-8")

    with pytest.raises(ValueError, match="hash mismatch"):
        binding.verify_integrity()
