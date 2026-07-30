from __future__ import annotations

import json
import math

import numpy as np
import pytest

from maple.function.calculator.extra_correction.implicit.route2_v0_bulk_liquid_state_source import (
    V0_BULK_LIQUID_STATE_SOURCE_CONSTRUCTION,
    V0_BULK_LIQUID_STATE_SOURCE_STATUS,
    parse_route2_v0_bulk_liquid_state_source,
)
from maple.function.calculator.extra_correction.implicit.route2_v0_lorentz_dielectric_source import (
    V0_LORENTZ_DIELECTRIC_SOURCE_CONSTRUCTION,
    V0_LORENTZ_DIELECTRIC_SOURCE_STATUS,
    parse_route2_v0_lorentz_nonlocal_dielectric_source,
)
from maple.function.calculator.extra_correction.implicit.route2_v0_nonlocal_dielectric import (
    V0_LORENTZ_NONLOCAL_DIELECTRIC_CONSTRUCTION,
    V0_LORENTZ_NONLOCAL_DIELECTRIC_SCOPE,
    V0_NONLOCAL_DIELECTRIC_CONSTRUCTION,
    V0_NONLOCAL_DIELECTRIC_SCOPE,
    Route2V0LorentzNonlocalDielectricSpectrum,
    Route2V0NonlocalDielectricOperator,
)
from maple.function.calculator.extra_correction.implicit.route2_v0_structured_solvent import (
    RegularCartesianGrid,
)


def _grid(
    shape: tuple[int, int, int] = (10, 8, 6),
    cell_lengths_bohr: tuple[float, float, float] = (4.0, 5.0, 6.0),
) -> RegularCartesianGrid:
    return RegularCartesianGrid(
        origin_bohr=np.array([-1.0, -1.5, -2.0]),
        spacing_bohr=np.asarray(cell_lengths_bohr) / np.asarray(shape),
        shape=shape,
    )


def _neutral_field(
    grid: RegularCartesianGrid,
    rng: np.random.Generator,
) -> np.ndarray:
    result = rng.normal(size=grid.shape)
    result -= np.mean(result)
    return result


def _reciprocal_even_dielectric(grid: RegularCartesianGrid) -> np.ndarray:
    reciprocal_axes = tuple(
        2.0 * math.pi * np.fft.fftfreq(grid.shape[axis], d=grid.spacing_bohr[axis])
        for axis in range(3)
    )
    wavevector_squared = sum(
        axis_values**2 for axis_values in np.meshgrid(*reciprocal_axes, indexing="ij")
    )
    return 1.0 + 3.0 * np.exp(-0.18 * wavevector_squared)


def _bulk_state_source(*, static: float = 30.0, optical: float = 1.8):
    property_names = (
        "temperature_kelvin",
        "pressure_bar",
        "molecular_number_density_angstrom3",
        "static_dielectric_constant",
        "optical_dielectric_constant",
        "isothermal_compressibility_pa_inverse",
        "surface_tension_newton_per_meter",
    )
    return parse_route2_v0_bulk_liquid_state_source(
        json.dumps(
            {
                "protocol_id": V0_BULK_LIQUID_STATE_SOURCE_CONSTRUCTION,
                "schema_version": 1,
                "status": V0_BULK_LIQUID_STATE_SOURCE_STATUS,
                "solvent_id": "lorentz-fixture",
                "model": {
                    "identifier": "lorentz-fixture-all-atom-v1",
                    "source_sha256": "a" * 64,
                },
                "state": {
                    "temperature_kelvin": 298.15,
                    "pressure_bar": 1.0,
                    "molecular_number_density_angstrom3": 0.025,
                    "static_dielectric_constant": static,
                    "optical_dielectric_constant": optical,
                    "isothermal_compressibility_pa_inverse": 5.0e-10,
                    "surface_tension_newton_per_meter": 0.03,
                },
                "property_sources": [
                    {
                        "property": name,
                        "origin": "upstream_model_validation",
                        "document_url": "https://example.org/lorentz-fixture.pdf",
                        "document_sha256": "b" * 64,
                        "source_locator": f"Table 1, {name}",
                        "retrieved_utc": "2026-07-30T00:00:00Z",
                    }
                    for name in property_names
                ],
                "no_target_policy": {
                    "post_training": False,
                    "fine_tuning": False,
                    "experimental_solvation_fit": False,
                    "map_or_uq_calibration": False,
                    "target_solvation_labels_used": False,
                },
                "claim_boundary": (
                    "This is a source-only state fixture, not a molecular liquid "
                    "or solvation endpoint."
                ),
                "not_claimed": [
                    "The state does not determine a finite-wavevector liquid functional."
                ],
            }
        )
    )


def _lorentz_response_source(
    *,
    static: float = 30.0,
    optical: float = 1.8,
    correlation_length_bohr: float | None = 0.65,
):
    return parse_route2_v0_lorentz_nonlocal_dielectric_source(
        json.dumps(
            {
                "protocol_id": V0_LORENTZ_DIELECTRIC_SOURCE_CONSTRUCTION,
                "schema_version": 1,
                "status": V0_LORENTZ_DIELECTRIC_SOURCE_STATUS,
                "solvent_id": "lorentz-fixture",
                "model": {
                    "identifier": "lorentz-fixture-all-atom-v1",
                    "source_sha256": "a" * 64,
                },
                "response": {
                    "static_dielectric_constant": static,
                    "optical_dielectric_constant": optical,
                    "orientational_correlation_length_bohr": correlation_length_bohr,
                },
                "correlation_length_source": (
                    {
                        "origin": "ab_initio",
                        "document_url": "https://example.org/lorentz-correlation.pdf",
                        "document_sha256": "c" * 64,
                        "source_locator": "Figure 3, longitudinal response fit",
                        "retrieved_utc": "2026-07-30T00:00:00Z",
                    }
                    if correlation_length_bohr is not None
                    else None
                ),
                "no_target_policy": {
                    "post_training": False,
                    "fine_tuning": False,
                    "experimental_solvation_fit": False,
                    "map_or_uq_calibration": False,
                    "target_solvation_labels_used": False,
                },
                "claim_boundary": (
                    "This is a source-only response fixture, not a molecular liquid "
                    "or solvation endpoint."
                ),
                "not_claimed": [
                    "The response does not determine a molecular liquid functional."
                ],
            }
        )
    )


def _wavevector_squared(grid: RegularCartesianGrid) -> np.ndarray:
    axes = tuple(
        2.0 * math.pi * np.fft.fftfreq(grid.shape[axis], d=grid.spacing_bohr[axis])
        for axis in range(3)
    )
    return sum(axis**2 for axis in np.meshgrid(*axes, indexing="ij"))


def test_nonlocal_dielectric_matches_one_exact_reciprocal_mode_and_zero_gauge():
    grid = _grid()
    dielectric = _reciprocal_even_dielectric(grid)
    operator = Route2V0NonlocalDielectricOperator(grid, dielectric)
    x = grid.spacing_bohr[0] * np.arange(grid.shape[0])
    wavevector = 2.0 * math.pi / operator.cell_lengths_bohr[0]
    charge_density = (
        0.02
        * np.cos(wavevector * x)[:, None, None]
        * np.ones((1, grid.shape[1], grid.shape[2]))
    )
    epsilon_at_mode = dielectric[1, 0, 0]
    multiplier = 4.0 * math.pi * (1.0 / epsilon_at_mode - 1.0) / wavevector**2

    potential = operator.reaction_potential_hartree_per_e(charge_density)

    np.testing.assert_allclose(
        potential,
        multiplier * charge_density,
        rtol=0.0,
        atol=3.0e-15,
    )
    assert abs(float(np.mean(potential))) < 3.0e-16
    assert operator.fourier_reaction_green_bohr2[0, 0, 0] == 0.0
    assert operator.construction == V0_NONLOCAL_DIELECTRIC_CONSTRUCTION
    assert operator.response_scope == V0_NONLOCAL_DIELECTRIC_SCOPE
    dielectric[1, 0, 0] = 1.0
    assert operator.fourier_dielectric_spectrum[1, 0, 0] == pytest.approx(
        epsilon_at_mode
    )
    with pytest.raises(ValueError, match="read-only"):
        operator.fourier_dielectric_spectrum[1, 0, 0] = 1.0


def test_lorentz_nonlocal_dielectric_fixes_the_full_spectrum_from_physical_limits():
    grid = _grid()
    static = 28.0
    optical = 1.7
    correlation_length = 0.85
    model = Route2V0LorentzNonlocalDielectricSpectrum(
        grid=grid,
        static_dielectric_constant=static,
        optical_dielectric_constant=optical,
        correlation_length_bohr=correlation_length,
    )
    expected = optical + (static - optical) / (
        1.0 + correlation_length**2 * _wavevector_squared(grid)
    )

    assert model.construction == V0_LORENTZ_NONLOCAL_DIELECTRIC_CONSTRUCTION
    assert model.response_scope == V0_LORENTZ_NONLOCAL_DIELECTRIC_SCOPE
    assert model.is_bulk_state_bound is False
    assert model.is_total_solvation_asset is False
    np.testing.assert_allclose(
        model.dielectric_spectrum,
        expected,
        rtol=0.0,
        atol=4.0e-15,
    )
    assert model.dielectric_spectrum[0, 0, 0] == pytest.approx(static)
    assert np.all(model.dielectric_spectrum >= optical)
    assert model.dielectric_spectrum[1, 0, 0] < static
    with pytest.raises(ValueError, match="source-bound"):
        model.require_bulk_state_source()
    with pytest.raises(ValueError, match="read-only"):
        model.dielectric_spectrum[1, 0, 0] = 1.0


def test_lorentz_spectrum_requires_both_state_and_correlation_sources():
    bulk_source = _bulk_state_source(static=30.0, optical=1.8)
    state_bound_only = (
        Route2V0LorentzNonlocalDielectricSpectrum.from_bulk_liquid_state_source(
            grid=_grid(),
            bulk_state_source=bulk_source,
            orientational_correlation_length_bohr=0.65,
        )
    )
    response_source = _lorentz_response_source(
        static=30.0,
        optical=1.8,
        correlation_length_bohr=0.65,
    )
    model = Route2V0LorentzNonlocalDielectricSpectrum.from_source_bound_records(
        grid=_grid(),
        bulk_state_source=bulk_source,
        lorentz_response_source=response_source,
    )
    operator = model.as_operator()
    rng = np.random.default_rng(20260730)
    left = _neutral_field(model.grid, rng)
    right = _neutral_field(model.grid, rng)

    assert model.is_bulk_state_bound is True
    assert model.is_fully_source_bound is True
    assert model.require_bulk_state_source() is bulk_source
    assert model.require_source_bound_records() == (bulk_source, response_source)
    assert state_bound_only.is_fully_source_bound is False
    with pytest.raises(ValueError, match="correlation length"):
        state_bound_only.require_source_bound_records()
    assert model.static_dielectric_constant == pytest.approx(
        bulk_source.static_dielectric_constant
    )
    assert model.optical_dielectric_constant == pytest.approx(
        bulk_source.optical_dielectric_constant
    )
    np.testing.assert_allclose(
        operator.fourier_dielectric_spectrum,
        model.dielectric_spectrum,
        rtol=0.0,
        atol=0.0,
    )
    assert operator.reaction_pairing_hartree(left, right) == pytest.approx(
        operator.reaction_pairing_hartree(right, left),
        rel=2.0e-13,
        abs=2.0e-13,
    )
    assert operator.polarization_energy_hartree(left) <= 1.0e-14

    with pytest.raises(ValueError, match="must equal the attached bulk-state"):
        Route2V0LorentzNonlocalDielectricSpectrum(
            grid=_grid(),
            static_dielectric_constant=bulk_source.static_dielectric_constant + 0.1,
            optical_dielectric_constant=bulk_source.optical_dielectric_constant,
            correlation_length_bohr=0.65,
            bulk_state_source=bulk_source,
        )

    with pytest.raises(ValueError, match="correlation length must equal"):
        Route2V0LorentzNonlocalDielectricSpectrum(
            grid=_grid(),
            static_dielectric_constant=bulk_source.static_dielectric_constant,
            optical_dielectric_constant=bulk_source.optical_dielectric_constant,
            correlation_length_bohr=0.7,
            bulk_state_source=bulk_source,
            lorentz_response_source=response_source,
        )


def test_lorentz_nonlocal_dielectric_rejects_unidentified_or_nonpassive_inputs():
    grid = _grid()

    with pytest.raises(ValueError, match="requires an independently sourced"):
        Route2V0LorentzNonlocalDielectricSpectrum(
            grid=grid,
            static_dielectric_constant=20.0,
            optical_dielectric_constant=1.8,
            correlation_length_bohr=None,
        )
    with pytest.raises(ValueError, match="unidentifiable"):
        Route2V0LorentzNonlocalDielectricSpectrum(
            grid=grid,
            static_dielectric_constant=1.8,
            optical_dielectric_constant=1.8,
            correlation_length_bohr=0.5,
        )
    constant = Route2V0LorentzNonlocalDielectricSpectrum(
        grid=grid,
        static_dielectric_constant=1.8,
        optical_dielectric_constant=1.8,
        correlation_length_bohr=None,
    )
    np.testing.assert_allclose(constant.dielectric_spectrum, 1.8)
    constant_bulk_source = _bulk_state_source(static=1.8, optical=1.8)
    constant_response_source = _lorentz_response_source(
        static=1.8,
        optical=1.8,
        correlation_length_bohr=None,
    )
    constant_source_bound = (
        Route2V0LorentzNonlocalDielectricSpectrum.from_source_bound_records(
            grid=grid,
            bulk_state_source=constant_bulk_source,
            lorentz_response_source=constant_response_source,
        )
    )
    assert constant_source_bound.is_fully_source_bound is True
    assert constant_source_bound.require_source_bound_records() == (
        constant_bulk_source,
        constant_response_source,
    )
    np.testing.assert_allclose(constant_source_bound.dielectric_spectrum, 1.8)
    with pytest.raises(ValueError, match="optical dielectric"):
        Route2V0LorentzNonlocalDielectricSpectrum(
            grid=grid,
            static_dielectric_constant=1.8,
            optical_dielectric_constant=2.0,
            correlation_length_bohr=None,
        )
    with pytest.raises(ValueError, match="static dielectric"):
        Route2V0LorentzNonlocalDielectricSpectrum(
            grid=grid,
            static_dielectric_constant=1.0,
            optical_dielectric_constant=1.0,
            correlation_length_bohr=None,
        )


def test_nonlocal_dielectric_scalar_derivative_is_its_reaction_potential():
    rng = np.random.default_rng(20260729)
    operator = Route2V0NonlocalDielectricOperator(
        _grid(),
        _reciprocal_even_dielectric(_grid()),
    )
    density = _neutral_field(operator.grid, rng)
    direction = _neutral_field(operator.grid, rng)
    epsilon = 1.0e-5

    finite_difference = (
        operator.polarization_energy_hartree(density + epsilon * direction)
        - operator.polarization_energy_hartree(density - epsilon * direction)
    ) / (2.0 * epsilon)
    analytic = operator.grid.volume_element_bohr3 * np.sum(
        operator.reaction_potential_hartree_per_e(density) * direction
    )

    assert finite_difference == pytest.approx(analytic, rel=2.0e-9, abs=2.0e-11)


def test_nonlocal_dielectric_pairing_is_reciprocal_passive_and_translation_covariant():
    rng = np.random.default_rng(41)
    grid = _grid()
    operator = Route2V0NonlocalDielectricOperator(
        grid,
        _reciprocal_even_dielectric(grid),
    )
    left = _neutral_field(grid, rng)
    right = _neutral_field(grid, rng)

    assert operator.reaction_pairing_hartree(left, right) == pytest.approx(
        operator.reaction_pairing_hartree(right, left),
        rel=2.0e-13,
        abs=2.0e-13,
    )
    assert operator.polarization_energy_hartree(left) <= 1.0e-14
    assert operator.polarization_energy_hartree(right) <= 1.0e-14

    shift = (3, -2, 1)
    shifted = np.roll(right, shift=shift, axis=(0, 1, 2))
    np.testing.assert_allclose(
        operator.reaction_potential_hartree_per_e(shifted),
        np.roll(
            operator.reaction_potential_hartree_per_e(right),
            shift=shift,
            axis=(0, 1, 2),
        ),
        rtol=0.0,
        atol=3.0e-15,
    )
    assert operator.polarization_energy_hartree(shifted) == pytest.approx(
        operator.polarization_energy_hartree(right),
        rel=2.0e-13,
        abs=2.0e-13,
    )


def test_nonlocal_dielectric_rejects_invalid_spectrum_or_non_neutral_source():
    grid = _grid()
    dielectric = _reciprocal_even_dielectric(grid)

    with pytest.raises(ValueError, match="shape"):
        Route2V0NonlocalDielectricOperator(grid, np.array(78.4))
    with pytest.raises(ValueError, match="epsilon"):
        Route2V0NonlocalDielectricOperator(grid, np.ones(grid.shape) - 1.0e-6)
    with pytest.raises(ValueError, match="real-valued"):
        Route2V0NonlocalDielectricOperator(grid, dielectric.astype(complex))
    with pytest.raises(ValueError, match="reciprocal-even"):
        non_even = dielectric.copy()
        non_even[1, 0, 0] += 0.01
        Route2V0NonlocalDielectricOperator(grid, non_even)
    with pytest.raises(ValueError, match="construction"):
        Route2V0NonlocalDielectricOperator(
            grid,
            dielectric,
            construction="not-route2-v0",
        )
    with pytest.raises(ValueError, match="response scope"):
        Route2V0NonlocalDielectricOperator(
            grid,
            dielectric,
            response_scope="total-solvation",
        )
    with pytest.raises(ValueError, match="reciprocity tolerance"):
        Route2V0NonlocalDielectricOperator(
            grid,
            dielectric,
            reciprocity_relative_tolerance=0.0,
        )

    operator = Route2V0NonlocalDielectricOperator(grid, dielectric)
    non_neutral = np.zeros(grid.shape)
    non_neutral[0, 0, 0] = 1.0
    with pytest.raises(ValueError, match="neutral"):
        operator.reaction_potential_hartree_per_e(non_neutral)
    with pytest.raises(ValueError, match="neutral"):
        operator.polarization_energy_hartree(non_neutral)
    with pytest.raises(ValueError, match="neutral"):
        operator.reaction_pairing_hartree(np.zeros(grid.shape), non_neutral)
    with pytest.raises(ValueError, match="tolerance"):
        operator.reaction_potential_hartree_per_e(
            np.zeros(grid.shape),
            neutrality_relative_tolerance=0.0,
        )
