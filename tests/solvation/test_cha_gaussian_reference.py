from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError, replace
import importlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from scipy import special

BENCHMARK_DIR = (
    Path(__file__).resolve().parents[2] / "docs/implicit-solvation/benchmarks"
)
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

from cha_gaussian_reference import (  # pyright: ignore[reportMissingImports]
    GaussianChaSizeDomainError,
    gaussian_cha_from_reference_geometry,
    gaussian_cha_reference,
    prepare_gaussian_reference_geometry,
)
from cha_continuum_reference import (  # pyright: ignore[reportMissingImports]
    UnsupportedR6Geometry,
    cha_continuum_reference,
)


def _single_site_inputs(charge: float = -0.7):
    return (
        np.array([[0.0, 0.0, 0.0]]),
        np.array([charge]),
        np.array([1.5]),
        np.array([1.82]),
        np.array([0.093]),
    )


def _water_inputs():
    return (
        np.array([[0.011, 0.404, 0.0], [0.777, -0.223, 0.0], [-0.788, -0.181, 0.0]]),
        np.array([-0.784666666667, 0.392333333333, 0.392333333333]),
        np.array([1.88, 1.04, 1.04]),
        np.array([1.82, 0.3019, 0.3019]),
        np.array([0.093, 0.0047, 0.0047]),
    )


@pytest.fixture(scope="module")
def single_site_geometry():
    return prepare_gaussian_reference_geometry(*_single_site_inputs())


def test_zero_sigma_is_rejected(single_site_geometry):
    with pytest.raises(ValueError, match="positive finite"):
        gaussian_cha_from_reference_geometry(single_site_geometry, sigma_e=0.0)


def test_boolean_sigma_is_rejected(single_site_geometry):
    with pytest.raises(TypeError, match="positive real"):
        gaussian_cha_from_reference_geometry(single_site_geometry, sigma_e=True)


@pytest.mark.parametrize("sigma", [math.inf, -math.inf, math.nan])
def test_nonfinite_sigma_is_rejected(single_site_geometry, sigma):
    with pytest.raises(ValueError, match="positive finite"):
        gaussian_cha_from_reference_geometry(single_site_geometry, sigma_e=sigma)


def test_gaussian_sign_is_zero_at_zero_effective_charge():
    result = gaussian_cha_reference(*_single_site_inputs(0.0), sigma_e=0.01)

    assert result.smoothed_signs == pytest.approx([0.0], abs=0.0)


def test_gaussian_sign_is_odd_in_effective_charge():
    positive = gaussian_cha_reference(*_single_site_inputs(0.003), sigma_e=0.01)
    negative = gaussian_cha_reference(*_single_site_inputs(-0.003), sigma_e=0.01)

    assert positive.smoothed_signs == pytest.approx(-negative.smoothed_signs, abs=1e-15)


def test_gaussian_sign_approaches_unit_sign_in_the_far_tail(single_site_geometry):
    result = gaussian_cha_from_reference_geometry(single_site_geometry, sigma_e=1e-6)

    assert result.smoothed_signs == pytest.approx([-1.0], abs=0.0)
    assert result.gaussian_tail[0] < 1e-100


def test_gaussian_tail_is_evaluated_with_direct_scipy_erfc(single_site_geometry):
    result = gaussian_cha_from_reference_geometry(single_site_geometry, sigma_e=0.5)
    expected = special.erfc(np.abs(result.effective_charges_e) / (math.sqrt(2.0) * 0.5))

    assert result.gaussian_tail == pytest.approx(expected, rel=0.0, abs=0.0)
    source = (BENCHMARK_DIR / "cha_gaussian_reference.py").read_text()
    assert "special.erfc(" in source
    assert "1.0 - special.erf" not in source


def test_reference_source_has_no_direct_torch_or_gaussian_runtime_imports():
    tree = ast.parse((BENCHMARK_DIR / "cha_gaussian_reference.py").read_text())
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    }

    assert "torch" not in imports
    assert not any("torch_chagb_gaussian" in name for name in imports)
    assert not any("gaussian_cha_correction" in name for name in imports)


def test_clean_import_closure_discloses_incidental_torch_without_gaussian_helpers():
    code = """
import importlib, json, sys
before = set(sys.modules)
module = importlib.import_module('docs.implicit-solvation.benchmarks.cha_gaussian_reference')
loaded = sorted(set(sys.modules) - before)
print(json.dumps({'loaded': loaded}))
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    loaded = set(json.loads(completed.stdout)["loaded"])

    assert "torch" in loaded
    assert "maple.function.calculator.set_calculator" in loaded
    assert not loaded.intersection(
        {
            "maple.function.calculator.extra_correction.implicit.torch_chagb_gaussian",
            "maple.function.calculator.extra_correction.implicit.torch_continuum_chagb_gaussian",
            "maple.function.calculator.extra_correction.implicit.gaussian_cha_correction",
            "maple.function.calculator.extra_correction.implicit.torch_continuum_ses_geometry",
            "maple.function.calculator.extra_correction.implicit.torch_continuum_r6",
        }
    )


def test_reference_supports_runner_package_import_path():
    module = importlib.import_module(
        "docs.implicit-solvation.benchmarks.cha_gaussian_reference"
    )

    assert module.__file__ == str(BENCHMARK_DIR / "cha_gaussian_reference.py")


def test_prepared_geometry_owns_immutable_parameter_copies():
    inputs = list(_single_site_inputs())
    geometry = prepare_gaussian_reference_geometry(*inputs)
    inputs[0][0, 0] = 99.0
    inputs[1][0] = 99.0

    assert geometry.positions_angstrom[0, 0] == 0.0
    assert geometry.charges_e[0] == -0.7
    with pytest.raises(ValueError, match="read-only"):
        geometry.positions_angstrom[0, 0] = 1.0
    with pytest.raises(FrozenInstanceError):
        geometry.identity_sha256 = "changed"


def test_readonly_geometry_cannot_reenable_writes():
    geometry = prepare_gaussian_reference_geometry(*_single_site_inputs())

    with pytest.raises(ValueError):
        geometry.positions_angstrom.setflags(write=True)


def test_readonly_geometry_cannot_change_array_metadata_shape():
    geometry = prepare_gaussian_reference_geometry(*_single_site_inputs())

    with pytest.raises((AttributeError, ValueError)):
        geometry.positions_angstrom.shape = (3,)


def test_geometry_payload_integrity_rejects_changed_coordinates_with_stale_born():
    geometry = prepare_gaussian_reference_geometry(
        *_water_inputs(), azimuth_orders=(32,)
    )
    changed = geometry.positions_angstrom.copy()
    changed[1, 0] += 0.05
    fresh = prepare_gaussian_reference_geometry(
        changed, *_water_inputs()[1:], azimuth_orders=(32,)
    )
    forged = replace(geometry, positions_angstrom=changed)

    assert not np.array_equal(
        fresh.inverse_born_per_angstrom, geometry.inverse_born_per_angstrom
    )
    with pytest.raises(RuntimeError, match="payload integrity"):
        gaussian_cha_from_reference_geometry(forged, sigma_e=0.01)


def test_width_results_do_not_alias_prepared_geometry(single_site_geometry):
    result = gaussian_cha_from_reference_geometry(single_site_geometry, sigma_e=0.01)

    assert not np.shares_memory(
        result.inverse_born_per_angstrom,
        single_site_geometry.inverse_born_per_angstrom,
    )
    with pytest.raises(ValueError, match="read-only"):
        result.born_radii_angstrom[0] = 1.0


def test_geometry_identity_changes_when_coordinates_change():
    first = prepare_gaussian_reference_geometry(*_single_site_inputs())
    inputs = list(_single_site_inputs())
    inputs[0][0, 0] = 0.125
    second = prepare_gaussian_reference_geometry(*inputs)

    assert first.coordinates_sha256 != second.coordinates_sha256
    assert first.identity_sha256 != second.identity_sha256
    assert first.parameters_sha256 == second.parameters_sha256


def test_coordinate_change_recomputes_every_water_born_radius():
    inputs = list(_water_inputs())
    reference = prepare_gaussian_reference_geometry(*inputs, azimuth_orders=(32,))
    inputs[0] = inputs[0].copy()
    inputs[0][1, 0] += 1.0e-3
    moved = prepare_gaussian_reference_geometry(*inputs, azimuth_orders=(32,))

    difference = np.abs(
        moved.inverse_born_per_angstrom - reference.inverse_born_per_angstrom
    )
    assert np.all(difference > 1.0e-9)


def test_prepared_geometry_can_be_reused_across_widths():
    geometry = prepare_gaussian_reference_geometry(*_single_site_inputs(-0.003))
    narrow = gaussian_cha_from_reference_geometry(geometry, sigma_e=0.001)
    broad = gaussian_cha_from_reference_geometry(geometry, sigma_e=0.01)

    assert narrow.geometry_identity_sha256 == broad.geometry_identity_sha256
    assert narrow.cavity_kcal_mol == broad.cavity_kcal_mol
    assert narrow.dispersion_kcal_mol == broad.dispersion_kcal_mol
    assert narrow.smoothed_signs[0] < broad.smoothed_signs[0]


def test_smallest_positive_sigma_saturates_tail_without_nonfinite_energy():
    geometry = prepare_gaussian_reference_geometry(*_single_site_inputs())
    result = gaussian_cha_from_reference_geometry(
        geometry, sigma_e=np.nextafter(0.0, 1.0)
    )

    assert math.isfinite(result.total_kcal_mol)
    assert np.all(np.isfinite(result.smoothed_signs))
    assert np.all(np.isfinite(result.gaussian_tail))
    assert result.diagnostics["charge_sigma_tail_saturated"] is True


def test_diagnostics_truthfully_describe_shared_and_incidental_helpers(
    single_site_geometry,
):
    diagnostics = single_site_geometry.diagnostics

    assert diagnostics["shared_numpy_reference_helpers"] == (
        "cha_continuum_reference.r6_inverse_born_reference",
        "sphere_union_volume.volume_and_gradient",
        "sphere_union_dispersion.dispersion_energy_and_gradient",
    )
    assert diagnostics["torch_gaussian_runtime_helpers_used"] is False
    assert diagnostics["torch_ses_helpers_used"] is False
    assert (
        diagnostics["incidental_torch_import_via_maple_package_initialization"] is True
    )


def test_size_equal_to_cap_fails_closed():
    inputs = list(_single_site_inputs())
    inputs[2] = np.array([9.5])

    with pytest.raises(GaussianChaSizeDomainError, match=r"strictly below 9\.5"):
        prepare_gaussian_reference_geometry(*inputs)


def test_more_than_three_sites_retains_reference_domain_failure():
    with pytest.raises(UnsupportedR6Geometry, match="at most three"):
        prepare_gaussian_reference_geometry(
            [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
            [0.1, -0.1, 0.1, -0.1],
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
            [0.1, 0.1, 0.1, 0.1],
        )


def test_expanded_sphere_tangency_retains_reference_domain_failure():
    probe = float(1.4 - 0.52)
    with pytest.raises(UnsupportedR6Geometry, match="tangency"):
        prepare_gaussian_reference_geometry(
            [[0.0, 0.0, 0.0], [2.0 * (1.0 + probe), 0.0, 0.0]],
            [0.1, -0.1],
            [1.0, 1.0],
            [1.0, 1.0],
            [0.1, 0.1],
        )


def test_components_sum_to_total(single_site_geometry):
    result = gaussian_cha_from_reference_geometry(single_site_geometry, sigma_e=0.01)

    assert result.energies_kcal_mol["total"] == pytest.approx(
        result.energies_kcal_mol["polar"]
        + result.energies_kcal_mol["cavity"]
        + result.energies_kcal_mol["dispersion"],
        abs=2e-12,
    )


def test_reference_retains_all_r6_refinement_levels(single_site_geometry):
    result = gaussian_cha_from_reference_geometry(single_site_geometry, sigma_e=0.01)

    assert [level.azimuth_order for level in result.r6_levels] == [64, 96, 128]
    assert all(
        level.inverse_cube_quad_error_estimate_per_angstrom3 >= 0.0
        for level in result.r6_levels
    )


def test_near_sign_event_uses_new_functional_not_exact_sign_parity():
    inputs = _single_site_inputs(0.001)
    gaussian = gaussian_cha_reference(*inputs, sigma_e=0.01)
    exact = cha_continuum_reference(*inputs)

    assert gaussian.smoothed_signs[0] == pytest.approx(
        special.erf(0.001 / (math.sqrt(2.0) * 0.01)), rel=2e-15
    )
    assert gaussian.energies_kcal_mol["polar"] != pytest.approx(
        exact.polar_kcal_mol, rel=0.0, abs=1e-12
    )


def test_narrow_width_recovers_exact_sign_pair_algebra():
    inputs = _water_inputs()
    gaussian = gaussian_cha_reference(*inputs, sigma_e=1.0e-6, azimuth_orders=(32,))
    exact = cha_continuum_reference(*inputs, azimuth_orders=(32,))

    assert gaussian.polar_kcal_mol == pytest.approx(exact.polar_kcal_mol, abs=1e-12)
