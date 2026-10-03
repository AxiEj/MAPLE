from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

BENCHMARK_DIR = (
    Path(__file__).resolve().parents[2] / "docs/implicit-solvation/benchmarks"
)
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

import cha_gaussian_second_derivative_validation as validation  # pyright: ignore[reportMissingImports]

POSITIONS = np.array(
    [[0.01, 0.39, 0.0], [0.77, -0.19, 0.0], [-0.78, -0.15, 0.0]],
    dtype=np.float64,
)
FORCE_STEPS = [5e-5, 2e-5, 1e-5]
CURVATURE_STEPS = [1e-4, 5e-5, 2e-5]
PARAMETERS = {
    "charges_e": [-0.8, 0.4, 0.4],
    "cha_radii_angstrom": [1.88, 1.04, 1.04],
    "lj_rmin_angstrom": [1.7683, 1.2, 1.2],
    "lj_epsilon_kcal_mol": [0.152, 0.02, 0.02],
}


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _hash_arrays(label, *arrays, extras=()):
    digest = hashlib.sha256(label.encode("ascii"))
    for array in arrays:
        contiguous = np.ascontiguousarray(array, dtype="<f8")
        digest.update(str(contiguous.shape).encode("ascii"))
        digest.update(contiguous.tobytes())
    for extra in extras:
        digest.update(repr(extra).encode("ascii"))
    return digest.hexdigest()


def _metadata(positions, order):
    positions = np.asarray(positions, dtype=np.float64)
    charges = np.asarray(PARAMETERS["charges_e"])
    radii = np.asarray(PARAMETERS["cha_radii_angstrom"])
    rmin = np.asarray(PARAMETERS["lj_rmin_angstrom"])
    epsilon = np.asarray(PARAMETERS["lj_epsilon_kcal_mol"])
    inverse_cube = np.asarray([0.4, 0.5, 0.6])
    inverse_born = np.cbrt(inverse_cube)
    effective = charges.copy()
    level = {
        "azimuth_order": order,
        "inverse_cube_per_angstrom3": inverse_cube.tolist(),
        "inverse_born_per_angstrom": inverse_born.tolist(),
        "gauss_closure_vector_angstrom2": [0.0, 0.0, 0.0],
        "inverse_cube_quad_error_estimate_per_angstrom3": 0.0,
        "meridian_evaluations": order,
    }
    constants = tuple(_protocol()["reference_constants"].values())
    coordinates_sha = _hash_arrays("gaussian-cha-coordinates-v1", positions)
    parameters_sha = _hash_arrays(
        "gaussian-cha-parameters-v1",
        charges,
        radii,
        rmin,
        epsilon,
        extras=((order,), 1e-12, 1e-12, *constants),
    )
    identity_sha = hashlib.sha256(
        f"gaussian-cha-geometry-v1:{coordinates_sha}:{parameters_sha}".encode("ascii")
    ).hexdigest()
    level_metadata = ((order, 0.0, order),)
    payload_sha = _hash_arrays(
        "gaussian-cha-prepared-payload-v1",
        positions,
        charges,
        radii,
        rmin,
        epsilon,
        inverse_cube,
        inverse_born,
        effective,
        inverse_cube,
        inverse_born,
        np.zeros(3),
        extras=(
            1.0,
            1.0,
            0.1,
            -0.1,
            coordinates_sha,
            parameters_sha,
            identity_sha,
            level_metadata,
        ),
    )
    value = {
        "positions_angstrom": positions.tolist(),
        "charges_e": charges.tolist(),
        "cha_radii_angstrom": radii.tolist(),
        "lj_rmin_angstrom": rmin.tolist(),
        "lj_epsilon_kcal_mol": epsilon.tolist(),
        "inverse_cube_per_angstrom3": inverse_cube.tolist(),
        "inverse_born_per_angstrom": inverse_born.tolist(),
        "effective_charges_e": effective.tolist(),
        "electrostatic_size_angstrom": 1.0,
        "cavity_volume_angstrom3": 1.0,
        "cavity_kcal_mol": 0.1,
        "dispersion_kcal_mol": -0.1,
        "r6_levels": [level],
        "r6_diagnostics": {"regular_pair_count": 1},
        "nonpolar_diagnostics": {
            "azimuth_orders": [order],
            "epsabs": 1e-12,
            "epsrel": 1e-12,
            "cavity_mixed_vector_quad_error": [0.0, 0.0, 0.0],
            "dispersion_mixed_vector_quad_error": [0.0, 0.0, 0.0],
            "mixed_vector_quad_errors_are_scalar_uncertainties": False,
            "error_estimates_are_rigorous_bounds": False,
        },
        "epsabs": 1e-12,
        "epsrel": 1e-12,
        "coordinates_sha256": coordinates_sha,
        "parameters_sha256": parameters_sha,
        "identity_sha256": identity_sha,
        "payload_sha256": payload_sha,
    }
    value["content_sha256"] = hashlib.sha256(_canonical(value)).hexdigest()
    return value


@lru_cache(maxsize=None)
def _cached_metadata(position_bytes, order):
    positions = np.frombuffer(position_bytes, dtype="<f8").reshape(3, 3)
    return _metadata(positions, order)


def _metadata_cached(positions, order):
    raw = np.asarray(positions, dtype="<f8").tobytes()
    return deepcopy(_cached_metadata(raw, order))


def _protocol():
    return {
        "profile_id": "gaussian-cha-r6-derivative-v2-numerical-profile-20261001.2-direct-complement",
        "topology_sha256": "t" * 64,
        "checkpoint_sha256": "c" * 64,
        "production_order": 64,
        "sigma_e": [0.001, 0.003, 0.01],
        "force_fd_steps_angstrom": FORCE_STEPS,
        "curvature_steps_angstrom": CURVATURE_STEPS,
        "independent_reference": {
            "order": 128,
            "epsabs": 1e-12,
            "epsrel": 1e-12,
        },
        "reference_parameters": PARAMETERS,
        "reference_constants": {
            "AMBER_CHARGE_SCALE": 18.2223,
            "ALPB_ALPHA": 0.571412,
            "CHA_TAU": 1.47,
            "CHA_ROH_ANGSTROM": 0.586,
            "EFFECTIVE_PROBE_ANGSTROM": float(1.4 - 0.52),
            "GAUSSIAN_SIZE_CAP_ANGSTROM": 9.5,
            "CAVITY_PROBE_ANGSTROM": 1.3,
            "CAVITY_COEFFICIENT_KCAL_MOL_ANGSTROM3": 0.0378,
            "CAVITY_OFFSET_KCAL_MOL": -0.5692,
            "DISPERSION_PROBE_ANGSTROM": 0.557,
            "WATER_OXYGEN_RMIN_ANGSTROM": 1.7683,
            "WATER_OXYGEN_EPSILON_KCAL_MOL": 0.152,
            "WATER_DENSITY_PER_ANGSTROM3": 0.03333 * 1.129,
        },
        "gates": {
            "hessian_symmetry_kcal_mol_per_angstrom2": 1e-8,
            "translation_block_sum_kcal_mol_per_angstrom2": 1e-8,
            "ward_relative": 1e-8,
            "hvp_dense_kcal_mol_per_angstrom2": 1e-8,
            "force_fd_max_kcal_mol_per_angstrom2": 5e-3,
            "force_fd_rms_kcal_mol_per_angstrom2": 1e-3,
            "force_fd_disagreement_kcal_mol_per_angstrom2": 5e-3,
            "independent_curvature_kcal_mol_per_angstrom2": 5e-3,
            "independent_disagreement_kcal_mol_per_angstrom2": 5e-3,
            "identity_relative": 1e-12,
        },
        "graph_budgets": {"per_cell_total": 256},
        "budgets": {
            "maximum": {
                "gas_graphs": 256,
                "solvent_graphs": 256,
                "independent_energy_calls": 40,
            },
            "nominal": {
                "gas_graphs": 30,
                "solvent_graphs": 60,
                "independent_energy_calls": 40,
            },
            "operations": [
                {
                    "operation_id": "gas-solvent",
                    "gas_graphs": 30,
                    "solvent_graphs": 60,
                    "independent_energy_calls": 0,
                },
                {
                    "operation_id": "independent",
                    "gas_graphs": 0,
                    "solvent_graphs": 0,
                    "independent_energy_calls": 40,
                },
            ],
        },
        "graph_seams": {
            "gas": {"module": "gas", "qualname": "build", "source_sha256": "g" * 64},
            "solvent": {
                "module": "solvent",
                "qualname": "scalar",
                "source_sha256": "s" * 64,
            },
            "independent": {
                "module": "oracle",
                "qualname": "prepare",
                "source_sha256": "o" * 64,
            },
        },
        "core_api_counter_expectations": {
            "calculator": {"direct_hvp": 18, "solvent_total": 38},
            "correction": {
                "dense_hessian": 2,
                "directional_hvp": 36,
                "total": 38,
            },
        },
        "source_identity_sha256": "s" * 64,
        "input_identity_sha256": "i" * 64,
        "covariance_transforms": [
            {
                "transform_id": "z-37deg-shift",
                "Q": [
                    [0.7986355100472928, -0.6018150231520483, 0.0],
                    [0.6018150231520483, 0.7986355100472928, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                "a": [0.3, -0.2, 0.1],
            },
            {
                "transform_id": "x-23deg-shift",
                "Q": [
                    [1.0, 0.0, 0.0],
                    [0.0, 0.9205048534524404, -0.39073112848927377],
                    [0.0, 0.39073112848927377, 0.9205048534524404],
                ],
                "a": [-0.17, 0.23, 0.31],
            },
        ],
    }


def _prereg_row():
    generated = validation.build_direction_inventory(POSITIONS, seed=2026100200)[-3:]
    return {
        "row_id": "center-00-sigma-00",
        "center_index": 0,
        "center_scale": 0.96,
        "sigma_e": 0.001,
        "terminal_positions_angstrom": POSITIONS.tolist(),
        "terminal_coordinate_sha256": validation.coordinate_sha256(POSITIONS),
        "direction_seed": 2026100200,
        "nonrigid_directions": [
            {
                "direction_id": item["direction_id"],
                "values": item["direction"],
            }
            for item in generated
        ],
    }


def _term(direction_inventory):
    zero9 = np.zeros(9).tolist()
    zero_h = np.zeros((9, 9)).tolist()
    return {
        "energy_kcal_mol": 0.0,
        "gradient_kcal_mol_per_angstrom": zero9,
        "hessian_kcal_mol_per_angstrom2": zero_h,
        "directional_hvp": [
            {
                "direction_id": item["direction_id"],
                "direction": item["direction"],
                "hvp_kcal_mol_per_angstrom2": zero9,
            }
            for item in direction_inventory
        ],
    }


def _center_term(direction_inventory):
    zero9 = np.zeros(9).tolist()
    return {
        "energy_hartree": 0.0,
        "gradient_hartree_per_angstrom": zero9,
        "forces_hartree_per_angstrom": np.zeros((3, 3)).tolist(),
        "hessian_hartree_per_angstrom2": np.zeros((9, 9)).tolist(),
        "directional_hvp": [
            {
                "direction_id": item["direction_id"],
                "direction": item["direction"],
                "dense_hessian_product_hartree_per_angstrom2": zero9,
                "direct_hvp_hartree_per_angstrom2": zero9,
            }
            for item in direction_inventory
        ],
    }


def _valid_receipt():
    prereg = _prereg_row()
    directions = validation.build_direction_inventory(
        POSITIONS,
        seed=prereg["direction_seed"],
        nonrigid_directions=prereg["nonrigid_directions"],
    )
    terms = {name: _term(directions) for name in ("polar", "cavity", "dispersion")}
    terms["total"] = _term(directions)
    center_terms = {
        name: _center_term(directions) for name in ("solvent", "gas", "composed")
    }
    force_validation = []
    for flat in range(9):
        direction = np.eye(9)[flat]
        force_validation.append(
            {
                "dof": flat,
                "direction": direction.tolist(),
                "steps": [
                    {
                        "step_angstrom": step,
                        "plus_positions_angstrom": (
                            POSITIONS.reshape(-1) + step * direction
                        )
                        .reshape(3, 3)
                        .tolist(),
                        "plus_coordinate_sha256": validation.coordinate_sha256(
                            (POSITIONS.reshape(-1) + step * direction).reshape(3, 3)
                        ),
                        "plus_composed_forces_hartree_per_angstrom": np.zeros(
                            (3, 3)
                        ).tolist(),
                        "minus_positions_angstrom": (
                            POSITIONS.reshape(-1) - step * direction
                        )
                        .reshape(3, 3)
                        .tolist(),
                        "minus_coordinate_sha256": validation.coordinate_sha256(
                            (POSITIONS.reshape(-1) - step * direction).reshape(3, 3)
                        ),
                        "minus_composed_forces_hartree_per_angstrom": np.zeros(
                            (3, 3)
                        ).tolist(),
                    }
                    for step in FORCE_STEPS
                ],
            }
        )
    independent = []
    for item in directions[-3:]:
        independent.append(
            {
                "direction_id": item["direction_id"],
                "direction": item["direction"],
                "steps": [
                    {
                        "step_angstrom": step,
                        "reference_order": 128,
                        "epsabs": 1e-12,
                        "epsrel": 1e-12,
                        "samples": [
                            {
                                "multiplier": multiplier,
                                "positions_angstrom": (
                                    POSITIONS.reshape(-1)
                                    + multiplier * step * np.asarray(item["direction"])
                                )
                                .reshape(3, 3)
                                .tolist(),
                                "coordinate_sha256": validation.coordinate_sha256(
                                    (
                                        POSITIONS.reshape(-1)
                                        + multiplier
                                        * step
                                        * np.asarray(item["direction"])
                                    ).reshape(3, 3)
                                ),
                                "energy_kcal_mol": 0.0,
                                "quadrature_error_estimate": 0.0,
                                "branch_metadata": {"domain": "admitted"},
                                "reference_geometry_metadata": _metadata_cached(
                                    (
                                        POSITIONS.reshape(-1)
                                        + multiplier
                                        * step
                                        * np.asarray(item["direction"])
                                    ).reshape(3, 3),
                                    128,
                                ),
                            }
                            for multiplier in (-2, -1, 0, 1, 2)
                        ],
                    }
                    for step in CURVATURE_STEPS
                ],
            }
        )
    covariance = []
    for transform in _protocol()["covariance_transforms"]:
        q = np.asarray(transform["Q"])
        shift = np.asarray(transform["a"])
        transformed = POSITIONS @ q.T + shift
        for name in ("solvent", "gas", "composed"):
            covariance.append(
                {
                    "term": name,
                    "transform_id": transform["transform_id"],
                    "Q": transform["Q"],
                    "a": transform["a"],
                    "transformed_positions_angstrom": transformed.tolist(),
                    "transformed_coordinate_sha256": validation.coordinate_sha256(
                        transformed
                    ),
                    "base": deepcopy(center_terms[name]),
                    "transformed": deepcopy(center_terms[name]),
                }
            )
    return {
        "case_identity": prereg,
        "preregistration_sha256": "p" * 64,
        "protocol_sha256": "q" * 64,
        "source_identity_sha256": "s" * 64,
        "input_identity_sha256": "i" * 64,
        "profile_id": _protocol()["profile_id"],
        "topology_sha256": "t" * 64,
        "checkpoint_sha256": "c" * 64,
        "production_order": 64,
        "analytic_identity_fingerprint": [
            "chagb-r6-pbsa-gaussian-sign-v1",
            "gaussian-cha-r6-derivative-v2-numerical-profile-20261001.2-direct-complement",
            "t" * 64,
            0.001,
            64,
            "cpu",
            "torch.float64",
        ],
        "analytic_identity_provenance": {
            "model_identity": "chagb-r6-pbsa-gaussian-sign-v1",
            "numerical_profile_id": "gaussian-cha-r6-derivative-v2-numerical-profile-20261001.2-direct-complement",
            "topology_sha256": "t" * 64,
            "sigma_e": 0.001,
            "order": 64,
            "device": "cpu",
            "dtype": "torch.float64",
        },
        "units": {
            "center_energy": "hartree",
            "center_force": "hartree/angstrom",
            "center_hessian": "hartree/angstrom^2",
            "solvent_terms": "kcal/mol,angstrom",
            "direction": "dimensionless",
        },
        "direction_inventory": directions,
        "solvent_terms": terms,
        "analytic_center": center_terms,
        "direct_hvp_records": [
            {
                "direction_id": item["direction_id"],
                "direction": item["direction"],
                "composed_hvp_hartree_per_angstrom2": np.zeros(9).tolist(),
                "tuple_forces_hartree_per_angstrom": np.zeros(9).tolist(),
                "tuple_energy_hartree": 0.0,
            }
            for item in directions
        ],
        "force_validation": force_validation,
        "independent_curvature": independent,
        "reference_center_diagnostics": [
            {
                "order": order,
                "total_energy_kcal_mol": 0.0,
                "epsabs": 1e-12,
                "epsrel": 1e-12,
                "quadrature_error_estimates": [0.0],
                "branch_metadata": {"domain": "admitted"},
                "coordinate_sha256": validation.coordinate_sha256(POSITIONS),
                "reference_geometry_metadata": _metadata_cached(POSITIONS, order),
            }
            for order in (64, 96, 128)
        ],
        "covariance": covariance,
        "fresh_identity": {
            "evaluation_count": 1,
            "fresh_gas_energy_hartree": 0.0,
            "fresh_solvent_energy_hartree": 0.0,
            "fresh_composed_energy_hartree": 0.0,
            "fresh_gas_forces_hartree_per_angstrom": np.zeros((3, 3)).tolist(),
            "fresh_solvent_forces_hartree_per_angstrom": np.zeros((3, 3)).tolist(),
            "fresh_composed_forces_hartree_per_angstrom": np.zeros((3, 3)).tolist(),
            "hvp_tuple_energy_hartree": 0.0,
            "hvp_tuple_forces_hartree_per_angstrom": np.zeros(9).tolist(),
        },
        "counters": {
            "gas_graphs": 30,
            "solvent_graphs": 60,
            "independent_energy_calls": 40,
            "hvp_calls": 54,
            "operations": deepcopy(_protocol()["budgets"]["operations"]),
            "graph_seams": deepcopy(_protocol()["graph_seams"]),
            "core_api_counters": deepcopy(_protocol()["core_api_counter_expectations"]),
        },
        "exceptions": [],
    }


def test_five_point_energy_curvature_is_independently_correct():
    direction = np.array([0.6, -0.8])
    hessian = np.array([[2.0, 0.3], [0.3, 1.5]])
    expected = float(direction @ hessian @ direction)
    for step in CURVATURE_STEPS:
        energies = []
        for multiplier in (-2, -1, 0, 1, 2):
            displacement = multiplier * step * direction
            energies.append(0.5 * float(displacement @ hessian @ displacement))
        assert validation.five_point_curvature(energies, step) == pytest.approx(
            expected, abs=1e-11
        )


def test_force_difference_hvp_uses_explicit_negative_sign():
    hessian = np.diag(np.arange(1.0, 10.0))
    direction = np.ones(9) / 3.0
    step = 2e-5
    plus_force = -(hessian @ (step * direction)).reshape(3, 3)
    minus_force = -(hessian @ (-step * direction)).reshape(3, 3)

    observed = validation.force_difference_hvp(plus_force, minus_force, step)

    assert observed == pytest.approx(hessian @ direction, abs=1e-12)


def test_valid_zero_curvature_stub_passes_all_rederived_gates():
    result = validation.validate_freq_ts_row(
        _valid_receipt(), _protocol(), _prereg_row()
    )

    assert result["status"] == "SECOND_DERIVATIVE_VALIDATED"
    assert result["passed"] is True
    assert result["direction_count"] == 18
    assert result["independent_direction_count"] == 3


@pytest.mark.parametrize(
    "mutation,match",
    [
        (
            lambda receipt: receipt["solvent_terms"]["total"][
                "hessian_kcal_mol_per_angstrom2"
            ][0].__setitem__(0, float("nan")),
            "finite",
        ),
        (
            lambda receipt: receipt["direction_inventory"].pop(),
            "direction inventory",
        ),
        (
            lambda receipt: receipt["force_validation"][0]["steps"].pop(),
            "force-validation steps",
        ),
        (
            lambda receipt: receipt["independent_curvature"][0]["steps"][0].update(
                reference_order=96
            ),
            "reference order",
        ),
        (
            lambda receipt: receipt["independent_curvature"][0]["steps"][0][
                "samples"
            ].pop(),
            "five energies",
        ),
    ],
)
def test_corrupt_or_incomplete_raw_evidence_is_unresolved(mutation, match):
    receipt = _valid_receipt()
    mutation(receipt)

    result = validation.validate_freq_ts_row(receipt, _protocol(), _prereg_row())

    assert result["status"] == "SECOND_DERIVATIVE_UNRESOLVED"
    assert match in result["reason"]


def test_component_hessian_closure_is_rederived_from_raw():
    receipt = _valid_receipt()
    receipt["solvent_terms"]["polar"]["hessian_kcal_mol_per_angstrom2"][0][0] = 1e-4

    result = validation.validate_freq_ts_row(receipt, _protocol(), _prereg_row())

    assert result["passed"] is False
    assert result["component_closure_passed"] is False


def test_direct_hvp_is_compared_to_dense_hessian_for_every_direction():
    receipt = _valid_receipt()
    receipt["solvent_terms"]["total"]["directional_hvp"][0][
        "hvp_kcal_mol_per_angstrom2"
    ][0] = 1e-4

    result = validation.validate_freq_ts_row(receipt, _protocol(), _prereg_row())

    assert result["passed"] is False
    assert result["solvent_term_metrics"]["total"]["hvp_dense_passed"] is False


def test_center_directional_schema_rejects_old_ambiguous_alias():
    receipt = _valid_receipt()
    record = receipt["analytic_center"]["composed"]["directional_hvp"][0]
    record["hvp_hartree_per_angstrom2"] = record["direct_hvp_hartree_per_angstrom2"]
    result = validation.validate_freq_ts_row(receipt, _protocol(), _prereg_row())
    assert result["status"] == "SECOND_DERIVATIVE_UNRESOLVED"
    assert "directional HVP schema" in result["reason"]


def test_all_18_direct_hvp_tuple_records_are_revalidated():
    receipt = _valid_receipt()
    receipt["direct_hvp_records"][10]["tuple_energy_hartree"] = 1e-3
    result = validation.validate_freq_ts_row(receipt, _protocol(), _prereg_row())
    assert result["passed"] is False
    assert "direct HVP tuple" in result["reason"]


@pytest.mark.parametrize(
    "mutation,match",
    [
        (
            lambda receipt: receipt["counters"]["operations"][0].update(gas_graphs=29),
            "operation inventory",
        ),
        (
            lambda receipt: receipt["counters"]["operations"][0].update(
                operation_id="renamed"
            ),
            "operation inventory",
        ),
        (
            lambda receipt: receipt["counters"]["graph_seams"]["gas"].update(
                source_sha256="x" * 64
            ),
            "graph seam",
        ),
    ],
)
def test_graph_operations_and_primitive_seams_are_exactly_bound(mutation, match):
    receipt = _valid_receipt()
    mutation(receipt)
    result = validation.validate_freq_ts_row(receipt, _protocol(), _prereg_row())
    assert result["status"] == "SECOND_DERIVATIVE_UNRESOLVED"
    assert match in result["reason"]


@pytest.mark.parametrize(
    "mutation,match",
    [
        (
            lambda metadata: metadata["positions_angstrom"][0].__setitem__(0, 9.0),
            "geometry coordinates",
        ),
        (
            lambda metadata: metadata["charges_e"].__setitem__(0, -0.7),
            "parameter",
        ),
        (
            lambda metadata: metadata["nonpolar_diagnostics"].update(
                error_estimates_are_rigorous_bounds=True
            ),
            "nonrigorous",
        ),
        (
            lambda metadata: metadata["r6_levels"][0].update(
                inverse_cube_quad_error_estimate_per_angstrom3=float("nan")
            ),
            "finite nonnegative",
        ),
    ],
)
def test_reference_geometry_metadata_is_rehashed_and_semantically_checked(
    mutation, match
):
    receipt = _valid_receipt()
    metadata = receipt["reference_center_diagnostics"][0]["reference_geometry_metadata"]
    mutation(metadata)
    metadata["content_sha256"] = hashlib.sha256(
        _canonical(
            {key: value for key, value in metadata.items() if key != "content_sha256"}
        )
    ).hexdigest()
    result = validation.validate_freq_ts_row(receipt, _protocol(), _prereg_row())
    assert result["status"] == "SECOND_DERIVATIVE_UNRESOLVED"
    assert match in result["reason"]


def test_covariance_base_and_transformed_coordinate_hash_bind_canonical_center():
    receipt = _valid_receipt()
    receipt["covariance"][0]["base"]["energy_hartree"] = 1e-3
    result = validation.validate_freq_ts_row(receipt, _protocol(), _prereg_row())
    assert result["status"] == "SECOND_DERIVATIVE_UNRESOLVED"
    assert "covariance embedded base" in result["reason"]

    receipt = _valid_receipt()
    receipt["covariance"][0]["transformed_coordinate_sha256"] = "x" * 64
    result = validation.validate_freq_ts_row(receipt, _protocol(), _prereg_row())
    assert result["status"] == "SECOND_DERIVATIVE_UNRESOLVED"
    assert "transformed coordinate" in result["reason"]


def test_every_independent_sample_metadata_binds_its_displaced_coordinates():
    receipt = _valid_receipt()
    metadata = receipt["independent_curvature"][1]["steps"][1]["samples"][3][
        "reference_geometry_metadata"
    ]
    metadata["coordinates_sha256"] = "x" * 64
    metadata["content_sha256"] = hashlib.sha256(
        _canonical(
            {key: value for key, value in metadata.items() if key != "content_sha256"}
        )
    ).hexdigest()
    result = validation.validate_freq_ts_row(receipt, _protocol(), _prereg_row())
    assert result["status"] == "SECOND_DERIVATIVE_UNRESOLVED"
    assert "reference geometry hash binding" in result["reason"]


def test_rotational_ward_identity_uses_h_omega_r_equals_omega_g():
    receipt = _valid_receipt()
    receipt["solvent_terms"]["total"]["gradient_kcal_mol_per_angstrom"][0] = 1.0

    result = validation.validate_freq_ts_row(receipt, _protocol(), _prereg_row())

    assert result["passed"] is False
    assert result["solvent_term_metrics"]["total"]["rotational_ward_passed"] is False


def test_covariance_rejects_improper_or_coordinate_mismatched_transform():
    receipt = _valid_receipt()
    receipt["covariance"][0]["Q"][0][0] = -1.0

    result = validation.validate_freq_ts_row(receipt, _protocol(), _prereg_row())

    assert result["status"] == "SECOND_DERIVATIVE_UNRESOLVED"
    assert "covariance" in result["reason"]


def test_fresh_tuple_and_graph_budgets_are_rederived():
    receipt = _valid_receipt()
    receipt["fresh_identity"]["evaluation_count"] = 2
    receipt["counters"]["solvent_graphs"] = 257

    result = validation.validate_freq_ts_row(receipt, _protocol(), _prereg_row())

    assert result["passed"] is False
    assert result["fresh_identity_passed"] is False
    assert result["budget_passed"] is False


def test_independent_curvature_noise_has_specific_unresolved_status():
    receipt = _valid_receipt()
    step = receipt["independent_curvature"][0]["steps"][-1]["step_angstrom"]
    for sample in receipt["independent_curvature"][0]["steps"][-1]["samples"]:
        sample["energy_kcal_mol"] = 0.5 * (sample["multiplier"] * step) ** 2 * 0.02

    result = validation.validate_freq_ts_row(receipt, _protocol(), _prereg_row())

    assert result["status"] == "INDEPENDENT_CURVATURE_UNRESOLVED"
    assert result["independent_curvature"]["passed"] is False


def test_source_identity_corruption_fails_closed():
    receipt = _valid_receipt()
    receipt["source_identity_sha256"] = "x" * 64

    result = validation.validate_freq_ts_row(receipt, _protocol(), _prereg_row())

    assert result["status"] == "SECOND_DERIVATIVE_UNRESOLVED"
    assert "source/input/profile" in result["reason"]


def test_exact_15_summary_retains_failed_denominator():
    rows = []
    expected = []
    for center in range(5):
        for width, sigma in enumerate((0.001, 0.003, 0.01)):
            row_id = f"center-{center:02d}-sigma-{width:02d}"
            expected.append({"row_id": row_id})
            rows.append(
                {
                    "row_id": row_id,
                    "status": "SECOND_DERIVATIVE_VALIDATED",
                    "passed": True,
                }
            )

    passed = validation.summarize_freq_ts_rows(rows, expected)
    rows[7]["passed"] = False
    rows[7]["status"] = "SECOND_DERIVATIVE_UNRESOLVED"
    failed = validation.summarize_freq_ts_rows(rows, expected)

    assert passed["status"] == "SECOND_DERIVATIVE_VALIDATED"
    assert failed["row_count"] == 15
    assert failed["passed_count"] == 14
    assert failed["status"] == "SECOND_DERIVATIVE_UNRESOLVED"


def test_live_protocol_keeps_force_and_independent_ladders_distinct_and_complete():
    protocol = json.loads(
        (BENCHMARK_DIR / "cha_gaussian_freq_ts_protocol.json").read_text()
    )

    assert protocol["approved_plan_sha256"] == (
        "2c872a6c3035b307098b7631434dfc8c88b6ace6bb91ca86d8361d1f15ade07c"
    )
    assert protocol["force_fd_steps_angstrom"] == [5e-5, 2e-5, 1e-5]
    assert protocol["curvature_steps_angstrom"] == [1e-4, 5e-5, 2e-5]
    assert protocol["reference_center_orders"] == [64, 96, 128]
    assert protocol["independent_reference"]["order"] == 128
    assert len(protocol["cases"]) == 15
    assert all(len(case["nonrigid_directions"]) == 3 for case in protocol["cases"])
    assert len(protocol["covariance_transforms"]) == 2
    operations = protocol["budgets"]["operations"]
    assert {
        key: sum(operation[key] for operation in operations)
        for key in ("gas_graphs", "solvent_graphs", "independent_energy_calls")
    } == protocol["budgets"]["nominal"]
