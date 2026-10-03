"""Pure validators for Gaussian-CHA analytic second-derivative evidence.

This module implements no runtime derivative capability.  It rederives gates
from raw create-only workflow receipts and treats the independent NumPy/SciPy
reference strictly as an energy directional-curvature diagnostic.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any, Iterable

import numpy as np

KCAL_PER_HARTREE = 627.5094740631
EXPECTED_UNITS = {
    "center_energy": "hartree",
    "center_force": "hartree/angstrom",
    "center_hessian": "hartree/angstrom^2",
    "solvent_terms": "kcal/mol,angstrom",
    "direction": "dimensionless",
}
COMPONENTS = ("polar", "cavity", "dispersion")
CENTER_TERMS = ("solvent", "gas", "composed")
REFERENCE_CONSTANT_NAMES = (
    "AMBER_CHARGE_SCALE",
    "ALPB_ALPHA",
    "CHA_TAU",
    "CHA_ROH_ANGSTROM",
    "EFFECTIVE_PROBE_ANGSTROM",
    "GAUSSIAN_SIZE_CAP_ANGSTROM",
    "CAVITY_PROBE_ANGSTROM",
    "CAVITY_COEFFICIENT_KCAL_MOL_ANGSTROM3",
    "CAVITY_OFFSET_KCAL_MOL",
    "DISPERSION_PROBE_ANGSTROM",
    "WATER_OXYGEN_RMIN_ANGSTROM",
    "WATER_OXYGEN_EPSILON_KCAL_MOL",
    "WATER_DENSITY_PER_ANGSTROM3",
)


def coordinate_sha256(positions: Any) -> str:
    array = np.asarray(positions, dtype=np.float64)
    if array.shape != (3, 3) or not np.isfinite(array).all():
        raise ValueError("terminal positions must have finite shape (3,3)")
    return hashlib.sha256(
        array.astype("<f8", copy=False).tobytes(order="C")
    ).hexdigest()


def _hash_arrays(label: str, *arrays: np.ndarray, extras=()) -> str:
    digest = hashlib.sha256(label.encode("ascii"))
    for array in arrays:
        contiguous = np.ascontiguousarray(array, dtype="<f8")
        digest.update(str(contiguous.shape).encode("ascii"))
        digest.update(contiguous.tobytes())
    for extra in extras:
        digest.update(repr(extra).encode("ascii"))
    return digest.hexdigest()


def _metadata_content_sha256(metadata: dict[str, Any]) -> str:
    import json

    payload = {key: value for key, value in metadata.items() if key != "content_sha256"}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _validate_error_diagnostics(value: Any, *, path: str = "diagnostics") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {
                "error_estimates_are_rigorous_bounds",
                "mixed_vector_quad_errors_are_scalar_uncertainties",
            }:
                if item is not False:
                    raise ValueError(
                        "reference errors must retain nonrigorous semantics"
                    )
            elif "error" in key.lower() and isinstance(item, (int, float, list, tuple)):
                array = np.asarray(item, dtype=np.float64)
                if not np.isfinite(array).all() or np.any(array < 0.0):
                    raise ValueError("reference errors must be finite nonnegative")
            else:
                _validate_error_diagnostics(item, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_error_diagnostics(item, path=f"{path}[{index}]")


def _validate_reference_geometry_metadata(
    metadata: dict[str, Any],
    expected_positions: np.ndarray,
    expected_order: int,
    protocol: dict[str, Any],
) -> None:
    if metadata.get("content_sha256") != _metadata_content_sha256(metadata):
        raise ValueError("reference geometry metadata seal differs")
    positions = _array(
        "reference geometry coordinates", metadata["positions_angstrom"], (3, 3)
    )
    if not np.array_equal(positions, expected_positions):
        raise ValueError("reference geometry coordinates differ")
    reference = protocol["independent_reference"]
    parameters = protocol["reference_parameters"]
    charges = _array("reference charges", metadata["charges_e"], (3,))
    radii = _array("reference CHA radii", metadata["cha_radii_angstrom"], (3,))
    rmin = _array("reference LJ rmin", metadata["lj_rmin_angstrom"], (3,))
    epsilon = _array("reference LJ epsilon", metadata["lj_epsilon_kcal_mol"], (3,))
    for name, observed in (
        ("charges_e", charges),
        ("cha_radii_angstrom", radii),
        ("lj_rmin_angstrom", rmin),
        ("lj_epsilon_kcal_mol", epsilon),
    ):
        if not np.array_equal(observed, np.asarray(parameters[name], dtype=np.float64)):
            raise ValueError("reference parameter vectors differ")
    epsabs = float(reference["epsabs"])
    epsrel = float(reference["epsrel"])
    if float(metadata["epsabs"]) != epsabs or float(metadata["epsrel"]) != epsrel:
        raise ValueError("reference quadrature tolerances differ")
    levels = metadata["r6_levels"]
    if not isinstance(levels, list) or [
        level.get("azimuth_order") for level in levels
    ] != [expected_order]:
        raise ValueError("reference geometry order differs")
    inverse_cube = _array(
        "reference inverse cube", metadata["inverse_cube_per_angstrom3"], (3,)
    )
    inverse_born = _array(
        "reference inverse Born", metadata["inverse_born_per_angstrom"], (3,)
    )
    effective = _array(
        "reference effective charges", metadata["effective_charges_e"], (3,)
    )
    level_arrays = []
    level_metadata = []
    for level in levels:
        level_inverse_cube = _array(
            "reference level inverse cube",
            level["inverse_cube_per_angstrom3"],
            (3,),
        )
        level_inverse_born = _array(
            "reference level inverse Born",
            level["inverse_born_per_angstrom"],
            (3,),
        )
        closure = _array(
            "reference Gauss closure",
            level["gauss_closure_vector_angstrom2"],
            (3,),
        )
        error = float(level["inverse_cube_quad_error_estimate_per_angstrom3"])
        evaluations = int(level["meridian_evaluations"])
        if not math.isfinite(error) or error < 0.0 or evaluations <= 0:
            raise ValueError("reference errors must be finite nonnegative")
        level_arrays.extend((level_inverse_cube, level_inverse_born, closure))
        level_metadata.append((expected_order, error, evaluations))
    coordinates_sha = _hash_arrays("gaussian-cha-coordinates-v1", positions)
    constants = protocol["reference_constants"]
    constant_values = tuple(float(constants[name]) for name in REFERENCE_CONSTANT_NAMES)
    parameters_sha = _hash_arrays(
        "gaussian-cha-parameters-v1",
        charges,
        radii,
        rmin,
        epsilon,
        extras=((expected_order,), epsabs, epsrel, *constant_values),
    )
    identity_sha = hashlib.sha256(
        f"gaussian-cha-geometry-v1:{coordinates_sha}:{parameters_sha}".encode("ascii")
    ).hexdigest()
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
        *level_arrays,
        extras=(
            float(metadata["electrostatic_size_angstrom"]),
            float(metadata["cavity_volume_angstrom3"]),
            float(metadata["cavity_kcal_mol"]),
            float(metadata["dispersion_kcal_mol"]),
            coordinates_sha,
            parameters_sha,
            identity_sha,
            tuple(level_metadata),
        ),
    )
    if (
        metadata["coordinates_sha256"] != coordinates_sha
        or metadata["parameters_sha256"] != parameters_sha
        or metadata["identity_sha256"] != identity_sha
        or metadata["payload_sha256"] != payload_sha
    ):
        raise ValueError("reference geometry hash binding differs")
    _validate_error_diagnostics(metadata["r6_diagnostics"])
    _validate_error_diagnostics(metadata["nonpolar_diagnostics"])


def five_point_curvature(energies: Iterable[float], step: float) -> float:
    values = np.asarray(list(energies), dtype=np.float64)
    if values.shape != (5,) or not np.isfinite(values).all():
        raise ValueError("independent curvature requires exactly five finite energies")
    h = float(step)
    if not math.isfinite(h) or h <= 0.0:
        raise ValueError("curvature step must be positive and finite")
    em2, em1, e0, ep1, ep2 = values
    return float((-ep2 + 16 * ep1 - 30 * e0 + 16 * em1 - em2) / (12 * h**2))


def force_difference_hvp(plus_force: Any, minus_force: Any, step: float) -> np.ndarray:
    plus = np.asarray(plus_force, dtype=np.float64)
    minus = np.asarray(minus_force, dtype=np.float64)
    if plus.shape != (3, 3) or minus.shape != (3, 3):
        raise ValueError("force validation samples must have shape (3,3)")
    if not np.isfinite(plus).all() or not np.isfinite(minus).all():
        raise ValueError("force validation samples must be finite")
    h = float(step)
    if not math.isfinite(h) or h <= 0.0:
        raise ValueError("force validation step must be positive and finite")
    # F=-grad(E), hence H v = -(F(R+h v)-F(R-h v))/(2h).
    return (-(plus - minus) / (2.0 * h)).reshape(-1)


def _rotation_generator(axis: int) -> np.ndarray:
    generators = (
        np.array([[0.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]),
        np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]),
        np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
    )
    return generators[axis]


def build_direction_inventory(
    positions: Any,
    *,
    seed: int = 20261002,
    nonrigid_directions: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    coordinates = np.asarray(positions, dtype=np.float64)
    if coordinates.shape != (3, 3) or not np.isfinite(coordinates).all():
        raise ValueError("direction inventory requires finite positions (3,3)")
    result = [
        {"direction_id": f"basis-{index:02d}", "direction": np.eye(9)[index].tolist()}
        for index in range(9)
    ]
    rigid = []
    for axis in range(3):
        direction = np.zeros((3, 3))
        direction[:, axis] = 1.0
        direction = direction.reshape(-1) / np.linalg.norm(direction)
        rigid.append(direction)
        result.append(
            {"direction_id": f"translation-{axis}", "direction": direction.tolist()}
        )
    centered = coordinates - coordinates.mean(axis=0)
    for axis in range(3):
        omega = _rotation_generator(axis)
        direction = (centered @ omega.T).reshape(-1)
        norm = float(np.linalg.norm(direction))
        if norm <= 0.0:
            raise ValueError("rotation generator is degenerate")
        direction /= norm
        rigid.append(direction)
        result.append(
            {"direction_id": f"rotation-{axis}", "direction": direction.tolist()}
        )
    rigid_basis, _ = np.linalg.qr(np.stack(rigid, axis=1))
    if nonrigid_directions is None:
        rng = np.random.default_rng(seed)
        supplied = [
            {"direction_id": f"seed-nonrigid-{index}", "values": rng.normal(size=9)}
            for index in range(3)
        ]
    else:
        supplied = nonrigid_directions
    if len(supplied) != 3 or len({item.get("direction_id") for item in supplied}) != 3:
        raise ValueError("nonrigid direction inventory must contain three unique IDs")
    nonrigid = []
    for item in supplied:
        trial = _array("nonrigid direction", item["values"], (9,)).copy()
        if nonrigid_directions is None:
            trial -= rigid_basis @ (rigid_basis.T @ trial)
            for prior in nonrigid:
                trial -= prior * float(np.dot(prior, trial))
        norm = float(np.linalg.norm(trial))
        if abs(norm - 1.0) > 2e-14 and nonrigid_directions is not None:
            raise ValueError("frozen nonrigid direction must be unit normalized")
        if norm <= 1e-12:
            raise RuntimeError("seeded nonrigid direction is degenerate")
        trial /= norm
        rigid_overlap = float(np.max(np.abs(rigid_basis.T @ trial)))
        if rigid_overlap > 2e-12:
            raise ValueError("frozen nonrigid direction contains a rigid component")
        nonrigid.append(trial)
        result.append(
            {"direction_id": item["direction_id"], "direction": trial.tolist()}
        )
    return result


def _array(name: str, value: Any, shape: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must have finite shape {shape}")
    return array


def _relative_close(
    first: np.ndarray | float, second: np.ndarray | float, tolerance: float
) -> bool:
    left = np.asarray(first, dtype=np.float64)
    right = np.asarray(second, dtype=np.float64)
    scale = max(1.0, float(np.max(np.abs(left))), float(np.max(np.abs(right))))
    return bool(np.max(np.abs(left - right)) <= tolerance * scale)


def _direction_map(
    records: list[dict[str, Any]], expected: list[dict[str, Any]], key: str
):
    if [record.get("direction_id") for record in records] != [
        item["direction_id"] for item in expected
    ]:
        raise ValueError("direction inventory is incomplete or reordered")
    result = {}
    for record, item in zip(records, expected, strict=True):
        observed = _array("direction", record["direction"], (9,))
        target = _array("expected direction", item["direction"], (9,))
        if not np.allclose(observed, target, rtol=0.0, atol=2e-15):
            raise ValueError("direction inventory vectors differ")
        result[item["direction_id"]] = _array(key, record[key], (9,))
    return result


def _hessian_metrics(
    hessian: np.ndarray,
    gradient: np.ndarray,
    positions: np.ndarray,
    directions: list[dict[str, Any]],
    hvp: dict[str, np.ndarray],
    gates: dict[str, float],
) -> dict[str, Any]:
    symmetry = float(np.max(np.abs(hessian - hessian.T)))
    blocks = hessian.reshape(3, 3, 3, 3)
    block_sum = float(np.max(np.abs(blocks.sum(axis=2))))
    hscale = max(1.0, float(np.max(np.abs(hessian))))
    translation_errors = []
    for axis in range(3):
        direction = np.asarray(directions[9 + axis]["direction"])
        translation_errors.append(float(np.max(np.abs(hessian @ direction))))
    centered = positions - positions.mean(axis=0)
    rotation_errors = []
    for axis in range(3):
        omega = _rotation_generator(axis)
        raw = (centered @ omega.T).reshape(-1)
        norm = float(np.linalg.norm(raw))
        direction = raw / norm
        rhs = (gradient.reshape(3, 3) @ omega.T).reshape(-1) / norm
        error = hessian @ direction - rhs
        scale = max(hscale, float(np.max(np.abs(gradient))) / norm, 1.0)
        rotation_errors.append(float(np.max(np.abs(error))) / scale)
    hvp_error = max(
        float(
            np.max(
                np.abs(
                    hvp[item["direction_id"]] - hessian @ np.asarray(item["direction"])
                )
            )
        )
        for item in directions
    )
    return {
        "symmetry_max": symmetry,
        "translation_block_sum_max": block_sum,
        "translation_ward_max": max(translation_errors),
        "rotational_ward_relative_max": max(rotation_errors),
        "hvp_dense_max": hvp_error,
        "symmetry_passed": symmetry <= gates["hessian_symmetry_kcal_mol_per_angstrom2"],
        "translation_block_sum_passed": block_sum
        <= gates["translation_block_sum_kcal_mol_per_angstrom2"],
        "translation_ward_passed": max(translation_errors)
        <= gates["ward_relative"] * hscale,
        "rotational_ward_passed": max(rotation_errors) <= gates["ward_relative"],
        "hvp_dense_passed": hvp_error <= gates["hvp_dense_kcal_mol_per_angstrom2"],
    }


def _validate_identity(
    receipt: dict[str, Any], protocol: dict[str, Any], prereg: dict[str, Any]
):
    case = receipt["case_identity"]
    for key in (
        "row_id",
        "center_index",
        "center_scale",
        "sigma_e",
        "terminal_positions_angstrom",
        "terminal_coordinate_sha256",
    ):
        if case.get(key) != prereg.get(key):
            raise ValueError(f"case identity differs: {key}")
    positions = _array(
        "terminal positions", case["terminal_positions_angstrom"], (3, 3)
    )
    if coordinate_sha256(positions) != case["terminal_coordinate_sha256"]:
        raise ValueError("terminal coordinate SHA256 differs")
    expected_topology = (
        protocol["topology_sha256"]
        if "topology_sha256" in protocol
        else protocol["topology_content_sha256"]
    )
    expected_provenance = {
        "model_identity": "chagb-r6-pbsa-gaussian-sign-v1",
        "numerical_profile_id": protocol["profile_id"],
        "topology_sha256": expected_topology,
        "sigma_e": case["sigma_e"],
        "order": int(protocol["production_order"]),
        "device": "cpu",
        "dtype": "torch.float64",
    }
    expected_fingerprint = [
        expected_provenance["model_identity"],
        expected_provenance["numerical_profile_id"],
        expected_provenance["topology_sha256"],
        expected_provenance["sigma_e"],
        expected_provenance["order"],
        expected_provenance["device"],
        expected_provenance["dtype"],
    ]
    if (
        receipt["source_identity_sha256"] != protocol["source_identity_sha256"]
        or receipt["input_identity_sha256"] != protocol["input_identity_sha256"]
        or receipt["profile_id"] != protocol["profile_id"]
        or receipt["topology_sha256"] != expected_topology
        or receipt["checkpoint_sha256"] != protocol["checkpoint_sha256"]
        or int(receipt["production_order"]) != int(protocol["production_order"])
        or case["sigma_e"] not in protocol["sigma_e"]
        or receipt["units"] != EXPECTED_UNITS
        or receipt["analytic_identity_provenance"] != expected_provenance
        or list(receipt["analytic_identity_fingerprint"]) != expected_fingerprint
    ):
        raise ValueError(
            "source/input/profile/topology/checkpoint/units identity differs"
        )
    return positions


def validate_freq_ts_row(
    receipt: dict[str, Any], protocol: dict[str, Any], prereg_row: dict[str, Any]
) -> dict[str, Any]:
    row_id = prereg_row.get("row_id")
    try:
        positions = _validate_identity(receipt, protocol, prereg_row)
        if receipt.get("exceptions"):
            raise ValueError("receipt retains a runtime exception")
        expected_directions = build_direction_inventory(
            positions,
            seed=int(prereg_row.get("direction_seed", 20261002)),
            nonrigid_directions=prereg_row.get("nonrigid_directions"),
        )
        observed_inventory = receipt["direction_inventory"]
        if len(observed_inventory) != 18:
            raise ValueError("direction inventory must contain exactly 18 directions")
        for observed, expected in zip(
            observed_inventory, expected_directions, strict=True
        ):
            if observed["direction_id"] != expected["direction_id"] or not np.allclose(
                _array("direction", observed["direction"], (9,)),
                np.asarray(expected["direction"]),
                rtol=0.0,
                atol=2e-15,
            ):
                raise ValueError("direction inventory differs")
        gates = protocol["gates"]
        term_data = {}
        term_metrics = {}
        for name in (*COMPONENTS, "total"):
            record = receipt["solvent_terms"][name]
            gradient = _array(
                f"{name} gradient", record["gradient_kcal_mol_per_angstrom"], (9,)
            )
            hessian = _array(
                f"{name} hessian", record["hessian_kcal_mol_per_angstrom2"], (9, 9)
            )
            hvp = _direction_map(
                record["directional_hvp"],
                expected_directions,
                "hvp_kcal_mol_per_angstrom2",
            )
            term_data[name] = {
                "energy": float(record["energy_kcal_mol"]),
                "gradient": gradient,
                "hessian": hessian,
                "hvp": hvp,
            }
            if not math.isfinite(term_data[name]["energy"]):
                raise ValueError(f"{name} energy must be finite")
            term_metrics[name] = _hessian_metrics(
                hessian, gradient, positions, expected_directions, hvp, gates
            )
        tolerance = gates["identity_relative"]
        component_closure = all(
            (
                _relative_close(
                    sum(term_data[name][field] for name in COMPONENTS),
                    term_data["total"][field],
                    tolerance,
                )
                if field != "hvp"
                else all(
                    _relative_close(
                        sum(term_data[name][field][direction] for name in COMPONENTS),
                        term_data["total"][field][direction],
                        tolerance,
                    )
                    for direction in term_data["total"][field]
                )
            )
            for field in ("energy", "gradient", "hessian", "hvp")
        )
        center_data = {}
        center_metrics = {}
        for name in CENTER_TERMS:
            record = receipt["analytic_center"][name]
            gradient = _array(
                f"{name} center gradient", record["gradient_hartree_per_angstrom"], (9,)
            )
            force = _array(
                f"{name} center forces", record["forces_hartree_per_angstrom"], (3, 3)
            )
            if not _relative_close(force.reshape(-1), -gradient, tolerance):
                raise ValueError(f"{name} force/gradient sign identity differs")
            hessian = _array(
                f"{name} center hessian",
                record["hessian_hartree_per_angstrom2"],
                (9, 9),
            )
            directional = record["directional_hvp"]
            if any("hvp_hartree_per_angstrom2" in item for item in directional):
                raise ValueError(
                    "center directional HVP schema retains ambiguous old alias"
                )
            dense_hvp_native = _direction_map(
                directional,
                expected_directions,
                "dense_hessian_product_hartree_per_angstrom2",
            )
            hvp_native = _direction_map(
                directional,
                expected_directions,
                "direct_hvp_hartree_per_angstrom2",
            )
            if not all(
                _relative_close(
                    dense_hvp_native[direction],
                    hessian
                    @ np.asarray(
                        next(
                            item["direction"]
                            for item in expected_directions
                            if item["direction_id"] == direction
                        )
                    ),
                    tolerance,
                )
                for direction in dense_hvp_native
            ):
                raise ValueError(f"{name} stored dense HVP differs from Hessian")
            center_data[name] = {
                "energy": float(record["energy_hartree"]),
                "gradient": gradient,
                "force": force.reshape(-1),
                "hessian": hessian,
                "hvp": hvp_native,
            }
            center_metrics[name] = _hessian_metrics(
                hessian * KCAL_PER_HARTREE,
                gradient * KCAL_PER_HARTREE,
                positions,
                expected_directions,
                {key: value * KCAL_PER_HARTREE for key, value in hvp_native.items()},
                gates,
            )
        direct_records = receipt["direct_hvp_records"]
        if [item.get("direction_id") for item in direct_records] != [
            item["direction_id"] for item in expected_directions
        ]:
            raise ValueError("direct HVP tuple inventory is incomplete or reordered")
        for record, expected in zip(direct_records, expected_directions, strict=True):
            direction = _array("direct HVP tuple direction", record["direction"], (9,))
            if not np.allclose(direction, expected["direction"], rtol=0.0, atol=2e-15):
                raise ValueError("direct HVP tuple direction differs")
            direction_id = expected["direction_id"]
            if not (
                _relative_close(
                    record["composed_hvp_hartree_per_angstrom2"],
                    center_data["composed"]["hvp"][direction_id],
                    tolerance,
                )
                and _relative_close(
                    record["tuple_forces_hartree_per_angstrom"],
                    center_data["composed"]["force"],
                    tolerance,
                )
                and _relative_close(
                    float(record["tuple_energy_hartree"]),
                    center_data["composed"]["energy"],
                    tolerance,
                )
            ):
                raise ValueError("direct HVP tuple differs from composed center")
        dense_composition = all(
            _relative_close(
                center_data["gas"][field] + center_data["solvent"][field],
                center_data["composed"][field],
                tolerance,
            )
            for field in ("energy", "gradient", "hessian")
        ) and all(
            _relative_close(
                center_data["gas"]["hvp"][direction]
                + center_data["solvent"]["hvp"][direction],
                center_data["composed"]["hvp"][direction],
                tolerance,
            )
            for direction in center_data["composed"]["hvp"]
        )
        solvent_native_identity = all(
            _relative_close(
                term_data["total"][field],
                center_data["solvent"][field] * KCAL_PER_HARTREE,
                tolerance,
            )
            for field in ("energy", "gradient", "hessian")
        ) and all(
            _relative_close(
                term_data["total"]["hvp"][direction],
                center_data["solvent"]["hvp"][direction] * KCAL_PER_HARTREE,
                tolerance,
            )
            for direction in term_data["total"]["hvp"]
        )
        force_metrics = _force_validation_metrics(
            receipt["force_validation"],
            center_data["composed"]["hessian"] * KCAL_PER_HARTREE,
            positions,
            protocol,
        )
        reference_center = _reference_center_diagnostics(
            receipt["reference_center_diagnostics"], protocol, positions
        )
        independent_metrics = _independent_curvature_metrics(
            receipt["independent_curvature"],
            term_data["total"]["hessian"],
            expected_directions[-3:],
            positions,
            protocol,
            reference_center[128],
        )
        covariance = _covariance_metrics(
            receipt["covariance"],
            positions,
            protocol,
            receipt["analytic_center"],
        )
        fresh_passed = _fresh_identity_passes(
            receipt["fresh_identity"], center_data, tolerance
        )
        counters = receipt["counters"]
        operations = counters["operations"]
        if not isinstance(operations, list) or not operations:
            raise ValueError("graph counter operation inventory is missing")
        recomputed_counts = {
            key: sum(int(operation[key]) for operation in operations)
            for key in ("gas_graphs", "solvent_graphs", "independent_energy_calls")
        }
        if any(
            int(operation[key]) < 0
            for operation in operations
            for key in ("gas_graphs", "solvent_graphs", "independent_energy_calls")
        ) or len({operation["operation_id"] for operation in operations}) != len(
            operations
        ):
            raise ValueError("graph counter operation inventory differs")
        if operations != protocol["budgets"]["operations"]:
            raise ValueError("graph operation inventory differs from protocol")
        if counters.get("graph_seams") != protocol["graph_seams"]:
            raise ValueError("graph seam identity differs from source freeze")
        if (
            counters.get("core_api_counters")
            != protocol["core_api_counter_expectations"]
        ):
            raise ValueError("core API counters differ from expected operations")
        nominal = protocol["budgets"]["nominal"]
        maximum = protocol["budgets"]["maximum"]
        budget_passed = bool(
            all(
                int(counters[key]) == recomputed_counts[key]
                for key in recomputed_counts
            )
            and recomputed_counts == nominal
            and all(
                recomputed_counts[key] <= int(maximum[key]) for key in recomputed_counts
            )
            and recomputed_counts["gas_graphs"] + recomputed_counts["solvent_graphs"]
            <= int(protocol["graph_budgets"]["per_cell_total"])
        )
        hessian_passed = all(
            all(value for key, value in metrics.items() if key.endswith("_passed"))
            for metrics in (*term_metrics.values(), *center_metrics.values())
        )
        passed = bool(
            hessian_passed
            and component_closure
            and dense_composition
            and solvent_native_identity
            and force_metrics["passed"]
            and independent_metrics["passed"]
            and covariance["passed"]
            and fresh_passed
            and budget_passed
        )
        status = (
            "SECOND_DERIVATIVE_VALIDATED"
            if passed
            else (
                "INDEPENDENT_CURVATURE_UNRESOLVED"
                if not independent_metrics["passed"]
                else "SECOND_DERIVATIVE_UNRESOLVED"
            )
        )
        return {
            "row_id": row_id,
            "passed": passed,
            "status": status,
            "direction_count": len(expected_directions),
            "independent_direction_count": 3,
            "solvent_term_metrics": term_metrics,
            "center_term_metrics": center_metrics,
            "component_closure_passed": component_closure,
            "dense_composition_passed": dense_composition,
            "solvent_native_identity_passed": solvent_native_identity,
            "force_validation": force_metrics,
            "independent_curvature": independent_metrics,
            "reference_center_diagnostics": reference_center,
            "covariance": covariance,
            "fresh_identity_passed": fresh_passed,
            "budget_passed": budget_passed,
        }
    except (KeyError, TypeError, ValueError, IndexError, OverflowError) as exc:
        return {
            "row_id": row_id,
            "passed": False,
            "status": "SECOND_DERIVATIVE_UNRESOLVED",
            "reason": str(exc),
            "error": type(exc).__name__,
        }


def _force_validation_metrics(
    raw: list[dict[str, Any]],
    hessian_kcal: np.ndarray,
    center_positions: np.ndarray,
    protocol,
):
    if len(raw) != 9 or [item.get("dof") for item in raw] != list(range(9)):
        raise ValueError("force-validation direction inventory differs")
    steps = [float(value) for value in protocol["force_fd_steps_angstrom"]]
    finest = np.zeros((9, 9))
    disagreements = []
    for item in raw:
        dof = int(item["dof"])
        direction = _array("force-validation direction", item["direction"], (9,))
        if not np.array_equal(direction, np.eye(9)[dof]):
            raise ValueError("force-validation direction differs")
        if [float(record.get("step_angstrom")) for record in item["steps"]] != steps:
            raise ValueError("force-validation steps differ")
        estimates = []
        for record in item["steps"]:
            h = float(record["step_angstrom"])
            expected_plus = (center_positions.reshape(-1) + h * direction).reshape(3, 3)
            expected_minus = (center_positions.reshape(-1) - h * direction).reshape(
                3, 3
            )
            plus_positions = _array(
                "plus force-validation positions",
                record["plus_positions_angstrom"],
                (3, 3),
            )
            minus_positions = _array(
                "minus force-validation positions",
                record["minus_positions_angstrom"],
                (3, 3),
            )
            if (
                not np.array_equal(plus_positions, expected_plus)
                or not np.array_equal(minus_positions, expected_minus)
                or coordinate_sha256(plus_positions) != record["plus_coordinate_sha256"]
                or coordinate_sha256(minus_positions)
                != record["minus_coordinate_sha256"]
            ):
                raise ValueError("force-validation displaced coordinates differ")
            estimate_hartree = force_difference_hvp(
                record["plus_composed_forces_hartree_per_angstrom"],
                record["minus_composed_forces_hartree_per_angstrom"],
                record["step_angstrom"],
            )
            estimates.append(estimate_hartree * KCAL_PER_HARTREE)
        finest[:, dof] = estimates[-1]
        disagreements.append(float(np.max(np.abs(estimates[-1] - estimates[-2]))))
    error = finest - hessian_kcal
    maximum = float(np.max(np.abs(error)))
    rms = float(np.sqrt(np.mean(error**2)))
    disagreement = max(disagreements)
    gates = protocol["gates"]
    return {
        "maximum_error_kcal_mol_per_angstrom2": maximum,
        "rms_error_kcal_mol_per_angstrom2": rms,
        "two_finest_disagreement_kcal_mol_per_angstrom2": disagreement,
        "passed": bool(
            maximum <= gates["force_fd_max_kcal_mol_per_angstrom2"]
            and rms <= gates["force_fd_rms_kcal_mol_per_angstrom2"]
            and disagreement <= gates["force_fd_disagreement_kcal_mol_per_angstrom2"]
        ),
    }


def _reference_center_diagnostics(raw, protocol, center_positions):
    if not isinstance(raw, list) or [item.get("order") for item in raw] != [
        64,
        96,
        128,
    ]:
        raise ValueError("reference center order inventory differs")
    result = {}
    epsabs = float(protocol["independent_reference"]["epsabs"])
    epsrel = float(protocol["independent_reference"]["epsrel"])
    for record in raw:
        energy = float(record["total_energy_kcal_mol"])
        errors = np.asarray(record["quadrature_error_estimates"], dtype=np.float64)
        if (
            not math.isfinite(energy)
            or float(record["epsabs"]) != epsabs
            or float(record["epsrel"]) != epsrel
            or errors.size == 0
            or not np.isfinite(errors).all()
            or np.any(errors < 0.0)
            or not isinstance(record.get("branch_metadata"), dict)
            or record["branch_metadata"].get("domain") != "admitted"
            or record.get("coordinate_sha256") != coordinate_sha256(center_positions)
        ):
            raise ValueError("reference center diagnostics differ")
        _validate_reference_geometry_metadata(
            record["reference_geometry_metadata"],
            center_positions,
            int(record["order"]),
            protocol,
        )
        result[int(record["order"])] = energy
    return result


def _independent_curvature_metrics(
    raw,
    hessian,
    expected_directions,
    center_positions,
    protocol,
    reference_center_energy,
):
    if len(raw) != 3 or [item.get("direction_id") for item in raw] != [
        item["direction_id"] for item in expected_directions
    ]:
        raise ValueError("independent direction inventory differs")
    steps = [float(value) for value in protocol["curvature_steps_angstrom"]]
    order = int(protocol["independent_reference"]["order"])
    epsabs = float(protocol["independent_reference"]["epsabs"])
    epsrel = float(protocol["independent_reference"]["epsrel"])
    errors = []
    disagreements = []
    for item, expected in zip(raw, expected_directions, strict=True):
        direction = _array("independent direction", item["direction"], (9,))
        if not np.allclose(direction, expected["direction"], rtol=0.0, atol=2e-15):
            raise ValueError("independent direction differs")
        if (
            len(item["steps"]) != 3
            or [float(record.get("step_angstrom")) for record in item["steps"]] != steps
        ):
            raise ValueError("independent curvature steps differ")
        estimates = []
        for record in item["steps"]:
            samples = record.get("samples", [])
            if len(samples) != 5:
                raise ValueError("independent curvature requires five energies")
            if [sample.get("multiplier") for sample in samples] != [-2, -1, 0, 1, 2]:
                raise ValueError("independent curvature multipliers differ")
            if int(record.get("reference_order")) != order:
                raise ValueError("independent reference order differs")
            if (
                float(record.get("epsabs")) != epsabs
                or float(record.get("epsrel")) != epsrel
            ):
                raise ValueError("independent quadrature tolerances differ")
            errors_raw = np.asarray(
                [sample["quadrature_error_estimate"] for sample in samples],
                dtype=np.float64,
            )
            if (
                errors_raw.shape != (5,)
                or not np.isfinite(errors_raw).all()
                or np.any(errors_raw < 0)
            ):
                raise ValueError(
                    "independent quadrature errors must be finite nonnegative"
                )
            if any(
                not isinstance(sample.get("branch_metadata"), dict)
                or sample["branch_metadata"].get("domain") != "admitted"
                for sample in samples
            ):
                raise ValueError("independent branch/domain metadata is unresolved")
            for sample in samples:
                expected_positions = (
                    center_positions.reshape(-1)
                    + float(sample["multiplier"])
                    * float(record["step_angstrom"])
                    * direction
                ).reshape(3, 3)
                observed_positions = _array(
                    "independent displaced positions",
                    sample["positions_angstrom"],
                    (3, 3),
                )
                if (
                    not np.array_equal(observed_positions, expected_positions)
                    or coordinate_sha256(observed_positions)
                    != sample["coordinate_sha256"]
                ):
                    raise ValueError("independent displaced coordinates differ")
                _validate_reference_geometry_metadata(
                    sample["reference_geometry_metadata"],
                    observed_positions,
                    order,
                    protocol,
                )
            estimates.append(
                five_point_curvature(
                    [sample["energy_kcal_mol"] for sample in samples],
                    record["step_angstrom"],
                )
            )
            if float(samples[2]["energy_kcal_mol"]) != float(reference_center_energy):
                raise ValueError("independent shared center energy differs")
        expected_curvature = float(direction @ hessian @ direction)
        errors.append(abs(estimates[-1] - expected_curvature))
        disagreements.append(abs(estimates[-1] - estimates[-2]))
    maximum = max(errors)
    disagreement = max(disagreements)
    gates = protocol["gates"]
    return {
        "maximum_error_kcal_mol_per_angstrom2": maximum,
        "two_finest_disagreement_kcal_mol_per_angstrom2": disagreement,
        "coverage": "three-seeded-directions-energy-only-not-full-hessian-oracle",
        "passed": bool(
            maximum <= gates["independent_curvature_kcal_mol_per_angstrom2"]
            and disagreement <= gates["independent_disagreement_kcal_mol_per_angstrom2"]
        ),
    }


def _covariance_metrics(raw, positions, protocol, analytic_center):
    expected_inventory = [
        (transform["transform_id"], term)
        for transform in protocol["covariance_transforms"]
        for term in CENTER_TERMS
    ]
    if (
        len(raw) != len(expected_inventory)
        or [(item.get("transform_id"), item.get("term")) for item in raw]
        != expected_inventory
    ):
        raise ValueError("covariance term inventory differs")
    gates = protocol["gates"]
    passed = True
    maximum_h = 0.0
    transforms = {
        transform["transform_id"]: transform
        for transform in protocol["covariance_transforms"]
    }
    for item in raw:
        expected_transform = transforms[item["transform_id"]]
        q = _array("covariance Q", item["Q"], (3, 3))
        shift = _array("covariance translation", item["a"], (3,))
        if not np.array_equal(
            q, np.asarray(expected_transform["Q"], dtype=np.float64)
        ) or not np.array_equal(
            shift, np.asarray(expected_transform["a"], dtype=np.float64)
        ):
            raise ValueError("covariance transform differs from protocol")
        if not np.allclose(q.T @ q, np.eye(3), rtol=0.0, atol=1e-12) or not np.isclose(
            np.linalg.det(q), 1.0, rtol=0.0, atol=1e-12
        ):
            raise ValueError("covariance Q must be a proper rotation")
        transformed_positions = _array(
            "transformed positions", item["transformed_positions_angstrom"], (3, 3)
        )
        if not np.allclose(
            transformed_positions, positions @ q.T + shift, rtol=0.0, atol=2e-14
        ):
            raise ValueError("covariance transformed coordinates differ")
        if coordinate_sha256(transformed_positions) != item.get(
            "transformed_coordinate_sha256"
        ):
            raise ValueError("covariance transformed coordinate hash differs")
        base = item["base"]
        if base != analytic_center[item["term"]]:
            raise ValueError("covariance embedded base differs from canonical center")
        moved = item["transformed"]
        base_force = _array(
            "base covariance force", base["forces_hartree_per_angstrom"], (3, 3)
        )
        moved_force = _array(
            "moved covariance force", moved["forces_hartree_per_angstrom"], (3, 3)
        )
        base_h = _array(
            "base covariance Hessian", base["hessian_hartree_per_angstrom2"], (9, 9)
        )
        moved_h = _array(
            "moved covariance Hessian", moved["hessian_hartree_per_angstrom2"], (9, 9)
        )
        q3 = np.kron(np.eye(3), q)
        h_error = float(np.max(np.abs(moved_h - q3 @ base_h @ q3.T))) * KCAL_PER_HARTREE
        maximum_h = max(maximum_h, h_error)
        force_error = (
            float(np.max(np.abs(moved_force - base_force @ q.T))) * KCAL_PER_HARTREE
        )
        energy_error = (
            abs(float(moved["energy_hartree"]) - float(base["energy_hartree"]))
            * KCAL_PER_HARTREE
        )
        base_centered = positions - positions.mean(axis=0)
        moved_centered = transformed_positions - transformed_positions.mean(axis=0)
        torque_base = np.cross(base_centered, base_force).sum(axis=0)
        torque_moved = np.cross(moved_centered, moved_force).sum(axis=0)
        torque_error = (
            float(np.max(np.abs(torque_moved - q @ torque_base))) * KCAL_PER_HARTREE
        )
        torque_zero = (
            max(
                float(np.max(np.abs(torque_base))),
                float(np.max(np.abs(torque_moved))),
            )
            * KCAL_PER_HARTREE
        )
        force_gate = gates.get("covariance_force_kcal_mol_per_angstrom", 2e-5)
        passed = bool(
            passed
            and h_error <= 1e-8
            and force_error <= force_gate
            and energy_error <= gates.get("covariance_energy_kcal_mol", 1e-6)
            and torque_error <= force_gate
            and torque_zero <= force_gate
        )
    return {"hessian_max_error_kcal_mol_per_angstrom2": maximum_h, "passed": passed}


def _fresh_identity_passes(raw, center_data, tolerance):
    if int(raw["evaluation_count"]) != 1:
        return False
    return bool(
        all(
            _relative_close(
                raw[f"fresh_{name}_energy_hartree"],
                center_data[name]["energy"],
                tolerance,
            )
            and _relative_close(
                raw[f"fresh_{name}_forces_hartree_per_angstrom"],
                center_data[name]["force"].reshape(3, 3),
                tolerance,
            )
            for name in CENTER_TERMS
        )
        and _relative_close(
            raw["fresh_gas_energy_hartree"] + raw["fresh_solvent_energy_hartree"],
            raw["fresh_composed_energy_hartree"],
            tolerance,
        )
        and _relative_close(
            np.asarray(raw["fresh_gas_forces_hartree_per_angstrom"])
            + np.asarray(raw["fresh_solvent_forces_hartree_per_angstrom"]),
            raw["fresh_composed_forces_hartree_per_angstrom"],
            tolerance,
        )
        and _relative_close(
            raw["hvp_tuple_energy_hartree"],
            center_data["composed"]["energy"],
            tolerance,
        )
        and _relative_close(
            raw["hvp_tuple_forces_hartree_per_angstrom"],
            center_data["composed"]["force"],
            tolerance,
        )
    )


def summarize_freq_ts_rows(
    rows: list[dict[str, Any]], expected_rows: list[dict[str, Any]]
):
    expected_ids = [row["row_id"] for row in expected_rows]
    observed_ids = [row.get("row_id") for row in rows]
    exact = observed_ids == expected_ids and len(set(observed_ids)) == 15
    passed_count = sum(bool(row.get("passed")) for row in rows)
    passed = exact and passed_count == 15
    return {
        "row_count": len(rows),
        "inventory_exact": exact,
        "passed_count": passed_count,
        "status": (
            "SECOND_DERIVATIVE_VALIDATED" if passed else "SECOND_DERIVATIVE_UNRESOLVED"
        ),
    }


__all__ = [
    "build_direction_inventory",
    "coordinate_sha256",
    "five_point_curvature",
    "force_difference_hvp",
    "summarize_freq_ts_rows",
    "validate_freq_ts_row",
]
