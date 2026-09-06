"""Pure preregistered reduction for AIMNet2 multi-solvent force evidence.

The helpers in this module select no chemistry and execute no model.  They
construct geometry-only directions and reduce raw same-scalar finite-
difference measurements against the frozen v1 thresholds.  A passing summary
is prequalification evidence only and never changes a MAPLE capability.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Mapping, Sequence

import numpy as np

from maple.function.calculator.aimnet._aimnet2_float64_source import (
    AIMNET_FIRST_ORDER_ENERGY_PARITY_ABSOLUTE_TOLERANCE_EV,
    AIMNET_SECOND_ORDER_CHARGE_PARITY_ABSOLUTE_TOLERANCE_E,
    AIMNET_SECOND_ORDER_CHARGE_VJP_PARITY_ABSOLUTE_TOLERANCE_EV_PER_A,
    AIMNET_SECOND_ORDER_ENERGY_PARITY_ABSOLUTE_TOLERANCE_EV,
    AIMNET_SECOND_ORDER_GRADIENT_PARITY_ABSOLUTE_TOLERANCE_EV_PER_A,
)
from maple.solvation.continuum.aimnet2_smooth_partition_ddpcm import (
    AIMNET2_SMOOTH_PARTITION_PARTITION_LMAX,
    AIMNET2_SMOOTH_PARTITION_TRANSITION_WIDTH_ANGSTROM2,
)
from maple.solvation.release.geometry_mediated import (
    summarize_geometry_mediated_reciprocity_audit,
)

FORCE_PREQUALIFICATION_CONTRACT_VERSION = (
    "route2-aimnet2-smooth-ddpcm-force-prequalification-v1"
)
FORCE_PREQUALIFICATION_STEPS_A = (4.0e-4, 2.0e-4, 1.0e-4)
FORCE_PREQUALIFICATION_LEAVES = (
    "vacuum",
    "continuum",
    "electrostatic_total",
    "nonpolar",
    "total",
)
FORCE_PREQUALIFICATION_THRESHOLDS = {
    "replay_energy_absolute_eV": 1.0e-10,
    "replay_source_max_absolute_e": 1.0e-10,
    "replay_gradient_max_absolute_eV_per_A": 1.0e-9,
    "directional_all_step_max_absolute_eV_per_A": 2.0e-3,
    "directional_terminal_absolute_eV_per_A": 5.0e-4,
    "directional_terminal_relative": 2.0e-3,
    "directional_relative_denominator_floor_eV_per_A": 1.0e-3,
    "cartesian_step_rms_eV_per_A": 5.0e-4,
    "cartesian_step_max_eV_per_A": 2.0e-3,
    "observed_central_order_minimum": 1.5,
    "numerical_plateau_eV_per_A": 5.0e-5,
    "terminal_growth_maximum": 1.25,
    "net_gradient_norm_eV_per_A": 1.0e-5,
    "torque_norm_eV": 1.0e-4,
    "maximum_algebraic_degree": 192,
}
NO_CAPABILITIES = {tier: False for tier in ("E", "F", "H", "V", "M")}
FORCE_SOURCE_PARITY_THRESHOLDS = {
    "energy_absolute_error_eV": (
        AIMNET_FIRST_ORDER_ENERGY_PARITY_ABSOLUTE_TOLERANCE_EV
    ),
    "charge_max_absolute_error_e": (
        AIMNET_SECOND_ORDER_CHARGE_PARITY_ABSOLUTE_TOLERANCE_E
    ),
    "intrinsic_gradient_max_absolute_error_eV_per_A": (
        AIMNET_SECOND_ORDER_GRADIENT_PARITY_ABSOLUTE_TOLERANCE_EV_PER_A
    ),
    "charge_vjp_max_absolute_error_eV_per_A": (
        AIMNET_SECOND_ORDER_CHARGE_VJP_PARITY_ABSOLUTE_TOLERANCE_EV_PER_A
    ),
    "repeat_energy_absolute_error_eV": (
        AIMNET_SECOND_ORDER_ENERGY_PARITY_ABSOLUTE_TOLERANCE_EV
    ),
    "repeat_charge_max_absolute_error_e": (
        AIMNET_SECOND_ORDER_CHARGE_PARITY_ABSOLUTE_TOLERANCE_E
    ),
    "repeat_intrinsic_gradient_max_absolute_error_eV_per_A": (
        AIMNET_SECOND_ORDER_GRADIENT_PARITY_ABSOLUTE_TOLERANCE_EV_PER_A
    ),
    "repeat_charge_vjp_max_absolute_error_eV_per_A": (
        AIMNET_SECOND_ORDER_CHARGE_VJP_PARITY_ABSOLUTE_TOLERANCE_EV_PER_A
    ),
}
FORCE_MEASUREMENT_ARTIFACT = (
    "route2-aimnet2-smooth-ddpcm-force-prequalification-shard-v1"
)
FORCE_MEASUREMENT_SCHEMA_VERSION = 1
FORCE_MEASUREMENT_STATUS = "private-measurement-complete"
FORCE_FINALIZED_STATUS = "private-cold-replay-finalized"
FORCE_MEASUREMENT_CLAIM_BOUNDARY = (
    "Private same-scalar force prequalification measurement only; no public "
    "E/F/H/V/M or daily-task admission."
)
_BINDING_KEYS = (
    "execution_git_commit",
    "execution_git_tree",
    "protocol_file_sha256",
    "selection_manifest_file_sha256",
    "selection_manifest_canonical_sha256",
    "checkpoint_sha256",
    "checkpoint_bytes",
    "aimnet2_runtime_kind",
    "aimnet2_runtime_provenance_sha256",
    "aimnet2_provider_id",
    "scalar_id",
    "profile_id",
    "scalar_fingerprint_sha256",
    "model_configuration_sha256",
    "continuum_configuration_sha256",
    "continuum_provenance_sha256",
    "nonpolar_provider_id",
    "nonpolar_profile_id",
    "nonpolar_configuration_sha256",
    "nonpolar_runtime_provenance_sha256",
    "center_model_topology_sha256",
    "center_model_minimum_cutoff_margin_angstrom",
    "center_continuum_topology_sha256",
    "numerical_runtime_sha256",
    "task",
    "stage",
    "mode",
)


def canonical_sha256(value: object) -> str:
    """Return the repository's stable canonical-JSON SHA256."""

    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _positions(value: object) -> np.ndarray:
    positions = np.asarray(value, dtype=float)
    if (
        positions.ndim != 2
        or positions.shape[0] < 2
        or positions.shape[1] != 3
        or not np.all(np.isfinite(positions))
    ):
        raise ValueError("positions must be finite with shape (N,3), N>=2.")
    return np.array(positions, copy=True)


def _rigid_basis(positions: np.ndarray) -> np.ndarray:
    count = positions.shape[0]
    centered = positions - np.mean(positions, axis=0)
    modes = []
    for axis in np.eye(3):
        modes.append(np.tile(axis, (count, 1)).reshape(-1))
    for axis in np.eye(3):
        modes.append(np.cross(np.tile(axis, (count, 1)), centered).reshape(-1))
    matrix = np.stack(modes, axis=1)
    left, singular, _right = np.linalg.svd(matrix, full_matrices=False)
    tolerance = max(matrix.shape) * np.finfo(float).eps * singular[0]
    return left[:, singular > tolerance]


def _project_internal(raw: np.ndarray, rigid: np.ndarray, *, name: str) -> np.ndarray:
    vector = np.asarray(raw, dtype=float).reshape(-1)
    vector = vector - rigid @ (rigid.T @ vector)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 1.0e-12:
        raise ValueError(f"{name} collapsed after rigid-motion projection.")
    result = (vector / norm).reshape(-1, 3)
    result.setflags(write=False)
    return result


def _hash_vector(geometry_sha256: str, *, lane: int, size: int) -> np.ndarray:
    if (
        not isinstance(geometry_sha256, str)
        or len(geometry_sha256) != 64
        or any(character not in "0123456789abcdef" for character in geometry_sha256)
    ):
        raise ValueError("geometry_sha256 must contain 64 lowercase hex digits.")
    values = []
    for index in range(size):
        digest = hashlib.sha256(
            f"{geometry_sha256}:force-direction-v1:{lane}:{index}".encode()
        ).digest()
        integer = int.from_bytes(digest[:8], byteorder="big", signed=False)
        values.append(2.0 * (integer / (2**64 - 1)) - 1.0)
    return np.asarray(values, dtype=float)


def geometry_internal_directions(
    positions: object, geometry_sha256: str
) -> dict[str, np.ndarray]:
    """Return three normalized geometry-only non-rigid Cartesian directions."""

    values = _positions(positions)
    rigid = _rigid_basis(values)
    breathing = (values - np.mean(values, axis=0)).reshape(-1)
    return {
        "projected_breathing": _project_internal(
            breathing, rigid, name="projected breathing direction"
        ),
        "sha256_internal_0": _project_internal(
            _hash_vector(geometry_sha256, lane=0, size=values.size),
            rigid,
            name="SHA256 direction 0",
        ),
        "sha256_internal_1": _project_internal(
            _hash_vector(geometry_sha256, lane=1, size=values.size),
            rigid,
            name="SHA256 direction 1",
        ),
    }


def cartesian_directions(atom_count: int) -> dict[str, np.ndarray]:
    if (
        isinstance(atom_count, bool)
        or not isinstance(atom_count, int)
        or atom_count < 1
    ):
        raise ValueError("atom_count must be a positive integer.")
    result = {}
    for atom in range(atom_count):
        for axis, label in enumerate("xyz"):
            vector = np.zeros((atom_count, 3), dtype=float)
            vector[atom, axis] = 1.0
            vector.setflags(write=False)
            result[f"atom_{atom:03d}_{label}"] = vector
    return result


def topology_preflight(
    positions: object,
    radii_angstrom: object,
    *,
    transition_width_angstrom2: float,
    partition_lmax: int,
) -> dict[str, object]:
    """Return the exact factor-degree preflight and pair-state digest."""

    values = _positions(positions)
    radii = np.asarray(radii_angstrom, dtype=float)
    if (
        radii.shape != (values.shape[0],)
        or not np.all(np.isfinite(radii))
        or np.any(radii <= 0.0)
    ):
        raise ValueError("radii_angstrom must be finite and positive with shape (N,).")
    width = float(transition_width_angstrom2)
    if not np.isfinite(width) or width <= 0.0:
        raise ValueError("transition_width_angstrom2 must be positive.")
    if isinstance(partition_lmax, bool) or not isinstance(partition_lmax, int):
        raise TypeError("partition_lmax must be an integer.")
    if partition_lmax < 1:
        raise ValueError("partition_lmax must be positive.")
    states = []
    maximum_factor_count = 0
    minimum_center_distance = math.inf
    for atom_i, radius_i in enumerate(radii):
        inside_count = 0
        centered_count = 0
        centered_buried = False
        for atom_j, radius_j in enumerate(radii):
            if atom_i == atom_j:
                continue
            distance = float(np.linalg.norm(values[atom_j] - values[atom_i]))
            if not np.isfinite(distance) or distance <= 1.0e-12:
                raise ValueError("sphere centres must remain distinct.")
            minimum_center_distance = min(minimum_center_distance, distance)
            minimum = (distance - radius_i) ** 2 - radius_j**2
            maximum = (distance + radius_i) ** 2 - radius_j**2
            if minimum >= width:
                state = "exposed"
            elif maximum <= -width:
                state = "buried"
            else:
                state = "transition"
            states.append((atom_i, atom_j, state))
            if minimum < 0.0:
                inside_count += 1
            if maximum <= -width:
                centered_buried = True
            elif minimum < width:
                centered_count += 1
        maximum_factor_count = max(
            maximum_factor_count,
            inside_count,
            0 if centered_buried else centered_count,
        )
    state_payload = [list(item) for item in states]
    state_sha256 = canonical_sha256(state_payload)
    return {
        "pair_state_sha256": state_sha256,
        "pair_state_counts": {
            state: sum(item[2] == state for item in states)
            for state in ("exposed", "transition", "buried")
        },
        "maximum_factor_count": maximum_factor_count,
        "required_algebraic_degree": (maximum_factor_count + 1) * partition_lmax,
        "minimum_center_distance_angstrom": minimum_center_distance,
    }


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _validate_continuum_topology(
    value: object,
    *,
    expected_configuration_sha256: str,
    expected_topology: Mapping[str, object] | None = None,
    name: str,
) -> dict[str, object]:
    topology = dict(_mapping(value, name=name))
    required = {
        "active_coefficient_deletion",
        "cavity_topology_sha256",
        "coefficient_count",
        "coefficient_topology_sha256",
        "configuration_sha256",
        "laboratory_fixed_surface_grid",
    }
    if set(topology) != required:
        raise ValueError(f"{name} fields drifted from the continuum contract.")
    if (
        topology["active_coefficient_deletion"] is not False
        or topology["laboratory_fixed_surface_grid"] is not False
        or topology["configuration_sha256"] != expected_configuration_sha256
    ):
        raise ValueError(f"{name} violates the smooth fixed-index topology contract.")
    _hex_digest(
        topology["cavity_topology_sha256"],
        name=f"{name} cavity topology SHA256",
        length=64,
    )
    _hex_digest(
        topology["coefficient_topology_sha256"],
        name=f"{name} coefficient topology SHA256",
        length=64,
    )
    _positive_integer(topology["coefficient_count"], name=f"{name} coefficient count")
    if expected_topology is not None and canonical_sha256(topology) != canonical_sha256(
        expected_topology
    ):
        raise ValueError(f"{name} drifted across the finite-difference stencil.")
    return topology


def _validate_preflight(
    observed: object,
    *,
    positions: np.ndarray,
    radii: np.ndarray,
    name: str,
) -> dict[str, object]:
    expected = topology_preflight(
        positions,
        radii,
        transition_width_angstrom2=(
            AIMNET2_SMOOTH_PARTITION_TRANSITION_WIDTH_ANGSTROM2
        ),
        partition_lmax=AIMNET2_SMOOTH_PARTITION_PARTITION_LMAX,
    )
    value = dict(_mapping(observed, name=name))
    if canonical_sha256(value) != canonical_sha256(expected):
        raise ValueError(f"{name} disagrees with exact geometry/radii recomputation.")
    degree = _positive_integer(
        value.get("required_algebraic_degree"),
        name=f"{name} required algebraic degree",
    )
    if degree > FORCE_PREQUALIFICATION_THRESHOLDS["maximum_algebraic_degree"]:
        raise ValueError(f"{name} exceeds the preregistered algebraic-degree bound.")
    return value


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping.")
    return value


def _sequence(value: object, *, name: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a sequence.")
    return value


def _finite(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric.")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _nonnegative_finite(value: object, *, name: str) -> float:
    result = _finite(value, name=name)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative.")
    return result


def _hex_digest(value: object, *, name: str, length: int) -> str:
    if (
        not isinstance(value, str)
        or len(value) != length
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must contain {length} lowercase hexadecimal digits.")
    return value


def _validate_runtime_provenance(record: Mapping[str, object]) -> None:
    provenance = _mapping(
        record.get("aimnet2_runtime_provenance"),
        name="AIMNet2 runtime provenance",
    )
    try:
        json.dumps(
            provenance,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "AIMNet2 runtime provenance must be finite JSON metadata."
        ) from exc
    observed = _hex_digest(
        record.get("aimnet2_runtime_provenance_sha256"),
        name="AIMNet2 runtime provenance SHA256",
        length=64,
    )
    if canonical_sha256(provenance) != observed:
        raise ValueError("AIMNet2 runtime provenance SHA256 is invalid.")
    if provenance.get("runtime_kind") != record.get("aimnet2_runtime_kind"):
        raise ValueError("AIMNet2 runtime kind and provenance disagree.")


def _validate_nonpolar_runtime_provenance(
    record: Mapping[str, object], task: Mapping[str, object]
) -> None:
    provenance = _mapping(
        record.get("nonpolar_runtime_provenance"),
        name="nonpolar runtime provenance",
    )
    try:
        json.dumps(
            provenance,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "nonpolar runtime provenance must be finite JSON metadata."
        ) from exc
    if set(provenance) != {
        "provider",
        "pyscf_version",
        "solvent",
        "pyscf_smd_solvent",
        "upstream_entrypoint",
    }:
        raise ValueError("nonpolar runtime provenance fields drifted.")
    observed = _hex_digest(
        record.get("nonpolar_runtime_provenance_sha256"),
        name="nonpolar runtime provenance SHA256",
        length=64,
    )
    if canonical_sha256(provenance) != observed:
        raise ValueError("nonpolar runtime provenance SHA256 is invalid.")
    if (
        provenance.get("provider") != "pyscf-smd-libsolvent-cds"
        or provenance.get("solvent") != task.get("canonical_solvent")
        or provenance.get("upstream_entrypoint") != "pyscf.solvent.smd.get_cds_legacy"
    ):
        raise ValueError("nonpolar runtime provenance identity drifted.")


def _validate_numerical_runtime(record: Mapping[str, object]) -> None:
    runtime = _mapping(record.get("numerical_runtime"), name="numerical runtime")
    try:
        json.dumps(
            runtime,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("numerical runtime must be finite JSON metadata.") from exc
    observed = _hex_digest(
        record.get("numerical_runtime_sha256"),
        name="numerical runtime SHA256",
        length=64,
    )
    if canonical_sha256(runtime) != observed:
        raise ValueError("numerical runtime SHA256 is invalid.")
    environment = _mapping(runtime.get("thread_environment"), name="thread environment")
    if any(
        environment.get(name) != "1"
        for name in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
        )
    ):
        raise ValueError("numerical thread environment is not single-threaded.")
    pools = _sequence(runtime.get("threadpools"), name="numerical thread pools")
    if not pools or any(
        not isinstance(pool, Mapping)
        or isinstance(pool.get("num_threads"), bool)
        or pool.get("num_threads") != 1
        for pool in pools
    ):
        raise ValueError("effective numerical thread pools are not single-threaded.")
    if (
        runtime.get("torch_num_threads") != 1
        or runtime.get("torch_num_interop_threads") != 1
    ):
        raise ValueError("effective Torch runtime is not single-threaded.")


def _validate_task(task: Mapping[str, object]) -> str:
    required = {
        "record_ordinal",
        "selection_index",
        "opaque_record_id",
        "partition",
        "geometry_handle",
        "geometry_sha256",
        "canonical_solvent",
        "stratum",
        "role",
        "atomic_numbers",
        "coordinates_angstrom",
        "cavity_radii_angstrom",
        "task_sha256",
    }
    if set(task) != required:
        raise ValueError("force task fields drifted from the preregistered schema.")
    observed = _hex_digest(task.get("task_sha256"), name="force task SHA256", length=64)
    expected = canonical_sha256(
        {key: value for key, value in task.items() if key != "task_sha256"}
    )
    if observed != expected:
        raise ValueError("force task SHA256 is invalid.")
    atomic_numbers = _sequence(task.get("atomic_numbers"), name="task atomic numbers")
    if not atomic_numbers or any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in atomic_numbers
    ):
        raise ValueError("task atomic numbers must be positive integers.")
    coordinates = _positions(task.get("coordinates_angstrom"))
    if len(atomic_numbers) != coordinates.shape[0]:
        raise ValueError("task atoms and coordinates have different lengths.")
    radii = np.asarray(task.get("cavity_radii_angstrom"), dtype=float)
    if (
        radii.shape != (coordinates.shape[0],)
        or not np.all(np.isfinite(radii))
        or np.any(radii <= 0.0)
    ):
        raise ValueError("task cavity radii are invalid.")
    _hex_digest(task.get("geometry_sha256"), name="geometry SHA256", length=64)
    _hex_digest(task.get("opaque_record_id"), name="opaque record ID", length=64)
    return observed


def _expected_directions(
    positions: np.ndarray, task: Mapping[str, object], mode: str
) -> dict[str, np.ndarray]:
    if mode == "directional":
        return geometry_internal_directions(
            positions, str(task.get("geometry_sha256", ""))
        )
    if mode == "full-cartesian":
        return cartesian_directions(positions.shape[0])
    raise ValueError("force record mode must be directional or full-cartesian.")


def validate_force_measurement_record(
    raw: Mapping[str, object],
    *,
    expected_bindings: Mapping[str, object],
    replicate: str,
) -> None:
    """Fail closed unless a private shard matches every external binding."""

    if set(expected_bindings) != set(_BINDING_KEYS):
        raise ValueError("expected force-record bindings are incomplete or excessive.")
    if replicate not in {"primary", "replay"}:
        raise ValueError("measurement replicate must be primary or replay.")
    if (
        raw.get("artifact") != FORCE_MEASUREMENT_ARTIFACT
        or raw.get("schema_version") != FORCE_MEASUREMENT_SCHEMA_VERSION
        or raw.get("status") != FORCE_MEASUREMENT_STATUS
        or raw.get("do_not_commit") is not True
        or raw.get("capabilities") != NO_CAPABILITIES
        or raw.get("claim_boundary") != FORCE_MEASUREMENT_CLAIM_BOUNDARY
        or raw.get("replicate") != replicate
    ):
        raise ValueError("private force measurement identity/status drifted.")
    for key in _BINDING_KEYS:
        if key == "task":
            continue
        if raw.get(key) != expected_bindings[key]:
            raise ValueError(f"private force measurement binding drifted: {key}.")
    task = _mapping(raw.get("task"), name="force task")
    expected_task = _mapping(expected_bindings["task"], name="expected force task")
    task_sha256 = _validate_task(task)
    expected_task_sha256 = _validate_task(expected_task)
    if task_sha256 != expected_task_sha256 or canonical_sha256(
        task
    ) != canonical_sha256(expected_task):
        raise ValueError("private force measurement task drifted.")
    for key, length in (
        ("execution_git_commit", 40),
        ("execution_git_tree", 40),
        ("protocol_file_sha256", 64),
        ("selection_manifest_file_sha256", 64),
        ("selection_manifest_canonical_sha256", 64),
        ("checkpoint_sha256", 64),
        ("scalar_fingerprint_sha256", 64),
        ("model_configuration_sha256", 64),
        ("continuum_configuration_sha256", 64),
        ("continuum_provenance_sha256", 64),
        ("nonpolar_configuration_sha256", 64),
        ("center_model_topology_sha256", 64),
        ("center_continuum_topology_sha256", 64),
        ("numerical_runtime_sha256", 64),
    ):
        _hex_digest(raw.get(key), name=key, length=length)
    checkpoint_bytes = raw.get("checkpoint_bytes")
    if (
        isinstance(checkpoint_bytes, bool)
        or not isinstance(checkpoint_bytes, int)
        or checkpoint_bytes <= 0
    ):
        raise ValueError("checkpoint_bytes must be a positive integer.")
    _validate_runtime_provenance(raw)
    _validate_nonpolar_runtime_provenance(raw, task)
    _validate_numerical_runtime(raw)
    process = _mapping(raw.get("process_identity"), name="process identity")
    pid = process.get("pid")
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        raise ValueError("process identity PID must be positive.")
    _hex_digest(
        process.get("process_import_token"),
        name="process import token",
        length=64,
    )
    if not isinstance(process.get("python_executable"), str) or not process.get(
        "python_executable"
    ):
        raise ValueError("process Python executable is missing.")

    center = _mapping(raw.get("center"), name="center")
    positions = _positions(center.get("positions_angstrom"))
    expected_positions = _positions(task.get("coordinates_angstrom"))
    if not np.array_equal(positions, expected_positions):
        raise ValueError("center positions drifted from the frozen force task.")
    _energy_leaves(center.get("energies_eV"), name="center energies")
    _gradient_leaves(
        center.get("gradients_eV_per_A"),
        name="center gradients",
        shape=positions.shape,
    )
    center_arrays = {}
    for name in ("source", "reaction_field"):
        values = np.asarray(center.get(name), dtype=float)
        if values.shape != (positions.shape[0], 4) or not np.all(np.isfinite(values)):
            raise ValueError(f"center {name} must be finite with shape (N,4).")
        center_arrays[name] = values
    if (
        not np.array_equal(
            center_arrays["source"][:, 1:],
            np.zeros_like(center_arrays["source"][:, 1:]),
        )
        or abs(float(np.sum(center_arrays["source"][:, 0]))) > 1.0e-10
    ):
        raise ValueError(
            "AIMNet2 source must remain charge-conserving point monopoles only."
        )
    parity = _mapping(
        center.get("aimnet2_source_response_parity"),
        name="AIMNet2 source-response parity",
    )
    required_parity = {
        "energy_absolute_error_eV",
        "charge_max_absolute_error_e",
        "intrinsic_gradient_max_absolute_error_eV_per_A",
        "charge_vjp_max_absolute_error_eV_per_A",
        "repeat_energy_absolute_error_eV",
        "repeat_charge_max_absolute_error_e",
        "repeat_intrinsic_gradient_max_absolute_error_eV_per_A",
        "repeat_charge_vjp_max_absolute_error_eV_per_A",
        "ordinary_forward_intrinsic_gradient_report_only_max_absolute_error_eV_per_A",
    }
    if set(parity) != required_parity or any(
        _finite(value, name=f"source-response parity {name}") < 0.0
        for name, value in parity.items()
    ):
        raise ValueError("AIMNet2 source-response parity ledger is malformed.")
    for name, threshold in FORCE_SOURCE_PARITY_THRESHOLDS.items():
        if _finite(parity[name], name=f"source-response parity {name}") > threshold:
            raise ValueError(
                f"AIMNet2 source-response parity hard gate failed: {name}."
            )
    reciprocity = _mapping(
        center.get("continuum_reciprocity_audit"),
        name="continuum reciprocity audit",
    )
    reciprocity_summary = summarize_geometry_mediated_reciprocity_audit(
        reciprocity,
        reaction_field=center_arrays["reaction_field"],
    )
    if reciprocity_summary["gate_passed"] is not True:
        raise ValueError("continuum reciprocity audit did not pass.")
    center_model_topology = _hex_digest(
        center.get("model_topology_sha256"),
        name="center model topology SHA256",
        length=64,
    )
    if center_model_topology != raw.get("center_model_topology_sha256"):
        raise ValueError("center AIMNet2 topology disagrees with external binding.")
    center_cutoff_margin = _finite(
        center.get("model_minimum_cutoff_margin_angstrom"),
        name="center model cutoff margin",
    )
    if center_cutoff_margin <= 0.0 or center_cutoff_margin != raw.get(
        "center_model_minimum_cutoff_margin_angstrom"
    ):
        raise ValueError("center AIMNet2 cutoff margin is invalid or unbound.")
    radii = np.asarray(task.get("cavity_radii_angstrom"), dtype=float)
    _validate_preflight(
        center.get("continuum_preflight"),
        positions=positions,
        radii=radii,
        name="center continuum preflight",
    )
    center_continuum_topology = _validate_continuum_topology(
        center.get("continuum_topology"),
        expected_configuration_sha256=str(raw["continuum_configuration_sha256"]),
        name="center continuum topology",
    )
    if canonical_sha256(center_continuum_topology) != raw.get(
        "center_continuum_topology_sha256"
    ):
        raise ValueError("center continuum topology disagrees with external binding.")

    directions = _mapping(raw.get("directions"), name="directions")
    expected_vectors = _expected_directions(positions, task, str(raw.get("mode")))
    if set(directions) != set(expected_vectors):
        raise ValueError("force direction names drifted from the deterministic policy.")
    for name, expected_vector in expected_vectors.items():
        direction = _mapping(directions[name], name=f"direction {name}")
        observed_vector = np.asarray(direction.get("vector"), dtype=float)
        if not np.array_equal(observed_vector, expected_vector):
            raise ValueError(
                f"force direction {name!r} drifted from the deterministic policy."
            )
        samples = _sequence(direction.get("samples"), name=f"samples {name}")
        if len(samples) != len(FORCE_PREQUALIFICATION_STEPS_A):
            raise ValueError(f"direction {name!r} has the wrong sample count.")
        for step_index, (step, raw_sample) in enumerate(
            zip(FORCE_PREQUALIFICATION_STEPS_A, samples, strict=True)
        ):
            sample = _mapping(raw_sample, name=f"sample {name}/{step_index}")
            if _finite(sample.get("step_angstrom"), name="step_angstrom") != step:
                raise ValueError("finite-difference step contract changed.")
            for side, sign in (("plus", 1.0), ("minus", -1.0)):
                _energy_leaves(
                    sample.get(f"{side}_energies_eV"),
                    name=f"{side} energies",
                )
                if sample.get(f"{side}_model_topology_sha256") != (
                    center_model_topology
                ):
                    raise ValueError(
                        "AIMNet2 hard topology changed across the stencil."
                    )
                if (
                    _finite(
                        sample.get(f"{side}_model_minimum_cutoff_margin_angstrom"),
                        name=f"{side} model cutoff margin",
                    )
                    <= 0.0
                ):
                    raise ValueError("AIMNet2 cutoff margin is not positive.")
                displaced = positions + sign * step * expected_vector
                _validate_preflight(
                    sample.get(f"{side}_continuum_preflight"),
                    positions=displaced,
                    radii=radii,
                    name=f"{side} continuum preflight {name}/{step_index}",
                )
                _validate_continuum_topology(
                    sample.get(f"{side}_continuum_topology"),
                    expected_configuration_sha256=str(
                        raw["continuum_configuration_sha256"]
                    ),
                    expected_topology=center_continuum_topology,
                    name=f"{side} continuum topology {name}/{step_index}",
                )
    observed_record_sha256 = _hex_digest(
        raw.get("record_sha256"), name="force record SHA256", length=64
    )
    expected_record_sha256 = canonical_sha256(
        {key: value for key, value in raw.items() if key != "record_sha256"}
    )
    if observed_record_sha256 != expected_record_sha256:
        raise ValueError("private force record SHA256 is invalid.")


def _energy_leaves(value: object, *, name: str) -> dict[str, float]:
    raw = _mapping(value, name=name)
    leaves = {
        leaf: _finite(raw.get(leaf), name=f"{name} {leaf}")
        for leaf in FORCE_PREQUALIFICATION_LEAVES
    }
    if not math.isclose(
        leaves["vacuum"] + leaves["continuum"],
        leaves["electrostatic_total"],
        rel_tol=0.0,
        abs_tol=1.0e-10,
    ) or not math.isclose(
        leaves["electrostatic_total"] + leaves["nonpolar"],
        leaves["total"],
        rel_tol=0.0,
        abs_tol=1.0e-10,
    ):
        raise ValueError(f"{name} energy-component ledger does not close.")
    return leaves


def _gradient_leaves(
    value: object, *, name: str, shape: tuple[int, int]
) -> dict[str, np.ndarray]:
    raw = _mapping(value, name=name)
    leaves: dict[str, np.ndarray] = {}
    for leaf in FORCE_PREQUALIFICATION_LEAVES:
        array = np.asarray(raw.get(leaf), dtype=float)
        if array.shape != shape or not np.all(np.isfinite(array)):
            raise ValueError(f"{name} leaf {leaf!r} is invalid.")
        leaves[leaf] = array
    if not np.allclose(
        leaves["vacuum"] + leaves["continuum"],
        leaves["electrostatic_total"],
        rtol=0.0,
        atol=2.0e-10,
    ) or not np.allclose(
        leaves["electrostatic_total"] + leaves["nonpolar"],
        leaves["total"],
        rtol=0.0,
        atol=2.0e-10,
    ):
        raise ValueError(f"{name} gradient-component ledger does not close.")
    return leaves


def _order_gate(errors: Sequence[float]) -> tuple[float | None, bool]:
    """Return a finite order, or None for a plateau/zero endpoint, and its gate.

    Raw errors and the explicit gate retain the zero-endpoint distinction;
    infinities must not enter hash-bound JSON evidence.
    """
    threshold = FORCE_PREQUALIFICATION_THRESHOLDS
    maximum = max(errors)
    if maximum <= threshold["numerical_plateau_eV_per_A"]:
        return None, True
    first, middle, terminal = errors
    if terminal == 0.0:
        if first == 0.0:
            return None, bool(middle <= threshold["numerical_plateau_eV_per_A"])
        middle_growth = middle <= max(
            threshold["terminal_growth_maximum"] * first,
            threshold["numerical_plateau_eV_per_A"],
        )
        return None, bool(middle_growth)
    if first == 0.0:
        return None, False
    order = (math.log(first) - math.log(terminal)) / math.log(4.0)
    growth = terminal <= max(
        threshold["terminal_growth_maximum"] * errors[1],
        threshold["numerical_plateau_eV_per_A"],
    )
    return order, bool(order >= threshold["observed_central_order_minimum"] and growth)


def _summarize_force_record(raw: Mapping[str, object]) -> dict[str, object]:
    """Reduce one already-validated directional or Cartesian force shard."""

    mode = raw.get("mode")
    if mode not in {"directional", "full-cartesian"}:
        raise ValueError("force record mode must be directional or full-cartesian.")
    center = _mapping(raw.get("center"), name="center")
    positions = _positions(center.get("positions_angstrom"))
    _energy_leaves(center.get("energies_eV"), name="center energies")
    gradients = _gradient_leaves(
        center.get("gradients_eV_per_A"),
        name="center gradients",
        shape=positions.shape,
    )
    source = np.asarray(center.get("source"), dtype=float)
    if source.shape != (positions.shape[0], 4) or not np.all(np.isfinite(source)):
        raise ValueError("center source must be finite with shape (N,4).")
    center_model_topology = str(center.get("model_topology_sha256", ""))
    if len(center_model_topology) != 64:
        raise ValueError("center model topology SHA256 is missing.")
    center_preflight = _mapping(
        center.get("continuum_preflight"), name="center continuum preflight"
    )
    center_degree = _positive_integer(
        center_preflight.get("required_algebraic_degree"),
        name="center required algebraic degree",
    )

    directions = _mapping(raw.get("directions"), name="directions")
    if not directions:
        raise ValueError("force record must contain at least one direction.")
    expected_direction_count = positions.size if mode == "full-cartesian" else 3
    if len(directions) != expected_direction_count:
        raise ValueError(
            f"{mode} force record requires {expected_direction_count} directions."
        )
    per_direction: dict[str, object] = {}
    aggregate: dict[str, list[list[float]]] = {
        leaf: [[] for _step in FORCE_PREQUALIFICATION_STEPS_A]
        for leaf in FORCE_PREQUALIFICATION_LEAVES
    }
    relative_terminal = {leaf: [] for leaf in FORCE_PREQUALIFICATION_LEAVES}
    topology_matches = True
    degree_within_bound = True
    per_direction_order_gates: list[bool] = []

    for name, raw_direction in directions.items():
        direction_record = _mapping(raw_direction, name=f"direction {name}")
        vector = np.asarray(direction_record.get("vector"), dtype=float)
        if vector.shape != positions.shape or not np.all(np.isfinite(vector)):
            raise ValueError(f"direction {name!r} vector is invalid.")
        if not np.isclose(np.linalg.norm(vector), 1.0, rtol=0.0, atol=1.0e-12):
            raise ValueError(f"direction {name!r} must be normalized.")
        samples = _sequence(direction_record.get("samples"), name=f"samples {name}")
        if len(samples) != len(FORCE_PREQUALIFICATION_STEPS_A):
            raise ValueError(f"direction {name!r} has the wrong sample count.")
        leaf_records: dict[str, object] = {}
        for leaf in FORCE_PREQUALIFICATION_LEAVES:
            analytic = float(np.sum(np.asarray(gradients[leaf]) * vector))
            errors = []
            finite_differences = []
            for step_index, (expected_step, raw_sample) in enumerate(
                zip(FORCE_PREQUALIFICATION_STEPS_A, samples, strict=True)
            ):
                sample = _mapping(raw_sample, name=f"sample {name}/{step_index}")
                step = _finite(sample.get("step_angstrom"), name="step_angstrom")
                if step != expected_step:
                    raise ValueError("finite-difference step contract changed.")
                plus = _energy_leaves(
                    sample.get("plus_energies_eV"), name="plus energies"
                )
                minus = _energy_leaves(
                    sample.get("minus_energies_eV"), name="minus energies"
                )
                fd = (
                    _finite(plus.get(leaf), name=f"plus {leaf}")
                    - _finite(minus.get(leaf), name=f"minus {leaf}")
                ) / (2.0 * step)
                error = abs(fd - analytic)
                finite_differences.append(fd)
                errors.append(error)
                aggregate[leaf][step_index].append(error)
                topology_matches &= (
                    sample.get("plus_model_topology_sha256")
                    == center_model_topology
                    == sample.get("minus_model_topology_sha256")
                )
                for side in ("plus", "minus"):
                    topology = _mapping(
                        sample.get(f"{side}_continuum_preflight"),
                        name=f"{side} continuum preflight",
                    )
                    degree_within_bound &= _positive_integer(
                        topology.get("required_algebraic_degree"),
                        name=f"{side} required algebraic degree",
                    ) <= int(
                        FORCE_PREQUALIFICATION_THRESHOLDS["maximum_algebraic_degree"]
                    )
            terminal_denominator = abs(finite_differences[-1])
            terminal_relative = (
                errors[-1] / terminal_denominator
                if terminal_denominator
                >= FORCE_PREQUALIFICATION_THRESHOLDS[
                    "directional_relative_denominator_floor_eV_per_A"
                ]
                else None
            )
            if terminal_relative is not None:
                relative_terminal[leaf].append(terminal_relative)
            order, order_passed = _order_gate(errors)
            per_direction_order_gates.append(order_passed)
            leaf_records[leaf] = {
                "analytic_eV_per_A": analytic,
                "finite_difference_eV_per_A": finite_differences,
                "absolute_errors_eV_per_A": errors,
                "terminal_relative_error": terminal_relative,
                "observed_first_to_last_order": order,
                "order_or_plateau_passed": order_passed,
            }
        per_direction[str(name)] = leaf_records

    aggregate_summary = {}
    derivative_gates = []
    for leaf, step_values in aggregate.items():
        rms = [float(np.sqrt(np.mean(np.square(values)))) for values in step_values]
        maximum = [float(np.max(values)) for values in step_values]
        order, order_passed = _order_gate(rms)
        relative_max = max(relative_terminal[leaf], default=0.0)
        if mode == "directional":
            leaf_gate = bool(
                max(maximum)
                <= FORCE_PREQUALIFICATION_THRESHOLDS[
                    "directional_all_step_max_absolute_eV_per_A"
                ]
                and maximum[-1]
                <= FORCE_PREQUALIFICATION_THRESHOLDS[
                    "directional_terminal_absolute_eV_per_A"
                ]
                and relative_max
                <= FORCE_PREQUALIFICATION_THRESHOLDS["directional_terminal_relative"]
                and order_passed
            )
        else:
            leaf_gate = bool(
                max(maximum)
                <= FORCE_PREQUALIFICATION_THRESHOLDS["cartesian_step_max_eV_per_A"]
                and max(rms)
                <= FORCE_PREQUALIFICATION_THRESHOLDS["cartesian_step_rms_eV_per_A"]
                and order_passed
            )
        derivative_gates.append(leaf_gate)
        aggregate_summary[leaf] = {
            "rms_absolute_error_eV_per_A": rms,
            "maximum_absolute_error_eV_per_A": maximum,
            "maximum_terminal_relative_error": relative_max,
            "observed_first_to_last_rms_order": order,
            "gate_passed": leaf_gate,
        }

    replay = _mapping(raw.get("deterministic_replay"), name="deterministic replay")
    replay_passed = bool(
        _nonnegative_finite(
            replay.get("energy_absolute_error_eV"), name="replay energy"
        )
        <= FORCE_PREQUALIFICATION_THRESHOLDS["replay_energy_absolute_eV"]
        and _nonnegative_finite(
            replay.get("source_max_absolute_error_e"), name="replay source"
        )
        <= FORCE_PREQUALIFICATION_THRESHOLDS["replay_source_max_absolute_e"]
        and _nonnegative_finite(
            replay.get("gradient_max_absolute_error_eV_per_A"),
            name="replay gradient",
        )
        <= FORCE_PREQUALIFICATION_THRESHOLDS["replay_gradient_max_absolute_eV_per_A"]
        and _nonnegative_finite(
            replay.get("stencil_energy_max_absolute_error_eV"),
            name="replay stencil energy",
        )
        <= FORCE_PREQUALIFICATION_THRESHOLDS["replay_energy_absolute_eV"]
    )
    total_gradient = np.asarray(gradients["total"], dtype=float)
    centered = positions - np.mean(positions, axis=0)
    net_gradient = float(np.linalg.norm(np.sum(total_gradient, axis=0)))
    torque = float(np.linalg.norm(np.sum(np.cross(centered, total_gradient), axis=0)))
    rigid_passed = bool(
        net_gradient <= FORCE_PREQUALIFICATION_THRESHOLDS["net_gradient_norm_eV_per_A"]
        and torque <= FORCE_PREQUALIFICATION_THRESHOLDS["torque_norm_eV"]
    )
    gates = {
        "aimnet2_source_response_parity": True,
        "deterministic_replay": replay_passed,
        "same_scalar_derivatives_all_leaves": all(derivative_gates),
        "every_direction_order_or_plateau": all(per_direction_order_gates),
        "model_hard_topology_unchanged": topology_matches,
        "continuum_algebraic_degree_within_bound": bool(
            degree_within_bound
            and center_degree
            <= FORCE_PREQUALIFICATION_THRESHOLDS["maximum_algebraic_degree"]
        ),
        "analytic_translation_and_torque": rigid_passed,
    }
    return {
        "contract_version": FORCE_PREQUALIFICATION_CONTRACT_VERSION,
        "mode": mode,
        "direction_count": len(directions),
        "steps_angstrom": list(FORCE_PREQUALIFICATION_STEPS_A),
        "per_direction": per_direction,
        "aggregate_derivative_errors": aggregate_summary,
        "net_gradient_norm_eV_per_A": net_gradient,
        "torque_norm_eV": torque,
        "gates": gates,
        "all_gates_passed": all(gates.values()),
        "capabilities": dict(NO_CAPABILITIES),
        "claim_boundary": (
            "One private first-derivative prequalification shard only; no public "
            "E/F/H/V/M or daily-task admission."
        ),
    }


def _replicate_identity(value: Mapping[str, object]) -> object:
    payload = copy.deepcopy(dict(value))
    for key in (
        "replicate",
        "runtime_seconds",
        "record_sha256",
        "deterministic_replay",
        "summary",
        "process_identity",
    ):
        payload.pop(key, None)
    center = payload.get("center")
    if isinstance(center, dict):
        center.pop("runtime_seconds", None)
    return payload


def finalize_force_replicates(
    primary: Mapping[str, object],
    replay: Mapping[str, object],
    *,
    expected_bindings: Mapping[str, object],
) -> dict[str, object]:
    """Bind two cold-process measurements and reduce the primary record."""

    validate_force_measurement_record(
        primary,
        expected_bindings=expected_bindings,
        replicate="primary",
    )
    validate_force_measurement_record(
        replay,
        expected_bindings=expected_bindings,
        replicate="replay",
    )
    if primary.get("replicate") != "primary" or replay.get("replicate") != "replay":
        raise ValueError("replicates must be labelled primary and replay.")
    primary_process = _mapping(
        primary.get("process_identity"), name="primary process identity"
    )
    replay_process = _mapping(
        replay.get("process_identity"), name="replay process identity"
    )
    if primary_process.get("process_import_token") == replay_process.get(
        "process_import_token"
    ) or primary_process.get("pid") == replay_process.get("pid"):
        raise ValueError("force replay must execute in a distinct cold process.")
    primary_identity = _replicate_identity(primary)
    replay_identity = _replicate_identity(replay)

    # Numerical arrays are compared below.  Remove them before checking that
    # the two cold processes executed the exact same task and stencil.
    def strip_measurements(payload: object) -> object:
        result = copy.deepcopy(payload)
        if not isinstance(result, dict):
            raise TypeError("replicate must be a mapping.")
        center = result.get("center")
        if isinstance(center, dict):
            for key in ("energies_eV", "gradients_eV_per_A", "source"):
                center.pop(key, None)
        directions = result.get("directions")
        if isinstance(directions, dict):
            for direction in directions.values():
                if not isinstance(direction, dict):
                    continue
                samples = direction.get("samples")
                if not isinstance(samples, list):
                    continue
                for sample in samples:
                    if isinstance(sample, dict):
                        sample.pop("plus_energies_eV", None)
                        sample.pop("minus_energies_eV", None)
        return result

    if canonical_sha256(strip_measurements(primary_identity)) != canonical_sha256(
        strip_measurements(replay_identity)
    ):
        raise ValueError("cold force replicates do not describe the same task/stencil.")

    primary_center = _mapping(primary.get("center"), name="primary center")
    replay_center = _mapping(replay.get("center"), name="replay center")
    energy_error = 0.0
    for leaf in FORCE_PREQUALIFICATION_LEAVES:
        energy_error = max(
            energy_error,
            abs(
                _finite(
                    _mapping(
                        primary_center.get("energies_eV"), name="primary energies"
                    ).get(leaf),
                    name=f"primary {leaf}",
                )
                - _finite(
                    _mapping(
                        replay_center.get("energies_eV"), name="replay energies"
                    ).get(leaf),
                    name=f"replay {leaf}",
                )
            ),
        )
    primary_source = np.asarray(primary_center.get("source"), dtype=float)
    replay_source = np.asarray(replay_center.get("source"), dtype=float)
    if (
        primary_source.shape != replay_source.shape
        or not np.all(np.isfinite(primary_source))
        or not np.all(np.isfinite(replay_source))
    ):
        raise ValueError("cold replay source arrays are invalid.")
    source_error = float(np.max(np.abs(primary_source - replay_source)))
    gradient_error = 0.0
    primary_gradients = _mapping(
        primary_center.get("gradients_eV_per_A"), name="primary gradients"
    )
    replay_gradients = _mapping(
        replay_center.get("gradients_eV_per_A"), name="replay gradients"
    )
    for leaf in FORCE_PREQUALIFICATION_LEAVES:
        first = np.asarray(primary_gradients.get(leaf), dtype=float)
        second = np.asarray(replay_gradients.get(leaf), dtype=float)
        if (
            first.shape != second.shape
            or not np.all(np.isfinite(first))
            or not np.all(np.isfinite(second))
        ):
            raise ValueError(f"cold replay gradient leaf {leaf!r} is invalid.")
        gradient_error = max(gradient_error, float(np.max(np.abs(first - second))))

    stencil_energy_error = 0.0
    primary_directions = _mapping(primary.get("directions"), name="primary directions")
    replay_directions = _mapping(replay.get("directions"), name="replay directions")
    for name, primary_direction in primary_directions.items():
        replay_direction = _mapping(
            replay_directions.get(name), name=f"replay direction {name}"
        )
        primary_samples = _sequence(
            _mapping(primary_direction, name=f"primary direction {name}").get(
                "samples"
            ),
            name=f"primary samples {name}",
        )
        replay_samples = _sequence(
            replay_direction.get("samples"), name=f"replay samples {name}"
        )
        for first, second in zip(primary_samples, replay_samples, strict=True):
            first_sample = _mapping(first, name="primary sample")
            second_sample = _mapping(second, name="replay sample")
            for side in ("plus", "minus"):
                first_energies = _mapping(
                    first_sample.get(f"{side}_energies_eV"),
                    name="primary stencil energies",
                )
                second_energies = _mapping(
                    second_sample.get(f"{side}_energies_eV"),
                    name="replay stencil energies",
                )
                for leaf in FORCE_PREQUALIFICATION_LEAVES:
                    stencil_energy_error = max(
                        stencil_energy_error,
                        abs(
                            _finite(first_energies.get(leaf), name=f"primary {leaf}")
                            - _finite(second_energies.get(leaf), name=f"replay {leaf}")
                        ),
                    )

    finalized = copy.deepcopy(dict(primary))
    finalized["status"] = FORCE_FINALIZED_STATUS
    finalized["replicate"] = "primary-with-cold-replay"
    finalized["deterministic_replay"] = {
        "energy_absolute_error_eV": energy_error,
        "source_max_absolute_error_e": source_error,
        "gradient_max_absolute_error_eV_per_A": gradient_error,
        "stencil_energy_max_absolute_error_eV": stencil_energy_error,
        "primary_record_sha256": primary.get("record_sha256"),
        "replay_record_sha256": replay.get("record_sha256"),
        "primary_process_identity": dict(primary_process),
        "replay_process_identity": dict(replay_process),
    }
    finalized["summary"] = _summarize_force_record(finalized)
    finalized["record_sha256"] = canonical_sha256(
        {key: value for key, value in finalized.items() if key != "record_sha256"}
    )
    return finalized


def summarize_force_record(
    finalized: Mapping[str, object],
    *,
    replay_measurement: Mapping[str, object],
    expected_bindings: Mapping[str, object],
) -> dict[str, object]:
    """Verify and independently re-reduce one finalized private force shard."""

    if (
        finalized.get("artifact") != FORCE_MEASUREMENT_ARTIFACT
        or finalized.get("schema_version") != FORCE_MEASUREMENT_SCHEMA_VERSION
        or finalized.get("status") != FORCE_FINALIZED_STATUS
        or finalized.get("replicate") != "primary-with-cold-replay"
        or finalized.get("do_not_commit") is not True
        or finalized.get("capabilities") != NO_CAPABILITIES
    ):
        raise ValueError("finalized force artifact identity/status drifted.")
    observed_sha256 = _hex_digest(
        finalized.get("record_sha256"),
        name="finalized force record SHA256",
        length=64,
    )
    expected_sha256 = canonical_sha256(
        {key: value for key, value in finalized.items() if key != "record_sha256"}
    )
    if observed_sha256 != expected_sha256:
        raise ValueError("finalized force record SHA256 is invalid.")
    replay = _mapping(
        finalized.get("deterministic_replay"), name="deterministic replay"
    )
    for key in (
        "energy_absolute_error_eV",
        "source_max_absolute_error_e",
        "gradient_max_absolute_error_eV_per_A",
        "stencil_energy_max_absolute_error_eV",
    ):
        _nonnegative_finite(replay.get(key), name=f"deterministic replay {key}")
    primary_sha256 = _hex_digest(
        replay.get("primary_record_sha256"),
        name="primary force record SHA256",
        length=64,
    )
    replay_sha256 = _hex_digest(
        replay.get("replay_record_sha256"),
        name="replay force record SHA256",
        length=64,
    )
    primary = copy.deepcopy(dict(finalized))
    primary["status"] = FORCE_MEASUREMENT_STATUS
    primary["replicate"] = "primary"
    primary["record_sha256"] = primary_sha256
    primary.pop("deterministic_replay", None)
    primary.pop("summary", None)
    validate_force_measurement_record(
        primary,
        expected_bindings=expected_bindings,
        replicate="primary",
    )
    validate_force_measurement_record(
        replay_measurement,
        expected_bindings=expected_bindings,
        replicate="replay",
    )
    if replay_measurement.get("record_sha256") != replay_sha256:
        raise ValueError("finalized replay SHA256 does not match the supplied replay.")
    independently_finalized = finalize_force_replicates(
        primary,
        replay_measurement,
        expected_bindings=expected_bindings,
    )
    if canonical_sha256(finalized) != canonical_sha256(independently_finalized):
        raise ValueError(
            "finalized force artifact does not match independent cold-replay reduction."
        )
    return dict(_mapping(independently_finalized.get("summary"), name="force summary"))


__all__ = [
    "FORCE_FINALIZED_STATUS",
    "FORCE_MEASUREMENT_ARTIFACT",
    "FORCE_MEASUREMENT_CLAIM_BOUNDARY",
    "FORCE_MEASUREMENT_SCHEMA_VERSION",
    "FORCE_MEASUREMENT_STATUS",
    "FORCE_PREQUALIFICATION_CONTRACT_VERSION",
    "FORCE_PREQUALIFICATION_LEAVES",
    "FORCE_PREQUALIFICATION_STEPS_A",
    "FORCE_PREQUALIFICATION_THRESHOLDS",
    "FORCE_SOURCE_PARITY_THRESHOLDS",
    "NO_CAPABILITIES",
    "canonical_sha256",
    "cartesian_directions",
    "finalize_force_replicates",
    "geometry_internal_directions",
    "summarize_force_record",
    "topology_preflight",
    "validate_force_measurement_record",
]
