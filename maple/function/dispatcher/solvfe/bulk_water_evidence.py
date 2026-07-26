from __future__ import annotations

import math
from typing import Any, Mapping, TYPE_CHECKING

import numpy as np
from ase import Atoms, units
from ase.geometry import find_mic

from .bulk_water import (
    BulkWaterValidationError,
    _blocked_rdf,
    _immutable_array,
    _oxygen_rdf_features,
    _rdf_histograms,
)
from .protocol import canonical_sha256

if TYPE_CHECKING:
    from .bulk_water_npt import BulkWaterNPTConfig


_AMU_PER_ANGSTROM3_TO_G_PER_ML = 1.66053906660
_OBSERVATION_FLOAT_NAMES = (
    "time_fs",
    "temperature_k",
    "potential_energy_ev",
    "kinetic_energy_ev",
    "total_energy_ev",
    "force_rms_ev_per_angstrom",
    "force_max_ev_per_angstrom",
    "pressure_bar",
    "volume_angstrom3",
    "density_g_per_ml",
)
_STEPWISE_FLOAT_NAMES = (
    "temperature_k",
    "potential_energy_ev",
    "velocity_max_angstrom_per_ase_time",
    "force_max_ev_per_angstrom",
    "oh_min_distance_angstrom",
    "oh_max_distance_angstrom",
    "hoh_min_angle_degrees",
    "hoh_max_angle_degrees",
    "minimum_cell_height_angstrom",
    "maximum_cell_height_angstrom",
    "volume_angstrom3",
    "density_g_per_ml",
)
_RDF_NAMES = (
    "rdf_r_angstrom",
    "rdf_oo_g",
    "rdf_oo_sem",
    "rdf_oh_g",
    "rdf_oh_sem",
    "rdf_hh_g",
    "rdf_hh_sem",
)
NPT_ARRAY_NAMES = frozenset(
    {
        "observation_stage",
        "observation_step",
        *(
            f"observation_{name}"
            for name in _OBSERVATION_FLOAT_NAMES
        ),
        "atomic_numbers",
        "production_positions_angstrom",
        "production_cells_angstrom",
        "production_velocities_angstrom_per_ase_time",
        "production_density_g_per_ml",
        "production_density_block_g_per_ml",
        "stepwise_stage_code",
        "stepwise_step",
        *(
            f"stepwise_{name}"
            for name in _STEPWISE_FLOAT_NAMES
        ),
        *_RDF_NAMES,
    }
)
_SINGLE_REPLICA_INTERPRETATION = (
    "One ASE-MTK NPT trajectory can establish engineering and "
    "single-replica density diagnostics only. It cannot reproduce "
    "the OpenMM Monte-Carlo-barostat paper protocol, freeze the "
    "Hamiltonian, or support Route A accuracy claims."
)


def _invalid(message: str) -> BulkWaterValidationError:
    return BulkWaterValidationError(
        f"CAMPAIGN_ARRAY_EVIDENCE_INVALID: {message}"
    )


def _array(
    arrays: Mapping[str, np.ndarray],
    name: str,
    *,
    dtype: np.dtype | None = None,
    shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    value = np.asarray(arrays[name])
    if dtype is not None and value.dtype != np.dtype(dtype):
        raise _invalid(
            f"{name} must have dtype {np.dtype(dtype)}, found {value.dtype}."
        )
    if shape is not None and value.shape != shape:
        raise _invalid(
            f"{name} must have shape {shape}, found {value.shape}."
        )
    return value


def _expected_observations(
    config: BulkWaterNPTConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    stages = ["initial"]
    steps = [0]
    times = [0.0]
    if config.precondition_steps:
        precondition_steps = np.arange(
            config.sample_interval_steps,
            config.precondition_steps + 1,
            config.sample_interval_steps,
            dtype=np.int64,
        )
        stages.extend(["precondition"] * len(precondition_steps))
        steps.extend(precondition_steps.tolist())
        times.extend(
            (
                precondition_steps
                * config.precondition_timestep_fs
            ).tolist()
        )
    dynamics_step = 0
    base_step = config.precondition_steps
    base_time = (
        config.precondition_steps
        * config.precondition_timestep_fs
    )
    for stage, stage_steps in (
        ("equilibration", config.equilibration_steps),
        ("production", config.production_steps),
    ):
        sampled = np.arange(
            config.sample_interval_steps,
            stage_steps + 1,
            config.sample_interval_steps,
            dtype=np.int64,
        )
        stages.extend([stage] * len(sampled))
        steps.extend((base_step + dynamics_step + sampled).tolist())
        times.extend(
            (
                base_time
                + (dynamics_step + sampled) * config.timestep_fs
            ).tolist()
        )
        dynamics_step += stage_steps
    return (
        np.asarray(stages, dtype="U16"),
        np.asarray(steps, dtype=np.int64),
        np.asarray(times, dtype=np.float64),
    )


def _expected_stepwise_stage_codes(
    config: BulkWaterNPTConfig,
) -> np.ndarray:
    return np.concatenate(
        (
            np.zeros(1, dtype=np.uint8),
            np.full(
                config.precondition_steps,
                1,
                dtype=np.uint8,
            ),
            np.full(
                config.equilibration_steps,
                2,
                dtype=np.uint8,
            ),
            np.full(
                config.production_steps,
                3,
                dtype=np.uint8,
            ),
        )
    )


def _cell_heights(cells: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    volumes = np.abs(np.linalg.det(cells))
    if not np.all(np.isfinite(volumes)) or np.any(volumes <= 0.0):
        raise _invalid("production cells must have finite positive volume.")
    heights = np.empty((len(cells), 3), dtype=np.float64)
    for axis in range(3):
        other = [index for index in range(3) if index != axis]
        face_areas = np.linalg.norm(
            np.cross(cells[:, other[0]], cells[:, other[1]]),
            axis=1,
        )
        if (
            not np.all(np.isfinite(face_areas))
            or np.any(face_areas <= 0.0)
        ):
            raise _invalid("production cells contain a degenerate face.")
        heights[:, axis] = volumes / face_areas
    return np.min(heights, axis=1), np.max(heights, axis=1)


def _same_numeric(
    actual: np.ndarray,
    expected: np.ndarray,
    *,
    label: str,
) -> None:
    if not np.allclose(
        actual,
        expected,
        rtol=1.0e-12,
        atol=1.0e-12,
    ):
        raise _invalid(f"{label} conflicts with retained numeric evidence.")


def _production_physical_state(
    *,
    numbers: np.ndarray,
    positions: np.ndarray,
    cells: np.ndarray,
    velocities: np.ndarray,
) -> dict[str, np.ndarray]:
    """Recompute kinetic and intramolecular evidence from retained states."""

    frame_count, atom_count, _ = positions.shape
    water_count = atom_count // 3
    masses = np.asarray(
        Atoms(numbers=numbers).get_masses(),
        dtype=np.float64,
    )
    kinetic_energy = 0.5 * np.einsum(
        "a,fai,fai->f",
        masses,
        velocities,
        velocities,
    )
    temperature = (
        2.0 * kinetic_energy
        / (3.0 * atom_count * units.kB)
    )
    velocity_max = np.max(
        np.linalg.norm(velocities, axis=2),
        axis=1,
    )
    oh_min = np.empty(frame_count, dtype=np.float64)
    oh_max = np.empty(frame_count, dtype=np.float64)
    hoh_min = np.empty(frame_count, dtype=np.float64)
    hoh_max = np.empty(frame_count, dtype=np.float64)
    for index, (frame_positions, frame_cell) in enumerate(
        zip(positions, cells, strict=True)
    ):
        oh_vectors = np.concatenate(
            (
                frame_positions[1::3] - frame_positions[0::3],
                frame_positions[2::3] - frame_positions[0::3],
            ),
            axis=0,
        )
        mic_vectors, oh_distances = find_mic(
            oh_vectors,
            frame_cell,
            pbc=True,
        )
        first = mic_vectors[:water_count]
        second = mic_vectors[water_count:]
        first_norm = np.linalg.norm(first, axis=1)
        second_norm = np.linalg.norm(second, axis=1)
        if np.any(first_norm <= 0.0) or np.any(second_norm <= 0.0):
            raise _invalid(
                "production positions contain a zero-length O-H bond."
            )
        cosine = np.sum(first * second, axis=1) / (
            first_norm * second_norm
        )
        angles = np.degrees(
            np.arccos(np.clip(cosine, -1.0, 1.0))
        )
        if (
            not np.all(np.isfinite(oh_distances))
            or not np.all(np.isfinite(angles))
        ):
            raise _invalid(
                "production positions do not define finite water topology."
            )
        oh_min[index] = np.min(oh_distances)
        oh_max[index] = np.max(oh_distances)
        hoh_min[index] = np.min(angles)
        hoh_max[index] = np.max(angles)
    return {
        "kinetic_energy_ev": kinetic_energy,
        "temperature_k": temperature,
        "velocity_max_angstrom_per_ase_time": velocity_max,
        "oh_min_distance_angstrom": oh_min,
        "oh_max_distance_angstrom": oh_max,
        "hoh_min_angle_degrees": hoh_min,
        "hoh_max_angle_degrees": hoh_max,
    }


def compute_bulk_water_rdf_evidence(
    positions: np.ndarray,
    cells: np.ndarray,
    numbers: np.ndarray,
    *,
    config: BulkWaterNPTConfig,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Derive the complete RDF artifact from retained production frames."""

    positions = np.asarray(positions, dtype=np.float64)
    cells = np.asarray(cells, dtype=np.float64)
    numbers = np.asarray(numbers, dtype=np.int64)
    if (
        positions.ndim != 3
        or positions.shape[2:] != (3,)
        or cells.shape != (len(positions), 3, 3)
        or numbers.shape != (positions.shape[1],)
    ):
        raise _invalid(
            "RDF inputs must contain matching frame, cell, and atom shapes."
        )
    if (
        len(positions) == 0
        or len(numbers) == 0
        or len(numbers) % 3
        or not np.array_equal(
            numbers.reshape((-1, 3)),
            np.tile(
                np.array([[8, 1, 1]], dtype=np.int64),
                (len(numbers) // 3, 1),
            ),
        )
    ):
        raise _invalid(
            "RDF inputs must contain retained contiguous O-H-H waters."
        )
    if (
        not np.all(np.isfinite(positions))
        or not np.all(np.isfinite(cells))
    ):
        raise _invalid("RDF positions and cells must be finite.")

    edges = np.arange(
        0.0,
        config.rdf_max_angstrom,
        config.rdf_bin_width_angstrom,
        dtype=np.float64,
    )
    edges = np.append(
        edges[edges < config.rdf_max_angstrom],
        config.rdf_max_angstrom,
    )
    radii = 0.5 * (edges[1:] + edges[:-1])
    molecule_ids = np.repeat(
        np.arange(len(numbers) // 3, dtype=np.int64),
        3,
    )
    arrays: dict[str, np.ndarray] = {
        "rdf_r_angstrom": _immutable_array(
            radii,
            dtype=np.float64,
        ),
    }
    summary: dict[str, Any] = {}
    for label, atomic_numbers in {
        "oo": (8, 8),
        "oh": (8, 1),
        "hh": (1, 1),
    }.items():
        histograms, normalizations = _rdf_histograms(
            positions,
            cells,
            numbers=numbers,
            molecule_ids=molecule_ids,
            first_atomic_number=atomic_numbers[0],
            second_atomic_number=atomic_numbers[1],
            edges=edges,
        )
        rdf, standard_error, block_count = _blocked_rdf(
            histograms,
            normalizations,
            block_count=config.rdf_block_count,
        )
        arrays[f"rdf_{label}_g"] = _immutable_array(
            rdf,
            dtype=np.float64,
        )
        arrays[f"rdf_{label}_sem"] = _immutable_array(
            standard_error,
            dtype=np.float64,
        )
        summary[label] = {
            "pair": list(atomic_numbers),
            "intermolecular_only": True,
            "block_count": block_count,
        }
    summary["oo_features"] = _oxygen_rdf_features(
        radii,
        arrays["rdf_oo_g"],
        edges,
        cells,
        oxygen_count=len(numbers) // 3,
    )
    return arrays, summary


def recompute_bulk_water_npt_evidence(
    arrays: Mapping[str, np.ndarray],
    *,
    config: BulkWaterNPTConfig,
    water_count: int,
) -> dict[str, Any]:
    """Validate the exact NPT array contract and recompute every campaign gate."""

    if set(arrays) != NPT_ARRAY_NAMES:
        missing = sorted(NPT_ARRAY_NAMES - arrays.keys())
        unexpected = sorted(arrays.keys() - NPT_ARRAY_NAMES)
        raise _invalid(
            f"array schema differs; missing={missing}, unexpected={unexpected}."
        )
    if type(water_count) is not int or water_count <= 0:
        raise _invalid("water_count must be a positive integer.")
    atom_count = 3 * water_count
    frame_count = config.production_frame_count
    expected_stages, expected_steps, expected_times = (
        _expected_observations(config)
    )
    observation_count = len(expected_stages)

    stages = np.asarray(arrays["observation_stage"])
    if stages.dtype.kind != "U" or stages.shape != (observation_count,):
        raise _invalid(
            "observation_stage must be a one-dimensional Unicode array."
        )
    if not np.array_equal(stages, expected_stages):
        raise _invalid("observation stages do not match the NPT protocol.")
    steps = _array(
        arrays,
        "observation_step",
        dtype=np.int64,
        shape=(observation_count,),
    )
    if not np.array_equal(steps, expected_steps):
        raise _invalid("observation steps do not match the NPT protocol.")
    observation_values = {
        name: _array(
            arrays,
            f"observation_{name}",
            dtype=np.float64,
            shape=(observation_count,),
        )
        for name in _OBSERVATION_FLOAT_NAMES
    }
    _same_numeric(
        observation_values["time_fs"],
        expected_times,
        label="observation time",
    )

    numbers = _array(
        arrays,
        "atomic_numbers",
        dtype=np.int64,
        shape=(atom_count,),
    )
    if not np.array_equal(
        numbers.reshape((-1, 3)),
        np.tile(np.array([[8, 1, 1]], dtype=np.int64), (water_count, 1)),
    ):
        raise _invalid("atomic_numbers must contain contiguous O-H-H waters.")
    positions = _array(
        arrays,
        "production_positions_angstrom",
        dtype=np.float64,
        shape=(frame_count, atom_count, 3),
    )
    cells = _array(
        arrays,
        "production_cells_angstrom",
        dtype=np.float64,
        shape=(frame_count, 3, 3),
    )
    velocities = _array(
        arrays,
        "production_velocities_angstrom_per_ase_time",
        dtype=np.float64,
        shape=(frame_count, atom_count, 3),
    )
    production_density = _array(
        arrays,
        "production_density_g_per_ml",
        dtype=np.float64,
        shape=(frame_count,),
    )
    density_blocks = _array(
        arrays,
        "production_density_block_g_per_ml",
        dtype=np.float64,
        shape=(config.rdf_block_count,),
    )

    recomputed_rdf_arrays, rdf_summary = compute_bulk_water_rdf_evidence(
        positions,
        cells,
        numbers,
        config=config,
    )
    for name, expected in recomputed_rdf_arrays.items():
        actual = _array(
            arrays,
            name,
            dtype=np.float64,
            shape=expected.shape,
        )
        _same_numeric(
            actual,
            expected,
            label=name,
        )

    total_steps = (
        config.precondition_steps
        + config.equilibration_steps
        + config.production_steps
    )
    stepwise_count = total_steps + 1
    stage_codes = _array(
        arrays,
        "stepwise_stage_code",
        dtype=np.uint8,
        shape=(stepwise_count,),
    )
    expected_stage_codes = _expected_stepwise_stage_codes(config)
    if not np.array_equal(stage_codes, expected_stage_codes):
        raise _invalid("stepwise stages do not match the NPT protocol.")
    stepwise_steps = _array(
        arrays,
        "stepwise_step",
        dtype=np.int64,
        shape=(stepwise_count,),
    )
    if not np.array_equal(
        stepwise_steps,
        np.arange(stepwise_count, dtype=np.int64),
    ):
        raise _invalid("stepwise evidence does not cover every MD step once.")
    stepwise = {
        name: _array(
            arrays,
            f"stepwise_{name}",
            dtype=np.float64,
            shape=(stepwise_count,),
        )
        for name in _STEPWISE_FLOAT_NAMES
    }

    numeric_arrays = [
        value
        for name, value in arrays.items()
        if name not in {"observation_stage"}
    ]
    if not all(np.all(np.isfinite(value)) for value in numeric_arrays):
        raise _invalid("all retained numeric evidence must be finite.")
    if (
        np.any(observation_values["temperature_k"] < 0.0)
        or np.any(observation_values["kinetic_energy_ev"] < 0.0)
        or np.any(
            observation_values["force_rms_ev_per_angstrom"] < 0.0
        )
        or np.any(
            observation_values["force_max_ev_per_angstrom"] < 0.0
        )
        or np.any(observation_values["volume_angstrom3"] <= 0.0)
        or np.any(observation_values["density_g_per_ml"] <= 0.0)
        or np.any(stepwise["temperature_k"] < 0.0)
        or np.any(
            stepwise["velocity_max_angstrom_per_ase_time"] < 0.0
        )
        or np.any(stepwise["force_max_ev_per_angstrom"] < 0.0)
        or np.any(stepwise["minimum_cell_height_angstrom"] <= 0.0)
        or np.any(
            stepwise["maximum_cell_height_angstrom"]
            < stepwise["minimum_cell_height_angstrom"]
        )
        or np.any(stepwise["volume_angstrom3"] <= 0.0)
        or np.any(stepwise["density_g_per_ml"] <= 0.0)
        or np.any(production_density <= 0.0)
        or np.any(density_blocks <= 0.0)
    ):
        raise _invalid("physical scalar evidence violates sign or order bounds.")
    _same_numeric(
        observation_values["total_energy_ev"],
        (
            observation_values["potential_energy_ev"]
            + observation_values["kinetic_energy_ev"]
        ),
        label="observation total energy",
    )
    if np.any(
        observation_values["force_rms_ev_per_angstrom"]
        > observation_values["force_max_ev_per_angstrom"] + 1.0e-12
    ):
        raise _invalid("force RMS cannot exceed maximum atomic force.")
    if (
        np.any(stepwise["oh_min_distance_angstrom"] < 0.70)
        or np.any(stepwise["oh_max_distance_angstrom"] > 1.30)
        or np.any(stepwise["hoh_min_angle_degrees"] < 70.0)
        or np.any(stepwise["hoh_max_angle_degrees"] > 140.0)
        or np.any(
            stepwise["oh_min_distance_angstrom"]
            > stepwise["oh_max_distance_angstrom"]
        )
        or np.any(
            stepwise["hoh_min_angle_degrees"]
            > stepwise["hoh_max_angle_degrees"]
        )
    ):
        raise _invalid("stepwise water topology evidence is invalid.")

    production_volumes = np.abs(np.linalg.det(cells))
    minimum_cell_heights, maximum_cell_heights = _cell_heights(cells)
    total_mass = float(
        np.sum(Atoms(numbers=numbers).get_masses())
    )
    _same_numeric(
        stepwise["density_g_per_ml"],
        (
            total_mass
            / stepwise["volume_angstrom3"]
            * _AMU_PER_ANGSTROM3_TO_G_PER_ML
        ),
        label="stepwise density/volume",
    )
    density_from_cells = (
        total_mass
        / production_volumes
        * _AMU_PER_ANGSTROM3_TO_G_PER_ML
    )
    production_mask = stages == "production"
    production_observation_indices = np.flatnonzero(production_mask)
    production_steps = steps[production_mask]
    if len(production_observation_indices) != frame_count:
        raise _invalid("production observation count differs from config.")
    _same_numeric(
        production_density,
        density_from_cells,
        label="production density/cell volume",
    )
    _same_numeric(
        production_density,
        observation_values["density_g_per_ml"][production_mask],
        label="production density/observations",
    )
    _same_numeric(
        production_volumes,
        observation_values["volume_angstrom3"][production_mask],
        label="production volume/observations",
    )
    _same_numeric(
        observation_values["density_g_per_ml"],
        stepwise["density_g_per_ml"][steps],
        label="sampled density",
    )
    _same_numeric(
        observation_values["volume_angstrom3"],
        stepwise["volume_angstrom3"][steps],
        label="sampled volume",
    )
    _same_numeric(
        production_density,
        stepwise["density_g_per_ml"][production_steps],
        label="production density/stepwise evidence",
    )
    _same_numeric(
        production_volumes,
        stepwise["volume_angstrom3"][production_steps],
        label="production volume/stepwise evidence",
    )
    _same_numeric(
        minimum_cell_heights,
        stepwise["minimum_cell_height_angstrom"][production_steps],
        label="production minimum cell height",
    )
    _same_numeric(
        maximum_cell_heights,
        stepwise["maximum_cell_height_angstrom"][production_steps],
        label="production maximum cell height",
    )
    _same_numeric(
        observation_values["temperature_k"],
        stepwise["temperature_k"][steps],
        label="sampled temperature",
    )
    _same_numeric(
        observation_values["potential_energy_ev"],
        stepwise["potential_energy_ev"][steps],
        label="sampled potential energy",
    )
    _same_numeric(
        observation_values["force_max_ev_per_angstrom"],
        stepwise["force_max_ev_per_angstrom"][steps],
        label="sampled maximum force",
    )
    physical_state = _production_physical_state(
        numbers=numbers,
        positions=positions,
        cells=cells,
        velocities=velocities,
    )
    _same_numeric(
        physical_state["kinetic_energy_ev"],
        observation_values["kinetic_energy_ev"][production_mask],
        label="production kinetic energy/velocities",
    )
    _same_numeric(
        physical_state["temperature_k"],
        observation_values["temperature_k"][production_mask],
        label="production temperature/velocities",
    )
    _same_numeric(
        physical_state["temperature_k"],
        stepwise["temperature_k"][production_steps],
        label="production temperature/stepwise evidence",
    )
    _same_numeric(
        physical_state["velocity_max_angstrom_per_ase_time"],
        stepwise["velocity_max_angstrom_per_ase_time"][
            production_steps
        ],
        label="production velocity maximum",
    )
    for name in (
        "oh_min_distance_angstrom",
        "oh_max_distance_angstrom",
        "hoh_min_angle_degrees",
        "hoh_max_angle_degrees",
    ):
        _same_numeric(
            physical_state[name],
            stepwise[name][production_steps],
            label=f"production {name}",
        )
    if (
        np.any(physical_state["oh_min_distance_angstrom"] < 0.70)
        or np.any(physical_state["oh_max_distance_angstrom"] > 1.30)
        or np.any(physical_state["hoh_min_angle_degrees"] < 70.0)
        or np.any(physical_state["hoh_max_angle_degrees"] > 140.0)
    ):
        raise _invalid(
            "production positions violate the water topology contract."
        )

    recomputed_blocks = np.asarray(
        [
            np.mean(block)
            for block in np.split(
                production_density,
                config.rdf_block_count,
            )
        ],
        dtype=np.float64,
    )
    _same_numeric(
        density_blocks,
        recomputed_blocks,
        label="density blocks",
    )
    density_mean = float(np.mean(production_density))
    density_block_sem = float(
        np.std(density_blocks, ddof=1)
        / math.sqrt(len(density_blocks))
    )
    half = frame_count // 2
    density_half_relative_drift = float(
        abs(
            np.mean(production_density[:half])
            - np.mean(production_density[half:])
        )
        / density_mean
    )
    density_relative_error = float(
        abs(density_mean - config.density_reference_g_per_ml)
        / config.density_reference_g_per_ml
    )
    production_temperature = observation_values["temperature_k"][
        production_mask
    ]
    production_pressure = observation_values["pressure_bar"][
        production_mask
    ]
    temperature_mean = float(np.mean(production_temperature))
    temperature_relative_error = float(
        abs(temperature_mean - config.temperature_k)
        / config.temperature_k
    )
    pressure_mean = float(np.mean(production_pressure))
    pressure_standard_deviation = float(
        np.std(production_pressure, ddof=1)
    )
    last_equilibration_time = float(
        observation_values["time_fs"][
            np.flatnonzero(stages == "equilibration")[-1]
        ]
    )
    production_duration_ps = float(
        (
            observation_values["time_fs"][
                production_observation_indices[-1]
            ]
            - last_equilibration_time
        )
        / 1_000.0
    )

    topology_preserved = bool(
        np.all(stepwise["oh_min_distance_angstrom"] >= 0.70)
        and np.all(stepwise["oh_max_distance_angstrom"] <= 1.30)
        and np.all(stepwise["hoh_min_angle_degrees"] >= 70.0)
        and np.all(stepwise["hoh_max_angle_degrees"] <= 140.0)
    )
    engineering_checks = {
        "finite_observations": True,
        "all_md_steps_checked": True,
        "temperature_below_emergency_limit": bool(
            np.max(stepwise["temperature_k"])
            <= config.maximum_temperature_k
        ),
        "force_below_emergency_limit": bool(
            np.max(stepwise["force_max_ev_per_angstrom"])
            <= config.maximum_force_ev_per_angstrom
        ),
        "water_topology_preserved": topology_preserved,
        "cell_cutoff_safe": bool(
            np.min(stepwise["minimum_cell_height_angstrom"])
            >= config.minimum_cell_height_angstrom
        ),
    }
    stepwise_summary = {
        "checked_unique_steps": stepwise_count,
        "expected_unique_steps": stepwise_count,
        "all_steps_checked": True,
        "maximum_temperature_k": float(
            np.max(stepwise["temperature_k"])
        ),
        "maximum_force_ev_per_angstrom": float(
            np.max(stepwise["force_max_ev_per_angstrom"])
        ),
        "oh_distance_range_angstrom": [
            float(np.min(stepwise["oh_min_distance_angstrom"])),
            float(np.max(stepwise["oh_max_distance_angstrom"])),
        ],
        "hoh_angle_range_degrees": [
            float(np.min(stepwise["hoh_min_angle_degrees"])),
            float(np.max(stepwise["hoh_max_angle_degrees"])),
        ],
        "topology_preserved": topology_preserved,
        "minimum_cell_height_angstrom": float(
            np.min(stepwise["minimum_cell_height_angstrom"])
        ),
        "maximum_cell_height_angstrom": float(
            np.max(stepwise["maximum_cell_height_angstrom"])
        ),
        "volume_range_angstrom3": [
            float(np.min(stepwise["volume_angstrom3"])),
            float(np.max(stepwise["volume_angstrom3"])),
        ],
        "density_range_g_per_ml": [
            float(np.min(stepwise["density_g_per_ml"])),
            float(np.max(stepwise["density_g_per_ml"])),
        ],
        "cell_cutoff_safe": engineering_checks["cell_cutoff_safe"],
    }
    diagnostic_checks = {
        "engineering_stability_passed": all(
            engineering_checks.values()
        ),
        "production_duration_sufficient": bool(
            production_duration_ps
            >= config.minimum_diagnostic_duration_ps
        ),
        "production_frame_count_sufficient": bool(
            frame_count >= config.minimum_diagnostic_frames
        ),
        "production_temperature_centered": bool(
            temperature_relative_error
            <= config.temperature_relative_tolerance
        ),
        "density_relative_error_passed": bool(
            density_relative_error
            <= config.density_relative_error_limit
        ),
        "density_half_stability_passed": bool(
            density_half_relative_drift
            <= config.density_half_drift_limit
        ),
        "production_pressure_centered": bool(
            abs(pressure_mean - config.pressure_bar)
            <= config.pressure_mean_tolerance_bar
        ),
    }
    gates = {
        "engineering_checks": engineering_checks,
        "engineering_stability_passed": all(
            engineering_checks.values()
        ),
        "minimum_npt_diagnostic_checks": diagnostic_checks,
        "minimum_npt_diagnostic_eligible": all(
            diagnostic_checks.values()
        ),
        "paper_duration_fidelity_passed": (
            config.paper_duration_fidelity
        ),
        "paper_integrator_fidelity_passed": (
            config.paper_integrator_fidelity
        ),
        "paper_protocol_reproduced": bool(
            config.paper_duration_fidelity
            and config.paper_integrator_fidelity
        ),
        "independent_replicas_passed": False,
        "npt_density_validation_passed": False,
        "finite_size_validation_passed": False,
        "external_rdf_validation_passed": False,
        "cross_engine_validation_passed": False,
        "hamiltonian_freeze_eligible": False,
        "claim_eligible": False,
        "interpretation": _SINGLE_REPLICA_INTERPRETATION,
    }
    return {
        "trajectory": {
            "production_duration_ps": production_duration_ps,
            "production_frame_count": frame_count,
        },
        "diagnostics": {
            "stepwise_stability": stepwise_summary,
            "rdf": rdf_summary,
            "minimum_cell_height_angstrom": (
                stepwise_summary["minimum_cell_height_angstrom"]
            ),
            "production_temperature_mean_k": temperature_mean,
            "production_temperature_relative_error": (
                temperature_relative_error
            ),
            "production_pressure_mean_bar": pressure_mean,
            "production_pressure_standard_deviation_bar": (
                pressure_standard_deviation
            ),
            "production_density_mean_g_per_ml": density_mean,
            "production_density_block_sem_g_per_ml": (
                density_block_sem
            ),
            "production_density_block_count": len(density_blocks),
            "production_density_half_relative_drift": (
                density_half_relative_drift
            ),
            "density_relative_error": density_relative_error,
        },
        "gates": gates,
    }


def require_bulk_water_npt_summary_consistency(
    summary: Mapping[str, Any],
    recomputed: Mapping[str, Any],
) -> None:
    """Fail if a JSON summary cannot be reproduced from retained arrays."""

    try:
        summary_subset = {
            "trajectory": {
                key: summary["trajectory"][key]
                for key in recomputed["trajectory"]
            },
            "diagnostics": {
                key: summary["diagnostics"][key]
                for key in recomputed["diagnostics"]
            },
            "gates": dict(summary["gates"]),
        }
    except (KeyError, TypeError) as exc:
        raise _invalid("summary omits recomputable NPT evidence.") from exc
    expected_subset = dict(recomputed)
    if canonical_sha256(summary_subset) != canonical_sha256(
        expected_subset
    ):
        raise _invalid(
            "summary claims conflict with values recomputed from arrays."
        )


__all__ = [
    "NPT_ARRAY_NAMES",
    "compute_bulk_water_rdf_evidence",
    "recompute_bulk_water_npt_evidence",
    "require_bulk_water_npt_summary_consistency",
]
