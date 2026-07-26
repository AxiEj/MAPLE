from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Callable, Mapping

import numpy as np
from ase import Atoms, units
from ase.calculators.calculator import Calculator
from ase.md.langevin import Langevin
from ase.md.nose_hoover_chain import IsotropicMTKNPT
from ase.md.velocitydistribution import (
    MaxwellBoltzmannDistribution,
    Stationary,
)

from .bulk_water import (
    BulkWaterValidationError,
    BulkWaterValidationResult,
    _array_sha256,
    _immutable_array,
    _observation_arrays,
    _observe,
    _StepwiseStabilityMonitor,
    _structure_evidence,
    _validated_run_provenance,
    validate_water_box,
)
from .bulk_water_evidence import (
    compute_bulk_water_rdf_evidence,
    recompute_bulk_water_npt_evidence,
)
from .protocol import canonical_sha256


IAPWS_DENSITY_298K_G_PER_ML = 0.997_047_013
IAPWS_DENSITY_REFERENCE = {
    "value_g_per_ml": IAPWS_DENSITY_298K_G_PER_ML,
    "temperature_k": 298.15,
    "pressure_mpa": 0.1,
    "source": (
        "IAPWS SR6-08(2011), Table 8, liquid water at 0.1 MPa"
    ),
    "url": "https://www.iapws.org/relguide/LiquidWater.pdf",
}
MACE_OFF24_WATER_DENSITY_PROTOCOL = {
    "identity": "mace-off24-si-water-density-500ps-v1",
    "box_edge_angstrom": 25.0,
    "timestep_fs": 1.0,
    "equilibration_ps": 100.0,
    "production_ps": 400.0,
    "density_sample_interval_steps": 100,
    "temperature_k": 298.0,
    "pressure_atm": 1.0,
    "dynamics": "Langevin equations",
    "barostat": "OpenMM Monte Carlo barostat",
    "source": (
        "MACE-OFF24 supporting information, condensed-phase simulations"
    ),
    "url": (
        "https://www.repository.cam.ac.uk/bitstreams/"
        "7e2a13f9-d1de-4814-9af7-e51de175a024/download"
    ),
}
ASE_MTK_INTEGRATOR = (
    "ase.md.nose_hoover_chain.IsotropicMTKNPT"
)
NPT_IMPLEMENTATION_PATHS = {
    "maple/function/dispatcher/solvfe/bulk_water.py",
    "maple/function/dispatcher/solvfe/bulk_water_evidence.py",
    "maple/function/dispatcher/solvfe/bulk_water_npt.py",
    "maple/function/dispatcher/solvfe/protocol.py",
    "maple/function/dispatcher/solvfe/provenance.py",
    "maple/function/calculator/mace/_mace_upstream_calculator.py",
    "examples/solvation/route_a/validate_bulk_water_npt.py",
}


def _positive_finite(value: float, *, label: str) -> float:
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0.0:
        raise BulkWaterValidationError(
            f"{label} must be finite and positive."
        )
    return normalized


def _cell_heights(cell: np.ndarray) -> np.ndarray:
    matrix = np.asarray(cell, dtype=float)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise BulkWaterValidationError(
            "CELL_INVALID: NPT cell must be a finite 3x3 matrix."
        )
    volume = abs(float(np.linalg.det(matrix)))
    if volume <= 0.0:
        raise BulkWaterValidationError(
            "CELL_INVALID: NPT cell must have positive volume."
        )
    heights = []
    for index in range(3):
        other = [axis for axis in range(3) if axis != index]
        face_area = float(
            np.linalg.norm(np.cross(matrix[other[0]], matrix[other[1]]))
        )
        if not math.isfinite(face_area) or face_area <= 0.0:
            raise BulkWaterValidationError(
                "CELL_INVALID: NPT cell has a degenerate face."
            )
        heights.append(volume / face_area)
    return np.asarray(heights, dtype=float)


@dataclass(frozen=True)
class BulkWaterNPTConfig:
    """Paper-derived, stress-aware liquid-water density protocol.

    The implementation uses ASE's correct-ensemble isotropic MTK integrator.
    It is deliberately not labeled an exact reproduction of the MACE-OFF24
    OpenMM Langevin/Monte-Carlo-barostat protocol.
    """

    temperature_k: float = 298.15
    pressure_bar: float = 1.01325
    timestep_fs: float = 1.0
    precondition_timestep_fs: float = 0.1
    precondition_steps: int = 1_000
    precondition_friction_per_fs: float = 0.05
    equilibration_steps: int = 100_000
    production_steps: int = 400_000
    sample_interval_steps: int = 100
    thermostat_damping_fs: float = 100.0
    barostat_damping_fs: float = 1_000.0
    thermostat_chain_length: int = 3
    barostat_chain_length: int = 3
    thermostat_substeps: int = 1
    barostat_substeps: int = 1
    seed: int = 20_260_727
    rdf_bin_width_angstrom: float = 0.05
    rdf_max_angstrom: float = 6.0
    rdf_block_count: int = 8
    interaction_cutoff_angstrom: float = 6.0
    cutoff_margin_angstrom: float = 0.10
    minimum_diagnostic_duration_ps: float = 400.0
    minimum_diagnostic_frames: int = 4_000
    temperature_relative_tolerance: float = 0.05
    maximum_temperature_k: float = 1_000.0
    maximum_force_ev_per_angstrom: float = 50.0
    density_reference_g_per_ml: float = IAPWS_DENSITY_298K_G_PER_ML
    density_relative_error_limit: float = 0.03
    density_half_drift_limit: float = 0.01
    pressure_mean_tolerance_bar: float = 500.0

    def __post_init__(self) -> None:
        for name in (
            "temperature_k",
            "pressure_bar",
            "timestep_fs",
            "precondition_timestep_fs",
            "precondition_friction_per_fs",
            "thermostat_damping_fs",
            "barostat_damping_fs",
            "rdf_bin_width_angstrom",
            "rdf_max_angstrom",
            "interaction_cutoff_angstrom",
            "cutoff_margin_angstrom",
            "minimum_diagnostic_duration_ps",
            "temperature_relative_tolerance",
            "maximum_temperature_k",
            "maximum_force_ev_per_angstrom",
            "density_reference_g_per_ml",
            "density_relative_error_limit",
            "density_half_drift_limit",
            "pressure_mean_tolerance_bar",
        ):
            _positive_finite(getattr(self, name), label=name)
        for name in (
            "equilibration_steps",
            "production_steps",
            "sample_interval_steps",
            "precondition_steps",
            "thermostat_chain_length",
            "barostat_chain_length",
            "thermostat_substeps",
            "barostat_substeps",
            "seed",
            "rdf_block_count",
            "minimum_diagnostic_frames",
        ):
            value = getattr(self, name)
            if type(value) is not int:
                raise BulkWaterValidationError(
                    f"{name} must be an integer."
                )
        for name in (
            "equilibration_steps",
            "production_steps",
            "sample_interval_steps",
            "thermostat_chain_length",
            "barostat_chain_length",
            "thermostat_substeps",
            "barostat_substeps",
            "rdf_block_count",
            "minimum_diagnostic_frames",
        ):
            if getattr(self, name) <= 0:
                raise BulkWaterValidationError(
                    f"{name} must be positive."
                )
        if self.precondition_steps < 0:
            raise BulkWaterValidationError(
                "precondition_steps must be non-negative."
            )
        if (
            self.precondition_steps
            and self.precondition_steps % self.sample_interval_steps
        ):
            raise BulkWaterValidationError(
                "precondition_steps must be divisible by "
                "sample_interval_steps."
            )
        if self.seed < 0:
            raise BulkWaterValidationError(
                "seed must be a non-negative integer."
            )
        if self.production_steps % self.sample_interval_steps:
            raise BulkWaterValidationError(
                "production_steps must be divisible by "
                "sample_interval_steps."
            )
        if self.equilibration_steps % self.sample_interval_steps:
            raise BulkWaterValidationError(
                "equilibration_steps must be divisible by "
                "sample_interval_steps."
            )
        frame_count = self.production_frame_count
        if frame_count < 2 or frame_count % 2:
            raise BulkWaterValidationError(
                "production must contain an even number of at least two "
                "sampled frames."
            )
        if (
            self.rdf_block_count < 2
            or frame_count % self.rdf_block_count
        ):
            raise BulkWaterValidationError(
                "production frame count must divide evenly into at least "
                "two RDF blocks."
            )
        if self.minimum_diagnostic_frames < 2:
            raise BulkWaterValidationError(
                "minimum_diagnostic_frames must be at least two."
            )

    @property
    def equilibration_duration_ps(self) -> float:
        return self.equilibration_steps * self.timestep_fs / 1_000.0

    @property
    def precondition_duration_ps(self) -> float:
        return (
            self.precondition_steps
            * self.precondition_timestep_fs
            / 1_000.0
        )

    @property
    def production_duration_ps(self) -> float:
        return self.production_steps * self.timestep_fs / 1_000.0

    @property
    def production_frame_count(self) -> int:
        return self.production_steps // self.sample_interval_steps

    @property
    def minimum_cell_height_angstrom(self) -> float:
        return 2.0 * max(
            self.interaction_cutoff_angstrom
            + self.cutoff_margin_angstrom,
            self.rdf_max_angstrom,
        )

    @property
    def paper_duration_fidelity(self) -> bool:
        return bool(
            math.isclose(self.timestep_fs, 1.0, abs_tol=1.0e-12)
            and self.equilibration_duration_ps >= 100.0
            and self.production_duration_ps >= 400.0
            and self.sample_interval_steps == 100
            and math.isclose(
                self.pressure_bar,
                1.01325,
                rel_tol=0.0,
                abs_tol=1.0e-8,
            )
            and abs(self.temperature_k - 298.0) <= 0.15 + 1.0e-12
        )

    @property
    def paper_integrator_fidelity(self) -> bool:
        return False

    @property
    def content_hash(self) -> str:
        return canonical_sha256(
            {
                "schema": "maple-route-a-bulk-water-npt-config-v1",
                **asdict(self),
            }
        )

    @property
    def content_hash_without_seed(self) -> str:
        values = asdict(self)
        values.pop("seed")
        return canonical_sha256(
            {
                "schema": "maple-route-a-bulk-water-npt-config-v1",
                **values,
            }
        )


@dataclass
class _NPTStepwiseMonitor:
    config: BulkWaterNPTConfig

    def __post_init__(self) -> None:
        self._water = _StepwiseStabilityMonitor(self.config)
        self._step_count = (
            self.config.precondition_steps
            + self.config.equilibration_steps
            + self.config.production_steps
            + 1
        )
        self._seen = np.zeros(self._step_count, dtype=bool)
        self._stage_code = np.empty(self._step_count, dtype=np.uint8)
        self._values = {
            name: np.empty(self._step_count, dtype=np.float64)
            for name in (
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
        }
    def inspect(self, atoms: Atoms, *, stage: str, step: int) -> None:
        normalized_step = int(step)
        if (
            normalized_step < 0
            or normalized_step >= self._step_count
            or self._seen[normalized_step]
        ):
            raise BulkWaterValidationError(
                "STEPWISE_EVIDENCE_INVALID: NPT steps must be recorded "
                "exactly once in protocol order."
            )
        if normalized_step == 0:
            expected_stage = "initial"
            stage_code = 0
        elif normalized_step <= self.config.precondition_steps:
            expected_stage = "precondition"
            stage_code = 1
        elif normalized_step <= (
            self.config.precondition_steps
            + self.config.equilibration_steps
        ):
            expected_stage = "equilibration"
            stage_code = 2
        else:
            expected_stage = "production"
            stage_code = 3
        if stage != expected_stage:
            raise BulkWaterValidationError(
                "STEPWISE_EVIDENCE_INVALID: "
                f"step {normalized_step} belongs to {expected_stage}, "
                f"not {stage}."
            )
        water_values = self._water.inspect(
            atoms,
            stage=stage,
            step=normalized_step,
        )
        heights = _cell_heights(atoms.cell.array)
        minimum_height = float(np.min(heights))
        volume = float(atoms.get_volume())
        density = float(
            np.sum(atoms.get_masses())
            / volume
            * 1.66053906660
        )
        if (
            minimum_height
            < self.config.minimum_cell_height_angstrom - 1.0e-12
        ):
            raise BulkWaterValidationError(
                "CELL_CUTOFF_UNSAFE: "
                f"stage={stage}, step={step}, minimum cell height "
                f"{minimum_height:.8g} A is below required "
                f"{self.config.minimum_cell_height_angstrom:.8g} A for "
                f"cutoff={self.config.interaction_cutoff_angstrom:.8g} A "
                f"with margin={self.config.cutoff_margin_angstrom:.8g} A "
                f"and RDF radius={self.config.rdf_max_angstrom:.8g} A."
            )
        self._seen[normalized_step] = True
        self._stage_code[normalized_step] = stage_code
        for name, value in water_values.items():
            self._values[name][normalized_step] = value
        self._values["minimum_cell_height_angstrom"][
            normalized_step
        ] = minimum_height
        self._values["maximum_cell_height_angstrom"][
            normalized_step
        ] = float(np.max(heights))
        self._values["volume_angstrom3"][normalized_step] = volume
        self._values["density_g_per_ml"][normalized_step] = density
    def as_dict(self, *, expected_last_step: int) -> dict[str, Any]:
        result = self._water.as_dict(
            expected_last_step=expected_last_step
        )
        result.update(
            {
                "minimum_cell_height_angstrom": (
                    float(
                        np.min(
                            self._values[
                                "minimum_cell_height_angstrom"
                            ]
                        )
                    )
                ),
                "maximum_cell_height_angstrom": (
                    float(
                        np.max(
                            self._values[
                                "maximum_cell_height_angstrom"
                            ]
                        )
                    )
                ),
                "volume_range_angstrom3": [
                    float(
                        np.min(self._values["volume_angstrom3"])
                    ),
                    float(
                        np.max(self._values["volume_angstrom3"])
                    ),
                ],
                "density_range_g_per_ml": [
                    float(
                        np.min(self._values["density_g_per_ml"])
                    ),
                    float(
                        np.max(self._values["density_g_per_ml"])
                    ),
                ],
                "cell_cutoff_safe": bool(
                    np.min(
                        self._values["minimum_cell_height_angstrom"]
                    )
                    >= self.config.minimum_cell_height_angstrom
                ),
            }
        )
        return result

    def arrays(self) -> dict[str, np.ndarray]:
        if not np.all(self._seen):
            raise BulkWaterValidationError(
                "STEPWISE_EVIDENCE_INVALID: NPT run did not retain every "
                "MD step."
            )
        return {
            "stepwise_stage_code": _immutable_array(
                self._stage_code,
                dtype=np.uint8,
            ),
            "stepwise_step": _immutable_array(
                np.arange(self._step_count),
                dtype=np.int64,
            ),
            **{
                f"stepwise_{name}": _immutable_array(
                    values,
                    dtype=np.float64,
                )
                for name, values in self._values.items()
            },
        }


def run_bulk_water_npt(
    initial_atoms: Atoms,
    *,
    calculator: Calculator,
    config: BulkWaterNPTConfig,
    source_provenance: Mapping[str, Any],
    calculator_provenance: Mapping[str, Any],
    implementation_provenance: Mapping[str, Any],
    progress_callback: (
        Callable[[Mapping[str, float | int | str]], None] | None
    ) = None,
) -> BulkWaterValidationResult:
    topology = validate_water_box(initial_atoms)
    source_record, calculator_record, implementation_record = (
        _validated_run_provenance(
            initial_atoms,
            calculator,
            source_provenance=source_provenance,
            calculator_provenance=calculator_provenance,
            implementation_provenance=implementation_provenance,
            required_implementation_paths=NPT_IMPLEMENTATION_PATHS,
        )
    )
    model_cutoff = float(
        calculator_record["calculator"]["interaction_cutoff_angstrom"]
    )
    if not math.isclose(
        config.interaction_cutoff_angstrom,
        model_cutoff,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise BulkWaterValidationError(
            "CALCULATOR_CUTOFF_MISMATCH: NPT config cutoff "
            f"{config.interaction_cutoff_angstrom:.8g} A differs from "
            f"the runtime calculator cutoff {model_cutoff:.8g} A."
        )
    initial_heights = _cell_heights(initial_atoms.cell.array)
    if (
        float(np.min(initial_heights))
        < config.minimum_cell_height_angstrom - 1.0e-12
    ):
        raise BulkWaterValidationError(
            "CELL_CUTOFF_UNSAFE: initial minimum cell height "
            f"{float(np.min(initial_heights)):.8g} A is below required "
            f"{config.minimum_cell_height_angstrom:.8g} A."
        )
    if config.rdf_max_angstrom > 0.5 * float(np.min(initial_heights)):
        raise BulkWaterValidationError(
            "RDF_RADIUS_INVALID: rdf_max_angstrom exceeds half the initial "
            "minimum periodic cell height."
        )

    atoms = initial_atoms.copy()
    atoms.calc = calculator
    rng = np.random.default_rng(config.seed)
    MaxwellBoltzmannDistribution(
        atoms,
        temperature_K=config.temperature_k,
        force_temp=True,
        rng=rng,
    )
    Stationary(atoms, preserve_temperature=True)

    observations = [
        _observe(
            atoms,
            calculator,
            stage="initial",
            step=0,
            timestep_fs=config.timestep_fs,
        )
    ]
    if progress_callback is not None:
        progress_callback(observations[0])
    monitor = _NPTStepwiseMonitor(config)
    monitor.inspect(atoms, stage="initial", step=0)
    global_step = 0
    physical_time_fs = 0.0
    if config.precondition_steps:
        precondition = Langevin(
            atoms,
            timestep=config.precondition_timestep_fs * units.fs,
            temperature_K=config.temperature_k,
            friction=config.precondition_friction_per_fs / units.fs,
            fixcm=True,
            rng=rng,
        )
        def inspect_precondition() -> None:
            if precondition.nsteps:
                monitor.inspect(
                    atoms,
                    stage="precondition",
                    step=int(precondition.nsteps),
                )

        precondition.attach(
            inspect_precondition,
            interval=1,
        )
        remaining = int(config.precondition_steps)
        while remaining:
            chunk = min(config.sample_interval_steps, remaining)
            precondition.run(chunk)
            remaining -= chunk
            global_step = int(precondition.nsteps)
            physical_time_fs = (
                global_step * config.precondition_timestep_fs
            )
            observation = _observe(
                atoms,
                calculator,
                stage="precondition",
                step=global_step,
                timestep_fs=config.precondition_timestep_fs,
            )
            observation["time_fs"] = physical_time_fs
            observations.append(observation)
            if progress_callback is not None:
                progress_callback(observation)

    stage_context = {"name": "equilibration"}
    dynamics = IsotropicMTKNPT(
        atoms,
        timestep=config.timestep_fs * units.fs,
        temperature_K=config.temperature_k,
        pressure_au=config.pressure_bar * units.bar,
        tdamp=config.thermostat_damping_fs * units.fs,
        pdamp=config.barostat_damping_fs * units.fs,
        tchain=config.thermostat_chain_length,
        pchain=config.barostat_chain_length,
        tloop=config.thermostat_substeps,
        ploop=config.barostat_substeps,
    )
    def inspect_dynamics() -> None:
        if dynamics.nsteps:
            monitor.inspect(
                atoms,
                stage=stage_context["name"],
                step=global_step + int(dynamics.nsteps),
            )

    dynamics.attach(
        inspect_dynamics,
        interval=1,
    )

    production_positions: list[np.ndarray] = []
    production_cells: list[np.ndarray] = []
    production_velocities: list[np.ndarray] = []
    for stage, stage_steps in (
        ("equilibration", config.equilibration_steps),
        ("production", config.production_steps),
    ):
        stage_context["name"] = stage
        remaining = int(stage_steps)
        while remaining:
            chunk = min(config.sample_interval_steps, remaining)
            dynamics.run(chunk)
            remaining -= chunk
            observation = _observe(
                atoms,
                calculator,
                stage=stage,
                step=global_step + int(dynamics.nsteps),
                timestep_fs=config.timestep_fs,
            )
            observation["time_fs"] = (
                physical_time_fs
                + int(dynamics.nsteps) * config.timestep_fs
            )
            observations.append(observation)
            if progress_callback is not None:
                progress_callback(observation)
            if stage == "production":
                production_positions.append(
                    np.asarray(atoms.positions, dtype=float).copy()
                )
                production_cells.append(
                    np.asarray(atoms.cell.array, dtype=float).copy()
                )
                production_velocities.append(
                    np.asarray(atoms.get_velocities(), dtype=float).copy()
                )

    positions = _immutable_array(
        production_positions,
        dtype=np.float64,
    )
    cells = _immutable_array(
        production_cells,
        dtype=np.float64,
    )
    velocities = _immutable_array(
        production_velocities,
        dtype=np.float64,
    )
    observation_arrays = _observation_arrays(observations)
    production_mask = (
        observation_arrays["observation_stage"] == "production"
    )
    production_density = _immutable_array(
        observation_arrays["observation_density_g_per_ml"][
            production_mask
        ],
        dtype=np.float64,
    )
    density_blocks = np.asarray(
        [
            np.mean(block)
            for block in np.split(
                production_density,
                config.rdf_block_count,
            )
        ],
        dtype=float,
    )
    rdf_arrays, _ = compute_bulk_water_rdf_evidence(
        positions,
        cells,
        np.asarray(atoms.numbers, dtype=np.int64),
        config=config,
    )
    arrays = {
        **observation_arrays,
        "atomic_numbers": _immutable_array(
            atoms.numbers,
            dtype=np.int64,
        ),
        "production_positions_angstrom": positions,
        "production_cells_angstrom": cells,
        "production_velocities_angstrom_per_ase_time": velocities,
        "production_density_g_per_ml": production_density,
        "production_density_block_g_per_ml": _immutable_array(
            density_blocks,
            dtype=np.float64,
        ),
        **monitor.arrays(),
        **rdf_arrays,
    }
    recomputed = recompute_bulk_water_npt_evidence(
        arrays,
        config=config,
        water_count=topology["water_count"],
    )
    runtime_stepwise = monitor.as_dict(
        expected_last_step=(
            config.precondition_steps
            + config.equilibration_steps
            + config.production_steps
        )
    )
    if canonical_sha256(runtime_stepwise) != canonical_sha256(
        recomputed["diagnostics"]["stepwise_stability"]
    ):
        raise BulkWaterValidationError(
            "STEPWISE_EVIDENCE_INVALID: retained arrays do not reproduce "
            "the runtime monitor."
        )
    semantic_hashes = {
        name: _array_sha256(values)
        for name, values in sorted(arrays.items())
    }
    summary = {
        "schema": "maple-route-a-bulk-water-npt-summary-v1",
        "config": {
            "schema": "maple-route-a-bulk-water-npt-config-v1",
            **asdict(config),
            "content_hash": config.content_hash,
            "content_hash_without_seed": (
                config.content_hash_without_seed
            ),
        },
        "literature_protocol": MACE_OFF24_WATER_DENSITY_PROTOCOL,
        "density_reference": IAPWS_DENSITY_REFERENCE,
        "source": source_record,
        "calculator": calculator_record,
        "implementation": implementation_record,
        "initial_structure": _structure_evidence(initial_atoms),
        "final_structure": _structure_evidence(atoms),
        "topology": topology,
        "trajectory": {
            "ensemble": "NPT",
            "integrator": ASE_MTK_INTEGRATOR,
            "equilibration_duration_ps": (
                config.equilibration_duration_ps
            ),
            "precondition_duration_ps": (
                config.precondition_duration_ps
            ),
            **recomputed["trajectory"],
            "semantic_array_sha256": semantic_hashes,
            "determinism_scope": (
                "The seed fixes initial velocities. MTK propagation is "
                "deterministic for one software/hardware stack; cross-stack "
                "bitwise identity is not claimed."
            ),
        },
        "diagnostics": recomputed["diagnostics"],
        "gates": recomputed["gates"],
    }
    result_hash = canonical_sha256(summary)
    return BulkWaterValidationResult(
        result_hash=result_hash,
        summary=summary,
        arrays=arrays,
        artifact_schema="maple-route-a-bulk-water-npt-artifact-v1",
    )
