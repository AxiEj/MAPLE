from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms, units
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import (
    MaxwellBoltzmannDistribution,
    Stationary,
    ZeroRotation,
)

from .alchemy import (
    AlchemicalState,
    GaussianRepulsiveCore,
    ManyBodyInteractionEvaluator,
    SequentialInsertionCalculator,
)
from .analysis import ReducedPotentialTable
from .artifacts import RestartHashMismatch
from .moves import RigidBodyMetropolis, RigidBodyMoveConfig
from .protocol import canonical_sha256


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_BASIS_COLUMNS = (
    "base_ev",
    "interaction_ev",
    "repulsive_ev",
    "restraint_ev",
)


def _array_sha256(values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(values, dtype="<f8")
    digest = hashlib.sha256()
    digest.update(str(contiguous.shape).encode("ascii"))
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _raw_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    payload = (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@dataclass(frozen=True)
class AlchemicalSchedule:
    states: tuple[AlchemicalState, ...]
    labels: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.states or len(self.states) != len(self.labels):
            raise ValueError(
                "Alchemical schedule requires one label per non-empty state."
            )
        if (
            any(not isinstance(label, str) or not label for label in self.labels)
            or len(set(self.labels)) != len(self.labels)
        ):
            raise ValueError(
                "Alchemical schedule labels must be unique non-empty strings."
            )
        if self.states[0].scales != (0.0, 0.0):
            raise ValueError("Alchemical schedule must begin at endpoint D.")
        if self.states[-1].scales != (0.0, 1.0):
            raise ValueError("Alchemical schedule must end at endpoint P.")
        scales = [state.scales for state in self.states]
        if scales.count((1.0, 0.0)) != 1 or scales.count((1.0, 1.0)) != 1:
            raise ValueError(
                "Alchemical schedule must contain shared R and I endpoints "
                "exactly once."
            )

    @classmethod
    def initial(cls, *, points_per_leg: int = 5) -> "AlchemicalSchedule":
        if (
            isinstance(points_per_leg, (bool, np.bool_))
            or int(points_per_leg) != points_per_leg
            or int(points_per_leg) < 2
        ):
            raise ValueError("points_per_leg must be an integer of at least two.")
        grid = np.linspace(0.0, 1.0, int(points_per_leg))
        states = tuple(
            [AlchemicalState("D_TO_R", float(value)) for value in grid]
            + [
                AlchemicalState("R_TO_I", float(value))
                for value in grid[1:]
            ]
            + [
                AlchemicalState("I_TO_P", float(value))
                for value in grid[1:]
            ]
        )
        labels = tuple(
            f"{state.leg}:{state.lambda_value:.8f}" for state in states
        )
        return cls(states=states, labels=labels)

    @property
    def content_hash(self) -> str:
        return canonical_sha256(
            {
                "contract_id": "sequential-alchemical-schedule-v1",
                "states": [
                    {
                        "label": label,
                        "leg": state.leg,
                        "lambda": float(state.lambda_value),
                        "repulsive_scale": state.scales[0],
                        "full_interaction_scale": state.scales[1],
                    }
                    for label, state in zip(self.labels, self.states)
                ],
            }
        )


@dataclass(frozen=True)
class WindowRunConfig:
    temperature_k: float
    timestep_fs: float
    friction_per_fs: float
    equilibration_steps: int
    production_steps: int
    sample_interval: int
    seed: int

    def __post_init__(self) -> None:
        for name, value in (
            ("temperature_k", self.temperature_k),
            ("timestep_fs", self.timestep_fs),
            ("friction_per_fs", self.friction_per_fs),
        ):
            if (
                isinstance(value, (bool, np.bool_))
                or not math.isfinite(float(value))
                or float(value) <= 0.0
            ):
                raise ValueError(f"{name} must be finite and positive.")
        for name, value, minimum in (
            ("equilibration_steps", self.equilibration_steps, 0),
            ("production_steps", self.production_steps, 1),
            ("sample_interval", self.sample_interval, 1),
        ):
            if (
                isinstance(value, (bool, np.bool_))
                or int(value) != value
                or int(value) < minimum
            ):
                raise ValueError(
                    f"{name} must be an integer of at least {minimum}."
                )
        if self.production_steps % self.sample_interval != 0:
            raise ValueError(
                "production_steps must be divisible by sample_interval."
            )
        if isinstance(self.seed, (bool, np.bool_)) or int(self.seed) != self.seed:
            raise ValueError("seed must be an integer.")

    @property
    def content_hash(self) -> str:
        return canonical_sha256(
            {
                "temperature_k": float(self.temperature_k),
                "timestep_fs": float(self.timestep_fs),
                "friction_per_fs": float(self.friction_per_fs),
                "equilibration_steps": int(self.equilibration_steps),
                "production_steps": int(self.production_steps),
                "sample_interval": int(self.sample_interval),
                "seed": int(self.seed),
            }
        )


@dataclass(frozen=True)
class AlchemicalSampleSet:
    schedule: AlchemicalSchedule
    N_k: tuple[int, ...]
    frame_ids: tuple[str, ...]
    positions_angstrom: np.ndarray
    basis_ev: np.ndarray
    atom_list_hash: str
    restraint_hash: str
    measure_id: str
    boundary_conditions: str
    content_hash: str

    @classmethod
    def create(
        cls,
        *,
        schedule: AlchemicalSchedule,
        N_k: tuple[int, ...],
        frame_ids: tuple[str, ...],
        positions_angstrom: np.ndarray,
        basis_ev: np.ndarray,
        atom_list_hash: str,
        restraint_hash: str,
        measure_id: str,
        boundary_conditions: str,
    ) -> "AlchemicalSampleSet":
        counts = tuple(int(value) for value in N_k)
        if (
            len(counts) != len(schedule.states)
            or any(
                isinstance(value, (bool, np.bool_))
                or int(value) != value
                or int(value) <= 0
                for value in N_k
            )
        ):
            raise ValueError(
                "N_k must contain one positive sample count per schedule state."
            )
        frame_count = sum(counts)
        frames = tuple(frame_ids)
        if (
            len(frames) != frame_count
            or any(not isinstance(value, str) or not value for value in frames)
            or len(set(frames)) != len(frames)
        ):
            raise ValueError(
                "frame_ids must contain one unique non-empty ID per sample."
            )
        positions = np.asarray(positions_angstrom, dtype=float)
        basis = np.asarray(basis_ev, dtype=float)
        if (
            positions.ndim != 3
            or positions.shape[0] != frame_count
            or positions.shape[2] != 3
            or basis.shape != (frame_count, len(_BASIS_COLUMNS))
            or not np.all(np.isfinite(positions))
            or not np.all(np.isfinite(basis))
        ):
            raise ValueError(
                "Sample arrays must be finite with positions shape "
                "(frames, atoms, 3) and basis shape (frames, 4)."
            )
        for name, value in (
            ("atom_list_hash", atom_list_hash),
            ("restraint_hash", restraint_hash),
        ):
            if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
                raise ValueError(f"{name} must be a lowercase SHA-256 hash.")
        if not isinstance(measure_id, str) or not measure_id:
            raise ValueError("measure_id must be a non-empty string.")
        if boundary_conditions != "nonperiodic":
            raise ValueError(
                "OFF24_PRODUCTION_STATE_FORBIDDEN: alchemical samples must "
                "use nonperiodic boundary conditions."
            )

        immutable_positions = np.array(positions, copy=True, order="C")
        immutable_basis = np.array(basis, copy=True, order="C")
        immutable_positions.setflags(write=False)
        immutable_basis.setflags(write=False)
        preimage = {
            "contract_id": "alchemical-sample-set-v1",
            "schedule_hash": schedule.content_hash,
            "N_k": list(counts),
            "frame_ids": list(frames),
            "positions_sha256": _array_sha256(immutable_positions),
            "basis_columns": list(_BASIS_COLUMNS),
            "basis_sha256": _array_sha256(immutable_basis),
            "atom_list_hash": atom_list_hash,
            "restraint_hash": restraint_hash,
            "measure_id": measure_id,
            "boundary_conditions": boundary_conditions,
        }
        return cls(
            schedule=schedule,
            N_k=counts,
            frame_ids=frames,
            positions_angstrom=immutable_positions,
            basis_ev=immutable_basis,
            atom_list_hash=atom_list_hash,
            restraint_hash=restraint_hash,
            measure_id=measure_id,
            boundary_conditions=boundary_conditions,
            content_hash=canonical_sha256(preimage),
        )

    def reduced_potential_table(
        self,
        *,
        beta_ev_inverse: float,
    ) -> ReducedPotentialTable:
        if (
            isinstance(beta_ev_inverse, (bool, np.bool_))
            or not math.isfinite(float(beta_ev_inverse))
            or float(beta_ev_inverse) <= 0.0
        ):
            raise ValueError("beta_ev_inverse must be finite and positive.")
        base = self.basis_ev[:, 0]
        interaction = self.basis_ev[:, 1]
        repulsive = self.basis_ev[:, 2]
        restraint = self.basis_ev[:, 3]
        energies = np.vstack(
            [
                base
                + state.scales[1] * interaction
                + state.scales[0] * repulsive
                + restraint
                for state in self.schedule.states
            ]
        )
        return ReducedPotentialTable.create(
            u_kn=float(beta_ev_inverse) * energies,
            N_k=self.N_k,
            row_labels=self.schedule.labels,
            frame_ids=self.frame_ids,
            beta=float(beta_ev_inverse),
            measure_id=self.measure_id,
            boundary_conditions=self.boundary_conditions,
        )

    def write(self, directory: str | Path) -> None:
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        metadata_path = root / "metadata.json"
        if metadata_path.exists():
            resumed = self.load(root)
            if resumed.content_hash != self.content_hash:
                raise RestartHashMismatch(
                    "RESTART_HASH_MISMATCH: alchemical sample set changed."
                )
            return

        arrays_path = root / "samples.npz"
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".samples.",
            suffix=".npz.tmp",
            dir=root,
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                np.savez_compressed(
                    handle,
                    positions_angstrom=self.positions_angstrom,
                    basis_ev=self.basis_ev,
                )
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, arrays_path)
        finally:
            temporary.unlink(missing_ok=True)

        metadata = {
            "schema_version": 1,
            "artifact_type": "route-a-alchemical-sample-set",
            "content_hash": self.content_hash,
            "arrays_path": arrays_path.name,
            "arrays_sha256": _raw_sha256(arrays_path),
            "basis_columns": list(_BASIS_COLUMNS),
            "schedule": [
                {
                    "label": label,
                    "leg": state.leg,
                    "lambda": float(state.lambda_value),
                }
                for label, state in zip(
                    self.schedule.labels,
                    self.schedule.states,
                )
            ],
            "schedule_hash": self.schedule.content_hash,
            "N_k": list(self.N_k),
            "frame_ids": list(self.frame_ids),
            "atom_list_hash": self.atom_list_hash,
            "restraint_hash": self.restraint_hash,
            "measure_id": self.measure_id,
            "boundary_conditions": self.boundary_conditions,
        }
        _atomic_json(metadata_path, metadata)

    @classmethod
    def load(cls, directory: str | Path) -> "AlchemicalSampleSet":
        root = Path(directory)
        metadata_path = root / "metadata.json"
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RestartHashMismatch(
                f"RESTART_HASH_MISMATCH: cannot read sample metadata: {exc}"
            ) from exc
        arrays_path = root / str(metadata.get("arrays_path", ""))
        if (
            not arrays_path.is_file()
            or _raw_sha256(arrays_path) != metadata.get("arrays_sha256")
        ):
            raise RestartHashMismatch(
                "RESTART_HASH_MISMATCH: sample array archive is missing or corrupt."
            )
        schedule = AlchemicalSchedule(
            states=tuple(
                AlchemicalState(str(item["leg"]), float(item["lambda"]))
                for item in metadata["schedule"]
            ),
            labels=tuple(str(item["label"]) for item in metadata["schedule"]),
        )
        if schedule.content_hash != metadata.get("schedule_hash"):
            raise RestartHashMismatch(
                "RESTART_HASH_MISMATCH: alchemical schedule hash changed."
            )
        with np.load(arrays_path, allow_pickle=False) as archive:
            result = cls.create(
                schedule=schedule,
                N_k=tuple(metadata["N_k"]),
                frame_ids=tuple(metadata["frame_ids"]),
                positions_angstrom=archive["positions_angstrom"],
                basis_ev=archive["basis_ev"],
                atom_list_hash=str(metadata["atom_list_hash"]),
                restraint_hash=str(metadata["restraint_hash"]),
                measure_id=str(metadata["measure_id"]),
                boundary_conditions=str(metadata["boundary_conditions"]),
            )
        if result.content_hash != metadata.get("content_hash"):
            raise RestartHashMismatch(
                "RESTART_HASH_MISMATCH: sample content hash changed."
            )
        return result


class AlchemicalWindowRunner:
    def __init__(
        self,
        *,
        evaluator: ManyBodyInteractionEvaluator,
        repulsive_core: GaussianRepulsiveCore,
        restraint: Any,
        schedule: AlchemicalSchedule,
        config: WindowRunConfig,
        rigid_body_moves: RigidBodyMoveConfig | None = None,
    ) -> None:
        self.evaluator = evaluator
        self.repulsive_core = repulsive_core
        self.restraint = restraint
        self.schedule = schedule
        self.config = config
        self.rigid_body_moves = rigid_body_moves
        self.last_move_diagnostics: tuple[dict[str, Any], ...] = ()

    def run(self, initial_atoms: Atoms) -> AlchemicalSampleSet:
        if bool(np.any(initial_atoms.get_pbc())):
            raise ValueError(
                "OFF24_PRODUCTION_STATE_FORBIDDEN: association windows are "
                "strictly nonperiodic."
            )
        self.evaluator.partition.validate(initial_atoms)
        current = initial_atoms.copy()
        positions: list[np.ndarray] = []
        basis_rows: list[tuple[float, float, float, float]] = []
        frame_ids: list[str] = []
        counts: list[int] = []
        move_diagnostics: list[dict[str, Any]] = []

        for state_index, (label, state) in enumerate(
            zip(self.schedule.labels, self.schedule.states)
        ):
            atoms = current.copy()
            calculator = SequentialInsertionCalculator(
                evaluator=self.evaluator,
                repulsive_core=self.repulsive_core,
                state=state,
                restraint=self.restraint,
                capture_full_basis=False,
            )
            atoms.calc = calculator
            rng = np.random.default_rng(int(self.config.seed) + state_index)
            MaxwellBoltzmannDistribution(
                atoms,
                temperature_K=float(self.config.temperature_k),
                force_temp=True,
                rng=rng,
            )
            Stationary(atoms)
            ZeroRotation(atoms)
            dynamics = Langevin(
                atoms,
                timestep=float(self.config.timestep_fs) * units.fs,
                temperature_K=float(self.config.temperature_k),
                friction=float(self.config.friction_per_fs) / units.fs,
                fixcm=True,
                rng=rng,
            )
            if self.config.equilibration_steps:
                dynamics.run(int(self.config.equilibration_steps))
            mover = (
                None
                if self.rigid_body_moves is None
                else RigidBodyMetropolis(
                    evaluator=self.evaluator,
                    repulsive_core=self.repulsive_core,
                    restraint=self.restraint,
                    state=state,
                    temperature_k=float(self.config.temperature_k),
                    config=self.rigid_body_moves,
                )
            )
            if mover is not None:
                equilibration_moves = mover.run(
                    atoms,
                    attempts=self.rigid_body_moves.equilibration_attempts,
                    rng=rng,
                )
            else:
                equilibration_moves = None

            state_count = 0
            production_move_records: list[dict[str, float | int]] = []
            for sample_index in range(
                self.config.production_steps // self.config.sample_interval
            ):
                dynamics.run(int(self.config.sample_interval))
                if mover is not None:
                    production_move_records.append(
                        mover.run(
                            atoms,
                            attempts=self.rigid_body_moves.attempts_per_sample,
                            rng=rng,
                        ).as_dict()
                    )
                components = calculator.evaluate_full_basis(atoms)
                energy = float(components["total_ev"])
                basis = (
                    float(components["base_ev"]),
                    float(components["interaction_ev"]),
                    float(components["repulsive_ev"]),
                    float(components["restraint_ev"]),
                )
                if not math.isfinite(energy) or not np.all(np.isfinite(basis)):
                    raise RuntimeError(
                        "ALCHEMY_NONFINITE: sampled energy basis is non-finite."
                    )
                frame_positions = np.asarray(atoms.positions, dtype=float).copy()
                frame_id = canonical_sha256(
                    {
                        "schedule_hash": self.schedule.content_hash,
                        "config_hash": self.config.content_hash,
                        "rigid_body_move_hash": (
                            None
                            if self.rigid_body_moves is None
                            else self.rigid_body_moves.content_hash
                        ),
                        "state_index": state_index,
                        "state_label": label,
                        "sample_index": sample_index,
                        "atomic_numbers": [
                            int(value) for value in atoms.numbers
                        ],
                        "positions_angstrom": frame_positions.tolist(),
                    }
                )
                positions.append(frame_positions)
                basis_rows.append(basis)
                frame_ids.append(frame_id)
                state_count += 1
            counts.append(state_count)
            move_diagnostics.append(
                {
                    "state": label,
                    "move_config_hash": (
                        None
                        if self.rigid_body_moves is None
                        else self.rigid_body_moves.content_hash
                    ),
                    "equilibration": (
                        None
                        if equilibration_moves is None
                        else equilibration_moves.as_dict()
                    ),
                    "production": production_move_records,
                }
            )
            current = atoms.copy()
            current.calc = None

        self.last_move_diagnostics = tuple(move_diagnostics)
        return AlchemicalSampleSet.create(
            schedule=self.schedule,
            N_k=tuple(counts),
            frame_ids=tuple(frame_ids),
            positions_angstrom=np.asarray(positions, dtype=float),
            basis_ev=np.asarray(basis_rows, dtype=float),
            atom_list_hash=self.evaluator.partition.atom_list_hash,
            restraint_hash=self.restraint.content_hash,
            measure_id=self.restraint.measure_id,
            boundary_conditions="nonperiodic",
        )
