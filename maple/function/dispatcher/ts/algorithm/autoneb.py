# -*- coding: utf-8 -*-
"""MAPLE-owned standard single-path AutoNEB scheduler.

The insertion policy follows the space and energy resolution rule used by ASE
3.27.0 ``ase.mep.autoneb.AutoNEB`` and E. L. Kolsbjerg, M. N. Groves, and
B. Hammer, J. Chem. Phys. 145, 094107 (2016), doi:10.1063/1.4961868. MAPLE owns
image metadata, complete-potential evaluation, stage results, and the NEB
optimizer kernel. This module does not use ASE's file-oriented AutoNEB runner.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np
from ase import Atoms

from maple.function.dispatcher.result_contracts import (
    PathStageResult,
    RunStatus,
    geometry_sha256,
)
from maple.function.utility import Molecules

from ...jobABC import JobABC


_OBSOLETE_FIELDS = frozenset(
    {
        "n_images",
        "ang_max",
        "ang_iter",
        "path_iter",
        "min_e_drop",
        "ep_iter",
        "ep_e_drop",
        "max_depth",
        "max_paths",
        "do_final_refine",
        "final_refine_factor",
        "final_refine_max_iter",
        "final_refine_ep_iter",
        "max_steps",
    }
)
_IDENTITY_MARKERS = ("source", "prepared", "parameter", "topology", "charge", "mass")


def _exact_nonnegative_int(value, *, name: str) -> int:
    if type(value) is not int or value < 0:
        raise TypeError(f"{name} must be a non-negative exact integer.")
    return value


def _positive_finite(value, *, name: str) -> float:
    raw = np.asarray(value)
    if raw.ndim != 0 or np.iscomplexobj(raw):
        raise ValueError(f"{name} must be a positive finite real number.")
    converted = float(raw)
    if not np.isfinite(converted) or converted <= 0.0:
        raise ValueError(f"{name} must be a positive finite real number.")
    return converted


@dataclass(frozen=True)
class AutoNEBParams:
    """Supported standard AutoNEB parameters."""

    n_max: int = 20
    max_iter: int = 256
    fmax: float = 9.5e-3
    rms_fmax: float = 5.0e-3
    climb: bool = True
    ci_max_iter: int = 200
    ci_fmax: float = 1.0e-2
    ci_rms_fmax: float = 1.0e-2
    ci_displacement_tol: float = 1.0e-3
    space_energy_ratio: float = 1.0
    interpolation: str = "idpp"
    verbose: int = 1

    def __post_init__(self) -> None:
        if type(self.n_max) is not int or self.n_max < 2:
            raise ValueError("n_max must be an exact integer of at least 2.")
        _exact_nonnegative_int(self.max_iter, name="max_iter")
        _exact_nonnegative_int(self.ci_max_iter, name="ci_max_iter")
        if type(self.climb) is not bool:
            raise TypeError("climb must be an exact boolean.")
        if self.climb and self.n_max < 3:
            raise ValueError("climb=true requires an interior image at n_max.")
        for name in (
            "fmax",
            "rms_fmax",
            "ci_fmax",
            "ci_rms_fmax",
            "ci_displacement_tol",
            "space_energy_ratio",
        ):
            object.__setattr__(self, name, _positive_finite(getattr(self, name), name=name))
        if self.interpolation not in {"linear", "idpp"}:
            raise ValueError("interpolation must be 'linear' or 'idpp'.")
        _exact_nonnegative_int(self.verbose, name="verbose")


@dataclass(frozen=True)
class AutoNEBResult:
    """Structured workflow result with immutable stage snapshots."""

    final_stage: PathStageResult
    base_stages: tuple[PathStageResult, ...]
    ci_stage: PathStageResult | None
    initial_image_count: int
    insertion_count: int
    termination_reason: str
    scheduler_counts: Mapping[str, int]

    def __post_init__(self) -> None:
        _exact_nonnegative_int(self.initial_image_count, name="initial_image_count")
        _exact_nonnegative_int(self.insertion_count, name="insertion_count")
        if len(self.final_stage.images) != self.initial_image_count + self.insertion_count:
            raise ValueError("AutoNEB image and insertion counts do not agree.")
        counts = dict(self.scheduler_counts)
        for name, value in counts.items():
            _exact_nonnegative_int(value, name=f"scheduler_counts[{name!r}]")
        if counts.get("base_stages") != len(self.base_stages):
            raise ValueError("AutoNEB base-stage count does not match its ledger.")
        if counts.get("ci_stages") != int(self.ci_stage is not None):
            raise ValueError("AutoNEB CI-stage count does not match its ledger.")
        object.__setattr__(self, "scheduler_counts", MappingProxyType(counts))

    @property
    def status(self) -> RunStatus:
        return self.final_stage.status

    @property
    def images(self) -> tuple[Atoms, ...]:
        return self.final_stage.images

    @property
    def energies_hartree(self) -> np.ndarray | None:
        return self.final_stage.energies_hartree

    @property
    def forces_hartree_per_angstrom(self) -> tuple[np.ndarray, ...] | None:
        return self.final_stage.forces_hartree_per_angstrom

    @property
    def candidate(self) -> Atoms | None:
        return self.final_stage.candidate

    @property
    def final_image_count(self) -> int:
        return len(self.final_stage.images)

    @property
    def base_stage_count(self) -> int:
        return len(self.base_stages)

    @property
    def certified_ts(self) -> bool:
        return False


def _positions(images: Sequence[Atoms]) -> list[np.ndarray]:
    return [np.asarray(image.get_positions(), dtype=np.float64) for image in images]


def select_insertion_interval(
    positions: Sequence[np.ndarray],
    energies: Sequence[float],
    *,
    space_energy_ratio: float,
) -> tuple[int, str]:
    """Select one adjacent interval by the pinned ASE 3.27 resolution rule."""
    ratio = _positive_finite(space_energy_ratio, name="space_energy_ratio")
    if len(positions) != len(energies) or len(positions) < 2:
        raise ValueError("Insertion selection requires aligned positions and energies.")
    coordinates = [np.asarray(value, dtype=np.float64) for value in positions]
    if any(value.shape != coordinates[0].shape for value in coordinates):
        raise ValueError("All AutoNEB images must have the same position shape.")
    if any(not np.isfinite(value).all() for value in coordinates):
        raise ValueError("AutoNEB positions must be finite.")
    energy_array = np.asarray(energies)
    if np.iscomplexobj(energy_array):
        raise ValueError("AutoNEB energies must be real and finite.")
    energy_array = np.asarray(energy_array, dtype=np.float64)
    if energy_array.shape != (len(coordinates),) or not np.isfinite(energy_array).all():
        raise ValueError("AutoNEB energies must be real and finite.")

    spring_lengths = np.asarray(
        [np.linalg.norm(right - left) for left, right in zip(coordinates, coordinates[1:])],
        dtype=np.float64,
    )
    spatial_index = int(np.argmax(spring_lengths))
    endpoint_distance = float(np.linalg.norm(coordinates[0] - coordinates[-1]))
    energy_minimum = float(np.min(energy_array))
    energy_range = float(np.max(energy_array) - energy_minimum)
    if energy_range <= np.finfo(np.float64).eps:
        return spatial_index, "space"

    energy_resolution = np.abs(
        np.diff(energy_array)
        * (energy_array[1:] + energy_array[:-1] - 2.0 * energy_minimum)
        / (2.0 * energy_range)
    )
    energy_index = int(np.argmax(energy_resolution))
    maximum_energy_resolution = float(energy_resolution[energy_index] / energy_range)
    if maximum_energy_resolution <= np.finfo(np.float64).eps:
        return spatial_index, "space"

    if endpoint_distance <= np.finfo(np.float64).eps:
        return spatial_index, "space"
    maximum_spatial_resolution = float(np.max(spring_lengths) / endpoint_distance)
    if maximum_spatial_resolution / maximum_energy_resolution > ratio:
        return spatial_index, "space"
    return energy_index, "energy"


def _copy_with_calculator(atoms: Atoms) -> Atoms:
    copied = atoms.copy()
    copied.calc = atoms.calc
    return copied


def _idpp_midpoint(left: Atoms, right: Atoms) -> np.ndarray:
    """Return one deterministic IDPP midpoint without ASE's AutoNEB runtime."""
    left_positions = np.asarray(left.get_positions(), dtype=np.float64)
    right_positions = np.asarray(right.get_positions(), dtype=np.float64)
    midpoint = 0.5 * (left_positions + right_positions)
    if len(left) < 2:
        return midpoint

    pair_i, pair_j = np.triu_indices(len(left), 1)
    left_distances = np.linalg.norm(left_positions[pair_i] - left_positions[pair_j], axis=1)
    right_distances = np.linalg.norm(right_positions[pair_i] - right_positions[pair_j], axis=1)
    targets = 0.5 * (left_distances + right_distances)
    if np.any(targets <= 1.0e-12):
        return midpoint

    positions = midpoint.copy()
    for _ in range(100):
        vectors = positions[pair_i] - positions[pair_j]
        distances = np.linalg.norm(vectors, axis=1)
        if np.any(distances <= 1.0e-12):
            return midpoint
        inverse_error = 1.0 / distances - 1.0 / targets
        derivatives = -2.0 * inverse_error / distances**2
        pair_gradients = derivatives[:, None] * vectors / distances[:, None]
        gradient = np.zeros_like(positions)
        np.add.at(gradient, pair_i, pair_gradients)
        np.add.at(gradient, pair_j, -pair_gradients)
        rms_gradient = float(np.sqrt(np.mean(gradient * gradient)))
        if rms_gradient < 1.0e-5:
            break
        step = -gradient
        maximum_component = float(np.max(np.abs(step)))
        if maximum_component > 0.05:
            step *= 0.05 / maximum_component
        positions += step
    return positions


def _insert_image(images: Sequence[Atoms], interval: int, interpolation: str) -> list[Atoms]:
    if interval < 0 or interval >= len(images) - 1:
        raise IndexError("AutoNEB insertion interval is outside the path.")
    left, right = images[interval], images[interval + 1]
    inserted = _copy_with_calculator(left)
    if interpolation == "linear":
        positions = 0.5 * (left.get_positions() + right.get_positions())
    elif interpolation == "idpp":
        positions = _idpp_midpoint(left, right)
    else:
        raise ValueError("interpolation must be 'linear' or 'idpp'.")
    inserted.set_positions(positions)
    inserted.info["path_role"] = "autoneb_inserted"
    return [
        *(_copy_with_calculator(image) for image in images[: interval + 1]),
        inserted,
        *(_copy_with_calculator(image) for image in images[interval + 1 :]),
    ]


def _load_neb_stage_api():
    from .neb import NEBParams, run_neb_stage

    return NEBParams, run_neb_stage


def _evaluate_path_images(images: Sequence[Atoms]):
    from maple.function.dispatcher.implicit_path import evaluate_path_images

    return evaluate_path_images(list(images), label="AutoNEB")


def _highest_energy_interior(stage: PathStageResult) -> Atoms | None:
    if len(stage.images) < 3 or stage.energies_hartree is None:
        return None
    index = 1 + int(np.argmax(stage.energies_hartree[1:-1]))
    return stage.images[index]


def _require_complete_stage(
    stage: PathStageResult,
    *,
    expected_images: int,
    endpoint_geometry: tuple[str, str],
) -> None:
    if not isinstance(stage, PathStageResult):
        raise TypeError("The NEB stage kernel must return PathStageResult.")
    if len(stage.images) != expected_images:
        raise RuntimeError("The NEB stage kernel changed the AutoNEB image count.")
    if stage.energies_hartree is None or stage.forces_hartree_per_angstrom is None:
        raise RuntimeError("The NEB stage kernel did not return aligned energy and force snapshots.")
    if (stage.geometry_sha256[0], stage.geometry_sha256[-1]) != endpoint_geometry:
        raise RuntimeError("The NEB stage kernel moved a fixed AutoNEB endpoint.")


class AutoNEB(JobABC):
    """Grow and relax one adaptive path, then optionally run one CI stage."""

    def __init__(self, output: str, atoms_or_molecules, paras: Mapping | None = None):
        super().__init__(output)
        if isinstance(atoms_or_molecules, Molecules):
            supplied = atoms_or_molecules.multiatoms
        elif isinstance(atoms_or_molecules, list):
            supplied = atoms_or_molecules
        else:
            raise ValueError("AutoNEB requires a Molecules object or a list of Atoms.")
        if len(supplied) < 2:
            raise ValueError("AutoNEB requires at least two images.")
        self.input_images = [_copy_with_calculator(image) for image in supplied]
        self.input_images[0].info["path_role"] = "reactant"
        self.input_images[-1].info["path_role"] = "product"
        for image in self.input_images[1:-1]:
            image.info.setdefault("path_role", "autoneb_input")
        self.params = self._parse_params(paras)
        if self.params.n_max < len(self.input_images):
            raise ValueError("n_max must be at least the initial image count.")
        self._validate_image_identity(self.input_images)
        self._calculator = self.input_images[0].calc
        self._reference_image = _copy_with_calculator(self.input_images[0])
        self.last_result: AutoNEBResult | None = None

    @staticmethod
    def _parse_params(paras: Mapping | None) -> AutoNEBParams:
        if paras is None:
            selected = {}
        elif not isinstance(paras, Mapping):
            raise TypeError("AutoNEB parameters must be a mapping.")
        else:
            lowered = {
                (key.lower() if isinstance(key, str) else key): value
                for key, value in paras.items()
            }
            nested = next(
                (lowered[name] for name in ("autoneb", "ts") if isinstance(lowered.get(name), Mapping)),
                None,
            )
            selected = dict(nested if nested is not None else lowered)
        selected = {
            (key.lower() if isinstance(key, str) else key): value
            for key, value in selected.items()
        }
        obsolete = sorted(str(key) for key in selected if key in _OBSOLETE_FIELDS)
        if obsolete:
            raise ValueError(f"Obsolete AutoNEB parameter: {obsolete[0]}")
        allowed = {field.name for field in fields(AutoNEBParams)}
        unknown = sorted(str(key) for key in selected if key not in allowed)
        if unknown:
            raise ValueError(f"Unknown AutoNEB parameter: {unknown[0]}")
        return AutoNEBParams(**selected)

    @staticmethod
    def _validate_image_identity(images: Sequence[Atoms]) -> None:
        calculator = images[0].calc
        if calculator is None or any(image.calc is not calculator for image in images):
            raise ValueError("All AutoNEB images must share one exact calculator instance.")
        reference = images[0]
        reference_arrays = {
            name: value for name, value in reference.arrays.items() if name != "positions"
        }
        identity_keys = {
            key
            for image in images
            for key in image.info
            if any(marker in str(key).lower() for marker in _IDENTITY_MARKERS)
        }
        if not np.isfinite(reference.get_positions()).all():
            raise ValueError("AutoNEB positions must be finite.")
        for image in images[1:]:
            if len(image) != len(reference) or not np.array_equal(
                image.get_atomic_numbers(), reference.get_atomic_numbers()
            ):
                raise ValueError("All AutoNEB images must share one topology.")
            arrays = {name: value for name, value in image.arrays.items() if name != "positions"}
            if arrays.keys() != reference_arrays.keys() or any(
                not np.array_equal(arrays[name], reference_arrays[name]) for name in arrays
            ):
                raise ValueError("All AutoNEB images must share frozen atomic metadata.")
            if any(image.info.get(key) != reference.info.get(key) for key in identity_keys):
                raise ValueError("All AutoNEB images must share frozen source metadata.")
            if not np.isfinite(image.get_positions()).all():
                raise ValueError("AutoNEB positions must be finite.")

    def _validate_stage_identity(
        self,
        images: Sequence[Atoms],
        *,
        expected_roles: Sequence[str],
    ) -> None:
        self._validate_image_identity(images)
        if [image.info.get("path_role") for image in images] != list(expected_roles):
            raise RuntimeError("The NEB stage kernel changed AutoNEB image roles.")
        if any(image.calc is not self._calculator for image in images):
            raise RuntimeError("The NEB stage kernel replaced the prepared calculator.")
        reference_arrays = {
            name: value
            for name, value in self._reference_image.arrays.items()
            if name != "positions"
        }
        identity_keys = {
            key
            for key in self._reference_image.info
            if any(marker in str(key).lower() for marker in _IDENTITY_MARKERS)
        }
        for image in images:
            arrays = {name: value for name, value in image.arrays.items() if name != "positions"}
            if arrays.keys() != reference_arrays.keys() or any(
                not np.array_equal(arrays[name], reference_arrays[name]) for name in arrays
            ):
                raise RuntimeError("The NEB stage kernel changed frozen atomic metadata.")
            if any(
                image.info.get(key) != self._reference_image.info.get(key)
                for key in identity_keys
            ):
                raise RuntimeError("The NEB stage kernel changed frozen source metadata.")

    def _log(self, lines: list[str]) -> None:
        if self.params.verbose:
            self.log_info(lines)

    def _neb_params(self):
        params_class, _ = _load_neb_stage_api()
        params = params_class()
        assignments = {
            "max_iter": self.params.max_iter,
            "neb_f_max_th": self.params.fmax,
            "neb_f_rms_th": self.params.rms_fmax,
            "cineb_max_iter": self.params.ci_max_iter,
            "cineb_f_max_th": self.params.ci_fmax,
            "cineb_f_rms_th": self.params.ci_rms_fmax,
            "ci_displacement_tol": self.params.ci_displacement_tol,
        }
        for name, value in assignments.items():
            if not hasattr(params, name):
                raise RuntimeError(
                    f"The reviewed NEB stage does not support AutoNEB parameter {name}."
                )
            setattr(params, name, value)
        return params

    def _result(
        self,
        *,
        final_stage: PathStageResult,
        base_stages: Sequence[PathStageResult],
        ci_stage: PathStageResult | None,
        initial_count: int,
        insertion_count: int,
        reason: str,
    ) -> AutoNEBResult:
        result = AutoNEBResult(
            final_stage=final_stage,
            base_stages=tuple(base_stages),
            ci_stage=ci_stage,
            initial_image_count=initial_count,
            insertion_count=insertion_count,
            termination_reason=reason,
            scheduler_counts={
                "base_stages": len(base_stages),
                "ci_stages": int(ci_stage is not None),
                "insertions": insertion_count,
                "final_images": len(final_stage.images),
            },
        )
        self.last_result = result
        return result

    def run(self) -> AutoNEBResult:
        images = [_copy_with_calculator(image) for image in self.input_images]
        initial_count = len(images)
        endpoint_geometry = (geometry_sha256(images[0]), geometry_sha256(images[-1]))
        energies, forces = _evaluate_path_images(images)
        initial_stage = PathStageResult.capture(
            status=RunStatus.BOUNDED_NONCONVERGED,
            images=images,
            energies_hartree=energies,
            forces_hartree_per_angstrom=forces,
            budgets={"maximum_iterations": self.params.max_iter, "used_iterations": 0},
            termination_reason="initial_path_evaluated",
        )
        if self.params.max_iter == 0:
            return self._result(
                final_stage=initial_stage,
                base_stages=(),
                ci_stage=None,
                initial_count=initial_count,
                insertion_count=0,
                reason="base_budget_exhausted_before_relaxation",
            )

        _, run_neb_stage = _load_neb_stage_api()
        neb_params = self._neb_params()
        base_stages: list[PathStageResult] = []
        insertion_count = 0
        current = initial_stage
        needs_initial_relaxation = len(images) == self.params.n_max

        while len(images) < self.params.n_max or needs_initial_relaxation:
            needs_initial_relaxation = False
            if len(images) < self.params.n_max:
                interval, basis = select_insertion_interval(
                    _positions(images),
                    current.energies_hartree,
                    space_energy_ratio=self.params.space_energy_ratio,
                )
                images = _insert_image(images, interval, self.params.interpolation)
                insertion_count += 1
                self._log(
                    [
                        f"AutoNEB insertion {insertion_count}: interval={interval}, ",
                        f"basis={basis}, images={len(images)}\n",
                    ]
                )
            expected_roles = [image.info["path_role"] for image in images]
            stage = run_neb_stage(
                images,
                params=neb_params,
                stage_kind="base",
                output=self.output,
            )
            _require_complete_stage(
                stage,
                expected_images=len(images),
                endpoint_geometry=endpoint_geometry,
            )
            self._validate_stage_identity(stage.images, expected_roles=expected_roles)
            base_stages.append(stage)
            current = stage
            images = [_copy_with_calculator(image) for image in stage.images]
            if stage.status is not RunStatus.CONVERGED:
                return self._result(
                    final_stage=stage,
                    base_stages=base_stages,
                    ci_stage=None,
                    initial_count=initial_count,
                    insertion_count=insertion_count,
                    reason="base_growth_stage_not_converged",
                )

        candidate = _highest_energy_interior(current)
        base_final = PathStageResult.capture(
            status=current.status,
            images=current.images,
            energies_hartree=current.energies_hartree,
            forces_hartree_per_angstrom=current.forces_hartree_per_angstrom,
            budgets=current.budgets,
            termination_reason=current.termination_reason,
            candidate=candidate,
        )
        if not self.params.climb:
            return self._result(
                final_stage=base_final,
                base_stages=base_stages,
                ci_stage=None,
                initial_count=initial_count,
                insertion_count=insertion_count,
                reason="growth_converged_ci_disabled",
            )
        if candidate is None:
            raise RuntimeError("AutoNEB CI requires an interior highest-energy candidate.")
        if self.params.ci_max_iter == 0:
            ci_stage = PathStageResult.capture(
                status=RunStatus.BOUNDED_NONCONVERGED,
                images=base_final.images,
                energies_hartree=base_final.energies_hartree,
                forces_hartree_per_angstrom=base_final.forces_hartree_per_angstrom,
                budgets={"maximum_iterations": 0, "used_iterations": 0},
                termination_reason="ci_budget_exhausted_before_relaxation",
                candidate=candidate,
            )
            return self._result(
                final_stage=ci_stage,
                base_stages=base_stages,
                ci_stage=ci_stage,
                initial_count=initial_count,
                insertion_count=insertion_count,
                reason="ci_budget_exhausted_before_relaxation",
            )

        expected_roles = [image.info["path_role"] for image in base_final.images]
        ci_stage = run_neb_stage(
            list(base_final.images),
            params=neb_params,
            stage_kind="ci",
            output=self.output,
        )
        _require_complete_stage(
            ci_stage,
            expected_images=len(base_final.images),
            endpoint_geometry=endpoint_geometry,
        )
        self._validate_stage_identity(
            ci_stage.images,
            expected_roles=expected_roles,
        )
        diagnostics = getattr(ci_stage, "_stage_diagnostics", None)
        if not isinstance(diagnostics, Mapping):
            raise RuntimeError("The CI stage did not return displacement diagnostics.")
        climbing_index = diagnostics.get("climbing_image_index")
        if (
            type(climbing_index) is not int
            or climbing_index <= 0
            or climbing_index >= len(ci_stage.images) - 1
        ):
            raise RuntimeError("The CI stage did not return an interior climbing-image index.")
        displacement = diagnostics.get("ci_max_displacement_angstrom")
        displacement_array = np.asarray(displacement)
        if displacement_array.ndim != 0 or np.iscomplexobj(displacement_array):
            raise RuntimeError("The CI stage did not return a valid displacement metric.")
        try:
            displacement_value = float(displacement_array)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "The CI stage did not return a valid displacement metric."
            ) from exc
        if not np.isfinite(displacement_value) or displacement_value < 0.0:
            raise RuntimeError("The CI stage did not return a valid displacement metric.")
        if (
            ci_stage.status is RunStatus.CONVERGED
            and displacement_value > self.params.ci_displacement_tol
        ):
            raise RuntimeError("The CI stage convergence contradicts its displacement metric.")
        ci_candidate = ci_stage.candidate
        if (
            ci_candidate is None
            or geometry_sha256(ci_candidate)
            != ci_stage.geometry_sha256[climbing_index]
        ):
            raise RuntimeError("The CI stage did not return an interior candidate.")
        final_stage = ci_stage
        return self._result(
            final_stage=final_stage,
            base_stages=base_stages,
            ci_stage=final_stage,
            initial_count=initial_count,
            insertion_count=insertion_count,
            reason=(
                "ci_converged_candidate_uncertified"
                if final_stage.status is RunStatus.CONVERGED
                else "ci_stage_not_converged"
            ),
        )
