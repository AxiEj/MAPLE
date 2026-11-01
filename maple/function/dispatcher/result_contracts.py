"""Immutable terminal states shared by reaction-path workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
from ase import Atoms

from .convergence import copy_convergence_context


class RunStatus(str, Enum):
    CONVERGED = "converged"
    BOUNDED_NONCONVERGED = "bounded_nonconverged"
    FAILED = "failed"
    REFUSED = "refused"


_HEX_DIGITS = frozenset("0123456789abcdef")
_PRECISION_PROFILES = frozenset(
    {
        "strict-generic",
        "qualified-aimnet2025-n32-gas",
        "qualified-aimnet2025-n32-torch-obc2",
    }
)


def _require_sha256(value: str, *, name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX_DIGITS for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase 64-character SHA256 digest.")


def _deep_freeze(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return tuple(_deep_freeze(item) for item in value.tolist())
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _deep_freeze(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_deep_freeze(item) for item in value)
    return value


def _freeze_budgets(value: Mapping[str, int]) -> Mapping[str, int]:
    if not isinstance(value, Mapping):
        raise TypeError("Result budgets must be a mapping.")
    budgets = dict(value)
    for name, count in budgets.items():
        if not isinstance(name, str) or not name:
            raise TypeError("Result budget names must be non-empty strings.")
        if type(count) is not int or count < 0:
            raise TypeError("Result budget counts must be non-negative exact integers.")
    maximum = budgets.get("maximum_iterations")
    used = budgets.get("used_iterations")
    if maximum is not None and used is not None and used > maximum:
        raise ValueError("Used iterations cannot exceed the maximum iteration budget.")
    return MappingProxyType(budgets)


def geometry_sha256(atoms: Atoms) -> str:
    identity = {
        "numbers": atoms.get_atomic_numbers().tolist(),
        "positions_angstrom": np.asarray(atoms.get_positions(), dtype=np.float64).tolist(),
        "cell_angstrom": np.asarray(atoms.cell, dtype=np.float64).tolist(),
        "pbc": np.asarray(atoms.pbc, dtype=bool).tolist(),
    }
    return hashlib.sha256(
        json.dumps(
            identity,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _snapshot_atoms(atoms: Atoms) -> Atoms:
    snapshot = atoms.copy()
    snapshot.calc = atoms.calc
    copy_convergence_context(atoms, snapshot)
    for name in ("max_f", "rms_f", "max_dp", "rms_dp"):
        if hasattr(atoms, name):
            setattr(snapshot, name, float(getattr(atoms, name)))
    return snapshot


def _readonly_array(value, *, shape: tuple[int, ...], name: str) -> np.ndarray:
    raw = np.asarray(value)
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must be real.")
    array = np.array(raw, dtype=np.float64, copy=True)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite array with shape {shape}.")
    array.setflags(write=False)
    return array


def _real_finite_scalar(value, *, name: str) -> float:
    raw = np.asarray(value)
    if raw.ndim != 0 or np.iscomplexobj(raw):
        raise ValueError(f"{name} must be a real finite scalar.")
    converted = np.asarray(raw, dtype=np.float64)
    if not np.isfinite(converted).all():
        raise ValueError(f"{name} must be a real finite scalar.")
    return float(converted.item())


@dataclass(frozen=True)
class SaddleCertification:
    geometry_sha256: str
    raw_hessian_sha256: str | None
    precision_profile: str | None
    material_negative_mode_count: int | None
    force_displacement_converged: bool
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    refusal_reason: str | None = None

    def __post_init__(self) -> None:
        _require_sha256(self.geometry_sha256, name="Certification geometry identity")
        if self.raw_hessian_sha256 is not None:
            _require_sha256(
                self.raw_hessian_sha256,
                name="Certification raw Hessian identity",
            )
        if self.precision_profile is not None and (
            not isinstance(self.precision_profile, str)
            or self.precision_profile not in _PRECISION_PROFILES
        ):
            raise ValueError("Certification precision profile is not recognized.")
        if self.material_negative_mode_count is not None and (
            type(self.material_negative_mode_count) is not int
            or self.material_negative_mode_count < 0
        ):
            raise TypeError(
                "A material negative-mode count must be a non-negative exact integer."
            )
        if type(self.force_displacement_converged) is not bool:
            raise TypeError("Certification convergence must be an exact boolean.")
        if self.refusal_reason is not None and (
            not isinstance(self.refusal_reason, str) or not self.refusal_reason
        ):
            raise ValueError("Certification refusal reason must be a non-empty string.")
        complete_curvature = (
            self.raw_hessian_sha256 is not None
            and self.precision_profile is not None
            and self.material_negative_mode_count is not None
        )
        if self.refusal_reason is None and not complete_curvature:
            raise ValueError(
                "A non-refused certification requires complete curvature evidence."
            )
        if self.refusal_reason is None and (
            not self.force_displacement_converged
            or self.material_negative_mode_count != 1
        ):
            raise ValueError(
                "A non-refused certification requires converged index-one evidence."
            )
        object.__setattr__(
            self,
            "diagnostics",
            _deep_freeze(dict(self.diagnostics)),
        )

    @property
    def certified_ts(self) -> bool:
        return (
            self.refusal_reason is None
            and self.force_displacement_converged
            and self.material_negative_mode_count == 1
            and bool(self.raw_hessian_sha256)
        )


@dataclass(frozen=True)
class PRFOResult:
    status: RunStatus
    atoms: Atoms
    iterations: int
    budgets: Mapping[str, int]
    force_displacement_converged: bool
    termination_reason: str
    geometry_sha256: str
    energy_hartree: float | None
    forces_hartree_per_angstrom: np.ndarray | None
    convergence_metrics: Mapping[str, float]
    certification: SaddleCertification | None = None
    output_paths: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.status, RunStatus):
            raise TypeError("PRFO status must be a RunStatus.")
        if type(self.iterations) is not int or self.iterations < 0:
            raise TypeError("PRFO iteration count must be a non-negative exact integer.")
        if type(self.force_displacement_converged) is not bool:
            raise TypeError("PRFO convergence must be an exact boolean.")
        snapshot = _snapshot_atoms(self.atoms)
        object.__setattr__(self, "atoms", snapshot)
        _require_sha256(self.geometry_sha256, name="PRFO geometry identity")
        if self.geometry_sha256 != geometry_sha256(self.atoms):
            raise ValueError("PRFO geometry identity does not match its snapshot.")
        if self.status is RunStatus.CONVERGED:
            if self.certification is None or not self.certification.certified_ts:
                raise ValueError("A converged PRFO result requires certified index-one curvature.")
            if not self.force_displacement_converged:
                raise ValueError("A certified TS requires force and displacement convergence.")
        elif self.certification is not None and self.certification.certified_ts:
            raise ValueError("A certified saddle cannot have a non-converged PRFO status.")
        if self.certification is not None and (
            self.certification.geometry_sha256 != self.geometry_sha256
        ):
            raise ValueError("PRFO certification and result geometry identities differ.")
        if self.energy_hartree is not None:
            object.__setattr__(
                self,
                "energy_hartree",
                _real_finite_scalar(self.energy_hartree, name="PRFO final energy"),
            )
        if self.forces_hartree_per_angstrom is not None:
            forces = _readonly_array(
                self.forces_hartree_per_angstrom,
                shape=(len(self.atoms), 3),
                name="PRFO final forces",
            )
            object.__setattr__(self, "forces_hartree_per_angstrom", forces)
        if self.status is RunStatus.CONVERGED and (
            self.energy_hartree is None or self.forces_hartree_per_angstrom is None
        ):
            raise ValueError("A certified TS requires finite final energy and forces.")
        budgets = _freeze_budgets(self.budgets)
        if "maximum_iterations" not in budgets or "used_iterations" not in budgets:
            raise ValueError(
                "PRFO budgets require maximum_iterations and used_iterations."
            )
        if budgets.get("used_iterations") != self.iterations:
            raise ValueError("PRFO used-iteration budget must equal its iteration count.")
        object.__setattr__(self, "budgets", budgets)
        object.__setattr__(
            self,
            "convergence_metrics",
            MappingProxyType(dict(self.convergence_metrics)),
        )
        object.__setattr__(self, "output_paths", MappingProxyType(dict(self.output_paths)))

    @property
    def certified_ts(self) -> bool:
        return self.status is RunStatus.CONVERGED and bool(
            self.certification and self.certification.certified_ts
        )

    @property
    def curvature(self) -> Mapping[str, Any] | None:
        return None if self.certification is None else self.certification.diagnostics

    @classmethod
    def capture(
        cls,
        *,
        status: RunStatus,
        atoms: Atoms,
        iterations: int,
        max_iterations: int,
        force_displacement_converged: bool,
        termination_reason: str,
        energy_hartree: float | None,
        forces_hartree_per_angstrom,
        certification: SaddleCertification | None,
        output_paths: Mapping[str, str],
    ) -> "PRFOResult":
        if type(iterations) is not int or type(max_iterations) is not int:
            raise TypeError("PRFO iteration counts must be exact integers.")
        if type(force_displacement_converged) is not bool:
            raise TypeError("PRFO convergence must be an exact boolean.")
        snapshot = _snapshot_atoms(atoms)
        geometry_id = geometry_sha256(snapshot)
        metrics = {
            name: float(getattr(snapshot, name))
            for name in ("max_f", "rms_f", "max_dp", "rms_dp")
            if hasattr(snapshot, name)
        }
        forces = None
        if forces_hartree_per_angstrom is not None:
            forces = _readonly_array(
                forces_hartree_per_angstrom,
                shape=(len(snapshot), 3),
                name="PRFO final forces",
            )
        return cls(
            status=status,
            atoms=snapshot,
            iterations=iterations,
            budgets={"maximum_iterations": max_iterations, "used_iterations": iterations},
            force_displacement_converged=force_displacement_converged,
            termination_reason=termination_reason,
            geometry_sha256=geometry_id,
            energy_hartree=(
                None
                if energy_hartree is None
                else _real_finite_scalar(
                    energy_hartree,
                    name="PRFO final energy",
                )
            ),
            forces_hartree_per_angstrom=forces,
            convergence_metrics=metrics,
            certification=certification,
            output_paths=dict(output_paths),
        )


@dataclass(frozen=True)
class PathStageResult:
    status: RunStatus
    images: tuple[Atoms, ...]
    energies_hartree: np.ndarray | None
    forces_hartree_per_angstrom: tuple[np.ndarray, ...] | None
    geometry_sha256: tuple[str, ...]
    budgets: Mapping[str, int]
    termination_reason: str
    candidate: Atoms | None = None
    candidate_geometry_sha256: str | None = None
    certification: SaddleCertification | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.status, RunStatus):
            raise TypeError("Path-stage status must be a RunStatus.")
        snapshots = tuple(_snapshot_atoms(image) for image in self.images)
        object.__setattr__(self, "images", snapshots)
        geometry_ids = tuple(self.geometry_sha256)
        object.__setattr__(self, "geometry_sha256", geometry_ids)
        if self.candidate is not None:
            object.__setattr__(self, "candidate", _snapshot_atoms(self.candidate))
        if len(self.geometry_sha256) != len(self.images):
            raise ValueError("Path-stage geometry identities must match the image count.")
        if any(
            digest != geometry_sha256(image)
            for image, digest in zip(self.images, self.geometry_sha256)
        ):
            raise ValueError("A path-stage geometry identity does not match its snapshot.")
        if self.energies_hartree is not None:
            object.__setattr__(
                self,
                "energies_hartree",
                _readonly_array(
                    self.energies_hartree,
                    shape=(len(self.images),),
                    name="path-stage energies",
                ),
            )
        if self.forces_hartree_per_angstrom is not None:
            if len(self.forces_hartree_per_angstrom) != len(self.images):
                raise ValueError("Path-stage force count must match the image count.")
            object.__setattr__(
                self,
                "forces_hartree_per_angstrom",
                tuple(
                    _readonly_array(value, shape=(len(image), 3), name="path-stage forces")
                    for image, value in zip(self.images, self.forces_hartree_per_angstrom)
                ),
            )
        object.__setattr__(self, "budgets", _freeze_budgets(self.budgets))
        if self.candidate is None:
            if self.candidate_geometry_sha256 is not None:
                raise ValueError("A path-stage candidate identity requires a candidate.")
        elif self.candidate_geometry_sha256 != geometry_sha256(self.candidate):
            raise ValueError("The path-stage candidate identity does not match its snapshot.")
        if self.certification is not None:
            designated_geometry = (
                self.candidate_geometry_sha256
                if self.candidate is not None
                else None
            )
            if designated_geometry is None:
                if self.certification.geometry_sha256 not in self.geometry_sha256:
                    raise ValueError(
                        "Path-stage certification does not match any stage geometry."
                    )
            elif self.certification.geometry_sha256 != designated_geometry:
                raise ValueError(
                    "Path-stage certification does not match the candidate geometry."
                )

    @classmethod
    def capture(
        cls,
        *,
        status: RunStatus,
        images,
        energies_hartree=None,
        forces_hartree_per_angstrom=None,
        budgets: Mapping[str, int],
        termination_reason: str,
        candidate: Atoms | None = None,
        certification: SaddleCertification | None = None,
    ) -> "PathStageResult":
        snapshots = tuple(_snapshot_atoms(image) for image in images)
        geometry_ids = tuple(geometry_sha256(image) for image in snapshots)
        energies = None
        if energies_hartree is not None:
            energies = _readonly_array(
                energies_hartree,
                shape=(len(snapshots),),
                name="path-stage energies",
            )
        forces = None
        if forces_hartree_per_angstrom is not None:
            if len(forces_hartree_per_angstrom) != len(snapshots):
                raise ValueError("Path-stage force count must match the image count.")
            forces = tuple(
                _readonly_array(value, shape=(len(image), 3), name="path-stage forces")
                for image, value in zip(snapshots, forces_hartree_per_angstrom)
            )
        candidate_snapshot = None if candidate is None else _snapshot_atoms(candidate)
        return cls(
            status=status,
            images=snapshots,
            energies_hartree=energies,
            forces_hartree_per_angstrom=forces,
            geometry_sha256=geometry_ids,
            budgets=dict(budgets),
            termination_reason=termination_reason,
            candidate=candidate_snapshot,
            candidate_geometry_sha256=(
                None if candidate_snapshot is None else geometry_sha256(candidate_snapshot)
            ),
            certification=certification,
        )
