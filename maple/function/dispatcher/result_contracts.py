"""Immutable terminal states shared by reaction-path workflows."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
from ase import Atoms


class RunStatus(str, Enum):
    CONVERGED = "converged"
    BOUNDED_NONCONVERGED = "bounded_nonconverged"
    FAILED = "failed"
    REFUSED = "refused"


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
        if self.material_negative_mode_count is not None and self.material_negative_mode_count < 0:
            raise ValueError("A material negative-mode count cannot be negative.")
        object.__setattr__(
            self,
            "diagnostics",
            MappingProxyType(deepcopy(dict(self.diagnostics))),
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
        snapshot = _snapshot_atoms(self.atoms)
        object.__setattr__(self, "atoms", snapshot)
        if self.iterations < 0:
            raise ValueError("PRFO iteration count cannot be negative.")
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
        if self.energy_hartree is not None and not np.isfinite(self.energy_hartree):
            raise ValueError("PRFO final energy must be finite when present.")
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
        object.__setattr__(self, "budgets", MappingProxyType(dict(self.budgets)))
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
            budgets={"maximum_iterations": int(max_iterations), "used_iterations": int(iterations)},
            force_displacement_converged=force_displacement_converged,
            termination_reason=termination_reason,
            geometry_sha256=geometry_id,
            energy_hartree=None if energy_hartree is None else float(energy_hartree),
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
        snapshots = tuple(_snapshot_atoms(image) for image in self.images)
        object.__setattr__(self, "images", snapshots)
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
        object.__setattr__(self, "budgets", MappingProxyType(dict(self.budgets)))
        if self.candidate is None:
            if self.candidate_geometry_sha256 is not None:
                raise ValueError("A path-stage candidate identity requires a candidate.")
        elif self.candidate_geometry_sha256 != geometry_sha256(self.candidate):
            raise ValueError("The path-stage candidate identity does not match its snapshot.")

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
