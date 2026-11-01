"""Typed result states shared by transition-state workflows."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
from typing import Any

import numpy as np
from ase import Atoms


class PRFOStatus(str, Enum):
    CERTIFIED_TS = "certified_ts"
    BOUNDED_NONCONVERGED = "bounded_nonconverged"
    REJECTED_NOT_INDEX_ONE = "rejected_not_index_one"


@dataclass(frozen=True)
class PRFOResult:
    """Terminal PRFO state; certification is represented by one enum member."""

    status: PRFOStatus
    atoms: Atoms
    iterations: int
    force_displacement_converged: bool
    certified_ts: bool
    termination_reason: str
    energy_hartree: float
    forces_hartree_per_angstrom: np.ndarray
    geometry_sha256: str
    convergence_metrics: dict[str, float]
    curvature: dict[str, Any] | None = None
    output_paths: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        certified_status = self.status is PRFOStatus.CERTIFIED_TS
        if self.certified_ts is not certified_status:
            raise ValueError("PRFO certification must agree with the terminal status.")
        if self.certified_ts and not self.force_displacement_converged:
            raise ValueError("A certified TS requires force and displacement convergence.")
        if self.iterations < 0:
            raise ValueError("PRFO iteration count cannot be negative.")
        if not np.isfinite(self.energy_hartree):
            raise ValueError("PRFO final energy must be finite.")
        if (
            self.forces_hartree_per_angstrom.shape != (len(self.atoms), 3)
            or not np.isfinite(self.forces_hartree_per_angstrom).all()
        ):
            raise ValueError("PRFO final forces must be a finite (N, 3) array.")
        negative_count = (
            self.curvature.get("negative_mode_count")
            if isinstance(self.curvature, dict)
            else None
        )
        if self.certified_ts and negative_count != 1:
            raise ValueError("A certified TS requires fresh index-one curvature.")
        if (
            self.status is PRFOStatus.REJECTED_NOT_INDEX_ONE
            and (not self.force_displacement_converged or negative_count == 1)
        ):
            raise ValueError("A non-index-one refusal requires converged E/F geometry evidence.")

    @classmethod
    def capture(
        cls,
        *,
        status: PRFOStatus,
        atoms: Atoms,
        iterations: int,
        force_displacement_converged: bool,
        termination_reason: str,
        energy_hartree: float,
        forces_hartree_per_angstrom,
        curvature: dict[str, Any] | None,
        output_paths: dict[str, str],
    ) -> "PRFOResult":
        """Freeze the final geometry, E/F values, and identity in one record."""
        snapshot = atoms.copy()
        snapshot.calc = atoms.calc
        positions = np.asarray(snapshot.get_positions(), dtype=np.float64)
        forces = np.array(forces_hartree_per_angstrom, dtype=np.float64, copy=True)
        forces.setflags(write=False)
        identity = {
            "numbers": snapshot.get_atomic_numbers().tolist(),
            "positions_angstrom": positions.tolist(),
            "cell_angstrom": np.asarray(snapshot.cell, dtype=np.float64).tolist(),
            "pbc": np.asarray(snapshot.pbc, dtype=bool).tolist(),
        }
        geometry_sha256 = hashlib.sha256(
            json.dumps(
                identity,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        metrics = {
            name: float(getattr(atoms, name))
            for name in ("max_f", "rms_f", "max_dp", "rms_dp")
            if hasattr(atoms, name)
        }
        for name, value in metrics.items():
            setattr(snapshot, name, value)
        return cls(
            status=status,
            atoms=snapshot,
            iterations=iterations,
            force_displacement_converged=force_displacement_converged,
            certified_ts=status is PRFOStatus.CERTIFIED_TS,
            termination_reason=termination_reason,
            energy_hartree=float(energy_hartree),
            forces_hartree_per_angstrom=forces,
            geometry_sha256=geometry_sha256,
            convergence_metrics=metrics,
            curvature=deepcopy(curvature),
            output_paths=dict(output_paths),
        )
