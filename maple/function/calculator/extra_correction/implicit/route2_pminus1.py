"""Frozen numerical profile for Route 2 P-minus-1 water diagnostics.

This profile is intentionally independent of the SMD/CDS profile registry.
It admits only neutral H/O, water, electrostatics-only response diagnostics and
constructs a fixed-cavity pyddx/ddPCM map for each geometry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from .pyddx_pcm_response import PyDDXPCMReactionFieldLinearMap
from .route2_engine import Route2EngineSettings
from .route2_fixed_point import SAFEGUARDED_ANDERSON_SOLVER
from .route2_plugin_contracts import P_MINUS_1_WATER_RESPONSE_ONLY_PROFILE

P_MINUS_1_WATER_CONTINUUM_SPEC_ID = "route2-pminus1-water-ddpcm-fixed-cavity-v1"


@dataclass(frozen=True)
class ElectrostaticsOnlyCDSResult:
    """Explicit zero CDS leaf for the P-minus-1 electrostatics-only ledger."""

    energy_hartree: float = 0.0


@dataclass(frozen=True)
class PMinus1WaterContinuumSpec:
    """All numerical choices needed to reproduce the primary P-minus-1 map."""

    identity: str = P_MINUS_1_WATER_CONTINUUM_SPEC_ID
    dielectric: float = 78.39
    lmax: int = 7
    n_lebedev: int = 302
    n_proc: int = 1
    solver_tolerance: float = 1.0e-12
    eta: float = 0.1
    radii_angstrom_by_atomic_number: Mapping[int, float] = field(
        default_factory=lambda: MappingProxyType({1: 1.20, 8: 1.52})
    )

    def __post_init__(self) -> None:
        if self.identity != P_MINUS_1_WATER_CONTINUUM_SPEC_ID:
            raise ValueError("unsupported P-minus-1 continuum identity.")
        if not np.isfinite(self.dielectric) or self.dielectric <= 1.0:
            raise ValueError("P-minus-1 dielectric must exceed one.")
        for name in ("lmax", "n_lebedev", "n_proc"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"P-minus-1 {name} must be a positive integer.")
        if self.lmax < 1:
            raise ValueError("P-minus-1 lmax must be at least one.")
        if not np.isfinite(self.solver_tolerance) or self.solver_tolerance <= 0.0:
            raise ValueError("P-minus-1 solver tolerance must be positive.")
        if not np.isfinite(self.eta) or not 0.0 <= self.eta <= 1.0:
            raise ValueError("P-minus-1 eta must lie in [0, 1].")
        radii = {
            int(number): float(radius)
            for number, radius in self.radii_angstrom_by_atomic_number.items()
        }
        if set(radii) != {1, 8} or any(
            number < 1 or not np.isfinite(radius) or radius <= 0.0
            for number, radius in radii.items()
        ):
            raise ValueError("P-minus-1 radii must define positive H/O values only.")
        object.__setattr__(
            self,
            "radii_angstrom_by_atomic_number",
            MappingProxyType(radii),
        )

    def radii_angstrom(self, atomic_numbers: object) -> np.ndarray:
        numbers = np.asarray(atomic_numbers)
        if (
            numbers.ndim != 1
            or numbers.size == 0
            or not np.issubdtype(numbers.dtype, np.integer)
        ):
            raise ValueError("P-minus-1 atomic numbers must be a nonempty vector.")
        unsupported = sorted(set(map(int, numbers)) - {1, 8})
        if unsupported:
            raise ValueError(
                "P-minus-1 fixed cavity excludes atomic numbers: "
                + ", ".join(map(str, unsupported))
                + "."
            )
        result = np.asarray(
            [self.radii_angstrom_by_atomic_number[int(number)] for number in numbers],
            dtype=float,
        )
        result.setflags(write=False)
        return result

    def as_provenance(self) -> dict[str, object]:
        return {
            "identity": self.identity,
            "plugin_profile": P_MINUS_1_WATER_RESPONSE_ONLY_PROFILE.name,
            "solvent": "water",
            "continuum_backend": "pyddx",
            "continuum_equation": "ddPCM",
            "dielectric": self.dielectric,
            "lmax": self.lmax,
            "n_lebedev": self.n_lebedev,
            "n_proc": self.n_proc,
            "solver_tolerance": self.solver_tolerance,
            "eta": self.eta,
            "radii_angstrom_by_atomic_number": {
                str(number): radius
                for number, radius in self.radii_angstrom_by_atomic_number.items()
            },
            "cavity_policy": P_MINUS_1_WATER_RESPONSE_ONLY_PROFILE.cavity_policy,
            "electrostatics_only": True,
            "include_cds": False,
        }

    def cavity_sha256(self, atoms: Any) -> str:
        positions = np.asarray(atoms.get_positions(), dtype="<f8")
        numbers = np.asarray(atoms.numbers, dtype="<i8")
        radii = np.asarray(self.radii_angstrom(numbers), dtype="<f8")
        if positions.shape != (numbers.size, 3) or not np.all(np.isfinite(positions)):
            raise ValueError("P-minus-1 cavity geometry is invalid.")
        digest = hashlib.sha256()
        digest.update(numbers.tobytes(order="C"))
        digest.update(positions.tobytes(order="C"))
        digest.update(radii.tobytes(order="C"))
        digest.update(
            json.dumps(
                self.as_provenance(),
                sort_keys=True,
                separators=(",", ":"),
            ).encode("ascii")
        )
        return digest.hexdigest()

    def build_reaction_field(self, atoms: Any) -> PyDDXPCMReactionFieldLinearMap:
        return PyDDXPCMReactionFieldLinearMap(
            np.asarray(atoms.get_positions(), dtype=float),
            self.radii_angstrom(atoms.numbers),
            dielectric=self.dielectric,
            lmax=self.lmax,
            n_lebedev=self.n_lebedev,
            n_proc=self.n_proc,
            solver_tolerance=self.solver_tolerance,
            eta=self.eta,
        )

    @staticmethod
    def engine_settings() -> Route2EngineSettings:
        return Route2EngineSettings(
            continuum_label="P-1 pyddx/ddPCM water",
            scf_mixing=1.0,
            scf_density_tolerance=2.0e-12,
            scf_dipole_tolerance_e_angstrom=2.0e-12,
            scf_energy_tolerance_ev=2.0e-12,
            scf_max_iterations=80,
            adjoint_relative_tolerance=1.0e-10,
            adjoint_absolute_tolerance=1.0e-13,
            adjoint_max_iterations=100,
            energy_identity_tolerance_ev=1.0e-10,
            force_state_energy_tolerance_ev=1.0e-10,
            neutral_density_tolerance=1.0e-8,
            scf_solver=SAFEGUARDED_ANDERSON_SOLVER,
            scf_anderson_depth=6,
            scf_anderson_regularization=1.0e-12,
            scf_anderson_coefficient_l1_limit=100.0,
            scf_anderson_step_ratio_limit=100.0,
            scf_anderson_residual_growth_limit=2.0,
            scf_total_charge_e=0.0,
            scf_raw_response_charge_tolerance_e=2.0e-12,
            scf_total_charge_residual_tolerance_e=2.0e-12,
            scf_molecular_dipole_tolerance_e_angstrom=2.0e-12,
            scf_require_two_energy_samples=True,
        )


P_MINUS_1_WATER_CONTINUUM_SPEC = PMinus1WaterContinuumSpec()


__all__ = [
    "P_MINUS_1_WATER_CONTINUUM_SPEC",
    "P_MINUS_1_WATER_CONTINUUM_SPEC_ID",
    "ElectrostaticsOnlyCDSResult",
    "PMinus1WaterContinuumSpec",
]
