"""Error-estimated forces obtained from one registered scalar.

This is deliberately a scalar-first fallback for providers whose native
runtime exposes energies but no nuclear derivative.  It never finite-
differences an independently assembled force or response map.  Every stencil
point rebuilds the provider state and evaluates the same scalar, so the result
is an explicitly error-bounded numerical approximation to ``-dE/dR``.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Protocol, runtime_checkable

import numpy as np

RICHARDSON_FORCE_CONTRACT = "scalar-central-richardson-force-v1"


def _digest(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA256 digest.")
    return value


def _canonical_sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _positions(geometry: object) -> np.ndarray:
    values = np.asarray(getattr(geometry, "positions", None), dtype=float)
    if values.ndim != 2 or values.shape[1] != 3 or not np.all(np.isfinite(values)):
        raise ValueError("geometry.positions must be finite with shape (N,3).")
    return values


def _displaced(geometry: object, atom: int, axis: int, delta: float) -> object:
    copier = getattr(geometry, "copy", None)
    if not callable(copier):
        raise TypeError("finite-difference geometry must provide copy().")
    result = copier()
    positions = _positions(result).copy()
    positions[atom, axis] += delta
    setter = getattr(result, "set_positions", None)
    if callable(setter):
        setter(positions)
    else:
        setattr(result, "positions", positions)
    return result


@dataclass(frozen=True, slots=True)
class ScalarEnergySample:
    """One immutable scalar evaluation used by a force stencil."""

    energy_eV: float
    state_sha256: str
    topology_id: str

    def __post_init__(self) -> None:
        energy = float(self.energy_eV)
        if not np.isfinite(energy):
            raise ValueError("energy_eV must be finite.")
        _digest(self.state_sha256, name="state_sha256")
        if not isinstance(self.topology_id, str) or not self.topology_id.strip():
            raise ValueError("topology_id must be a non-empty string.")
        object.__setattr__(self, "energy_eV", energy)
        object.__setattr__(self, "topology_id", self.topology_id.strip())


@runtime_checkable
class ScalarEnergySampler(Protocol):
    """Content-addressed geometry-to-scalar provider."""

    provider_id: str

    def configuration_sha256(self) -> str: ...

    def sample(self, geometry: object) -> ScalarEnergySample: ...


@dataclass(frozen=True, slots=True)
class RichardsonScalarForceComponentEvaluation:
    """One Cartesian Richardson derivative used by focused admission probes."""

    atom_index: int
    axis_index: int
    force_eV_per_A: float
    error_estimate_eV_per_A: float
    displaced_state_sha256: tuple[str, str, str, str]

    def __post_init__(self) -> None:
        if type(self.atom_index) is not int or self.atom_index < 0:
            raise ValueError("atom_index must be a non-negative integer.")
        if type(self.axis_index) is not int or self.axis_index not in (0, 1, 2):
            raise ValueError("axis_index must be 0, 1, or 2.")
        force = float(self.force_eV_per_A)
        error = float(self.error_estimate_eV_per_A)
        if not np.isfinite(force) or not np.isfinite(error) or error < 0.0:
            raise ValueError("component force and error estimate must be finite.")
        state_ids = tuple(self.displaced_state_sha256)
        if len(state_ids) != 4:
            raise ValueError("A Richardson component requires four states.")
        for index, digest in enumerate(state_ids):
            _digest(digest, name=f"displaced_state_sha256[{index}]")
        object.__setattr__(self, "force_eV_per_A", force)
        object.__setattr__(self, "error_estimate_eV_per_A", error)
        object.__setattr__(self, "displaced_state_sha256", state_ids)


@dataclass(frozen=True, slots=True)
class RichardsonScalarForceEvaluation:
    """One fourth-order central-difference force and its local error estimate."""

    contract_id: str
    provider_configuration_sha256: str
    central_sample: ScalarEnergySample
    coarse_step_angstrom: float
    fine_step_angstrom: float
    forces_eV_per_A: np.ndarray
    error_estimates_eV_per_A: np.ndarray
    displaced_state_sha256: tuple[str, ...]
    evaluation_sha256: str = ""

    def __post_init__(self) -> None:
        if self.contract_id != RICHARDSON_FORCE_CONTRACT:
            raise ValueError("Unknown Richardson force contract.")
        _digest(
            self.provider_configuration_sha256,
            name="provider_configuration_sha256",
        )
        if not isinstance(self.central_sample, ScalarEnergySample):
            raise TypeError("central_sample must be ScalarEnergySample.")
        coarse = float(self.coarse_step_angstrom)
        fine = float(self.fine_step_angstrom)
        if not np.isfinite(coarse) or not np.isfinite(fine):
            raise ValueError("finite-difference steps must be finite.")
        if coarse <= 0.0 or fine <= 0.0 or coarse != 2.0 * fine:
            raise ValueError("Richardson force requires coarse_step == 2*fine_step.")
        forces = np.asarray(self.forces_eV_per_A, dtype=float)
        errors = np.asarray(self.error_estimates_eV_per_A, dtype=float)
        if (
            forces.ndim != 2
            or forces.shape[1] != 3
            or errors.shape != forces.shape
            or not np.all(np.isfinite(forces))
            or not np.all(np.isfinite(errors))
            or np.any(errors < 0.0)
        ):
            raise ValueError("force and error arrays must be finite matching (N,3).")
        state_ids = tuple(self.displaced_state_sha256)
        if len(state_ids) != 4 * forces.size:
            raise ValueError("Every Cartesian force component requires four states.")
        for index, digest in enumerate(state_ids):
            _digest(digest, name=f"displaced_state_sha256[{index}]")
        forces_contiguous = np.ascontiguousarray(forces, dtype=np.float64)
        errors_contiguous = np.ascontiguousarray(errors, dtype=np.float64)
        forces_copy = np.frombuffer(
            forces_contiguous.tobytes(), dtype=np.float64
        ).reshape(forces.shape)
        errors_copy = np.frombuffer(
            errors_contiguous.tobytes(), dtype=np.float64
        ).reshape(errors.shape)
        payload = {
            "contract_id": self.contract_id,
            "provider_configuration_sha256": self.provider_configuration_sha256,
            "central_state_sha256": self.central_sample.state_sha256,
            "central_energy_eV": self.central_sample.energy_eV,
            "topology_id": self.central_sample.topology_id,
            "coarse_step_angstrom": coarse,
            "fine_step_angstrom": fine,
            "forces_eV_per_A": forces_copy.tolist(),
            "error_estimates_eV_per_A": errors_copy.tolist(),
            "displaced_state_sha256": list(state_ids),
        }
        expected = _canonical_sha256(payload)
        if self.evaluation_sha256 and self.evaluation_sha256 != expected:
            raise ValueError("evaluation_sha256 does not match force content.")
        object.__setattr__(self, "coarse_step_angstrom", coarse)
        object.__setattr__(self, "fine_step_angstrom", fine)
        object.__setattr__(self, "forces_eV_per_A", forces_copy)
        object.__setattr__(self, "error_estimates_eV_per_A", errors_copy)
        object.__setattr__(self, "displaced_state_sha256", state_ids)
        object.__setattr__(self, "evaluation_sha256", expected)

    @property
    def maximum_error_estimate_eV_per_A(self) -> float:
        return float(np.max(self.error_estimates_eV_per_A))


@dataclass(frozen=True, slots=True)
class RichardsonScalarForce:
    """Fourth-order central scalar differentiation with fail-closed guards."""

    coarse_step_angstrom: float = 5.0e-4
    maximum_error_eV_per_A: float = 2.0e-4

    def __post_init__(self) -> None:
        coarse = float(self.coarse_step_angstrom)
        error = float(self.maximum_error_eV_per_A)
        if not np.isfinite(coarse) or coarse <= 0.0:
            raise ValueError("coarse_step_angstrom must be positive and finite.")
        if not np.isfinite(error) or error <= 0.0:
            raise ValueError("maximum_error_eV_per_A must be positive and finite.")
        object.__setattr__(self, "coarse_step_angstrom", coarse)
        object.__setattr__(self, "maximum_error_eV_per_A", error)

    @property
    def fine_step_angstrom(self) -> float:
        return 0.5 * self.coarse_step_angstrom

    def evaluate_component(
        self,
        provider: ScalarEnergySampler,
        geometry: object,
        *,
        atom_index: int,
        axis_index: int,
        central_sample: ScalarEnergySample | None = None,
    ) -> RichardsonScalarForceComponentEvaluation:
        if not isinstance(getattr(provider, "provider_id", None), str):
            raise TypeError("scalar provider requires a stable provider_id.")
        configuration = provider.configuration_sha256()
        _digest(configuration, name="provider.configuration_sha256()")
        center = provider.sample(geometry) if central_sample is None else central_sample
        if not isinstance(center, ScalarEnergySample):
            raise TypeError("provider.sample() must return ScalarEnergySample.")
        positions = _positions(geometry)
        if type(atom_index) is not int or not 0 <= atom_index < len(positions):
            raise ValueError("atom_index is outside the geometry.")
        if type(axis_index) is not int or axis_index not in (0, 1, 2):
            raise ValueError("axis_index must be 0, 1, or 2.")
        coarse = self.coarse_step_angstrom
        fine = self.fine_step_angstrom
        samples = []
        for delta in (coarse, -coarse, fine, -fine):
            sample = provider.sample(
                _displaced(geometry, atom_index, axis_index, delta)
            )
            if sample.topology_id != center.topology_id:
                raise RuntimeError(
                    "finite-difference displacement changed the continuum "
                    "topology; conservative force fails closed."
                )
            samples.append(sample)
        plus_coarse, minus_coarse, plus_fine, minus_fine = samples
        derivative_coarse = (plus_coarse.energy_eV - minus_coarse.energy_eV) / (
            2.0 * coarse
        )
        derivative_fine = (plus_fine.energy_eV - minus_fine.energy_eV) / (2.0 * fine)
        derivative = (4.0 * derivative_fine - derivative_coarse) / 3.0
        correction = abs(derivative - derivative_fine)
        if correction > self.maximum_error_eV_per_A:
            raise RuntimeError(
                "Richardson force error estimate exceeds the admitted bound: "
                f"{correction:.6e} > {self.maximum_error_eV_per_A:.6e} eV/A."
            )
        if provider.configuration_sha256() != configuration:
            raise RuntimeError(
                "scalar provider configuration drifted during force evaluation."
            )
        return RichardsonScalarForceComponentEvaluation(
            atom_index=atom_index,
            axis_index=axis_index,
            force_eV_per_A=-derivative,
            error_estimate_eV_per_A=correction,
            displaced_state_sha256=tuple(sample.state_sha256 for sample in samples),
        )

    def evaluate(
        self,
        provider: ScalarEnergySampler,
        geometry: object,
        *,
        central_sample: ScalarEnergySample | None = None,
    ) -> RichardsonScalarForceEvaluation:
        if not isinstance(getattr(provider, "provider_id", None), str):
            raise TypeError("scalar provider requires a stable provider_id.")
        configuration = provider.configuration_sha256()
        _digest(configuration, name="provider.configuration_sha256()")
        center = provider.sample(geometry) if central_sample is None else central_sample
        if not isinstance(center, ScalarEnergySample):
            raise TypeError("provider.sample() must return ScalarEnergySample.")
        positions = _positions(geometry)
        forces = np.zeros_like(positions)
        errors = np.zeros_like(positions)
        state_ids: list[str] = []
        coarse = self.coarse_step_angstrom
        fine = self.fine_step_angstrom

        for atom in range(len(positions)):
            for axis in range(3):
                component = self.evaluate_component(
                    provider,
                    geometry,
                    atom_index=atom,
                    axis_index=axis,
                    central_sample=center,
                )
                forces[atom, axis] = component.force_eV_per_A
                errors[atom, axis] = component.error_estimate_eV_per_A
                state_ids.extend(component.displaced_state_sha256)

        maximum_error = float(np.max(errors))
        if maximum_error > self.maximum_error_eV_per_A:
            raise RuntimeError(
                "Richardson force error estimate exceeds the admitted bound: "
                f"{maximum_error:.6e} > {self.maximum_error_eV_per_A:.6e} eV/A."
            )
        if provider.configuration_sha256() != configuration:
            raise RuntimeError(
                "scalar provider configuration drifted during force evaluation."
            )
        return RichardsonScalarForceEvaluation(
            contract_id=RICHARDSON_FORCE_CONTRACT,
            provider_configuration_sha256=configuration,
            central_sample=center,
            coarse_step_angstrom=coarse,
            fine_step_angstrom=fine,
            forces_eV_per_A=forces,
            error_estimates_eV_per_A=errors,
            displaced_state_sha256=tuple(state_ids),
        )


__all__ = [
    "RICHARDSON_FORCE_CONTRACT",
    "RichardsonScalarForce",
    "RichardsonScalarForceComponentEvaluation",
    "RichardsonScalarForceEvaluation",
    "ScalarEnergySample",
    "ScalarEnergySampler",
]
