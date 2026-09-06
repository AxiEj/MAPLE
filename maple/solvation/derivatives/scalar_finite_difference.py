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

RICHARDSON_FORCE_CONTRACT = "scalar-central-richardson-force-v2"
RICHARDSON_FORCE_COMPONENT_CONTRACT = "scalar-central-richardson-force-component-v2"
RICHARDSON_HVP_CONTRACT = "scalar-force-central-richardson-hvp-v2"
RICHARDSON_HESSIAN_CONTRACT = "scalar-force-central-richardson-hessian-v2"
REQUIRE_COMPLETE_TOPOLOGY_OBSERVATION_V1 = "require-complete-topology-observation-v1"
OBSERVED_COMPONENTS_ONLY_EXPERIMENTAL_V1 = "observed-components-only-experimental-v1"
BOUNDED_TOPOLOGY_NOISE_EXPERIMENTAL_V1 = "bounded-topology-noise-experimental-v1"
TOPOLOGY_OBSERVATION_COVERAGES = ("complete", "partial", "unobservable")
_TOPOLOGY_GUARD_POLICIES = (
    REQUIRE_COMPLETE_TOPOLOGY_OBSERVATION_V1,
    OBSERVED_COMPONENTS_ONLY_EXPERIMENTAL_V1,
)
_LEGACY_UNOBSERVABLE_COMPONENT = "legacy-unspecified-topology-observation"


class FiniteDifferenceTopologyChangeError(RuntimeError):
    """A finite-difference stencil crossed a discrete continuum topology."""


class FiniteDifferenceTopologyObservationError(RuntimeError):
    """The requested derivative requires unavailable topology observations."""


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


def normalize_topology_observation(
    coverage: object,
    components: object,
) -> tuple[str, tuple[str, ...]]:
    if coverage not in TOPOLOGY_OBSERVATION_COVERAGES:
        raise ValueError(
            "topology_observation_coverage must be one of "
            f"{TOPOLOGY_OBSERVATION_COVERAGES}."
        )
    if isinstance(components, str):
        raise TypeError("unobservable_topology_components must be a tuple of strings.")
    try:
        raw_components = tuple(components)
    except TypeError as exc:
        raise TypeError(
            "unobservable_topology_components must be an iterable of strings."
        ) from exc
    if any(not isinstance(value, str) for value in raw_components):
        raise TypeError("unobservable topology component names must be strings.")
    normalized = tuple(value.strip() for value in raw_components)
    if any(not value for value in normalized):
        raise ValueError("unobservable topology component names must be non-empty.")
    if len(set(normalized)) != len(normalized):
        raise ValueError("unobservable topology component names must be unique.")
    normalized = tuple(sorted(normalized))
    if coverage == "complete" and normalized:
        raise ValueError(
            "complete topology observation cannot list unobservable components."
        )
    if coverage != "complete" and not normalized:
        raise ValueError(
            "partial or unobservable topology observation requires at least one "
            "unobservable component."
        )
    return str(coverage), normalized


def _topology_guard_policy(
    value: object,
    *,
    allow_bounded_noise: bool = False,
) -> str:
    policies = _TOPOLOGY_GUARD_POLICIES
    if allow_bounded_noise:
        policies += (BOUNDED_TOPOLOGY_NOISE_EXPERIMENTAL_V1,)
    if value not in policies:
        raise ValueError("topology_guard_policy must be one of " f"{policies}.")
    return str(value)


def _topology_guard_status(coverage: str, policy: str) -> str:
    _topology_guard_policy(policy, allow_bounded_noise=True)
    if policy == BOUNDED_TOPOLOGY_NOISE_EXPERIMENTAL_V1:
        return "bounded-topology-noise-experimental"
    if coverage == "complete":
        return "complete"
    if policy == REQUIRE_COMPLETE_TOPOLOGY_OBSERVATION_V1:
        raise FiniteDifferenceTopologyObservationError(
            "finite-difference derivative requires complete topology observation; "
            f"the scalar reports {coverage} observation."
        )
    if coverage == "partial":
        return "partial-experimental"
    return "unobservable-experimental"


def _validate_topology_stencil_sample(
    center: "ScalarEnergySample",
    sample: "ScalarEnergySample",
    *,
    derivative_name: str,
    reject_topology_change: bool = True,
) -> None:
    if (
        sample.topology_observation_coverage != center.topology_observation_coverage
        or sample.unobservable_topology_components
        != center.unobservable_topology_components
    ):
        raise RuntimeError(
            "finite-difference topology observation metadata drifted during "
            f"{derivative_name} evaluation."
        )
    if reject_topology_change and sample.topology_id != center.topology_id:
        raise FiniteDifferenceTopologyChangeError(
            "finite-difference displacement changed the continuum topology "
            f"observable projection; {derivative_name} fails closed for observed "
            "components."
        )


def _same_energy_sample(
    left: "ScalarEnergySample",
    right: "ScalarEnergySample",
) -> bool:
    return (
        left.state_sha256 == right.state_sha256
        and left.energy_eV == right.energy_eV
        and left.topology_id == right.topology_id
        and left.topology_observation_coverage == right.topology_observation_coverage
        and left.unobservable_topology_components
        == right.unobservable_topology_components
    )


def _validated_energy_center(
    provider: "ScalarEnergySampler",
    geometry: object,
    central_sample: "ScalarEnergySample | None",
) -> "ScalarEnergySample":
    if central_sample is None:
        center = provider.sample(geometry)
        if not isinstance(center, ScalarEnergySample):
            raise TypeError("provider.sample() must return ScalarEnergySample.")
        return center
    if not isinstance(central_sample, ScalarEnergySample):
        raise TypeError("central_sample must be ScalarEnergySample.")
    replay = provider.sample(geometry)
    if not isinstance(replay, ScalarEnergySample):
        raise TypeError("provider.sample() must return ScalarEnergySample.")
    if not _same_energy_sample(central_sample, replay):
        raise ValueError(
            "central_sample did not replay for the current provider and geometry."
        )
    return replay


def _validated_force_center(
    provider: "ScalarForceSampler",
    geometry: object,
    central_sample: "ScalarForceSample | None",
) -> "ScalarForceSample":
    if central_sample is None:
        center = provider.force_sample(geometry)
        if not isinstance(center, ScalarForceSample):
            raise TypeError("provider.force_sample() must return ScalarForceSample.")
        return center
    if not isinstance(central_sample, ScalarForceSample):
        raise TypeError("central_sample must be ScalarForceSample.")
    replay = provider.force_sample(geometry)
    if not isinstance(replay, ScalarForceSample):
        raise TypeError("provider.force_sample() must return ScalarForceSample.")
    if (
        not _same_energy_sample(central_sample.energy_sample, replay.energy_sample)
        or central_sample.evaluation_sha256 != replay.evaluation_sha256
        or not np.array_equal(
            central_sample.forces_eV_per_A,
            replay.forces_eV_per_A,
        )
    ):
        raise ValueError(
            "central_sample did not replay for the current provider and geometry."
        )
    return replay


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


def _directionally_displaced(
    geometry: object, direction: np.ndarray, delta: float
) -> object:
    copier = getattr(geometry, "copy", None)
    if not callable(copier):
        raise TypeError("finite-difference geometry must provide copy().")
    result = copier()
    positions = _positions(result)
    values = np.asarray(direction, dtype=float)
    if values.shape != positions.shape or not np.all(np.isfinite(values)):
        raise ValueError("direction must be finite with the geometry position shape.")
    displaced = positions + float(delta) * values
    setter = getattr(result, "set_positions", None)
    if callable(setter):
        setter(displaced)
    else:
        setattr(result, "positions", displaced)
    return result


@dataclass(frozen=True, slots=True)
class ScalarEnergySample:
    """One immutable scalar evaluation used by a force stencil."""

    energy_eV: float
    state_sha256: str
    topology_id: str
    topology_observation_coverage: str = "unobservable"
    unobservable_topology_components: tuple[str, ...] = (
        _LEGACY_UNOBSERVABLE_COMPONENT,
    )

    def __post_init__(self) -> None:
        energy = float(self.energy_eV)
        if not np.isfinite(energy):
            raise ValueError("energy_eV must be finite.")
        _digest(self.state_sha256, name="state_sha256")
        if not isinstance(self.topology_id, str) or not self.topology_id.strip():
            raise ValueError("topology_id must be a non-empty string.")
        coverage, components = normalize_topology_observation(
            self.topology_observation_coverage,
            self.unobservable_topology_components,
        )
        object.__setattr__(self, "energy_eV", energy)
        object.__setattr__(self, "topology_id", self.topology_id.strip())
        object.__setattr__(self, "topology_observation_coverage", coverage)
        object.__setattr__(self, "unobservable_topology_components", components)


@runtime_checkable
class ScalarEnergySampler(Protocol):
    """Content-addressed geometry-to-scalar provider."""

    provider_id: str

    def configuration_sha256(self) -> str: ...

    def sample(self, geometry: object) -> ScalarEnergySample: ...


@dataclass(frozen=True, slots=True)
class ScalarForceSample:
    """A conservative force sampled from the same content-addressed scalar."""

    energy_sample: ScalarEnergySample
    forces_eV_per_A: np.ndarray
    evaluation_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.energy_sample, ScalarEnergySample):
            raise TypeError("energy_sample must be ScalarEnergySample.")
        forces = np.asarray(self.forces_eV_per_A, dtype=float)
        if forces.ndim != 2 or forces.shape[1] != 3 or not np.all(np.isfinite(forces)):
            raise ValueError("forces_eV_per_A must be finite with shape (N,3).")
        _digest(self.evaluation_sha256, name="evaluation_sha256")
        copied = np.frombuffer(
            np.ascontiguousarray(forces, dtype=np.float64).tobytes(),
            dtype=np.float64,
        ).reshape(forces.shape)
        object.__setattr__(self, "forces_eV_per_A", copied)


@runtime_checkable
class ScalarForceSampler(ScalarEnergySampler, Protocol):
    """Provider of a scalar and its conservative first derivative."""

    def force_sample(self, geometry: object) -> ScalarForceSample: ...


@dataclass(frozen=True, slots=True)
class RichardsonScalarForceComponentEvaluation:
    """One Cartesian Richardson derivative used by focused admission probes."""

    contract_id: str
    provider_configuration_sha256: str
    derivative_policy_sha256: str
    central_sample: ScalarEnergySample
    topology_observation_coverage: str
    unobservable_topology_components: tuple[str, ...]
    topology_guard_status: str
    atom_index: int
    axis_index: int
    coarse_step_angstrom: float
    fine_step_angstrom: float
    force_eV_per_A: float
    error_estimate_eV_per_A: float
    displaced_state_sha256: tuple[str, str, str, str]
    evaluation_sha256: str = ""

    def __post_init__(self) -> None:
        if self.contract_id != RICHARDSON_FORCE_COMPONENT_CONTRACT:
            raise ValueError("Unknown Richardson force-component contract.")
        _digest(
            self.provider_configuration_sha256,
            name="provider_configuration_sha256",
        )
        _digest(self.derivative_policy_sha256, name="derivative_policy_sha256")
        if not isinstance(self.central_sample, ScalarEnergySample):
            raise TypeError("central_sample must be ScalarEnergySample.")
        coverage, components = normalize_topology_observation(
            self.topology_observation_coverage,
            self.unobservable_topology_components,
        )
        if (
            coverage != self.central_sample.topology_observation_coverage
            or components != self.central_sample.unobservable_topology_components
        ):
            raise ValueError(
                "force-component topology observation metadata is inconsistent."
            )
        expected_status = {
            "complete": "complete",
            "partial": "partial-experimental",
            "unobservable": "unobservable-experimental",
        }[coverage]
        if self.topology_guard_status != expected_status:
            raise ValueError(
                "topology_guard_status is inconsistent with force-component "
                "coverage."
            )
        if type(self.atom_index) is not int or self.atom_index < 0:
            raise ValueError("atom_index must be a non-negative integer.")
        if type(self.axis_index) is not int or self.axis_index not in (0, 1, 2):
            raise ValueError("axis_index must be 0, 1, or 2.")
        coarse = float(self.coarse_step_angstrom)
        fine = float(self.fine_step_angstrom)
        if (
            not np.isfinite(coarse)
            or not np.isfinite(fine)
            or coarse <= 0.0
            or fine <= 0.0
            or coarse != 2.0 * fine
        ):
            raise ValueError(
                "Richardson force component requires coarse_step == 2*fine_step."
            )
        force = float(self.force_eV_per_A)
        error = float(self.error_estimate_eV_per_A)
        if not np.isfinite(force) or not np.isfinite(error) or error < 0.0:
            raise ValueError("component force and error estimate must be finite.")
        state_ids = tuple(self.displaced_state_sha256)
        if len(state_ids) != 4:
            raise ValueError("A Richardson component requires four states.")
        for index, digest in enumerate(state_ids):
            _digest(digest, name=f"displaced_state_sha256[{index}]")
        payload = {
            "contract_id": self.contract_id,
            "provider_configuration_sha256": self.provider_configuration_sha256,
            "derivative_policy_sha256": self.derivative_policy_sha256,
            "central_state_sha256": self.central_sample.state_sha256,
            "central_energy_eV": self.central_sample.energy_eV,
            "topology_id": self.central_sample.topology_id,
            "topology_observation_coverage": coverage,
            "unobservable_topology_components": list(components),
            "topology_guard_status": self.topology_guard_status,
            "atom_index": self.atom_index,
            "axis_index": self.axis_index,
            "coarse_step_angstrom": coarse,
            "fine_step_angstrom": fine,
            "force_eV_per_A": force,
            "error_estimate_eV_per_A": error,
            "displaced_state_sha256": list(state_ids),
        }
        expected = _canonical_sha256(payload)
        if self.evaluation_sha256 and self.evaluation_sha256 != expected:
            raise ValueError(
                "evaluation_sha256 does not match force-component content."
            )
        object.__setattr__(self, "topology_observation_coverage", coverage)
        object.__setattr__(self, "unobservable_topology_components", components)
        object.__setattr__(self, "coarse_step_angstrom", coarse)
        object.__setattr__(self, "fine_step_angstrom", fine)
        object.__setattr__(self, "force_eV_per_A", force)
        object.__setattr__(self, "error_estimate_eV_per_A", error)
        object.__setattr__(self, "displaced_state_sha256", state_ids)
        object.__setattr__(self, "evaluation_sha256", expected)


@dataclass(frozen=True, slots=True)
class RichardsonScalarForceEvaluation:
    """One fourth-order central-difference force and its local error estimate."""

    contract_id: str
    provider_configuration_sha256: str
    derivative_policy_sha256: str
    central_sample: ScalarEnergySample
    topology_observation_coverage: str
    unobservable_topology_components: tuple[str, ...]
    topology_guard_status: str
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
        _digest(self.derivative_policy_sha256, name="derivative_policy_sha256")
        if not isinstance(self.central_sample, ScalarEnergySample):
            raise TypeError("central_sample must be ScalarEnergySample.")
        coverage, components = normalize_topology_observation(
            self.topology_observation_coverage,
            self.unobservable_topology_components,
        )
        if (
            coverage != self.central_sample.topology_observation_coverage
            or components != self.central_sample.unobservable_topology_components
        ):
            raise ValueError("force topology observation metadata is inconsistent.")
        expected_status = {
            "complete": "complete",
            "partial": "partial-experimental",
            "unobservable": "unobservable-experimental",
        }[coverage]
        if self.topology_guard_status != expected_status:
            raise ValueError(
                "topology_guard_status is inconsistent with force coverage."
            )
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
            "derivative_policy_sha256": self.derivative_policy_sha256,
            "central_state_sha256": self.central_sample.state_sha256,
            "central_energy_eV": self.central_sample.energy_eV,
            "topology_id": self.central_sample.topology_id,
            "topology_observation_coverage": coverage,
            "unobservable_topology_components": list(components),
            "topology_guard_status": self.topology_guard_status,
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
        object.__setattr__(self, "topology_observation_coverage", coverage)
        object.__setattr__(self, "unobservable_topology_components", components)
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
    topology_guard_policy: str = REQUIRE_COMPLETE_TOPOLOGY_OBSERVATION_V1

    def __post_init__(self) -> None:
        coarse = float(self.coarse_step_angstrom)
        error = float(self.maximum_error_eV_per_A)
        if not np.isfinite(coarse) or coarse <= 0.0:
            raise ValueError("coarse_step_angstrom must be positive and finite.")
        if not np.isfinite(error) or error <= 0.0:
            raise ValueError("maximum_error_eV_per_A must be positive and finite.")
        policy = _topology_guard_policy(self.topology_guard_policy)
        object.__setattr__(self, "coarse_step_angstrom", coarse)
        object.__setattr__(self, "maximum_error_eV_per_A", error)
        object.__setattr__(self, "topology_guard_policy", policy)

    @property
    def fine_step_angstrom(self) -> float:
        return 0.5 * self.coarse_step_angstrom

    def policy_payload(self) -> dict[str, object]:
        return {
            "contract": "scalar-richardson-force-derivative-policy-v1",
            "coarse_step_angstrom": self.coarse_step_angstrom,
            "maximum_error_eV_per_A": self.maximum_error_eV_per_A,
            "topology_guard_policy": self.topology_guard_policy,
        }

    def policy_sha256(self) -> str:
        return _canonical_sha256(self.policy_payload())

    def _evaluate_component_from_validated_center(
        self,
        provider: ScalarEnergySampler,
        geometry: object,
        *,
        atom_index: int,
        axis_index: int,
        center: ScalarEnergySample,
        configuration: str,
        topology_guard_status: str,
    ) -> RichardsonScalarForceComponentEvaluation:
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
            _validate_topology_stencil_sample(
                center,
                sample,
                derivative_name="conservative force",
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
            contract_id=RICHARDSON_FORCE_COMPONENT_CONTRACT,
            provider_configuration_sha256=configuration,
            derivative_policy_sha256=self.policy_sha256(),
            central_sample=center,
            topology_observation_coverage=center.topology_observation_coverage,
            unobservable_topology_components=(center.unobservable_topology_components),
            topology_guard_status=topology_guard_status,
            atom_index=atom_index,
            axis_index=axis_index,
            coarse_step_angstrom=coarse,
            fine_step_angstrom=fine,
            force_eV_per_A=-derivative,
            error_estimate_eV_per_A=correction,
            displaced_state_sha256=tuple(sample.state_sha256 for sample in samples),
        )

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
        center = _validated_energy_center(provider, geometry, central_sample)
        guard_status = _topology_guard_status(
            center.topology_observation_coverage,
            self.topology_guard_policy,
        )
        return self._evaluate_component_from_validated_center(
            provider,
            geometry,
            atom_index=atom_index,
            axis_index=axis_index,
            center=center,
            configuration=configuration,
            topology_guard_status=guard_status,
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
        center = _validated_energy_center(provider, geometry, central_sample)
        guard_status = _topology_guard_status(
            center.topology_observation_coverage,
            self.topology_guard_policy,
        )
        positions = _positions(geometry)
        forces = np.zeros_like(positions)
        errors = np.zeros_like(positions)
        state_ids: list[str] = []
        coarse = self.coarse_step_angstrom
        fine = self.fine_step_angstrom

        for atom in range(len(positions)):
            for axis in range(3):
                component = self._evaluate_component_from_validated_center(
                    provider,
                    geometry,
                    atom_index=atom,
                    axis_index=axis,
                    center=center,
                    configuration=configuration,
                    topology_guard_status=guard_status,
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
            derivative_policy_sha256=self.policy_sha256(),
            central_sample=center,
            topology_observation_coverage=(center.topology_observation_coverage),
            unobservable_topology_components=(center.unobservable_topology_components),
            topology_guard_status=guard_status,
            coarse_step_angstrom=coarse,
            fine_step_angstrom=fine,
            forces_eV_per_A=forces,
            error_estimates_eV_per_A=errors,
            displaced_state_sha256=tuple(state_ids),
        )


@dataclass(frozen=True, slots=True)
class RichardsonScalarHVPEvaluation:
    """One fourth-order Hessian-vector product from conservative forces.

    Bounded-noise diagnostics use the same ``(+h, -h, +h/2, -h/2)`` order as
    ``displaced_force_sha256`` and are included in ``evaluation_sha256``.
    Legacy policies leave those tuples empty and retain their historical hash.
    """

    contract_id: str
    provider_configuration_sha256: str
    derivative_policy_sha256: str
    central_sample: ScalarForceSample
    topology_observation_coverage: str
    unobservable_topology_components: tuple[str, ...]
    topology_guard_status: str
    topology_step_reductions_used: int
    coarse_step_angstrom: float
    fine_step_angstrom: float
    direction: np.ndarray
    hvp_eV_per_A2: np.ndarray
    error_estimates_eV_per_A2: np.ndarray
    displaced_force_sha256: tuple[str, str, str, str]
    displaced_topology_ids: tuple[str, ...] = ()
    displaced_topology_changed: tuple[bool, ...] = ()
    energy_force_discrepancies_eV_per_A: tuple[float, ...] = ()
    evaluation_sha256: str = ""

    def __post_init__(self) -> None:
        if self.contract_id != RICHARDSON_HVP_CONTRACT:
            raise ValueError("Unknown Richardson HVP contract.")
        _digest(
            self.provider_configuration_sha256,
            name="provider_configuration_sha256",
        )
        _digest(self.derivative_policy_sha256, name="derivative_policy_sha256")
        if not isinstance(self.central_sample, ScalarForceSample):
            raise TypeError("central_sample must be ScalarForceSample.")
        coverage, components = normalize_topology_observation(
            self.topology_observation_coverage,
            self.unobservable_topology_components,
        )
        if (
            coverage != self.central_sample.energy_sample.topology_observation_coverage
            or components
            != self.central_sample.energy_sample.unobservable_topology_components
        ):
            raise ValueError("HVP topology observation metadata is inconsistent.")
        expected_status = {
            "complete": "complete",
            "partial": "partial-experimental",
            "unobservable": "unobservable-experimental",
        }[coverage]
        bounded_status = "bounded-topology-noise-experimental"
        if self.topology_guard_status not in (expected_status, bounded_status):
            raise ValueError("topology_guard_status is inconsistent with HVP coverage.")
        reductions = self.topology_step_reductions_used
        if type(reductions) is not int or reductions < 0:
            raise ValueError("topology_step_reductions_used must be non-negative.")
        coarse = float(self.coarse_step_angstrom)
        fine = float(self.fine_step_angstrom)
        if coarse <= 0.0 or fine <= 0.0 or coarse != 2.0 * fine:
            raise ValueError("Richardson HVP requires coarse_step == 2*fine_step.")
        direction = np.asarray(self.direction, dtype=float)
        hvp = np.asarray(self.hvp_eV_per_A2, dtype=float)
        errors = np.asarray(self.error_estimates_eV_per_A2, dtype=float)
        expected_shape = self.central_sample.forces_eV_per_A.shape
        if (
            direction.shape != expected_shape
            or hvp.shape != expected_shape
            or errors.shape != expected_shape
            or not np.all(np.isfinite(direction))
            or not np.all(np.isfinite(hvp))
            or not np.all(np.isfinite(errors))
            or np.any(errors < 0.0)
        ):
            raise ValueError("Richardson HVP arrays must be finite matching (N,3).")
        state_ids = tuple(self.displaced_force_sha256)
        if len(state_ids) != 4:
            raise ValueError("A Richardson HVP requires four displaced forces.")
        for index, digest in enumerate(state_ids):
            _digest(digest, name=f"displaced_force_sha256[{index}]")
        topology_ids = tuple(self.displaced_topology_ids)
        topology_changed = tuple(self.displaced_topology_changed)
        discrepancies = tuple(
            float(value) for value in self.energy_force_discrepancies_eV_per_A
        )
        bounded_diagnostics = bool(topology_ids or topology_changed or discrepancies)
        if bounded_diagnostics:
            if self.topology_guard_status != bounded_status:
                raise ValueError(
                    "bounded topology diagnostics require the bounded-noise status."
                )
            if not (
                len(topology_ids)
                == len(topology_changed)
                == len(discrepancies)
                == len(state_ids)
            ):
                raise ValueError(
                    "bounded topology diagnostics must cover every displaced force."
                )
            if any(not isinstance(value, str) or not value for value in topology_ids):
                raise ValueError("displaced topology IDs must be non-empty strings.")
            if any(type(value) is not bool for value in topology_changed):
                raise ValueError("displaced topology change flags must be booleans.")
            central_topology = self.central_sample.energy_sample.topology_id
            if topology_changed != tuple(
                value != central_topology for value in topology_ids
            ):
                raise ValueError("displaced topology change flags are inconsistent.")
            if any(not np.isfinite(value) or value < 0.0 for value in discrepancies):
                raise ValueError(
                    "energy-force discrepancies must be finite and non-negative."
                )
        elif self.topology_guard_status == bounded_status:
            raise ValueError("bounded-noise HVP requires displaced diagnostics.")
        direction_copy = np.frombuffer(
            np.ascontiguousarray(direction, dtype=np.float64).tobytes(),
            dtype=np.float64,
        ).reshape(direction.shape)
        hvp_copy = np.frombuffer(
            np.ascontiguousarray(hvp, dtype=np.float64).tobytes(), dtype=np.float64
        ).reshape(hvp.shape)
        errors_copy = np.frombuffer(
            np.ascontiguousarray(errors, dtype=np.float64).tobytes(), dtype=np.float64
        ).reshape(errors.shape)
        payload = {
            "contract_id": self.contract_id,
            "provider_configuration_sha256": self.provider_configuration_sha256,
            "derivative_policy_sha256": self.derivative_policy_sha256,
            "central_force_sha256": self.central_sample.evaluation_sha256,
            "topology_observation_coverage": coverage,
            "unobservable_topology_components": list(components),
            "topology_guard_status": self.topology_guard_status,
            "topology_step_reductions_used": reductions,
            "coarse_step_angstrom": coarse,
            "fine_step_angstrom": fine,
            "direction": direction_copy.tolist(),
            "hvp_eV_per_A2": hvp_copy.tolist(),
            "error_estimates_eV_per_A2": errors_copy.tolist(),
            "displaced_force_sha256": list(state_ids),
        }
        if bounded_diagnostics:
            payload.update(
                {
                    "displaced_topology_ids": list(topology_ids),
                    "displaced_topology_changed": list(topology_changed),
                    "energy_force_discrepancies_eV_per_A": list(discrepancies),
                }
            )
        expected = _canonical_sha256(payload)
        if self.evaluation_sha256 and self.evaluation_sha256 != expected:
            raise ValueError("evaluation_sha256 does not match HVP content.")
        object.__setattr__(self, "coarse_step_angstrom", coarse)
        object.__setattr__(self, "fine_step_angstrom", fine)
        object.__setattr__(self, "topology_observation_coverage", coverage)
        object.__setattr__(self, "unobservable_topology_components", components)
        object.__setattr__(self, "topology_step_reductions_used", reductions)
        object.__setattr__(self, "direction", direction_copy)
        object.__setattr__(self, "hvp_eV_per_A2", hvp_copy)
        object.__setattr__(self, "error_estimates_eV_per_A2", errors_copy)
        object.__setattr__(self, "displaced_force_sha256", state_ids)
        object.__setattr__(self, "displaced_topology_ids", topology_ids)
        object.__setattr__(self, "displaced_topology_changed", topology_changed)
        object.__setattr__(self, "energy_force_discrepancies_eV_per_A", discrepancies)
        object.__setattr__(self, "evaluation_sha256", expected)

    @property
    def maximum_error_estimate_eV_per_A2(self) -> float:
        return float(np.max(self.error_estimates_eV_per_A2))


@dataclass(frozen=True, slots=True)
class RichardsonScalarHessianEvaluation:
    """Full symmetric Cartesian Hessian with numerical diagnostics.

    Bounded-noise displaced diagnostics are concatenated column by column;
    each column uses ``(+h, -h, +h/2, -h/2)`` stencil order.
    """

    contract_id: str
    provider_configuration_sha256: str
    derivative_policy_sha256: str
    central_sample: ScalarForceSample
    topology_observation_coverage: str
    unobservable_topology_components: tuple[str, ...]
    topology_guard_status: str
    topology_step_reductions_used: int
    coarse_step_angstrom: float
    fine_step_angstrom: float
    raw_hessian_eV_per_A2: np.ndarray
    hessian_eV_per_A2: np.ndarray
    error_estimates_eV_per_A2: np.ndarray
    displaced_force_sha256: tuple[str, ...]
    maximum_antisymmetry_eV_per_A2: float
    displaced_topology_ids: tuple[str, ...] = ()
    displaced_topology_changed: tuple[bool, ...] = ()
    energy_force_discrepancies_eV_per_A: tuple[float, ...] = ()
    evaluation_sha256: str = ""

    def __post_init__(self) -> None:
        if self.contract_id != RICHARDSON_HESSIAN_CONTRACT:
            raise ValueError("Unknown Richardson Hessian contract.")
        _digest(
            self.provider_configuration_sha256,
            name="provider_configuration_sha256",
        )
        _digest(self.derivative_policy_sha256, name="derivative_policy_sha256")
        if not isinstance(self.central_sample, ScalarForceSample):
            raise TypeError("central_sample must be ScalarForceSample.")
        coverage, components = normalize_topology_observation(
            self.topology_observation_coverage,
            self.unobservable_topology_components,
        )
        if (
            coverage != self.central_sample.energy_sample.topology_observation_coverage
            or components
            != self.central_sample.energy_sample.unobservable_topology_components
        ):
            raise ValueError("Hessian topology observation metadata is inconsistent.")
        expected_status = {
            "complete": "complete",
            "partial": "partial-experimental",
            "unobservable": "unobservable-experimental",
        }[coverage]
        bounded_status = "bounded-topology-noise-experimental"
        if self.topology_guard_status not in (expected_status, bounded_status):
            raise ValueError(
                "topology_guard_status is inconsistent with Hessian coverage."
            )
        reductions = self.topology_step_reductions_used
        if type(reductions) is not int or reductions < 0:
            raise ValueError("topology_step_reductions_used must be non-negative.")
        coarse = float(self.coarse_step_angstrom)
        fine = float(self.fine_step_angstrom)
        if coarse <= 0.0 or fine <= 0.0 or coarse != 2.0 * fine:
            raise ValueError("Richardson Hessian requires coarse_step == 2*fine_step.")
        dimension = self.central_sample.forces_eV_per_A.size
        expected_shape = (dimension, dimension)
        raw = np.asarray(self.raw_hessian_eV_per_A2, dtype=float)
        symmetric = np.asarray(self.hessian_eV_per_A2, dtype=float)
        errors = np.asarray(self.error_estimates_eV_per_A2, dtype=float)
        if (
            raw.shape != expected_shape
            or symmetric.shape != expected_shape
            or errors.shape != expected_shape
            or not np.all(np.isfinite(raw))
            or not np.all(np.isfinite(symmetric))
            or not np.all(np.isfinite(errors))
            or np.any(errors < 0.0)
        ):
            raise ValueError("Hessian arrays must be finite square matrices.")
        if not np.array_equal(symmetric, 0.5 * (raw + raw.T)):
            raise ValueError(
                "hessian_eV_per_A2 must be the exact symmetric raw Hessian."
            )
        antisymmetry = float(self.maximum_antisymmetry_eV_per_A2)
        expected_antisymmetry = float(np.max(np.abs(raw - raw.T)))
        if not np.isfinite(antisymmetry) or antisymmetry != expected_antisymmetry:
            raise ValueError("maximum_antisymmetry_eV_per_A2 is inconsistent.")
        state_ids = tuple(self.displaced_force_sha256)
        if len(state_ids) != 4 * dimension:
            raise ValueError("Every Hessian column requires four displaced forces.")
        for index, digest in enumerate(state_ids):
            _digest(digest, name=f"displaced_force_sha256[{index}]")
        topology_ids = tuple(self.displaced_topology_ids)
        topology_changed = tuple(self.displaced_topology_changed)
        discrepancies = tuple(
            float(value) for value in self.energy_force_discrepancies_eV_per_A
        )
        bounded_diagnostics = bool(topology_ids or topology_changed or discrepancies)
        if bounded_diagnostics:
            if self.topology_guard_status != bounded_status:
                raise ValueError(
                    "bounded topology diagnostics require the bounded-noise status."
                )
            if not (
                len(topology_ids)
                == len(topology_changed)
                == len(discrepancies)
                == len(state_ids)
            ):
                raise ValueError(
                    "bounded topology diagnostics must cover every Hessian stencil."
                )
            if any(not isinstance(value, str) or not value for value in topology_ids):
                raise ValueError("displaced topology IDs must be non-empty strings.")
            if any(type(value) is not bool for value in topology_changed):
                raise ValueError("displaced topology change flags must be booleans.")
            central_topology = self.central_sample.energy_sample.topology_id
            if topology_changed != tuple(
                value != central_topology for value in topology_ids
            ):
                raise ValueError("displaced topology change flags are inconsistent.")
            if any(not np.isfinite(value) or value < 0.0 for value in discrepancies):
                raise ValueError(
                    "energy-force discrepancies must be finite and non-negative."
                )
        elif self.topology_guard_status == bounded_status:
            raise ValueError("bounded-noise Hessian requires displaced diagnostics.")
        raw_copy = np.frombuffer(
            np.ascontiguousarray(raw, dtype=np.float64).tobytes(), dtype=np.float64
        ).reshape(raw.shape)
        symmetric_copy = np.frombuffer(
            np.ascontiguousarray(symmetric, dtype=np.float64).tobytes(),
            dtype=np.float64,
        ).reshape(symmetric.shape)
        errors_copy = np.frombuffer(
            np.ascontiguousarray(errors, dtype=np.float64).tobytes(), dtype=np.float64
        ).reshape(errors.shape)
        payload = {
            "contract_id": self.contract_id,
            "provider_configuration_sha256": self.provider_configuration_sha256,
            "derivative_policy_sha256": self.derivative_policy_sha256,
            "central_force_sha256": self.central_sample.evaluation_sha256,
            "topology_observation_coverage": coverage,
            "unobservable_topology_components": list(components),
            "topology_guard_status": self.topology_guard_status,
            "topology_step_reductions_used": reductions,
            "coarse_step_angstrom": coarse,
            "fine_step_angstrom": fine,
            "raw_hessian_eV_per_A2": raw_copy.tolist(),
            "hessian_eV_per_A2": symmetric_copy.tolist(),
            "error_estimates_eV_per_A2": errors_copy.tolist(),
            "maximum_antisymmetry_eV_per_A2": antisymmetry,
            "displaced_force_sha256": list(state_ids),
        }
        if bounded_diagnostics:
            payload.update(
                {
                    "displaced_topology_ids": list(topology_ids),
                    "displaced_topology_changed": list(topology_changed),
                    "energy_force_discrepancies_eV_per_A": list(discrepancies),
                }
            )
        expected = _canonical_sha256(payload)
        if self.evaluation_sha256 and self.evaluation_sha256 != expected:
            raise ValueError("evaluation_sha256 does not match Hessian content.")
        object.__setattr__(self, "coarse_step_angstrom", coarse)
        object.__setattr__(self, "fine_step_angstrom", fine)
        object.__setattr__(self, "topology_observation_coverage", coverage)
        object.__setattr__(self, "unobservable_topology_components", components)
        object.__setattr__(self, "topology_step_reductions_used", reductions)
        object.__setattr__(self, "raw_hessian_eV_per_A2", raw_copy)
        object.__setattr__(self, "hessian_eV_per_A2", symmetric_copy)
        object.__setattr__(self, "error_estimates_eV_per_A2", errors_copy)
        object.__setattr__(self, "displaced_force_sha256", state_ids)
        object.__setattr__(self, "displaced_topology_ids", topology_ids)
        object.__setattr__(self, "displaced_topology_changed", topology_changed)
        object.__setattr__(self, "energy_force_discrepancies_eV_per_A", discrepancies)
        object.__setattr__(self, "maximum_antisymmetry_eV_per_A2", antisymmetry)
        object.__setattr__(self, "evaluation_sha256", expected)

    @property
    def maximum_error_estimate_eV_per_A2(self) -> float:
        return float(np.max(self.error_estimates_eV_per_A2))


@dataclass(frozen=True, slots=True)
class RichardsonScalarHessian:
    """Differentiate conservative forces from one scalar with Richardson error control."""

    coarse_step_angstrom: float = 2.0e-3
    maximum_error_eV_per_A2: float = 5.0e-3
    maximum_antisymmetry_eV_per_A2: float = 5.0e-3
    maximum_topology_step_reductions: int = 0
    topology_guard_policy: str = REQUIRE_COMPLETE_TOPOLOGY_OBSERVATION_V1
    maximum_energy_force_discrepancy_eV_per_A: float | None = None

    def __post_init__(self) -> None:
        coarse = float(self.coarse_step_angstrom)
        error = float(self.maximum_error_eV_per_A2)
        antisymmetry = float(self.maximum_antisymmetry_eV_per_A2)
        reductions = self.maximum_topology_step_reductions
        if not np.isfinite(coarse) or coarse <= 0.0:
            raise ValueError("coarse_step_angstrom must be positive and finite.")
        if not np.isfinite(error) or error <= 0.0:
            raise ValueError("maximum_error_eV_per_A2 must be positive and finite.")
        if not np.isfinite(antisymmetry) or antisymmetry <= 0.0:
            raise ValueError(
                "maximum_antisymmetry_eV_per_A2 must be positive and finite."
            )
        if type(reductions) is not int or not 0 <= reductions <= 20:
            raise ValueError(
                "maximum_topology_step_reductions must be an integer in [0,20]."
            )
        policy = _topology_guard_policy(
            self.topology_guard_policy,
            allow_bounded_noise=True,
        )
        discrepancy = self.maximum_energy_force_discrepancy_eV_per_A
        if policy == BOUNDED_TOPOLOGY_NOISE_EXPERIMENTAL_V1:
            if discrepancy is None:
                discrepancy = 3.0e-3
            discrepancy = float(discrepancy)
            if not np.isfinite(discrepancy) or discrepancy <= 0.0:
                raise ValueError(
                    "maximum_energy_force_discrepancy_eV_per_A must be positive "
                    "and finite."
                )
        elif discrepancy is not None:
            raise ValueError(
                "maximum_energy_force_discrepancy_eV_per_A is only valid with "
                f"{BOUNDED_TOPOLOGY_NOISE_EXPERIMENTAL_V1}."
            )
        object.__setattr__(self, "coarse_step_angstrom", coarse)
        object.__setattr__(self, "maximum_error_eV_per_A2", error)
        object.__setattr__(self, "maximum_antisymmetry_eV_per_A2", antisymmetry)
        object.__setattr__(self, "maximum_topology_step_reductions", reductions)
        object.__setattr__(self, "topology_guard_policy", policy)
        object.__setattr__(
            self,
            "maximum_energy_force_discrepancy_eV_per_A",
            discrepancy,
        )

    @property
    def fine_step_angstrom(self) -> float:
        return 0.5 * self.coarse_step_angstrom

    def policy_payload(self) -> dict[str, object]:
        payload = {
            "contract": "scalar-richardson-hessian-derivative-policy-v1",
            "requested_coarse_step_angstrom": self.coarse_step_angstrom,
            "maximum_error_eV_per_A2": self.maximum_error_eV_per_A2,
            "maximum_antisymmetry_eV_per_A2": (self.maximum_antisymmetry_eV_per_A2),
            "maximum_topology_step_reductions": (self.maximum_topology_step_reductions),
            "topology_guard_policy": self.topology_guard_policy,
        }
        if self.topology_guard_policy == BOUNDED_TOPOLOGY_NOISE_EXPERIMENTAL_V1:
            payload["maximum_energy_force_discrepancy_eV_per_A"] = (
                self.maximum_energy_force_discrepancy_eV_per_A
            )
        return payload

    def policy_sha256(self) -> str:
        return _canonical_sha256(self.policy_payload())

    def _candidate_coarse_steps(self) -> tuple[float, ...]:
        return tuple(
            self.coarse_step_angstrom / (2.0**reduction)
            for reduction in range(self.maximum_topology_step_reductions + 1)
        )

    def _evaluate_hvp_at_step(
        self,
        provider: ScalarForceSampler,
        geometry: object,
        vector: np.ndarray,
        *,
        center: ScalarForceSample,
        configuration: str,
        coarse_step_angstrom: float,
        topology_guard_status: str,
        topology_step_reductions_used: int,
    ) -> RichardsonScalarHVPEvaluation:
        direction_norm = float(np.linalg.norm(vector))
        stencil_direction = vector / direction_norm
        fine_step_angstrom = 0.5 * coarse_step_angstrom
        bounded_noise = (
            self.topology_guard_policy == BOUNDED_TOPOLOGY_NOISE_EXPERIMENTAL_V1
        )
        samples = []
        deltas = (
            coarse_step_angstrom,
            -coarse_step_angstrom,
            fine_step_angstrom,
            -fine_step_angstrom,
        )
        try:
            for delta in deltas:
                sample = provider.force_sample(
                    _directionally_displaced(geometry, stencil_direction, delta)
                )
                _validate_topology_stencil_sample(
                    center.energy_sample,
                    sample.energy_sample,
                    derivative_name="Hessian",
                    reject_topology_change=not bounded_noise,
                )
                samples.append(sample)
        except BaseException:
            # Providers commonly cache their most recent geometry.  A rejected
            # stencil must not leave that cache silently pointing at a displacement.
            if bounded_noise:
                provider.force_sample(geometry)
            raise
        plus_coarse, minus_coarse, plus_fine, minus_fine = samples
        discrepancies: tuple[float, ...] = ()
        topology_ids: tuple[str, ...] = ()
        topology_changed: tuple[bool, ...] = ()
        if bounded_noise:
            center_energy = center.energy_sample.energy_eV
            center_force_along_direction = float(
                np.sum(center.forces_eV_per_A * stencil_direction)
            )
            discrepancies = tuple(
                abs(
                    sample.energy_sample.energy_eV
                    - center_energy
                    + 0.5
                    * delta
                    * (
                        center_force_along_direction
                        + float(np.sum(sample.forces_eV_per_A * stencil_direction))
                    )
                )
                / abs(delta)
                for delta, sample in zip(deltas, samples)
            )
            maximum_discrepancy = max(discrepancies)
            assert self.maximum_energy_force_discrepancy_eV_per_A is not None
            if maximum_discrepancy > self.maximum_energy_force_discrepancy_eV_per_A:
                provider.force_sample(geometry)
                raise RuntimeError(
                    "Hessian stencil energy-force discrepancy exceeds the admitted "
                    f"bound: {maximum_discrepancy:.6e} > "
                    f"{self.maximum_energy_force_discrepancy_eV_per_A:.6e} eV/A."
                )
            topology_ids = tuple(sample.energy_sample.topology_id for sample in samples)
            topology_changed = tuple(
                topology_id != center.energy_sample.topology_id
                for topology_id in topology_ids
            )
        derivative_coarse = (
            plus_coarse.forces_eV_per_A - minus_coarse.forces_eV_per_A
        ) / (2.0 * coarse_step_angstrom)
        derivative_fine = (plus_fine.forces_eV_per_A - minus_fine.forces_eV_per_A) / (
            2.0 * fine_step_angstrom
        )
        # F=-grad(E), hence H v = -dF/ds.
        hvp_fine = -direction_norm * derivative_fine
        hvp = -direction_norm * (4.0 * derivative_fine - derivative_coarse) / 3.0
        errors = np.abs(hvp - hvp_fine)
        maximum_error = float(np.max(errors))
        if maximum_error > self.maximum_error_eV_per_A2:
            if bounded_noise:
                provider.force_sample(geometry)
            raise RuntimeError(
                "Richardson HVP error estimate exceeds the admitted bound: "
                f"{maximum_error:.6e} > {self.maximum_error_eV_per_A2:.6e} eV/A^2."
            )
        if provider.configuration_sha256() != configuration:
            if bounded_noise:
                provider.force_sample(geometry)
            raise RuntimeError(
                "scalar provider configuration drifted during HVP evaluation."
            )
        return RichardsonScalarHVPEvaluation(
            contract_id=RICHARDSON_HVP_CONTRACT,
            provider_configuration_sha256=configuration,
            derivative_policy_sha256=self.policy_sha256(),
            central_sample=center,
            topology_observation_coverage=(
                center.energy_sample.topology_observation_coverage
            ),
            unobservable_topology_components=(
                center.energy_sample.unobservable_topology_components
            ),
            topology_guard_status=topology_guard_status,
            topology_step_reductions_used=topology_step_reductions_used,
            coarse_step_angstrom=coarse_step_angstrom,
            fine_step_angstrom=fine_step_angstrom,
            direction=vector,
            hvp_eV_per_A2=hvp,
            error_estimates_eV_per_A2=errors,
            displaced_force_sha256=tuple(
                sample.evaluation_sha256 for sample in samples
            ),
            displaced_topology_ids=topology_ids,
            displaced_topology_changed=topology_changed,
            energy_force_discrepancies_eV_per_A=discrepancies,
        )

    def evaluate_hvp(
        self,
        provider: ScalarForceSampler,
        geometry: object,
        direction: object,
        *,
        central_sample: ScalarForceSample | None = None,
    ) -> RichardsonScalarHVPEvaluation:
        if not isinstance(getattr(provider, "provider_id", None), str):
            raise TypeError("scalar provider requires a stable provider_id.")
        configuration = provider.configuration_sha256()
        _digest(configuration, name="provider.configuration_sha256()")
        center = _validated_force_center(provider, geometry, central_sample)
        guard_status = _topology_guard_status(
            center.energy_sample.topology_observation_coverage,
            self.topology_guard_policy,
        )
        vector = np.asarray(direction, dtype=float)
        positions = _positions(geometry)
        if vector.shape != positions.shape or not np.all(np.isfinite(vector)):
            raise ValueError(
                "direction must be finite with the geometry position shape."
            )
        direction_norm = float(np.linalg.norm(vector))
        if not np.isfinite(direction_norm) or direction_norm == 0.0:
            raise ValueError("direction must be nonzero.")
        last_topology_error: FiniteDifferenceTopologyChangeError | None = None
        for reduction, coarse_step in enumerate(self._candidate_coarse_steps()):
            try:
                return self._evaluate_hvp_at_step(
                    provider,
                    geometry,
                    vector,
                    center=center,
                    configuration=configuration,
                    coarse_step_angstrom=coarse_step,
                    topology_guard_status=guard_status,
                    topology_step_reductions_used=reduction,
                )
            except FiniteDifferenceTopologyChangeError as exc:
                last_topology_error = exc
        if self.maximum_topology_step_reductions == 0:
            assert last_topology_error is not None
            raise last_topology_error
        raise FiniteDifferenceTopologyChangeError(
            "finite-difference displacement changed the continuum topology at "
            "every admitted adaptive Hessian step through coarse_step="
            f"{self._candidate_coarse_steps()[-1]:.6e} A."
        ) from last_topology_error

    def evaluate(
        self,
        provider: ScalarForceSampler,
        geometry: object,
        *,
        central_sample: ScalarForceSample | None = None,
    ) -> RichardsonScalarHessianEvaluation:
        if not isinstance(getattr(provider, "provider_id", None), str):
            raise TypeError("scalar provider requires a stable provider_id.")
        configuration = provider.configuration_sha256()
        _digest(configuration, name="provider.configuration_sha256()")
        center = _validated_force_center(provider, geometry, central_sample)
        guard_status = _topology_guard_status(
            center.energy_sample.topology_observation_coverage,
            self.topology_guard_policy,
        )
        positions = _positions(geometry)
        dimension = positions.size
        last_topology_error: FiniteDifferenceTopologyChangeError | None = None
        for reduction, coarse_step in enumerate(self._candidate_coarse_steps()):
            raw = np.empty((dimension, dimension), dtype=float)
            errors = np.empty_like(raw)
            state_ids: list[str] = []
            topology_ids: list[str] = []
            topology_changed: list[bool] = []
            discrepancies: list[float] = []
            try:
                for column in range(dimension):
                    direction = np.zeros_like(positions)
                    direction.reshape(-1)[column] = 1.0
                    evaluated = self._evaluate_hvp_at_step(
                        provider,
                        geometry,
                        direction,
                        center=center,
                        configuration=configuration,
                        coarse_step_angstrom=coarse_step,
                        topology_guard_status=guard_status,
                        topology_step_reductions_used=reduction,
                    )
                    raw[:, column] = evaluated.hvp_eV_per_A2.reshape(-1)
                    errors[:, column] = evaluated.error_estimates_eV_per_A2.reshape(-1)
                    state_ids.extend(evaluated.displaced_force_sha256)
                    topology_ids.extend(evaluated.displaced_topology_ids)
                    topology_changed.extend(evaluated.displaced_topology_changed)
                    discrepancies.extend(evaluated.energy_force_discrepancies_eV_per_A)
            except FiniteDifferenceTopologyChangeError as exc:
                last_topology_error = exc
                continue
            antisymmetry = float(np.max(np.abs(raw - raw.T)))
            if antisymmetry > self.maximum_antisymmetry_eV_per_A2:
                if self.topology_guard_policy == BOUNDED_TOPOLOGY_NOISE_EXPERIMENTAL_V1:
                    provider.force_sample(geometry)
                raise RuntimeError(
                    "Numerical scalar Hessian antisymmetry exceeds the admitted bound: "
                    f"{antisymmetry:.6e} > "
                    f"{self.maximum_antisymmetry_eV_per_A2:.6e} eV/A^2."
                )
            symmetric = 0.5 * (raw + raw.T)
            if provider.configuration_sha256() != configuration:
                if self.topology_guard_policy == BOUNDED_TOPOLOGY_NOISE_EXPERIMENTAL_V1:
                    provider.force_sample(geometry)
                raise RuntimeError(
                    "scalar provider configuration drifted during Hessian evaluation."
                )
            return RichardsonScalarHessianEvaluation(
                contract_id=RICHARDSON_HESSIAN_CONTRACT,
                provider_configuration_sha256=configuration,
                derivative_policy_sha256=self.policy_sha256(),
                central_sample=center,
                topology_observation_coverage=(
                    center.energy_sample.topology_observation_coverage
                ),
                unobservable_topology_components=(
                    center.energy_sample.unobservable_topology_components
                ),
                topology_guard_status=guard_status,
                topology_step_reductions_used=reduction,
                coarse_step_angstrom=coarse_step,
                fine_step_angstrom=0.5 * coarse_step,
                raw_hessian_eV_per_A2=raw,
                hessian_eV_per_A2=symmetric,
                error_estimates_eV_per_A2=errors,
                displaced_force_sha256=tuple(state_ids),
                maximum_antisymmetry_eV_per_A2=antisymmetry,
                displaced_topology_ids=tuple(topology_ids),
                displaced_topology_changed=tuple(topology_changed),
                energy_force_discrepancies_eV_per_A=tuple(discrepancies),
            )
        if self.maximum_topology_step_reductions == 0:
            assert last_topology_error is not None
            raise last_topology_error
        raise FiniteDifferenceTopologyChangeError(
            "finite-difference displacement changed the continuum topology at "
            "every admitted adaptive Hessian step through coarse_step="
            f"{self._candidate_coarse_steps()[-1]:.6e} A."
        ) from last_topology_error


__all__ = [
    "BOUNDED_TOPOLOGY_NOISE_EXPERIMENTAL_V1",
    "FiniteDifferenceTopologyChangeError",
    "FiniteDifferenceTopologyObservationError",
    "OBSERVED_COMPONENTS_ONLY_EXPERIMENTAL_V1",
    "RICHARDSON_FORCE_COMPONENT_CONTRACT",
    "RICHARDSON_FORCE_CONTRACT",
    "RICHARDSON_HESSIAN_CONTRACT",
    "RICHARDSON_HVP_CONTRACT",
    "REQUIRE_COMPLETE_TOPOLOGY_OBSERVATION_V1",
    "RichardsonScalarForce",
    "RichardsonScalarForceComponentEvaluation",
    "RichardsonScalarForceEvaluation",
    "RichardsonScalarHessian",
    "RichardsonScalarHessianEvaluation",
    "RichardsonScalarHVPEvaluation",
    "ScalarEnergySample",
    "ScalarEnergySampler",
    "ScalarForceSample",
    "ScalarForceSampler",
    "TOPOLOGY_OBSERVATION_COVERAGES",
    "normalize_topology_observation",
]
