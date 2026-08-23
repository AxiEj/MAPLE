"""Generic same-scalar E/F/virial/HVP/H facade for operational roots.

The fixed-point equation selects a state; a registered ledger selects the
scalar.  This module never infers an energy from a response equation.  It
only composes an immutable equation/ledger builder, the deterministic reduced
root solver, the exact implicit first derivative of that same ledger, and the
existing Richardson HVP/Hessian differentiation of conservative forces.

Both pure MACE-POLAR and permanent/induced MLIP combinations use this facade.
Model- and continuum-specific construction belongs in small builders.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from maple.solvation.coupling.fixed_point import FixedPointOptions
from maple.solvation.coupling.operator import canonical_metadata_sha256
from maple.solvation.coupling.separated_fixed_point import (
    solve_separated_fixed_point,
)
from maple.solvation.coupling.separated_ledgers import (
    OperationalLedgerEvaluation,
    SeparatedOperationalGradientResult,
)
from maple.solvation.coupling.separated_state import SeparatedReducedStateEquation
from maple.solvation.coupling.state_equation import (
    geometry_sha256,
    provider_behavior_sha256,
)
from maple.solvation.derivatives import (
    MolecularVirialEvaluation,
    RichardsonScalarHessian,
    RichardsonScalarHessianEvaluation,
    RichardsonScalarHVPEvaluation,
    ScalarEnergySample,
    ScalarForceSample,
    evaluate_molecular_virial,
)


OPERATIONAL_IMPLICIT_PES_CONTRACT_ID = "route2-operational-implicit-scalar-pes-v1"
_MODULE_PATH = Path(__file__).resolve()


def _text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")
    return value.strip()


def _digest(value: object, *, name: str) -> str:
    result = _text(value, name=name).lower()
    if len(result) != 64 or any(
        character not in "0123456789abcdef" for character in result
    ):
        raise ValueError(f"{name} must be a lowercase SHA256 digest.")
    return result


def _root_options_payload(options: FixedPointOptions) -> dict[str, object]:
    return {
        "method": options.method,
        "tolerance": options.tolerance,
        "max_iterations": options.max_iterations,
        "damping": options.damping,
        "history": options.history,
    }


def _hessian_payload(backend: RichardsonScalarHessian) -> dict[str, float]:
    return {
        "coarse_step_angstrom": backend.coarse_step_angstrom,
        "maximum_error_eV_per_A2": backend.maximum_error_eV_per_A2,
        "maximum_antisymmetry_eV_per_A2": (
            backend.maximum_antisymmetry_eV_per_A2
        ),
    }


@dataclass(frozen=True, slots=True)
class OperationalEquationLedgerBundle:
    """One geometry-bound equation and its explicitly selected scalar ledger."""

    equation: SeparatedReducedStateEquation
    ledger: object
    scalar_id: str
    topology_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.equation, SeparatedReducedStateEquation):
            raise TypeError("equation must satisfy SeparatedReducedStateEquation.")
        scalar_id = _text(self.scalar_id, name="scalar_id")
        topology_id = _text(self.topology_id, name="topology_id")
        if getattr(self.ledger, "equation", None) is not self.equation:
            raise ValueError("ledger must bind the exact equation object.")
        if getattr(self.ledger, "scalar_id", None) != scalar_id:
            raise ValueError("ledger scalar identity does not match the bundle.")
        for name in ("configuration_sha256", "evaluate_root", "implicit_gradient"):
            if not callable(getattr(self.ledger, name, None)):
                raise TypeError(f"ledger requires callable {name}().")
        self.equation.fingerprint_sha256()
        _digest(self.ledger.configuration_sha256(), name="ledger configuration")
        object.__setattr__(self, "scalar_id", scalar_id)
        object.__setattr__(self, "topology_id", topology_id)


@runtime_checkable
class OperationalEquationLedgerBuilder(Protocol):
    """Geometry-to-equation/ledger factory for one immutable profile."""

    provider_id: str
    scalar_id: str

    def configuration_sha256(self) -> str: ...

    def build(self, geometry: object) -> OperationalEquationLedgerBundle: ...


@dataclass(frozen=True, slots=True)
class OperationalImplicitEvaluation:
    """Content-addressed same-scalar root, energy, and optional analytic force."""

    provider_id: str
    scalar_id: str
    provider_configuration_sha256: str
    geometry_sha256: str
    topology_id: str
    root_hash: str
    ledger_sha256: str
    energy_eV: float
    root_residual_norm: float
    forces_eV_per_A: np.ndarray | None = None
    adjoint_true_residual_norm: float | None = None
    state_sha256: str = ""
    evaluation_sha256: str = ""

    def __post_init__(self) -> None:
        provider_id = _text(self.provider_id, name="provider_id")
        scalar_id = _text(self.scalar_id, name="scalar_id")
        topology_id = _text(self.topology_id, name="topology_id")
        for name in (
            "provider_configuration_sha256",
            "geometry_sha256",
            "root_hash",
            "ledger_sha256",
        ):
            _digest(getattr(self, name), name=name)
        energy = float(self.energy_eV)
        residual = float(self.root_residual_norm)
        if not np.isfinite(energy):
            raise ValueError("energy_eV must be finite.")
        if not np.isfinite(residual) or residual < 0.0:
            raise ValueError("root_residual_norm must be finite and non-negative.")
        state_payload = {
            "contract": OPERATIONAL_IMPLICIT_PES_CONTRACT_ID,
            "provider_id": provider_id,
            "scalar_id": scalar_id,
            "provider_configuration_sha256": self.provider_configuration_sha256,
            "geometry_sha256": self.geometry_sha256,
            "topology_id": topology_id,
            "root_hash": self.root_hash,
            "ledger_sha256": self.ledger_sha256,
            "energy_eV": energy,
            "root_residual_norm": residual,
        }
        expected_state = canonical_metadata_sha256(state_payload)
        if self.state_sha256 and self.state_sha256 != expected_state:
            raise ValueError("state_sha256 does not match the energy state.")

        force_copy = None
        adjoint = self.adjoint_true_residual_norm
        if self.forces_eV_per_A is None:
            if adjoint is not None:
                raise ValueError("adjoint residual requires a force evaluation.")
        else:
            values = np.asarray(self.forces_eV_per_A, dtype=float)
            if values.ndim != 2 or values.shape[1] != 3 or not np.all(
                np.isfinite(values)
            ):
                raise ValueError("forces_eV_per_A must be finite with shape (N,3).")
            force_copy = np.frombuffer(
                np.ascontiguousarray(values, dtype=np.float64).tobytes(),
                dtype=np.float64,
            ).reshape(values.shape)
            adjoint = float(adjoint) if adjoint is not None else np.nan
            if not np.isfinite(adjoint) or adjoint < 0.0:
                raise ValueError(
                    "adjoint_true_residual_norm must be finite and non-negative."
                )
        evaluation_payload = {
            **state_payload,
            "state_sha256": expected_state,
            "forces_eV_per_A": (
                None if force_copy is None else force_copy.tolist()
            ),
            "adjoint_true_residual_norm": (
                None if force_copy is None else adjoint
            ),
        }
        expected_evaluation = canonical_metadata_sha256(evaluation_payload)
        if self.evaluation_sha256 and self.evaluation_sha256 != expected_evaluation:
            raise ValueError("evaluation_sha256 does not match evaluation content.")
        object.__setattr__(self, "provider_id", provider_id)
        object.__setattr__(self, "scalar_id", scalar_id)
        object.__setattr__(self, "topology_id", topology_id)
        object.__setattr__(self, "energy_eV", energy)
        object.__setattr__(self, "root_residual_norm", residual)
        object.__setattr__(self, "forces_eV_per_A", force_copy)
        object.__setattr__(self, "adjoint_true_residual_norm", adjoint)
        object.__setattr__(self, "state_sha256", expected_state)
        object.__setattr__(self, "evaluation_sha256", expected_evaluation)

    @property
    def energy_sample(self) -> ScalarEnergySample:
        return ScalarEnergySample(
            energy_eV=self.energy_eV,
            state_sha256=self.state_sha256,
            topology_id=self.topology_id,
        )

    @property
    def force_sample(self) -> ScalarForceSample:
        if self.forces_eV_per_A is None:
            raise ValueError("this evaluation does not contain forces.")
        return ScalarForceSample(
            energy_sample=self.energy_sample,
            forces_eV_per_A=self.forces_eV_per_A,
            evaluation_sha256=self.evaluation_sha256,
        )


class OperationalImplicitPES:
    """Common callable E/F/virial/HVP/H surface for one operational ledger."""

    __slots__ = (
        "_builder",
        "_configuration_sha256",
        "_hessian_backend",
        "_root_options",
        "_sealed",
        "provider_id",
        "scalar_id",
    )

    contract_id = OPERATIONAL_IMPLICIT_PES_CONTRACT_ID
    scientific_status = (
        "internal-same-scalar-E-F-virial-HVP-H; public admission remains separate"
    )

    def __init__(
        self,
        builder: OperationalEquationLedgerBuilder,
        *,
        root_options: FixedPointOptions = FixedPointOptions(),
        hessian_backend: RichardsonScalarHessian = RichardsonScalarHessian(),
    ) -> None:
        if not isinstance(builder, OperationalEquationLedgerBuilder):
            raise TypeError("builder must satisfy OperationalEquationLedgerBuilder.")
        if not isinstance(root_options, FixedPointOptions):
            raise TypeError("root_options must be FixedPointOptions.")
        if not isinstance(hessian_backend, RichardsonScalarHessian):
            raise TypeError("hessian_backend must be RichardsonScalarHessian.")
        provider_id = _text(builder.provider_id, name="builder.provider_id")
        scalar_id = _text(builder.scalar_id, name="builder.scalar_id")
        _digest(builder.configuration_sha256(), name="builder configuration")
        object.__setattr__(self, "_builder", builder)
        object.__setattr__(self, "_root_options", root_options)
        object.__setattr__(self, "_hessian_backend", hessian_backend)
        object.__setattr__(self, "provider_id", provider_id)
        object.__setattr__(self, "scalar_id", scalar_id)
        object.__setattr__(
            self, "_configuration_sha256", self._current_configuration_sha256()
        )
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("OperationalImplicitPES is immutable.")
        object.__setattr__(self, name, value)

    def _current_configuration_sha256(self) -> str:
        return canonical_metadata_sha256(
            {
                "contract": self.contract_id,
                "provider_id": self.provider_id,
                "scalar_id": self.scalar_id,
                "builder_configuration_sha256": (
                    self._builder.configuration_sha256()
                ),
                "builder_behavior_sha256": provider_behavior_sha256(
                    self._builder,
                    ("configuration_sha256", "build"),
                    label="operational_equation_ledger_builder",
                ),
                "root_options": _root_options_payload(self._root_options),
                "hessian_backend": _hessian_payload(self._hessian_backend),
                "implementation_sha256": canonical_metadata_sha256(
                    {"bytes": _MODULE_PATH.read_bytes().hex()}
                ),
                "capabilities": "none",
            }
        )

    def configuration_sha256(self) -> str:
        current = self._current_configuration_sha256()
        if current != self._configuration_sha256:
            raise RuntimeError("operational implicit PES configuration drifted.")
        return current

    def evaluate(
        self, geometry: object, *, need_forces: bool
    ) -> OperationalImplicitEvaluation:
        if type(need_forces) is not bool:
            raise TypeError("need_forces must be exactly bool.")
        configuration = self.configuration_sha256()
        bundle = self._builder.build(geometry)
        if not isinstance(bundle, OperationalEquationLedgerBundle):
            raise TypeError("builder.build() must return OperationalEquationLedgerBundle.")
        if bundle.scalar_id != self.scalar_id:
            raise ValueError("builder returned a different scalar identity.")
        geometry_digest = geometry_sha256(geometry)
        root = solve_separated_fixed_point(
            bundle.equation,
            geometry,
            root_context_id=(
                f"{self.provider_id}:{configuration[:16]}:{geometry_digest[:16]}"
            ),
            options=self._root_options,
        )
        ledger_evaluation = bundle.ledger.evaluate_root(
            geometry,
            root.y_array(),
            root_tolerance=self._root_options.tolerance,
        )
        if not isinstance(ledger_evaluation, OperationalLedgerEvaluation):
            raise TypeError("ledger.evaluate_root() returned an invalid evaluation.")
        if ledger_evaluation.scalar_id != self.scalar_id:
            raise ValueError("ledger evaluation changed scalar identity.")

        forces = None
        adjoint_residual = None
        if need_forces:
            gradient = bundle.ledger.implicit_gradient(geometry, root)
            if not isinstance(gradient, SeparatedOperationalGradientResult):
                raise TypeError("ledger.implicit_gradient() returned an invalid result.")
            if gradient.ledger.ledger_sha256 != ledger_evaluation.ledger_sha256:
                raise RuntimeError("energy and force did not replay the same ledger.")
            if gradient.ledger.total_energy_eV != ledger_evaluation.total_energy_eV:
                raise RuntimeError("energy and force scalar values differ.")
            forces = gradient.forces_array()
            adjoint_residual = gradient.adjoint.true_residual_norm

        if self.configuration_sha256() != configuration:
            raise RuntimeError("operational PES configuration drifted during evaluation.")
        return OperationalImplicitEvaluation(
            provider_id=self.provider_id,
            scalar_id=self.scalar_id,
            provider_configuration_sha256=configuration,
            geometry_sha256=geometry_digest,
            topology_id=bundle.topology_id,
            root_hash=root.root_hash,
            ledger_sha256=ledger_evaluation.ledger_sha256,
            energy_eV=ledger_evaluation.total_energy_eV,
            root_residual_norm=root.actual_unmixed_residual_norm,
            forces_eV_per_A=forces,
            adjoint_true_residual_norm=adjoint_residual,
        )

    def sample(self, geometry: object) -> ScalarEnergySample:
        return self.evaluate(geometry, need_forces=False).energy_sample

    def force_sample(self, geometry: object) -> ScalarForceSample:
        return self.evaluate(geometry, need_forces=True).force_sample

    def get_potential_energy(self, geometry: object) -> float:
        return self.sample(geometry).energy_eV

    def get_forces(self, geometry: object) -> np.ndarray:
        return np.array(self.force_sample(geometry).forces_eV_per_A, copy=True)

    def molecular_virial(
        self,
        geometry: object,
        *,
        origin_angstrom: object | None = None,
        central_force: ScalarForceSample | None = None,
    ) -> MolecularVirialEvaluation:
        center = self.force_sample(geometry) if central_force is None else central_force
        if not isinstance(center, ScalarForceSample):
            raise TypeError("central_force must be ScalarForceSample.")
        return evaluate_molecular_virial(
            geometry,
            center,
            origin_angstrom=origin_angstrom,
        )

    def hessian_vector_product(
        self,
        geometry: object,
        direction: object,
        *,
        central_force: ScalarForceSample | None = None,
    ) -> RichardsonScalarHVPEvaluation:
        center = self.force_sample(geometry) if central_force is None else central_force
        if not isinstance(center, ScalarForceSample):
            raise TypeError("central_force must be ScalarForceSample.")
        return self._hessian_backend.evaluate_hvp(
            self, geometry, direction, central_sample=center
        )

    def evaluate_hessian(
        self,
        geometry: object,
        *,
        central_force: ScalarForceSample | None = None,
    ) -> RichardsonScalarHessianEvaluation:
        center = self.force_sample(geometry) if central_force is None else central_force
        if not isinstance(center, ScalarForceSample):
            raise TypeError("central_force must be ScalarForceSample.")
        return self._hessian_backend.evaluate(
            self, geometry, central_sample=center
        )

    def get_hessian(self, geometry: object) -> np.ndarray:
        return np.array(self.evaluate_hessian(geometry).hessian_eV_per_A2, copy=True)


__all__ = [
    "OPERATIONAL_IMPLICIT_PES_CONTRACT_ID",
    "OperationalEquationLedgerBuilder",
    "OperationalEquationLedgerBundle",
    "OperationalImplicitEvaluation",
    "OperationalImplicitPES",
]
