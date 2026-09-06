"""Zero-training MDP molecular-susceptibility correction for MACE-POLAR.

This module implements one deliberately narrow operational hypothesis.  The
unchanged MACE-POLAR source ``M_P(R, u)`` retains all of its atomwise topology
and every second and higher field derivative.  Only its geometry-local,
zero-field uniform tangent is corrected so that the molecular dipole response
equals the symmetric molecular polarizability predicted by MACE-MDP::

    M_corr(R, u) = M_P(R, u) + B(R) [X(R)-I] G(R) u

where ``B = dM_P/du U``, ``C B X = -alpha_MDP``, ``U`` embeds an affine
potential and ``G U = I`` averages only the dimensionally homogeneous native
gradient channels.  The minus sign follows from ``E = -grad(phi)``.

This is not a common energy functional and is not a capability admission.  In
particular, the chart depends on geometry through both checkpoints, so a
complete coordinate VJP is unavailable until those derivatives are derived.
The provider therefore fails closed for coordinate derivatives.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import threading
from typing import Any, Protocol, runtime_checkable

import numpy as np

from maple.solvation.coupling.operator import canonical_metadata_sha256
from maple.solvation.coupling.separated_operators import NativeFieldSpace
from maple.solvation.coupling.spaces import SourceSpace
from maple.solvation.coupling.state_equation import geometry_sha256
from maple.solvation.release.uniform_response_manifold import (
    affine_uniform_native_field,
)
from maple.solvation.release.uniform_susceptibility_replacement import (
    UniformSusceptibilityFieldTransform,
    prepare_uniform_susceptibility_field_transform,
)

from .base import atom_count

MDP_POLAR_ARITHMETIC_TANGENT_CONTRACT = (
    "mace-mdp-molecular-alpha-mace-polar-additive-uniform-tangent-v1"
)
MDP_POLAR_ARITHMETIC_TANGENT_RESPONSE_PROVIDER_ID = (
    "maple.route2.model.mace-mdp-alpha-mace-polar-additive-tangent.impl.v1"
)
MDP_POLAR_ARITHMETIC_TANGENT_MODEL_PROFILE_ID = (
    "route2-research-mace-mdp-alpha-macepolar-additive-tangent-v1"
)
MDP_POLAR_ARITHMETIC_TANGENT_COUPLING_ID = (
    "route2-coupling-mace-mdp-alpha-macepolar-additive-tangent-v1"
)


@runtime_checkable
class MolecularPolarizabilityProvider(Protocol):
    """Geometry model exposing one molecular polarizability tensor."""

    provider_id: str
    model_profile_id: str

    def configuration_sha256(self) -> str: ...

    def evaluate(self, geometry: Any) -> Any: ...


@runtime_checkable
class NativeFieldSourceResponse(Protocol):
    """Four-channel source response to the native eight-channel field."""

    provider_id: str
    model_profile_id: str
    provenance_sha256: str
    source_space: SourceSpace
    receiver_space: NativeFieldSpace
    long_range_evaluator_profile: str

    def configuration_sha256(self) -> str: ...

    def vacuum_energy_ev(self, geometry: Any) -> float: ...

    def vacuum_forces_ev_per_angstrom(self, geometry: Any) -> np.ndarray: ...

    def evaluate_source(self, geometry: Any, field: np.ndarray) -> np.ndarray: ...

    def field_jvp(
        self, geometry: Any, field: np.ndarray, field_direction: np.ndarray
    ) -> np.ndarray: ...

    def field_vjp(
        self, geometry: Any, field: np.ndarray, source_cotangent: np.ndarray
    ) -> np.ndarray: ...


def _public_molecular_polarizability(state: object) -> np.ndarray:
    value = getattr(state, "public_polarizability_eangstrom2_per_volt", None)
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3, 3) or not np.all(np.isfinite(result)):
        raise RuntimeError(
            "Molecular-polarizability provider returned no finite public (3,3) tensor."
        )
    return result


class MDPPolarArithmeticTangentResponse:
    """Add the unique arithmetic-chart uniform tangent to MACE-POLAR.

    The one-entry cache is only a performance cache.  Its value is an immutable,
    content-addressed chart and never participates in configuration identity.
    Exact geometry and provider hashes are checked before every reuse.
    """

    __slots__ = (
        "_base",
        "_cache_geometry_sha256",
        "_cache_lock",
        "_cache_transform",
        "_configuration_sha256",
        "_mdp",
        "_sealed",
        "coupling_id",
        "model_profile_id",
        "provenance_sha256",
        "provider_id",
    )

    capabilities = ()
    variational_functional_admitted = False
    coordinate_derivative_available = False
    correction_kind = "additive-zero-field-uniform-tangent"
    higher_field_derivatives_changed = False

    def __init__(
        self,
        *,
        mdp: MolecularPolarizabilityProvider,
        base: NativeFieldSourceResponse,
    ) -> None:
        for owner, names in (
            (mdp, ("configuration_sha256", "evaluate")),
            (
                base,
                (
                    "configuration_sha256",
                    "vacuum_energy_ev",
                    "vacuum_forces_ev_per_angstrom",
                    "evaluate_source",
                    "field_jvp",
                    "field_vjp",
                ),
            ),
        ):
            for name in names:
                if not callable(getattr(owner, name, None)):
                    raise TypeError(
                        f"susceptibility provider requires callable {name}()."
                    )
        source_space = getattr(base, "source_space", None)
        receiver_space = getattr(base, "receiver_space", None)
        if not isinstance(source_space, SourceSpace):
            raise TypeError("base response must expose SourceSpace.")
        if not isinstance(receiver_space, NativeFieldSpace):
            raise TypeError("base response must expose NativeFieldSpace.")
        if source_space.component_count != 4 or receiver_space.component_count != 8:
            raise ValueError("arithmetic tangent requires source4 and native field8.")
        mdp.configuration_sha256()
        base.configuration_sha256()
        object.__setattr__(self, "_mdp", mdp)
        object.__setattr__(self, "_base", base)
        object.__setattr__(
            self, "provider_id", MDP_POLAR_ARITHMETIC_TANGENT_RESPONSE_PROVIDER_ID
        )
        object.__setattr__(
            self, "model_profile_id", MDP_POLAR_ARITHMETIC_TANGENT_MODEL_PROFILE_ID
        )
        object.__setattr__(
            self, "coupling_id", MDP_POLAR_ARITHMETIC_TANGENT_COUPLING_ID
        )
        provenance = canonical_metadata_sha256(
            {
                "contract": MDP_POLAR_ARITHMETIC_TANGENT_CONTRACT,
                "provider_id": self.provider_id,
                "model_profile_id": self.model_profile_id,
                "coupling_id": self.coupling_id,
                "mdp_provider_id": getattr(mdp, "provider_id", None),
                "mdp_model_profile_id": getattr(mdp, "model_profile_id", None),
                "mdp_configuration_sha256": mdp.configuration_sha256(),
                "base_provider_id": getattr(base, "provider_id", None),
                "base_model_profile_id": getattr(base, "model_profile_id", None),
                "base_provenance_sha256": getattr(base, "provenance_sha256", None),
                "base_configuration_sha256": base.configuration_sha256(),
                "source_space_sha256": source_space.metadata_hash(),
                "receiver_space_sha256": receiver_space.metadata_hash(),
                "continuation": self.correction_kind,
                "uniform_chart": "arithmetic-average-native-gradient-only-v1",
                "coordinate_derivative_available": False,
                "variational_functional_admitted": False,
                "capabilities": "none",
                "implementation_sha256": hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
            }
        )
        object.__setattr__(self, "provenance_sha256", provenance)
        object.__setattr__(self, "_cache_geometry_sha256", None)
        object.__setattr__(self, "_cache_transform", None)
        object.__setattr__(self, "_cache_lock", threading.RLock())
        object.__setattr__(self, "_configuration_sha256", self._current_configuration())
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("MDPPolarArithmeticTangentResponse is immutable.")
        object.__setattr__(self, name, value)

    @property
    def source_space(self) -> SourceSpace:
        return self._base.source_space

    @property
    def receiver_space(self) -> NativeFieldSpace:
        return self._base.receiver_space

    @property
    def long_range_evaluator_profile(self) -> str:
        return str(self._base.long_range_evaluator_profile)

    @property
    def field_energy_pairing_sha256(self) -> str:
        value = getattr(self._base, "field_energy_pairing_sha256", None)
        if not isinstance(value, str) or len(value) != 64:
            raise RuntimeError("base response exposes no valid field pairing digest.")
        return value

    def _current_configuration(self) -> str:
        return canonical_metadata_sha256(
            {
                "contract": MDP_POLAR_ARITHMETIC_TANGENT_CONTRACT,
                "provider_id": self.provider_id,
                "model_profile_id": self.model_profile_id,
                "coupling_id": self.coupling_id,
                "provenance_sha256": self.provenance_sha256,
                "mdp_configuration_sha256": self._mdp.configuration_sha256(),
                "base_configuration_sha256": self._base.configuration_sha256(),
                "source_space_sha256": self.source_space.metadata_hash(),
                "receiver_space_sha256": self.receiver_space.metadata_hash(),
                "long_range_evaluator_profile": self.long_range_evaluator_profile,
                "continuation": self.correction_kind,
                "coordinate_derivative_available": False,
                "capabilities": "none",
            }
        )

    def configuration_sha256(self) -> str:
        current = self._current_configuration()
        if current != self._configuration_sha256:
            raise RuntimeError("MDP/POLAR susceptibility configuration drifted.")
        return current

    def _transform(self, geometry: object) -> UniformSusceptibilityFieldTransform:
        self.configuration_sha256()
        before = geometry_sha256(geometry)
        with self._cache_lock:
            if self._cache_geometry_sha256 == before and isinstance(
                self._cache_transform, UniformSusceptibilityFieldTransform
            ):
                return self._cache_transform

            count = atom_count(geometry)
            positions = np.asarray(
                getattr(geometry, "positions", None), dtype=np.float64
            )
            if positions.shape != (count, 3) or not np.all(np.isfinite(positions)):
                raise ValueError("geometry positions must be finite with shape (N,3).")
            basis = np.stack(
                [
                    affine_uniform_native_field(positions, np.eye(3)[axis])
                    for axis in range(3)
                ],
                axis=-1,
            )
            zero = np.zeros(self.receiver_space.shape(count), dtype=np.float64)
            source_jacobian = np.stack(
                [
                    self._base.field_jvp(geometry, zero, basis[:, :, axis])
                    for axis in range(3)
                ],
                axis=-1,
            )
            alpha = _public_molecular_polarizability(self._mdp.evaluate(geometry))
            transform = prepare_uniform_susceptibility_field_transform(
                positions_angstrom=positions,
                uniform_native_basis=basis,
                polar_zero_uniform_source_jacobian=source_jacobian,
                mdp_molecular_polarizability_eangstrom2_per_volt=alpha,
            )
            after = geometry_sha256(geometry)
            if after != before:
                raise RuntimeError(
                    "geometry changed while building susceptibility chart."
                )
            object.__setattr__(self, "_cache_geometry_sha256", before)
            object.__setattr__(self, "_cache_transform", transform)
            return transform

    def chart_for_geometry(
        self, geometry: object
    ) -> UniformSusceptibilityFieldTransform:
        """Return the immutable geometry-local chart for structural audits."""

        return self._transform(geometry)

    def vacuum_energy_ev(self, geometry: object) -> float:
        self.configuration_sha256()
        return float(self._base.vacuum_energy_ev(geometry))

    def vacuum_forces_ev_per_angstrom(self, geometry: object) -> np.ndarray:
        self.configuration_sha256()
        result = np.asarray(
            self._base.vacuum_forces_ev_per_angstrom(geometry), dtype=np.float64
        )
        if result.shape != (atom_count(geometry), 3) or not np.all(np.isfinite(result)):
            raise RuntimeError("base vacuum forces must be finite with shape (N,3).")
        return result.copy()

    def evaluate_source(self, geometry: object, field: object) -> np.ndarray:
        count = atom_count(geometry)
        values = self.receiver_space.validate(field, atom_count=count)
        source = self.source_space.validate(
            self._base.evaluate_source(geometry, values), atom_count=count
        )
        return self.source_space.validate(
            self._transform(geometry).tangent_corrected_source(source, values),
            atom_count=count,
            name="susceptibility-corrected source",
        )

    def field_jvp(
        self, geometry: object, field: object, field_direction: object
    ) -> np.ndarray:
        count = atom_count(geometry)
        values = self.receiver_space.validate(field, atom_count=count)
        direction = self.receiver_space.validate(
            field_direction, atom_count=count, name="native field direction"
        )
        base = self.source_space.validate(
            self._base.field_jvp(geometry, values, direction), atom_count=count
        )
        correction = self._transform(geometry).tangent_source_direction(direction)
        return self.source_space.validate(
            base + correction,
            atom_count=count,
            name="susceptibility-corrected source JVP",
        )

    def field_vjp(
        self, geometry: object, field: object, source_cotangent: object
    ) -> np.ndarray:
        count = atom_count(geometry)
        values = self.receiver_space.validate(field, atom_count=count)
        cotangent = self.source_space.validate(
            source_cotangent, atom_count=count, name="source cotangent"
        )
        base = self.receiver_space.validate(
            self._base.field_vjp(geometry, values, cotangent), atom_count=count
        )
        correction = self._transform(geometry).pullback_tangent_source_cotangent(
            cotangent
        )
        return self.receiver_space.validate(
            base + correction,
            atom_count=count,
            name="susceptibility-corrected field VJP",
        )

    def coordinate_vjp(
        self, geometry: object, field: object, source_cotangent: object
    ) -> np.ndarray:
        del geometry, field, source_cotangent
        raise NotImplementedError(
            "The MDP/POLAR additive tangent depends on geometry through alpha, "
            "the uniform basis, and the zero-field POLAR Jacobian; its complete "
            "coordinate VJP is not implemented."
        )

    def conditioned_raw_energy_ev(self, geometry: object, field: object) -> float:
        method = getattr(self._base, "conditioned_raw_energy_ev", None)
        if not callable(method):
            raise NotImplementedError(
                "base response exposes no conditioned raw energy."
            )
        return float(method(geometry, field))

    def dense_source_jacobian(self, geometry: object, field: object) -> np.ndarray:
        method = getattr(self._base, "dense_source_jacobian", None)
        if not callable(method):
            raise NotImplementedError("base response exposes no dense source Jacobian.")
        count = atom_count(geometry)
        values = self.receiver_space.validate(field, atom_count=count)
        base = np.asarray(method(geometry, values), dtype=np.float64)
        expected_shape = (count * 4, count * 8)
        if base.shape != expected_shape or not np.all(np.isfinite(base)):
            raise RuntimeError("base dense source Jacobian has invalid shape/content.")
        chart = self._transform(geometry)
        b = chart.polar_zero_uniform_source_jacobian.reshape(count * 4, 3)
        correction = (
            b
            @ (chart.uniform_coordinate_transform - np.eye(3))
            @ chart.uniform_gradient_left_inverse
        )
        return base + correction


__all__ = [
    "MDP_POLAR_ARITHMETIC_TANGENT_CONTRACT",
    "MDP_POLAR_ARITHMETIC_TANGENT_COUPLING_ID",
    "MDP_POLAR_ARITHMETIC_TANGENT_MODEL_PROFILE_ID",
    "MDP_POLAR_ARITHMETIC_TANGENT_RESPONSE_PROVIDER_ID",
    "MDPPolarArithmeticTangentResponse",
    "MolecularPolarizabilityProvider",
    "NativeFieldSourceResponse",
]
