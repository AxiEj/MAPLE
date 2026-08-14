"""Disabled stationary scalar over an external harmonic-Galerkin snapshot.

The coefficient representation lives in :mod:`harmonic_coefficients`.  This
module owns only immutable matrix state, identity binding, and the sealed
same-scalar continuum functional.  Geometry-dependent analytic assembly is
not implemented, so all Route-2 E/F/H/V/M capabilities remain disabled.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import numpy as np

from maple.solvation.api.capabilities import CapabilityStatus
from maple.solvation.api.profiles import (
    FIXED_EXTERNAL_HARMONIC_CAVITY_PROFILE_ID,
    FIXED_HARMONIC_GALERKIN_CPCM_CONTINUUM_PROFILE_ID,
    FIXED_HARMONIC_GALERKIN_CONFIGURATION_CONTRACT_ID,
    MACE_POLAR_RADIAL_GTO_COUPLING_ID,
)
from maple.solvation.api.scalar_registry import (
    VARIATIONAL_MACEPOLAR_ENERGYGRADIENT_FIXEDCAVITY_HARMONIC_GALERKIN_CPCM_V1,
)
from maple.solvation.coupling.metrics import MACE_POLAR_RADIAL_GTO_PAIRING
from maple.solvation.coupling.spaces import (
    MACE_POLAR_RADIAL_GTO_FIELD_DUAL_SPACE,
    MACE_POLAR_RADIAL_GTO_SOURCE_SPACE,
)

from .functional import ContinuumEnergyFunctional
from .harmonic_coefficients import (
    HARMONIC_GALERKIN_COEFFICIENT_CONTRACT_ID,
    PerAtomHarmonicSpace,
    radial_gto_source_rotation_matrix,
    real_wigner_generators,
    real_wigner_matrix,
)

HARMONIC_GALERKIN_CPCM_FUNCTIONAL_PROVIDER_ID = (
    "maple.route2.continuum.fixed-harmonic-galerkin-cpcm-candidate.impl.v1"
)
HARMONIC_GALERKIN_STATIONARY_SCALAR_CONTRACT_ID = (
    "maple.route2.continuum-functional.symmetric-harmonic-galerkin.v1"
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _sha(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _digest(value: object, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA256 digest.")
    return value


def _nonempty(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")
    return value.strip()


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values, dtype="<f8")
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("ascii"))
    digest.update(b"|<f8|")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _implementation_sha256() -> tuple[tuple[str, str], ...]:
    directory = Path(__file__).resolve().parent
    return tuple(
        (name, hashlib.sha256((directory / name).read_bytes()).hexdigest())
        for name in (
            "functional.py",
            "harmonic_coefficients.py",
            "harmonic_galerkin.py",
        )
    )


@dataclass(frozen=True, slots=True, init=False)
class FixedHarmonicGalerkinSnapshot:
    """Immutable externally assembled coefficient matrices at one fixed cavity."""

    atomic_numbers: tuple[int, ...]
    coefficient_space: PerAtomHarmonicSpace
    cavity_descriptor_sha256: str
    assembly_contract_id: str
    minimum_eigenvalue: float
    configuration_sha256: str
    provenance_sha256: str
    state_sha256: str
    _surface_values: tuple[float, ...]
    _source_values: tuple[float, ...]

    def __init__(
        self,
        *,
        atomic_numbers: object,
        coefficient_space: PerAtomHarmonicSpace,
        surface_operator: object,
        source_operator: object,
        cavity_descriptor_sha256: str,
        assembly_contract_id: str,
    ) -> None:
        if not isinstance(coefficient_space, PerAtomHarmonicSpace):
            raise TypeError("coefficient_space must be PerAtomHarmonicSpace.")
        numbers = tuple(atomic_numbers)
        if len(numbers) != coefficient_space.atom_count or any(
            isinstance(number, bool)
            or not isinstance(number, (int, np.integer))
            or int(number) < 1
            for number in numbers
        ):
            raise ValueError(
                "atomic_numbers must match coefficient_space.atom_count and be positive."
            )
        numbers = tuple(int(number) for number in numbers)
        cavity_hash = _digest(cavity_descriptor_sha256, name="cavity_descriptor_sha256")
        assembly_id = _nonempty(assembly_contract_id, name="assembly_contract_id")

        surface = np.asarray(surface_operator, dtype=float)
        expected_surface_shape = (
            coefficient_space.dimension,
            coefficient_space.dimension,
        )
        if surface.shape != expected_surface_shape or not np.all(np.isfinite(surface)):
            raise ValueError(
                "surface_operator must be finite with shape "
                f"{expected_surface_shape}."
            )
        if not np.allclose(surface, surface.T, atol=2.0e-13, rtol=0.0):
            raise ValueError("surface_operator must be symmetric.")
        surface = 0.5 * (surface + surface.T)
        eigenvalues = np.linalg.eigvalsh(surface)
        minimum_eigenvalue = float(eigenvalues[0])
        if not np.isfinite(minimum_eigenvalue) or minimum_eigenvalue <= 1.0e-12:
            raise ValueError("surface_operator must be strictly positive definite.")

        source_dimension = coefficient_space.atom_count * (
            MACE_POLAR_RADIAL_GTO_SOURCE_SPACE.component_count
        )
        source = np.asarray(source_operator, dtype=float)
        expected_source_shape = (coefficient_space.dimension, source_dimension)
        if source.shape != expected_source_shape or not np.all(np.isfinite(source)):
            raise ValueError(
                "source_operator must be finite with shape " f"{expected_source_shape}."
            )
        surface = np.ascontiguousarray(surface, dtype=float)
        source = np.ascontiguousarray(source, dtype=float)
        configuration = _sha(
            {
                "coefficient_space_sha256": coefficient_space.metadata_sha256(),
                "cavity_descriptor_sha256": cavity_hash,
                "assembly_contract_id": assembly_id,
                "source_space_sha256": (
                    MACE_POLAR_RADIAL_GTO_SOURCE_SPACE.metadata_hash()
                ),
                "field_space_sha256": (
                    MACE_POLAR_RADIAL_GTO_FIELD_DUAL_SPACE.metadata_hash()
                ),
                "pairing_sha256": MACE_POLAR_RADIAL_GTO_PAIRING.metadata_hash(),
                "surface_operator_sha256": _array_sha256(surface),
                "source_operator_sha256": _array_sha256(source),
                "stationary_scalar": "-1/2 (S c)^T A^-1 (S c)",
                "geometry_assembly": "external-coefficient-snapshot-only",
            }
        )
        provenance = _sha(
            {
                "configuration_sha256": configuration,
                "implementation_sha256": _implementation_sha256(),
                "capabilities": "none",
            }
        )
        state = _sha(
            {
                "configuration_sha256": configuration,
                "provenance_sha256": provenance,
                "atomic_numbers": numbers,
            }
        )

        object.__setattr__(self, "atomic_numbers", numbers)
        object.__setattr__(self, "coefficient_space", coefficient_space)
        object.__setattr__(self, "cavity_descriptor_sha256", cavity_hash)
        object.__setattr__(self, "assembly_contract_id", assembly_id)
        object.__setattr__(self, "minimum_eigenvalue", minimum_eigenvalue)
        object.__setattr__(self, "configuration_sha256", configuration)
        object.__setattr__(self, "provenance_sha256", provenance)
        object.__setattr__(self, "state_sha256", state)
        object.__setattr__(
            self, "_surface_values", tuple(float(value) for value in surface.ravel())
        )
        object.__setattr__(
            self, "_source_values", tuple(float(value) for value in source.ravel())
        )
        self.validate()

    @property
    def source_dimension(self) -> int:
        return self.coefficient_space.atom_count * (
            MACE_POLAR_RADIAL_GTO_SOURCE_SPACE.component_count
        )

    @property
    def surface_operator(self) -> np.ndarray:
        result = np.asarray(self._surface_values, dtype=float).reshape(
            self.coefficient_space.dimension, self.coefficient_space.dimension
        )
        result.setflags(write=False)
        return result

    @property
    def source_operator(self) -> np.ndarray:
        result = np.asarray(self._source_values, dtype=float).reshape(
            self.coefficient_space.dimension, self.source_dimension
        )
        result.setflags(write=False)
        return result

    def validate(self) -> None:
        recomputed_minimum = float(np.linalg.eigvalsh(self.surface_operator)[0])
        if recomputed_minimum <= 1.0e-12 or not np.isclose(
            recomputed_minimum,
            self.minimum_eigenvalue,
            atol=1.0e-15,
            rtol=1.0e-13,
        ):
            raise RuntimeError("harmonic Galerkin eigenvalue certificate drifted.")
        reconstructed = FixedHarmonicGalerkinSnapshot._identity_payload(
            atomic_numbers=self.atomic_numbers,
            coefficient_space=self.coefficient_space,
            cavity_descriptor_sha256=self.cavity_descriptor_sha256,
            assembly_contract_id=self.assembly_contract_id,
            surface_operator=self.surface_operator,
            source_operator=self.source_operator,
        )
        if reconstructed[0] != self.configuration_sha256:
            raise RuntimeError("harmonic Galerkin snapshot configuration drifted.")
        if reconstructed[1] != self.provenance_sha256:
            raise RuntimeError("harmonic Galerkin snapshot provenance drifted.")
        if reconstructed[2] != self.state_sha256:
            raise RuntimeError("harmonic Galerkin snapshot state identity drifted.")

    @staticmethod
    def _identity_payload(
        *,
        atomic_numbers: tuple[int, ...],
        coefficient_space: PerAtomHarmonicSpace,
        cavity_descriptor_sha256: str,
        assembly_contract_id: str,
        surface_operator: np.ndarray,
        source_operator: np.ndarray,
    ) -> tuple[str, str, str]:
        configuration = _sha(
            {
                "coefficient_space_sha256": coefficient_space.metadata_sha256(),
                "cavity_descriptor_sha256": cavity_descriptor_sha256,
                "assembly_contract_id": assembly_contract_id,
                "source_space_sha256": (
                    MACE_POLAR_RADIAL_GTO_SOURCE_SPACE.metadata_hash()
                ),
                "field_space_sha256": (
                    MACE_POLAR_RADIAL_GTO_FIELD_DUAL_SPACE.metadata_hash()
                ),
                "pairing_sha256": MACE_POLAR_RADIAL_GTO_PAIRING.metadata_hash(),
                "surface_operator_sha256": _array_sha256(surface_operator),
                "source_operator_sha256": _array_sha256(source_operator),
                "stationary_scalar": "-1/2 (S c)^T A^-1 (S c)",
                "geometry_assembly": "external-coefficient-snapshot-only",
            }
        )
        provenance = _sha(
            {
                "configuration_sha256": configuration,
                "implementation_sha256": _implementation_sha256(),
                "capabilities": "none",
            }
        )
        state = _sha(
            {
                "configuration_sha256": configuration,
                "provenance_sha256": provenance,
                "atomic_numbers": atomic_numbers,
            }
        )
        return configuration, provenance, state

    def rotated(self, rotation: object) -> "FixedHarmonicGalerkinSnapshot":
        self.validate()
        surface_rotation = self.coefficient_space.representation_matrix(rotation)
        source_rotation = radial_gto_source_rotation_matrix(
            rotation, atom_count=self.coefficient_space.atom_count
        )
        rotated_surface = surface_rotation @ self.surface_operator @ surface_rotation.T
        rotated_source = surface_rotation @ self.source_operator @ source_rotation.T
        return FixedHarmonicGalerkinSnapshot(
            atomic_numbers=self.atomic_numbers,
            coefficient_space=self.coefficient_space,
            surface_operator=rotated_surface,
            source_operator=rotated_source,
            cavity_descriptor_sha256=_sha(
                {
                    "contract": "coefficient-conjugated-fixed-cavity.v1",
                    "coefficient_space_sha256": (
                        self.coefficient_space.metadata_sha256()
                    ),
                    "surface_operator_sha256": _array_sha256(rotated_surface),
                    "source_operator_sha256": _array_sha256(rotated_source),
                }
            ),
            assembly_contract_id=self.assembly_contract_id,
        )


class FixedHarmonicGalerkinCPCMCandidate(ContinuumEnergyFunctional):
    """One fixed coefficient snapshot; deliberately not a geometry assembler."""

    __slots__ = (
        "_candidate_configuration_sha256",
        "_candidate_provenance_sha256",
        "_runtime_device",
        "_runtime_dtype",
        "_snapshot",
    )

    provider_id = HARMONIC_GALERKIN_CPCM_FUNCTIONAL_PROVIDER_ID
    scalar_id = (
        VARIATIONAL_MACEPOLAR_ENERGYGRADIENT_FIXEDCAVITY_HARMONIC_GALERKIN_CPCM_V1
    )
    functional_contract_id = HARMONIC_GALERKIN_STATIONARY_SCALAR_CONTRACT_ID
    continuum_profile_id = FIXED_HARMONIC_GALERKIN_CPCM_CONTINUUM_PROFILE_ID
    cavity_profile_id = FIXED_EXTERNAL_HARMONIC_CAVITY_PROFILE_ID
    coupling_id = MACE_POLAR_RADIAL_GTO_COUPLING_ID
    configuration_contract_id = FIXED_HARMONIC_GALERKIN_CONFIGURATION_CONTRACT_ID
    capabilities = CapabilityStatus()
    scalar_first = True
    reciprocal = True
    linear_response = True
    fixed_cavity_descriptor = True
    source_dependent_geometry = False
    electrostatics_only = True
    include_nonpolar = False
    coefficient_action_is_so3_representation = True
    full_geometry_intertwiner_assembly_available = False
    moving_cavity_coordinate_derivative_available = False
    tier_v_rotation_admitted = False

    def __init__(
        self,
        snapshot: FixedHarmonicGalerkinSnapshot,
        *,
        dtype: object,
        device: object,
    ) -> None:
        if not isinstance(snapshot, FixedHarmonicGalerkinSnapshot):
            raise TypeError("snapshot must be FixedHarmonicGalerkinSnapshot.")
        snapshot.validate()
        runtime_dtype = str(dtype)
        runtime_device = str(device)
        configuration = _sha(
            self._configuration_payload(
                snapshot=snapshot,
                runtime_dtype=runtime_dtype,
                runtime_device=runtime_device,
            )
        )
        provenance = _sha(
            {
                "provider_id": self.provider_id,
                "configuration_sha256": configuration,
                "snapshot_provenance_sha256": snapshot.provenance_sha256,
            }
        )
        object.__setattr__(self, "_snapshot", snapshot)
        object.__setattr__(self, "_runtime_dtype", runtime_dtype)
        object.__setattr__(self, "_runtime_device", runtime_device)
        object.__setattr__(self, "_candidate_configuration_sha256", configuration)
        object.__setattr__(self, "_candidate_provenance_sha256", provenance)
        super().__init__(
            source_space=MACE_POLAR_RADIAL_GTO_SOURCE_SPACE,
            field_space=MACE_POLAR_RADIAL_GTO_FIELD_DUAL_SPACE,
            pairing=MACE_POLAR_RADIAL_GTO_PAIRING,
            dtype=dtype,
            device=device,
            expected_atomic_numbers=snapshot.atomic_numbers,
        )

    @property
    def snapshot(self) -> FixedHarmonicGalerkinSnapshot:
        self.configuration_sha256()
        return self._snapshot

    @property
    def provenance_sha256(self) -> str:
        self.configuration_sha256()
        return self._candidate_provenance_sha256

    @classmethod
    def _configuration_payload(
        cls,
        *,
        snapshot: FixedHarmonicGalerkinSnapshot,
        runtime_dtype: str,
        runtime_device: str,
    ) -> dict[str, object]:
        return {
            "provider_id": cls.provider_id,
            "scalar_id": cls.scalar_id,
            "functional_contract_id": cls.functional_contract_id,
            "continuum_profile_id": cls.continuum_profile_id,
            "cavity_profile_id": cls.cavity_profile_id,
            "coupling_id": cls.coupling_id,
            "configuration_contract_id": cls.configuration_contract_id,
            "snapshot_state_sha256": snapshot.state_sha256,
            "source_space_sha256": MACE_POLAR_RADIAL_GTO_SOURCE_SPACE.metadata_hash(),
            "field_space_sha256": (
                MACE_POLAR_RADIAL_GTO_FIELD_DUAL_SPACE.metadata_hash()
            ),
            "pairing_sha256": MACE_POLAR_RADIAL_GTO_PAIRING.metadata_hash(),
            "runtime_dtype": runtime_dtype,
            "runtime_device": runtime_device,
            "implementation_sha256": _implementation_sha256(),
            "geometry_assembly": "external-coefficient-snapshot-only",
            "tier_v_admission": "disabled",
        }

    def configuration_sha256(self) -> str:
        self._snapshot.validate()
        current = _sha(
            self._configuration_payload(
                snapshot=self._snapshot,
                runtime_dtype=self._runtime_dtype,
                runtime_device=self._runtime_device,
            )
        )
        if current != self._candidate_configuration_sha256:
            raise RuntimeError("harmonic Galerkin functional configuration drifted.")
        expected_provenance = _sha(
            {
                "provider_id": self.provider_id,
                "configuration_sha256": current,
                "snapshot_provenance_sha256": self._snapshot.provenance_sha256,
            }
        )
        if expected_provenance != self._candidate_provenance_sha256:
            raise RuntimeError("harmonic Galerkin functional provenance drifted.")
        return current

    def _energy_torch(self, positions: Any, source: Any):
        del positions
        torch = __import__("torch")
        surface = source.new_tensor(
            np.array(self._snapshot.surface_operator, copy=True)
        )
        source_operator = source.new_tensor(
            np.array(self._snapshot.source_operator, copy=True)
        )
        source_vector = source.reshape(-1)
        right_hand_side = source_operator @ source_vector
        surface_state = torch.linalg.solve(surface, right_hand_side)
        return -0.5 * (right_hand_side @ surface_state)

    def rotated(self, rotation: object) -> "FixedHarmonicGalerkinCPCMCandidate":
        self.configuration_sha256()
        return type(self)(
            self._snapshot.rotated(rotation),
            dtype=self._torch_dtype,
            device=self._torch_device,
        )

    def runtime_provenance(self) -> tuple[tuple[str, object], ...]:
        return tuple(
            sorted(
                {
                    "provider_id": self.provider_id,
                    "provenance_sha256": self.provenance_sha256,
                    "configuration_sha256": self.configuration_sha256(),
                    "snapshot_state_sha256": self._snapshot.state_sha256,
                    "coefficient_contract_id": (
                        self._snapshot.coefficient_space.coefficient_contract_id
                    ),
                    "stationary_scalar": "-1/2 (S c)^T A^-1 (S c)",
                    "derivative_route": "sealed-same-scalar-autograd",
                    "geometry_assembly": "external-coefficient-snapshot-only",
                    "tier_v_admission": "disabled",
                    "capabilities": "none",
                }.items()
            )
        )


__all__ = [
    "FixedHarmonicGalerkinCPCMCandidate",
    "FixedHarmonicGalerkinSnapshot",
    "HARMONIC_GALERKIN_COEFFICIENT_CONTRACT_ID",
    "HARMONIC_GALERKIN_CPCM_FUNCTIONAL_PROVIDER_ID",
    "HARMONIC_GALERKIN_STATIONARY_SCALAR_CONTRACT_ID",
    "PerAtomHarmonicSpace",
    "radial_gto_source_rotation_matrix",
    "real_wigner_generators",
    "real_wigner_matrix",
]
