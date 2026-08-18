"""Disabled smooth-partition harmonic ddCOSMO scalar candidate.

Unlike the earlier weighted-shell charge functional, this implementation
retains one fixed block of local reaction-potential coefficients per atom.
Complete burial replaces a boundary equation by a smooth Schwarz consistency
equation; it never removes a coefficient block.  The discrete physical scalar
is

``G(R,c) = f(epsilon)/2 * c.T C(R) X`` with ``L(R) X = -B(R) c``.

``L`` is generally nonsymmetric.  All source and coordinate derivatives are
therefore obtained by differentiating this scalar through the solve, which is
equivalent to the standard ddCOSMO primal-adjoint Lagrangian.  This research
candidate has no public E/F/H/V/M admission.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from maple.solvation.api.capabilities import CapabilityStatus
from maple.solvation.coupling.metrics import ATOMIC_L1_PAIRING
from maple.solvation.coupling.spaces import (
    ATOMIC_L1_FIELD_DUAL_SPACE,
    ATOMIC_L1_SOURCE_SPACE,
)

from .functional import ContinuumEnergyFunctional
from .harmonic_coefficients import _bounded_lmax, _positive_int
from .harmonic_schwarz_primitives import _assemble_point_ddcosmo

SMOOTH_PARTITION_HARMONIC_DDCOSMO_PROVIDER_ID = (
    "maple.route2.continuum.smooth-partition-harmonic-ddcosmo.impl.v1"
)
SMOOTH_PARTITION_HARMONIC_DDCOSMO_CONTRACT_ID = (
    "maple.route2.continuum.fixed-local-potential-schwarz-scalar.v1"
)
SMOOTH_PARTITION_HARMONIC_DDCOSMO_PROFILE_ID = (
    "smooth-partition-harmonic-ddcosmo-point-l1-v1"
)
SMOOTH_PARTITION_HARMONIC_DDCOSMO_SCALAR_ID = (
    "route2-research-macepolar-frozen-point-l1-smoothharmonic-ddcosmo-v1"
)
_MINIMUM_RELATIVE_SINGULAR_VALUE = 1.0e-10
_MAXIMUM_CONDITION_NUMBER = 1.0e10


def _sha(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _implementation_sha256() -> tuple[tuple[str, str], ...]:
    directory = Path(__file__).resolve().parent
    names = (
        "functional.py",
        "harmonic_coefficients.py",
        "harmonic_ddcosmo_functional.py",
        "harmonic_exposure.py",
        "harmonic_point_source.py",
        "harmonic_schwarz_primitives.py",
        "harmonic_torch_primitives.py",
    )
    return tuple(
        (name, hashlib.sha256((directory / name).read_bytes()).hexdigest())
        for name in names
    )


def _positive_float(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise TypeError(f"{name} must be a real scalar.")
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return result


def _bounded_order(value: object, *, name: str) -> int:
    result = _positive_int(value, name=name)
    if result > 4096:
        raise ValueError(f"{name} exceeds the bounded contract.")
    return result


class SmoothPartitionHarmonicDDCOSMOFunctionalCandidate(ContinuumEnergyFunctional):
    """Fixed-coefficient, scalar-first, point-l<=1 harmonic ddCOSMO."""

    __slots__ = (
        "_candidate_configuration_sha256",
        "_candidate_provenance_sha256",
        "_dielectric",
        "_partition_lmax",
        "_partition_radial_order",
        "_radii_angstrom",
        "_runtime_device",
        "_runtime_dtype",
        "_screening_factor",
        "_source_radial_order",
        "_surface_lmax",
        "_transition_width_angstrom2",
    )

    provider_id = SMOOTH_PARTITION_HARMONIC_DDCOSMO_PROVIDER_ID
    functional_contract_id = SMOOTH_PARTITION_HARMONIC_DDCOSMO_CONTRACT_ID
    continuum_profile_id = SMOOTH_PARTITION_HARMONIC_DDCOSMO_PROFILE_ID
    scalar_id = SMOOTH_PARTITION_HARMONIC_DDCOSMO_SCALAR_ID
    capabilities = CapabilityStatus()
    scalar_first = True
    linear_response = True
    reciprocal_energy_hessian = True
    conventional_receiver_equals_energy_gradient = False
    fixed_topology = True
    source_dependent_geometry = False
    electrostatics_only = True
    include_nonpolar = False
    coefficient_action_is_so3_representation = True
    laboratory_fixed_surface_grid = False
    smooth_partition = True
    derivatives_generated_from_same_scalar = True
    tier_v_admitted = False

    def __init__(
        self,
        *,
        atomic_numbers: tuple[int, ...],
        radii_angstrom: tuple[float, ...],
        transition_width_angstrom2: float,
        surface_lmax: int,
        partition_lmax: int,
        partition_radial_quadrature_order: int = 96,
        source_radial_quadrature_order: int = 128,
        dielectric: float,
        dtype: object,
        device: object,
    ) -> None:
        numbers = tuple(atomic_numbers)
        if not numbers or any(
            isinstance(number, bool) or not isinstance(number, int) or number < 1
            for number in numbers
        ):
            raise ValueError("atomic_numbers must contain positive integers.")
        radii = tuple(
            _positive_float(value, name=f"radii_angstrom[{index}]")
            for index, value in enumerate(radii_angstrom)
        )
        if len(radii) != len(numbers):
            raise ValueError("radii_angstrom must have one value per atom.")
        maximum = _bounded_lmax(surface_lmax)
        partition_maximum = _bounded_lmax(partition_lmax)
        if partition_maximum < 2 * maximum:
            raise ValueError("partition_lmax must be at least twice surface_lmax.")
        _bounded_lmax(maximum + partition_maximum)
        partition_order = _bounded_order(
            partition_radial_quadrature_order,
            name="partition_radial_quadrature_order",
        )
        source_order = _bounded_order(
            source_radial_quadrature_order,
            name="source_radial_quadrature_order",
        )
        width = _positive_float(
            transition_width_angstrom2, name="transition_width_angstrom2"
        )
        epsilon = _positive_float(dielectric, name="dielectric")
        if epsilon < 1.0:
            raise ValueError("dielectric must be at least one.")
        screening = (epsilon - 1.0) / epsilon
        runtime_dtype = str(dtype)
        runtime_device = str(device)
        payload = {
            "provider_id": self.provider_id,
            "functional_contract_id": self.functional_contract_id,
            "continuum_profile_id": self.continuum_profile_id,
            "scalar_id": self.scalar_id,
            "atomic_numbers": numbers,
            "radii_angstrom": radii,
            "transition_width_angstrom2": width,
            "surface_lmax": maximum,
            "partition_lmax": partition_maximum,
            "partition_radial_quadrature_order": partition_order,
            "source_radial_quadrature_order": source_order,
            "dielectric": epsilon,
            "screening_factor": screening,
            "source_space_sha256": ATOMIC_L1_SOURCE_SPACE.metadata_hash(),
            "field_space_sha256": ATOMIC_L1_FIELD_DUAL_SPACE.metadata_hash(),
            "pairing_sha256": ATOMIC_L1_PAIRING.metadata_hash(),
            "runtime_dtype": runtime_dtype,
            "runtime_device": runtime_device,
            "assembly": "L X=-B c; G=f_eps/2 c.T C X",
            "partition": "polynomial-Shapley-coefficient-partition-v1",
            "angular_contractions": "finite-band-exact",
            "capabilities": "none",
            "implementation_sha256": _implementation_sha256(),
        }
        configuration = _sha(payload)
        provenance = _sha(
            {
                "provider_id": self.provider_id,
                "configuration_sha256": configuration,
                "implementation_sha256": _implementation_sha256(),
                "scientific_status": (
                    "disabled-research-ddcosmo; finite-dielectric-ddpcm-not-yet-implemented"
                ),
            }
        )
        object.__setattr__(self, "_radii_angstrom", radii)
        object.__setattr__(self, "_transition_width_angstrom2", width)
        object.__setattr__(self, "_surface_lmax", maximum)
        object.__setattr__(self, "_partition_lmax", partition_maximum)
        object.__setattr__(self, "_partition_radial_order", partition_order)
        object.__setattr__(self, "_source_radial_order", source_order)
        object.__setattr__(self, "_dielectric", epsilon)
        object.__setattr__(self, "_screening_factor", screening)
        object.__setattr__(self, "_runtime_dtype", runtime_dtype)
        object.__setattr__(self, "_runtime_device", runtime_device)
        object.__setattr__(self, "_candidate_configuration_sha256", configuration)
        object.__setattr__(self, "_candidate_provenance_sha256", provenance)
        super().__init__(
            source_space=ATOMIC_L1_SOURCE_SPACE,
            field_space=ATOMIC_L1_FIELD_DUAL_SPACE,
            pairing=ATOMIC_L1_PAIRING,
            dtype=dtype,
            device=device,
            expected_atomic_numbers=numbers,
        )

    @property
    def radii_angstrom(self) -> tuple[float, ...]:
        return self._radii_angstrom

    @property
    def atomic_numbers(self) -> tuple[int, ...]:
        return self._expected_atomic_numbers

    @property
    def transition_width_angstrom2(self) -> float:
        return self._transition_width_angstrom2

    @property
    def surface_lmax(self) -> int:
        return self._surface_lmax

    @property
    def partition_lmax(self) -> int:
        return self._partition_lmax

    @property
    def partition_radial_quadrature_order(self) -> int:
        return self._partition_radial_order

    @property
    def source_radial_quadrature_order(self) -> int:
        return self._source_radial_order

    @property
    def runtime_dtype(self) -> str:
        return self._runtime_dtype

    @property
    def runtime_device(self) -> str:
        return self._runtime_device

    @property
    def dielectric(self) -> float:
        return self._dielectric

    @property
    def screening_factor(self) -> float:
        return self._screening_factor

    @property
    def provenance_sha256(self) -> str:
        self.configuration_sha256()
        return self._candidate_provenance_sha256

    def configuration_sha256(self) -> str:
        current = _sha(
            {
                "provider_id": self.provider_id,
                "functional_contract_id": self.functional_contract_id,
                "continuum_profile_id": self.continuum_profile_id,
                "scalar_id": self.scalar_id,
                "atomic_numbers": self._expected_atomic_numbers,
                "radii_angstrom": self._radii_angstrom,
                "transition_width_angstrom2": self._transition_width_angstrom2,
                "surface_lmax": self._surface_lmax,
                "partition_lmax": self._partition_lmax,
                "partition_radial_quadrature_order": self._partition_radial_order,
                "source_radial_quadrature_order": self._source_radial_order,
                "dielectric": self._dielectric,
                "screening_factor": self._screening_factor,
                "source_space_sha256": self.source_space.metadata_hash(),
                "field_space_sha256": self.field_space.metadata_hash(),
                "pairing_sha256": self.pairing.metadata_hash(),
                "runtime_dtype": self._runtime_dtype,
                "runtime_device": self._runtime_device,
                "assembly": "L X=-B c; G=f_eps/2 c.T C X",
                "partition": "polynomial-Shapley-coefficient-partition-v1",
                "angular_contractions": "finite-band-exact",
                "capabilities": "none",
                "implementation_sha256": _implementation_sha256(),
            }
        )
        if current != self._candidate_configuration_sha256:
            raise RuntimeError("smooth harmonic ddCOSMO configuration drifted.")
        expected_provenance = _sha(
            {
                "provider_id": self.provider_id,
                "configuration_sha256": current,
                "implementation_sha256": _implementation_sha256(),
                "scientific_status": (
                    "disabled-research-ddcosmo; finite-dielectric-ddpcm-not-yet-implemented"
                ),
            }
        )
        if expected_provenance != self._candidate_provenance_sha256:
            raise RuntimeError("smooth harmonic ddCOSMO provenance drifted.")
        return current

    def _assemble_torch(self, positions: Any):
        matrices = _assemble_point_ddcosmo(
            positions,
            radii=self._radii_angstrom,
            transition_width=self._transition_width_angstrom2,
            lmax=self._surface_lmax,
            partition_lmax=self._partition_lmax,
            partition_radial_order=self._partition_radial_order,
            source_radial_order=self._source_radial_order,
        )
        schwarz, source_operator, receiver, partition = matrices
        singular_values = __import__("torch").linalg.svdvals(schwarz).detach()
        relative = float((singular_values[-1] / singular_values[0]).cpu())
        condition = float((singular_values[0] / singular_values[-1]).cpu())
        if (
            not np.isfinite(relative)
            or relative <= _MINIMUM_RELATIVE_SINGULAR_VALUE
            or not np.isfinite(condition)
            or condition > _MAXIMUM_CONDITION_NUMBER
        ):
            raise ValueError(
                "smooth harmonic ddCOSMO Schwarz system failed its invertibility gate."
            )
        return schwarz, source_operator, receiver, partition

    def _energy_torch(self, positions: Any, source: Any):
        self.configuration_sha256()
        schwarz, source_operator, receiver, _partition = self._assemble_torch(positions)
        vector = source.reshape(-1)
        reaction_coefficients = __import__("torch").linalg.solve(
            schwarz, -(source_operator @ vector)
        )
        return (
            0.5 * self._screening_factor * (vector @ (receiver @ reaction_coefficients))
        )

    def debug_geometry_matrices(self, geometry: object) -> dict[str, np.ndarray]:
        positions = self._positions_tensor(
            geometry,
            atom_count=len(self._radii_angstrom),
            requires_grad=False,
        )
        schwarz, source_operator, receiver, partition = self._assemble_torch(positions)
        exposed = __import__("torch").stack(tuple(item[0] for item in partition))
        return {
            "schwarz_operator": np.asarray(schwarz.detach().cpu(), dtype=float).copy(),
            "source_operator": np.asarray(
                source_operator.detach().cpu(), dtype=float
            ).copy(),
            "receiver": np.asarray(receiver.detach().cpu(), dtype=float).copy(),
            "exposed_coefficients": np.asarray(
                exposed.detach().cpu(), dtype=float
            ).copy(),
        }

    def runtime_provenance(self) -> tuple[tuple[str, object], ...]:
        return tuple(
            sorted(
                {
                    "provider_id": self.provider_id,
                    "configuration_sha256": self.configuration_sha256(),
                    "provenance_sha256": self.provenance_sha256,
                    "continuum_profile_id": self.continuum_profile_id,
                    "scalar_id": self.scalar_id,
                    "stationary_equation": "L X=-B c",
                    "scalar": "f_eps/2 c.T C X",
                    "surface_lmax": self._surface_lmax,
                    "partition_lmax": self._partition_lmax,
                    "laboratory_fixed_surface_grid": False,
                    "capabilities": "none",
                }.items()
            )
        )


__all__ = [
    "SMOOTH_PARTITION_HARMONIC_DDCOSMO_CONTRACT_ID",
    "SMOOTH_PARTITION_HARMONIC_DDCOSMO_PROFILE_ID",
    "SMOOTH_PARTITION_HARMONIC_DDCOSMO_PROVIDER_ID",
    "SMOOTH_PARTITION_HARMONIC_DDCOSMO_SCALAR_ID",
    "SmoothPartitionHarmonicDDCOSMOFunctionalCandidate",
]
