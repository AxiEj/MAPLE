"""Direct-sum ddX continuum for point, radial, and canonical-ADT sources.

The operational MDP/POLAR response has three categorically different source
branches:

* permanent MACE-MDP atom-centred point monopoles/dipoles;
* the nonuniform MACE-POLAR residual in its first 1.5-A radial-GTO block;
* canonical atomic-density-translation (ADT) dipoles.

The ADT branch is deliberately heterogeneous.  Its ddX ``psi`` data are the
exact exterior point-dipole multipoles, while its boundary ``phi`` data are the
analytic finite-density translation-tangent potential.  This is the same
general-source split used by ddPCM: the far-field multipole moments and the
finite-density molecular electrostatic potential are supplied independently.

All three branches share one ddX state and one stationary continuum scalar.
The eight-channel field returned to MACE-POLAR remains the radial external-MEP
receiver owned by :mod:`separated_source_ddx`; it is not confused with the
energy gradient of any source branch.

This module admits no public capability.  In particular, coordinate
derivatives of the moving cavity and of the canonical ADT chart are not yet
implemented, and the finite laboratory-fixed ddX quadrature is not a structural
SO(3) guarantee.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from ase.data import atomic_numbers
from ase.units import Bohr

from maple.function.calculator.extra_correction.implicit.gto_density import (
    external_field_to_density_order,
)
from maple.solvation.api.capabilities import CapabilityStatus
from maple.solvation.coupling.atomic_displacement_lift import (
    CanonicalAtomicDisplacementLift,
)
from maple.solvation.coupling.spaces import (
    ATOMIC_L1_SOURCE_SPACE,
    MACE_POLAR_RADIAL_GTO_FIELD_DUAL_SPACE,
)

from .separated_source_ddx import (
    PreparedSeparatedSourceDDX,
    embed_atomic_l1_in_first_radial_channel,
)

CANONICAL_ADT_SEPARATED_DDX_PROVIDER_ID = (
    "maple.route2.continuum.ddx-separated-point-radial-canonical-adt.impl.v1"
)
CANONICAL_ADT_SEPARATED_DDX_CONTRACT_ID = (
    "maple.route2.continuum.ddx-point-permanent-radial-residual-adt.v1"
)
CANONICAL_ADT_SEPARATED_DDX_PROFILE_ID = (
    "route2-research-ddx-point-permanent-radial-residual-canonical-adt-v1"
)

_ADT_DATA_CONTRACT = "ddx-canonical-adt-general-source-data-v1"
_STATE_CONTRACT = "ddx-point-radial-canonical-adt-state-v1"
_VJP_CONTRACT = "ddx-radial-residual-adt-model-field-vjp-v1"


def _sha(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _array_sha(values: object) -> str:
    array = np.ascontiguousarray(values, dtype="<f8")
    return hashlib.sha256(
        f"{array.dtype.str}:{array.shape}".encode() + b"\0" + array.tobytes(order="C")
    ).hexdigest()


def _digest(value: object, *, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be a SHA256 digest.")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{name} must be a SHA256 digest.") from exc
    return value.lower()


def _readonly(values: object, *, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite with shape {shape}.")
    contiguous = np.ascontiguousarray(array, dtype=np.float64)
    return np.frombuffer(contiguous.tobytes(), dtype=np.float64).reshape(shape)


def _dipoles(values: object, *, atom_count: int, name: str) -> np.ndarray:
    return _readonly(values, shape=(atom_count, 3), name=name)


def _raw_point_dipole_source(dipoles_eangstrom: object) -> np.ndarray:
    dipoles = np.asarray(dipoles_eangstrom, dtype=np.float64)
    if dipoles.ndim != 2 or dipoles.shape[1] != 3 or not np.all(np.isfinite(dipoles)):
        raise ValueError("Cartesian point dipoles must be finite with shape (N,3).")
    external = np.zeros((len(dipoles), 4), dtype=np.float64)
    external[:, 1:] = dipoles
    return ATOMIC_L1_SOURCE_SPACE.validate(
        external_field_to_density_order(external),
        atom_count=len(dipoles),
        name="raw point-dipole source",
    )


def _embedding_matrix(atom_count: int) -> np.ndarray:
    result = np.empty((atom_count * 8, atom_count * 4), dtype=np.float64)
    for column in range(atom_count * 4):
        basis = np.zeros((atom_count, 4), dtype=np.float64)
        basis.reshape(-1)[column] = 1.0
        result[:, column] = embed_atomic_l1_in_first_radial_channel(basis).reshape(-1)
    return _readonly(
        result,
        shape=(atom_count * 8, atom_count * 4),
        name="first-radial embedding matrix",
    )


@dataclass(frozen=True, slots=True)
class CanonicalADTDDXProblemData:
    """Content-addressed ADT maps into one prepared ddX problem."""

    geometry_sha256: str
    continuum_configuration_sha256: str
    cavity_topology_sha256: str
    lift_configuration_sha256: str
    atom_count: int
    basis_count: int
    cavity_point_count: int
    psi_matrix: np.ndarray
    phi_matrix: np.ndarray
    configuration_sha256: str = ""

    def __post_init__(self) -> None:
        geometry = _digest(self.geometry_sha256, name="geometry_sha256")
        continuum = _digest(
            self.continuum_configuration_sha256,
            name="continuum_configuration_sha256",
        )
        topology = _digest(self.cavity_topology_sha256, name="cavity_topology_sha256")
        lift = _digest(self.lift_configuration_sha256, name="lift_configuration_sha256")
        for name in ("atom_count", "basis_count", "cavity_point_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        dimension = self.atom_count * 3
        psi = _readonly(
            self.psi_matrix,
            shape=(self.basis_count * self.atom_count, dimension),
            name="ADT psi matrix",
        )
        phi = _readonly(
            self.phi_matrix,
            shape=(self.cavity_point_count, dimension),
            name="ADT phi matrix",
        )
        expected = _sha(
            {
                "contract": _ADT_DATA_CONTRACT,
                "geometry_sha256": geometry,
                "continuum_configuration_sha256": continuum,
                "cavity_topology_sha256": topology,
                "lift_configuration_sha256": lift,
                "atom_count": self.atom_count,
                "basis_count": self.basis_count,
                "cavity_point_count": self.cavity_point_count,
                "source_coordinates": "atom-cartesian-dipoles-e-angstrom",
                "psi_semantics": "exact-exterior-point-dipole-multipoles",
                "phi_semantics": "analytic-finite-density-translation-tangent-mep",
                "psi_matrix_sha256": _array_sha(psi),
                "phi_matrix_sha256": _array_sha(phi),
            }
        )
        if self.configuration_sha256 and self.configuration_sha256 != expected:
            raise ValueError("configuration_sha256 does not match ADT problem data.")
        object.__setattr__(self, "geometry_sha256", geometry)
        object.__setattr__(self, "continuum_configuration_sha256", continuum)
        object.__setattr__(self, "cavity_topology_sha256", topology)
        object.__setattr__(self, "lift_configuration_sha256", lift)
        object.__setattr__(self, "psi_matrix", psi)
        object.__setattr__(self, "phi_matrix", phi)
        object.__setattr__(self, "configuration_sha256", expected)

    def problem_data(
        self, atomic_dipoles_eangstrom: object
    ) -> tuple[np.ndarray, np.ndarray]:
        dipoles = _dipoles(
            atomic_dipoles_eangstrom,
            atom_count=self.atom_count,
            name="ADT atomic dipoles",
        )
        vector = dipoles.reshape(-1)
        psi = (self.psi_matrix @ vector).reshape(self.basis_count, self.atom_count)
        phi = self.phi_matrix @ vector
        return (
            _readonly(
                psi,
                shape=(self.basis_count, self.atom_count),
                name="ADT psi",
            ),
            _readonly(phi, shape=(self.cavity_point_count,), name="ADT phi"),
        )


@dataclass(frozen=True, slots=True)
class CanonicalADTModelFieldCotangents:
    """Transpose result for the two induced direct-sum branches."""

    radial_residual_source4: np.ndarray
    adt_atomic_dipoles_eangstrom: np.ndarray
    configuration_sha256: str
    state_sha256: str = ""

    def __post_init__(self) -> None:
        radial = np.asarray(self.radial_residual_source4, dtype=np.float64)
        if radial.ndim != 2 or radial.shape[1] != 4:
            raise ValueError("radial_residual_source4 must have shape (N,4).")
        count = len(radial)
        radial = _readonly(
            radial, shape=(count, 4), name="radial residual source cotangent"
        )
        dipoles = _dipoles(
            self.adt_atomic_dipoles_eangstrom,
            atom_count=count,
            name="ADT atomic-dipole cotangent",
        )
        configuration = _digest(self.configuration_sha256, name="configuration_sha256")
        expected = _sha(
            {
                "contract": _VJP_CONTRACT,
                "configuration_sha256": configuration,
                "radial_residual_source4_sha256": _array_sha(radial),
                "adt_atomic_dipoles_eangstrom_sha256": _array_sha(dipoles),
            }
        )
        if self.state_sha256 and self.state_sha256 != expected:
            raise ValueError("state_sha256 does not match model-field cotangents.")
        object.__setattr__(self, "radial_residual_source4", radial)
        object.__setattr__(self, "adt_atomic_dipoles_eangstrom", dipoles)
        object.__setattr__(self, "configuration_sha256", configuration)
        object.__setattr__(self, "state_sha256", expected)


@dataclass(frozen=True, slots=True)
class CanonicalADTSeparatedDDXState:
    """One immutable point + radial-residual + ADT continuum state."""

    geometry_sha256: str
    configuration_sha256: str
    cavity_topology_sha256: str
    permanent_source4: np.ndarray
    radial_residual_source4: np.ndarray
    adt_atomic_dipoles_eangstrom: np.ndarray
    model_field8: np.ndarray
    permanent_energy_gradient4: np.ndarray
    radial_residual_energy_gradient4: np.ndarray
    adt_energy_gradient_eV_per_eangstrom: np.ndarray
    polarization_energy_ev: float
    general_solution_sha256: str
    state_sha256: str = ""

    def __post_init__(self) -> None:
        permanent = np.asarray(self.permanent_source4, dtype=np.float64)
        if permanent.ndim != 2 or permanent.shape[1] != 4:
            raise ValueError("permanent_source4 must have shape (N,4).")
        count = len(permanent)
        permanent = _readonly(
            permanent, shape=(count, 4), name="permanent point source"
        )
        radial = _readonly(
            self.radial_residual_source4,
            shape=(count, 4),
            name="radial residual source",
        )
        dipoles = _dipoles(
            self.adt_atomic_dipoles_eangstrom,
            atom_count=count,
            name="ADT atomic dipoles",
        )
        field = _readonly(
            self.model_field8, shape=(count, 8), name="native model field"
        )
        permanent_gradient = _readonly(
            self.permanent_energy_gradient4,
            shape=(count, 4),
            name="permanent energy gradient",
        )
        radial_gradient = _readonly(
            self.radial_residual_energy_gradient4,
            shape=(count, 4),
            name="radial residual energy gradient",
        )
        adt_gradient = _dipoles(
            self.adt_energy_gradient_eV_per_eangstrom,
            atom_count=count,
            name="ADT energy gradient",
        )
        energy = float(self.polarization_energy_ev)
        if not math.isfinite(energy):
            raise ValueError("polarization_energy_ev must be finite.")
        geometry = _digest(self.geometry_sha256, name="geometry_sha256")
        configuration = _digest(self.configuration_sha256, name="configuration_sha256")
        topology = _digest(self.cavity_topology_sha256, name="cavity_topology_sha256")
        solution = _digest(self.general_solution_sha256, name="general_solution_sha256")
        work = float(np.vdot(permanent, permanent_gradient))
        work += float(np.vdot(radial, radial_gradient))
        work += float(np.vdot(dipoles, adt_gradient))
        tolerance = max(3.0e-10, 3.0e-10 * abs(energy))
        if abs(work - 2.0 * energy) > tolerance:
            raise ValueError("direct-sum state violates the exact half-work identity.")
        expected = _sha(
            {
                "contract": _STATE_CONTRACT,
                "geometry_sha256": geometry,
                "configuration_sha256": configuration,
                "cavity_topology_sha256": topology,
                "permanent_source4_sha256": _array_sha(permanent),
                "radial_residual_source4_sha256": _array_sha(radial),
                "adt_atomic_dipoles_eangstrom_sha256": _array_sha(dipoles),
                "model_field8_sha256": _array_sha(field),
                "permanent_energy_gradient4_sha256": _array_sha(permanent_gradient),
                "radial_residual_energy_gradient4_sha256": _array_sha(radial_gradient),
                "adt_energy_gradient_sha256": _array_sha(adt_gradient),
                "polarization_energy_ev": energy,
                "general_solution_sha256": solution,
            }
        )
        if self.state_sha256 and self.state_sha256 != expected:
            raise ValueError("state_sha256 does not match the direct-sum state.")
        object.__setattr__(self, "geometry_sha256", geometry)
        object.__setattr__(self, "configuration_sha256", configuration)
        object.__setattr__(self, "cavity_topology_sha256", topology)
        object.__setattr__(self, "permanent_source4", permanent)
        object.__setattr__(self, "radial_residual_source4", radial)
        object.__setattr__(self, "adt_atomic_dipoles_eangstrom", dipoles)
        object.__setattr__(self, "model_field8", field)
        object.__setattr__(self, "permanent_energy_gradient4", permanent_gradient)
        object.__setattr__(self, "radial_residual_energy_gradient4", radial_gradient)
        object.__setattr__(self, "adt_energy_gradient_eV_per_eangstrom", adt_gradient)
        object.__setattr__(self, "polarization_energy_ev", energy)
        object.__setattr__(self, "general_solution_sha256", solution)
        object.__setattr__(self, "state_sha256", expected)

    @property
    def energy_dual_work_ev(self) -> float:
        return (
            float(np.vdot(self.permanent_source4, self.permanent_energy_gradient4))
            + float(
                np.vdot(
                    self.radial_residual_source4,
                    self.radial_residual_energy_gradient4,
                )
            )
            + float(
                np.vdot(
                    self.adt_atomic_dipoles_eangstrom,
                    self.adt_energy_gradient_eV_per_eangstrom,
                )
            )
        )

    @property
    def half_work_identity_residual_ev(self) -> float:
        return self.energy_dual_work_ev - 2.0 * self.polarization_energy_ev


class PreparedCanonicalADTSeparatedDDX:
    """Geometry-bound three-branch ddX continuum operator."""

    __slots__ = (
        "_adt_data",
        "_configuration_sha256",
        "_embedding",
        "_implementation_sha256",
        "_lift",
        "_point_phi_matrix",
        "_point_psi_matrix",
        "_prepared",
        "_radial4_phi_matrix",
        "_radial4_psi_matrix",
        "_sealed",
    )

    provider_id = CANONICAL_ADT_SEPARATED_DDX_PROVIDER_ID
    continuum_profile_id = CANONICAL_ADT_SEPARATED_DDX_PROFILE_ID
    contract_id = CANONICAL_ADT_SEPARATED_DDX_CONTRACT_ID
    capabilities = CapabilityStatus()
    coordinate_derivative_available = False
    fixed_topology = False
    source_dependent_geometry = False
    structurally_rotation_equivariant = False
    linear_response = True

    def __init__(
        self,
        prepared: PreparedSeparatedSourceDDX,
        adt_lift: CanonicalAtomicDisplacementLift,
    ) -> None:
        if not isinstance(prepared, PreparedSeparatedSourceDDX):
            raise TypeError("prepared must be PreparedSeparatedSourceDDX.")
        if not isinstance(adt_lift, CanonicalAtomicDisplacementLift):
            raise TypeError("adt_lift must be CanonicalAtomicDisplacementLift.")
        if prepared.atom_count != adt_lift.atom_count:
            raise ValueError("prepared continuum and ADT lift atom counts differ.")
        expected_numbers = np.asarray(
            [atomic_numbers[symbol] for symbol in prepared.symbols], dtype=np.int64
        )
        if not np.array_equal(expected_numbers, adt_lift.atomic_numbers):
            raise ValueError("prepared continuum symbols and ADT elements differ.")

        point_psi, point_phi = prepared.point_problem_matrices()
        radial8_psi, radial8_phi = prepared.radial_problem_matrices()
        embedding = _embedding_matrix(prepared.atom_count)
        radial4_psi = radial8_psi @ embedding
        radial4_phi = radial8_phi @ embedding

        raw_dipole_map = np.empty(
            (prepared.atom_count * 4, prepared.atom_count * 3), dtype=np.float64
        )
        for column in range(prepared.atom_count * 3):
            basis = np.zeros((prepared.atom_count, 3), dtype=np.float64)
            basis.reshape(-1)[column] = 1.0
            raw_dipole_map[:, column] = _raw_point_dipole_source(basis).reshape(-1)
        adt_psi = point_psi @ raw_dipole_map
        # The lift operator is per e*bohr.  Direct-sum response coordinates are
        # e*angstrom, so p[e*bohr] = p[e*angstrom] / Bohr.
        adt_phi = (
            adt_lift.atomic_surface_operator(
                points_bohr=prepared.cavity_points_bohr,
                centers_bohr=prepared.positions_angstrom / Bohr,
            )
            / Bohr
        )
        adt_data = CanonicalADTDDXProblemData(
            geometry_sha256=prepared.geometry_sha256,
            continuum_configuration_sha256=prepared.configuration_sha256,
            cavity_topology_sha256=prepared.cavity_topology_sha256,
            lift_configuration_sha256=adt_lift.configuration_sha256,
            atom_count=prepared.atom_count,
            basis_count=prepared.basis_count,
            cavity_point_count=prepared.cavity_point_count,
            psi_matrix=adt_psi,
            phi_matrix=adt_phi,
        )
        implementation = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        object.__setattr__(self, "_prepared", prepared)
        object.__setattr__(self, "_lift", adt_lift)
        object.__setattr__(self, "_embedding", embedding)
        object.__setattr__(self, "_point_psi_matrix", point_psi)
        object.__setattr__(self, "_point_phi_matrix", point_phi)
        object.__setattr__(
            self,
            "_radial4_psi_matrix",
            _readonly(
                radial4_psi,
                shape=(
                    prepared.basis_count * prepared.atom_count,
                    prepared.atom_count * 4,
                ),
                name="radial residual psi matrix",
            ),
        )
        object.__setattr__(
            self,
            "_radial4_phi_matrix",
            _readonly(
                radial4_phi,
                shape=(prepared.cavity_point_count, prepared.atom_count * 4),
                name="radial residual phi matrix",
            ),
        )
        object.__setattr__(self, "_adt_data", adt_data)
        object.__setattr__(self, "_implementation_sha256", implementation)
        object.__setattr__(self, "_configuration_sha256", self._current_configuration())
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("PreparedCanonicalADTSeparatedDDX is immutable.")
        object.__setattr__(self, name, value)

    @property
    def atom_count(self) -> int:
        return self._prepared.atom_count

    @property
    def permanent_source4(self) -> np.ndarray:
        return self._prepared.permanent_source

    @property
    def adt_problem_data(self) -> CanonicalADTDDXProblemData:
        return self._adt_data

    @property
    def adt_lift(self) -> CanonicalAtomicDisplacementLift:
        return self._lift

    def _current_configuration(self) -> str:
        return _sha(
            {
                "contract": self.contract_id,
                "provider_id": self.provider_id,
                "continuum_profile_id": self.continuum_profile_id,
                "prepared_geometry_sha256": self._prepared.geometry_sha256,
                "prepared_configuration_sha256": self._prepared.configuration_sha256,
                "cavity_topology_sha256": self._prepared.cavity_topology_sha256,
                "permanent_source_sha256": _array_sha(self._prepared.permanent_source),
                "adt_lift_configuration_sha256": self._lift.configuration_sha256,
                "adt_problem_data_sha256": self._adt_data.configuration_sha256,
                "radial4_psi_matrix_sha256": _array_sha(self._radial4_psi_matrix),
                "radial4_phi_matrix_sha256": _array_sha(self._radial4_phi_matrix),
                "implementation_sha256": self._implementation_sha256,
                "coordinate_derivative_available": False,
                "capabilities": "none",
            }
        )

    def configuration_sha256(self) -> str:
        current = self._current_configuration()
        if current != self._configuration_sha256:
            raise RuntimeError("canonical ADT separated ddX configuration drifted.")
        return current

    def _radial_problem_data(self, source4: object) -> tuple[np.ndarray, np.ndarray]:
        source = ATOMIC_L1_SOURCE_SPACE.validate(
            source4,
            atom_count=self.atom_count,
            name="radial residual source",
        )
        vector = source.reshape(-1)
        return (
            _readonly(
                (self._radial4_psi_matrix @ vector).reshape(
                    self._prepared.basis_count, self.atom_count
                ),
                shape=(self._prepared.basis_count, self.atom_count),
                name="radial residual psi",
            ),
            _readonly(
                self._radial4_phi_matrix @ vector,
                shape=(self._prepared.cavity_point_count,),
                name="radial residual phi",
            ),
        )

    def solve(
        self,
        radial_residual_source4: object,
        adt_atomic_dipoles_eangstrom: object,
    ) -> CanonicalADTSeparatedDDXState:
        self.configuration_sha256()
        radial = ATOMIC_L1_SOURCE_SPACE.validate(
            radial_residual_source4,
            atom_count=self.atom_count,
            name="radial residual source",
        )
        dipoles = _dipoles(
            adt_atomic_dipoles_eangstrom,
            atom_count=self.atom_count,
            name="ADT atomic dipoles",
        )
        permanent_psi, permanent_phi = self._prepared.bound_permanent_problem_data()
        radial_psi, radial_phi = self._radial_problem_data(radial)
        adt_psi, adt_phi = self._adt_data.problem_data(dipoles)
        solution = self._prepared.solve_problem_data(
            permanent_psi + radial_psi + adt_psi,
            permanent_phi + radial_phi + adt_phi,
        )
        permanent_gradient = self._prepared.problem_data_energy_gradient(
            solution,
            psi_matrix=self._point_psi_matrix,
            phi_matrix=self._point_phi_matrix,
            source_shape=(self.atom_count, 4),
            name="permanent point source",
        )
        radial_gradient = self._prepared.problem_data_energy_gradient(
            solution,
            psi_matrix=self._radial4_psi_matrix,
            phi_matrix=self._radial4_phi_matrix,
            source_shape=(self.atom_count, 4),
            name="radial residual source",
        )
        adt_gradient = self._prepared.problem_data_energy_gradient(
            solution,
            psi_matrix=self._adt_data.psi_matrix,
            phi_matrix=self._adt_data.phi_matrix,
            source_shape=(self.atom_count, 3),
            name="canonical ADT source",
        )
        return CanonicalADTSeparatedDDXState(
            geometry_sha256=self._prepared.geometry_sha256,
            configuration_sha256=self._configuration_sha256,
            cavity_topology_sha256=self._prepared.cavity_topology_sha256,
            permanent_source4=self._prepared.permanent_source,
            radial_residual_source4=radial,
            adt_atomic_dipoles_eangstrom=dipoles,
            model_field8=solution.model_field,
            permanent_energy_gradient4=permanent_gradient,
            radial_residual_energy_gradient4=radial_gradient,
            adt_energy_gradient_eV_per_eangstrom=adt_gradient,
            polarization_energy_ev=solution.polarization_energy_ev,
            general_solution_sha256=solution.state_sha256,
        )

    def model_field_jvp(
        self,
        radial_residual_direction4: object,
        adt_atomic_dipole_direction_eangstrom: object,
    ) -> np.ndarray:
        self.configuration_sha256()
        radial_psi, radial_phi = self._radial_problem_data(radial_residual_direction4)
        adt_psi, adt_phi = self._adt_data.problem_data(
            adt_atomic_dipole_direction_eangstrom
        )
        return self._prepared.problem_data_model_field_jvp(
            radial_psi + adt_psi, radial_phi + adt_phi
        )

    def model_field_vjp(
        self, field_cotangent: object
    ) -> CanonicalADTModelFieldCotangents:
        self.configuration_sha256()
        cotangent = MACE_POLAR_RADIAL_GTO_FIELD_DUAL_SPACE.validate(
            field_cotangent,
            atom_count=self.atom_count,
            name="native model-field cotangent",
        )
        radial = self._prepared.problem_data_model_field_vjp(
            cotangent,
            source_psi_matrix=self._radial4_psi_matrix,
            source_shape=(self.atom_count, 4),
            name="radial residual source",
        )
        adt = self._prepared.problem_data_model_field_vjp(
            cotangent,
            source_psi_matrix=self._adt_data.psi_matrix,
            source_shape=(self.atom_count, 3),
            name="canonical ADT source",
        )
        return CanonicalADTModelFieldCotangents(
            radial_residual_source4=radial,
            adt_atomic_dipoles_eangstrom=adt,
            configuration_sha256=self._configuration_sha256,
        )

    def coordinate_vjp(self, *args: Any, **kwargs: Any) -> np.ndarray:
        del args, kwargs
        raise NotImplementedError(
            "The canonical ADT direct-sum ddX coordinate VJP is not implemented."
        )


def prepare_canonical_adt_separated_ddx(
    prepared: PreparedSeparatedSourceDDX,
    adt_lift: CanonicalAtomicDisplacementLift,
) -> PreparedCanonicalADTSeparatedDDX:
    """Build the immutable three-branch continuum at one geometry."""

    return PreparedCanonicalADTSeparatedDDX(prepared, adt_lift)


__all__ = [
    "CANONICAL_ADT_SEPARATED_DDX_CONTRACT_ID",
    "CANONICAL_ADT_SEPARATED_DDX_PROFILE_ID",
    "CANONICAL_ADT_SEPARATED_DDX_PROVIDER_ID",
    "CanonicalADTDDXProblemData",
    "CanonicalADTModelFieldCotangents",
    "CanonicalADTSeparatedDDXState",
    "PreparedCanonicalADTSeparatedDDX",
    "prepare_canonical_adt_separated_ddx",
]
