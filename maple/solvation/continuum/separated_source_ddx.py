"""ddX continuum for heterogeneous permanent and induced source kernels.

The scientific contract is the direct sum

``permanent atomic l<=1 point multipoles + induced radial-GTO multipoles``.

The two branches share the same ddPCM/ddCOSMO state but not the same
source-to-boundary map.  In particular, permanent moments enter through
ddX's point-multipole ``phi`` while the radial source retains its finite-width
Gaussian ``phi``.  Both branches contribute the corresponding ``psi`` data.
The eight-channel field returned to a response model is the external-MEP
boundary adjoint, not the general-source energy gradient.  This distinction is
required because a Gaussian has nonzero density outside every finite cavity,
whereas ddX's general-source energy assumes compact support inside the cavity.
The state therefore exposes both objects under different names:

* ``model_field`` drives a field-responsive MLIP through the ``phi``-side
  boundary adjoint;
* ``energy_source_gradient`` differentiates the frozen operational energy
  ledger and is used only by its implicit adjoint.

This module is model agnostic: it knows neither MACE-MDP nor MACE-POLAR.  A
model adapter owns the permanent moments and the induced-source fixed point.
The finite ddX cavity discretization is still not a structural SO(3)/global
smoothness proof, so release capabilities remain closed while chemistry and
derivative evidence are accumulated.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Literal, Sequence

import numpy as np
from ase.units import Bohr

from maple.function.calculator.extra_correction.implicit.gto_density import (
    cartesian_multipoles,
)
from maple.solvation.api.capabilities import CapabilityStatus
from maple.solvation.api.units import HARTREE_TO_EV
from maple.solvation.coupling.exact_gto import (
    mace_polar_learned_source_embedding_matrix,
)
from maple.solvation.coupling.spaces import (
    ATOMIC_L1_SOURCE_SPACE,
    MACE_POLAR_RADIAL_GTO_FIELD_DUAL_SPACE,
    MACE_POLAR_RADIAL_GTO_SOURCE_SPACE,
)
from maple.solvation.coupling.state_equation import geometry_sha256

from .radial_gto_ddx import (
    MACE_POLAR_RADIAL_SIGMAS_ANGSTROM,
    RadialGTODDXBackend,
    _radial_coefficients,
    gaussian_multipole_potential_displacement_gradient,
    gaussian_multipole_potential_position_vjp,
)

SEPARATED_SOURCE_DDX_PROVIDER_ID = (
    "maple.route2.continuum.ddx-separated-point-radial.impl.v2"
)
SEPARATED_SOURCE_DDX_CONTRACT_ID = (
    "maple.route2.continuum.ddx-point-permanent-plus-radial-induced.v2"
)
_STATE_CONTRACT = "ddx-separated-source-state-v2"


def _sha(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _array_sha(values: object) -> str:
    array = np.ascontiguousarray(values, dtype="<f8")
    return hashlib.sha256(
        f"{array.dtype.str}:{array.shape}".encode() + b"\0" + array.tobytes(order="C")
    ).hexdigest()


def _readonly(values: object, *, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite with shape {shape}.")
    result = np.ascontiguousarray(array, dtype=float)
    result.setflags(write=False)
    return result


def embed_atomic_l1_in_first_radial_channel(source: object) -> np.ndarray:
    """Embed atom-centred ``[q,y,z,x]`` values into the first radial block."""

    values = np.asarray(source, dtype=float)
    if values.ndim != 2 or values.shape[1] != 4 or not np.all(np.isfinite(values)):
        raise ValueError("atomic l<=1 source must be finite with shape (N,4).")
    return values @ mace_polar_learned_source_embedding_matrix().T


def extract_atomic_l1_first_radial_cotangent(cotangent: object) -> np.ndarray:
    """Pull an eight-channel radial cotangent back to atomic ``[q,y,z,x]``."""

    values = np.asarray(cotangent, dtype=float)
    if values.ndim != 2 or values.shape[1] != 8 or not np.all(np.isfinite(values)):
        raise ValueError("radial cotangent must be finite with shape (N,8).")
    return values @ mace_polar_learned_source_embedding_matrix()


@dataclass(frozen=True, slots=True)
class SeparatedSourceDDXState:
    """Immutable operational field and scalar derivative at one geometry."""

    geometry_sha256: str
    configuration_sha256: str
    permanent_source: np.ndarray
    radial_source: np.ndarray
    model_field: np.ndarray
    permanent_energy_gradient: np.ndarray
    energy_source_gradient: np.ndarray
    polarization_energy_ev: float
    cavity_point_count: int
    state_sha256: str = ""

    def __post_init__(self) -> None:
        permanent = np.asarray(self.permanent_source, dtype=float)
        radial = np.asarray(self.radial_source, dtype=float)
        model_field = np.asarray(self.model_field, dtype=float)
        permanent_gradient = np.asarray(self.permanent_energy_gradient, dtype=float)
        energy_gradient = np.asarray(self.energy_source_gradient, dtype=float)
        if permanent.ndim != 2 or permanent.shape[1] != 4:
            raise ValueError("permanent_source must have shape (N,4).")
        atom_count = len(permanent)
        permanent = _readonly(permanent, shape=(atom_count, 4), name="permanent_source")
        radial = _readonly(radial, shape=(atom_count, 8), name="radial_source")
        model_field = _readonly(model_field, shape=(atom_count, 8), name="model_field")
        permanent_gradient = _readonly(
            permanent_gradient,
            shape=(atom_count, 4),
            name="permanent_energy_gradient",
        )
        energy_gradient = _readonly(
            energy_gradient,
            shape=(atom_count, 8),
            name="energy_source_gradient",
        )
        energy = float(self.polarization_energy_ev)
        if not math.isfinite(energy):
            raise ValueError("polarization_energy_ev must be finite.")
        if type(self.cavity_point_count) is not int or self.cavity_point_count < 1:
            raise ValueError("cavity_point_count must be a positive integer.")
        payload = {
            "contract": _STATE_CONTRACT,
            "geometry_sha256": self.geometry_sha256,
            "configuration_sha256": self.configuration_sha256,
            "permanent_source_sha256": _array_sha(permanent),
            "radial_source_sha256": _array_sha(radial),
            "model_field_sha256": _array_sha(model_field),
            "permanent_energy_gradient_sha256": _array_sha(permanent_gradient),
            "energy_source_gradient_sha256": _array_sha(energy_gradient),
            "polarization_energy_ev": energy,
            "cavity_point_count": self.cavity_point_count,
        }
        expected = _sha(payload)
        if self.state_sha256 and self.state_sha256 != expected:
            raise ValueError("state_sha256 does not match the state contents.")
        object.__setattr__(self, "permanent_source", permanent)
        object.__setattr__(self, "radial_source", radial)
        object.__setattr__(self, "model_field", model_field)
        object.__setattr__(self, "permanent_energy_gradient", permanent_gradient)
        object.__setattr__(self, "energy_source_gradient", energy_gradient)
        object.__setattr__(self, "polarization_energy_ev", energy)
        object.__setattr__(self, "state_sha256", expected)


class PreparedSeparatedSourceDDX:
    """Geometry/permanent-source-bound ddX problem reusable across SCF steps."""

    __slots__ = (
        "_backend",
        "_configuration_sha256",
        "_geometry_sha256",
        "_permanent_electric_field",
        "_permanent_phi",
        "_permanent_phi_matrix",
        "_permanent_psi",
        "_permanent_psi_matrix",
        "_permanent_multipoles",
        "_permanent_source",
        "_problem",
    )

    def __init__(
        self,
        backend: "SeparatedSourceDDXBackend",
        geometry: Any,
        permanent_source: object,
    ) -> None:
        count = len(backend.symbols)
        permanent = ATOMIC_L1_SOURCE_SPACE.validate(
            permanent_source, atom_count=count, name="permanent point source"
        )
        problem = backend.radial_backend.prepare_problem(geometry)
        radial_alias = embed_atomic_l1_in_first_radial_channel(permanent)
        multipoles = backend.radial_backend._ddx_total_multipoles(radial_alias)
        electrostatics = problem.model.multipole_electrostatics(
            multipoles, derivative_order=1
        )
        phi = np.asarray(electrostatics["phi"], dtype=float)
        electric_field = np.asarray(electrostatics["e"], dtype=float)
        psi = np.asarray(problem.model.multipole_psi(multipoles), dtype=float)
        expected_phi = (int(problem.model.n_cav),)
        expected_electric_field = (3, int(problem.model.n_cav))
        expected_psi = (int(problem.model.n_basis), count)
        if (
            phi.shape != expected_phi
            or electric_field.shape != expected_electric_field
            or psi.shape != expected_psi
        ):
            raise RuntimeError("ddX point-source construction returned wrong shapes.")

        source_dimension = count * 4
        phi_matrix = np.empty((expected_phi[0], source_dimension), dtype=float)
        psi_matrix = np.empty(
            (expected_psi[0] * expected_psi[1], source_dimension), dtype=float
        )
        for column in range(source_dimension):
            direction = np.zeros((count, 4), dtype=float)
            direction.reshape(-1)[column] = 1.0
            direction_multipoles = backend.radial_backend._ddx_total_multipoles(
                embed_atomic_l1_in_first_radial_channel(direction)
            )
            phi_matrix[:, column] = np.asarray(
                problem.model.multipole_electrostatics(
                    direction_multipoles, derivative_order=0
                )["phi"],
                dtype=float,
            )
            psi_matrix[:, column] = np.asarray(
                problem.model.multipole_psi(direction_multipoles), dtype=float
            ).reshape(-1)
        if not np.allclose(
            phi_matrix @ permanent.reshape(-1), phi, rtol=0.0, atol=2.0e-13
        ) or not np.allclose(
            (psi_matrix @ permanent.reshape(-1)).reshape(expected_psi),
            psi,
            rtol=0.0,
            atol=2.0e-13,
        ):
            raise RuntimeError("ddX point-source matrices do not replay the source.")
        self._backend = backend
        self._problem = problem
        self._permanent_source = _readonly(
            permanent, shape=(count, 4), name="permanent source"
        )
        self._permanent_phi = _readonly(phi, shape=expected_phi, name="permanent phi")
        self._permanent_psi = _readonly(psi, shape=expected_psi, name="permanent psi")
        self._permanent_electric_field = _readonly(
            electric_field,
            shape=expected_electric_field,
            name="permanent electric field",
        )
        self._permanent_multipoles = _readonly(
            multipoles,
            shape=(4, count),
            name="permanent multipoles",
        )
        self._permanent_phi_matrix = _readonly(
            phi_matrix,
            shape=phi_matrix.shape,
            name="permanent phi matrix",
        )
        self._permanent_psi_matrix = _readonly(
            psi_matrix,
            shape=psi_matrix.shape,
            name="permanent psi matrix",
        )
        self._geometry_sha256 = geometry_sha256(geometry)
        self._configuration_sha256 = backend.configuration_sha256()

    @property
    def permanent_source(self) -> np.ndarray:
        return self._permanent_source

    @property
    def cavity_point_count(self) -> int:
        return int(self._problem.model.n_cav)

    @property
    def cavity_topology_sha256(self) -> str:
        """Return the exposed ddX-node topology bound to this geometry."""

        return self._problem.cavity_topology_sha256

    def _radial_problem_data(
        self, radial_source: object
    ) -> tuple[np.ndarray, np.ndarray]:
        values = MACE_POLAR_RADIAL_GTO_SOURCE_SPACE.validate(
            radial_source,
            atom_count=len(self._permanent_source),
            name="radial induced source",
        )
        vector = values.reshape(-1)
        phi = self._problem.phi_matrix @ vector
        psi = (self._problem.psi_matrix @ vector).reshape(
            int(self._problem.model.n_basis), len(self._permanent_source)
        )
        return psi, phi

    def _solve_raw(self, radial_source: object):
        values = MACE_POLAR_RADIAL_GTO_SOURCE_SPACE.validate(
            radial_source,
            atom_count=len(self._permanent_source),
            name="radial induced source",
        )
        psi, phi = self._radial_problem_data(values)
        raw_state, energy, energy_gradient = self._problem.solve_general(
            self._permanent_psi + psi, self._permanent_phi + phi
        )
        return values, psi, phi, raw_state, energy, energy_gradient

    def _permanent_energy_gradient(self, raw_state: object) -> np.ndarray:
        forward = np.asarray(getattr(raw_state, "x", None), dtype=float).reshape(-1)
        adjoint = np.asarray(getattr(raw_state, "xi", None), dtype=float).reshape(-1)
        gradient = 0.5 * (
            self._permanent_psi_matrix.T @ forward
            - self._permanent_phi_matrix.T @ adjoint
        )
        gradient *= self._backend.radial_backend._dielectric_scaling * HARTREE_TO_EV
        return ATOMIC_L1_SOURCE_SPACE.validate(
            gradient.reshape(self._permanent_source.shape),
            atom_count=len(self._permanent_source),
            name="permanent-source energy gradient",
        )

    def _state_from_raw(
        self,
        values: np.ndarray,
        raw_state: object,
        energy: float,
        energy_gradient: object,
    ) -> SeparatedSourceDDXState:
        model_field = MACE_POLAR_RADIAL_GTO_FIELD_DUAL_SPACE.validate(
            self._problem.external_mep_model_field(raw_state),
            atom_count=len(self._permanent_source),
            name="external-MEP model field",
        )
        energy_gradient = MACE_POLAR_RADIAL_GTO_FIELD_DUAL_SPACE.validate(
            energy_gradient,
            atom_count=len(self._permanent_source),
            name="operational energy source gradient",
        )
        return SeparatedSourceDDXState(
            geometry_sha256=self._geometry_sha256,
            configuration_sha256=self._configuration_sha256,
            permanent_source=self._permanent_source,
            radial_source=values,
            model_field=model_field,
            permanent_energy_gradient=self._permanent_energy_gradient(raw_state),
            energy_source_gradient=energy_gradient,
            polarization_energy_ev=energy,
            cavity_point_count=self.cavity_point_count,
        )

    def solve(self, radial_source: object) -> SeparatedSourceDDXState:
        values, _psi, _phi, raw_state, energy, energy_gradient = self._solve_raw(
            radial_source
        )
        return self._state_from_raw(values, raw_state, energy, energy_gradient)

    def radial_jvp(self, radial_direction: object) -> np.ndarray:
        """Apply the exact fixed-geometry external-MEP model-field JVP."""

        psi, phi = self._radial_problem_data(radial_direction)
        state, _energy, _energy_gradient = self._problem.solve_general(psi, phi)
        return MACE_POLAR_RADIAL_GTO_FIELD_DUAL_SPACE.validate(
            self._problem.external_mep_model_field(state),
            atom_count=len(self._permanent_source),
            name="external-MEP model-field JVP",
        )

    def radial_vjp(self, field_cotangent: object) -> np.ndarray:
        """Apply the true transpose of the external-MEP model-field map."""

        cotangent = MACE_POLAR_RADIAL_GTO_FIELD_DUAL_SPACE.validate(
            field_cotangent,
            atom_count=len(self._permanent_source),
            name="radial field cotangent",
        )
        psi, phi = self._radial_problem_data(cotangent)
        return MACE_POLAR_RADIAL_GTO_SOURCE_SPACE.validate(
            self._problem.external_mep_model_field_vjp(psi, phi),
            atom_count=len(self._permanent_source),
            name="external-MEP model-field VJP",
        )

    def permanent_model_field_vjp(self, field_cotangent: object) -> np.ndarray:
        """Pull a model-field cotangent back to the point permanent source."""

        return ATOMIC_L1_SOURCE_SPACE.validate(
            extract_atomic_l1_first_radial_cotangent(self.radial_vjp(field_cotangent)),
            atom_count=len(self._permanent_source),
            name="permanent model-field VJP",
        )

    def _radial_electric_field(self, radial_source: object) -> np.ndarray:
        values = MACE_POLAR_RADIAL_GTO_SOURCE_SPACE.validate(
            radial_source,
            atom_count=len(self._permanent_source),
            name="radial source for electric field",
        )
        cavity = np.asarray(self._problem.model.cavity, dtype=float).T
        electric_field = np.zeros_like(cavity)
        coefficients = _radial_coefficients(values)
        for radial_index, sigma in enumerate(MACE_POLAR_RADIAL_SIGMAS_ANGSTROM):
            for center, raw in zip(
                self._problem.positions / Bohr,
                coefficients[:, radial_index],
                strict=True,
            ):
                charge, dipole_angstrom = cartesian_multipoles(raw[None, :])
                electric_field -= gaussian_multipole_potential_displacement_gradient(
                    cavity - center,
                    float(charge[0]),
                    dipole_angstrom[0] / Bohr,
                    sigma_bohr=sigma / Bohr,
                )
        return electric_field

    def _radial_phi_position_gradient(
        self, radial_source: object, adjoint: object
    ) -> np.ndarray:
        values = MACE_POLAR_RADIAL_GTO_SOURCE_SPACE.validate(
            radial_source,
            atom_count=len(self._permanent_source),
            name="radial source for position gradient",
        )
        xi = np.asarray(adjoint, dtype=float)
        expected = (int(self._problem.model.n_cav),)
        if xi.shape != expected or not np.all(np.isfinite(xi)):
            raise RuntimeError("ddX adjoint must be finite on every cavity point.")
        cavity = np.asarray(self._problem.model.cavity, dtype=float).T
        result = np.zeros_like(self._problem.positions)
        coefficients = _radial_coefficients(values)
        for radial_index, sigma in enumerate(MACE_POLAR_RADIAL_SIGMAS_ANGSTROM):
            result += gaussian_multipole_potential_position_vjp(
                cavity,
                self._problem.positions,
                coefficients[:, radial_index],
                -0.5 * xi,
                sigma_angstrom=sigma,
            )
        return result

    def _fixed_source_energy_coordinate_gradient_from_raw(
        self, radial_source: object, raw_state: object
    ) -> np.ndarray:
        radial_electric = self._radial_electric_field(radial_source)
        total_electric = self._permanent_electric_field.T + radial_electric
        cavity_term = np.asarray(
            raw_state.solvation_force_terms({"e": total_electric.T}), dtype=float
        ).T
        permanent_term = np.asarray(
            raw_state.multipole_force_terms(self._permanent_multipoles), dtype=float
        ).T
        radial_term = self._radial_phi_position_gradient(
            radial_source, getattr(raw_state, "xi", None)
        )
        result = (
            self._backend.radial_backend._dielectric_scaling
            * HARTREE_TO_EV
            * (cavity_term / Bohr + permanent_term / Bohr + radial_term)
        )
        if result.shape != self._problem.positions.shape or not np.all(
            np.isfinite(result)
        ):
            raise RuntimeError("separated ddX energy coordinate gradient is invalid.")
        return result

    def solve_with_fixed_source_energy_derivatives(
        self, radial_source: object
    ) -> tuple[SeparatedSourceDDXState, np.ndarray]:
        """Return the state and ``partial G/partial R`` at fixed source values."""

        values, _psi, _phi, raw_state, energy, energy_gradient = self._solve_raw(
            radial_source
        )
        state = self._state_from_raw(values, raw_state, energy, energy_gradient)
        gradient = self._fixed_source_energy_coordinate_gradient_from_raw(
            values, raw_state
        )
        return state, gradient.copy()

    def model_field_coordinate_vjp(
        self,
        radial_source: object,
        field_cotangent: object,
    ) -> np.ndarray:
        """Return ``partial_R <bar_u,u_model>`` at fixed source coefficients.

        The nonsymmetric external-MEP receiver is represented by the exact
        cross stationary scalar ``2 E(psi_source, phi_bar_u)``.  This avoids a
        derivative of a matrix inverse and reuses ddX's analytic force
        primitives without assuming that the model drive is the energy
        gradient of the Gaussian source.
        """

        values = MACE_POLAR_RADIAL_GTO_SOURCE_SPACE.validate(
            radial_source,
            atom_count=len(self._permanent_source),
            name="radial source for model-field coordinate VJP",
        )
        cotangent = MACE_POLAR_RADIAL_GTO_FIELD_DUAL_SPACE.validate(
            field_cotangent,
            atom_count=len(self._permanent_source),
            name="model-field coordinate cotangent",
        )
        source_psi, _source_phi = self._radial_problem_data(values)
        _cotangent_psi, cotangent_phi = self._radial_problem_data(cotangent)
        raw_state, _energy, _gradient = self._problem.solve_general(
            self._permanent_psi + source_psi, cotangent_phi
        )
        cotangent_electric = self._radial_electric_field(cotangent)
        cavity_term = np.asarray(
            raw_state.solvation_force_terms({"e": cotangent_electric.T}),
            dtype=float,
        ).T
        source_term = self._radial_phi_position_gradient(
            cotangent, getattr(raw_state, "xi", None)
        )
        result = (
            2.0
            * self._backend.radial_backend._dielectric_scaling
            * HARTREE_TO_EV
            * (cavity_term / Bohr + source_term)
        )
        if result.shape != self._problem.positions.shape or not np.all(
            np.isfinite(result)
        ):
            raise RuntimeError("separated ddX model-field coordinate VJP is invalid.")
        return result.copy()


class SeparatedSourceDDXBackend:
    """Model-agnostic point-permanent/radial-induced ddPCM/ddCOSMO backend."""

    __slots__ = (
        "_configuration_sha256",
        "_provenance_sha256",
        "_radial_backend",
        "_sealed",
    )

    provider_id = SEPARATED_SOURCE_DDX_PROVIDER_ID
    contract_id = SEPARATED_SOURCE_DDX_CONTRACT_ID
    permanent_source_space = ATOMIC_L1_SOURCE_SPACE
    radial_source_space = MACE_POLAR_RADIAL_GTO_SOURCE_SPACE
    radial_field_space = MACE_POLAR_RADIAL_GTO_FIELD_DUAL_SPACE
    capabilities = CapabilityStatus()
    linear_response = True
    model_field_is_energy_gradient = False
    source_dependent_geometry = False
    structurally_rotation_equivariant = False

    def __init__(
        self,
        symbols: Sequence[str],
        cavity_radii_angstrom: object,
        *,
        continuum_model: Literal["pcm", "cosmo"],
        dielectric: float,
        lmax: int,
        n_lebedev: int,
        solver_tolerance: float = 1.0e-12,
        eta: float = 0.1,
        n_proc: int = 1,
    ) -> None:
        radial = RadialGTODDXBackend(
            symbols,
            cavity_radii_angstrom,
            continuum_model=continuum_model,
            dielectric=dielectric,
            lmax=lmax,
            n_lebedev=n_lebedev,
            solver_tolerance=solver_tolerance,
            eta=eta,
            n_proc=n_proc,
        )
        configuration = _sha(
            {
                "provider_id": self.provider_id,
                "contract_id": self.contract_id,
                "radial_backend_configuration_sha256": radial.configuration_sha256(),
                "permanent_source_space_sha256": self.permanent_source_space.metadata_hash(),
                "radial_source_space_sha256": self.radial_source_space.metadata_hash(),
                "radial_field_space_sha256": self.radial_field_space.metadata_hash(),
                "permanent_kernel": "ddx-point-multipole-phi-and-psi",
                "induced_kernel": "finite-width-radial-gto-phi-and-multipole-psi",
                "receiver": "external-mep-phi-adjoint-not-energy-gradient",
                "energy_source_gradient": "general-source-psi-plus-phi-adjoint",
                "capabilities": "none",
            }
        )
        provenance = _sha(
            {
                "provider_id": self.provider_id,
                "configuration_sha256": configuration,
                "radial_backend_provenance_sha256": radial.provenance_sha256,
            }
        )
        object.__setattr__(self, "_radial_backend", radial)
        object.__setattr__(self, "_configuration_sha256", configuration)
        object.__setattr__(self, "_provenance_sha256", provenance)
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("SeparatedSourceDDXBackend is immutable.")
        object.__setattr__(self, name, value)

    @property
    def symbols(self) -> tuple[str, ...]:
        return self._radial_backend.symbols

    @property
    def radial_backend(self) -> RadialGTODDXBackend:
        return self._radial_backend

    @property
    def provenance_sha256(self) -> str:
        self.configuration_sha256()
        return self._provenance_sha256

    def configuration_sha256(self) -> str:
        radial = self._radial_backend.configuration_sha256()
        current = _sha(
            {
                "provider_id": self.provider_id,
                "contract_id": self.contract_id,
                "radial_backend_configuration_sha256": radial,
                "permanent_source_space_sha256": self.permanent_source_space.metadata_hash(),
                "radial_source_space_sha256": self.radial_source_space.metadata_hash(),
                "radial_field_space_sha256": self.radial_field_space.metadata_hash(),
                "permanent_kernel": "ddx-point-multipole-phi-and-psi",
                "induced_kernel": "finite-width-radial-gto-phi-and-multipole-psi",
                "receiver": "external-mep-phi-adjoint-not-energy-gradient",
                "energy_source_gradient": "general-source-psi-plus-phi-adjoint",
                "capabilities": "none",
            }
        )
        if current != self._configuration_sha256:
            raise RuntimeError("separated ddX configuration drifted.")
        return current

    def prepare(
        self, geometry: Any, permanent_source: object
    ) -> PreparedSeparatedSourceDDX:
        return PreparedSeparatedSourceDDX(self, geometry, permanent_source)


__all__ = [
    "PreparedSeparatedSourceDDX",
    "SEPARATED_SOURCE_DDX_CONTRACT_ID",
    "SEPARATED_SOURCE_DDX_PROVIDER_ID",
    "SeparatedSourceDDXBackend",
    "SeparatedSourceDDXState",
    "embed_atomic_l1_in_first_radial_channel",
    "extract_atomic_l1_first_radial_cotangent",
]
