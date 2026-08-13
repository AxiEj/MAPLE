"""Disabled scalar-first wrapper for fixed-topology reciprocal radial C-PCM.

This module re-expresses the existing audited C-PCM map as the one scalar

``G(R,c) = 1/2 <c, P_R c>_Q``.

The drive, source HVP, and fixed-source coordinate partial are inherited from
:class:`ContinuumEnergyFunctional` and therefore cannot be supplied by a
second response implementation.  The frozen matrix at one geometry is the
same matrix returned by the production radial-GTO backend.  All release
capabilities remain closed.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from ase.data import atomic_numbers
from ase.units import Bohr

from maple.solvation.api.capabilities import CapabilityStatus
from maple.solvation.api.units import HARTREE_TO_EV
from maple.solvation.api.scalar_registry import (
    VARIATIONAL_MACEPOLAR_ENERGYGRADIENT_FIXEDCAVITY_CPCM_V1,
)
from .conjugate_fixed_topology_cpcm import (
    ConjugateRadialGTOFixedTopologyCPCMBackend,
)
from .functional import ContinuumEnergyFunctional

FIXED_RECIPROCAL_CPCM_FUNCTIONAL_PROVIDER_ID = (
    "maple.route2.continuum.fixed-reciprocal-radial-gto-cpcm-functional.impl.v1"
)
_FUNCTIONAL_CONTRACT_ID = "maple.route2.continuum-functional.fixed-linear-reciprocal.v1"


def _sha(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _implementation_sha256() -> tuple[tuple[str, str], ...]:
    directory = Path(__file__).resolve().parent
    return tuple(
        (name, hashlib.sha256((directory / name).read_bytes()).hexdigest())
        for name in ("functional.py", "fixed_reciprocal_functional.py")
    )


class FixedReciprocalCPCMFunctional(ContinuumEnergyFunctional):
    """One immutable fixed-cavity C-PCM scalar; no capability is admitted."""

    __slots__ = (
        "_backend",
        "_configuration_sha256",
        "_functional_provenance_sha256",
        "_runtime_device",
        "_runtime_dtype",
    )

    provider_id = FIXED_RECIPROCAL_CPCM_FUNCTIONAL_PROVIDER_ID
    scalar_id = VARIATIONAL_MACEPOLAR_ENERGYGRADIENT_FIXEDCAVITY_CPCM_V1
    functional_contract_id = _FUNCTIONAL_CONTRACT_ID
    capabilities = CapabilityStatus()
    fixed_topology = True
    source_dependent_geometry = False
    scalar_first = True
    reciprocal = True
    linear_response = True
    electrostatics_only = True
    include_nonpolar = False

    def __init__(
        self,
        backend: ConjugateRadialGTOFixedTopologyCPCMBackend,
        *,
        dtype: object,
        device: object,
    ) -> None:
        if not isinstance(backend, ConjugateRadialGTOFixedTopologyCPCMBackend):
            raise TypeError(
                "backend must be ConjugateRadialGTOFixedTopologyCPCMBackend."
            )
        if (
            backend.fixed_topology is not True
            or backend.linear_response is not True
            or backend.reciprocal is not True
            or backend.source_dependent_geometry is not False
        ):
            raise ValueError(
                "fixed reciprocal functional requires a fixed, linear, reciprocal backend."
            )
        backend_configuration = backend.configuration_sha256()
        runtime_dtype = str(dtype)
        runtime_device = str(device)
        payload = {
            "provider_id": self.provider_id,
            "scalar_id": self.scalar_id,
            "functional_contract_id": self.functional_contract_id,
            "backend_provider_id": backend.provider_id,
            "backend_scalar_id": backend.scalar_id,
            "backend_configuration_sha256": backend_configuration,
            "backend_provenance_sha256": backend.provenance_sha256,
            "source_space_sha256": backend.source_space.metadata_hash(),
            "field_space_sha256": backend.field_space.metadata_hash(),
            "pairing_sha256": backend.pairing.metadata_hash(),
            "hartree_to_ev": HARTREE_TO_EV,
            "bohr_angstrom": Bohr,
            "runtime_dtype": runtime_dtype,
            "runtime_device": runtime_device,
            "implementation_sha256": _implementation_sha256(),
            "derivatives": "sealed-autograd-from-one-half-coupling-scalar",
            "capabilities": "none",
        }
        object.__setattr__(self, "_backend", backend)
        object.__setattr__(self, "_runtime_dtype", runtime_dtype)
        object.__setattr__(self, "_runtime_device", runtime_device)
        object.__setattr__(self, "_configuration_sha256", _sha(payload))
        object.__setattr__(
            self,
            "_functional_provenance_sha256",
            _sha(
                {
                    "provider_id": self.provider_id,
                    "configuration_sha256": self._configuration_sha256,
                    "backend_provenance_sha256": backend.provenance_sha256,
                }
            ),
        )
        super().__init__(
            source_space=backend.source_space,
            field_space=backend.field_space,
            pairing=backend.pairing,
            dtype=dtype,
            device=device,
            expected_atomic_numbers=tuple(
                atomic_numbers[symbol] for symbol in backend.symbols
            ),
        )

    @property
    def backend(self) -> ConjugateRadialGTOFixedTopologyCPCMBackend:
        return self._backend

    @property
    def continuum_profile_id(self) -> str:
        return self.backend.continuum_profile_id

    @property
    def cavity_profile_id(self) -> str:
        return self.backend.cavity_profile_id

    @property
    def coupling_id(self) -> str:
        return self.backend.coupling_id

    @property
    def configuration_contract_id(self) -> str:
        return self.backend.configuration_contract_id

    @property
    def provenance_sha256(self) -> str:
        self.configuration_sha256()
        return self._functional_provenance_sha256

    def configuration_sha256(self) -> str:
        backend_configuration = self.backend.configuration_sha256()
        current = _sha(
            {
                "provider_id": self.provider_id,
                "scalar_id": self.scalar_id,
                "functional_contract_id": self.functional_contract_id,
                "backend_provider_id": self.backend.provider_id,
                "backend_scalar_id": self.backend.scalar_id,
                "backend_configuration_sha256": backend_configuration,
                "backend_provenance_sha256": self.backend.provenance_sha256,
                "source_space_sha256": self.backend.source_space.metadata_hash(),
                "field_space_sha256": self.backend.field_space.metadata_hash(),
                "pairing_sha256": self.backend.pairing.metadata_hash(),
                "hartree_to_ev": HARTREE_TO_EV,
                "bohr_angstrom": Bohr,
                "runtime_dtype": self._runtime_dtype,
                "runtime_device": self._runtime_device,
                "implementation_sha256": _implementation_sha256(),
                "derivatives": "sealed-autograd-from-one-half-coupling-scalar",
                "capabilities": "none",
            }
        )
        if current != self._configuration_sha256:
            raise RuntimeError("Fixed reciprocal continuum configuration drifted.")
        expected_provenance = _sha(
            {
                "provider_id": self.provider_id,
                "configuration_sha256": current,
                "backend_provenance_sha256": self.backend.provenance_sha256,
            }
        )
        if expected_provenance != self._functional_provenance_sha256:
            raise RuntimeError("Fixed reciprocal continuum provenance drifted.")
        return current

    @staticmethod
    def _constant(values: object, *, like: Any):
        torch = __import__("torch")
        return torch.tensor(
            np.array(values, dtype=float, copy=True),
            dtype=like.dtype,
            device=like.device,
        )

    @staticmethod
    def _amplitude_switch(clearance: Any):
        torch = __import__("torch")
        x = torch.clamp(clearance, min=0.0, max=1.0)
        polynomial = x**4 * (35.0 - x * (84.0 - x * (70.0 - 20.0 * x)))
        return torch.where(
            clearance <= 0.0,
            torch.zeros_like(clearance),
            torch.where(clearance >= 1.0, torch.ones_like(clearance), polynomial),
        )

    @staticmethod
    def _gaussian_kernels(radius: Any, sigma_bohr: float):
        torch = __import__("torch")
        nonzero = radius > 1.0e-14
        safe_radius = torch.where(nonzero, radius, torch.ones_like(radius))
        x = safe_radius / (math.sqrt(2.0) * sigma_bohr)
        exponential = torch.exp(-(x**2))
        monopole_regular = torch.special.erf(x) / safe_radius
        monopole = torch.where(
            nonzero,
            monopole_regular,
            torch.full_like(radius, math.sqrt(2.0 / math.pi) / sigma_bohr),
        )
        dipole_regular = (
            torch.special.erf(x) - (2.0 * x / math.sqrt(math.pi)) * exponential
        ) / safe_radius**3
        dipole = torch.where(nonzero, dipole_regular, torch.zeros_like(radius))
        return monopole, dipole

    def _energy_torch(self, positions: Any, source: Any):
        torch = __import__("torch")
        self.configuration_sha256()
        surface = self.backend.surface_provider
        atom_count = len(self.backend.symbols)
        angular = self._constant(surface.unit_sphere, like=positions)
        directions = angular[:, :3]
        weights_one_atom = angular[:, 3]
        grid_count = int(angular.shape[0])
        parents = torch.arange(atom_count, device=positions.device).repeat_interleave(
            grid_count
        )
        directions = directions.repeat(atom_count, 1)
        weights = (4.0 * math.pi * weights_one_atom).repeat(atom_count)
        radii = self._constant(
            self.backend.cavity_radii_angstrom / Bohr, like=positions
        )
        positions_bohr = positions / Bohr
        points = positions_bohr[parents] + radii[parents, None] * directions

        widths = radii * math.sqrt(14.0 / grid_count)
        ratio = radii / widths
        alpha = 0.5 + ratio - torch.sqrt(ratio**2 - 1.0 / 28.0)
        inner_radii = radii - alpha * widths
        displacement_to_atoms = points[:, None, :] - positions_bohr[None, :, :]
        distance_to_atoms = torch.linalg.vector_norm(displacement_to_atoms, dim=2)
        parent_mask = torch.nn.functional.one_hot(parents, num_classes=atom_count).to(
            dtype=torch.bool
        )
        clearance = (distance_to_atoms - inner_radii[None, :]) / widths[None, :]
        clearance = torch.where(parent_mask, torch.ones_like(clearance), clearance)
        amplitudes = torch.prod(self._amplitude_switch(clearance), dim=1)
        exponents = float(surface.switching_constant) / (
            radii[parents] * torch.sqrt(weights)
        )

        node_delta = points[:, None, :] - points[None, :, :]
        node_distance = torch.linalg.vector_norm(node_delta, dim=2)
        eye = torch.eye(len(parents), dtype=torch.bool, device=positions.device)
        if bool(((node_distance <= 1.0e-12) & ~eye).any().item()):
            raise ValueError(
                "Fixed-topology amplitude-SWIG has coincident candidate centres."
            )
        safe_node_distance = torch.where(
            eye, torch.ones_like(node_distance), node_distance
        )
        pair_exponent = (
            exponents[:, None]
            * exponents[None, :]
            / torch.sqrt(exponents[:, None] ** 2 + exponents[None, :] ** 2)
        )
        off_diagonal = (
            torch.special.erf(pair_exponent * safe_node_distance) / safe_node_distance
        )
        off_diagonal = torch.where(eye, torch.zeros_like(off_diagonal), off_diagonal)
        diagonal = exponents * math.sqrt(2.0 / math.pi)
        surface_hessian = (
            torch.diag(diagonal)
            + amplitudes[:, None] * off_diagonal * amplitudes[None, :]
        )
        surface_hessian = 0.5 * (surface_hessian + surface_hessian.T)

        radius = torch.linalg.vector_norm(displacement_to_atoms, dim=2)
        potential = torch.zeros_like(radius[:, 0])
        radial_layout = (
            (0, (4, 2, 3), 1.5),
            (1, (7, 5, 6), 3.0),
        )
        for charge_index, dipole_indices, sigma_angstrom in radial_layout:
            monopole_kernel, dipole_kernel = self._gaussian_kernels(
                radius, sigma_angstrom / Bohr
            )
            charge = source[:, charge_index]
            # ``dipole_indices`` is ordered as public [x,y,z] for each
            # radial block: first (4,2,3), then (7,5,6).
            dipole_bohr = source[:, dipole_indices] / Bohr
            projection = torch.einsum("sai,ai->sa", displacement_to_atoms, dipole_bohr)
            potential = potential + torch.sum(
                charge[None, :] * monopole_kernel + projection * dipole_kernel,
                dim=1,
            )

        dielectric_factor = (self.backend.dielectric - 1.0) / self.backend.dielectric
        # The production radial coupling maps source to surface MEP in eV/e.
        # Its stationary C-PCM solve first converts that MEP to Hartree/e, so
        # the scalar contains one net Hartree-to-eV factor rather than two.
        potential_ev = potential * HARTREE_TO_EV
        amplitude_charge = torch.linalg.solve(
            surface_hessian,
            -dielectric_factor * amplitudes * (potential_ev / HARTREE_TO_EV),
        )
        physical_charge = amplitudes * amplitude_charge
        return 0.5 * torch.dot(potential_ev, physical_charge)

    def runtime_provenance(self) -> tuple[tuple[str, str], ...]:
        self.configuration_sha256()
        return tuple(
            sorted(
                {
                    "provider_id": self.provider_id,
                    "scalar_id": self.scalar_id,
                    "functional_contract_id": self.functional_contract_id,
                    "configuration_sha256": self.configuration_sha256(),
                    "provenance_sha256": self.provenance_sha256,
                    "backend_provider_id": self.backend.provider_id,
                    "backend_configuration_sha256": (
                        self.backend.configuration_sha256()
                    ),
                    "derivative_route": "sealed-same-scalar-autograd",
                    "capabilities": "none-pending-strict-variational-gates",
                }.items()
            )
        )


__all__ = [
    "FIXED_RECIPROCAL_CPCM_FUNCTIONAL_PROVIDER_ID",
    "FixedReciprocalCPCMFunctional",
]
