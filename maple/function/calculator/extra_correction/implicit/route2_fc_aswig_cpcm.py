"""Fixed-cardinality amplitude-SWIG C-PCM response for Route 2 research.

The response is built directly in amplitude-charge variables.  For an
all-candidate surface with exposure amplitudes ``g`` and Gaussian Coulomb
matrix ``C = D + C_off``, it solves

``H y = -f_epsilon G v``, where ``H = D + G C_off G`` and ``q = G y``.

This avoids constructing the singular legacy diagonal ``D / F`` at a buried
node.  It is intentionally C-PCM-only: the positive-definite construction
does not authorize an implicit substitution for the nonsymmetric IEFPCM
equation.  The class has no coordinate-derivative contract yet and therefore
cannot enable public Route-2 forces.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
from ase.data import atomic_numbers as ASE_ATOMIC_NUMBERS
from ase.units import Bohr
from scipy.linalg import LinAlgError, cho_factor, cho_solve
from scipy.special import erf

from .continuum_response import (
    EXTERNAL_MEP_RESPONSE_CONTRACT_VERSION,
    SurfaceChargeState,
)
from .route2_fixed_topology_surface import (
    FixedTopologyAmplitudeSWIGSurface,
    build_fixed_topology_amplitude_swig_surface,
    load_pyscf_amplitude_swig_angular_grid,
)


def _validated_vector(
    values: np.ndarray,
    *,
    name: str,
    length: int | None = None,
) -> np.ndarray:
    vector = np.asarray(values, dtype=float)
    expected = "(n,)" if length is None else f"({length},)"
    if vector.ndim != 1 or (length is not None and vector.shape != (length,)):
        raise ValueError(f"{name} must have shape {expected}; received {vector.shape}.")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be finite.")
    return vector


def _atomic_numbers(symbols: tuple[str, ...]) -> np.ndarray:
    numbers = np.empty(len(symbols), dtype=int)
    for index, symbol in enumerate(symbols):
        try:
            number = int(ASE_ATOMIC_NUMBERS[str(symbol)])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Cannot resolve a real atomic number for {symbol!r}.") from exc
        if number <= 0:
            raise ValueError(f"Cannot resolve a real atomic number for {symbol!r}.")
        numbers[index] = number
    return numbers


def _gaussian_coulomb_off_diagonal(
    points_bohr: np.ndarray,
    exponents_bohr_inverse: np.ndarray,
) -> np.ndarray:
    """Return the off-diagonal Gaussian Coulomb kernel in atomic units."""

    points = np.asarray(points_bohr, dtype=float)
    exponents = np.asarray(exponents_bohr_inverse, dtype=float)
    displacement = points[:, None, :] - points[None, :, :]
    distance = np.linalg.norm(displacement, axis=2)
    np.fill_diagonal(distance, np.inf)
    if np.any(distance <= 1.0e-12):
        raise ValueError(
            "Fixed-topology amplitude-SWIG has coincident candidate Gaussian "
            "centres; reject this geometry rather than regularizing it."
        )
    pair_exponent = (
        exponents[:, None]
        * exponents[None, :]
        / np.sqrt(exponents[:, None] ** 2 + exponents[None, :] ** 2)
    )
    kernel = erf(pair_exponent * distance) / distance
    np.fill_diagonal(kernel, 0.0)
    if not np.all(np.isfinite(kernel)):
        raise RuntimeError("Gaussian C-PCM off-diagonal kernel is non-finite.")
    return kernel


@dataclass(frozen=True)
class AmplitudeSurfaceChargeState:
    """One C-PCM amplitude solve retained for diagnostics and VJP development."""

    surface_potential_hartree_per_e: np.ndarray
    amplitude_charge_e: np.ndarray
    physical_surface_charge_e: np.ndarray
    polarization_energy_hartree: float

    def __post_init__(self) -> None:
        potential = _validated_vector(
            self.surface_potential_hartree_per_e,
            name="surface_potential_hartree_per_e",
        )
        amplitude = _validated_vector(
            self.amplitude_charge_e,
            name="amplitude_charge_e",
            length=potential.size,
        )
        charge = _validated_vector(
            self.physical_surface_charge_e,
            name="physical_surface_charge_e",
            length=potential.size,
        )
        energy = float(self.polarization_energy_hartree)
        if not math.isfinite(energy):
            raise ValueError("polarization_energy_hartree must be finite.")
        expected = 0.5 * float(np.dot(potential, charge))
        if abs(energy - expected) > max(1.0e-12, 1.0e-10 * abs(expected)):
            raise ValueError(
                "Amplitude C-PCM energy must equal 0.5*dot(surface_potential, q)."
            )
        for name, value in (
            ("surface_potential_hartree_per_e", potential),
            ("amplitude_charge_e", amplitude),
            ("physical_surface_charge_e", charge),
        ):
            immutable = np.array(value, dtype=float, copy=True)
            immutable.setflags(write=False)
            object.__setattr__(self, name, immutable)
        object.__setattr__(self, "polarization_energy_hartree", energy)


class FixedTopologyAmplitudeSWIGCPCMResponse:
    """One immutable fixed-topology C-PCM external-MEP response.

    This experimental class implements the existing external-MEP response
    contract for energy and fixed-cavity reaction maps.  It intentionally does
    not expose ``operator_position_vjp`` or a surface-motion derivative
    contract, so force admission remains fail-closed.
    """

    contract_version = EXTERNAL_MEP_RESPONSE_CONTRACT_VERSION
    energy_response_is_reciprocal = True

    def __init__(
        self,
        symbols: Sequence[str],
        atom_positions_angstrom: np.ndarray,
        cavity_radii_angstrom: np.ndarray,
        *,
        dielectric: float,
        lebedev_order: int | None = None,
        _unit_sphere: np.ndarray | None = None,
        _switching_constant: float | None = None,
        _runtime_version: str | None = None,
    ) -> None:
        symbol_tuple = tuple(str(symbol) for symbol in symbols)
        atom_count = len(symbol_tuple)
        if atom_count == 0:
            raise ValueError("Fixed-topology amplitude-SWIG requires at least one atom.")
        positions = np.asarray(atom_positions_angstrom, dtype=float)
        if positions.shape != (atom_count, 3) or not np.all(np.isfinite(positions)):
            raise ValueError(
                "atom_positions_angstrom must be finite with shape "
                f"({atom_count}, 3)."
            )
        radii = np.asarray(cavity_radii_angstrom, dtype=float)
        if (
            radii.shape != (atom_count,)
            or not np.all(np.isfinite(radii))
            or np.any(radii <= 0.0)
        ):
            raise ValueError(
                "cavity_radii_angstrom must be finite and positive with shape "
                f"({atom_count},)."
            )
        dielectric_value = float(dielectric)
        if not math.isfinite(dielectric_value) or dielectric_value <= 1.0:
            raise ValueError("C-PCM dielectric must be finite and greater than 1.")
        injected_grid = _unit_sphere is not None or _switching_constant is not None
        if injected_grid and (_unit_sphere is None or _switching_constant is None):
            raise ValueError(
                "_unit_sphere and _switching_constant must be supplied together."
            )
        if injected_grid:
            angular_grid = np.asarray(_unit_sphere, dtype=float)
            switching_constant = float(_switching_constant)
            runtime_version = str(_runtime_version or "injected-test-grid")
            grid_order: int | None = None
        else:
            if lebedev_order is None:
                raise ValueError("lebedev_order is required without an injected grid.")
            angular_grid, switching_constant, runtime_version = (
                load_pyscf_amplitude_swig_angular_grid(lebedev_order)
            )
            grid_order = int(lebedev_order)

        surface = build_fixed_topology_amplitude_swig_surface(
            positions / Bohr,
            radii / Bohr,
            angular_grid,
            switching_constant=switching_constant,
        )
        diagonal = surface.charge_exponents_bohr_inverse * math.sqrt(
            2.0 / math.pi
        )
        off_diagonal = _gaussian_coulomb_off_diagonal(
            surface.surface_points_bohr,
            surface.charge_exponents_bohr_inverse,
        )
        amplitudes = surface.exposure_amplitudes
        hessian = np.diag(diagonal) + (
            amplitudes[:, None] * off_diagonal * amplitudes[None, :]
        )
        hessian = 0.5 * (hessian + hessian.T)
        try:
            factor = cho_factor(hessian, lower=True, check_finite=True)
        except (LinAlgError, ValueError) as exc:
            raise RuntimeError(
                "Fixed-topology amplitude-SWIG C-PCM curvature is not positive "
                "definite for this geometry."
            ) from exc

        self._atomic_numbers = _atomic_numbers(symbol_tuple)
        self._positions_angstrom = positions.copy()
        self._radii_angstrom = radii.copy()
        self._dielectric = dielectric_value
        self._f_epsilon = (dielectric_value - 1.0) / dielectric_value
        self._surface = surface
        self._hessian = hessian
        self._factor = factor
        self._runtime_version = runtime_version
        self._lebedev_order = grid_order
        self.atom_count = atom_count
        self.surface_size = surface.surface_size

    @property
    def atomic_numbers(self) -> np.ndarray:
        return self._atomic_numbers.copy()

    @property
    def reference_positions_bohr(self) -> np.ndarray:
        return self._positions_angstrom.copy() / Bohr

    @property
    def cavity_radii_angstrom(self) -> np.ndarray:
        return self._radii_angstrom.copy()

    @property
    def surface_points_bohr(self) -> np.ndarray:
        return self._surface.surface_points_bohr.copy()

    @property
    def surface_areas_bohr2(self) -> np.ndarray:
        """Return effective areas, including zero-area buried candidates."""

        return self._surface.effective_areas_bohr2.copy()

    @property
    def surface_parent_atom_indices(self) -> np.ndarray:
        return self._surface.parent_atom_indices.copy()

    @property
    def exposure_amplitudes(self) -> np.ndarray:
        return self._surface.exposure_amplitudes.copy()

    @property
    def switching_weights(self) -> np.ndarray:
        return self._surface.switching_weights

    @property
    def surface_hessian(self) -> np.ndarray:
        """Return ``H = D + G C_off G`` for structural diagnostics only."""

        return self._hessian.copy()

    @property
    def runtime_provenance(self) -> dict[str, str | int | float | None]:
        return {
            "provider": "fixed-topology-aswig",
            "profile": "cpcm-fc-aswig-v1-experimental",
            "continuum_model": "cpcm",
            "pyscf_version": self._runtime_version,
            "lebedev_order": self._lebedev_order,
            "grid_points_per_atom": self._surface.grid_points_per_atom,
            "surface_size": self.surface_size,
            "static_dielectric": self._dielectric,
            "dielectric_scaling": self._f_epsilon,
            "surface_cardinality_policy": "all-candidates-retained-v1",
            "switching_weight": "F=product(C3-amplitude-neighbor-switches)^2",
            "equation": "variational-amplitude-cpcm-v1",
            "upstream_equivalence": "new-discretization-not-pruned-swig-reparameterization",
            "force_capability": "energy-only-no-coordinate-vjp-yet",
        }

    def _validated_potential(self, values: np.ndarray) -> np.ndarray:
        return _validated_vector(
            values,
            name="surface_potential_hartree_per_e",
            length=self.surface_size,
        )

    def solve_amplitude_state(
        self,
        surface_potential_hartree_per_e: np.ndarray,
    ) -> AmplitudeSurfaceChargeState:
        """Solve the direct amplitude-charge C-PCM stationary equation."""

        potential = self._validated_potential(surface_potential_hartree_per_e)
        amplitudes = self._surface.exposure_amplitudes
        rhs = -self._f_epsilon * amplitudes * potential
        try:
            amplitude_charge = cho_solve(self._factor, rhs, check_finite=True)
        except (LinAlgError, ValueError) as exc:
            raise RuntimeError("Fixed-topology amplitude C-PCM solve failed.") from exc
        charge = amplitudes * amplitude_charge
        if not np.all(np.isfinite(charge)):
            raise RuntimeError("Fixed-topology amplitude C-PCM produced non-finite ASC.")
        return AmplitudeSurfaceChargeState(
            surface_potential_hartree_per_e=potential,
            amplitude_charge_e=amplitude_charge,
            physical_surface_charge_e=charge,
            polarization_energy_hartree=0.5 * float(np.dot(potential, charge)),
        )

    def apply_energy_conjugate(
        self,
        surface_potential_hartree_per_e: np.ndarray,
    ) -> np.ndarray:
        return self.solve_amplitude_state(
            surface_potential_hartree_per_e
        ).physical_surface_charge_e.copy()

    def solve(
        self,
        surface_potential_hartree_per_e: np.ndarray,
    ) -> SurfaceChargeState:
        state = self.solve_amplitude_state(surface_potential_hartree_per_e)
        return SurfaceChargeState(
            surface_potential_hartree_per_e=(
                state.surface_potential_hartree_per_e
            ),
            direct_surface_charge_e=state.physical_surface_charge_e,
            adjoint_surface_charge_e=state.physical_surface_charge_e,
            energy_conjugate_surface_charge_e=state.physical_surface_charge_e,
            polarization_energy_hartree=state.polarization_energy_hartree,
        )

    def reaction_field_linear_map(self, atom_positions_angstrom: np.ndarray):
        """Build an energy-only Route-2 local-jet reaction-field map."""

        from .route2_pcm_response import FixedCavityPCMReactionFieldLinearMap

        return FixedCavityPCMReactionFieldLinearMap(
            self,
            atom_positions_angstrom,
        )


__all__ = [
    "AmplitudeSurfaceChargeState",
    "FixedTopologyAmplitudeSWIGCPCMResponse",
]
