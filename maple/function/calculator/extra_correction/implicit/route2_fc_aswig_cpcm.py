"""Fixed-cardinality amplitude-SWIG C-PCM response for Route 2 research.

The response is built directly in amplitude-charge variables.  For an
all-candidate surface with exposure amplitudes ``g`` and Gaussian Coulomb
matrix ``C = D + C_off``, it solves

``H y = -f_epsilon G v``, where ``H = D + G C_off G`` and ``q = G y``.

This avoids constructing the singular legacy diagonal ``D / F`` at a buried
node.  It is intentionally C-PCM-only: the positive-definite construction
does not authorize an implicit substitution for the nonsymmetric IEFPCM
equation.  It provides a same-scalar operator coordinate VJP for research
validation, but remains ineligible for public Route-2 forces until it is
integrated with a same-geometry CDS term and passes the complete rotation,
PES, root-uniqueness, and thermodynamic-semantics gates.
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
from .continuum_derivative import (
    EXTERNAL_MEP_OPERATOR_DERIVATIVE_CONTRACT_VERSION,
)
from .route2_fixed_topology_surface import (
    FixedTopologyAmplitudeSWIGSurface,
    build_fixed_topology_amplitude_swig_surface,
    load_pyscf_amplitude_swig_angular_grid,
)
from .route2_pcm_response import ATOM_CENTERED_SURFACE_MOTION_CONTRACT_VERSION


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


def _gaussian_coulomb_off_diagonal_point_vjp(
    points_bohr: np.ndarray,
    exponents_bohr_inverse: np.ndarray,
    left_amplitude_charge_e: np.ndarray,
    right_amplitude_charge_e: np.ndarray,
) -> np.ndarray:
    """Return ``d(left.T C_off right)/d(surface points)`` in hartree/bohr.

    ``C_off`` is the symmetric Gaussian Coulomb matrix used verbatim in the
    amplitude C-PCM curvature.  This is a bilinear VJP rather than a dense
    third-rank derivative.  It is intentionally separate from the exposure
    derivative, whose coordinates enter through ``G`` rather than ``C_off``.
    """

    points = np.asarray(points_bohr, dtype=float)
    exponents = np.asarray(exponents_bohr_inverse, dtype=float)
    left = _validated_vector(
        left_amplitude_charge_e,
        name="left_amplitude_charge_e",
        length=len(points),
    )
    right = _validated_vector(
        right_amplitude_charge_e,
        name="right_amplitude_charge_e",
        length=len(points),
    )
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
    radial_derivative = (
        (2.0 * pair_exponent / math.sqrt(math.pi))
        * np.exp(-(pair_exponent * distance) ** 2)
        / distance
        - erf(pair_exponent * distance) / distance**2
    )
    np.fill_diagonal(radial_derivative, 0.0)
    radial_over_distance = radial_derivative / distance
    np.fill_diagonal(radial_over_distance, 0.0)
    pair_weight = (
        left[:, None] * right[None, :]
        + right[:, None] * left[None, :]
    ) * radial_over_distance
    result = np.einsum("ij,ijk->ik", pair_weight, displacement)
    if not np.all(np.isfinite(result)):
        raise RuntimeError(
            "Fixed-topology amplitude C-PCM point-kernel VJP is non-finite."
        )
    return result


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

    This experimental class implements one common C-PCM scalar for the energy,
    reaction map, and ``operator_position_vjp``.  Its all-candidate surface
    gives a fixed continuum dimension at a fixed atom count.  This mathematical
    component is still not a public Route-2 force provider: the total model
    lacks a same-geometry CDS implementation and has not passed the broader
    PES/rotation/root/semantic admission gates.
    """

    contract_version = EXTERNAL_MEP_RESPONSE_CONTRACT_VERSION
    energy_response_is_reciprocal = True
    operator_derivative_contract_version = (
        EXTERNAL_MEP_OPERATOR_DERIVATIVE_CONTRACT_VERSION
    )
    surface_motion_contract_version = ATOM_CENTERED_SURFACE_MOTION_CONTRACT_VERSION

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
        self._off_diagonal = np.array(off_diagonal, dtype=float, copy=True)
        self._hessian = hessian
        self._factor = factor
        self._runtime_version = runtime_version
        self._lebedev_order = grid_order
        self.atom_count = atom_count
        self.surface_size = surface.surface_size
        self._off_diagonal.setflags(write=False)

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
            "force_capability": "same-energy-full-continuum-vjp-experimental",
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

    def operator_position_vjp(
        self,
        left_surface_potential_hartree_per_e: np.ndarray,
        right_surface_potential_hartree_per_e: np.ndarray,
    ) -> np.ndarray:
        """Differentiate the exact amplitude-CPCM response operator.

        With ``Q=-f G H^-1 G`` and the stationary amplitude states

        ``H y_a=-f G a`` and ``H y_b=-f G b``, this evaluates

        ``d(a.T Q b) = y_b.T dG a + y_a.T dG b + y_a.T dH y_b / f``.

        Surface-potential values are held fixed in the candidate-node basis.
        Thus this method owns precisely the exposure and Gaussian-kernel
        terms of the same scalar C-PCM discretization; source and reaction
        kernel motion are supplied by ``AtomCenteredSurfacePCMReactionFieldLinearMap``.
        """

        left = self._validated_potential(left_surface_potential_hartree_per_e)
        right = self._validated_potential(right_surface_potential_hartree_per_e)
        left_state = self.solve_amplitude_state(left)
        right_state = self.solve_amplitude_state(right)
        y_left = left_state.amplitude_charge_e
        y_right = right_state.amplitude_charge_e
        amplitudes = self._surface.exposure_amplitudes
        left_weighted = amplitudes * y_left
        right_weighted = amplitudes * y_right

        # H = D + G C_off G.  Collect all dG terms before asking the surface
        # object to differentiate the compact amplitude products.
        amplitude_cotangent = (
            y_left * (self._off_diagonal @ right_weighted)
            + y_right * (self._off_diagonal @ left_weighted)
        ) / self._f_epsilon
        amplitude_cotangent += y_left * right + y_right * left
        amplitude_gradient_bohr = self._surface.exposure_amplitude_position_vjp(
            amplitude_cotangent
        )

        # The remaining dH term is G dC_off G.  Each candidate point moves
        # rigidly with its parent atom; exposure motion has already been
        # included above.
        point_gradient_bohr = (
            _gaussian_coulomb_off_diagonal_point_vjp(
                self._surface.surface_points_bohr,
                self._surface.charge_exponents_bohr_inverse,
                left_weighted,
                right_weighted,
            )
            / self._f_epsilon
        )
        np.add.at(
            amplitude_gradient_bohr,
            self._surface.parent_atom_indices,
            point_gradient_bohr,
        )
        result = amplitude_gradient_bohr / Bohr
        expected_shape = (self.atom_count, 3)
        if result.shape != expected_shape or not np.all(np.isfinite(result)):
            raise RuntimeError(
                "Fixed-topology amplitude C-PCM operator VJP must be finite "
                f"with shape {expected_shape}; received {result.shape}."
            )
        return result

    def reaction_field_linear_map(self, atom_positions_angstrom: np.ndarray):
        """Build a same-energy local-jet reaction-field map with full VJP."""

        from .route2_pcm_response import (
            AtomCenteredSurfacePCMReactionFieldLinearMap,
        )

        return AtomCenteredSurfacePCMReactionFieldLinearMap(
            self,
            atom_positions_angstrom,
        )


__all__ = [
    "AmplitudeSurfaceChargeState",
    "FixedTopologyAmplitudeSWIGCPCMResponse",
]
