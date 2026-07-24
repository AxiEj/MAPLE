"""Fixed-surface energy-conjugate PCM response map for the Route-2 adjoint.

This module owns the linear density-to-node-field response for one already-built
cavity/operator and the explicit solute-kernel coordinate VJP with that surface
held fixed.  It is not a cavity derivative and does not make Route 2
force-capable by itself.
"""

from __future__ import annotations

import numpy as np
from ase.units import Bohr, Hartree

from .continuum_response import (
    EXTERNAL_MEP_RESPONSE_CONTRACT_VERSION,
    ExternalMEPCavityResponse,
)
from .gto_density import (
    external_field_to_density_order,
    point_asc_reaction_position_vjp,
    point_asc_reaction_potential_gradient,
    point_multipole_potential,
    point_multipole_potential_position_vjp,
)


def _validated_atom_block(
    values: np.ndarray,
    *,
    atom_count: int,
    name: str,
) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    expected_shape = (atom_count, 4)
    if array.shape != expected_shape or not np.all(np.isfinite(array)):
        raise ValueError(
            f"{name} must be finite with shape {expected_shape}; "
            f"received {array.shape}."
        )
    return array


class FixedCavityPCMReactionFieldLinearMap:
    """Apply a hermitivized fixed-cavity PCM response and its adjoint.

    Density blocks use MACE-POLAR's raw real-spherical order.  Node fields use
    ``[V, dV/dx, dV/dy, dV/dz]`` in eV/e and eV/(e Angstrom).  The continuum
    adapter must expose a reciprocal energy-conjugate surface response.  The
    current PCMSolver adapter admits only ``MATRIXSYMM=TRUE`` sessions.
    """

    reciprocal_energy_pairing = True

    def __init__(
        self,
        response: ExternalMEPCavityResponse,
        atom_positions_angstrom: np.ndarray,
        *,
        geometry_tolerance_angstrom: float = 1.0e-12,
    ):
        if geometry_tolerance_angstrom < 0.0:
            raise ValueError("Geometry tolerance cannot be negative.")
        if (
            getattr(response, "contract_version", None)
            != EXTERNAL_MEP_RESPONSE_CONTRACT_VERSION
        ):
            raise ValueError(
                "Unsupported external-MEP continuum-response contract version."
            )
        if not response.energy_response_is_reciprocal:
            raise ValueError(
                "Route-2 PCM adjoints require a reciprocal energy-conjugate "
                "surface response."
            )

        positions = np.asarray(atom_positions_angstrom, dtype=float)
        atomic_numbers = np.asarray(response.atomic_numbers, dtype=float)
        expected_shape = (atomic_numbers.size, 3)
        if positions.shape != expected_shape or not np.all(np.isfinite(positions)):
            raise ValueError(
                "Fixed-cavity atom positions must be finite with shape "
                f"{expected_shape}; received {positions.shape}."
            )
        response_positions = (
            np.asarray(response.reference_positions_bohr, dtype=float) * Bohr
        )
        if not np.allclose(
            positions,
            response_positions,
            rtol=0.0,
            atol=geometry_tolerance_angstrom,
        ):
            raise ValueError(
                "Fixed-cavity PCM response geometry does not match the open "
                "continuum-response geometry."
            )

        # Access through the response contract snapshots the fixed surface used
        # by every Krylov application.
        centers = np.asarray(response.surface_points_bohr, dtype=float)
        if centers.ndim != 2 or centers.shape[1] != 3 or not np.all(
            np.isfinite(centers)
        ):
            raise RuntimeError(
                "Continuum surface points must be finite with shape "
                "(n_surface, 3)."
            )
        self._response = response
        self._positions_angstrom = positions.copy()
        self._centers_bohr = centers.copy()
        self._surface_readback_tolerance_bohr = (
            geometry_tolerance_angstrom / Bohr
        )
        self.atom_count = positions.shape[0]

    def _compute_asc(self, density: np.ndarray) -> np.ndarray:
        mep = point_multipole_potential(
            self._centers_bohr,
            self._positions_angstrom,
            density,
        )
        asc = np.asarray(
            self._response.apply_energy_conjugate(mep),
            dtype=float,
        )
        if asc.shape != (self._centers_bohr.shape[0],) or not np.all(
            np.isfinite(asc)
        ):
            raise RuntimeError(
                "Continuum surface-charge response must be finite with one value per "
                "fixed cavity point."
            )
        return asc

    def apply(self, density_direction: np.ndarray) -> np.ndarray:
        """Map a density direction to the atom-centred reaction field."""

        density = _validated_atom_block(
            density_direction,
            atom_count=self.atom_count,
            name="density_direction",
        )
        asc = self._compute_asc(density)
        potential, gradient = point_asc_reaction_potential_gradient(
            self._positions_angstrom,
            self._centers_bohr,
            asc,
        )
        field = np.concatenate(
            (
                (potential * Hartree)[:, None],
                gradient * Hartree / Bohr,
            ),
            axis=1,
        )
        return _validated_atom_block(
            field,
            atom_count=self.atom_count,
            name="reaction-field response",
        )

    def adjoint(self, field_cotangent: np.ndarray) -> np.ndarray:
        """Apply the discrete transpose of the fixed-cavity response map.

        The MACE ``l=1`` order maps to Cartesian ``x/y/z`` as ``[2, 0, 1]``.
        With a reciprocal energy-conjugate surface response, the numeric
        transpose is therefore the same forward response bracketed by the inverse
        permutation ``[0, 2, 3, 1]``.  The Hartree and Bohr conversion factors
        cancel between the two brackets; no empirical scaling is introduced.
        """

        cotangent = _validated_atom_block(
            field_cotangent,
            atom_count=self.atom_count,
            name="field_cotangent",
        )
        density_order = external_field_to_density_order(cotangent)
        response = self.apply(density_order)
        return external_field_to_density_order(response)

    def at_solute_positions(
        self,
        atom_positions_angstrom: np.ndarray,
    ) -> "FixedCavityPCMReactionFieldLinearMap":
        """Reuse this fixed surface/operator at displaced solute positions.

        This is an explicit fixed-surface derivative helper.  It changes only
        the atom centres used by the solute-MEP and ASC-back-projection kernels;
        the continuum surface points and response operator are shared unchanged.
        It must not be used as a substitute for rebuilding the physical cavity
        at a new geometry.
        """

        positions = np.asarray(atom_positions_angstrom, dtype=float)
        expected_shape = (self.atom_count, 3)
        if positions.shape != expected_shape or not np.all(np.isfinite(positions)):
            raise ValueError(
                "Displaced fixed-surface solute positions must be finite with "
                f"shape {expected_shape}; received {positions.shape}."
            )
        current_centers = np.asarray(
            self._response.surface_points_bohr,
            dtype=float,
        )
        centers_unchanged = (
            current_centers.shape == self._centers_bohr.shape
            and np.all(np.isfinite(current_centers))
            and np.allclose(
                current_centers,
                self._centers_bohr,
                rtol=0.0,
                atol=self._surface_readback_tolerance_bohr,
            )
        )
        if not centers_unchanged:
            raise RuntimeError(
                "The open continuum surface changed after the fixed-surface "
                "response map was constructed."
            )

        displaced = object.__new__(type(self))
        displaced._response = self._response
        displaced._positions_angstrom = positions.copy()
        displaced._centers_bohr = self._centers_bohr.copy()
        displaced._surface_readback_tolerance_bohr = (
            self._surface_readback_tolerance_bohr
        )
        displaced.atom_count = self.atom_count
        return displaced

    def position_vjp(
        self,
        density: np.ndarray,
        field_cotangent: np.ndarray,
    ) -> np.ndarray:
        """Differentiate a field pairing at fixed surface and PCM operator.

        This returns
        ``d <field_cotangent, P_R(density)> / dR`` in eV/Angstrom while
        holding density coefficients, tessera centres, and the PCM surface
        response matrix fixed.  Both the solute-MEP and ASC-back-projection
        kernel positions are differentiated.  Cavity motion and operator
        derivatives are deliberately absent.
        """

        coefficients = _validated_atom_block(
            density,
            atom_count=self.atom_count,
            name="density",
        )
        cotangent = _validated_atom_block(
            field_cotangent,
            atom_count=self.atom_count,
            name="field_cotangent",
        )

        asc = self._compute_asc(coefficients)
        # The stored field gradient is eV/(e Angstrom), whereas the kernel VJP
        # pairs its gradient with dipoles in e*bohr. Convert that cotangent by
        # 1/Bohr here, then convert both Hartree/Angstrom kernel terms to
        # eV/Angstrom once after they are summed.
        reaction_projection_vjp = point_asc_reaction_position_vjp(
            self._positions_angstrom,
            self._centers_bohr,
            asc,
            cotangent[:, 0],
            cotangent[:, 1:] / Bohr,
        )

        adjoint_density = external_field_to_density_order(cotangent)
        adjoint_asc = self._compute_asc(adjoint_density)
        solute_mep_vjp = point_multipole_potential_position_vjp(
            self._centers_bohr,
            self._positions_angstrom,
            coefficients,
            adjoint_asc,
        )

        result = (reaction_projection_vjp + solute_mep_vjp) * Hartree
        expected_shape = (self.atom_count, 3)
        if result.shape != expected_shape or not np.all(np.isfinite(result)):
            raise RuntimeError(
                "Fixed-surface PCM position VJP must be finite with shape "
                f"{expected_shape}; received {result.shape}."
            )
        return result


__all__ = ["FixedCavityPCMReactionFieldLinearMap"]
