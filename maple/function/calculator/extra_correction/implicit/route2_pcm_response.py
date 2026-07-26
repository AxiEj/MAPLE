"""Fixed-surface energy-conjugate PCM response map for the Route-2 adjoint.

This module owns the linear density-to-node-field response for one already-built
cavity/operator and the explicit solute-kernel coordinate VJP with that surface
held fixed.  It is not a cavity derivative and does not make Route 2
force-capable by itself.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ase.units import Bohr, Hartree

from .continuum_derivative import (
    EXTERNAL_MEP_OPERATOR_DERIVATIVE_CONTRACT_VERSION,
    continuum_operator_position_vjp as _continuum_operator_position_vjp,
)
from .continuum_response import (
    EXTERNAL_MEP_RESPONSE_CONTRACT_VERSION,
    ExternalMEPCavityResponse,
)
from .gto_density import (
    density_reaction_coupling,
    external_field_to_density_order,
    point_asc_reaction_position_vjp,
    point_asc_reaction_potential_gradient,
    point_multipole_potential,
    point_multipole_potential_position_vjp,
    point_multipole_potential_surface_position_vjp,
)
from .gto_field_projection import (
    ATOMIC_CENTER_MEAN_GAUGE,
    ExactGTOFieldProjector,
)
from .route2_derivative import (
    FULL_REACTION_FIELD_POSITION_DERIVATIVE_CONTRACT_VERSION,
)
from .route2_field_state import ReactionFieldDrive


ATOM_CENTERED_SURFACE_MOTION_CONTRACT_VERSION = 1


@dataclass(frozen=True)
class FixedCavityPCMSnapshot:
    """PCM energy, field, and surface state from one explicit root density."""

    density_coefficients: np.ndarray
    mep_hartree_per_e: np.ndarray
    asc_e: np.ndarray
    polarization_energy_hartree: float
    reaction_potential_hartree_per_e: np.ndarray
    reaction_gradient_hartree_per_e_bohr: np.ndarray
    density_reaction_coupling_hartree: float

    def __post_init__(self) -> None:
        for name in (
            "density_coefficients",
            "mep_hartree_per_e",
            "asc_e",
            "reaction_potential_hartree_per_e",
            "reaction_gradient_hartree_per_e_bohr",
        ):
            values = np.asarray(getattr(self, name), dtype=float)
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{name} must contain only finite values.")
            immutable = np.array(values, copy=True)
            immutable.setflags(write=False)
            object.__setattr__(self, name, immutable)
        for name in (
            "polarization_energy_hartree",
            "density_reaction_coupling_hartree",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite.")
            object.__setattr__(self, name, value)


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
        model_field_projector: ExactGTOFieldProjector | None = None,
        model_field_gauge: str = "continuum-zero-at-infinity",
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
        self._model_field_projector = model_field_projector
        self._model_field_gauge = str(model_field_gauge)
        if self._model_field_gauge not in {
            "continuum-zero-at-infinity",
            ATOMIC_CENTER_MEAN_GAUGE,
        }:
            raise ValueError(
                f"Unsupported Route-2 model-field gauge: "
                f"{self._model_field_gauge}."
            )
        if (
            self._model_field_projector is not None
            and self._model_field_gauge != ATOMIC_CENTER_MEAN_GAUGE
        ):
            raise ValueError(
                "The exact-GTO projector currently requires the "
                "atomic-center-mean-zero-v1 model-field gauge."
            )
        self._scf_snapshot: FixedCavityPCMSnapshot | None = None

    @property
    def model_field_projection_provenance(
        self,
    ) -> dict[str, object] | None:
        """Return the exact projector contract, if this map owns one."""

        projector = self._model_field_projector
        if projector is None:
            return None
        return dict(projector.spec.provenance)

    def _surface_potential(self, density: np.ndarray) -> np.ndarray:
        return point_multipole_potential(
            self._centers_bohr,
            self._positions_angstrom,
            density,
        )

    def _reaction_potential_gradient(
        self,
        asc: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        potential, gradient = point_asc_reaction_potential_gradient(
            self._positions_angstrom,
            self._centers_bohr,
            asc,
        )
        return (
            np.asarray(potential, dtype=float),
            np.asarray(gradient, dtype=float),
        )

    def _compute_asc(self, density: np.ndarray) -> np.ndarray:
        mep = self._surface_potential(density)
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

    def _required_model_field_projector(self) -> ExactGTOFieldProjector:
        projector = self._model_field_projector
        if projector is None:
            raise NotImplementedError(
                "The exact-GTO model-feature derivative requires an "
                "ExactGTOFieldProjector."
            )
        return projector

    def _solve_snapshot(
        self,
        density_coefficients: np.ndarray,
    ) -> FixedCavityPCMSnapshot:
        density = _validated_atom_block(
            density_coefficients,
            atom_count=self.atom_count,
            name="density_coefficients",
        )
        mep = self._surface_potential(density)
        solved = self._response.solve(mep)
        solved_mep = np.asarray(
            solved.surface_potential_hartree_per_e,
            dtype=float,
        )
        if solved_mep.shape != mep.shape or not np.array_equal(solved_mep, mep):
            raise RuntimeError(
                "Continuum response returned a state for a different surface "
                "potential."
            )
        asc = np.asarray(
            solved.energy_conjugate_surface_charge_e,
            dtype=float,
        )
        polarization_energy = float(solved.polarization_energy_hartree)
        surface_coupling = float(np.dot(mep, asc))
        expected_coupling = 2.0 * polarization_energy
        tolerance = max(1.0e-10, 1.0e-8 * abs(expected_coupling))
        if abs(surface_coupling - expected_coupling) > tolerance:
            raise RuntimeError(
                "Continuum polarization-energy convention check failed: "
                "E_pol must equal 0.5*dot(MEP,q_energy)."
            )

        reaction_potential, reaction_gradient = (
            self._reaction_potential_gradient(asc)
        )
        multipole_coupling = density_reaction_coupling(
            density,
            reaction_potential,
            reaction_gradient,
        )
        if abs(multipole_coupling - surface_coupling) > max(
            1.0e-10, 1.0e-8 * abs(surface_coupling)
        ):
            raise RuntimeError(
                "MACE-POLAR source/ASC reciprocity check failed; the "
                "reaction-field projection is inconsistent with the cavity MEP."
            )
        return FixedCavityPCMSnapshot(
            density_coefficients=density,
            mep_hartree_per_e=mep,
            asc_e=asc,
            polarization_energy_hartree=polarization_energy,
            reaction_potential_hartree_per_e=reaction_potential,
            reaction_gradient_hartree_per_e_bohr=reaction_gradient,
            density_reaction_coupling_hartree=surface_coupling,
        )

    def scf_snapshot(
        self,
        density_coefficients: np.ndarray,
    ) -> FixedCavityPCMSnapshot:
        """Return the cached PCM state for exactly one SCF root density."""

        density = _validated_atom_block(
            density_coefficients,
            atom_count=self.atom_count,
            name="density_coefficients",
        )
        cached = self._scf_snapshot
        if cached is not None and np.array_equal(
            cached.density_coefficients,
            density,
        ):
            return cached
        snapshot = self._solve_snapshot(density)
        self._scf_snapshot = snapshot
        return snapshot

    @staticmethod
    def _field_from_snapshot(
        snapshot: FixedCavityPCMSnapshot,
    ) -> np.ndarray:
        return np.concatenate(
            (
                (
                    snapshot.reaction_potential_hartree_per_e * Hartree
                )[:, None],
                (
                    snapshot.reaction_gradient_hartree_per_e_bohr
                    * Hartree
                    / Bohr
                ),
            ),
            axis=1,
        )

    def apply_scf(self, density_direction: np.ndarray) -> np.ndarray:
        """Map one SCF root and retain its exact continuum energy state."""

        snapshot = self.scf_snapshot(density_direction)
        return _validated_atom_block(
            self._field_from_snapshot(snapshot),
            atom_count=self.atom_count,
            name="reaction-field response",
        )

    def apply_scf_drive(
        self,
        density_direction: np.ndarray,
    ) -> ReactionFieldDrive:
        """Return one same-root energy field and optional exact model features."""

        snapshot = self.scf_snapshot(density_direction)
        field = _validated_atom_block(
            self._field_from_snapshot(snapshot),
            atom_count=self.atom_count,
            name="reaction-field response",
        )
        projector = self._model_field_projector
        if self._model_field_gauge == "continuum-zero-at-infinity":
            gauge_reference_ev = 0.0
        else:
            gauge_reference_ev = float(np.mean(field[:, 0]))
        if projector is None:
            return ReactionFieldDrive.local_jet(
                field,
                model_field_gauge=self._model_field_gauge,
                model_field_gauge_reference_ev=gauge_reference_ev,
            )
        features, gauge_reference_ev = projector.project_asc_with_gauge(
            self._positions_angstrom,
            self._centers_bohr,
            snapshot.asc_e,
        )
        expected_reference_ev = float(np.mean(field[:, 0]))
        if abs(gauge_reference_ev - expected_reference_ev) > 1.0e-12:
            raise RuntimeError(
                "The exact-GTO and local-jet projectors disagree on the "
                "atomic-center mean model-field gauge."
            )
        return ReactionFieldDrive(
            density_dual_field_ev=field,
            model_local_field_ev=None,
            model_field_features=features,
            projector="exact-gto-v1",
            model_field_gauge=self._model_field_gauge,
            model_field_gauge_reference_ev=gauge_reference_ev,
        )

    def scf_polarization_energy_hartree(
        self,
        density_coefficients: np.ndarray,
    ) -> float:
        """Return the polarization energy from the same cached SCF root."""

        return self.scf_snapshot(
            density_coefficients
        ).polarization_energy_hartree

    def apply(self, density_direction: np.ndarray) -> np.ndarray:
        """Map a density direction to the atom-centred reaction field."""

        density = _validated_atom_block(
            density_direction,
            atom_count=self.atom_count,
            name="density_direction",
        )
        asc = self._compute_asc(density)
        potential, gradient = self._reaction_potential_gradient(asc)
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

    def model_feature_jvp(
        self,
        density_direction: np.ndarray,
    ) -> np.ndarray:
        """Map a density direction to exact-GTO model features."""

        density = _validated_atom_block(
            density_direction,
            atom_count=self.atom_count,
            name="density_direction",
        )
        projector = self._required_model_field_projector()
        features = np.asarray(
            projector.project_asc(
                self._positions_angstrom,
                self._centers_bohr,
                self._compute_asc(density),
            ),
            dtype=float,
        )
        expected_shape = (self.atom_count, projector.feature_count)
        if features.shape != expected_shape or not np.all(np.isfinite(features)):
            raise RuntimeError(
                "Continuum exact-GTO feature JVP must be finite with shape "
                f"{expected_shape}; received {features.shape}."
            )
        return features

    def model_feature_vjp(
        self,
        feature_cotangent: np.ndarray,
    ) -> np.ndarray:
        """Apply the exact-GTO-feature-to-density continuum transpose."""

        projector = self._required_model_field_projector()
        cotangent = np.asarray(feature_cotangent, dtype=float)
        expected_shape = (self.atom_count, projector.feature_count)
        if cotangent.shape != expected_shape or not np.all(
            np.isfinite(cotangent)
        ):
            raise ValueError(
                "feature_cotangent must be finite with shape "
                f"{expected_shape}; received {cotangent.shape}."
            )
        asc_cotangent = projector.project_asc_adjoint(
            self._positions_angstrom,
            self._centers_bohr,
            cotangent,
        )
        surface_cotangent = np.asarray(
            self._response.apply_energy_conjugate(asc_cotangent),
            dtype=float,
        )
        expected_surface_shape = (self._centers_bohr.shape[0],)
        if surface_cotangent.shape != expected_surface_shape or not np.all(
            np.isfinite(surface_cotangent)
        ):
            raise RuntimeError(
                "Continuum exact-GTO surface adjoint must be finite with "
                f"shape {expected_surface_shape}; received "
                f"{surface_cotangent.shape}."
            )
        potential, gradient = self._reaction_potential_gradient(
            surface_cotangent
        )
        density_cotangent = external_field_to_density_order(
            np.concatenate(
                (
                    potential[:, None],
                    gradient / Bohr,
                ),
                axis=1,
            )
        )
        return _validated_atom_block(
            density_cotangent,
            atom_count=self.atom_count,
            name="exact-GTO density cotangent",
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
        displaced._model_field_projector = self._model_field_projector
        displaced._model_field_gauge = self._model_field_gauge
        displaced._scf_snapshot = None
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
        adjoint_density = external_field_to_density_order(cotangent)
        adjoint_asc = self._compute_asc(adjoint_density)
        return self._fixed_surface_position_vjp_from_asc(
            coefficients,
            cotangent,
            asc,
            adjoint_asc,
        )

    def _fixed_surface_position_vjp_from_asc(
        self,
        coefficients: np.ndarray,
        cotangent: np.ndarray,
        asc: np.ndarray,
        adjoint_asc: np.ndarray,
    ) -> np.ndarray:
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

    def continuum_operator_position_vjp(
        self,
        density: np.ndarray,
        field_cotangent: np.ndarray,
    ) -> np.ndarray:
        """Differentiate only the continuum operator in a field pairing.

        This returns the ``dQ_sym/dR`` contribution to
        ``d<field_cotangent, P_R(density)>/dR`` in eV/Angstrom.  Surface MEP and
        reaction-field projection kernels are held fixed; their solute-centre
        derivative is provided by :meth:`position_vjp`, while moving-surface
        kernel terms remain a separate future contribution.
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
        right_surface_potential = point_multipole_potential(
            self._centers_bohr,
            self._positions_angstrom,
            coefficients,
        )
        left_surface_potential = point_multipole_potential(
            self._centers_bohr,
            self._positions_angstrom,
            external_field_to_density_order(cotangent),
        )
        result = (
            _continuum_operator_position_vjp(
                self._response,
                left_surface_potential,
                right_surface_potential,
            )
            * Hartree
        )
        expected_shape = (self.atom_count, 3)
        if result.shape != expected_shape or not np.all(np.isfinite(result)):
            raise RuntimeError(
                "Continuum operator position VJP must be finite with shape "
                f"{expected_shape}; received {result.shape}."
            )
        return result


class AtomCenteredSurfacePCMReactionFieldLinearMap(
    FixedCavityPCMReactionFieldLinearMap
):
    """Reaction map with an exact atom-centred surface-coordinate VJP.

    The continuum response must own the exact operator derivative used by its
    energy and expose one parent atom for every surface point.  Surface nodes are
    assumed to translate rigidly with that parent at fixed angular quadrature;
    switching weights, areas, and response matrices remain the backend's
    operator-derivative responsibility.
    """

    full_position_derivative_contract_version = (
        FULL_REACTION_FIELD_POSITION_DERIVATIVE_CONTRACT_VERSION
    )

    def __init__(
        self,
        response: ExternalMEPCavityResponse,
        atom_positions_angstrom: np.ndarray,
        *,
        geometry_tolerance_angstrom: float = 1.0e-12,
    ):
        super().__init__(
            response,
            atom_positions_angstrom,
            geometry_tolerance_angstrom=geometry_tolerance_angstrom,
        )
        motion_version = getattr(
            response,
            "surface_motion_contract_version",
            None,
        )
        if motion_version is None:
            raise NotImplementedError(
                "The continuum backend does not provide an atom-centered "
                "surface-motion derivative."
            )
        if motion_version != ATOM_CENTERED_SURFACE_MOTION_CONTRACT_VERSION:
            raise ValueError(
                "Unsupported atom-centered surface-motion contract version."
            )
        operator_version = getattr(
            response,
            "operator_derivative_contract_version",
            None,
        )
        if operator_version is None:
            raise NotImplementedError(
                "The continuum backend does not provide an operator derivative."
            )
        if operator_version != (
            EXTERNAL_MEP_OPERATOR_DERIVATIVE_CONTRACT_VERSION
        ):
            raise ValueError(
                "Unsupported external-MEP operator-derivative contract version."
            )

        parents = np.asarray(
            getattr(response, "surface_parent_atom_indices", None)
        )
        expected_shape = (self._centers_bohr.shape[0],)
        valid_parent_indices = (
            parents.shape == expected_shape
            and np.issubdtype(parents.dtype, np.integer)
            and np.all(parents >= 0)
            and np.all(parents < self.atom_count)
        )
        if not valid_parent_indices:
            raise ValueError(
                "Surface parent atom indices must be integers with shape "
                f"{expected_shape} and values in [0, {self.atom_count})."
            )
        self._surface_parent_atom_indices = np.array(
            parents,
            dtype=int,
            copy=True,
        )

    def full_position_vjp(
        self,
        density: np.ndarray,
        field_cotangent: np.ndarray,
    ) -> np.ndarray:
        """Differentiate the complete same-provider reaction-field pairing."""

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
        adjoint_density = external_field_to_density_order(cotangent)
        asc = self._compute_asc(coefficients)
        adjoint_asc = self._compute_asc(adjoint_density)

        fixed_surface = self._fixed_surface_position_vjp_from_asc(
            coefficients,
            cotangent,
            asc,
            adjoint_asc,
        )
        surface_point_vjp = (
            point_multipole_potential_surface_position_vjp(
                self._centers_bohr,
                self._positions_angstrom,
                adjoint_density,
                asc,
            )
            + point_multipole_potential_surface_position_vjp(
                self._centers_bohr,
                self._positions_angstrom,
                coefficients,
                adjoint_asc,
            )
        )
        moving_surface = np.zeros((self.atom_count, 3), dtype=float)
        np.add.at(
            moving_surface,
            self._surface_parent_atom_indices,
            surface_point_vjp,
        )
        moving_surface *= Hartree / Bohr

        continuum_operator = self.continuum_operator_position_vjp(
            coefficients,
            cotangent,
        )
        result = fixed_surface + moving_surface + continuum_operator
        expected_shape = (self.atom_count, 3)
        if result.shape != expected_shape or not np.all(np.isfinite(result)):
            raise RuntimeError(
                "Full reaction-field position VJP must be finite with shape "
                f"{expected_shape}; received {result.shape}."
            )
        return result


__all__ = [
    "ATOM_CENTERED_SURFACE_MOTION_CONTRACT_VERSION",
    "AtomCenteredSurfacePCMReactionFieldLinearMap",
    "FixedCavityPCMSnapshot",
    "FixedCavityPCMReactionFieldLinearMap",
]
