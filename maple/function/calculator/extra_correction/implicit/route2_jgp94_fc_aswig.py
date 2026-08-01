"""JGP94 body-frame wrapper for fixed-topology amplitude-SWIG C-PCM.

The all-candidate amplitude-SWIG discretization removes the discontinuous
surface-node deletion of the legacy pyddx/PySCF paths.  Its finite angular grid
is nevertheless tied to the laboratory axes, so it retains a small, purely
numerical rigid-rotation error.  This module composes the existing
fixed-topology C-PCM operator with the nondegenerate Johnson--Gill--Pople
principal-axis frame from :mod:`route2_body_frame`.

The composition is deliberately narrow:

* it only wraps the fixed-topology amplitude C-PCM response;
* it keeps the existing point-``l<=1`` source and local-jet receiver;
* it supplies the complete coordinate VJP of the *same* discrete reaction map;
* it fails closed when the nuclear-charge moment tensor is near degenerate.

It does not claim a common variational MACE--PCM electronic free energy, a
complete SMD implementation, or public forces.  Those are separate Route-2
admission gates.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
from ase.data import atomic_numbers
from ase.units import Bohr

from .gto_density import (
    density_to_external_field_order,
    external_field_to_density_order,
)
from .route2_body_frame import (
    JGP94Frame,
    body_dipoles,
    build_jgp94_frame,
    jgp94_frame_vjp,
)
from .route2_fc_aswig_cpcm import FixedTopologyAmplitudeSWIGCPCMResponse
from .route2_fc_aswig_smd_cds import (
    FixedTopologyAqueousSMDCDS,
    FixedTopologyAqueousSMDCDSResult,
)
from .route2_field_state import ReactionFieldDrive
from .route2_pcm_response import (
    FULL_REACTION_FIELD_POSITION_DERIVATIVE_CONTRACT_VERSION,
)


JGP94_FC_ASWIG_PROFILE = "jgp94-cpcm-fc-aswig-v1-experimental"
"""Stable identity for this nondegenerate, fixed-topology research profile."""

JGP94_DEFAULT_MINIMUM_RELATIVE_EIGENGAP = 0.05
"""Predeclared JGP94 local-chart conditioning guard, not a fitted parameter."""


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


def _validated_positions(
    values: np.ndarray,
    *,
    atom_count: int,
    name: str,
) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    expected_shape = (atom_count, 3)
    if array.shape != expected_shape or not np.all(np.isfinite(array)):
        raise ValueError(
            f"{name} must be finite with shape {expected_shape}; "
            f"received {array.shape}."
        )
    return array


def _nuclear_charges_from_symbols(symbols: Sequence[str]) -> np.ndarray:
    values: list[float] = []
    for symbol in symbols:
        atomic_number = int(atomic_numbers.get(str(symbol), 0))
        if atomic_number <= 0:
            raise ValueError(f"Unsupported element symbol for JGP94 frame: {symbol!r}.")
        values.append(float(atomic_number))
    return np.asarray(values, dtype=float)


@dataclass(frozen=True)
class JGP94BodyFrameReactionFieldMap:
    """Lab-frame reaction map backed by one body-frame fixed-topology map.

    Let ``O`` map lab row vectors to body row vectors.  The wrapper implements

    ``c_b = S(O)c_lab``, ``f_lab = S(O)^T P_b c_b``.

    The full coordinate VJP contains the body continuum VJP, the rotation of
    input dipoles, and the rotation of the returned reaction-field gradients.
    All three are reduced through the reusable JGP94 eigenvector VJP.
    """

    atom_count: int
    frame: JGP94Frame
    nuclear_charges: np.ndarray
    body_map: object
    runtime_provenance: dict[str, object]

    reciprocal_energy_pairing: bool = True
    full_position_derivative_contract_version: int = (
        FULL_REACTION_FIELD_POSITION_DERIVATIVE_CONTRACT_VERSION
    )

    def __post_init__(self) -> None:
        if self.atom_count <= 0:
            raise ValueError("JGP94 body-frame map requires at least one atom.")
        charges = np.asarray(self.nuclear_charges, dtype=float)
        if (
            charges.shape != (self.atom_count,)
            or not np.all(np.isfinite(charges))
            or np.any(charges <= 0.0)
        ):
            raise ValueError(
                "nuclear_charges must be finite and positive with one value per atom."
            )
        object.__setattr__(self, "nuclear_charges", np.array(charges, copy=True))
        if int(getattr(self.body_map, "atom_count", -1)) != self.atom_count:
            raise ValueError("The body reaction map does not match the frame atom count.")
        if not bool(getattr(self.body_map, "reciprocal_energy_pairing", False)):
            raise ValueError("The body reaction map must be energy reciprocal.")
        if getattr(
            self.body_map,
            "full_position_derivative_contract_version",
            None,
        ) != FULL_REACTION_FIELD_POSITION_DERIVATIVE_CONTRACT_VERSION:
            raise ValueError(
                "The body reaction map must provide the supported full coordinate VJP."
            )
        object.__setattr__(self, "runtime_provenance", dict(self.runtime_provenance))

    @property
    def orientation(self) -> np.ndarray:
        """Return a defensive copy of the lab-to-body proper rotation."""

        return np.array(self.frame.orientation, copy=True)

    def _body_density(self, density_lab: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        density = _validated_atom_block(
            density_lab,
            atom_count=self.atom_count,
            name="density_direction",
        )
        external = density_to_external_field_order(density)
        body_external = np.array(external, copy=True)
        body_external[:, 1:] = external[:, 1:] @ self.frame.orientation
        return external_field_to_density_order(body_external), external[:, 1:]

    def _body_field(self, field_lab: np.ndarray) -> np.ndarray:
        field = _validated_atom_block(
            field_lab,
            atom_count=self.atom_count,
            name="field_cotangent",
        )
        body = np.array(field, copy=True)
        body[:, 1:] = field[:, 1:] @ self.frame.orientation
        return body

    def _lab_field(self, field_body: np.ndarray) -> np.ndarray:
        field = _validated_atom_block(
            field_body,
            atom_count=self.atom_count,
            name="body reaction field",
        )
        lab = np.array(field, copy=True)
        lab[:, 1:] = field[:, 1:] @ self.frame.orientation.T
        return lab

    def _lab_density_cotangent(self, cotangent_body: np.ndarray) -> np.ndarray:
        raw = _validated_atom_block(
            cotangent_body,
            atom_count=self.atom_count,
            name="body density cotangent",
        )
        external = density_to_external_field_order(raw)
        lab_external = np.array(external, copy=True)
        lab_external[:, 1:] = external[:, 1:] @ self.frame.orientation.T
        return external_field_to_density_order(lab_external)

    def apply(self, density_direction: np.ndarray) -> np.ndarray:
        """Map a lab-frame density direction to a lab-frame reaction field."""

        body_density, _ = self._body_density(density_direction)
        return self._lab_field(self.body_map.apply(body_density))

    def apply_scf(self, density_direction: np.ndarray) -> np.ndarray:
        """Map an SCF density and retain the same body continuum state."""

        body_density, _ = self._body_density(density_direction)
        return self._lab_field(self.body_map.apply_scf(body_density))

    def apply_scf_drive(self, density_direction: np.ndarray) -> ReactionFieldDrive:
        """Expose the same lab-frame local-jet drive to MACE-POLAR."""

        return ReactionFieldDrive.local_jet(self.apply_scf(density_direction))

    def scf_polarization_energy_hartree(
        self,
        density_coefficients: np.ndarray,
    ) -> float:
        """Return the same C-PCM half-coupling energy in the body frame."""

        body_density, _ = self._body_density(density_coefficients)
        energy = float(
            self.body_map.scf_polarization_energy_hartree(body_density)
        )
        if not math.isfinite(energy):
            raise RuntimeError("JGP94 body-frame C-PCM energy is non-finite.")
        return energy

    def adjoint(self, field_cotangent: np.ndarray) -> np.ndarray:
        """Apply the exact lab-frame transpose of :meth:`apply`."""

        body_field = self._body_field(field_cotangent)
        return self._lab_density_cotangent(self.body_map.adjoint(body_field))

    def full_position_vjp(
        self,
        density: np.ndarray,
        field_cotangent: np.ndarray,
    ) -> np.ndarray:
        """Differentiate ``<field_cotangent, P_R density>`` in lab coordinates.

        The inner body map differentiates its source, moving surface, and
        amplitude-C-PCM operator at fixed body coordinates.  The outer JGP94
        VJP adds the exact derivative of ``R -> (Y(R), O(R))``.  In particular,
        the returned field-gradient rotation has an explicit orientation
        cotangent, so it is not silently omitted from the force.
        """

        body_density, lab_dipoles = self._body_density(density)
        lab_field_cotangent = _validated_atom_block(
            field_cotangent,
            atom_count=self.atom_count,
            name="field_cotangent",
        )
        body_field_cotangent = self._body_field(lab_field_cotangent)

        body_position_cotangent = self.body_map.full_position_vjp(
            body_density,
            body_field_cotangent,
        )
        body_density_cotangent = self.body_map.adjoint(body_field_cotangent)
        body_density_cotangent_external = density_to_external_field_order(
            body_density_cotangent
        )
        body_field = self.body_map.apply(body_density)

        # f_b is the derivative with respect to the body cotangent.  Because
        # w_b = w_lab @ O for the three gradient columns, this contracts the
        # otherwise easy-to-miss output-frame contribution to dO.
        output_orientation_cotangent = (
            lab_field_cotangent[:, 1:].T @ body_field[:, 1:]
        )
        frame_vjp = jgp94_frame_vjp(
            self.frame,
            self.nuclear_charges,
            lab_dipoles,
            body_position_cotangent,
            body_density_cotangent_external[:, 1:],
            additional_orientation_cotangent=output_orientation_cotangent,
        )
        result = _validated_positions(
            frame_vjp.position_cotangent,
            atom_count=self.atom_count,
            name="JGP94 full reaction-field position VJP",
        )
        return result


class JGP94BodyFrameFixedTopologyAmplitudeSWIGCPCMResponse:
    """Build a nondegenerate JGP94-wrapped fixed-topology C-PCM map.

    A new instance must be constructed for each molecular geometry.  This is
    intentional: both the all-candidate surface and the JGP94 frame are part
    of the single scalar that is differentiated.  The wrapper accepts an
    optional local signed-axis reference for coordinate scans; if absent, the
    proper eigensystem's sign variants are harmless for this centrally
    symmetric angular grid and are nevertheless covered by explicit tests.
    """

    def __init__(
        self,
        symbols: Sequence[str],
        atom_positions_angstrom: np.ndarray,
        cavity_radii_angstrom: np.ndarray,
        *,
        dielectric: float,
        nuclear_charges: np.ndarray | None = None,
        minimum_relative_eigengap: float = JGP94_DEFAULT_MINIMUM_RELATIVE_EIGENGAP,
        reference_orientation: np.ndarray | None = None,
        minimum_reference_axis_overlap: float | None = None,
        lebedev_order: int | None = None,
        _unit_sphere: np.ndarray | None = None,
        _switching_constant: float | None = None,
        _runtime_version: str | None = None,
    ) -> None:
        self._symbols = tuple(str(symbol) for symbol in symbols)
        self.atom_count = len(self._symbols)
        if self.atom_count == 0:
            raise ValueError("JGP94 body-frame C-PCM requires at least one atom.")
        self._positions_angstrom = _validated_positions(
            atom_positions_angstrom,
            atom_count=self.atom_count,
            name="atom_positions_angstrom",
        ).copy()
        radii = np.asarray(cavity_radii_angstrom, dtype=float)
        if (
            radii.shape != (self.atom_count,)
            or not np.all(np.isfinite(radii))
            or np.any(radii <= 0.0)
        ):
            raise ValueError(
                "cavity_radii_angstrom must be finite and positive with one value "
                "per atom."
            )
        self._radii_angstrom = np.array(radii, copy=True)
        if nuclear_charges is None:
            charges = _nuclear_charges_from_symbols(self._symbols)
        else:
            charges = np.asarray(nuclear_charges, dtype=float)
        if (
            charges.shape != (self.atom_count,)
            or not np.all(np.isfinite(charges))
            or np.any(charges <= 0.0)
        ):
            raise ValueError(
                "nuclear_charges must be finite and positive with one value per atom."
            )
        self._nuclear_charges = np.array(charges, copy=True)
        self._minimum_relative_eigengap = float(minimum_relative_eigengap)
        self._frame = build_jgp94_frame(
            self._positions_angstrom,
            self._nuclear_charges,
            minimum_relative_eigengap=self._minimum_relative_eigengap,
            reference_orientation=reference_orientation,
            minimum_reference_axis_overlap=minimum_reference_axis_overlap,
        )
        self._body_response = FixedTopologyAmplitudeSWIGCPCMResponse(
            self._symbols,
            self._frame.body_positions,
            self._radii_angstrom,
            dielectric=dielectric,
            lebedev_order=lebedev_order,
            _unit_sphere=_unit_sphere,
            _switching_constant=_switching_constant,
            _runtime_version=_runtime_version,
        )

    @property
    def frame(self) -> JGP94Frame:
        """Return the checked local standard frame for diagnostics."""

        return self._frame

    @property
    def atomic_numbers(self) -> np.ndarray:
        return self._body_response.atomic_numbers

    @property
    def reference_positions_bohr(self) -> np.ndarray:
        return self._positions_angstrom.copy() / Bohr

    @property
    def cavity_radii_angstrom(self) -> np.ndarray:
        return self._radii_angstrom.copy()

    @property
    def surface_points_bohr(self) -> np.ndarray:
        body_points = self._body_response.surface_points_bohr
        lab_points = (
            self._frame.center[None, :]
            + body_points * Bohr @ self._frame.orientation.T
        )
        return lab_points / Bohr

    @property
    def surface_areas_bohr2(self) -> np.ndarray:
        return self._body_response.surface_areas_bohr2

    @property
    def surface_parent_atom_indices(self) -> np.ndarray:
        return self._body_response.surface_parent_atom_indices

    @property
    def runtime_provenance(self) -> dict[str, object]:
        provenance = dict(self._body_response.runtime_provenance)
        provenance.update(
            {
                "provider": "jgp94-body-frame-fixed-topology-aswig",
                "profile": JGP94_FC_ASWIG_PROFILE,
                "laboratory_grid_rotation_remedy": "jgp94-nuclear-charge-principal-frame-v1",
                "minimum_relative_eigengap": self._frame.relative_minimum_eigengap,
                "minimum_relative_eigengap_guard": self._minimum_relative_eigengap,
                "frame_orthogonality_error": self._frame.orthogonality_error,
                "frame_determinant_error": self._frame.determinant_error,
                "force_capability": "same-energy-full-continuum-vjp-experimental",
                "public_force_status": "closed-pending-total-pes-admission",
            }
        )
        return provenance

    def reaction_field_linear_map(
        self,
        atom_positions_angstrom: np.ndarray,
    ) -> JGP94BodyFrameReactionFieldMap:
        """Return the exact lab-frame map for this immutable geometry."""

        positions = _validated_positions(
            atom_positions_angstrom,
            atom_count=self.atom_count,
            name="atom_positions_angstrom",
        )
        if not np.array_equal(positions, self._positions_angstrom):
            raise ValueError(
                "JGP94 body-frame response geometry does not match its fixed "
                "continuum geometry."
            )
        body_map = self._body_response.reaction_field_linear_map(
            self._frame.body_positions
        )
        return JGP94BodyFrameReactionFieldMap(
            atom_count=self.atom_count,
            frame=self._frame,
            nuclear_charges=self._nuclear_charges,
            body_map=body_map,
            runtime_provenance=self.runtime_provenance,
        )


class JGP94BodyFrameFixedTopologyAqueousSMDCDS:
    """Rotate the fixed-topology aqueous SMD-CDS scalar with the same frame.

    The electrostatic and CDS discretizations deliberately retain their own
    physical radii, just as conventional SMD does.  They do, however, share
    the *same* JGP94 frame object so their angular grids cannot introduce
    inconsistent lab-frame rotation errors into a total-force check.
    """

    def __init__(
        self,
        symbols: Sequence[str],
        frame: JGP94Frame,
        nuclear_charges: np.ndarray,
        *,
        lebedev_order: int | None = None,
        _unit_sphere: np.ndarray | None = None,
        _switching_constant: float | None = None,
        _runtime_version: str | None = None,
    ) -> None:
        self._symbols = tuple(str(symbol) for symbol in symbols)
        self.atom_count = len(self._symbols)
        if self.atom_count == 0:
            raise ValueError("JGP94 body-frame SMD-CDS requires at least one atom.")
        if frame.body_positions.shape != (self.atom_count, 3):
            raise ValueError("The JGP94 frame does not match the CDS atom count.")
        charges = np.asarray(nuclear_charges, dtype=float)
        if (
            charges.shape != (self.atom_count,)
            or not np.all(np.isfinite(charges))
            or np.any(charges <= 0.0)
        ):
            raise ValueError(
                "nuclear_charges must be finite and positive with one value per atom."
            )
        self._frame = frame
        self._nuclear_charges = np.array(charges, copy=True)
        self._body_cds = FixedTopologyAqueousSMDCDS(
            self._symbols,
            frame.body_positions,
            lebedev_order=lebedev_order,
            _unit_sphere=_unit_sphere,
            _switching_constant=_switching_constant,
            _runtime_version=_runtime_version,
        )
        self._result: FixedTopologyAqueousSMDCDSResult | None = None

    @property
    def runtime_provenance(self) -> dict[str, object]:
        provenance = dict(self._body_cds.runtime_provenance)
        provenance.update(
            {
                "provider": "jgp94-body-frame-fixed-topology-aqueous-smd-cds",
                "profile": "jgp94-aqueous-smd-cds-fixed-topology-c3-area-v1-experimental",
                "laboratory_grid_rotation_remedy": "jgp94-nuclear-charge-principal-frame-v1",
                "frame_relative_minimum_eigengap": (
                    self._frame.relative_minimum_eigengap
                ),
            }
        )
        return provenance

    def result(self) -> FixedTopologyAqueousSMDCDSResult:
        """Return the lab-frame gradient of the same body-frame CDS scalar."""

        cached = self._result
        if cached is not None:
            return cached
        body = self._body_cds.result()
        zeros = np.zeros((self.atom_count, 3), dtype=float)
        frame_vjp = jgp94_frame_vjp(
            self._frame,
            self._nuclear_charges,
            zeros,
            body.position_gradient_hartree_per_angstrom,
            zeros,
        )
        result = FixedTopologyAqueousSMDCDSResult(
            energy_hartree=body.energy_hartree,
            energy_kcal_mol=body.energy_kcal_mol,
            atom_areas_angstrom2=body.atom_areas_angstrom2,
            atom_tensions_cal_mol_angstrom2=(
                body.atom_tensions_cal_mol_angstrom2
            ),
            position_gradient_hartree_per_angstrom=(
                frame_vjp.position_cotangent
            ),
            grid_points_per_atom=body.grid_points_per_atom,
            surface_size=body.surface_size,
        )
        self._result = result
        return result


__all__ = [
    "JGP94_DEFAULT_MINIMUM_RELATIVE_EIGENGAP",
    "JGP94_FC_ASWIG_PROFILE",
    "JGP94BodyFrameFixedTopologyAmplitudeSWIGCPCMResponse",
    "JGP94BodyFrameFixedTopologyAqueousSMDCDS",
    "JGP94BodyFrameReactionFieldMap",
]
