"""D2-symmetrised JGP94 canonical frame for the Route-2 MACE boundary.

The official MACE-POLAR-1-M molecular real-space evaluator contains a
finite-difference electrostatic-feature block with fixed laboratory ``x/y/z``
displacements.  Consequently, despite the equivariant MACE trunk, its
real-space density/energy map is not exactly SO(3) covariant at finite
displacement.  A continuum-only body-frame wrapper cannot repair that error:
the learned map itself must receive equivalent molecules in equivalent
coordinates.

This module defines a narrow, no-training canonicalisation layer.  For a
nondegenerate JGP94 principal-axis frame ``O`` it evaluates all proper
diagonal sign variants

``O @ diag(1, 1, 1)``, ``O @ diag(1, -1, -1)``,
``O @ diag(-1, 1, -1)``, and ``O @ diag(-1, -1, 1)``.

Their equal-weight average is the Reynolds projection over the residual D2
sign gauge of a proper principal-axis frame.  It therefore removes both the
laboratory-axis artefact and the otherwise arbitrary eigenvector signs without
selecting an atom-index-dependent sign convention.  The construction changes
neither checkpoint weights nor any solvent/experimental parameter.  It does
define a distinct, explicitly versioned *evaluation operator*, so callers
must record it and must not identify it with the upstream raw real-space
operator.

The implementation is deliberately pure NumPy/ASE plumbing.  Torch/MACE
autograd remains owned by ``_macepol_calculator``; this module only supplies
the linear coordinate, density, field, and reverse-VJP transformations needed
to compose that autograd with the JGP94 chart.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any, Protocol, Sequence

import numpy as np

from .gto_density import (
    density_to_external_field_order,
    external_field_to_density_order,
)
from .route2_body_frame import JGP94Frame, build_jgp94_frame, jgp94_frame_vjp


JGP94_D2_CANONICAL_MACE_FRAME_POLICY = "jgp94-d2-canonical-v1"
"""Versioned MACE spatial-evaluation policy for the fixed-topology profile."""

JGP94_D2_CANONICAL_MINIMUM_RELATIVE_EIGENGAP = 0.05
"""Predeclared nondegenerate-chart guard shared with the JGP94 continuum."""

JGP94_D2_SIGN_MATRICES: tuple[np.ndarray, ...] = tuple(
    np.diag(values).astype(float)
    for values in (
        (1.0, 1.0, 1.0),
        (1.0, -1.0, -1.0),
        (-1.0, 1.0, -1.0),
        (-1.0, -1.0, 1.0),
    )
)
"""The four proper diagonal sign matrices of the principal-axis D2 gauge."""


def _validated_positions(
    values: np.ndarray,
    *,
    atom_count: int,
    name: str,
) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if (
        array.shape != (atom_count, 3)
        or not np.all(np.isfinite(array))
    ):
        raise ValueError(
            f"{name} must be finite with shape {(atom_count, 3)}; "
            f"received {array.shape}."
        )
    return array


def _validated_atom_block(
    values: np.ndarray,
    *,
    atom_count: int,
    name: str,
) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if (
        array.shape != (atom_count, 4)
        or not np.all(np.isfinite(array))
    ):
        raise ValueError(
            f"{name} must be finite with shape {(atom_count, 4)}; "
            f"received {array.shape}."
        )
    return array


def _validated_potential_gradient(
    potential: np.ndarray,
    gradient: np.ndarray,
    *,
    atom_count: int,
    potential_name: str,
    gradient_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    potential_array = np.asarray(potential, dtype=float)
    gradient_array = _validated_positions(
        gradient,
        atom_count=atom_count,
        name=gradient_name,
    )
    if (
        potential_array.shape != (atom_count,)
        or not np.all(np.isfinite(potential_array))
    ):
        raise ValueError(
            f"{potential_name} must be finite with shape {(atom_count,)}; "
            f"received {potential_array.shape}."
        )
    return potential_array, gradient_array


@dataclass(frozen=True)
class JGP94D2MACEBranch:
    """One proper-sign branch of a nondegenerate principal-axis chart."""

    index: int
    sign_matrix: np.ndarray
    frame: JGP94Frame

    def __post_init__(self) -> None:
        if self.index < 0 or self.index >= len(JGP94_D2_SIGN_MATRICES):
            raise ValueError("JGP94 D2 branch index is outside the four-branch orbit.")
        sign = np.asarray(self.sign_matrix, dtype=float)
        if sign.shape != (3, 3) or not np.all(np.isfinite(sign)):
            raise ValueError("JGP94 D2 sign matrix must be finite with shape (3, 3).")
        if abs(float(np.linalg.det(sign)) - 1.0) > 1.0e-14:
            raise ValueError("JGP94 D2 sign matrix must be a proper rotation.")
        object.__setattr__(self, "sign_matrix", np.array(sign, copy=True))


@dataclass(frozen=True)
class JGP94D2CanonicalMACEContext:
    """One four-branch, fixed-geometry MACE frame context.

    All transformations use row-vector Cartesian conventions.  Raw
    MACE-POLAR ``l=1`` coefficients are converted through the canonical
    density/field permutation helpers before any Cartesian rotation.
    """

    atom_count: int
    nuclear_charges: np.ndarray
    base_frame: JGP94Frame
    branches: tuple[JGP94D2MACEBranch, ...]
    policy: str = JGP94_D2_CANONICAL_MACE_FRAME_POLICY

    def __post_init__(self) -> None:
        if self.atom_count <= 0:
            raise ValueError("JGP94 D2 MACE context requires at least one atom.")
        charges = np.asarray(self.nuclear_charges, dtype=float)
        if (
            charges.shape != (self.atom_count,)
            or not np.all(np.isfinite(charges))
            or np.any(charges <= 0.0)
        ):
            raise ValueError(
                "nuclear_charges must be finite and positive with one value per atom."
            )
        if len(self.branches) != len(JGP94_D2_SIGN_MATRICES):
            raise ValueError("JGP94 D2 MACE context requires all four proper sign branches.")
        if self.policy != JGP94_D2_CANONICAL_MACE_FRAME_POLICY:
            raise ValueError("Unsupported JGP94 D2 MACE frame policy.")
        object.__setattr__(self, "nuclear_charges", np.array(charges, copy=True))

    @classmethod
    def from_atoms(
        cls,
        atoms: Any,
        *,
        minimum_relative_eigengap: float,
    ) -> "JGP94D2CanonicalMACEContext":
        positions = np.asarray(atoms.get_positions(), dtype=float)
        nuclear_charges = np.asarray(atoms.numbers, dtype=float)
        atom_count = int(nuclear_charges.shape[0])
        _validated_positions(
            positions,
            atom_count=atom_count,
            name="atom_positions_angstrom",
        )
        if (
            nuclear_charges.shape != (atom_count,)
            or not np.all(np.isfinite(nuclear_charges))
            or np.any(nuclear_charges <= 0.0)
        ):
            raise ValueError(
                "atoms must provide finite positive nuclear charges for JGP94."
            )
        base = build_jgp94_frame(
            positions,
            nuclear_charges,
            minimum_relative_eigengap=minimum_relative_eigengap,
        )
        branches: list[JGP94D2MACEBranch] = []
        for index, sign in enumerate(JGP94_D2_SIGN_MATRICES):
            orientation = base.orientation @ sign
            branch_frame = replace(
                base,
                orientation=orientation,
                body_positions=base.centered_positions @ orientation,
                reference_axis_overlaps=None,
            )
            branches.append(
                JGP94D2MACEBranch(
                    index=index,
                    sign_matrix=sign,
                    frame=branch_frame,
                )
            )
        return cls(
            atom_count=atom_count,
            nuclear_charges=nuclear_charges,
            base_frame=base,
            branches=tuple(branches),
        )

    @property
    def branch_weight(self) -> float:
        return 1.0 / float(len(self.branches))

    @property
    def provenance(self) -> dict[str, object]:
        return {
            "mace_geometry_frame_policy": self.policy,
            "mace_geometry_frame_branch_count": len(self.branches),
            "mace_geometry_frame_group": "proper-principal-axis-D2",
            "mace_geometry_frame_minimum_relative_eigengap": (
                self.base_frame.relative_minimum_eigengap
            ),
            "mace_geometry_frame_laboratory_axis_remedy": (
                "equal-weight-D2-Reynolds-projection"
            ),
        }

    def body_atoms(self, atoms: Any, branch: JGP94D2MACEBranch) -> Any:
        """Return a detached atoms object in one canonical branch frame."""

        body = atoms.copy()
        body.set_positions(branch.frame.body_positions)
        return body

    def field_to_body(
        self,
        potential_lab: np.ndarray,
        gradient_lab: np.ndarray,
        branch: JGP94D2MACEBranch,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Rotate atom-centred local-field gradients into one body frame."""

        potential, gradient = _validated_potential_gradient(
            potential_lab,
            gradient_lab,
            atom_count=self.atom_count,
            potential_name="node_potential_ev",
            gradient_name="node_gradient_ev_per_angstrom",
        )
        return potential.copy(), gradient @ branch.frame.orientation

    def field_block_to_body(
        self,
        field_lab: np.ndarray,
        branch: JGP94D2MACEBranch,
    ) -> np.ndarray:
        """Rotate one external-order ``[V,gx,gy,gz]`` block into body axes."""

        field = _validated_atom_block(
            field_lab,
            atom_count=self.atom_count,
            name="external_field",
        )
        result = np.array(field, copy=True)
        result[:, 1:] = field[:, 1:] @ branch.frame.orientation
        return result

    def field_block_from_body(
        self,
        field_body: np.ndarray,
        branch: JGP94D2MACEBranch,
    ) -> np.ndarray:
        """Rotate one body-frame external field/cotangent back to lab axes."""

        field = _validated_atom_block(
            field_body,
            atom_count=self.atom_count,
            name="body_external_field",
        )
        result = np.array(field, copy=True)
        result[:, 1:] = field[:, 1:] @ branch.frame.orientation.T
        return result

    def density_from_body(
        self,
        density_body: np.ndarray,
        branch: JGP94D2MACEBranch,
    ) -> np.ndarray:
        """Rotate raw MACE ``l<=1`` density coefficients body -> lab."""

        external = density_to_external_field_order(
            _validated_atom_block(
                density_body,
                atom_count=self.atom_count,
                name="body_density_coefficients",
            )
        )
        lab_external = np.array(external, copy=True)
        lab_external[:, 1:] = external[:, 1:] @ branch.frame.orientation.T
        return external_field_to_density_order(lab_external)

    def density_cotangent_to_body(
        self,
        density_cotangent_lab: np.ndarray,
        branch: JGP94D2MACEBranch,
    ) -> np.ndarray:
        """Apply the transpose body<-lab map to a raw density cotangent."""

        external = density_to_external_field_order(
            _validated_atom_block(
                density_cotangent_lab,
                atom_count=self.atom_count,
                name="density_cotangent",
            )
        )
        body_external = np.array(external, copy=True)
        body_external[:, 1:] = external[:, 1:] @ branch.frame.orientation
        return external_field_to_density_order(body_external)

    def dipole_from_body(
        self,
        dipole_body: np.ndarray,
        branch: JGP94D2MACEBranch,
    ) -> np.ndarray:
        dipole = np.asarray(dipole_body, dtype=float)
        if dipole.shape != (3,) or not np.all(np.isfinite(dipole)):
            raise ValueError("body molecular dipole must be finite with shape (3,).")
        return dipole @ branch.frame.orientation.T

    def output_density_orientation_cotangent(
        self,
        density_cotangent_lab: np.ndarray,
        density_body: np.ndarray,
    ) -> np.ndarray:
        """Return ``d <bar_c_lab,c_lab> / dO`` for one branch.

        ``c_lab[:,1:] = c_body[:,1:] @ O.T`` after the raw real-spherical
        permutation.  The scalar ``l=0`` component has no orientation term.
        """

        cotangent_external = density_to_external_field_order(
            _validated_atom_block(
                density_cotangent_lab,
                atom_count=self.atom_count,
                name="density_cotangent",
            )
        )
        body_external = density_to_external_field_order(
            _validated_atom_block(
                density_body,
                atom_count=self.atom_count,
                name="body_density_coefficients",
            )
        )
        return cotangent_external[:, 1:].T @ body_external[:, 1:]

    def reduce_position_vjp(
        self,
        branch: JGP94D2MACEBranch,
        *,
        body_position_cotangent: np.ndarray,
        gradient_lab: np.ndarray,
        body_gradient_cotangent: np.ndarray,
        additional_orientation_cotangent: np.ndarray | None = None,
    ) -> np.ndarray:
        """Pull one branch's body-coordinate scalar VJP back to lab geometry."""

        position_bar = _validated_positions(
            body_position_cotangent,
            atom_count=self.atom_count,
            name="body_position_cotangent",
        )
        gradient = _validated_positions(
            gradient_lab,
            atom_count=self.atom_count,
            name="node_gradient_ev_per_angstrom",
        )
        gradient_bar = _validated_positions(
            body_gradient_cotangent,
            atom_count=self.atom_count,
            name="body_gradient_cotangent",
        )
        result = jgp94_frame_vjp(
            branch.frame,
            self.nuclear_charges,
            gradient,
            position_bar,
            gradient_bar,
            additional_orientation_cotangent=additional_orientation_cotangent,
        ).position_cotangent
        return _validated_positions(
            result,
            atom_count=self.atom_count,
            name="JGP94 D2 position VJP",
        )


class _DensityResponseLike(Protocol):
    """Small structural protocol for MACE's native local-jet response."""

    def jvp(self, field_direction: np.ndarray) -> np.ndarray: ...

    def vjp(self, density_cotangent: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class JGP94D2CanonicalDensityResponse:
    """Exact fixed-geometry JVP/VJP of the D2-averaged density response."""

    context: JGP94D2CanonicalMACEContext
    branch_responses: tuple[_DensityResponseLike, ...]

    def __post_init__(self) -> None:
        if len(self.branch_responses) != len(self.context.branches):
            raise ValueError("One native density response is required per D2 branch.")

    @property
    def atom_count(self) -> int:
        return self.context.atom_count

    def jvp(self, field_direction: np.ndarray) -> np.ndarray:
        direction = _validated_atom_block(
            field_direction,
            atom_count=self.atom_count,
            name="field_direction",
        )
        result = np.zeros((self.atom_count, 4), dtype=float)
        for branch, response in zip(
            self.context.branches,
            self.branch_responses,
            strict=True,
        ):
            body_direction = self.context.field_block_to_body(direction, branch)
            body_density = response.jvp(body_direction)
            result += self.context.branch_weight * self.context.density_from_body(
                body_density,
                branch,
            )
        return _validated_atom_block(
            result,
            atom_count=self.atom_count,
            name="D2 canonical density JVP",
        )

    def vjp(self, density_cotangent: np.ndarray) -> np.ndarray:
        cotangent = _validated_atom_block(
            density_cotangent,
            atom_count=self.atom_count,
            name="density_cotangent",
        )
        result = np.zeros((self.atom_count, 4), dtype=float)
        for branch, response in zip(
            self.context.branches,
            self.branch_responses,
            strict=True,
        ):
            body_cotangent = self.context.density_cotangent_to_body(
                cotangent,
                branch,
            )
            body_field = response.vjp(body_cotangent)
            result += self.context.branch_weight * self.context.field_block_from_body(
                body_field,
                branch,
            )
        return _validated_atom_block(
            result,
            atom_count=self.atom_count,
            name="D2 canonical density VJP",
        )


def average_finite_scalars(values: Sequence[float], *, name: str) -> float:
    """Return an equal-weight finite branch average with a precise failure."""

    array = np.asarray(values, dtype=float)
    if array.shape != (len(JGP94_D2_SIGN_MATRICES),) or not np.all(
        np.isfinite(array)
    ):
        raise ValueError(
            f"{name} must contain four finite JGP94 D2 branch values."
        )
    result = float(np.mean(array))
    if not math.isfinite(result):
        raise RuntimeError(f"{name} D2 branch average is non-finite.")
    return result


__all__ = [
    "JGP94_D2_CANONICAL_MINIMUM_RELATIVE_EIGENGAP",
    "JGP94_D2_CANONICAL_MACE_FRAME_POLICY",
    "JGP94_D2_SIGN_MATRICES",
    "JGP94D2CanonicalDensityResponse",
    "JGP94D2CanonicalMACEContext",
    "JGP94D2MACEBranch",
    "average_finite_scalars",
]
