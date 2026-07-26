"""Analytic point-ASC projection onto MACE-POLAR receiver GTO features.

The continuum apparent surface charges are integrated point charges.  Their
Coulomb potential can therefore be convolved analytically with each spherical
Gaussian receiver.  For ``l=1``, integration by parts turns the receiver
integral into the gradient of the same Gaussian-smoothed potential.  The final
normalization and real-spherical ordering are taken from the immutable
upstream MACE-POLAR projector matrix rather than reimplemented here.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
from ase.units import Bohr, Hartree

from .electrostatic_pairing import (
    MACE_POLAR_MODEL_FEATURE_FIELD_INDICES,
)
from .gto_density import (
    asc_reaction_potential_gradient,
    point_asc_reaction_potential_gradient,
)


ATOMIC_CENTER_MEAN_GAUGE = "atomic-center-mean-zero-v1"
EXACT_GTO_GRAPH_LONGRANGE_VERSION = "0.4.0"
EXACT_GTO_FEATURE_LAYOUT = (
    "graph-longrange-0.4.0-l0-radial-first-l1-radial-cartesian-v1"
)


def _matrix_sha256(values: np.ndarray) -> str:
    """Return a platform-stable fingerprint of one float64 matrix."""

    matrix = np.ascontiguousarray(values, dtype="<f8")
    shape = np.ascontiguousarray(matrix.shape, dtype="<i8")
    digest = hashlib.sha256()
    digest.update(shape.tobytes())
    digest.update(matrix.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class MACEPolarGTOFieldProjectionSpec:
    """Checkpoint-derived receiver basis needed for exact field projection."""

    receiver_sigmas_angstrom: tuple[float, ...]
    receiver_max_l: int
    receiver_normalization: str
    upstream_matrix: np.ndarray
    graph_longrange_version: str = EXACT_GTO_GRAPH_LONGRANGE_VERSION
    feature_layout: str = EXACT_GTO_FEATURE_LAYOUT
    contract_version: int = 1

    def __post_init__(self) -> None:
        sigmas = tuple(float(value) for value in self.receiver_sigmas_angstrom)
        if not sigmas or any(
            not np.isfinite(value) or value <= 0.0 for value in sigmas
        ):
            raise ValueError(
                "Receiver GTO widths must be finite positive values."
            )
        if self.receiver_max_l != 1:
            raise ValueError(
                "Exact Route-2 GTO projection currently requires receiver "
                "max_l=1."
            )
        if self.receiver_normalization != "receiver":
            raise ValueError(
                "Exact Route-2 GTO projection requires upstream "
                "normalization='receiver'."
            )
        if self.graph_longrange_version != EXACT_GTO_GRAPH_LONGRANGE_VERSION:
            raise ValueError(
                "Exact Route-2 GTO projection is validated only against "
                f"graph-longrange {EXACT_GTO_GRAPH_LONGRANGE_VERSION}; "
                f"received {self.graph_longrange_version}."
            )
        if self.feature_layout != EXACT_GTO_FEATURE_LAYOUT:
            raise ValueError(
                "Unsupported exact GTO receiver-feature layout."
            )
        if self.contract_version != 1:
            raise ValueError(
                "Unsupported exact GTO field-projection contract version."
            )

        matrix = np.asarray(self.upstream_matrix, dtype=float)
        expected_shape = (
            len(sigmas) * (self.receiver_max_l + 1) ** 2,
            4,
        )
        if matrix.shape != expected_shape or not np.all(np.isfinite(matrix)):
            raise ValueError(
                "Upstream GTO projector matrix shape must be "
                f"{expected_shape}; received {matrix.shape}."
            )
        immutable = np.array(matrix, copy=True)
        immutable.setflags(write=False)
        object.__setattr__(self, "receiver_sigmas_angstrom", sigmas)
        object.__setattr__(self, "upstream_matrix", immutable)

    @property
    def upstream_matrix_sha256(self) -> str:
        """Fingerprint the live checkpoint projector used by this contract."""

        return _matrix_sha256(self.upstream_matrix)

    @property
    def provenance(self) -> dict[str, object]:
        """Return the complete versioned ordering/matrix contract."""

        return {
            "contract_version": self.contract_version,
            "graph_longrange_version": self.graph_longrange_version,
            "feature_layout": self.feature_layout,
            "receiver_sigmas_angstrom": list(
                self.receiver_sigmas_angstrom
            ),
            "receiver_max_l": self.receiver_max_l,
            "receiver_normalization": self.receiver_normalization,
            "upstream_matrix_shape": list(self.upstream_matrix.shape),
            "upstream_matrix_sha256": self.upstream_matrix_sha256,
        }


@dataclass(frozen=True)
class ExactGTOFieldProjector:
    """Project an ASC into the checkpoint's complete receiver feature tensor."""

    spec: MACEPolarGTOFieldProjectionSpec

    @property
    def feature_count(self) -> int:
        return int(self.spec.upstream_matrix.shape[0])

    def _feature_sigma_indices(self) -> np.ndarray:
        sigma_count = len(self.spec.receiver_sigmas_angstrom)
        # Upstream orders all l=0 radial channels first, followed by one
        # three-component l=1 block per radial channel.
        return np.concatenate(
            (
                np.arange(sigma_count, dtype=int),
                np.repeat(np.arange(sigma_count, dtype=int), 3),
            )
        )

    def smoothed_reaction_fields_hartree(
        self,
        atom_positions_angstrom: np.ndarray,
        surface_centers_bohr: np.ndarray,
        apparent_surface_charges: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return one analytic Gaussian-smoothed ``[V,grad V]`` per width.

        Potentials have units Hartree/e and gradients Hartree/(e bohr).
        Array shapes are ``(n_sigma,n_atoms)`` and
        ``(n_sigma,n_atoms,3)``.
        """

        potentials = []
        gradients = []
        for sigma in self.spec.receiver_sigmas_angstrom:
            potential, gradient = asc_reaction_potential_gradient(
                atom_positions_angstrom,
                surface_centers_bohr,
                apparent_surface_charges,
                sigma_angstrom=sigma,
            )
            potentials.append(potential)
            gradients.append(gradient)
        return np.stack(potentials), np.stack(gradients)

    def project_smoothed_fields(
        self,
        potentials_ev: np.ndarray,
        gradients_ev_per_angstrom: np.ndarray,
        *,
        scalar_potential_gauge_reference_ev: float = 0.0,
    ) -> np.ndarray:
        """Map per-width smoothed fields to model-native receiver features."""

        potentials = np.asarray(potentials_ev, dtype=float)
        gradients = np.asarray(gradients_ev_per_angstrom, dtype=float)
        sigma_count = len(self.spec.receiver_sigmas_angstrom)
        if (
            potentials.ndim != 2
            or potentials.shape[0] != sigma_count
            or not np.all(np.isfinite(potentials))
        ):
            raise ValueError(
                "Smoothed potentials must be finite with shape "
                "(n_receiver_sigmas, n_atoms)."
            )
        expected_gradient_shape = (*potentials.shape, 3)
        if (
            gradients.shape != expected_gradient_shape
            or not np.all(np.isfinite(gradients))
        ):
            raise ValueError(
                "Smoothed gradients must be finite with shape "
                "(n_receiver_sigmas, n_atoms, 3)."
            )
        gauge_reference = float(scalar_potential_gauge_reference_ev)
        if not np.isfinite(gauge_reference):
            raise ValueError(
                "The scalar-potential gauge reference must be finite."
            )

        external_order = np.concatenate(
            ((potentials - gauge_reference)[..., None], gradients),
            axis=2,
        )
        # Match graph_longrange.GTOInternalFieldtoFeaturesBlock exactly.
        upstream_input_order = external_order[
            ...,
            list(MACE_POLAR_MODEL_FEATURE_FIELD_INDICES),
        ]
        feature_fields = upstream_input_order[
            self._feature_sigma_indices(),
            :,
            :,
        ]
        return np.einsum(
            "pf,pnf->np",
            self.spec.upstream_matrix,
            feature_fields,
        )

    @staticmethod
    def atomic_center_mean_gauge_reference_hartree(
        atom_positions_angstrom: np.ndarray,
        surface_centers_bohr: np.ndarray,
        apparent_surface_charges: np.ndarray,
    ) -> float:
        """Return the mean point-ASC potential at the atomic centres.

        The atom centres are protected by the continuum cavity radii.  This
        avoids evaluating the point-ASC potential at an arithmetic barycentre
        that can approach a tessera in a non-convex or hollow molecule.
        """

        positions = np.asarray(atom_positions_angstrom, dtype=float)
        if (
            positions.ndim != 2
            or positions.shape[1] != 3
            or positions.shape[0] == 0
            or not np.all(np.isfinite(positions))
        ):
            raise ValueError(
                "Atom positions must be finite with shape (n_atoms, 3)."
            )
        potential, _ = point_asc_reaction_potential_gradient(
            positions,
            surface_centers_bohr,
            apparent_surface_charges,
        )
        return float(np.mean(potential))

    def project_asc_with_gauge(
        self,
        atom_positions_angstrom: np.ndarray,
        surface_centers_bohr: np.ndarray,
        apparent_surface_charges: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """Return model features and the common subtracted gauge in eV."""

        potentials, gradients = self.smoothed_reaction_fields_hartree(
            atom_positions_angstrom,
            surface_centers_bohr,
            apparent_surface_charges,
        )
        gauge_reference_ev = (
            self.atomic_center_mean_gauge_reference_hartree(
                atom_positions_angstrom,
                surface_centers_bohr,
                apparent_surface_charges,
            )
            * Hartree
        )
        return (
            self.project_smoothed_fields(
                potentials * Hartree,
                gradients * Hartree / Bohr,
                scalar_potential_gauge_reference_ev=gauge_reference_ev,
            ),
            gauge_reference_ev,
        )

    def project_asc(
        self,
        atom_positions_angstrom: np.ndarray,
        surface_centers_bohr: np.ndarray,
        apparent_surface_charges: np.ndarray,
    ) -> np.ndarray:
        """Return model-native reaction-field features from point ASC."""

        features, _ = self.project_asc_with_gauge(
            atom_positions_angstrom,
            surface_centers_bohr,
            apparent_surface_charges,
        )
        return features


__all__ = [
    "ATOMIC_CENTER_MEAN_GAUGE",
    "EXACT_GTO_FEATURE_LAYOUT",
    "EXACT_GTO_GRAPH_LONGRANGE_VERSION",
    "ExactGTOFieldProjector",
    "MACEPolarGTOFieldProjectionSpec",
]
