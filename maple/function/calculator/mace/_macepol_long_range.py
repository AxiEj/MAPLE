"""Isolated long-range evaluator policy for MACE-POLAR Route 2.

The default policy is an exact no-op.  The only alternative is the
evidence-gated, fixed-40-A reciprocal evaluator used by one explicit Route-2
profile.  It changes no learned weights and owns all fixed-box/dtype bridging
details so they do not leak into the calculator or continuum providers.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import numpy as np
import torch

from ...route2_smd_profiles import (
    MACEPOL_FORCED_RECIPROCAL_FIXED_BOX40_PROFILE,
    MACEPOL_MOLECULAR_REALSPACE_PROFILE,
)


_VALIDATED_GRAPH_LONGRANGE_VERSION = "0.4.0"
_FIXED_BOX_LENGTH_ANGSTROM = 40.0


class _DtypeSafeInternalFieldProjection(torch.nn.Module):
    """Cast only the reciprocal molecular correction field for projection."""

    def __init__(self, upstream: torch.nn.Module):
        super().__init__()
        if not hasattr(upstream, "matrix") or not hasattr(
            upstream,
            "projections_dim",
        ):
            raise RuntimeError(
                "The graph_longrange reciprocal correction projector is "
                "incompatible with the validated dtype bridge."
            )
        self.upstream = upstream
        self.projections_dim = upstream.projections_dim

    def forward(self, batch, positions, node_fields):
        return self.upstream(
            batch=batch,
            positions=positions,
            node_fields=node_fields.to(
                dtype=self.upstream.matrix.dtype,
                device=self.upstream.matrix.device,
            ),
        )


@dataclass(frozen=True)
class MACEPolarLongRangeEvaluator:
    """One closed, versioned MACE-POLAR long-range evaluation policy."""

    profile: str
    use_pbc_evaluator: bool
    box_length_angstrom: float | None

    def __post_init__(self) -> None:
        expected = {
            MACEPOL_MOLECULAR_REALSPACE_PROFILE: (False, None),
            MACEPOL_FORCED_RECIPROCAL_FIXED_BOX40_PROFILE: (
                True,
                _FIXED_BOX_LENGTH_ANGSTROM,
            ),
        }
        try:
            expected_pbc_evaluator, expected_box_length = expected[self.profile]
        except KeyError as exc:
            raise ValueError(
                "Unsupported MACE-POLAR long-range evaluator profile: "
                f"{self.profile}."
            ) from exc
        if (
            self.use_pbc_evaluator is not expected_pbc_evaluator
            or self.box_length_angstrom != expected_box_length
        ):
            raise ValueError(
                "The MACE-POLAR long-range evaluator fields are inconsistent "
                f"with profile={self.profile}."
            )

    @classmethod
    def from_profile(cls, profile: str) -> "MACEPolarLongRangeEvaluator":
        normalized = str(profile).strip().lower()
        if normalized == MACEPOL_MOLECULAR_REALSPACE_PROFILE:
            return cls(
                profile=normalized,
                use_pbc_evaluator=False,
                box_length_angstrom=None,
            )
        if normalized == MACEPOL_FORCED_RECIPROCAL_FIXED_BOX40_PROFILE:
            return cls(
                profile=normalized,
                use_pbc_evaluator=True,
                box_length_angstrom=_FIXED_BOX_LENGTH_ANGSTROM,
            )
        raise ValueError(
            f"Unsupported MACE-POLAR long-range evaluator profile: {profile}."
        )

    @property
    def is_default(self) -> bool:
        return self.profile == MACEPOL_MOLECULAR_REALSPACE_PROFILE

    @property
    def model_forward_kwargs(self) -> dict[str, bool]:
        if self.is_default:
            return {}
        return {"use_pbc_evaluator": True}

    def coordinate_vjp(self, cotangent: np.ndarray) -> np.ndarray:
        """Pull one coordinate cotangent back to the input geometry.

        The molecular real-space profile uses the identity transform.  The
        fixed-box reciprocal profile evaluates the model at
        ``R - mean(R)`` and therefore applies the corresponding centering
        projector to every returned coordinate cotangent.
        """

        values = np.asarray(cotangent, dtype=float)
        if (
            values.ndim != 2
            or values.shape[1] != 3
            or values.shape[0] == 0
            or not np.all(np.isfinite(values))
        ):
            raise ValueError(
                "Coordinate cotangents must be finite with shape "
                "(n_atoms, 3)."
            )
        result = values.copy()
        if not self.is_default:
            result -= np.mean(result, axis=0, keepdims=True)
        return result

    def coordinate_hessian_pullback(
        self,
        hessian: np.ndarray,
    ) -> np.ndarray:
        """Pull one Cartesian Hessian back through the coordinate transform."""

        values = np.asarray(hessian, dtype=float)
        if (
            values.ndim != 2
            or values.shape[0] == 0
            or values.shape[0] != values.shape[1]
            or values.shape[0] % 3 != 0
            or not np.all(np.isfinite(values))
        ):
            raise ValueError(
                "Coordinate Hessians must be finite square (3N, 3N) arrays."
            )
        if self.is_default:
            return values.copy()

        atom_count = values.shape[0] // 3
        blocks = values.reshape(atom_count, 3, atom_count, 3)
        row_mean = np.mean(blocks, axis=0, keepdims=True)
        column_mean = np.mean(blocks, axis=2, keepdims=True)
        total_mean = np.mean(
            blocks,
            axis=(0, 2),
            keepdims=True,
        )
        return (
            blocks - row_mean - column_mean + total_mean
        ).reshape(values.shape)

    @property
    def provenance(self) -> dict[str, Any]:
        if self.is_default:
            return {
                "profile": self.profile,
                "operator": "graph_longrange molecular real-space evaluator",
                "use_pbc_evaluator": False,
                "box_length_angstrom": None,
                "coordinate_policy": "upstream molecular batch",
                "coordinate_derivative_policy": "identity pullback",
                "pbc": False,
                "dtype_bridge": False,
                "equivalent_to_default_evaluator": True,
                "experimental": False,
            }
        return {
            "profile": self.profile,
            "operator": (
                "graph_longrange forced periodic reciprocal-space evaluator "
                "for an FFF molecular graph"
            ),
            "use_pbc_evaluator": True,
            "box_length_angstrom": self.box_length_angstrom,
            "coordinate_policy": "arithmetic-mean centering in fixed cubic box",
            "coordinate_derivative_policy": (
                "centering-projector VJP and double-sided Hessian pullback"
            ),
            "pbc": False,
            "dtype_bridge": True,
            "graph_longrange_version": _VALIDATED_GRAPH_LONGRANGE_VERSION,
            "equivalent_to_default_evaluator": False,
            "experimental": True,
            "claim_boundary": (
                "This profile changes the long-range evaluation operator for "
                "FFF systems from the default real-space path to a forced "
                "periodic reciprocal-space path. It is an experimental "
                "fixed-box evaluator, not a proof of equivalence to the "
                "default evaluator. Results are box- and placement-dependent "
                "and require explicit convergence validation."
            ),
        }

    def configure_model(self, model) -> None:
        """Install the narrow dtype bridge required by the reciprocal path."""

        if self.is_default:
            return
        try:
            installed_version = version("graph-longrange")
        except PackageNotFoundError as exc:
            raise RuntimeError(
                "The reciprocal MACE-POLAR profile requires graph_longrange."
            ) from exc
        if installed_version != _VALIDATED_GRAPH_LONGRANGE_VERSION:
            raise RuntimeError(
                "The reciprocal MACE-POLAR profile is validated only with "
                "graph_longrange "
                f"{_VALIDATED_GRAPH_LONGRANGE_VERSION}; found "
                f"{installed_version}."
            )
        try:
            correction_terms = (
                model.electric_potential_descriptor
                .non_periodic_correction_terms
            )
            projection = correction_terms.displaced_interactions
        except AttributeError as exc:
            raise RuntimeError(
                "The loaded MACE-POLAR model lacks the validated "
                "graph_longrange reciprocal correction path."
            ) from exc
        if isinstance(projection, _DtypeSafeInternalFieldProjection):
            return
        correction_terms.displaced_interactions = (
            _DtypeSafeInternalFieldProjection(projection)
        )

    def prepare_batch(
        self,
        batch: dict[str, torch.Tensor],
        *,
        r_max: float,
    ) -> dict[str, torch.Tensor]:
        """Return the exact upstream batch or one validated fixed-box batch."""

        if self.is_default:
            return batch
        required = {
            "positions",
            "batch",
            "cell",
            "rcell",
            "volume",
            "pbc",
        }
        missing = sorted(required.difference(batch))
        if missing:
            raise RuntimeError(
                "The reciprocal MACE-POLAR evaluator batch is missing: "
                + ", ".join(missing)
                + "."
            )
        positions = batch["positions"]
        graph_index = batch["batch"]
        if (
            positions.ndim != 2
            or positions.shape[1] != 3
            or not torch.is_floating_point(positions)
        ):
            raise RuntimeError(
                "The reciprocal MACE-POLAR evaluator requires floating-point "
                "positions with shape (n_atoms, 3)."
            )
        if graph_index.shape != (positions.shape[0],) or not bool(
            torch.all(graph_index == 0)
        ):
            raise RuntimeError(
                "The reciprocal MACE-POLAR profile supports one molecular "
                "graph per evaluation."
            )
        pbc = batch["pbc"]
        if (
            not isinstance(pbc, torch.Tensor)
            or pbc.numel() != 3
            or bool(torch.any(pbc))
        ):
            raise RuntimeError(
                "The reciprocal MACE-POLAR profile requires one explicitly "
                "non-periodic FFF molecular graph."
            )

        centered = positions - torch.mean(positions, dim=0)
        half_box = float(self.box_length_angstrom) / 2.0
        maximum_extent = float(torch.max(torch.abs(centered)).detach().cpu())
        # This is a supported-domain fit gate, not a box-convergence proof.
        if maximum_extent + float(r_max) >= half_box:
            raise RuntimeError(
                "The molecule plus the MACE cutoff does not fit inside the "
                "validated 40 A fixed box."
            )

        result = dict(batch)
        result["positions"] = centered
        cell_template = batch["cell"]
        rcell_template = batch["rcell"]
        volume_template = batch["volume"]
        if cell_template.numel() != 9 or rcell_template.numel() != 9:
            raise RuntimeError(
                "The reciprocal MACE-POLAR evaluator requires one 3x3 cell."
            )
        if volume_template.numel() != 1:
            raise RuntimeError(
                "The reciprocal MACE-POLAR evaluator requires one cell volume."
            )
        identity = torch.eye(
            3,
            dtype=positions.dtype,
            device=positions.device,
        )
        box_length = float(self.box_length_angstrom)
        cell = identity * box_length
        rcell = identity * (2.0 * torch.pi / box_length)
        volume = positions.new_tensor(box_length**3)
        result["cell"] = cell.reshape_as(cell_template)
        result["rcell"] = rcell.reshape_as(rcell_template)
        result["volume"] = volume.reshape_as(volume_template)
        return result

    def forward_model(self, model, batch, **kwargs):
        return model(
            batch,
            **kwargs,
            **self.model_forward_kwargs,
        )


__all__ = ["MACEPolarLongRangeEvaluator"]
