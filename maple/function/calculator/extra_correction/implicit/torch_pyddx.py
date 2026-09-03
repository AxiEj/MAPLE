"""First-order PyTorch autograd bridge to the standard pyddx ddPCM backend.

The continuum mathematics, cavity construction, forward solve, adjoint source
gradient, and analytic coordinate derivative remain owned by the pinned
``pyddx==0.8.0`` implementation.  This module only translates those already
validated derivatives into a ``torch.autograd.Function``.  It is therefore a
PyTorch-differentiable wrapper around standard ddPCM, not a second PCM solver.

Only first derivatives are exposed.  Double backward is rejected explicitly
because pyddx does not provide the higher-order operator derivatives required
to implement it without finite-difference approximations.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any

import numpy as np
import torch
from torch.autograd.function import once_differentiable

from .gto_density import external_field_to_density_order
from .pyddx_pcm_response import (
    TESTED_PYDDX_VERSION,
    PyDDXPCMReactionFieldLinearMap,
)

TORCH_PYDDX_AUTOGRAD_CONTRACT_ID = (
    "maple.route2.continuum.pyddx-ddpcm-first-order-autograd.v1"
)


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class TorchPyDDXPCMConfig:
    """Immutable construction data for one standard pyddx ddPCM operator."""

    radii_angstrom: tuple[float, ...]
    dielectric: float
    lmax: int
    n_lebedev: int
    n_proc: int = 1
    solver_tolerance: float = 1.0e-12
    eta: float = 0.1

    def __post_init__(self) -> None:
        radii = tuple(float(value) for value in self.radii_angstrom)
        if not radii or any(
            not math.isfinite(value) or value <= 0.0 for value in radii
        ):
            raise ValueError("radii_angstrom must contain finite positive values.")
        object.__setattr__(self, "radii_angstrom", radii)

        dielectric = float(self.dielectric)
        if not math.isfinite(dielectric) or dielectric <= 1.0:
            raise ValueError("ddPCM dielectric must be finite and greater than one.")
        object.__setattr__(self, "dielectric", dielectric)

        for name in ("lmax", "n_lebedev", "n_proc"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")

        tolerance = float(self.solver_tolerance)
        if not math.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("solver_tolerance must be finite and positive.")
        object.__setattr__(self, "solver_tolerance", tolerance)

        eta = float(self.eta)
        if not math.isfinite(eta) or not 0.0 <= eta <= 1.0:
            raise ValueError("eta must be finite and lie in [0, 1].")
        object.__setattr__(self, "eta", eta)

    @property
    def atom_count(self) -> int:
        return len(self.radii_angstrom)

    @property
    def configuration_sha256(self) -> str:
        return _canonical_sha256(self.as_dict())

    def as_dict(self) -> dict[str, object]:
        return {
            "contract_id": TORCH_PYDDX_AUTOGRAD_CONTRACT_ID,
            "backend": "pyddx",
            "pyddx_version": TESTED_PYDDX_VERSION,
            "continuum_model": "ddPCM",
            "radii_angstrom": list(self.radii_angstrom),
            "dielectric": self.dielectric,
            "lmax": self.lmax,
            "n_lebedev": self.n_lebedev,
            "n_proc": self.n_proc,
            "solver_tolerance": self.solver_tolerance,
            "eta": self.eta,
            "autograd_order": 1,
            "double_backward": False,
        }

    def build(self, positions_angstrom: np.ndarray) -> PyDDXPCMReactionFieldLinearMap:
        return PyDDXPCMReactionFieldLinearMap(
            positions_angstrom,
            np.asarray(self.radii_angstrom, dtype=float),
            dielectric=self.dielectric,
            lmax=self.lmax,
            n_lebedev=self.n_lebedev,
            n_proc=self.n_proc,
            solver_tolerance=self.solver_tolerance,
            eta=self.eta,
        )


def _validate_inputs(
    positions: torch.Tensor,
    source: torch.Tensor,
    config: TorchPyDDXPCMConfig,
) -> None:
    if not isinstance(config, TorchPyDDXPCMConfig):
        raise TypeError("config must be TorchPyDDXPCMConfig.")
    if positions.dtype != torch.float64 or source.dtype != torch.float64:
        raise TypeError("standard ddPCM autograd inputs must use torch.float64.")
    if positions.device != source.device:
        raise ValueError("positions and source must be on the same Torch device.")
    if positions.shape != (config.atom_count, 3):
        raise ValueError("positions must have shape " f"({config.atom_count}, 3).")
    if source.shape != (config.atom_count, 4):
        raise ValueError("source must have shape " f"({config.atom_count}, 4).")
    if not bool(torch.isfinite(positions).all()) or not bool(
        torch.isfinite(source).all()
    ):
        raise ValueError("positions and source must contain only finite values.")


class _PyDDXPCMPolarizationEnergy(torch.autograd.Function):
    """Standard ddPCM energy with analytic first-order pyddx derivatives."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any,
        positions: torch.Tensor,
        source: torch.Tensor,
        config: TorchPyDDXPCMConfig,
    ) -> torch.Tensor:
        _validate_inputs(positions, source, config)
        positions_numpy = np.asarray(positions.detach().cpu(), dtype=float)
        source_numpy = np.asarray(source.detach().cpu(), dtype=float)

        reaction_field = config.build(positions_numpy)
        from ase.units import Hartree

        energy_ev = float(
            reaction_field.polarization_energy_hartree(source_numpy) * Hartree
        )
        field_cartesian = reaction_field.apply(source_numpy)
        source_gradient_raw = external_field_to_density_order(field_cartesian)
        position_gradient = (
            reaction_field.polarization_position_gradient_ev_per_angstrom(source_numpy)
        )

        if not math.isfinite(energy_ev):
            raise FloatingPointError("pyddx ddPCM energy is non-finite.")
        if not np.all(np.isfinite(source_gradient_raw)) or not np.all(
            np.isfinite(position_gradient)
        ):
            raise FloatingPointError("pyddx ddPCM derivative is non-finite.")

        source_gradient_tensor = torch.as_tensor(
            np.array(source_gradient_raw, dtype=float, copy=True),
            dtype=source.dtype,
            device=source.device,
        )
        position_gradient_tensor = torch.as_tensor(
            np.array(position_gradient, dtype=float, copy=True),
            dtype=positions.dtype,
            device=positions.device,
        )
        ctx.save_for_backward(position_gradient_tensor, source_gradient_tensor)
        return torch.tensor(energy_ev, dtype=positions.dtype, device=positions.device)

    @staticmethod
    @once_differentiable
    def backward(  # type: ignore[override]
        ctx: Any,
        gradient_output: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, None]:
        position_gradient, source_gradient = ctx.saved_tensors
        if gradient_output.numel() != 1 or not bool(torch.isfinite(gradient_output)):
            raise ValueError("ddPCM energy cotangent must be one finite scalar.")
        scale = gradient_output.reshape(())
        return scale * position_gradient, scale * source_gradient, None


def pyddx_pcm_energy(
    positions: torch.Tensor,
    source: torch.Tensor,
    config: TorchPyDDXPCMConfig,
) -> torch.Tensor:
    """Return standard ddPCM polarization energy with first-order autograd.

    ``positions`` use angstrom and ``source`` uses MAPLE's raw point-``l<=1``
    order ``[q, y, z, x]``.  The scalar is returned in eV.  Backward returns
    ``dE/dR`` in eV/angstrom and ``dE/dsource`` in the raw source-dual order.
    """

    return _PyDDXPCMPolarizationEnergy.apply(positions, source, config)


__all__ = [
    "TORCH_PYDDX_AUTOGRAD_CONTRACT_ID",
    "TorchPyDDXPCMConfig",
    "pyddx_pcm_energy",
]
