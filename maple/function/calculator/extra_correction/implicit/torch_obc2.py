"""Torch automatic-differentiation backend for OpenMM's OBC-II/ACE model.

The OBC-II and ACE expressions in this file are derived from OpenMM 8.5.2
``openmm.app.internal.customgbforces`` (pinned source SHA256
``f96a49a9e9ef0528e937481b1c1331b43ad03e6ec3018b478c642b301f07391a``).

Portions copyright (c) 2012-2022 University of Virginia and the Authors.
Authors: Christoph Klein, Michael R. Shirts
Contributors: Jason M. Swails, Peter Eastman, Justin L. MacCallum

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

from __future__ import annotations

import numpy as np

from .common import KJ_PER_MOL_PER_HARTREE
from .obc2_parameters import OBC_OFFSET_NM, OBC2Parameters
from .openmm_compat import require_verified_obc2_expression_source
from .result import SolvationDirectionalResult, SolvationResult

KINK_MARGIN_NM = 1.0e-8
COULOMB_KJ_MOL_NM_E2 = 138.935485
ACE_KJ_MOL_NM2 = 28.3919551


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised without optional extra
        raise ImportError(
            "Torch OBC-II requires PyTorch. Install a MAPLE MLIP extra that "
            "provides torch."
        ) from exc
    return torch


class TorchOBC2:
    """Exact float64 Torch transcription of zero-salt OpenMM OBC-II/ACE."""

    supported_properties = frozenset({"energy", "forces", "hessian", "hvp"})

    def __init__(self, parameters: OBC2Parameters, *, device="cpu", dtype=None):
        torch = _torch()
        if dtype is None:
            dtype = torch.float64
        if dtype != torch.float64:
            raise ValueError("Torch OBC-II requires torch.float64 for parity.")
        if parameters.nonpolar not in {"ace", "none"}:
            raise ValueError("Torch OBC-II nonpolar must be 'ace' or 'none'.")
        self.parameters = parameters
        self.source_compatibility = require_verified_obc2_expression_source()
        self.device = torch.device(device)
        self.dtype = dtype
        self._charges = torch.as_tensor(
            np.array(parameters.charges, copy=True), dtype=dtype, device=self.device
        )
        self._or = torch.as_tensor(
            np.array(parameters.offset_radii_nm, copy=True),
            dtype=dtype,
            device=self.device,
        )
        self._sr = torch.as_tensor(
            np.array(parameters.scaled_offset_radii_nm, copy=True),
            dtype=dtype,
            device=self.device,
        )
        self._provenance = {
            "provider": "torch-obc2",
            "method": "gb",
            "model": "obc2",
            "amber_igb": 5,
            "nonpolar": parameters.nonpolar,
            "derivatives": "torch automatic differentiation",
            "source_compatibility": dict(self.source_compatibility),
            "dtype": str(dtype),
            "device": str(self.device),
            "solvent_dielectric": parameters.solvent_dielectric,
            "solute_dielectric": parameters.solute_dielectric,
        }
        self.last_derivative_provenance = None

    @property
    def provenance(self):
        return dict(self._provenance)

    def _positions(self, atoms, *, requires_grad: bool):
        torch = _torch()
        positions = np.asarray(atoms.get_positions(), dtype=np.float64)
        expected = (self._charges.numel(), 3)
        if positions.shape != expected:
            raise ValueError(
                f"Torch OBC-II positions must have shape {expected}, got {positions.shape}."
            )
        if not np.isfinite(positions).all():
            raise ValueError("Torch OBC-II positions must be finite.")
        return torch.tensor(
            positions,
            dtype=self.dtype,
            device=self.device,
            requires_grad=requires_grad,
        )

    def _geometry(self, positions_angstrom, *, second_derivatives: bool):
        torch = _torch()
        positions_nm = positions_angstrom * 0.1
        displacement = positions_nm[:, None, :] - positions_nm[None, :, :]
        squared = torch.sum(displacement * displacement, dim=-1)
        diagonal = torch.eye(squared.shape[0], dtype=torch.bool, device=squared.device)
        # The diagonal is excluded analytically.  Replacing it before sqrt keeps
        # both first and second autograd passes away from sqrt'(0).
        safe_squared = torch.where(diagonal, torch.ones_like(squared), squared)
        distance = torch.sqrt(safe_squared)
        off_diagonal = ~diagonal
        collision_margin = (
            torch.min(distance[off_diagonal])
            if bool(torch.any(off_diagonal).item())
            else distance.new_tensor(float("inf"))
        )
        if bool((collision_margin <= KINK_MARGIN_NM).item()):
            raise ValueError("OBC-II atom collision lies within the branch margin.")

        source_sr = self._sr[None, :]
        target_or = self._or[:, None]
        difference = torch.abs(distance - source_sr)
        lower = torch.maximum(target_or, difference)
        upper = distance + source_sr
        active = (distance + source_sr - target_or) >= 0.0

        step_margin_values = torch.abs(distance + source_sr - target_or)
        max_switch_margin_values = torch.abs(target_or - difference)
        step_margin = (
            torch.min(step_margin_values[off_diagonal])
            if bool(torch.any(off_diagonal).item())
            else distance.new_tensor(float("inf"))
        )
        effective_max_switch = off_diagonal & active
        max_switch_margin = (
            torch.min(max_switch_margin_values[effective_max_switch])
            if bool(torch.any(effective_max_switch).item())
            else distance.new_tensor(float("inf"))
        )
        diagnostics = {
            "minimum_collision_margin_nm": float(collision_margin.detach().cpu()),
            "minimum_step_margin_nm": float(step_margin.detach().cpu()),
            "minimum_max_switch_margin_nm": float(max_switch_margin.detach().cpu()),
            "rejection_margin_nm": KINK_MARGIN_NM,
        }
        if second_derivatives:
            if bool((step_margin <= KINK_MARGIN_NM).item()) or bool(
                (max_switch_margin <= KINK_MARGIN_NM).item()
            ):
                raise ValueError(
                    "OBC-II geometry lies within the requested branch/kink margin."
                )
        return distance, off_diagonal, lower, upper, active, diagnostics

    def _components_kj(self, positions_angstrom, *, second_derivatives=False):
        torch = _torch()
        distance, off_diagonal, lower, upper, active, diagnostics = self._geometry(
            positions_angstrom, second_derivatives=second_derivatives
        )
        sr = self._sr[None, :]
        integral_expression = 0.5 * (
            1.0 / lower
            - 1.0 / upper
            + 0.25
            * (distance - sr * sr / distance)
            * (1.0 / (upper * upper) - 1.0 / (lower * lower))
            + 0.5 * torch.log(lower / upper) / distance
        )
        pair_integrals = torch.where(
            off_diagonal & active,
            integral_expression,
            torch.zeros_like(integral_expression),
        )
        integral = torch.sum(pair_integrals, dim=1)
        psi = integral * self._or
        radius = self._or + OBC_OFFSET_NM
        denominator = (
            1.0 / self._or
            - torch.tanh(psi - 0.8 * psi * psi + 4.85 * psi * psi * psi) / radius
        )
        if not bool(torch.all(torch.isfinite(denominator)).item()) or bool(
            torch.any(denominator <= 0.0).item()
        ):
            raise ValueError("OBC-II effective Born-radius domain is invalid.")
        born = 1.0 / denominator

        dielectric = (
            1.0 / self.parameters.solute_dielectric
            - 1.0 / self.parameters.solvent_dielectric
        )
        self_energy = (
            -0.5
            * COULOMB_KJ_MOL_NM_E2
            * dielectric
            * torch.sum(self._charges * self._charges / born)
        )
        born_product = born[:, None] * born[None, :]
        f_squared = squared_distance = distance * distance
        f_squared = squared_distance + born_product * torch.exp(
            -squared_distance / (4.0 * born_product)
        )
        # As above, exclude the diagonal before sqrt to avoid creating an
        # irrelevant diagonal branch in the autograd graph.
        safe_f_squared = torch.where(
            off_diagonal, f_squared, torch.ones_like(f_squared)
        )
        if not bool(torch.all(torch.isfinite(safe_f_squared)).item()) or bool(
            torch.any(safe_f_squared <= 0.0).item()
        ):
            raise ValueError("OBC-II pair-function square-root domain is invalid.")
        pair_f = torch.sqrt(safe_f_squared)
        charge_products = self._charges[:, None] * self._charges[None, :]
        upper_triangle = torch.triu(
            torch.ones_like(pair_f, dtype=torch.bool), diagonal=1
        )
        pair_energy = (
            -COULOMB_KJ_MOL_NM_E2
            * dielectric
            * torch.sum(
                torch.where(
                    upper_triangle,
                    charge_products / pair_f,
                    torch.zeros_like(pair_f),
                )
            )
        )
        polar = self_energy + pair_energy
        if self.parameters.nonpolar == "ace":
            nonpolar = ACE_KJ_MOL_NM2 * torch.sum(
                (radius + 0.14) ** 2 * (radius / born) ** 6
            )
        else:
            nonpolar = torch.zeros((), dtype=self.dtype, device=self.device)
        if not bool(torch.isfinite(polar + nonpolar).item()):
            raise ValueError("Torch OBC-II produced a non-finite energy.")
        return polar, nonpolar, diagnostics

    def _energy_hartree(self, positions_angstrom, *, second_derivatives=False):
        polar, nonpolar, _diagnostics = self._components_kj(
            positions_angstrom, second_derivatives=second_derivatives
        )
        return (polar + nonpolar) / KJ_PER_MOL_PER_HARTREE

    def branch_diagnostics(self, atoms) -> dict[str, float]:
        positions = self._positions(atoms, requires_grad=False)
        *_geometry, diagnostics = self._geometry(positions, second_derivatives=False)
        return diagnostics

    def evaluate(self, atoms, need_forces: bool = False, calculator=None):
        torch = _torch()
        positions = self._positions(atoms, requires_grad=need_forces)
        polar, nonpolar, diagnostics = self._components_kj(positions)
        total = polar + nonpolar
        forces = None
        if need_forces:
            gradient = torch.autograd.grad(total, positions)[0]
            forces = (-gradient / KJ_PER_MOL_PER_HARTREE).detach().cpu().numpy()
        polar_value = float((polar / KJ_PER_MOL_PER_HARTREE).detach().cpu())
        nonpolar_value = float((nonpolar / KJ_PER_MOL_PER_HARTREE).detach().cpu())
        return SolvationResult(
            energy_hartree=polar_value + nonpolar_value,
            forces_hartree_per_angstrom=forces,
            components_hartree={
                "polar": polar_value,
                "nonpolar": nonpolar_value,
            },
            provenance={**self.provenance, "branch_diagnostics": diagnostics},
        )

    def hessian(self, atoms) -> np.ndarray:
        """Return the row-wise analytic Cartesian Hessian in Hartree/A^2."""
        torch = _torch()
        positions = self._positions(atoms, requires_grad=True)
        energy = self._energy_hartree(positions, second_derivatives=True)
        gradient = torch.autograd.grad(energy, positions, create_graph=True)[0].reshape(
            -1
        )
        rows = [
            torch.autograd.grad(component, positions, retain_graph=True)[0].reshape(-1)
            for component in gradient
        ]
        hessian = torch.stack(rows).detach().cpu().numpy()
        self.last_derivative_provenance = {
            **self.provenance,
            "derivative": "hessian",
            "branch_diagnostics": self.branch_diagnostics(atoms),
        }
        return hessian

    def hvp(self, atoms, direction) -> np.ndarray:
        """Apply the analytic Cartesian Hessian without materializing it."""
        return self.directional_derivatives(atoms, direction).hvp_hartree_per_angstrom2

    def directional_derivatives(self, atoms, direction) -> SolvationDirectionalResult:
        """Return solvent E/F/HVP from one scalar autograd graph."""
        torch = _torch()
        size = 3 * self._charges.numel()
        values = np.asarray(direction, dtype=np.float64)
        if values.shape != (size,):
            raise ValueError(
                f"Torch OBC-II HVP direction shape must be ({size},), got {values.shape}."
            )
        if not np.isfinite(values).all():
            raise ValueError("Torch OBC-II HVP direction must be finite.")
        positions = self._positions(atoms, requires_grad=True)
        energy = self._energy_hartree(positions, second_derivatives=True)
        gradient = torch.autograd.grad(energy, positions, create_graph=True)[0]
        vector = torch.as_tensor(
            values.reshape(positions.shape), dtype=self.dtype, device=self.device
        )
        product = torch.autograd.grad(
            gradient,
            positions,
            grad_outputs=vector,
        )[0]
        provenance = {
            **self.provenance,
            "derivative": "directional-hvp",
            "branch_diagnostics": self.branch_diagnostics(atoms),
        }
        self.last_derivative_provenance = provenance
        return SolvationDirectionalResult(
            hvp_hartree_per_angstrom2=(product.reshape(-1).detach().cpu().numpy()),
            forces_hartree_per_angstrom=(-gradient.detach().cpu().numpy()),
            energy_hartree=float(energy.detach().cpu()),
            provenance=provenance,
        )
