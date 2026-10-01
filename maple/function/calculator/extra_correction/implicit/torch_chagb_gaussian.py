"""Gaussian-sign CHA-GB/ALPB polar algebra with supplied inverse Born inputs.

This separately versioned research scalar changes only the discontinuous CHA
sign factor.  It is not a provider and it does not claim complete coordinate
forces unless its inverse Born inputs are themselves live functions of the
coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING, ClassVar

from .torch_chagb import (
    ALPB_ALPHA,
    AMBER_CHARGE_SCALE,
    CHA_ROH_ANGSTROM,
    CHA_TAU,
    EFFECTIVE_PROBE_ANGSTROM,
    cha_electrostatic_size,
    cha_inverse_born_shift,
)
from .torch_dense_budget import DEFAULT_DENSE_TORCH_BUDGET, DenseTorchBudget

if TYPE_CHECKING:
    from torch import Tensor


GAUSSIAN_CHA_MODEL_IDENTITY = "chagb-r6-pbsa-gaussian-sign-v1"
GAUSSIAN_SIZE_LIMIT_ANGSTROM = 9.5
GAUSSIAN_SIZE_ROUNDOFF_GUARD_ANGSTROM = (
    8.0 * 2.220446049250313e-16 * GAUSSIAN_SIZE_LIMIT_ANGSTROM
)


class GaussianChaSizeDomainError(ValueError):
    """The new identity's strict electrostatic-size domain was exceeded."""

    def __init__(self, electrostatic_size_angstrom: float):
        self.electrostatic_size_angstrom = electrostatic_size_angstrom
        self.limit_angstrom = GAUSSIAN_SIZE_LIMIT_ANGSTROM
        super().__init__(
            "Gaussian CHA v1 requires electrostatic_size strictly below "
            f"{self.limit_angstrom} angstrom; got {electrostatic_size_angstrom}."
        )


@dataclass(frozen=True)
class GaussianChaPolarAlgebraResult:
    """Graph-preserving Gaussian-sign polar values and diagnostics."""

    polar_kcal_mol: Tensor
    self_kcal_mol: Tensor
    pair_kcal_mol: Tensor
    electrostatic_size_angstrom: Tensor
    inverse_born_shift_per_angstrom: Tensor
    shifted_inverse_born_per_angstrom: Tensor
    born_radii_angstrom: Tensor
    effective_charges_e: Tensor
    cha_factors: Tensor
    smoothed_signs: Tensor
    charge_sigma_e: float
    weighted_signs_over_sigma: Tensor
    gaussian_erfc_tails: Tensor
    gaussian_erfcx_tails: Tensor
    scope: ClassVar[str] = "gaussian-sign-polar-algebra-with-supplied-inverse-born"
    complete_coordinate_graph: ClassVar[bool] = False
    model_identity: ClassVar[str] = GAUSSIAN_CHA_MODEL_IDENTITY


def validate_gaussian_sigma_e(sigma_e: float) -> float:
    """Return a validated Gaussian width in electrons; there is no default."""
    if (
        isinstance(sigma_e, bool)
        or not isinstance(sigma_e, (int, float))
        or not math.isfinite(float(sigma_e))
        or float(sigma_e) <= 0.0
    ):
        raise ValueError("sigma_e must be a finite positive scalar in electrons.")
    width = float(sigma_e)
    if not math.isfinite(1.0 / width):
        raise ValueError("sigma_e must have a finite reciprocal in float64.")
    return width


def _require_cpu_float64(name, value, *, shape=None, positive=False, device=None):
    import torch

    if not isinstance(value, torch.Tensor) or value.dtype != torch.float64:
        raise TypeError(f"{name} must be an explicit torch.float64 tensor.")
    if value.device.type != "cpu":
        raise ValueError(f"{name} must be on CPU for Gaussian CHA v1.")
    if shape is not None and tuple(value.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(value.shape)}.")
    if device is not None and value.device != device:
        raise ValueError(f"{name} must use device {device}; no transfer is made.")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite.")
    if positive and not bool((value > 0.0).all()):
        raise ValueError(f"{name} must be positive.")


def gaussian_cha_polar_from_inverse_born(
    positions_angstrom: Tensor,
    charges_e: Tensor,
    effective_cha_radii_angstrom: Tensor,
    unshifted_inverse_born_per_angstrom: Tensor,
    *,
    sigma_e: float,
    resource_budget: DenseTorchBudget = DEFAULT_DENSE_TORCH_BUDGET,
) -> GaussianChaPolarAlgebraResult:
    """Evaluate Gaussian-sign CHA polar energy in kcal/mol.

    ``sigma_e`` is mandatory and belongs to this model identity.  The smooth
    sign is ``erf(S_e/(sqrt(2)*sigma_e))`` where ``S_e`` is the legacy weighted
    effective charge.  Gaussian CHA v1 is restricted to electrostatic size
    strictly below 9.5 A, so the preserved legacy 10 A shift evaluates to zero.
    """
    import torch

    sigma = validate_gaussian_sigma_e(sigma_e)
    if not isinstance(resource_budget, DenseTorchBudget):
        raise TypeError("resource_budget must be a DenseTorchBudget.")
    _require_cpu_float64("positions_angstrom", positions_angstrom)
    if positions_angstrom.ndim != 2 or positions_angstrom.shape[1] != 3:
        raise ValueError("positions_angstrom must have shape (N, 3).")
    count = len(positions_angstrom)
    if count == 0:
        raise ValueError("Gaussian CHA algebra requires a non-empty input.")
    device = positions_angstrom.device
    _require_cpu_float64("charges_e", charges_e, shape=(count,), device=device)
    _require_cpu_float64(
        "effective_cha_radii_angstrom",
        effective_cha_radii_angstrom,
        shape=(count,),
        positive=True,
        device=device,
    )
    _require_cpu_float64(
        "unshifted_inverse_born_per_angstrom",
        unshifted_inverse_born_per_angstrom,
        shape=(count,),
        positive=True,
        device=device,
    )
    derivative_order = (
        2
        if any(
            value.requires_grad
            for value in (
                positions_angstrom,
                charges_e,
                effective_cha_radii_angstrom,
                unshifted_inverse_born_per_angstrom,
            )
        )
        else 0
    )
    resource_budget.admit_pair_graph(
        count, derivative_order=derivative_order, label="Gaussian CHA"
    )

    size = cha_electrostatic_size(positions_angstrom, effective_cha_radii_angstrom)
    if bool(
        size >= GAUSSIAN_SIZE_LIMIT_ANGSTROM - GAUSSIAN_SIZE_ROUNDOFF_GUARD_ANGSTROM
    ):
        raise GaussianChaSizeDomainError(float(size.detach().cpu()))
    differences = positions_angstrom[:, None, :] - positions_angstrom[None, :, :]
    distances_squared = differences.square().sum(dim=-1)
    first, second = torch.triu_indices(count, count, offset=1, device=device)
    if bool((distances_squared[first, second] <= 0.0).any()):
        raise ValueError(
            "Coincident distinct atoms are outside the CHA algebra domain."
        )

    shift = cha_inverse_born_shift(size)
    shifted_inverse = unshifted_inverse_born_per_angstrom + shift
    born = shifted_inverse.reciprocal()
    _require_cpu_float64("shifted Born radii", born, positive=True)
    born_products = born[:, None] * born[None, :]
    native_charges = charges_e * AMBER_CHARGE_SCALE
    weights = torch.exp(-CHA_TAU * distances_squared / born_products)
    effective_native = (weights * native_charges[None, :]).sum(dim=1)
    effective_e = effective_native / AMBER_CHARGE_SCALE
    charge_sigma = effective_e / sigma
    smoothed_signs = torch.erf(charge_sigma / math.sqrt(2.0))
    absolute_normalized = charge_sigma.abs() / math.sqrt(2.0)
    gaussian_tails = torch.erfc(absolute_normalized)
    gaussian_scaled_tails = torch.special.erfcx(absolute_normalized)
    _require_cpu_float64("Gaussian CHA effective charges", effective_e)
    _require_cpu_float64("Gaussian CHA normalized effective charges", charge_sigma)
    _require_cpu_float64("Gaussian CHA smoothed signs", smoothed_signs)
    _require_cpu_float64("Gaussian CHA erfc tails", gaussian_tails)
    _require_cpu_float64("Gaussian CHA erfcx tails", gaussian_scaled_tails)
    mu = 1.0 + smoothed_signs * CHA_ROH_ANGSTROM / (born + EFFECTIVE_PROBE_ANGSTROM)
    _require_cpu_float64("Gaussian CHA factors", mu, positive=True)

    beta = ALPB_ALPHA / 78.5
    dielectric = (1.0 - 1.0 / 78.5) / (1.0 + beta)
    size_term = beta / size
    self_energy = (
        -0.5
        * dielectric
        * (native_charges.square() * (shifted_inverse / mu + size_term)).sum()
    )
    pair_r2 = distances_squared[first, second]
    pair_born = born_products[first, second]
    radicand = (
        pair_r2
        + pair_born * torch.exp(-0.25 * pair_r2 / pair_born) * mu[first] * mu[second]
    )
    _require_cpu_float64("Gaussian CHA pair radicand", radicand, positive=True)
    pair_energy = (
        -dielectric
        * (
            native_charges[first]
            * native_charges[second]
            * (radicand.rsqrt() + size_term)
        ).sum()
    )
    polar = self_energy + pair_energy
    _require_cpu_float64("Gaussian CHA polar energy", polar)
    return GaussianChaPolarAlgebraResult(
        polar_kcal_mol=polar,
        self_kcal_mol=self_energy,
        pair_kcal_mol=pair_energy,
        electrostatic_size_angstrom=size,
        inverse_born_shift_per_angstrom=shift,
        shifted_inverse_born_per_angstrom=shifted_inverse,
        born_radii_angstrom=born,
        effective_charges_e=effective_e,
        cha_factors=mu,
        smoothed_signs=smoothed_signs,
        charge_sigma_e=sigma,
        weighted_signs_over_sigma=charge_sigma,
        gaussian_erfc_tails=gaussian_tails,
        gaussian_erfcx_tails=gaussian_scaled_tails,
    )
