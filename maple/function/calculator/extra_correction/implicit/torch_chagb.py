"""CHA-GB/ALPB *polar algebra* with explicitly supplied inverse Born inputs.

This is not a solvent provider. NSR6 surface construction and PBSA nonpolar
terms are not implemented here. Differentiating with imported/frozen inverse
Born values yields partial derivatives, NOT complete coordinate forces.

The mathematical definitions are independently expressed in tensor form from
Mukhopadhyay et al., JCTC 2014, doi:10.1021/ct4010917 and the versioned numerical
conventions of AmberTools26 RC7. A separately built, source-pinned Fortran
oracle validates the algebra. No upstream Fortran bodies are vendored here.
See TORCH_CHA_REWRITE.md for effective-radius/probe and source/licence boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass
import struct
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from torch import Tensor


# Native default-real constants are promoted after rounding to real(4). These
# are versioned evaluation conventions, not fitted parameters or new controls.
_SHIFT_SLOPE = struct.unpack("f", struct.pack("f", 0.0015))[0]
_SHIFT_INTERCEPT = struct.unpack("f", struct.pack("f", 0.01))[0]
AMBER_CHARGE_SCALE = 18.2223
ALPB_ALPHA = 0.571412
CHA_TAU = 1.47
CHA_ROH_ANGSTROM = 0.586
CHA_RS_ANGSTROM = 0.52
EFFECTIVE_PROBE_ANGSTROM = 1.4 - CHA_RS_ANGSTROM
SIZE_SWITCH_ANGSTROM = 10.0
SIZE_GUARD_ANGSTROM = 1.0e-8
CHARGE_SIGN_GUARD_E = 1.0e-10


@dataclass(frozen=True)
class ChaPolarAlgebraResult:
    """Graph-preserving intermediate tensors; no complete-force capability."""

    polar_kcal_mol: Tensor
    self_kcal_mol: Tensor
    pair_kcal_mol: Tensor
    electrostatic_size_angstrom: Tensor
    inverse_born_shift_per_angstrom: Tensor
    shifted_inverse_born_per_angstrom: Tensor
    born_radii_angstrom: Tensor
    effective_charges_e: Tensor
    cha_factors: Tensor
    scope: ClassVar[str] = "polar-algebra-with-supplied-inverse-born"
    complete_coordinate_graph: ClassVar[bool] = False


def _require_tensor(name, value, *, shape=None, positive=False, device=None):
    import torch

    if not isinstance(value, torch.Tensor) or value.dtype != torch.float64:
        raise TypeError(f"{name} must be an explicit torch.float64 tensor.")
    if shape is not None and tuple(value.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(value.shape)}.")
    if device is not None and value.device != device:
        raise ValueError(
            f"{name} must use device {device}; no implicit transfer is made."
        )
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite.")
    if positive and not bool((value > 0).all()):
        raise ValueError(f"{name} must be positive.")


def cha_electrostatic_size(
    positions_angstrom: Tensor, effective_cha_radii_angstrom: Tensor
) -> Tensor:
    """Radius-cubed weighted electrostatic size, in angstrom.

    Radii are native ``radi`` AFTER cha_rad remapping and +Rs, not the Bondi
    values originally stored in prmtop. No atom typing is performed here.
    """
    import torch

    _require_tensor("positions_angstrom", positions_angstrom)
    if positions_angstrom.ndim != 2 or positions_angstrom.shape[1] != 3:
        raise ValueError("positions_angstrom must have shape (N, 3).")
    count = len(positions_angstrom)
    if count == 0:
        raise ValueError("CHA algebra requires a non-empty input.")
    radii = effective_cha_radii_angstrom
    _require_tensor(
        "effective_cha_radii_angstrom",
        radii,
        shape=(count,),
        positive=True,
        device=positions_angstrom.device,
    )
    weights = radii**3
    mass = weights.sum()
    center = (weights[:, None] * positions_angstrom).sum(dim=0) / mass
    centered = positions_angstrom - center
    second_moment = centered.T @ (weights[:, None] * centered)
    sphere_moment = (2.0 / 5.0) * (weights * radii**2).sum()
    identity = torch.eye(3, dtype=radii.dtype, device=radii.device)
    inertia = (torch.trace(second_moment) + sphere_moment) * identity - second_moment
    determinant = torch.linalg.det(inertia)
    _require_tensor("electrostatic inertia determinant", determinant, positive=True)
    size = (2.5 / mass).sqrt() * determinant.pow(1.0 / 6.0)
    _require_tensor("electrostatic size", size, positive=True)
    return size


def cha_inverse_born_shift(electrostatic_size_angstrom: Tensor) -> Tensor:
    """Return native pre-inversion shift [A^-1], preserving the hard A=10 branch."""
    import torch

    size = electrostatic_size_angstrom
    _require_tensor("electrostatic_size_angstrom", size, positive=True)
    return torch.where(
        size < SIZE_SWITCH_ANGSTROM,
        torch.zeros_like(size),
        _SHIFT_SLOPE * size + _SHIFT_INTERCEPT,
    )


def validate_cha_derivative_branch(result: ChaPolarAlgebraResult, charges_e: Tensor):
    """Reject active hard-switch neighborhoods before interpreting algebra partials.

    Passing this guard is NOT sufficient to establish complete coordinate
    forces: the R6 geometry and PBSA chains are still required.
    """
    import torch

    _require_tensor(
        "charges_e",
        charges_e,
        shape=tuple(result.effective_charges_e.shape),
        device=result.effective_charges_e.device,
    )
    active = charges_e != 0.0
    if bool((active & (result.effective_charges_e.abs() <= CHARGE_SIGN_GUARD_E)).any()):
        raise ValueError(
            "CHA algebra derivative is undefined near an effective-charge sign switch."
        )
    if bool(
        torch.abs(result.electrostatic_size_angstrom - SIZE_SWITCH_ANGSTROM)
        <= SIZE_GUARD_ANGSTROM
    ):
        raise ValueError(
            "CHA algebra derivative is undefined near the size-shift switch."
        )


def cha_polar_from_inverse_born(
    positions_angstrom: Tensor,
    charges_e: Tensor,
    effective_cha_radii_angstrom: Tensor,
    unshifted_inverse_born_per_angstrom: Tensor,
) -> ChaPolarAlgebraResult:
    """Evaluate fixed-profile CHA polar algebra in kcal/mol without detaching inputs.

    The fourth input is NSR6 ``onereff`` BEFORE the CHA size-dependent shift.
    Supplying printed post-shift ``rinv`` here would apply the shift twice.
    Charges are in electrons; the Amber native-charge conversion is internal.
    The effective probe is 1.4 - 0.52 A, matching gb_read for chagb=1.

    Requires float64 tensors on one explicit device. A differentiable upstream
    R6 tensor keeps its graph, but graph presence alone is NOT certification
    of that upstream algorithm. No force adapter is exposed by this module.
    """
    import torch

    size = cha_electrostatic_size(positions_angstrom, effective_cha_radii_angstrom)
    count = len(positions_angstrom)
    _require_tensor(
        "charges_e", charges_e, shape=(count,), device=positions_angstrom.device
    )
    raw_inverse = unshifted_inverse_born_per_angstrom
    _require_tensor(
        "unshifted_inverse_born_per_angstrom",
        raw_inverse,
        shape=(count,),
        positive=True,
        device=positions_angstrom.device,
    )
    differences = positions_angstrom[:, None, :] - positions_angstrom[None, :, :]
    distances_squared = differences.square().sum(dim=-1)
    first, second = torch.triu_indices(
        count, count, offset=1, device=positions_angstrom.device
    )
    if bool((distances_squared[first, second] <= 0.0).any()):
        raise ValueError(
            "Coincident distinct atoms are outside the CHA algebra domain."
        )

    shift = cha_inverse_born_shift(size)
    shifted_inverse = raw_inverse + shift
    born = shifted_inverse.reciprocal()
    _require_tensor("shifted Born radii", born, positive=True)
    born_products = born[:, None] * born[None, :]
    native_charges = charges_e * AMBER_CHARGE_SCALE
    weights = torch.exp(-CHA_TAU * distances_squared / born_products)
    effective_native = (weights * native_charges[None, :]).sum(dim=1)
    effective_e = effective_native / AMBER_CHARGE_SCALE
    mu = 1.0 + effective_native.sign() * CHA_ROH_ANGSTROM / (
        born + EFFECTIVE_PROBE_ANGSTROM
    )
    _require_tensor("CHA factors", mu, positive=True)
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
    _require_tensor("CHA pair radicand", radicand, positive=True)
    pair_energy = (
        -dielectric
        * (
            native_charges[first]
            * native_charges[second]
            * (radicand.rsqrt() + size_term)
        ).sum()
    )
    result = ChaPolarAlgebraResult(
        polar_kcal_mol=self_energy + pair_energy,
        self_kcal_mol=self_energy,
        pair_kcal_mol=pair_energy,
        electrostatic_size_angstrom=size,
        inverse_born_shift_per_angstrom=shift,
        shifted_inverse_born_per_angstrom=shifted_inverse,
        born_radii_angstrom=born,
        effective_charges_e=effective_e,
        cha_factors=mu,
    )
    _require_tensor("polar energy", result.polar_kcal_mol)
    if any(
        t.requires_grad
        for t in (
            positions_angstrom,
            charges_e,
            effective_cha_radii_angstrom,
            raw_inverse,
        )
    ):
        validate_cha_derivative_branch(result, charges_e)
    return result
