"""Exact sigma-split dispersion integral on one certified covering SAS sphere."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING

from .torch_continuum_dispersion import (
    WATER_DENSITY_PER_ANGSTROM3,
    WATER_OXYGEN_EPSILON_KCAL_MOL,
    WATER_OXYGEN_RMIN_ANGSTROM,
    _TWO_TO_NEGATIVE_ONE_SIXTH,
)
from .torch_continuum_dispersion_domain import (
    SingleCoverSASCertificate,
    SingleCoverSASRejection,
    certify_single_covering_dispersion_sas,
)

if TYPE_CHECKING:
    from torch import Tensor


KERNEL_IDENTITY = "torch-single-cover-sas-dispersion-v3"


@dataclass(frozen=True)
class SingleSASDispersionResult:
    energy_kcal_mol: Tensor
    receiver_energies_kcal_mol: Tensor
    certificate: SingleCoverSASCertificate
    branch_labels: tuple[str, ...]
    kernel_identity: str
    runtime_finite_difference: bool
    legacy_quadrature_used: bool


def _select_covering_sas_sphere(
    certificate: SingleCoverSASCertificate, positions_angstrom: Tensor
) -> tuple[int, Tensor, Tensor]:
    """Select the already-certified cover exactly once per public scalar call."""
    index = certificate.covering_index
    return (
        index,
        positions_angstrom[index],
        certificate.sas_radii_angstrom[index],
    )


def _all_inner_surface_integral(
    displacement_squared: Tensor,
    sigma_angstrom: Tensor,
    mixed_epsilon_kcal_mol: Tensor,
) -> Tensor:
    inside_coefficient = -8.0 * mixed_epsilon_kcal_mol * sigma_angstrom.pow(3) / 9.0
    return 4.0 * math.pi * inside_coefficient + 0.0 * displacement_squared


def _all_outer_surface_integral(
    displacement_squared: Tensor,
    radius_angstrom: Tensor,
    sigma_angstrom: Tensor,
    mixed_epsilon_kcal_mol: Tensor,
) -> Tensor:
    radius_squared = radius_angstrom.square()
    c = radius_squared - displacement_squared
    raw_b = 4.0 * mixed_epsilon_kcal_mol * sigma_angstrom.pow(6)
    raw_a = raw_b * sigma_angstrom.pow(6)
    j6 = 4.0 * math.pi * radius_angstrom.pow(3) / c.pow(3)
    numerator = (
        5.0 * radius_angstrom.pow(6)
        + 45.0 * radius_angstrom.pow(4) * displacement_squared
        + 63.0 * radius_squared * displacement_squared.square()
        + 15.0 * displacement_squared.pow(3)
    )
    j12 = 4.0 * math.pi * radius_angstrom.pow(3) * numerator / (5.0 * c.pow(9))
    return -raw_b * j6 / 3.0 + raw_a * j12 / 9.0


def _crossing_primitive_difference(
    q_angstrom: Tensor,
    c_angstrom2: Tensor,
    sigma_angstrom: Tensor,
    mixed_epsilon_kcal_mol: Tensor,
) -> Tensor:
    """Stable ``(Fout-Fin)(q) - (Fout-Fin)(sigma)``."""
    t = q_angstrom / sigma_angstrom
    k = c_angstrom2 / sigma_angstrom.square()
    polynomial = (
        54.0 * k * t.pow(7)
        + 82.0 * k * t.pow(6)
        + 84.0 * k * t.pow(5)
        + 60.0 * k * t.pow(4)
        + 40.0 * k * t.pow(3)
        + 24.0 * k * t.square()
        + 12.0 * k * t
        + 4.0 * k
        + 80.0 * t.pow(8)
        + 105.0 * t.pow(7)
        + 75.0 * t.pow(6)
        + 50.0 * t.pow(5)
        + 30.0 * t.pow(4)
        + 15.0 * t.pow(3)
        + 5.0 * t.square()
    )
    return (
        mixed_epsilon_kcal_mol
        * sigma_angstrom.pow(4)
        * (t - 1.0).pow(3)
        * polynomial
        / (90.0 * t.pow(10))
    )


def _single_receiver_surface_integral(
    displacement_angstrom: Tensor,
    radius_angstrom: Tensor,
    sigma_angstrom: Tensor,
    mixed_epsilon_kcal_mol: Tensor,
) -> tuple[Tensor, str]:
    """Return one receiver's raw surface integral before water density."""
    import torch

    if (
        not isinstance(displacement_angstrom, torch.Tensor)
        or displacement_angstrom.dtype != torch.float64
        or displacement_angstrom.shape != (3,)
    ):
        raise TypeError("receiver displacement must be float64 Torch [3]")
    for name, value in (
        ("radius", radius_angstrom),
        ("sigma", sigma_angstrom),
        ("mixed_epsilon", mixed_epsilon_kcal_mol),
    ):
        if (
            not isinstance(value, torch.Tensor)
            or value.dtype != torch.float64
            or value.shape != ()
            or value.device != displacement_angstrom.device
        ):
            raise TypeError(f"receiver {name} must be a same-device float64 scalar")
    if not bool(
        torch.isfinite(displacement_angstrom).all()
        and torch.isfinite(radius_angstrom)
        and torch.isfinite(sigma_angstrom)
        and torch.isfinite(mixed_epsilon_kcal_mol)
    ):
        raise ValueError("receiver inputs must be finite")
    if not bool(
        (radius_angstrom > 0.0)
        and (sigma_angstrom > 0.0)
        and (mixed_epsilon_kcal_mol >= 0.0)
    ):
        raise ValueError(
            "receiver radius/sigma must be positive and epsilon nonnegative"
        )

    displacement_squared = displacement_angstrom.square().sum()
    if float(displacement_squared.detach()) == 0.0:
        radius_value = float(radius_angstrom.detach())
        sigma_value = float(sigma_angstrom.detach())
        if radius_value == sigma_value:
            raise ValueError("d=0, R=sigma is outside the v3 C2 domain")
        if radius_value < sigma_value:
            return (
                _all_inner_surface_integral(
                    displacement_squared,
                    sigma_angstrom,
                    mixed_epsilon_kcal_mol,
                ),
                "all_inner",
            )
        return (
            _all_outer_surface_integral(
                displacement_squared,
                radius_angstrom,
                sigma_angstrom,
                mixed_epsilon_kcal_mol,
            ),
            "all_outer",
        )

    distance = torch.sqrt(displacement_squared)
    if bool(distance >= radius_angstrom):
        raise ValueError("receiver center must lie strictly inside the covering sphere")
    lower = radius_angstrom - distance
    upper = radius_angstrom + distance
    if bool(upper <= sigma_angstrom):
        return (
            _all_inner_surface_integral(
                displacement_squared,
                sigma_angstrom,
                mixed_epsilon_kcal_mol,
            ),
            "all_inner",
        )
    if bool(lower >= sigma_angstrom):
        return (
            _all_outer_surface_integral(
                displacement_squared,
                radius_angstrom,
                sigma_angstrom,
                mixed_epsilon_kcal_mol,
            ),
            "all_outer",
        )

    c = radius_angstrom.square() - displacement_squared
    base = _all_inner_surface_integral(
        displacement_squared, sigma_angstrom, mixed_epsilon_kcal_mol
    )
    correction = _crossing_primitive_difference(
        upper, c, sigma_angstrom, mixed_epsilon_kcal_mol
    )
    return base + math.pi * correction / distance, "crossing"


def single_sas_dispersion_from_rmin_epsilon(
    positions_angstrom: Tensor,
    lj_rmin_angstrom: Tensor,
    lj_epsilon_kcal_mol: Tensor,
) -> SingleSASDispersionResult:
    """Evaluate the exact frozen dispersion scalar on its certified domain."""
    import torch

    certificate = certify_single_covering_dispersion_sas(
        positions_angstrom, lj_rmin_angstrom
    )
    if isinstance(certificate, SingleCoverSASRejection):
        raise ValueError(
            "v3 single-cover dispersion domain rejected: "
            f"{certificate.reason}: {certificate.detail}"
        )
    if (
        not isinstance(lj_epsilon_kcal_mol, torch.Tensor)
        or lj_epsilon_kcal_mol.dtype != torch.float64
        or lj_epsilon_kcal_mol.device != positions_angstrom.device
        or lj_epsilon_kcal_mol.shape != (len(positions_angstrom),)
    ):
        raise TypeError("epsilon must be same-device CPU float64 [N]")
    if not bool(torch.isfinite(lj_epsilon_kcal_mol).all()):
        raise ValueError("epsilon must be finite")
    if not bool((lj_epsilon_kcal_mol >= 0.0).all()):
        raise ValueError("epsilon must be nonnegative")

    _, cover_center, cover_radius = _select_covering_sas_sphere(
        certificate, positions_angstrom
    )
    sigma = (lj_rmin_angstrom + WATER_OXYGEN_RMIN_ANGSTROM) * _TWO_TO_NEGATIVE_ONE_SIXTH
    mixed_epsilon = torch.sqrt(lj_epsilon_kcal_mol * WATER_OXYGEN_EPSILON_KCAL_MOL)
    receiver_integrals = []
    branch_labels = []
    for receiver in range(len(positions_angstrom)):
        integral, label = _single_receiver_surface_integral(
            cover_center - positions_angstrom[receiver],
            cover_radius,
            sigma[receiver],
            mixed_epsilon[receiver],
        )
        receiver_integrals.append(integral)
        branch_labels.append(label)
    receiver_energies = WATER_DENSITY_PER_ANGSTROM3 * torch.stack(receiver_integrals)
    energy = receiver_energies.sum()
    if not bool(torch.isfinite(energy)):
        raise RuntimeError("single-cover dispersion produced a nonfinite energy")
    return SingleSASDispersionResult(
        energy_kcal_mol=energy,
        receiver_energies_kcal_mol=receiver_energies,
        certificate=certificate,
        branch_labels=tuple(branch_labels),
        kernel_identity=KERNEL_IDENTITY,
        runtime_finite_difference=False,
        legacy_quadrature_used=False,
    )
