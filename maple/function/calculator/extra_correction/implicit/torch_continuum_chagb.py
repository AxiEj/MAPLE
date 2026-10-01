"""Pure, unregistered complete scalar for a narrowly certified CHA point domain.

No provider, force adapter, optimizer contract, or supported-property claim is
defined here.  Coordinate derivatives are obtained by differentiating the one
returned scalar; finite differences are validation-only.

Radius provenance is intentionally component-specific: the R6 solvent-
excluded surface and CHA polar algebra both use native remapped ``radi``
(``cha_radii_angstrom``, including Rs); cavity uses ``lj_rmin + 1.3 A`` and
dispersion uses ``lj_rmin + 0.557 A``.  Bondi radii remain frozen input
provenance but do not enter this continuum scalar.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, ClassVar

from .continuum_chagb_inputs import ContinuumChaTopology
from .torch_chagb import (
    CHARGE_SIGN_GUARD_E,
    SIZE_GUARD_ANGSTROM,
    SIZE_SWITCH_ANGSTROM,
    ChaPolarAlgebraResult,
    cha_polar_from_inverse_born,
)
from .torch_continuum_chagb_domain import (
    CertifiedLocalPatchScope,
    DomainCertificationFailure,
    certify_three_site_domain,
)
from .torch_continuum_dispersion import dispersion_from_rmin_epsilon
from .torch_continuum_r6_patches import (
    contact_patch_inverse_cube,
    pair_torus_inverse_cube,
)
from .torch_continuum_sav import cavity_from_rmin

if TYPE_CHECKING:
    from torch import Tensor


# Every restored R6 patch quadrature constructs dense order-by-order
# Gauss-Legendre eigenproblem inputs.  The frozen 24/48/64 production study and
# 8/32 diagnostics fit this explicit envelope; arbitrary user-sized allocation
# is not part of the unregistered research contract.
MAX_QUADRATURE_ORDER = 128


@dataclass(frozen=True)
class ContinuumChaQuadratureIdentity:
    r6_order: int
    phi_order: int
    nonpolar_atol: float = 1.0e-11
    nonpolar_rtol: float = 1.0e-11
    component_sum_residual_kcal_mol: float = 0.0


@dataclass(frozen=True)
class ContinuumChaResourceAccounting:
    site_count: int
    active_pair_count: int
    cavity_area_evaluations: int
    dispersion_z_nodes_evaluated: int
    dispersion_ad_site_cap: int = 6


@dataclass(frozen=True)
class ContinuumChaRadiusProvenance:
    r6_ses: str = "cha_radii_angstrom (native radi: CHA map + Rs)"
    polar_algebra: str = "cha_radii_angstrom (native radi: CHA map + Rs)"
    cavity: str = "lj_rmin_angstrom + 1.3 angstrom"
    dispersion: str = "lj_rmin_angstrom + 0.557 angstrom"
    bondi: str = "frozen topology provenance only; unused by continuum scalar"


@dataclass(frozen=True)
class ChaBranchDiagnostics:
    weighted_signs_e: Tensor
    absolute_sign_margins_e: Tensor
    electrostatic_size_angstrom: Tensor
    size_switch_margin_angstrom: Tensor


@dataclass(frozen=True)
class ContinuumChaScalarResult:
    polar_kcal_mol: Tensor
    cavity_kcal_mol: Tensor
    dispersion_kcal_mol: Tensor
    total_kcal_mol: Tensor
    point_domain: CertifiedLocalPatchScope
    cha: ChaBranchDiagnostics
    quadrature_identity: ContinuumChaQuadratureIdentity
    resources: ContinuumChaResourceAccounting
    radius_provenance: ContinuumChaRadiusProvenance
    scope: ClassVar[str] = "unregistered-three-site-point-scalar"
    supported_properties: ClassVar[tuple[str, ...]] = ()


class ContinuumChaDomainError(ValueError):
    """Raised before scalar assembly when point geometry is uncertified."""

    def __init__(self, failure: DomainCertificationFailure):
        super().__init__(
            f"CHA point domain rejected ({failure.reason.value}): {failure.message}"
        )
        self.failure = failure


class ChaBranchFailureReason(str, Enum):
    WEIGHTED_SIGN_ZERO = "weighted-sign-zero"
    SIZE_SWITCH = "size-switch"


class ContinuumChaBranchError(ValueError):
    """Typed CHA point-branch rejection, separate from R6 geometry scope."""

    def __init__(
        self,
        reason: ChaBranchFailureReason,
        message: str,
        raw_margins: tuple[tuple[str, float], ...],
    ):
        super().__init__(message)
        self.reason = reason
        self.raw_margins = tuple(raw_margins)


def _r6_inverse_born(positions, intrinsic_radii, domain, order):
    """Assemble the complete admitted SES flux without inventing new patches."""
    import torch

    inverse_cube = contact_patch_inverse_cube(
        positions, intrinsic_radii, mu_order=order, phi_order=order
    ).inverse_cube_per_angstrom3
    for pair in domain.active_pairs:
        inverse_cube = (
            inverse_cube
            + pair_torus_inverse_cube(
                positions,
                intrinsic_radii,
                pair,
                theta_order=order,
                meridian_order=order,
            ).inverse_cube_per_angstrom3
        )
    if not bool(torch.isfinite(inverse_cube).all() and (inverse_cube > 0.0).all()):
        raise RuntimeError("Certified R6 patches did not yield positive inverse cubes.")
    return inverse_cube.pow(1.0 / 3.0)


def _branch_diagnostics(
    polar: ChaPolarAlgebraResult, charges_e: Tensor
) -> ChaBranchDiagnostics:
    import torch

    active = charges_e != 0.0
    sign_margins = polar.effective_charges_e.abs()
    relevant = sign_margins[active]
    minimum_sign_margin = (
        float(relevant.detach().min()) if len(relevant) else float("inf")
    )
    size_margin = torch.abs(polar.electrostatic_size_angstrom - SIZE_SWITCH_ANGSTROM)
    raw_margins = tuple(
        (f"weighted_sign_{index}_abs_e", float(value.detach()))
        for index, value in enumerate(sign_margins)
    ) + (
        ("minimum_relevant_sign_margin_e", minimum_sign_margin),
        ("size_switch_margin_angstrom", float(size_margin.detach())),
    )
    if bool((active & (sign_margins <= CHARGE_SIGN_GUARD_E)).any()):
        raise ContinuumChaBranchError(
            ChaBranchFailureReason.WEIGHTED_SIGN_ZERO,
            "A relevant CHA weighted sign is at or inside its zero guard.",
            raw_margins,
        )
    if bool(size_margin <= SIZE_GUARD_ANGSTROM):
        raise ContinuumChaBranchError(
            ChaBranchFailureReason.SIZE_SWITCH,
            "The CHA size branch is at its hard switch guard.",
            raw_margins,
        )
    return ChaBranchDiagnostics(
        weighted_signs_e=polar.effective_charges_e,
        absolute_sign_margins_e=sign_margins,
        electrostatic_size_angstrom=polar.electrostatic_size_angstrom,
        size_switch_margin_angstrom=size_margin,
    )


def _polar_with_structured_branch_failure(
    positions, charges, cha_radii, inverse_born
) -> ChaPolarAlgebraResult:
    """Preserve frozen algebra while typing its early AD branch rejection.

    The frozen polar routine validates active AD branches before returning its
    diagnostic tensors.  On only those two known guard failures, reevaluate
    the identical scalar inputs detached and without AD solely to recover the
    same single-point margins, then raise a structured error.  No detached
    energy is ever returned and no force fallback exists.
    """
    try:
        return cha_polar_from_inverse_born(positions, charges, cha_radii, inverse_born)
    except ValueError as error:
        message = str(error)
        if not (
            "effective-charge sign switch" in message or "size-shift switch" in message
        ):
            raise
        detached = cha_polar_from_inverse_born(
            positions.detach(),
            charges.detach(),
            cha_radii.detach(),
            inverse_born.detach(),
        )
        try:
            _branch_diagnostics(detached, charges.detach())
        except ContinuumChaBranchError as structured:
            raise structured from error
        raise


def continuum_cha_scalar(
    positions_angstrom: Tensor,
    topology: ContinuumChaTopology,
    *,
    expected_topology_sha256: str,
    order: int = 48,
) -> ContinuumChaScalarResult:
    """Evaluate the complete continuum scalar at one certified three-site point."""
    import torch

    if not isinstance(topology, ContinuumChaTopology):
        raise TypeError("topology must be a ContinuumChaTopology.")
    topology.assert_current(expected_topology_sha256)
    if isinstance(order, bool) or not isinstance(order, int):
        raise TypeError("order must be an integer.")
    if order < 8:
        raise ValueError("order must be at least 8.")
    if order > MAX_QUADRATURE_ORDER:
        raise ValueError(
            f"order must be at most {MAX_QUADRATURE_ORDER}; restored quadratures "
            "form dense order-squared Gauss-Legendre work arrays."
        )
    if (
        not isinstance(positions_angstrom, torch.Tensor)
        or positions_angstrom.dtype != torch.float64
        or positions_angstrom.shape != (topology.atom_count, 3)
        or not bool(torch.isfinite(positions_angstrom).all())
    ):
        raise ValueError("positions must be a finite float64 [N,3] tensor.")

    fixed = topology.tensors(
        expected_content_sha256=expected_topology_sha256,
        device=str(positions_angstrom.device),
    )
    point_domain = certify_three_site_domain(
        positions_angstrom, fixed.cha_radii_angstrom
    )
    if isinstance(point_domain, DomainCertificationFailure):
        raise ContinuumChaDomainError(point_domain)

    inverse_born = _r6_inverse_born(
        positions_angstrom, fixed.cha_radii_angstrom, point_domain, order
    )
    polar = _polar_with_structured_branch_failure(
        positions_angstrom,
        fixed.charges_e,
        fixed.cha_radii_angstrom,
        inverse_born,
    )
    cha = _branch_diagnostics(polar, fixed.charges_e)
    nonpolar_atol = 1.0e-11
    nonpolar_rtol = 1.0e-11
    cavity = cavity_from_rmin(
        positions_angstrom,
        fixed.lj_rmin_angstrom,
        atol=nonpolar_atol,
        rtol=nonpolar_rtol,
    )
    dispersion = dispersion_from_rmin_epsilon(
        positions_angstrom,
        fixed.lj_rmin_angstrom,
        fixed.lj_epsilon_kcal_mol,
        atol=nonpolar_atol,
        rtol=nonpolar_rtol,
        phi_order=order,
    )
    total = polar.polar_kcal_mol + cavity.energy_kcal_mol + dispersion.energy_kcal_mol
    residual = float(
        (
            total
            - (
                polar.polar_kcal_mol
                + cavity.energy_kcal_mol
                + dispersion.energy_kcal_mol
            )
        )
        .detach()
        .abs()
    )
    return ContinuumChaScalarResult(
        polar_kcal_mol=polar.polar_kcal_mol,
        cavity_kcal_mol=cavity.energy_kcal_mol,
        dispersion_kcal_mol=dispersion.energy_kcal_mol,
        total_kcal_mol=total,
        point_domain=point_domain,
        cha=cha,
        quadrature_identity=ContinuumChaQuadratureIdentity(
            r6_order=order,
            phi_order=order,
            nonpolar_atol=nonpolar_atol,
            nonpolar_rtol=nonpolar_rtol,
            component_sum_residual_kcal_mol=residual,
        ),
        resources=ContinuumChaResourceAccounting(
            site_count=topology.atom_count,
            active_pair_count=len(point_domain.active_pairs),
            cavity_area_evaluations=cavity.area_evaluations,
            dispersion_z_nodes_evaluated=dispersion.z_nodes_evaluated,
        ),
        radius_provenance=ContinuumChaRadiusProvenance(),
    )
