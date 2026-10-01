"""Complete three-site continuum scalar for Gaussian-sign CHA v1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from .continuum_chagb_inputs import ContinuumChaTopology
from .gaussian_cha_profiles import (
    GAUSSIAN_CHA_R6_V1_PROFILE_ID,
    R6BackendDiagnostics,
    resolve_gaussian_cha_profile,
)
from .torch_chagb_gaussian import (
    GAUSSIAN_CHA_MODEL_IDENTITY,
    GaussianChaPolarAlgebraResult,
    gaussian_cha_polar_from_inverse_born,
    validate_gaussian_sigma_e,
)
from .torch_continuum_chagb import (
    MAX_QUADRATURE_ORDER,
    ContinuumChaDomainError,
    ContinuumChaQuadratureIdentity,
    ContinuumChaRadiusProvenance,
    ContinuumChaResourceAccounting,
)
from .torch_continuum_chagb_domain import (
    CertifiedLocalPatchScope,
    DomainCertificationFailure,
    certify_three_site_domain,
)
from .torch_continuum_dispersion import dispersion_from_rmin_epsilon
from .torch_continuum_sav import cavity_from_rmin

if TYPE_CHECKING:
    from torch import Tensor


@dataclass(frozen=True)
class GaussianChaDiagnostics:
    """Live polar intermediates needed to audit smoothing and Born response."""

    weighted_signs_e: Tensor
    smoothed_signs: Tensor
    charge_sigma_e: float
    weighted_signs_over_sigma: Tensor
    gaussian_erfc_tails: Tensor
    gaussian_erfcx_tails: Tensor
    electrostatic_size_angstrom: Tensor
    born_radii_angstrom: Tensor
    cha_factors: Tensor


@dataclass(frozen=True)
class ContinuumGaussianChaScalarResult:
    polar_kcal_mol: Tensor
    cavity_kcal_mol: Tensor
    dispersion_kcal_mol: Tensor
    total_kcal_mol: Tensor
    point_domain: CertifiedLocalPatchScope
    cha: GaussianChaDiagnostics
    quadrature_identity: ContinuumChaQuadratureIdentity
    resources: ContinuumChaResourceAccounting
    radius_provenance: ContinuumChaRadiusProvenance
    numerical_profile_id: str
    r6_backend_diagnostics: R6BackendDiagnostics
    scope: ClassVar[str] = "unregistered-three-site-gaussian-sign-point-scalar"
    model_identity: ClassVar[str] = GAUSSIAN_CHA_MODEL_IDENTITY
    supported_properties: ClassVar[tuple[str, ...]] = ()


def _diagnostics(polar: GaussianChaPolarAlgebraResult) -> GaussianChaDiagnostics:
    return GaussianChaDiagnostics(
        weighted_signs_e=polar.effective_charges_e,
        smoothed_signs=polar.smoothed_signs,
        charge_sigma_e=polar.charge_sigma_e,
        weighted_signs_over_sigma=polar.weighted_signs_over_sigma,
        gaussian_erfc_tails=polar.gaussian_erfc_tails,
        gaussian_erfcx_tails=polar.gaussian_erfcx_tails,
        electrostatic_size_angstrom=polar.electrostatic_size_angstrom,
        born_radii_angstrom=polar.born_radii_angstrom,
        cha_factors=polar.cha_factors,
    )


def continuum_gaussian_cha_scalar(
    positions_angstrom: Tensor,
    topology: ContinuumChaTopology,
    *,
    expected_topology_sha256: str,
    sigma_e: float,
    order: int = 64,
    numerical_profile_id: str = GAUSSIAN_CHA_R6_V1_PROFILE_ID,
) -> ContinuumGaussianChaScalarResult:
    """Evaluate one complete live R6/Born/polar/cavity/dispersion scalar."""
    import torch

    sigma = validate_gaussian_sigma_e(sigma_e)
    numerical_profile = resolve_gaussian_cha_profile(numerical_profile_id)
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
        or positions_angstrom.device.type != "cpu"
        or positions_angstrom.shape != (topology.atom_count, 3)
        or not bool(torch.isfinite(positions_angstrom).all())
    ):
        raise ValueError("positions must be a finite CPU float64 [N,3] tensor.")

    fixed = topology.tensors(
        expected_content_sha256=expected_topology_sha256, device="cpu"
    )
    point_domain = certify_three_site_domain(
        positions_angstrom, fixed.cha_radii_angstrom
    )
    if isinstance(point_domain, DomainCertificationFailure):
        raise ContinuumChaDomainError(point_domain)

    r6_backend = numerical_profile.evaluate_inverse_born(
        positions_angstrom, fixed.cha_radii_angstrom, point_domain, order
    )
    inverse_born = r6_backend.inverse_born_per_angstrom
    polar = gaussian_cha_polar_from_inverse_born(
        positions_angstrom,
        fixed.charges_e,
        fixed.cha_radii_angstrom,
        inverse_born,
        sigma_e=sigma,
    )
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
    return ContinuumGaussianChaScalarResult(
        polar_kcal_mol=polar.polar_kcal_mol,
        cavity_kcal_mol=cavity.energy_kcal_mol,
        dispersion_kcal_mol=dispersion.energy_kcal_mol,
        total_kcal_mol=total,
        point_domain=point_domain,
        cha=_diagnostics(polar),
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
        numerical_profile_id=numerical_profile.profile_id,
        r6_backend_diagnostics=r6_backend.diagnostics,
    )
