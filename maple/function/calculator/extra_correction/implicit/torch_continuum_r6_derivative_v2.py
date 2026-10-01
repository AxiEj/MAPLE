"""R6 derivative-v2 assembly: new contact root plus unchanged admitted tori."""

from __future__ import annotations

from .torch_continuum_chagb_domain import CertifiedLocalPatchScope
from .torch_continuum_r6_contact_v2 import contact_patch_inverse_cube_v2
from .torch_continuum_r6_patches import pair_torus_inverse_cube


def _r6_inverse_born_v2(positions, intrinsic_radii, domain, order):
    """Assemble live inverse Born radii for an admitted three-site point."""
    import torch

    if not isinstance(domain, CertifiedLocalPatchScope):
        raise TypeError("domain must be a CertifiedLocalPatchScope")
    contact = contact_patch_inverse_cube_v2(positions, intrinsic_radii, order=order)
    inverse_cube = contact.inverse_cube_per_angstrom3
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
        raise RuntimeError(
            "Certified contact-v2 plus unchanged tori did not yield positive inverse cubes."
        )
    return inverse_cube.pow(1.0 / 3.0)
