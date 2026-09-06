"""Scalar-only P13/R8 split-order electrostatic head algebra.

The permanent source is ``radial8 + l2`` while the induced response is radial8
only.  The l2 field enters affinely, so its susceptibility and every radial-l2
cross block are exactly zero.  Point versus fixed-width Gaussian l2 is a
separate analytic source/receiver choice and does not alter this scalar
identity.
"""

from __future__ import annotations


def _torch():
    return __import__("torch")


def split_order_passive_energy_torch(
    *,
    vacuum_energy,
    permanent_radial_source,
    permanent_l2_source,
    radial_field,
    l2_field,
    radial_response_factor,
):
    """Return the sole P13/R8 scalar.

    Shapes are ``(N,8)`` for radial source/field, ``(N,5)`` for l2
    source/field, and any nonempty vector for ``radial_response_factor``.
    The factor must be computed from the radial field by the same live graph.
    """

    torch = _torch()
    values = {
        "vacuum_energy": vacuum_energy,
        "permanent_radial_source": permanent_radial_source,
        "permanent_l2_source": permanent_l2_source,
        "radial_field": radial_field,
        "l2_field": l2_field,
        "radial_response_factor": radial_response_factor,
    }
    if not all(
        torch.is_tensor(value) and torch.is_floating_point(value)
        for value in values.values()
    ):
        raise TypeError("split-order scalar inputs must be floating Torch tensors.")
    reference = radial_field
    if not all(
        value.dtype == reference.dtype and value.device == reference.device
        for value in values.values()
    ):
        raise ValueError("split-order scalar inputs must share dtype and device.")
    atom_count = int(radial_field.shape[0]) if radial_field.ndim == 2 else -1
    if (
        atom_count < 1
        or tuple(radial_field.shape) != (atom_count, 8)
        or tuple(permanent_radial_source.shape) != (atom_count, 8)
        or tuple(l2_field.shape) != (atom_count, 5)
        or tuple(permanent_l2_source.shape) != (atom_count, 5)
        or vacuum_energy.ndim != 0
        or radial_response_factor.ndim != 1
        or radial_response_factor.numel() < 1
    ):
        raise ValueError("split-order scalar input shapes are inconsistent.")
    if not all(bool(torch.isfinite(value).all()) for value in values.values()):
        raise ValueError("split-order scalar inputs must be finite.")
    return (
        vacuum_energy
        + torch.sum(permanent_radial_source * radial_field)
        + torch.sum(permanent_l2_source * l2_field)
        - 0.5 * torch.dot(radial_response_factor, radial_response_factor)
    )


def p13_passive_energy_torch(
    *,
    vacuum_energy,
    permanent_radial_source,
    permanent_l2_source,
    radial_field,
    l2_field,
    full_response_factor,
):
    """Return a P13 scalar whose squared factor may depend on both fields.

    The first full-response candidate uses a block factor (radial sparse rows
    plus local l2 rows).  A future cross factor would still enter through this
    same scalar; no separate source implementation is allowed.
    """

    return split_order_passive_energy_torch(
        vacuum_energy=vacuum_energy,
        permanent_radial_source=permanent_radial_source,
        permanent_l2_source=permanent_l2_source,
        radial_field=radial_field,
        l2_field=l2_field,
        radial_response_factor=full_response_factor,
    )


__all__ = ["p13_passive_energy_torch", "split_order_passive_energy_torch"]
