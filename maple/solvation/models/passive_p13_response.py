"""Observable helpers for the block-passive P13 response scalar.

The response energy is defined only through the factor norm

``E_response(eta) = -1/2 ||C_theta(R) eta||^2``.

The induced radial and quadrupole sources are gradients of that scalar.  This
module intentionally does not expose a separately parameterized response
source.  It also provides the exact total-charge and molecular-dipole maps for
the repository's two-width radial-GTO source convention.

Torch is imported lazily so dependency-light MAPLE imports remain unchanged.
"""

from __future__ import annotations

from .sparse_passive_factor import sparse_passive_p13_factor_torch


def _torch():
    return __import__("torch")


def radial_source_total_charge_torch(radial_source):
    """Return total charge for ``(..., N, 8)`` radial source tensors."""

    torch = _torch()
    if not torch.is_tensor(radial_source) or not torch.is_floating_point(
        radial_source
    ):
        raise TypeError("radial_source must be a floating Torch tensor.")
    if radial_source.ndim < 2 or radial_source.shape[-1] != 8:
        raise ValueError("radial_source must have shape (..., N, 8).")
    if not bool(torch.isfinite(radial_source).all()):
        raise ValueError("radial_source must be finite.")
    return torch.sum(radial_source[..., 0] + radial_source[..., 1], dim=-1)


def radial_source_molecular_dipole_torch(
    *,
    radial_source,
    positions_angstrom,
):
    """Return the physical molecular dipole in ``e*angstrom``.

    The public raw real-``l=1`` order is ``(y,z,x)`` for each radial block.
    Both scalar radial channels contribute to the physical atomic charge and
    both vector radial channels contribute to the physical atomic dipole.
    """

    torch = _torch()
    if not (
        torch.is_tensor(radial_source)
        and torch.is_floating_point(radial_source)
        and torch.is_tensor(positions_angstrom)
        and torch.is_floating_point(positions_angstrom)
    ):
        raise TypeError("source and positions must be floating Torch tensors.")
    if radial_source.ndim < 2 or radial_source.shape[-1] != 8:
        raise ValueError("radial_source must have shape (..., N, 8).")
    atom_count = int(radial_source.shape[-2])
    if tuple(positions_angstrom.shape) != (atom_count, 3):
        raise ValueError("positions_angstrom must have shape (N,3).")
    if (
        radial_source.dtype != positions_angstrom.dtype
        or radial_source.device != positions_angstrom.device
        or not bool(torch.isfinite(radial_source).all())
        or not bool(torch.isfinite(positions_angstrom).all())
    ):
        raise ValueError("source and positions must share finite dtype/device.")

    charge = radial_source[..., 0] + radial_source[..., 1]
    atomic_dipole = torch.stack(
        (
            radial_source[..., 4] + radial_source[..., 7],
            radial_source[..., 2] + radial_source[..., 5],
            radial_source[..., 3] + radial_source[..., 6],
        ),
        dim=-1,
    )
    return torch.einsum("...n,nc->...c", charge, positions_angstrom) + torch.sum(
        atomic_dipole,
        dim=-2,
    )


def p13_response_mep_torch(
    *,
    radial_source,
    l2_source,
    radial_operator,
    l2_operator,
):
    """Project one or more P13 sources to probe MEP values."""

    torch = _torch()
    tensors = (radial_source, l2_source, radial_operator, l2_operator)
    if not all(
        torch.is_tensor(value) and torch.is_floating_point(value)
        for value in tensors
    ):
        raise TypeError("P13 source and operator values must be floating tensors.")
    if radial_source.ndim < 2 or radial_source.shape[-1] != 8:
        raise ValueError("radial_source must have shape (...,N,8).")
    if l2_source.shape[:-1] != radial_source.shape[:-1] or l2_source.shape[-1] != 5:
        raise ValueError("l2_source must have shape (...,N,5).")
    atom_count = int(radial_source.shape[-2])
    if (
        radial_operator.ndim != 2
        or l2_operator.ndim != 2
        or radial_operator.shape[0] != l2_operator.shape[0]
        or radial_operator.shape[1] != 8 * atom_count
        or l2_operator.shape[1] != 5 * atom_count
    ):
        raise ValueError("P13 operator dimensions do not match the source.")
    reference = radial_source
    if not all(
        value.dtype == reference.dtype and value.device == reference.device
        for value in tensors
    ) or not all(bool(torch.isfinite(value).all()) for value in tensors):
        raise ValueError("P13 source/operator tensors must share finite dtype/device.")
    radial_flat = radial_source.reshape(*radial_source.shape[:-2], 8 * atom_count)
    l2_flat = l2_source.reshape(*l2_source.shape[:-2], 5 * atom_count)
    return (
        torch.einsum("pd,...d->...p", radial_operator, radial_flat)
        + torch.einsum("pd,...d->...p", l2_operator, l2_flat)
    )


def p13_induced_source_from_scalar_torch(
    *,
    radial_fields,
    l2_fields,
    l2_gain,
    create_graph: bool,
    **radial_factor_inputs,
):
    """Differentiate the block-passive scalar for a batch of field modes.

    ``radial_fields`` and ``l2_fields`` have shapes ``(M,N,8)`` and
    ``(M,N,5)``.  Geometry-dependent factor coefficients are shared across the
    modes.  Differentiating the sum of independent mode energies yields the
    response source for every mode in one reverse pass.
    """

    torch = _torch()
    if type(create_graph) is not bool:
        raise TypeError("create_graph must be exactly bool.")
    if not (
        torch.is_tensor(radial_fields)
        and torch.is_floating_point(radial_fields)
        and torch.is_tensor(l2_fields)
        and torch.is_floating_point(l2_fields)
    ):
        raise TypeError("P13 fields must be floating Torch tensors.")
    if radial_fields.ndim != 3 or radial_fields.shape[-1] != 8:
        raise ValueError("radial_fields must have shape (M,N,8).")
    mode_count, atom_count, _ = radial_fields.shape
    if tuple(l2_fields.shape) != (mode_count, atom_count, 5):
        raise ValueError("l2_fields must have shape (M,N,5).")
    if mode_count < 1 or atom_count < 1:
        raise ValueError("at least one field mode and atom are required.")
    if (
        radial_fields.dtype != l2_fields.dtype
        or radial_fields.device != l2_fields.device
        or not bool(torch.isfinite(radial_fields).all())
        or not bool(torch.isfinite(l2_fields).all())
    ):
        raise ValueError("P13 fields must share finite dtype/device.")

    if not radial_fields.requires_grad:
        radial_fields.requires_grad_(True)
    if not l2_fields.requires_grad:
        l2_fields.requires_grad_(True)
    mode_energies = []
    for mode in range(mode_count):
        factor = sparse_passive_p13_factor_torch(
            **radial_factor_inputs,
            radial_field=radial_fields[mode],
            l2_field=l2_fields[mode],
            l2_gain=l2_gain,
        )
        mode_energies.append(-0.5 * torch.dot(factor, factor))
    energies = torch.stack(mode_energies)
    radial_source, l2_source = torch.autograd.grad(
        torch.sum(energies),
        (radial_fields, l2_fields),
        create_graph=create_graph,
        retain_graph=create_graph,
    )
    return energies, radial_source, l2_source


def p13_induced_source_analytic_torch(
    *,
    radial_fields,
    l2_fields,
    positions_angstrom,
    edge_index,
    atomic_polarizability,
    radial_gain,
    vector_coefficients,
    edge_coefficients,
    edge_cutoff,
    scalar_radial_reshape,
    vector_radial_reshape,
    l2_gain,
):
    """Return the exact algebraic derivative of the P13 factor scalar.

    This is a performance backend for the same
    ``-1/2 ||C_theta(R) eta||^2`` scalar, not an independently parameterized
    source.  The implementation is vectorized over the leading field-mode
    dimension and remains differentiable with respect to all factor and
    geometry inputs.
    """

    torch = _torch()
    values = {
        "radial_fields": radial_fields,
        "l2_fields": l2_fields,
        "positions_angstrom": positions_angstrom,
        "atomic_polarizability": atomic_polarizability,
        "radial_gain": radial_gain,
        "vector_coefficients": vector_coefficients,
        "edge_coefficients": edge_coefficients,
        "edge_cutoff": edge_cutoff,
        "scalar_radial_reshape": scalar_radial_reshape,
        "vector_radial_reshape": vector_radial_reshape,
        "l2_gain": l2_gain,
    }
    if not all(
        torch.is_tensor(value) and torch.is_floating_point(value)
        for value in values.values()
    ):
        raise TypeError("analytic P13 values except edge_index must be floating tensors.")
    if radial_fields.ndim != 3 or radial_fields.shape[-1] != 8:
        raise ValueError("radial_fields must have shape (M,N,8).")
    mode_count, atom_count, _ = radial_fields.shape
    if tuple(l2_fields.shape) != (mode_count, atom_count, 5):
        raise ValueError("l2_fields must have shape (M,N,5).")
    edge_count = int(edge_index.shape[1]) if edge_index.ndim == 2 else -1
    expected = {
        "positions_angstrom": (atom_count, 3),
        "atomic_polarizability": (atom_count, 3, 3),
        "radial_gain": (atom_count,),
        "vector_coefficients": (atom_count, 2, 2, 2),
        "edge_coefficients": (edge_count, 2),
        "edge_cutoff": (edge_count,),
        "scalar_radial_reshape": (),
        "vector_radial_reshape": (),
        "l2_gain": (atom_count,),
    }
    if mode_count < 1 or atom_count < 1 or edge_count < 0:
        raise ValueError("analytic P13 dimensions are invalid.")
    if not torch.is_tensor(edge_index) or torch.is_floating_point(edge_index):
        raise TypeError("edge_index must be an integer Torch tensor.")
    reference = radial_fields
    if edge_index.device != reference.device:
        raise ValueError("edge_index must share the P13 device.")
    if not all(
        value.dtype == reference.dtype and value.device == reference.device
        for value in values.values()
    ):
        raise ValueError("analytic P13 tensors must share dtype/device.")
    for name, shape in expected.items():
        if tuple(values[name].shape) != shape:
            raise ValueError(f"{name} must have shape {shape}.")
    if not all(bool(torch.isfinite(value).all()) for value in values.values()):
        raise ValueError("analytic P13 tensors must be finite.")
    if edge_count:
        if bool(torch.any(edge_index < 0)) or bool(torch.any(edge_index >= atom_count)):
            raise ValueError("edge_index contains an atom outside the system.")
        if bool(torch.any(edge_index[0] == edge_index[1])):
            raise ValueError("self edges are not allowed.")

    root_two = torch.sqrt(
        torch.as_tensor(2.0, dtype=reference.dtype, device=reference.device)
    )
    moment = torch.full((2,), 0.5, dtype=reference.dtype, device=reference.device)
    radial_null = torch.stack((1.0 / root_two, -1.0 / root_two))
    scalar_mix = moment + scalar_radial_reshape * radial_null
    vector_mix = moment + vector_radial_reshape * radial_null

    phi = radial_fields[..., :2]
    electric = torch.stack(
        (
            radial_fields[..., (4, 2, 3)],
            radial_fields[..., (7, 5, 6)],
        ),
        dim=2,
    )
    alpha = 0.5 * (
        atomic_polarizability + atomic_polarizability.transpose(-1, -2)
    )

    radial_projection = torch.einsum("mna,a->mn", phi, radial_null)
    radial_rows = radial_gain[None, :] * radial_projection
    phi_source = -(
        radial_gain.square()[None, :, None]
        * radial_projection[:, :, None]
        * radial_null[None, None, :]
    )

    alpha_electric = torch.einsum("nij,mnaj->mnai", alpha, electric)
    identity_coefficients = vector_coefficients[..., 0]
    alpha_coefficients = vector_coefficients[..., 1]
    vector_rows = (
        torch.einsum("nka,mnac->mnkc", identity_coefficients, electric)
        + torch.einsum("nka,mnac->mnkc", alpha_coefficients, alpha_electric)
    )
    electric_source = -(
        torch.einsum("nka,mnkc->mnac", identity_coefficients, vector_rows)
        + torch.einsum(
            "nka,nij,mnki->mnaj",
            alpha_coefficients,
            alpha,
            vector_rows,
        )
    )

    if edge_count:
        atom_i = edge_index[0].to(dtype=torch.long)
        atom_j = edge_index[1].to(dtype=torch.long)
        displacement = positions_angstrom[atom_j] - positions_angstrom[atom_i]
        distance = torch.linalg.vector_norm(displacement, dim=1)
        if bool(torch.any(distance <= 0)):
            raise ValueError("edge endpoints must have positive separation.")
        direction = displacement / distance[:, None]
        scalar_field = torch.einsum("mna,a->mn", phi, scalar_mix)
        vector_field = torch.einsum("mnac,a->mnc", electric, vector_mix)
        scalar_difference = scalar_field[:, atom_i] - scalar_field[:, atom_j]
        average_vector = 0.5 * (
            vector_field[:, atom_i] + vector_field[:, atom_j]
        )
        longitudinal_vector = torch.einsum(
            "ec,mec->me", direction, average_vector
        )
        edge_rows = edge_cutoff[None, :] * (
            edge_coefficients[None, :, 0] * scalar_difference
            + edge_coefficients[None, :, 1] * longitudinal_vector
        )
        row_cotangent = -edge_rows * edge_cutoff[None, :]
        scalar_edge = (
            row_cotangent[:, :, None]
            * edge_coefficients[None, :, 0, None]
            * scalar_mix[None, None, :]
        )
        phi_source = phi_source.index_add(1, atom_i, scalar_edge)
        phi_source = phi_source.index_add(1, atom_j, -scalar_edge)
        vector_edge = (
            0.5
            * row_cotangent[:, :, None, None]
            * edge_coefficients[None, :, 1, None, None]
            * vector_mix[None, None, :, None]
            * direction[None, :, None, :]
        )
        electric_source = electric_source.index_add(1, atom_i, vector_edge)
        electric_source = electric_source.index_add(1, atom_j, vector_edge)
    else:
        edge_rows = torch.empty(
            (mode_count, 0), dtype=reference.dtype, device=reference.device
        )

    first_raw = torch.stack(
        (
            electric_source[:, :, 0, 1],
            electric_source[:, :, 0, 2],
            electric_source[:, :, 0, 0],
        ),
        dim=2,
    )
    second_raw = torch.stack(
        (
            electric_source[:, :, 1, 1],
            electric_source[:, :, 1, 2],
            electric_source[:, :, 1, 0],
        ),
        dim=2,
    )
    radial_source = torch.cat((phi_source, first_raw, second_raw), dim=2)
    l2_rows = l2_gain[None, :, None] * l2_fields
    l2_source = -l2_gain.square()[None, :, None] * l2_fields
    energies = -0.5 * (
        torch.sum(radial_rows.square(), dim=1)
        + torch.sum(vector_rows.square(), dim=(1, 2, 3))
        + torch.sum(edge_rows.square(), dim=1)
        + torch.sum(l2_rows.square(), dim=(1, 2))
    )
    return energies, radial_source, l2_source


__all__ = [
    "p13_induced_source_analytic_torch",
    "p13_induced_source_from_scalar_torch",
    "p13_response_mep_torch",
    "radial_source_molecular_dipole_torch",
    "radial_source_total_charge_torch",
]
