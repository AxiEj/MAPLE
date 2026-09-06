"""Sparse variable-N factor for a passive two-width radial response.

The only public operation in this module is ``B(R) @ field``.  A scalar model
uses ``-1/2 ||B field||^2`` so its susceptibility is structurally
``-B.T @ B``.  No hardness matrix is inverted and no dense variable-N response
matrix is predicted.

Torch is imported lazily to keep dependency-light MAPLE imports unchanged.
The caller owns smooth geometry descriptors and cutoff values; this module
owns the exact radial-null, vector, and edge contractions.
"""

from __future__ import annotations


def _torch():
    return __import__("torch")


def sparse_passive_radial_factor_torch(
    *,
    radial_field,
    positions_angstrom,
    edge_index,
    atomic_polarizability,
    radial_gain,
    vector_coefficients,
    edge_coefficients,
    edge_cutoff,
    scalar_radial_reshape,
    vector_radial_reshape,
):
    """Apply the minimum atom/edge factor to one physical radial field.

    Shapes are

    ``radial_field`` ``(N,8)``;
    ``positions_angstrom`` ``(N,3)``;
    ``edge_index`` ``(2,E)`` with one orientation per unordered edge;
    ``atomic_polarizability`` ``(N,3,3)``;
    ``radial_gain`` ``(N,)``;
    ``vector_coefficients`` ``(N,2,2,2)`` for row/radial/(I,alpha);
    ``edge_coefficients`` ``(E,2)`` for scalar/vector terms; and
    ``edge_cutoff`` ``(E,)``.

    The vector output rows remain Cartesian vectors.  Flattening them inside a
    Euclidean norm is rotation invariant.
    """

    torch = _torch()
    floating = {
        "radial_field": radial_field,
        "positions_angstrom": positions_angstrom,
        "atomic_polarizability": atomic_polarizability,
        "radial_gain": radial_gain,
        "vector_coefficients": vector_coefficients,
        "edge_coefficients": edge_coefficients,
        "edge_cutoff": edge_cutoff,
        "scalar_radial_reshape": scalar_radial_reshape,
        "vector_radial_reshape": vector_radial_reshape,
    }
    if not all(
        torch.is_tensor(value) and torch.is_floating_point(value)
        for value in floating.values()
    ):
        raise TypeError("all factor values except edge_index must be floating tensors.")
    reference = radial_field
    if not all(
        value.dtype == reference.dtype and value.device == reference.device
        for value in floating.values()
    ):
        raise ValueError("factor tensors must share dtype and device.")
    if not torch.is_tensor(edge_index) or torch.is_floating_point(edge_index):
        raise TypeError("edge_index must be an integer Torch tensor.")
    if edge_index.device != reference.device:
        raise ValueError("edge_index must share the factor device.")
    atom_count = int(radial_field.shape[0]) if radial_field.ndim == 2 else -1
    edge_count = int(edge_index.shape[1]) if edge_index.ndim == 2 else -1
    expected = {
        "radial_field": (atom_count, 8),
        "positions_angstrom": (atom_count, 3),
        "atomic_polarizability": (atom_count, 3, 3),
        "radial_gain": (atom_count,),
        "vector_coefficients": (atom_count, 2, 2, 2),
        "edge_coefficients": (edge_count, 2),
        "edge_cutoff": (edge_count,),
        "scalar_radial_reshape": (),
        "vector_radial_reshape": (),
    }
    if atom_count < 1 or edge_count < 0 or edge_index.shape != (2, edge_count):
        raise ValueError("radial field or edge-index shape is invalid.")
    for name, shape in expected.items():
        if tuple(floating[name].shape) != shape:
            raise ValueError(f"{name} must have shape {shape}.")
        if not bool(torch.isfinite(floating[name]).all()):
            raise ValueError(f"{name} must be finite.")
    if edge_index.numel():
        if bool(torch.any(edge_index < 0)) or bool(torch.any(edge_index >= atom_count)):
            raise ValueError("edge_index contains an atom outside the system.")
        if bool(torch.any(edge_index[0] == edge_index[1])):
            raise ValueError("self edges are not allowed.")
    if bool(torch.any(edge_cutoff < 0)):
        raise ValueError("edge_cutoff must be nonnegative.")

    root_two = torch.sqrt(torch.as_tensor(2.0, dtype=reference.dtype, device=reference.device))
    moment = torch.full((2,), 0.5, dtype=reference.dtype, device=reference.device)
    radial_null = torch.stack((1.0 / root_two, -1.0 / root_two))
    scalar_mix = moment + scalar_radial_reshape * radial_null
    vector_mix = moment + vector_radial_reshape * radial_null

    phi = radial_field[:, :2]
    # Public raw l=1 order is y,z,x; convert both radial blocks to x,y,z.
    electric = torch.stack(
        (
            radial_field[:, (4, 2, 3)],
            radial_field[:, (7, 5, 6)],
        ),
        dim=1,
    )
    alpha = 0.5 * (
        atomic_polarizability + atomic_polarizability.transpose(-1, -2)
    )

    radial_rows = radial_gain * torch.einsum("na,a->n", phi, radial_null)
    alpha_electric = torch.einsum("nij,naj->nai", alpha, electric)
    identity_coefficients = vector_coefficients[..., 0]
    alpha_coefficients = vector_coefficients[..., 1]
    vector_rows = (
        torch.einsum("nka,nac->nkc", identity_coefficients, electric)
        + torch.einsum("nka,nac->nkc", alpha_coefficients, alpha_electric)
    )

    if edge_count:
        atom_i = edge_index[0].to(dtype=torch.long)
        atom_j = edge_index[1].to(dtype=torch.long)
        displacement = positions_angstrom[atom_j] - positions_angstrom[atom_i]
        distance = torch.linalg.vector_norm(displacement, dim=1)
        if bool(torch.any(distance <= 0)):
            raise ValueError("edge endpoints must have positive separation.")
        direction = displacement / distance[:, None]
        scalar_field = torch.einsum("na,a->n", phi, scalar_mix)
        vector_field = torch.einsum("nac,a->nc", electric, vector_mix)
        scalar_difference = scalar_field[atom_i] - scalar_field[atom_j]
        average_vector = 0.5 * (vector_field[atom_i] + vector_field[atom_j])
        longitudinal_vector = torch.einsum("ec,ec->e", direction, average_vector)
        edge_rows = edge_cutoff * (
            edge_coefficients[:, 0] * scalar_difference
            + edge_coefficients[:, 1] * longitudinal_vector
        )
    else:
        edge_rows = torch.empty((0,), dtype=reference.dtype, device=reference.device)

    return torch.cat(
        (
            radial_rows.reshape(-1),
            vector_rows.reshape(-1),
            edge_rows.reshape(-1),
        )
    )


def sparse_passive_p13_factor_torch(
    *,
    l2_field,
    l2_gain,
    **radial_factor_inputs,
):
    """Apply the radial sparse factor plus one local isotropic l2 row block.

    ``l2_field`` has shape ``(N,5)`` in the orthonormal STF basis and
    ``l2_gain`` has shape ``(N,)``.  The initial P13 response is block diagonal:
    it adds local induced quadrupoles without an unvalidated radial-l2 cross
    susceptibility.  Both sectors remain part of one squared factor norm.
    """

    torch = _torch()
    radial_field = radial_factor_inputs.get("radial_field")
    if not torch.is_tensor(radial_field):
        raise TypeError("radial_factor_inputs must include radial_field.")
    if not (
        torch.is_tensor(l2_field)
        and torch.is_floating_point(l2_field)
        and torch.is_tensor(l2_gain)
        and torch.is_floating_point(l2_gain)
    ):
        raise TypeError("l2 field and gain must be floating Torch tensors.")
    atom_count = int(radial_field.shape[0])
    if (
        tuple(l2_field.shape) != (atom_count, 5)
        or tuple(l2_gain.shape) != (atom_count,)
        or l2_field.dtype != radial_field.dtype
        or l2_gain.dtype != radial_field.dtype
        or l2_field.device != radial_field.device
        or l2_gain.device != radial_field.device
        or not bool(torch.isfinite(l2_field).all())
        or not bool(torch.isfinite(l2_gain).all())
    ):
        raise ValueError("l2 factor shapes, values, dtype, or device are invalid.")
    radial_rows = sparse_passive_radial_factor_torch(**radial_factor_inputs)
    l2_rows = l2_gain[:, None] * l2_field
    return torch.cat((radial_rows, l2_rows.reshape(-1)))


__all__ = [
    "sparse_passive_p13_factor_torch",
    "sparse_passive_radial_factor_torch",
]
