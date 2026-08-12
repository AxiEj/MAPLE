"""Independent differentiable Torch C-PCM derivative oracles.

This module is deliberately independent of the production continuum providers.
It imports Torch lazily and contains two explicitly distinct ``torch.float64``
test assets:

* :class:`TorchSyntheticDenseCPCM` is a regularized monopole toy used only for
  generic autograd identities.  It is not production parity.
* :class:`TorchFixedTopologyAmplitudeSWIGCPCMOracle` independently reproduces
  the production fixed-topology amplitude-SWIG C-PCM scalar from an immutable
  surface snapshot and configuration.

The synthetic topology stores reference node positions, positive quadrature weights
and widths, and one owner atom per node.  At another geometry each node follows
its owner by rigid translation; node count, owner assignment, and ordering never
change.  In internally consistent reference units the surface potential and
surface-charge equation are

``v = B c`` and ``A sigma = -f_epsilon v``, where
``f_epsilon = (epsilon - 1) / epsilon``.

``B`` is a smooth weighted regularized-Coulomb matrix.  ``A`` has the same
off-diagonal kernel and a geometry-independent diagonal chosen from a strict
Gershgorin bound, so it is symmetric positive definite for every geometry.
With the positive energy-dual field convention ``field = B.T @ sigma``, the
polarization energy is exactly

``E = 0.5 * c.T @ field = 0.5 * sigma.T @ v``.

The resulting energy is normally negative for a dielectric because of the
minus sign in the C-PCM equation.  Neither asset opens a Route 2 capability;
the parity oracle is validation evidence, not a production provider.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
import json
from typing import Any

import numpy as np


def _torch():
    """Import Torch only when an oracle operation is actually requested."""

    try:
        return importlib.import_module("torch")
    except ImportError as exc:  # pragma: no cover - exercised on no-Torch CI
        raise ImportError(
            "The optional Torch C-PCM test oracles require torch."
        ) from exc


def _immutable_array(
    values: object,
    *,
    name: str,
    dtype: np.dtype[Any],
    shape_tail: tuple[int, ...] = (),
) -> np.ndarray:
    array = np.asarray(values, dtype=dtype)
    if array.ndim != 1 + len(shape_tail) or array.shape[1:] != shape_tail:
        expected = (
            "(N,)" if not shape_tail else f"(N, {', '.join(map(str, shape_tail))})"
        )
        raise ValueError(f"{name} must have shape {expected}; received {array.shape}.")
    if array.shape[0] == 0:
        raise ValueError(f"{name} must be non-empty.")
    if np.issubdtype(array.dtype, np.floating) and not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    result = np.array(array, copy=True)
    result.setflags(write=False)
    return result


def _positive_scalar(value: object, *, name: str, lower: float = 0.0) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real scalar.")
    number = float(value)
    if not np.isfinite(number) or number <= lower:
        raise ValueError(f"{name} must be finite and greater than {lower}.")
    return number


def _immutable_owner_array(values: object) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != 1 or raw.shape[0] == 0:
        raise ValueError(f"node_owners must have shape (N,); received {raw.shape}.")
    try:
        numeric = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError("node_owners must contain only integers.") from exc
    if not np.all(np.isfinite(numeric)) or not np.all(numeric == np.floor(numeric)):
        raise ValueError("node_owners must contain only finite integers.")
    result = numeric.astype(np.int64)
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class FixedCPCMTopology:
    """Immutable atom/node geometry and regularization for the dense oracle."""

    reference_atom_positions: np.ndarray
    node_positions: np.ndarray
    node_weights: np.ndarray
    node_widths: np.ndarray
    node_owners: np.ndarray
    source_widths: np.ndarray
    epsilon: float = 80.0
    spd_margin: float = 1.0
    coulomb_scale: float = 1.0

    def __post_init__(self) -> None:
        atoms = _immutable_array(
            self.reference_atom_positions,
            name="reference_atom_positions",
            dtype=np.dtype(np.float64),
            shape_tail=(3,),
        )
        nodes = _immutable_array(
            self.node_positions,
            name="node_positions",
            dtype=np.dtype(np.float64),
            shape_tail=(3,),
        )
        weights = _immutable_array(
            self.node_weights, name="node_weights", dtype=np.dtype(np.float64)
        )
        widths = _immutable_array(
            self.node_widths, name="node_widths", dtype=np.dtype(np.float64)
        )
        owners = _immutable_owner_array(self.node_owners)
        source_widths = _immutable_array(
            self.source_widths, name="source_widths", dtype=np.dtype(np.float64)
        )
        node_count = nodes.shape[0]
        if (
            weights.shape != (node_count,)
            or widths.shape != (node_count,)
            or owners.shape != (node_count,)
        ):
            raise ValueError(
                "node positions, weights, widths, and owners must have the same node count."
            )
        if source_widths.shape != (atoms.shape[0],):
            raise ValueError("source_widths must contain one width per atom.")
        if (
            np.any(weights <= 0.0)
            or np.any(widths <= 0.0)
            or np.any(source_widths <= 0.0)
        ):
            raise ValueError(
                "all node weights and node/source widths must be positive."
            )
        if np.any(owners < 0) or np.any(owners >= atoms.shape[0]):
            raise ValueError("every node owner must be a valid atom index.")
        object.__setattr__(self, "reference_atom_positions", atoms)
        object.__setattr__(self, "node_positions", nodes)
        object.__setattr__(self, "node_weights", weights)
        object.__setattr__(self, "node_widths", widths)
        object.__setattr__(self, "node_owners", owners)
        object.__setattr__(self, "source_widths", source_widths)
        object.__setattr__(
            self, "epsilon", _positive_scalar(self.epsilon, name="epsilon", lower=1.0)
        )
        object.__setattr__(
            self, "spd_margin", _positive_scalar(self.spd_margin, name="spd_margin")
        )
        object.__setattr__(
            self,
            "coulomb_scale",
            _positive_scalar(self.coulomb_scale, name="coulomb_scale"),
        )

    @property
    def atom_count(self) -> int:
        return self.reference_atom_positions.shape[0]

    @property
    def node_count(self) -> int:
        return self.node_positions.shape[0]

    @property
    def dielectric_factor(self) -> float:
        return (self.epsilon - 1.0) / self.epsilon

    @property
    def topology_hash(self) -> str:
        header = json.dumps(
            {
                "schema": "maple.route2.torch-reference-cpcm.fixed-topology.v1",
                "atom_count": self.atom_count,
                "node_count": self.node_count,
                "epsilon": self.epsilon,
                "spd_margin": self.spd_margin,
                "coulomb_scale": self.coulomb_scale,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        digest = hashlib.sha256(header)
        for array in (
            self.reference_atom_positions,
            self.node_positions,
            self.node_weights,
            self.node_widths,
            self.node_owners,
            self.source_widths,
        ):
            digest.update(array.dtype.str.encode())
            digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
            digest.update(array.tobytes(order="C"))
        return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class CPCMReferenceState:
    """One dense forward evaluation; tensors retain their autograd graph."""

    node_positions: Any
    A: Any
    B: Any
    potential: Any
    surface_charge: Any
    field: Any
    energy: Any
    topology_hash: str


class TorchSyntheticDenseCPCM:
    """Synthetic dense C-PCM derivative fixture, not a Phase-4 parity oracle."""

    provider_id = "maple.route2.oracle.synthetic-dense-cpcm.v1"
    scalar_id = "synthetic-regularized-cpcm-not-production-parity-v1"
    capability_admitted = False

    def __init__(self, topology: FixedCPCMTopology):
        if not isinstance(topology, FixedCPCMTopology):
            raise TypeError("topology must be a FixedCPCMTopology.")
        self.topology = topology

    def _tensor(self, value: object, *, shape: tuple[int, ...], name: str):
        torch = _torch()
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor.")
        if value.dtype != torch.float64:
            raise TypeError(f"{name} must use torch.float64; received {value.dtype}.")
        if tuple(value.shape) != shape:
            raise ValueError(
                f"{name} must have shape {shape}; received {tuple(value.shape)}."
            )
        if not bool(torch.isfinite(value).all().item()):
            raise ValueError(f"{name} must contain only finite values.")
        return value

    def _constant(self, values: np.ndarray, *, like: Any):
        torch = _torch()
        # The topology arrays are intentionally read-only.  Copy here rather
        # than exposing that storage to a writable Torch tensor.
        return torch.tensor(
            np.array(values, copy=True), dtype=torch.float64, device=like.device
        )

    def moved_node_positions(self, atom_positions: Any):
        positions = self._tensor(
            atom_positions,
            shape=(self.topology.atom_count, 3),
            name="atom_positions",
        )
        owners = self._constant(self.topology.node_owners, like=positions).to(
            dtype=_torch().long
        )
        reference_atoms = self._constant(
            self.topology.reference_atom_positions, like=positions
        )
        reference_nodes = self._constant(self.topology.node_positions, like=positions)
        return reference_nodes + positions[owners] - reference_atoms[owners]

    def assemble(self, atom_positions: Any) -> tuple[Any, Any, Any]:
        """Return moved nodes, SPD ``A``, and source-to-surface ``B``."""

        torch = _torch()
        positions = self._tensor(
            atom_positions,
            shape=(self.topology.atom_count, 3),
            name="atom_positions",
        )
        nodes = self.moved_node_positions(positions)
        weights = self._constant(self.topology.node_weights, like=positions)
        node_widths = self._constant(self.topology.node_widths, like=positions)
        source_widths = self._constant(self.topology.source_widths, like=positions)
        sqrt_weights = torch.sqrt(weights)

        node_source_delta = nodes[:, None, :] - positions[None, :, :]
        node_source_r2 = torch.sum(node_source_delta * node_source_delta, dim=-1)
        b_softening2 = node_widths[:, None] ** 2 + source_widths[None, :] ** 2
        B = (
            self.topology.coulomb_scale
            * sqrt_weights[:, None]
            / torch.sqrt(node_source_r2 + b_softening2)
        )

        node_delta = nodes[:, None, :] - nodes[None, :, :]
        node_r2 = torch.sum(node_delta * node_delta, dim=-1)
        a_softening2 = node_widths[:, None] ** 2 + node_widths[None, :] ** 2
        kernel = (
            self.topology.coulomb_scale
            * sqrt_weights[:, None]
            * sqrt_weights[None, :]
            / torch.sqrt(node_r2 + a_softening2)
        )
        mask = ~torch.eye(
            self.topology.node_count, dtype=torch.bool, device=positions.device
        )
        off_diagonal = torch.where(mask, kernel, torch.zeros_like(kernel))

        # Geometry-independent upper bound for every off-diagonal entry.  The
        # resulting strict row diagonal dominance proves A is SPD because A is
        # real symmetric and has positive diagonal.
        upper = (
            self.topology.coulomb_scale
            * sqrt_weights[:, None]
            * sqrt_weights[None, :]
            / torch.sqrt(a_softening2)
        )
        row_bound = torch.sum(torch.where(mask, upper, torch.zeros_like(upper)), dim=1)
        diagonal = row_bound + self.topology.spd_margin
        A = off_diagonal + torch.diag(diagonal)
        return nodes, A, B

    def solve(self, atom_positions: Any, source: Any) -> CPCMReferenceState:
        torch = _torch()
        positions = self._tensor(
            atom_positions,
            shape=(self.topology.atom_count, 3),
            name="atom_positions",
        )
        source_tensor = self._tensor(
            source, shape=(self.topology.atom_count,), name="source"
        )
        if source_tensor.device != positions.device:
            raise ValueError("atom_positions and source must be on the same device.")
        nodes, A, B = self.assemble(positions)
        potential = B @ source_tensor
        surface_charge = torch.linalg.solve(
            A, -self.topology.dielectric_factor * potential
        )
        field = B.transpose(0, 1) @ surface_charge
        energy = 0.5 * torch.dot(source_tensor, field)
        return CPCMReferenceState(
            node_positions=nodes,
            A=A,
            B=B,
            potential=potential,
            surface_charge=surface_charge,
            field=field,
            energy=energy,
            topology_hash=self.topology.topology_hash,
        )

    def energy(self, atom_positions: Any, source: Any):
        return self.solve(atom_positions, source).energy

    def dense_source_jacobian(self, atom_positions: Any):
        """Return the exact symmetric local map ``d field / d source``."""

        torch = _torch()
        _, A, B = self.assemble(atom_positions)
        return -self.topology.dielectric_factor * (
            B.transpose(0, 1) @ torch.linalg.solve(A, B)
        )

    def autograd_energy_gradients(
        self, atom_positions: Any, source: Any
    ) -> tuple[Any, Any]:
        """Return unrolled-autograd gradients with respect to geometry and source."""

        torch = _torch()
        positions = self._tensor(
            atom_positions,
            shape=(self.topology.atom_count, 3),
            name="atom_positions",
        )
        source_tensor = self._tensor(
            source, shape=(self.topology.atom_count,), name="source"
        )
        return torch.autograd.grad(
            self.energy(positions, source_tensor),
            (positions, source_tensor),
            create_graph=True,
        )

    def implicit_coordinate_gradient(self, atom_positions: Any, source: Any):
        """Dense stationary/implicit coordinate derivative with ``sigma`` held fixed.

        For ``E = -0.5 f v.T A^-1 v``, differentiation gives
        ``dE = sigma.T dv + (1/(2f)) sigma.T dA sigma``.
        """

        torch = _torch()
        positions = self._tensor(
            atom_positions,
            shape=(self.topology.atom_count, 3),
            name="atom_positions",
        )
        source_tensor = self._tensor(
            source, shape=(self.topology.atom_count,), name="source"
        )
        state = self.solve(positions, source_tensor)
        sigma = state.surface_charge.detach()
        surrogate = torch.dot(sigma, state.potential) + (
            0.5 / self.topology.dielectric_factor
        ) * torch.dot(sigma, state.A @ sigma)
        return torch.autograd.grad(surrogate, positions, create_graph=False)[0]

    def field_jvp(
        self,
        atom_positions: Any,
        source: Any,
        position_direction: Any,
        source_direction: Any,
    ) -> tuple[Any, Any]:
        """Return field and its JVP for a joint geometry/source direction."""

        torch = _torch()
        positions = self._tensor(
            atom_positions, shape=(self.topology.atom_count, 3), name="atom_positions"
        )
        source_tensor = self._tensor(
            source, shape=(self.topology.atom_count,), name="source"
        )
        d_positions = self._tensor(
            position_direction, shape=tuple(positions.shape), name="position_direction"
        )
        d_source = self._tensor(
            source_direction, shape=tuple(source_tensor.shape), name="source_direction"
        )
        return torch.autograd.functional.jvp(
            lambda r, c: self.solve(r, c).field,
            (positions, source_tensor),
            (d_positions, d_source),
            create_graph=True,
            strict=True,
        )

    def field_vjp(
        self, atom_positions: Any, source: Any, cotangent: Any
    ) -> tuple[Any, Any]:
        """Return the joint geometry/source VJP of the nonlinear field map."""

        torch = _torch()
        positions = self._tensor(
            atom_positions, shape=(self.topology.atom_count, 3), name="atom_positions"
        )
        source_tensor = self._tensor(
            source, shape=(self.topology.atom_count,), name="source"
        )
        output_cotangent = self._tensor(
            cotangent, shape=tuple(source_tensor.shape), name="cotangent"
        )
        _, vjp = torch.autograd.functional.vjp(
            lambda r, c: self.solve(r, c).field,
            (positions, source_tensor),
            v=output_cotangent,
            create_graph=True,
            strict=True,
        )
        return vjp

    def energy_hvp(
        self,
        atom_positions: Any,
        source: Any,
        position_direction: Any,
        source_direction: Any,
    ) -> tuple[Any, tuple[Any, Any]]:
        """Return energy and its exact joint geometry/source Hessian-vector product."""

        torch = _torch()
        positions = self._tensor(
            atom_positions, shape=(self.topology.atom_count, 3), name="atom_positions"
        )
        source_tensor = self._tensor(
            source, shape=(self.topology.atom_count,), name="source"
        )
        d_positions = self._tensor(
            position_direction, shape=tuple(positions.shape), name="position_direction"
        )
        d_source = self._tensor(
            source_direction, shape=tuple(source_tensor.shape), name="source_direction"
        )
        return torch.autograd.functional.hvp(
            lambda r, c: self.energy(r, c),
            (positions, source_tensor),
            (d_positions, d_source),
            create_graph=False,
            strict=True,
        )


# Compatibility for callers of the first unreviewed prototype.  Metadata on the
# class is intentionally explicit that this is synthetic and not Phase-4 parity.
TorchReferenceCPCM = TorchSyntheticDenseCPCM


class TorchFixedTopologyAmplitudeSWIGCPCMOracle:
    """Independent Torch replica of the production amplitude-SWIG C-PCM scalar.

    ``surface_snapshot`` is the immutable
    ``FixedTopologySurfaceSnapshot`` emitted by the production surface provider.
    This oracle reads its frozen parent-major topology, directions, weights and
    radii, but does not call any production continuum forward or derivative.
    Every geometry rebuilds the same fixed-cardinality surface in Torch.

    Sources have the authoritative raw MACE-POLAR order
    ``[q, l1_m0(y), l1_m1(z), l1_m-1(x)]`` and units ``[e,e*A,e*A,e*A]``.
    Returned fields have external order ``[V,dV/dx,dV/dy,dV/dz]`` and units
    ``[eV/e,eV/(e*A),eV/(e*A),eV/(e*A)]``.  The energy uses exactly
    ``0.5*c.T@Q@field`` with ``Q`` permutation ``(0,2,3,1)``.
    """

    provider_id = "maple.route2.oracle.torch-fixed-topology-amplitude-swig-cpcm.v1"
    scalar_id = "route2-operational-cpcm-fixedtopology-electrostatic-v1"
    capability_admitted = False
    hessian_capability_admitted = False
    topology_input_scope = "validated-production-snapshot-not-independent-grid-asset"
    source_order = ("net_monopole", "real_l1_m0", "real_l1_m1", "real_l1_m_minus1")
    field_order = (
        "potential",
        "potential_gradient_x",
        "potential_gradient_y",
        "potential_gradient_z",
    )
    field_to_source_indices = (0, 2, 3, 1)
    source_units = ("e", "e*angstrom", "e*angstrom", "e*angstrom")
    field_units = ("eV/e", "eV/(e*angstrom)", "eV/(e*angstrom)", "eV/(e*angstrom)")
    bohr_angstrom = 0.5291772105638411
    hartree_to_ev = 27.211386245988

    __slots__ = (
        "atom_count",
        "grid_points_per_atom",
        "node_count",
        "dielectric",
        "switching_constant",
        "dielectric_factor",
        "topology_hash",
        "_snapshot_identity",
        "_reference_positions_bohr",
        "_radii_bohr",
        "_parents",
        "_directions",
        "_weights",
        "_configuration_sha256",
        "_provenance_sha256",
        "_sealed",
    )

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError(
                "TorchFixedTopologyAmplitudeSWIGCPCMOracle is immutable."
            )
        object.__setattr__(self, name, value)

    def __init__(
        self, surface_snapshot: object, *, dielectric: float, switching_constant: float
    ):
        from maple.solvation.surfaces.fixed_topology import (
            FixedTopologySurfaceSnapshot,
        )

        if not isinstance(surface_snapshot, FixedTopologySurfaceSnapshot):
            raise TypeError(
                "surface_snapshot must be a validated FixedTopologySurfaceSnapshot."
            )
        surface_snapshot.validate_integrity()
        atom_count = getattr(surface_snapshot, "atom_count", None)
        grid_count = getattr(surface_snapshot, "grid_points_per_atom", None)
        if (
            isinstance(atom_count, bool)
            or not isinstance(atom_count, int)
            or atom_count < 1
        ):
            raise ValueError(
                "surface_snapshot must declare a positive integer atom_count."
            )
        if (
            isinstance(grid_count, bool)
            or not isinstance(grid_count, int)
            or grid_count < 1
        ):
            raise ValueError("surface_snapshot must declare grid_points_per_atom.")
        if getattr(surface_snapshot, "fixed_topology", None) is not True:
            raise ValueError("surface_snapshot must declare fixed_topology=True.")
        object.__setattr__(self, "atom_count", atom_count)
        object.__setattr__(self, "grid_points_per_atom", grid_count)
        object.__setattr__(self, "node_count", atom_count * grid_count)
        object.__setattr__(
            self,
            "dielectric",
            _positive_scalar(dielectric, name="dielectric", lower=1.0),
        )
        object.__setattr__(
            self,
            "switching_constant",
            _positive_scalar(switching_constant, name="switching_constant"),
        )
        object.__setattr__(
            self, "dielectric_factor", (self.dielectric - 1.0) / self.dielectric
        )
        object.__setattr__(
            self, "topology_hash", str(getattr(surface_snapshot, "topology_hash", ""))
        )
        if len(self.topology_hash) != 64:
            raise ValueError("surface_snapshot must carry a SHA256 topology_hash.")

        object.__setattr__(
            self,
            "_snapshot_identity",
            (
                surface_snapshot.provider_id,
                surface_snapshot.cavity_profile_id,
                surface_snapshot.configuration_sha256,
                surface_snapshot.provenance_sha256,
                surface_snapshot.topology_hash,
                surface_snapshot.state_hash,
            ),
        )

        object.__setattr__(
            self,
            "_reference_positions_bohr",
            _immutable_array(
                getattr(surface_snapshot, "reference_positions_bohr"),
                name="reference_positions_bohr",
                dtype=np.dtype(np.float64),
                shape_tail=(3,),
            ),
        )
        object.__setattr__(
            self,
            "_radii_bohr",
            _immutable_array(
                getattr(surface_snapshot, "radii_bohr"),
                name="radii_bohr",
                dtype=np.dtype(np.float64),
            ),
        )
        object.__setattr__(
            self,
            "_parents",
            _immutable_owner_array(getattr(surface_snapshot, "parent_atom_indices")),
        )
        object.__setattr__(
            self,
            "_directions",
            _immutable_array(
                getattr(surface_snapshot, "unit_directions"),
                name="unit_directions",
                dtype=np.dtype(np.float64),
                shape_tail=(3,),
            ),
        )
        object.__setattr__(
            self,
            "_weights",
            _immutable_array(
                getattr(surface_snapshot, "quadrature_weights"),
                name="quadrature_weights",
                dtype=np.dtype(np.float64),
            ),
        )
        if self._reference_positions_bohr.shape != (atom_count, 3):
            raise ValueError("snapshot reference-position count is inconsistent.")
        if self._radii_bohr.shape != (atom_count,) or np.any(self._radii_bohr <= 0.0):
            raise ValueError("snapshot radii must be positive with one value per atom.")
        if (
            self._parents.shape != (self.node_count,)
            or self._directions.shape != (self.node_count, 3)
            or self._weights.shape != (self.node_count,)
        ):
            raise ValueError(
                "snapshot candidate arrays do not match fixed cardinality."
            )
        expected_parents = np.repeat(np.arange(atom_count), grid_count)
        if not np.array_equal(self._parents, expected_parents):
            raise ValueError("snapshot order must be parent-major/grid-minor.")
        if np.any(self._weights <= 0.0):
            raise ValueError("snapshot quadrature weights must be positive.")
        if not np.allclose(
            np.linalg.norm(self._directions, axis=1), 1.0, atol=2e-12, rtol=0
        ):
            raise ValueError("snapshot unit directions must have unit norm.")
        object.__setattr__(
            self, "_configuration_sha256", self._current_configuration_sha256()
        )
        object.__setattr__(
            self,
            "_provenance_sha256",
            hashlib.sha256(
                json.dumps(
                    {
                        "provider_id": self.provider_id,
                        "configuration_sha256": self._configuration_sha256,
                        "scope": (
                            "independent-continuum-algebra-and-derivatives;"
                            "validated-production-topology-input"
                        ),
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest(),
        )
        object.__setattr__(self, "_sealed", True)

    def _current_configuration_sha256(self) -> str:
        digest = hashlib.sha256(
            json.dumps(
                {
                    "provider_id": self.provider_id,
                    "scalar_id": self.scalar_id,
                    "snapshot_identity": self._snapshot_identity,
                    "atom_count": self.atom_count,
                    "grid_points_per_atom": self.grid_points_per_atom,
                    "dielectric": self.dielectric,
                    "switching_constant": self.switching_constant,
                    "source_order": self.source_order,
                    "field_order": self.field_order,
                    "field_to_source_indices": self.field_to_source_indices,
                    "source_units": self.source_units,
                    "field_units": self.field_units,
                    "bohr_angstrom": self.bohr_angstrom,
                    "hartree_to_ev": self.hartree_to_ev,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        )
        for array in (
            self._reference_positions_bohr,
            self._radii_bohr,
            self._parents,
            self._directions,
            self._weights,
        ):
            canonical = np.ascontiguousarray(array)
            digest.update(canonical.dtype.str.encode())
            digest.update(np.asarray(canonical.shape, dtype=np.int64).tobytes())
            digest.update(canonical.tobytes())
        return digest.hexdigest()

    def _validate_configuration(self) -> None:
        current = self._current_configuration_sha256()
        if current != self._configuration_sha256:
            raise RuntimeError("Torch C-PCM oracle configuration fingerprint changed.")
        expected_provenance = hashlib.sha256(
            json.dumps(
                {
                    "provider_id": self.provider_id,
                    "configuration_sha256": current,
                    "scope": (
                        "independent-continuum-algebra-and-derivatives;"
                        "validated-production-topology-input"
                    ),
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        if expected_provenance != self._provenance_sha256:
            raise RuntimeError("Torch C-PCM oracle provenance fingerprint changed.")

    @property
    def configuration_sha256(self) -> str:
        self._validate_configuration()
        return self._configuration_sha256

    @property
    def provenance_sha256(self) -> str:
        self._validate_configuration()
        return self._provenance_sha256

    def _tensor(self, value: object, *, shape: tuple[int, ...], name: str):
        torch = _torch()
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor.")
        if value.dtype != torch.float64:
            raise TypeError(f"{name} must use torch.float64; received {value.dtype}.")
        if tuple(value.shape) != shape:
            raise ValueError(
                f"{name} must have shape {shape}; received {tuple(value.shape)}."
            )
        if not bool(torch.isfinite(value).all().item()):
            raise ValueError(f"{name} must contain only finite values.")
        return value

    @staticmethod
    def _constant(values: np.ndarray, like: Any):
        torch = _torch()
        return torch.tensor(
            np.array(values, copy=True), dtype=torch.float64, device=like.device
        )

    @staticmethod
    def _amplitude_switch(clearance: Any):
        torch = _torch()
        x = torch.clamp(clearance, min=0.0, max=1.0)
        polynomial = x**4 * (35.0 - x * (84.0 - x * (70.0 - 20.0 * x)))
        return torch.where(
            clearance <= 0.0,
            torch.zeros_like(clearance),
            torch.where(clearance >= 1.0, torch.ones_like(clearance), polynomial),
        )

    def _surface(self, positions_angstrom: Any) -> tuple[Any, Any, Any, Any]:
        self._validate_configuration()
        torch = _torch()
        positions = self._tensor(
            positions_angstrom, shape=(self.atom_count, 3), name="positions_angstrom"
        )
        positions_bohr = positions / self.bohr_angstrom
        parents = self._constant(self._parents, positions).to(torch.long)
        radii = self._constant(self._radii_bohr, positions)
        directions = self._constant(self._directions, positions)
        weights = self._constant(self._weights, positions)
        points = positions_bohr[parents] + radii[parents, None] * directions

        widths = radii * np.sqrt(14.0 / self.grid_points_per_atom)
        alpha = 0.5 + radii / widths - torch.sqrt((radii / widths) ** 2 - 1.0 / 28.0)
        inner_radii = radii - alpha * widths
        displacement = points[:, None, :] - positions_bohr[None, :, :]
        distance = torch.linalg.vector_norm(displacement, dim=2)
        if bool((distance <= 1.0e-14).any().item()):
            rows, columns = torch.where(distance <= 1.0e-14)
            if bool((columns != parents[rows]).any().item()):
                raise ValueError("A SWIG candidate coincides with an occluding atom.")
        clearance = (distance - inner_radii[None, :]) / widths[None, :]
        parent_mask = torch.nn.functional.one_hot(
            parents, num_classes=self.atom_count
        ).to(dtype=torch.bool)
        clearance = torch.where(parent_mask, torch.ones_like(clearance), clearance)
        amplitudes = torch.prod(self._amplitude_switch(clearance), dim=1)
        exponents = self.switching_constant / (radii[parents] * torch.sqrt(weights))
        return positions_bohr, points, amplitudes, exponents

    def assemble(self, positions_angstrom: Any) -> tuple[Any, Any, Any, Any]:
        """Return surface points, amplitudes, production-parity ``H`` and dense ``B``.

        ``B`` maps flattened raw ``(N,4)`` sources directly to the surface MEP
        in Hartree/e.  It is formed by Torch autograd from the exact analytic
        point charge/dipole formula, never by calling production code.
        """

        torch = _torch()
        positions = self._tensor(
            positions_angstrom, shape=(self.atom_count, 3), name="positions_angstrom"
        )
        positions_bohr, points, amplitudes, exponents = self._surface(positions)
        delta = points[:, None, :] - positions_bohr[None, :, :]
        radius = torch.linalg.vector_norm(delta, dim=2)
        if bool((radius <= 1.0e-14).any().item()):
            raise ValueError("A PCM surface point coincides with an atomic centre.")
        inv_r = 1.0 / radius
        # Raw columns [q,y,z,x]; Cartesian dipoles in bohr are [x,y,z]/Bohr.
        B_blocks = torch.stack(
            (
                inv_r,
                delta[..., 1] * inv_r**3 / self.bohr_angstrom,
                delta[..., 2] * inv_r**3 / self.bohr_angstrom,
                delta[..., 0] * inv_r**3 / self.bohr_angstrom,
            ),
            dim=2,
        )
        B = B_blocks.reshape(self.node_count, 4 * self.atom_count)

        node_delta = points[:, None, :] - points[None, :, :]
        node_distance = torch.linalg.vector_norm(node_delta, dim=2)
        eye = torch.eye(self.node_count, dtype=torch.bool, device=positions.device)
        if bool(((node_distance <= 1.0e-12) & ~eye).any().item()):
            raise ValueError("Fixed-topology SWIG has coincident candidate centres.")
        safe_distance = torch.where(eye, torch.ones_like(node_distance), node_distance)
        pair_exponent = (
            exponents[:, None]
            * exponents[None, :]
            / torch.sqrt(exponents[:, None] ** 2 + exponents[None, :] ** 2)
        )
        off_diagonal = torch.special.erf(pair_exponent * safe_distance) / safe_distance
        off_diagonal = torch.where(eye, torch.zeros_like(off_diagonal), off_diagonal)
        diagonal = exponents * np.sqrt(2.0 / np.pi)
        H = (
            torch.diag(diagonal)
            + amplitudes[:, None] * off_diagonal * amplitudes[None, :]
        )
        H = 0.5 * (H + H.T)
        return points, amplitudes, H, B

    def solve(self, positions_angstrom: Any, source: Any) -> CPCMReferenceState:
        torch = _torch()
        positions = self._tensor(
            positions_angstrom, shape=(self.atom_count, 3), name="positions_angstrom"
        )
        source_tensor = self._tensor(source, shape=(self.atom_count, 4), name="source")
        if source_tensor.device != positions.device:
            raise ValueError(
                "positions_angstrom and source must be on the same device."
            )
        points, amplitudes, H, B = self.assemble(positions)
        potential = B @ source_tensor.reshape(-1)
        amplitude_charge = torch.linalg.solve(
            H, -self.dielectric_factor * amplitudes * potential
        )
        physical_charge = amplitudes * amplitude_charge
        raw_dual_hartree = B.T @ physical_charge
        raw_dual_ev = raw_dual_hartree.reshape(self.atom_count, 4) * self.hartree_to_ev
        # B.T is already in raw source-dual order. Convert it to public external
        # field order [V,gx,gy,gz], the inverse of Q=(0,2,3,1).
        field = raw_dual_ev[:, (0, 3, 1, 2)]
        energy = 0.5 * torch.dot(potential, physical_charge) * self.hartree_to_ev
        return CPCMReferenceState(
            node_positions=points,
            A=H,
            B=B,
            potential=potential,
            surface_charge=physical_charge,
            field=field,
            energy=energy,
            topology_hash=self.topology_hash,
        )

    def energy(self, positions_angstrom: Any, source: Any):
        return self.solve(positions_angstrom, source).energy

    def field(self, positions_angstrom: Any, source: Any):
        return self.solve(positions_angstrom, source).field

    def energy_gradients(
        self, positions_angstrom: Any, source: Any, *, create_graph: bool = True
    ):
        torch = _torch()
        positions = self._tensor(
            positions_angstrom, shape=(self.atom_count, 3), name="positions_angstrom"
        )
        source_tensor = self._tensor(source, shape=(self.atom_count, 4), name="source")
        return torch.autograd.grad(
            self.energy(positions, source_tensor),
            (positions, source_tensor),
            create_graph=create_graph,
        )

    def source_jvp(self, positions_angstrom: Any, source: Any, direction: Any):
        torch = _torch()
        positions = self._tensor(
            positions_angstrom, shape=(self.atom_count, 3), name="positions_angstrom"
        )
        source_tensor = self._tensor(source, shape=(self.atom_count, 4), name="source")
        source_direction = self._tensor(
            direction, shape=(self.atom_count, 4), name="source_direction"
        )
        return torch.autograd.functional.jvp(
            lambda c: self.field(positions, c),
            source_tensor,
            source_direction,
            create_graph=True,
            strict=True,
        )[1]

    def source_vjp(self, positions_angstrom: Any, source: Any, cotangent: Any):
        torch = _torch()
        positions = self._tensor(
            positions_angstrom, shape=(self.atom_count, 3), name="positions_angstrom"
        )
        source_tensor = self._tensor(source, shape=(self.atom_count, 4), name="source")
        field_cotangent = self._tensor(
            cotangent, shape=(self.atom_count, 4), name="field_cotangent"
        )
        return torch.autograd.functional.vjp(
            lambda c: self.field(positions, c),
            source_tensor,
            v=field_cotangent,
            create_graph=True,
            strict=True,
        )[1]

    def energy_hvp(
        self,
        positions_angstrom: Any,
        source: Any,
        position_direction: Any,
        source_direction: Any,
    ):
        """Raw oracle HVP; deliberately not admitted as a production capability."""

        torch = _torch()
        positions = self._tensor(
            positions_angstrom, shape=(self.atom_count, 3), name="positions_angstrom"
        )
        source_tensor = self._tensor(source, shape=(self.atom_count, 4), name="source")
        d_positions = self._tensor(
            position_direction, shape=(self.atom_count, 3), name="position_direction"
        )
        d_source = self._tensor(
            source_direction, shape=(self.atom_count, 4), name="source_direction"
        )
        return torch.autograd.functional.hvp(
            lambda r, c: self.energy(r, c),
            (positions, source_tensor),
            (d_positions, d_source),
            strict=True,
        )


__all__ = [
    "CPCMReferenceState",
    "FixedCPCMTopology",
    "TorchFixedTopologyAmplitudeSWIGCPCMOracle",
    "TorchReferenceCPCM",
    "TorchSyntheticDenseCPCM",
]
