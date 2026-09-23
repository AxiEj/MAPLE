"""Exact dense Torch implementation of the ddX 0.8.0 ddPCM scalar.

This is a bounded correctness backend, not a scalable PCM solver.  It follows
the operator definitions in ddX 0.8.0 (commit
``4d79e3d9caeae5e602683572a71cb550414f9b09``): the regularized union-of-
spheres characteristic function, full single-layer operator, double-layer
self jump, two dense solves, and the general-source half coupling.  pyddx is
an optional test oracle only and is never called by this module.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Sequence

import numpy as np
from ase.data import atomic_numbers
from ase.units import Bohr

from maple.solvation.api.units import HARTREE_TO_EV
from maple.solvation.coupling.metrics import MACE_POLAR_RADIAL_GTO_PAIRING
from maple.solvation.coupling.operator import source_files_sha256
from maple.solvation.coupling.spaces import (
    MACE_POLAR_RADIAL_GTO_FIELD_DUAL_SPACE,
    MACE_POLAR_RADIAL_GTO_SOURCE_SPACE,
)
from maple.solvation.surfaces.lebedev import ordered_lebedev_grid

from .functional import ContinuumEnergyFunctional
from .harmonic_torch_primitives import _torch_real_harmonic_design

TORCH_DDPCM_PROVIDER_ID = "maple.route2.continuum.torch-ddpcm-dense.impl.v1"
TORCH_DDPCM_PROVENANCE = (
    "ddx-0.8.0-4d79e3d9caeae5e602683572a71cb550414f9b09-equations-v1"
)
_UNUSED_SOURCE_INDICES = (1, 5, 6, 7)
_LEARNED_SOURCE_INDICES = (0, 2, 3, 4)


def _sha(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _immutable_float64(values: object, shape: tuple[int, ...]) -> np.ndarray:
    contiguous = np.ascontiguousarray(values, dtype=np.float64)
    if contiguous.shape != shape or not np.all(np.isfinite(contiguous)):
        raise ValueError(f"immutable array must be finite with shape {shape}.")
    return np.frombuffer(contiguous.tobytes(order="C"), dtype=np.float64).reshape(shape)


def _runtime_versions() -> tuple[tuple[str, str], ...]:
    result = []
    for distribution in ("ase", "numpy", "pyscf", "torch"):
        try:
            result.append((distribution, version(distribution)))
        except PackageNotFoundError as exc:  # pragma: no cover - required runtime
            raise RuntimeError(
                f"TorchDDPCM runtime dependency {distribution!r} is unavailable."
            ) from exc
    return tuple(result)


def _implementation_files_sha256() -> tuple[tuple[str, str], ...]:
    solvation = Path(__file__).resolve().parents[1]
    return source_files_sha256(
        {
            "solvation.api.units": solvation / "api" / "units.py",
            "solvation.continuum.functional": solvation / "continuum" / "functional.py",
            "solvation.continuum.harmonic_coefficients": solvation
            / "continuum"
            / "harmonic_coefficients.py",
            "solvation.continuum.harmonic_torch_primitives": solvation
            / "continuum"
            / "harmonic_torch_primitives.py",
            "solvation.continuum.torch_ddpcm": Path(__file__),
            "solvation.coupling.metrics": solvation / "coupling" / "metrics.py",
            "solvation.coupling.operator": solvation / "coupling" / "operator.py",
            "solvation.coupling.spaces": solvation / "coupling" / "spaces.py",
            "solvation.surfaces.lebedev": solvation / "surfaces" / "lebedev.py",
        }
    )


def _dense_resource_estimate(
    atom_count: int,
    lmax: int,
    n_lebedev: int,
    limit_bytes: int,
    derivative_order: int,
) -> DenseResourceEstimate:
    basis = atom_count * (lmax + 1) ** 2
    matrix_bytes = 8 * basis * basis
    # Retained harmonic recurrences dominate the first reverse pass.  The
    # factor 12 is calibrated conservatively against measured float64 CPU and
    # CUDA energy+gradient peaks; it is deliberately separate from process
    # baseline memory and from a caller's full-Hessian row retention.
    first_order_peak = 16 * matrix_bytes + 12 * 8 * atom_count * n_lebedev * (
        basis + 32
    )
    forward_peak = max(matrix_bytes * 8, (first_order_peak + 1) // 2)
    second_order_peak = 4 * first_order_peak
    selected = (forward_peak, first_order_peak, second_order_peak)[derivative_order]
    return DenseResourceEstimate(
        atom_count=atom_count,
        basis_dimension=basis,
        bytes_per_matrix=matrix_bytes,
        forward_peak_bytes=forward_peak,
        first_order_peak_bytes=first_order_peak,
        second_order_peak_bytes=second_order_peak,
        derivative_order=derivative_order,
        conservative_peak_bytes=selected,
        limit_bytes=limit_bytes,
    )


@dataclass(frozen=True, slots=True)
class DenseResourceEstimate:
    atom_count: int
    basis_dimension: int
    bytes_per_matrix: int
    forward_peak_bytes: int
    first_order_peak_bytes: int
    second_order_peak_bytes: int
    derivative_order: int
    conservative_peak_bytes: int
    limit_bytes: int

    @property
    def within_limit(self) -> bool:
        return self.conservative_peak_bytes <= self.limit_bytes


@dataclass(frozen=True, slots=True)
class DDPCMTopologyCertificate:
    contract: str
    topology_sha256: str
    active_node_count: int
    buried_plateau_count: int
    transition_node_count: int
    minimum_switch_margin: float | None
    minimum_active_f_margin: float | None


@dataclass(frozen=True, slots=True)
class MatrixStabilityDiagnostics:
    matrix_name: str
    dimension: int
    numerical_rank: int
    condition_number: float
    reciprocal_condition: float
    minimum_reciprocal_condition: float
    rank_tolerance: float
    maximum_singular_value: float
    minimum_singular_value: float


@dataclass(frozen=True, slots=True)
class TorchDDPCMState:
    """Graph-bearing internal state plus detached observability metadata."""

    energy: Any
    L: Any
    D: Any
    rhs: Any
    dielectric_rhs: Any
    solution: Any
    topology: DDPCMTopologyCertificate
    resource: DenseResourceEstimate
    l_stability: MatrixStabilityDiagnostics
    reps_stability: MatrixStabilityDiagnostics
    reps_relative_residual: float
    l_relative_residual: float


class TorchDDPCM(ContinuumEnergyFunctional):
    """Dense differentiable ddPCM for the frozen MACE-POLAR point source."""

    __slots__ = (
        "_configuration_sha256",
        "_dielectric",
        "_eta",
        "_grid_directions",
        "_grid_sha256",
        "_grid_weights",
        "_lmax",
        "_max_dense_bytes",
        "_n_lebedev",
        "_numerical_policy",
        "_provenance_sha256",
        "_radii_angstrom",
        "_solve_residual_tolerance",
        "_symbols",
        "_topology_margin",
    )

    provider_id = TORCH_DDPCM_PROVIDER_ID
    capability_admitted = False

    def __init__(
        self,
        symbols: Sequence[str],
        radii_angstrom: object,
        *,
        dielectric: float,
        lmax: int = 15,
        n_lebedev: int = 1202,
        eta: float = 0.1,
        device: object = "cpu",
        dtype: object = None,
        max_dense_bytes: int = 1_000_000_000,
        solve_residual_tolerance: float = 1.0e-12,
        topology_margin: float = 1.0e-8,
    ) -> None:
        torch = __import__("torch")
        dtype = torch.float64 if dtype is None else dtype
        if dtype is not torch.float64:
            raise TypeError("TorchDDPCM supports torch.float64 only.")
        requested_device = torch.device(device)
        if requested_device.type not in {"cpu", "cuda"}:
            raise ValueError("TorchDDPCM device must be CPU or CUDA.")
        if requested_device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA was requested for TorchDDPCM but is unavailable."
                )
            index = (
                torch.cuda.current_device()
                if requested_device.index is None
                else requested_device.index
            )
            if index < 0 or index >= torch.cuda.device_count():
                raise RuntimeError(
                    f"requested CUDA device cuda:{index} is unavailable."
                )
            requested_device = torch.device("cuda", index)

        symbol_tuple = tuple(str(symbol) for symbol in symbols)
        if not symbol_tuple or any(
            symbol not in atomic_numbers for symbol in symbol_tuple
        ):
            raise ValueError("symbols must contain recognized element symbols.")
        radii = np.asarray(radii_angstrom, dtype=np.float64)
        if (
            radii.shape != (len(symbol_tuple),)
            or not np.all(np.isfinite(radii))
            or np.any(radii <= 0.0)
        ):
            raise ValueError("radii_angstrom must be finite and positive per atom.")
        if isinstance(lmax, bool) or not isinstance(lmax, int) or lmax < 1:
            raise ValueError("lmax must be a positive integer.")
        dielectric_value = float(dielectric)
        if not math.isfinite(dielectric_value) or dielectric_value <= 1.0:
            raise ValueError("dielectric must be finite and greater than one.")
        eta_value = float(eta)
        if not math.isfinite(eta_value) or not 0.0 < eta_value <= 1.0:
            raise ValueError("eta must lie in (0, 1].")
        if (
            isinstance(max_dense_bytes, bool)
            or not isinstance(max_dense_bytes, int)
            or max_dense_bytes <= 0
        ):
            raise ValueError("max_dense_bytes must be a positive integer.")
        residual_tolerance = float(solve_residual_tolerance)
        if not math.isfinite(residual_tolerance) or residual_tolerance <= 0.0:
            raise ValueError("solve_residual_tolerance must be positive.")
        margin = float(topology_margin)
        if not math.isfinite(margin) or margin <= 0.0:
            raise ValueError("topology_margin must be positive.")

        preflight = _dense_resource_estimate(
            len(symbol_tuple), lmax, n_lebedev, max_dense_bytes, 1
        )
        if not preflight.within_limit:
            raise MemoryError(
                "TorchDDPCM dense preflight rejected allocation: estimated "
                f"{preflight.conservative_peak_bytes} bytes exceeds limit "
                f"{preflight.limit_bytes} bytes (B={preflight.basis_dimension})."
            )
        grid = ordered_lebedev_grid(n_lebedev)
        radii = _immutable_float64(radii, (len(symbol_tuple),))
        directions = _immutable_float64(grid.directions, (n_lebedev, 3))
        weights = _immutable_float64(grid.weights, (n_lebedev,))
        policy = {
            "policy_id": "torch-ddpcm-dense-numerical-policy-v2",
            "backend": "dense-direct-two-solve",
            "dtype": "torch.float64",
            "device": str(requested_device),
            "max_dense_bytes": max_dense_bytes,
            "solve_residual_tolerance": residual_tolerance,
            "topology_margin": margin,
            "matrix_peak_factor": 16,
            "condition_estimator": "torch.linalg.svdvals-on-requested-device-detached",
            "minimum_reciprocal_condition": math.sqrt(np.finfo(np.float64).eps),
            "rank_tolerance": "dimension*float64-epsilon*maximum-singular-value",
            "conditioning_claim": "numerical-stability-guard-not-Hessian-error-bound",
            "fallback": "forbidden",
        }
        object.__setattr__(self, "_symbols", symbol_tuple)
        object.__setattr__(self, "_radii_angstrom", radii)
        object.__setattr__(self, "_dielectric", dielectric_value)
        object.__setattr__(self, "_lmax", lmax)
        object.__setattr__(self, "_n_lebedev", n_lebedev)
        object.__setattr__(self, "_eta", eta_value)
        object.__setattr__(self, "_max_dense_bytes", max_dense_bytes)
        object.__setattr__(self, "_solve_residual_tolerance", residual_tolerance)
        object.__setattr__(self, "_topology_margin", margin)
        object.__setattr__(self, "_grid_directions", directions)
        object.__setattr__(self, "_grid_weights", weights)
        object.__setattr__(self, "_grid_sha256", grid.sha256)
        object.__setattr__(self, "_numerical_policy", MappingProxyType(policy))
        configuration_sha256 = _sha(self._current_configuration_payload())
        object.__setattr__(self, "_configuration_sha256", configuration_sha256)
        object.__setattr__(
            self,
            "_provenance_sha256",
            _sha(
                {
                    "configuration_sha256": configuration_sha256,
                    "source": TORCH_DDPCM_PROVENANCE,
                }
            ),
        )
        super().__init__(
            source_space=MACE_POLAR_RADIAL_GTO_SOURCE_SPACE,
            field_space=MACE_POLAR_RADIAL_GTO_FIELD_DUAL_SPACE,
            pairing=MACE_POLAR_RADIAL_GTO_PAIRING,
            dtype=dtype,
            device=requested_device,
            expected_atomic_numbers=tuple(atomic_numbers[s] for s in symbol_tuple),
        )

    @property
    def symbols(self) -> tuple[str, ...]:
        return self._symbols

    @property
    def radii_angstrom(self) -> np.ndarray:
        return self._radii_angstrom

    def _current_configuration_payload(self) -> dict[str, object]:
        return {
            "provider_id": self.provider_id,
            "provenance": TORCH_DDPCM_PROVENANCE,
            "symbols": self._symbols,
            "radii_angstrom": self._radii_angstrom.tolist(),
            "dielectric": self._dielectric,
            "lmax": self._lmax,
            "n_lebedev": self._n_lebedev,
            "eta": self._eta,
            "grid_sha256": self._grid_sha256,
            "source_layout": {
                "learned": _LEARNED_SOURCE_INDICES,
                "zero": _UNUSED_SOURCE_INDICES,
            },
            "numerical_policy": dict(self._numerical_policy),
            "runtime_versions": dict(_runtime_versions()),
            "source_files_sha256": dict(_implementation_files_sha256()),
        }

    def configuration_sha256(self) -> str:
        if _sha(self._current_configuration_payload()) != self._configuration_sha256:
            raise RuntimeError("TorchDDPCM implementation configuration changed.")
        return self._configuration_sha256

    @property
    def provenance_sha256(self) -> str:
        self.configuration_sha256()
        return self._provenance_sha256

    @property
    def numerical_policy(self):
        return self._numerical_policy

    @property
    def device(self):
        return self._torch_device

    @property
    def dtype(self):
        return self._torch_dtype

    @property
    def lmax(self) -> int:
        return self._lmax

    @property
    def n_lebedev(self) -> int:
        return self._n_lebedev

    @property
    def dielectric(self) -> float:
        return self._dielectric

    def estimate_resources(
        self,
        atom_count: int | None = None,
        *,
        derivative_order: int = 1,
        limit_bytes: int | None = None,
    ) -> DenseResourceEstimate:
        count = len(self._symbols) if atom_count is None else atom_count
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError("atom_count must be a positive integer.")
        if (
            isinstance(derivative_order, bool)
            or not isinstance(derivative_order, int)
            or derivative_order not in (0, 1, 2)
        ):
            raise ValueError("derivative_order must be exactly 0, 1, or 2.")
        limit = self._max_dense_bytes if limit_bytes is None else limit_bytes
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError("limit_bytes must be a positive integer.")
        return _dense_resource_estimate(
            count, self._lmax, self._n_lebedev, limit, derivative_order
        )

    def preflight_resources(
        self,
        *,
        derivative_order: int,
        atom_count: int | None = None,
        limit_bytes: int | None = None,
    ) -> DenseResourceEstimate:
        estimate = self.estimate_resources(
            atom_count,
            derivative_order=derivative_order,
            limit_bytes=limit_bytes,
        )
        if not estimate.within_limit:
            raise MemoryError(
                "TorchDDPCM dense preflight rejected derivative order "
                f"{derivative_order}: estimated {estimate.conservative_peak_bytes} "
                f"bytes exceeds limit {estimate.limit_bytes} bytes "
                f"(B={estimate.basis_dimension})."
            )
        return estimate

    def _constant(self, reference: Any, values: object):
        torch = __import__("torch")
        return torch.tensor(
            np.array(values, copy=True), dtype=reference.dtype, device=reference.device
        )

    def _switch(self, distance_ratio: Any):
        torch = __import__("torch")
        shifted = distance_ratio - 0.5 * self._eta
        a = 1.0 - shifted
        z = a / self._eta
        polynomial = z**3 * (z * (6.0 * z - 15.0) + 10.0)
        return torch.where(
            a <= 0.0,
            torch.zeros_like(a),
            torch.where(a >= self._eta, torch.ones_like(a), polynomial),
        )

    def _geometry(self, positions: Any):
        torch = __import__("torch")
        directions = self._constant(positions, self._grid_directions)
        weights = self._constant(positions, self._grid_weights)
        radii_bohr = self._constant(positions, self._radii_angstrom / Bohr)
        centres_bohr = positions / Bohr
        nodes = centres_bohr[:, None, :] + radii_bohr[:, None, None] * directions
        pair_delta = nodes[:, :, None, :] - centres_bohr[None, None, :, :]
        pair_distance = torch.linalg.vector_norm(pair_delta, dim=-1)
        atom_count = len(self._symbols)
        owner_mask = torch.eye(atom_count, dtype=torch.bool, device=positions.device)[
            :, None, :
        ]
        safe_distance = torch.where(
            owner_mask, torch.ones_like(pair_distance), pair_distance
        )
        if bool(
            (safe_distance[~owner_mask.expand_as(safe_distance)] <= 1.0e-13)
            .any()
            .detach()
            .cpu()
        ):
            raise RuntimeError(
                "a ddPCM grid node coincides with a distinct sphere centre."
            )
        ratios = safe_distance / radii_bohr[None, None, :]
        chi = torch.where(owner_mask, torch.zeros_like(ratios), self._switch(ratios))
        fi = torch.sum(chi, dim=-1)
        ui = torch.clamp(1.0 - fi, min=0.0)
        return (
            directions,
            weights,
            radii_bohr,
            centres_bohr,
            nodes,
            pair_delta,
            pair_distance,
            chi,
            fi,
            ui,
        )

    def _certify_topology(
        self, ratios: Any, chi: Any, fi: Any, ui: Any
    ) -> DDPCMTopologyCertificate:
        torch = __import__("torch")
        count = len(self._symbols)
        nonself = ~torch.eye(count, dtype=torch.bool, device=fi.device)[:, None, :]
        relevant_ratios = ratios[nonself.expand_as(ratios)]
        lower = 1.0 - 0.5 * self._eta
        upper = 1.0 + 0.5 * self._eta
        switch_margin = torch.minimum(
            torch.abs(relevant_ratios - lower), torch.abs(relevant_ratios - upper)
        )
        transition = (chi > 0.0) & (chi < 1.0) & nonself
        transition_per_node = torch.any(transition, dim=-1)
        plateau_buried = (fi == 1.0) & ~transition_per_node
        active_f_margin_values = torch.abs(fi[transition_per_node] - 1.0)
        minimum_active = (
            float(active_f_margin_values.min().detach().cpu())
            if active_f_margin_values.numel()
            else None
        )
        minimum_switch = (
            float(switch_margin.min().detach().cpu()) if switch_margin.numel() else None
        )
        ambiguous_f = transition_per_node & (
            torch.abs(fi - 1.0) <= self._topology_margin
        )
        if (
            minimum_switch is not None and minimum_switch <= self._topology_margin
        ) or bool(ambiguous_f.any().detach().cpu()):
            raise RuntimeError(
                "ddPCM topology is within the certified branch-change margin."
            )
        support = torch.where(
            chi == 0.0,
            torch.zeros_like(chi, dtype=torch.int8),
            torch.where(
                chi == 1.0,
                torch.full_like(chi, 2, dtype=torch.int8),
                torch.ones_like(chi, dtype=torch.int8),
            ),
        )
        topology_digest = hashlib.sha256(b"ddx-v0.8.0-discrete-topology-v1")
        for metadata in (support, ui > 0.0, fi > 1.0):
            array = np.ascontiguousarray(metadata.detach().cpu().numpy())
            topology_digest.update(str(array.shape).encode())
            topology_digest.update(b"\0")
            topology_digest.update(array.tobytes(order="C"))
        return DDPCMTopologyCertificate(
            contract="ddx-v0.8.0-fixed-branch-margin-v1",
            topology_sha256=topology_digest.hexdigest(),
            active_node_count=int((ui > 0.0).sum().detach().cpu()),
            buried_plateau_count=int(plateau_buried.sum().detach().cpu()),
            transition_node_count=int(transition_per_node.sum().detach().cpu()),
            minimum_switch_margin=minimum_switch,
            minimum_active_f_margin=minimum_active,
        )

    def _point_source(
        self, nodes: Any, centres_bohr: Any, radii_bohr: Any, source: Any
    ):
        torch = __import__("torch")
        delta = nodes[:, :, None, :] - centres_bohr[None, None, :, :]
        distance = torch.linalg.vector_norm(delta, dim=-1)
        if bool((distance <= 1.0e-13).any().detach().cpu()):
            raise RuntimeError("a point source coincides with a ddPCM surface node.")
        charges = source[:, 0]
        dipoles_bohr = (
            torch.stack((source[:, 4], source[:, 2], source[:, 3]), dim=-1) / Bohr
        )
        phi = torch.sum(
            charges[None, None, :] / distance
            + torch.sum(delta * dipoles_bohr[None, None, :, :], dim=-1) / distance**3,
            dim=-1,
        )

        basis = (self._lmax + 1) ** 2
        psi = source.new_zeros((len(self._symbols), basis))
        psi[:, 0] = math.sqrt(4.0 * math.pi) * source[:, 0]
        dipole_normalizer = Bohr * math.sqrt(4.0 * math.pi / 3.0)
        multipoles_l1 = source[:, (2, 3, 4)] / dipole_normalizer
        psi[:, 1:4] = (4.0 * math.pi / 3.0) * multipoles_l1 / radii_bohr[:, None]
        return phi, psi.reshape(-1)

    def _operators(self, positions: Any, geometry: tuple[Any, ...]):
        torch = __import__("torch")
        (
            directions,
            weights,
            radii,
            centres,
            nodes,
            pair_delta,
            pair_distance,
            chi,
            fi,
            ui,
        ) = geometry
        atom_count = len(self._symbols)
        design = _torch_real_harmonic_design(directions, lmax=self._lmax)
        ell = self._constant(
            positions,
            np.concatenate(
                [np.full(2 * l + 1, l, dtype=np.float64) for l in range(self._lmax + 1)]
            ),
        )
        single_scale = 4.0 * math.pi / (2.0 * ell + 1.0)
        diagonal_l = torch.diag(single_scale)
        l_rows = []
        d_rows = []
        for target in range(atom_count):
            l_blocks = []
            d_blocks = []
            target_weight = weights * ui[target]
            for source_index in range(atom_count):
                if target == source_index:
                    l_blocks.append(diagonal_l)
                    self_potential = -0.5 * design * single_scale[None, :]
                    d_blocks.append(
                        design.T @ (target_weight[:, None] * self_potential)
                    )
                    continue
                distance = pair_distance[target, :, source_index]
                unit = pair_delta[target, :, source_index, :] / distance[:, None]
                source_design = _torch_real_harmonic_design(unit, lmax=self._lmax)
                ratio = distance / radii[source_index]
                overlap = chi[target, :, source_index] / torch.clamp(
                    fi[target], min=1.0
                )
                l_potential = -source_design * (
                    overlap[:, None]
                    * ratio[:, None] ** ell[None, :]
                    * single_scale[None, :]
                )
                l_blocks.append(design.T @ (weights[:, None] * l_potential))
                d_scale = (
                    4.0
                    * math.pi
                    * ell
                    / (2.0 * ell + 1.0)
                    * ratio[:, None] ** (-(ell[None, :] + 1.0))
                )
                d_blocks.append(
                    design.T @ (target_weight[:, None] * source_design * d_scale)
                )
            l_rows.append(torch.cat(l_blocks, dim=1))
            d_rows.append(torch.cat(d_blocks, dim=1))
        return torch.cat(l_rows, dim=0), torch.cat(d_rows, dim=0), design, ui

    @staticmethod
    def _relative_residual(matrix: Any, solution: Any, rhs: Any) -> float:
        torch = __import__("torch")
        residual = torch.linalg.vector_norm(matrix @ solution - rhs)
        scale = torch.maximum(torch.linalg.vector_norm(rhs), rhs.new_tensor(1.0))
        return float((residual / scale).detach().cpu())

    @staticmethod
    def _validate_matrix_stability(
        matrix_name: str, matrix: Any
    ) -> MatrixStabilityDiagnostics:
        """Certify numerical rank/conditioning without entering the energy graph."""

        torch = __import__("torch")
        if (
            not torch.is_tensor(matrix)
            or matrix.ndim != 2
            or matrix.shape[0] != matrix.shape[1]
            or matrix.shape[0] < 1
        ):
            raise ValueError(f"{matrix_name} must be a nonempty square tensor.")
        with torch.no_grad():
            singular_values = torch.linalg.svdvals(matrix.detach())
            if not bool(torch.isfinite(singular_values).all().detach().cpu()):
                raise RuntimeError(
                    f"TorchDDPCM {matrix_name} singular values are non-finite."
                )
            maximum = float(singular_values[0].detach().cpu())
            minimum = float(singular_values[-1].detach().cpu())
            dimension = int(matrix.shape[0])
            epsilon = np.finfo(np.float64).eps
            rank_tolerance = dimension * epsilon * maximum
            numerical_rank = int(
                torch.count_nonzero(singular_values > rank_tolerance).detach().cpu()
            )
            reciprocal_condition = minimum / maximum if maximum > 0.0 else 0.0
            minimum_reciprocal = math.sqrt(epsilon)
            condition_number = maximum / minimum if minimum > 0.0 else math.inf
        diagnostics = MatrixStabilityDiagnostics(
            matrix_name=matrix_name,
            dimension=dimension,
            numerical_rank=numerical_rank,
            condition_number=condition_number,
            reciprocal_condition=reciprocal_condition,
            minimum_reciprocal_condition=minimum_reciprocal,
            rank_tolerance=rank_tolerance,
            maximum_singular_value=maximum,
            minimum_singular_value=minimum,
        )
        if numerical_rank != dimension or reciprocal_condition <= minimum_reciprocal:
            raise RuntimeError(
                f"TorchDDPCM {matrix_name} is rank deficient or ill-conditioned: "
                f"rank={numerical_rank}/{dimension}, rcond={reciprocal_condition:.3e}, "
                f"required>{minimum_reciprocal:.3e}."
            )
        return diagnostics

    def _state_torch(self, positions: Any, source: Any) -> TorchDDPCMState:
        torch = __import__("torch")
        self.configuration_sha256()
        if bool(torch.count_nonzero(source[:, _UNUSED_SOURCE_INDICES]).detach().cpu()):
            raise ValueError(
                "TorchDDPCM requires exact zeros in source columns (1,5,6,7)."
            )
        resource = self.estimate_resources(len(self._symbols))
        if not resource.within_limit:
            raise MemoryError(
                "TorchDDPCM dense preflight rejected allocation: estimated "
                f"{resource.conservative_peak_bytes} bytes exceeds limit "
                f"{resource.limit_bytes} bytes (B={resource.basis_dimension})."
            )
        geometry = self._geometry(positions)
        (
            directions,
            weights,
            radii,
            centres,
            nodes,
            pair_delta,
            pair_distance,
            chi,
            fi,
            ui,
        ) = geometry
        owner = torch.eye(
            len(self._symbols), dtype=torch.bool, device=positions.device
        )[:, None, :]
        ratios = torch.where(
            owner, torch.ones_like(pair_distance), pair_distance / radii[None, None, :]
        )
        topology = self._certify_topology(ratios, chi, fi, ui)
        phi, psi = self._point_source(nodes, centres, radii, source)
        L, D, design, ui = self._operators(positions, geometry)
        rhs_blocks = [
            -(design.T @ (weights * ui[i] * phi[i])) for i in range(len(self._symbols))
        ]
        rhs = torch.cat(rhs_blocks)
        identity = torch.eye(
            resource.basis_dimension, dtype=positions.dtype, device=positions.device
        )
        r_inf = 2.0 * math.pi * identity - D
        r_eps = (
            2.0
            * math.pi
            * (self._dielectric + 1.0)
            / (self._dielectric - 1.0)
            * identity
            - D
        )
        l_stability = self._validate_matrix_stability("L", L)
        reps_stability = self._validate_matrix_stability("R_epsilon", r_eps)
        try:
            dielectric_rhs = torch.linalg.solve(r_eps, r_inf @ rhs)
            solution = torch.linalg.solve(L, dielectric_rhs)
        except RuntimeError as exc:
            raise RuntimeError("TorchDDPCM dense linear solve failed.") from exc
        reps_residual = self._relative_residual(r_eps, dielectric_rhs, r_inf @ rhs)
        l_residual = self._relative_residual(L, solution, dielectric_rhs)
        # Direct LAPACK solves normally reach machine precision.  The factor
        # 100 protects a strict 1e-12 scientific contract from residual-rounding
        # noise without relaxing it to iterative-solver accuracy.
        limit = max(100.0 * np.finfo(np.float64).eps, self._solve_residual_tolerance)
        if reps_residual > limit or l_residual > limit:
            raise RuntimeError(
                "TorchDDPCM dense solve residual exceeded its declared tolerance."
            )
        energy = 0.5 * torch.dot(psi, solution) * HARTREE_TO_EV
        return TorchDDPCMState(
            energy=energy,
            L=L,
            D=D,
            rhs=rhs,
            dielectric_rhs=dielectric_rhs,
            solution=solution,
            topology=topology,
            resource=resource,
            l_stability=l_stability,
            reps_stability=reps_stability,
            reps_relative_residual=reps_residual,
            l_relative_residual=l_residual,
        )

    def _energy_torch(self, positions: Any, source: Any):
        return self._state_torch(positions, source).energy

    def diagnostics(self, positions: Any, source: Any) -> dict[str, object]:
        """Evaluate once and return detached diagnostics; never used for derivatives."""

        torch = __import__("torch")
        with torch.no_grad():
            state = self._state_torch(positions.detach(), source.detach())
        return {
            "provider_id": self.provider_id,
            "configuration_sha256": self.configuration_sha256(),
            "provenance_sha256": self.provenance_sha256,
            "energy_eV": float(state.energy.detach().cpu()),
            "topology": asdict(state.topology),
            "resource": asdict(state.resource),
            "L_stability": asdict(state.l_stability),
            "R_epsilon_stability": asdict(state.reps_stability),
            "reps_relative_residual": state.reps_relative_residual,
            "l_relative_residual": state.l_relative_residual,
        }


__all__ = [
    "DDPCMTopologyCertificate",
    "DenseResourceEstimate",
    "MatrixStabilityDiagnostics",
    "TORCH_DDPCM_PROVIDER_ID",
    "TorchDDPCM",
    "TorchDDPCMState",
]
