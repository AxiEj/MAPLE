"""Exact dense, structured-response ddPCM scalar.

This module deliberately owns only the certified linear solves and the
stationary-response algebra.  Geometry/operator derivatives are supplied by
``DDPCMResponseOperators`` and contain no autograd tape.  The resulting state
is evaluation-local: factors are never cached on the provider or returned by
public methods.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np
from ase.data import atomic_numbers

from maple.solvation.api.units import HARTREE_TO_EV
from maple.solvation.coupling.metrics import MACE_POLAR_RADIAL_GTO_PAIRING
from maple.solvation.coupling.spaces import (
    MACE_POLAR_RADIAL_GTO_FIELD_DUAL_SPACE,
    MACE_POLAR_RADIAL_GTO_SOURCE_SPACE,
)

from .functional import ContinuumEnergyFunctional
from .response_tensor_binding import TensorStateBinding, iter_named_tensors
from .torch_ddpcm import TorchDDPCM, _runtime_versions

DDPCM_RESPONSE_PROVIDER_ID = "maple.route2.continuum.torch-ddpcm-response.impl.v1"
DDPCM_RESPONSE_PROVENANCE = "ddx-0.8.0-exact-structured-stationary-response-v1"
_ACTIVE = (0, 2, 3, 4)
_INACTIVE = (1, 5, 6, 7)


def _sha(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tensor_sha(tensor: Any) -> str:
    array = np.ascontiguousarray(tensor.detach().cpu().numpy())
    digest = hashlib.sha256(str(array.shape).encode())
    digest.update(str(array.dtype).encode())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _immutable_json(value: object) -> object:
    """Return an immutable, JSON-value-only deep copy."""

    def plain(item: object) -> object:
        if isinstance(item, Mapping):
            return {str(key): plain(child) for key, child in item.items()}
        if isinstance(item, (tuple, list)):
            return [plain(child) for child in item]
        return item

    copied = json.loads(json.dumps(plain(value), sort_keys=True, allow_nan=False))

    def freeze(item: object) -> object:
        if isinstance(item, dict):
            return MappingProxyType({str(k): freeze(v) for k, v in item.items()})
        if isinstance(item, list):
            return tuple(freeze(v) for v in item)
        return item

    return freeze(copied)


@dataclass(frozen=True, slots=True)
class ResponseResourceEstimate:
    atom_count: int
    basis_dimension: int
    bytes_per_matrix: int
    operator_matrix_bytes: int
    factorization_bytes: int
    retained_matrix_bytes: int
    geometry_pair_grid_bytes: int
    geometry_node_field_bytes: int
    geometry_transient_bytes: int
    geometry_partial_bytes: int
    derivative_array_bytes: int
    tangent_solution_bytes: int
    harmonic_tile_bytes: int
    coefficient_table_bytes: int
    h0_bytes: int
    workspace_bytes: int
    host_peak_bytes: int
    device_peak_bytes: int
    conservative_peak_bytes: int
    derivative_order: int
    limit_bytes: int

    @property
    def within_limit(self) -> bool:
        return self.conservative_peak_bytes <= self.limit_bytes


@dataclass(slots=True)
class _ResponseState:
    energy: Any
    L: Any
    A: Any
    C: Any
    lu_l: Any
    piv_l: Any
    lu_a: Any
    piv_a: Any
    positions_sha256: str
    source_sha256: str
    source_active_sha256: str
    configuration_sha256: str
    diagnostics_base: Mapping[str, object]
    solve_records: list[dict[str, object]]
    atom_count: int
    derivative_order: int
    provider: Any
    primal: Any
    mu: Any
    z: Any
    lam: Any
    v0: Any
    alpha: Any
    solve_residual_limit: float
    factor_policy: str
    topology_sha256: str
    gradient: Any = None
    h0: Any = None
    T: Any = None
    Q: Any = None
    V: Any = None
    K: Any = None
    _tensor_binding: TensorStateBinding | None = None
    _control_binding: tuple[object, ...] | None = None

    def __setattr__(self, name: str, value: object) -> None:
        if name in {"_tensor_binding", "_control_binding"} and hasattr(self, name):
            raise AttributeError(f"{name} is a protected response-state binding.")
        object.__setattr__(self, name, value)

    def _named_tensors(self) -> tuple[tuple[str, Any], ...]:
        direct = tuple(
            (name, getattr(self, name))
            for name in (
                "energy",
                "L",
                "A",
                "C",
                "lu_l",
                "piv_l",
                "lu_a",
                "piv_a",
                "mu",
                "z",
                "lam",
                "v0",
                "alpha",
                "gradient",
                "h0",
                "T",
                "Q",
                "V",
                "K",
            )
            if getattr(self, name) is not None
        )
        return direct + iter_named_tensors(self.primal, "primal")

    def _controls(self) -> tuple[object, ...]:
        return (
            id(self.provider),
            self.positions_sha256,
            self.source_sha256,
            self.source_active_sha256,
            self.configuration_sha256,
            id(self.diagnostics_base),
            _control_sha(self.diagnostics_base),
            id(self.solve_records),
            _control_sha(self.solve_records),
            self.atom_count,
            self.derivative_order,
            self.solve_residual_limit,
            self.factor_policy,
            self.topology_sha256,
            _sha(asdict(self.primal.geometry.topology)),
        )

    def _refresh_bindings(self) -> None:
        object.__setattr__(
            self, "_tensor_binding", TensorStateBinding.capture(self._named_tensors())
        )
        object.__setattr__(self, "_control_binding", self._controls())

    def _validate(self) -> None:
        if self._tensor_binding is None or self._control_binding is None:
            raise RuntimeError("DDPCMResponse state is not integrity-bound.")
        if self._controls() != self._control_binding:
            raise RuntimeError("DDPCMResponse state controls changed after binding.")
        self._tensor_binding.validate(self._named_tensors())
        if self.configuration_sha256 != self.provider.configuration_sha256():
            raise RuntimeError(
                "DDPCMResponse configuration changed after linearization."
            )

    def _ensure(self, order: int) -> None:
        self._validate()
        if self.derivative_order < order:
            self.provider._upgrade_state(self, order)
            self._validate()

    def gradient_partial(self) -> tuple[Any, Any]:
        self._ensure(1)
        n = self.atom_count
        source_gradient = self.gradient.new_zeros((n, 8))
        source_gradient[:, _ACTIVE] = self.gradient[3 * n :].reshape(n, 4)
        return self.gradient[: 3 * n].reshape(n, 3), source_gradient

    def hessian_partial(self) -> Any:
        self._ensure(2)
        limit = self.solve_residual_limit
        Y = _lu_solve_checked(
            self.lu_a,
            self.piv_a,
            self.A,
            self.T,
            transpose=False,
            label="A-matrix-tangent",
            limit=limit,
            records=self.solve_records,
        )
        Z = _lu_solve_checked(
            self.lu_l,
            self.piv_l,
            self.L,
            Y - self.V,
            transpose=False,
            label="L-matrix-tangent",
            limit=limit,
            records=self.solve_records,
        )
        response = self.K.mT @ Y + self.Q.mT @ Z
        result = self.h0 + response + response.mT
        self._refresh_bindings()
        return result

    def hvp_partial(self, direction: Any) -> Any:
        self._ensure(2)
        torch = __import__("torch")
        vector = torch.as_tensor(
            direction, dtype=self.gradient.dtype, device=self.gradient.device
        )
        if vector.shape != (7 * self.atom_count,) or not bool(
            torch.isfinite(vector).all()
        ):
            raise ValueError(
                f"direction must be finite with shape ({7 * self.atom_count},)."
            )
        return _hvp_from_state(self, vector)

    def diagnostics(self) -> Mapping[str, object]:
        self._validate()
        payload = dict(self.diagnostics_base)
        payload["resource"] = asdict(
            self.provider.estimate_resources(derivative_order=self.derivative_order)
        )
        payload["solve_records"] = tuple(dict(record) for record in self.solve_records)
        return _immutable_json(payload)

    def tensors_for_autograd(self) -> tuple[Any, ...]:
        self._validate()
        assert self._tensor_binding is not None
        return self._tensor_binding.tensors


def _as_matrix(rhs: Any) -> tuple[Any, bool]:
    return (rhs[:, None], True) if rhs.ndim == 1 else (rhs, False)


def _control_sha(value: object) -> str:
    """Hash JSON-like controls, including immutable mapping proxies."""

    def plain(item: object) -> object:
        if isinstance(item, Mapping):
            return {str(key): plain(child) for key, child in item.items()}
        if isinstance(item, (tuple, list)):
            return [plain(child) for child in item]
        return item

    return _sha(plain(value))


def _relative_residual_columns(
    matrix: Any, solution: Any, rhs: Any
) -> tuple[float, ...]:
    torch = __import__("torch")
    x, squeezed = _as_matrix(solution)
    b, _ = _as_matrix(rhs)
    residual = torch.linalg.vector_norm(matrix @ x - b, dim=0)
    scale = torch.maximum(torch.linalg.vector_norm(b, dim=0), b.new_ones((b.shape[1],)))
    values = residual / scale
    if not bool(torch.isfinite(values).all().detach().cpu()):
        raise RuntimeError("DDPCMResponse solve produced a non-finite residual.")
    result = tuple(float(v) for v in values.detach().cpu())
    return result[:1] if squeezed else result


def _lu_solve_checked(
    lu: Any,
    pivots: Any,
    matrix: Any,
    rhs: Any,
    *,
    transpose: bool,
    label: str,
    limit: float,
    records: list[dict[str, object]],
) -> Any:
    torch = __import__("torch")
    b, squeezed = _as_matrix(rhs)
    try:
        solution = torch.linalg.lu_solve(lu, pivots, b, adjoint=transpose)
    except RuntimeError as exc:
        raise RuntimeError(f"DDPCMResponse {label} solve failed.") from exc
    original = matrix.mT if transpose else matrix
    residuals = _relative_residual_columns(original, solution, b)
    worst = max(range(len(residuals)), key=residuals.__getitem__)
    records.append(
        {
            "equation": label,
            "transpose": transpose,
            "relative_residuals": residuals,
            "maximum_relative_residual": residuals[worst],
            "worst_rhs_index": worst,
        }
    )
    if any(value > limit for value in residuals):
        raise RuntimeError(
            f"DDPCMResponse {label} solve residual exceeded {limit:.3e}."
        )
    return solution[:, 0] if squeezed else solution


class _GradientFunction(__import__("torch").autograd.Function):
    @staticmethod
    def forward(
        ctx, positions: Any, source: Any, state: _ResponseState, grad_output: Any
    ):
        state._ensure(1)
        ctx.state = state
        ctx.save_for_backward(
            positions, source, grad_output, *state.tensors_for_autograd()
        )
        n = positions.shape[0]
        gradient = state.gradient
        active = gradient[3 * n :].reshape(n, 4)
        source_gradient = source.new_zeros(source.shape)
        source_gradient[:, _ACTIVE] = active
        return (
            grad_output * gradient[: 3 * n].reshape_as(positions),
            grad_output * source_gradient,
        )

    @staticmethod
    @__import__("torch").autograd.function.once_differentiable
    def backward(ctx, position_cotangent: Any, source_cotangent: Any):
        saved = ctx.saved_tensors
        positions, source, grad_output = saved[:3]
        state = ctx.state
        state._ensure(2)
        n = positions.shape[0]
        if position_cotangent is None:
            position_cotangent = __import__("torch").zeros_like(positions)
        if source_cotangent is None:
            source_cotangent = __import__("torch").zeros_like(source)
        direction = __import__("torch").cat(
            (
                position_cotangent.reshape(-1),
                source_cotangent[:, _ACTIVE].reshape(-1),
            )
        )
        action = _hvp_from_state(state, direction)
        position_action = grad_output * action[: 3 * n].reshape_as(positions)
        source_action = source.new_zeros(source.shape)
        source_action[:, _ACTIVE] = (grad_output * action[3 * n :]).reshape(n, 4)
        grad_output_cotangent = __import__("torch").dot(direction, state.gradient)
        return position_action, source_action, None, grad_output_cotangent


class _EnergyFunction(__import__("torch").autograd.Function):
    @staticmethod
    def forward(ctx, positions: Any, source: Any, provider: "TorchDDPCMResponse"):
        state = provider._linearize(positions, source, derivative_order=0)
        if state.configuration_sha256 != provider.configuration_sha256():
            raise RuntimeError("DDPCMResponse configuration changed during evaluation.")
        ctx.state = state
        ctx.provider = provider
        ctx.save_for_backward(positions, source, *state.tensors_for_autograd())
        # Autograd attaches this Function to its returned tensor.  Keep the
        # private numerical energy detached, avoiding a context/output cycle
        # that would otherwise retain all dense factors until cyclic GC.
        return state.energy.clone()

    @staticmethod
    def backward(ctx, grad_output: Any):
        saved = ctx.saved_tensors
        positions, source = saved[:2]
        if ctx.state.configuration_sha256 != ctx.provider.configuration_sha256():
            raise RuntimeError("DDPCMResponse configuration changed after forward.")
        # A create_graph=True request announces that this first backward will
        # itself be differentiated.  Build the complete order-2 response once
        # rather than first running the order-1 kernel and recomputing it when
        # _GradientFunction.backward is entered.  Ordinary force evaluation
        # keeps the cheaper order-1 path.
        if __import__("torch").is_grad_enabled():
            ctx.state._ensure(2)
        gradient_r, gradient_s = _GradientFunction.apply(
            positions, source, ctx.state, grad_output
        )
        return gradient_r, gradient_s, None


def _hvp_from_state(state: _ResponseState, direction: Any) -> Any:
    state._ensure(2)
    limit = state.solve_residual_limit
    dy = _lu_solve_checked(
        state.lu_a,
        state.piv_a,
        state.A,
        state.T @ direction,
        transpose=False,
        label="A-tangent",
        limit=limit,
        records=state.solve_records,
    )
    dz = _lu_solve_checked(
        state.lu_l,
        state.piv_l,
        state.L,
        dy - state.V @ direction,
        transpose=False,
        label="L-tangent",
        limit=limit,
        records=state.solve_records,
    )
    dmu = _lu_solve_checked(
        state.lu_l,
        state.piv_l,
        state.L,
        state.Q @ direction,
        transpose=True,
        label="L-adjoint-tangent",
        limit=limit,
        records=state.solve_records,
    )
    dlam = _lu_solve_checked(
        state.lu_a,
        state.piv_a,
        state.A,
        state.K @ direction + dmu,
        transpose=True,
        label="A-adjoint-tangent",
        limit=limit,
        records=state.solve_records,
    )
    result = (
        state.h0 @ direction
        + state.K.mT @ dy
        + state.Q.mT @ dz
        + state.T.mT @ dlam
        - state.V.mT @ dmu
    )
    state._refresh_bindings()
    return result


class TorchDDPCMResponse(ContinuumEnergyFunctional):
    """Experimental exact structured-response ddPCM functional."""

    __slots__ = (
        "_configuration_sha256",
        "_dielectric",
        "_eta",
        "_lmax",
        "_max_dense_bytes",
        "_n_lebedev",
        "_operator",
        "_provenance_sha256",
        "_radii_angstrom",
        "_solve_residual_tolerance",
        "_solid_harmonic_contracted_table_sha256",
        "_solid_harmonic_table_sha256",
        "_symbols",
        "_topology_margin",
    )

    provider_id = DDPCM_RESPONSE_PROVIDER_ID
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
            raise TypeError("TorchDDPCMResponse supports torch.float64 only.")
        requested_device = torch.device(device)
        if requested_device.type not in {"cpu", "cuda"}:
            raise ValueError("TorchDDPCMResponse device must be CPU or CUDA.")
        if requested_device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("requested CUDA device is unavailable.")
            requested_device = torch.device(
                "cuda",
                (
                    torch.cuda.current_device()
                    if requested_device.index is None
                    else requested_device.index
                ),
            )
        symbol_tuple = tuple(str(s) for s in symbols)
        if not symbol_tuple or any(s not in atomic_numbers for s in symbol_tuple):
            raise ValueError("symbols must contain recognized element symbols.")
        radii = np.asarray(radii_angstrom, dtype=np.float64)
        if (
            radii.shape != (len(symbol_tuple),)
            or not np.all(np.isfinite(radii))
            or np.any(radii <= 0)
        ):
            raise ValueError("radii_angstrom must be finite and positive per atom.")
        if isinstance(lmax, bool) or not isinstance(lmax, int) or lmax < 1:
            raise ValueError("lmax must be a positive integer.")
        if (
            isinstance(n_lebedev, bool)
            or not isinstance(n_lebedev, int)
            or n_lebedev < 1
        ):
            raise ValueError("n_lebedev must be a positive integer.")
        dielectric = float(dielectric)
        if not math.isfinite(dielectric) or dielectric <= 1:
            raise ValueError("dielectric must be finite and greater than one.")
        eta = float(eta)
        if not math.isfinite(eta) or not 0 < eta <= 1:
            raise ValueError("eta must lie in (0, 1].")
        if (
            isinstance(max_dense_bytes, bool)
            or not isinstance(max_dense_bytes, int)
            or max_dense_bytes <= 0
        ):
            raise ValueError("max_dense_bytes must be a positive integer.")
        solve_residual_tolerance = float(solve_residual_tolerance)
        topology_margin = float(topology_margin)
        if not math.isfinite(solve_residual_tolerance) or solve_residual_tolerance <= 0:
            raise ValueError("solve_residual_tolerance must be positive.")
        if not math.isfinite(topology_margin) or topology_margin <= 0:
            raise ValueError("topology_margin must be positive.")

        estimate = self._resource_estimate(
            len(symbol_tuple),
            lmax,
            n_lebedev,
            2,
            max_dense_bytes,
            requested_device.type,
        )
        if not estimate.within_limit:
            raise MemoryError("DDPCMResponse preflight rejected allocation.")
        from .ddpcm_response_operators import DDPCMResponseOperators

        operator = DDPCMResponseOperators(
            symbol_tuple,
            radii,
            dielectric=dielectric,
            lmax=lmax,
            n_lebedev=n_lebedev,
            eta=eta,
            device=requested_device,
            topology_margin=topology_margin,
        )
        object.__setattr__(self, "_symbols", symbol_tuple)
        object.__setattr__(
            self, "_radii_angstrom", np.frombuffer(radii.tobytes(), dtype=np.float64)
        )
        object.__setattr__(self, "_dielectric", dielectric)
        object.__setattr__(self, "_lmax", lmax)
        object.__setattr__(self, "_n_lebedev", n_lebedev)
        object.__setattr__(self, "_eta", eta)
        object.__setattr__(self, "_max_dense_bytes", max_dense_bytes)
        object.__setattr__(self, "_solve_residual_tolerance", solve_residual_tolerance)
        object.__setattr__(self, "_topology_margin", topology_margin)
        object.__setattr__(self, "_operator", operator)
        from .solid_harmonic_response import (
            contracted_hessian_table_sha256,
            derivative_table_sha256,
        )

        object.__setattr__(
            self,
            "_solid_harmonic_table_sha256",
            MappingProxyType(
                {
                    "regular": derivative_table_sha256(lmax, "regular"),
                    "irregular": derivative_table_sha256(lmax, "irregular"),
                }
            ),
        )
        object.__setattr__(
            self,
            "_solid_harmonic_contracted_table_sha256",
            MappingProxyType(
                {
                    "regular": contracted_hessian_table_sha256(lmax, "regular"),
                    "irregular": contracted_hessian_table_sha256(lmax, "irregular"),
                }
            ),
        )
        # Needed while constructing the configuration payload; the base class
        # records the identical device again after identity has been frozen.
        object.__setattr__(self, "_torch_device", requested_device)
        configuration = _sha(self._configuration_payload())
        object.__setattr__(self, "_configuration_sha256", configuration)
        object.__setattr__(
            self,
            "_provenance_sha256",
            _sha(
                {
                    "configuration": configuration,
                    "provenance": DDPCM_RESPONSE_PROVENANCE,
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

    @staticmethod
    def _resource_estimate(
        atom_count: int,
        lmax: int,
        n_lebedev: int,
        derivative_order: int,
        limit: int,
        device_type: str = "cpu",
    ) -> ResponseResourceEstimate:
        basis = atom_count * (lmax + 1) ** 2
        matrix_bytes = 8 * basis * basis
        dimension = 7 * atom_count
        # ``_geometry`` is built before derivative dispatch and always retains
        # the same 19 FP64 pair-grid equivalents. Node/point fields and the
        # largest RHS scratch family are reported separately rather than being
        # hidden in a matrix multiplier.
        pair_grid = 19 * 8 * atom_count * atom_count * n_lebedev
        node_fields = 8 * (7 * atom_count * n_lebedev + 3 * atom_count)
        geometry_transient = 18 * 8 * atom_count * atom_count * n_lebedev
        geometry = pair_grid + node_fields + geometry_transient
        # Jr/Jp, four unpadded action Jacobians, padded V/W/K/M, and retained
        # T/Q overlap. Ten Bxd FP64 arrays conservatively cover that peak
        # without pretending they are dense BxB operators. Full-Hessian Y/Z
        # tangent solutions add two more Bxd arrays at order two.
        derivative_arrays = 0 if derivative_order == 0 else 10 * basis * dimension * 8
        tangent_solutions = 2 * basis * dimension * 8 if derivative_order == 2 else 0
        # ``harmonic_tile_bytes`` is a conservative simultaneous-allocation
        # upper bound, not a claim about arrays retained after the helper
        # returns.  Order two retains basis values/gradients and only two Gx3x3
        # contracted Hessians.  The bound also counts the irregular internal
        # lmax+2 values, a second physical-value view, and the largest q_internal
        # by nine transformed-coefficient workspace.
        physical_basis = (lmax + 1) ** 2
        internal_basis = (lmax + derivative_order + 1) ** 2
        if derivative_order == 2:
            grid_components = 8 * physical_basis + internal_basis + 18
            transformed_coefficients = 9 * internal_basis
        else:
            jet_components = (1, 4)[derivative_order]
            grid_components = 2 * jet_components * physical_basis + internal_basis
            transformed_coefficients = 0
        tile = 8 * (n_lebedev * grid_components + transformed_coefficients)
        from .solid_harmonic_response import (
            contracted_hessian_table_nbytes,
            derivative_table_nbytes,
        )

        coefficient_tables = sum(
            derivative_table_nbytes(lmax, kind)
            + contracted_hessian_table_nbytes(lmax, kind)
            for kind in ("regular", "irregular")
        )
        h0 = 8 * dimension * dimension if derivative_order == 2 else 0
        # Four named operators (L,D,A,C) and two reusable LU factors are
        # persistent. The remaining ten-matrix allowance covers identity,
        # SVD/LU, and batched-solve workspaces, preserving the existing 16M
        # conservative peak without misreporting ten retained matrices.
        operators = 4 * matrix_bytes
        factors = 2 * matrix_bytes
        retained_matrices = operators + factors
        workspace = 10 * matrix_bytes
        peak = (
            retained_matrices
            + geometry
            + derivative_arrays
            + tangent_solutions
            + tile
            + coefficient_tables
            + h0
            + workspace
        )
        host_peak = peak if device_type == "cpu" else coefficient_tables
        device_peak = 0 if device_type == "cpu" else peak - coefficient_tables
        return ResponseResourceEstimate(
            atom_count,
            basis,
            matrix_bytes,
            operators,
            factors,
            retained_matrices,
            pair_grid,
            node_fields,
            geometry_transient,
            geometry,
            derivative_arrays,
            tangent_solutions,
            tile,
            coefficient_tables,
            h0,
            workspace,
            host_peak,
            device_peak,
            peak,
            derivative_order,
            limit,
        )

    def estimate_resources(
        self,
        atom_count: int | None = None,
        *,
        derivative_order: int = 2,
        limit_bytes: int | None = None,
    ) -> ResponseResourceEstimate:
        count = len(self._symbols) if atom_count is None else atom_count
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError("atom_count must be a positive integer.")
        if derivative_order not in (0, 1, 2) or isinstance(derivative_order, bool):
            raise ValueError("derivative_order must be exactly 0, 1, or 2.")
        limit = self._max_dense_bytes if limit_bytes is None else limit_bytes
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError("limit_bytes must be a positive integer.")
        return self._resource_estimate(
            count,
            self._lmax,
            self._n_lebedev,
            derivative_order,
            limit,
            self._torch_device.type,
        )

    def preflight_resources(
        self,
        *,
        derivative_order: int,
        atom_count: int | None = None,
        limit_bytes: int | None = None,
    ) -> ResponseResourceEstimate:
        result = self.estimate_resources(
            atom_count, derivative_order=derivative_order, limit_bytes=limit_bytes
        )
        if not result.within_limit:
            raise MemoryError(
                f"DDPCMResponse preflight rejected derivative order {derivative_order}: estimated {result.conservative_peak_bytes} bytes exceeds limit {result.limit_bytes} bytes (B={result.basis_dimension})."
            )
        return result

    @property
    def symbols(self) -> tuple[str, ...]:
        return self._symbols

    @property
    def radii_angstrom(self) -> np.ndarray:
        return self._radii_angstrom

    @property
    def provenance_sha256(self) -> str:
        self.configuration_sha256()
        return self._provenance_sha256

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

    def _configuration_payload(self) -> dict[str, object]:
        continuum = Path(__file__).resolve().parent
        solvation = continuum.parent
        operator_path = continuum / "ddpcm_response_operators.py"
        operator_payload_getter = getattr(self._operator, "configuration_payload", None)
        operator_payload = (
            operator_payload_getter()
            if callable(operator_payload_getter)
            else {"operator_class": type(self._operator).__qualname__}
        )
        implementation_paths = {
            "solvation.api.units": solvation / "api" / "units.py",
            "solvation.continuum.ddpcm_response": Path(__file__),
            "solvation.continuum.ddpcm_response_operators": operator_path,
            "solvation.continuum.response_tensor_binding": continuum
            / "response_tensor_binding.py",
            "solvation.continuum.functional": continuum / "functional.py",
            "solvation.continuum.harmonic_coefficients": continuum
            / "harmonic_coefficients.py",
            "solvation.continuum.harmonic_torch_primitives": continuum
            / "harmonic_torch_primitives.py",
            "solvation.continuum.solid_harmonic_response": continuum
            / "solid_harmonic_response.py",
            "solvation.continuum.torch_ddpcm_guard": continuum / "torch_ddpcm.py",
            "solvation.coupling.metrics": solvation / "coupling" / "metrics.py",
            "solvation.coupling.spaces": solvation / "coupling" / "spaces.py",
            "solvation.surfaces.lebedev": solvation / "surfaces" / "lebedev.py",
        }
        implementation_hashes = {
            name: _file_sha(path)
            for name, path in implementation_paths.items()
            if path.is_file()
        }
        if not operator_path.is_file():
            implementation_hashes["solvation.continuum.ddpcm_response_operators"] = (
                _sha({"operator_class": type(self._operator).__qualname__})
            )
        return {
            "provider_id": self.provider_id,
            "provenance": DDPCM_RESPONSE_PROVENANCE,
            "symbols": self._symbols,
            "radii_angstrom": self._radii_angstrom.tolist(),
            "dielectric": self._dielectric,
            "lmax": self._lmax,
            "n_lebedev": self._n_lebedev,
            "eta": self._eta,
            "device": (
                str(self._torch_device)
                if hasattr(self, "_torch_device")
                else str(getattr(self._operator, "device", "cpu"))
            ),
            "source_layout": {"active": _ACTIVE, "inactive": _INACTIVE},
            "operator_configuration": operator_payload,
            "runtime_versions": dict(_runtime_versions()),
            "solid_harmonic_derivative_tables": {
                "regular": self._solid_harmonic_table_sha256["regular"],
                "irregular": self._solid_harmonic_table_sha256["irregular"],
            },
            "solid_harmonic_contracted_hessian_tables": {
                "regular": self._solid_harmonic_contracted_table_sha256["regular"],
                "irregular": self._solid_harmonic_contracted_table_sha256["irregular"],
            },
            "numerical_policy": {
                "dtype": "torch.float64",
                "factorizations": 2,
                "max_dense_bytes": self._max_dense_bytes,
                "matrix_peak_factor": 16,
                "resource_model": "named-retained-plus-workspace-and-contracted-harmonic-upper-bound-v3",
                "stability": "full-svd",
                "minimum_rcond": math.sqrt(np.finfo(np.float64).eps),
                "solve_residual_tolerance": self._solve_residual_tolerance,
                "topology_margin": self._topology_margin,
                "fallback": "forbidden",
            },
            "implementation_sha256": implementation_hashes,
        }

    def configuration_sha256(self) -> str:
        if _sha(self._configuration_payload()) != self._configuration_sha256:
            raise RuntimeError("DDPCMResponse implementation configuration changed.")
        return self._configuration_sha256

    def _validate_inputs(self, positions: Any, source: Any) -> Any:
        torch = __import__("torch")
        if not torch.is_tensor(positions) or not torch.is_tensor(source):
            raise TypeError("positions and source must be Torch tensors.")
        if positions.shape != (len(self._symbols), 3) or source.shape != (
            len(self._symbols),
            8,
        ):
            raise ValueError("positions/source shapes must be (N,3) and (N,8).")
        if (
            positions.dtype is not torch.float64
            or source.dtype is not torch.float64
            or positions.device != source.device
            or positions.device != self._torch_device
            or not bool(torch.isfinite(positions).all())
            or not bool(torch.isfinite(source).all())
        ):
            raise ValueError(
                "positions/source must be finite FP64 tensors on the configured device."
            )
        if bool(torch.count_nonzero(source[:, _INACTIVE]).detach().cpu()):
            raise ValueError(
                "DDPCMResponse requires exact zeros in source columns (1,5,6,7)."
            )
        return source[:, _ACTIVE]

    def _upgrade_state(self, state: _ResponseState, derivative_order: int) -> None:
        """Populate response arrays without repeating certification/factorization."""

        state._validate()
        if derivative_order not in (1, 2):
            raise ValueError("response-state upgrade order must be 1 or 2.")
        if state.derivative_order >= derivative_order:
            return
        self.preflight_resources(derivative_order=derivative_order)
        torch = __import__("torch")
        with torch.no_grad():
            if state.mu is None:
                limit = state.solve_residual_limit
                state.mu = _lu_solve_checked(
                    state.lu_l,
                    state.piv_l,
                    state.L,
                    (HARTREE_TO_EV / 2.0) * state.primal.psi,
                    transpose=True,
                    label="L-adjoint",
                    limit=limit,
                    records=state.solve_records,
                )
                state.lam = _lu_solve_checked(
                    state.lu_a,
                    state.piv_a,
                    state.A,
                    state.mu,
                    transpose=True,
                    label="A-adjoint",
                    limit=limit,
                    records=state.solve_records,
                )
                state.alpha = state.C.mT @ state.lam
            derivatives = self._operator.derivatives(
                state.primal,
                mu=state.mu,
                z=state.z,
                lam=state.lam,
                v0=state.v0,
                alpha=state.alpha,
                order=derivative_order,
            )
            Jr = derivatives.rhs_jacobian
            Jp = derivatives.psi_jacobian
            basis = state.L.shape[0]
            source_zeros = state.L.new_zeros((basis, 4 * state.atom_count))
            V = torch.cat((derivatives.l_action_jacobian, source_zeros), dim=1)
            W = torch.cat((derivatives.d_action_jacobian, source_zeros), dim=1)
            K = torch.cat((derivatives.dt_action_jacobian, source_zeros), dim=1)
            M = torch.cat((derivatives.lt_action_jacobian, source_zeros), dim=1)
            state.V = V
            state.K = K
            state.T = state.C @ Jr + W
            state.Q = (HARTREE_TO_EV / 2.0) * Jp - M
            state.gradient = (
                (HARTREE_TO_EV / 2.0) * (Jp.mT @ state.z)
                + Jr.mT @ state.alpha
                - V.mT @ state.mu
                + W.mT @ state.lam
            )
            if derivative_order == 2:
                dimension = 7 * state.atom_count
                coordinate_dimension = 3 * state.atom_count
                operator_hessian = state.L.new_zeros((dimension, dimension))
                operator_hessian[:coordinate_dimension, :coordinate_dimension] = (
                    -derivatives.l_contraction_hessian
                    + derivatives.d_contraction_hessian
                )
                state.h0 = (
                    derivatives.rhs_contraction_hessian
                    + operator_hessian
                    - K.mT @ Jr
                    - Jr.mT @ K
                )
            state.derivative_order = derivative_order
            state._refresh_bindings()

    def _linearize(
        self, positions: Any, source: Any, derivative_order: int = 2
    ) -> _ResponseState:
        torch = __import__("torch")
        configuration = self.configuration_sha256()
        self.preflight_resources(derivative_order=derivative_order)
        active = self._validate_inputs(positions, source)
        with torch.no_grad():
            primal = self._operator.build(positions.detach(), active.detach())
            L, D, rhs, psi = primal.L, primal.D, primal.rhs, primal.psi
            basis = L.shape[0]
            identity = torch.eye(basis, dtype=L.dtype, device=L.device)
            A = (
                2.0
                * math.pi
                * (self._dielectric + 1.0)
                / (self._dielectric - 1.0)
                * identity
                - D
            )
            C = 2.0 * math.pi * identity - D
            l_stability = TorchDDPCM._validate_matrix_stability("L", L)
            a_stability = TorchDDPCM._validate_matrix_stability("R_epsilon", A)
            try:
                lu_a, piv_a = torch.linalg.lu_factor(A)
                lu_l, piv_l = torch.linalg.lu_factor(L)
            except RuntimeError as exc:
                raise RuntimeError("DDPCMResponse LU factorization failed.") from exc
            limit = max(
                100.0 * np.finfo(np.float64).eps, self._solve_residual_tolerance
            )
            records: list[dict[str, object]] = []
            y = _lu_solve_checked(
                lu_a,
                piv_a,
                A,
                C @ rhs,
                transpose=False,
                label="A-primal",
                limit=limit,
                records=records,
            )
            z = _lu_solve_checked(
                lu_l,
                piv_l,
                L,
                y,
                transpose=False,
                label="L-primal",
                limit=limit,
                records=records,
            )
            v0 = y - rhs
            energy = (HARTREE_TO_EV / 2.0) * torch.dot(psi, z)
            resource = self.estimate_resources(derivative_order=derivative_order)
            diagnostics = _immutable_json(
                {
                    "provider_id": self.provider_id,
                    "factor_policy": "two-full-svd-certificates-two-reusable-lu-factors-v1",
                    "solid_harmonic_derivative_tables": dict(
                        self._solid_harmonic_table_sha256
                    ),
                    "solid_harmonic_contracted_hessian_tables": dict(
                        self._solid_harmonic_contracted_table_sha256
                    ),
                    "configuration_sha256": configuration,
                    "provenance_sha256": self._provenance_sha256,
                    "energy_eV": float(energy.cpu()),
                    "topology": asdict(primal.geometry.topology),
                    "resource": asdict(resource),
                    "L_stability": asdict(l_stability),
                    "R_epsilon_stability": asdict(a_stability),
                    "solve_residual_limit": limit,
                }
            )
            state = _ResponseState(
                energy=energy,
                L=L,
                A=A,
                C=C,
                lu_l=lu_l,
                piv_l=piv_l,
                lu_a=lu_a,
                piv_a=piv_a,
                positions_sha256=_tensor_sha(positions),
                source_sha256=_tensor_sha(source),
                source_active_sha256=_tensor_sha(active),
                configuration_sha256=configuration,
                diagnostics_base=diagnostics,
                solve_records=records,
                atom_count=len(self._symbols),
                derivative_order=0,
                provider=self,
                primal=primal,
                mu=None,
                z=z,
                lam=None,
                v0=v0,
                alpha=None,
                solve_residual_limit=limit,
                factor_policy="two-full-svd-certificates-two-reusable-lu-factors-v1",
                topology_sha256=_sha(asdict(primal.geometry.topology)),
            )
            state._refresh_bindings()
            if derivative_order:
                self._upgrade_state(state, derivative_order)
            return state

    def _energy_torch(self, positions: Any, source: Any):
        return _EnergyFunction.apply(positions, source, self)

    def gradient_partial(self, positions: Any, source: Any) -> tuple[Any, Any]:
        return self._linearize(positions, source, derivative_order=1).gradient_partial()

    def hessian_partial(self, positions: Any, source: Any) -> Any:
        return self._linearize(positions, source, derivative_order=2).hessian_partial()

    def hvp_partial(self, positions: Any, source: Any, direction: Any) -> Any:
        state = self._linearize(positions, source, derivative_order=2)
        return state.hvp_partial(direction)

    def diagnostics(self, positions: Any, source: Any) -> Mapping[str, object]:
        return self._linearize(positions, source, derivative_order=1).diagnostics()


__all__ = [
    "DDPCM_RESPONSE_PROVIDER_ID",
    "DDPCM_RESPONSE_PROVENANCE",
    "ResponseResourceEstimate",
    "TorchDDPCMResponse",
]
