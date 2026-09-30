"""Identity-neutral mechanics for structured ddPCM response compositions."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import InitVar, asdict, dataclass, field
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Mapping

import numpy as np

from maple.function.calculator.extra_correction.implicit.route2_domain import (
    validate_route2_domain,
)
from maple.solvation.coupling.exact_gto import (
    mace_polar_learned_source_embedding_matrix,
)
from maple.solvation.coupling.operator import source_files_sha256
from maple.solvation.coupling.state_equation import geometry_sha256
from maple.solvation.derivatives.response import ResponseHessianEvaluation

from .mace_polar_torch import (
    _available_host_memory_bytes,
    _component_hash,
    _graph_has_path,
    _require_graph_inspection_api,
)

_TEST_COMPONENTS_TOKEN = object()
_BUILDER_TOKEN = object()
_ACTIVE_SOURCE_COLUMNS = (0, 2, 3, 4)
_SOURCE_CHARGE_ATOL_E = 1.0e-8
_HESSIAN_PARITY_BUDGET_EV_A2 = 1.0e-4
_SECOND_ORDER_CONTINUUM_LIMIT_BYTES = 4_000_000_000
_CPU_MODEL_RUNTIME_BASE_BYTES = 2 * 1024**3
_CPU_MODEL_RUNTIME_PER_ATOM_BYTES = 64 * 1024**2
_CUDA_MODEL_RUNTIME_BASE_BYTES = 1 * 1024**3
_CUDA_MODEL_RUNTIME_PER_ATOM_BYTES = 32 * 1024**2


@dataclass(frozen=True, slots=True)
class ResponseCompositionPolicy:
    """Closed identity and component policy supplied by a composition module."""

    policy_id: str
    provider_id: str
    ids_for_device: Callable[[str], Mapping[str, str]] = field(
        repr=False, compare=False
    )
    validate_components: Callable[[object, object, object], None] = field(
        repr=False, compare=False
    )
    source_files: Callable[[], Mapping[str, Path]] = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.policy_id or not self.provider_id:
            raise ValueError("Response composition policy requires explicit IDs.")

    def ids(self, device: str) -> Mapping[str, str]:
        ids = MappingProxyType(dict(self.ids_for_device(device)))
        if set(ids) != {"provider_id", "profile_id", "scalar_contract_id"}:
            raise ValueError("Response composition policy returned invalid IDs.")
        if ids["provider_id"] != self.provider_id or len(set(ids.values())) != 3:
            raise ValueError(
                "Response composition identities must be explicit/distinct."
            )
        return ids


def _sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _tensor_sha256(value) -> str:
    array = np.ascontiguousarray(value.detach().cpu().numpy())
    digest = hashlib.sha256(str(array.shape).encode())
    digest.update(str(array.dtype).encode())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _model_output_sha256(vacuum, learned) -> str:
    return _sha(
        {
            "vacuum": _tensor_sha256(vacuum.reshape(1)),
            "learned_source": _tensor_sha256(learned),
        }
    )


def _response_resource_preflight(
    continuum, device: str, atom_count: int, derivative_order: int
) -> dict:
    estimate = continuum.preflight_resources(
        derivative_order=derivative_order,
        atom_count=atom_count,
        limit_bytes=_SECOND_ORDER_CONTINUUM_LIMIT_BYTES,
    )
    host_available = _available_host_memory_bytes()
    host_reserve = (
        _CPU_MODEL_RUNTIME_BASE_BYTES + atom_count * _CPU_MODEL_RUNTIME_PER_ATOM_BYTES
    )
    device_reserve = (
        host_reserve
        if device == "cpu"
        else _CUDA_MODEL_RUNTIME_BASE_BYTES
        + atom_count * _CUDA_MODEL_RUNTIME_PER_ATOM_BYTES
    )
    required = estimate.conservative_peak_bytes + device_reserve
    if device == "cpu":
        available = host_available
    else:
        import torch

        free, _ = torch.cuda.mem_get_info(device)
        reusable = torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(
            device
        )
        available = int(free + reusable)
        if host_available < host_reserve:
            raise MemoryError("Insufficient host memory for response evaluation.")
    if required > available:
        raise MemoryError(
            "Structured-response preflight requires an estimated "
            f"{required} bytes including runtime reserve, but {device} has "
            f"{available} bytes available."
        )
    return {
        "policy": "torch-ddpcm-structured-response-memory-envelope-v1",
        "continuum": asdict(estimate),
        "model_runtime_reserve_bytes": device_reserve,
        "host_runtime_reserve_bytes": host_reserve,
        "estimated_total_bytes": required,
        "device_available_bytes": available,
        "host_available_bytes": host_available,
        "claim_boundary": "engineering estimate, not an allocator guarantee",
    }


@dataclass(frozen=True, slots=True)
class ResponsePESCore:
    """Fixed-component total PES with stationary-response full Hessians."""

    model: object
    continuum: object
    solvent_term: object
    symbols: tuple[str, ...]
    device: str = "cpu"
    solvent: str = "water"
    _policy: InitVar[ResponseCompositionPolicy | None] = None
    _testing_token: InitVar[object | None] = None
    _builder_token: InitVar[object | None] = None
    scalar_contract_id: str = field(init=False)
    profile_id: str = field(init=False)
    provider_id: str = field(init=False)
    _component_hashes: tuple[str, str, str] = field(init=False, repr=False)
    _configuration_hash: str = field(init=False, repr=False)
    _testing: bool = field(init=False, repr=False)
    _composition_policy: ResponseCompositionPolicy = field(init=False, repr=False)

    def __post_init__(self, _policy, _testing_token, _builder_token) -> None:
        import torch

        if not isinstance(_policy, ResponseCompositionPolicy):
            raise TypeError("Response PES requires a closed composition policy.")
        _require_graph_inspection_api()
        device = "cuda:0" if self.device == "cuda" else str(self.device)
        if device != "cpu" and not (
            device.startswith("cuda:") and device[5:].isdigit()
        ):
            raise ValueError("Response PES requires explicit cpu or cuda:N device.")
        if device.startswith("cuda:"):
            from .cuda_execution import prepare_cuda_execution

            prepare_cuda_execution()
            if (
                not torch.cuda.is_available()
                or int(device[5:]) >= torch.cuda.device_count()
            ):
                raise ValueError(f"Requested response device {device} is unavailable.")
        symbols = tuple(self.symbols)
        if not symbols or any(not isinstance(symbol, str) for symbol in symbols):
            raise ValueError("Response PES requires a nonempty ordered symbols tuple.")
        testing = _testing_token is _TEST_COMPONENTS_TOKEN
        if _testing_token is not None and not testing:
            raise ValueError("Invalid private response component injection token.")
        if not testing and _builder_token is not _BUILDER_TOKEN:
            raise ValueError(
                "Response PES construction requires its programmatic builder; "
                "arbitrary components cannot receive the experimental identity."
            )
        if not testing:
            _policy.validate_components(self.model, self.continuum, self.solvent_term)
        if (
            str(self.model.device) != device
            or str(self.model.dtype).removeprefix("torch.") != "float64"
        ):
            raise ValueError(
                "Response model must match the requested device and float64."
            )
        object.__setattr__(self, "device", device)
        object.__setattr__(self, "symbols", symbols)
        object.__setattr__(self, "_testing", testing)
        object.__setattr__(self, "_composition_policy", _policy)
        kind = "cpu" if device == "cpu" else "cuda"
        if testing:
            identity = f"unregistered-engineering-test-response-{kind}"
            object.__setattr__(self, "provider_id", identity)
            object.__setattr__(self, "profile_id", identity)
            object.__setattr__(self, "scalar_contract_id", identity)
        else:
            ids = _policy.ids(kind)
            for name, value in ids.items():
                object.__setattr__(self, name, value)
        object.__setattr__(self, "_component_hashes", self._current_component_hashes())
        object.__setattr__(
            self, "_configuration_hash", self._current_configuration_hash()
        )

    @classmethod
    def _for_testing(cls, **kwargs):
        return cls(**kwargs, _testing_token=_TEST_COMPONENTS_TOKEN)

    def _current_component_hashes(self) -> tuple[str, str, str]:
        return tuple(
            _component_hash(component)
            for component in (self.model, self.continuum, self.solvent_term)
        )

    def _current_configuration_hash(self) -> str:
        import torch

        solvation = Path(__file__).resolve().parents[1]
        return _sha(
            {
                "provider": self.provider_id,
                "composition_policy_id": self._composition_policy.policy_id,
                "scalar": self.scalar_contract_id,
                "profile": self.profile_id,
                "symbols": self.symbols,
                "solvent": self.solvent,
                "device": self.device,
                "dtype": "float64",
                "torch_default_dtype": str(torch.get_default_dtype()),
                "components": self._current_component_hashes(),
                "embedding": mace_polar_learned_source_embedding_matrix().tolist(),
                "active_source_columns": _ACTIVE_SOURCE_COLUMNS,
                "source_charge_atol_e": _SOURCE_CHARGE_ATOL_E,
                "maximum_raw_antisymmetry_eV_per_A2": _HESSIAN_PARITY_BUDGET_EV_A2,
                "second_order_continuum_limit_bytes": _SECOND_ORDER_CONTINUUM_LIMIT_BYTES,
                "cpu_model_runtime_base_bytes": _CPU_MODEL_RUNTIME_BASE_BYTES,
                "cpu_model_runtime_per_atom_bytes": _CPU_MODEL_RUNTIME_PER_ATOM_BYTES,
                "cuda_model_runtime_base_bytes": _CUDA_MODEL_RUNTIME_BASE_BYTES,
                "cuda_model_runtime_per_atom_bytes": _CUDA_MODEL_RUNTIME_PER_ATOM_BYTES,
                "source_files_sha256": source_files_sha256(
                    {
                        "response_core": Path(__file__),
                        "response_result": solvation / "derivatives" / "response.py",
                        "response_continuum": solvation
                        / "continuum"
                        / "ddpcm_response.py",
                        "response_operators": solvation
                        / "continuum"
                        / "ddpcm_response_operators.py",
                        "response_tensor_binding": solvation
                        / "continuum"
                        / "response_tensor_binding.py",
                        "response_topology": solvation
                        / "continuum"
                        / "response_topology.py",
                        "response_harmonics": solvation
                        / "continuum"
                        / "solid_harmonic_response.py",
                        "harmonic_primitives": solvation
                        / "continuum"
                        / "harmonic_torch_primitives.py",
                        "lebedev_tables": solvation / "surfaces" / "lebedev.py",
                        "energy_units": solvation / "api" / "units.py",
                        "source_embedding": solvation / "coupling" / "exact_gto.py",
                        "domain_guard": solvation.parent
                        / "function"
                        / "calculator"
                        / "extra_correction"
                        / "implicit"
                        / "route2_domain.py",
                        "cuda_execution": solvation
                        / "experimental"
                        / "cuda_execution.py",
                        **self._composition_policy.source_files(),
                    }
                ),
                "engineering_test_injection": self._testing,
            }
        )

    def configuration_sha256(self) -> str:
        if self._current_configuration_hash() != self._configuration_hash:
            raise RuntimeError(
                "Response PES component or implementation configuration changed."
            )
        return self._configuration_hash

    def _execution_context(self):
        if self.device == "cpu":
            return nullcontext()
        from .cuda_execution import cuda_model_execution

        return cuda_model_execution()

    def _preflight(self, derivative_order: int) -> dict:
        if self._testing:
            return {"engineering_test_injection": True}
        return _response_resource_preflight(
            self.continuum, self.device, len(self.symbols), derivative_order
        )

    def _positions(self, atoms, *, requires_grad: bool):
        import torch

        self.configuration_sha256()
        if tuple(atoms.get_chemical_symbols()) != self.symbols:
            raise ValueError("Response PES ordered symbols changed.")
        validate_route2_domain(atoms)
        return torch.tensor(
            atoms.get_positions(),
            dtype=torch.float64,
            device=self.device,
            requires_grad=requires_grad,
        )

    def _model_and_source(self, atoms, positions):
        import torch

        vacuum, learned = self.model.energy_source_torch(atoms, positions)
        if (
            not torch.is_tensor(learned)
            or learned.shape != (len(self.symbols), 4)
            or learned.dtype != positions.dtype
            or learned.device != positions.device
            or not bool(torch.isfinite(learned).all())
        ):
            raise RuntimeError(
                "Response source has invalid shape, dtype, device or values."
            )
        if abs(float(learned[:, 0].sum().detach())) > _SOURCE_CHARGE_ATOL_E:
            raise RuntimeError(
                "Response source violates the unchanged neutral charge contract."
            )
        if positions.requires_grad and not _graph_has_path(learned, positions):
            raise RuntimeError("MACE source is disconnected from the coordinate graph.")
        embedding = torch.tensor(
            mace_polar_learned_source_embedding_matrix(),
            dtype=positions.dtype,
            device=positions.device,
        )
        return vacuum, learned, learned @ embedding.T

    @staticmethod
    def _validate_scalar(name, value, positions) -> None:
        import torch

        if (
            not torch.is_tensor(value)
            or value.ndim != 0
            or value.dtype != positions.dtype
            or value.device != positions.device
            or not bool(torch.isfinite(value))
        ):
            raise RuntimeError(
                f"{name} must be a finite scalar on the coordinate device."
            )
        if positions.requires_grad and not _graph_has_path(value, positions):
            raise RuntimeError(f"{name} is disconnected from the coordinate graph.")

    def _energy_components(self, atoms, positions):
        vacuum, _, source = self._model_and_source(atoms, positions)
        continuum = self.continuum.energy_torch(positions, source)
        cds = self.solvent_term.energy_torch(positions)
        for name, value in (("vacuum", vacuum), ("ddpcm", continuum), ("cds", cds)):
            self._validate_scalar(name, value, positions)
        if positions.requires_grad:
            if not _graph_has_path(continuum, source):
                raise RuntimeError(
                    "ddPCM response energy is disconnected from its source."
                )
            if not _graph_has_path(continuum, positions, stop_at=(source,)):
                raise RuntimeError("ddPCM response lacks its explicit coordinate path.")
        terms = {"vacuum": vacuum, "ddpcm": continuum, "cds": cds}
        return sum(terms.values()), terms

    def get_potential_energy(self, atoms) -> float:
        with self._execution_context():
            self._preflight(0)
            positions = self._positions(atoms, requires_grad=False)
            total, _ = self._energy_components(atoms, positions)
            result = float(total.detach())
            self.configuration_sha256()
            return result

    def get_solvation_energy(self, atoms) -> float:
        with self._execution_context():
            self._preflight(0)
            positions = self._positions(atoms, requires_grad=False)
            _, terms = self._energy_components(atoms, positions)
            result = float((terms["ddpcm"] + terms["cds"]).detach())
            self.configuration_sha256()
            return result

    def get_forces(self, atoms) -> np.ndarray:
        import torch

        with self._execution_context():
            self._preflight(1)
            positions = self._positions(atoms, requires_grad=True)
            total, _ = self._energy_components(atoms, positions)
            gradient = torch.autograd.grad(total, positions)[0]
            if not bool(torch.isfinite(gradient).all()):
                raise RuntimeError("Response total-PES gradient is not finite.")
            self.configuration_sha256()
            return -gradient.detach().cpu().numpy().copy()

    @staticmethod
    def _gradient(value, positions, *, create_graph=False, retain_graph=False):
        import torch

        if not value.requires_grad:
            return torch.zeros_like(positions)
        result = torch.autograd.grad(
            value,
            positions,
            create_graph=create_graph,
            retain_graph=retain_graph,
            allow_unused=True,
        )[0]
        return torch.zeros_like(positions) if result is None else result

    def hessian_vector_product(self, atoms, direction) -> np.ndarray:
        import torch

        vector = np.asarray(direction, dtype=float)
        if vector.shape != (len(self.symbols), 3) or not np.all(np.isfinite(vector)):
            raise ValueError("HVP direction must be finite with shape (N,3).")
        with self._execution_context():
            self._preflight(2)
            positions = self._positions(atoms, requires_grad=True)
            total, _ = self._energy_components(atoms, positions)
            gradient = torch.autograd.grad(total, positions, create_graph=True)[0]
            probe = torch.tensor(vector, dtype=positions.dtype, device=positions.device)
            product = self._gradient((gradient * probe).sum(), positions)
            if not bool(torch.isfinite(product).all()):
                raise RuntimeError(
                    "Response total-PES Hessian-vector product is not finite."
                )
            self.configuration_sha256()
            return product.detach().cpu().numpy().copy()

    def evaluate_hessian(self, atoms) -> ResponseHessianEvaluation:
        import torch

        with self._execution_context():
            resources = self._preflight(2)
            positions = self._positions(atoms, requires_grad=True)
            vacuum, learned, source = self._model_and_source(atoms, positions)
            if self._testing:
                cds_state = None
                cds = self.solvent_term.energy_torch(positions)
            else:
                cds_state = self.solvent_term.evaluate_torch(positions)
                cds = cds_state.energy_eV
            self._validate_scalar("vacuum", vacuum, positions)
            self._validate_scalar("cds", cds, positions)
            state = self.continuum._linearize(
                positions.detach(), source.detach(), derivative_order=2
            )
            g_r, g_s = state.gradient_partial()
            h_theta = state.hessian_partial()
            continuum_energy = state.energy
            if (
                not torch.is_tensor(continuum_energy)
                or continuum_energy.ndim != 0
                or continuum_energy.dtype != positions.dtype
                or continuum_energy.device != positions.device
                or not bool(torch.isfinite(continuum_energy))
            ):
                raise RuntimeError(
                    "Structured response returned an invalid continuum energy."
                )
            for name, tensor in (
                ("continuum partial gradient R", g_r),
                ("continuum partial gradient source", g_s),
                ("continuum partial Hessian", h_theta),
            ):
                if (
                    not torch.is_tensor(tensor)
                    or tensor.dtype != positions.dtype
                    or tensor.device != positions.device
                    or not bool(torch.isfinite(tensor).all())
                ):
                    raise RuntimeError(f"{name} is not finite.")
            n = len(self.symbols)
            if (
                g_r.shape != (n, 3)
                or g_s.shape != (n, 8)
                or h_theta.shape != (7 * n, 7 * n)
            ):
                raise RuntimeError(
                    "Structured response returned inconsistent derivative shapes."
                )
            if bool(torch.count_nonzero(g_s[:, (1, 5, 6, 7)])):
                raise RuntimeError(
                    "Structured response returned nonzero inactive-source derivatives."
                )

            learned_entries = learned.reshape(-1)
            source_rows = [
                self._gradient(value, positions, retain_graph=True).reshape(-1)
                for value in learned_entries
            ]
            source_jacobian = torch.stack(source_rows)
            g_active = g_s[:, _ACTIVE_SOURCE_COLUMNS]
            witnesses = self._source_witnesses(
                positions, learned, source, source_jacobian, g_active, h_theta
            )
            auxiliary = vacuum + cds + torch.sum(g_active.detach() * learned)
            auxiliary_gradient = torch.autograd.grad(
                auxiliary, positions, create_graph=True, retain_graph=True
            )[0]
            gradient = auxiliary_gradient + g_r
            auxiliary_entries = auxiliary_gradient.reshape(-1)
            auxiliary_hessian = torch.stack(
                [
                    self._gradient(
                        value,
                        positions,
                        retain_graph=index + 1 < len(auxiliary_entries),
                    ).reshape(-1)
                    for index, value in enumerate(auxiliary_entries)
                ]
            )
            identity = torch.eye(3 * n, dtype=positions.dtype, device=positions.device)
            outer_chain = torch.cat((identity, source_jacobian), dim=0)
            hessian = auxiliary_hessian + outer_chain.mT @ h_theta @ outer_chain
            if not bool(torch.isfinite(hessian).all()):
                raise RuntimeError("Response total-PES Hessian is not finite.")
            raw_asymmetry = float((hessian - hessian.mT).abs().max().detach())
            if raw_asymmetry > _HESSIAN_PARITY_BUDGET_EV_A2:
                raise RuntimeError(
                    "Raw response Hessian antisymmetry exceeds the parity budget."
                )
            total = vacuum + continuum_energy + cds
            diagnostics = {
                "coordinate_finite_differences": False,
                "hessian_symmetrization": "none",
                "source_coordinate_path": True,
                "continuum_direct_coordinate_path": True,
                "source_jacobian_frobenius_norm": float(
                    torch.linalg.vector_norm(source_jacobian).detach()
                ),
                "source_active_sha256": _tensor_sha256(learned),
                "source_embedded_sha256": _tensor_sha256(source),
                "mace_output_sha256": _model_output_sha256(vacuum, learned),
                "source_active_columns": _ACTIVE_SOURCE_COLUMNS,
                "source_witnesses": witnesses,
                "dtype": "float64",
                "torch_default_dtype": str(torch.get_default_dtype()),
                "device": self.device,
                "component_configuration_sha256": dict(
                    zip(("model", "ddpcm_response", "cds"), self._component_hashes)
                ),
                "scientific_release_admitted": False,
                "second_order_resources": resources,
                "ddpcm_response": dict(state.diagnostics()),
            }
            if not self._testing:
                diagnostics["cds"] = asdict(cds_state.diagnostics)
                diagnostics["model_topology"] = self.model.topology_diagnostics(atoms)
                diagnostics["model_provenance"] = self.model.metadata()
            self.configuration_sha256()
            return ResponseHessianEvaluation(
                hessian_eV_per_A2=hessian.detach().cpu().numpy(),
                forces_eV_per_A=-gradient.detach().cpu().numpy(),
                energy_eV=float(total.detach()),
                geometry_sha256=geometry_sha256(atoms),
                configuration_sha256=self._configuration_hash,
                scalar_contract_id=self.scalar_contract_id,
                provider_id=self.provider_id,
                profile_id=self.profile_id,
                device=self.device,
                component_energies_eV={
                    "vacuum": float(vacuum.detach()),
                    "ddpcm": float(continuum_energy.detach()),
                    "cds": float(cds.detach()),
                },
                diagnostics=diagnostics,
            )

    def _source_witnesses(self, positions, learned, source, jacobian, weight, htheta):
        """Observe the live source chain before releasing its graph.

        These witnesses do not enter the energy or Hessian. Nonzero floors are
        qualification-panel requirements, not universal physical assumptions.
        """
        import torch

        probe = torch.linspace(
            -1.0, 1.0, positions.numel(), dtype=positions.dtype, device=positions.device
        ).reshape_as(positions)
        probe = probe / torch.linalg.vector_norm(probe)
        weighted = torch.sum(weight.detach() * learned)
        weighted_gradient = self._gradient(
            weighted, positions, create_graph=True, retain_graph=True
        )
        curvature = self._gradient(
            torch.sum(weighted_gradient * probe), positions, retain_graph=True
        )
        norms = {
            "source_jacobian_frobenius_norm": torch.linalg.vector_norm(jacobian),
            "source_jvp_norm": torch.linalg.vector_norm(jacobian @ probe.reshape(-1)),
            "source_vjp_norm": torch.linalg.vector_norm(
                jacobian.mT @ weight.reshape(-1)
            ),
            "mixed_R_source_max_abs": htheta[: positions.numel(), positions.numel() :]
            .abs()
            .max(),
            "weighted_source_curvature_hvp_norm": torch.linalg.vector_norm(curvature),
        }
        if any(not bool(torch.isfinite(value)) for value in norms.values()):
            raise RuntimeError("Response source-chain witness is not finite.")
        inactive_count = int(torch.count_nonzero(source[:, (1, 5, 6, 7)]))
        charge = float(learned[:, 0].sum().detach())
        return {
            "schema": "macepolar-response-source-witness-v1",
            "active_columns": _ACTIVE_SOURCE_COLUMNS,
            "inactive_columns": (1, 5, 6, 7),
            "inactive_nonzero_count": inactive_count,
            "inactive_exact_zero": inactive_count == 0,
            "charge_sum_e": charge,
            "charge_atol_e": _SOURCE_CHARGE_ATOL_E,
            "neutral_charge_pass": abs(charge) <= _SOURCE_CHARGE_ATOL_E,
            "probe_policy": "normalized-linspace-minus1-plus1-v1",
            "probe_sha256": _tensor_sha256(probe),
            **{key: float(value.detach()) for key, value in norms.items()},
        }


__all__ = [
    "ResponseCompositionPolicy",
    "ResponsePESCore",
    "_BUILDER_TOKEN",
    "_TEST_COMPONENTS_TOKEN",
]
