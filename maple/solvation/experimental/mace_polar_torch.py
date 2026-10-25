"""One coordinate-connected Torch scalar for the pure frozen solution PES.

The legacy evaluator remains independent. This backend composes the same
zero-field MACE-POLAR energy/source, ddPCM, and legacy SMD-CDS equations;
forces and second derivatives are generated from their summed scalar. No
coordinate finite differences or compiled-solvent fallback are used here.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import InitVar, asdict, dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

from maple.function.calculator.extra_correction.implicit.route2_domain import (
    validate_route2_domain,
)
from maple.solvation.coupling.exact_gto import (
    mace_polar_learned_source_embedding_matrix,
)
from maple.solvation.coupling.state_equation import geometry_sha256
from maple.solvation.coupling.operator import source_files_sha256
from maple.solvation.derivatives.analytic import (
    AnalyticForceEvaluation,
    AnalyticHessianEvaluation,
    AnalyticHVPEvaluation,
)
from maple.solvation.models.base import model_input_sha256

_TEST_COMPONENTS_TOKEN = object()
_REGISTERED_BUILDER_TOKEN = object()
_SOURCE_CHARGE_ATOL_E = 1.0e-8  # Existing frozen-source charge contract.
_HESSIAN_PARITY_BUDGET_EV_A2 = 1.0e-4
_PROVIDER_ID = "maple.pure-macepolar-frozen-ddpcm-legacy-cds.torch-v3"
_SECOND_ORDER_CONTINUUM_LIMIT_BYTES = 4_000_000_000
_MODEL_RUNTIME_RESERVE_BYTES = 2 * 1024**3
_CUDA_MODEL_RUNTIME_RESERVE_BYTES = 512 * 1024**2
_CONTRACT_EVIDENCE_ID = "route2-pure-macepolar-torch-analytic-v3-contract-tests"


class _HVPTensors(NamedTuple):
    """Transient graph data, consumed by one public HVP return and never cached."""

    total: Any
    terms: dict[str, Any]
    central_gradient: Any
    product: Any
    probe: Any
    input_sha256: str
    direction_snapshot: np.ndarray


def _require_graph_inspection_api() -> None:
    import torch

    if not callable(getattr(torch.autograd.graph, "get_gradient_edge", None)):
        raise RuntimeError(
            "Torch v3 requires PyTorch >=2.2 with autograd.graph.get_gradient_edge; "
            "no legacy or finite-difference fallback is available."
        )


def _graph_has_path(output, target, *, stop_at=()) -> bool:
    """Check graph reachability without extra backward passes or detached probes.

    A tensor merely having ``requires_grad`` is insufficient: it may be a new
    detached leaf. Stopping at the source also distinguishes the continuum's
    explicit geometry path from the geometry path through model multipoles.
    """
    import torch

    if not output.requires_grad or not target.requires_grad:
        return False
    target_node = torch.autograd.graph.get_gradient_edge(target).node
    blocked = {torch.autograd.graph.get_gradient_edge(value).node for value in stop_at}
    pending = [torch.autograd.graph.get_gradient_edge(output).node]
    seen = set()
    while pending:
        node = pending.pop()
        if node is target_node:
            return True
        if node in seen or node in blocked:
            continue
        seen.add(node)
        pending.extend(child for child, _ in node.next_functions if child is not None)
    return False


def _available_host_memory_bytes() -> int:
    """Use Linux's reclaim-aware MemAvailable, not misleading free pages."""
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError) as error:
        raise RuntimeError(
            "Cannot establish the Torch v3 host-memory preflight."
        ) from error
    raise RuntimeError("Torch v3 memory preflight requires Linux MemAvailable.")


def _second_order_resource_preflight(continuum, device: str) -> dict[str, object]:
    """Separate second-backward budget from the unchanged 1 GB forward cap.

    The continuum estimate is an engineering envelope, not a formal allocator
    upper bound. The reserve covers the model, Python/autograd and runtime;
    an allocation failure still propagates and never triggers another backend.
    """
    estimate = continuum.preflight_resources(
        derivative_order=2, limit_bytes=_SECOND_ORDER_CONTINUUM_LIMIT_BYTES
    )
    host_available = _available_host_memory_bytes()
    reserve = (
        _MODEL_RUNTIME_RESERVE_BYTES
        if device == "cpu"
        else _CUDA_MODEL_RUNTIME_RESERVE_BYTES
    )
    required = estimate.conservative_peak_bytes + reserve
    if device == "cpu":
        available = host_available
    else:
        import torch

        free, _ = torch.cuda.mem_get_info(device)
        reusable = torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(
            device
        )
        available = int(free + reusable)
        if host_available < _MODEL_RUNTIME_RESERVE_BYTES:
            raise MemoryError(
                "Insufficient host memory for the Torch second-order runtime."
            )
    if required > available:
        raise MemoryError(
            f"Torch second-order preflight requires an estimated {required} bytes "
            f"including runtime reserve, but {device} has {available} bytes available."
        )
    return {
        "policy": "torch-v3-second-order-memory-envelope-v1",
        "continuum": asdict(estimate),
        "model_runtime_reserve_bytes": reserve,
        "host_runtime_reserve_bytes": _MODEL_RUNTIME_RESERVE_BYTES,
        "estimated_total_bytes": required,
        "device_available_bytes": available,
        "host_available_bytes": host_available,
        "claim_boundary": "engineering estimate, not a guaranteed allocator upper bound",
    }


def _sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _component_hash(component: object) -> str:
    value = component.configuration_sha256()
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ValueError("Torch PES component configuration must be a SHA256 digest.")
    return value


@dataclass(frozen=True, slots=True)
class MACEPolarTorchPES:
    """Explicit experimental total PES with a fixed component configuration."""

    model: object
    continuum: object
    solvent_term: object
    symbols: tuple[str, ...]
    device: str = "cpu"
    solvent: str = "water"
    _testing_token: InitVar[object | None] = None
    _builder_token: InitVar[object | None] = None
    requested_profile: InitVar[str | None] = None
    _fp32: bool = field(init=False, repr=False)
    scalar_contract_id: str = field(init=False)
    profile_id: str = field(init=False)
    provider_id: str = field(init=False, default=_PROVIDER_ID)
    _component_hashes: tuple[str, str, str] = field(init=False, repr=False)
    _configuration_hash: str = field(init=False, repr=False)
    _testing: bool = field(init=False, repr=False)

    def __post_init__(self, _testing_token, _builder_token, requested_profile) -> None:
        import torch

        _require_graph_inspection_api()
        device = "cuda:0" if self.device == "cuda" else str(self.device)
        if device != "cpu" and not (
            device.startswith("cuda:") and device[5:].isdigit()
        ):
            raise ValueError("Torch PES requires explicit cpu or cuda:N device.")
        if device.startswith("cuda:"):
            from .cuda_execution import prepare_cuda_execution

            prepare_cuda_execution()
            if (
                not torch.cuda.is_available()
                or int(device[5:]) >= torch.cuda.device_count()
            ):
                raise ValueError(f"Requested Torch PES device {device} is unavailable.")
        kind = "cpu" if device == "cpu" else "cuda"
        v3 = f"pure-macepolar-frozen-point-l1-ddpcm-smd-torch-{kind}-v3"
        v4 = f"pure-macepolar-frozen-point-l1-ddpcm-smd-torch-{kind}-v4"
        selected = v3 if requested_profile is None else requested_profile
        if selected not in {v3, v4}:
            raise ValueError(
                "Torch PES profile must match the requested device/version."
            )
        fp32 = selected == v4
        object.__setattr__(self, "_fp32", fp32)
        symbols = tuple(self.symbols)
        if not symbols or any(not isinstance(symbol, str) for symbol in symbols):
            raise ValueError("Torch PES requires a nonempty ordered symbols tuple.")
        testing = _testing_token is _TEST_COMPONENTS_TOKEN
        if _testing_token is not None and not testing:
            raise ValueError("Invalid private Torch PES component injection token.")
        if not testing and _builder_token is not _REGISTERED_BUILDER_TOKEN:
            raise ValueError(
                "Registered Torch PES construction requires "
                "build_smd_mace_polar_torch_pes; arbitrary components cannot "
                "receive a registered scientific identity."
            )
        if not testing:
            from maple.solvation.continuum.torch_ddpcm import TorchDDPCM
            from maple.solvation.models.mace_polar_torch import (
                MACEPolarTorchGraphAdapter,
            )
            from maple.solvation.nonpolar.legacy_smd_cds import TorchLegacySMDCDS

            from maple.solvation.models.mace_polar_fp32 import MACEPolarFP32Adapter

            if not (
                isinstance(
                    self.model,
                    MACEPolarFP32Adapter if fp32 else MACEPolarTorchGraphAdapter,
                )
                and isinstance(self.continuum, TorchDDPCM)
                and isinstance(self.solvent_term, TorchLegacySMDCDS)
            ):
                raise TypeError(
                    "Registered Torch PES requires the exact Torch components."
                )
        if str(self.model.device) != device or str(self.model.dtype).removeprefix(
            "torch."
        ) != ("float32" if fp32 else "float64"):
            raise ValueError(
                "Torch PES model must match the requested device and profile dtype."
            )
        object.__setattr__(self, "device", device)
        object.__setattr__(self, "symbols", symbols)
        object.__setattr__(self, "_testing", testing)
        if not testing:
            expected_dtype = torch.float32 if fp32 else torch.float64
            if (
                self.continuum.dtype is not expected_dtype
                or self.solvent_term.dtype is not expected_dtype
            ):
                raise ValueError("Torch PES components must share the profile dtype.")
            object.__setattr__(
                self,
                "provider_id",
                (
                    _PROVIDER_ID
                    if not fp32
                    else "maple.pure-macepolar-frozen-ddpcm-legacy-cds.torch-v4"
                ),
            )
        if testing:
            object.__setattr__(
                self, "provider_id", "unregistered-engineering-test-torch-pes"
            )
        object.__setattr__(
            self,
            "profile_id",
            (f"unregistered-engineering-test-torch-{kind}" if testing else selected),
        )
        object.__setattr__(
            self,
            "scalar_contract_id",
            (
                f"unregistered-engineering-test-torch-{kind}"
                if testing
                else "route2-experimental-pure-macepolar-frozen-point-l1-ddpcm-smd-"
                f"torch-{kind}-v{4 if fp32 else 3}"
            ),
        )
        object.__setattr__(self, "_component_hashes", self._current_component_hashes())
        object.__setattr__(
            self, "_configuration_hash", self._current_configuration_hash()
        )

    @classmethod
    def _for_testing(cls, **kwargs):
        """Private engineering injection; public construction binds real components."""
        return cls(**kwargs, _testing_token=_TEST_COMPONENTS_TOKEN)

    def _current_component_hashes(self) -> tuple[str, str, str]:
        return tuple(
            _component_hash(component)
            for component in (self.model, self.continuum, self.solvent_term)
        )

    def _current_configuration_hash(self) -> str:
        root = Path(__file__).resolve().parents[1]
        return _sha(
            {
                "provider": self.provider_id,
                "scalar": self.scalar_contract_id,
                "profile": self.profile_id,
                "symbols": self.symbols,
                "solvent": self.solvent,
                "device": self.device,
                "dtype": "float32" if self._fp32 else "float64",
                "components": self._current_component_hashes(),
                "embedding": mace_polar_learned_source_embedding_matrix().tolist(),
                "source_charge_policy": (
                    "fp32-roundoff-scaled" if self._fp32 else "fixed-1e-8"
                ),
                "source_charge_atol_e": _SOURCE_CHARGE_ATOL_E,
                "base_absolute_antisymmetry_budget_eV_per_A2": _HESSIAN_PARITY_BUDGET_EV_A2,
                "hessian_antisymmetry_policy": (
                    "max(1e-4,100eps*maxabsH)" if self._fp32 else "fixed-1e-4"
                ),
                "second_order_continuum_limit_bytes": _SECOND_ORDER_CONTINUUM_LIMIT_BYTES,
                "model_runtime_reserve_bytes": _MODEL_RUNTIME_RESERVE_BYTES,
                "cuda_model_runtime_reserve_bytes": _CUDA_MODEL_RUNTIME_RESERVE_BYTES,
                "source_files_sha256": source_files_sha256(
                    {
                        "total_pes": Path(__file__),
                        "cuda_execution": root / "experimental" / "cuda_execution.py",
                        "analytic_result": root / "derivatives" / "analytic.py",
                        "source_embedding": root / "coupling" / "exact_gto.py",
                        "precision_policy": root / "numerical_precision.py",
                        "smd_solvent_data": root.parent
                        / "function"
                        / "smd_solvent_data.py",
                    }
                ),
                "engineering_test_injection": self._testing,
            }
        )

    def configuration_sha256(self) -> str:
        if self._current_configuration_hash() != self._configuration_hash:
            raise RuntimeError(
                "Torch PES component or implementation configuration changed."
            )
        return self._configuration_hash

    def _execution_context(self):
        if self.device == "cpu":
            return nullcontext()
        from .cuda_execution import cuda_model_execution

        # The context intentionally spans both reverse passes, not only forward.
        return cuda_model_execution()

    def _positions(self, atoms, *, requires_grad: bool):
        import torch

        self.configuration_sha256()
        if tuple(atoms.get_chemical_symbols()) != self.symbols:
            raise ValueError("Torch PES ordered symbols changed.")
        validate_route2_domain(atoms, allow_charged=self._fp32)
        return torch.tensor(
            atoms.get_positions(),
            dtype=torch.float32 if self._fp32 else torch.float64,
            device=self.device,
            requires_grad=requires_grad,
        )

    def _energy_components(self, atoms, positions):
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
                "Torch source has invalid shape, dtype, device or values."
            )
        from maple.solvation.numerical_precision import source_charge_tolerance

        charge_tolerance = (
            source_charge_tolerance(learned[:, 0])
            if self._fp32
            else _SOURCE_CHARGE_ATOL_E
        )
        from maple.solvation.models.base import model_charge_and_multiplicity

        requested_charge, _ = model_charge_and_multiplicity(atoms)
        source_charge = float(learned[:, 0].sum().detach())
        if abs(source_charge - requested_charge) > charge_tolerance:
            raise RuntimeError(
                "Torch source violates the molecular charge contract "
                f"(source={source_charge:.8g} e, requested={requested_charge} e, "
                f"tolerance={charge_tolerance:.3e} e)."
            )
        if positions.requires_grad and not _graph_has_path(learned, positions):
            raise RuntimeError("MACE source is disconnected from the coordinate graph.")
        embedding = torch.tensor(
            mace_polar_learned_source_embedding_matrix(),
            dtype=positions.dtype,
            device=positions.device,
        )
        source = learned @ embedding.T
        energies = {
            "vacuum": vacuum,
            "ddpcm": self.continuum.energy_torch(positions, source),
            "cds": self.solvent_term.energy_torch(positions),
        }
        for name, energy in energies.items():
            if (
                not torch.is_tensor(energy)
                or energy.ndim != 0
                or energy.device != positions.device
                or energy.dtype != positions.dtype
                or not bool(torch.isfinite(energy))
            ):
                raise RuntimeError(
                    f"{name} must return a finite scalar on the coordinate device."
                )
            if positions.requires_grad and not _graph_has_path(energy, positions):
                raise RuntimeError(
                    f"{name} energy is disconnected from the differentiable graph."
                )
        if positions.requires_grad:
            if not _graph_has_path(energies["ddpcm"], source):
                raise RuntimeError(
                    "ddPCM energy is disconnected from the source graph."
                )
            if not _graph_has_path(energies["ddpcm"], positions, stop_at=(source,)):
                raise RuntimeError("ddPCM lacks its explicit coordinate graph path.")
        return sum(energies.values()), energies, source

    def get_potential_energy(self, atoms) -> float:
        with self._execution_context():
            positions = self._positions(atoms, requires_grad=False)
            total, _, _ = self._energy_components(atoms, positions)
            result = float(total.detach())
            self.configuration_sha256()
            return result

    def get_solvation_energy(self, atoms) -> float:
        with self._execution_context():
            positions = self._positions(atoms, requires_grad=False)
            _, terms, _ = self._energy_components(atoms, positions)
            result = float((terms["ddpcm"] + terms["cds"]).detach())
            self.configuration_sha256()
            return result

    def evaluate_forces(self, atoms) -> AnalyticForceEvaluation:
        """Return energy and analytic forces from one transient forward graph."""
        import torch

        with self._execution_context():
            input_sha256 = model_input_sha256(atoms)
            positions = self._positions(atoms, requires_grad=True)
            total, terms, _ = self._energy_components(atoms, positions)
            gradient = torch.autograd.grad(total, positions)[0]
            return self._force_evaluation_from_graph(
                atoms, total, terms, gradient, input_sha256
            )

    def _force_evaluation_from_graph(
        self, atoms, total, terms, gradient, input_sha256: str
    ) -> AnalyticForceEvaluation:
        """Detach a first derivative already computed by a force or HVP request."""
        import torch

        if not bool(torch.isfinite(gradient).all()):
            raise RuntimeError("Torch total-PES coordinate gradient is not finite.")
        self.configuration_sha256()
        if model_input_sha256(atoms) != input_sha256:
            raise RuntimeError(
                "Torch PES model inputs changed during force evaluation."
            )
        return AnalyticForceEvaluation(
            energy_eV=float(total.detach()),
            forces_eV_per_A=-gradient.detach().cpu().numpy(),
            geometry_sha256=geometry_sha256(atoms),
            model_input_sha256=input_sha256,
            configuration_sha256=self._configuration_hash,
            scalar_contract_id=self.scalar_contract_id,
            device=self.device,
            component_energies_eV={
                name: float(value.detach()) for name, value in terms.items()
            },
            provider_id=self.provider_id,
            profile_id=self.profile_id,
            verification_evidence_id=(
                "route2-pure-macepolar-torch-fp32-v4-engineering-tests"
                if self._fp32
                else _CONTRACT_EVIDENCE_ID
            ),
            computation_dtype="float32" if self._fp32 else "float64",
        )

    def get_forces(self, atoms) -> np.ndarray:
        evaluation = self.evaluate_forces(atoms)
        # Result records normalize immutable storage to FP64. Preserve the
        # standalone compatibility API's original computation-dtype array.
        dtype = np.float32 if self._fp32 else np.float64
        return np.array(evaluation.forces_eV_per_A, dtype=dtype, copy=True)

    @staticmethod
    def _gradient_derivative(value, positions, *, retain_graph: bool):
        """Exact zero for an analytically constant gradient, never an FD fallback."""
        import torch

        if not value.requires_grad:
            return torch.zeros_like(positions)
        derivative = torch.autograd.grad(
            value, positions, retain_graph=retain_graph, allow_unused=True
        )[0]
        return torch.zeros_like(positions) if derivative is None else derivative

    def evaluate_hvp(self, atoms, direction) -> AnalyticHVPEvaluation:
        """Evaluate HVP and its central energy/forces from one transient graph."""
        tensors = self._evaluate_hvp_tensors(
            atoms, direction, include_central_force=True
        )
        evaluation = AnalyticHVPEvaluation(
            force_evaluation=self._force_evaluation_from_graph(
                atoms,
                tensors.total,
                tensors.terms,
                tensors.central_gradient,
                tensors.input_sha256,
            ),
            direction=tensors.probe.detach().cpu().numpy(),
            hvp_eV_per_A2=tensors.product.detach().cpu().numpy(),
        )
        self._validate_hvp_request(atoms, direction, tensors)
        return evaluation

    def _evaluate_hvp_tensors(self, atoms, direction, *, include_central_force: bool):
        """Share the forward, not two numerically distinct reverse-mode paths.

        Legacy forces use create_graph=False; the gradient for a direct HVP
        needs create_graph=True. Reusing the latter for forces changes FP32
        rounding. A composed request therefore retains the original force
        reverse pass on this same forward graph. A standalone HVP omits it.
        The returned tensors are transient and are never retained on the PES.
        """
        import torch

        vector = np.array(direction, dtype=np.float64, copy=True, order="C")
        if vector.shape != (len(self.symbols), 3) or not np.all(np.isfinite(vector)):
            raise ValueError("HVP direction must be finite with shape (N,3).")
        with self._execution_context():
            if not self._testing:
                _second_order_resource_preflight(self.continuum, self.device)
            input_sha256 = model_input_sha256(atoms)
            positions = self._positions(atoms, requires_grad=True)
            total, terms, _ = self._energy_components(atoms, positions)
            central_gradient = (
                torch.autograd.grad(
                    total, positions, create_graph=False, retain_graph=True
                )[0]
                if include_central_force
                else None
            )
            gradient = torch.autograd.grad(total, positions, create_graph=True)[0]
            probe = torch.tensor(vector, dtype=positions.dtype, device=positions.device)
            product = self._gradient_derivative(
                (gradient * probe).sum(), positions, retain_graph=False
            )
            if not bool(torch.isfinite(product).all()):
                raise RuntimeError(
                    "Torch total-PES Hessian-vector product is not finite."
                )
            if central_gradient is not None and not bool(
                torch.isfinite(central_gradient).all()
            ):
                raise RuntimeError("Torch total-PES coordinate gradient is not finite.")
            return _HVPTensors(
                total, terms, central_gradient, product, probe, input_sha256, vector
            )

    def _validate_hvp_request(self, atoms, direction, tensors: _HVPTensors) -> None:
        """Recheck the complete request after public result materialization."""
        self.configuration_sha256()
        if model_input_sha256(atoms) != tensors.input_sha256:
            raise RuntimeError("Torch PES model inputs changed during HVP evaluation.")
        if not np.array_equal(
            np.asarray(direction, dtype=np.float64), tensors.direction_snapshot
        ):
            raise RuntimeError("HVP direction changed during evaluation.")

    def hessian_vector_product(self, atoms, direction) -> np.ndarray:
        tensors = self._evaluate_hvp_tensors(
            atoms, direction, include_central_force=False
        )
        result = tensors.product.detach().cpu().numpy().copy()
        self._validate_hvp_request(atoms, direction, tensors)
        return result

    def evaluate_hessian(self, atoms) -> AnalyticHessianEvaluation:
        import torch

        with self._execution_context():
            resources = (
                {"engineering_test_injection": True}
                if self._testing
                else _second_order_resource_preflight(self.continuum, self.device)
            )
            positions = self._positions(atoms, requires_grad=True)
            total, terms, source = self._energy_components(atoms, positions)
            gradient = torch.autograd.grad(total, positions, create_graph=True)[0]
            entries = gradient.reshape(-1)
            rows = [
                self._gradient_derivative(
                    value, positions, retain_graph=index + 1 < len(entries)
                ).reshape(-1)
                for index, value in enumerate(entries)
            ]
            hessian = torch.stack(rows)
            if not bool(torch.isfinite(hessian).all()):
                raise RuntimeError("Torch total-PES Hessian is not finite.")
            raw_asymmetry = float((hessian - hessian.T).abs().max().detach())
            from maple.solvation.numerical_precision import (
                hessian_antisymmetry_tolerance,
            )

            symmetry_budget = hessian_antisymmetry_tolerance(hessian)
            if raw_asymmetry > symmetry_budget:
                raise RuntimeError(
                    f"Raw Torch Hessian antisymmetry {raw_asymmetry:.3e} exceeds "
                    f"the dtype-aware parity budget {symmetry_budget:.3e}."
                )
            diagnostics = {
                "coordinate_finite_differences": False,
                "raw_hessian_antisymmetry_eV_per_A2": raw_asymmetry,
                "hessian_antisymmetry_budget_eV_per_A2": symmetry_budget,
                "component_configuration_sha256": dict(
                    zip(("model", "ddpcm", "cds"), self._component_hashes)
                ),
                "model_device": self.device,
                "continuum_device": self.device,
                "cds_device": self.device,
                "scientific_release_admitted": False,
                "second_order_resources": resources,
            }
            if not self._testing:
                # A detached *second* observation does not supply any derivative.
                # It avoids retaining graphs/mutable caches inside scalar providers.
                diagnostics["ddpcm"] = self.continuum.diagnostics(
                    positions.detach(), source.detach()
                )
                diagnostics["cds"] = asdict(
                    self.solvent_term.evaluate_torch(positions.detach()).diagnostics
                )
                diagnostics["model_topology"] = self.model.topology_diagnostics(atoms)
                diagnostics["model_provenance"] = self.model.metadata()
            self.configuration_sha256()
            return AnalyticHessianEvaluation(
                computation_dtype="float32" if self._fp32 else "float64",
                hessian_eV_per_A2=hessian.detach().cpu().numpy(),
                forces_eV_per_A=-gradient.detach().cpu().numpy(),
                energy_eV=float(total.detach()),
                geometry_sha256=geometry_sha256(atoms),
                configuration_sha256=self._configuration_hash,
                scalar_contract_id=self.scalar_contract_id,
                device=self.device,
                component_energies_eV={
                    name: float(value.detach()) for name, value in terms.items()
                },
                diagnostics=diagnostics,
                provider_id=self.provider_id,
                profile_id=self.profile_id,
                verification_evidence_id=(
                    "route2-pure-macepolar-torch-fp32-v4-engineering-tests"
                    if self._fp32
                    else _CONTRACT_EVIDENCE_ID
                ),
            )


def build_smd_mace_polar_torch_pes(
    symbols,
    *,
    solvent: str,
    device: str = "cpu",
    checkpoint_path=None,
    profile: str | None = None,
) -> MACEPolarTorchPES:
    """Build default FP32/5/194 v4, or an explicitly selected historical v3."""
    import torch

    _require_graph_inspection_api()
    from maple.function.calculator.extra_correction.implicit.smd_cds import (
        smd_coulomb_radii,
    )
    from maple.function.route2_solvents import route2_solvent_spec
    from maple.solvation.continuum.torch_ddpcm import TorchDDPCM
    from maple.solvation.models.mace_polar import build_official_mace_polar_1_m_adapter
    from maple.solvation.models.mace_polar_torch import MACEPolarTorchGraphAdapter
    from maple.solvation.nonpolar.legacy_smd_cds import TorchLegacySMDCDS

    specification = route2_solvent_spec(solvent)
    normalized_device = "cuda:0" if device == "cuda" else str(device)
    if normalized_device.startswith("cuda:"):
        from .cuda_execution import prepare_cuda_execution

        prepare_cuda_execution()
    kind = "cpu" if normalized_device == "cpu" else "cuda"
    v3 = f"pure-macepolar-frozen-point-l1-ddpcm-smd-torch-{kind}-v3"
    v4 = f"pure-macepolar-frozen-point-l1-ddpcm-smd-torch-{kind}-v4"
    selected = v4 if profile is None else profile
    if selected not in {v3, v4}:
        raise ValueError("Torch PES profile must match the requested device/version.")
    fp32 = selected == v4
    if not fp32 and specification.name not in {"water", "hexane"}:
        raise ValueError("Historical Torch v3 supports water and hexane only.")
    dtype = torch.float32 if fp32 else torch.float64
    symbols = tuple(symbols)
    # Reject the dense resource/domain limit before loading the large checkpoint.
    continuum = TorchDDPCM(
        symbols,
        smd_coulomb_radii(symbols, solvent=specification.name),
        dielectric=specification.descriptors.dielectric,
        lmax=5 if fp32 else 15,
        n_lebedev=194 if fp32 else 1202,
        dtype=dtype,
        device=normalized_device,
    )
    cds = TorchLegacySMDCDS(
        symbols, specification.name, device=normalized_device, dtype=dtype
    )
    if fp32:
        from maple.solvation.models.mace_polar_fp32 import (
            build_official_mace_polar_fp32_adapter,
        )

        model = build_official_mace_polar_fp32_adapter(
            device=normalized_device, checkpoint_path=checkpoint_path
        )
    else:
        base = build_official_mace_polar_1_m_adapter(
            device=normalized_device, checkpoint_path=checkpoint_path, torch_graph=True
        )
        model = MACEPolarTorchGraphAdapter(base)
    return MACEPolarTorchPES(
        model=model,
        continuum=continuum,
        solvent_term=cds,
        symbols=symbols,
        device=normalized_device,
        solvent=specification.name,
        _builder_token=_REGISTERED_BUILDER_TOKEN,
        requested_profile=selected,
    )


__all__ = ["MACEPolarTorchPES", "build_smd_mace_polar_torch_pes"]
