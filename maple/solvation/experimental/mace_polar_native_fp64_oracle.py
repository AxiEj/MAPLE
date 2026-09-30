"""Bounded direct-autograd oracle for the native-FP64 v2 scalar.

This module is deliberately independent of the structured-response algebra.
It is a qualification oracle for water and methane (at most five atoms), not a
registered production backend.  Library import and evaluation never change
Torch's process-global default dtype; the dedicated qualification process owns
that precondition.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import InitVar, asdict, dataclass, field
import hashlib
import json
from pathlib import Path

import numpy as np

from maple.function.calculator.extra_correction.implicit.route2_domain import (
    validate_route2_domain,
)
from maple.solvation.coupling.exact_gto import (
    mace_polar_learned_source_embedding_matrix,
)
from maple.solvation.coupling.operator import source_files_sha256
from maple.solvation.coupling.state_equation import geometry_sha256
from maple.solvation.derivatives.analytic import AnalyticHessianEvaluation
from maple.solvation.experimental.mace_polar_torch import (
    _graph_has_path,
    _second_order_resource_preflight,
)

_BUILDER_TOKEN = object()
_TEST_TOKEN = object()
_SOURCE_CHARGE_ATOL_E = 1.0e-8
_HESSIAN_ANTISYMMETRY_LIMIT_EV_A2 = 1.0e-4
_PROVIDER_ID = "maple.native-fp64-direct-autograd-oracle.v2"
_PROFILE_PREFIX = "pure-macepolar-native-fp64-direct-autograd-oracle"
_SCALAR_PREFIX = "route2-experimental-pure-macepolar-native-fp64-ddpcm-smd"
_EVIDENCE_ID = "route2-native-fp64-v2-direct-autograd-qualification"
_ALLOWED_CASES = {
    ("O", "H", "H"): "water",
    ("C", "H", "H", "H", "H"): "methane",
}


def _sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _component_hash(component: object) -> str:
    digest = component.configuration_sha256()
    if not (
        isinstance(digest, str)
        and len(digest) == 64
        and all(character in "0123456789abcdef" for character in digest)
    ):
        raise ValueError("Oracle component configuration must be a SHA256 digest.")
    return digest


def _tensor_sha256(value) -> str:
    array = np.ascontiguousarray(value.detach().cpu().numpy())
    digest = hashlib.sha256(str(array.shape).encode())
    digest.update(str(array.dtype).encode())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def validate_oracle_case(symbols) -> str:
    """Return the fixed case name or reject anything outside the 4 GB panel."""
    normalized = tuple(symbols)
    try:
        return _ALLOWED_CASES[normalized]
    except KeyError as error:
        raise ValueError(
            "The direct-autograd oracle is restricted to ordered water and methane "
            "cases (N<=5) under the unchanged 4 GB second-order cap."
        ) from error


@dataclass(frozen=True, slots=True)
class MACEPolarNativeFP64DirectOracle:
    """Exact direct-AD total scalar used only as the v2 comparison oracle."""

    admitted_capabilities = ()
    scientific_release_admitted = False

    model: object
    continuum: object
    solvent_term: object
    symbols: tuple[str, ...]
    solvent: str = "water"
    device: str = "cpu"
    _builder_token: InitVar[object | None] = None
    _testing_token: InitVar[object | None] = None
    provider_id: str = field(init=False, default=_PROVIDER_ID)
    profile_id: str = field(init=False)
    scalar_contract_id: str = field(init=False)
    _component_hashes: tuple[str, str, str] = field(init=False, repr=False)
    _configuration_hash: str = field(init=False, repr=False)
    _testing: bool = field(init=False, repr=False)

    def __post_init__(self, _builder_token, _testing_token) -> None:
        import torch

        testing = _testing_token is _TEST_TOKEN
        if _testing_token is not None and not testing:
            raise ValueError("Invalid private direct-oracle injection token.")
        if not testing and _builder_token is not _BUILDER_TOKEN:
            raise ValueError("Direct oracle construction requires its v2 builder.")
        symbols = tuple(self.symbols)
        case = validate_oracle_case(symbols)
        device = "cuda:0" if self.device == "cuda" else str(self.device)
        if device != "cpu" and not (
            device.startswith("cuda:") and device[5:].isdigit()
        ):
            raise ValueError("Direct oracle requires explicit cpu or cuda:N device.")
        if self.solvent != "water":
            raise ValueError(
                "The water/methane direct-oracle panel uses water solvent."
            )
        if str(self.model.device) != device or self.model.dtype is not torch.float64:
            raise ValueError("Direct oracle model must match the device and float64.")
        if not testing:
            from maple.solvation.continuum.torch_ddpcm import TorchDDPCM
            from maple.solvation.models.mace_polar_native_fp64 import (
                MACEPolarNativeFP64Adapter,
            )
            from maple.solvation.nonpolar.native_smd_cds import TorchNativeSMDCDS

            if not (
                type(self.model) is MACEPolarNativeFP64Adapter
                and type(self.continuum) is TorchDDPCM
                and type(self.solvent_term) is TorchNativeSMDCDS
            ):
                raise TypeError(
                    "Direct oracle requires the exact native-FP64 model, unchanged "
                    "TorchDDPCM, and native v2 CDS."
                )
        kind = "cpu" if device == "cpu" else "cuda"
        object.__setattr__(self, "symbols", symbols)
        object.__setattr__(self, "device", device)
        object.__setattr__(self, "_testing", testing)
        if testing:
            identity = f"unregistered-engineering-test-direct-autograd-{kind}"
            object.__setattr__(self, "provider_id", identity)
            object.__setattr__(self, "profile_id", identity)
            object.__setattr__(self, "scalar_contract_id", identity)
        else:
            object.__setattr__(
                self, "profile_id", f"{_PROFILE_PREFIX}-{case}-{kind}-v2"
            )
            object.__setattr__(
                self, "scalar_contract_id", f"{_SCALAR_PREFIX}-{kind}-v2"
            )
        object.__setattr__(
            self,
            "_component_hashes",
            tuple(
                _component_hash(component)
                for component in (self.model, self.continuum, self.solvent_term)
            ),
        )
        object.__setattr__(self, "_configuration_hash", self._current_hash())

    @classmethod
    def _for_testing(cls, **kwargs):
        return cls(**kwargs, _testing_token=_TEST_TOKEN)

    def _current_hash(self) -> str:
        root = Path(__file__).resolve().parents[1]
        return _sha(
            {
                "provider_id": self.provider_id,
                "profile_id": self.profile_id,
                "scalar_contract_id": self.scalar_contract_id,
                "symbols": self.symbols,
                "solvent": self.solvent,
                "device": self.device,
                "dtype": "float64",
                "implementation": "direct-autograd",
                "structured_response": False,
                "second_order_limit_bytes": 4_000_000_000,
                "components": tuple(
                    _component_hash(component)
                    for component in (self.model, self.continuum, self.solvent_term)
                ),
                "source_files_sha256": source_files_sha256(
                    {
                        "direct_oracle": Path(__file__),
                        "analytic_result": root / "derivatives" / "analytic.py",
                        "source_embedding": root / "coupling" / "exact_gto.py",
                        "torch_ddpcm": root / "continuum" / "torch_ddpcm.py",
                    }
                ),
                "engineering_test_injection": self._testing,
            }
        )

    def configuration_sha256(self) -> str:
        if self._current_hash() != self._configuration_hash:
            raise RuntimeError(
                "Direct-oracle component or source configuration changed."
            )
        return self._configuration_hash

    def _execution_context(self):
        if self.device == "cpu":
            return nullcontext()
        from .cuda_execution import cuda_model_execution

        return cuda_model_execution()

    def _positions(self, atoms, *, requires_grad: bool):
        import torch

        self.configuration_sha256()
        if tuple(atoms.get_chemical_symbols()) != self.symbols:
            raise ValueError("Direct-oracle ordered symbols changed.")
        validate_route2_domain(atoms)
        return torch.tensor(
            atoms.get_positions(),
            dtype=torch.float64,
            device=self.device,
            requires_grad=requires_grad,
        )

    def _energy_components(self, atoms, positions):
        import torch

        vacuum, learned = self.model.energy_source_torch(atoms, positions)
        if not (
            torch.is_tensor(learned)
            and learned.shape == (len(self.symbols), 4)
            and learned.dtype == positions.dtype
            and learned.device == positions.device
            and bool(torch.isfinite(learned).all())
        ):
            raise RuntimeError("Native-FP64 model returned an invalid learned source.")
        if abs(float(learned[:, 0].sum().detach())) > _SOURCE_CHARGE_ATOL_E:
            raise RuntimeError("Native-FP64 learned charge violates neutrality.")
        if positions.requires_grad and not _graph_has_path(learned, positions):
            raise RuntimeError("Native-FP64 source is disconnected from coordinates.")
        embedding = torch.tensor(
            mace_polar_learned_source_embedding_matrix(),
            dtype=positions.dtype,
            device=positions.device,
        )
        source = learned @ embedding.T
        terms = {
            "vacuum": vacuum,
            "ddpcm": self.continuum.energy_torch(positions, source),
            "cds": self.solvent_term.energy_torch(positions),
        }
        for name, energy in terms.items():
            if not (
                torch.is_tensor(energy)
                and energy.ndim == 0
                and energy.dtype == positions.dtype
                and energy.device == positions.device
                and bool(torch.isfinite(energy))
            ):
                raise RuntimeError(f"{name} did not return a finite FP64 scalar.")
            if positions.requires_grad and not _graph_has_path(energy, positions):
                raise RuntimeError(f"{name} is disconnected from coordinates.")
        return sum(terms.values()), terms, source, learned

    @staticmethod
    def _differentiate(value, positions, *, retain_graph: bool):
        import torch

        if not value.requires_grad:
            return torch.zeros_like(positions)
        derivative = torch.autograd.grad(
            value, positions, retain_graph=retain_graph, allow_unused=True
        )[0]
        return torch.zeros_like(positions) if derivative is None else derivative

    def get_potential_energy(self, atoms) -> float:
        with self._execution_context():
            positions = self._positions(atoms, requires_grad=False)
            total, _, _, _ = self._energy_components(atoms, positions)
            result = float(total.detach())
            if not np.isfinite(result):
                raise RuntimeError("Direct-oracle total energy is not finite.")
            self.configuration_sha256()
            return result

    def get_solvation_energy(self, atoms) -> float:
        with self._execution_context():
            positions = self._positions(atoms, requires_grad=False)
            _, terms, _, _ = self._energy_components(atoms, positions)
            result = float((terms["ddpcm"] + terms["cds"]).detach())
            if not np.isfinite(result):
                raise RuntimeError("Direct-oracle solvation energy is not finite.")
            self.configuration_sha256()
            return result

    def get_forces(self, atoms) -> np.ndarray:
        import torch

        with self._execution_context():
            positions = self._positions(atoms, requires_grad=True)
            total, _, _, _ = self._energy_components(atoms, positions)
            gradient = torch.autograd.grad(total, positions)[0]
            if not bool(torch.isfinite(gradient).all()):
                raise RuntimeError("Direct-oracle forces are not finite.")
            result = -gradient.detach().cpu().numpy().copy()
            self.configuration_sha256()
            return result

    def _resource_preflight(self):
        if self._testing:
            return {"engineering_test_injection": True, "limit_bytes": 4_000_000_000}
        resources = _second_order_resource_preflight(self.continuum, self.device)
        return {
            **resources,
            "policy": "native-fp64-v2-direct-autograd-memory-envelope-v1",
            "reused_estimate_policy": resources["policy"],
        }

    def hessian_vector_product(self, atoms, direction) -> np.ndarray:
        import torch

        vector = np.asarray(direction, dtype=float)
        if vector.shape != (len(self.symbols), 3) or not np.all(np.isfinite(vector)):
            raise ValueError("HVP direction must be finite with shape (N,3).")
        with self._execution_context():
            self._resource_preflight()
            positions = self._positions(atoms, requires_grad=True)
            total, _, _, _ = self._energy_components(atoms, positions)
            gradient = torch.autograd.grad(total, positions, create_graph=True)[0]
            probe = torch.as_tensor(
                vector, dtype=positions.dtype, device=positions.device
            )
            product = self._differentiate(
                (gradient * probe).sum(), positions, retain_graph=False
            )
            if not bool(torch.isfinite(product).all()):
                raise RuntimeError("Direct-oracle HVP is not finite.")
            result = product.detach().cpu().numpy().copy()
            self.configuration_sha256()
            return result

    def evaluate_hessian(self, atoms) -> AnalyticHessianEvaluation:
        import torch

        with self._execution_context():
            resources = self._resource_preflight()
            positions = self._positions(atoms, requires_grad=True)
            total, terms, source, learned = self._energy_components(atoms, positions)
            gradient = torch.autograd.grad(total, positions, create_graph=True)[0]
            entries = gradient.reshape(-1)
            rows = [
                self._differentiate(
                    entry, positions, retain_graph=index + 1 < len(entries)
                ).reshape(-1)
                for index, entry in enumerate(entries)
            ]
            hessian = torch.stack(rows)
            if not bool(torch.isfinite(hessian).all()):
                raise RuntimeError("Direct-autograd Hessian is not finite.")
            asymmetry = float((hessian - hessian.T).abs().max().detach())
            if asymmetry > _HESSIAN_ANTISYMMETRY_LIMIT_EV_A2:
                raise RuntimeError("Direct-autograd Hessian antisymmetry exceeds gate.")
            diagnostics = {
                "implementation": "direct-autograd",
                "structured_response": False,
                "coordinate_finite_differences": False,
                "scientific_release_admitted": False,
                "component_configuration_sha256": dict(
                    zip(("model", "ddpcm", "cds"), self._component_hashes)
                ),
                "second_order_resources": resources,
                "source_active_sha256": _tensor_sha256(learned),
                "source_embedded_sha256": _tensor_sha256(source),
            }
            if not self._testing:
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
                verification_evidence_id=_EVIDENCE_ID,
            )


def build_smd_mace_polar_native_fp64_v2_oracle(
    symbols, *, solvent: str, device: str = "cpu", checkpoint_path=None
) -> MACEPolarNativeFP64DirectOracle:
    """Build the N<=5 direct-AD oracle from the corrected v2 components."""
    import torch

    if torch.get_default_dtype() is not torch.float64:
        raise RuntimeError(
            "Native-FP64 oracle requires process default torch.float64 before build."
        )
    symbols = tuple(symbols)
    validate_oracle_case(symbols)
    if solvent != "water":
        raise ValueError("The water/methane oracle panel is fixed to water solvent.")
    from maple.function.calculator.extra_correction.implicit.smd_cds import (
        smd_coulomb_radii,
    )
    from maple.function.route2_solvents import route2_solvent_spec
    from maple.solvation.continuum.torch_ddpcm import TorchDDPCM
    from maple.solvation.models.mace_polar_native_fp64 import (
        build_official_mace_polar_native_fp64_adapter,
    )
    from maple.solvation.nonpolar.native_smd_cds import TorchNativeSMDCDS

    normalized_device = "cuda:0" if device == "cuda" else str(device)
    if normalized_device.startswith("cuda:"):
        from .cuda_execution import prepare_cuda_execution

        prepare_cuda_execution()
    specification = route2_solvent_spec(solvent)
    continuum = TorchDDPCM(
        symbols,
        smd_coulomb_radii(symbols, solvent=specification.name),
        dielectric=specification.descriptors.dielectric,
        device=normalized_device,
    )
    continuum.preflight_resources(derivative_order=2, limit_bytes=4_000_000_000)
    model = build_official_mace_polar_native_fp64_adapter(
        device=normalized_device, checkpoint_path=checkpoint_path
    )
    cds = TorchNativeSMDCDS(symbols, specification.name, device=normalized_device)
    return MACEPolarNativeFP64DirectOracle(
        model=model,
        continuum=continuum,
        solvent_term=cds,
        symbols=symbols,
        solvent=specification.name,
        device=normalized_device,
        _builder_token=_BUILDER_TOKEN,
    )


__all__ = [
    "MACEPolarNativeFP64DirectOracle",
    "build_smd_mace_polar_native_fp64_v2_oracle",
    "validate_oracle_case",
]
