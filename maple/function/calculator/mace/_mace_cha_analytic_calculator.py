"""Pinned programmatic MACE-OFF23-M plus analytic Gaussian-CHA composition."""

from __future__ import annotations

import hashlib
import inspect
from pathlib import Path

import numpy as np

from ..calculator_base import EV2HARTREE
from ..extra_correction.implicit.gaussian_cha_analytic_correction import (
    GaussianChaAnalyticCorrection,
)
from ._mace_calculator import MACECalculator, build_data_from_atoms

EXPECTED_MACEOFF23M_SHA256 = (
    "ac172fdf9b5173fef4c64667739dbd06f230b0167b4ae2c67e8c08033254c9ee"
)
PINNED_MACEOFF23M_PATH = (
    Path(__file__).resolve().parents[1] / "model" / "maceoff23m.pt"
).resolve()
_EXPECTED_NATIVE_IDENTITY = {
    "model": "maceoff23m",
    "device": "cpu",
    "dtype": "torch.float64",
    "parameter_tensor_count": 29,
    "float64_buffer_count": 48,
    "int64_buffer_count": 2,
    "other_buffer_dtypes": (),
    "r_max_angstrom": 5.0,
    "atomic_numbers": (1, 6, 7, 8, 9, 15, 16, 17, 35, 53),
    "training": False,
    "all_parameter_requires_grad_false": True,
}
_CORRECTION_INTERFACE_METHODS = (
    "preflight_analytic_atoms",
    "get_hessian",
    "get_directional_derivatives",
)
_TRUSTED_MACE_ANALYTIC_INTERFACES = (
    "_require_current_identity",
    "_collect_native_model_identity",
    "_native_model_state_signature",
    "_strict_direction",
    "_gas_energy_leaf_hartree",
    "_analytic_hessian",
    "prepare_analytic_derivatives",
    "get_hessian",
    "get_hvp",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class MACEChaAnalyticCalculator(MACECalculator):
    """One strict unregistered analytic composition for pinned water studies."""

    ANALYTIC_ONLY = True

    def __getattribute__(self, name):
        if name in _TRUSTED_MACE_ANALYTIC_INTERFACES:
            try:
                frozen = object.__getattribute__(self, "_construction_mace_interfaces")
            except AttributeError:
                frozen = ()
            for frozen_name, descriptor in frozen:
                if frozen_name == name:
                    return descriptor.__get__(self, type(self))
        return super().__getattribute__(name)

    def __init__(
        self,
        *,
        correction: GaussianChaAnalyticCorrection,
        model_path: str | Path,
        device: str = "cpu",
        model: str = "maceoff23m",
    ):
        if type(correction) is not GaussianChaAnalyticCorrection:
            raise TypeError(
                "MACE CHA analytic composition requires the exact reviewed "
                "GaussianChaAnalyticCorrection type."
            )
        if device != "cpu":
            raise ValueError(
                "MACE CHA analytic composition requires literal CPU device."
            )
        if model != "maceoff23m":
            raise ValueError("MACE CHA analytic composition requires model maceoff23m.")
        requested = Path(model_path).expanduser()
        if not requested.exists() or requested.resolve() != PINNED_MACEOFF23M_PATH:
            raise ValueError(
                "MACE CHA analytic composition requires the exact local maceoff23m path."
            )
        resolved = requested.resolve()
        observed_sha256 = _sha256_file(resolved)
        if observed_sha256 != EXPECTED_MACEOFF23M_SHA256:
            raise ValueError("Pinned maceoff23m checkpoint SHA256 differs before load.")

        import torch

        super().__init__(
            device=torch.device("cpu"),
            model="maceoff23m",
            model_path=str(resolved),
            implicit="none",
            solvent="none",
        )
        native_identity = self._collect_native_model_identity()
        if native_identity != _EXPECTED_NATIVE_IDENTITY:
            raise ValueError(
                f"Pinned maceoff23m native identity differs: {native_identity!r}."
            )
        self.checkpoint_path = str(resolved)
        self.checkpoint_sha256 = observed_sha256
        self._construction_checkpoint_path = str(resolved)
        self._construction_checkpoint_sha256 = observed_sha256
        self._native_model_identity_frozen = dict(native_identity)
        self._native_model_state_signature_frozen = self._native_model_state_signature()
        self.solvent_correction = correction
        self._construction_correction = correction
        self._construction_correction_fingerprint = (
            correction.analytic_identity_fingerprint
        )
        self._construction_correction_interface = tuple(
            inspect.getattr_static(type(correction), name)
            for name in _CORRECTION_INTERFACE_METHODS
        )
        self._construction_mace_interfaces = tuple(
            (name, inspect.getattr_static(type(self), name))
            for name in _TRUSTED_MACE_ANALYTIC_INTERFACES
        )
        self._bound_correction = None
        self._bound_correction_fingerprint = None
        self._analytic_prepared = False
        self._analytic_binding_provenance = None
        self.last_composed_hvp_provenance = None
        self._direct_hvp_call_count = 0
        self._gas_dense_hessian_graph_count = 0
        self._gas_direct_hvp_graph_count = 0
        self.hessian = "analytic"

    @property
    def analytic_implicit_derivatives_admitted(self) -> bool:
        """Report prepared instance compatibility, not scientific qualification."""
        try:
            self._require_current_identity(require_prepared=True)
        except (RuntimeError, TypeError, ValueError):
            return False
        return True

    @analytic_implicit_derivatives_admitted.setter
    def analytic_implicit_derivatives_admitted(self, value) -> None:
        if bool(value):
            raise AttributeError(
                "Call prepare_analytic_derivatives(); the admission flag cannot prepare itself."
            )
        self._analytic_prepared = False
        self._bound_correction = None
        self._bound_correction_fingerprint = None
        self._analytic_binding_provenance = {
            "prepared": False,
            "analytic_interface_status": "experimental-programmatic-api",
            "scientific_qualification_receipt": False,
        }

    def _collect_native_model_identity(self) -> dict[str, object]:
        import torch

        parameters = tuple(self.model.parameters())
        buffers = tuple(self.model.buffers())
        floating_buffers = tuple(
            value for value in buffers if value.is_floating_point()
        )
        int64_buffers = tuple(value for value in buffers if value.dtype is torch.int64)
        other_buffer_dtypes = tuple(
            sorted(
                {
                    str(value.dtype)
                    for value in buffers
                    if not value.is_floating_point() and value.dtype is not torch.int64
                }
            )
        )
        devices = {str(value.device) for value in (*parameters, *buffers)}
        parameter_dtypes = {str(value.dtype) for value in parameters}
        floating_buffer_dtypes = {str(value.dtype) for value in floating_buffers}
        if devices != {"cpu"}:
            raise ValueError(
                f"Pinned maceoff23m tensors are not all on CPU: {devices}."
            )
        if parameter_dtypes != {"torch.float64"} or floating_buffer_dtypes != {
            "torch.float64"
        }:
            raise ValueError(
                "Pinned maceoff23m parameters and floating buffers must be float64."
            )
        if self.dtype is not torch.float64:
            raise ValueError("Pinned maceoff23m effective dtype must be torch.float64.")
        return {
            "model": "maceoff23m",
            "device": str(self.device),
            "dtype": str(self.dtype),
            "parameter_tensor_count": len(parameters),
            "float64_buffer_count": len(floating_buffers),
            "int64_buffer_count": len(int64_buffers),
            "other_buffer_dtypes": other_buffer_dtypes,
            "r_max_angstrom": float(self.r_max),
            "atomic_numbers": tuple(int(value) for value in self.atomic_numbers),
            "training": bool(self.model.training),
            "all_parameter_requires_grad_false": all(
                not value.requires_grad for value in parameters
            ),
        }

    def _native_model_state_signature(self) -> tuple[object, ...]:
        """Bind the actual loaded object and tensor storage/version identities."""

        def tensor_signature(value):
            return (
                id(value),
                int(value.data_ptr()),
                tuple(int(size) for size in value.shape),
                str(value.dtype),
                str(value.device),
                bool(value.requires_grad),
                int(value._version),
            )

        return (
            id(self.model),
            bool(self.model.training),
            tuple(tensor_signature(value) for value in self.model.parameters()),
            tuple(tensor_signature(value) for value in self.model.buffers()),
        )

    @property
    def native_model_identity(self) -> dict[str, object]:
        return dict(self._native_model_identity_frozen)

    @property
    def analytic_binding_provenance(self) -> dict[str, object]:
        return dict(self._analytic_binding_provenance or {})

    @property
    def direct_hvp_call_count(self) -> int:
        return int(self._direct_hvp_call_count)

    @property
    def analytic_graph_counters(self) -> dict[str, int]:
        return {
            "direct_hvp": self.direct_hvp_call_count,
            "gas_dense_hessian": int(self._gas_dense_hessian_graph_count),
            "gas_direct_hvp": int(self._gas_direct_hvp_graph_count),
            "solvent_total": self.solvent_correction.analytic_graph_counters["total"],
        }

    def _require_current_identity(self, *, require_prepared: bool) -> None:
        instance_values = object.__getattribute__(self, "__dict__")
        for name, descriptor in self._construction_mace_interfaces:
            if name in instance_values:
                self._analytic_prepared = False
                raise RuntimeError(
                    f"MACE CHA reviewed analytic interface {name!r} was shadowed."
                )
            if inspect.getattr_static(type(self), name) is not descriptor:
                self._analytic_prepared = False
                raise RuntimeError(
                    f"MACE CHA reviewed analytic interface {name!r} drifted."
                )
        if (
            self.checkpoint_path != self._construction_checkpoint_path
            or self.checkpoint_sha256 != self._construction_checkpoint_sha256
        ):
            self._analytic_prepared = False
            raise RuntimeError("Pinned MACE checkpoint binding attributes drifted.")
        if self.hessian != "analytic":
            self._analytic_prepared = False
            raise RuntimeError("MACE CHA analytic Hessian mode drifted from analytic.")
        current_sha256 = _sha256_file(Path(self._construction_checkpoint_path))
        if current_sha256 != self.checkpoint_sha256:
            self._analytic_prepared = False
            raise RuntimeError("Pinned MACE checkpoint identity drifted.")
        if self._native_model_state_signature() != (
            self._native_model_state_signature_frozen
        ):
            self._analytic_prepared = False
            raise RuntimeError("Pinned MACE actual loaded model state drifted.")
        if self._collect_native_model_identity() != self._native_model_identity_frozen:
            self._analytic_prepared = False
            raise RuntimeError("Pinned MACE native model identity drifted.")
        correction = getattr(self, "solvent_correction", None)
        if type(correction) is not GaussianChaAnalyticCorrection:
            self._analytic_prepared = False
            raise RuntimeError("MACE CHA analytic binding correction was replaced.")
        if (
            tuple(
                inspect.getattr_static(type(correction), name)
                for name in _CORRECTION_INTERFACE_METHODS
            )
            != self._construction_correction_interface
        ):
            self._analytic_prepared = False
            raise RuntimeError("MACE CHA reviewed correction interface drifted.")
        current_fingerprint = correction.analytic_identity_fingerprint
        if (
            correction is not self._construction_correction
            or current_fingerprint != self._construction_correction_fingerprint
        ):
            self._analytic_prepared = False
            raise RuntimeError(
                "MACE CHA analytic binding construction identity drifted."
            )
        if require_prepared and (
            not self._analytic_prepared
            or correction is not self._bound_correction
            or current_fingerprint != self._bound_correction_fingerprint
        ):
            raise RuntimeError("MACE CHA analytic binding is not currently prepared.")

    def prepare_analytic_derivatives(self) -> None:
        """Bind this exact model and correction instance for later H/HVP calls."""
        self._require_current_identity(require_prepared=False)
        correction = self.solvent_correction
        if correction.analytic_task_derivatives_admitted is not True:
            raise RuntimeError("Gaussian-CHA analytic correction is not admitted.")
        fingerprint = correction.analytic_identity_fingerprint
        self._bound_correction = correction
        self._bound_correction_fingerprint = fingerprint
        self._analytic_prepared = True
        self._analytic_binding_provenance = {
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_sha256": self.checkpoint_sha256,
            "native_model_identity": self.native_model_identity,
            "correction_fingerprint": fingerprint,
            "prepared": True,
            "analytic_interface_status": "experimental-programmatic-api",
            "scientific_qualification_receipt": False,
            "hessian_mode": "analytic",
            "runtime_numerical_hessian_allowed": False,
        }
        self._require_current_identity(require_prepared=True)

    def get_hessian(self, atoms, delta: float | None = None):
        """Use CalcABC's existing gas-plus-solvent dense-H composition."""
        self._require_current_identity(require_prepared=True)
        self.solvent_correction.preflight_analytic_atoms(atoms)
        return super().get_hessian(atoms, delta=delta)

    def _analytic_hessian(self, atoms):
        self._gas_dense_hessian_graph_count += 1
        return super()._analytic_hessian(atoms)

    @staticmethod
    def _strict_direction(direction, atom_count: int) -> np.ndarray:
        if not isinstance(direction, np.ndarray) or direction.dtype != np.float64:
            raise TypeError("MACE CHA HVP direction must be an explicit float64 array.")
        expected = (3 * atom_count,)
        if direction.shape != expected or not np.isfinite(direction).all():
            raise ValueError(
                f"MACE CHA HVP direction must be finite flat shape {expected}."
            )
        return np.array(direction, dtype=np.float64, copy=True)

    def _gas_energy_leaf_hartree(self, atoms):
        import torch

        self._gas_direct_hvp_graph_count += 1
        positions = torch.tensor(
            atoms.get_positions(),
            dtype=torch.float64,
            device="cpu",
            requires_grad=True,
        )
        data_dict, local_or_ghost = build_data_from_atoms(
            atoms,
            self.model,
            device="cpu",
            positions=positions,
            dtype=torch.float64,
        )
        energy_electron_volt = self.model.forward(
            data=data_dict,
            local_or_ghost=local_or_ghost,
            compute_virials=False,
        ).sum()
        energy_hartree = energy_electron_volt * EV2HARTREE
        return positions, energy_hartree

    def get_hvp(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, atoms, n: np.ndarray
    ):
        """Return direct composed `(Hn, force, energy)` without dense H or FD."""
        import torch

        self._require_current_identity(require_prepared=True)
        direction = self._strict_direction(n, len(atoms))
        self.solvent_correction.preflight_analytic_atoms(atoms)
        self._reject_unsupported_pbc(atoms)
        positions, gas_energy = self._gas_energy_leaf_hartree(atoms)
        gas_gradient = torch.autograd.grad(gas_energy, positions, create_graph=True)[0]
        vector = torch.as_tensor(
            direction.reshape(positions.shape), dtype=torch.float64, device="cpu"
        )
        gas_hvp = torch.autograd.grad(
            gas_gradient,
            positions,
            grad_outputs=vector,
        )[0]
        correction = self.solvent_correction
        solvent = correction.get_directional_derivatives(atoms, direction)
        total_hvp = gas_hvp.reshape(-1).detach().cpu().numpy() + np.asarray(
            solvent.hvp_hartree_per_angstrom2, dtype=np.float64
        )
        total_forces = (-gas_gradient).reshape(-1).detach().cpu().numpy() + np.asarray(
            solvent.forces_hartree_per_angstrom, dtype=np.float64
        ).reshape(-1)
        total_energy = float(gas_energy.detach().cpu()) + float(solvent.energy_hartree)
        if (
            total_hvp.shape != direction.shape
            or total_forces.shape != direction.shape
            or not np.isfinite(total_hvp).all()
            or not np.isfinite(total_forces).all()
            or not np.isfinite(total_energy)
        ):
            raise RuntimeError(
                "MACE CHA composed directional derivatives are nonfinite."
            )
        self._direct_hvp_call_count += 1
        self.last_composed_hvp_provenance = {
            "derivative": "direct-composed-hvp",
            "runtime_finite_difference": False,
            "dense_hessian_materialized": False,
            "analytic_interface_status": "experimental-programmatic-api",
            "scientific_qualification_receipt": False,
            "checkpoint_sha256": self.checkpoint_sha256,
            "correction_fingerprint": correction.analytic_identity_fingerprint,
            "solvent_derivative": solvent.provenance.get("derivative"),
            "call_count": self._direct_hvp_call_count,
        }
        return total_hvp, total_forces, total_energy
