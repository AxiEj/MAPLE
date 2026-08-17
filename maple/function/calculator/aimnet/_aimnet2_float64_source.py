"""Source-bound float64 reconstruction of the legacy AIMNet2 checkpoint.

The historical MAPLE TorchScript asset hard-codes ``torch.float32`` in its
serialized ``_prepare_dtype`` method.  Casting that module with ``double()``
therefore does *not* create a float64 coordinate graph: every forward converts
the coordinates back to float32 before evaluating distances.

This module provides a deliberately narrow research runtime.  It rebuilds the
same AIMNet2 wB97M-D3 architecture from the configuration shipped by the
official ``aimnet==0.2.0`` package, copies the unchanged legacy checkpoint
state dictionary into that architecture, and then evaluates the complete
model in float64.  It exposes only the energy/charge coordinate primitives
needed by the geometry-mediated Route-2 adapter.  It is not registered as a
MAPLE calculator and explicitly refuses the public ASE ``calculate`` path.

The package version and all AIMNet configuration, source, kernel-registration,
and DFT-D3 data files used by this path are content-addressed below.  A changed
upstream runtime fails closed instead of silently defining another numerical
model.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from importlib import import_module, metadata
import hashlib
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from maple.function.calculator.aimnet._aimnet2_calculator import (
    AIMNet2Calculator,
    AIMNet2ChargePositionResponse,
    AIMNet2ChargeState,
    nblist_dense_padded,
    pad_dim0,
)
from maple.function.calculator.calculator_base import CalcABC

AIMNET_FLOAT64_RUNTIME_VERSION = "aimnet-reconstructed-float64-runtime-v3"
AIMNET_REQUIRED_PACKAGE_VERSION = "0.2.0"
AIMNET_MODEL_CONFIGURATION = "models/aimnet2_dftd3_wb97m.yaml"
AIMNET_FIRST_ORDER_ENERGY_PARITY_ABSOLUTE_TOLERANCE_EV = 1.0e-6
AIMNET_SECOND_ORDER_ENERGY_PARITY_ABSOLUTE_TOLERANCE_EV = 1.0e-8
AIMNET_SECOND_ORDER_CHARGE_PARITY_ABSOLUTE_TOLERANCE_E = 1.0e-10
AIMNET_SECOND_ORDER_GRADIENT_PARITY_ABSOLUTE_TOLERANCE_EV_PER_A = 1.0e-7
AIMNET_SECOND_ORDER_CHARGE_VJP_PARITY_ABSOLUTE_TOLERANCE_EV_PER_A = 1.0e-10
AIMNET_SECOND_ORDER_CHARGE_TANGENT_TOLERANCE_E_PER_A = 1.0e-10

# A version string alone is too weak for a scientific forward graph.  Bind the
# upstream Python modules, architecture configuration, kernel registration,
# and DFT-D3 data asset used by the dense molecular AIMNet2 path.
AIMNET_REQUIRED_FILE_SHA256 = MappingProxyType(
    {
        "__init__.py": "bb18b97049cb5f783f0bb9d49f11b050b845fb0c8d214c1c6142260d8867a0a0",
        "config.py": "0232457935c9e08aae689d778de70a872b5ecf409b1d138ace5c3271f63b9360",
        "constants.py": "4fa79bfc21b9764c6e520db65dff16885d22181f892e82cdcc7abd47c491ceb4",
        "dftd3_data.pt": "83d1d0ff69de90cb3ded49c80c8151786106f2aea7eb3425abe625e4b1909ce2",
        "kernels/__init__.py": (
            "7d692e675f0fde3bfea8d1815ea4fee2dceb41e6e6d3e993cb742e2843ef39be"
        ),
        "kernels/conv_sv_2d_sp_wp.py": (
            "b67d5a342f602f91514785c13be20a00ab9c5840f2995729d518e5668a069e12"
        ),
        "models/__init__.py": (
            "fdc6de2603dfdf9c505bb9e26f5ce7d15784be04f12b8997464c18178ef6b5db"
        ),
        "models/base.py": "f22ed8577cff74cc63e7b51897e231565733732f61f8377db9bf616b8d73bf37",
        "models/aimnet2.py": "a6fcfb549ab3906165f1721a2de1332078f63c5d17dd276fe6d67887e1a9658f",
        "models/utils.py": (
            "4670c83ab1c37459526c60b9da0298252a76d2a79feffb571e1d99987c221d0f"
        ),
        "models/aimnet2_dftd3_wb97m.yaml": (
            "1a4f3af7fd914085f3947c1135b7ec6a2e2ef703d7de803220d2b523bd5e0452"
        ),
        "modules/__init__.py": (
            "23ef617aa45d9d77b2a4c523cde43a6381004ccffd9f8345c820ba3a6e87d95c"
        ),
        "modules/aev.py": "c283deaa5a4cb3f9faccec6b04c457848ab421a07629e639c3b89391458efdd7",
        "modules/core.py": "e95b05a9add7534c3b785ea1743af3f5e269cb20afa228ff5f55d929a5aa239a",
        "modules/lr.py": "fe31354eede1dc4d8042ff5f0faab8d1b719b03a347e2227e2c7cdd912ac076f",
        "nbops.py": "4a6f78489e719aa8cac37d6b7ac09dfdcda6368aa8aec4984f88c299f51c7c50",
        "ops.py": "5e06d7cb2fe7e231ec336282b505bdf1236f082c55389c2ace5f2abc5165ecda",
    }
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _upstream_runtime() -> tuple[Path, dict[str, str], dict[str, str]]:
    """Return and validate the exact optional upstream AIMNet runtime."""

    try:
        actual_version = metadata.version("aimnet")
    except metadata.PackageNotFoundError as exc:  # pragma: no cover - host optional
        raise ModuleNotFoundError(
            "The float64 AIMNet2 research runtime requires aimnet==0.2.0."
        ) from exc
    if actual_version != AIMNET_REQUIRED_PACKAGE_VERSION:
        raise RuntimeError(
            "The float64 AIMNet2 research runtime is source-bound to "
            f"aimnet=={AIMNET_REQUIRED_PACKAGE_VERSION}; found {actual_version}."
        )

    aimnet = import_module("aimnet")
    module_file = getattr(aimnet, "__file__", None)
    if not isinstance(module_file, str) or not module_file:
        raise RuntimeError("The installed AIMNet package has no filesystem origin.")
    root = Path(module_file).resolve().parent
    observed: dict[str, str] = {}
    for relative, expected in AIMNET_REQUIRED_FILE_SHA256.items():
        path = root / relative
        if not path.is_file():
            raise RuntimeError(f"Required AIMNet runtime file is missing: {relative}.")
        digest = _sha256_file(path)
        if digest != expected:
            raise RuntimeError(
                "Installed AIMNet file differs from the source-bound runtime: "
                f"{relative}."
            )
        observed[relative] = digest

    package_versions: dict[str, str] = {}
    for distribution in (
        "aimnet",
        "torch",
        "warp-lang",
        "nvalchemi-toolkit-ops",
        "pyyaml",
    ):
        try:
            package_versions[distribution] = metadata.version(distribution)
        except metadata.PackageNotFoundError as exc:
            raise ModuleNotFoundError(
                f"The float64 AIMNet2 runtime is missing {distribution}."
            ) from exc
    return root, observed, package_versions


@dataclass(frozen=True, slots=True)
class AIMNet2ChargePositionSecondOrderResponse:
    """One source-bound AIMNet2 coordinate HVP ledger.

    The charge cotangent is held fixed in
    ``contracted_charge_hessian_ev_per_angstrom2``.  Consequently that term is
    exactly ``D_R[J_q(R).T v][h]`` rather than the derivative of a
    geometry-dependent reaction potential.  The latter response belongs to
    the continuum HVP and is composed at the scalar layer.
    """

    charge_state: AIMNet2ChargeState
    charge_cotangent_ev_per_e: np.ndarray
    coordinate_direction: np.ndarray
    intrinsic_energy_gradient_ev_per_angstrom: np.ndarray
    charge_position_vjp_ev_per_angstrom: np.ndarray
    charge_position_jvp_e_per_angstrom: np.ndarray
    intrinsic_energy_hvp_ev_per_angstrom2: np.ndarray
    contracted_charge_hessian_ev_per_angstrom2: np.ndarray
    standard_decomposed_energy_absolute_error_ev: float
    standard_decomposed_charge_max_absolute_error_e: float
    standard_decomposed_intrinsic_gradient_max_absolute_error_ev_per_angstrom: float
    standard_decomposed_charge_vjp_max_absolute_error_ev_per_angstrom: float
    charge_tangent_residual_e_per_angstrom: float

    def __post_init__(self) -> None:
        if not isinstance(self.charge_state, AIMNet2ChargeState):
            raise TypeError("charge_state must be AIMNet2ChargeState.")
        count = self.charge_state.charges_e.size
        arrays = {
            "charge_cotangent_ev_per_e": ((count,), self.charge_cotangent_ev_per_e),
            "coordinate_direction": ((count, 3), self.coordinate_direction),
            "intrinsic_energy_gradient_ev_per_angstrom": (
                (count, 3),
                self.intrinsic_energy_gradient_ev_per_angstrom,
            ),
            "charge_position_vjp_ev_per_angstrom": (
                (count, 3),
                self.charge_position_vjp_ev_per_angstrom,
            ),
            "charge_position_jvp_e_per_angstrom": (
                (count,),
                self.charge_position_jvp_e_per_angstrom,
            ),
            "intrinsic_energy_hvp_ev_per_angstrom2": (
                (count, 3),
                self.intrinsic_energy_hvp_ev_per_angstrom2,
            ),
            "contracted_charge_hessian_ev_per_angstrom2": (
                (count, 3),
                self.contracted_charge_hessian_ev_per_angstrom2,
            ),
        }
        for name, (shape, raw) in arrays.items():
            values = np.asarray(raw, dtype=float)
            if values.shape != shape or not np.all(np.isfinite(values)):
                raise ValueError(f"{name} must be finite with shape {shape}.")
            values = np.array(values, copy=True)
            values.setflags(write=False)
            object.__setattr__(self, name, values)
        for name in (
            "standard_decomposed_energy_absolute_error_ev",
            "standard_decomposed_charge_max_absolute_error_e",
            "standard_decomposed_intrinsic_gradient_max_absolute_error_ev_per_angstrom",
            "standard_decomposed_charge_vjp_max_absolute_error_ev_per_angstrom",
            "charge_tangent_residual_e_per_angstrom",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative.")
            object.__setattr__(self, name, value)
        if (
            self.charge_tangent_residual_e_per_angstrom
            > AIMNET_SECOND_ORDER_CHARGE_TANGENT_TOLERANCE_E_PER_A
        ):
            raise ValueError(
                "AIMNet2 charge JVP leaves the fixed-total-charge tangent space."
            )


class AIMNet2ReconstructedFloat64SourceCalculator(AIMNet2Calculator):
    """Float64 energy/charge primitive with unchanged AIMNet2 checkpoint weights.

    This class intentionally supports neither ASE calculation nor public
    Hessians.  Its admitted consumers are the research source adapter's
    energy/charge first-order calls and the explicitly diagnostic
    ``charge_position_second_order`` ledger.
    """

    implemented_properties: list[str] = []
    inference_dtype = "float64"
    runtime_kind = AIMNET_FLOAT64_RUNTIME_VERSION

    def __init__(self, *, model_path: str | Path, device: str = "cpu") -> None:
        torch = import_module("torch")
        yaml = import_module("yaml")
        build_module = import_module("aimnet.config").build_module

        resolved_device = torch.device(device)
        if resolved_device.type != "cpu":
            raise NotImplementedError(
                "The source-bound AIMNet2 float64 runtime is currently validated "
                "only on CPU."
            )
        checkpoint = Path(model_path).expanduser().resolve(strict=True)
        root, source_hashes, package_versions = _upstream_runtime()
        configuration_path = root / AIMNET_MODEL_CONFIGURATION
        configuration = yaml.safe_load(configuration_path.read_text(encoding="utf-8"))
        if not isinstance(configuration, dict):
            raise RuntimeError("The source-bound AIMNet architecture is invalid.")

        CalcABC.__init__(self)
        model = build_module(copy.deepcopy(configuration))
        if not hasattr(model, "outputs") or not hasattr(model.outputs, "atomic_shift"):
            raise RuntimeError("AIMNet architecture omitted the atomic-shift head.")

        # Preserve the checkpoint's float64 self-atomic-energy table before
        # load_state_dict; loading it into a float32 destination would truncate
        # a geometry-independent but provenance-relevant energy constant.
        model.outputs.atomic_shift.double()
        legacy = torch.jit.load(str(checkpoint), map_location="cpu")
        load_result = model.load_state_dict(legacy.state_dict(), strict=False)
        del legacy
        if load_result.missing_keys or set(load_result.unexpected_keys) != {
            "impemented_species"
        }:
            raise RuntimeError(
                "Legacy AIMNet2 state dictionary does not exactly match the "
                "source-bound Python architecture."
            )

        model = model.to(device=resolved_device, dtype=torch.float64).eval()
        model.requires_grad_(False)
        # AIMNet2Base reads these instance attributes in _prepare_dtype.  The
        # upstream defaults are captured as float32 at module import, so merely
        # converting parameters is insufficient.
        model._required_keys_dtype = [  # noqa: SLF001 - version-bound upstream API
            torch.float64,
            torch.int64,
            torch.float64,
        ]
        model._optional_keys_dtype = [  # noqa: SLF001 - version-bound upstream API
            torch.float64,
            torch.int32,
            torch.int32,
            torch.int64,
            torch.float64,
            torch.float64,
            torch.float64,
            torch.int32,
            torch.float64,
            torch.float64,
            torch.int32,
            torch.float64,
            torch.float64,
            torch.bool,
        ]
        floating_tensors = tuple(model.parameters()) + tuple(model.buffers())
        if any(
            tensor.is_floating_point() and tensor.dtype != torch.float64
            for tensor in floating_tensors
        ):
            raise RuntimeError("AIMNet2 reconstruction retained a non-float64 tensor.")

        # The embedded DFT-D3 coordinate derivative is a custom first-order
        # autograd function whose sub-millimilliangstrom energy differences are
        # noisy even in the otherwise float64 graph.  The same source-bound
        # upstream module exposes a smooth differentiable ``hessian=True``
        # energy.  Preserve the ordinary forward as a per-geometry parity oracle
        # and return first/second derivatives from a separate frozen graph in
        # which DFT-D3 is removed and reapplied through that upstream path.
        second_order_model = copy.deepcopy(model)
        second_order_dftd3 = getattr(second_order_model.outputs, "dftd3", None)
        if second_order_dftd3 is None or not callable(second_order_dftd3):
            raise RuntimeError(
                "The source-bound AIMNet2 architecture omitted its DFT-D3 module."
            )
        second_order_model.outputs.dftd3 = torch.nn.Identity()
        second_order_model.eval().requires_grad_(False)
        second_order_dftd3.eval().requires_grad_(False)

        self.device = resolved_device
        self.model_name = "aimnet2"
        self.model_path = str(checkpoint)
        self.model = model
        self._second_order_model = second_order_model
        self._second_order_dftd3 = second_order_dftd3
        self._supports_atom_charges = True
        self._coulomb_method = "simple"
        self.cutoff = float(configuration["kwargs"]["aev"]["rc_s"])
        self.cutoff_lr = float("inf")
        self.solvent_correction = None
        self._last_charge_state: AIMNet2ChargeState | None = None
        self._last_ordinary_decomposed_response_parity: dict[str, float] | None = None
        self._runtime_provenance = {
            "runtime_kind": AIMNET_FLOAT64_RUNTIME_VERSION,
            "aimnet_package_version": AIMNET_REQUIRED_PACKAGE_VERSION,
            "aimnet_runtime_files_sha256": dict(sorted(source_hashes.items())),
            "package_versions": dict(sorted(package_versions.items())),
            "architecture_configuration": AIMNET_MODEL_CONFIGURATION,
            "architecture_configuration_sha256": source_hashes[
                AIMNET_MODEL_CONFIGURATION
            ],
            "checkpoint_state_dict_policy": (
                "unchanged-legacy-state-dict; expected unused sentinel buffer "
                "impemented_species only"
            ),
            "coordinate_dtype": "torch.float64",
            "parameter_dtype": "torch.float64",
            "first_order_coordinate_graph": (
                "frozen-deep-copy-with-embedded-dftd3-replaced-by-identity; "
                "source-bound-upstream-dftd3-reapplied-with-hessian-true"
            ),
            "ordinary_forward_role": (
                "per-geometry energy-charge-gradient-vjp parity oracle only"
            ),
            "first_order_ordinary_decomposed_parity_tolerances": {
                "energy_absolute_eV": (
                    AIMNET_FIRST_ORDER_ENERGY_PARITY_ABSOLUTE_TOLERANCE_EV
                ),
                "charge_absolute_e": (
                    AIMNET_SECOND_ORDER_CHARGE_PARITY_ABSOLUTE_TOLERANCE_E
                ),
                "intrinsic_gradient_absolute_eV_per_A": (
                    AIMNET_SECOND_ORDER_GRADIENT_PARITY_ABSOLUTE_TOLERANCE_EV_PER_A
                ),
                "charge_vjp_absolute_eV_per_A": (
                    AIMNET_SECOND_ORDER_CHARGE_VJP_PARITY_ABSOLUTE_TOLERANCE_EV_PER_A
                ),
            },
            "second_order_coordinate_graph": (
                "frozen-deep-copy-with-embedded-dftd3-replaced-by-identity; "
                "source-bound-upstream-dftd3-reapplied-with-hessian-true"
            ),
            "second_order_standard_forward_parity_tolerances": {
                "energy_absolute_eV": (
                    AIMNET_SECOND_ORDER_ENERGY_PARITY_ABSOLUTE_TOLERANCE_EV
                ),
                "charge_absolute_e": (
                    AIMNET_SECOND_ORDER_CHARGE_PARITY_ABSOLUTE_TOLERANCE_E
                ),
                "intrinsic_gradient_absolute_eV_per_A": (
                    AIMNET_SECOND_ORDER_GRADIENT_PARITY_ABSOLUTE_TOLERANCE_EV_PER_A
                ),
                "charge_vjp_absolute_eV_per_A": (
                    AIMNET_SECOND_ORDER_CHARGE_VJP_PARITY_ABSOLUTE_TOLERANCE_EV_PER_A
                ),
                "charge_tangent_absolute_e_per_A": (
                    AIMNET_SECOND_ORDER_CHARGE_TANGENT_TOLERANCE_E_PER_A
                ),
            },
            "public_hessian": False,
            "public_hvp": False,
            "public_ase_calculator": False,
            "route2_public_ase_admitted": False,
        }

    def runtime_provenance(self) -> dict[str, Any]:
        """Return a detached description of the external numerical runtime."""

        return copy.deepcopy(self._runtime_provenance)

    def last_ordinary_decomposed_parity(self) -> dict[str, float]:
        """Return the last full first-order ordinary/decomposed parity ledger."""

        if self._last_ordinary_decomposed_response_parity is None:
            raise RuntimeError(
                "No ordinary/decomposed first-order response parity is available."
            )
        return dict(self._last_ordinary_decomposed_response_parity)

    def calculate(self, atoms=None, properties=None, system_changes=None):
        raise NotImplementedError(
            "AIMNet2ReconstructedFloat64SourceCalculator is a Route-2 research "
            "source primitive, not a public ASE calculator."
        )

    def _build_data(self, coord, atoms):
        torch = import_module("torch")

        if coord.dtype != torch.float64 or coord.device != self.device:
            raise ValueError("AIMNet2 float64 coordinates have the wrong runtime.")
        numbers = torch.as_tensor(
            atoms.get_atomic_numbers(), dtype=torch.int32, device=self.device
        )
        count = int(coord.shape[0])
        molecule_index = torch.zeros(count, dtype=torch.int64, device=self.device)
        neighbor_matrix = nblist_dense_padded(coord, self.cutoff)
        data = {
            "coord": pad_dim0(coord, value=0.0),
            "numbers": pad_dim0(numbers, value=0),
            "charge": torch.tensor(
                [self._total_charge_from_atoms(atoms)],
                dtype=torch.float64,
                device=self.device,
            ),
            "mult": torch.tensor(
                [float(atoms.info.get("mult", 1.0))],
                dtype=torch.float64,
                device=self.device,
            ),
            "mol_idx": pad_dim0(molecule_index, value=0),
            "nbmat": neighbor_matrix,
            "nbmat_lr": neighbor_matrix,
            "cutoff_lr": torch.tensor(
                self.cutoff, dtype=torch.float64, device=self.device
            ),
        }
        return data

    def _validated_forward(self, atoms, *, requires_grad: bool):
        torch = import_module("torch")

        self._reject_unsupported_pbc(atoms)
        self._validate_charge_output_domain(atoms)
        coordinate = torch.tensor(
            atoms.get_positions(),
            dtype=torch.float64,
            device=self.device,
            requires_grad=requires_grad,
        )
        data = self._build_data(coordinate, atoms)
        output = self._forward_output(data)
        for key in ("coord", "energy", "charges"):
            value = output.get(key)
            if value is None or value.dtype != torch.float64:
                raise RuntimeError(f"AIMNet2 forward did not preserve float64 {key}.")
        return data, output

    def _validated_decomposed_forward(self, atoms):
        """Return the smooth DFT-D3-decomposed graph before parity comparison."""

        torch = import_module("torch")

        self._reject_unsupported_pbc(atoms)
        self._validate_charge_output_domain(atoms)
        coordinate = torch.tensor(
            atoms.get_positions(),
            dtype=torch.float64,
            device=self.device,
            requires_grad=True,
        )
        data = self._build_data(coordinate, atoms)
        output = self._second_order_model(data)
        output = self._second_order_dftd3(output, hessian=True)
        if not isinstance(output, dict):
            raise RuntimeError("AIMNet2 second-order forward did not return a mapping.")
        for key in ("coord", "energy", "charges"):
            value = output.get(key)
            if value is None or value.dtype != torch.float64:
                raise RuntimeError(
                    f"AIMNet2 second-order forward did not preserve float64 {key}."
                )
        return data, output

    def _response_from_output(
        self,
        atoms,
        charge_cotangent_ev_per_e: np.ndarray,
        data,
        output,
    ) -> AIMNet2ChargePositionResponse:
        """Differentiate one already validated energy/charge forward graph."""

        torch = import_module("torch")

        atom_count = len(atoms)
        cotangent = np.asarray(charge_cotangent_ev_per_e, dtype=float)
        state = self._charge_state_from_output(
            output,
            atom_count=atom_count,
            requested_total_charge_e=self._total_charge_from_atoms(atoms),
        )
        charges = self._differentiable_charges(
            output,
            atom_count=atom_count,
            requested_total_charge_e=state.requested_total_charge_e,
        )
        energy = self._energy_from_output(output)
        intrinsic = torch.autograd.grad(energy, data["coord"], retain_graph=True)[0][
            :atom_count
        ]
        pairing = torch.sum(
            charges
            * torch.as_tensor(cotangent, dtype=charges.dtype, device=charges.device)
        )
        response = torch.autograd.grad(pairing, data["coord"])[0][:atom_count]
        return AIMNet2ChargePositionResponse(
            charge_state=state,
            charge_cotangent_ev_per_e=np.array(cotangent, copy=True),
            intrinsic_energy_gradient_ev_per_angstrom=(
                intrinsic.detach().cpu().numpy()
            ),
            charge_position_vjp_ev_per_angstrom=response.detach().cpu().numpy(),
        )

    @staticmethod
    def _state_parity_errors(
        ordinary: AIMNet2ChargeState,
        decomposed: AIMNet2ChargeState,
    ) -> dict[str, float]:
        return {
            "energy_absolute_error_eV": abs(
                float(decomposed.energy_ev) - float(ordinary.energy_ev)
            ),
            "charge_max_absolute_error_e": max(
                float(
                    np.max(np.abs(decomposed.raw_charges_e - ordinary.raw_charges_e))
                ),
                float(np.max(np.abs(decomposed.charges_e - ordinary.charges_e))),
            ),
        }

    @staticmethod
    def _validate_parity_errors(
        errors: dict[str, float],
        *,
        energy_tolerance_eV: float = (
            AIMNET_FIRST_ORDER_ENERGY_PARITY_ABSOLUTE_TOLERANCE_EV
        ),
        comparison: str = "ordinary-forward",
    ) -> None:
        tolerances = {
            "energy_absolute_error_eV": float(energy_tolerance_eV),
            "charge_max_absolute_error_e": (
                AIMNET_SECOND_ORDER_CHARGE_PARITY_ABSOLUTE_TOLERANCE_E
            ),
            "intrinsic_gradient_max_absolute_error_eV_per_A": (
                AIMNET_SECOND_ORDER_GRADIENT_PARITY_ABSOLUTE_TOLERANCE_EV_PER_A
            ),
            "charge_vjp_max_absolute_error_eV_per_A": (
                AIMNET_SECOND_ORDER_CHARGE_VJP_PARITY_ABSOLUTE_TOLERANCE_EV_PER_A
            ),
        }
        failures = [
            f"{name}={error:.6e}>{tolerances[name]:.6e}"
            for name, error in errors.items()
            if error > tolerances[name]
        ]
        if failures:
            raise RuntimeError(
                f"AIMNet2 differentiable DFT-D3 graph failed {comparison} "
                f"parity: {', '.join(failures)}."
            )

    def _differentiable_charges(
        self,
        output,
        *,
        atom_count: int,
        requested_total_charge_e: float,
    ):
        torch = import_module("torch")

        raw = output.get("charges")
        if raw is None or not raw.requires_grad:
            raise RuntimeError("AIMNet2 float64 charges are not differentiable.")
        raw = raw[:atom_count]
        target = torch.as_tensor(
            requested_total_charge_e, dtype=raw.dtype, device=raw.device
        )
        return raw + (target - raw.sum()) / atom_count

    def _second_order_charge_map(self, coordinate, atoms):
        output = self._second_order_model(self._build_data(coordinate, atoms))
        return self._differentiable_charges(
            output,
            atom_count=len(atoms),
            requested_total_charge_e=self._total_charge_from_atoms(atoms),
        )

    def charge_state(self, atoms) -> AIMNet2ChargeState:
        """Evaluate one smooth, ordinary-forward-parity-gated float64 state."""

        torch = import_module("torch")

        with torch.no_grad():
            _, ordinary_output = self._validated_forward(atoms, requires_grad=False)
        ordinary = self._charge_state_from_output(
            ordinary_output,
            atom_count=len(atoms),
            requested_total_charge_e=self._total_charge_from_atoms(atoms),
        )
        _, decomposed_output = self._validated_decomposed_forward(atoms)
        decomposed = self._charge_state_from_output(
            decomposed_output,
            atom_count=len(atoms),
            requested_total_charge_e=self._total_charge_from_atoms(atoms),
        )
        self._validate_parity_errors(self._state_parity_errors(ordinary, decomposed))
        self._last_charge_state = decomposed
        return decomposed

    def charge_position_response(
        self,
        atoms,
        charge_cotangent_ev_per_e: np.ndarray,
    ) -> AIMNet2ChargePositionResponse:
        """Differentiate the smooth graph after ordinary-forward parity checks."""

        atom_count = len(atoms)
        cotangent = np.array(charge_cotangent_ev_per_e, dtype=float, copy=True)
        if cotangent.shape != (atom_count,) or not np.all(np.isfinite(cotangent)):
            raise ValueError(
                "AIMNet2 charge cotangent must be finite with shape "
                f"{(atom_count,)}."
            )
        ordinary_data, ordinary_output = self._validated_forward(
            atoms, requires_grad=True
        )
        ordinary = self._response_from_output(
            atoms,
            cotangent,
            ordinary_data,
            ordinary_output,
        )
        decomposed_data, decomposed_output = self._validated_decomposed_forward(atoms)
        decomposed = self._response_from_output(
            atoms,
            cotangent,
            decomposed_data,
            decomposed_output,
        )
        parity = self._state_parity_errors(
            ordinary.charge_state,
            decomposed.charge_state,
        )
        parity.update(
            {
                "intrinsic_gradient_max_absolute_error_eV_per_A": float(
                    np.max(
                        np.abs(
                            decomposed.intrinsic_energy_gradient_ev_per_angstrom
                            - ordinary.intrinsic_energy_gradient_ev_per_angstrom
                        )
                    )
                ),
                "charge_vjp_max_absolute_error_eV_per_A": float(
                    np.max(
                        np.abs(
                            decomposed.charge_position_vjp_ev_per_angstrom
                            - ordinary.charge_position_vjp_ev_per_angstrom
                        )
                    )
                ),
            }
        )
        self._validate_parity_errors(parity)
        self._last_ordinary_decomposed_response_parity = dict(parity)
        self._last_charge_state = decomposed.charge_state
        return decomposed

    def charge_position_second_order(
        self,
        atoms,
        charge_cotangent_ev_per_e: np.ndarray,
        coordinate_direction: np.ndarray,
    ) -> AIMNet2ChargePositionSecondOrderResponse:
        """Apply the intrinsic and fixed-cotangent charge coordinate Hessians.

        The public first-order response is the smooth DFT-D3-decomposed graph,
        already parity-gated against the ordinary AIMNet2 graph.  A fresh copy
        of that smooth graph must reproduce the public response before any
        second-order quantity is exposed.
        """

        torch = import_module("torch")

        atom_count = len(atoms)
        cotangent = np.array(charge_cotangent_ev_per_e, dtype=float, copy=True)
        direction = np.array(coordinate_direction, dtype=float, copy=True)
        if cotangent.shape != (atom_count,) or not np.all(np.isfinite(cotangent)):
            raise ValueError(
                "AIMNet2 charge cotangent must be finite with shape "
                f"{(atom_count,)}."
            )
        if direction.shape != (atom_count, 3) or not np.all(np.isfinite(direction)):
            raise ValueError(
                "AIMNet2 coordinate direction must be finite with shape "
                f"{(atom_count, 3)}."
            )

        reference = self.charge_position_response(atoms, cotangent)
        ordinary_decomposed_parity = self.last_ordinary_decomposed_parity()
        data, output = self._validated_decomposed_forward(atoms)
        candidate_state = self._charge_state_from_output(
            output,
            atom_count=atom_count,
            requested_total_charge_e=self._total_charge_from_atoms(atoms),
        )
        charges = self._differentiable_charges(
            output,
            atom_count=atom_count,
            requested_total_charge_e=candidate_state.requested_total_charge_e,
        )
        energy = self._energy_from_output(output)
        intrinsic_full = torch.autograd.grad(
            energy,
            data["coord"],
            create_graph=True,
            retain_graph=True,
        )[0]
        intrinsic = intrinsic_full[:atom_count]
        cotangent_tensor = torch.as_tensor(
            cotangent, dtype=charges.dtype, device=charges.device
        )
        pairing = torch.sum(charges * cotangent_tensor)
        charge_vjp_full = torch.autograd.grad(
            pairing,
            data["coord"],
            create_graph=True,
            retain_graph=True,
        )[0]
        charge_vjp = charge_vjp_full[:atom_count]
        direction_tensor = torch.as_tensor(
            direction, dtype=charges.dtype, device=charges.device
        )

        def directional_hvp(gradient, *, retain_graph: bool):
            contraction = torch.sum(gradient[:atom_count] * direction_tensor)
            if not contraction.requires_grad:
                return torch.zeros_like(data["coord"][:atom_count])
            result = torch.autograd.grad(
                contraction,
                data["coord"],
                retain_graph=retain_graph,
                allow_unused=True,
            )[0]
            if result is None:
                return torch.zeros_like(data["coord"][:atom_count])
            return result[:atom_count]

        intrinsic_hvp = directional_hvp(intrinsic_full, retain_graph=True)
        contracted_charge_hessian = directional_hvp(charge_vjp_full, retain_graph=False)

        jvp_coordinate = torch.tensor(
            atoms.get_positions(),
            dtype=torch.float64,
            device=self.device,
        )
        _, charge_jvp = torch.autograd.functional.jvp(
            lambda candidate: self._second_order_charge_map(candidate, atoms),
            jvp_coordinate,
            direction_tensor,
            create_graph=False,
            strict=False,
        )

        energy_error = abs(
            float(candidate_state.energy_ev) - float(reference.charge_state.energy_ev)
        )
        charge_error = max(
            float(
                np.max(
                    np.abs(
                        candidate_state.raw_charges_e
                        - reference.charge_state.raw_charges_e
                    )
                )
            ),
            float(
                np.max(
                    np.abs(candidate_state.charges_e - reference.charge_state.charges_e)
                )
            ),
        )
        intrinsic_values = np.asarray(intrinsic.detach().cpu(), dtype=float)
        charge_vjp_values = np.asarray(charge_vjp.detach().cpu(), dtype=float)
        intrinsic_error = float(
            np.max(
                np.abs(
                    intrinsic_values
                    - reference.intrinsic_energy_gradient_ev_per_angstrom
                )
            )
        )
        charge_vjp_error = float(
            np.max(
                np.abs(
                    charge_vjp_values - reference.charge_position_vjp_ev_per_angstrom
                )
            )
        )
        self._validate_parity_errors(
            {
                "energy_absolute_error_eV": energy_error,
                "charge_max_absolute_error_e": charge_error,
                "intrinsic_gradient_max_absolute_error_eV_per_A": intrinsic_error,
                "charge_vjp_max_absolute_error_eV_per_A": charge_vjp_error,
            },
            energy_tolerance_eV=(
                AIMNET_SECOND_ORDER_ENERGY_PARITY_ABSOLUTE_TOLERANCE_EV
            ),
            comparison="public-first-order repeat",
        )
        charge_jvp_values = np.asarray(charge_jvp.detach().cpu(), dtype=float)
        tangent_residual = abs(float(np.sum(charge_jvp_values)))
        if tangent_residual > AIMNET_SECOND_ORDER_CHARGE_TANGENT_TOLERANCE_E_PER_A:
            raise RuntimeError(
                "AIMNet2 charge JVP violates fixed-total-charge tangency: "
                f"{tangent_residual:.6e} e/A."
            )

        result = AIMNet2ChargePositionSecondOrderResponse(
            charge_state=reference.charge_state,
            charge_cotangent_ev_per_e=np.array(cotangent, copy=True),
            coordinate_direction=np.array(direction, copy=True),
            intrinsic_energy_gradient_ev_per_angstrom=(
                reference.intrinsic_energy_gradient_ev_per_angstrom
            ),
            charge_position_vjp_ev_per_angstrom=(
                reference.charge_position_vjp_ev_per_angstrom
            ),
            charge_position_jvp_e_per_angstrom=charge_jvp_values,
            intrinsic_energy_hvp_ev_per_angstrom2=(
                intrinsic_hvp.detach().cpu().numpy()
            ),
            contracted_charge_hessian_ev_per_angstrom2=(
                contracted_charge_hessian.detach().cpu().numpy()
            ),
            standard_decomposed_energy_absolute_error_ev=(
                ordinary_decomposed_parity["energy_absolute_error_eV"]
            ),
            standard_decomposed_charge_max_absolute_error_e=(
                ordinary_decomposed_parity["charge_max_absolute_error_e"]
            ),
            standard_decomposed_intrinsic_gradient_max_absolute_error_ev_per_angstrom=(
                ordinary_decomposed_parity[
                    "intrinsic_gradient_max_absolute_error_eV_per_A"
                ]
            ),
            standard_decomposed_charge_vjp_max_absolute_error_ev_per_angstrom=(
                ordinary_decomposed_parity["charge_vjp_max_absolute_error_eV_per_A"]
            ),
            charge_tangent_residual_e_per_angstrom=tangent_residual,
        )
        self._last_charge_state = reference.charge_state
        return result

    def _analytic_hessian(self, atoms):
        raise NotImplementedError(
            "The float64 research source runtime does not expose a public Hessian."
        )

    def get_hvp(self, atoms, direction):
        raise NotImplementedError(
            "The float64 research source runtime does not expose public HVPs; "
            "Route-2 research code must use charge_position_second_order()."
        )


__all__ = [
    "AIMNET_FLOAT64_RUNTIME_VERSION",
    "AIMNET_FIRST_ORDER_ENERGY_PARITY_ABSOLUTE_TOLERANCE_EV",
    "AIMNET_MODEL_CONFIGURATION",
    "AIMNET_REQUIRED_PACKAGE_VERSION",
    "AIMNET_REQUIRED_FILE_SHA256",
    "AIMNET_SECOND_ORDER_CHARGE_PARITY_ABSOLUTE_TOLERANCE_E",
    "AIMNET_SECOND_ORDER_CHARGE_TANGENT_TOLERANCE_E_PER_A",
    "AIMNET_SECOND_ORDER_CHARGE_VJP_PARITY_ABSOLUTE_TOLERANCE_EV_PER_A",
    "AIMNET_SECOND_ORDER_ENERGY_PARITY_ABSOLUTE_TOLERANCE_EV",
    "AIMNET_SECOND_ORDER_GRADIENT_PARITY_ABSOLUTE_TOLERANCE_EV_PER_A",
    "AIMNet2ChargePositionSecondOrderResponse",
    "AIMNet2ReconstructedFloat64SourceCalculator",
]
