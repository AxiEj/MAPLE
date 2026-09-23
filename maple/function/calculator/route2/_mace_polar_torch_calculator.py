"""ASE bridge for the explicit full-Torch analytic MACE-POLAR v3 scalar."""

from __future__ import annotations

import numpy as np
from ase.calculators.calculator import Calculator, all_changes

from maple.function.calculator.extra_correction.implicit.route2_domain import (
    validate_route2_domain,
)
from maple.function.route2_smd_profiles import (
    PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_TORCH_CPU_V3_PROFILE,
    PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_TORCH_CUDA_V3_PROFILE,
    Route2SMDProfileSpec,
    pure_macepolar_expected_device,
    route2_smd_profile_spec,
)
from maple.function.route2_solvents import normalize_route2_solvent_name
from maple.solvation.models.base import model_input_sha256

_TEST_PES_FACTORY_TOKEN = object()
_TORCH_PROFILE_NAMES = frozenset(
    {
        PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_TORCH_CPU_V3_PROFILE,
        PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_TORCH_CUDA_V3_PROFILE,
    }
)


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _normalize_torch_device(device: object) -> str:
    text = str(device).strip().lower()
    if text == "cpu":
        return text
    if text == "cuda":
        return "cuda:0"
    if text.startswith("cuda:") and text[5:].isdigit():
        return f"cuda:{int(text[5:])}"
    raise ValueError("Pure Torch MACE-POLAR device must be cpu, cuda, or cuda:N.")


def _validate_cuda_device(device: str) -> None:
    if not device.startswith("cuda:"):
        return
    import torch

    if not torch.cuda.is_available():
        raise ValueError(f"Requested {device}, but CUDA is not available.")
    index = int(device.split(":", 1)[1])
    count = int(torch.cuda.device_count())
    if index >= count:
        raise ValueError(
            f"Requested {device}, but only {count} CUDA device(s) are available."
        )


class PureMACEPolarTorchCalculator(Calculator):
    """Expose the v3 single-graph Torch scalar through ASE's eV contract.

    This facade is intentionally separate from the legacy pyddx/PySCF wrapper.
    It never constructs a Richardson derivative backend and has no fallback to
    legacy continuum or CDS kernels.
    """

    implemented_properties = ["energy", "free_energy", "forces"]
    workflow_kind = "pure-torch-analytic-total-pes"
    MODEL_ENERGY_UNIT = "eV"
    SUPPORTS_PBC = False
    SUPPORTS_CHARGE_MULT = False
    SUPPORTED_HESSIAN_MODES = ("analytic",)

    def __init__(
        self,
        *,
        atoms,
        solvent: str,
        device: object = "cpu",
        model: str = "macepolm",
        profile_spec: Route2SMDProfileSpec,
        _testing_pes: object | None = None,
        _testing_token: object | None = None,
    ) -> None:
        super().__init__()
        normalized_device = _normalize_torch_device(device)
        if str(model).strip().lower() != "macepolm":
            raise ValueError("The pure Torch workflow requires model=macepolm.")
        if (
            not isinstance(profile_spec, Route2SMDProfileSpec)
            or profile_spec.name not in _TORCH_PROFILE_NAMES
            or profile_spec is not route2_smd_profile_spec(profile_spec.name)
            or profile_spec.provider != "torch"
            or profile_spec.execution_route != self.workflow_kind
            or profile_spec.allowed_response_modes != ("frozen",)
        ):
            raise ValueError("profile_spec is not a registered pure Torch v3 profile.")
        expected_device = pure_macepolar_expected_device(profile_spec)
        if (expected_device == "cpu" and normalized_device != "cpu") or (
            expected_device == "cuda" and not normalized_device.startswith("cuda:")
        ):
            raise ValueError(
                f"Route 2 profile={profile_spec.name} requires device={expected_device}."
            )
        _validate_cuda_device(normalized_device)

        normalized_solvent = normalize_route2_solvent_name(solvent)
        if not profile_spec.supports_solvent(normalized_solvent):
            raise ValueError(
                f"Route 2 profile={profile_spec.name} does not support solvent={solvent}."
            )
        validate_route2_domain(atoms)
        if _testing_pes is not None and _testing_token is not _TEST_PES_FACTORY_TOKEN:
            raise ValueError("test PES injection is not a public construction path.")
        if _testing_pes is None:
            from maple.solvation.experimental.mace_polar_torch import (
                build_smd_mace_polar_torch_pes,
            )

            pes = build_smd_mace_polar_torch_pes(
                tuple(atoms.get_chemical_symbols()),
                solvent=normalized_solvent,
                device=normalized_device,
            )
        else:
            pes = _testing_pes
        if getattr(pes, "scalar_contract_id", None) != profile_spec.scalar_contract_id:
            raise ValueError(
                "Torch PES does not implement the profile scalar contract."
            )
        pes_device = getattr(pes, "device", None)
        if (
            pes_device is not None
            and _normalize_torch_device(pes_device) != normalized_device
        ):
            raise ValueError(
                "Torch PES device does not match the requested profile device."
            )

        self._device = normalized_device
        self._model_name = "macepolm"
        self._solvent = normalized_solvent
        self._profile_spec = profile_spec
        self._pes = pes
        self.solvent_correction = None
        self.chargecalc = None
        self._configured_symbols = tuple(atoms.get_chemical_symbols())
        self._cache_key: tuple[str, str, str, str, str] | None = None
        self._hessian_cache_key: tuple[str, str, str, str, str] | None = None
        self._hessian_evaluation: object | None = None

    @classmethod
    def _from_test_pes(
        cls,
        *,
        atoms,
        solvent: str,
        pes: object,
        device: object = "cpu",
        model: str = "macepolm",
        profile_spec: Route2SMDProfileSpec,
    ) -> "PureMACEPolarTorchCalculator":
        """Build an engineering-test facade without loading the real model."""

        return cls(
            atoms=atoms,
            solvent=solvent,
            device=device,
            model=model,
            profile_spec=profile_spec,
            _testing_pes=pes,
            _testing_token=_TEST_PES_FACTORY_TOKEN,
        )

    @property
    def device(self) -> str:
        return self._device

    @property
    def model(self) -> str:
        return self._model_name

    @property
    def solvent(self) -> str:
        return self._solvent

    @property
    def profile_spec(self) -> Route2SMDProfileSpec:
        return self._profile_spec

    @property
    def pes(self) -> object:
        return self._pes

    @property
    def last_hessian_evaluation(self) -> object | None:
        return self._hessian_evaluation

    def _validate_atoms(self, atoms) -> tuple[str, str, str, str, str]:
        expected_device = pure_macepolar_expected_device(self._profile_spec)
        if (
            self._profile_spec.provider != "torch"
            or self._profile_spec.execution_route != self.workflow_kind
            or self._profile_spec.allowed_response_modes != ("frozen",)
            or getattr(self._pes, "scalar_contract_id", None)
            != self._profile_spec.scalar_contract_id
            or (expected_device == "cpu" and self._device != "cpu")
            or (expected_device == "cuda" and not self._device.startswith("cuda:"))
        ):
            raise RuntimeError("pure Torch workflow configuration drifted.")
        pes_device = getattr(self._pes, "device", None)
        if (
            pes_device is not None
            and _normalize_torch_device(pes_device) != self._device
        ):
            raise RuntimeError("pure Torch PES device drifted.")
        validate_route2_domain(atoms)
        if tuple(atoms.get_chemical_symbols()) != self._configured_symbols:
            raise ValueError("atoms differ from the calculator's configured symbols.")
        configuration = getattr(self._pes, "configuration_sha256", None)
        configuration_sha256 = (
            str(configuration())
            if callable(configuration)
            else str(configuration or "")
        )
        if not _is_sha256(configuration_sha256):
            raise RuntimeError("pure Torch PES configuration identity is not SHA256.")
        return (
            model_input_sha256(atoms),
            configuration_sha256,
            self._profile_spec.name,
            str(self._profile_spec.scalar_contract_id),
            self._device,
        )

    @staticmethod
    def _reject_hessian_constraints(atoms) -> None:
        if tuple(getattr(atoms, "constraints", ()) or ()):
            raise ValueError(
                "The pure Torch workflow does not support constrained Hessians."
            )

    def get_property(self, name, atoms=None, allow_calculation=True):
        target = atoms if atoms is not None else self.atoms
        if target is not None:
            self._validate_atoms(target)
        return super().get_property(name, atoms, allow_calculation)

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        target = atoms if atoms is not None else self.atoms
        if target is None:
            raise ValueError("PureMACEPolarTorchCalculator requires atoms.")
        cache_key = self._validate_atoms(target)
        requested = frozenset(properties or self.implemented_properties)
        super().calculate(target, properties, system_changes)

        energy = float(self._pes.get_potential_energy(target))
        if not np.isfinite(energy):
            raise RuntimeError("pure Torch total-PES energy is invalid.")
        results: dict[str, object] = {"energy": energy, "free_energy": energy}
        if "forces" in requested:
            forces = np.asarray(self._pes.get_forces(target), dtype=float)
            if forces.shape != (len(target), 3) or not np.all(np.isfinite(forces)):
                raise RuntimeError("pure Torch total-PES forces are invalid.")
            results["forces"] = np.array(forces, copy=True)
        self.results = results
        self._cache_key = cache_key
        if cache_key != self._hessian_cache_key:
            self._hessian_cache_key = None
            self._hessian_evaluation = None

    def get_hessian_evaluation(self, atoms):
        self._reject_hessian_constraints(atoms)
        cache_key = self._validate_atoms(atoms)
        if (
            cache_key == self._hessian_cache_key
            and self._hessian_evaluation is not None
        ):
            return self._hessian_evaluation
        evaluation = self._pes.evaluate_hessian(atoms)
        if getattr(evaluation, "derivative_method", None) != "torch-autograd":
            raise RuntimeError("pure Torch PES returned a non-analytic Hessian result.")
        hessian = np.asarray(getattr(evaluation, "hessian_eV_per_A2"), dtype=float)
        if hessian.shape != (3 * len(atoms), 3 * len(atoms)) or not np.all(
            np.isfinite(hessian)
        ):
            raise RuntimeError("pure Torch total-PES Hessian is invalid.")
        self._hessian_cache_key = cache_key
        self._hessian_evaluation = evaluation
        return evaluation

    def get_hessian(self, atoms, delta: float | None = None) -> np.ndarray:
        if delta is not None:
            raise ValueError(
                "The analytic Torch workflow does not accept a finite-difference step."
            )
        evaluation = self.get_hessian_evaluation(atoms)
        return np.array(evaluation.hessian_eV_per_A2, dtype=float, copy=True)

    def get_hvp(self, atoms, n: np.ndarray):
        self._reject_hessian_constraints(atoms)
        self._validate_atoms(atoms)
        direction = np.asarray(n, dtype=float)
        if direction.size != 3 * len(atoms) or not np.all(np.isfinite(direction)):
            raise ValueError("HVP direction must contain 3N finite components.")
        method = getattr(self._pes, "hessian_vector_product", None)
        if not callable(method):
            raise RuntimeError("pure Torch PES does not expose analytic HVP.")
        hvp = np.asarray(method(atoms, direction.reshape(len(atoms), 3)), dtype=float)
        if hvp.shape != (len(atoms), 3) or not np.all(np.isfinite(hvp)):
            raise RuntimeError("pure Torch total-PES HVP is invalid.")
        forces = np.asarray(self._pes.get_forces(atoms), dtype=float)
        energy = float(self._pes.get_potential_energy(atoms))
        if forces.shape != (len(atoms), 3) or not np.all(np.isfinite(forces)):
            raise RuntimeError("pure Torch total-PES forces are invalid.")
        if not np.isfinite(energy):
            raise RuntimeError("pure Torch total-PES energy is invalid.")
        try:
            import torch

            return (
                torch.as_tensor(hvp.reshape(-1), dtype=torch.float64),
                torch.as_tensor(forces.reshape(-1), dtype=torch.float64),
                torch.as_tensor(energy, dtype=torch.float64),
            )
        except ImportError:  # pragma: no cover - Torch is a v3 prerequisite
            return hvp.reshape(-1), forces.reshape(-1), energy


def is_pure_mace_polar_torch_calculator(calculator: object) -> bool:
    """Return whether *calculator* is the exact canonical v3 Torch facade."""

    from maple.function.dispatcher.legacy_units import LegacyHartreeJobView

    candidate = calculator
    if type(candidate) is LegacyHartreeJobView:
        candidate = candidate.raw_calculator
    if type(candidate) is not PureMACEPolarTorchCalculator:
        return False
    spec = getattr(candidate, "profile_spec", None)
    if (
        not isinstance(spec, Route2SMDProfileSpec)
        or spec.name not in _TORCH_PROFILE_NAMES
    ):
        return False
    canonical = route2_smd_profile_spec(spec.name)
    expected_device = pure_macepolar_expected_device(canonical)
    return bool(
        spec is canonical
        and canonical.provider == "torch"
        and canonical.execution_route == candidate.workflow_kind
        and canonical.allowed_response_modes == ("frozen",)
        and (
            candidate.device == "cpu"
            if expected_device == "cpu"
            else candidate.device.startswith("cuda:")
        )
        and candidate.model == "macepolm"
        and canonical.supports_solvent(candidate.solvent)
        and getattr(candidate.pes, "scalar_contract_id", None)
        == canonical.scalar_contract_id
    )


__all__ = [
    "PureMACEPolarTorchCalculator",
    "is_pure_mace_polar_torch_calculator",
]
