"""ASE calculator for the exact local MACE-POLAR-EF-v2 TorchScript asset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from ase.calculators.calculator import all_changes

from ...route2_model_contracts import ROUTE2_MACE_POLAR_EF_V2_PROFILE_BINDING
from ...route2_smd_profiles import validate_route2_smd_profile
from ..calculator_base import (
    CalcABC,
    _IMPLICIT_SOLVENT_FACTORY_TOKEN,
    register_calculator,
)
from ..extra_correction.implicit.mace_polar_ef import (
    MACEPolarEFConfig,
    MACEPolarEFEnergyModel,
)
from ..extra_correction.implicit.mace_polar_ef_specs import (
    MACE_POLAR_EF_V2_CHECKPOINT_SPEC,
    MACEPolarEFCheckpointSpec,
)


@dataclass(frozen=True, slots=True)
class _GasState:
    atomic_numbers: np.ndarray
    positions_angstrom: np.ndarray
    energy_ev: float
    coordinate_gradient_ev_per_angstrom: np.ndarray | None

    def __post_init__(self) -> None:
        numbers = np.array(self.atomic_numbers, dtype=int, copy=True)
        positions = np.array(self.positions_angstrom, dtype=float, copy=True)
        if numbers.ndim != 1 or positions.shape != (len(numbers), 3):
            raise ValueError("MACE-POLAR-EF gas-state geometry is malformed.")
        if not np.all(np.isfinite(positions)) or not np.isfinite(self.energy_ev):
            raise ValueError("MACE-POLAR-EF gas state must be finite.")
        numbers.setflags(write=False)
        positions.setflags(write=False)
        object.__setattr__(self, "atomic_numbers", numbers)
        object.__setattr__(self, "positions_angstrom", positions)
        object.__setattr__(self, "energy_ev", float(self.energy_ev))
        gradient = self.coordinate_gradient_ev_per_angstrom
        if gradient is not None:
            values = np.array(gradient, dtype=float, copy=True)
            if values.shape != positions.shape or not np.all(np.isfinite(values)):
                raise ValueError("MACE-POLAR-EF gas gradient is malformed.")
            values.setflags(write=False)
            object.__setattr__(
                self,
                "coordinate_gradient_ev_per_angstrom",
                values,
            )

    def matches(self, atoms, *, require_gradient: bool) -> bool:
        return (
            np.array_equal(self.atomic_numbers, np.asarray(atoms.numbers, dtype=int))
            and np.array_equal(
                self.positions_angstrom,
                np.asarray(atoms.get_positions(), dtype=float),
            )
            and (
                not require_gradient
                or self.coordinate_gradient_ev_per_angstrom is not None
            )
        )


@register_calculator
class MACEPolarEFCalculator(CalcABC):
    """Reusable atomwise-potential MACE-EF calculator.

    Compatible checkpoint families may subclass this class and override the
    registry-facing class attributes without copying inference or solvent code.
    """

    implemented_properties = ["energy", "forces", "free_energy"]
    MODEL_NAMES = ("macepolarefv2",)
    MODEL_ENERGY_UNIT = "eV"
    SUPPORTED_HESSIAN_MODES = ("numerical",)
    SUPPORTS_CHARGE_MULT = True
    SUPPORTS_PBC = False
    CHECKPOINT_FILENAME = None
    REQUIRES_LOCAL_MODEL_FILE = False
    OPTION_KEYS = ()
    MODEL_PATH_OPTION = "model_path"
    CHECKPOINT_SPEC: MACEPolarEFCheckpointSpec = MACE_POLAR_EF_V2_CHECKPOINT_SPEC
    ROUTE2_PROFILE_BINDING = ROUTE2_MACE_POLAR_EF_V2_PROFILE_BINDING
    SUPPORTED_TOTAL_CHARGES = (0,)
    SUPPORTED_SPIN_MULTIPLICITIES = (1,)

    @classmethod
    def build_kwargs_from_options(
        cls,
        model,
        model_options,
        *,
        resolved_model_path=None,
    ):
        del model, model_options
        if resolved_model_path is None:
            raise ValueError("MACE-POLAR-EF-v2 requires an explicit model_path.")
        return {"model_path": resolved_model_path}

    @classmethod
    def build_implicit_solvent_kwargs(cls, solvation_options):
        provider = str(solvation_options.get("provider", "")).lower()
        profile = str(solvation_options.get("profile", "")).lower()
        spec = validate_route2_smd_profile(provider, profile)
        if (
            not spec.known_nonpassive_diagnostic
            or spec.electronic_profile_binding != cls.ROUTE2_PROFILE_BINDING
            or spec.electronic_model_family != cls.CHECKPOINT_SPEC.model_family
        ):
            raise ValueError(
                "MACE-POLAR-EF-v2 is exposed only through its exact "
                "known-nonpassive smooth-PCM diagnostic profile."
            )
        return {}

    def __init__(
        self,
        device: torch.device | str,
        model: str = "macepolarefv2",
        model_path: str | None = None,
        implicit: Literal["smd", "cosmo", "none"] = "none",
        solvent: str = "none",
        _implicit_solvent_factory_token=None,
    ) -> None:
        super().__init__()
        del model
        if model_path is None:
            raise ValueError("MACE-POLAR-EF-v2 requires an explicit model_path.")
        path = Path(model_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        device_value = torch.device(device)
        checkpoint_spec = type(self).CHECKPOINT_SPEC
        if device_value.type != checkpoint_spec.device_type:
            raise ValueError(
                f"{checkpoint_spec.name} requires device type "
                f"{checkpoint_spec.device_type}."
            )
        if device_value.index is None:
            device_value = torch.device(
                checkpoint_spec.device_type,
                checkpoint_spec.device_index,
            )
        if device_value.index != checkpoint_spec.device_index:
            raise ValueError(
                f"{checkpoint_spec.name} requires device index "
                f"{checkpoint_spec.device_index}."
            )
        route2_continuum = str(implicit).strip().lower() in {"smd", "cosmo"}
        if route2_continuum and (
            _implicit_solvent_factory_token is not _IMPLICIT_SOLVENT_FACTORY_TOKEN
        ):
            raise ValueError(
                "Direct MACEPolarEFCalculator implicit-solvent construction "
                "is disabled; use MAPLE's SetCalculator factory."
            )

        self.device = device_value
        self.model_path = str(path)
        self.route2_smd_profile = (
            type(self).ROUTE2_PROFILE_BINDING if route2_continuum else None
        )
        self._evaluator: MACEPolarEFEnergyModel | None = None
        self._atomic_numbers: tuple[int, ...] | None = None
        self._last_gas_state: _GasState | None = None
        self.implicit_solv_init(implicit=implicit, solvent=solvent)

    @classmethod
    def _state_metadata(cls, atoms) -> tuple[int, int]:
        try:
            charge = int(atoms.info.get("charge", 0))
            multiplicity = int(atoms.info.get("mult", 1))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "MACE-POLAR-EF-v2 requires integer charge and multiplicity."
            ) from exc
        if (
            charge not in cls.SUPPORTED_TOTAL_CHARGES
            or multiplicity not in cls.SUPPORTED_SPIN_MULTIPLICITIES
        ):
            raise ValueError(
                f"The exposed {cls.CHECKPOINT_SPEC.name} adapter supports "
                f"charges={cls.SUPPORTED_TOTAL_CHARGES} and "
                "spin multiplicities="
                f"{cls.SUPPORTED_SPIN_MULTIPLICITIES}."
            )
        return charge, multiplicity

    def mace_polar_ef_evaluator(self, atoms) -> MACEPolarEFEnergyModel:
        numbers = tuple(int(value) for value in atoms.numbers)
        charge, multiplicity = self._state_metadata(atoms)
        if self._evaluator is None:
            config = MACEPolarEFConfig(
                checkpoint_path=self.model_path,
                atomic_numbers=numbers,
                checkpoint_spec=type(self).CHECKPOINT_SPEC,
                total_charge=charge,
                spin_multiplicity=multiplicity,
                device=str(self.device),
            )
            self._evaluator = MACEPolarEFEnergyModel(config)
            self._atomic_numbers = numbers
        elif numbers != self._atomic_numbers:
            raise ValueError(
                "MACE-POLAR-EF-v2 calculator does not permit atom identity/order changes."
            )
        return self._evaluator

    def _gas_state(self, atoms, *, require_gradient: bool) -> _GasState:
        cached = self._last_gas_state
        if cached is not None and cached.matches(
            atoms,
            require_gradient=require_gradient,
        ):
            return cached
        evaluator = self.mace_polar_ef_evaluator(atoms)
        positions = torch.tensor(
            np.asarray(atoms.get_positions(), dtype=np.float32),
            dtype=torch.float32,
            device=evaluator.device,
            requires_grad=require_gradient,
        )
        field = torch.zeros(
            (len(atoms), 4),
            dtype=torch.float32,
            device=evaluator.device,
        )
        energy, _node_energy, _density = evaluator.energy_torch(positions, field)
        gradient = None
        if require_gradient:
            (gradient_tensor,) = torch.autograd.grad(energy, positions)
            gradient = np.asarray(gradient_tensor.detach().cpu(), dtype=float)
        state = _GasState(
            atomic_numbers=np.asarray(atoms.numbers, dtype=int),
            positions_angstrom=np.asarray(atoms.get_positions(), dtype=float),
            energy_ev=float(energy.detach().cpu()),
            coordinate_gradient_ev_per_angstrom=gradient,
        )
        self._last_gas_state = state
        return state

    def mace_polar_ef_gas_energy_ev(self, atoms) -> float:
        return self._gas_state(atoms, require_gradient=False).energy_ev

    def mace_polar_ef_gas_gradient_ev_per_angstrom(
        self,
        atoms,
    ) -> np.ndarray:
        gradient = self._gas_state(
            atoms,
            require_gradient=True,
        ).coordinate_gradient_ev_per_angstrom
        if gradient is None:
            raise RuntimeError("MACE-POLAR-EF gas gradient is unavailable.")
        return np.array(gradient, dtype=float, copy=True)

    def calculate(
        self,
        atoms=None,
        properties=("energy",),
        system_changes=all_changes,
    ) -> None:
        properties = self._normalize_properties(properties)
        atoms = super().calculate(atoms, properties, system_changes)
        needs_forces = "forces" in properties
        state = self._gas_state(atoms, require_gradient=needs_forces)
        forces = None
        if needs_forces:
            gradient = state.coordinate_gradient_ev_per_angstrom
            if gradient is None:
                raise RuntimeError("MACE-POLAR-EF gas gradient is unavailable.")
            forces = -gradient
        self._finalize_results(
            atoms,
            energy=state.energy_ev,
            forces=forces,
        )


__all__ = ["MACEPolarEFCalculator"]
