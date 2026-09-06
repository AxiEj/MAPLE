"""MAPLE/ASE entry point for the admitted hybrid harmonic E/F profile.

The calculator intentionally exposes only the two capabilities admitted by the
registered profile: the operational electrostatic energy and its guarded
same-scalar implicit-adjoint force.  It does not advertise Hessian, stress,
virial, frequency, MD, finite-dielectric, or complete-solvation capability.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from ase.calculators.calculator import all_changes

from ..calculator_base import CalcABC, register_calculator


@register_calculator
class MACE_MDPPOLARHybridCalculator(CalcABC):
    """ASE calculator for the admitted MACE-MDP + MACE-POLAR harmonic PES."""

    implemented_properties = ["energy", "forces", "free_energy"]

    MODEL_NAMES = ("macemdppolarhybrid",)
    MODEL_ENERGY_UNIT = "eV"
    SUPPORTED_HESSIAN_MODES: tuple[str, ...] = ()
    SUPPORTS_CHARGE_MULT = False
    SUPPORTS_PBC = False
    CHECKPOINT_FILENAME = None
    REQUIRES_LOCAL_MODEL_FILE = False
    OPTION_KEYS = ("mdp_checkpoint_path", "polar_checkpoint_path")
    MAPLE_DAILY_JOB_CAPABILITIES = ("sp", "opt:first-order")

    def validate_maple_job(self, *, jobtype: object, params: object = None) -> None:
        """Fail closed outside the MAPLE jobs supported by admitted E/F."""

        normalized_job = str(jobtype).strip().lower()
        options = params if isinstance(params, dict) else {}
        if normalized_job == "sp":
            return
        if normalized_job == "opt":
            method = str(options.get("method") or "lbfgs").strip().lower()
            if method in {"lbfgs", "sd", "sdcg", "cg"}:
                return
            raise NotImplementedError(
                "macemdppolarhybrid OPT supports only first-order "
                "LBFGS/SD/SDCG/CG methods; RFO requires H and remains closed."
            )
        raise NotImplementedError(
            "macemdppolarhybrid currently supports MAPLE SP and first-order OPT "
            "only; FREQ/TS/IRC/MD and scan workflows remain closed."
        )

    @classmethod
    def build_kwargs_from_options(
        cls, model: str, model_options: dict[str, Any], *, resolved_model_path=None
    ) -> dict[str, Any]:
        del model, resolved_model_path
        result: dict[str, Any] = {}
        for name in cls.OPTION_KEYS:
            value = model_options.get(name)
            if value is not None:
                result[name] = value
        return result

    @staticmethod
    def _load_hybrid(
        *, device: str, mdp_checkpoint_path: Path, polar_checkpoint_path: Path
    ) -> tuple[object, object]:
        from maple.solvation.models import (
            MACEPolarOriginalSourceNativeFieldAdapter,
            build_mace_mdp_anchored_mace_polar_hybrid,
            build_mace_mdp_moment_adapter,
            build_official_mace_polar_1_m_radial_gto_adapter,
        )
        from maple.solvation.models.runtime.analytic_gaussian_multipole import (
            MACE_POLAR_ANALYTIC_GAUSSIAN_MULTIPOLE_EVALUATOR_ID,
        )

        mdp = build_mace_mdp_moment_adapter(
            checkpoint_path=mdp_checkpoint_path,
            device="cpu",
        )
        radial = build_official_mace_polar_1_m_radial_gto_adapter(
            checkpoint_path=polar_checkpoint_path,
            device=device,
            long_range_evaluator_profile=(
                MACE_POLAR_ANALYTIC_GAUSSIAN_MULTIPOLE_EVALUATOR_ID
            ),
        )
        hybrid = build_mace_mdp_anchored_mace_polar_hybrid(
            permanent=mdp,
            response=MACEPolarOriginalSourceNativeFieldAdapter(radial),
        )
        return hybrid, radial

    def __init__(
        self,
        device: object,
        model: str = "macemdppolarhybrid",
        implicit: str = "none",
        solvent: str = "none",
        mdp_checkpoint_path: str | Path | None = None,
        polar_checkpoint_path: str | Path | None = None,
    ) -> None:
        super().__init__()
        normalized_device = str(device)
        if normalized_device != "cuda":
            raise ValueError(
                "The admitted hybrid harmonic E/F profile requires device='cuda'."
            )
        if str(model).strip().lower() not in self.MODEL_NAMES:
            raise ValueError("Unsupported hybrid calculator model identity.")
        if str(implicit).strip().lower() not in {"", "none"}:
            raise ValueError(
                "macemdppolarhybrid already owns its Route-2 continuum; "
                "set implicit='none'."
            )
        if str(solvent).strip().lower() not in {"", "none", "water"}:
            raise ValueError(
                "The admitted hybrid harmonic profile is restricted to water-cavity "
                "conductor-limit electrostatics."
            )

        mdp_path = (
            Path(mdp_checkpoint_path or Path.home() / ".cache/mace/MACE-MDP.model")
            .expanduser()
            .resolve(strict=True)
        )
        polar_path = (
            Path(polar_checkpoint_path or Path.home() / ".cache/mace/MACEPOLAR1Mmodel")
            .expanduser()
            .resolve(strict=True)
        )
        hybrid, radial = self._load_hybrid(
            device=normalized_device,
            mdp_checkpoint_path=mdp_path,
            polar_checkpoint_path=polar_path,
        )

        self.device = normalized_device
        self.model_name = self.MODEL_NAMES[0]
        self.solvent = "water"
        self.solvent_correction = None
        self._hybrid = hybrid
        self._radial = radial
        self._pes = None
        self._atomic_numbers: tuple[int, ...] | None = None
        self.route2_result = None

    def _build_pes(self, atoms: object):
        from maple.function.calculator.extra_correction.implicit.smd_cds import (
            smd_water_coulomb_radii,
        )
        from maple.solvation.experimental import (
            MACE_MDPPolarHybridSmoothHarmonicPES,
        )

        numbers = tuple(int(value) for value in atoms.get_atomic_numbers())
        symbols = tuple(str(value) for value in atoms.get_chemical_symbols())
        return MACE_MDPPolarHybridSmoothHarmonicPES(
            hybrid=self._hybrid,
            atomic_numbers=numbers,
            cavity_radii_angstrom=smd_water_coulomb_radii(symbols),
            dtype=self._radial.dtype,
            device=self._radial.device,
        )

    def _pes_for(self, atoms: object):
        numbers = tuple(int(value) for value in atoms.get_atomic_numbers())
        if self._pes is None or self._atomic_numbers != numbers:
            self._pes = self._build_pes(atoms)
            self._atomic_numbers = numbers
        return self._pes

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        requested = self._normalize_properties(properties)
        target = super().calculate(atoms, requested, system_changes)
        unsupported = sorted(
            set(str(value).lower() for value in requested)
            - {"energy", "free_energy", "forces"}
        )
        if unsupported:
            raise NotImplementedError(
                "The admitted hybrid harmonic calculator exposes only E/F; "
                f"unsupported properties: {', '.join(unsupported)}."
            )

        need_forces = "forces" in requested
        result = self._pes_for(target).evaluate(target, need_forces=need_forces)
        forces = None
        if need_forces:
            values = result.total_forces_eV_per_A
            if values is None:
                raise RuntimeError("Admitted hybrid result omitted its force leaves.")
            forces = np.asarray(values, dtype=float)

        self._finalize_results(
            target,
            energy=result.total_energy_eV,
            forces=forces,
            unit="eV",
        )
        self.route2_result = result
        self.results["route2"] = {
            "profile_id": result.profile_id,
            "scalar_id": result.scalar_id,
            "state_equation_id": result.state_equation_id,
            "root_sha256": result.root_sha256,
            "primal_residual": result.primal_residual,
            "adjoint_residual": result.adjoint_residual,
            "force_derivative_kind": self._pes_for(target).force_derivative_kind,
            "force_error_estimate_eV_per_A": (result.force_error_estimate_eV_per_A),
            "force_components": tuple(item.name for item in result.force_components),
            "energy_components_eV": {
                item.name: item.value_eV for item in result.energy_components
            },
            "warnings": tuple(result.warnings),
        }

    def get_hessian(self, atoms, delta: float = 0.002):
        del atoms, delta
        raise NotImplementedError(
            "Hybrid harmonic H is not admitted; the first-order analytic force "
            "does not by itself admit a daily Hessian."
        )
