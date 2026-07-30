"""Sealed UMA + AniSolv compact single-point energy calculator.

AniSolv compact supplies a geometry-level, solvent-conditioned correction to a
vacuum potential.  This class pins that composition to the existing molecular
UMA-S-1P2 path rather than exposing an arbitrary ``MLIP + correction``
combination.  It is deliberately not an absolute free-energy protocol and
does not expose AniSolv's orientation-defective upstream force path.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

from ..calculator_base import register_calculator
from ..extra_correction.implicit import AniSolvCompactBackend
from ..model_capabilities import load_model_provenance_card
from ._uma_calculator import UMACalculator

ANISOLV_UMA_BASE_SIZE = "uma-s-1p2"
UMA_S1P2_COMPAT_CHECKPOINT_SHA256 = (
    "41d0354c9d8635859b13b05c719b39a368f158c638c0960380990c8455e094cb"
)
MODEL_CARD_ROOT = Path(__file__).resolve().parents[1] / "model_cards"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_scalar(value, *, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"AniSolv UMA {label} must be a finite scalar.") from exc
    if not math.isfinite(number):
        raise ValueError(f"AniSolv UMA {label} must be a finite scalar.")
    return number


@register_calculator
class AniSolvUMACalculator(UMACalculator):
    """Authorized UMA-S-1P2 plus pinned scalar AniSolv compact correction.

    The exposed energy is a geometry-level solution correction for single
    points only.  The
    parent UMA calculator supplies the gas-phase term and retains its tested
    eV-to-Hartree conversion/composition implementation; this wrapper only
    fixes the allowed base model and attaches the identity-checked solvent
    correction.
    """

    MODEL_NAMES = ("anisolv-uma",)
    SUPPORTED_HESSIAN_MODES = ()
    SUPPORTS_PBC = False
    SUPPORTS_CHARGE_MULT = False
    OPTION_KEYS = (
        "task",
        "size",
        "inference",
        "base_model_path",
        "solvation_model_path",
    )
    MODEL_PATH_OPTION = None

    @classmethod
    def build_kwargs_from_options(cls, model, options, *, resolved_model_path=None):
        del model, resolved_model_path
        return {
            "task": options.get("task"),
            "size": options.get("size"),
            "inference_settings": options.get("inference"),
            "base_model_path": options.get("base_model_path"),
            "solvation_model_path": options.get("solvation_model_path"),
        }

    def __init__(
        self,
        device,
        model: str = "anisolv-uma",
        overrides=None,
        implicit: str = "anisolv",
        solvent: str = "none",
        task=None,
        size=None,
        checkpoint_path=None,
        inference_settings=None,
        base_model_path=None,
        solvation_model_path=None,
        execution_task=None,
    ):
        if str(model).strip().lower() != "anisolv-uma":
            raise ValueError("AniSolv UMA is pinned to model='anisolv-uma'.")
        provenance = load_model_provenance_card("anisolv-uma", MODEL_CARD_ROOT)
        if execution_task is not None:
            provenance.validate_task(execution_task)
        provenance.validate_device(
            device,
            task=execution_task,
            inference_mode=inference_settings,
        )
        self.execution_task = execution_task
        if str(implicit).strip().lower() != "anisolv":
            raise ValueError("AniSolv UMA requires implicit='anisolv'.")
        if checkpoint_path is not None:
            raise ValueError(
                "AniSolv UMA uses a hash-pinned UMA-S-1P2 checkpoint; "
                "custom checkpoint_path is disabled."
            )
        if overrides is not None:
            raise ValueError(
                "AniSolv UMA uses an unmodified hash-pinned UMA-S-1P2 checkpoint; "
                "overrides are disabled."
            )
        if size is not None and str(size).strip().lower() != ANISOLV_UMA_BASE_SIZE:
            raise ValueError("AniSolv UMA is pinned to base size='uma-s-1p2'.")
        if task is not None and str(task).strip().lower() != "omol":
            raise ValueError(
                "AniSolv UMA is molecular and only permits UMA task='omol'."
            )
        if not solvation_model_path:
            raise ValueError(
                "AniSolv UMA requires model option solvation_model_path=<official "
                "model1_compact.pt>."
            )
        if not base_model_path:
            raise ValueError(
                "AniSolv UMA requires model option base_model_path=<verified "
                "UMA-S-1P2 compatibility checkpoint>."
            )
        base_checkpoint = Path(str(base_model_path).strip()).expanduser()
        if not base_checkpoint.is_file():
            raise FileNotFoundError(
                f"AniSolv UMA base model file not found: {base_checkpoint}"
            )
        base_checksum = _sha256(base_checkpoint)
        if base_checksum != UMA_S1P2_COMPAT_CHECKPOINT_SHA256:
            raise ValueError(
                "AniSolv UMA base model checksum mismatch: expected "
                f"{UMA_S1P2_COMPAT_CHECKPOINT_SHA256}, got {base_checksum}."
            )

        super().__init__(
            device=device,
            model="uma",
            overrides=overrides,
            implicit="none",
            solvent="none",
            task="omol",
            size=ANISOLV_UMA_BASE_SIZE,
            checkpoint_path=str(base_checkpoint),
            inference_settings=inference_settings,
        )
        self.solvent_correction = AniSolvCompactBackend(
            model_path=str(solvation_model_path),
            solvent=solvent,
            device=str(self.device),
            execution_task=execution_task,
        )
        self.implemented_properties = [
            property_name
            for property_name in self.implemented_properties
            if property_name in {"energy", "free_energy"}
        ]
        self.chargecalc = None

    def get_hessian(self, atoms, delta: float = 0.002):
        del atoms, delta
        raise NotImplementedError(
            "AniSolv UMA Hessians are disabled because the upstream AniSolv "
            "force path is not rotation-covariant."
        )

    def calculate(self, atoms, properties=None, system_changes=None):
        """Evaluate and annotate the energy-only sealed composition.

        The base calculator may eagerly populate forces even for an energy
        request.  They are removed before returning so ASE cannot later reuse
        an uncorrected base force as a solvent-composite force.
        """

        requested = {
            str(property_name).strip().lower()
            for property_name in (properties or ("energy",))
        }
        if "forces" in requested:
            raise NotImplementedError(
                "AniSolv UMA forces are disabled because the exact upstream "
                "AniSolv force path is not rotation-covariant. Use single-point "
                "energy only."
            )

        correction = self.solvent_correction
        self.solvent_correction = None
        try:
            results = super().calculate(atoms, properties, system_changes)
        finally:
            self.solvent_correction = correction

        try:
            base_energies = {
                name: _finite_scalar(self.results[name], label=f"base {name}")
                for name in ("energy", "free_energy")
                if name in self.results
            }
            evaluated_atoms = (
                atoms if atoms is not None else getattr(self, "atoms", None)
            )
            if evaluated_atoms is None:
                raise ValueError(
                    "AniSolvUMACalculator.calculate requires an Atoms object."
                )
            solvent_result = correction.evaluate(
                evaluated_atoms,
                need_forces=False,
            )
            correction_energy = _finite_scalar(
                solvent_result.energy_hartree,
                label="correction energy",
            )
            components = {
                name: _finite_scalar(value, label=f"correction component {name}")
                for name, value in dict(solvent_result.components_hartree).items()
            }
            combined_energies = {
                name: _finite_scalar(
                    value + correction_energy,
                    label=f"combined {name}",
                )
                for name, value in base_energies.items()
            }

            quantity = "geometry_level_solution_pmf"
            solvation = {
                "energy_hartree": correction_energy,
                "components_hartree": components,
                "provenance": {
                    **dict(solvent_result.provenance),
                    "quantity": quantity,
                    "energy_reference": "relative",
                    "supports_absolute_solvation": False,
                },
                "quantity": quantity,
                "frequency_type": None,
                "forces_exposed": False,
                "ase_free_energy_is_thermochemical_gibbs": False,
            }
            if "energy" in combined_energies:
                combined = combined_energies["energy"]
                solvation["solution_pmf_correction_hartree"] = correction_energy
                solvation["gas_energy_hartree"] = _finite_scalar(
                    combined - correction_energy,
                    label="reconstructed gas energy",
                )
                solvation["combined_energy_hartree"] = combined
        except Exception:
            self.results = {}
            raise

        self.results.pop("forces", None)
        self.results.update(combined_energies)
        self.results["solvation"] = solvation
        return results
