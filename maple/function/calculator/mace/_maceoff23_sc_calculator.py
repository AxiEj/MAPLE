"""Pinned CPU-default adapter for the public MACE-OFF23-SC checkpoint.

This adapter exposes the official full-coupling potential-energy surface for
explicit systems.  It does not reproduce the separate ``mace-md`` alchemical
sampling protocol and therefore cannot emit an absolute solvation free energy.
The official CUDA runtime is executable; production CUDA selection remains
fail-closed until a broad experimental no-accuracy-degradation gate passes.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Callable

import numpy as np
from ase.calculators.calculator import all_changes

from ..calculator_base import CalcABC, register_calculator
from ..model_capabilities import ModelProvenanceCard

MACE_OFF23_SC_SHA256 = (
    "32c9fb51704f96da855c67e0cdc9894f3e41694e98b7a0ed8813388b9f21db33"
)
MACE_OFF23_SC_OFFICIAL_IDENTITY = {
    "model_id": "mace-off23-sc",
    "version": "MACE-OFF23-SC_swa",
    "source_revision": "efbb20c930462bc37a247998a27814dff35d474f",
    "protocol_revision": "c3056287622ba18f9b905e9df45affdff46cb147",
}
MACE_OFF23_SC_CARD = (
    Path(__file__).resolve().parents[1] / "model_cards" / "mace-off23-sc.yaml"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_card(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"MACE-OFF23-SC model card must use JSON-compatible YAML syntax: {path}"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError("MACE-OFF23-SC model card must be a JSON object.")
    return payload


def _validate_official_card(card: dict) -> ModelProvenanceCard:
    identity = MACE_OFF23_SC_OFFICIAL_IDENTITY
    if str(card.get("model_id", "")).strip().lower() != identity["model_id"]:
        raise ValueError(
            "MACE-OFF23-SC model card identity mismatch: this adapter is pinned "
            f"to {identity['model_id']!r}."
        )
    if str(card.get("version", "")).strip() != identity["version"]:
        raise ValueError(
            "MACE-OFF23-SC model card version does not match the pinned checkpoint."
        )
    source = card.get("source")
    if not isinstance(source, dict):
        raise ValueError("MACE-OFF23-SC model card must define source metadata.")
    if str(source.get("revision", "")).strip() != identity["source_revision"]:
        raise ValueError(
            "MACE-OFF23-SC model card revision does not match the pinned source."
        )
    protocol = card.get("protocol_source")
    if not isinstance(protocol, dict):
        raise ValueError(
            "MACE-OFF23-SC model card must define protocol-source metadata."
        )
    if str(protocol.get("revision", "")).strip() != identity["protocol_revision"]:
        raise ValueError(
            "MACE-OFF23-SC protocol revision does not match the audited source."
        )
    if str(card.get("checkpoint_sha256", "")).strip().lower() != (MACE_OFF23_SC_SHA256):
        raise ValueError(
            "MACE-OFF23-SC model card checksum does not match the pinned "
            f"official digest {MACE_OFF23_SC_SHA256}."
        )
    elements = card.get("elements")
    if not isinstance(elements, list) or not elements:
        raise ValueError("MACE-OFF23-SC model card must define supported elements.")

    provenance = ModelProvenanceCard.from_payload(
        card,
        model_name=identity["model_id"],
    )
    capabilities = provenance.capabilities
    if (
        not capabilities.energy
        or not capabilities.forces
        or not capabilities.conservative_forces
        or not capabilities.supports_md
        or not capabilities.supports_pbc
        or not capabilities.supports_multifragment
        or capabilities.requires_topology
        or capabilities.hessian != "none"
        or capabilities.solvation_mode != "none"
        or capabilities.supports_absolute_solvation
        or capabilities.supports_alchemical_lambda
    ):
        raise ValueError(
            "MACE-OFF23-SC card must describe the full-coupling explicit-system "
            "PES adapter, not the unavailable free-energy protocol."
        )
    return provenance


def _official_calculator_factory(model_path: str, device: str):
    try:
        from mace.calculators import MACECalculator
    except ImportError as exc:
        raise ImportError(
            "MACE-OFF23-SC requires the optional official mace-torch runtime."
        ) from exc
    return MACECalculator(
        model_paths=model_path,
        device=device,
        default_dtype="float64",
    )


@register_calculator
class MACEOFF23SCCalculator(CalcABC):
    """Official MACE-OFF23-SC full-coupling explicit-system PES."""

    implemented_properties = ["energy", "forces", "free_energy"]
    MODEL_NAMES = ("mace-off23-sc",)
    MODEL_ENERGY_UNIT = "eV"
    SUPPORTED_HESSIAN_MODES = ()
    SUPPORTS_CHARGE_MULT = False
    SUPPORTS_PBC = True
    OPTION_KEYS = ()
    MODEL_PATH_OPTION = "model_path"

    @classmethod
    def build_kwargs_from_options(
        cls,
        model,
        options,
        *,
        resolved_model_path=None,
    ):
        if options.get("d4", False):
            raise ValueError(
                "MACE-OFF23-SC does not support D4; refusing to ignore the "
                "requested dispersion correction."
            )
        del model, options
        kwargs = {}
        if resolved_model_path is not None:
            kwargs["model_path"] = resolved_model_path
        return kwargs

    def __init__(
        self,
        device,
        model: str = "mace-off23-sc",
        model_path: str | None = None,
        implicit: str = "none",
        solvent: str = "none",
        model_card_path: str | None = None,
        calculator_factory: Callable[[str, str], object] | None = None,
        execution_task: str | None = None,
    ) -> None:
        super().__init__()
        requested = (
            str(model)
            .strip()
            .lower()
            .replace("-", "")
            .replace("_", "")
            .replace(".", "")
        )
        if requested != "maceoff23sc":
            raise ValueError(
                "MACE-OFF23-SC is pinned to model identity 'mace-off23-sc'; "
                f"received {model!r}."
            )
        implicit_name = str(implicit).strip().lower()
        solvent_name = str(solvent).strip().lower()
        if implicit_name not in {"", "none"} or solvent_name not in {"", "none"}:
            raise ValueError(
                "MACE-OFF23-SC is an explicit-system PES and cannot be composed "
                "with a MAPLE implicit backend or named implicit solvent."
            )
        if model_path is None:
            raise ValueError(
                "MACE-OFF23-SC requires an explicit model_path to the official "
                "MACE-OFF23-SC_swa.model checkpoint."
            )
        if (
            model_card_path is not None
            and Path(model_card_path).expanduser().resolve()
            != MACE_OFF23_SC_CARD.resolve()
        ):
            raise ValueError(
                "MACE-OFF23-SC binds a fixed official model card and does not "
                "accept a user-supplied model_card_path."
            )

        checkpoint = Path(model_path).expanduser()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"MACE-OFF23-SC checkpoint not found: {checkpoint}")
        card = _load_card(MACE_OFF23_SC_CARD)
        self.model_provenance = _validate_official_card(card)
        if execution_task is not None:
            self.model_provenance.validate_task(execution_task)
        self.model_provenance.validate_device(device, task=execution_task)

        actual = _sha256(checkpoint)
        if actual != MACE_OFF23_SC_SHA256:
            raise ValueError(
                "MACE-OFF23-SC checkpoint checksum mismatch: expected "
                f"{MACE_OFF23_SC_SHA256}, got {actual}."
            )

        self.device = device
        self.model_name = MACE_OFF23_SC_OFFICIAL_IDENTITY["model_id"]
        self.model_path = str(checkpoint)
        self.model_card_path = str(MACE_OFF23_SC_CARD)
        self.model_card = card
        self.execution_task = execution_task
        self.capabilities = self.model_provenance.capabilities
        self.solvent_correction = None
        factory = calculator_factory or _official_calculator_factory
        self._delegate = factory(self.model_path, str(device))

    @property
    def provenance(self) -> dict:
        return {
            "provider": "mace-off23-sc",
            "checkpoint_sha256": self.model_card["checkpoint_sha256"],
            "checkpoint_path": self.model_path,
            "source_revision": self.model_card["source"]["revision"],
            "protocol_revision": self.model_card["protocol_source"]["revision"],
            "model_card": self.model_card_path,
            "runtime_scope": "full_coupling_explicit_system_pes_only",
        }

    def validate_task(self, task: str) -> None:
        self.model_provenance.validate_task(task)

    def _validate_system(self, atoms) -> None:
        if len(atoms) == 0:
            raise ValueError("MACE-OFF23-SC requires at least one atom.")
        if "charge" not in atoms.info or "mult" not in atoms.info:
            raise ValueError(
                "MACE-OFF23-SC requires explicit atoms.info['charge'] and "
                "atoms.info['mult']; MAPLE will not assume neutral singlet metadata."
            )
        try:
            charge = float(atoms.info["charge"])
            multiplicity = float(atoms.info["mult"])
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "MACE-OFF23-SC requires finite integer charge and multiplicity."
            ) from exc
        if (
            not math.isfinite(charge)
            or not charge.is_integer()
            or not math.isfinite(multiplicity)
            or not multiplicity.is_integer()
        ):
            raise ValueError(
                "MACE-OFF23-SC requires finite integer charge and multiplicity."
            )
        if charge != 0.0 or multiplicity != 1.0:
            raise ValueError(
                "MACE-OFF23-SC is admitted only for explicit neutral singlets."
            )

        supported = set(self.model_card["elements"])
        unsupported = sorted(set(atoms.get_chemical_symbols()).difference(supported))
        if unsupported:
            raise ValueError(
                "MACE-OFF23-SC model card does not enable elements: "
                + ", ".join(unsupported)
            )

        pbc = np.asarray(atoms.get_pbc(), dtype=bool)
        if pbc.any() and not pbc.all():
            raise ValueError(
                "MACE-OFF23-SC accepts either a fully periodic or non-periodic system."
            )
        if pbc.all():
            cell = np.asarray(atoms.cell.array, dtype=float)
            if (
                cell.shape != (3, 3)
                or not np.isfinite(cell).all()
                or not math.isfinite(float(atoms.cell.volume))
                or float(atoms.cell.volume) <= 0.0
            ):
                raise ValueError(
                    "Periodic MACE-OFF23-SC systems require a finite positive-volume cell."
                )

    def _validate_properties(self, properties) -> tuple[str, ...]:
        normalized = tuple(str(item).strip().lower() for item in properties)
        unsupported = sorted(
            set(normalized).difference({"energy", "free_energy", "forces"})
        )
        if unsupported:
            raise ValueError(
                "MACE-OFF23-SC received unsupported properties: "
                + ", ".join(unsupported)
            )
        self.validate_task("sp")
        if "forces" in normalized:
            self.validate_task("opt")
        return normalized

    def calculate(
        self,
        atoms=None,
        properties=("energy",),
        system_changes=all_changes,
    ):
        properties = self._validate_properties(self._normalize_properties(properties))
        atoms = super().calculate(atoms, properties, system_changes)
        self._validate_system(atoms)

        delegated = ["energy"]
        if "forces" in properties:
            delegated.append("forces")
        self._delegate.calculate(
            atoms,
            properties=delegated,
            system_changes=system_changes,
        )
        raw = getattr(self._delegate, "results", {})
        if "energy" not in raw:
            raise RuntimeError("MACE-OFF23-SC delegate did not return energy.")
        try:
            energy = float(raw["energy"])
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("MACE-OFF23-SC delegate energy must be finite.") from exc
        if not math.isfinite(energy):
            raise ValueError("MACE-OFF23-SC delegate energy must be finite.")

        forces = None
        if "forces" in properties:
            if "forces" not in raw:
                raise RuntimeError("MACE-OFF23-SC delegate did not return forces.")
            forces = np.asarray(raw["forces"], dtype=float)
            if forces.shape != (len(atoms), 3):
                raise ValueError(
                    "MACE-OFF23-SC delegate forces must have shape (N, 3)."
                )
            if not np.isfinite(forces).all():
                raise ValueError("MACE-OFF23-SC delegate forces must be finite.")

        self._finalize_results(atoms, energy=energy, forces=forces)
        self.results["model_metadata"] = self.provenance
