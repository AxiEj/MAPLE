from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Dict, Literal

import torch
from ase.calculators.calculator import all_changes

from ..calculator_base import CalcABC, register_calculator
from ..model_capabilities import ModelProvenanceCard

DEFAULT_CARD_PATH = (
    Path(__file__).resolve().parents[1] / "model_cards" / "aimnet2-cpcms-v2.yaml"
)
OFFICIAL_IDENTITY = {
    "model_id": "aimnet2-cpcms-v2",
    "version": "wb97m_cpcms_v2_0",
    "source_revision": "3f1c3fe015ca07b0d7f907d8f6ebc4b5dea7ba83",
}
DEFAULT_CHECKPOINT_SHA256 = (
    "32f73fcd68e76f67a6e088841304ca0adc96066cca83216dd9ed992e9e4dec87"
)


def _read_json_model_card(path: Path) -> dict:
    """Read the model-card payload using JSON-compatible syntax only."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Model-card file at {path} must be JSON-compatible (JSON only)."
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Model-card payload at {path} must be a JSON object.")
    return payload


def _validate_official_identity(payload: dict) -> None:
    payload_model_id = str(payload.get("model_id", "")).strip().lower()
    if payload_model_id != OFFICIAL_IDENTITY["model_id"]:
        raise ValueError(
            f"Model-card model_id mismatch for official AIMNet2-CPCMS: "
            f"expected {OFFICIAL_IDENTITY['model_id']!r}, got {payload_model_id!r}."
        )

    payload_version = str(payload.get("version", "")).strip()
    if payload_version != OFFICIAL_IDENTITY["version"]:
        raise ValueError(
            f"Model-card version mismatch for official AIMNet2-CPCMS: "
            f"expected {OFFICIAL_IDENTITY['version']!r}, got {payload_version!r}."
        )

    source = payload.get("source", {})
    if not isinstance(source, dict):
        raise ValueError("AIMNet2-CPCMS model card must include a source mapping.")
    payload_revision = str(source.get("revision", "")).strip()
    if payload_revision != OFFICIAL_IDENTITY["source_revision"]:
        raise ValueError(
            "AIMNet2-CPCMS model card source revision mismatch for the official "
            f"identity: expected {OFFICIAL_IDENTITY['source_revision']!r}, "
            f"got {payload_revision!r}."
        )

    payload_checksum = str(payload.get("checkpoint_sha256", "")).strip().lower()
    if payload_checksum != DEFAULT_CHECKPOINT_SHA256:
        raise ValueError(
            "AIMNet2-CPCMS model-card checkpoint checksum does not match "
            f"the pinned official digest {DEFAULT_CHECKPOINT_SHA256}."
        )


def _coerce_bool(value, name: str) -> bool:
    if isinstance(value, bool):
        return value
    raise TypeError(f"Model-card field {name!r} must be boolean.")


def _coerce_capability_map(payload: dict) -> dict:
    raw_caps = payload.get("capabilities")
    if raw_caps is None:
        raw_caps = {}
    if not isinstance(raw_caps, dict):
        raise ValueError(
            "Model-card field 'capabilities' must be a mapping if present."
        )

    top_level = payload
    return {
        "supports_absolute_solvation": _coerce_bool(
            top_level.get("supports_absolute_solvation", False),
            "supports_absolute_solvation",
        ),
        "supports_free_energy": _coerce_bool(
            top_level.get("supports_free_energy", False),
            "supports_free_energy",
        ),
        "supports_single_point": _coerce_bool(
            raw_caps.get("supports_single_point", True),
            "capabilities.supports_single_point",
        ),
        "supports_energy_derived_forces": _coerce_bool(
            raw_caps.get(
                "supports_energy_derived_forces",
                top_level.get("supports_energy_derived_forces", False),
            ),
            "capabilities.supports_energy_derived_forces",
        ),
        "supports_numerical_hessian": _coerce_bool(
            raw_caps.get("supports_numerical_hessian", False),
            "capabilities.supports_numerical_hessian",
        ),
    }


def _sha256_for_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().lower()


def _coerce_device(value):
    return torch.device(value)


@register_calculator
class AIMNet2CPCMSCalculator(CalcABC):
    """CalcABC adapter for LoQI-style scripted AIMNet2-CPCMS checkpoints.

    The wrapper is intentionally conservative: native solution-phase only,
    explicit checkpoint + model-card paths, explicit checksum assertion, fixed
    multiplicity=1, and fail-closed derivative/task gates.
    """

    implemented_properties = ["energy", "forces", "free_energy", "hessian"]

    MODEL_NAMES = ("aimnet2-cpcms-v2",)
    MODEL_ENERGY_UNIT = "eV"
    SUPPORTED_HESSIAN_MODES = ("numerical",)
    SUPPORTS_CHARGE_MULT = True
    SUPPORTS_PBC = False
    OPTION_KEYS = ()
    MODEL_PATH_OPTION = "model_path"

    @classmethod
    def build_kwargs_from_options(cls, model, options, *, resolved_model_path=None):
        kwargs = {}
        if resolved_model_path is not None:
            kwargs["model_path"] = resolved_model_path
        return kwargs

    def __init__(
        self,
        device: torch.device,
        model: str = "aimnet2-cpcms-v2",
        model_path: str | None = None,
        implicit: Literal["gbsa", "none"] = "none",
        solvent: str = "none",
        model_card_path: str | None = None,
        execution_task: str | None = None,
    ):
        super().__init__()

        requested_model = (
            str(model)
            .strip()
            .lower()
            .replace("-", "")
            .replace("_", "")
            .replace(".", "")
        )
        if requested_model != "aimnet2cpcmsv2":
            raise ValueError(
                "AIMNet2-CPCMS is pinned to model identity 'aimnet2-cpcms-v2'; "
                f"received {model!r}."
            )
        if implicit != "none":
            raise ValueError(
                "AIMNet2-CPCMS is already native implicit-solvent and "
                "does not support additive implicit controls."
            )
        if solvent not in (None, "none", "None", "NONE"):
            raise ValueError(
                "AIMNet2-CPCMS has native fixed/unknown solvent and "
                "does not accept additive solvent keywords."
            )
        if model_path is None:
            raise ValueError(
                "AIMNet2-CPCMS requires explicit model_path; "
                "model weights are not auto-selected in this route."
            )
        if (
            model_card_path is not None
            and Path(model_card_path).expanduser().resolve()
            != DEFAULT_CARD_PATH.resolve()
        ):
            raise ValueError(
                "AIMNet2-CPCMS binds a fixed official model card and does "
                "not accept user-supplied model_card_path."
            )

        self.device = _coerce_device(device)
        self.model_name = OFFICIAL_IDENTITY["model_id"]
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"AIMNet2-CPCMS model_path does not exist or is not a file: {self.model_path}"
            )

        card_path = DEFAULT_CARD_PATH
        if not card_path.is_file():
            raise FileNotFoundError(
                f"AIMNet2-CPCMS model card does not exist or is not a file: {card_path}"
            )

        payload = _read_json_model_card(card_path)
        _validate_official_identity(payload)
        self.model_provenance = ModelProvenanceCard.from_payload(
            payload,
            model_name=OFFICIAL_IDENTITY["model_id"],
        )
        self.model_provenance.validate_device(self.device, task=execution_task)
        self.execution_task = execution_task
        self.capabilities = self.model_provenance.capabilities
        expected_checksum = DEFAULT_CHECKPOINT_SHA256
        actual_checksum = _sha256_for_path(self.model_path)
        if expected_checksum != actual_checksum:
            raise ValueError(
                f"Model checksum mismatch for AIMNet2-CPCMS: expected {expected_checksum}, got {actual_checksum}."
            )

        if not _coerce_bool(payload.get("audited", False), "audited"):
            raise RuntimeError(
                "AIMNet2-CPCMS metadata card is not marked audited "
                "and cannot be used in this route."
            )

        caps = _coerce_capability_map(payload)
        if caps["supports_absolute_solvation"]:
            raise RuntimeError(
                "AIMNet2-CPCMS card disallows absolute solvation claims "
                "in this route."
            )
        if caps["supports_free_energy"]:
            raise RuntimeError(
                "AIMNet2-CPCMS card disallows free-energy certification "
                "in this route."
            )

        self.model_card_path = card_path
        self.model_card = payload
        self.model_card_checksum = expected_checksum
        self.model_card_checksum_actual = actual_checksum

        self._supports_single_point = caps["supports_single_point"]
        self._supports_energy_derived_forces = caps["supports_energy_derived_forces"]
        self._supports_numerical_hessian = caps["supports_numerical_hessian"]
        if (
            self._supports_numerical_hessian
            and not self._supports_energy_derived_forces
        ):
            raise ValueError(
                "AIMNet2-CPCMS numerical Hessian capability requires "
                "energy-derived conservative forces."
            )

        self.model = torch.jit.load(
            str(self.model_path), map_location=self.device
        ).eval()
        self.hessian = "numerical"

        self.model_metadata = {
            "model_name": self.model_name,
            "model_path": str(self.model_path),
            "model_card_path": str(self.model_card_path),
            "model_card_checksum_expected": expected_checksum,
            "model_card_checksum_actual": actual_checksum,
            "solvent_scope": payload.get("solvent_scope", "fixed-or-model-defined"),
            "experimental": bool(payload.get("experimental", True)),
            "capabilities": caps,
            "forbidden_tasks": list(self.model_provenance.forbidden_tasks),
        }

    def validate_task(self, task: str) -> None:
        self.model_provenance.validate_task(task)

    def _validate_capability_tasks(self, properties: list) -> None:
        if "hessian" in properties:
            self.validate_task("frequency")
            if not (
                self._supports_single_point
                and self._supports_energy_derived_forces
                and self._supports_numerical_hessian
            ):
                raise ValueError(
                    "AIMNet2-CPCMS frequency workflow requires "
                    "single-point, conservative forces, and numerical Hessian "
                    "capability in the pinned model card."
                )

        if "forces" in properties:
            self.validate_task("opt")
            if not (
                self._supports_single_point and self._supports_energy_derived_forces
            ):
                raise ValueError(
                    "AIMNet2-CPCMS requires energy and conservative forces "
                    "to support force workflows."
                )

        if "energy" in properties or "free_energy" in properties:
            self.validate_task("sp")
            if not self._supports_single_point:
                raise RuntimeError(
                    "AIMNet2-CPCMS single-point workflow is disabled in model-card provenance."
                )

    def _validate_request(self, atoms) -> None:
        self._validate_capability_tasks(["energy"])

        info = getattr(atoms, "info", {})
        if not isinstance(info, dict) or "charge" not in info or "mult" not in info:
            raise ValueError(
                "AIMNet2-CPCMS requires explicit atoms.info['charge'] and "
                "atoms.info['mult']; MAPLE will not assume neutral singlet metadata."
            )

        mult = info["mult"]
        try:
            mult_numeric = float(mult)
        except (TypeError, ValueError) as exc:
            raise ValueError("AIMNet2-CPCMS requires an integer multiplicity.") from exc
        if not mult_numeric.is_integer() or int(mult_numeric) != 1:
            raise ValueError(
                "AIMNet2-CPCMS supports multiplicity=1 only in this route."
            )

        charge = float(self._total_charge_from_atoms(atoms))
        if not math.isfinite(charge) or not charge.is_integer() or charge != 0.0:
            raise ValueError(
                "AIMNet2-CPCMS upstream charge range is unaudited; Route 4 "
                "currently enables neutral molecules only (charge=0)."
            )

        supported_elements = self.model_card.get("elements")
        if isinstance(supported_elements, list) and supported_elements:
            unsupported = sorted(
                set(atoms.get_chemical_symbols()).difference(supported_elements)
            )
            if unsupported:
                raise ValueError(
                    "AIMNet2-CPCMS model card does not enable elements: "
                    + ", ".join(unsupported)
                )

    def _build_input(self, coord: torch.Tensor, atoms) -> Dict[str, torch.Tensor]:
        numbers = torch.tensor(
            atoms.get_atomic_numbers(),
            dtype=torch.int64,
            device=self.device,
        )
        charge = float(self._total_charge_from_atoms(atoms))
        return {
            "coord": coord.unsqueeze(0),
            "numbers": numbers.unsqueeze(0),
            "charge": torch.tensor([charge], dtype=torch.float32, device=self.device),
        }

    def _forward_energy(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        with torch.jit.optimized_execution(False):
            raw = self.model(data)

        if isinstance(raw, torch.Tensor):
            energy = raw
        elif isinstance(raw, dict):
            if "energy" not in raw:
                raise KeyError(
                    'TorchScript LoQI model output must contain key "energy".'
                )
            energy = raw["energy"]
        else:
            raise TypeError(
                "TorchScript LoQI model output must be a tensor or a dict "
                'with key "energy".'
            )

        if isinstance(energy, torch.Tensor):
            return energy.sum()
        return torch.as_tensor(energy, device=self.device).sum()

    def calculate(self, atoms=None, properties=["energy"], system_changes=all_changes):
        properties = self._normalize_properties(properties)
        atoms = super().calculate(atoms, properties, system_changes)

        self._validate_request(atoms)
        self._validate_capability_tasks(properties)

        if "forces" in properties and not self._supports_energy_derived_forces:
            raise NotImplementedError(
                "AIMNet2-CPCMS backend is currently marked "
                "supports_energy_derived_forces=false; force/gradient workflows are disabled "
                "until model-card audit updates."
            )
        if "hessian" in properties and not self._supports_numerical_hessian:
            raise NotImplementedError(
                "AIMNet2-CPCMS backend is currently marked "
                "supports_numerical_hessian=false; Hessian workflows are disabled "
                "until numerical-hessian audit updates."
            )

        coord = torch.tensor(
            atoms.get_positions(),
            dtype=torch.float32,
            device=self.device,
            requires_grad=("forces" in properties or "hessian" in properties),
        )
        energy_eV = self._forward_energy(self._build_input(coord, atoms))
        if not bool(torch.isfinite(energy_eV).item()):
            raise ValueError("AIMNet2-CPCMS energy must be finite.")

        forces_np = None
        if "forces" in properties:
            grad = torch.autograd.grad(
                energy_eV,
                coord,
                create_graph=("hessian" in properties),
            )[0]
            if not bool(torch.isfinite(grad).all().item()):
                raise ValueError("AIMNet2-CPCMS forces must be finite.")
            forces_np = (-grad).detach().cpu().numpy()

        hessian = None
        if "hessian" in properties:
            hessian = super().get_hessian(atoms)

        self._finalize_results(
            atoms,
            energy=float(energy_eV.item()),
            forces=forces_np,
            hessian=hessian,
            unit="eV",
        )

        self.results["model_metadata"] = dict(self.model_metadata)
        self.results["provenance"] = {
            "calculator": type(self).__name__,
            "model_id": self.model_metadata["model_name"],
            "solvent": self.model_card.get("solvent_scope"),
            "supports_absolute_solvation": False,
            "supports_free_energy": False,
            "supports_numerical_hessian": self._supports_numerical_hessian,
            "supports_energy_derived_forces": self._supports_energy_derived_forces,
            "model_card_checksum": self.model_card_checksum,
            "checkpoint_checksum": self.model_card_checksum_actual,
            "model_version": self.model_card.get("version"),
            "scientific_status": self.model_card.get(
                "scientific_status", "experimental"
            ),
        }

    def get_hessian(self, atoms, delta: float = 0.002):
        self._validate_request(atoms)
        self._validate_capability_tasks(["hessian"])
        if not self._supports_numerical_hessian:
            raise NotImplementedError(
                "AIMNet2-CPCMS backend is currently marked "
                "supports_numerical_hessian=false; Hessian workflows are disabled "
                "until numerical-hessian audit updates."
            )

        return super().get_hessian(atoms, delta=delta)
