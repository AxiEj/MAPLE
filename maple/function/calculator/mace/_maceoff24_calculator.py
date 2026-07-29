"""Lazy ASE adapter for the public MACE-OFF24 Medium checkpoint."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Callable

import numpy as np
from ase.calculators.calculator import all_changes

from ..calculator_base import CalcABC, register_calculator
from ..extra_correction.implicit.topology import TopologyProvider
from ..model_capabilities import ModelProvenanceCard

MACE_OFF24_MEDIUM_SHA256 = (
    "e5ccf5837f685899811a68754e7c994393bfd1a81720393b03c643b46c70bc69"
)
MACE_OFF24_OFFICIAL_IDENTITY = {
    "model_id": "mace-off24-medium",
    "version": "MACE-OFF24(M)",
    "source_revision": "91a78c5a9c300d1104700d9352c8bfe449227737",
}
MACE_OFF24_CARD = (
    Path(__file__).resolve().parents[1] / "model_cards" / "mace-off24-medium.yaml"
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
            f"MACE-OFF24 model card must use JSON-compatible YAML syntax: {path}"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError("MACE-OFF24 model card must be a JSON object.")
    return payload


def _validate_official_card(card: dict) -> ModelProvenanceCard:
    if (
        str(card.get("model_id", "")).strip().lower()
        != MACE_OFF24_OFFICIAL_IDENTITY["model_id"]
    ):
        raise ValueError(
            "MACE-OFF24 model card identity mismatch: this adapter is pinned to "
            f"{MACE_OFF24_OFFICIAL_IDENTITY['model_id']!r}."
        )
    if str(card.get("version", "")).strip() != MACE_OFF24_OFFICIAL_IDENTITY["version"]:
        raise ValueError(
            "MACE-OFF24 card version mismatch for official identity: "
            f"expected {MACE_OFF24_OFFICIAL_IDENTITY['version']!r}, got {str(card.get('version', '')).strip()!r}."
        )
    source = card.get("source", {})
    if not isinstance(source, dict):
        raise ValueError("MACE-OFF24 model card must define source metadata.")
    if (
        str(source.get("revision", "")).strip()
        != MACE_OFF24_OFFICIAL_IDENTITY["source_revision"]
    ):
        raise ValueError(
            "MACE-OFF24 model card revision does not match the pinned upstream commit."
        )
    if (
        str(card.get("checkpoint_sha256", "")).strip().lower()
        != MACE_OFF24_MEDIUM_SHA256
    ):
        raise ValueError(
            "MACE-OFF24 model card checksum does not match the pinned official digest "
            f"{MACE_OFF24_MEDIUM_SHA256}."
        )
    elements = card.get("elements")
    if not isinstance(elements, list) or not elements:
        raise ValueError("MACE-OFF24 model card must define its supported elements.")
    provenance = ModelProvenanceCard.from_payload(
        card,
        model_name=MACE_OFF24_OFFICIAL_IDENTITY["model_id"],
    )
    if (
        not provenance.capabilities.requires_topology
        or provenance.capabilities.supports_multifragment
    ):
        raise ValueError(
            "MACE-OFF24 Route 4 requires topology-backed single-fragment domain gating."
        )
    return provenance


def _official_calculator_factory(
    model_path: str,
    device: str,
):
    try:
        from mace.calculators import mace_off
    except ImportError as exc:
        raise ImportError(
            "MACE-OFF24 requires the optional mace-torch package."
        ) from exc
    return mace_off(
        model=model_path,
        device=device,
        default_dtype="float64",
    )


@register_calculator
class MACEOFF24MediumCalculator(CalcABC):
    """Complete conservative PES; solvent terms remain separate backends."""

    implemented_properties = ["energy", "forces", "free_energy", "hessian"]
    MODEL_NAMES = ("mace-off24-medium",)
    MODEL_ENERGY_UNIT = "eV"
    SUPPORTED_HESSIAN_MODES = ("numerical",)
    SUPPORTS_CHARGE_MULT = False
    SUPPORTS_PBC = False
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
        kwargs = {}
        if resolved_model_path is not None:
            kwargs["model_path"] = resolved_model_path
        return kwargs

    def __init__(
        self,
        device,
        model: str = "mace-off24-medium",
        model_path: str | None = None,
        implicit: str = "none",
        solvent: str = "none",
        model_card_path: str | None = None,
        calculator_factory: Callable[[str, str], object] | None = None,
    ) -> None:
        super().__init__()
        implicit = str(implicit).strip().lower()
        solvent = str(solvent).strip().lower()
        if implicit not in {"none", "gb", "pb"}:
            raise ValueError(
                "MACE-OFF24 accepts no native solvent term; use MAPLE's separately "
                "validated gb/pb composition or implicit='none'."
            )
        if implicit == "none" and solvent not in {"", "none"}:
            raise ValueError("A solvent name requires an explicit solvent backend.")
        if model_path is None:
            raise ValueError(
                "MACE-OFF24 Medium requires an explicit model_path to "
                "MACE-OFF24_medium.model; mace_off(model='medium') resolves OFF23."
            )
        if (
            model_card_path is not None
            and Path(model_card_path).expanduser().resolve()
            != MACE_OFF24_CARD.resolve()
        ):
            raise ValueError(
                "MACE-OFF24 Medium binds a fixed official model card and "
                "does not accept user-supplied model_card_path."
            )

        checkpoint = Path(model_path).expanduser()
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"MACE-OFF24 Medium checkpoint not found: {checkpoint}"
            )
        card_path = MACE_OFF24_CARD
        card = _load_card(card_path)
        self.model_provenance = _validate_official_card(card)
        self.capabilities = self.model_provenance.capabilities
        expected = MACE_OFF24_MEDIUM_SHA256
        actual = _sha256(checkpoint)
        if actual != expected:
            raise ValueError(
                "MACE-OFF24 checkpoint checksum mismatch: "
                f"expected {expected}, got {actual}."
            )

        self.device = device
        self.model_name = model
        self.model_path = str(checkpoint)
        self.model_card_path = str(card_path)
        self.model_card = card
        self.hessian = "numerical"
        self.solvent_correction = None
        factory = calculator_factory or _official_calculator_factory
        self._delegate = factory(self.model_path, str(device))

    def _validate_system(self, atoms) -> None:
        try:
            charge = float(atoms.info.get("charge", 0))
            multiplicity = float(atoms.info.get("mult", 1))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "MACE-OFF24 Medium requires finite integer charge and multiplicity."
            ) from exc
        if (
            not math.isfinite(charge)
            or not charge.is_integer()
            or not math.isfinite(multiplicity)
            or not multiplicity.is_integer()
            or charge != 0.0
            or multiplicity != 1.0
        ):
            raise ValueError(
                "MACE-OFF24 Medium has no public charge/spin contract; only "
                "neutral singlets are enabled."
            )

        supported = set(self.model_card["elements"])
        unsupported = sorted(set(atoms.get_chemical_symbols()).difference(supported))
        if unsupported:
            raise ValueError(
                "MACE-OFF24 Medium model card does not enable elements: "
                + ", ".join(unsupported)
            )
        if not isinstance(atoms.info.get("mol2"), dict):
            raise ValueError(
                "MACE-OFF24 Medium requires explicit MOL2 topology metadata "
                "for single-fragment domain validation."
            )
        try:
            TopologyProvider.from_mol2_atoms(
                atoms,
                require_single_fragment=True,
            )
        except ValueError as exc:
            if "disconnected fragments" in str(exc):
                raise ValueError(
                    "MACE-OFF24 Medium accepts one molecular fragment per ASE Atoms."
                ) from exc
            raise

    def validate_task(self, task: str) -> None:
        self.model_provenance.validate_task(task)

    def _validate_capabilities(self, properties: list) -> None:
        self.validate_task("sp")
        if "forces" in properties:
            self.validate_task("opt")
        if "hessian" in properties:
            self.validate_task("frequency")

    @property
    def provenance(self) -> dict:
        return {
            "provider": "mace-off24-medium",
            "checkpoint_sha256": self.model_card["checkpoint_sha256"],
            "checkpoint_path": self.model_path,
            "source_revision": self.model_card["source"]["revision"],
            "model_card": self.model_card_path,
        }

    def calculate(
        self,
        atoms=None,
        properties=("energy",),
        system_changes=all_changes,
    ):
        properties = self._normalize_properties(properties)
        atoms = super().calculate(atoms, properties, system_changes)
        self._validate_system(atoms)
        self._validate_capabilities(properties)

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
            raise RuntimeError("MACE-OFF24 delegate did not return energy.")
        forces = None
        if "forces" in properties:
            if "forces" not in raw:
                raise RuntimeError("MACE-OFF24 delegate did not return forces.")
            forces = np.asarray(raw["forces"], dtype=float)
        hessian = self.get_hessian(atoms) if "hessian" in properties else None
        self._finalize_results(
            atoms,
            energy=float(raw["energy"]),
            forces=forces,
            hessian=hessian,
        )
        self.results["model_metadata"] = self.provenance

    def get_hessian(self, atoms, delta: float = 0.002):
        self._validate_capabilities(["hessian"])
        return super().get_hessian(atoms, delta=delta)
