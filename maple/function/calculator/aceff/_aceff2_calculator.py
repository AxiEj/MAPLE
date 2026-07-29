"""Lazy ASE adapter for the public AceFF 2.0 checkpoint."""

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

ACEFF2_SHA256 = "877af6bf84cc69d6d5cbd377a61cfc19c45fb87a00583f7c867e2f9d244cc8fd"
ACEFF2_OFFICIAL_IDENTITY = {
    "model_id": "aceff-2.0",
    "version": "2.0",
    "source_revision": "67ad6a3b06ee9b44cfbac3bc32117dd6276e9122",
}
ACEFF2_UNIT_EVIDENCE_REVISION = "3c59fb367903011e58abc51a7445cff85f7bf3df"
ACEFF2_CARD = Path(__file__).resolve().parents[1] / "model_cards" / "aceff-2.0.yaml"


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
            f"AceFF 2.0 model card must use JSON-compatible YAML syntax: {path}"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError("AceFF 2.0 model card must be a JSON object.")
    return payload


def _validate_official_card(card: dict) -> ModelProvenanceCard:
    if (
        str(card.get("model_id", "")).strip().lower()
        != ACEFF2_OFFICIAL_IDENTITY["model_id"]
    ):
        raise ValueError(
            "AceFF 2.0 model card identity mismatch for the pinned official adapter."
        )
    if str(card.get("version", "")).strip() != ACEFF2_OFFICIAL_IDENTITY["version"]:
        raise ValueError(
            "AceFF 2.0 model card version mismatch for the pinned official adapter."
        )
    source = card.get("source", {})
    if not isinstance(source, dict):
        raise ValueError("AceFF 2.0 model card must define source metadata.")
    if (
        str(source.get("revision", "")).strip()
        != ACEFF2_OFFICIAL_IDENTITY["source_revision"]
    ):
        raise ValueError(
            "AceFF 2.0 model card revision does not match the pinned upstream commit."
        )
    if str(card.get("checkpoint_sha256", "")).strip().lower() != ACEFF2_SHA256:
        raise ValueError(
            "AceFF 2.0 model card checksum does not match the pinned official digest "
            f"{ACEFF2_SHA256}."
        )
    unit_evidence = card.get("unit_evidence")
    if not isinstance(unit_evidence, dict) or (
        str(unit_evidence.get("revision", "")).strip()
        != ACEFF2_UNIT_EVIDENCE_REVISION
    ):
        raise ValueError(
            "AceFF 2.0 model card must retain the pinned official unit evidence."
        )
    elements = card.get("elements")
    if not isinstance(elements, list) or not elements:
        raise ValueError("AceFF 2.0 model card must define supported elements.")
    charge_range = card.get("charge_range")
    if (
        not isinstance(charge_range, list)
        or len(charge_range) != 2
        or any(isinstance(value, bool) for value in charge_range)
    ):
        raise ValueError("AceFF 2.0 model card must define a two-integer charge_range.")
    try:
        low, high = (int(value) for value in charge_range)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "AceFF 2.0 model card must define a two-integer charge_range."
        ) from exc
    if [low, high] != charge_range or low > high:
        raise ValueError("AceFF 2.0 model card charge_range is invalid.")
    provenance = ModelProvenanceCard.from_payload(
        card,
        model_name=ACEFF2_OFFICIAL_IDENTITY["model_id"],
    )
    if (
        not provenance.capabilities.requires_topology
        or provenance.capabilities.supports_multifragment
    ):
        raise ValueError(
            "AceFF 2.0 Route 4 requires topology-backed single-fragment domain gating."
        )
    return provenance


def _official_calculator_factory(model_path: str, device: str):
    try:
        from torchmdnet.calculators import TMDNETCalculator
    except ImportError as exc:
        raise ImportError(
            "AceFF 2.0 requires the optional torchmd-net package."
        ) from exc
    return TMDNETCalculator(model_file=model_path, device=device)


@register_calculator
class AceFF2Calculator(CalcABC):
    """Small-molecule conservative PES with strict public-domain gates."""

    implemented_properties = ["energy", "forces", "free_energy", "hessian"]
    MODEL_NAMES = ("aceff-2.0",)
    MODEL_ENERGY_UNIT = "eV"
    SUPPORTED_HESSIAN_MODES = ("numerical",)
    SUPPORTS_CHARGE_MULT = True
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
        model: str = "aceff-2.0",
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
                "AceFF 2.0 accepts no native solvent term; use a separately "
                "validated gb/pb composition or implicit='none'."
            )
        if implicit == "none" and solvent not in {"", "none"}:
            raise ValueError("A solvent name requires an explicit solvent backend.")
        if model_path is None:
            raise ValueError("AceFF 2.0 requires an explicit local model_path.")
        if (
            model_card_path is not None
            and Path(model_card_path).expanduser().resolve() != ACEFF2_CARD.resolve()
        ):
            raise ValueError(
                "AceFF 2.0 binds a fixed official model card and does not accept "
                "user-supplied model_card_path."
            )

        checkpoint = Path(model_path).expanduser()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"AceFF 2.0 checkpoint not found: {checkpoint}")
        card_path = ACEFF2_CARD
        card = _load_card(card_path)
        self.model_provenance = _validate_official_card(card)
        self.capabilities = self.model_provenance.capabilities
        actual = _sha256(checkpoint)
        if actual != ACEFF2_SHA256:
            raise ValueError(
                "AceFF 2.0 checkpoint checksum mismatch: "
                f"expected {ACEFF2_SHA256}, got {actual}."
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
        if "charge" not in atoms.info:
            raise ValueError("AceFF 2.0 requires explicit atoms.info['charge'].")
        charge_raw = float(atoms.info["charge"])
        if not math.isfinite(charge_raw) or not charge_raw.is_integer():
            raise ValueError("AceFF 2.0 total charge must be a finite integer.")
        charge = int(charge_raw)
        low, high = self.model_card["charge_range"]
        if not low <= charge <= high:
            raise ValueError(
                "AceFF 2.0 total charge must be an integer in {-2,-1,0,1,2}."
            )
        multiplicity = float(atoms.info.get("mult", 1))
        if (
            not math.isfinite(multiplicity)
            or not multiplicity.is_integer()
            or multiplicity != 1.0
        ):
            raise ValueError(
                "AceFF 2.0 has no public spin/multiplicity contract; only mult=1 is enabled."
            )
        supported = set(self.model_card["elements"])
        unsupported = sorted(set(atoms.get_chemical_symbols()).difference(supported))
        if unsupported:
            raise ValueError(
                "AceFF 2.0 model card does not enable elements: "
                + ", ".join(unsupported)
            )
        try:
            TopologyProvider.from_mol2_atoms(
                atoms,
                require_single_fragment=True,
            )
        except ValueError as exc:
            if "atoms.info['mol2'] metadata" in str(exc):
                raise ValueError(
                    "AceFF 2.0 requires explicit MOL2 topology metadata."
                ) from exc
            if "disconnected fragments" in str(exc):
                raise ValueError(
                    "AceFF 2.0 accepts one small-molecule fragment per ASE Atoms."
                ) from exc
            raise

    def validate_task(self, task: str) -> None:
        self.model_provenance.validate_task(task)

    def _validate_capabilities(self, properties: list[str]) -> None:
        self.validate_task("sp")
        if "forces" in properties:
            self.validate_task("opt")
        if "hessian" in properties:
            self.validate_task("frequency")

    @property
    def provenance(self) -> dict:
        return {
            "provider": "aceff-2.0",
            "checkpoint_sha256": self.model_card["checkpoint_sha256"],
            "checkpoint_path": self.model_path,
            "source_revision": self.model_card["source"]["revision"],
            "model_card": self.model_card_path,
            "forbidden_tasks": list(self.model_provenance.forbidden_tasks),
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
            raise RuntimeError("AceFF 2.0 delegate did not return energy.")
        forces = None
        if "forces" in properties:
            if "forces" not in raw:
                raise RuntimeError("AceFF 2.0 delegate did not return forces.")
            forces = np.asarray(raw["forces"], dtype=float)
        hessian = self.get_hessian(atoms) if "hessian" in properties else None
        self._finalize_results(
            atoms,
            energy=float(raw["energy"]),
            forces=forces,
            hessian=hessian,
        )
        self.results["model_metadata"] = self.provenance
