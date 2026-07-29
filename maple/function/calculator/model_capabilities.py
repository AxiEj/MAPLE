"""Machine-readable capability and provenance contracts for model composition."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping, Sequence

MODEL_CARD_EXTENSIONS = (".yaml", ".yml", ".json")
CONSERVATIVE_NUMERICAL_FREQUENCY_TYPE = "effective_solution_pmf"

HessianMode = Literal["analytic", "autograd", "finite_difference", "none"]
SolvationMode = Literal["none", "native", "additive", "alchemical", "property_only"]
EnergyReference = Literal["absolute", "relative", "unknown"]


class ModelCardError(ValueError):
    """Raised when a model-provenance card cannot be loaded or validated."""


def _strict_bool(
    payload: Mapping[str, object],
    name: str,
    *,
    default: bool = False,
) -> bool:
    value = payload.get(name, default)
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    raise ModelCardError(
        f"Model-card field {name!r} must be a boolean, got {type(value).__name__}."
    )


@dataclass(frozen=True)
class ModelCapabilities:
    """Capabilities of one potential or property model.

    The defaults are deliberately closed.  A task is enabled only by explicit
    card evidence; absence of a field never implies support.
    """

    energy: bool = False
    forces: bool = False
    conservative_forces: bool = False
    hessian: HessianMode = "none"
    supports_md: bool = False
    supports_pbc: bool = False
    supports_charge: bool = False
    supports_multiplicity: bool = False
    supports_multifragment: bool = False
    requires_topology: bool = False
    requires_partial_charges: bool = False
    solvation_mode: SolvationMode = "none"
    energy_reference: EnergyReference = "unknown"
    supports_absolute_solvation: bool = False
    supports_alchemical_lambda: bool = False

    @property
    def supports_numerical_hessian(self) -> bool:
        return self.hessian == "finite_difference"

    @property
    def supports_energy_derived_forces(self) -> bool:
        return self.forces and self.conservative_forces

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "ModelCapabilities":
        raw = payload.get("capabilities", payload)
        if not isinstance(raw, Mapping):
            raise ModelCardError("Model-card capabilities must be a mapping.")

        hessian = raw.get("hessian")
        if hessian is None:
            hessian = (
                "finite_difference"
                if _strict_bool(raw, "supports_numerical_hessian", default=False)
                else "none"
            )
        hessian = str(hessian).strip().lower()
        if hessian not in {"analytic", "autograd", "finite_difference", "none"}:
            raise ModelCardError(f"Unsupported hessian capability: {hessian!r}.")

        conservative = _strict_bool(
            raw,
            "conservative_forces",
            default=_strict_bool(raw, "supports_energy_derived_forces", default=False),
        )
        forces = _strict_bool(raw, "forces", default=conservative)
        solvation_mode = str(raw.get("solvation_mode", "none")).strip().lower()
        if solvation_mode not in {
            "none",
            "native",
            "additive",
            "alchemical",
            "property_only",
        }:
            raise ModelCardError(
                f"Unsupported solvation_mode capability: {solvation_mode!r}."
            )
        energy_reference = (
            str(raw.get("energy_reference", payload.get("energy_reference", "unknown")))
            .strip()
            .lower()
        )
        if energy_reference == "relative_or_unknown":
            energy_reference = "unknown"
        if energy_reference not in {"absolute", "relative", "unknown"}:
            raise ModelCardError(
                f"Unsupported energy_reference capability: {energy_reference!r}."
            )

        capabilities = cls(
            energy=_strict_bool(raw, "energy", default=False),
            forces=forces,
            conservative_forces=conservative,
            hessian=hessian,  # type: ignore[arg-type]
            supports_md=_strict_bool(raw, "supports_md", default=False),
            supports_pbc=_strict_bool(raw, "supports_pbc", default=False),
            supports_charge=_strict_bool(raw, "supports_charge", default=False),
            supports_multiplicity=_strict_bool(
                raw, "supports_multiplicity", default=False
            ),
            supports_multifragment=_strict_bool(
                raw, "supports_multifragment", default=False
            ),
            requires_topology=_strict_bool(raw, "requires_topology", default=False),
            requires_partial_charges=_strict_bool(
                raw, "requires_partial_charges", default=False
            ),
            solvation_mode=solvation_mode,  # type: ignore[arg-type]
            energy_reference=energy_reference,  # type: ignore[arg-type]
            supports_absolute_solvation=_strict_bool(
                raw,
                "supports_absolute_solvation",
                default=_strict_bool(
                    payload, "supports_absolute_solvation", default=False
                ),
            ),
            supports_alchemical_lambda=_strict_bool(
                raw,
                "supports_alchemical_lambda",
                default=_strict_bool(
                    payload, "supports_alchemical_lambda", default=False
                ),
            ),
        )
        capabilities._validate_consistency()
        return capabilities

    def _validate_consistency(self) -> None:
        if self.conservative_forces and not (self.energy and self.forces):
            raise ModelCardError(
                "conservative_forces=true requires both energy=true and forces=true."
            )
        if self.hessian != "none" and not (
            self.energy and self.forces and self.conservative_forces
        ):
            raise ModelCardError(
                "Hessian support requires an energy model with conservative forces."
            )

    def validate_task(self, task: str) -> None:
        task = str(task).strip().lower()
        if task in {"sp", "single_point", "energy"} and not self.energy:
            raise ValueError("The model card does not enable energy evaluation.")
        if task in {"opt", "optimization"} and not (
            self.energy and self.forces and self.conservative_forces
        ):
            raise ValueError(
                "Optimization requires energy-derived conservative forces."
            )
        if task in {"frequency", "freq"} and not (
            self.energy
            and self.forces
            and self.conservative_forces
            and self.hessian != "none"
        ):
            raise ValueError(
                "Frequency analysis requires a conservative energy/force model and Hessian support."
            )
        if task == "md" and not self.supports_md:
            raise ValueError("The model card does not enable molecular dynamics.")
        if (
            task
            in {
                "absolute_solvation",
                "absolute_solvation_free_energy",
            }
            and not self.supports_absolute_solvation
        ):
            raise ValueError(
                "The model card does not enable absolute solvation free energy."
            )
        if task in {"alchemical", "alchemical_free_energy"} and not (
            self.supports_alchemical_lambda
        ):
            raise ValueError("The model card does not enable alchemical lambda.")


@dataclass(frozen=True)
class SolvationCapabilities:
    """Capabilities of an additive solvent backend."""

    energy: bool = True
    forces: bool = False
    conservative_forces: bool = False
    hessian: HessianMode = "none"
    supports_pbc: bool = False
    multisolvent: bool = False
    energy_reference: EnergyReference = "unknown"
    supports_absolute_solvation: bool = False

    @classmethod
    def from_backend(cls, backend) -> "SolvationCapabilities":
        supported = {
            str(item).strip().lower()
            for item in getattr(backend, "supported_properties", {"energy"})
        }
        return cls(
            energy="energy" in supported,
            forces="forces" in supported,
            conservative_forces=bool(getattr(backend, "conservative_forces", False)),
            hessian=getattr(backend, "hessian_capability", "none"),
            supports_pbc=bool(getattr(backend, "supports_pbc", False)),
            multisolvent=bool(getattr(backend, "multisolvent", False)),
            energy_reference=getattr(backend, "energy_reference", "unknown"),
            supports_absolute_solvation=bool(
                getattr(backend, "supports_absolute_solvation", False)
            ),
        )


def _normalize_task_name(task: object) -> str:
    normalized = str(task).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "single_point": "sp",
        "energy": "sp",
        "optimization": "opt",
        "geometry_optimization": "opt",
        "freq": "frequency",
        "numerical_frequency": "frequency",
        "short_md": "md",
        "molecular_dynamics": "md",
        "absolute_solvation": "absolute_solvation_free_energy",
        "alchemical": "alchemical_free_energy",
    }
    return aliases.get(normalized, normalized)


def _normalize_forbidden_tasks(raw: object) -> tuple[str, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        raise ModelCardError("Model-card forbidden_tasks must be a sequence.")
    normalized = {_normalize_task_name(item) for item in raw if str(item).strip()}
    return tuple(sorted(normalized))


def _capability_enables_task(capabilities: ModelCapabilities, task: str) -> bool:
    try:
        capabilities.validate_task(task)
    except ValueError:
        return False
    return True


@dataclass(frozen=True)
class ModelProvenanceCard:
    """Structured model metadata used for scientific capability checks."""

    model_id: str
    version: str
    capabilities: ModelCapabilities
    payload: Mapping[str, object]
    forbidden_tasks: tuple[str, ...] = ()

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, object],
        model_name: str,
    ) -> "ModelProvenanceCard":
        if not isinstance(payload, Mapping):
            raise ModelCardError("Model provenance payload must be a mapping.")
        model_id = str(payload.get("model_id", model_name)).strip().lower()
        if not model_id:
            raise ModelCardError("Model card model_id cannot be empty.")
        version = str(payload.get("version", "unknown")).strip() or "unknown"
        card = cls(
            model_id=model_id,
            version=version,
            capabilities=ModelCapabilities.from_payload(payload),
            payload=dict(payload),
            forbidden_tasks=_normalize_forbidden_tasks(payload.get("forbidden_tasks")),
        )
        card._validate_consistency()
        return card

    def _validate_consistency(self) -> None:
        # `forbidden_tasks` is an explicit runtime authority layer. A card may
        # advertise broad model capabilities while still failing closed for
        # route-specific tasks such as certified frequency, MD, or free-energy
        # workflows.
        return None

    def validate_task(self, task: str) -> None:
        normalized = _normalize_task_name(task)
        if normalized in self.forbidden_tasks:
            raise ValueError(
                f"Model card {self.model_id!r} explicitly forbids task {normalized!r}."
            )
        self.capabilities.validate_task(normalized)


def read_json_compatible_yaml(path: Path) -> Mapping[str, object]:
    """Load JSON syntax from a .yaml/.yml/.json card without adding PyYAML."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ModelCardError(f"Model card not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ModelCardError(
            f"Model card at {path} must use JSON-compatible YAML syntax."
        ) from exc
    if not isinstance(payload, Mapping):
        raise ModelCardError(f"Model card payload at {path} must be a JSON object.")
    return dict(payload)


def _normalize_model_name(model_name: str) -> str:
    return str(model_name).strip().lower().replace("_", "-")


def load_model_provenance_card(
    model_name: str,
    model_card_root: Path,
) -> ModelProvenanceCard:
    normalized = _normalize_model_name(model_name)
    for extension in MODEL_CARD_EXTENSIONS:
        card_path = model_card_root / f"{normalized}{extension}"
        if card_path.is_file():
            return ModelProvenanceCard.from_payload(
                read_json_compatible_yaml(card_path),
                model_name=normalized,
            )
    raise ModelCardError(
        f"Model provenance card for model {model_name!r} was not found in {model_card_root}."
    )


@dataclass(frozen=True)
class SolventCombinationProfile:
    """Metadata emitted after a legal potential/solvent composition."""

    frequency_type: str | None = None


class CombinationValidator:
    """Validate potential, solvent, derivative, and free-energy task boundaries."""

    def __init__(
        self,
        *,
        model_capabilities: ModelCapabilities,
        implicit: str = "none",
        hessian_mode: str | None = None,
        solvation_capabilities: SolvationCapabilities | None = None,
        task: str | None = None,
    ) -> None:
        self.model_capabilities = model_capabilities
        self.implicit = str(implicit).strip().lower()
        self.hessian_mode = (
            hessian_mode.lower() if isinstance(hessian_mode, str) else None
        )
        self.solvation_capabilities = solvation_capabilities
        self.task = str(task).strip().lower() if task is not None else None

    def validate(self) -> SolventCombinationProfile | None:
        if self.task is not None:
            self.model_capabilities.validate_task(self.task)

        additive_requested = self.implicit not in {"", "none"}
        if additive_requested and self.model_capabilities.solvation_mode == "native":
            raise ValueError(
                "Native solution-phase potentials cannot be combined with an additive solvent backend."
            )
        if self.model_capabilities.solvation_mode == "property_only" and self.task:
            raise ValueError(
                "Property-only solvation predictors are benchmark baselines, not calculator backends."
            )

        if self.hessian_mode != "numerical" or not additive_requested:
            return None
        if self.solvation_capabilities is None:
            raise ValueError(
                "Numerical solvent Hessian requires explicit solvent-backend capabilities."
            )
        if not (
            self.model_capabilities.energy
            and self.model_capabilities.forces
            and self.model_capabilities.conservative_forces
            and self.model_capabilities.hessian == "finite_difference"
        ):
            raise ValueError(
                "Numerical solvent Hessian requires a conservative base model with finite-difference Hessian support."
            )
        if not (
            self.solvation_capabilities.energy
            and self.solvation_capabilities.forces
            and self.solvation_capabilities.conservative_forces
        ):
            raise ValueError(
                "Numerical solvent Hessian requires an energy-consistent solvent force backend."
            )
        return SolventCombinationProfile(
            frequency_type=CONSERVATIVE_NUMERICAL_FREQUENCY_TYPE
        )


def validate_model_card_task(card: ModelProvenanceCard, task: str) -> None:
    card.validate_task(task)
