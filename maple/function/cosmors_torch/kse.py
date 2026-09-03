"""Provider-neutral kinetic-solvent-effect arithmetic for fixed structures."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence

GAS_CONSTANT_KCAL_PER_MOL_K = 1.98720425864083e-3


@dataclass(frozen=True, slots=True)
class SolvationFreeEnergy:
    species: str
    solvent: str
    delta_g_solvation_kcal_mol: float
    temperature_k: float
    standard_state: str
    provider_identity: str
    stoichiometric_coefficient: float = 1.0

    def __post_init__(self) -> None:
        for name in ("species", "solvent", "standard_state", "provider_identity"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} must be non-empty.")
        for name in (
            "delta_g_solvation_kcal_mol",
            "temperature_k",
            "stoichiometric_coefficient",
        ):
            if not math.isfinite(float(getattr(self, name))):
                raise ValueError(f"{name} must be finite.")
        if self.temperature_k <= 0.0 or self.stoichiometric_coefficient <= 0.0:
            raise ValueError(
                "Temperature and stoichiometric coefficient must be positive."
            )


@dataclass(frozen=True, slots=True)
class ActivationSolvationFreeEnergy:
    solvent: str
    temperature_k: float
    standard_state: str
    provider_identity: str
    transition_state: SolvationFreeEnergy
    reactants: tuple[SolvationFreeEnergy, ...]
    delta_g_activation_solvation_kcal_mol: float

    @classmethod
    def from_states(
        cls,
        transition_state: SolvationFreeEnergy,
        reactants: Sequence[SolvationFreeEnergy],
    ) -> "ActivationSolvationFreeEnergy":
        terms = tuple(reactants)
        if not terms:
            raise ValueError("At least one reactant solvation energy is required.")
        identity = (
            transition_state.solvent,
            transition_state.temperature_k,
            transition_state.standard_state,
            transition_state.provider_identity,
        )
        for reactant in terms:
            observed = (
                reactant.solvent,
                reactant.temperature_k,
                reactant.standard_state,
                reactant.provider_identity,
            )
            if observed != identity:
                raise ValueError(
                    "TS and reactants must share solvent, temperature, standard "
                    "state, and provider identity."
                )
        value = transition_state.delta_g_solvation_kcal_mol - sum(
            reactant.stoichiometric_coefficient * reactant.delta_g_solvation_kcal_mol
            for reactant in terms
        )
        return cls(
            solvent=identity[0],
            temperature_k=identity[1],
            standard_state=identity[2],
            provider_identity=identity[3],
            transition_state=transition_state,
            reactants=terms,
            delta_g_activation_solvation_kcal_mol=value,
        )


@dataclass(frozen=True, slots=True)
class RelativeKineticSolventEffect:
    target: ActivationSolvationFreeEnergy
    reference: ActivationSolvationFreeEnergy
    delta_delta_g_activation_solvation_kcal_mol: float
    ln_target_over_reference_rate: float
    log10_target_over_reference_rate: float
    target_over_reference_rate: float | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "target_solvent": self.target.solvent,
            "reference_solvent": self.reference.solvent,
            "temperature_k": self.target.temperature_k,
            "standard_state": self.target.standard_state,
            "provider_identity": self.target.provider_identity,
            "target_delta_g_activation_solvation_kcal_mol": (
                self.target.delta_g_activation_solvation_kcal_mol
            ),
            "reference_delta_g_activation_solvation_kcal_mol": (
                self.reference.delta_g_activation_solvation_kcal_mol
            ),
            "delta_delta_g_activation_solvation_kcal_mol": (
                self.delta_delta_g_activation_solvation_kcal_mol
            ),
            "ln_target_over_reference_rate": self.ln_target_over_reference_rate,
            "log10_target_over_reference_rate": (self.log10_target_over_reference_rate),
            "target_over_reference_rate": self.target_over_reference_rate,
            "reaction": {
                "transition_state": self.target.transition_state.species,
                "reactants": [
                    {
                        "species": item.species,
                        "stoichiometric_coefficient": (item.stoichiometric_coefficient),
                    }
                    for item in self.target.reactants
                ],
            },
        }


def _reaction_identity(
    value: ActivationSolvationFreeEnergy,
) -> tuple[str, tuple[tuple[str, float], ...]]:
    return (
        value.transition_state.species,
        tuple(
            sorted(
                (
                    item.species,
                    item.stoichiometric_coefficient,
                )
                for item in value.reactants
            )
        ),
    )


def compute_relative_kinetic_solvent_effect(
    target: ActivationSolvationFreeEnergy,
    reference: ActivationSolvationFreeEnergy,
) -> RelativeKineticSolventEffect:
    """Apply transition-state theory to two solvent activation corrections."""

    if _reaction_identity(target) != _reaction_identity(reference):
        raise ValueError("Target and reference must describe the same reaction.")
    if target.temperature_k != reference.temperature_k:
        raise ValueError("Target and reference temperatures must match.")
    if target.standard_state != reference.standard_state:
        raise ValueError("Target and reference standard states must match.")
    if target.provider_identity != reference.provider_identity:
        raise ValueError("Target and reference provider identities must match.")
    delta_delta = (
        target.delta_g_activation_solvation_kcal_mol
        - reference.delta_g_activation_solvation_kcal_mol
    )
    ln_ratio = -delta_delta / (GAS_CONSTANT_KCAL_PER_MOL_K * target.temperature_k)
    log10_ratio = ln_ratio / math.log(10.0)
    ratio = math.exp(ln_ratio) if abs(ln_ratio) < 700.0 else None
    return RelativeKineticSolventEffect(
        target=target,
        reference=reference,
        delta_delta_g_activation_solvation_kcal_mol=delta_delta,
        ln_target_over_reference_rate=ln_ratio,
        log10_target_over_reference_rate=log10_ratio,
        target_over_reference_rate=ratio,
    )


def _state_from_dict(
    payload: dict[str, Any],
    *,
    solvent: str,
    temperature_k: float,
    standard_state: str,
    provider_identity: str,
) -> SolvationFreeEnergy:
    return SolvationFreeEnergy(
        species=str(payload["species"]),
        solvent=solvent,
        delta_g_solvation_kcal_mol=float(payload["delta_g_solvation_kcal_mol"]),
        temperature_k=temperature_k,
        standard_state=standard_state,
        provider_identity=provider_identity,
        stoichiometric_coefficient=float(
            payload.get("stoichiometric_coefficient", 1.0)
        ),
    )


def _activation_from_dict(
    payload: dict[str, Any],
    *,
    temperature_k: float,
    standard_state: str,
    provider_identity: str,
) -> ActivationSolvationFreeEnergy:
    solvent = str(payload["solvent"])
    transition_state = _state_from_dict(
        dict(payload["transition_state"]),
        solvent=solvent,
        temperature_k=temperature_k,
        standard_state=standard_state,
        provider_identity=provider_identity,
    )
    reactants = tuple(
        _state_from_dict(
            dict(item),
            solvent=solvent,
            temperature_k=temperature_k,
            standard_state=standard_state,
            provider_identity=provider_identity,
        )
        for item in payload["reactants"]
    )
    return ActivationSolvationFreeEnergy.from_states(
        transition_state,
        reactants,
    )


def evaluate_kse_payload(payload: dict[str, Any]) -> dict[str, Any]:
    temperature = float(payload["temperature_k"])
    standard_state = str(payload["standard_state"])
    provider = str(payload["provider_identity"])
    target = _activation_from_dict(
        dict(payload["target"]),
        temperature_k=temperature,
        standard_state=standard_state,
        provider_identity=provider,
    )
    reference = _activation_from_dict(
        dict(payload["reference"]),
        temperature_k=temperature,
        standard_state=standard_state,
        provider_identity=provider,
    )
    result = compute_relative_kinetic_solvent_effect(target, reference).as_dict()
    canonical_input = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    result["input_sha256"] = hashlib.sha256(canonical_input).hexdigest()
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compute a relative kinetic solvent effect from COSMO-RS energies."
    )
    parser.add_argument("input_json")
    parser.add_argument("output_json")
    args = parser.parse_args(argv)
    input_path = Path(args.input_json).expanduser().resolve()
    output_path = Path(args.output_json).expanduser().resolve()
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    result = evaluate_kse_payload(payload)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ActivationSolvationFreeEnergy",
    "GAS_CONSTANT_KCAL_PER_MOL_K",
    "RelativeKineticSolventEffect",
    "SolvationFreeEnergy",
    "compute_relative_kinetic_solvent_effect",
    "evaluate_kse_payload",
    "main",
]
