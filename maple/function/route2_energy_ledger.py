"""Versioned electrostatic-energy ledgers for experimental Route 2.

The profile owns this choice so an input cannot silently exchange a legacy
MACE-field-energy term for a PCM polarization term after a benchmark has been
run.  Neither ledger turns the legacy MACE-POLAR fixed point into a common
stationary electronic functional; that remains a separate V0/KKT gate.
"""

from __future__ import annotations

from typing import Literal

Route2ElectrostaticEnergyLedger = Literal[
    "legacy-mace-field-energy-plus-pcm-v1",
    "pcm-half-coupling-only-v1",
]

LEGACY_MACE_FIELD_ENERGY_PLUS_PCM_V1: Route2ElectrostaticEnergyLedger = (
    "legacy-mace-field-energy-plus-pcm-v1"
)
PCM_HALF_COUPLING_ONLY_V1: Route2ElectrostaticEnergyLedger = "pcm-half-coupling-only-v1"
SUPPORTED_ROUTE2_ELECTROSTATIC_ENERGY_LEDGERS = frozenset(
    {
        LEGACY_MACE_FIELD_ENERGY_PLUS_PCM_V1,
        PCM_HALF_COUPLING_ONLY_V1,
    }
)


def validate_route2_electrostatic_energy_ledger(
    ledger: str,
) -> Route2ElectrostaticEnergyLedger:
    """Return one registered ledger or fail closed."""

    normalized = str(ledger).strip().lower()
    if normalized not in SUPPORTED_ROUTE2_ELECTROSTATIC_ENERGY_LEDGERS:
        supported = ", ".join(sorted(SUPPORTED_ROUTE2_ELECTROSTATIC_ENERGY_LEDGERS))
        raise ValueError(
            "Unsupported Route-2 electrostatic energy ledger: "
            f"{ledger!r}; expected one of: {supported}."
        )
    return normalized  # type: ignore[return-value]


def route2_energy_composition_description(
    ledger: str,
    *,
    continuum_symbol: str,
) -> str:
    """Describe the selected leaf-only electrostatic accounting rule."""

    selected = validate_route2_electrostatic_energy_ledger(ledger)
    if selected == PCM_HALF_COUPLING_ONLY_V1:
        return (
            "delta_G_solv = 0.5*<c_MACE-POLAR, f_reac_PCM> "
            f"(= E_{continuum_symbol}) + G_CDS; "
            "E_MACE_intrinsic[V_reac]-E_MACE_gas is recorded only as a "
            "diagnostic and is excluded from the reported ledger"
        )
    return (
        "delta_G_solv = (E_MACE_intrinsic[V_reac]-E_MACE_gas) "
        f"+ E_{continuum_symbol} + G_CDS"
    )


__all__ = [
    "LEGACY_MACE_FIELD_ENERGY_PLUS_PCM_V1",
    "PCM_HALF_COUPLING_ONLY_V1",
    "SUPPORTED_ROUTE2_ELECTROSTATIC_ENERGY_LEDGERS",
    "Route2ElectrostaticEnergyLedger",
    "route2_energy_composition_description",
    "validate_route2_electrostatic_energy_ledger",
]
