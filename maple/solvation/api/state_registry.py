"""Versioned state equations used by registered Route-2 scalars."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping


OPERATIONAL_STATE_EQUATION_ID = "route2-constrained-mutual-polarization-root-v1"
VARIATIONAL_STATE_EQUATION_ID = "route2-common-functional-stationarity-v1"


@dataclass(frozen=True, slots=True)
class StateEquationDefinition:
    state_equation_id: str
    exact_formula: str
    coordinates: str
    constraints: tuple[str, ...]
    enabled: bool = False

    def __post_init__(self) -> None:
        for name in ("state_equation_id", "exact_formula", "coordinates"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string.")
        constraints = tuple(self.constraints)
        if not constraints or any(not isinstance(item, str) or not item.strip() for item in constraints):
            raise ValueError("constraints must contain non-empty strings.")
        if len(set(constraints)) != len(constraints):
            raise ValueError("state constraints must be unique.")
        object.__setattr__(self, "constraints", constraints)
        if type(self.enabled) is not bool:
            raise TypeError("enabled must be a bool.")


_STATE_ENTRIES = (
    StateEquationDefinition(
        state_equation_id=OPERATIONAL_STATE_EQUATION_ID,
        exact_formula=(
            "r(R,y)=T^+[c_ref(R)+T(R)y-Pi_q M_theta(R,P_R(c_ref(R)+T(R)y))]=0"
        ),
        coordinates="dimensionless reduced source coordinates y",
        constraints=("A c = q_tot", "A T = 0", "unique smooth admitted root"),
    ),
    StateEquationDefinition(
        state_equation_id=VARIATIONAL_STATE_EQUATION_ID,
        exact_formula="D_c F_var(R,c)=0 with c=M_theta(R,u) and u=P_R(c)",
        coordinates="charge-constrained source c and energy-dual field u",
        constraints=(
            "A c = q_tot",
            "field-energy/source identity",
            "reciprocity, stability, local invertibility, and complete coordinate derivative",
        ),
    ),
)

STATE_REGISTRY: Mapping[str, StateEquationDefinition] = MappingProxyType(
    {entry.state_equation_id: entry for entry in _STATE_ENTRIES}
)


def get_state_equation(state_equation_id: str) -> StateEquationDefinition:
    try:
        return STATE_REGISTRY[state_equation_id]
    except KeyError as exc:
        raise KeyError(f"Unregistered Route-2 state equation: {state_equation_id!r}.") from exc


__all__ = [
    "OPERATIONAL_STATE_EQUATION_ID",
    "STATE_REGISTRY",
    "VARIATIONAL_STATE_EQUATION_ID",
    "StateEquationDefinition",
    "get_state_equation",
]
