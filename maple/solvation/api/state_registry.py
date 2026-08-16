"""Versioned state equations used by registered Route-2 scalars."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

OPERATIONAL_STATE_EQUATION_ID = "route2-constrained-mutual-polarization-root-v1"
SEPARATED_OPERATIONAL_STATE_EQUATION_ID = (
    "route2-separated-source-boundary-nativefield-root-v1"
)
MACE_MDP_POLAR_HYBRID_STATE_EQUATION_ID = (
    "route2-mace-mdp-permanent-macepolar-induced-pcmsolver-field-root-v1"
)
MACE_MDP_POLAR_HYBRID_HARMONIC_STATE_EQUATION_ID = (
    "route2-mace-mdp-permanent-macepolar-induced-harmonic-field-root-v1"
)
PURE_MACEPOLAR_FROZEN_SOURCE_STATE_EQUATION_ID = (
    "route2-pure-macepolar-zero-field-frozen-source-ddx-v1"
)
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
        if not constraints or any(
            not isinstance(item, str) or not item.strip() for item in constraints
        ):
            raise ValueError("constraints must contain non-empty strings.")
        if len(set(constraints)) != len(constraints):
            raise ValueError("state constraints must be unique.")
        object.__setattr__(self, "constraints", constraints)
        if type(self.enabled) is not bool:
            raise TypeError("enabled must be a bool.")


_STATE_ENTRIES = (
    StateEquationDefinition(
        state_equation_id=PURE_MACEPOLAR_FROZEN_SOURCE_STATE_EQUATION_ID,
        exact_formula=(
            "c0(R)=M_MACE-POLAR(R,u=0); s_ddX(R)=Solve_ddX[R,c0(R)]; "
            "there is no coupled ML/continuum root"
        ),
        coordinates=(
            "geometry R, deterministic zero-field MACE-POLAR source c0(R), "
            "and the uniquely solved linear ddX response state"
        ),
        constraints=(
            "pure MACE-POLAR provider and no MACE-MDP source",
            "the external MACE-POLAR receiver field is exactly zero",
            "the declared ddX linear response solve is unique and deterministic",
            "the additive solvent term is geometry-only and separately identified",
            "no mutual ML/continuum fixed point or common-functional claim",
        ),
    ),
    StateEquationDefinition(
        state_equation_id=OPERATIONAL_STATE_EQUATION_ID,
        exact_formula=(
            "r(R,y)=T^+[c_ref(R)+T(R)y-Pi_q M_theta(R,P_R(c_ref(R)+T(R)y))]=0"
        ),
        coordinates="dimensionless reduced source coordinates y",
        constraints=("A c = q_tot", "A T = 0", "unique smooth admitted root"),
    ),
    StateEquationDefinition(
        state_equation_id=SEPARATED_OPERATIONAL_STATE_EQUATION_ID,
        exact_formula=(
            "c=c_ref+Ty; A_R sigma=B_R c; u=L_R sigma; "
            "r(R,y)=y-T_plus[Pi_q M_theta(R,u)-c_ref]=0"
        ),
        coordinates="dimensionless reduced original-source coordinates y",
        constraints=(
            "source C, continuum Sigma, and native field U are distinct spaces",
            "A c = q_tot and A T = 0",
            "unique smooth admitted root",
            "no operational assertion that L_R equals B_R adjoint",
        ),
    ),
    StateEquationDefinition(
        state_equation_id=MACE_MDP_POLAR_HYBRID_STATE_EQUATION_ID,
        exact_formula=(
            "c_perm=M_MDP(R); delta_c(u)=M_POLAR(R,u)-M_POLAR(R,0); "
            "v=B_point(R)c_perm+B_GTO1p5(R)delta_c(u); "
            "q=Q_PCMSolver(R)v; r(R,u)=u-L_radial(R)q=0"
        ),
        coordinates="MACE-POLAR native two-width radial receiver field u",
        constraints=(
            "neutral singlet and fixed total charge",
            "MACE-MDP permanent point kernel remains distinct from induced GTO kernel",
            "symmetric content-addressed PCMSolver external-MEP response",
            "two deterministic starts agree and the final residual is below tolerance",
            "numerical scalar-gradient force rebuilds the cavity and resolves the root at every stencil point",
            "no analytic coordinate derivative, Hessian, MD, or strict common-functional claim",
        ),
    ),
    StateEquationDefinition(
        state_equation_id=MACE_MDP_POLAR_HYBRID_HARMONIC_STATE_EQUATION_ID,
        exact_formula=(
            "c_perm=M_MDP(R); delta_c(u)=M_POLAR(R,u)-M_POLAR(R,0); "
            "b=B_point^harm(R)c_perm+B_GTO1p5^harm(R)delta_c(u); "
            "A_harm(R)sigma=b; r(R,u)=u+S8_harm(R)^T sigma=0"
        ),
        coordinates="MACE-POLAR native two-width radial receiver field u",
        constraints=(
            "neutral singlet and fixed total charge",
            "MACE-MDP permanent point kernel remains distinct from induced GTO kernel",
            "fixed-dimensional smooth weighted harmonic coefficient topology",
            "two deterministic starts agree and the final residual is below tolerance",
            "numerical scalar-gradient force rebuilds the full harmonic scalar at every stencil point",
            "no Hessian, MD, or strict common-functional claim",
        ),
        enabled=True,
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
        raise KeyError(
            f"Unregistered Route-2 state equation: {state_equation_id!r}."
        ) from exc


__all__ = [
    "MACE_MDP_POLAR_HYBRID_HARMONIC_STATE_EQUATION_ID",
    "MACE_MDP_POLAR_HYBRID_STATE_EQUATION_ID",
    "OPERATIONAL_STATE_EQUATION_ID",
    "PURE_MACEPOLAR_FROZEN_SOURCE_STATE_EQUATION_ID",
    "SEPARATED_OPERATIONAL_STATE_EQUATION_ID",
    "STATE_REGISTRY",
    "VARIATIONAL_STATE_EQUATION_ID",
    "StateEquationDefinition",
    "get_state_equation",
]
