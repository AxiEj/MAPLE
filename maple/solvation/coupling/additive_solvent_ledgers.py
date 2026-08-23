"""Exact PySCF SMD-CDS addition over the registered hybrid Phi0 scalar.

The electronic/continuum equation and root remain owned by the base ledger.
This adapter creates one distinct total-SMD scalar identity.  It accepts only
the registered ``PySCFSMDCDSTerm`` provider and never changes the base source,
receiver, continuum, cavity, or root equation.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from maple.solvation.api.scalar_registry import (
    OPERATIONAL_MACE_MDP_POLAR_HYBRID_PHI0_SMOOTH_HARMONIC_DDPCM_V2,
    OPERATIONAL_MACE_MDP_POLAR_HYBRID_SMD_TOTAL_SMOOTH_HARMONIC_DDPCM_V1,
    get_scalar_definition,
)
from maple.solvation.coupling.permanent_induced_ledgers import (
    HybridHarmonicDDPCMPhi0Ledger,
)
from maple.solvation.coupling.separated_fixed_point import SeparatedFixedPointState
from maple.solvation.coupling.separated_ledgers import (
    OperationalLedgerEvaluation,
    SeparatedOperationalGradientResult,
)
from maple.solvation.coupling.state_equation import (
    geometry_sha256,
    provider_behavior_sha256,
)
from maple.solvation.solvent_terms import (
    PYSCF_SMD_CDS_PROVIDER_ID,
    PySCFSMDCDSTerm,
    SolventEnergyState,
)

ADDITIVE_SOLVENT_OPERATIONAL_LEDGER_CONTRACT_ID = (
    "route2-additive-solvent-operational-ledger-v1"
)
_MODULE_PATH = Path(__file__).resolve()
_EXPECTED_BASE_COMPONENTS = (
    "macepolar_vacuum_energy",
    "point_permanent_gaussian_induced_smooth_harmonic_ddpcm_energy",
)
_SOLVENT_COMPONENT = "pyscf_smd_cds_energy"


def _array_rows(values: object, *, name: str) -> tuple[tuple[float, ...], ...]:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.shape[1] != 3 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite atom-by-3 array.")
    return tuple(tuple(float(value) for value in row) for row in array)


class AdditiveSolventOperationalLedger:
    """Compose hybrid Phi0 with the exact immutable PySCF SMD-CDS term."""

    __slots__ = (
        "_base",
        "_configuration_sha256",
        "_sealed",
        "_solvent_term",
        "equation",
    )

    implementation_entry_point = (
        "maple.solvation.coupling.additive_solvent_ledgers:"
        "AdditiveSolventOperationalLedger"
    )
    base_scalar_id = OPERATIONAL_MACE_MDP_POLAR_HYBRID_PHI0_SMOOTH_HARMONIC_DDPCM_V2
    scalar_id = OPERATIONAL_MACE_MDP_POLAR_HYBRID_SMD_TOTAL_SMOOTH_HARMONIC_DDPCM_V1

    def __init__(
        self,
        base_ledger: HybridHarmonicDDPCMPhi0Ledger,
        solvent_term: PySCFSMDCDSTerm,
    ) -> None:
        if not isinstance(base_ledger, HybridHarmonicDDPCMPhi0Ledger):
            raise TypeError("base_ledger must be HybridHarmonicDDPCMPhi0Ledger.")
        if base_ledger.scalar_id != self.base_scalar_id:
            raise ValueError("additive ledger received the wrong base scalar.")
        if type(solvent_term) is not PySCFSMDCDSTerm:
            raise TypeError("this exact total-SMD scalar requires PySCFSMDCDSTerm.")
        if solvent_term.provider_id != PYSCF_SMD_CDS_PROVIDER_ID:
            raise ValueError("total-SMD scalar solvent provider identity drifted.")
        definition = get_scalar_definition(self.scalar_id)
        if definition.implementation_entry_point != self.implementation_entry_point:
            raise ValueError("total scalar registry entry point does not match ledger.")
        if definition.state_equation_id != base_ledger.equation.state_equation_id:
            raise ValueError("total scalar changed the base state equation.")
        if definition.enabled or definition.admitted_capabilities.enabled_tiers:
            raise ValueError(
                "additive total scalar must remain disabled before admission."
            )
        base_ledger.configuration_sha256()
        solvent_term.configuration_sha256()
        object.__setattr__(self, "_base", base_ledger)
        object.__setattr__(self, "_solvent_term", solvent_term)
        object.__setattr__(self, "equation", base_ledger.equation)
        object.__setattr__(self, "_configuration_sha256", self._current_configuration())
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("additive solvent operational ledger is immutable.")
        object.__setattr__(self, name, value)

    @property
    def base_ledger(self) -> HybridHarmonicDDPCMPhi0Ledger:
        return self._base

    @property
    def solvent_term(self) -> PySCFSMDCDSTerm:
        return self._solvent_term

    def _current_configuration(self) -> str:
        payload = {
            "contract": ADDITIVE_SOLVENT_OPERATIONAL_LEDGER_CONTRACT_ID,
            "scalar_id": self.scalar_id,
            "registry_formula": get_scalar_definition(self.scalar_id).exact_formula,
            "implementation_entry_point": self.implementation_entry_point,
            "base_scalar_id": self.base_scalar_id,
            "base_ledger_sha256": self._base.configuration_sha256(),
            "solvent_provider_id": self._solvent_term.provider_id,
            "solvent_configuration_sha256": (self._solvent_term.configuration_sha256()),
            "solvent_behavior_sha256": provider_behavior_sha256(
                self._solvent_term,
                ("configuration_sha256", "evaluate"),
                label="additive_solvent_term",
            ),
            "implementation_sha256": hashlib.sha256(
                _MODULE_PATH.read_bytes()
            ).hexdigest(),
            "capabilities": "none",
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def configuration_sha256(self) -> str:
        current = self._current_configuration()
        if current != self._configuration_sha256:
            raise RuntimeError("additive solvent operational ledger drifted.")
        return current

    def _solvent_state(
        self, geometry: object, *, need_gradient: bool
    ) -> SolventEnergyState:
        state = self._solvent_term.evaluate(geometry, need_gradient=need_gradient)
        if not isinstance(state, SolventEnergyState):
            raise TypeError("solvent term returned an invalid state.")
        if state.provider_id != PYSCF_SMD_CDS_PROVIDER_ID:
            raise ValueError("solvent state provider differs from total-SMD scalar.")
        if state.geometry_sha256 != geometry_sha256(geometry):
            raise ValueError("solvent state geometry differs from the base scalar.")
        if state.configuration_sha256 != self._solvent_term.configuration_sha256():
            raise ValueError("solvent state configuration differs from its provider.")
        return state

    def evaluate_root(
        self,
        geometry: object,
        reduced_coordinates: object,
        *,
        root_tolerance: float,
    ) -> OperationalLedgerEvaluation:
        self.configuration_sha256()
        base = self._base.evaluate_root(
            geometry, reduced_coordinates, root_tolerance=root_tolerance
        )
        if tuple(name for name, _value in base.components_eV) != (
            _EXPECTED_BASE_COMPONENTS
        ):
            raise ValueError("base Phi0 components changed before solvent addition.")
        solvent = self._solvent_state(geometry, need_gradient=False)
        components = (*base.components_eV, (_SOLVENT_COMPONENT, solvent.energy_eV))
        return OperationalLedgerEvaluation(
            scalar_id=self.scalar_id,
            state_equation_id=base.state_equation_id,
            geometry_sha256=base.geometry_sha256,
            equation_sha256=base.equation_sha256,
            ledger_sha256=self.configuration_sha256(),
            field_semantics_sha256=base.field_semantics_sha256,
            reduced_coordinates_sha256=base.reduced_coordinates_sha256,
            root_residual_norm=base.root_residual_norm,
            root_tolerance=base.root_tolerance,
            components_eV=components,
            total_energy_eV=float(sum(value for _name, value in components)),
        )

    def solvation_energy_eV(self, evaluation: OperationalLedgerEvaluation) -> float:
        if not isinstance(evaluation, OperationalLedgerEvaluation):
            raise TypeError("evaluation must be OperationalLedgerEvaluation.")
        if (
            evaluation.scalar_id != self.scalar_id
            or evaluation.ledger_sha256 != self.configuration_sha256()
        ):
            raise ValueError("evaluation does not belong to this additive ledger.")
        components = dict(evaluation.components_eV)
        if tuple(components) != (*_EXPECTED_BASE_COMPONENTS, _SOLVENT_COMPONENT):
            raise ValueError("additive evaluation components changed.")
        return float(
            components[_EXPECTED_BASE_COMPONENTS[1]] + components[_SOLVENT_COMPONENT]
        )

    def implicit_gradient(
        self, geometry: object, state: SeparatedFixedPointState
    ) -> SeparatedOperationalGradientResult:
        self.configuration_sha256()
        base = self._base.implicit_gradient(geometry, state)
        solvent = self._solvent_state(geometry, need_gradient=True)
        if solvent.gradient_eV_per_A is None:
            raise ValueError("solvent term omitted its requested gradient.")
        evaluation = self.evaluate_root(
            geometry,
            state.y_array(),
            root_tolerance=state.primal_tolerance,
        )
        evaluated_solvent_energy = dict(evaluation.components_eV)[_SOLVENT_COMPONENT]
        if solvent.energy_eV != evaluated_solvent_energy:
            raise RuntimeError(
                "gradient and energy evaluations changed the SMD-CDS scalar."
            )
        direct = np.asarray(
            base.direct_coordinate_gradient_eV_per_angstrom, dtype=float
        ) + np.asarray(solvent.gradient_eV_per_A, dtype=float)
        residual = np.asarray(
            base.residual_coordinate_pullback_eV_per_angstrom, dtype=float
        )
        total = np.asarray(
            base.total_coordinate_gradient_eV_per_angstrom, dtype=float
        ) + np.asarray(solvent.gradient_eV_per_A, dtype=float)
        return SeparatedOperationalGradientResult(
            ledger=evaluation,
            adjoint=base.adjoint,
            direct_coordinate_gradient_eV_per_angstrom=_array_rows(
                direct, name="additive direct gradient"
            ),
            residual_coordinate_pullback_eV_per_angstrom=_array_rows(
                residual, name="additive residual pullback"
            ),
            total_coordinate_gradient_eV_per_angstrom=_array_rows(
                total, name="additive total gradient"
            ),
        )


__all__ = [
    "ADDITIVE_SOLVENT_OPERATIONAL_LEDGER_CONTRACT_ID",
    "AdditiveSolventOperationalLedger",
]
