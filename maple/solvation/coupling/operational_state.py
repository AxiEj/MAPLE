"""Fail-closed assembly of one registered operational Route-2 scalar.

This module joins the generic model adapters, a scalar-first continuum, and
the charge-constrained reduced equation.  It does not admit a capability or
publish an ASE calculator.  In particular, the electronic source may remain
the original response head: only the continuum drive is required to come from
the same continuum scalar used by :class:`OperationalElectrostaticScalar`.
"""

from __future__ import annotations

from maple.solvation.api.profiles import get_solvation_profile
from maple.solvation.api.scalar_registry import get_scalar_definition
from maple.solvation.api.state_registry import OPERATIONAL_STATE_EQUATION_ID
from maple.solvation.continuum.functional import ContinuumEnergyFunctional
from maple.solvation.models.base import (
    FieldResponsiveModel,
    atom_count,
    model_charge_and_multiplicity,
    validate_model_identity,
    validate_source_model_identity,
)
from maple.solvation.models.equation_adapter import (
    ElectronicResponseEquationAdapter,
    VacuumScalarEquationAdapter,
)

from .energy import OperationalElectrostaticScalar
from .metrics import get_pairing_metric
from .spaces import get_coordinate_contract
from .state_equation import ReducedStateEquation
from .variational_adapters import ScalarFirstContinuumResponseAdapter


def build_disabled_operational_electrostatic_scalar(
    model: FieldResponsiveModel,
    continuum: ContinuumEnergyFunctional,
    geometry: object,
    *,
    scalar_id: str,
    profile_id: str,
) -> OperationalElectrostaticScalar:
    """Bind one disabled operational scalar to its exact registry identities.

    The returned object evaluates

    ``E_vac(R) + G_cont(R, c*(R))``

    with ``c*=M(R, grad_Q G_cont(R,c*))``.  Since the continuum is linear and
    reciprocal, ``G_cont=0.5<c,grad_Q G_cont>_Q`` and the existing implicit
    adjoint differentiates this single operational PES.  The function rejects
    enabled/admitted registry entries so construction cannot silently become a
    public capability.
    """

    if not isinstance(continuum, ContinuumEnergyFunctional):
        raise TypeError("continuum must be a ContinuumEnergyFunctional.")
    profile = get_solvation_profile(profile_id)
    definition = get_scalar_definition(scalar_id)
    if profile.scalar_id != scalar_id:
        raise ValueError("profile does not bind the requested scalar.")
    if (
        profile.state_equation_id != OPERATIONAL_STATE_EQUATION_ID
        or definition.state_equation_id != OPERATIONAL_STATE_EQUATION_ID
    ):
        raise ValueError("builder accepts only the operational state equation.")
    if (
        profile.enabled
        or profile.capabilities.enabled_tiers
        or definition.enabled
        or definition.admitted_capabilities.enabled_tiers
    ):
        raise ValueError(
            "this internal operational assembly requires closed capabilities."
        )

    source_provenance = validate_source_model_identity(model)
    vacuum_provenance = validate_model_identity(model)
    if source_provenance.sha256 != vacuum_provenance.sha256:
        raise ValueError("vacuum and response must use one model provenance.")
    source_provenance.domain.validate_atoms(geometry)
    count = atom_count(geometry)
    total_charge = float(model_charge_and_multiplicity(geometry)[0])
    coordinates = get_coordinate_contract(profile.coordinate_contract_id).build(
        count, total_charge
    )
    electronic = ElectronicResponseEquationAdapter(model, profile.coupling_id)
    response = ScalarFirstContinuumResponseAdapter(continuum)
    equation = ReducedStateEquation(
        coordinates,
        electronic,
        response,
        state_equation_id=OPERATIONAL_STATE_EQUATION_ID,
    )
    return OperationalElectrostaticScalar(
        equation=equation,
        vacuum=VacuumScalarEquationAdapter(model),
        scalar_id=scalar_id,
        profile_id=profile_id,
        metric=get_pairing_metric(profile.pairing_id),
    )


__all__ = ["build_disabled_operational_electrostatic_scalar"]
