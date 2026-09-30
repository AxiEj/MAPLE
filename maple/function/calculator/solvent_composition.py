"""Model-neutral orchestration for composing gas and solvent results.

The physics and typed algebra live in the solvent provider and
``analytic_protocol`` respectively.  This module only validates an explicit
operation, requests the matching solvent operation once, and returns detached
values for a calculator wrapper to commit transactionally.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from .analytic_protocol import (
    AnalyticCapabilityError,
    AnalyticCompositionAdmission,
    AnalyticDerivativeAdmission,
    AnalyticIdentityError,
    AnalyticOperationResult,
    ScalarPointIdentity,
    compose_energy_forces,
    compose_hessian,
    compose_hvp,
    identity_record,
)


_EV2HARTREE = 1.0 / 27.211386245988
_IMPLICIT_SOLVENT_FORCE_ERROR = (
    "The selected implicit-solvent provider does not supply the requested "
    "energy-consistent derivative."
)


def _mapping_copy(value: Mapping[str, Any] | None, *, name: str) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping.")
    return dict(value)


def _gas_point(
    operation: AnalyticOperationResult,
    *,
    expected: str,
    gas_admission: AnalyticDerivativeAdmission | None,
) -> ScalarPointIdentity:
    if not isinstance(operation, AnalyticOperationResult):
        raise TypeError(f"gas_operation must be an AnalyticOperationResult for {expected}.")
    if operation.operation != expected:
        raise ValueError(
            f"Expected analytic operation {expected!r}, got {operation.operation!r}."
        )
    if not isinstance(operation.identity, ScalarPointIdentity):
        raise AnalyticIdentityError(
            "An already composed analytic result cannot be reused as a gas component."
        )
    if operation.identity.cell.component != "gas":
        raise AnalyticIdentityError("Analytic gas operation has a non-gas identity.")
    if gas_admission is not None:
        if not isinstance(gas_admission, AnalyticDerivativeAdmission):
            raise TypeError("gas_admission must be an AnalyticDerivativeAdmission.")
        gas_admission.require(operation.identity, expected)
    return operation.identity


def _strict_solvent_point(
    atoms,
    *,
    gas_point: ScalarPointIdentity,
    operation: str,
    correction,
    composition_admission: AnalyticCompositionAdmission,
    gas_admission: AnalyticDerivativeAdmission | None,
) -> ScalarPointIdentity:
    if not isinstance(composition_admission, AnalyticCompositionAdmission):
        raise TypeError(
            "composition_admission must be an AnalyticCompositionAdmission."
        )
    if gas_admission != composition_admission.gas_admission:
        raise AnalyticIdentityError(
            "The live gas admission drifted from the composition profile."
        )
    if (
        getattr(correction, "analytic_derivative_admission", None)
        != composition_admission.solvent_admission
    ):
        raise AnalyticIdentityError(
            "The live solvent admission drifted from the composition profile."
        )
    identity_fn = getattr(correction, "analytic_scalar_identity", None)
    if not callable(identity_fn):
        raise AnalyticCapabilityError(
            "The strict solvent provider does not expose analytic_scalar_identity()."
        )
    solvent_point = identity_fn(atoms)
    if not isinstance(solvent_point, ScalarPointIdentity):
        raise TypeError("analytic_scalar_identity() must return ScalarPointIdentity.")
    composition_admission.require(gas_point, solvent_point, operation)
    required_property = {
        "energy_forces": "forces",
        "hessian": "hessian",
        "hvp": "hvp",
    }[operation]
    if required_property not in set(
        getattr(correction, "supported_properties", {"energy"})
    ):
        raise AnalyticCapabilityError(
            f"The solvent provider does not advertise {required_property!r}."
        )
    return solvent_point


def _convert_energy_forces(energy, forces, *, source_unit: str):
    if source_unit == "hartree":
        return energy, forces
    if source_unit == "eV":
        return (
            energy * _EV2HARTREE,
            None if forces is None else forces * _EV2HARTREE,
        )
    raise ValueError(
        f"Unknown source_unit: {source_unit!r}; expected 'eV' or 'hartree'."
    )


def _structured_solvation(
    solvent_result,
    *,
    gas_energy_hartree: float,
    combined_energy_hartree: float,
    precision_provenance: Mapping[str, Any] | None,
) -> dict[str, Any]:
    provenance = dict(solvent_result.provenance)
    structured: dict[str, Any] = {
        "energy_hartree": float(solvent_result.energy_hartree),
        "gas_energy_hartree": gas_energy_hartree,
        "combined_energy_hartree": combined_energy_hartree,
        "components_hartree": dict(solvent_result.components_hartree),
        "provenance": provenance,
        "ase_free_energy_is_thermochemical_gibbs": False,
    }
    if precision_provenance is not None:
        structured["gas_inference_precision"] = dict(precision_provenance)
    if provenance.get("reference_only") is True:
        structured["reference_correction_hartree"] = float(
            solvent_result.energy_hartree
        )
    elif provenance.get("absolute_solvation_free_energy_claim") is False:
        structured["cluster_continuum_correction_hartree"] = float(
            solvent_result.energy_hartree
        )
    else:
        structured["delta_g_solv_hartree"] = float(solvent_result.energy_hartree)
    return structured


def finalize_energy_forces(
    atoms,
    *,
    correction=None,
    energy=None,
    forces=None,
    source_unit: str | None = None,
    gas_operation: AnalyticOperationResult | None = None,
    raw_hessian=None,
    composition_admission: AnalyticCompositionAdmission | None = None,
    gas_admission: AnalyticDerivativeAdmission | None = None,
    execution_provenance: Mapping[str, Any] | None = None,
    precision_provenance: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], Any | None]:
    """Return a fresh ASE result payload and its structured solvent result.

    Raw gas values use ``source_unit``.  A typed gas operation is already in
    Hartree and cannot be mixed with raw values.  ``raw_hessian`` is retained
    solely for the legacy gas-only finalization contract; solvent Hessians use
    :func:`compose_analytic_hessian`.
    """
    _mapping_copy(execution_provenance, name="execution_provenance")
    precision = _mapping_copy(
        precision_provenance, name="precision_provenance"
    )
    gas_point = None
    if gas_operation is not None:
        if energy is not None or forces is not None or source_unit is not None:
            raise ValueError(
                "Typed gas energy/forces cannot be mixed with raw energy, forces, or units."
            )
        gas_point = _gas_point(
            gas_operation,
            expected="energy_forces",
            gas_admission=gas_admission,
        )
        energy_hartree = float(gas_operation.energy_hartree)
        forces_hartree = gas_operation.forces_hartree_per_angstrom
    else:
        if energy is None:
            raise ValueError("Raw result finalization requires an energy value.")
        if source_unit is None:
            raise ValueError("Raw result finalization requires source_unit.")
        if composition_admission is not None or gas_admission is not None:
            raise ValueError("Raw gas values cannot carry analytic admissions.")
        energy_hartree, forces_hartree = _convert_energy_forces(
            energy, forces, source_unit=source_unit
        )

    if raw_hessian is not None and correction is not None:
        raise NotImplementedError(_IMPLICIT_SOLVENT_FORCE_ERROR)
    gas_energy_hartree = float(energy_hartree)
    solvent_result = None
    composite_identity = None

    if correction is None:
        if composition_admission is not None:
            raise AnalyticCapabilityError(
                "A composition admission cannot be used without a solvent correction."
            )
    else:
        evaluate = getattr(correction, "evaluate", None)
        if not callable(evaluate):
            raise RuntimeError(
                "Unstructured legacy implicit-solvent corrections are no longer "
                "supported; attach an ImplicitSolvationCorrection."
            )
        strict_solvent_point = None
        if composition_admission is not None:
            if gas_point is None:
                raise ValueError(
                    "An exact composition admission requires a typed gas operation."
                )
            strict_solvent_point = _strict_solvent_point(
                atoms,
                gas_point=gas_point,
                operation="energy_forces",
                correction=correction,
                composition_admission=composition_admission,
                gas_admission=gas_admission,
            )
        elif forces_hartree is not None and "forces" not in set(
            getattr(correction, "supported_properties", {"energy"})
        ):
            raise AnalyticCapabilityError(
                "The solvent provider does not advertise 'forces'."
            )

        # No supported provider consumes a gas-calculator owner.  Omitting it
        # keeps this boundary model-neutral and prevents hidden owner coupling.
        solvent_result = evaluate(
            atoms,
            need_forces=forces_hartree is not None,
        )
        if strict_solvent_point is not None:
            composed = compose_energy_forces(
                gas_operation, solvent_result, strict_solvent_point
            )
            energy_hartree = composed.energy_hartree
            forces_hartree = composed.forces_hartree_per_angstrom
            composite_identity = composed.identity
        else:
            energy_hartree = energy_hartree + float(solvent_result.energy_hartree)
            if forces_hartree is not None:
                if solvent_result.forces_hartree_per_angstrom is None:
                    raise AnalyticCapabilityError(_IMPLICIT_SOLVENT_FORCE_ERROR)
                forces_hartree = np.asarray(forces_hartree) + np.asarray(
                    solvent_result.forces_hartree_per_angstrom
                )

    payload: dict[str, Any] = {
        "energy": float(energy_hartree),
        "free_energy": float(energy_hartree),
    }
    if forces_hartree is not None:
        payload["forces"] = forces_hartree
    if raw_hessian is not None:
        payload["hessian"] = raw_hessian
    if composite_identity is not None:
        payload["analytic_composition_identity"] = identity_record(
            composite_identity
        )
    if precision is not None:
        payload["inference_precision"] = precision
    if solvent_result is not None:
        payload["solvation"] = _structured_solvation(
            solvent_result,
            gas_energy_hartree=gas_energy_hartree,
            combined_energy_hartree=float(energy_hartree),
            precision_provenance=precision,
        )
    return payload, solvent_result


def _operation_provenance(
    *,
    gas_point: ScalarPointIdentity,
    composed_identity=None,
    composition_admission: AnalyticCompositionAdmission | None,
    operation_provenance: Mapping[str, Any] | None,
) -> dict[str, Any]:
    operation = _mapping_copy(operation_provenance, name="operation_provenance")
    if composition_admission is None:
        provenance: dict[str, Any] = {
            "composition": "gas-analytic",
            "gas": identity_record(gas_point),
        }
    else:
        provenance = {
            "composition": composition_admission.profile_id,
            "evidence_id": composition_admission.evidence_id,
            "identity": identity_record(composed_identity),
        }
    if operation is not None:
        provenance["operation"] = operation
    return provenance


def compose_analytic_hessian(
    atoms,
    *,
    gas_operation: AnalyticOperationResult,
    correction=None,
    composition_admission: AnalyticCompositionAdmission | None = None,
    gas_admission: AnalyticDerivativeAdmission | None = None,
    operation_provenance: Mapping[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Compose one typed gas Hessian with exactly one solvent Hessian."""
    operation_record = _mapping_copy(
        operation_provenance, name="operation_provenance"
    )
    gas_point = _gas_point(
        gas_operation, expected="hessian", gas_admission=gas_admission
    )
    if correction is None:
        if composition_admission is not None:
            raise AnalyticCapabilityError(
                "A composition admission cannot be used without a solvent correction."
            )
        value = gas_operation.hessian_hartree_per_angstrom2
        identity = None
    else:
        if composition_admission is None:
            raise AnalyticCapabilityError(
                "Analytic solvent Hessian composition requires an exact admission."
            )
        solvent_point = _strict_solvent_point(
            atoms,
            gas_point=gas_point,
            operation="hessian",
            correction=correction,
            composition_admission=composition_admission,
            gas_admission=gas_admission,
        )
        get_hessian = getattr(correction, "get_hessian", None)
        if not callable(get_hessian):
            raise AnalyticCapabilityError(
                "The admitted solvent provider has no analytic Hessian entry point."
            )
        composed = compose_hessian(gas_operation, get_hessian(atoms), solvent_point)
        value = composed.hessian_hartree_per_angstrom2
        identity = composed.identity
    return value, _operation_provenance(
        gas_point=gas_point,
        composed_identity=identity,
        composition_admission=composition_admission,
        operation_provenance=operation_record,
    )


def compose_direct_hvp(
    atoms,
    direction,
    *,
    gas_operation: AnalyticOperationResult,
    correction=None,
    composition_admission: AnalyticCompositionAdmission | None = None,
    gas_admission: AnalyticDerivativeAdmission | None = None,
    operation_provenance: Mapping[str, Any] | None = None,
) -> tuple[tuple[np.ndarray, np.ndarray, float], dict[str, Any]]:
    """Compose one typed direct HVP without constructing a dense Hessian."""
    operation_record = _mapping_copy(
        operation_provenance, name="operation_provenance"
    )
    gas_point = _gas_point(gas_operation, expected="hvp", gas_admission=gas_admission)
    requested_direction = np.asarray(direction, dtype=np.float64).reshape(-1)
    if not np.array_equal(requested_direction, np.asarray(gas_operation.direction)):
        raise AnalyticIdentityError(
            "The gas HVP direction does not match the requested direction."
        )
    if correction is None:
        if composition_admission is not None:
            raise AnalyticCapabilityError(
                "A composition admission cannot be used without a solvent correction."
            )
        operation = gas_operation
        identity = None
    else:
        if composition_admission is None:
            raise AnalyticCapabilityError(
                "Analytic solvent HVP composition requires an exact admission."
            )
        solvent_point = _strict_solvent_point(
            atoms,
            gas_point=gas_point,
            operation="hvp",
            correction=correction,
            composition_admission=composition_admission,
            gas_admission=gas_admission,
        )
        directional = getattr(correction, "get_directional_derivatives", None)
        if not callable(directional):
            raise AnalyticCapabilityError(
                "The attached solvent correction has no direct-HVP entry point."
            )
        operation = compose_hvp(
            gas_operation,
            directional(atoms, requested_direction),
            solvent_point,
        )
        identity = operation.identity
    result = (
        operation.hvp_hartree_per_angstrom2,
        operation.forces_hartree_per_angstrom.reshape(-1),
        float(operation.energy_hartree),
    )
    return result, _operation_provenance(
        gas_point=gas_point,
        composed_identity=identity,
        composition_admission=composition_admission,
        operation_provenance=operation_record,
    )


__all__ = [
    "compose_analytic_hessian",
    "compose_direct_hvp",
    "finalize_energy_forces",
]
