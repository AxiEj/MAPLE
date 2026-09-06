"""Zero-training MACE-MDP polarizability-weighted dipole closure.

MACE-MDP exposes a molecular dipole, a molecular polarizability, and atomwise
polarizability contributions whose sum is the molecular tensor.  For a source
provided by another frozen checkpoint, the unique *uniform linear-response*
correction represented by those tensors is

``E = alpha_total^{-1} (mu_target - mu_source)``
``delta p_A = alpha_A E``.

The construction changes no atomic charge, closes the molecular dipole exactly,
and introduces no fitted parameter.  Individual ``alpha_A`` contributions are
not interpreted as positive atomic response tensors; only their exact additive
identity is used.  This is a diagnostic source composition, not a trained model
or a public scientific capability.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math

import numpy as np

_CONTRACT = "route2-mdp-polarizability-dipole-closure-v1"


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values, dtype="<f8")
    return hashlib.sha256(
        f"{array.dtype.str}:{array.shape}".encode() + b"\0" + array.tobytes(order="C")
    ).hexdigest()


def _readonly(values: object, *, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite with shape {shape}.")
    result = np.ascontiguousarray(array, dtype=np.float64)
    result.setflags(write=False)
    return result


def _canonical_sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _cartesian_source(source4_raw_l1: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    charges = source4_raw_l1[:, 0]
    dipoles = source4_raw_l1[:, (3, 1, 2)]
    return charges, dipoles


def _raw_l1(charges: np.ndarray, dipoles_cartesian: np.ndarray) -> np.ndarray:
    result = np.column_stack(
        (
            charges,
            dipoles_cartesian[:, 1],
            dipoles_cartesian[:, 2],
            dipoles_cartesian[:, 0],
        )
    )
    result = np.ascontiguousarray(result, dtype=np.float64)
    result.setflags(write=False)
    return result


def molecular_dipole_eangstrom(
    source4_raw_l1: object, positions_angstrom: object
) -> np.ndarray:
    """Return ``sum_A(q_A R_A + p_A)`` in e Angstrom."""

    source = np.asarray(source4_raw_l1, dtype=np.float64)
    if source.ndim != 2 or source.shape[1] != 4 or not np.all(np.isfinite(source)):
        raise ValueError("source4_raw_l1 must be finite with shape (N, 4).")
    positions = _readonly(
        positions_angstrom, shape=(len(source), 3), name="positions_angstrom"
    )
    charges, dipoles = _cartesian_source(source)
    result = np.sum(charges[:, None] * positions + dipoles, axis=0)
    result = np.ascontiguousarray(result, dtype=np.float64)
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class MDPPolarizabilityDipoleClosure:
    """Immutable result of one zero-training uniform-response composition."""

    source4_raw_l1: np.ndarray
    source_molecular_dipole_eangstrom: np.ndarray
    target_molecular_dipole_eangstrom: np.ndarray
    dipole_correction_eangstrom: np.ndarray
    equivalent_uniform_field_volt_per_angstrom: np.ndarray
    molecular_polarizability_condition_number: float
    closure_residual_eangstrom: float
    total_charge_change_e: float
    state_sha256: str = ""

    def __post_init__(self) -> None:
        source = np.asarray(self.source4_raw_l1, dtype=np.float64)
        if source.ndim != 2 or source.shape[1] != 4:
            raise ValueError("source4_raw_l1 must have shape (N, 4).")
        source = _readonly(source, shape=source.shape, name="source4_raw_l1")
        source_mu = _readonly(
            self.source_molecular_dipole_eangstrom,
            shape=(3,),
            name="source_molecular_dipole_eangstrom",
        )
        target_mu = _readonly(
            self.target_molecular_dipole_eangstrom,
            shape=(3,),
            name="target_molecular_dipole_eangstrom",
        )
        correction = _readonly(
            self.dipole_correction_eangstrom,
            shape=(len(source), 3),
            name="dipole_correction_eangstrom",
        )
        field = _readonly(
            self.equivalent_uniform_field_volt_per_angstrom,
            shape=(3,),
            name="equivalent_uniform_field_volt_per_angstrom",
        )
        condition = float(self.molecular_polarizability_condition_number)
        residual = float(self.closure_residual_eangstrom)
        charge_change = float(self.total_charge_change_e)
        if (
            not math.isfinite(condition)
            or condition < 1.0
            or not math.isfinite(residual)
            or residual < 0.0
            or not math.isfinite(charge_change)
        ):
            raise ValueError("Closure diagnostics are invalid.")
        payload = {
            "contract": _CONTRACT,
            "source_sha256": _array_sha256(source),
            "source_molecular_dipole_sha256": _array_sha256(source_mu),
            "target_molecular_dipole_sha256": _array_sha256(target_mu),
            "dipole_correction_sha256": _array_sha256(correction),
            "equivalent_uniform_field_sha256": _array_sha256(field),
            "molecular_polarizability_condition_number": condition,
            "closure_residual_eangstrom": residual,
            "total_charge_change_e": charge_change,
        }
        expected = _canonical_sha256(payload)
        if self.state_sha256 and self.state_sha256 != expected:
            raise ValueError("state_sha256 does not match the closure contents.")
        object.__setattr__(self, "source4_raw_l1", source)
        object.__setattr__(self, "source_molecular_dipole_eangstrom", source_mu)
        object.__setattr__(self, "target_molecular_dipole_eangstrom", target_mu)
        object.__setattr__(self, "dipole_correction_eangstrom", correction)
        object.__setattr__(self, "equivalent_uniform_field_volt_per_angstrom", field)
        object.__setattr__(self, "molecular_polarizability_condition_number", condition)
        object.__setattr__(self, "closure_residual_eangstrom", residual)
        object.__setattr__(self, "total_charge_change_e", charge_change)
        object.__setattr__(self, "state_sha256", expected)


def close_dipole_with_mdp_polarizability(
    *,
    source4_raw_l1: object,
    positions_angstrom: object,
    target_molecular_dipole_eangstrom: object,
    atomic_polarizabilities_eangstrom2_per_volt: object,
    molecular_polarizability_eangstrom2_per_volt: object,
    dipole_tolerance_eangstrom: float = 1.0e-10,
    charge_tolerance_e: float = 1.0e-12,
    maximum_condition_number: float = 100.0,
    maximum_equivalent_field_volt_per_angstrom: float = 0.25,
) -> MDPPolarizabilityDipoleClosure:
    """Close a source dipole using only frozen MACE-MDP response tensors."""

    source = np.asarray(source4_raw_l1, dtype=np.float64)
    if source.ndim != 2 or source.shape[1] != 4 or not np.all(np.isfinite(source)):
        raise ValueError("source4_raw_l1 must be finite with shape (N, 4).")
    atom_count = len(source)
    positions = _readonly(
        positions_angstrom, shape=(atom_count, 3), name="positions_angstrom"
    )
    target = _readonly(
        target_molecular_dipole_eangstrom,
        shape=(3,),
        name="target_molecular_dipole_eangstrom",
    )
    atomic = _readonly(
        atomic_polarizabilities_eangstrom2_per_volt,
        shape=(atom_count, 3, 3),
        name="atomic_polarizabilities_eangstrom2_per_volt",
    )
    total = _readonly(
        molecular_polarizability_eangstrom2_per_volt,
        shape=(3, 3),
        name="molecular_polarizability_eangstrom2_per_volt",
    )
    for value, name in (
        (dipole_tolerance_eangstrom, "dipole_tolerance_eangstrom"),
        (charge_tolerance_e, "charge_tolerance_e"),
        (maximum_condition_number, "maximum_condition_number"),
        (
            maximum_equivalent_field_volt_per_angstrom,
            "maximum_equivalent_field_volt_per_angstrom",
        ),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be positive and finite.")
    if not np.allclose(np.sum(atomic, axis=0), total, rtol=0.0, atol=2.0e-11):
        raise ValueError(
            "Atomic polarizabilities do not reconstruct the molecular tensor."
        )
    symmetric = 0.5 * (total + total.T)
    if not np.allclose(total, symmetric, rtol=0.0, atol=2.0e-12):
        raise ValueError("Molecular polarizability is not symmetric.")
    eigenvalues = np.linalg.eigvalsh(symmetric)
    if eigenvalues[0] <= 0.0:
        raise ValueError("Molecular polarizability is not positive definite.")
    condition = float(eigenvalues[-1] / eigenvalues[0])
    if condition > maximum_condition_number:
        raise ValueError("Molecular polarizability condition number exceeds the gate.")

    source_mu = molecular_dipole_eangstrom(source, positions)
    delta_mu = target - source_mu
    equivalent_field = np.linalg.solve(symmetric, delta_mu)
    if (
        not np.all(np.isfinite(equivalent_field))
        or np.linalg.norm(equivalent_field) > maximum_equivalent_field_volt_per_angstrom
    ):
        raise ValueError("Equivalent uniform field exceeds the physical gate.")
    corrections = np.einsum("aij,j->ai", atomic, equivalent_field)
    charges, dipoles = _cartesian_source(source)
    closed_source = _raw_l1(charges, dipoles + corrections)
    closed_mu = molecular_dipole_eangstrom(closed_source, positions)
    residual = float(np.linalg.norm(closed_mu - target))
    charge_change = float(np.sum(closed_source[:, 0]) - np.sum(source[:, 0]))
    if residual > dipole_tolerance_eangstrom:
        raise RuntimeError("Polarizability-weighted source did not close the dipole.")
    if abs(charge_change) > charge_tolerance_e:
        raise RuntimeError("Polarizability-weighted source changed total charge.")
    return MDPPolarizabilityDipoleClosure(
        source4_raw_l1=closed_source,
        source_molecular_dipole_eangstrom=closed_mu,
        target_molecular_dipole_eangstrom=target,
        dipole_correction_eangstrom=corrections,
        equivalent_uniform_field_volt_per_angstrom=equivalent_field,
        molecular_polarizability_condition_number=condition,
        closure_residual_eangstrom=residual,
        total_charge_change_e=charge_change,
    )
