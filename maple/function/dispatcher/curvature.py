"""Shared validation and mode analysis for Cartesian curvature."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import numpy as np

from ..calculator.calculator_base import _is_factory_hessian_precision_binding
from ..calculator.analytic_protocol import (
    AnalyticCapabilityError,
    AnalyticIdentityError,
)


_HARTREE_J = 4.3597447222071e-18
_AMU_KG = 1.66053906660e-27
_ANGSTROM2_M2 = 1.0e-20
_LIGHT_SPEED_M_S = 2.99792458e8
HARTREE_AMU_ANGSTROM2_TO_CM1 = (
    np.sqrt(_HARTREE_J / (_AMU_KG * _ANGSTROM2_M2))
    / (2.0 * np.pi * _LIGHT_SPEED_M_S * 100.0)
)


class CurvatureValidationError(ValueError):
    """Expected scientific refusal at the Cartesian-curvature boundary."""


@dataclass(frozen=True)
class HessianPrecisionProfile:
    """A source-derived raw-skew policy for one Hessian consumer."""

    identity: str
    relative_skew_tolerance: float
    source_bound: bool
    calculator_token: int | None = None
    atoms_token: int | None = None
    runtime_identity_sha256: str | None = None
    receipt_sha256: str | None = None
    composition: str | None = None
    composition_identity_sha256: str | None = None
    atom_domain_sha256: str | None = None
    _authority: object | None = None


@dataclass(frozen=True)
class ValidatedCartesianHessian:
    """Raw diagnostics plus the read-only physical symmetric Hessian."""

    physical_hessian: np.ndarray
    raw_sha256: str
    raw_maximum_absolute: float
    raw_maximum_skew: float
    raw_relative_skew: float
    precision_profile: HessianPrecisionProfile


@dataclass(frozen=True)
class MassWeightedModeBasis:
    """One mass metric, optimization basis, and eigensystem."""

    basis: np.ndarray
    rigid_rank: int
    inverse_sqrt_mass: np.ndarray
    weighted_hessian: np.ndarray
    reduced_hessian: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    frequencies_cm1: np.ndarray


_STRICT_PROFILE = HessianPrecisionProfile(
    identity="strict-generic",
    relative_skew_tolerance=1.0e-8,
    source_bound=False,
)
_PROFILE_AUTHORITY = object()
_PROFILE_RELATIVE_SKEW_TOLERANCES = {
    "strict-generic": 1.0e-8,
    "qualified-aimnet2025-n32-gas": 2.0e-6,
    "qualified-aimnet2025-n32-torch-obc2": 2.0e-6,
}


def _same_identity(left, right) -> bool:
    return isinstance(left, dict) and isinstance(right, dict) and left == right


def _json_sha256(value) -> str:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise CurvatureValidationError(
            "Hessian provenance identity is not canonical JSON."
        ) from exc
    return hashlib.sha256(encoded).hexdigest()


def _live_runtime_identity(calculator) -> tuple[dict, str]:
    """Re-run the cheap analytic identity guards and return their canonical SHA."""
    try:
        identity = calculator.analytic_cell_identity
        runtime_identity = calculator.runtime_profile_identity
    except (AnalyticIdentityError, AnalyticCapabilityError) as exc:
        raise CurvatureValidationError(
            "Exact AIMNet2025 Hessian runtime binding validation failed."
        ) from exc
    if not _same_identity(identity, runtime_identity):
        raise CurvatureValidationError(
            "Exact AIMNet2025 Hessian claim has a drifted runtime binding."
        )
    return identity, _json_sha256(identity)


def _atom_domain(atoms, *, total_charge: float | None = None) -> dict:
    info = getattr(atoms, "info", {})
    try:
        charge = (
            float(info.get("charge", 0.0))
            if total_charge is None
            else float(total_charge)
        )
        multiplicity = info.get("mult", info.get("multiplicity", 1))
    except (TypeError, ValueError) as exc:
        raise CurvatureValidationError(
            "Hessian Atoms charge and multiplicity must be numeric."
        ) from exc
    return {
        "atomic_numbers": atoms.get_atomic_numbers().tolist(),
        "charge": charge,
        "multiplicity": multiplicity,
        "pbc": np.asarray(atoms.get_pbc(), dtype=bool).tolist(),
    }


def resolve_hessian_precision_profile(
    atoms,
    *,
    purpose: str,
) -> HessianPrecisionProfile:
    """Resolve a raw-skew policy from a live, exact calculator binding.

    User parameters and self-reported dictionaries cannot select the N32
    allowance.  The exact shipped AIMNet2025 implementation re-runs its bound
    runtime/source guard when either identity property is read.  Composed cells
    must also retain both existing admission decisions.
    """
    if not isinstance(purpose, str) or not purpose.strip():
        raise ValueError("Hessian precision resolution requires a purpose.")
    calculator = getattr(atoms, "calc", None)
    if calculator is None:
        return _STRICT_PROFILE
    try:
        from ..calculator.aimnet._aimnet2025_calculator import AIMNet2025Calculator
    except ImportError:
        return _STRICT_PROFILE
    if type(calculator) is not AIMNet2025Calculator:
        return _STRICT_PROFILE
    if str(getattr(calculator, "hessian", "")).lower() != "analytic":
        return _STRICT_PROFILE

    identity, identity_sha = _live_runtime_identity(calculator)
    binding = getattr(calculator, "_hessian_precision_binding", None)
    if not _is_factory_hessian_precision_binding(binding):
        raise CurvatureValidationError("Exact AIMNet2025 Hessian claim lacks a factory binding.")
    if (
        binding.profile_id
        not in {
            "qualified-aimnet2025-n32-gas",
            "qualified-aimnet2025-n32-torch-obc2",
        }
        or binding.runtime_identity_sha256 != identity_sha
        or binding.owner_token != id(calculator)
    ):
        raise CurvatureValidationError("Exact AIMNet2025 Hessian claim has an invalid source binding.")
    if (
        identity.get("model") != "aimnet2025"
        or identity.get("effective_ad_class") != "N32"
        or identity.get("effective_nn_dtype") != "float32"
        or str(getattr(calculator, "dtype", "")) != "torch.float32"
        or getattr(calculator, "d4", None) is not False
        or str(getattr(calculator, "hessian", "")).lower() != "analytic"
    ):
        raise CurvatureValidationError("Exact AIMNet2025 Hessian claim is outside the qualified N32 cell.")
    charge = AIMNet2025Calculator._total_charge_from_atoms(atoms)
    multiplicity = getattr(atoms, "info", {}).get(
        "mult", getattr(atoms, "info", {}).get("multiplicity", 1)
    )
    if not np.isfinite(charge) or charge != 0.0 or multiplicity != 1:
        raise CurvatureValidationError(
            "Qualified AIMNet2025 curvature requires a neutral closed-shell domain."
        )
    if np.any(atoms.get_pbc()):
        raise CurvatureValidationError("Qualified AIMNet2025 curvature requires a nonperiodic domain.")

    correction = getattr(calculator, "solvent_correction", None)
    if correction is not None:
        if binding.composition != "torch-obc2":
            raise CurvatureValidationError("Exact AIMNet2025 composed Hessian binding has drifted.")
        if (
            getattr(calculator, "analytic_implicit_derivatives_admitted", False)
            is not True
            or getattr(correction, "analytic_task_derivatives_admitted", False)
            is not True
        ):
            raise CurvatureValidationError("Exact AIMNet2025 composed derivative admission has drifted.")
        provider = getattr(correction, "underlying_provider", None)
        provenance = getattr(provider, "provenance", None)
        provenance = provenance() if callable(provenance) else provenance
        if not isinstance(provenance, dict):
            raise CurvatureValidationError("Exact AIMNet2025 composed provenance is missing.")
        if (
            provenance.get("runtime_provider") != "torch-obc2"
            or provenance.get("equation_profile") != "obc2-zero-salt-v1"
            or provenance.get("dtype") != "torch.float32"
            or provenance.get("nonpolar") not in {"ace", "none"}
        ):
            raise CurvatureValidationError("Exact AIMNet2025 composed provenance has drifted.")
        composition_identity_sha = _json_sha256(provenance)
    elif binding.composition != "gas":
        raise CurvatureValidationError("Exact AIMNet2025 gas Hessian binding has drifted.")
    else:
        composition_identity_sha = _json_sha256({"composition": "gas"})
    if composition_identity_sha != binding.composition_identity_sha256:
        raise CurvatureValidationError(
            "Exact AIMNet2025 Hessian composition changed after factory binding."
        )
    return HessianPrecisionProfile(
        identity=binding.profile_id,
        relative_skew_tolerance=2.0e-6,
        source_bound=True,
        calculator_token=id(calculator),
        atoms_token=id(atoms),
        runtime_identity_sha256=identity_sha,
        receipt_sha256=binding.receipt_sha256,
        composition=binding.composition,
        composition_identity_sha256=composition_identity_sha,
        atom_domain_sha256=_json_sha256(
            _atom_domain(atoms, total_charge=charge)
        ),
        _authority=_PROFILE_AUTHORITY,
    )


def _validation_policy(
    atoms,
    profile: HessianPrecisionProfile,
) -> tuple[str, float]:
    """Derive the numerical policy from the live binding, never display data."""
    calculator = getattr(atoms, "calc", None)
    binding = getattr(calculator, "_hessian_precision_binding", None)
    if profile._authority is not _PROFILE_AUTHORITY:
        if profile.source_bound:
            raise CurvatureValidationError(
                "A source-bound Hessian precision profile must be factory-bound."
            )
        return "strict-generic", _PROFILE_RELATIVE_SKEW_TOLERANCES["strict-generic"]
    if not _is_factory_hessian_precision_binding(binding):
        raise CurvatureValidationError("The Hessian precision profile lacks a live factory binding.")
    expected_tolerance = _PROFILE_RELATIVE_SKEW_TOLERANCES.get(binding.profile_id)
    if expected_tolerance is None:
        raise CurvatureValidationError("The Hessian source binding names an unknown precision policy.")
    if profile.atoms_token != id(atoms):
        raise CurvatureValidationError(
            "A bound Hessian profile cannot validate a different Atoms object."
        )
    expected_fields = (
        binding.profile_id,
        expected_tolerance,
        True,
        id(calculator),
        id(atoms),
        binding.runtime_identity_sha256,
        binding.receipt_sha256,
        binding.composition,
        binding.composition_identity_sha256,
    )
    observed_fields = (
        profile.identity,
        profile.relative_skew_tolerance,
        profile.source_bound,
        profile.calculator_token,
        profile.atoms_token,
        profile.runtime_identity_sha256,
        profile.receipt_sha256,
        profile.composition,
        profile.composition_identity_sha256,
    )
    if observed_fields != expected_fields:
        raise CurvatureValidationError(
            "The Hessian precision profile does not match its immutable live policy."
        )
    return binding.profile_id, expected_tolerance


def validate_cartesian_hessian(
    atoms,
    raw_hessian,
    profile: HessianPrecisionProfile,
) -> ValidatedCartesianHessian:
    """Validate raw curvature before returning its physical symmetric part."""
    if not isinstance(profile, HessianPrecisionProfile):
        raise TypeError("Hessian validation requires a precision profile.")
    policy_identity, relative_skew_tolerance = _validation_policy(atoms, profile)
    if policy_identity != "strict-generic":
        if profile.atoms_token != id(atoms):
            raise CurvatureValidationError("A bound Hessian profile cannot validate a different Atoms object.")
        calculator = getattr(atoms, "calc", None)
        if profile.calculator_token != id(calculator):
            raise CurvatureValidationError("A bound Hessian profile cannot validate a different calculator.")
        binding = getattr(calculator, "_hessian_precision_binding", None)
        if (
            not _is_factory_hessian_precision_binding(binding)
            or binding.owner_token != profile.calculator_token
            or binding.runtime_identity_sha256 != profile.runtime_identity_sha256
            or binding.receipt_sha256 != profile.receipt_sha256
            or binding.composition != profile.composition
            or binding.composition_identity_sha256
            != profile.composition_identity_sha256
        ):
            raise CurvatureValidationError("The Hessian source binding changed after resolution.")
        _, live_identity_sha = _live_runtime_identity(calculator)
        if (
            live_identity_sha != binding.runtime_identity_sha256
            or live_identity_sha != profile.runtime_identity_sha256
        ):
            raise CurvatureValidationError(
                "The Hessian runtime identity changed after profile resolution."
            )
        if (
            str(getattr(calculator, "hessian", "")).lower() != "analytic"
            or str(getattr(calculator, "dtype", "")) != "torch.float32"
            or getattr(calculator, "d4", None) is not False
        ):
            raise CurvatureValidationError("The Hessian runtime identity changed after resolution.")
        try:
            from ..calculator.aimnet._aimnet2025_calculator import AIMNet2025Calculator
            charge = AIMNet2025Calculator._total_charge_from_atoms(atoms)
        except (TypeError, ValueError) as exc:
            raise CurvatureValidationError(
                "The Hessian Atoms charge is invalid."
            ) from exc
        if _json_sha256(
            _atom_domain(atoms, total_charge=charge)
        ) != profile.atom_domain_sha256:
            raise CurvatureValidationError("The Hessian Atoms domain changed after resolution.")
        correction = getattr(calculator, "solvent_correction", None)
        if profile.composition == "gas":
            current_composition_sha = _json_sha256({"composition": "gas"})
        else:
            provider = getattr(correction, "underlying_provider", None)
            provenance = getattr(provider, "provenance", None)
            provenance = provenance() if callable(provenance) else provenance
            if not isinstance(provenance, dict):
                raise CurvatureValidationError("The Hessian composition provenance is missing.")
            current_composition_sha = _json_sha256(provenance)
        if current_composition_sha != profile.composition_identity_sha256:
            raise CurvatureValidationError("The Hessian composition changed after resolution.")
    raw_input = np.asarray(raw_hessian)
    if np.iscomplexobj(raw_input):
        raise CurvatureValidationError("Cartesian Hessian must be real; complex values are not supported.")
    raw = np.asarray(raw_input, dtype=np.float64)
    if raw.ndim == 3 and raw.shape[0] == 1:
        raw = raw[0]
    expected = (3 * len(atoms), 3 * len(atoms))
    if raw.shape != expected:
        raise CurvatureValidationError(
            f"Cartesian Hessian shape {raw.shape} does not match exact 3N shape {expected}."
        )
    if not np.isfinite(raw).all():
        raise CurvatureValidationError("Cartesian Hessian must contain only finite values.")
    raw_maximum_absolute = float(np.max(np.abs(raw))) if raw.size else 0.0
    scale = max(1.0, raw_maximum_absolute)
    raw_maximum_skew = float(np.max(np.abs(raw - raw.T))) if raw.size else 0.0
    raw_relative_skew = raw_maximum_skew / scale
    if raw_relative_skew > relative_skew_tolerance:
        raise CurvatureValidationError(
            "Cartesian Hessian has unresolved asymmetry "
            f"({raw_relative_skew:.8e} > {relative_skew_tolerance:.8e}; "
            f"profile={policy_identity})."
        )
    raw_bytes = np.ascontiguousarray(raw).tobytes()
    physical = np.array(0.5 * (raw + raw.T), dtype=np.float64, copy=True)
    physical.setflags(write=False)
    return ValidatedCartesianHessian(
        physical_hessian=physical,
        raw_sha256=hashlib.sha256(raw_bytes).hexdigest(),
        raw_maximum_absolute=raw_maximum_absolute,
        raw_maximum_skew=raw_maximum_skew,
        raw_relative_skew=raw_relative_skew,
        precision_profile=profile,
    )


def _validated_mode_domain(
    atoms, *, project_rigid_modes: bool
) -> tuple[np.ndarray, np.ndarray]:
    masses = np.asarray(atoms.get_masses(), dtype=np.float64)
    if (
        masses.shape != (len(atoms),)
        or not np.isfinite(masses).all()
        or np.any(masses <= 0.0)
    ):
        raise CurvatureValidationError(
            "Mode analysis requires finite positive atomic masses."
        )
    positions = np.asarray(atoms.get_positions(), dtype=np.float64)
    if positions.shape != (len(atoms), 3) or not np.isfinite(positions).all():
        raise CurvatureValidationError("Mode analysis requires finite positions.")
    if project_rigid_modes and (
        np.any(atoms.get_pbc()) or bool(getattr(atoms, "constraints", None))
    ):
        raise CurvatureValidationError(
            "Rigid-mode projection is implemented only for isolated "
            "unconstrained molecules."
        )
    return masses, positions


def _optimization_basis(
    atoms,
    *,
    project_rigid_modes: bool,
    masses: np.ndarray,
    positions: np.ndarray,
) -> tuple[np.ndarray, int]:
    n3 = 3 * len(atoms)
    if not project_rigid_modes:
        return np.eye(n3, dtype=np.float64), 0
    center = np.sum(positions * masses[:, None], axis=0) / np.sum(masses)
    rigid = np.zeros((n3, 6), dtype=np.float64)
    for index, (mass, position) in enumerate(zip(masses, positions)):
        root_mass = np.sqrt(mass)
        relative = position - center
        block = slice(3 * index, 3 * index + 3)
        rigid[block, :3] = root_mass * np.eye(3)
        rigid[block, 3:] = root_mass * np.asarray(
            [
                [0.0, -relative[2], relative[1]],
                [relative[2], 0.0, -relative[0]],
                [-relative[1], relative[0], 0.0],
            ]
        )
    left_vectors, singular_values, _ = np.linalg.svd(rigid, full_matrices=True)
    tolerance = (
        np.finfo(np.float64).eps * max(rigid.shape) * singular_values[0]
        if singular_values.size
        else 0.0
    )
    rigid_rank = int(np.count_nonzero(singular_values > tolerance))
    return left_vectors[:, rigid_rank:], rigid_rank


def build_mass_weighted_mode_basis(
    atoms,
    validated_hessian: ValidatedCartesianHessian,
    *,
    project_rigid_modes: bool,
) -> MassWeightedModeBasis:
    """Build the shared mass-weighted eigensystem for validated curvature."""
    if not isinstance(validated_hessian, ValidatedCartesianHessian):
        raise TypeError("Mode analysis requires a validated Cartesian Hessian.")
    masses, positions = _validated_mode_domain(
        atoms,
        project_rigid_modes=project_rigid_modes,
    )
    inverse_sqrt_mass = 1.0 / np.sqrt(np.repeat(masses, 3))
    weighted = (
        inverse_sqrt_mass[:, None]
        * validated_hessian.physical_hessian
        * inverse_sqrt_mass[None, :]
    )
    basis, rigid_rank = _optimization_basis(
        atoms,
        project_rigid_modes=project_rigid_modes,
        masses=masses,
        positions=positions,
    )
    reduced = basis.T @ weighted @ basis
    reduced = 0.5 * (reduced + reduced.T)
    eigenvalues, reduced_vectors = np.linalg.eigh(reduced)
    eigenvectors = basis @ reduced_vectors
    frequencies = (
        np.sign(eigenvalues)
        * np.sqrt(np.abs(eigenvalues))
        * HARTREE_AMU_ANGSTROM2_TO_CM1
    )
    return MassWeightedModeBasis(
        basis=basis,
        rigid_rank=rigid_rank,
        inverse_sqrt_mass=inverse_sqrt_mass,
        weighted_hessian=weighted,
        reduced_hessian=reduced,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        frequencies_cm1=frequencies,
    )


def bofill_update(
    hessian: np.ndarray,
    displacement: np.ndarray,
    gradient_change: np.ndarray,
) -> np.ndarray:
    """Return a symmetric Bofill update, or the input for a null secant.

    The null-secant case is legitimate when two evaluations have identical
    gradients.  Returning the unchanged Hessian avoids the ``0 / 0`` terms in
    both the Murtagh-Sargent update and the Bofill mixing factor.
    """
    hessian = np.asarray(hessian, dtype=np.float64)
    dx = np.asarray(displacement, dtype=np.float64).reshape(-1)
    dg = np.asarray(gradient_change, dtype=np.float64).reshape(-1)
    dx2 = float(dx @ dx)
    dg2 = float(dg @ dg)
    if dx2 <= 1e-16 or dg2 <= 1e-16:
        return hessian.copy()

    residual = dg - hessian @ dx
    residual2 = float(residual @ residual)
    residual_dot_step = float(residual @ dx)
    psb = (
        (np.outer(residual, dx) + np.outer(dx, residual)) / dx2
        - residual_dot_step * np.outer(dx, dx) / (dx2 * dx2)
    )
    if abs(residual_dot_step) > 1e-8 and residual2 > 1e-16:
        ms = np.outer(residual, residual) / residual_dot_step
        ratio = residual_dot_step**2 / (dx2 * residual2)
        phi = float(np.clip(1.0 - ratio, 0.0, 1.0))
    else:
        ms = np.zeros_like(hessian)
        phi = 1.0
    updated = hessian + (1.0 - phi) * ms + phi * psb
    return 0.5 * (updated + updated.T)
