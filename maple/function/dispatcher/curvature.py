"""Shared validation and mode analysis for Cartesian curvature."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import numpy as np

from ..calculator.calculator_base import _is_factory_hessian_precision_binding


_HARTREE_J = 4.3597447222071e-18
_AMU_KG = 1.66053906660e-27
_ANGSTROM2_M2 = 1.0e-20
_LIGHT_SPEED_M_S = 2.99792458e8
HARTREE_AMU_ANGSTROM2_TO_CM1 = (
    np.sqrt(_HARTREE_J / (_AMU_KG * _ANGSTROM2_M2))
    / (2.0 * np.pi * _LIGHT_SPEED_M_S * 100.0)
)


@dataclass(frozen=True)
class HessianPrecisionProfile:
    """A source-derived raw-skew policy for one Hessian consumer."""

    identity: str
    relative_skew_tolerance: float
    source_bound: bool


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
_AIMNET2025_N32_PROFILE = HessianPrecisionProfile(
    identity="qualified-aimnet2025-n32",
    relative_skew_tolerance=2.0e-6,
    source_bound=True,
)


def _same_identity(left, right) -> bool:
    return isinstance(left, dict) and isinstance(right, dict) and left == right


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

    identity = calculator.analytic_cell_identity
    runtime_identity = calculator.runtime_profile_identity
    if not _same_identity(identity, runtime_identity):
        return _STRICT_PROFILE
    binding = getattr(calculator, "_hessian_precision_binding", None)
    if not _is_factory_hessian_precision_binding(binding):
        return _STRICT_PROFILE
    identity_sha = hashlib.sha256(
        json.dumps(
            identity,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    if (
        binding.profile_id != _AIMNET2025_N32_PROFILE.identity
        or binding.runtime_identity_sha256 != identity_sha
    ):
        return _STRICT_PROFILE
    if (
        identity.get("model") != "aimnet2025"
        or identity.get("effective_ad_class") != "N32"
        or identity.get("effective_nn_dtype") != "float32"
        or str(getattr(calculator, "dtype", "")) != "torch.float32"
        or getattr(calculator, "d4", None) is not False
        or str(getattr(calculator, "hessian", "")).lower() != "analytic"
    ):
        return _STRICT_PROFILE

    correction = getattr(calculator, "solvent_correction", None)
    if correction is not None:
        if binding.composition != "torch-obc2":
            return _STRICT_PROFILE
        if (
            getattr(calculator, "analytic_implicit_derivatives_admitted", False)
            is not True
            or getattr(correction, "analytic_task_derivatives_admitted", False)
            is not True
        ):
            return _STRICT_PROFILE
        provider = getattr(correction, "underlying_provider", None)
        provenance = getattr(provider, "provenance", None)
        provenance = provenance() if callable(provenance) else provenance
        if not isinstance(provenance, dict):
            return _STRICT_PROFILE
        if (
            provenance.get("runtime_provider") != "torch-obc2"
            or provenance.get("equation_profile") != "obc2-zero-salt-v1"
            or provenance.get("dtype") != "torch.float32"
            or provenance.get("nonpolar") not in {"ace", "none"}
        ):
            return _STRICT_PROFILE
    elif binding.composition != "gas":
        return _STRICT_PROFILE
    return _AIMNET2025_N32_PROFILE


def validate_cartesian_hessian(
    atoms,
    raw_hessian,
    profile: HessianPrecisionProfile,
) -> ValidatedCartesianHessian:
    """Validate raw curvature before returning its physical symmetric part."""
    if not isinstance(profile, HessianPrecisionProfile):
        raise TypeError("Hessian validation requires a precision profile.")
    if (
        profile.relative_skew_tolerance > _STRICT_PROFILE.relative_skew_tolerance
        and profile is not _AIMNET2025_N32_PROFILE
    ):
        raise ValueError("A relaxed Hessian precision profile must be factory-bound.")
    raw = np.asarray(raw_hessian, dtype=np.float64)
    if raw.ndim == 3 and raw.shape[0] == 1:
        raw = raw[0]
    expected = (3 * len(atoms), 3 * len(atoms))
    if raw.shape != expected:
        raise ValueError(
            f"Cartesian Hessian shape {raw.shape} does not match exact 3N shape {expected}."
        )
    if not np.isfinite(raw).all():
        raise ValueError("Cartesian Hessian must contain only finite values.")
    raw_maximum_absolute = float(np.max(np.abs(raw))) if raw.size else 0.0
    scale = max(1.0, raw_maximum_absolute)
    raw_maximum_skew = float(np.max(np.abs(raw - raw.T))) if raw.size else 0.0
    raw_relative_skew = raw_maximum_skew / scale
    if raw_relative_skew > profile.relative_skew_tolerance:
        raise ValueError(
            "Cartesian Hessian has unresolved asymmetry "
            f"({raw_relative_skew:.8e} > {profile.relative_skew_tolerance:.8e}; "
            f"profile={profile.identity})."
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


def _optimization_basis(atoms, *, project_rigid_modes: bool) -> tuple[np.ndarray, int]:
    n3 = 3 * len(atoms)
    if not project_rigid_modes:
        return np.eye(n3, dtype=np.float64), 0
    if np.any(atoms.get_pbc()) or bool(getattr(atoms, "constraints", None)):
        raise ValueError(
            "Rigid-mode projection is implemented only for isolated "
            "unconstrained molecules."
        )
    masses = np.asarray(atoms.get_masses(), dtype=np.float64)
    if (
        masses.shape != (len(atoms),)
        or not np.isfinite(masses).all()
        or np.any(masses <= 0.0)
    ):
        raise ValueError("Mode analysis requires finite positive atomic masses.")
    positions = np.asarray(atoms.get_positions(), dtype=np.float64)
    if positions.shape != (len(atoms), 3) or not np.isfinite(positions).all():
        raise ValueError("Rigid-mode projection requires finite positions.")
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
    masses = np.asarray(atoms.get_masses(), dtype=np.float64)
    if (
        masses.shape != (len(atoms),)
        or not np.isfinite(masses).all()
        or np.any(masses <= 0.0)
    ):
        raise ValueError("Mode analysis requires finite positive atomic masses.")
    inverse_sqrt_mass = 1.0 / np.sqrt(np.repeat(masses, 3))
    weighted = (
        inverse_sqrt_mass[:, None]
        * validated_hessian.physical_hessian
        * inverse_sqrt_mass[None, :]
    )
    basis, rigid_rank = _optimization_basis(
        atoms,
        project_rigid_modes=project_rigid_modes,
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
