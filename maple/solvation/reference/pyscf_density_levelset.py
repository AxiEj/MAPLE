"""PySCF AO-density level set for fixed-geometry rho-DROP diagnostics.

This module provides an independent QM reference for assessing MAPLE's
reconstructed-density cavity.  It is deliberately not a production electronic
model: the density is frozen from a content-addressed PySCF checkpoint, source
and coordinate response methods are unavailable, and no experimental
solvation labels enter the construction.

The level-set convention matches the Route-2 reconstructed-density provider,

``S(r) = 1 - n(r) / n_iso``,

so the zero sets can be compared without conflating the cavity definition with
a different DROP scaling convention.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from maple.function.calculator.extra_correction.implicit.route2_density_levelset import (
    SpatialLSFJet,
)
from maple.solvation.reference.pyscf_pcmsolver import (
    array_sha256,
    closed_shell_density_from_orbitals,
)


QM_AO_DENSITY_LEVEL_SET_ID = "route2-qm-ao-density-isodensity-level-set-diagnostic-v1"

_GRADIENT_COMPONENTS = (1, 2, 3)
_HESSIAN_COMPONENTS = {
    (0, 0): 4,
    (0, 1): 5,
    (0, 2): 6,
    (1, 1): 7,
    (1, 2): 8,
    (2, 2): 9,
}
_THIRD_COMPONENTS = {
    (0, 0, 0): 10,
    (0, 0, 1): 11,
    (0, 0, 2): 12,
    (0, 1, 1): 13,
    (0, 1, 2): 14,
    (0, 2, 2): 15,
    (1, 1, 1): 16,
    (1, 1, 2): 17,
    (1, 2, 2): 18,
    (2, 2, 2): 19,
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()


def _points(values: object) -> np.ndarray:
    points = np.asarray(values, dtype=float)
    if (
        points.ndim != 2
        or points.shape[0] == 0
        or points.shape[1] != 3
        or not np.all(np.isfinite(points))
    ):
        raise ValueError("QM density evaluation points must be finite with shape (M,3).")
    return points


def _symmetric_component(
    derivatives: np.ndarray,
    components: dict[tuple[int, ...], int],
    indices: tuple[int, ...],
) -> np.ndarray:
    return derivatives[components[tuple(sorted(indices))]]


@dataclass(frozen=True, slots=True)
class QMDensityBoundaryDiagnostics:
    minimum_density_e_per_bohr3: float
    minimum_level_set_gradient_norm_bohr: float
    maximum_level_set_residual: float
    electron_count_e: float
    expected_electron_count_e: float
    electron_count_absolute_error_e: float
    passed: bool
    reasons: tuple[str, ...]


class PySCFAODensityLevelSet:
    """Frozen closed-shell AO-density level set for reference cavity builds."""

    density_identity = "frozen-pyscf-ao-electron-density"
    level_set_identity = QM_AO_DENSITY_LEVEL_SET_ID

    def __init__(
        self,
        molecule: Any,
        mo_coeff: object,
        mo_occ: object,
        *,
        n_iso_e_per_bohr3: float,
        checkpoint_sha256: str,
        checkpoint_path: str | Path | None = None,
        surface_tolerance: float = 1.0e-9,
        minimum_gradient_norm_bohr: float = 1.0e-10,
        minimum_density_e_per_bohr3: float = -1.0e-12,
        electron_count_tolerance_e: float = 1.0e-7,
    ) -> None:
        overlap = np.asarray(molecule.intor_symmetric("int1e_ovlp"), dtype=float)
        density, density_audit = closed_shell_density_from_orbitals(
            mo_coeff,
            mo_occ,
            overlap,
        )
        coefficients = np.asarray(mo_coeff, dtype=float)
        occupations = np.asarray(mo_occ, dtype=float)
        occupied = occupations > 1.0e-12
        weighted = coefficients[:, occupied] * np.sqrt(occupations[occupied])[None, :]
        if weighted.shape[1] == 0:
            raise ValueError("QM density level set requires occupied orbitals.")

        n_iso = float(n_iso_e_per_bohr3)
        if not np.isfinite(n_iso) or n_iso <= 0.0:
            raise ValueError("QM density isovalue must be positive and finite.")
        checkpoint_digest = str(checkpoint_sha256)
        if len(checkpoint_digest) != 64 or any(
            character not in "0123456789abcdef" for character in checkpoint_digest
        ):
            raise ValueError("QM checkpoint identity must be a lowercase SHA256.")

        self.molecule = molecule
        self.atomic_numbers = np.asarray(molecule.atom_charges(), dtype=np.int32)
        self.atomic_numbers.setflags(write=False)
        self.positions_bohr = np.asarray(molecule.atom_coords(), dtype=float)
        self.positions_bohr.setflags(write=False)
        self.density_matrix = np.asarray(density, dtype=float)
        self.density_matrix.setflags(write=False)
        self.weighted_occupied_coefficients = np.asarray(weighted, dtype=float)
        self.weighted_occupied_coefficients.setflags(write=False)
        self.n_iso_e_per_bohr3 = n_iso
        self.checkpoint_sha256 = checkpoint_digest
        self.checkpoint_path = (
            None if checkpoint_path is None else str(Path(checkpoint_path).resolve())
        )
        self.surface_tolerance = float(surface_tolerance)
        self.minimum_gradient_norm_bohr = float(minimum_gradient_norm_bohr)
        self.minimum_density_e_per_bohr3 = float(minimum_density_e_per_bohr3)
        self.electron_count_tolerance_e = float(electron_count_tolerance_e)
        self.electron_count_e = float(density_audit["electron_count_e"])
        self.expected_electron_count_e = float(density_audit["occupation_sum_e"])
        self.density_sha256 = str(density_audit["density_sha256"])
        self.state_sha256 = _canonical_sha256(
            {
                "level_set_identity": self.level_set_identity,
                "checkpoint_sha256": self.checkpoint_sha256,
                "density_sha256": self.density_sha256,
                "atomic_numbers_sha256": array_sha256(self.atomic_numbers),
                "positions_bohr_sha256": array_sha256(self.positions_bohr),
                "n_iso_e_per_bohr3": self.n_iso_e_per_bohr3,
                "pyscf_version": self._pyscf_version(),
            }
        )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        n_iso_e_per_bohr3: float,
        expected_checkpoint_sha256: str | None = None,
        **kwargs: object,
    ) -> "PySCFAODensityLevelSet":
        from pyscf import lib, scf

        path = Path(checkpoint_path).expanduser().resolve(strict=True)
        digest = _sha256_file(path)
        if expected_checkpoint_sha256 is not None and digest != str(
            expected_checkpoint_sha256
        ):
            raise ValueError("QM checkpoint SHA256 does not match the frozen input.")
        molecule = scf.chkfile.load_mol(str(path))
        payload = lib.chkfile.load(str(path), "scf")
        try:
            mo_coeff = payload["mo_coeff"]
            mo_occ = payload["mo_occ"]
        except (KeyError, TypeError) as exc:
            raise ValueError("QM checkpoint omits restricted orbitals/occupations.") from exc
        return cls(
            molecule,
            mo_coeff,
            mo_occ,
            n_iso_e_per_bohr3=n_iso_e_per_bohr3,
            checkpoint_sha256=digest,
            checkpoint_path=path,
            **kwargs,
        )

    @staticmethod
    def _pyscf_version() -> str:
        import pyscf

        return str(pyscf.__version__)

    @property
    def atom_count(self) -> int:
        return int(self.atomic_numbers.size)

    def evaluate_density(self, points_bohr: object):
        from pyscf.dft import numint

        points = _points(points_bohr)
        ao = np.asarray(numint.eval_ao(self.molecule, points, deriv=3), dtype=float)
        if ao.shape != (20, len(points), self.density_matrix.shape[0]):
            raise RuntimeError("PySCF returned an unexpected third-order AO layout.")

        coeff = self.weighted_occupied_coefficients
        value_orbitals = ao[0] @ coeff
        gradient_orbitals = np.stack(
            [ao[index] @ coeff for index in _GRADIENT_COMPONENTS], axis=1
        )
        hessian_orbitals = np.empty(
            (len(points), 3, 3, coeff.shape[1]), dtype=float
        )
        third_orbitals = np.empty(
            (len(points), 3, 3, 3, coeff.shape[1]), dtype=float
        )
        for i in range(3):
            for j in range(3):
                hessian_orbitals[:, i, j] = (
                    _symmetric_component(ao, _HESSIAN_COMPONENTS, (i, j)) @ coeff
                )
                for k in range(3):
                    third_orbitals[:, i, j, k] = (
                        _symmetric_component(ao, _THIRD_COMPONENTS, (i, j, k))
                        @ coeff
                    )

        density = np.einsum("mo,mo->m", value_orbitals, value_orbitals)
        gradient = 2.0 * np.einsum(
            "mo,mio->mi", value_orbitals, gradient_orbitals
        )
        hessian = 2.0 * (
            np.einsum("mo,mijo->mij", value_orbitals, hessian_orbitals)
            + np.einsum("mio,mjo->mij", gradient_orbitals, gradient_orbitals)
        )
        third = np.empty((len(points), 3, 3, 3), dtype=float)
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    third[:, i, j, k] = 2.0 * (
                        np.einsum(
                            "mo,mo->m",
                            value_orbitals,
                            third_orbitals[:, i, j, k],
                        )
                        + np.einsum(
                            "mo,mo->m",
                            hessian_orbitals[:, i, j],
                            gradient_orbitals[:, k],
                        )
                        + np.einsum(
                            "mo,mo->m",
                            hessian_orbitals[:, i, k],
                            gradient_orbitals[:, j],
                        )
                        + np.einsum(
                            "mo,mo->m",
                            hessian_orbitals[:, j, k],
                            gradient_orbitals[:, i],
                        )
                    )
        return density, gradient, hessian, third

    def evaluate_spatial(self, points_bohr: object) -> SpatialLSFJet:
        density, gradient, hessian, third = self.evaluate_density(points_bohr)
        scale = -1.0 / self.n_iso_e_per_bohr3
        return SpatialLSFJet(
            value=1.0 + scale * density,
            gradient=scale * gradient,
            hessian=scale * hessian,
            third=scale * third,
        )

    def diagnose(
        self,
        points_bohr: object,
        *,
        require_surface: bool,
    ) -> QMDensityBoundaryDiagnostics:
        points = _points(points_bohr)
        density, _, _, _ = self.evaluate_density(points)
        level_set = self.evaluate_spatial(points)
        minimum_density = float(np.min(density))
        minimum_gradient = float(np.min(np.linalg.norm(level_set.gradient, axis=1)))
        maximum_residual = float(np.max(np.abs(level_set.value)))
        electron_count_error = abs(
            self.electron_count_e - self.expected_electron_count_e
        )
        reasons: list[str] = []
        if minimum_density < self.minimum_density_e_per_bohr3:
            reasons.append(f"negative QM density: {minimum_density:.6e} e/bohr^3")
        if minimum_gradient < self.minimum_gradient_norm_bohr:
            reasons.append(
                f"near-critical QM level-set gradient: {minimum_gradient:.6e} bohr^-1"
            )
        if require_surface and maximum_residual > self.surface_tolerance:
            reasons.append(
                f"QM surface level-set residual is too large: {maximum_residual:.6e}"
            )
        if electron_count_error > self.electron_count_tolerance_e:
            reasons.append(
                f"QM density electron-count error is too large: {electron_count_error:.6e}"
            )
        return QMDensityBoundaryDiagnostics(
            minimum_density_e_per_bohr3=minimum_density,
            minimum_level_set_gradient_norm_bohr=minimum_gradient,
            maximum_level_set_residual=maximum_residual,
            electron_count_e=self.electron_count_e,
            expected_electron_count_e=self.expected_electron_count_e,
            electron_count_absolute_error_e=electron_count_error,
            passed=not reasons,
            reasons=tuple(reasons),
        )

    def require_valid_boundary(
        self,
        points_bohr: object,
        validation_shell_points_bohr: object | None = None,
    ) -> QMDensityBoundaryDiagnostics:
        boundary = self.diagnose(points_bohr, require_surface=True)
        reasons = list(boundary.reasons)
        if validation_shell_points_bohr is not None:
            shell = self.diagnose(
                validation_shell_points_bohr,
                require_surface=False,
            )
            reasons.extend(f"validation shell: {reason}" for reason in shell.reasons)
        if reasons:
            raise ValueError("QM rho-DROP boundary admission failed: " + "; ".join(reasons))
        return boundary

    # These methods make the fixed-density diagnostic satisfy the structural
    # level-set protocol.  Invoking response derivatives would change the
    # scientific question and therefore fails loudly.
    def source_jvp(self, points_bohr: object, source_direction: object):
        raise NotImplementedError("Frozen QM density exposes no MACE-source JVP.")

    def source_vjp(self, points_bohr: object, weights: object):
        raise NotImplementedError("Frozen QM density exposes no MACE-source VJP.")

    def nuclear_vjp(self, points_bohr: object, weights: object):
        raise NotImplementedError("QM-density nuclear response is outside this diagnostic.")


__all__ = [
    "QM_AO_DENSITY_LEVEL_SET_ID",
    "QMDensityBoundaryDiagnostics",
    "PySCFAODensityLevelSet",
]
