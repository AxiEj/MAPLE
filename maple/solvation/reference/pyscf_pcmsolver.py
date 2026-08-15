"""Exact fixed-geometry PySCF--PCMSolver electrostatic reference coupling.

The production Route-2 runtime must not depend on PySCF.  This module therefore
imports only NumPy at module import time and accepts the small structural
interfaces supplied by a pinned PySCF molecule and a pinned PCMSolver session.

For one fixed nuclear geometry, PCMSolver receives the *total* molecular MEP

``v_surface = v_nuclear - Tr[D r_surface^-1]``.

With a symmetric PCMSolver response matrix, the polarization energy is
``1/2 v_surface.T q`` and its derivative with respect to the AO density is the
one-electron reaction operator ``-sum_i q_i r_i^-1``.  ``PCMSolverSCFSolvent``
exposes exactly that energy/operator pair to PySCF's solvent attachment hook.
No fitted charge or MAPLE model source appears in this reference construction.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Protocol

import numpy as np

REFERENCE_COUPLING_VERSION = "route2-pyscf-pcmsolver-fixed-geometry-v1"


class _Molecule(Protocol):
    natm: int

    def nao_nr(self) -> int: ...

    def atom_coords(self) -> np.ndarray: ...

    def atom_charges(self) -> np.ndarray: ...

    def intor(self, name: str) -> np.ndarray: ...

    def with_rinv_origin(self, origin: np.ndarray): ...


class _PCMSolverSession(Protocol):
    @property
    def cavity_centers_bohr(self) -> np.ndarray: ...

    @property
    def response_operator_is_symmetric(self) -> bool: ...

    def compute_asc(
        self,
        mep: np.ndarray,
        *,
        mep_label: bytes | str = ...,
        asc_label: bytes | str = ...,
    ) -> np.ndarray: ...

    def solve(
        self,
        mep: np.ndarray,
        *,
        mep_label: bytes | str = ...,
        asc_label: bytes | str = ...,
    ) -> dict[str, np.ndarray | float]: ...


def array_sha256(values: object) -> str:
    """Hash a finite array including its dtype, shape, and exact bytes."""

    array = np.ascontiguousarray(np.asarray(values))
    if array.size == 0 or not np.issubdtype(array.dtype, np.number):
        raise ValueError("array hash input must be a non-empty numeric array.")
    if not np.all(np.isfinite(array)):
        raise ValueError("array hash input must be finite.")
    header = json.dumps(
        {
            "kind": "route2-reference-array-v1",
            "dtype": array.dtype.str,
            "shape": list(array.shape),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(header + b"\0" + array.tobytes()).hexdigest()


def _finite_square(values: object, *, size: int, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (size, size) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite with shape ({size}, {size}).")
    return np.ascontiguousarray(array)


def closed_shell_density_from_orbitals(
    mo_coeff: object,
    mo_occ: object,
    overlap: object,
) -> tuple[np.ndarray, dict[str, float | int | str]]:
    """Build and validate one restricted closed-shell AO density."""

    coefficients = np.asarray(mo_coeff, dtype=np.float64)
    occupations = np.asarray(mo_occ, dtype=np.float64)
    if (
        coefficients.ndim != 2
        or coefficients.shape[0] < 1
        or occupations.shape != (coefficients.shape[1],)
        or not np.all(np.isfinite(coefficients))
        or not np.all(np.isfinite(occupations))
    ):
        raise ValueError(
            "restricted orbitals/occupations have invalid shape or values."
        )
    occupation_distance = np.minimum(np.abs(occupations), np.abs(occupations - 2.0))
    if float(np.max(occupation_distance, initial=0.0)) > 1.0e-8:
        raise ValueError("restricted occupations must be integer closed-shell 0/2.")
    metric = _finite_square(overlap, size=coefficients.shape[0], name="AO overlap")
    orthonormality = float(
        np.max(
            np.abs(
                coefficients.T @ metric @ coefficients
                - np.eye(coefficients.shape[1], dtype=np.float64)
            )
        )
    )
    if orthonormality > 1.0e-7:
        raise ValueError("checkpoint orbitals are not orthonormal in the AO metric.")
    density = (coefficients * occupations[None, :]) @ coefficients.T
    density = np.ascontiguousarray(0.5 * (density + density.T))
    electron_count = float(np.einsum("ij,ji->", density, metric))
    occupation_sum = float(np.sum(occupations))
    binding = abs(electron_count - occupation_sum)
    if binding > 1.0e-7:
        raise ValueError("AO density electron count differs from its occupations.")
    return density, {
        "density_source": "checkpoint:scf/mo_coeff+scf/mo_occ",
        "electron_count_e": electron_count,
        "occupation_sum_e": occupation_sum,
        "occupied_orbital_count": int(np.count_nonzero(occupations > 1.0)),
        "mo_orthonormality_inf": orthonormality,
        "density_binding_residual_e": binding,
        "density_sha256": array_sha256(density),
    }


class AOInverseDistanceIntegralCache:
    """Fixed-surface AO ``1/|r-s_i|`` integrals with bounded-memory batches."""

    def __init__(
        self,
        molecule: _Molecule,
        surface_points_bohr: object,
        *,
        path: str | Path | None = None,
        batch_size: int = 16,
    ) -> None:
        points = np.asarray(surface_points_bohr, dtype=np.float64)
        if (
            points.ndim != 2
            or points.shape[0] < 1
            or points.shape[1] != 3
            or not np.all(np.isfinite(points))
        ):
            raise ValueError("surface points must be finite with shape (M,3).")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int):
            raise TypeError("batch_size must be an integer.")
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        nao = int(molecule.nao_nr())
        if nao < 1:
            raise ValueError("molecule must contain at least one AO.")
        self.molecule = molecule
        self.surface_points_bohr = np.ascontiguousarray(points)
        self.nao = nao
        self.batch_size = batch_size
        self.path = None if path is None else Path(path).expanduser().resolve()
        shape = (len(points), nao, nao)
        if self.path is None:
            storage: np.ndarray = np.empty(shape, dtype=np.float64)
            self.storage_kind = "memory"
        else:
            if self.path.exists():
                raise FileExistsError(self.path)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            storage = np.lib.format.open_memmap(
                self.path, mode="w+", dtype=np.float64, shape=shape
            )
            self.storage_kind = "npy-memmap"
        for index, point in enumerate(self.surface_points_bohr):
            with molecule.with_rinv_origin(point):
                integral = np.asarray(molecule.intor("int1e_rinv"), dtype=np.float64)
            if integral.shape != (nao, nao) or not np.all(np.isfinite(integral)):
                raise RuntimeError(
                    "PySCF returned an invalid AO inverse-distance block."
                )
            symmetry_error = float(np.max(np.abs(integral - integral.T)))
            if symmetry_error > 1.0e-10:
                raise RuntimeError("AO inverse-distance block is not symmetric.")
            storage[index] = 0.5 * (integral + integral.T)
        if isinstance(storage, np.memmap):
            storage.flush()
        self._storage = storage

    @property
    def shape(self) -> tuple[int, int, int]:
        return self._storage.shape

    @property
    def bytes(self) -> int:
        return int(np.prod(self.shape, dtype=np.int64) * np.dtype(np.float64).itemsize)

    def _batches(self):
        for start in range(0, self.shape[0], self.batch_size):
            stop = min(start + self.batch_size, self.shape[0])
            yield start, stop, np.asarray(self._storage[start:stop])

    def electronic_surface_potential(self, density: object) -> np.ndarray:
        dm = _finite_square(density, size=self.nao, name="AO density")
        result = np.empty(self.shape[0], dtype=np.float64)
        for start, stop, integrals in self._batches():
            result[start:stop] = np.einsum("xij,ji->x", integrals, dm, optimize=True)
        return result

    def reaction_potential_matrix(self, surface_charges: object) -> np.ndarray:
        charges = np.asarray(surface_charges, dtype=np.float64)
        if charges.shape != (self.shape[0],) or not np.all(np.isfinite(charges)):
            raise ValueError("surface charges have invalid shape or values.")
        result = np.zeros((self.nao, self.nao), dtype=np.float64)
        for start, stop, integrals in self._batches():
            result -= np.einsum(
                "x,xij->ij", charges[start:stop], integrals, optimize=True
            )
        return np.ascontiguousarray(0.5 * (result + result.T))

    def close(self) -> None:
        if isinstance(self._storage, np.memmap):
            self._storage.flush()


@dataclass(frozen=True, slots=True)
class PCMSolverDensityResponse:
    """One exact PCMSolver response to a fixed-geometry AO density."""

    total_surface_mep_hartree_per_e: np.ndarray
    apparent_surface_charge_e: np.ndarray
    polarization_energy_hartree: float
    reaction_potential_ao_hartree: np.ndarray

    def __post_init__(self) -> None:
        mep = np.asarray(self.total_surface_mep_hartree_per_e, dtype=np.float64)
        charge = np.asarray(self.apparent_surface_charge_e, dtype=np.float64)
        potential = np.asarray(self.reaction_potential_ao_hartree, dtype=np.float64)
        energy = float(self.polarization_energy_hartree)
        if (
            mep.ndim != 1
            or mep.size < 1
            or charge.shape != mep.shape
            or potential.ndim != 2
            or potential.shape[0] != potential.shape[1]
            or not np.all(np.isfinite(mep))
            or not np.all(np.isfinite(charge))
            or not np.all(np.isfinite(potential))
            or not np.isfinite(energy)
        ):
            raise ValueError("PCMSolver density response has invalid shape or values.")
        for name, value in (
            ("total_surface_mep_hartree_per_e", mep),
            ("apparent_surface_charge_e", charge),
            ("reaction_potential_ao_hartree", potential),
        ):
            copied = np.array(value, copy=True)
            copied.setflags(write=False)
            object.__setattr__(self, name, copied)
        object.__setattr__(self, "polarization_energy_hartree", energy)

    @property
    def direct_half_coupling_hartree(self) -> float:
        return float(
            0.5
            * np.dot(
                self.total_surface_mep_hartree_per_e,
                self.apparent_surface_charge_e,
            )
        )

    @property
    def half_coupling_residual_hartree(self) -> float:
        return abs(self.polarization_energy_hartree - self.direct_half_coupling_hartree)


class PCMSolverSCFSolvent:
    """Duck-typed PySCF solvent object backed by one PCMSolver session."""

    frozen = False
    equilibrium_solvation = True
    method = "IEFPCM"
    max_cycle = 100
    conv_tol = 1.0e-10
    state_id = 0
    _keys = {"method", "frozen", "equilibrium_solvation"}

    def __init__(
        self,
        molecule: _Molecule,
        session: _PCMSolverSession,
        integral_cache: AOInverseDistanceIntegralCache,
    ) -> None:
        if not bool(session.response_operator_is_symmetric):
            raise ValueError(
                "QM/PCMSolver reference SCF requires MATRIXSYMM=True; a "
                "non-symmetric response has no single density derivative here."
            )
        points = np.asarray(session.cavity_centers_bohr, dtype=np.float64)
        if (
            points.shape != integral_cache.surface_points_bohr.shape
            or not np.array_equal(points, integral_cache.surface_points_bohr)
        ):
            raise ValueError("integral cache and PCMSolver surface differ.")
        coordinates = np.asarray(molecule.atom_coords(), dtype=np.float64)
        charges = np.asarray(molecule.atom_charges(), dtype=np.float64)
        if (
            coordinates.shape != (int(molecule.natm), 3)
            or charges.shape != (int(molecule.natm),)
            or not np.all(np.isfinite(coordinates))
            or not np.all(np.isfinite(charges))
        ):
            raise ValueError("molecule coordinates/charges are invalid.")
        distances = np.linalg.norm(points[:, None, :] - coordinates[None, :, :], axis=2)
        if np.any(distances <= 1.0e-12):
            raise ValueError("a PCMSolver surface point coincides with a nucleus.")
        self.mol = molecule
        self.session = session
        self.integral_cache = integral_cache
        self.nuclear_surface_mep = (1.0 / distances) @ charges
        self.e: float | None = None
        self.v: np.ndarray | None = None
        self.last_response: PCMSolverDensityResponse | None = None

    def evaluate_density(self, density: object) -> PCMSolverDensityResponse:
        electronic = self.integral_cache.electronic_surface_potential(density)
        total_mep = self.nuclear_surface_mep - electronic
        solved = self.session.solve(total_mep)
        try:
            asc = np.asarray(solved["asc"], dtype=np.float64)
            energy = float(solved["polarization_energy"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                "PCMSolver returned an invalid response payload."
            ) from exc
        reaction = self.integral_cache.reaction_potential_matrix(asc)
        response = PCMSolverDensityResponse(
            total_surface_mep_hartree_per_e=total_mep,
            apparent_surface_charge_e=asc,
            polarization_energy_hartree=energy,
            reaction_potential_ao_hartree=reaction,
        )
        self.e = response.polarization_energy_hartree
        self.v = response.reaction_potential_ao_hartree
        self.last_response = response
        return response

    def kernel(self, density: object) -> tuple[float, np.ndarray]:
        response = self.evaluate_density(density)
        return (
            response.polarization_energy_hartree,
            np.asarray(response.reaction_potential_ao_hartree),
        )

    def response_potential(self, density_direction: object) -> np.ndarray:
        electronic = self.integral_cache.electronic_surface_potential(density_direction)
        delta_mep = -electronic
        delta_asc = self.session.compute_asc(
            delta_mep,
            mep_label="MAPLE_DM1_MEP",
            asc_label="MAPLE_DM1_ASC",
        )
        return self.integral_cache.reaction_potential_matrix(delta_asc)

    def _B_dot_x(self, density_direction: object) -> np.ndarray:
        return self.response_potential(density_direction)

    def check_sanity(self):
        return self

    def dump_flags(self, verbose=None):
        return self

    def reset(self, molecule=None):
        if molecule is not None and molecule is not self.mol:
            raise RuntimeError("PCMSolverSCFSolvent is fixed to one geometry.")
        return self


def attach_pcmsolver_to_scf(mean_field, solvent: PCMSolverSCFSolvent):
    """Attach the reference solvent through pinned PySCF's official mixin."""

    try:
        from pyscf.solvent import _attach_solvent
    except ImportError as exc:  # pragma: no cover - optional runtime guard
        raise RuntimeError(
            "PySCF is required for matched QM/PCM reference SCF."
        ) from exc
    return _attach_solvent._for_scf(mean_field, solvent)


def response_symmetry_defect(
    session: _PCMSolverSession,
    *,
    surface_size: int,
    seed: int = 20260815,
) -> dict[str, float]:
    """Evaluate a deterministic dot-product symmetry canary for PCMSolver."""

    if surface_size < 1:
        raise ValueError("surface_size must be positive.")
    rng = np.random.default_rng(seed)
    left = rng.standard_normal(surface_size)
    right = rng.standard_normal(surface_size)
    left_response = session.compute_asc(
        left, mep_label="MAPLE_SYM_LEFT", asc_label="MAPLE_SYM_LEFT_ASC"
    )
    right_response = session.compute_asc(
        right, mep_label="MAPLE_SYM_RIGHT", asc_label="MAPLE_SYM_RIGHT_ASC"
    )
    lhs = float(np.dot(left, right_response))
    rhs = float(np.dot(right, left_response))
    absolute = abs(lhs - rhs)
    relative = absolute / max(abs(lhs), abs(rhs), np.finfo(float).tiny)
    return {
        "left_dot_response_right": lhs,
        "right_dot_response_left": rhs,
        "absolute_defect": absolute,
        "relative_defect": relative,
    }


def solvent_energy_directional_derivative_error(
    solvent: PCMSolverSCFSolvent,
    *,
    density: object,
    direction: object,
    step: float = 1.0e-4,
) -> dict[str, float]:
    """Compare the solvent energy FD with its reaction-potential contraction."""

    dm = _finite_square(density, size=solvent.integral_cache.nao, name="AO density")
    delta = _finite_square(
        direction, size=solvent.integral_cache.nao, name="AO density direction"
    )
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("finite-difference step must be positive and finite.")
    norm = float(np.linalg.norm(delta))
    if norm <= 0.0:
        raise ValueError("AO density direction must be nonzero.")
    delta = delta / norm
    base = solvent.evaluate_density(dm)
    analytic = float(np.einsum("ij,ji->", base.reaction_potential_ao_hartree, delta))
    plus = solvent.evaluate_density(dm + step * delta).polarization_energy_hartree
    minus = solvent.evaluate_density(dm - step * delta).polarization_energy_hartree
    finite_difference = float((plus - minus) / (2.0 * step))
    solvent.evaluate_density(dm)
    absolute = abs(analytic - finite_difference)
    relative = absolute / max(
        abs(analytic), abs(finite_difference), np.finfo(float).tiny
    )
    return {
        "step": float(step),
        "analytic_hartree": analytic,
        "finite_difference_hartree": finite_difference,
        "absolute_error_hartree": absolute,
        "relative_error": relative,
    }


__all__ = [
    "AOInverseDistanceIntegralCache",
    "PCMSolverDensityResponse",
    "PCMSolverSCFSolvent",
    "REFERENCE_COUPLING_VERSION",
    "array_sha256",
    "attach_pcmsolver_to_scf",
    "closed_shell_density_from_orbitals",
    "response_symmetry_defect",
    "solvent_energy_directional_derivative_error",
]
