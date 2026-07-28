"""Optional PySCF SWIG PCM responses for the Route-2 research force path.

PySCF is imported lazily and is not a MAPLE runtime dependency.  This adapter
uses PySCF's own SWIG surface, PCM matrix definitions, and analytic operator
gradient; MAPLE supplies only an external surface MEP and keeps the direct,
transpose, and energy-conjugate responses distinct.  IEFPCM, C-PCM, and COSMO
remain separately named equations.  The private PySCF gradient-intermediate
layout is version-gated until upstream exposes a public external-MEP derivative
API.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import math
from typing import Any, Literal, Sequence
import warnings

import numpy as np
from numpy.linalg import LinAlgError
from ase.units import Bohr
from scipy.linalg import LinAlgWarning, lu_factor, lu_solve

from .continuum_derivative import (
    EXTERNAL_MEP_OPERATOR_DERIVATIVE_CONTRACT_VERSION,
)
from .continuum_response import (
    EXTERNAL_MEP_RESPONSE_CONTRACT_VERSION,
    SurfaceChargeState,
)
from .pyscf_runtime import (
    TESTED_PYSCF_VERSION,
    require_tested_pyscf_version,
)
from .route2_pcm_response import (
    ATOM_CENTERED_SURFACE_MOTION_CONTRACT_VERSION,
    AtomCenteredSurfacePCMReactionFieldLinearMap,
)


@dataclass(frozen=True)
class _PySCFRuntime:
    version: str
    gto: Any
    gen_grid: Any
    pcm: Any
    pcm_grad: Any


def _load_pyscf_runtime() -> _PySCFRuntime:
    try:
        pyscf = importlib.import_module("pyscf")
        gto = importlib.import_module("pyscf.gto")
        gen_grid = importlib.import_module("pyscf.dft.gen_grid")
        pcm = importlib.import_module("pyscf.solvent.pcm")
        pcm_grad = importlib.import_module("pyscf.solvent.grad.pcm")
    except ImportError as exc:
        raise ImportError(
            "The optional PySCF SWIG PCM research provider requires "
            f"PySCF {TESTED_PYSCF_VERSION}; MAPLE does not install it "
            "automatically."
        ) from exc
    return _PySCFRuntime(
        version=str(pyscf.__version__),
        gto=gto,
        gen_grid=gen_grid,
        pcm=pcm,
        pcm_grad=pcm_grad,
    )


def _validated_surface_vector(
    values: np.ndarray,
    *,
    surface_size: int,
    name: str,
) -> np.ndarray:
    vector = np.asarray(values, dtype=float)
    expected_shape = (surface_size,)
    if vector.shape != expected_shape or not np.all(np.isfinite(vector)):
        raise ValueError(
            f"{name} must be finite with shape {expected_shape}; "
            f"received {vector.shape}."
        )
    return vector


def _atomic_numbers(
    runtime: _PySCFRuntime,
    symbols: tuple[str, ...],
) -> np.ndarray:
    """Resolve real atomic numbers without coupling radii to elements."""

    atomic_numbers = np.empty(len(symbols), dtype=int)
    for index, symbol in enumerate(symbols):
        try:
            atomic_number = int(runtime.gto.charge(symbol))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"PySCF cannot resolve element {symbol}."
            ) from exc
        if atomic_number <= 0:
            raise ValueError(
                f"PySCF cannot resolve element {symbol} to a real atom."
            )
        atomic_numbers[index] = atomic_number
    return atomic_numbers


class _AtomIndexedSurfaceMolecule:
    """Minimal PySCF ``gen_surface`` view for per-atom cavity radii.

    PySCF 2.13.1 maps ``mol.elements`` through ``charge()`` before indexing
    the supplied radius vector.  Integer element labels pass through that
    function unchanged, so the labels below make the upstream routine index
    radii by atom rather than by element.  Coordinates and all subsequent
    surface, matrix, and gradient work remain in PySCF; the real molecule is
    retained separately for nuclear identities and gradient bookkeeping.
    """

    def __init__(self, molecule: Any) -> None:
        self._molecule = molecule
        self.natm = int(molecule.natm)
        self.elements = tuple(range(self.natm))

    def atom_coords(self, unit: str = "B") -> np.ndarray:
        return self._molecule.atom_coords(unit=unit)


def _generate_per_atom_radius_surface(
    runtime: _PySCFRuntime,
    molecule: Any,
    radii_angstrom: np.ndarray,
    *,
    grid_points_per_atom: int,
) -> dict[str, Any]:
    """Delegate SWIG construction to PySCF with one radius per atom."""

    atom_indices = tuple(range(int(molecule.natm)))
    try:
        resolved_indices = tuple(
            int(runtime.gto.charge(index)) for index in atom_indices
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "PySCF no longer accepts atom indices in its surface-radius lookup."
        ) from exc
    if resolved_indices != atom_indices:
        raise RuntimeError(
            "PySCF no longer preserves atom indices in its surface-radius lookup."
        )

    return runtime.pcm.gen_surface(
        _AtomIndexedSurfaceMolecule(molecule),
        ng=grid_points_per_atom,
        rad=np.asarray(radii_angstrom, dtype=float) / Bohr,
        surface_discretization_method="SWIG",
    )


def _surface_parent_indices(
    slices: Sequence[Sequence[int]],
    *,
    atom_count: int,
    surface_size: int,
) -> np.ndarray:
    if len(slices) != atom_count:
        raise RuntimeError(
            "PySCF SWIG surface must provide one grid slice per atom."
        )
    parents = np.empty(surface_size, dtype=int)
    expected_start = 0
    for atom_index, bounds in enumerate(slices):
        if len(bounds) != 2:
            raise RuntimeError("A PySCF SWIG grid slice must contain two bounds.")
        if not all(
            isinstance(bound, (int, np.integer))
            for bound in bounds
        ):
            raise RuntimeError("PySCF SWIG grid-slice bounds must be integers.")
        start, stop = (int(bounds[0]), int(bounds[1]))
        if start != expected_start or stop < start or stop > surface_size:
            raise RuntimeError(
                "PySCF SWIG grid slices must be contiguous and cover the surface."
            )
        parents[start:stop] = atom_index
        expected_start = stop
    if expected_start != surface_size:
        raise RuntimeError(
            "PySCF SWIG grid slices do not cover the complete surface."
        )
    return parents


_SUPPORTED_CONTINUUM_MODELS = frozenset({"iefpcm", "cpcm", "cosmo"})
_PYSCF_METHODS = {
    "iefpcm": "IEFPCM",
    "cpcm": "C-PCM",
    "cosmo": "COSMO",
}
_DISPLAY_NAMES = {
    "iefpcm": "IEFPCM",
    "cpcm": "C-PCM",
    "cosmo": "COSMO",
}


class PySCFSWIGPCMResponse:
    """One immutable PySCF SWIG PCM energy-conjugate response operator."""

    contract_version = EXTERNAL_MEP_RESPONSE_CONTRACT_VERSION
    operator_derivative_contract_version = (
        EXTERNAL_MEP_OPERATOR_DERIVATIVE_CONTRACT_VERSION
    )
    surface_motion_contract_version = (
        ATOM_CENTERED_SURFACE_MOTION_CONTRACT_VERSION
    )
    energy_response_is_reciprocal = True

    def __init__(
        self,
        symbols: Sequence[str],
        atom_positions_angstrom: np.ndarray,
        cavity_radii_angstrom: np.ndarray,
        *,
        continuum_model: Literal["iefpcm", "cpcm", "cosmo"],
        dielectric: float,
        lebedev_order: int,
        _runtime: _PySCFRuntime | None = None,
    ) -> None:
        runtime = _load_pyscf_runtime() if _runtime is None else _runtime
        require_tested_pyscf_version(
            runtime.version,
            feature="The private PySCF external-MEP gradient bridge",
        )

        normalized_model = str(continuum_model).strip().lower()
        if normalized_model not in _SUPPORTED_CONTINUUM_MODELS:
            raise ValueError(
                "continuum_model must be iefpcm, cpcm, or cosmo; "
                f"received {continuum_model!r}."
            )
        pyscf_method = _PYSCF_METHODS[normalized_model]
        display_name = _DISPLAY_NAMES[normalized_model]
        symbol_tuple = tuple(str(symbol) for symbol in symbols)
        positions = np.asarray(atom_positions_angstrom, dtype=float)
        radii = np.asarray(cavity_radii_angstrom, dtype=float)
        atom_count = len(symbol_tuple)
        if atom_count == 0:
            raise ValueError(
                f"PySCF SWIG/{display_name} requires at least one atom."
            )
        if positions.shape != (atom_count, 3) or not np.all(
            np.isfinite(positions)
        ):
            raise ValueError(
                "Atom positions must be finite with shape "
                f"({atom_count}, 3); received {positions.shape}."
            )
        if (
            radii.shape != (atom_count,)
            or not np.all(np.isfinite(radii))
            or np.any(radii <= 0.0)
        ):
            raise ValueError(
                "Cavity radii must be finite and positive with shape "
                f"({atom_count},); received {radii.shape}."
            )
        dielectric_value = float(dielectric)
        if not np.isfinite(dielectric_value) or dielectric_value <= 1.0:
            raise ValueError("The static dielectric must be finite and greater than 1.")
        order = int(lebedev_order)
        try:
            grid_points_per_atom = int(runtime.gen_grid.LEBEDEV_ORDER[order])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Unsupported PySCF Lebedev order: {lebedev_order}."
            ) from exc

        atomic_numbers = _atomic_numbers(runtime, symbol_tuple)
        mol = runtime.gto.M(
            atom=list(zip(symbol_tuple, positions.tolist(), strict=True)),
            unit="Angstrom",
            basis="sto-3g",
            charge=0,
            spin=0,
            verbose=0,
        )
        surface = _generate_per_atom_radius_surface(
            runtime,
            mol,
            radii,
            grid_points_per_atom=grid_points_per_atom,
        )
        try:
            points = np.asarray(surface["grid_coords"], dtype=float)
            slices = surface["gslice_by_atom"]
        except (KeyError, TypeError) as exc:
            raise RuntimeError(
                "PySCF SWIG surface omitted grid coordinates or atom slices."
            ) from exc
        if (
            points.ndim != 2
            or points.shape[0] == 0
            or points.shape[1] != 3
            or not np.all(np.isfinite(points))
        ):
            raise RuntimeError(
                "PySCF SWIG surface points must be finite with shape "
                "(n_surface, 3)."
            )
        parents = _surface_parent_indices(
            slices,
            atom_count=atom_count,
            surface_size=points.shape[0],
        )

        _, area = runtime.pcm.get_F_A(surface)
        D, S = runtime.pcm.get_D_S(
            surface,
            with_S=True,
            with_D=True,
        )
        area = np.asarray(area, dtype=float)
        D = np.asarray(D, dtype=float)
        S = np.asarray(S, dtype=float)
        surface_size = points.shape[0]
        if (
            area.shape != (surface_size,)
            or not np.all(np.isfinite(area))
            or np.any(area <= 0.0)
        ):
            raise RuntimeError(
                "PySCF SWIG surface areas must be finite and positive."
            )
        expected_matrix_shape = (surface_size, surface_size)
        if (
            D.shape != expected_matrix_shape
            or S.shape != expected_matrix_shape
            or not np.all(np.isfinite(D))
            or not np.all(np.isfinite(S))
        ):
            raise RuntimeError(
                f"PySCF {display_name} D/S matrices must be finite square "
                "surface matrices."
            )

        if normalized_model == "cpcm":
            f_epsilon = (dielectric_value - 1.0) / dielectric_value
            K = S
            R = -f_epsilon * np.eye(surface_size)
        elif normalized_model == "cosmo":
            f_epsilon = (
                (dielectric_value - 1.0) / (dielectric_value + 0.5)
            )
            K = S
            R = -f_epsilon * np.eye(surface_size)
        else:
            f_epsilon = (
                (dielectric_value - 1.0) / (dielectric_value + 1.0)
            )
            DA = D * area
            K = S - f_epsilon / (2.0 * math.pi) * (DA @ S)
            R = -f_epsilon * (
                np.eye(surface_size) - DA / (2.0 * math.pi)
            )
        if not np.all(np.isfinite(K)) or not np.all(np.isfinite(R)):
            raise RuntimeError(
                f"PySCF {display_name} K/R matrices contain non-finite values."
            )

        self._runtime = runtime
        self._continuum_model = normalized_model
        self._pyscf_method = pyscf_method
        self._display_name = display_name
        self._symbols = symbol_tuple
        self._positions_angstrom = positions.copy()
        self._radii_angstrom = radii.copy()
        self._dielectric = dielectric_value
        self._lebedev_order = order
        self._mol = mol
        self._surface = surface
        self._surface_points_bohr = points.copy()
        self._surface_parent_atom_indices = parents
        self._atomic_numbers = atomic_numbers
        self._area = area.copy()
        self._D = D.copy()
        self._S = S.copy()
        self._K = K
        self._R = R
        self._f_epsilon = f_epsilon
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", LinAlgWarning)
                self._lu_and_piv = lu_factor(K, check_finite=True)
        except (LinAlgError, LinAlgWarning, ValueError) as exc:
            raise RuntimeError(
                f"PySCF {display_name} response matrix factorization failed."
            ) from exc
        self.atom_count = atom_count
        self.surface_size = surface_size

    @property
    def atomic_numbers(self) -> np.ndarray:
        return self._atomic_numbers.copy()

    @property
    def reference_positions_bohr(self) -> np.ndarray:
        return self._positions_angstrom.copy() / Bohr

    @property
    def cavity_radii_angstrom(self) -> np.ndarray:
        return self._radii_angstrom.copy()

    @property
    def surface_points_bohr(self) -> np.ndarray:
        return self._surface_points_bohr.copy()

    @property
    def surface_areas_bohr2(self) -> np.ndarray:
        return self._area.copy()

    @property
    def surface_parent_atom_indices(self) -> np.ndarray:
        return self._surface_parent_atom_indices.copy()

    @property
    def runtime_provenance(self) -> dict[str, str | int | float]:
        return {
            "provider": f"pyscf-swig-{self._continuum_model}",
            "pyscf_version": self._runtime.version,
            "continuum_model": self._continuum_model,
            "pyscf_pcm_method": self._pyscf_method,
            "lebedev_order": self._lebedev_order,
            "static_dielectric": self._dielectric,
            "dielectric_scaling": self._f_epsilon,
            "equation_source": "pyscf.solvent.pcm.PCM.build",
            "cavity_radius_assignment": "per-atom",
        }

    def _solve_components(
        self,
        surface_potential_hartree_per_e: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        potential = _validated_surface_vector(
            surface_potential_hartree_per_e,
            surface_size=self.surface_size,
            name="surface_potential_hartree_per_e",
        )
        try:
            direct = lu_solve(
                self._lu_and_piv,
                self._R @ potential,
                check_finite=True,
            )
            inverse_transpose_potential = lu_solve(
                self._lu_and_piv,
                potential,
                trans=1,
                check_finite=True,
            )
        except (LinAlgError, ValueError) as exc:
            raise RuntimeError(
                f"PySCF {self._display_name} response solve failed."
            ) from exc
        adjoint = self._R.T @ inverse_transpose_potential
        conjugate = 0.5 * (direct + adjoint)
        if not all(
            np.all(np.isfinite(values))
            for values in (direct, adjoint, conjugate)
        ):
            raise RuntimeError(
                f"PySCF {self._display_name} response solve produced "
                "non-finite values."
            )
        return direct, adjoint, conjugate

    def apply_energy_conjugate(
        self,
        surface_potential_hartree_per_e: np.ndarray,
    ) -> np.ndarray:
        return self._solve_components(
            surface_potential_hartree_per_e
        )[2].copy()

    def solve(
        self,
        surface_potential_hartree_per_e: np.ndarray,
    ) -> SurfaceChargeState:
        potential = _validated_surface_vector(
            surface_potential_hartree_per_e,
            surface_size=self.surface_size,
            name="surface_potential_hartree_per_e",
        )
        direct, adjoint, conjugate = self._solve_components(potential)
        return SurfaceChargeState(
            surface_potential_hartree_per_e=potential,
            direct_surface_charge_e=direct,
            adjoint_surface_charge_e=adjoint,
            energy_conjugate_surface_charge_e=conjugate,
            polarization_energy_hartree=0.5 * float(
                np.dot(potential, conjugate)
            ),
        )

    def _polarization_operator_position_gradient(
        self,
        surface_potential_hartree_per_e: np.ndarray,
    ) -> np.ndarray:
        potential = _validated_surface_vector(
            surface_potential_hartree_per_e,
            surface_size=self.surface_size,
            name="surface_potential_hartree_per_e",
        )
        direct, _, conjugate = self._solve_components(potential)
        dm = np.zeros((self._mol.nao, self._mol.nao), dtype=float)
        pcm_object = self._runtime.pcm.PCM(self._mol)
        pcm_object.method = self._pyscf_method
        pcm_object.eps = self._dielectric
        pcm_object.surface_discretization_method = "SWIG"
        pcm_object.surface = self._surface
        pcm_object._intermediates = {
            "A": self._area,
            "D": self._D,
            "S": self._S,
            "K": self._K,
            "R": self._R,
            "f_epsilon": self._f_epsilon,
            "q": direct,
            "q_sym": conjugate,
            "v_grids": potential,
            "dm": dm.copy(),
        }
        gradient_hartree_per_bohr = np.asarray(
            self._runtime.pcm_grad.grad_solver(pcm_object, dm),
            dtype=float,
        )
        expected_shape = (self.atom_count, 3)
        if (
            gradient_hartree_per_bohr.shape != expected_shape
            or not np.all(np.isfinite(gradient_hartree_per_bohr))
        ):
            raise RuntimeError(
                "PySCF operator gradient must be finite with shape "
                f"{expected_shape}; received "
                f"{gradient_hartree_per_bohr.shape}."
            )
        return gradient_hartree_per_bohr / Bohr

    def operator_position_vjp(
        self,
        left_surface_potential_hartree_per_e: np.ndarray,
        right_surface_potential_hartree_per_e: np.ndarray,
    ) -> np.ndarray:
        left = _validated_surface_vector(
            left_surface_potential_hartree_per_e,
            surface_size=self.surface_size,
            name="left_surface_potential_hartree_per_e",
        )
        right = _validated_surface_vector(
            right_surface_potential_hartree_per_e,
            surface_size=self.surface_size,
            name="right_surface_potential_hartree_per_e",
        )
        return 0.5 * (
            self._polarization_operator_position_gradient(left + right)
            - self._polarization_operator_position_gradient(left - right)
        )

    def reaction_field_linear_map(
        self,
    ) -> AtomCenteredSurfacePCMReactionFieldLinearMap:
        """Return the same-energy forward/adjoint/full-VJP reaction map."""

        return AtomCenteredSurfacePCMReactionFieldLinearMap(
            self,
            self._positions_angstrom,
        )


class PySCFSWIGIEFPCMResponse(PySCFSWIGPCMResponse):
    """Version-locked PySCF SWIG/IEFPCM response."""

    def __init__(
        self,
        symbols: Sequence[str],
        atom_positions_angstrom: np.ndarray,
        cavity_radii_angstrom: np.ndarray,
        *,
        dielectric: float,
        lebedev_order: int,
        _runtime: _PySCFRuntime | None = None,
    ) -> None:
        super().__init__(
            symbols,
            atom_positions_angstrom,
            cavity_radii_angstrom,
            continuum_model="iefpcm",
            dielectric=dielectric,
            lebedev_order=lebedev_order,
            _runtime=_runtime,
        )


class PySCFSWIGCPCMResponse(PySCFSWIGPCMResponse):
    """Version-locked PySCF SWIG/C-PCM response."""

    def __init__(
        self,
        symbols: Sequence[str],
        atom_positions_angstrom: np.ndarray,
        cavity_radii_angstrom: np.ndarray,
        *,
        dielectric: float,
        lebedev_order: int,
        _runtime: _PySCFRuntime | None = None,
    ) -> None:
        super().__init__(
            symbols,
            atom_positions_angstrom,
            cavity_radii_angstrom,
            continuum_model="cpcm",
            dielectric=dielectric,
            lebedev_order=lebedev_order,
            _runtime=_runtime,
        )


class PySCFSWIGCOSMOResponse(PySCFSWIGPCMResponse):
    """Version-locked PySCF SWIG/COSMO response."""

    def __init__(
        self,
        symbols: Sequence[str],
        atom_positions_angstrom: np.ndarray,
        cavity_radii_angstrom: np.ndarray,
        *,
        dielectric: float,
        lebedev_order: int,
        _runtime: _PySCFRuntime | None = None,
    ) -> None:
        super().__init__(
            symbols,
            atom_positions_angstrom,
            cavity_radii_angstrom,
            continuum_model="cosmo",
            dielectric=dielectric,
            lebedev_order=lebedev_order,
            _runtime=_runtime,
        )


__all__ = [
    "TESTED_PYSCF_VERSION",
    "PySCFSWIGCPCMResponse",
    "PySCFSWIGCOSMOResponse",
    "PySCFSWIGIEFPCMResponse",
    "PySCFSWIGPCMResponse",
]
