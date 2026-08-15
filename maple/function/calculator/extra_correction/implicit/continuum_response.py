"""Provider-neutral external-MEP continuum response for Route 2.

The contract in this module is deliberately smaller than a complete implicit
solvent provider.  It owns one fixed cavity/operator and maps an externally
supplied molecular electrostatic potential (MEP) to the surface charge that is
conjugate to the polarization energy.  Solute projection, ML density response,
CDS, and nuclear derivatives remain separate Route-2 components.

For a nonsymmetric IEFPCM discretization the direct and transpose solves need
not agree.  The energy-conjugate charge is therefore

``q_sym = 0.5 * (q_direct + q_adjoint)``

and the discrete polarization energy is

``E_pol = 0.5 * dot(v_surface, q_sym)``.

The current PCMSolver profile requests ``MATRIXSYMM=TRUE``, so all three
charges coincide.  Keeping the distinction explicit prevents a future smooth
SWIG provider from silently using the wrong energy or reaction field.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Protocol

import numpy as np

from .pcmsolver import PCMSolverSession


EXTERNAL_MEP_RESPONSE_CONTRACT_VERSION = 1
PCMSOLVER_EXTERNAL_MEP_PROVIDER_ID = (
    "maple.route2.continuum.pcmsolver-symmetric-external-mep.impl.v1"
)
PCMSOLVER_EXTERNAL_MEP_CONTINUUM_PROFILE_ID = (
    "pcmsolver-symmetric-external-mep-electrostatic-v1"
)
PCMSOLVER_EXTERNAL_MEP_CAVITY_PROFILE_ID = (
    "pcmsolver-input-defined-gepol-cavity-v1"
)


def _sha256_file(path: Path, *, name: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"{name} is unavailable: {path}.")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(values: object, *, name: str) -> str:
    array = np.asarray(values)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a nonempty finite array.")
    contiguous = np.ascontiguousarray(array)
    header = json.dumps(
        {"dtype": contiguous.dtype.str, "shape": list(contiguous.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(header + b"\0" + contiguous.tobytes()).hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _immutable_vector(
    values: np.ndarray,
    *,
    name: str,
    length: int | None = None,
    positive: bool = False,
) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    valid_length = length is None or array.shape == (length,)
    if array.ndim != 1 or array.size == 0 or not valid_length:
        expected = "(n,)" if length is None else f"({length},)"
        raise ValueError(
            f"{name} must have shape {expected}; received {array.shape}."
        )
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    if positive and np.any(array <= 0.0):
        raise ValueError(f"{name} must contain only positive values.")
    result = np.array(array, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _immutable_matrix(
    values: np.ndarray,
    *,
    name: str,
    rows: int | None = None,
) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    expected_shape = None if rows is None else (rows, 3)
    valid_shape = (
        array.ndim == 2
        and array.shape[0] > 0
        and array.shape[1] == 3
        and (expected_shape is None or array.shape == expected_shape)
    )
    if not valid_shape or not np.all(np.isfinite(array)):
        expected = "(n, 3)" if rows is None else str(expected_shape)
        raise ValueError(
            f"{name} must be finite with shape {expected}; "
            f"received {array.shape}."
        )
    result = np.array(array, dtype=float, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class SurfaceChargeState:
    """Validated external-MEP response for one fixed continuum operator."""

    surface_potential_hartree_per_e: np.ndarray
    direct_surface_charge_e: np.ndarray
    adjoint_surface_charge_e: np.ndarray
    energy_conjugate_surface_charge_e: np.ndarray
    polarization_energy_hartree: float

    def __post_init__(self) -> None:
        potential = _immutable_vector(
            self.surface_potential_hartree_per_e,
            name="surface_potential_hartree_per_e",
        )
        direct = _immutable_vector(
            self.direct_surface_charge_e,
            name="direct_surface_charge_e",
            length=potential.size,
        )
        adjoint = _immutable_vector(
            self.adjoint_surface_charge_e,
            name="adjoint_surface_charge_e",
            length=potential.size,
        )
        conjugate = _immutable_vector(
            self.energy_conjugate_surface_charge_e,
            name="energy_conjugate_surface_charge_e",
            length=potential.size,
        )
        expected_conjugate = 0.5 * (direct + adjoint)
        if not np.allclose(
            conjugate,
            expected_conjugate,
            rtol=1.0e-12,
            atol=1.0e-14,
        ):
            raise ValueError(
                "energy_conjugate_surface_charge_e must be the arithmetic "
                "mean of the direct and adjoint surface charges."
            )

        energy = float(self.polarization_energy_hartree)
        if not np.isfinite(energy):
            raise ValueError("polarization_energy_hartree must be finite.")
        expected_energy = 0.5 * float(np.dot(potential, conjugate))
        tolerance = max(1.0e-10, 1.0e-8 * abs(expected_energy))
        if abs(energy - expected_energy) > tolerance:
            raise ValueError(
                "polarization_energy_hartree must equal "
                "0.5*dot(surface_potential, energy_conjugate_surface_charge)."
            )

        object.__setattr__(self, "surface_potential_hartree_per_e", potential)
        object.__setattr__(self, "direct_surface_charge_e", direct)
        object.__setattr__(self, "adjoint_surface_charge_e", adjoint)
        object.__setattr__(
            self,
            "energy_conjugate_surface_charge_e",
            conjugate,
        )
        object.__setattr__(self, "polarization_energy_hartree", energy)


class ExternalMEPCavityResponse(Protocol):
    """Fixed continuum cavity/operator driven by an external surface MEP."""

    contract_version: int
    energy_response_is_reciprocal: bool
    atom_count: int
    provider_id: str
    continuum_profile_id: str
    cavity_profile_id: str

    def configuration_sha256(self) -> str:
        """Content address the live continuum and cavity configuration."""
        ...

    def cavity_configuration_sha256(self) -> str:
        """Content address only the live cavity geometry and input policy."""
        ...

    @property
    def atomic_numbers(self) -> np.ndarray:
        """Atomic numbers defining the fixed cavity."""
        ...

    @property
    def reference_positions_bohr(self) -> np.ndarray:
        """Solute geometry at which the cavity/operator was built."""
        ...

    @property
    def cavity_radii_angstrom(self) -> np.ndarray:
        """Per-atom radii used to define the cavity."""
        ...

    @property
    def surface_points_bohr(self) -> np.ndarray:
        """Current fixed surface points."""
        ...

    @property
    def surface_areas_bohr2(self) -> np.ndarray:
        """Current fixed surface areas."""
        ...

    def apply_energy_conjugate(
        self,
        surface_potential_hartree_per_e: np.ndarray,
    ) -> np.ndarray:
        """Apply the reciprocal energy-conjugate surface response."""
        ...

    def solve(
        self,
        surface_potential_hartree_per_e: np.ndarray,
    ) -> SurfaceChargeState:
        """Return direct, adjoint, conjugate charge, and polarization energy."""
        ...


class PCMSolverExternalMEPCavityResponse:
    """Adapt one open symmetric PCMSolver session to the external-MEP contract."""

    contract_version = EXTERNAL_MEP_RESPONSE_CONTRACT_VERSION
    energy_response_is_reciprocal = True
    provider_id = PCMSOLVER_EXTERNAL_MEP_PROVIDER_ID
    continuum_profile_id = PCMSOLVER_EXTERNAL_MEP_CONTINUUM_PROFILE_ID
    cavity_profile_id = PCMSOLVER_EXTERNAL_MEP_CAVITY_PROFILE_ID

    def __init__(
        self,
        session: PCMSolverSession,
        *,
        cavity_radii_angstrom: np.ndarray,
    ) -> None:
        if not session.response_operator_is_symmetric:
            raise ValueError(
                "Route-2 external-MEP response requires PCMSolver "
                "MATRIXSYMM=TRUE."
            )
        atomic_numbers = _immutable_vector(
            session.atomic_numbers,
            name="atomic_numbers",
            positive=True,
        )
        positions = _immutable_matrix(
            session.coordinates_bohr,
            name="reference_positions_bohr",
            rows=atomic_numbers.size,
        )
        radii = _immutable_vector(
            cavity_radii_angstrom,
            name="cavity_radii_angstrom",
            length=atomic_numbers.size,
            positive=True,
        )
        points = _immutable_matrix(
            session.cavity_centers_bohr,
            name="surface_points_bohr",
        )
        _immutable_vector(
            session.cavity_areas_bohr2,
            name="surface_areas_bohr2",
            length=points.shape[0],
            positive=True,
        )

        self._session = session
        self._atomic_numbers = atomic_numbers
        self._reference_positions_bohr = positions
        self._cavity_radii_angstrom = radii
        self.atom_count = int(atomic_numbers.size)
        self._surface_size = points.shape[0]
        try:
            self._configuration_digest = self._live_configuration_sha256()
            self._cavity_configuration_digest = (
                self._live_cavity_configuration_sha256()
            )
        except (AttributeError, FileNotFoundError):
            # Lightweight algebra tests use an intentionally file-free fake
            # session.  It may exercise the numerical response, but it cannot
            # be promoted to a content-addressed public runtime.
            self._configuration_digest = None
            self._cavity_configuration_digest = None

    def _runtime_file_hashes(self) -> tuple[str, str]:
        parsed_input = Path(self._session.parsed_input_path).expanduser().resolve()
        library = Path(self._session.library_source).expanduser().resolve()
        return (
            _sha256_file(parsed_input, name="PCMSolver parsed input"),
            _sha256_file(library, name="PCMSolver shared library"),
        )

    def _live_cavity_configuration_sha256(self) -> str:
        parsed_input_sha256, _library_sha256 = self._runtime_file_hashes()
        return _canonical_sha256(
            {
                "contract": "pcmsolver-input-defined-gepol-cavity-v1",
                "cavity_profile_id": self.cavity_profile_id,
                "parsed_input_sha256": parsed_input_sha256,
                "atomic_numbers_sha256": _array_sha256(
                    self._atomic_numbers, name="atomic numbers"
                ),
                "reference_positions_bohr_sha256": _array_sha256(
                    self._reference_positions_bohr,
                    name="reference positions",
                ),
                "cavity_radii_angstrom_sha256": _array_sha256(
                    self._cavity_radii_angstrom,
                    name="cavity radii",
                ),
                "surface_points_bohr_sha256": _array_sha256(
                    self.surface_points_bohr,
                    name="surface points",
                ),
                "surface_areas_bohr2_sha256": _array_sha256(
                    self.surface_areas_bohr2,
                    name="surface areas",
                ),
            }
        )

    def _live_configuration_sha256(self) -> str:
        parsed_input_sha256, library_sha256 = self._runtime_file_hashes()
        return _canonical_sha256(
            {
                "contract": "pcmsolver-symmetric-external-mep-response-v1",
                "contract_version": self.contract_version,
                "provider_id": self.provider_id,
                "continuum_profile_id": self.continuum_profile_id,
                "cavity_profile_id": self.cavity_profile_id,
                "energy_response_is_reciprocal": (
                    self.energy_response_is_reciprocal
                ),
                "parsed_input_sha256": parsed_input_sha256,
                "library_sha256": library_sha256,
                "cavity_configuration_sha256": (
                    self._live_cavity_configuration_sha256()
                ),
            }
        )

    def configuration_sha256(self) -> str:
        if self._configuration_digest is None:
            raise RuntimeError(
                "PCMSolver response lacks content-addressed input/library provenance."
            )
        current = self._live_configuration_sha256()
        if current != self._configuration_digest:
            raise RuntimeError("PCMSolver external-MEP configuration drifted.")
        return current

    def cavity_configuration_sha256(self) -> str:
        if self._cavity_configuration_digest is None:
            raise RuntimeError(
                "PCMSolver cavity lacks content-addressed input provenance."
            )
        current = self._live_cavity_configuration_sha256()
        if current != self._cavity_configuration_digest:
            raise RuntimeError("PCMSolver cavity configuration drifted.")
        return current

    @property
    def atomic_numbers(self) -> np.ndarray:
        return self._atomic_numbers.copy()

    @property
    def reference_positions_bohr(self) -> np.ndarray:
        return self._reference_positions_bohr.copy()

    @property
    def cavity_radii_angstrom(self) -> np.ndarray:
        return self._cavity_radii_angstrom.copy()

    @property
    def surface_points_bohr(self) -> np.ndarray:
        return _immutable_matrix(
            self._session.cavity_centers_bohr,
            name="surface_points_bohr",
        ).copy()

    @property
    def surface_areas_bohr2(self) -> np.ndarray:
        points = self.surface_points_bohr
        return _immutable_vector(
            self._session.cavity_areas_bohr2,
            name="surface_areas_bohr2",
            length=points.shape[0],
            positive=True,
        ).copy()

    def _validated_surface_potential(
        self,
        values: np.ndarray,
    ) -> np.ndarray:
        return _immutable_vector(
            values,
            name="surface_potential_hartree_per_e",
            length=self._surface_size,
        )

    def apply_energy_conjugate(
        self,
        surface_potential_hartree_per_e: np.ndarray,
    ) -> np.ndarray:
        potential = self._validated_surface_potential(
            surface_potential_hartree_per_e
        )
        charge = _immutable_vector(
            self._session.compute_asc(potential),
            name="energy_conjugate_surface_charge_e",
            length=potential.size,
        )
        return charge.copy()

    def solve(
        self,
        surface_potential_hartree_per_e: np.ndarray,
    ) -> SurfaceChargeState:
        potential = self._validated_surface_potential(
            surface_potential_hartree_per_e
        )
        solved = self._session.solve(potential)
        try:
            direct = solved["asc"]
            energy = solved["polarization_energy"]
        except (KeyError, TypeError) as exc:
            raise RuntimeError(
                "PCMSolver solve() must return asc and polarization_energy."
            ) from exc
        direct_charge = _immutable_vector(
            np.asarray(direct, dtype=float),
            name="direct_surface_charge_e",
            length=potential.size,
        )
        return SurfaceChargeState(
            surface_potential_hartree_per_e=potential,
            direct_surface_charge_e=direct_charge,
            adjoint_surface_charge_e=direct_charge,
            energy_conjugate_surface_charge_e=direct_charge,
            polarization_energy_hartree=float(energy),
        )


__all__ = [
    "EXTERNAL_MEP_RESPONSE_CONTRACT_VERSION",
    "ExternalMEPCavityResponse",
    "PCMSOLVER_EXTERNAL_MEP_CAVITY_PROFILE_ID",
    "PCMSOLVER_EXTERNAL_MEP_CONTINUUM_PROFILE_ID",
    "PCMSOLVER_EXTERNAL_MEP_PROVIDER_ID",
    "PCMSolverExternalMEPCavityResponse",
    "SurfaceChargeState",
]
