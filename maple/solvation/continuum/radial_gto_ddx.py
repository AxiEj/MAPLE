"""Disabled eight-channel radial-GTO adapter for ddX PCM/COSMO.

The legacy pyddx adapter represents the solute by one atom-centred point
``l<=1`` multipole block.  That map cannot be reused for the vNext
MACE-POLAR radial source because it would collapse the two Gaussian widths.

This adapter instead supplies both pieces of ddX's public general-source
state, using one linear map ``D`` throughout:

* ``phi`` is the finite-width radial-GTO potential at the exposed ddX cavity
  nodes;
* ``psi`` is the exact density-pairing coefficient of the same normalized
  Gaussian multipoles.  Outside their centres a spherical Gaussian has the
  same Coulomb multipoles, so the two widths contribute separately to
  ``phi`` but add to the corresponding local ``psi`` coefficient.

The ddX scalar is ``0.5 * <psi, x>``.  Its source derivative is therefore
obtained from the same ``D`` and the ddX adjoint, not from a separate receiver
model.  All E/F/H/V/M release capabilities remain closed: a finite ddX
Lebedev/active-set discretization is not structurally rotation equivariant or
globally smooth merely because this fixed-geometry coupling is conjugate.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
import json
import math
from pathlib import Path
import re
from typing import Any, Literal, Sequence

import numpy as np
from ase.data import atomic_numbers
from ase.units import Bohr

from maple.function.calculator.extra_correction.implicit.gto_density import (
    cartesian_multipoles,
)
from maple.function.calculator.extra_correction.implicit.smd_cds import (
    smd_water_coulomb_radii,
)
from maple.solvation.api.capabilities import CapabilityStatus
from maple.solvation.api.profiles import (
    DDX_WATER_DDPCM_194_CONFIGURATION_CONTRACT_ID,
    UNBOUND_CONTINUUM_CONFIGURATION_CONTRACT_ID,
)
from maple.solvation.api.scalar_registry import (
    DIAGNOSTIC_DDX_DDCOSMO_RADIAL_GTO_ELECTROSTATIC_V1,
    DIAGNOSTIC_DDX_DDPCM_RADIAL_GTO_ELECTROSTATIC_V1,
)
from maple.solvation.api.units import HARTREE_TO_EV
from maple.solvation.coupling.exact_gto import (
    MACE_POLAR_RADIAL_GTO_COUPLING_ID,
    MACE_POLAR_RADIAL_SIGMAS_ANGSTROM,
    MACEPolarRadialGTOCoupling,
    OwnedFixedSurfaceGeometry,
)
from maple.solvation.coupling.gaussian_multipole_derivatives import (
    gaussian_multipole_potential_displacement_gradient,
    gaussian_multipole_potential_position_vjp,
)
from maple.solvation.coupling.metrics import MACE_POLAR_RADIAL_GTO_PAIRING
from maple.solvation.coupling.spaces import (
    MACE_POLAR_RADIAL_GTO_FIELD_DUAL_SPACE,
    MACE_POLAR_RADIAL_GTO_SOURCE_SPACE,
)

TESTED_PYDDX_VERSION = "0.8.0"
DDX_SOURCE_COMMIT = "4d79e3d9caeae5e602683572a71cb550414f9b09"
DDX_RADIAL_PROVIDER_ID = "maple.route2.continuum.ddx-radial-gto.impl.v1"
DDX_PCM_PROFILE_ID = "ddx-ddpcm-radial-gto-v1"
DDX_COSMO_PROFILE_ID = "ddx-ddcosmo-radial-gto-v1"
DDX_CAVITY_PROFILE_ID = "ddx-union-of-spheres-exposed-lebedev-v0p8p0"
# Compatibility spelling retained for the first local prototype.  The value is
# authoritative in the dependency-light profile registry.
DDX_WATER_194_CONFIGURATION_CONTRACT_ID = DDX_WATER_DDPCM_194_CONFIGURATION_CONTRACT_ID
DDX_WATER_DIELECTRIC = 78.39
DDX_WATER_LMAX = 8
DDX_WATER_LEBEDEV_POINTS = 194
_STATE_CONTRACT = "ddx-radial-gto-state-v2"
_IMPLEMENTATION_FILES = (
    "continuum/radial_gto_ddx.py",
    "coupling/exact_gto.py",
    "coupling/gaussian_multipole_derivatives.py",
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _sha(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _digest(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA256 digest.")
    return value


def _tuple(values: object, shape: tuple[int, ...], name: str) -> tuple[float, ...]:
    array = np.asarray(values, dtype=float)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite with shape {shape}.")
    return tuple(float(value) for value in array.reshape(-1))


def _readonly(values: tuple[float, ...], shape: tuple[int, ...]) -> np.ndarray:
    result = np.asarray(values, dtype=float).reshape(shape).copy()
    result.setflags(write=False)
    return result


def _implementation_sha256() -> tuple[tuple[str, str], ...]:
    root = Path(__file__).resolve().parents[1]
    result = []
    for logical_name in _IMPLEMENTATION_FILES:
        path = root / logical_name
        if not path.is_file():
            raise FileNotFoundError(
                f"ddX radial implementation dependency is unavailable: {logical_name}."
            )
        result.append((logical_name, hashlib.sha256(path.read_bytes()).hexdigest()))
    return tuple(result)


def _positions(geometry: Any, symbols: tuple[str, ...]) -> np.ndarray:
    getter = getattr(geometry, "get_positions", None)
    values = getter() if callable(getter) else geometry
    result = np.asarray(values, dtype=float)
    if result.shape != (len(symbols), 3) or not np.all(np.isfinite(result)):
        raise ValueError(
            f"geometry positions must be finite with shape ({len(symbols)}, 3)."
        )
    number_getter = getattr(geometry, "get_atomic_numbers", None)
    if callable(number_getter):
        expected = np.asarray([atomic_numbers[symbol] for symbol in symbols], dtype=int)
        actual = np.asarray(number_getter(), dtype=int)
        if actual.shape != expected.shape or not np.array_equal(actual, expected):
            raise ValueError(
                "ASE geometry atomic numbers do not match backend symbols."
            )
    pbc_getter = getattr(geometry, "get_pbc", None)
    if callable(pbc_getter):
        pbc = np.asarray(pbc_getter(), dtype=bool)
        if pbc.shape != (3,) or np.any(pbc):
            raise ValueError(
                "The ddX radial candidate accepts only nonperiodic geometry."
            )
    return np.array(result, copy=True)


def _geometry_sha256(positions_angstrom: np.ndarray, symbols: tuple[str, ...]) -> str:
    """Bind the exact nonperiodic geometry consumed by the ddX adapter."""

    positions = np.ascontiguousarray(positions_angstrom, dtype="<f8")
    return _sha(
        {
            "contract": "ddx-radial-gto-geometry-v1",
            "symbols": list(symbols),
            "positions_angstrom_hex": positions.tobytes(order="C").hex(),
        }
    )


def _cavity_topology_sha256(
    cavity_points_bohr: np.ndarray,
    positions_angstrom: np.ndarray,
    radii_angstrom: np.ndarray,
    owners: np.ndarray,
) -> str:
    """Hash only the discrete exposed-node topology, not its coordinates.

    ddX uses laboratory-fixed Lebedev directions.  Expressing each exposed
    point in its parent-sphere frame therefore identifies the active grid
    nodes while remaining invariant to continuous translations of the
    parent centres.  A changed exposed-node set changes this digest even when
    the total number of cavity points happens to remain constant.
    """

    positions_bohr = np.asarray(positions_angstrom, dtype=float) / Bohr
    radii_bohr = np.asarray(radii_angstrom, dtype=float) / Bohr
    parent = np.asarray(owners, dtype=np.int64)
    directions = (
        np.asarray(cavity_points_bohr, dtype=float) - positions_bohr[parent]
    ) / radii_bohr[parent, None]
    if directions.shape != (len(parent), 3) or not np.all(np.isfinite(directions)):
        raise RuntimeError("ddX cavity topology directions are invalid.")
    # The public pyddx cavity coordinates contain roundoff at roughly 1e-15.
    # Twelve decimals retain the exact Lebedev-node identity without hashing
    # irrelevant reconstruction noise.
    quantized = np.round(directions, decimals=12)
    return _sha(
        {
            "contract": "ddx-exposed-lebedev-node-topology-v1",
            "owners": parent.tolist(),
            "parent_directions": quantized.tolist(),
        }
    )


def _runtime() -> Any:
    try:
        module = importlib.import_module("pyddx")
    except ImportError as exc:  # pragma: no cover - optional runtime boundary
        raise RuntimeError(
            "The radial-GTO ddX candidate requires optional pyddx==0.8.0."
        ) from exc
    version = str(getattr(module, "__version__", "unknown"))
    if version != TESTED_PYDDX_VERSION:
        raise RuntimeError(
            "The radial-GTO ddX candidate is tested only with pyddx 0.8.0; "
            f"received {version}."
        )
    return module


def _radial_coefficients(values: np.ndarray) -> np.ndarray:
    source = np.asarray(values, dtype=float)
    if source.ndim != 2 or source.shape[1] != 8 or not np.all(np.isfinite(source)):
        raise ValueError("radial source must be finite with shape (n_atoms, 8).")
    result = np.empty((len(source), 2, 4), dtype=float)
    result[:, 0] = source[:, (0, 2, 3, 4)]
    result[:, 1] = source[:, (1, 5, 6, 7)]
    return result


class RadialDDXProblem:
    """One prepared ddX geometry plus the exact radial source/receiver map.

    ``solve_general`` deliberately accepts an independently assembled ddX
    ``(psi, phi)`` pair.  This is the small reusable boundary needed by
    heterogeneous solute models: a permanent point-multipole source can be
    combined with a finite-width induced source without pretending that both
    branches share one source kernel.  The returned eight-channel field is
    still the exact derivative of the same ddX scalar with respect to the
    radial-GTO source coordinates.
    """

    __slots__ = (
        "_dielectric_scaling",
        "_runtime",
        "_solver_tolerance",
        "cavity_topology_sha256",
        "model",
        "phi_matrix",
        "positions",
        "psi_matrix",
    )

    def __init__(
        self,
        model: Any,
        positions: np.ndarray,
        phi_matrix: np.ndarray,
        psi_matrix: np.ndarray,
        *,
        runtime: Any,
        solver_tolerance: float,
        dielectric_scaling: float,
        cavity_topology_sha256: str,
    ) -> None:
        self.model = model
        self.positions = np.array(positions, copy=True)
        self.phi_matrix = np.array(phi_matrix, copy=True)
        self.psi_matrix = np.array(psi_matrix, copy=True)
        self._runtime = runtime
        self._solver_tolerance = float(solver_tolerance)
        self._dielectric_scaling = float(dielectric_scaling)
        self.cavity_topology_sha256 = _digest(
            cavity_topology_sha256, "cavity_topology_sha256"
        )
        for array in (self.positions, self.phi_matrix, self.psi_matrix):
            array.setflags(write=False)

    def solve_general(
        self,
        psi: object,
        phi: object,
    ) -> tuple[Any, float, np.ndarray]:
        """Solve one general-source ddX state and return ``(state,E,dE/dc)``.

        ``psi`` and ``phi`` are the two source objects required by ddX.  The
        gradient is only with respect to the prepared radial-GTO coordinates;
        any independent permanent-source derivative remains owned by that
        source adapter.
        """

        psi_values = np.asarray(psi, dtype=float)
        phi_values = np.asarray(phi, dtype=float)
        expected_psi_shape = (int(self.model.n_basis), len(self.positions))
        expected_phi_shape = (int(self.model.n_cav),)
        if psi_values.shape != expected_psi_shape or not np.all(
            np.isfinite(psi_values)
        ):
            raise ValueError(
                "ddX psi must be finite with shape " f"{expected_psi_shape}."
            )
        if phi_values.shape != expected_phi_shape or not np.all(
            np.isfinite(phi_values)
        ):
            raise ValueError(
                "ddX phi must be finite with shape " f"{expected_phi_shape}."
            )
        state = self._runtime.State(self.model, psi_values, phi_values)
        state.fill_guess(self._solver_tolerance)
        state.solve(self._solver_tolerance)
        state.fill_guess_adjoint(self._solver_tolerance)
        state.solve_adjoint(self._solver_tolerance)
        forward = np.asarray(state.x, dtype=float).reshape(-1)
        adjoint = np.asarray(state.xi, dtype=float)
        if not np.all(np.isfinite(forward)) or not np.all(np.isfinite(adjoint)):
            raise RuntimeError("ddX general-source solve returned non-finite states.")
        raw_field = 0.5 * (self.psi_matrix.T @ forward - self.phi_matrix.T @ adjoint)
        field = (
            self._dielectric_scaling
            * HARTREE_TO_EV
            * raw_field.reshape(len(self.positions), 8)
        )
        energy = self._dielectric_scaling * HARTREE_TO_EV * float(state.energy())
        if not math.isfinite(energy) or not np.all(np.isfinite(field)):
            raise RuntimeError("ddX general-source energy/field is non-finite.")
        return state, energy, field

    def external_mep_model_field(self, state: Any) -> np.ndarray:
        """Return the boundary-MEP receiver used by an operational model.

        ddX's general-source energy derivative contains equal ``psi`` and
        ``phi`` contributions only when the represented density is supported
        inside the cavity.  A normalized Gaussian has nonzero tails outside
        every finite union of spheres, so that identity cannot be assumed for
        MACE-POLAR's radial source.  The operational external-MEP receiver is
        therefore the complete ``phi``-side adjoint contraction,

        ``-D_phi.T @ xi``,

        rather than the gradient of the general-source energy.  The two
        objects are intentionally exposed separately by the higher-level
        separated-source adapter.
        """

        adjoint = np.asarray(getattr(state, "xi", None), dtype=float)
        if adjoint.shape != (int(self.model.n_cav),) or not np.all(
            np.isfinite(adjoint)
        ):
            raise RuntimeError("ddX external-MEP adjoint state is invalid.")
        field = (
            -self._dielectric_scaling
            * HARTREE_TO_EV
            * (self.phi_matrix.T @ adjoint).reshape(len(self.positions), 8)
        )
        if not np.all(np.isfinite(field)):
            raise RuntimeError("ddX external-MEP model field is non-finite.")
        return field

    def external_mep_model_field_vjp(
        self,
        psi: object,
        phi: object,
    ) -> np.ndarray:
        """Apply the transpose of the external-MEP radial field map.

        If ``K = -D_phi.T L^{-T} D_psi`` is the operational receiver map,
        this method returns ``K.T`` applied to the radial cotangent encoded by
        ``(psi, phi)``.  The forward ddX solution supplies the required
        ``D_psi.T L^{-1} D_phi`` contraction without forming ``K``.
        """

        state, _energy, _energy_gradient = self.solve_general(psi, phi)
        forward = np.asarray(state.x, dtype=float).reshape(-1)
        result = (
            self._dielectric_scaling
            * HARTREE_TO_EV
            * (self.psi_matrix.T @ forward).reshape(len(self.positions), 8)
        )
        if not np.all(np.isfinite(result)):
            raise RuntimeError("ddX external-MEP model-field VJP is non-finite.")
        return result


def _cavity_parent_indices(
    cavity_points_bohr: np.ndarray,
    positions_angstrom: np.ndarray,
    radii_angstrom: np.ndarray,
) -> np.ndarray:
    """Recover the exact ddX sphere owner of each exposed cavity point."""

    positions_bohr = positions_angstrom / Bohr
    radii_bohr = radii_angstrom / Bohr
    distances = np.linalg.norm(
        cavity_points_bohr[:, None, :] - positions_bohr[None, :, :], axis=2
    )
    residual = np.abs(distances - radii_bohr[None, :])
    owners = np.argmin(residual, axis=1).astype(np.int64)
    maximum_residual = float(np.max(residual[np.arange(len(owners)), owners]))
    if not math.isfinite(maximum_residual) or maximum_residual > 2.0e-11:
        raise RuntimeError("ddX cavity points could not be bound to their spheres.")
    return owners


@dataclass(frozen=True, slots=True)
class RadialGTODDXState:
    """Immutable evidence snapshot for one fixed-geometry ddX solve."""

    provider_id: str
    continuum_profile_id: str
    cavity_profile_id: str
    configuration_contract_id: str
    configuration_sha256: str
    provenance_sha256: str
    geometry_sha256: str
    cavity_topology_sha256: str
    state_hash: str
    pyddx_version: str
    continuum_model: str
    atom_count: int
    cavity_point_count: int
    source_values: tuple[float, ...]
    field_values: tuple[float, ...]
    polarization_energy_ev: float
    scalar_id: str = DIAGNOSTIC_DDX_DDPCM_RADIAL_GTO_ELECTROSTATIC_V1
    coupling_id: str = MACE_POLAR_RADIAL_GTO_COUPLING_ID
    capabilities: CapabilityStatus = CapabilityStatus()

    def __post_init__(self) -> None:
        if self.provider_id != DDX_RADIAL_PROVIDER_ID:
            raise ValueError("ddX radial state provider identity is invalid.")
        expected_continuum_profile_id = (
            DDX_PCM_PROFILE_ID
            if self.continuum_model == "pcm"
            else DDX_COSMO_PROFILE_ID
        )
        if self.continuum_profile_id != expected_continuum_profile_id:
            raise ValueError("ddX radial continuum profile identity is invalid.")
        if self.cavity_profile_id != DDX_CAVITY_PROFILE_ID:
            raise ValueError("ddX radial cavity identity is invalid.")
        expected_scalar_id = (
            DIAGNOSTIC_DDX_DDPCM_RADIAL_GTO_ELECTROSTATIC_V1
            if self.continuum_model == "pcm"
            else DIAGNOSTIC_DDX_DDCOSMO_RADIAL_GTO_ELECTROSTATIC_V1
        )
        if self.scalar_id != expected_scalar_id:
            raise ValueError("ddX radial scalar identity is invalid.")
        if self.coupling_id != MACE_POLAR_RADIAL_GTO_COUPLING_ID:
            raise ValueError("ddX radial coupling identity is invalid.")
        if self.pyddx_version != TESTED_PYDDX_VERSION:
            raise ValueError("ddX radial state runtime version is invalid.")
        if (
            not isinstance(self.configuration_contract_id, str)
            or not self.configuration_contract_id.strip()
        ):
            raise ValueError(
                "ddX radial configuration contract identity must be non-empty."
            )
        if self.continuum_model not in ("pcm", "cosmo"):
            raise ValueError("ddX radial state equation is invalid.")
        if isinstance(self.atom_count, bool) or self.atom_count < 1:
            raise ValueError("ddX radial atom_count must be positive.")
        if isinstance(self.cavity_point_count, bool) or self.cavity_point_count < 1:
            raise ValueError("ddX radial cavity_point_count must be positive.")
        for name in (
            "configuration_sha256",
            "provenance_sha256",
            "geometry_sha256",
            "cavity_topology_sha256",
            "state_hash",
        ):
            _digest(getattr(self, name), name)
        if self.capabilities.enabled_tiers:
            raise ValueError("ddX radial E/F/H/V/M capabilities remain closed.")
        energy = float(self.polarization_energy_ev)
        if not math.isfinite(energy):
            raise ValueError("ddX radial energy must be finite.")
        paired = 0.5 * MACE_POLAR_RADIAL_GTO_PAIRING.pair(
            self.source, self.reaction_field
        )
        tolerance = max(2.0e-10, 2.0e-10 * abs(energy))
        if abs(energy - paired) > tolerance:
            raise ValueError("ddX radial state violates source/field half coupling.")
        expected = _sha(
            {
                "contract": _STATE_CONTRACT,
                "provider_id": self.provider_id,
                "continuum_profile_id": self.continuum_profile_id,
                "cavity_profile_id": self.cavity_profile_id,
                "scalar_id": self.scalar_id,
                "configuration_contract_id": self.configuration_contract_id,
                "configuration_sha256": self.configuration_sha256,
                "provenance_sha256": self.provenance_sha256,
                "geometry_sha256": self.geometry_sha256,
                "cavity_topology_sha256": self.cavity_topology_sha256,
                "pyddx_version": self.pyddx_version,
                "continuum_model": self.continuum_model,
                "atom_count": self.atom_count,
                "cavity_point_count": self.cavity_point_count,
                "source": self.source.tolist(),
                "field": self.reaction_field.tolist(),
                "polarization_energy_ev": energy,
            }
        )
        if self.state_hash != expected:
            raise ValueError("ddX radial state hash does not match its contents.")

    @property
    def source(self) -> np.ndarray:
        return _readonly(self.source_values, (self.atom_count, 8))

    @property
    def reaction_field(self) -> np.ndarray:
        return _readonly(self.field_values, (self.atom_count, 8))


class RadialGTODDXBackend:
    """Disabled ddPCM/ddCOSMO backend in the full eight-channel radial space."""

    __slots__ = (
        "_configuration",
        "_configuration_sha256",
        "_continuum_model",
        "_coupling",
        "_dielectric",
        "_dielectric_scaling",
        "_eta",
        "_lmax",
        "_n_lebedev",
        "_n_proc",
        "_provenance_sha256",
        "_pyddx",
        "_radii",
        "_sealed",
        "_solver_tolerance",
        "_symbols",
        "configuration_contract_id",
        "continuum_profile_id",
    )
    provider_id = DDX_RADIAL_PROVIDER_ID
    cavity_profile_id = DDX_CAVITY_PROFILE_ID
    coupling_id = MACE_POLAR_RADIAL_GTO_COUPLING_ID
    source_space = MACE_POLAR_RADIAL_GTO_SOURCE_SPACE
    field_space = MACE_POLAR_RADIAL_GTO_FIELD_DUAL_SPACE
    pairing = MACE_POLAR_RADIAL_GTO_PAIRING
    capabilities = CapabilityStatus()
    fixed_topology = False
    linear_response = True
    reciprocal = True
    source_dependent_geometry = False
    structurally_rotation_equivariant = False
    achieved_algebraic_residual_available = False

    def __init__(
        self,
        symbols: Sequence[str],
        cavity_radii_angstrom: object,
        *,
        continuum_model: Literal["pcm", "cosmo"],
        dielectric: float,
        lmax: int,
        n_lebedev: int,
        solver_tolerance: float = 1.0e-12,
        eta: float = 0.1,
        n_proc: int = 1,
        configuration_contract_id: str = UNBOUND_CONTINUUM_CONFIGURATION_CONTRACT_ID,
    ) -> None:
        normalized_symbols = tuple(str(symbol) for symbol in symbols)
        if not normalized_symbols or any(
            symbol not in atomic_numbers for symbol in normalized_symbols
        ):
            raise ValueError("ddX radial symbols must be known element symbols.")
        radii = np.asarray(cavity_radii_angstrom, dtype=float)
        if (
            radii.shape != (len(normalized_symbols),)
            or not np.all(np.isfinite(radii))
            or np.any(radii <= 0.0)
        ):
            raise ValueError("ddX radial radii must be finite and positive per atom.")
        method = str(continuum_model).strip().lower()
        if method not in ("pcm", "cosmo"):
            raise ValueError("continuum_model must be 'pcm' or 'cosmo'.")
        dielectric_value = float(dielectric)
        tolerance = float(solver_tolerance)
        eta_value = float(eta)
        if not math.isfinite(dielectric_value) or dielectric_value <= 1.0:
            raise ValueError("ddX radial dielectric must be greater than one.")
        if not math.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("ddX radial solver tolerance must be positive.")
        if not math.isfinite(eta_value) or not 0.0 <= eta_value <= 1.0:
            raise ValueError("ddX radial eta must lie in [0, 1].")
        for value, name in (
            (lmax, "lmax"),
            (n_lebedev, "n_lebedev"),
            (n_proc, "n_proc"),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, np.integer))
                or int(value) < 1
            ):
                raise ValueError(f"ddX radial {name} must be a positive integer.")
        if (
            not isinstance(configuration_contract_id, str)
            or not configuration_contract_id.strip()
        ):
            raise ValueError("configuration_contract_id must be non-empty.")
        if configuration_contract_id == DDX_WATER_194_CONFIGURATION_CONTRACT_ID:
            if (
                method != "pcm"
                or dielectric_value != DDX_WATER_DIELECTRIC
                or int(lmax) != DDX_WATER_LMAX
                or int(n_lebedev) != DDX_WATER_LEBEDEV_POINTS
                or tolerance != 1.0e-12
                or eta_value != 0.1
                or int(n_proc) != 1
                or not np.array_equal(
                    radii, smd_water_coulomb_radii(normalized_symbols)
                )
            ):
                raise ValueError(
                    "The water ddX-194 contract requires its exact frozen settings."
                )

        pyddx = _runtime()
        coupling = MACEPolarRadialGTOCoupling()
        scaling = (
            1.0 if method == "pcm" else (dielectric_value - 1.0) / dielectric_value
        )
        configuration = (
            normalized_symbols,
            tuple(float(value) for value in radii),
            method,
            dielectric_value,
            int(lmax),
            int(n_lebedev),
            tolerance,
            eta_value,
            int(n_proc),
            configuration_contract_id,
            TESTED_PYDDX_VERSION,
            tuple(MACE_POLAR_RADIAL_SIGMAS_ANGSTROM),
            coupling.configuration_sha256(),
            coupling.provenance_sha256,
            self.source_space.metadata_hash(),
            self.field_space.metadata_hash(),
            self.pairing.metadata_hash(),
            HARTREE_TO_EV,
            _implementation_sha256(),
        )
        configuration_sha256 = _sha(configuration)
        provenance_sha256 = _sha(
            {
                "provider_id": self.provider_id,
                "configuration_sha256": configuration_sha256,
                "pyddx_version": TESTED_PYDDX_VERSION,
                "ddx_source_commit": DDX_SOURCE_COMMIT,
                "source_map": "joint-radial-gto-psi-phi-D-and-D-star-v1",
                "scientific_status": "disabled-finite-grid-not-rotation-admitted",
            }
        )
        object.__setattr__(self, "_symbols", normalized_symbols)
        object.__setattr__(self, "_radii", tuple(float(value) for value in radii))
        object.__setattr__(self, "_continuum_model", method)
        object.__setattr__(self, "_dielectric", dielectric_value)
        object.__setattr__(self, "_dielectric_scaling", scaling)
        object.__setattr__(self, "_lmax", int(lmax))
        object.__setattr__(self, "_n_lebedev", int(n_lebedev))
        object.__setattr__(self, "_solver_tolerance", tolerance)
        object.__setattr__(self, "_eta", eta_value)
        object.__setattr__(self, "_n_proc", int(n_proc))
        object.__setattr__(self, "configuration_contract_id", configuration_contract_id)
        object.__setattr__(
            self,
            "continuum_profile_id",
            DDX_PCM_PROFILE_ID if method == "pcm" else DDX_COSMO_PROFILE_ID,
        )
        object.__setattr__(self, "_pyddx", pyddx)
        object.__setattr__(self, "_coupling", coupling)
        object.__setattr__(self, "_configuration", configuration)
        object.__setattr__(self, "_configuration_sha256", configuration_sha256)
        object.__setattr__(self, "_provenance_sha256", provenance_sha256)
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("RadialGTODDXBackend is immutable.")
        object.__setattr__(self, name, value)

    @property
    def symbols(self) -> tuple[str, ...]:
        return self._symbols

    @property
    def scalar_id(self) -> str:
        if self._continuum_model == "pcm":
            return DIAGNOSTIC_DDX_DDPCM_RADIAL_GTO_ELECTROSTATIC_V1
        return DIAGNOSTIC_DDX_DDCOSMO_RADIAL_GTO_ELECTROSTATIC_V1

    @property
    def cavity_radii_angstrom(self) -> np.ndarray:
        result = np.asarray(self._radii, dtype=float)
        result.setflags(write=False)
        return result

    @property
    def provenance_sha256(self) -> str:
        return self._provenance_sha256

    def configuration_sha256(self) -> str:
        current = (
            self.symbols,
            tuple(float(value) for value in self.cavity_radii_angstrom),
            self._continuum_model,
            self._dielectric,
            self._lmax,
            self._n_lebedev,
            self._solver_tolerance,
            self._eta,
            self._n_proc,
            self.configuration_contract_id,
            TESTED_PYDDX_VERSION,
            tuple(MACE_POLAR_RADIAL_SIGMAS_ANGSTROM),
            self._coupling.configuration_sha256(),
            self._coupling.provenance_sha256,
            self.source_space.metadata_hash(),
            self.field_space.metadata_hash(),
            self.pairing.metadata_hash(),
            HARTREE_TO_EV,
            _implementation_sha256(),
        )
        if (
            current != self._configuration
            or _sha(current) != self._configuration_sha256
        ):
            raise RuntimeError("ddX radial configuration fingerprint changed.")
        return self._configuration_sha256

    def prepare_problem(self, geometry: Any) -> RadialDDXProblem:
        """Prepare one reusable geometry-local ddX general-source problem."""

        self.configuration_sha256()
        positions = _positions(geometry, self.symbols)
        model = self._pyddx.Model(
            self._continuum_model,
            (positions / Bohr).T,
            self.cavity_radii_angstrom / Bohr,
            self._dielectric,
            eta=self._eta,
            shift=0.0,
            lmax=self._lmax,
            n_lebedev=self._n_lebedev,
            enable_fmm=False,
            n_proc=self._n_proc,
            enable_force=True,
        )
        if int(model.n_spheres) != len(self.symbols) or not bool(
            model.has_force_enabled
        ):
            raise RuntimeError("pyddx did not retain the radial model contract.")
        cavity = np.asarray(model.cavity, dtype=float).T
        owners = _cavity_parent_indices(cavity, positions, self.cavity_radii_angstrom)
        topology_sha256 = _cavity_topology_sha256(
            cavity,
            positions,
            self.cavity_radii_angstrom,
            owners,
        )
        owned = OwnedFixedSurfaceGeometry(
            positions,
            cavity,
            owners,
        )
        phi_matrix = self._coupling.surface_operator(owned) / HARTREE_TO_EV
        source_dimension = len(self.symbols) * 8
        psi_matrix = np.empty(
            (int(model.n_basis) * len(self.symbols), source_dimension)
        )
        for column in range(source_dimension):
            direction = np.zeros((len(self.symbols), 8), dtype=float)
            direction.reshape(-1)[column] = 1.0
            multipoles = self._ddx_total_multipoles(direction)
            psi_matrix[:, column] = np.asarray(
                model.multipole_psi(multipoles), dtype=float
            ).reshape(-1)
        return RadialDDXProblem(
            model,
            positions,
            phi_matrix,
            psi_matrix,
            runtime=self._pyddx,
            solver_tolerance=self._solver_tolerance,
            dielectric_scaling=self._dielectric_scaling,
            cavity_topology_sha256=topology_sha256,
        )

    def _problem(self, geometry: Any) -> RadialDDXProblem:
        """Compatibility alias for the pre-vNext internal spelling."""

        return self.prepare_problem(geometry)

    @staticmethod
    def _ddx_total_multipoles(source: np.ndarray) -> np.ndarray:
        """Convert radial raw-l1 blocks to ddX real-spherical multipoles.

        Both APIs use the checkpoint/raw order ``[q,y,z,x]``.  Cartesian
        reordering is needed only for pointwise gradients, not for ddX's local
        spherical coefficients; this matches the independently exercised
        legacy ``mace_polar_density_to_pyddx_multipoles`` convention.
        """

        coefficients = _radial_coefficients(source).sum(axis=1)
        multipoles = np.empty((4, len(source)), dtype=float)
        multipoles[0] = coefficients[:, 0] / math.sqrt(4.0 * math.pi)
        multipoles[1:] = coefficients[:, 1:].T / (Bohr * math.sqrt(4.0 * math.pi / 3.0))
        return multipoles

    def _solve(self, geometry: Any, source: object, *, coordinate: bool = False):
        problem = self._problem(geometry)
        values = self.source_space.validate(source, atom_count=len(self.symbols))
        vector = values.reshape(-1)
        phi = problem.phi_matrix @ vector
        psi = (problem.psi_matrix @ vector).reshape(
            int(problem.model.n_basis), len(self.symbols)
        )
        # pyddx 0.8.0's tested binding is State(model, psi, phi).  An older
        # upstream prose example accidentally reversed the two arrays; their
        # unequal public shapes make that typo fail closed here.
        state = self._pyddx.State(problem.model, psi, phi)
        state.solve(self._solver_tolerance)
        state.solve_adjoint(self._solver_tolerance)
        if not np.all(np.isfinite(np.asarray(state.x))) or not np.all(
            np.isfinite(np.asarray(state.xi))
        ):
            raise RuntimeError("ddX radial solve returned non-finite solutions.")
        raw_field = 0.5 * (
            problem.psi_matrix.T @ np.asarray(state.x, dtype=float).reshape(-1)
            - problem.phi_matrix.T @ np.asarray(state.xi, dtype=float)
        )
        field = (
            self._dielectric_scaling * HARTREE_TO_EV * raw_field.reshape(values.shape)
        )
        field = self.field_space.validate(
            field, atom_count=len(self.symbols), name="ddX radial reaction field"
        )
        energy = self._dielectric_scaling * HARTREE_TO_EV * float(state.energy())
        paired = 0.5 * self.pairing.pair(values, field)
        tolerance = max(2.0e-10, 100.0 * self._solver_tolerance) * max(
            1.0, abs(energy), abs(paired)
        )
        if not math.isfinite(energy) or abs(energy - paired) > tolerance:
            raise RuntimeError("ddX radial solve violates the half-coupling identity.")
        if coordinate:
            return problem, values, state, energy, field
        return problem, values, state, energy, field

    def _state_from_solution(
        self,
        problem: RadialDDXProblem,
        values: np.ndarray,
        energy: float,
        field: np.ndarray,
    ) -> RadialGTODDXState:
        geometry_digest = _geometry_sha256(problem.positions, self.symbols)
        payload = {
            "contract": _STATE_CONTRACT,
            "provider_id": self.provider_id,
            "continuum_profile_id": self.continuum_profile_id,
            "cavity_profile_id": self.cavity_profile_id,
            "scalar_id": self.scalar_id,
            "configuration_contract_id": self.configuration_contract_id,
            "configuration_sha256": self.configuration_sha256(),
            "provenance_sha256": self.provenance_sha256,
            "geometry_sha256": geometry_digest,
            "cavity_topology_sha256": problem.cavity_topology_sha256,
            "pyddx_version": TESTED_PYDDX_VERSION,
            "continuum_model": self._continuum_model,
            "atom_count": len(self.symbols),
            "cavity_point_count": int(problem.model.n_cav),
            "source": values.tolist(),
            "field": field.tolist(),
            "polarization_energy_ev": energy,
        }
        return RadialGTODDXState(
            provider_id=self.provider_id,
            continuum_profile_id=self.continuum_profile_id,
            cavity_profile_id=self.cavity_profile_id,
            configuration_contract_id=self.configuration_contract_id,
            configuration_sha256=self.configuration_sha256(),
            provenance_sha256=self.provenance_sha256,
            geometry_sha256=geometry_digest,
            cavity_topology_sha256=problem.cavity_topology_sha256,
            state_hash=_sha(payload),
            pyddx_version=TESTED_PYDDX_VERSION,
            continuum_model=self._continuum_model,
            atom_count=len(self.symbols),
            cavity_point_count=int(problem.model.n_cav),
            source_values=_tuple(values, (len(self.symbols), 8), "source"),
            field_values=_tuple(field, (len(self.symbols), 8), "field"),
            polarization_energy_ev=energy,
            scalar_id=self.scalar_id,
        )

    def build_state(self, geometry: Any, source: object) -> RadialGTODDXState:
        problem, values, _state, energy, field = self._solve(geometry, source)
        return self._state_from_solution(problem, values, energy, field)

    def energy(self, geometry: Any, source: object) -> float:
        return float(self._solve(geometry, source)[3])

    def evaluate_field(self, geometry: Any, source: object) -> np.ndarray:
        return np.array(self._solve(geometry, source)[4], copy=True)

    field = evaluate_field

    def source_jvp(
        self, geometry: Any, source: object, source_direction: object
    ) -> np.ndarray:
        self.source_space.validate(source, atom_count=len(self.symbols))
        return self.evaluate_field(geometry, source_direction)

    def source_vjp(
        self, geometry: Any, source: object, field_cotangent: object
    ) -> np.ndarray:
        self.source_space.validate(source, atom_count=len(self.symbols))
        cotangent = self.field_space.validate(
            field_cotangent, atom_count=len(self.symbols), name="field cotangent"
        )
        # The authoritative radial pairing is Q=I, so the field cotangent has
        # the same numeric coordinates as the reciprocal map's source input.
        # Exact metadata validation above prevents cross-space reinterpretation.
        return self.source_space.validate(
            self.evaluate_field(geometry, cotangent),
            atom_count=len(self.symbols),
            name="source VJP",
        )

    def _energy_coordinate_gradient_from_solution(
        self,
        problem: RadialDDXProblem,
        values: np.ndarray,
        state: Any,
    ) -> np.ndarray:
        coefficients = _radial_coefficients(values)
        cavity = np.asarray(problem.model.cavity, dtype=float).T
        electric_field = np.zeros_like(cavity)
        for radial_index, sigma in enumerate(MACE_POLAR_RADIAL_SIGMAS_ANGSTROM):
            for center, raw in zip(
                problem.positions / Bohr, coefficients[:, radial_index], strict=True
            ):
                charge, dipole_angstrom = cartesian_multipoles(raw[None, :])
                electric_field -= gaussian_multipole_potential_displacement_gradient(
                    cavity - center,
                    float(charge[0]),
                    dipole_angstrom[0] / Bohr,
                    sigma_bohr=sigma / Bohr,
                )
        cavity_term = np.asarray(
            state.solvation_force_terms({"e": electric_field.T}), dtype=float
        ).T
        xi = np.asarray(state.xi, dtype=float)
        source_term = np.zeros_like(problem.positions)
        for radial_index, sigma in enumerate(MACE_POLAR_RADIAL_SIGMAS_ANGSTROM):
            source_term += gaussian_multipole_potential_position_vjp(
                cavity,
                problem.positions,
                coefficients[:, radial_index],
                -0.5 * xi,
                sigma_angstrom=sigma,
            )
        result = (
            self._dielectric_scaling
            * HARTREE_TO_EV
            * (cavity_term / Bohr + source_term)
        )
        if result.shape != problem.positions.shape or not np.all(np.isfinite(result)):
            raise RuntimeError("ddX radial coordinate gradient is invalid.")
        return result

    def _energy_coordinate_gradient(self, geometry: Any, source: object) -> np.ndarray:
        problem, values, state, _energy, _field = self._solve(
            geometry, source, coordinate=True
        )
        return self._energy_coordinate_gradient_from_solution(problem, values, state)

    def fixed_source_coordinate_gradient(
        self, geometry: Any, source: object
    ) -> np.ndarray:
        """Return ``partial G(R,c)/partial R`` at fixed radial source ``c``.

        This is the nuclear derivative of the same ddX polarization scalar
        returned by :meth:`energy`.  It deliberately excludes the derivative
        of any geometry-dependent model source; a higher-level PES must add
        ``(dc/dR)^T dG/dc`` exactly once.
        """

        return self._energy_coordinate_gradient(geometry, source).copy()

    def build_state_with_fixed_source_coordinate_gradient(
        self, geometry: Any, source: object
    ) -> tuple[RadialGTODDXState, np.ndarray]:
        """Solve once and return the immutable state plus its fixed-source gradient."""

        problem, values, raw_state, energy, field = self._solve(
            geometry, source, coordinate=True
        )
        state = self._state_from_solution(problem, values, energy, field)
        gradient = self._energy_coordinate_gradient_from_solution(
            problem, values, raw_state
        )
        return state, gradient.copy()

    def coordinate_vjp(
        self, geometry: Any, source: object, field_cotangent: object
    ) -> np.ndarray:
        right = self.source_space.validate(source, atom_count=len(self.symbols))
        left = self.field_space.validate(
            field_cotangent, atom_count=len(self.symbols), name="field cotangent"
        )
        # Under the same Q=I pairing, polarization of the scalar coordinate
        # gradient yields left^T (dP/dR) right without a second receiver model.
        result = 0.5 * (
            self._energy_coordinate_gradient(geometry, left + right)
            - self._energy_coordinate_gradient(geometry, left - right)
        )
        return result.reshape(-1).copy()

    @property
    def runtime_provenance(self) -> tuple[tuple[str, str], ...]:
        return tuple(
            sorted(
                {
                    "provider_id": self.provider_id,
                    "continuum_profile_id": self.continuum_profile_id,
                    "cavity_profile_id": self.cavity_profile_id,
                    "scalar_id": self.scalar_id,
                    "configuration_contract_id": self.configuration_contract_id,
                    "configuration_sha256": self.configuration_sha256(),
                    "provenance_sha256": self.provenance_sha256,
                    "pyddx_version": TESTED_PYDDX_VERSION,
                    "ddx_source_commit": DDX_SOURCE_COMMIT,
                    "source_map": "joint-radial-gto-psi-phi-D-and-D-star-v1",
                    "coordinate_derivative": "same-energy-polarization-identity-v1",
                    "fixed_source_coordinate_gradient": "public-same-scalar-v1",
                    "rotation_status": "finite-grid-not-structurally-equivariant",
                    "inner_solve_evidence": "requested-tolerance-only",
                    "capabilities": "none",
                }.items()
            )
        )


def build_water_radial_gto_ddpcm_194_candidate(
    symbols: Sequence[str],
) -> RadialGTODDXBackend:
    """Build the exact disabled water/ddPCM 194-node candidate."""

    normalized = tuple(symbols)
    return RadialGTODDXBackend(
        normalized,
        smd_water_coulomb_radii(normalized),
        continuum_model="pcm",
        dielectric=DDX_WATER_DIELECTRIC,
        lmax=DDX_WATER_LMAX,
        n_lebedev=DDX_WATER_LEBEDEV_POINTS,
        solver_tolerance=1.0e-12,
        eta=0.1,
        n_proc=1,
        configuration_contract_id=DDX_WATER_194_CONFIGURATION_CONTRACT_ID,
    )


__all__ = [
    "DDX_CAVITY_PROFILE_ID",
    "DDX_COSMO_PROFILE_ID",
    "DDX_SOURCE_COMMIT",
    "DDX_PCM_PROFILE_ID",
    "DDX_RADIAL_PROVIDER_ID",
    "DDX_WATER_194_CONFIGURATION_CONTRACT_ID",
    "RadialDDXProblem",
    "RadialGTODDXBackend",
    "RadialGTODDXState",
    "build_water_radial_gto_ddpcm_194_candidate",
]
