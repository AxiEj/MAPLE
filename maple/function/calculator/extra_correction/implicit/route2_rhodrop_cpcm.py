"""Source-dependent rho-DROP C-PCM forward map for Route 2.

The reconstructed density and MOIST DROP adapter live behind separate module
boundaries.  This module owns only the conventional electrostatic C-PCM
forward scalar for one source-bound surface:

``v = B_Gamma c``
``A_Gamma q = -((epsilon - 1) / epsilon) v``
``f = B_Gamma* q``
``E_pol = 0.5 q.T v = 0.5 <c, f>``

Because ``Gamma`` depends on the source, this is not a linear response map.
The public ``apply``/``adjoint`` methods are therefore the local reaction-map
JVP/VJP at the latest exact forward state.  The pinned MOIST API exposes the
efficient reverse contraction but not a forward-mode surface JVP, so
``apply`` is an exact reference implementation assembled from transpose
probes and is not admitted for production Krylov solves.  This module does
not advertise a coordinate VJP or analytic-force capability.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from types import ModuleType
from typing import Any

import numpy as np
from ase.units import Bohr, Hartree
from scipy.linalg import LinAlgError, cho_factor, cho_solve

from .gto_density import (
    density_reaction_coupling,
    external_field_to_density_order,
    point_asc_reaction_potential_gradient,
    point_multipole_potential,
    point_multipole_potential_surface_position_vjp,
)
from .route2_atomic_reference_density import AtomicReferenceDensityAsset
from .route2_density_levelset import ReconstructedMacePolarDensityLevelSet
from .route2_field_state import ReactionFieldDrive
from .route2_moist_drop import (
    MoistDropAdapter,
    MoistDropSettings,
    MoistDropSurfaceSnapshot,
    MoistDropSurfaceState,
    MoistRuntimeProvenance,
    MoistSurfaceWeights,
    reconstructed_level_set_state_sha256,
    route2_source_state_sha256,
)

RHODROP_CPCM_FORWARD_CONTRACT_VERSION = 2
RHODROP_CPCM_MODEL_DRIVE = "conventional-electrostatic-reaction-field"
RHODROP_CPCM_FORCE_STATUS = "blocked-pending-complete-drop-coordinate-vjp"
RHODROP_CPCM_JVP_IMPLEMENTATION = "analytic-vjp-transpose-probe-reference-v1"
RHODROP_CPCM_OPERATIONAL_PROFILE_KIND = "route2-rhodrop-cpcm-operational-v1"


def _sha256_json(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _hash_arrays(payload: object, arrays: tuple[np.ndarray, ...]) -> str:
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
    )
    for array in arrays:
        canonical = np.ascontiguousarray(array)
        digest.update(canonical.dtype.str.encode("ascii"))
        digest.update(json.dumps(canonical.shape).encode("ascii"))
        digest.update(canonical.tobytes())
    return digest.hexdigest()


def _readonly_finite_array(
    values: object,
    *,
    name: str,
    shape: tuple[int, ...] | None = None,
    dtype: Any = float,
) -> np.ndarray:
    array = np.asarray(values, dtype=dtype)
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}; received {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    result = np.array(array, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _source_block(
    values: object,
    *,
    atom_count: int,
    name: str,
) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    expected = (atom_count, 4)
    if array.shape != expected or not np.all(np.isfinite(array)):
        raise ValueError(
            f"{name} must be finite with shape {expected}; received {array.shape}."
        )
    return array


def _identity_tolerance(
    reference: float,
    *,
    absolute: float,
    relative: float,
) -> float:
    return max(absolute, relative * abs(reference))


@dataclass(frozen=True)
class RhoDropCPCMSettings:
    """Numerical settings for the stationary frozen-source C-PCM solve."""

    dielectric: float = 80.0
    linear_solve_relative_tolerance: float = 5.0e-12
    linear_solve_absolute_tolerance: float = 1.0e-13
    energy_identity_relative_tolerance: float = 1.0e-10
    energy_identity_absolute_tolerance_hartree: float = 1.0e-12
    maximum_condition_number_2: float = 1.0e12
    require_exact_cold_replay: bool = True

    def __post_init__(self) -> None:
        dielectric = float(self.dielectric)
        if not math.isfinite(dielectric) or dielectric <= 1.0:
            raise ValueError(
                "rho-DROP C-PCM dielectric must be finite and greater than 1."
            )
        object.__setattr__(self, "dielectric", dielectric)
        for name in (
            "linear_solve_relative_tolerance",
            "linear_solve_absolute_tolerance",
            "energy_identity_relative_tolerance",
            "energy_identity_absolute_tolerance_hartree",
            "maximum_condition_number_2",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite.")
            object.__setattr__(self, name, value)
        if not isinstance(self.require_exact_cold_replay, bool):
            raise TypeError("require_exact_cold_replay must be bool.")

    @property
    def dielectric_factor(self) -> float:
        return (self.dielectric - 1.0) / self.dielectric

    @property
    def identity_sha256(self) -> str:
        return _sha256_json(asdict(self))


@dataclass(frozen=True)
class RhoDropCPCMForwardState:
    """Immutable stationary C-PCM state for one exact source and DROP surface."""

    density_coefficients: np.ndarray
    surface_potential_hartree_per_e: np.ndarray
    surface_charge_e: np.ndarray
    reaction_potential_hartree_per_e: np.ndarray
    reaction_gradient_hartree_per_e_bohr: np.ndarray
    reaction_field_ev: np.ndarray
    polarization_energy_hartree: float
    surface_coupling_hartree: float
    density_reaction_coupling_hartree: float
    dielectric: float
    dielectric_factor: float
    linear_residual_absolute: float
    linear_residual_relative: float
    condition_number_2: float
    solver_settings_sha256: str
    surface_state: MoistDropSurfaceState = field(repr=False, compare=False)
    source_sha256: str = field(init=False)
    operator_sha256: str = field(init=False)
    state_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.surface_state, MoistDropSurfaceState):
            raise TypeError("rho-DROP C-PCM state requires a MOIST DROP surface state.")
        surface = self.surface_state.snapshot
        atom_count = surface.atom_count
        surface_size = surface.surface_size
        source = _readonly_finite_array(
            self.density_coefficients,
            name="rho-DROP source",
            shape=(atom_count, 4),
        )
        surface_potential = _readonly_finite_array(
            self.surface_potential_hartree_per_e,
            name="rho-DROP surface potential",
            shape=(surface_size,),
        )
        surface_charge = _readonly_finite_array(
            self.surface_charge_e,
            name="rho-DROP surface charge",
            shape=(surface_size,),
        )
        reaction_potential = _readonly_finite_array(
            self.reaction_potential_hartree_per_e,
            name="rho-DROP reaction potential",
            shape=(atom_count,),
        )
        reaction_gradient = _readonly_finite_array(
            self.reaction_gradient_hartree_per_e_bohr,
            name="rho-DROP reaction-potential gradient",
            shape=(atom_count, 3),
        )
        reaction_field = _readonly_finite_array(
            self.reaction_field_ev,
            name="rho-DROP Route-2 reaction field",
            shape=(atom_count, 4),
        )

        scalar_names = (
            "polarization_energy_hartree",
            "surface_coupling_hartree",
            "density_reaction_coupling_hartree",
            "dielectric",
            "dielectric_factor",
            "linear_residual_absolute",
            "linear_residual_relative",
            "condition_number_2",
        )
        scalars: dict[str, float] = {}
        for name in scalar_names:
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"rho-DROP C-PCM {name} must be finite.")
            scalars[name] = value
            object.__setattr__(self, name, value)
        if scalars["linear_residual_absolute"] < 0.0:
            raise ValueError("rho-DROP C-PCM absolute residual cannot be negative.")
        if scalars["linear_residual_relative"] < 0.0:
            raise ValueError("rho-DROP C-PCM relative residual cannot be negative.")
        if scalars["condition_number_2"] < 1.0:
            raise ValueError("rho-DROP C-PCM condition number cannot be below one.")
        expected_factor = (scalars["dielectric"] - 1.0) / scalars["dielectric"]
        if (
            scalars["dielectric"] <= 1.0
            or scalars["dielectric_factor"] != expected_factor
        ):
            raise ValueError("rho-DROP C-PCM dielectric factor is inconsistent.")

        settings_hash = str(self.solver_settings_sha256)
        if len(settings_hash) != 64 or any(
            char not in "0123456789abcdef" for char in settings_hash
        ):
            raise ValueError("rho-DROP C-PCM settings identity must be a SHA256.")

        expected_field = np.column_stack(
            (
                reaction_potential * Hartree,
                reaction_gradient * Hartree / Bohr,
            )
        )
        if not np.array_equal(reaction_field, expected_field):
            raise ValueError(
                "rho-DROP reaction field has inconsistent units or ordering."
            )

        expected_surface_coupling = float(np.dot(surface_potential, surface_charge))
        tolerance = _identity_tolerance(
            expected_surface_coupling,
            absolute=1.0e-12,
            relative=1.0e-10,
        )
        if (
            abs(scalars["surface_coupling_hartree"] - expected_surface_coupling)
            > tolerance
        ):
            raise ValueError("rho-DROP surface coupling is inconsistent with q.T v.")
        if (
            abs(
                2.0 * scalars["polarization_energy_hartree"] - expected_surface_coupling
            )
            > tolerance
        ):
            raise ValueError("rho-DROP energy must equal 0.5*q.T*v.")
        if (
            abs(
                scalars["density_reaction_coupling_hartree"] - expected_surface_coupling
            )
            > tolerance
        ):
            raise ValueError("rho-DROP source/field and surface couplings disagree.")

        object.__setattr__(self, "density_coefficients", source)
        object.__setattr__(self, "surface_potential_hartree_per_e", surface_potential)
        object.__setattr__(self, "surface_charge_e", surface_charge)
        object.__setattr__(self, "reaction_potential_hartree_per_e", reaction_potential)
        object.__setattr__(
            self,
            "reaction_gradient_hartree_per_e_bohr",
            reaction_gradient,
        )
        object.__setattr__(self, "reaction_field_ev", reaction_field)
        object.__setattr__(self, "solver_settings_sha256", settings_hash)

        source_hash = route2_source_state_sha256(source)
        operator_hash = _sha256_json(
            {
                "contract_version": RHODROP_CPCM_FORWARD_CONTRACT_VERSION,
                "surface_snapshot_sha256": surface.snapshot_sha256,
                "solver_settings_sha256": settings_hash,
            }
        )
        state_hash = _hash_arrays(
            {
                "contract_version": RHODROP_CPCM_FORWARD_CONTRACT_VERSION,
                "source_sha256": source_hash,
                "operator_sha256": operator_hash,
                **scalars,
            },
            (
                source,
                surface_potential,
                surface_charge,
                reaction_potential,
                reaction_gradient,
                reaction_field,
            ),
        )
        object.__setattr__(self, "source_sha256", source_hash)
        object.__setattr__(self, "operator_sha256", operator_hash)
        object.__setattr__(self, "state_sha256", state_hash)

    @property
    def surface_snapshot(self) -> MoistDropSurfaceSnapshot:
        return self.surface_state.snapshot


def solve_frozen_source_rhodrop_cpcm(
    surface_state: MoistDropSurfaceState,
    atom_positions_angstrom: object,
    density_coefficients: object,
    *,
    settings: RhoDropCPCMSettings = RhoDropCPCMSettings(),
) -> RhoDropCPCMForwardState:
    """Solve the stationary C-PCM scalar on one already-built DROP surface."""

    if not isinstance(surface_state, MoistDropSurfaceState):
        raise TypeError("Frozen-source C-PCM requires a MOIST DROP surface state.")
    if not isinstance(settings, RhoDropCPCMSettings):
        raise TypeError("Frozen-source C-PCM requires RhoDropCPCMSettings.")
    snapshot = surface_state.snapshot
    positions = np.asarray(atom_positions_angstrom, dtype=float)
    expected_position_shape = (snapshot.atom_count, 3)
    if positions.shape != expected_position_shape or not np.all(np.isfinite(positions)):
        raise ValueError(
            "rho-DROP C-PCM atom positions must be finite with shape "
            f"{expected_position_shape}; received {positions.shape}."
        )
    if not np.array_equal(positions / Bohr, snapshot.atom_positions_bohr):
        raise RuntimeError("rho-DROP surface and C-PCM solute geometries differ.")
    source = _source_block(
        density_coefficients,
        atom_count=snapshot.atom_count,
        name="rho-DROP C-PCM source",
    )
    source_hash = route2_source_state_sha256(source)
    if snapshot.bound_source_sha256 is None:
        raise RuntimeError(
            "rho-DROP C-PCM refuses a surface without an explicit bound source."
        )
    if snapshot.bound_source_sha256 != source_hash:
        raise RuntimeError(
            "rho-DROP C-PCM source does not own the supplied cavity/operator."
        )

    potential = point_multipole_potential(
        snapshot.surface_points_bohr,
        positions,
        source,
    )
    matrix = np.asarray(snapshot.amat, dtype=float)
    try:
        factor = cho_factor(matrix, lower=True, overwrite_a=False, check_finite=False)
    except LinAlgError as exc:
        raise RuntimeError(
            "MOIST DROP C-PCM A-matrix is not positive definite."
        ) from exc

    condition_number = float(np.linalg.cond(matrix, p=2))
    if (
        not math.isfinite(condition_number)
        or condition_number > settings.maximum_condition_number_2
    ):
        raise RuntimeError(
            "MOIST DROP C-PCM A-matrix condition number exceeds the admitted limit."
        )
    right_hand_side = -settings.dielectric_factor * potential
    charge = np.asarray(
        cho_solve(factor, right_hand_side, check_finite=False),
        dtype=float,
    )
    if charge.shape != potential.shape or not np.all(np.isfinite(charge)):
        raise RuntimeError("rho-DROP C-PCM solve returned invalid surface charges.")
    residual = matrix @ charge - right_hand_side
    residual_absolute = float(np.linalg.norm(residual, ord=np.inf))
    right_hand_side_norm = float(np.linalg.norm(right_hand_side, ord=np.inf))
    residual_relative = residual_absolute / max(right_hand_side_norm, 1.0e-300)
    residual_limit = max(
        settings.linear_solve_absolute_tolerance,
        settings.linear_solve_relative_tolerance * right_hand_side_norm,
    )
    if residual_absolute > residual_limit:
        raise RuntimeError(
            "rho-DROP C-PCM linear solve failed its residual tolerance "
            f"({residual_absolute:.3e} > {residual_limit:.3e})."
        )

    surface_coupling = float(np.dot(potential, charge))
    energy = 0.5 * surface_coupling
    energy_tolerance = _identity_tolerance(
        surface_coupling,
        absolute=settings.energy_identity_absolute_tolerance_hartree,
        relative=settings.energy_identity_relative_tolerance,
    )
    if energy > energy_tolerance:
        raise RuntimeError("Stationary C-PCM polarization energy must be non-positive.")

    reaction_potential, reaction_gradient = point_asc_reaction_potential_gradient(
        positions,
        snapshot.surface_points_bohr,
        charge,
    )
    density_coupling = density_reaction_coupling(
        source,
        reaction_potential,
        reaction_gradient,
    )
    if abs(density_coupling - surface_coupling) > energy_tolerance:
        raise RuntimeError(
            "rho-DROP C-PCM failed the source/field reciprocity identity."
        )
    reaction_field = np.column_stack(
        (
            reaction_potential * Hartree,
            reaction_gradient * Hartree / Bohr,
        )
    )
    return RhoDropCPCMForwardState(
        density_coefficients=source,
        surface_potential_hartree_per_e=potential,
        surface_charge_e=charge,
        reaction_potential_hartree_per_e=reaction_potential,
        reaction_gradient_hartree_per_e_bohr=reaction_gradient,
        reaction_field_ev=reaction_field,
        polarization_energy_hartree=energy,
        surface_coupling_hartree=surface_coupling,
        density_reaction_coupling_hartree=density_coupling,
        dielectric=settings.dielectric,
        dielectric_factor=settings.dielectric_factor,
        linear_residual_absolute=residual_absolute,
        linear_residual_relative=residual_relative,
        condition_number_2=condition_number,
        solver_settings_sha256=settings.identity_sha256,
        surface_state=surface_state,
    )


class RhoDropCPCMReactionField:
    """Conventional Route-2 reaction field on a source-dependent DROP cavity."""

    # The frozen-surface electrostatic kernels retain the reciprocal pairing
    # needed by the stationary half-coupling energy.  The full nonlinear
    # Jacobian also contains cavity response and is not assumed self-adjoint.
    reciprocal_energy_pairing = True
    reaction_jacobian_self_adjoint = False
    source_dependent_geometry = True
    model_drive = RHODROP_CPCM_MODEL_DRIVE
    profile_kind = RHODROP_CPCM_OPERATIONAL_PROFILE_KIND
    electrostatics_only = True
    include_cds = False
    reaction_map_derivative_available = True
    operational_jvp_efficiency_admitted = False
    jvp_implementation = RHODROP_CPCM_JVP_IMPLEMENTATION
    complete_position_derivative_available = False
    analytic_force_status = RHODROP_CPCM_FORCE_STATUS

    def __init__(
        self,
        asset: AtomicReferenceDensityAsset,
        atomic_numbers: object,
        atom_positions_angstrom: object,
        *,
        expected_total_charge_e: float,
        runtime: MoistRuntimeProvenance,
        n_iso_e_per_bohr3: float = 1.0e-3,
        cpcm_settings: RhoDropCPCMSettings = RhoDropCPCMSettings(),
        drop_settings: MoistDropSettings = MoistDropSettings(),
        sigma_angstrom: float = 1.5,
        minimum_density_e_per_bohr3: float = 0.0,
        electron_count_tolerance: float = 5.0e-12,
        moist_module: ModuleType | Any | None = None,
        _allow_synthetic_runtime: bool = False,
    ) -> None:
        if not isinstance(asset, AtomicReferenceDensityAsset):
            raise TypeError(
                "rho-DROP C-PCM requires a verified reference-density asset."
            )
        if not isinstance(runtime, MoistRuntimeProvenance):
            raise TypeError("rho-DROP C-PCM requires verified MOIST provenance.")
        if not isinstance(cpcm_settings, RhoDropCPCMSettings):
            raise TypeError("cpcm_settings must be RhoDropCPCMSettings.")
        if not isinstance(drop_settings, MoistDropSettings):
            raise TypeError("drop_settings must be MoistDropSettings.")
        positions = np.asarray(atom_positions_angstrom, dtype=float)
        if (
            positions.ndim != 2
            or positions.shape[0] == 0
            or positions.shape[1] != 3
            or not np.all(np.isfinite(positions))
        ):
            raise ValueError("rho-DROP atom positions must have shape (n_atoms, 3).")
        numbers = np.asarray(atomic_numbers)
        if (
            numbers.shape != (positions.shape[0],)
            or not np.all(np.isfinite(numbers))
            or np.any(numbers != np.rint(numbers))
            or np.any(numbers < 1)
        ):
            raise ValueError(
                "rho-DROP atomic numbers must be positive integers per atom."
            )
        charge = float(expected_total_charge_e)
        if not math.isfinite(charge):
            raise ValueError("rho-DROP expected total charge must be finite.")
        n_iso = float(n_iso_e_per_bohr3)
        if not math.isfinite(n_iso) or n_iso <= 0.0:
            raise ValueError("rho-DROP isodensity value must be positive and finite.")
        sigma = float(sigma_angstrom)
        if not math.isfinite(sigma) or sigma <= 0.0:
            raise ValueError("rho-DROP residual Gaussian width must be positive.")
        minimum_density = float(minimum_density_e_per_bohr3)
        if not math.isfinite(minimum_density) or minimum_density < 0.0:
            raise ValueError(
                "rho-DROP minimum reconstructed density must be nonnegative."
            )
        electron_tolerance = float(electron_count_tolerance)
        if not math.isfinite(electron_tolerance) or electron_tolerance <= 0.0:
            raise ValueError("rho-DROP electron-count tolerance must be positive.")
        if not isinstance(_allow_synthetic_runtime, bool):
            raise TypeError("_allow_synthetic_runtime must be bool.")

        self.asset = asset
        self.atomic_numbers = np.array(numbers, dtype=np.int64, copy=True)
        self.atomic_numbers.setflags(write=False)
        self.atom_positions_angstrom = np.array(positions, dtype=float, copy=True)
        self.atom_positions_angstrom.setflags(write=False)
        self.atom_count = int(positions.shape[0])
        self.expected_total_charge_e = charge
        self.n_iso_e_per_bohr3 = n_iso
        self.cpcm_settings = cpcm_settings
        self.drop_settings = drop_settings
        self.runtime = runtime
        self.sigma_angstrom = sigma
        self.minimum_density_e_per_bohr3 = minimum_density
        self.electron_count_tolerance = electron_tolerance
        self._moist_module = moist_module
        self._allow_synthetic_runtime = bool(_allow_synthetic_runtime)
        self._scf_snapshot: RhoDropCPCMForwardState | None = None
        self._linearization_factor: tuple[str, tuple[np.ndarray, bool]] | None = None
        self._configuration_sha256 = self._current_configuration_sha256()

    def _current_configuration_sha256(self) -> str:
        """Fingerprint every immutable input that defines this provider."""

        try:
            if not isinstance(self.asset, AtomicReferenceDensityAsset):
                raise TypeError("reference-density asset type changed")
            if not isinstance(self.cpcm_settings, RhoDropCPCMSettings):
                raise TypeError("C-PCM settings type changed")
            if not isinstance(self.drop_settings, MoistDropSettings):
                raise TypeError("DROP settings type changed")
            if not isinstance(self.runtime, MoistRuntimeProvenance):
                raise TypeError("MOIST runtime provenance type changed")
            self.asset.require_content_integrity()
            numbers = np.asarray(self.atomic_numbers, dtype=np.int64)
            positions = np.asarray(self.atom_positions_angstrom, dtype=float)
            if numbers.shape != (self.atom_count,) or positions.shape != (
                self.atom_count,
                3,
            ):
                raise ValueError("provider geometry shape changed")
            if not np.all(np.isfinite(positions)):
                raise ValueError("provider geometry became non-finite")
            payload = {
                "contract_version": RHODROP_CPCM_FORWARD_CONTRACT_VERSION,
                "asset_artifact": self.asset.artifact,
                "asset_construction": self.asset.construction,
                "asset_table_sha256": self.asset.table_sha256,
                "asset_manifest_sha256": self.asset.manifest_sha256,
                "asset_content_sha256": self.asset.content_sha256,
                "expected_total_charge_e": float(self.expected_total_charge_e),
                "n_iso_e_per_bohr3": float(self.n_iso_e_per_bohr3),
                "sigma_angstrom": float(self.sigma_angstrom),
                "minimum_density_e_per_bohr3": float(self.minimum_density_e_per_bohr3),
                "electron_count_tolerance": float(self.electron_count_tolerance),
                "cpcm_settings_sha256": self.cpcm_settings.identity_sha256,
                "drop_settings_sha256": self.drop_settings.identity_sha256,
                "runtime_sha256": self.runtime.identity_sha256,
                "allow_synthetic_runtime": self._allow_synthetic_runtime,
            }
            return _hash_arrays(payload, (numbers, positions))
        except (AttributeError, TypeError, ValueError) as exc:
            raise RuntimeError(
                "rho-DROP provider configuration is no longer valid."
            ) from exc

    def _require_configuration_identity(self) -> str:
        observed = self._current_configuration_sha256()
        if observed != self._configuration_sha256:
            raise RuntimeError(
                "rho-DROP provider configuration changed after construction; "
                "the cached cavity/operator state is invalid."
            )
        return observed

    def _build_once(self, source: np.ndarray) -> RhoDropCPCMForwardState:
        level_set = ReconstructedMacePolarDensityLevelSet(
            self.asset,
            self.atomic_numbers,
            self.atom_positions_angstrom,
            source,
            self.n_iso_e_per_bohr3,
            expected_total_charge_e=self.expected_total_charge_e,
            sigma_angstrom=self.sigma_angstrom,
            minimum_density_e_per_bohr3=self.minimum_density_e_per_bohr3,
            minimum_gradient_norm_bohr=self.drop_settings.minimum_gradient_norm_bohr,
            surface_tolerance=self.drop_settings.surface_tolerance,
            electron_count_tolerance=self.electron_count_tolerance,
        )
        level_set_hash = reconstructed_level_set_state_sha256(level_set)
        surface_state = MoistDropAdapter(
            level_set,
            self.atomic_numbers,
            self.atom_positions_angstrom,
            runtime=self.runtime,
            settings=self.drop_settings,
            level_set_state_sha256=level_set_hash,
            moist_module=self._moist_module,
            _allow_synthetic_runtime=self._allow_synthetic_runtime,
        ).build_surface()
        if surface_state.snapshot.level_set_state_sha256 != level_set_hash:
            raise RuntimeError(
                "MOIST DROP surface is stale with respect to its source."
            )
        return solve_frozen_source_rhodrop_cpcm(
            surface_state,
            self.atom_positions_angstrom,
            source,
            settings=self.cpcm_settings,
        )

    def _solve_snapshot(self, source: np.ndarray) -> RhoDropCPCMForwardState:
        first = self._build_once(source)
        if self.cpcm_settings.require_exact_cold_replay:
            replay = self._build_once(source)
            if (
                replay.surface_snapshot.snapshot_sha256
                != first.surface_snapshot.snapshot_sha256
            ):
                raise RuntimeError(
                    "rho-DROP cold replay produced a different cavity/operator."
                )
            if replay.state_sha256 != first.state_sha256:
                raise RuntimeError(
                    "rho-DROP cold replay produced a different C-PCM field."
                )
        return first

    def scf_snapshot(self, density_coefficients: object) -> RhoDropCPCMForwardState:
        self._require_configuration_identity()
        source = _source_block(
            density_coefficients,
            atom_count=self.atom_count,
            name="rho-DROP SCF source",
        )
        cached = self._scf_snapshot
        if cached is not None and np.array_equal(cached.density_coefficients, source):
            return cached
        snapshot = self._solve_snapshot(source)
        self._scf_snapshot = snapshot
        self._linearization_factor = None
        return snapshot

    def apply_scf(self, density_coefficients: object) -> np.ndarray:
        """Return the conventional reaction-potential jet for one SCF source."""

        return np.array(
            self.scf_snapshot(density_coefficients).reaction_field_ev,
            copy=True,
        )

    def apply_scf_drive(self, density_coefficients: object) -> ReactionFieldDrive:
        """Keep the energy-dual and model-driving fields explicitly identical."""

        return ReactionFieldDrive.local_jet(self.apply_scf(density_coefficients))

    def scf_polarization_energy_hartree(
        self,
        density_coefficients: object,
    ) -> float:
        self.assert_forward_state(density_coefficients)
        return self._required_linearization_state().polarization_energy_hartree

    def assert_forward_state(self, density_coefficients: object) -> str:
        """Return the exact bound state hash or reject a stale source."""

        state = self._required_linearization_state()
        source = _source_block(
            density_coefficients,
            atom_count=self.atom_count,
            name="rho-DROP forward-state source",
        )
        if not np.array_equal(source, state.density_coefficients):
            raise RuntimeError(
                "rho-DROP source does not match the cached cavity/operator state."
            )
        return state.state_sha256

    def audit_snapshot(self) -> dict[str, object]:
        """Return JSON-safe provenance and diagnostics for the cached state."""

        state = self._required_linearization_state()
        surface = state.surface_snapshot
        return {
            "contract_version": RHODROP_CPCM_FORWARD_CONTRACT_VERSION,
            "profile_kind": self.profile_kind,
            "model_drive": self.model_drive,
            "source_dependent_geometry": self.source_dependent_geometry,
            "reaction_jacobian_self_adjoint": (self.reaction_jacobian_self_adjoint),
            "electrostatics_only": self.electrostatics_only,
            "include_cds": self.include_cds,
            "reaction_map_derivative_available": (
                self.reaction_map_derivative_available
            ),
            "jvp_implementation": self.jvp_implementation,
            "operational_jvp_efficiency_admitted": (
                self.operational_jvp_efficiency_admitted
            ),
            "complete_position_derivative_available": (
                self.complete_position_derivative_available
            ),
            "analytic_force_status": self.analytic_force_status,
            "runtime_sha256": surface.runtime_sha256,
            "provider_configuration_sha256": self._configuration_sha256,
            "drop_parameter_sha256": surface.parameter_sha256,
            "cpcm_parameter_sha256": state.solver_settings_sha256,
            "reference_density_artifact": self.asset.artifact,
            "reference_density_table_sha256": self.asset.table_sha256,
            "reference_density_manifest_sha256": self.asset.manifest_sha256,
            "reference_density_content_sha256": self.asset.content_sha256,
            "expected_total_charge_e": self.expected_total_charge_e,
            "n_iso_e_per_bohr3": self.n_iso_e_per_bohr3,
            "sigma_angstrom": self.sigma_angstrom,
            "source_sha256": state.source_sha256,
            "geometry_sha256": surface.geometry_sha256,
            "level_set_state_sha256": surface.level_set_state_sha256,
            "surface_snapshot_sha256": surface.snapshot_sha256,
            "operator_sha256": state.operator_sha256,
            "forward_state_sha256": state.state_sha256,
            "surface_size": surface.surface_size,
            "surface_area_bohr2": surface.area_bohr2,
            "surface_volume_bohr3": surface.volume_bohr3,
            "dielectric": state.dielectric,
            "dielectric_factor": state.dielectric_factor,
            "linear_residual_absolute": state.linear_residual_absolute,
            "linear_residual_relative": state.linear_residual_relative,
            "condition_number_2": state.condition_number_2,
            "cold_replay_required": (self.cpcm_settings.require_exact_cold_replay),
        }

    def _required_linearization_state(self) -> RhoDropCPCMForwardState:
        self._require_configuration_identity()
        state = self._scf_snapshot
        if state is None:
            raise RuntimeError(
                "rho-DROP reaction-map derivatives require a preceding "
                "apply_scf(source) state."
            )
        return state

    def validate_linearization_state(
        self,
        density_coefficients: object,
        reaction_field_values: object,
    ) -> None:
        """Require source and field to match the cached nonlinear-map state."""

        state = self._required_linearization_state()
        source = _source_block(
            density_coefficients,
            atom_count=self.atom_count,
            name="rho-DROP linearization source",
        )
        field_values = _source_block(
            reaction_field_values,
            atom_count=self.atom_count,
            name="rho-DROP linearization field",
        )
        if not np.array_equal(source, state.density_coefficients):
            raise RuntimeError(
                "rho-DROP derivative source differs from the cached forward state."
            )
        if not np.array_equal(field_values, state.reaction_field_ev):
            raise RuntimeError(
                "rho-DROP derivative field differs from the cached forward state."
            )

    def _factor_for_state(
        self,
        state: RhoDropCPCMForwardState,
    ) -> tuple[np.ndarray, bool]:
        cached = self._linearization_factor
        if cached is not None and cached[0] == state.state_sha256:
            return cached[1]
        try:
            factor = cho_factor(
                state.surface_snapshot.amat,
                lower=True,
                overwrite_a=False,
                check_finite=False,
            )
        except LinAlgError as exc:
            raise RuntimeError(
                "Cached rho-DROP C-PCM operator is not positive definite."
            ) from exc
        self._linearization_factor = (state.state_sha256, factor)
        return factor

    @staticmethod
    def _point_only_surface_weights(points_bohr: np.ndarray) -> MoistSurfaceWeights:
        point_weights = np.asarray(points_bohr, dtype=float)
        surface_size = point_weights.shape[0]
        return MoistSurfaceWeights(
            xi=np.zeros(surface_size),
            switching=np.zeros(surface_size),
            points_bohr=point_weights,
        )

    def _adjoint_from_state(
        self,
        state: RhoDropCPCMForwardState,
        field_cotangent: np.ndarray,
    ) -> np.ndarray:
        """Apply the exact first-order source VJP at one cached forward state."""

        cotangent = _source_block(
            field_cotangent,
            atom_count=self.atom_count,
            name="rho-DROP reaction-field cotangent",
        )
        surface_state = state.surface_state
        surface = state.surface_snapshot
        source = state.density_coefficients

        # Numerically, Route 2's field cotangent becomes a raw l<=1 source by
        # the same permutation used by the fixed-cavity transpose.  The point
        # multipole kernel then supplies B_Gamma y, including the angstrom-to-
        # bohr dipole conversion.  The final Hartree factor is applied once.
        adjoint_source = external_field_to_density_order(cotangent)
        output_surface_potential = point_multipole_potential(
            surface.surface_points_bohr,
            self.atom_positions_angstrom,
            adjoint_source,
        )
        z = np.asarray(
            cho_solve(
                self._factor_for_state(state),
                output_surface_potential,
                check_finite=False,
            ),
            dtype=float,
        )
        surface_potential_cotangent = -state.dielectric_factor * z

        direct_potential, direct_gradient = point_asc_reaction_potential_gradient(
            self.atom_positions_angstrom,
            surface.surface_points_bohr,
            surface_potential_cotangent,
        )
        direct_source_cotangent = external_field_to_density_order(
            np.column_stack(
                (
                    direct_potential * Hartree,
                    direct_gradient * Hartree / Bohr,
                )
            )
        )

        # q.T d(B y): motion of the reaction-field back-projection kernel.
        output_kernel_weights = self._point_only_surface_weights(
            point_multipole_potential_surface_position_vjp(
                surface.surface_points_bohr,
                self.atom_positions_angstrom,
                adjoint_source,
                state.surface_charge_e,
            )
        )
        # (-f_epsilon z).T d(B c): motion of the solute-MEP kernel.
        source_kernel_weights = self._point_only_surface_weights(
            point_multipole_potential_surface_position_vjp(
                surface.surface_points_bohr,
                self.atom_positions_angstrom,
                source,
                surface_potential_cotangent,
            )
        )
        # -z.T dA q: source-induced motion/width response of the C-PCM operator.
        operator_weights = surface_state.contract_amat_surface_weights(
            -z,
            state.surface_charge_e,
        )
        total_surface_weights = output_kernel_weights.plus(source_kernel_weights).plus(
            operator_weights
        )
        surface_source_cotangent = np.asarray(
            surface_state.source_vjp(total_surface_weights),
            dtype=float,
        )
        expected_shape = (self.atom_count, 4)
        if surface_source_cotangent.shape != expected_shape or not np.all(
            np.isfinite(surface_source_cotangent)
        ):
            raise RuntimeError(
                "MOIST DROP source VJP returned an invalid Route-2 cotangent."
            )
        result = direct_source_cotangent + Hartree * surface_source_cotangent
        if result.shape != expected_shape or not np.all(np.isfinite(result)):
            raise RuntimeError("rho-DROP reaction-map VJP is non-finite.")
        return result

    def apply(self, source_direction: object) -> np.ndarray:
        """Apply the local nonlinear-map JVP at the latest forward state.

        The pinned MOIST API currently exports the efficient DROP reverse
        contraction but not its forward-mode counterpart.  This reference JVP
        obtains ``J d`` analytically from ``d.T J.T e_i`` for the 4N output
        basis vectors.  It neither finite-differences the cavity nor builds
        ``dGamma/dc``.  Its O(4N) reverse sweeps are deliberately marked as not
        efficiency-admitted for production Krylov iterations.
        """

        state = self._required_linearization_state()
        direction = _source_block(
            source_direction,
            atom_count=self.atom_count,
            name="rho-DROP source direction",
        )
        flat_result = np.empty(4 * self.atom_count, dtype=float)
        basis = np.zeros((self.atom_count, 4), dtype=float)
        for index in range(flat_result.size):
            basis.flat[index] = 1.0
            flat_result[index] = float(
                np.vdot(self._adjoint_from_state(state, basis), direction)
            )
            basis.flat[index] = 0.0
        result = flat_result.reshape(self.atom_count, 4)
        if not np.all(np.isfinite(result)):
            raise RuntimeError("rho-DROP reaction-map JVP is non-finite.")
        return result

    def adjoint(self, field_cotangent: object) -> np.ndarray:
        """Apply ``J_f(c,R).T`` at the latest exact forward state."""

        state = self._required_linearization_state()
        cotangent = _source_block(
            field_cotangent,
            atom_count=self.atom_count,
            name="rho-DROP reaction-field cotangent",
        )
        return self._adjoint_from_state(state, cotangent)


__all__ = [
    "RHODROP_CPCM_FORCE_STATUS",
    "RHODROP_CPCM_FORWARD_CONTRACT_VERSION",
    "RHODROP_CPCM_JVP_IMPLEMENTATION",
    "RHODROP_CPCM_MODEL_DRIVE",
    "RHODROP_CPCM_OPERATIONAL_PROFILE_KIND",
    "RhoDropCPCMForwardState",
    "RhoDropCPCMReactionField",
    "RhoDropCPCMSettings",
    "solve_frozen_source_rhodrop_cpcm",
]
