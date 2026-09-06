"""Canonical atomic-density-translation response for the MDP/POLAR hybrid.

This module implements the zero-training response selected by the terminal
uniform-susceptibility audit.  It deliberately keeps three physical categories
separate:

* MACE-MDP supplies the permanent point multipoles and one supervised
  *molecular* polarizability tensor;
* MACE-POLAR supplies its nonlinear four-channel induced residual outside the
  selected uniform-field tangent;
* a canonical free-atom density-translation (ADT) lift supplies the missing
  molecular uniform response as atom-centred analytic dipoles.

For native field ``u`` and the arithmetic affine-potential chart ``g = G u``,
the two induced source branches are

``delta_c_radial = M_P(u) - M_P(0) - (J_P(0) U) g``

and

``p_ADT = P_ADT (-alpha_MDP g)``.

Consequently the residual branch has zero molecular uniform susceptibility,
the ADT branch closes exactly to ``-alpha_MDP``, and every field in ``ker(G)``
retains the original MACE-POLAR response.  The two branches must remain a
direct sum in the continuum; this class intentionally exposes no collapsed
``evaluate_source`` method.

The construction is operational, not a common variational functional.  Its
geometry-dependent chart contains MDP and MACE-POLAR derivatives, so nuclear
coordinate derivatives fail closed until the complete chain rule is
implemented.  No accuracy, force, Hessian, variational, or MD capability is
admitted here.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import threading
from types import MappingProxyType
from typing import Mapping

import numpy as np

from maple.function.calculator.extra_correction.implicit.route2_atomic_reference_density import (
    GaussianMixtureAtom,
)
from maple.solvation.coupling.adt_radial_shape import (
    ADTRadialShapeRegistry,
    load_repository_adt_radial_shape_registry,
)
from maple.solvation.coupling.atomic_displacement_lift import (
    CANONICAL_ADT_LIFT_CONTRACT,
    ROLE_SEPARATED_ADT_LIFT_CONTRACT,
    CanonicalAtomicDisplacementLift,
)
from maple.solvation.coupling.neutral_atom_penetration import (
    NEUTRAL_ATOM_PENETRATION_CONTENT_SHA256,
    NEUTRAL_ATOM_PENETRATION_MANIFEST_REPO_PATH,
    NEUTRAL_ATOM_PENETRATION_TABLE_REPO_PATH,
    load_neutral_atom_penetration_mixtures,
)
from maple.solvation.coupling.operator import canonical_metadata_sha256
from maple.solvation.coupling.separated_operators import NativeFieldSpace
from maple.solvation.coupling.spaces import SourceSpace
from maple.solvation.coupling.state_equation import geometry_sha256
from maple.solvation.release.uniform_response_manifold import (
    affine_uniform_native_field,
)
from maple.solvation.release.uniform_susceptibility_replacement import (
    arithmetic_uniform_gradient_left_inverse,
)

from .base import atom_count
from .mace_mdp_polar_susceptibility import (
    MolecularPolarizabilityProvider,
    NativeFieldSourceResponse,
)

MDP_POLAR_CANONICAL_ADT_CONTRACT = (
    "mace-mdp-molecular-alpha-mace-polar-canonical-adt-tangent-v1"
)
MDP_POLAR_CANONICAL_ADT_RESPONSE_PROVIDER_ID = (
    "maple.route2.model.mace-mdp-alpha-macepolar-canonical-adt-tangent.impl.v1"
)
MDP_POLAR_CANONICAL_ADT_MODEL_PROFILE_ID = (
    "route2-research-mace-mdp-point-permanent-macepolar-canonical-adt-tangent-v1"
)
MDP_POLAR_CANONICAL_ADT_COUPLING_ID = (
    "route2-coupling-mace-mdp-alpha-canonical-adt-macepolar-tangent-v1"
)
MDP_POLAR_ROLE_SEPARATED_ADT_CONTRACT = (
    "mace-mdp-molecular-alpha-mace-polar-role-separated-adt-tangent-v2"
)
MDP_POLAR_ROLE_SEPARATED_ADT_RESPONSE_PROVIDER_ID = (
    "maple.route2.model.mace-mdp-alpha-macepolar-role-separated-adt-tangent.impl.v2"
)
MDP_POLAR_ROLE_SEPARATED_ADT_MODEL_PROFILE_ID = (
    "route2-research-mace-mdp-point-permanent-macepolar-role-separated-adt-tangent-v2"
)
MDP_POLAR_ROLE_SEPARATED_ADT_COUPLING_ID = (
    "route2-coupling-mace-mdp-alpha-role-separated-adt-macepolar-tangent-v2"
)
MACE_POLAR_ZERO_POINT_PERMANENT_CONTRACT = (
    "mace-polar-zero-field-point-permanent-bound-to-canonical-adt-v1"
)
MACE_POLAR_ZERO_POINT_PERMANENT_PROVIDER_ID = (
    "maple.route2.model.macepolar-zero-field-point-permanent.impl.v1"
)
MACE_POLAR_ZERO_POINT_PERMANENT_MODEL_PROFILE_ID = (
    "route2-research-macepolar-zero-field-point-permanent-for-mdp-alpha-adt-v1"
)
CANONICAL_ADT_ATOMIC_DIPOLE_SPACE_ID = (
    "maple.route2.canonical-adt-atomic-cartesian-dipoles.v1"
)

_CHARGE_RESPONSE_ATOL_E = 1.0e-10
_DIPOLE_CLOSURE_ATOL_EANGSTROM = 3.0e-12
_ALPHA_SYMMETRY_RELATIVE_ATOL = 1.0e-10


def _readonly(values: object, *, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite with shape {shape}.")
    contiguous = np.ascontiguousarray(array, dtype=np.float64)
    return np.frombuffer(contiguous.tobytes(), dtype=np.float64).reshape(shape)


def _atomic_numbers(geometry: object) -> np.ndarray:
    getter = getattr(geometry, "get_atomic_numbers", None)
    if not callable(getter):
        raise TypeError("geometry must expose get_atomic_numbers().")
    raw = np.asarray(getter())
    numeric = np.asarray(raw, dtype=np.float64)
    count = atom_count(geometry)
    if (
        numeric.shape != (count,)
        or not np.all(np.isfinite(numeric))
        or not np.array_equal(numeric, np.rint(numeric))
        or np.any(numeric < 1.0)
    ):
        raise ValueError("geometry atomic numbers must be positive integers.")
    contiguous = np.ascontiguousarray(numeric, dtype=np.int64)
    return np.frombuffer(contiguous.tobytes(), dtype=np.int64)


def _positions_angstrom(geometry: object) -> np.ndarray:
    getter = getattr(geometry, "get_positions", None)
    if not callable(getter):
        raise TypeError("geometry must expose get_positions().")
    return _readonly(
        getter(), shape=(atom_count(geometry), 3), name="geometry positions"
    )


def _copy_mixtures(
    values: Mapping[int, GaussianMixtureAtom],
) -> Mapping[int, GaussianMixtureAtom]:
    if not isinstance(values, Mapping) or not values:
        raise TypeError("mixtures_by_atomic_number must be a non-empty mapping.")
    copied: dict[int, GaussianMixtureAtom] = {}
    for raw_number, mixture in values.items():
        if (
            isinstance(raw_number, bool)
            or not isinstance(raw_number, int)
            or raw_number < 1
        ):
            raise ValueError("mixture atomic numbers must be positive integers.")
        if not isinstance(mixture, GaussianMixtureAtom):
            raise TypeError("mixture entries must be GaussianMixtureAtom.")
        if abs(mixture.electron_count - float(raw_number)) > 5.0e-12:
            raise ValueError(f"Gaussian mixture for Z={raw_number} is not normalized.")
        copied[raw_number] = GaussianMixtureAtom(
            np.asarray(mixture.electron_counts, dtype=np.float64),
            np.asarray(mixture.gaussian_exponents_bohr2, dtype=np.float64),
        )
    return MappingProxyType(dict(sorted(copied.items())))


def _mixture_payload(
    mixtures: Mapping[int, GaussianMixtureAtom],
) -> dict[str, object]:
    return {
        str(number): {
            "electron_counts": mixture.electron_counts.tolist(),
            "gaussian_exponents_bohr2": (mixture.gaussian_exponents_bohr2.tolist()),
        }
        for number, mixture in mixtures.items()
    }


def _public_molecular_polarizability(state: object) -> np.ndarray:
    # Deliberately access only the public molecular observable.  The MDP
    # checkpoint's atomwise alpha decomposition is a latent gauge and must not
    # determine the ADT source partition.
    alpha = np.asarray(
        getattr(state, "public_polarizability_eangstrom2_per_volt", None),
        dtype=np.float64,
    )
    if alpha.shape != (3, 3) or not np.all(np.isfinite(alpha)):
        raise RuntimeError("MDP returned no finite public molecular polarizability.")
    symmetric = 0.5 * (alpha + alpha.T)
    scale = max(1.0, float(np.linalg.norm(symmetric, ord=2)))
    if not np.allclose(
        alpha,
        symmetric,
        rtol=0.0,
        atol=_ALPHA_SYMMETRY_RELATIVE_ATOL * scale,
    ):
        raise RuntimeError("MDP molecular polarizability is not symmetric.")
    if float(np.min(np.linalg.eigvalsh(symmetric))) <= 1.0e-12 * scale:
        raise RuntimeError("MDP molecular polarizability is not positive definite.")
    return symmetric


@dataclass(frozen=True, slots=True)
class CanonicalADTResponseChart:
    """One immutable geometry-local arithmetic/ADT response chart."""

    geometry_sha256: str
    atomic_numbers: np.ndarray
    positions_angstrom: np.ndarray
    uniform_native_basis: np.ndarray
    uniform_gradient_left_inverse: np.ndarray
    polar_zero_source4: np.ndarray
    polar_zero_uniform_source_jacobian: np.ndarray
    mdp_molecular_polarizability_eangstrom2_per_volt: np.ndarray
    adt_lift: CanonicalAtomicDisplacementLift
    adt_atomic_dipole_jacobian_eangstrom: np.ndarray
    state_sha256: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.geometry_sha256, str) or len(self.geometry_sha256) != 64:
            raise ValueError("geometry_sha256 must be a SHA256 digest.")
        numbers = np.asarray(self.atomic_numbers)
        if (
            numbers.ndim != 1
            or len(numbers) < 1
            or not np.issubdtype(numbers.dtype, np.integer)
            or np.any(numbers < 1)
        ):
            raise ValueError("atomic_numbers must be positive integers.")
        numbers = np.frombuffer(
            np.ascontiguousarray(numbers, dtype=np.int64).tobytes(), dtype=np.int64
        )
        count = len(numbers)
        positions = _readonly(
            self.positions_angstrom,
            shape=(count, 3),
            name="positions_angstrom",
        )
        basis = _readonly(
            self.uniform_native_basis,
            shape=(count, 8, 3),
            name="uniform_native_basis",
        )
        left = _readonly(
            self.uniform_gradient_left_inverse,
            shape=(3, count * 8),
            name="uniform_gradient_left_inverse",
        )
        zero = _readonly(
            self.polar_zero_source4,
            shape=(count, 4),
            name="polar_zero_source4",
        )
        polar_tangent = _readonly(
            self.polar_zero_uniform_source_jacobian,
            shape=(count, 4, 3),
            name="polar_zero_uniform_source_jacobian",
        )
        alpha = _readonly(
            self.mdp_molecular_polarizability_eangstrom2_per_volt,
            shape=(3, 3),
            name="mdp_molecular_polarizability_eangstrom2_per_volt",
        )
        adt_jacobian = _readonly(
            self.adt_atomic_dipole_jacobian_eangstrom,
            shape=(count, 3, 3),
            name="adt_atomic_dipole_jacobian_eangstrom",
        )
        if not isinstance(self.adt_lift, CanonicalAtomicDisplacementLift):
            raise TypeError("adt_lift must be CanonicalAtomicDisplacementLift.")
        if not np.array_equal(numbers, self.adt_lift.atomic_numbers):
            raise ValueError("ADT lift atomic numbers do not match the chart.")
        flat_basis = basis.reshape(count * 8, 3)
        if not np.allclose(left @ flat_basis, np.eye(3), rtol=0.0, atol=2.0e-14):
            raise ValueError("uniform gradient chart is not a left inverse.")
        if np.max(np.abs(np.sum(polar_tangent[:, 0, :], axis=0))) > (
            _CHARGE_RESPONSE_ATOL_E
        ):
            raise ValueError("MACE-POLAR uniform tangent changes total charge.")
        symmetric = 0.5 * (alpha + alpha.T)
        scale = max(1.0, float(np.linalg.norm(symmetric, ord=2)))
        if (
            not np.allclose(
                alpha,
                symmetric,
                rtol=0.0,
                atol=_ALPHA_SYMMETRY_RELATIVE_ATOL * scale,
            )
            or float(np.min(np.linalg.eigvalsh(symmetric))) <= 1.0e-12 * scale
        ):
            raise ValueError(
                "chart molecular polarizability must be symmetric positive."
            )
        expected_adt = -np.einsum(
            "a,ij->aij", self.adt_lift.atomic_dipole_weights, symmetric
        )
        if not np.allclose(adt_jacobian, expected_adt, rtol=0.0, atol=2.0e-14):
            raise ValueError("ADT atomic dipole Jacobian is not canonical.")
        if not np.allclose(
            np.sum(adt_jacobian, axis=0),
            -symmetric,
            rtol=0.0,
            atol=_DIPOLE_CLOSURE_ATOL_EANGSTROM,
        ):
            raise ValueError("ADT response does not close to molecular alpha.")

        response_contract = (
            MDP_POLAR_ROLE_SEPARATED_ADT_CONTRACT
            if self.adt_lift.contract_id == ROLE_SEPARATED_ADT_LIFT_CONTRACT
            else MDP_POLAR_CANONICAL_ADT_CONTRACT
        )
        expected_hash = canonical_metadata_sha256(
            {
                "contract": response_contract,
                "geometry_sha256": self.geometry_sha256,
                "atomic_numbers": numbers.tolist(),
                "positions_angstrom": positions.tolist(),
                "uniform_native_basis": basis.tolist(),
                "uniform_gradient_left_inverse": left.tolist(),
                "polar_zero_source4": zero.tolist(),
                "polar_zero_uniform_source_jacobian": polar_tangent.tolist(),
                "mdp_molecular_polarizability": symmetric.tolist(),
                "adt_lift_configuration_sha256": self.adt_lift.configuration_sha256,
                "adt_atomic_dipole_jacobian_eangstrom": adt_jacobian.tolist(),
                "adt_atomic_dipole_space_id": CANONICAL_ADT_ATOMIC_DIPOLE_SPACE_ID,
            }
        )
        if self.state_sha256 and self.state_sha256 != expected_hash:
            raise ValueError("state_sha256 does not match the ADT response chart.")
        object.__setattr__(self, "atomic_numbers", numbers)
        object.__setattr__(self, "positions_angstrom", positions)
        object.__setattr__(self, "uniform_native_basis", basis)
        object.__setattr__(self, "uniform_gradient_left_inverse", left)
        object.__setattr__(self, "polar_zero_source4", zero)
        object.__setattr__(self, "polar_zero_uniform_source_jacobian", polar_tangent)
        object.__setattr__(
            self, "mdp_molecular_polarizability_eangstrom2_per_volt", symmetric
        )
        object.__setattr__(self, "adt_atomic_dipole_jacobian_eangstrom", adt_jacobian)
        object.__setattr__(self, "state_sha256", expected_hash)

    @property
    def atom_count(self) -> int:
        return int(len(self.atomic_numbers))

    def _field(self, values: object, *, name: str) -> np.ndarray:
        return _readonly(values, shape=(self.atom_count, 8), name=name)

    def uniform_coordinates(self, field: object) -> np.ndarray:
        values = self._field(field, name="native field")
        result = self.uniform_gradient_left_inverse @ values.reshape(-1)
        result.setflags(write=False)
        return result

    def polar_uniform_tangent(self, field: object) -> np.ndarray:
        result = np.einsum(
            "nsc,c->ns",
            self.polar_zero_uniform_source_jacobian,
            self.uniform_coordinates(field),
        )
        result.setflags(write=False)
        return result

    def adt_atomic_dipoles(self, field: object) -> np.ndarray:
        result = np.einsum(
            "aic,c->ai",
            self.adt_atomic_dipole_jacobian_eangstrom,
            self.uniform_coordinates(field),
        )
        result.setflags(write=False)
        return result

    def pullback_polar_uniform_tangent(self, cotangent: object) -> np.ndarray:
        source_bar = _readonly(
            cotangent,
            shape=(self.atom_count, 4),
            name="radial residual cotangent",
        )
        coordinate_bar = np.einsum(
            "nsc,ns->c", self.polar_zero_uniform_source_jacobian, source_bar
        )
        result = (self.uniform_gradient_left_inverse.T @ coordinate_bar).reshape(
            self.atom_count, 8
        )
        result.setflags(write=False)
        return result

    def pullback_adt_atomic_dipoles(self, cotangent: object) -> np.ndarray:
        dipole_bar = _readonly(
            cotangent,
            shape=(self.atom_count, 3),
            name="ADT atomic-dipole cotangent",
        )
        coordinate_bar = np.einsum(
            "aic,ai->c", self.adt_atomic_dipole_jacobian_eangstrom, dipole_bar
        )
        result = (self.uniform_gradient_left_inverse.T @ coordinate_bar).reshape(
            self.atom_count, 8
        )
        result.setflags(write=False)
        return result


@dataclass(frozen=True, slots=True)
class CanonicalADTInducedComponents:
    """Direct-sum induced source returned by the canonical ADT response."""

    radial_residual_source4: np.ndarray
    adt_atomic_dipoles_eangstrom: np.ndarray
    chart_sha256: str

    def __post_init__(self) -> None:
        radial = np.asarray(self.radial_residual_source4, dtype=np.float64)
        dipoles = np.asarray(self.adt_atomic_dipoles_eangstrom, dtype=np.float64)
        if radial.ndim != 2 or radial.shape[1] != 4 or not np.all(np.isfinite(radial)):
            raise ValueError("radial_residual_source4 must be finite with shape (N,4).")
        count = len(radial)
        radial = _readonly(radial, shape=(count, 4), name="radial_residual_source4")
        dipoles = _readonly(
            dipoles,
            shape=(count, 3),
            name="adt_atomic_dipoles_eangstrom",
        )
        if not isinstance(self.chart_sha256, str) or len(self.chart_sha256) != 64:
            raise ValueError("chart_sha256 must be a SHA256 digest.")
        if abs(float(np.sum(radial[:, 0]))) > _CHARGE_RESPONSE_ATOL_E:
            raise ValueError("radial residual changes total charge.")
        object.__setattr__(self, "radial_residual_source4", radial)
        object.__setattr__(self, "adt_atomic_dipoles_eangstrom", dipoles)

    @property
    def atom_count(self) -> int:
        return int(len(self.radial_residual_source4))


class MDPPolarCanonicalADTResponse:
    """Separate residual-radial and canonical-ADT induced response branches."""

    __slots__ = (
        "_base",
        "_cache_chart",
        "_cache_geometry_sha256",
        "_cache_lock",
        "_configuration_sha256",
        "_contract_id",
        "_mdp",
        "_mixtures",
        "_radial_shape_registry",
        "_sealed",
        "_source_asset_sha256",
        "coupling_id",
        "model_profile_id",
        "provenance_sha256",
        "provider_id",
    )

    capabilities = ()
    variational_functional_admitted = False
    coordinate_derivative_available = False
    response_kind = "canonical-adt-plus-polar-nonuniform-residual"
    adt_atomic_dipole_space_id = CANONICAL_ADT_ATOMIC_DIPOLE_SPACE_ID

    def __init__(
        self,
        *,
        mdp: MolecularPolarizabilityProvider,
        base: NativeFieldSourceResponse,
        mixtures_by_atomic_number: Mapping[int, GaussianMixtureAtom] | None = None,
        source_asset_sha256: str | None = None,
        radial_shape_registry: ADTRadialShapeRegistry | None = None,
    ) -> None:
        for owner, names in (
            (mdp, ("configuration_sha256", "evaluate")),
            (
                base,
                (
                    "configuration_sha256",
                    "vacuum_energy_ev",
                    "vacuum_forces_ev_per_angstrom",
                    "evaluate_source",
                    "field_jvp",
                    "field_vjp",
                ),
            ),
        ):
            for name in names:
                if not callable(getattr(owner, name, None)):
                    raise TypeError(
                        f"canonical ADT response requires callable {name}()."
                    )
        source_space = getattr(base, "source_space", None)
        receiver_space = getattr(base, "receiver_space", None)
        if not isinstance(source_space, SourceSpace):
            raise TypeError("base response must expose SourceSpace.")
        if not isinstance(receiver_space, NativeFieldSpace):
            raise TypeError("base response must expose NativeFieldSpace.")
        if source_space.component_count != 4 or receiver_space.component_count != 8:
            raise ValueError(
                "canonical ADT response requires source4 and native field8."
            )
        role_separated = radial_shape_registry is not None
        if role_separated:
            if mixtures_by_atomic_number is not None or source_asset_sha256 is not None:
                raise ValueError(
                    "role-separated ADT registry cannot be combined with legacy mixtures."
                )
            if not isinstance(radial_shape_registry, ADTRadialShapeRegistry):
                raise TypeError("radial_shape_registry must be ADTRadialShapeRegistry.")
            registry: ADTRadialShapeRegistry | None = radial_shape_registry
            mixtures = MappingProxyType(
                {
                    number: GaussianMixtureAtom(
                        np.asarray(shape.mixture.electron_counts, dtype=np.float64),
                        np.asarray(
                            shape.mixture.gaussian_exponents_bohr2, dtype=np.float64
                        ),
                    )
                    for number, shape in registry.shapes_by_atomic_number.items()
                }
            )
            asset_sha256 = registry.configuration_sha256
            contract_id = MDP_POLAR_ROLE_SEPARATED_ADT_CONTRACT
            provider_id = MDP_POLAR_ROLE_SEPARATED_ADT_RESPONSE_PROVIDER_ID
            model_profile_id = MDP_POLAR_ROLE_SEPARATED_ADT_MODEL_PROFILE_ID
            coupling_id = MDP_POLAR_ROLE_SEPARATED_ADT_COUPLING_ID
            lift_contract = ROLE_SEPARATED_ADT_LIFT_CONTRACT
        else:
            if mixtures_by_atomic_number is None:
                raise TypeError(
                    "legacy canonical ADT requires mixtures_by_atomic_number."
                )
            if (
                not isinstance(source_asset_sha256, str)
                or len(source_asset_sha256) != 64
            ):
                raise ValueError("source_asset_sha256 must be a SHA256 digest.")
            int(source_asset_sha256, 16)
            registry = None
            mixtures = _copy_mixtures(mixtures_by_atomic_number)
            asset_sha256 = source_asset_sha256.lower()
            contract_id = MDP_POLAR_CANONICAL_ADT_CONTRACT
            provider_id = MDP_POLAR_CANONICAL_ADT_RESPONSE_PROVIDER_ID
            model_profile_id = MDP_POLAR_CANONICAL_ADT_MODEL_PROFILE_ID
            coupling_id = MDP_POLAR_CANONICAL_ADT_COUPLING_ID
            lift_contract = CANONICAL_ADT_LIFT_CONTRACT
        mdp.configuration_sha256()
        base.configuration_sha256()
        object.__setattr__(self, "_mdp", mdp)
        object.__setattr__(self, "_base", base)
        object.__setattr__(self, "_mixtures", mixtures)
        object.__setattr__(self, "_radial_shape_registry", registry)
        object.__setattr__(self, "_source_asset_sha256", asset_sha256)
        object.__setattr__(self, "_contract_id", contract_id)
        object.__setattr__(self, "provider_id", provider_id)
        object.__setattr__(self, "model_profile_id", model_profile_id)
        object.__setattr__(self, "coupling_id", coupling_id)
        provenance = canonical_metadata_sha256(
            {
                "contract": contract_id,
                "provider_id": self.provider_id,
                "model_profile_id": self.model_profile_id,
                "coupling_id": self.coupling_id,
                "mdp_provider_id": getattr(mdp, "provider_id", None),
                "mdp_model_profile_id": getattr(mdp, "model_profile_id", None),
                "mdp_configuration_sha256": mdp.configuration_sha256(),
                "base_provider_id": getattr(base, "provider_id", None),
                "base_model_profile_id": getattr(base, "model_profile_id", None),
                "base_provenance_sha256": getattr(base, "provenance_sha256", None),
                "base_configuration_sha256": base.configuration_sha256(),
                "source_space_sha256": source_space.metadata_hash(),
                "receiver_space_sha256": receiver_space.metadata_hash(),
                "adt_lift_contract": lift_contract,
                "adt_source_asset_sha256": self._source_asset_sha256,
                "adt_mixtures": _mixture_payload(mixtures),
                "adt_radial_shape_registry_sha256": (
                    registry.configuration_sha256 if registry is not None else None
                ),
                "uniform_chart": "arithmetic-average-native-gradient-only-v1",
                "direct_sum": (
                    "polar-source4-nonuniform-residual+canonical-adt-atomic-dipoles"
                ),
                "coordinate_derivative_available": False,
                "variational_functional_admitted": False,
                "capabilities": "none",
                "implementation_sha256": hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
            }
        )
        object.__setattr__(self, "provenance_sha256", provenance)
        object.__setattr__(self, "_cache_geometry_sha256", None)
        object.__setattr__(self, "_cache_chart", None)
        object.__setattr__(self, "_cache_lock", threading.RLock())
        object.__setattr__(self, "_configuration_sha256", self._current_configuration())
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("MDPPolarCanonicalADTResponse is immutable.")
        object.__setattr__(self, name, value)

    @classmethod
    def from_repository_assets(
        cls,
        *,
        mdp: MolecularPolarizabilityProvider,
        base: NativeFieldSourceResponse,
        source_root: str | Path | None = None,
    ) -> "MDPPolarCanonicalADTResponse":
        root = (
            Path(source_root).expanduser().resolve()
            if source_root is not None
            else Path(__file__).resolve().parents[3]
        )
        mixtures = load_neutral_atom_penetration_mixtures(
            table_path=root / NEUTRAL_ATOM_PENETRATION_TABLE_REPO_PATH,
            manifest_path=root / NEUTRAL_ATOM_PENETRATION_MANIFEST_REPO_PATH,
        )
        return cls(
            mdp=mdp,
            base=base,
            mixtures_by_atomic_number=mixtures,
            source_asset_sha256=NEUTRAL_ATOM_PENETRATION_CONTENT_SHA256,
        )

    @classmethod
    def from_role_separated_repository_assets(
        cls,
        *,
        mdp: MolecularPolarizabilityProvider,
        base: NativeFieldSourceResponse,
        source_root: str | Path | None = None,
    ) -> "MDPPolarCanonicalADTResponse":
        """Build v2 with an ADT-only iodine ECP-valence response shape."""

        registry = load_repository_adt_radial_shape_registry(source_root=source_root)
        return cls(mdp=mdp, base=base, radial_shape_registry=registry)

    @property
    def contract_id(self) -> str:
        return self._contract_id

    @property
    def source_space(self) -> SourceSpace:
        return self._base.source_space

    @property
    def mdp_configuration_sha256(self) -> str:
        """Return the exact MDP identity supplying molecular polarizability."""

        return self._mdp.configuration_sha256()

    @property
    def receiver_space(self) -> NativeFieldSpace:
        return self._base.receiver_space

    @property
    def long_range_evaluator_profile(self) -> str:
        return str(self._base.long_range_evaluator_profile)

    @property
    def field_energy_pairing_sha256(self) -> str:
        value = getattr(self._base, "field_energy_pairing_sha256", None)
        if not isinstance(value, str) or len(value) != 64:
            raise RuntimeError("base response exposes no valid field pairing digest.")
        return value

    def _current_configuration(self) -> str:
        return canonical_metadata_sha256(
            {
                "contract": self.contract_id,
                "provider_id": self.provider_id,
                "model_profile_id": self.model_profile_id,
                "coupling_id": self.coupling_id,
                "provenance_sha256": self.provenance_sha256,
                "mdp_configuration_sha256": self._mdp.configuration_sha256(),
                "base_configuration_sha256": self._base.configuration_sha256(),
                "source_space_sha256": self.source_space.metadata_hash(),
                "receiver_space_sha256": self.receiver_space.metadata_hash(),
                "long_range_evaluator_profile": self.long_range_evaluator_profile,
                "adt_source_asset_sha256": self._source_asset_sha256,
                "adt_mixtures": _mixture_payload(self._mixtures),
                "adt_radial_shape_registry_sha256": (
                    self._radial_shape_registry.configuration_sha256
                    if self._radial_shape_registry is not None
                    else None
                ),
                "coordinate_derivative_available": False,
                "capabilities": "none",
            }
        )

    def configuration_sha256(self) -> str:
        current = self._current_configuration()
        if current != self._configuration_sha256:
            raise RuntimeError("MDP/POLAR canonical ADT configuration drifted.")
        return current

    def _chart(self, geometry: object) -> CanonicalADTResponseChart:
        self.configuration_sha256()
        before = geometry_sha256(geometry)
        with self._cache_lock:
            if self._cache_geometry_sha256 == before and isinstance(
                self._cache_chart, CanonicalADTResponseChart
            ):
                return self._cache_chart
            count = atom_count(geometry)
            numbers = _atomic_numbers(geometry)
            positions = _positions_angstrom(geometry)
            missing = sorted(set(numbers.tolist()) - set(self._mixtures))
            if missing:
                raise ValueError(f"canonical ADT asset has no elements {missing}.")
            basis = np.stack(
                [
                    affine_uniform_native_field(positions, np.eye(3)[axis])
                    for axis in range(3)
                ],
                axis=-1,
            )
            left = arithmetic_uniform_gradient_left_inverse(count)
            zero_field = np.zeros(self.receiver_space.shape(count), dtype=np.float64)
            zero_source = self.source_space.validate(
                self._base.evaluate_source(geometry, zero_field),
                atom_count=count,
                name="MACE-POLAR zero-field source",
            )
            polar_tangent = np.stack(
                [
                    self.source_space.validate(
                        self._base.field_jvp(geometry, zero_field, basis[:, :, axis]),
                        atom_count=count,
                        name="MACE-POLAR zero-field uniform JVP",
                    )
                    for axis in range(3)
                ],
                axis=-1,
            )
            alpha = _public_molecular_polarizability(self._mdp.evaluate(geometry))
            if self._radial_shape_registry is None:
                lift = CanonicalAtomicDisplacementLift.from_mixtures(
                    atomic_numbers=numbers,
                    mixtures_by_atomic_number=self._mixtures,
                    source_asset_sha256=self._source_asset_sha256,
                )
            else:
                lift = CanonicalAtomicDisplacementLift.from_radial_shapes(
                    atomic_numbers=numbers,
                    registry=self._radial_shape_registry,
                )
            adt_jacobian = -np.einsum("a,ij->aij", lift.atomic_dipole_weights, alpha)
            chart = CanonicalADTResponseChart(
                geometry_sha256=before,
                atomic_numbers=numbers,
                positions_angstrom=positions,
                uniform_native_basis=basis,
                uniform_gradient_left_inverse=left,
                polar_zero_source4=zero_source,
                polar_zero_uniform_source_jacobian=polar_tangent,
                mdp_molecular_polarizability_eangstrom2_per_volt=alpha,
                adt_lift=lift,
                adt_atomic_dipole_jacobian_eangstrom=adt_jacobian,
            )
            after = geometry_sha256(geometry)
            if after != before:
                raise RuntimeError(
                    "geometry changed while building canonical ADT chart."
                )
            object.__setattr__(self, "_cache_geometry_sha256", before)
            object.__setattr__(self, "_cache_chart", chart)
            return chart

    def chart_for_geometry(self, geometry: object) -> CanonicalADTResponseChart:
        """Return the immutable chart for target-free structural audits."""

        return self._chart(geometry)

    def vacuum_energy_ev(self, geometry: object) -> float:
        self.configuration_sha256()
        value = float(self._base.vacuum_energy_ev(geometry))
        if not np.isfinite(value):
            raise RuntimeError("base vacuum energy is non-finite.")
        return value

    def vacuum_forces_ev_per_angstrom(self, geometry: object) -> np.ndarray:
        self.configuration_sha256()
        result = np.asarray(
            self._base.vacuum_forces_ev_per_angstrom(geometry), dtype=np.float64
        )
        if result.shape != (atom_count(geometry), 3) or not np.all(np.isfinite(result)):
            raise RuntimeError("base vacuum forces must be finite with shape (N,3).")
        return result.copy()

    def evaluate_components(
        self, geometry: object, field: object
    ) -> CanonicalADTInducedComponents:
        chart = self._chart(geometry)
        values = self.receiver_space.validate(
            field, atom_count=chart.atom_count, name="native field"
        )
        response = self.source_space.validate(
            self._base.evaluate_source(geometry, values),
            atom_count=chart.atom_count,
            name="MACE-POLAR source",
        )
        radial = (
            response - chart.polar_zero_source4 - chart.polar_uniform_tangent(values)
        )
        radial = self.source_space.validate(
            radial,
            atom_count=chart.atom_count,
            name="MACE-POLAR nonuniform residual source",
        )
        return CanonicalADTInducedComponents(
            radial_residual_source4=radial,
            adt_atomic_dipoles_eangstrom=chart.adt_atomic_dipoles(values),
            chart_sha256=chart.state_sha256,
        )

    def original_induced_source(self, geometry: object, field: object) -> np.ndarray:
        chart = self._chart(geometry)
        values = self.receiver_space.validate(field, atom_count=chart.atom_count)
        result = (
            self.source_space.validate(
                self._base.evaluate_source(geometry, values),
                atom_count=chart.atom_count,
            )
            - chart.polar_zero_source4
        )
        return self.source_space.validate(
            result, atom_count=chart.atom_count, name="original MACE-POLAR increment"
        )

    def field_jvp_components(
        self, geometry: object, field: object, field_direction: object
    ) -> CanonicalADTInducedComponents:
        chart = self._chart(geometry)
        values = self.receiver_space.validate(field, atom_count=chart.atom_count)
        direction = self.receiver_space.validate(
            field_direction,
            atom_count=chart.atom_count,
            name="native field direction",
        )
        radial = self.source_space.validate(
            self._base.field_jvp(geometry, values, direction),
            atom_count=chart.atom_count,
            name="MACE-POLAR source JVP",
        ) - chart.polar_uniform_tangent(direction)
        return CanonicalADTInducedComponents(
            radial_residual_source4=radial,
            adt_atomic_dipoles_eangstrom=chart.adt_atomic_dipoles(direction),
            chart_sha256=chart.state_sha256,
        )

    def field_vjp_components(
        self,
        geometry: object,
        field: object,
        *,
        radial_source_cotangent: object,
        adt_atomic_dipole_cotangent: object,
    ) -> np.ndarray:
        chart = self._chart(geometry)
        values = self.receiver_space.validate(field, atom_count=chart.atom_count)
        radial_bar = self.source_space.validate(
            radial_source_cotangent,
            atom_count=chart.atom_count,
            name="radial residual cotangent",
        )
        adt_bar = _readonly(
            adt_atomic_dipole_cotangent,
            shape=(chart.atom_count, 3),
            name="ADT atomic-dipole cotangent",
        )
        result = self.receiver_space.validate(
            self._base.field_vjp(geometry, values, radial_bar),
            atom_count=chart.atom_count,
            name="base native-field VJP",
        )
        result = (
            result
            - chart.pullback_polar_uniform_tangent(radial_bar)
            + chart.pullback_adt_atomic_dipoles(adt_bar)
        )
        return self.receiver_space.validate(
            result,
            atom_count=chart.atom_count,
            name="canonical ADT direct-sum field VJP",
        )

    def dense_component_jacobians(
        self, geometry: object, field: object
    ) -> tuple[np.ndarray, np.ndarray]:
        chart = self._chart(geometry)
        values = self.receiver_space.validate(field, atom_count=chart.atom_count)
        field_dimension = chart.atom_count * 8
        radial = np.empty((chart.atom_count * 4, field_dimension), dtype=np.float64)
        adt = np.empty((chart.atom_count * 3, field_dimension), dtype=np.float64)
        for column in range(field_dimension):
            direction = np.zeros((chart.atom_count, 8), dtype=np.float64)
            direction.reshape(-1)[column] = 1.0
            components = self.field_jvp_components(geometry, values, direction)
            radial[:, column] = components.radial_residual_source4.reshape(-1)
            adt[:, column] = components.adt_atomic_dipoles_eangstrom.reshape(-1)
        radial.setflags(write=False)
        adt.setflags(write=False)
        return radial, adt

    def coordinate_vjp(self, *args: object, **kwargs: object) -> np.ndarray:
        del args, kwargs
        raise NotImplementedError(
            "The canonical ADT chart depends on MDP alpha, the zero-field POLAR "
            "Jacobian, the arithmetic field chart, and translated atomic density; "
            "its complete coordinate VJP is not implemented."
        )


class MACEPolarZeroFieldPointPermanentSource:
    """Permanent point multipoles taken from the same POLAR response chart.

    MACE-MDP remains the provider of the supervised molecular polarizability
    used by :class:`MDPPolarCanonicalADTResponse`; it is deliberately *not*
    treated as an atomwise electrostatic-density model.  This immutable view
    exposes the zero-field MACE-POLAR source already cached by that exact
    response object, preventing a second checkpoint evaluation or a hidden
    adapter mismatch.

    The source is interpreted as point monopoles/dipoles only by the separated
    continuum profile that consumes this provider.  No energy, force,
    variational, or production capability is admitted here.
    """

    __slots__ = (
        "_configuration_sha256",
        "_response",
        "_sealed",
        "model_profile_id",
        "provenance_sha256",
        "provider_id",
    )

    capabilities = ()
    coordinate_derivative_available = False
    variational_functional_admitted = False
    source_representation = "point-monopoles-and-cartesian-dipoles"

    def __init__(self, response: MDPPolarCanonicalADTResponse) -> None:
        if not isinstance(response, MDPPolarCanonicalADTResponse):
            raise TypeError("response must be MDPPolarCanonicalADTResponse.")
        response_configuration = response.configuration_sha256()
        object.__setattr__(self, "_response", response)
        object.__setattr__(
            self, "provider_id", MACE_POLAR_ZERO_POINT_PERMANENT_PROVIDER_ID
        )
        object.__setattr__(
            self,
            "model_profile_id",
            MACE_POLAR_ZERO_POINT_PERMANENT_MODEL_PROFILE_ID,
        )
        provenance = canonical_metadata_sha256(
            {
                "contract": MACE_POLAR_ZERO_POINT_PERMANENT_CONTRACT,
                "provider_id": self.provider_id,
                "model_profile_id": self.model_profile_id,
                "response_provider_id": response.provider_id,
                "response_model_profile_id": response.model_profile_id,
                "response_provenance_sha256": response.provenance_sha256,
                "response_configuration_sha256": response_configuration,
                "source_space_sha256": response.source_space.metadata_hash(),
                "source_representation": self.source_representation,
                "coordinate_derivative_available": False,
                "variational_functional_admitted": False,
                "capabilities": "none",
                "implementation_sha256": hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
            }
        )
        object.__setattr__(self, "provenance_sha256", provenance)
        object.__setattr__(self, "_configuration_sha256", self._current_configuration())
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("MACEPolarZeroFieldPointPermanentSource is immutable.")
        object.__setattr__(self, name, value)

    @property
    def source_space(self) -> SourceSpace:
        return self._response.source_space

    @property
    def response_configuration_sha256(self) -> str:
        return self._response.configuration_sha256()

    def _current_configuration(self) -> str:
        return canonical_metadata_sha256(
            {
                "contract": MACE_POLAR_ZERO_POINT_PERMANENT_CONTRACT,
                "provider_id": self.provider_id,
                "model_profile_id": self.model_profile_id,
                "provenance_sha256": self.provenance_sha256,
                "response_configuration_sha256": (
                    self._response.configuration_sha256()
                ),
                "source_space_sha256": self.source_space.metadata_hash(),
                "source_representation": self.source_representation,
                "coordinate_derivative_available": False,
                "capabilities": "none",
            }
        )

    def configuration_sha256(self) -> str:
        current = self._current_configuration()
        if current != self._configuration_sha256:
            raise RuntimeError("POLAR zero-field permanent-source view drifted.")
        return current

    def evaluate_source(self, geometry: object) -> np.ndarray:
        chart = self._response.chart_for_geometry(geometry)
        return _readonly(
            chart.polar_zero_source4,
            shape=(chart.atom_count, 4),
            name="MACE-POLAR zero-field point permanent source",
        )

    def coordinate_vjp(self, *args: object, **kwargs: object) -> np.ndarray:
        del args, kwargs
        raise NotImplementedError(
            "The zero-field MACE-POLAR source coordinate VJP is not yet "
            "implemented for this research profile."
        )


def build_mdp_polar_canonical_adt_response(
    *,
    mdp: MolecularPolarizabilityProvider,
    base: NativeFieldSourceResponse,
    source_root: str | Path | None = None,
) -> MDPPolarCanonicalADTResponse:
    """Build the content-addressed repository-asset canonical ADT response."""

    return MDPPolarCanonicalADTResponse.from_repository_assets(
        mdp=mdp, base=base, source_root=source_root
    )


def build_mdp_polar_role_separated_adt_response(
    *,
    mdp: MolecularPolarizabilityProvider,
    base: NativeFieldSourceResponse,
    source_root: str | Path | None = None,
) -> MDPPolarCanonicalADTResponse:
    """Build v2 with role-separated ADT shapes, including iodine ECP valence."""

    return MDPPolarCanonicalADTResponse.from_role_separated_repository_assets(
        mdp=mdp, base=base, source_root=source_root
    )


__all__ = [
    "CANONICAL_ADT_ATOMIC_DIPOLE_SPACE_ID",
    "CanonicalADTInducedComponents",
    "CanonicalADTResponseChart",
    "MDP_POLAR_CANONICAL_ADT_CONTRACT",
    "MDP_POLAR_CANONICAL_ADT_COUPLING_ID",
    "MDP_POLAR_CANONICAL_ADT_MODEL_PROFILE_ID",
    "MDP_POLAR_CANONICAL_ADT_RESPONSE_PROVIDER_ID",
    "MDP_POLAR_ROLE_SEPARATED_ADT_CONTRACT",
    "MDP_POLAR_ROLE_SEPARATED_ADT_COUPLING_ID",
    "MDP_POLAR_ROLE_SEPARATED_ADT_MODEL_PROFILE_ID",
    "MDP_POLAR_ROLE_SEPARATED_ADT_RESPONSE_PROVIDER_ID",
    "MACE_POLAR_ZERO_POINT_PERMANENT_CONTRACT",
    "MACE_POLAR_ZERO_POINT_PERMANENT_MODEL_PROFILE_ID",
    "MACE_POLAR_ZERO_POINT_PERMANENT_PROVIDER_ID",
    "MACEPolarZeroFieldPointPermanentSource",
    "MDPPolarCanonicalADTResponse",
    "build_mdp_polar_canonical_adt_response",
    "build_mdp_polar_role_separated_adt_response",
]
