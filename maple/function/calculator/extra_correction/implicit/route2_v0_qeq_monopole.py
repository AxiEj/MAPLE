"""Published-QEq-hardness monopole tangent for the no-training Route-2 V0-Q.

This module is a deliberately limited physical curvature binding.  It combines
the published Rappé--Goddard atomic QEq hardnesses with the *same* single
Gaussian GTO density metric used by the frozen MACE-POLAR source and the
Galerkin PCM.  Only monopole coefficients are allowed to change; all ``l=1``
components are exact homogeneous constraints, not a large penalty or a fitted
polarizability.

It is an internal fixed-geometry falsifier.  It has no nonpolar term and must
not be reported as a complete solvation model, a force/PES implementation, or
an accuracy improvement.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import math
from pathlib import Path

import numpy as np
from ase.units import Bohr, Hartree

from .gto_density import MACE_POLAR_DENSITY_SIGMA_ANGSTROM
from .gto_galerkin import FixedCavityGTOGalerkinOperator
from .route2_v0_variational_quadratic import (
    Route2V0VariationalQuadraticState,
    embed_mace_polar_l1_density_in_single_radial_gto,
    solve_route2_v0_variational_quadratic,
)


RAPPE_GODDARD_QEQ_TANGENT_CONSTRUCTION = (
    "route2-v0-rappe-goddard-hardness-same-basis-monopole-v1"
)
RAPPE_GODDARD_QEQ_PARAMETER_SHA256 = (
    "5d2b405b78dd59b95da89b69fb10b409bcb3f0f85921fc8cc4a67dd2aba8a618"
)
RAPPE_GODDARD_QEQ_DOI = "10.1021/j100161a070"
OPENBABEL_GAUSSIAN_QEQ_REFERENCE_DOI = "10.1007/978-90-481-2596-8_19"

# Open Babel documents that only these rows are the published Rappé--Goddard
# set; the remaining qeq.dat rows come from unpublished UFF-distributed values.
RAPPE_GODDARD_PUBLISHED_QEQ_SYMBOLS = frozenset(
    {
        "H",
        "Li",
        "C",
        "N",
        "O",
        "F",
        "Na",
        "Si",
        "P",
        "S",
        "Cl",
        "K",
        "Br",
        "Rb",
        "I",
        "Cs",
    }
)


def _parameter_path() -> Path:
    return Path(__file__).resolve().parents[1] / "charge" / "data" / "qeq.dat"


def _immutable_array(
    values: np.ndarray,
    *,
    name: str,
    shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if (
        (shape is not None and array.shape != shape)
        or not np.all(np.isfinite(array))
    ):
        expected = "a finite array" if shape is None else f"shape {shape}"
        raise ValueError(f"{name} must be finite with {expected}.")
    result = np.array(array, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _require_single_native_radial_basis(
    operator: FixedCavityGTOGalerkinOperator,
) -> None:
    basis = operator.basis
    if basis.radial_count != 1:
        raise ValueError(
            "The Rappé-Goddard monopole tangent requires the native "
            "single-radial MACE-POLAR GTO basis."
        )
    if not math.isclose(
        basis.sigmas_angstrom[0],
        MACE_POLAR_DENSITY_SIGMA_ANGSTROM,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError(
            "The Rappé-Goddard monopole tangent requires the native "
            "MACE-POLAR Gaussian density width."
        )


@dataclass(frozen=True)
class RappeGoddardQEqAtomicParameter:
    """One pinned atomic QEq row in its tabulated native units."""

    symbol: str
    electronegativity_ev: float
    hardness_ev_per_e2: float
    screening_radius_angstrom: float

    def __post_init__(self) -> None:
        if self.symbol not in RAPPE_GODDARD_PUBLISHED_QEQ_SYMBOLS:
            raise ValueError("Unsupported published Rappé-Goddard QEq element.")
        for name in (
            "electronegativity_ev",
            "hardness_ev_per_e2",
            "screening_radius_angstrom",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or (
                name != "electronegativity_ev" and value <= 0.0
            ):
                raise ValueError(f"{name} must be finite and positive.")
            object.__setattr__(self, name, value)


@lru_cache(maxsize=1)
def _published_parameter_table() -> dict[str, RappeGoddardQEqAtomicParameter]:
    path = _parameter_path()
    contents = path.read_bytes()
    actual_hash = hashlib.sha256(contents).hexdigest()
    if actual_hash != RAPPE_GODDARD_QEQ_PARAMETER_SHA256:
        raise RuntimeError(
            "The QEq parameter table changed; update its physical provenance "
            "before using the Route-2 V0-Q tangent."
        )
    result: dict[str, RappeGoddardQEqAtomicParameter] = {}
    for line in contents.decode("utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        fields = stripped.split()
        if len(fields) < 4:
            raise RuntimeError("Pinned QEq parameter table has a malformed row.")
        symbol = fields[0]
        if symbol not in RAPPE_GODDARD_PUBLISHED_QEQ_SYMBOLS:
            continue
        if symbol in result:
            raise RuntimeError("Pinned QEq parameter table has a duplicate row.")
        result[symbol] = RappeGoddardQEqAtomicParameter(
            symbol=symbol,
            electronegativity_ev=float(fields[1]),
            hardness_ev_per_e2=float(fields[2]),
            screening_radius_angstrom=float(fields[3]),
        )
    if set(result) != RAPPE_GODDARD_PUBLISHED_QEQ_SYMBOLS:
        raise RuntimeError("Pinned QEq parameter table is missing a published row.")
    return result


def _published_parameters(
    symbols: tuple[str, ...],
) -> tuple[RappeGoddardQEqAtomicParameter, ...]:
    if not symbols or any(not isinstance(symbol, str) for symbol in symbols):
        raise ValueError("QEq symbols must be one or more element strings.")
    unsupported = sorted(set(symbols) - RAPPE_GODDARD_PUBLISHED_QEQ_SYMBOLS)
    if unsupported:
        raise ValueError(
            "Route-2 V0-Q admits only published Rappé-Goddard QEq rows; "
            f"unsupported={unsupported}."
        )
    table = _published_parameter_table()
    return tuple(table[symbol] for symbol in symbols)


@dataclass(frozen=True)
class SameBasisQEqHardnessCurvature:
    """No-fit monopole curvature with explicit source and basis provenance."""

    symbols: tuple[str, ...]
    atom_positions_angstrom: np.ndarray
    parameters: tuple[RappeGoddardQEqAtomicParameter, ...]
    hardness_matrix_hartree_per_e2: np.ndarray
    parameter_file_sha256: str = RAPPE_GODDARD_QEQ_PARAMETER_SHA256
    construction: str = RAPPE_GODDARD_QEQ_TANGENT_CONSTRUCTION

    def __post_init__(self) -> None:
        if not self.symbols or len(self.symbols) != len(self.parameters):
            raise ValueError("QEq curvature symbols and parameters must agree.")
        positions = _immutable_array(
            self.atom_positions_angstrom,
            name="QEq curvature positions",
            shape=(len(self.symbols), 3),
        )
        matrix = _immutable_array(
            self.hardness_matrix_hartree_per_e2,
            name="QEq hardness matrix",
            shape=(len(self.symbols), len(self.symbols)),
        )
        if not np.allclose(matrix, matrix.T, rtol=0.0, atol=1.0e-14):
            raise ValueError("QEq hardness matrix must be symmetric.")
        if self.parameter_file_sha256 != RAPPE_GODDARD_QEQ_PARAMETER_SHA256:
            raise ValueError("Unexpected QEq parameter-table provenance.")
        if self.construction != RAPPE_GODDARD_QEQ_TANGENT_CONSTRUCTION:
            raise ValueError("Unsupported QEq tangent construction.")
        object.__setattr__(self, "atom_positions_angstrom", positions)
        object.__setattr__(self, "hardness_matrix_hartree_per_e2", matrix)


def build_same_basis_qeq_hardness_curvature(
    symbols: tuple[str, ...] | list[str],
    atom_positions_angstrom: np.ndarray,
) -> SameBasisQEqHardnessCurvature:
    """Build the published-hardness, MACE-GTO-metric monopole curvature.

    The QEq hardness diagonals are taken from the pinned published
    Rappé--Goddard rows.  Off-diagonal vacuum interactions use the exact
    Coulomb integral between the **same** normalized Gaussian monopoles used
    by the current MACE-POLAR density source, rather than a second atom-radius
    density representation.  For common width ``sigma`` this is

    ``erf(R / (2 sigma)) / R``

    in atomic units.  This makes the internal curvature and continuum source
    share one density coordinate system while retaining the independently
    sourced local hardness scale.
    """

    symbol_tuple = tuple(symbols)
    parameters = _published_parameters(symbol_tuple)
    positions = _immutable_array(
        atom_positions_angstrom,
        name="QEq curvature positions",
        shape=(len(symbol_tuple), 3),
    )
    sigma_bohr = MACE_POLAR_DENSITY_SIGMA_ANGSTROM / Bohr
    position_bohr = positions / Bohr
    matrix = np.diag(
        [parameter.hardness_ev_per_e2 / Hartree for parameter in parameters]
    )
    for first in range(len(symbol_tuple)):
        for second in range(first):
            distance = float(
                np.linalg.norm(position_bohr[first] - position_bohr[second])
            )
            if distance <= 1.0e-14:
                interaction = 1.0 / (math.sqrt(math.pi) * sigma_bohr)
            else:
                interaction = math.erf(distance / (2.0 * sigma_bohr)) / distance
            matrix[first, second] = interaction
            matrix[second, first] = interaction
    return SameBasisQEqHardnessCurvature(
        symbols=symbol_tuple,
        atom_positions_angstrom=positions,
        parameters=parameters,
        hardness_matrix_hartree_per_e2=matrix,
    )


def _monopole_indices(
    operator: FixedCavityGTOGalerkinOperator,
) -> np.ndarray:
    shape = operator.basis.coefficient_shape(operator.atom_count)
    return np.asarray(
        [
            np.ravel_multi_index((atom_index, 0, 0), shape)
            for atom_index in range(operator.atom_count)
        ],
        dtype=int,
    )


def _freeze_l1_induced_constraints(
    operator: FixedCavityGTOGalerkinOperator,
) -> np.ndarray:
    shape = operator.basis.coefficient_shape(operator.atom_count)
    frozen_indices = [
        np.ravel_multi_index((atom_index, 0, component), shape)
        for atom_index in range(operator.atom_count)
        for component in (1, 2, 3)
    ]
    return np.eye(operator.coefficient_count, dtype=float)[frozen_indices]


@dataclass(frozen=True)
class Route2V0QEqMonopoleState:
    """One auditable same-basis QEq-hardness V0-Q evaluation."""

    qeq_curvature: SameBasisQEqHardnessCurvature
    monopole_hardness_matrix_hartree_per_e2: np.ndarray
    response_state: Route2V0VariationalQuadraticState
    construction: str = RAPPE_GODDARD_QEQ_TANGENT_CONSTRUCTION

    def __post_init__(self) -> None:
        matrix = _immutable_array(
            self.monopole_hardness_matrix_hartree_per_e2,
            name="Monopole hardness matrix",
            shape=self.qeq_curvature.hardness_matrix_hartree_per_e2.shape,
        )
        if not np.array_equal(
            matrix,
            self.qeq_curvature.hardness_matrix_hartree_per_e2,
        ):
            raise ValueError("Monopole hardness matrix must equal QEq provenance.")
        if self.construction != RAPPE_GODDARD_QEQ_TANGENT_CONSTRUCTION:
            raise ValueError("Unsupported Route-2 V0 QEq-monopole construction.")
        if np.max(
            np.abs(self.response_state.induced_density_coefficients[:, :, 1:])
        ) > 1.0e-10:
            raise ValueError("QEq-monopole construction must freeze all l=1 response.")
        object.__setattr__(self, "monopole_hardness_matrix_hartree_per_e2", matrix)


def evaluate_route2_v0_rappe_goddard_monopole(
    *,
    symbols: tuple[str, ...] | list[str],
    frozen_density_coefficients: np.ndarray,
    operator: FixedCavityGTOGalerkinOperator,
    target_total_charge_e: float = 0.0,
    external_coefficient_dual_hartree: np.ndarray | None = None,
) -> Route2V0QEqMonopoleState:
    """Evaluate the no-training, monopole-only QEq-hardness V0-Q candidate.

    This binds a physical local-hardness reference without changing MACE
    weights or using solvation data.  It is intentionally fail-closed for any
    non-native radial basis or element lacking a published Rappé--Goddard row.
    The result remains fixed-geometry electrostatics only.
    """

    if not isinstance(operator, FixedCavityGTOGalerkinOperator):
        raise TypeError("QEq-monopole V0-Q requires a GTO Galerkin operator.")
    _require_single_native_radial_basis(operator)
    frozen = operator.basis.validate_coefficients(
        frozen_density_coefficients,
        atom_count=operator.atom_count,
        name="Frozen density coefficients",
    )
    symbol_tuple = tuple(symbols)
    if len(symbol_tuple) != operator.atom_count:
        raise ValueError("QEq symbols must match the frozen density atom count.")
    curvature = build_same_basis_qeq_hardness_curvature(
        symbol_tuple,
        operator.atom_positions_angstrom,
    )
    monopole_indices = _monopole_indices(operator)
    full_curvature = np.zeros(
        (operator.coefficient_count, operator.coefficient_count),
        dtype=float,
    )
    full_curvature[np.ix_(monopole_indices, monopole_indices)] = (
        curvature.hardness_matrix_hartree_per_e2
    )
    response_state = solve_route2_v0_variational_quadratic(
        frozen_density_coefficients=frozen,
        operator=operator,
        electronic_curvature_coefficient_dual=full_curvature,
        target_total_charge_e=target_total_charge_e,
        external_coefficient_dual_hartree=external_coefficient_dual_hartree,
        induced_constraints=_freeze_l1_induced_constraints(operator),
    )
    return Route2V0QEqMonopoleState(
        qeq_curvature=curvature,
        monopole_hardness_matrix_hartree_per_e2=(
            curvature.hardness_matrix_hartree_per_e2
        ),
        response_state=response_state,
    )


__all__ = [
    "OPENBABEL_GAUSSIAN_QEQ_REFERENCE_DOI",
    "RAPPE_GODDARD_PUBLISHED_QEQ_SYMBOLS",
    "RAPPE_GODDARD_QEQ_DOI",
    "RAPPE_GODDARD_QEQ_PARAMETER_SHA256",
    "RAPPE_GODDARD_QEQ_TANGENT_CONSTRUCTION",
    "RappeGoddardQEqAtomicParameter",
    "Route2V0QEqMonopoleState",
    "SameBasisQEqHardnessCurvature",
    "build_same_basis_qeq_hardness_curvature",
    "embed_mace_polar_l1_density_in_single_radial_gto",
    "evaluate_route2_v0_rappe_goddard_monopole",
]
