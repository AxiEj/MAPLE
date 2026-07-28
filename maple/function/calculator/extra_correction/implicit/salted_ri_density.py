"""Optional SALTED RI-density to continuum-surface MEP bridge.

SALTED predicts coefficients of an atom-centred auxiliary Gaussian basis.  The
bridge evaluates that density directly on the continuum surface through
two-centre Coulomb integrals.  It intentionally does not fit atom-centred
charges or truncate the prediction to low-order multipoles.

PySCF is imported lazily and remains an optional research dependency.  SALTED
itself is not imported: a trained SALTED model may supply its coefficient
vector through a file, subprocess, or Python caller without becoming a MAPLE
runtime dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from typing import Any, Iterable, Literal, Sequence

import numpy as np

from .pyscf_runtime import require_tested_pyscf_version
from .source.frozen_density_mep import FrozenDensityMEPSource


@dataclass(frozen=True)
class SaltedRISurfaceMEP:
    """Validated surface MEP reconstructed from one SALTED RI density."""

    source: FrozenDensityMEPSource
    raw_electron_count_e: float
    electron_count_e: float
    auxiliary_basis: str
    coefficient_count: int
    charge_constraint: str
    charge_correction_coulomb_norm: float
    max_abs_coefficient_correction: float
    pyscf_version: str

    @property
    def surface_potential_hartree_per_e(self) -> np.ndarray:
        return self.source.surface_potential_hartree_per_e.copy()

    @property
    def observed_total_charge_e(self) -> float:
        return self.source.observed_total_charge_e


@dataclass(frozen=True)
class RIChargeProjection:
    """Coulomb-metric minimum correction enforcing one electron count."""

    coefficients: np.ndarray
    raw_electron_count_e: float
    projected_electron_count_e: float
    target_electron_count_e: float
    correction_coulomb_norm: float
    max_abs_coefficient_correction: float


def project_ri_coefficients_to_electron_count(
    coefficients: np.ndarray,
    *,
    integrated_basis: np.ndarray,
    coulomb_metric: np.ndarray,
    target_electron_count_e: float,
) -> RIChargeProjection:
    """Project RI coefficients onto an exact electron-count hyperplane.

    The correction minimizes ``delta.T @ J @ delta`` under the constraint
    ``integrated_basis.T @ (coefficients + delta) == target``.  This keeps the
    correction explicit and reproducible instead of relaxing the molecular
    charge tolerance.
    """

    values = np.asarray(coefficients, dtype=float)
    integrals = np.asarray(integrated_basis, dtype=float)
    metric = np.asarray(coulomb_metric, dtype=float)
    target = float(target_electron_count_e)
    if (
        values.ndim != 1
        or values.size == 0
        or not np.all(np.isfinite(values))
    ):
        raise ValueError("RI charge projection requires finite coefficients.")
    if integrals.shape != values.shape or not np.all(np.isfinite(integrals)):
        raise ValueError(
            "RI charge projection basis integrals must match the coefficients."
        )
    if (
        metric.shape != (values.size, values.size)
        or not np.all(np.isfinite(metric))
    ):
        raise ValueError(
            "RI charge projection Coulomb metric has an incompatible shape."
        )
    if not np.isfinite(target):
        raise ValueError("RI target electron count must be finite.")
    if not np.allclose(metric, metric.T, rtol=0.0, atol=1.0e-11):
        raise ValueError("RI charge projection Coulomb metric must be symmetric.")

    raw_count = float(np.dot(integrals, values))
    direction = np.linalg.solve(metric, integrals)
    denominator = float(np.dot(integrals, direction))
    if not np.isfinite(denominator) or denominator <= 0.0:
        raise ValueError(
            "RI charge projection Coulomb metric is not positive on the "
            "electron-count constraint."
        )
    correction = direction * ((target - raw_count) / denominator)
    projected = values + correction
    projected_count = float(np.dot(integrals, projected))
    correction_norm_squared = float(correction @ metric @ correction)
    if correction_norm_squared < -1.0e-12:
        raise RuntimeError(
            "RI charge projection produced a negative Coulomb norm."
        )
    immutable = np.array(projected, dtype=float, copy=True)
    immutable.setflags(write=False)
    return RIChargeProjection(
        coefficients=immutable,
        raw_electron_count_e=raw_count,
        projected_electron_count_e=projected_count,
        target_electron_count_e=target,
        correction_coulomb_norm=float(
            np.sqrt(max(correction_norm_squared, 0.0))
        ),
        max_abs_coefficient_correction=float(
            np.max(np.abs(correction), initial=0.0)
        ),
    )


def _validated_l1_slices(
    l1_slices: Iterable[slice],
    *,
    coefficient_count: int,
) -> tuple[slice, ...]:
    validated: list[slice] = []
    occupied: set[int] = set()
    for block in l1_slices:
        if (
            not isinstance(block, slice)
            or block.step not in (None, 1)
            or block.start is None
            or block.stop is None
            or block.start < 0
            or block.stop - block.start != 3
            or block.stop > coefficient_count
        ):
            raise ValueError(
                "Each SALTED l=1 block must be one in-range three-element slice."
            )
        indexes = set(range(block.start, block.stop))
        if occupied.intersection(indexes):
            raise ValueError("SALTED l=1 coefficient slices must not overlap.")
        occupied.update(indexes)
        validated.append(slice(block.start, block.stop))
    return tuple(validated)


def pyscf_coefficients_to_salted_order(
    coefficients: np.ndarray,
    *,
    l1_slices: Iterable[slice],
) -> np.ndarray:
    """Convert PySCF ``(+1,-1,0)`` p blocks to SALTED ``(-1,0,+1)``."""

    source = np.asarray(coefficients, dtype=float)
    if source.ndim != 1 or source.size == 0 or not np.all(np.isfinite(source)):
        raise ValueError(
            "RI density coefficients must be one finite non-empty vector."
        )
    blocks = _validated_l1_slices(
        l1_slices,
        coefficient_count=source.size,
    )
    result = np.array(source, dtype=float, copy=True)
    for block in blocks:
        values = np.array(source[block], copy=True)
        result[block] = values[[1, 2, 0]]
    return result


def salted_coefficients_to_pyscf_order(
    coefficients: np.ndarray,
    *,
    l1_slices: Iterable[slice],
) -> np.ndarray:
    """Convert SALTED ``(-1,0,+1)`` p blocks to PySCF ``(+1,-1,0)``."""

    source = np.asarray(coefficients, dtype=float)
    if source.ndim != 1 or source.size == 0 or not np.all(np.isfinite(source)):
        raise ValueError(
            "SALTED RI density coefficients must be one finite non-empty vector."
        )
    blocks = _validated_l1_slices(
        l1_slices,
        coefficient_count=source.size,
    )
    result = np.array(source, dtype=float, copy=True)
    for block in blocks:
        values = np.array(source[block], copy=True)
        result[block] = values[[2, 0, 1]]
    return result


def _pyscf_l1_slices(auxiliary_molecule: Any) -> tuple[slice, ...]:
    ao_locations = np.asarray(auxiliary_molecule.ao_loc_nr(), dtype=int)
    if ao_locations.shape != (int(auxiliary_molecule.nbas) + 1,):
        raise RuntimeError("PySCF auxiliary AO locations have an unexpected shape.")
    blocks: list[slice] = []
    for shell in range(int(auxiliary_molecule.nbas)):
        angular_momentum = int(auxiliary_molecule.bas_angular(shell))
        contraction_count = int(auxiliary_molecule.bas_nctr(shell))
        component_count = 2 * angular_momentum + 1
        start = int(ao_locations[shell])
        stop = int(ao_locations[shell + 1])
        if stop - start != component_count * contraction_count:
            raise RuntimeError(
                "PySCF auxiliary shell layout is incompatible with SALTED "
                "real-spherical coefficient blocks."
            )
        if angular_momentum == 1:
            for contraction in range(contraction_count):
                offset = start + 3 * contraction
                blocks.append(slice(offset, offset + 3))
    return tuple(blocks)


def _validated_symbols(symbols: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(str(symbol) for symbol in symbols)
    if not normalized or any(not symbol for symbol in normalized):
        raise ValueError("SALTED RI density requires non-empty atomic symbols.")
    return normalized


def salted_ri_surface_mep(
    *,
    symbols: Sequence[str],
    atom_positions_angstrom: np.ndarray,
    surface_points_bohr: np.ndarray,
    salted_coefficients: np.ndarray,
    auxiliary_basis: str,
    source_model: str,
    declared_total_charge_e: float,
    charge_tolerance_e: float = 1.0e-4,
    charge_constraint: Literal["strict", "coulomb-metric"] = "strict",
    point_chunk_size: int = 400,
) -> SaltedRISurfaceMEP:
    """Evaluate a SALTED auxiliary-basis density on one PCM surface."""

    if not auxiliary_basis:
        raise ValueError("SALTED RI density requires a PySCF auxiliary basis.")
    if not source_model:
        raise ValueError("SALTED RI density requires a trained-model identity.")
    if charge_constraint not in {"strict", "coulomb-metric"}:
        raise ValueError(
            "SALTED RI charge constraint must be 'strict' or 'coulomb-metric'."
        )
    if point_chunk_size <= 0:
        raise ValueError("SALTED RI surface point chunk size must be positive.")

    normalized_symbols = _validated_symbols(symbols)
    positions = np.asarray(atom_positions_angstrom, dtype=float)
    points = np.asarray(surface_points_bohr, dtype=float)
    coefficients = np.asarray(salted_coefficients, dtype=float)
    if (
        positions.shape != (len(normalized_symbols), 3)
        or not np.all(np.isfinite(positions))
    ):
        raise ValueError(
            "SALTED atom positions must be finite with shape (n_atoms, 3)."
        )
    if (
        points.ndim != 2
        or points.shape[0] == 0
        or points.shape[1] != 3
        or not np.all(np.isfinite(points))
    ):
        raise ValueError(
            "SALTED continuum points must be finite with shape (n_surface, 3)."
        )

    try:
        pyscf = importlib.import_module("pyscf")
        gto = importlib.import_module("pyscf.gto")
        ft_ao = importlib.import_module("pyscf.gto.ft_ao")
    except ImportError as exc:
        raise ImportError(
            "The SALTED RI-density bridge requires optional PySCF 2.13.1."
        ) from exc
    version = require_tested_pyscf_version(
        pyscf.__version__,
        feature="SALTED RI-density surface-MEP bridge",
    )

    auxiliary = gto.M(
        atom=list(zip(normalized_symbols, positions, strict=True)),
        basis=auxiliary_basis,
        unit="Angstrom",
        charge=int(round(float(declared_total_charge_e))),
        spin=0,
        verbose=0,
    )
    coefficient_count = int(auxiliary.nao_nr())
    if (
        coefficients.shape != (coefficient_count,)
        or not np.all(np.isfinite(coefficients))
    ):
        raise ValueError(
            "SALTED coefficient vector does not match the PySCF auxiliary "
            f"basis (expected {(coefficient_count,)}, "
            f"received {coefficients.shape})."
        )
    pyscf_coefficients = salted_coefficients_to_pyscf_order(
        coefficients,
        l1_slices=_pyscf_l1_slices(auxiliary),
    )

    zero_wavevector = np.zeros((1, 3), dtype=float)
    integrated_basis = np.asarray(
        ft_ao.ft_ao(auxiliary, zero_wavevector)[0]
    )
    if float(np.max(np.abs(np.imag(integrated_basis)), initial=0.0)) > 1.0e-10:
        raise RuntimeError(
            "PySCF auxiliary-basis zero-wavevector integral is unexpectedly complex."
        )
    integrated_basis = np.real(integrated_basis)
    nuclear_charges = np.asarray(auxiliary.atom_charges(), dtype=float)
    target_electron_count = float(
        np.sum(nuclear_charges) - float(declared_total_charge_e)
    )
    raw_electron_count = float(
        np.dot(integrated_basis, pyscf_coefficients)
    )
    correction_norm = 0.0
    max_abs_correction = 0.0
    if charge_constraint == "coulomb-metric":
        projection = project_ri_coefficients_to_electron_count(
            pyscf_coefficients,
            integrated_basis=integrated_basis,
            coulomb_metric=np.asarray(
                auxiliary.intor(auxiliary._add_suffix("int2c2e")),
                dtype=float,
            ),
            target_electron_count_e=target_electron_count,
        )
        pyscf_coefficients = projection.coefficients
        electron_count = projection.projected_electron_count_e
        correction_norm = projection.correction_coulomb_norm
        max_abs_correction = projection.max_abs_coefficient_correction
    else:
        electron_count = raw_electron_count
    coordinates_bohr = np.asarray(
        auxiliary.atom_coords(unit="Bohr"),
        dtype=float,
    )
    observed_total_charge = float(np.sum(nuclear_charges) - electron_count)

    nuclear_potential = np.zeros(points.shape[0], dtype=float)
    for charge, center in zip(
        nuclear_charges,
        coordinates_bohr,
        strict=True,
    ):
        distances = np.linalg.norm(points - center, axis=1)
        if np.any(distances <= 1.0e-12):
            raise ValueError("A continuum surface point coincides with a nucleus.")
        nuclear_potential += charge / distances

    electronic_potential = np.empty(points.shape[0], dtype=float)
    intor = auxiliary._add_suffix("int2c2e")
    for start in range(0, points.shape[0], point_chunk_size):
        stop = min(start + point_chunk_size, points.shape[0])
        fake_points = gto.fakemol_for_charges(points[start:stop])
        kernel = np.asarray(
            gto.mole.intor_cross(
                intor,
                auxiliary,
                fake_points,
            ),
            dtype=float,
        )
        if kernel.shape != (coefficient_count, stop - start):
            raise RuntimeError(
                "PySCF returned an unexpected auxiliary-density Coulomb kernel."
            )
        electronic_potential[start:stop] = pyscf_coefficients @ kernel

    source = FrozenDensityMEPSource(
        surface_points_bohr=points,
        surface_potential_hartree_per_e=(
            nuclear_potential - electronic_potential
        ),
        declared_total_charge_e=declared_total_charge_e,
        observed_total_charge_e=observed_total_charge,
        source_model=source_model,
        density_representation=(
            f"salted-ri-gto:{auxiliary_basis}:{charge_constraint}"
        ),
        charge_tolerance_e=charge_tolerance_e,
    )
    return SaltedRISurfaceMEP(
        source=source,
        raw_electron_count_e=raw_electron_count,
        electron_count_e=electron_count,
        auxiliary_basis=auxiliary_basis,
        coefficient_count=coefficient_count,
        charge_constraint=charge_constraint,
        charge_correction_coulomb_norm=correction_norm,
        max_abs_coefficient_correction=max_abs_correction,
        pyscf_version=version,
    )


__all__ = [
    "SaltedRISurfaceMEP",
    "RIChargeProjection",
    "project_ri_coefficients_to_electron_count",
    "pyscf_coefficients_to_salted_order",
    "salted_coefficients_to_pyscf_order",
    "salted_ri_surface_mep",
]
