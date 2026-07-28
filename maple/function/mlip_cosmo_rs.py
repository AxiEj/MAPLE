"""Experimental MLIP-to-COSMO-RS surface bridge.

The production COSMO-RS parameterization is trained on conductor surfaces from
BP86/def2-TZVPD ORCA calculations.  This module deliberately does *not* claim
parameterization equivalence.  It provides a small, audited research boundary
for asking a narrower question: can an AIMNet2 or MACE-POLAR electrostatic
source generate the conductor screening-charge segments consumed by
openCOSMO-RS without a solute quantum-chemistry calculation?

Only the solute-side conductor profile is replaced.  A precomputed solvent
profile may still come from the parameterized openCOSMO-RS solvent library.
Any resulting free energy is therefore an out-of-parameterization diagnostic,
not an openCOSMO-RS 24a accuracy claim.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

import numpy as np
from ase.units import Bohr

from .calculator.extra_correction.implicit.gto_density import (
    point_multipole_potential,
)


MLIP_COSMORS_BRIDGE_CONTRACT_VERSION = 1
OPEN_COSMORS_MINIMUM_SEGMENT_AREA_ANGSTROM2 = 0.01

# Radius values read from the ORCA 6/openCOSMO-RS 24a conductor assets used by
# the frozen ten-record MNSol panel.  The deliberately narrow element domain
# prevents silently applying unverified defaults.
OPEN_COSMORS_24A_AUDITED_RADII_ANGSTROM = {
    "H": 1.30,
    "C": 2.00,
    "N": 1.83,
    "O": 1.72,
    "S": 2.16,
    "Cl": 2.05,
    "Br": 2.16,
}

# Exact nine-decimal radius tokens observed in the audited ORCA 6 conductor
# assets.  openCOSMO-RS 24a compares parsed radii with exact floating-point
# equality across every component, so recomputing these tokens with a newer
# physical-constant table can spuriously reject an otherwise identical radius
# set.
OPEN_COSMORS_24A_AUDITED_RADII_BOHR = {
    "H": 2.456643974,
    "C": 3.779452268,
    "N": 3.458198825,
    "O": 3.250328950,
    "S": 4.081808449,
    "Cl": 3.873938575,
    "Br": 4.081808449,
}


def open_cosmors_24a_cavity_radii(
    symbols: Sequence[str],
) -> np.ndarray:
    """Return the audited ORCA/openCOSMO-RS radii for a bounded element set."""

    normalized = tuple(str(symbol) for symbol in symbols)
    if not normalized:
        raise ValueError("MLIP COSMO-RS requires at least one atom.")
    unsupported = sorted(
        {
            symbol
            for symbol in normalized
            if symbol not in OPEN_COSMORS_24A_AUDITED_RADII_ANGSTROM
        }
    )
    if unsupported:
        raise ValueError(
            "No audited openCOSMO-RS 24a cavity radius is available for: "
            + ", ".join(unsupported)
        )
    return np.asarray(
        [OPEN_COSMORS_24A_AUDITED_RADII_ANGSTROM[symbol] for symbol in normalized],
        dtype=float,
    )


def _immutable_vector(
    values: Any,
    *,
    name: str,
    length: int | None = None,
    positive: bool = False,
) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    expected = None if length is None else (length,)
    if (
        array.ndim != 1
        or array.size == 0
        or (expected is not None and array.shape != expected)
        or not np.all(np.isfinite(array))
        or (positive and np.any(array <= 0.0))
    ):
        shape = "(n,)" if expected is None else str(expected)
        qualifier = " finite positive" if positive else " finite"
        raise ValueError(f"{name} must be{qualifier} with shape {shape}.")
    result = np.array(array, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _immutable_matrix(
    values: Any,
    *,
    name: str,
    rows: int | None = None,
    columns: int,
) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    valid_rows = array.shape[0] > 0 and (rows is None or array.shape[0] == rows)
    if (
        array.ndim != 2
        or array.shape[1:] != (columns,)
        or not valid_rows
        or not np.all(np.isfinite(array))
    ):
        row_label = "n" if rows is None else str(rows)
        raise ValueError(
            f"{name} must be finite with shape ({row_label}, {columns})."
        )
    result = np.array(array, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _surface_volume_bohr3(
    points_bohr: np.ndarray,
    areas_bohr2: np.ndarray,
    parent_atom_indices: np.ndarray,
    atom_positions_angstrom: np.ndarray,
    cavity_radii_angstrom: np.ndarray,
) -> float:
    """Approximate a closed atom-centred SWIG volume by the divergence theorem."""

    atom_positions_bohr = atom_positions_angstrom / Bohr
    radii_bohr = cavity_radii_angstrom / Bohr
    relative = points_bohr - atom_positions_bohr[parent_atom_indices]
    normals = relative / radii_bohr[parent_atom_indices, None]
    normal_norms = np.linalg.norm(normals, axis=1)
    if np.any(normal_norms <= 0.0) or not np.all(np.isfinite(normal_norms)):
        raise ValueError("MLIP COSMO-RS surface normals are invalid.")
    normals = normals / normal_norms[:, None]
    origin = np.mean(atom_positions_bohr, axis=0)
    volume = float(
        np.dot(
            areas_bohr2,
            np.einsum("ij,ij->i", points_bohr - origin, normals),
        )
        / 3.0
    )
    if not math.isfinite(volume) or volume <= 0.0:
        raise ValueError("MLIP COSMO-RS cavity volume must be finite and positive.")
    return volume


@dataclass(frozen=True)
class MLIPCOSMORSSurface:
    """One fixed-source conductor surface consumable by openCOSMO-RS."""

    source_id: str
    symbols: tuple[str, ...]
    atom_positions_angstrom: np.ndarray
    cavity_radii_angstrom: np.ndarray
    surface_points_bohr: np.ndarray
    surface_areas_bohr2: np.ndarray
    surface_parent_atom_indices: np.ndarray
    surface_potential_hartree_per_e: np.ndarray
    surface_charge_e: np.ndarray
    gas_energy_hartree: float
    conductor_energy_hartree: float
    polarization_energy_hartree: float
    cavity_volume_bohr3: float
    discarded_segment_count: int
    pre_correction_surface_charge_e: float

    def __post_init__(self) -> None:
        symbols = tuple(str(symbol) for symbol in self.symbols)
        atom_count = len(symbols)
        if atom_count == 0:
            raise ValueError("MLIP COSMO-RS surface requires at least one atom.")
        positions = _immutable_matrix(
            self.atom_positions_angstrom,
            name="atom_positions_angstrom",
            rows=atom_count,
            columns=3,
        )
        radii = _immutable_vector(
            self.cavity_radii_angstrom,
            name="cavity_radii_angstrom",
            length=atom_count,
            positive=True,
        )
        points = _immutable_matrix(
            self.surface_points_bohr,
            name="surface_points_bohr",
            columns=3,
        )
        areas = _immutable_vector(
            self.surface_areas_bohr2,
            name="surface_areas_bohr2",
            length=points.shape[0],
            positive=True,
        )
        parents = np.asarray(self.surface_parent_atom_indices, dtype=int)
        if (
            parents.shape != (points.shape[0],)
            or np.any(parents < 0)
            or np.any(parents >= atom_count)
        ):
            raise ValueError(
                "surface_parent_atom_indices must map every segment to one atom."
            )
        parents = np.array(parents, dtype=int, copy=True)
        parents.setflags(write=False)
        potential = _immutable_vector(
            self.surface_potential_hartree_per_e,
            name="surface_potential_hartree_per_e",
            length=points.shape[0],
        )
        charge = _immutable_vector(
            self.surface_charge_e,
            name="surface_charge_e",
            length=points.shape[0],
        )
        finite_scalars = (
            float(self.gas_energy_hartree),
            float(self.conductor_energy_hartree),
            float(self.polarization_energy_hartree),
            float(self.cavity_volume_bohr3),
            float(self.pre_correction_surface_charge_e),
        )
        if not all(math.isfinite(value) for value in finite_scalars):
            raise ValueError("MLIP COSMO-RS scalar values must be finite.")
        if self.cavity_volume_bohr3 <= 0.0:
            raise ValueError("MLIP COSMO-RS cavity volume must be positive.")
        if int(self.discarded_segment_count) < 0:
            raise ValueError("discarded_segment_count cannot be negative.")
        if not math.isclose(
            self.conductor_energy_hartree - self.gas_energy_hartree,
            self.polarization_energy_hartree,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError(
                "Conductor minus gas energy must equal the MLIP polarization energy."
            )
        if abs(float(np.sum(charge))) > 1.0e-10:
            raise ValueError(
                "The neutral MLIP COSMO-RS screening surface must have zero net charge."
            )

        object.__setattr__(self, "source_id", str(self.source_id))
        object.__setattr__(self, "symbols", symbols)
        object.__setattr__(self, "atom_positions_angstrom", positions)
        object.__setattr__(self, "cavity_radii_angstrom", radii)
        object.__setattr__(self, "surface_points_bohr", points)
        object.__setattr__(self, "surface_areas_bohr2", areas)
        object.__setattr__(self, "surface_parent_atom_indices", parents)
        object.__setattr__(self, "surface_potential_hartree_per_e", potential)
        object.__setattr__(self, "surface_charge_e", charge)

    @classmethod
    def from_fixed_multipoles(
        cls,
        response: Any,
        symbols: Sequence[str],
        atom_positions_angstrom: np.ndarray,
        density_coefficients: np.ndarray,
        *,
        source_id: str,
        gas_energy_hartree: float = 0.0,
        minimum_segment_area_angstrom2: float = (
            OPEN_COSMORS_MINIMUM_SEGMENT_AREA_ANGSTROM2
        ),
    ) -> "MLIPCOSMORSSurface":
        """Solve one neutral fixed ML source against a conductor boundary."""

        symbol_tuple = tuple(str(symbol) for symbol in symbols)
        positions = np.asarray(atom_positions_angstrom, dtype=float)
        coefficients = np.asarray(density_coefficients, dtype=float)
        atom_count = len(symbol_tuple)
        if (
            positions.shape != (atom_count, 3)
            or coefficients.shape != (atom_count, 4)
            or not np.all(np.isfinite(positions))
            or not np.all(np.isfinite(coefficients))
        ):
            raise ValueError(
                "MLIP COSMO-RS positions and l<=1 coefficients must have "
                "shapes (n_atoms, 3) and (n_atoms, 4)."
            )
        if abs(float(np.sum(coefficients[:, 0]))) > 1.0e-8:
            raise ValueError(
                "The current MLIP COSMO-RS bridge accepts neutral sources."
            )
        provenance = dict(response.runtime_provenance)
        if provenance.get("conductor_limit") is not True:
            raise ValueError(
                "MLIP COSMO-RS requires an explicitly identified conductor response."
            )
        radii = np.asarray(response.cavity_radii_angstrom, dtype=float)
        audited_radii = open_cosmors_24a_cavity_radii(symbol_tuple)
        if not np.allclose(radii, audited_radii, rtol=0.0, atol=1.0e-12):
            raise ValueError(
                "MLIP COSMO-RS requires the audited openCOSMO-RS 24a "
                "cavity radii."
            )
        points = np.asarray(response.surface_points_bohr, dtype=float)
        areas = np.asarray(response.surface_areas_bohr2, dtype=float)
        parents = np.asarray(response.surface_parent_atom_indices, dtype=int)
        surface_potential = point_multipole_potential(
            points,
            positions,
            coefficients,
        )
        state = response.solve(surface_potential)
        direct_charge = np.asarray(state.direct_surface_charge_e, dtype=float)
        conjugate_charge = np.asarray(
            state.energy_conjugate_surface_charge_e,
            dtype=float,
        )
        if not np.allclose(
            direct_charge,
            conjugate_charge,
            rtol=2.0e-11,
            atol=2.0e-13,
        ):
            raise RuntimeError(
                "The conductor surface response must be reciprocal before "
                "constructing a COSMO-RS profile."
            )

        minimum_area = float(minimum_segment_area_angstrom2)
        if not math.isfinite(minimum_area) or minimum_area <= 0.0:
            raise ValueError("minimum_segment_area_angstrom2 must be positive.")
        keep = areas * Bohr**2 >= minimum_area
        if not np.any(keep):
            raise RuntimeError("The segment-area filter removed the complete surface.")
        filtered_points = points[keep]
        filtered_areas = areas[keep]
        filtered_parents = parents[keep]
        filtered_potential = surface_potential[keep]
        filtered_charge = direct_charge[keep].copy()
        pre_correction_charge = float(np.sum(filtered_charge))
        filtered_charge -= (
            pre_correction_charge * filtered_areas / float(np.sum(filtered_areas))
        )
        if abs(float(np.sum(filtered_charge))) > 1.0e-12:
            raise RuntimeError("Neutral conductor charge correction did not close.")

        polarization_energy = 0.5 * float(
            np.dot(filtered_potential, filtered_charge)
        )
        gas_energy = float(gas_energy_hartree)
        volume = _surface_volume_bohr3(
            filtered_points,
            filtered_areas,
            filtered_parents,
            positions,
            radii,
        )
        return cls(
            source_id=source_id,
            symbols=symbol_tuple,
            atom_positions_angstrom=positions,
            cavity_radii_angstrom=radii,
            surface_points_bohr=filtered_points,
            surface_areas_bohr2=filtered_areas,
            surface_parent_atom_indices=filtered_parents,
            surface_potential_hartree_per_e=filtered_potential,
            surface_charge_e=filtered_charge,
            gas_energy_hartree=gas_energy,
            conductor_energy_hartree=gas_energy + polarization_energy,
            polarization_energy_hartree=polarization_energy,
            cavity_volume_bohr3=volume,
            discarded_segment_count=int(np.count_nonzero(~keep)),
            pre_correction_surface_charge_e=pre_correction_charge,
        )

    @property
    def cavity_area_bohr2(self) -> float:
        return float(np.sum(self.surface_areas_bohr2))

    def as_manifest(self) -> dict[str, object]:
        return {
            "contract_version": MLIP_COSMORS_BRIDGE_CONTRACT_VERSION,
            "scientific_role": "experimental-mlip-solute-cosmors-profile",
            "source_id": self.source_id,
            "qm_solute_calculation_required": False,
            "strict_open_cosmors_24a_parameterization_equivalence": False,
            "atom_count": len(self.symbols),
            "surface_segment_count": int(self.surface_charge_e.size),
            "discarded_segment_count": int(self.discarded_segment_count),
            "minimum_segment_area_angstrom2": (
                OPEN_COSMORS_MINIMUM_SEGMENT_AREA_ANGSTROM2
            ),
            "pre_correction_surface_charge_e": (
                self.pre_correction_surface_charge_e
            ),
            "post_correction_surface_charge_e": float(
                np.sum(self.surface_charge_e)
            ),
            "polarization_energy_hartree": self.polarization_energy_hartree,
            "cavity_area_bohr2": self.cavity_area_bohr2,
            "cavity_volume_bohr3": self.cavity_volume_bohr3,
            "claim_boundary": (
                "The solute screening profile is generated from a fixed MLIP "
                "l<=1 source on a PySCF SWIG conductor surface. openCOSMO-RS "
                "24a was fitted to ORCA BP86/def2-TZVPD profiles, so this "
                "bridge is an out-of-parameterization diagnostic."
            ),
        }


def render_mlip_orcacosmo(
    profile: MLIPCOSMORSSurface,
    *,
    name: str,
) -> str:
    """Render the minimal ORCA-COSMO text consumed by openCOSMO-RS."""

    safe_name = str(name).strip()
    if not safe_name or any(character in safe_name for character in "\r\n:"):
        raise ValueError("MLIP COSMO-RS profile name must be one safe line.")
    lines = [
        f"{safe_name} : MLIP_{profile.source_id}_CPCM_CONDUCTOR",
        "",
        "##################################################",
        "#ENERGY",
        (
            "FINAL SINGLE POINT ENERGY      "
            f"{profile.conductor_energy_hartree:.15f}"
        ),
        "",
        "##################################################",
        "#XYZ_FILE",
        str(len(profile.symbols)),
        f"Coordinates from MAPLE MLIP source {profile.source_id}",
    ]
    for symbol, position in zip(
        profile.symbols,
        profile.atom_positions_angstrom,
        strict=True,
    ):
        lines.append(
            f"{symbol:>3s} {position[0]:20.14f} {position[1]:20.14f} "
            f"{position[2]:20.14f}"
        )
    lines.extend(
        [
            "",
            "##################################################",
            "#COSMO",
            f"{len(profile.symbols)} # Number of atoms",
            f"{profile.surface_charge_e.size} # Number of surface points",
            f"{profile.cavity_volume_bohr3:.12f} # Volume",
            f"{profile.cavity_area_bohr2:.12f} # Area",
            (
                f"{profile.polarization_energy_hartree:.15f} "
                "# CPCM dielectric energy"
            ),
            "",
            "#------------------------------------------------------------",
            "# CARTESIAN COORDINATES (A.U.) + RADII (A.U.) + ATOMIC NUMBER",
            "#------------------------------------------------------------",
        ]
    )
    positions_bohr = profile.atom_positions_angstrom / Bohr
    for symbol, position in zip(
        profile.symbols,
        positions_bohr,
        strict=True,
    ):
        atomic_number = _atomic_number(symbol)
        radius = OPEN_COSMORS_24A_AUDITED_RADII_BOHR[symbol]
        lines.append(
            f"{position[0]:18.12f} {position[1]:18.12f} "
            f"{position[2]:18.12f} {radius:18.9f} {atomic_number:d}"
        )
    lines.extend(
        [
            "",
            "#------------------------------------------------------------",
            "# SURFACE POINTS (A.U.)    (Hint - charge NOT scaled by FEps)",
            "#------------------------------------------------------------",
            (
                "          X                 Y                 Z"
                "               area            potential          charge"
                "            w_leb             Switch_F          G_width       atom"
            ),
        ]
    )
    for point, area, potential, charge, parent in zip(
        profile.surface_points_bohr,
        profile.surface_areas_bohr2,
        profile.surface_potential_hartree_per_e,
        profile.surface_charge_e,
        profile.surface_parent_atom_indices,
        strict=True,
    ):
        lines.append(
            f"{point[0]:18.12f} {point[1]:18.12f} {point[2]:18.12f} "
            f"{area:18.12f} {potential:18.12f} {charge:18.12f} "
            f"{0.0:18.12f} {1.0:18.12f} {1.0:18.12f} {int(parent):6d}"
        )
    lines.extend(
        [
            "",
            "##################################################",
            "#COSMO_corrected",
            (
                "Corrected dielectric energy   =     "
                f"{profile.polarization_energy_hartree:.15f}"
            ),
            (
                "Total C-PCM charge            =      "
                f"{float(np.sum(profile.surface_charge_e)):.15f}"
            ),
            "C-PCM corrected charges:",
        ]
    )
    lines.extend(f"{charge:20.15f}" for charge in profile.surface_charge_e)
    lines.extend(["##################################################", ""])
    return "\n".join(lines)


def _atomic_number(symbol: str) -> int:
    numbers = {
        "H": 1,
        "C": 6,
        "N": 7,
        "O": 8,
        "S": 16,
        "Cl": 17,
        "Br": 35,
    }
    try:
        return numbers[symbol]
    except KeyError as exc:
        raise ValueError(
            f"No audited MLIP COSMO-RS atomic number mapping for {symbol!r}."
        ) from exc


__all__ = [
    "MLIP_COSMORS_BRIDGE_CONTRACT_VERSION",
    "MLIPCOSMORSSurface",
    "OPEN_COSMORS_24A_AUDITED_RADII_ANGSTROM",
    "OPEN_COSMORS_24A_AUDITED_RADII_BOHR",
    "OPEN_COSMORS_MINIMUM_SEGMENT_AREA_ANGSTROM2",
    "open_cosmors_24a_cavity_radii",
    "render_mlip_orcacosmo",
]
