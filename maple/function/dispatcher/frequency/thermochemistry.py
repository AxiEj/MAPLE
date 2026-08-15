"""Ideal-gas rigid-rotor/harmonic-oscillator thermochemistry.

The normal-mode driver supplies an exact rigid-body prefix followed by the
physical vibrational spectrum.  This module owns only the molecular RRHO
partition functions and reports energy corrections in kJ/mol and entropies in
J/(mol K).  Empirical quasi-RRHO models and transition-state partition
functions are deliberately outside this contract.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from ase import Atoms

from .normal_modes import rigid_body_subspaces

PLANCK_J_S = 6.62607015e-34
BOLTZMANN_J_K = 1.380649e-23
LIGHT_M_S = 2.99792458e8
AVOGADRO_MOL_INV = 6.02214076e23
CENTIMETER_M = 1.0e-2
GAS_CONSTANT_J_MOL_K = BOLTZMANN_J_K * AVOGADRO_MOL_INV
ATOMIC_MASS_UNIT_KG = 1.66053906660e-27
ANGSTROM_SQUARED_M2 = 1.0e-20


@dataclass
class ThermoResults:
    """Gas-phase RRHO correction components."""

    zpe_kjmol: float
    h_trans_kjmol: float
    h_rot_kjmol: float
    h_vib_thermal_kjmol: float
    s_trans_jmolK: float = 0.0
    s_rot_jmolK: float = 0.0
    s_vib_jmolK: float = 0.0
    s_elec_jmolK: float = 0.0
    g_correction_kjmol: float = 0.0

    @property
    def h_total_kjmol(self) -> float:
        return (
            self.h_trans_kjmol
            + self.h_rot_kjmol
            + self.zpe_kjmol
            + self.h_vib_thermal_kjmol
        )

    @property
    def s_total_jmolK(self) -> float:
        return (
            self.s_trans_jmolK + self.s_rot_jmolK + self.s_vib_jmolK + self.s_elec_jmolK
        )


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer.")
    try:
        converted = int(value)
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer.") from exc
    if not math.isfinite(numeric) or numeric != converted or converted <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return converted


def rigid_mode_count(atoms: Atoms) -> int:
    """Return the mass-metric translation/rotation rank for ``atoms``."""

    return rigid_body_subspaces(
        atoms.get_masses(),
        atoms.get_positions(),
    ).rigid_rank


def geometry_kind(atoms: Atoms) -> str:
    """Return ASE's monatomic/linear/nonlinear RRHO geometry class."""

    if len(atoms) == 1:
        return "monatomic"
    return (
        "linear"
        if rigid_body_subspaces(
            atoms.get_masses(),
            atoms.get_positions(),
        ).is_linear
        else "nonlinear"
    )


def _translational_rotational_entropy(
    atoms: Atoms,
    *,
    temperature_K: float,
    pressure_kPa: float,
    symmetry_number: int,
) -> tuple[float, float]:
    pressure_Pa = pressure_kPa * 1000.0
    total_mass_kg = float(np.sum(atoms.get_masses())) * ATOMIC_MASS_UNIT_KG
    principal_moments = (
        np.asarray(atoms.get_moments_of_inertia(vectors=False), dtype=float)
        * ATOMIC_MASS_UNIT_KG
        * ANGSTROM_SQUARED_M2
    )

    translation_partition = (
        (2.0 * np.pi * total_mass_kg * BOLTZMANN_J_K * temperature_K) / PLANCK_J_S**2
    ) ** 1.5 * (BOLTZMANN_J_K * temperature_K / pressure_Pa)
    translation_entropy = GAS_CONSTANT_J_MOL_K * (np.log(translation_partition) + 2.5)

    geometry = geometry_kind(atoms)
    if geometry == "monatomic":
        rotation_entropy = 0.0
    elif geometry == "linear":
        moment = float(np.max(principal_moments))
        rotation_partition = (
            8.0
            * np.pi**2
            * moment
            * BOLTZMANN_J_K
            * temperature_K
            / (symmetry_number * PLANCK_J_S**2)
        )
        rotation_entropy = GAS_CONSTANT_J_MOL_K * (np.log(rotation_partition) + 1.0)
    else:
        rotation_partition = (
            np.sqrt(np.pi)
            / symmetry_number
            * (8.0 * np.pi**2 * BOLTZMANN_J_K * temperature_K / PLANCK_J_S**2) ** 1.5
            * np.sqrt(np.prod(principal_moments))
        )
        rotation_entropy = GAS_CONSTANT_J_MOL_K * (np.log(rotation_partition) + 1.5)

    return float(translation_entropy), float(rotation_entropy)


def compute_ideal_gas_rrho(
    atoms: Atoms,
    frequencies_cm1: object,
    *,
    temperature_K: float,
    pressure_kPa: float,
    symmetry_number: int,
) -> ThermoResults:
    """Compute a minimum's ideal-gas RRHO thermochemical correction.

    The input is the complete ``3N`` spectrum produced by MAPLE's normal-mode
    kernel: exact projected rigid-body zeros first, followed by vibrational
    modes.  All vibrational modes must be strictly positive.  Consequently a
    transition state or a zero-frequency floppy mode fails closed instead of
    being silently omitted.
    """

    temperature = float(temperature_K)
    pressure = float(pressure_kPa)
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature_K must be finite and positive.")
    if not math.isfinite(pressure) or pressure <= 0.0:
        raise ValueError("pressure_kPa must be finite and positive.")
    symmetry = _positive_integer(symmetry_number, "symmetry_number")

    frequencies = np.asarray(frequencies_cm1, dtype=float)
    expected_mode_count = 3 * len(atoms)
    if frequencies.shape != (expected_mode_count,) or not np.all(
        np.isfinite(frequencies)
    ):
        raise ValueError(
            "frequencies_cm1 must be finite with one entry per Cartesian "
            "degree of freedom."
        )

    rigid_rank = rigid_mode_count(atoms)
    if np.any(np.abs(frequencies[:rigid_rank]) > 1.0e-8):
        raise ValueError(
            "RRHO input must begin with the exact projected rigid-body zeros."
        )
    vibrations = frequencies[rigid_rank:]
    if np.any(vibrations <= 0.0):
        raise ValueError(
            "RRHO thermochemistry requires strictly positive vibrational "
            "frequencies at a minimum; TS thermochemistry needs a separate "
            "contract."
        )

    zpe_jmol = (
        0.5
        * PLANCK_J_S
        * LIGHT_M_S
        * AVOGADRO_MOL_INV
        * float(np.sum(vibrations))
        / CENTIMETER_M
    )
    characteristic_temperature = (
        PLANCK_J_S * LIGHT_M_S * (vibrations / CENTIMETER_M) / BOLTZMANN_J_K
    )
    reduced_frequency = characteristic_temperature / temperature

    bose_term = np.zeros_like(reduced_frequency)
    finite_exponential = reduced_frequency <= 700.0
    bose_term[finite_exponential] = reduced_frequency[finite_exponential] / np.expm1(
        reduced_frequency[finite_exponential]
    )
    vibrational_entropy = GAS_CONSTANT_J_MOL_K * np.sum(
        bose_term - np.log1p(-np.exp(-reduced_frequency))
    )
    vibrational_thermal_energy = GAS_CONSTANT_J_MOL_K * temperature * np.sum(bose_term)

    kind = geometry_kind(atoms)
    rotational_dof = {"monatomic": 0, "linear": 2, "nonlinear": 3}[kind]
    translation_entropy, rotation_entropy = _translational_rotational_entropy(
        atoms,
        temperature_K=temperature,
        pressure_kPa=pressure,
        symmetry_number=symmetry,
    )
    multiplicity = _positive_integer(
        atoms.info.get("mult", 1),
        "atoms.info['mult']",
    )

    result = ThermoResults(
        zpe_kjmol=float(zpe_jmol * 1.0e-3),
        h_trans_kjmol=float(2.5 * GAS_CONSTANT_J_MOL_K * temperature * 1.0e-3),
        h_rot_kjmol=float(
            0.5 * rotational_dof * GAS_CONSTANT_J_MOL_K * temperature * 1.0e-3
        ),
        h_vib_thermal_kjmol=float(vibrational_thermal_energy * 1.0e-3),
        s_trans_jmolK=translation_entropy,
        s_rot_jmolK=rotation_entropy,
        s_vib_jmolK=float(vibrational_entropy),
        s_elec_jmolK=float(GAS_CONSTANT_J_MOL_K * np.log(multiplicity)),
    )
    result.g_correction_kjmol = (
        result.h_total_kjmol - temperature * result.s_total_jmolK * 1.0e-3
    )
    return result


__all__ = [
    "ThermoResults",
    "compute_ideal_gas_rrho",
    "geometry_kind",
    "rigid_mode_count",
]
