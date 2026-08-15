"""Formatting for molecular frequencies and RRHO corrections."""

from __future__ import annotations

from dataclasses import dataclass
import os

import numpy as np
from ase import Atoms

from .thermochemistry import ThermoResults

KJ_PER_MOL_TO_KCAL_PER_MOL = 0.23900573614


@dataclass
class PrintParams:
    """Display-only frequency settings."""

    n_freqs_to_print: int = 10
    imag_tol_cm1: float = 10.0


def _kjmol_to_kcalmol(value: float) -> float:
    return value * KJ_PER_MOL_TO_KCAL_PER_MOL


def _reported_mode_count(
    total_mode_count: int,
    rigid_mode_count: int,
    verbosity: int,
    requested_vibrations: int,
) -> int:
    if verbosity == 0:
        vibration_count = 0
    elif verbosity == 1:
        vibration_count = requested_vibrations
    else:
        vibration_count = total_mode_count - rigid_mode_count
    return min(rigid_mode_count + vibration_count, total_mode_count)


def _format_frequencies(
    frequencies_cm1: np.ndarray,
    *,
    rigid_mode_count: int,
    verbosity: int,
    requested_vibrations: int,
) -> str:
    lines = [
        "-" * 60 + "\n",
        "VIBRATIONAL FREQUENCIES\n",
        "-" * 60 + "\n\n",
        "Scaling factor for frequencies = 1.000000000 (already applied!)\n\n",
    ]
    count = _reported_mode_count(
        len(frequencies_cm1),
        rigid_mode_count,
        verbosity,
        requested_vibrations,
    )
    for index, frequency in enumerate(frequencies_cm1[:count]):
        if index < rigid_mode_count:
            displayed = 0.0
            tag = ""
        else:
            displayed = float(frequency)
            tag = "  ***imaginary mode***" if frequency < 0.0 else ""
        lines.append(f"{index:6d}:  {displayed:10.2f} cm**-1{tag}\n")
    lines.append("\n\n")
    return "".join(lines)


def _format_thermochemistry(
    atoms: Atoms,
    thermo: ThermoResults,
    *,
    temperature_K: float,
    pressure_kPa: float,
) -> str:
    zpe = _kjmol_to_kcalmol(thermo.zpe_kjmol)
    h_trans = _kjmol_to_kcalmol(thermo.h_trans_kjmol)
    h_rot = _kjmol_to_kcalmol(thermo.h_rot_kjmol)
    h_vib = _kjmol_to_kcalmol(thermo.h_vib_thermal_kjmol)
    h_total = _kjmol_to_kcalmol(thermo.h_total_kjmol)
    g_correction = _kjmol_to_kcalmol(thermo.g_correction_kjmol)
    entropy_factor = 1.0e-3 * KJ_PER_MOL_TO_KCAL_PER_MOL
    s_trans = thermo.s_trans_jmolK * entropy_factor
    s_rot = thermo.s_rot_jmolK * entropy_factor
    s_vib = thermo.s_vib_jmolK * entropy_factor
    s_elec = thermo.s_elec_jmolK * entropy_factor
    s_total = s_trans + s_rot + s_vib + s_elec
    rule = "-" * 60 + "\n"
    return "".join(
        [
            "\n",
            rule,
            f"THERMOCHEMISTRY AT {temperature_K:.2f}K\n",
            rule,
            "\n",
            f"Temperature         ...   {temperature_K:.2f} K\n",
            f"Pressure            ...   {pressure_kPa / 101.325:.2f} atm\n",
            f"Total Mass          ...   {np.sum(atoms.get_masses()):.2f} AMU\n\n",
            rule,
            "ENTHALPY CORRECTION\n",
            rule,
            "\n",
            f"Zero point energy                ...   {zpe:10.2f} kcal/mol\n",
            f"Thermal vibrational correction   ...   {h_vib:10.2f} kcal/mol\n",
            f"Thermal rotational correction    ...   {h_rot:10.2f} kcal/mol\n",
            f"Thermal translational correction ...   {h_trans:10.2f} kcal/mol\n",
            rule,
            f"Total enthalpy correction        ...   {h_total:10.2f} kcal/mol\n\n",
            rule,
            "ENTROPY\n",
            rule,
            "\n",
            f"Translational entropy            ...   {s_trans:10.6f} kcal/(mol*K)\n",
            f"Rotational entropy               ...   {s_rot:10.6f} kcal/(mol*K)\n",
            f"Vibrational entropy              ...   {s_vib:10.6f} kcal/(mol*K)\n",
            f"Electronic entropy               ...   {s_elec:10.6f} kcal/(mol*K)\n",
            rule,
            f"Total entropy                    ...   {s_total:10.6f} kcal/(mol*K)\n\n",
            rule,
            "GIBBS FREE ENERGY\n",
            rule,
            "\n",
            f"Total enthalpy correction        ...   {h_total:10.2f} kcal/mol\n",
            f"Total entropy correction         ...   {-temperature_K * s_total:10.2f} kcal/mol\n",
            rule,
            f"Final Gibbs free energy corr.    ...   {g_correction:10.2f} kcal/mol\n\n",
        ]
    )


def _format_normal_modes(
    frequencies_cm1: np.ndarray,
    modes_cartesian: np.ndarray,
    *,
    rigid_mode_count: int,
    verbosity: int,
    requested_vibrations: int,
) -> str:
    if verbosity <= 0:
        return ""
    count = _reported_mode_count(
        len(frequencies_cm1),
        rigid_mode_count,
        verbosity,
        requested_vibrations,
    )
    displayed_modes = np.array(modes_cartesian, copy=True)
    displayed_modes[:rigid_mode_count] = 0.0
    lines = [
        "\n",
        "-" * 60 + "\n",
        "NORMAL MODES\n",
        "-" * 60 + "\n\n",
        "These are Cartesian displacement vectors obtained from mass-weighted modes.\n",
        "They are orthonormal in the atomic-mass metric, not the Euclidean metric.\n\n",
    ]
    for block_start in range(0, count, 6):
        block_end = min(block_start + 6, count)
        lines.append(
            "       "
            + "".join(
                f"{mode_index:11d}    " for mode_index in range(block_start, block_end)
            )
            + "\n"
        )
        for coordinate_index in range(modes_cartesian.shape[1]):
            values = "".join(
                f"{displayed_modes[mode_index, coordinate_index]:13.6f}  "
                for mode_index in range(block_start, block_end)
            )
            lines.append(f"{coordinate_index:6d}  {values}\n")
        lines.append("\n")
    return "".join(lines)


def render_frequency_report(
    atoms: Atoms,
    frequencies_cm1: object,
    modes_cartesian: object,
    thermo: ThermoResults,
    *,
    temperature_K: float,
    pressure_kPa: float,
    rigid_mode_count: int,
    verbosity: int,
    print_params: PrintParams,
) -> str:
    """Render the append-only main FREQ report."""

    frequencies = np.asarray(frequencies_cm1, dtype=float)
    modes = np.asarray(modes_cartesian, dtype=float)
    expected_shape = (frequencies.size, 3 * len(atoms))
    if modes.shape != expected_shape:
        raise ValueError(f"modes_cartesian must have shape {expected_shape}.")
    return "".join(
        [
            _format_frequencies(
                frequencies,
                rigid_mode_count=rigid_mode_count,
                verbosity=verbosity,
                requested_vibrations=print_params.n_freqs_to_print,
            ),
            _format_thermochemistry(
                atoms,
                thermo,
                temperature_K=temperature_K,
                pressure_kPa=pressure_kPa,
            ),
            _format_normal_modes(
                frequencies,
                modes,
                rigid_mode_count=rigid_mode_count,
                verbosity=verbosity,
                requested_vibrations=print_params.n_freqs_to_print,
            ),
        ]
    )


def write_frequency_summary(
    output_path: str,
    atoms: Atoms,
    frequencies_cm1: object,
    modes_cartesian: object,
    thermo: ThermoResults,
    *,
    temperature_K: float,
    pressure_kPa: float,
    rigid_mode_count: int,
) -> str:
    """Write the verbosity-10 summary and return its path."""

    frequencies = np.asarray(frequencies_cm1, dtype=float)[rigid_mode_count:]
    modes = np.asarray(modes_cartesian, dtype=float)[rigid_mode_count:]
    symbols = atoms.get_chemical_symbols()
    positions = atoms.get_positions()
    atom_count = len(atoms)
    base, _ = os.path.splitext(output_path)
    summary_path = f"{base}.sum"

    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write("=" * 70 + "\nTHERMODYNAMIC SUMMARY\n" + "=" * 70 + "\n\n")
        handle.write(f"Temperature:            {temperature_K:.2f} K\n")
        handle.write(
            f"Pressure:               {pressure_kPa:.3f} kPa "
            f"({pressure_kPa / 101.325:.3f} atm)\n"
        )
        handle.write(f"Total Mass:             {np.sum(atoms.get_masses()):.4f} amu\n")
        handle.write(f"Number of Atoms:        {atom_count}\n")
        handle.write(f"Number of Vib. Modes:   {len(frequencies)}\n\n")
        handle.write("-" * 40 + "\nKey Thermodynamic Values\n" + "-" * 40 + "\n")
        handle.write(
            f"ZPE:                    {thermo.zpe_kjmol:12.4f} kJ/mol  "
            f"({_kjmol_to_kcalmol(thermo.zpe_kjmol):10.4f} kcal/mol)\n"
        )
        handle.write(
            f"H_corr (total):         {thermo.h_total_kjmol:12.4f} kJ/mol  "
            f"({_kjmol_to_kcalmol(thermo.h_total_kjmol):10.4f} kcal/mol)\n"
        )
        handle.write(
            f"G_corr (total):         {thermo.g_correction_kjmol:12.4f} kJ/mol  "
            f"({_kjmol_to_kcalmol(thermo.g_correction_kjmol):10.4f} kcal/mol)\n"
        )
        handle.write(
            f"S_total:                {thermo.s_total_jmolK:12.4f} J/mol/K\n\n"
        )
        handle.write("-" * 40 + "\nEnthalpy Contributions (kJ/mol)\n" + "-" * 40 + "\n")
        handle.write(f"  H_trans:              {thermo.h_trans_kjmol:12.4f}\n")
        handle.write(f"  H_rot:                {thermo.h_rot_kjmol:12.4f}\n")
        handle.write(f"  H_vib (thermal):      {thermo.h_vib_thermal_kjmol:12.4f}\n")
        handle.write(f"  ZPE:                  {thermo.zpe_kjmol:12.4f}\n\n")
        handle.write("-" * 40 + "\nEntropy Contributions (J/mol/K)\n" + "-" * 40 + "\n")
        handle.write(f"  S_trans:              {thermo.s_trans_jmolK:12.4f}\n")
        handle.write(f"  S_rot:                {thermo.s_rot_jmolK:12.4f}\n")
        handle.write(f"  S_vib:                {thermo.s_vib_jmolK:12.4f}\n")
        handle.write(f"  S_elec:               {thermo.s_elec_jmolK:12.4f}\n\n")
        handle.write(
            "=" * 70 + "\nVIBRATIONAL FREQUENCIES (cm^-1)\n" + "=" * 70 + "\n\n"
        )
        handle.write(
            f"{'Mode':>6}  {'Frequency':>12}  {'Type':<15}\n" + "-" * 40 + "\n"
        )
        for index, frequency in enumerate(frequencies, start=1):
            mode_type = "imaginary" if frequency < 0.0 else "real"
            handle.write(f"{index:>6}  {frequency:>12.2f}  {mode_type:<15}\n")

        handle.write(
            "\n"
            + "=" * 70
            + "\nNORMAL MODES (XYZ Trajectory Format)\n"
            + "=" * 70
            + "\n"
        )
        handle.write(
            "# Equilibrium coordinates followed by a Cartesian mode vector.\n\n"
        )
        for index, (frequency, mode) in enumerate(zip(frequencies, modes), start=1):
            handle.write(f"{atom_count}\nMode {index}: {frequency:.2f} cm^-1\n")
            for symbol, position, displacement in zip(
                symbols,
                positions,
                mode.reshape(atom_count, 3),
            ):
                handle.write(
                    f"{symbol:2s}  "
                    + "  ".join(
                        f"{value:12.6f}" for value in (*position, *displacement)
                    )
                    + "\n"
                )
        handle.write("\n# End of normal modes trajectory\n")

    return summary_path


__all__ = [
    "PrintParams",
    "render_frequency_report",
    "write_frequency_summary",
]
