"""Molecular normal modes and stationary-point analysis.

The public calculator boundary is ASE-native: forces are eV/Angstrom and the
Cartesian Hessian is eV/Angstrom**2.  The only admitted eigensystem is the
mass-weighted generalized eigenproblem implemented in :mod:`normal_modes`.
Minimum RRHO and first-order-saddle validation remain separate contracts.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import math
from typing import Optional

import numpy as np
from ase import Atoms

from ..jobABC import JobABC
from ..legacy_units import LegacyHartreeJobView
from .normal_modes import (
    EV_PER_ANGSTROM2_AMU_TO_WAVENUMBER_CM1,
    analyze_cartesian_hessian,
    mass_weighted_basis_to_cartesian,
)
from .reporting import (
    PrintParams,
    render_frequency_report,
    write_frequency_summary,
)
from .stationary_points import (
    StationaryPointAssessment,
    assess_stationary_point,
    normalize_stationary_point_target,
)
from .thermochemistry import (
    ThermoResults,
    compute_ideal_gas_rrho,
    rigid_mode_count,
)

from maple.function.timer import timer

_ALIASES = ("freq", "frequency", "frequency_analysis")
_IGNORED_GLOBAL_PARAMETERS = {
    "model",
    "model_options",
    "device",
    "gpuid",
    "d4",
    "pbc",
    "solv",
    "charge",
    "level",
}


def _lower_keys(values: object) -> dict:
    if not isinstance(values, dict):
        return {}
    return {
        key.lower() if isinstance(key, str) else key: value
        for key, value in values.items()
    }


def _select_subdict(parameters: object) -> dict:
    lowered = _lower_keys(parameters)
    for alias in _ALIASES:
        nested = lowered.get(alias)
        if isinstance(nested, dict):
            return _lower_keys(nested)
    return lowered


def _update_dataclass(instance: object, values: dict) -> None:
    field_names = {field.name.lower(): field.name for field in fields(instance)}
    for key, value in values.items():
        if key in field_names:
            setattr(instance, field_names[key], value)


def _positive_float(value: object, name: str) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite positive number.") from exc
    if not math.isfinite(converted) or converted <= 0.0:
        raise ValueError(f"{name} must be a finite positive number.")
    return converted


def _nonnegative_float(value: object, name: str) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite non-negative number.") from exc
    if not math.isfinite(converted) or converted < 0.0:
        raise ValueError(f"{name} must be a finite non-negative number.")
    return converted


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer.")
    try:
        converted = int(value)
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer.") from exc
    if not math.isfinite(numeric) or numeric != converted:
        raise ValueError(f"{name} must be an integer.")
    return converted


def _positive_integer(value: object, name: str) -> int:
    try:
        converted = _integer(value, name)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer.") from exc
    if converted <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return converted


def _nonnegative_integer(value: object, name: str) -> int:
    converted = _integer(value, name)
    if converted < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return converted


def _validate_molecular_atoms(atoms: Atoms) -> None:
    if np.any(atoms.get_pbc()):
        raise ValueError(
            "Molecular RRHO frequency analysis requires a non-periodic system."
        )
    if atoms.constraints:
        raise NotImplementedError(
            "Frequency analysis with ASE constraints is not implemented; "
            "a reduced-coordinate Hessian/mass contract is required."
        )


@dataclass
class FrequencyParams:
    """Validated public FREQ parameters."""

    method: str = "mw"
    temperature: float = 298.15
    pressure_kpa: float = 101.325
    symmetry_number: int = 1
    stationarity_tolerance_eV_per_A: float = 1.0e-3
    hessian_symmetry_relative_tolerance: float = 1.0e-6
    rigid_mode_tolerance_cm1: float = 5.0
    stationary_point: str = "minimum"
    transition_state_imaginary_threshold_cm1: float = 50.0
    ilowfreq: int = 0
    verbose: int = 1
    treat_imag_as_real: bool = False


class FrequencyBase(JobABC):
    """Shared ASE-unit molecular frequency workflow."""

    def __init__(
        self,
        output: str,
        atoms: Atoms,
        *,
        temperature: float = 298.15,
        pressure_kpa: float = 101.325,
        symmetry_number: int = 1,
        ilowfreq: int = 0,
        stationarity_tolerance_eV_per_A: float = 1.0e-3,
        hessian_symmetry_relative_tolerance: float = 1.0e-6,
        rigid_mode_tolerance_cm1: float = 5.0,
        stationary_point: str = "minimum",
        transition_state_imaginary_threshold_cm1: float = 50.0,
    ):
        super().__init__(output)
        _validate_molecular_atoms(atoms)
        self.atoms = atoms
        self.temperature = _positive_float(temperature, "temperature")
        self.pressure_kpa = _positive_float(pressure_kpa, "pressure_kpa")
        self.symmetry_number = _positive_integer(
            symmetry_number,
            "symmetry_number",
        )
        self.stationarity_tolerance_eV_per_A = _positive_float(
            stationarity_tolerance_eV_per_A,
            "stationarity_tolerance_eV_per_A",
        )
        self.hessian_symmetry_relative_tolerance = _nonnegative_float(
            hessian_symmetry_relative_tolerance,
            "hessian_symmetry_relative_tolerance",
        )
        self.rigid_mode_tolerance_cm1 = _nonnegative_float(
            rigid_mode_tolerance_cm1,
            "rigid_mode_tolerance_cm1",
        )
        self.stationary_point = normalize_stationary_point_target(stationary_point)
        self.transition_state_imaginary_threshold_cm1 = _positive_float(
            transition_state_imaginary_threshold_cm1,
            "transition_state_imaginary_threshold_cm1",
        )
        self.ilowfreq = _integer(ilowfreq, "ilowfreq")
        if self.ilowfreq != 0:
            raise ValueError(
                "Only standard RRHO thermochemistry (ilowfreq=0) is admitted; "
                "quasi-RRHO models need a separately validated implementation."
            )
        self.verbosity = 1
        self.treat_imag_as_real = False
        self._print = PrintParams()

    def run(self) -> None:
        self.log_info(
            [
                "Starting frequency analysis calculation, "
                f"Number of atoms: {len(self.atoms)}\n"
            ]
        )
        try:
            self._validate_stationarity()
            if self.stationary_point == "transition_state" and self.treat_imag_as_real:
                raise ValueError(
                    "stationary_point='transition_state' cannot be combined with "
                    "treat_imag_as_real; the reaction mode must remain explicit."
                )
            frequencies, modes = self.compute_frequencies(self.get_hessian())
            reinterpreted_negative_frequencies: tuple[float, ...] = ()
            reinterpretation_threshold = None
            if self.treat_imag_as_real:
                tolerance = self._print.imag_tol_cm1
                reinterpretation_mask = (frequencies < 0.0) & (
                    frequencies >= -tolerance
                )
                reinterpreted_negative_frequencies = tuple(
                    float(value) for value in frequencies[reinterpretation_mask]
                )
                reinterpretation_threshold = tolerance
                frequencies = np.where(
                    frequencies < -tolerance,
                    frequencies,
                    np.abs(frequencies),
                )
            assessment = assess_stationary_point(
                frequencies[self._rigid_mode_count() :],
                target=self.stationary_point,
                imaginary_threshold_cm1=(self.transition_state_imaginary_threshold_cm1),
                reinterpreted_negative_frequencies_cm1=(
                    reinterpreted_negative_frequencies
                ),
                reinterpretation_threshold_cm1=reinterpretation_threshold,
            )
            thermo = (
                self.compute_thermo(frequencies)
                if assessment.thermochemistry_admitted
                else None
            )
            self._write_output(frequencies, modes, thermo, assessment)
            self.log_info(
                [
                    "Frequency analysis completed\n",
                    f"Output file: {self.output}\n",
                ]
            )
            if self.verbosity == 10:
                summary_path = write_frequency_summary(
                    self.output,
                    self.atoms,
                    frequencies,
                    modes,
                    thermo,
                    assessment,
                    temperature_K=self.temperature,
                    pressure_kPa=self.pressure_kpa,
                    rigid_mode_count=self._rigid_mode_count(),
                )
                self.log_info([f"Summary file written: {summary_path}\n"])
        except Exception as exc:
            self.log_error(f"Frequency analysis failed: {exc}")
            raise

    def _validate_stationarity(self) -> None:
        try:
            forces = np.asarray(self.atoms.get_forces(), dtype=float)
        except Exception as exc:
            raise RuntimeError(
                "Frequency analysis requires calculator forces to verify the "
                "stationary geometry before Hessian evaluation."
            ) from exc
        expected_shape = (len(self.atoms), 3)
        if forces.shape != expected_shape or not np.all(np.isfinite(forces)):
            raise ValueError(
                f"stationarity forces must be finite with shape {expected_shape}."
            )
        maximum_force = float(np.max(np.abs(forces), initial=0.0))
        if maximum_force > self.stationarity_tolerance_eV_per_A:
            raise ValueError(
                "Frequency analysis requires a stationary geometry: maximum "
                f"|force|={maximum_force:.6g} eV/Angstrom exceeds "
                f"{self.stationarity_tolerance_eV_per_A:.6g} eV/Angstrom."
            )

    def get_hessian(self) -> np.ndarray:
        """Return a finite ``(3N, 3N)`` raw ASE Hessian in eV/Angstrom**2."""

        calculator = self.atoms.calc
        if calculator is None or not hasattr(calculator, "get_hessian"):
            raise RuntimeError("Atom calculator must implement get_hessian method.")
        if isinstance(calculator, LegacyHartreeJobView):
            raise RuntimeError(
                "Frequency analysis requires the raw ASE calculator Hessian in "
                "eV/Angstrom**2, not the private legacy Hartree job view."
            )
        hessian = calculator.get_hessian(self.atoms)
        if hasattr(hessian, "detach"):
            hessian = hessian.detach().cpu().numpy()
        elif hasattr(hessian, "numpy"):
            hessian = hessian.numpy()
        hessian = np.asarray(hessian, dtype=float)
        if hessian.ndim == 3 and hessian.shape[0] == 1:
            hessian = hessian[0]
        expected_shape = (3 * len(self.atoms), 3 * len(self.atoms))
        if hessian.shape != expected_shape:
            raise ValueError(
                f"Hessian shape {hessian.shape} does not match {expected_shape}."
            )
        if not np.all(np.isfinite(hessian)):
            raise ValueError("Cartesian Hessian must contain only finite values.")
        return hessian

    def compute_frequencies(
        self,
        hessian_matrix: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def compute_thermo(self, frequencies_cm1: np.ndarray) -> ThermoResults:
        return compute_ideal_gas_rrho(
            self.atoms,
            frequencies_cm1,
            temperature_K=self.temperature,
            pressure_kPa=self.pressure_kpa,
            symmetry_number=self.symmetry_number,
        )

    def _rigid_mode_count(self) -> int:
        return rigid_mode_count(self.atoms)

    def _write_output(
        self,
        frequencies_cm1: np.ndarray,
        modes_cartesian: np.ndarray,
        thermo: Optional[ThermoResults],
        assessment: StationaryPointAssessment,
    ) -> None:
        report = render_frequency_report(
            self.atoms,
            frequencies_cm1,
            modes_cartesian,
            thermo,
            assessment,
            temperature_K=self.temperature,
            pressure_kPa=self.pressure_kpa,
            rigid_mode_count=self._rigid_mode_count(),
            verbosity=self.verbosity,
            print_params=self._print,
        )
        self.log_info([report])


class MWFrequency(FrequencyBase):
    """Mass-weighted molecular vibrational analysis in public ASE units."""

    def compute_frequencies(
        self,
        hessian_matrix: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        masses = np.asarray(self.atoms.get_masses(), dtype=float)
        hessian = np.asarray(hessian_matrix, dtype=float)
        hessian_scale = float(np.max(np.abs(hessian), initial=0.0))
        symmetry_tolerance = (
            1.0e-8 + self.hessian_symmetry_relative_tolerance * hessian_scale
        )
        analysis = analyze_cartesian_hessian(
            hessian,
            masses,
            self.atoms.get_positions(),
            symmetry_tolerance_eV_per_A2=symmetry_tolerance,
        )
        rigid_basis = np.concatenate(
            (
                analysis.subspaces.translation_basis_mass_weighted,
                analysis.subspaces.rotation_basis_mass_weighted,
            ),
            axis=1,
        )
        rigid_residual = analysis.mass_weighted_hessian_eV_per_A2_amu @ rigid_basis
        residual_eigenvalue = float(np.linalg.norm(rigid_residual, ord=2))
        residual_cm1 = (
            np.sqrt(residual_eigenvalue) * EV_PER_ANGSTROM2_AMU_TO_WAVENUMBER_CM1
        )
        if residual_cm1 > self.rigid_mode_tolerance_cm1:
            raise ValueError(
                "Cartesian Hessian violates stationary rigid-body invariance: "
                f"operator residual is equivalent to {residual_cm1:.6g} cm^-1, "
                f"above {self.rigid_mode_tolerance_cm1:.6g} cm^-1."
            )

        rigid_modes = mass_weighted_basis_to_cartesian(rigid_basis, masses)
        modes = np.concatenate(
            (rigid_modes, analysis.modes_cartesian_per_sqrt_amu),
            axis=1,
        ).T
        frequencies = np.concatenate(
            (
                np.zeros(analysis.subspaces.rigid_rank, dtype=float),
                analysis.frequencies_cm1,
            )
        )
        if self.verbosity >= 1:
            tolerance = self._print.imag_tol_cm1
            imaginary_count = int(
                np.count_nonzero(analysis.frequencies_cm1 < -tolerance)
            )
            positive_count = int(np.count_nonzero(analysis.frequencies_cm1 > tolerance))
            self.log_info(
                [
                    "\nFrequency analysis summary:\n",
                    f"  Projected rigid modes:                   {analysis.subspaces.rigid_rank}\n",
                    "  Raw Hessian symmetry max defect:         "
                    f"{analysis.hessian_symmetry_max_abs_eV_per_A2:.6g} eV/Angstrom^2\n",
                    f"  Raw rigid residual equivalent:          {residual_cm1:.6g} cm^-1\n",
                    f"  Imaginary vibrations (< -{tolerance:g}):     {imaginary_count}\n",
                    f"  Positive vibrations (> {tolerance:g}):       {positive_count}\n\n",
                ]
            )
        return frequencies, modes


class Frequency:
    """Validated front driver for the single admitted molecular FREQ path."""

    def __init__(
        self,
        output: str,
        atoms: Atoms,
        params: Optional[FrequencyParams] = None,
        paras: Optional[dict] = None,
    ):
        _validate_molecular_atoms(atoms)
        if params is not None and not isinstance(params, FrequencyParams):
            raise TypeError("params must be a FrequencyParams instance.")
        self.output = output
        self.atoms = atoms
        self.params = params if params is not None else FrequencyParams()
        self.print_params = PrintParams()
        user = _select_subdict(paras)
        self._validate_parameter_names(user)
        self._apply_aliases(user)
        _update_dataclass(self.params, user)
        _update_dataclass(self.print_params, user)
        self._validate_parameters()

    @staticmethod
    def _validate_parameter_names(user: dict) -> None:
        allowed = {field.name.lower() for field in fields(FrequencyParams)}
        allowed.update(field.name.lower() for field in fields(PrintParams))
        allowed.update({"verbosity", "mode", "pressure_pa"})
        allowed.update(_IGNORED_GLOBAL_PARAMETERS)
        unknown = sorted(set(user).difference(allowed))
        if unknown:
            rendered = ", ".join(repr(name) for name in unknown)
            raise ValueError(f"Unknown FREQ parameter(s): {rendered}")

    @staticmethod
    def _apply_aliases(user: dict) -> None:
        alias_pairs = (
            ("verbosity", "verbose"),
            ("mode", "method"),
            ("pressure_pa", "pressure_kpa"),
        )
        for alias, canonical in alias_pairs:
            if alias in user and canonical in user:
                raise ValueError(f"Use either {alias} or {canonical}, not both.")
        if "verbosity" in user:
            user["verbose"] = user["verbosity"]
        if "mode" in user:
            user["method"] = user["mode"]
        if "pressure_pa" in user:
            user["pressure_kpa"] = (
                _positive_float(user["pressure_pa"], "pressure_pa") / 1000.0
            )

    def _validate_parameters(self) -> None:
        self.params.temperature = _positive_float(
            self.params.temperature,
            "temperature",
        )
        self.params.pressure_kpa = _positive_float(
            self.params.pressure_kpa,
            "pressure_kpa",
        )
        self.params.symmetry_number = _positive_integer(
            self.params.symmetry_number,
            "symmetry_number",
        )
        self.params.stationarity_tolerance_eV_per_A = _positive_float(
            self.params.stationarity_tolerance_eV_per_A,
            "stationarity_tolerance_eV_per_A",
        )
        self.params.hessian_symmetry_relative_tolerance = _nonnegative_float(
            self.params.hessian_symmetry_relative_tolerance,
            "hessian_symmetry_relative_tolerance",
        )
        self.params.rigid_mode_tolerance_cm1 = _nonnegative_float(
            self.params.rigid_mode_tolerance_cm1,
            "rigid_mode_tolerance_cm1",
        )
        self.params.stationary_point = normalize_stationary_point_target(
            self.params.stationary_point
        )
        self.params.transition_state_imaginary_threshold_cm1 = _positive_float(
            self.params.transition_state_imaginary_threshold_cm1,
            "transition_state_imaginary_threshold_cm1",
        )
        self.params.ilowfreq = _integer(self.params.ilowfreq, "ilowfreq")
        self.params.verbose = _nonnegative_integer(
            self.params.verbose,
            "verbosity",
        )
        self.print_params.n_freqs_to_print = _nonnegative_integer(
            self.print_params.n_freqs_to_print,
            "n_freqs_to_print",
        )
        self.print_params.imag_tol_cm1 = _nonnegative_float(
            self.print_params.imag_tol_cm1,
            "imag_tol_cm1",
        )
        if type(self.params.treat_imag_as_real) is not bool:
            raise ValueError("treat_imag_as_real must be true or false.")
        if (
            self.params.stationary_point == "transition_state"
            and self.params.treat_imag_as_real
        ):
            raise ValueError(
                "stationary_point='transition_state' cannot be combined with "
                "treat_imag_as_real; the reaction mode must remain explicit."
            )
        if self.params.ilowfreq != 0:
            raise ValueError(
                "Only standard RRHO thermochemistry (ilowfreq=0) is admitted; "
                "quasi-RRHO models need a separately validated implementation."
            )
        self.params.method = str(self.params.method).lower()
        if self.params.method != "mw":
            raise ValueError(
                "Frequency analysis supports only the physically mass-weighted "
                "method='mw'; nonmw/both are not scientific task modes."
            )

    def run(self) -> None:
        with timer("Frequency Calculation"):
            job = MWFrequency(
                output=self.output,
                atoms=self.atoms,
                temperature=self.params.temperature,
                pressure_kpa=self.params.pressure_kpa,
                symmetry_number=self.params.symmetry_number,
                ilowfreq=self.params.ilowfreq,
                stationarity_tolerance_eV_per_A=(
                    self.params.stationarity_tolerance_eV_per_A
                ),
                hessian_symmetry_relative_tolerance=(
                    self.params.hessian_symmetry_relative_tolerance
                ),
                rigid_mode_tolerance_cm1=self.params.rigid_mode_tolerance_cm1,
                stationary_point=self.params.stationary_point,
                transition_state_imaginary_threshold_cm1=(
                    self.params.transition_state_imaginary_threshold_cm1
                ),
            )
            job.verbosity = self.params.verbose
            job.treat_imag_as_real = self.params.treat_imag_as_real
            job._print = self.print_params
            job.run()


__all__ = [
    "Frequency",
    "FrequencyBase",
    "FrequencyParams",
    "MWFrequency",
    "PrintParams",
    "ThermoResults",
]
