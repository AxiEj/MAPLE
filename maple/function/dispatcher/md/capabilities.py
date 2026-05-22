"""Calculator-capability gating and MD parameter admission.

Owns the calculator-metadata helpers, the stress-tensor contract check
(``validate_stress_tensor`` — a calculator-capability check, kept here with the
other capability helpers so there is no ``pressure``<->``capabilities`` import
cycle), the per-ensemble capability gate, the dataclass range validator, and the
PBC runtime-COM default policy.  Re-exported by ``...md.utils``.
"""

from typing import Optional

import numpy as np
from ase import Atoms
from ase.calculators.calculator import PropertyNotImplementedError

from maple.function.calculator._ase_unit_contract import (
    ASE_STRESS_UNIT,
    MAPLE_ENERGY_UNIT,
    MAPLE_FORCE_UNIT,
)


def _calc_model_name(calc) -> Optional[str]:
    """Return a normalized MAPLE model name attached to a calculator."""
    if calc is None:
        return None

    for attr in ("maple_model_name", "model_name", "model"):
        candidate = getattr(calc, attr, None)
        if isinstance(candidate, str):
            return candidate.lower()
    return None


def _calc_capability(calc, explicit_attr: str) -> bool:
    """Return a calculator-declared MAPLE capability, defaulting to False."""
    if calc is None:
        return False
    return bool(getattr(calc, explicit_attr, False))


def _calc_label(calc) -> str:
    """Return a stable model/calculator label for capability errors."""
    model_name = _calc_model_name(calc)
    if model_name:
        return model_name
    if calc is None:
        return "<none>"
    return calc.__class__.__name__


class MDStressUnavailableError(ValueError):
    """Raised when NPT pressure needs stress but the calculator cannot provide it."""


class MDUnitContractError(ValueError):
    """Raised when a calculator entering MD does not declare the Ha/Ha·Å contract."""


def validate_energy_force_units(atoms: Atoms) -> None:
    """Require the MAPLE MD energy/force unit contract before any ensemble runs.

    The MD layer reads ``get_potential_energy()`` as Hartree and ``get_forces()`` as
    Hartree/Å (``units.forces_au`` multiplies by ``HA_PER_ANG_TO_AU`` unconditionally).
    A raw ASE calculator returns eV / eV·Å and would be silently mis-scaled, so every
    calculator must *declare* the contract (MAPLE-native calculators inherit it from
    ``CalcABC``; a raw ASE calculator must be wrapped by ``wrap_ase_calculator`` which
    converts and then declares it).  Mirrors :func:`validate_stress_tensor`.
    """
    calc = atoms.calc
    energy_unit = getattr(calc, "maple_energy_unit", None)
    force_unit = getattr(calc, "maple_force_unit", None)
    if energy_unit is None or force_unit is None:
        raise MDUnitContractError(
            f"Calculator '{_calc_label(calc)}' does not declare the MAPLE MD unit "
            "contract (maple_energy_unit / maple_force_unit). MD consumes energy as "
            f"{MAPLE_ENERGY_UNIT!r} and forces as {MAPLE_FORCE_UNIT!r}; a raw ASE "
            "calculator (eV / eV·Å) would be silently mis-scaled. Wrap it with "
            "maple.function.calculator.wrap_ase_calculator(...) to convert and declare "
            "its units explicitly."
        )
    if energy_unit != MAPLE_ENERGY_UNIT or force_unit != MAPLE_FORCE_UNIT:
        raise MDUnitContractError(
            f"Calculator '{_calc_label(calc)}' unit contract mismatch: MD requires "
            f"energy={MAPLE_ENERGY_UNIT!r}, force={MAPLE_FORCE_UNIT!r}; got "
            f"energy={energy_unit!r}, force={force_unit!r}."
        )


def validate_stress_tensor(atoms: Atoms) -> np.ndarray:
    """Return a finite Voigt stress tensor or raise a hard NPT startup error."""
    calc = atoms.calc
    if _calc_capability(calc, "maple_stress_supported") and not hasattr(calc, "maple_stress_unit"):
        raise MDStressUnavailableError(
            "Calculator declares MAPLE stress support but does not declare "
            f"the required maple_stress_unit contract ({ASE_STRESS_UNIT!r})."
        )

    try:
        stress = atoms.get_stress(voigt=True)
    except (PropertyNotImplementedError, NotImplementedError, RuntimeError) as exc:
        raise MDStressUnavailableError(
            "Calculator does not provide a stress tensor; NPT pressure cannot be "
            "computed without real calculator stress."
        ) from exc

    stress = np.asarray(stress, dtype=float).reshape(-1)
    if stress.shape != (6,):
        raise MDStressUnavailableError(
            f"Calculator returned invalid stress shape {stress.shape}; expected Voigt length 6."
        )
    if not np.all(np.isfinite(stress)):
        raise MDStressUnavailableError("Calculator returned non-finite stress values.")
    stress_unit = getattr(calc, "maple_stress_unit", ASE_STRESS_UNIT)
    if stress_unit != ASE_STRESS_UNIT:
        raise MDStressUnavailableError(
            "Calculator stress unit contract mismatch: expected "
            f"{ASE_STRESS_UNIT!r}, got {stress_unit!r}."
        )
    return stress


def validate_md_capabilities(atoms: Atoms, ensemble: str, params=None) -> None:
    """Validate calculator capabilities needed by the requested MD ensemble.

    ``params`` carries the per-gate overrides (e.g. ``allow_unknown_cutoff``) and is
    optional: capability-only callers/tests may omit it and get the strict default.
    Must be called *after* the ensemble resolves ``params`` so the overrides reach
    the gate.
    """
    if atoms is None or atoms.calc is None:
        raise ValueError("Atoms object must have a calculator attached")

    ensemble_name = str(ensemble).lower()

    from maple.function.calculator.set_calculator import (
        validate_pbc_capabilities,
        validate_pbc_cell_geometry,
        validate_pbc_neighbor_cutoff,
    )

    # Energy/force unit contract for every ensemble (PBC or not): the MD layer
    # consumes energy/forces as Ha / Ha·Å⁻¹ and would silently mis-scale a raw
    # eV/eV·Å calculator that does not declare the contract.
    validate_energy_force_units(atoms)

    validate_pbc_capabilities(atoms, ensemble_name)
    # Shared periodic-cell geometry gate for every ensemble (NVE/NVT/NPT), so
    # the wrap/unwrap reconstruction never runs on a rank-deficient or
    # degenerate cell.  NPT's full-3-D-PBC requirement stays in NPT.__init__.
    validate_pbc_cell_geometry(atoms)

    # Minimum-image neighbor-cutoff gate on the MD admission path itself (not only
    # when MAPLE builds the calculator via SetCalculator), so a user-supplied
    # calculator is gated too.  Unknown cutoff is rejected for PBC MD unless the
    # caller opts in via params.allow_unknown_cutoff.
    validate_pbc_neighbor_cutoff(
        atoms,
        atoms.calc,
        allow_unknown_cutoff=bool(getattr(params, "allow_unknown_cutoff", False)),
        require_known_cutoff=True,
    )

    if ensemble_name != "npt" or not any(atoms.pbc):
        return

    calc = atoms.calc
    model_label = _calc_label(calc)
    if not _calc_capability(calc, "maple_stress_supported"):
        raise MDStressUnavailableError(
            "NPT requires a calculator with real stress support. "
            f"Model/calculator '{model_label}' is not declared stress capable."
        )
    validate_stress_tensor(atoms)


def validate_md_parameter_ranges(params, ensemble: str) -> None:
    """Validate common MD dataclass parameter ranges before components start.

    The dispatcher accepts user dictionaries and updates dataclasses directly.
    Centralizing the physical/numerical range checks keeps NVE/NVT/NPT startup
    failures deterministic instead of letting invalid values fail later inside
    a thermostat, barostat, logger, or trajectory writer.
    """

    ensemble_name = str(ensemble).lower()

    def finite_float(name: str) -> float:
        value = getattr(params, name)
        try:
            out = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"MD parameter '{name}' must be a finite number, got {value!r}.") from exc
        if not np.isfinite(out):
            raise ValueError(f"MD parameter '{name}' must be finite, got {value!r}.")
        return out

    def positive_float(name: str) -> float:
        value = finite_float(name)
        if value <= 0.0:
            raise ValueError(f"MD parameter '{name}' must be > 0, got {value!r}.")
        return value

    def nonnegative_float(name: str) -> float:
        value = finite_float(name)
        if value < 0.0:
            raise ValueError(f"MD parameter '{name}' must be >= 0, got {value!r}.")
        return value

    def integer_value(name: str) -> int:
        value = getattr(params, name)
        if isinstance(value, bool):
            raise ValueError(f"MD parameter '{name}' must be an integer, got {value!r}.")
        try:
            out = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"MD parameter '{name}' must be an integer, got {value!r}.") from exc
        if out != value and not (isinstance(value, float) and value.is_integer()):
            raise ValueError(f"MD parameter '{name}' must be an integer, got {value!r}.")
        return out

    def positive_int(name: str) -> int:
        value = integer_value(name)
        if value <= 0:
            raise ValueError(f"MD parameter '{name}' must be > 0, got {value!r}.")
        return value

    def nonnegative_int(name: str) -> int:
        value = integer_value(name)
        if value < 0:
            raise ValueError(f"MD parameter '{name}' must be >= 0, got {value!r}.")
        return value

    positive_float("timestep")
    nonnegative_int("steps")
    positive_int("traj_every")
    positive_int("log_every")
    nonnegative_int("rst_every")
    nonnegative_int("remove_com_every")
    nonnegative_int("remove_angular_every")
    nonnegative_int("verbose")
    if getattr(params, "random_seed", None) is not None:
        integer_value("random_seed")

    if str(getattr(params, "traj_format", "")).lower() not in {"xyz", "dcd"}:
        raise ValueError(
            f"MD parameter 'traj_format' must be 'xyz' or 'dcd', got {getattr(params, 'traj_format')!r}."
        )

    temperature = nonnegative_float("temperature")
    if ensemble_name in {"nvt", "npt"} and temperature <= 0.0:
        raise ValueError(
            f"MD parameter 'temperature' must be > 0 for {ensemble_name.upper()}, got {temperature!r}."
        )

    thermostat = str(getattr(params, "thermostat", "")).lower()
    if thermostat == "langevin":
        nonnegative_float("friction")
    elif thermostat == "v-rescale":
        positive_float("tau_t")

    if ensemble_name == "npt":
        finite_float("pressure")
        positive_float("tau_p")
        positive_float("compressibility")


def pbc_com_default_note(atoms: Atoms, params, user_set_remove_com_every: bool) -> Optional[str]:
    """Default runtime COM removal off under PBC unless the user set it (WS4b).

    Runtime COM removal under PBC subtracts total momentum during dynamics, which
    changes diffusion/VACF and other transport observables, so it must be an
    explicit opt-in rather than a silent default.  Mutates ``params`` in place and
    returns a one-line note when the default was applied (value forced to 0); the
    non-PBC default and any explicit user value are left untouched.
    """
    if not any(atoms.pbc) or user_set_remove_com_every:
        return None
    if int(getattr(params, "remove_com_every", 0) or 0) == 0:
        return None
    params.remove_com_every = 0
    return (
        "PBC default: runtime COM removal disabled (remove_com_every=0); it "
        "subtracts total momentum and perturbs diffusion/VACF under PBC. Pass "
        "remove_com_every explicitly to override."
    )
