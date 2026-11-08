"""Immutable, per-frame single-point results for same-identity trajectories."""

from __future__ import annotations

from dataclasses import dataclass, fields
import hashlib
import json
from collections.abc import Mapping
from typing import Any, Iterable

import numpy as np

from maple.solvation.models.base import model_charge_and_multiplicity

RAW_TOTAL_PES_FORCE_SCOPE = "raw_cartesian_total_pes_force_negative_energy_gradient"


def _is_sha256(value: object) -> bool:
    text = str(value)
    return len(text) == 64 and all(
        character in "0123456789abcdef" for character in text
    )


def _cartesian_rows(values: np.ndarray) -> tuple[tuple[float, float, float], ...]:
    return tuple((float(row[0]), float(row[1]), float(row[2])) for row in values)


def _raw_calculator(atoms: Any) -> Any:
    calculator = getattr(atoms, "calc", None)
    return getattr(calculator, "raw_calculator", calculator)


def _domain_identity(calculator: Any) -> str:
    explicit = getattr(calculator, "domain_identity", None)
    if explicit is not None:
        value = explicit() if callable(explicit) else explicit
        return str(value)
    payload = {
        "calculator_type": (
            f"{type(calculator).__module__}.{type(calculator).__qualname__}"
        ),
        "configured_symbols": list(getattr(calculator, "_configured_symbols", ())),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class SPFrameContext:
    """Geometry and calculator identity frozen before any frame evaluation."""

    frame_index: int
    symbols: tuple[str, ...]
    positions_angstrom: tuple[tuple[float, float, float], ...]
    masses_amu: tuple[float, ...]
    charge: int
    multiplicity: int
    model_input_sha256: str
    configuration_sha256: str
    profile: str
    scalar_contract_id: str
    actual_device: str
    domain_identity: str


@dataclass(frozen=True, slots=True)
class SPFrameResult:
    """One authoritative raw-ASE-unit result detached from calculator caches."""

    frame_index: int
    energy_eV: float
    forces_eV_per_A: tuple[tuple[float, float, float], ...] | None
    force_scope: str | None
    symbols: tuple[str, ...]
    positions_angstrom: tuple[tuple[float, float, float], ...]
    masses_amu: tuple[float, ...]
    charge: int
    multiplicity: int
    model_input_sha256: str
    configuration_sha256: str
    profile: str
    scalar_contract_id: str
    actual_device: str
    domain_identity: str

    def to_record(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "energy_eV": self.energy_eV,
            "forces_eV_per_A": (
                None
                if self.forces_eV_per_A is None
                else [list(row) for row in self.forces_eV_per_A]
            ),
            "force_scope": self.force_scope,
            "symbols": list(self.symbols),
            "positions_angstrom": [list(row) for row in self.positions_angstrom],
            "masses_amu": list(self.masses_amu),
            "charge": self.charge,
            "multiplicity": self.multiplicity,
            "model_input_sha256": self.model_input_sha256,
            "configuration_sha256": self.configuration_sha256,
            "profile": self.profile,
            "scalar_contract_id": self.scalar_contract_id,
            "actual_device": self.actual_device,
            "domain_identity": self.domain_identity,
        }


@dataclass(frozen=True, slots=True)
class SPBatchResult:
    """Ordered same-identity frame results owned by one trajectory job."""

    frames: tuple[SPFrameResult, ...]

    def __post_init__(self) -> None:
        if not self.frames:
            raise ValueError("A single-point trajectory requires at least one frame.")
        if tuple(frame.frame_index for frame in self.frames) != tuple(
            range(len(self.frames))
        ):
            raise ValueError(
                "Single-point frame indices must be ordered and contiguous."
            )

    def to_records(self) -> list[dict[str, Any]]:
        return [frame.to_record() for frame in self.frames]


@dataclass(frozen=True, slots=True)
class SPRequest:
    """Normalized V4 request whose mode, not container, owns result shape."""

    mode: str
    frames: tuple[Any, ...]

    @property
    def returns_batch(self) -> bool:
        return self.mode == "trajectory"


def sp_request_frames(atoms: Any) -> tuple[Any, ...]:
    if hasattr(atoms, "multiatoms"):
        return tuple(atoms.multiatoms)
    if isinstance(atoms, (list, tuple)):
        return tuple(atoms)
    return (atoms,)


def normalize_v4_sp_request(atoms: Any, params: Mapping[str, Any]) -> SPRequest:
    """Validate the complete V4 SP request before any physical evaluation."""

    if isinstance(atoms, tuple):
        raise ValueError(
            "V4 SP tuple inputs are unsupported; use Atoms, list[Atoms], "
            "or the existing Molecules input helper."
        )
    raw_mode = params.get("mode", "single")
    if not isinstance(raw_mode, str):
        raise ValueError("V4 SP mode must be 'single' or 'trajectory'.")
    mode = raw_mode.strip().lower()
    if mode not in {"single", "trajectory"}:
        raise ValueError(
            "V4 SP mode must be 'single' or 'trajectory'; heterogeneous "
            "collections remain closed."
        )
    frames = sp_request_frames(atoms)
    if not frames or any(frame is None for frame in frames):
        raise ValueError("V4 SP requires at least one molecular frame.")
    if mode == "single" and len(frames) != 1:
        raise ValueError("V4 SP mode=single requires exactly one frame.")
    return SPRequest(mode=mode, frames=frames)


def frame_context(atoms: Any, frame_index: int) -> SPFrameContext:
    """Capture one canonical non-PES frame context without an energy call."""

    calculator = _raw_calculator(atoms)
    validate = getattr(calculator, "_validate_atoms", None)
    if not callable(validate):
        raise ValueError(
            "V4 single-point frames require canonical identity validation."
        )
    validated = validate(atoms)
    if not isinstance(validated, tuple) or len(validated) != 5:
        raise ValueError("V4 single-point calculator identity is invalid.")
    model_input, configuration, profile, scalar, device = validated
    if not _is_sha256(model_input) or not _is_sha256(configuration):
        raise ValueError(
            "V4 single-point model-input and configuration identities must be SHA256."
        )
    if any(not str(value) for value in (profile, scalar, device)):
        raise ValueError("V4 single-point profile/scalar/device identity is invalid.")
    positions = np.asarray(atoms.get_positions(), dtype=np.float64)
    masses = np.asarray(atoms.get_masses(), dtype=np.float64)
    if positions.shape != (len(atoms), 3) or not np.isfinite(positions).all():
        raise ValueError("V4 single-point frame geometry must be finite N-by-3.")
    if (
        masses.shape != (len(atoms),)
        or not np.isfinite(masses).all()
        or np.any(masses <= 0.0)
    ):
        raise ValueError("V4 single-point frame masses must be finite and positive.")
    charge, multiplicity = model_charge_and_multiplicity(atoms)
    return SPFrameContext(
        frame_index=frame_index,
        symbols=tuple(atoms.get_chemical_symbols()),
        positions_angstrom=_cartesian_rows(positions),
        masses_amu=tuple(float(x) for x in masses),
        charge=charge,
        multiplicity=multiplicity,
        model_input_sha256=str(model_input),
        configuration_sha256=str(configuration),
        profile=str(profile),
        scalar_contract_id=str(scalar),
        actual_device=str(device),
        domain_identity=_domain_identity(calculator),
    )


def require_unchanged_frame_context(atoms: Any, expected: SPFrameContext) -> None:
    """Fail closed if a frame changed after the all-frame preflight."""

    if frame_context(atoms, expected.frame_index) != expected:
        raise RuntimeError(
            f"V4 SP frame {expected.frame_index} identity/geometry drifted "
            "after trajectory preflight."
        )


def preflight_same_identity_frames(frames: Iterable[Any]) -> tuple[SPFrameContext, ...]:
    """Validate every V4 trajectory frame before the first physical call."""

    images = tuple(frames)
    if not images:
        raise ValueError("A single-point trajectory requires at least one frame.")
    contexts = [frame_context(atoms, index) for index, atoms in enumerate(images)]
    first = contexts[0]
    identity_fields = (
        "symbols",
        "masses_amu",
        "charge",
        "multiplicity",
        "configuration_sha256",
        "profile",
        "scalar_contract_id",
        "actual_device",
        "domain_identity",
    )
    for context in contexts[1:]:
        mismatches = [
            name
            for name in identity_fields
            if getattr(context, name) != getattr(first, name)
        ]
        if mismatches:
            raise ValueError(
                "V4 trajectory frames must share one physical/configuration "
                "identity; mismatched " + ", ".join(mismatches) + "."
            )
    return tuple(contexts)


def capture_raw_frame_result(
    atoms: Any,
    context: SPFrameContext,
    *,
    forces_requested: bool,
) -> SPFrameResult:
    """Copy canonical raw eV E/F immediately, before another frame advances."""

    results = getattr(_raw_calculator(atoms), "results", {}) or {}
    if "energy" not in results:
        raise RuntimeError("Single-point calculator did not retain raw energy_eV.")
    energy = float(results["energy"])
    if not np.isfinite(energy):
        raise RuntimeError("Single-point raw energy_eV is nonfinite.")
    forces_record = None
    force_scope = None
    if forces_requested:
        forces = np.array(results.get("forces"), dtype=np.float64, copy=True)
        if forces.shape != (len(atoms), 3) or not np.isfinite(forces).all():
            raise RuntimeError("Single-point raw forces must be finite N-by-3.")
        forces_record = _cartesian_rows(forces)
        force_scope = RAW_TOTAL_PES_FORCE_SCOPE
    context_values = {
        item.name: getattr(context, item.name) for item in fields(context)
    }
    return SPFrameResult(
        **context_values,
        energy_eV=energy,
        forces_eV_per_A=forces_record,
        force_scope=force_scope,
    )
