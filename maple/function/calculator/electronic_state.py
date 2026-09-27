from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

IDENTITY_SCHEMA_VERSION = 2
_CALCULATOR_IDENTITY_ATTRIBUTE = "maple_pes_identity"


def solvation_identity_settings(implicit: str, solvent: str) -> dict[str, str]:
    method = str(implicit).strip().lower()
    if method in {"", "none"}:
        return {"implicit": "none"}
    return {"implicit": method, "solvent": str(solvent).strip().lower()}


def checkpoint_fingerprint(path: str | Path) -> dict[str, str]:
    """Return a content identity for the checkpoint bytes read at construction."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "algorithm": "sha256",
        "digest": digest.hexdigest(),
        "source": "loaded_checkpoint",
    }


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError("PES identity values must be finite.")
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_value(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    raise TypeError(f"PES identity value is not JSON-compatible: {type(value).__name__}")


def attach_calculator_identity(
    calculator,
    *,
    backend: str,
    implementation_version: str,
    checkpoint_path: str | Path | None = None,
    model_fingerprint: dict[str, str] | None = None,
    relevant_settings: dict[str, Any] | None = None,
) -> None:
    """Attach the immutable identity of weights and scientific adapter code."""
    if (checkpoint_path is None) == (model_fingerprint is None):
        raise ValueError("Provide exactly one checkpoint path or loaded fingerprint.")
    fingerprint = (
        checkpoint_fingerprint(checkpoint_path)
        if checkpoint_path is not None else _json_value(model_fingerprint)
    )
    if not isinstance(implementation_version, str) or not implementation_version.strip():
        raise ValueError("implementation_version must be a non-empty stable identifier.")
    version = implementation_version.strip()
    setattr(
        calculator,
        _CALCULATOR_IDENTITY_ATTRIBUTE,
        {
            "backend": str(backend).strip().lower(),
            "implementation_version": version,
            "model_fingerprint": fingerprint,
            "relevant_settings": _json_value(relevant_settings or {}),
        },
    )


def _calculator_identity(calculator) -> dict[str, Any]:
    if calculator is None:
        raise ValueError("A PES identity requires an attached calculator.")

    provider = getattr(calculator, "get_pes_identity", None)
    if callable(provider):
        identity = provider()
    else:
        identity = getattr(calculator, _CALCULATOR_IDENTITY_ATTRIBUTE, None)
    if not isinstance(identity, dict):
        raise TypeError(
            f"Calculator {type(calculator).__name__} has no durable PES identity. "
            "External calculators must provide get_pes_identity() or maple_pes_identity."
        )

    required = {
        "backend",
        "implementation_version",
        "model_fingerprint",
        "relevant_settings",
    }
    missing = sorted(required - identity.keys())
    if missing:
        raise ValueError(f"Calculator PES identity is missing: {', '.join(missing)}")
    normalized = _json_value(identity)
    if not isinstance(normalized["implementation_version"], str) or not normalized[
        "implementation_version"
    ].strip():
        raise ValueError(
            "Calculator PES identity requires a non-empty implementation_version."
        )
    fingerprint = normalized["model_fingerprint"]
    if (
        not isinstance(fingerprint, dict)
        or not fingerprint.get("algorithm")
        or not fingerprint.get("digest")
        or not fingerprint.get("source")
    ):
        raise ValueError("Calculator PES identity requires a content fingerprint.")
    return normalized


def _integer_state(atoms, key: str, default: int) -> int:
    value = atoms.info.get(key, default)
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"atoms.info['{key}'] must be an integer, not a boolean.")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"atoms.info['{key}'] must be an integer; got {value!r}.") from exc
    if not np.isfinite(numeric) or not numeric.is_integer():
        raise ValueError(f"atoms.info['{key}'] must be an integer; got {value!r}.")
    return int(numeric)


def requested_electronic_state(atoms) -> tuple[int, int]:
    """Resolve the explicit state without guessing from partial atomic charges."""
    multiplicity = _integer_state(atoms, "mult", 1)
    if multiplicity < 1:
        raise ValueError("atoms.info['mult'] must be at least 1.")
    charge_value = atoms.info.get("charge")
    if charge_value is None and hasattr(atoms, "get_initial_charges"):
        initial_charges = np.asarray(atoms.get_initial_charges(), dtype=float)
        if (
            initial_charges.size
            and np.all(np.isfinite(initial_charges))
            and not np.isclose(initial_charges.sum(), 0.0, rtol=0.0, atol=1e-12)
        ):
            raise ValueError(
                "Nonzero initial charges make the electronic state ambiguous; "
                "set atoms.info['charge'] explicitly for path or restart use."
            )
    charge = _integer_state(atoms, "charge", 0) if charge_value is not None else 0
    return charge, multiplicity


def validate_electronic_state(atoms, backend, *, model_options=None) -> tuple[int, int]:
    """Reject a requested state that the energy backend cannot represent."""
    charge, multiplicity = requested_electronic_state(atoms)
    periodic = bool(np.any(np.asarray(atoms.get_pbc(), dtype=bool)))
    if periodic:
        if (charge != 0 or multiplicity != 1) and not getattr(
            backend, "SUPPORTS_PERIODIC_CHARGE_MULT", False
        ):
            name = backend.__name__ if isinstance(backend, type) else type(backend).__name__
            raise ValueError(
                f"{name} does not declare a periodic/fractional-occupation electronic-state "
                f"model for charge={charge}, mult={multiplicity}."
            )
    else:
        electron_count = int(np.asarray(atoms.get_atomic_numbers(), dtype=int).sum()) - charge
        spin_excess = multiplicity - 1
        if electron_count < 0:
            raise ValueError(
                f"Requested molecular charge={charge} gives a negative electron count "
                f"({electron_count})."
            )
        if spin_excess > electron_count:
            raise ValueError(
                f"Requested multiplicity={multiplicity} exceeds the maximum allowed by "
                f"the molecular electron count ({electron_count})."
            )
        if (electron_count - spin_excess) % 2:
            raise ValueError(
                f"Requested multiplicity={multiplicity} has inconsistent electron/spin "
                f"parity for {electron_count} molecular electrons."
            )
    if (charge != 0 or multiplicity != 1) and not getattr(backend, "SUPPORTS_CHARGE_MULT", False):
        name = backend.__name__ if isinstance(backend, type) else type(backend).__name__
        raise ValueError(
            f"{name} does not support the requested electronic state "
            f"(charge={charge}, mult={multiplicity}); charge/multiplicity cannot be ignored."
        )
    settings = model_options or {}
    settings_provider = getattr(backend, "electronic_state_settings", None)
    if not isinstance(backend, type) and callable(settings_provider):
        settings = settings_provider()
    validator = getattr(backend, "validate_electronic_state_request", None)
    if callable(validator):
        validator(charge, multiplicity, settings)
    return charge, multiplicity


def electronic_state_identity(atoms) -> dict[str, Any]:
    """Return the canonical, JSON-compatible electronic/PES identity."""
    calculator = getattr(atoms, "calc", None)
    calculator_identity = _calculator_identity(calculator)
    charge, multiplicity = validate_electronic_state(atoms, calculator)
    return {
        "schema_version": IDENTITY_SCHEMA_VERSION,
        "charge": charge,
        "multiplicity": multiplicity,
        "backend": calculator_identity["backend"],
        "implementation_version": calculator_identity["implementation_version"],
        "model_fingerprint": calculator_identity["model_fingerprint"],
        "relevant_settings": calculator_identity["relevant_settings"],
    }


def canonical_identity_json(identity: dict[str, Any]) -> str:
    return json.dumps(_json_value(identity), sort_keys=True, separators=(",", ":"))


def validate_path_contract(images: Iterable, *, method: str = "reaction path") -> dict[str, Any]:
    """Require a non-periodic path with one ordering and content-addressed PES."""
    path = list(images)
    if len(path) < 2:
        raise ValueError(f"{method} requires at least two structures.")

    reference = path[0]
    periodic_images = [
        index
        for index, image in enumerate(path)
        if np.any(np.asarray(image.get_pbc(), dtype=bool))
    ]
    if periodic_images:
        raise ValueError(
            f"{method} uses a Cartesian non-periodic path implementation and does not "
            f"support PBC (periodic image indices: {periodic_images})."
        )
    if reference.calc is None:
        raise ValueError(f"{method} requires a calculator on the first structure.")
    reference_numbers = np.asarray(reference.get_atomic_numbers(), dtype=int)
    reference_pbc = np.asarray(reference.get_pbc(), dtype=bool)
    reference_cell = np.asarray(reference.get_cell(), dtype=float)
    reference_identity = electronic_state_identity(reference)
    reference_json = canonical_identity_json(reference_identity)

    for index, image in enumerate(path[1:], start=1):
        numbers = np.asarray(image.get_atomic_numbers(), dtype=int)
        if not np.array_equal(numbers, reference_numbers):
            raise ValueError(
                f"{method} image {index} has a different atomic-number ordering."
            )
        pbc = np.asarray(image.get_pbc(), dtype=bool)
        if not np.array_equal(pbc, reference_pbc):
            raise ValueError(f"{method} image {index} has different PBC flags.")
        if not np.allclose(
            np.asarray(image.get_cell(), dtype=float), reference_cell, rtol=0.0, atol=1e-12
        ):
            raise ValueError(f"{method} image {index} has a different cell.")

        if image.calc is None:
            image.calc = reference.calc
        identity = electronic_state_identity(image)
        if canonical_identity_json(identity) != reference_json:
            raise ValueError(
                f"{method} image {index} is on a different electronic state or PES: "
                f"expected {reference_identity}, got {identity}."
            )
    return reference_identity
