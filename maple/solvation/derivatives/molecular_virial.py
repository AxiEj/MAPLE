"""Origin-declared molecular virial from one conservative force sample."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from maple.solvation.coupling.operator import canonical_metadata_sha256

from .scalar_finite_difference import ScalarForceSample


def _digest(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA256 digest.")
    return value


def _readonly(values: object, *, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite with shape {shape}.")
    return np.frombuffer(
        np.ascontiguousarray(array, dtype=np.float64).tobytes(), dtype=np.float64
    ).reshape(shape)


@dataclass(frozen=True, slots=True)
class MolecularVirialEvaluation:
    """Molecular virial and infinitesimal-strain gradient from one force."""

    force_evaluation_sha256: str
    origin_angstrom: np.ndarray
    net_force_eV_per_A: np.ndarray
    raw_virial_eV: np.ndarray
    symmetric_virial_eV: np.ndarray
    strain_gradient_eV: np.ndarray
    maximum_antisymmetry_eV: float
    evaluation_sha256: str = ""

    def __post_init__(self) -> None:
        _digest(self.force_evaluation_sha256, name="force_evaluation_sha256")
        origin = _readonly(self.origin_angstrom, shape=(3,), name="origin_angstrom")
        net = _readonly(
            self.net_force_eV_per_A, shape=(3,), name="net_force_eV_per_A"
        )
        raw = _readonly(self.raw_virial_eV, shape=(3, 3), name="raw_virial_eV")
        symmetric = _readonly(
            self.symmetric_virial_eV, shape=(3, 3), name="symmetric_virial_eV"
        )
        strain = _readonly(
            self.strain_gradient_eV, shape=(3, 3), name="strain_gradient_eV"
        )
        if not np.array_equal(symmetric, 0.5 * (raw + raw.T)):
            raise ValueError("symmetric_virial_eV must symmetrize raw_virial_eV.")
        if not np.array_equal(strain, -raw):
            raise ValueError("strain_gradient_eV must equal -raw_virial_eV.")
        antisymmetry = float(self.maximum_antisymmetry_eV)
        if not np.isfinite(antisymmetry) or antisymmetry < 0.0:
            raise ValueError("maximum_antisymmetry_eV must be finite and non-negative.")
        if antisymmetry != float(np.max(np.abs(raw - raw.T))):
            raise ValueError("maximum_antisymmetry_eV is inconsistent.")
        payload = {
            "contract": "molecular-virial-from-conservative-force-v1",
            "force_evaluation_sha256": self.force_evaluation_sha256,
            "origin_angstrom": origin.tolist(),
            "net_force_eV_per_A": net.tolist(),
            "raw_virial_eV": raw.tolist(),
            "symmetric_virial_eV": symmetric.tolist(),
            "strain_gradient_eV": strain.tolist(),
            "maximum_antisymmetry_eV": antisymmetry,
        }
        expected = canonical_metadata_sha256(payload)
        if self.evaluation_sha256 and self.evaluation_sha256 != expected:
            raise ValueError("evaluation_sha256 does not match virial contents.")
        object.__setattr__(self, "origin_angstrom", origin)
        object.__setattr__(self, "net_force_eV_per_A", net)
        object.__setattr__(self, "raw_virial_eV", raw)
        object.__setattr__(self, "symmetric_virial_eV", symmetric)
        object.__setattr__(self, "strain_gradient_eV", strain)
        object.__setattr__(self, "maximum_antisymmetry_eV", antisymmetry)
        object.__setattr__(self, "evaluation_sha256", expected)


def evaluate_molecular_virial(
    geometry: object,
    force_sample: ScalarForceSample,
    *,
    origin_angstrom: object | None = None,
) -> MolecularVirialEvaluation:
    """Contract one same-scalar molecular force with Cartesian positions."""

    if not isinstance(force_sample, ScalarForceSample):
        raise TypeError("force_sample must be ScalarForceSample.")
    positions = np.asarray(getattr(geometry, "positions", None), dtype=float)
    forces = np.asarray(force_sample.forces_eV_per_A, dtype=float)
    if positions.shape != forces.shape or not np.all(np.isfinite(positions)):
        raise ValueError("geometry positions must match the finite force shape.")
    origin = (
        np.mean(positions, axis=0)
        if origin_angstrom is None
        else np.asarray(origin_angstrom, dtype=float)
    )
    if origin.shape != (3,) or not np.all(np.isfinite(origin)):
        raise ValueError("origin_angstrom must be finite with shape (3,).")
    raw = forces.T @ (positions - origin)
    return MolecularVirialEvaluation(
        force_evaluation_sha256=force_sample.evaluation_sha256,
        origin_angstrom=origin,
        net_force_eV_per_A=np.sum(forces, axis=0),
        raw_virial_eV=raw,
        symmetric_virial_eV=0.5 * (raw + raw.T),
        strain_gradient_eV=-raw,
        maximum_antisymmetry_eV=float(np.max(np.abs(raw - raw.T))),
    )


__all__ = ["MolecularVirialEvaluation", "evaluate_molecular_virial"]
