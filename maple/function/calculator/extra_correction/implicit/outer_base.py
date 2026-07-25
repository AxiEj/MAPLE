from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from ase import Atoms

from maple.function.dispatcher.solvfe.protocol import canonical_sha256


_REQUIRED_PROVENANCE = {
    "source_commit",
    "source_blob_sha256",
    "model_sha256",
    "pcmsolver_library_sha256",
    "profile",
    "cavity_policy",
}


def atom_list_sha256(atoms: Atoms) -> str:
    return canonical_sha256(
        {
            "atomic_numbers": [int(value) for value in atoms.numbers],
            "atom_count": len(atoms),
        }
    )


@dataclass(frozen=True)
class OuterDeltaResult:
    atom_list_hash: str
    solute_polarization_hartree: float
    pcm_polarization_hartree: float
    electrostatic_hartree: float
    cds_hartree: float
    standard_state_hartree: float
    delta_g_solv_hartree: float
    provenance: Mapping[str, Any]
    whole_cluster: bool = True
    force_support: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "provenance",
            MappingProxyType(dict(self.provenance)),
        )


class SMDPolarDeltaProvider:
    """Strict Route-2-compatible adapter around one stateless backend call.

    The backend owns all SMD/PCMSolver physics.  This class only freezes units,
    component identities, warning policy, whole-cluster semantics, and
    provenance so Route A cannot duplicate or silently alter Route 2 math.
    """

    def __init__(
        self,
        backend: Callable[[Atoms], Mapping[str, Any]],
    ) -> None:
        if not callable(backend):
            raise TypeError("SMDPolarDeltaProvider backend must be callable.")
        self._backend = backend

    def evaluate_delta(self, atoms: Atoms) -> OuterDeltaResult:
        if not isinstance(atoms, Atoms):
            raise TypeError("Outer delta evaluation requires one ASE Atoms object.")
        raw = self._backend(atoms.copy())
        return self._validate_result(atoms, raw)

    def evaluate_many(self, atoms: list[Atoms]) -> tuple[OuterDeltaResult, ...]:
        """Evaluate one topology-preserving frame batch through the backend.

        A backend may expose ``evaluate_many`` to keep a heavy MACE-POLAR model
        resident across frames.  The exact same validation is applied to every
        returned frame, so batching cannot weaken the public component,
        warning, or provenance contract.
        """

        if not isinstance(atoms, list) or not atoms:
            raise TypeError(
                "Outer delta batch evaluation requires a non-empty list of "
                "ASE Atoms objects."
            )
        if not all(isinstance(value, Atoms) for value in atoms):
            raise TypeError(
                "Outer delta batch evaluation requires only ASE Atoms objects."
            )
        batched = getattr(self._backend, "evaluate_many", None)
        if callable(batched):
            raw_results = batched([value.copy() for value in atoms])
        else:
            raw_results = [self._backend(value.copy()) for value in atoms]
        if not isinstance(raw_results, (list, tuple)):
            raise TypeError("Route 2 outer batch backend must return a sequence.")
        if len(raw_results) != len(atoms):
            raise ValueError(
                "Route 2 outer batch backend returned the wrong frame count."
            )
        return tuple(
            self._validate_result(value, raw)
            for value, raw in zip(atoms, raw_results, strict=True)
        )

    @staticmethod
    def _validate_result(
        atoms: Atoms,
        raw: Mapping[str, Any],
    ) -> OuterDeltaResult:
        if not isinstance(raw, Mapping):
            raise TypeError("Route 2 outer backend must return a mapping.")
        if raw.get("units") != "hartree":
            raise ValueError("Route 2 outer backend units must be hartree.")
        warnings = raw.get("warnings")
        if not isinstance(warnings, list) or warnings:
            raise ValueError("SMD_WARNING: every native or PEDRA warning is fatal.")

        names = (
            "solute_polarization",
            "pcm_polarization",
            "electrostatic",
            "cds",
            "standard_state",
            "delta_g_solv",
        )
        try:
            values = {name: float(raw[name]) for name in names}
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "Route 2 outer backend omitted a required numeric component."
            ) from exc
        if not all(math.isfinite(value) for value in values.values()):
            raise ValueError("Route 2 outer backend returned a non-finite component.")
        if not math.isclose(
            values["electrostatic"],
            values["solute_polarization"] + values["pcm_polarization"],
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError(
                "Route 2 outer identity failed: electrostatic != "
                "solute_polarization + pcm_polarization."
            )
        if not math.isclose(
            values["delta_g_solv"],
            values["electrostatic"]
            + values["cds"]
            + values["standard_state"],
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError(
                "Route 2 outer identity failed: delta_g_solv != "
                "electrostatic + cds + standard_state."
            )

        provenance = raw.get("provenance")
        if not isinstance(provenance, Mapping):
            raise ValueError("Route 2 outer backend provenance is required.")
        missing = sorted(_REQUIRED_PROVENANCE - set(provenance))
        if missing:
            raise ValueError(
                "Route 2 outer backend provenance is incomplete: "
                + ", ".join(missing)
            )

        return OuterDeltaResult(
            atom_list_hash=atom_list_sha256(atoms),
            solute_polarization_hartree=values["solute_polarization"],
            pcm_polarization_hartree=values["pcm_polarization"],
            electrostatic_hartree=values["electrostatic"],
            cds_hartree=values["cds"],
            standard_state_hartree=values["standard_state"],
            delta_g_solv_hartree=values["delta_g_solv"],
            provenance=provenance,
        )
