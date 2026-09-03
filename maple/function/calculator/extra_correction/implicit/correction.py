"""Route-2 composition boundary for field-aware MLIPs and SMD continuum."""

from __future__ import annotations

import hashlib
import json
import os
import threading
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .ddpcm_smd import PyDDXSMDImplicitSolvation
from .fc_aswig_smd import FixedTopologyASWIGAqueousSMDImplicitSolvation
from .mace_polar_ef_smooth_smd import (
    MACEPolarEFSmoothPCMKnownNonpassiveDiagnostic,
)
from .result import SolvationResult
from .smd import SMDImplicitSolvation
from .source_receiver_contract import route2_source_receiver_contract


def _json_ready(value: Any) -> Any:
    """Preserve the existing JSON boundary before deriving an artifact hash."""

    return json.loads(
        json.dumps(
            value,
            sort_keys=True,
            default=str,
            allow_nan=False,
        )
    )


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        _json_ready(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _evaluation_geometry(atoms) -> dict[str, Any]:
    """Return the complete current ASE geometry used by one evaluation."""

    atomic_numbers = np.asarray(atoms.get_atomic_numbers(), dtype=int)
    positions = np.asarray(atoms.get_positions(), dtype=float)
    cell = np.asarray(atoms.get_cell(), dtype=float)
    periodic = np.asarray(atoms.get_pbc(), dtype=bool)
    if (
        atomic_numbers.ndim != 1
        or positions.shape != (len(atomic_numbers), 3)
        or cell.shape != (3, 3)
        or periodic.shape != (3,)
        or not np.all(np.isfinite(positions))
        or not np.all(np.isfinite(cell))
    ):
        raise ValueError(
            "Route 2 evaluation geometry must contain finite ASE atoms, "
            "positions, cell, and periodicity."
        )
    return {
        "atomic_numbers": atomic_numbers.tolist(),
        "chemical_symbols": list(atoms.get_chemical_symbols()),
        "positions_angstrom": positions.tolist(),
        "cell_angstrom": cell.tolist(),
        "pbc": periodic.tolist(),
    }


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically publish one human-readable, canonically hashable JSON record."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary_path.write_text(
        json.dumps(
            _json_ready(payload),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _public_force_record(result: SolvationResult) -> dict[str, Any]:
    """Return the auditable property scope for one public result record.

    A force array is publishable only together with the per-geometry,
    fail-closed Route-2 admission certificate that authorized it.  Energy-only
    evaluations carry an explicit null certificate so consumers never infer a
    force claim from profile-level metadata alone.
    """

    forces_evaluated = result.forces_hartree_per_angstrom is not None
    force_admission = result.provenance.get("force_admission")
    if not forces_evaluated:
        return {
            "forces_evaluated": False,
            "force_admission": None,
        }
    if not isinstance(force_admission, Mapping):
        raise ValueError(
            "A public Route-2 force result requires a per-geometry "
            "force_admission certificate."
        )
    if force_admission.get("release_admitted") is not True:
        raise ValueError(
            "A public Route-2 force result requires release_admitted=true."
        )
    return {
        "forces_evaluated": True,
        "force_admission": _json_ready(dict(force_admission)),
    }


class ImplicitSolvationCorrection:
    """Prepare and evaluate one explicitly selected Route-2 SMD provider."""

    provider_api_version = 1
    _audit_write_lock = threading.RLock()

    def __init__(
        self,
        atoms,
        charge_options: dict[str, Any],
        solvation_options: dict[str, Any],
        *,
        output: str | os.PathLike[str] | None = None,
    ):
        self.atoms = atoms
        self.charge_options = dict(charge_options)
        self.solvation_options = dict(solvation_options)
        if self.charge_options:
            raise ValueError(
                "Route 2 obtains its electrostatic source from the selected "
                "electronic-model adapter; remove #charge(...)."
            )
        if self.solvation_options.get("experimental") is not True:
            raise ValueError(
                "Route 2 is an uncertified research path; "
                "set experimental=true explicitly."
            )
        method = str(self.solvation_options.get("method", "")).lower()
        if method != "smd":
            raise ValueError("The Route-2 branch supports method='smd' only.")
        if "profile" not in self.solvation_options:
            raise ValueError(
                "Route 2 SMD requires an explicit versioned profile."
            )
        provider_name = str(
            self.solvation_options.get("provider", "pcmsolver")
        ).lower()
        providers = {
            "pcmsolver": SMDImplicitSolvation,
            "pyddx": PyDDXSMDImplicitSolvation,
            "fc-aswig": FixedTopologyASWIGAqueousSMDImplicitSolvation,
            "torch-smooth-pcm": MACEPolarEFSmoothPCMKnownNonpassiveDiagnostic,
        }
        try:
            provider_type = providers[provider_name]
        except KeyError as exc:
            raise ValueError(
                "Route 2 provider must be pcmsolver, pyddx, fc-aswig, "
                "or torch-smooth-pcm."
            ) from exc

        self.method = "smd"
        self.mode = str(self.solvation_options.get("response", "scf")).lower()
        self.source_receiver_contract = route2_source_receiver_contract(
            str(self.solvation_options["profile"])
        )
        output_path = Path(output).resolve() if output else Path.cwd() / "maple.out"
        self.audit_dir = output_path.with_suffix(output_path.suffix + ".implicit")
        self.audit_dir.mkdir(parents=True, exist_ok=True)
        self.provider = provider_type(
            atoms,
            self.solvation_options,
            audit_dir=self.audit_dir,
        )
        self.supported_properties = set(self.provider.supported_properties)
        self._write_audit_manifest()

    def _write_audit_manifest(self) -> None:
        provenance = getattr(self.provider, "provenance", None)
        if callable(provenance):
            provenance = provenance()
        geometry = _evaluation_geometry(self.atoms)
        manifest = {
            "schema_version": 4,
            "charge": None,
            "solvation": provenance,
            "charge_options": {},
            "solvation_options": self.solvation_options,
            "solvation_options_sha256": _canonical_json_sha256(
                self.solvation_options
            ),
            "source_receiver_contract": self.source_receiver_contract.as_provenance(),
            "energy_composition": "E_MAPLE_gas + delta_G_solv",
            "response_lifecycle": f"density-coupled-{self.mode}",
            "initial_geometry": geometry,
            "initial_geometry_sha256": _canonical_json_sha256(geometry),
            "public_evaluation_record_directory": "route2-public-results",
            "public_evaluation_record_contract": (
                "each record carries current geometry, evaluated-property "
                "scope, an optional per-geometry force-admission certificate, "
                "and immutable content/manifest integrity digests"
            ),
        }
        self._manifest_sha256 = _canonical_json_sha256(manifest)
        _atomic_write_json(
            self.audit_dir / "manifest.json",
            {
                **manifest,
                "manifest_sha256": self._manifest_sha256,
            },
        )

    def _write_public_result_ledger(self, atoms, result: SolvationResult) -> None:
        """Persist the checked leaf/derived energy split for one evaluation."""

        if not result.leaf_components_hartree:
            return
        run_id = uuid.uuid4().hex
        geometry = _evaluation_geometry(atoms)
        geometry_sha256 = _canonical_json_sha256(geometry)
        payload = {
            "schema_version": 3,
            "run_id": run_id,
            "energy_hartree": float(result.energy_hartree),
            "leaf_components_hartree": dict(result.leaf_components_hartree),
            "derived_totals_hartree": dict(result.derived_totals_hartree),
            "profile": result.provenance.get("profile"),
            "provider": result.provenance.get("provider"),
            "evaluation_geometry": geometry,
            "geometry_sha256": geometry_sha256,
            "solvation_options_sha256": _canonical_json_sha256(
                self.solvation_options
            ),
            "base_manifest_sha256": self._manifest_sha256,
            "source_receiver_contract": self.source_receiver_contract.as_provenance(),
            "component_contract": (
                "derived totals are checked from leaves; do not sum the "
                "legacy flat components map"
            ),
            **_public_force_record(result),
        }
        result_content_sha256 = _canonical_json_sha256(payload)
        evaluation_manifest = {
            "schema_version": 1,
            "run_id": run_id,
            "base_manifest_sha256": self._manifest_sha256,
            "geometry_sha256": geometry_sha256,
            "solvation_options_sha256": payload["solvation_options_sha256"],
            "result_content_sha256": result_content_sha256,
        }
        evaluation_manifest_sha256 = _canonical_json_sha256(evaluation_manifest)
        record_directory = self.audit_dir / "route2-public-results"
        manifest_path = record_directory / f"{run_id}.manifest.json"
        record_path = record_directory / f"{run_id}.json"
        published_payload = {
            **payload,
            "result_content_sha256": result_content_sha256,
            "evaluation_manifest_path": str(
                manifest_path.relative_to(self.audit_dir)
            ),
            "evaluation_manifest_sha256": evaluation_manifest_sha256,
        }
        with self._audit_write_lock:
            _atomic_write_json(
                manifest_path,
                {
                    **evaluation_manifest,
                    "evaluation_manifest_sha256": evaluation_manifest_sha256,
                },
            )
            _atomic_write_json(record_path, published_payload)
            _atomic_write_json(
                self.audit_dir / "route2-public-result-ledger.json",
                published_payload,
            )

    def evaluate(
        self,
        atoms,
        *,
        need_forces: bool = False,
        calculator=None,
    ) -> SolvationResult:
        if need_forces and "forces" not in self.supported_properties:
            raise NotImplementedError(
                "Route 2 SMD does not expose forces before the "
                "solution-phase PES validation gate passes."
            )
        result = self.provider.evaluate(
            atoms,
            need_forces=need_forces,
            calculator=calculator,
        )
        self._write_public_result_ledger(atoms, result)
        return result
