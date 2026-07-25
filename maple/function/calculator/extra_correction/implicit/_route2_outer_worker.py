"""Private child-process entry point for the frozen Route 2 outer provider."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _verify_source(request: dict[str, Any]) -> None:
    root = Path(request["source_root"]).resolve()
    commit = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout.strip()
    if commit != request["source_commit"] or dirty:
        raise RuntimeError(
            "Frozen Route 2 source changed after parent validation."
        )
    for relative_path, expected in request["source_sha256"].items():
        if _sha256(root / relative_path) != expected:
            raise RuntimeError(
                f"Frozen Route 2 source changed: {relative_path}."
            )


def _frame_atoms(topology: dict[str, Any], positions: np.ndarray):
    from ase import Atoms

    atoms = Atoms(
        numbers=topology["atomic_numbers"],
        positions=positions,
        pbc=topology["pbc"],
    )
    atoms.info.update(
        {
            "charge": topology["charge"],
            "mult": topology["multiplicity"],
            "mol2": {"atom_types": topology["atom_types"]},
        }
    )
    return atoms


def _scaled_supermolecule_provider(
    route2_smd,
    *,
    atoms,
    config: dict[str, Any],
    audit_dir: Path,
    rule: dict[str, Any],
    scaled_radii_angstrom: np.ndarray,
):
    class _ScaledSupermoleculePCM(route2_smd.SMDImplicitSolvation):
        def __init__(self) -> None:
            super().__init__(atoms, config, audit_dir=audit_dir)
            expected = (
                np.asarray(self.coulomb_radii_angstrom, dtype=float)
                * float(rule["radius_scale"])
            )
            supplied = np.asarray(scaled_radii_angstrom, dtype=float)
            if (
                supplied.shape != expected.shape
                or not np.allclose(
                    supplied,
                    expected,
                    rtol=0.0,
                    atol=1.0e-12,
                )
            ):
                raise RuntimeError(
                    "Route A scaled cavity radii do not match the frozen "
                    "Route 2 base profile."
                )
            self.coulomb_radii_angstrom = supplied.copy()
            self.provenance.update(
                {
                    "profile": (
                        "scaled-supermolecule-pcm-electrostatic-v2"
                    ),
                    "cavity_policy": (
                        "per-frame-whole-supermolecule-gepol"
                    ),
                    "cavity_radii": (
                        "frozen SMD/GAFF2-O Coulomb radii scaled by "
                        f"{float(rule['radius_scale']):.10g}"
                    ),
                    "cds": "excluded from Route A v2 core",
                    "energy_composition": (
                        "delta_G_outer = "
                        "(E_MACE_intrinsic[V_reac]-E_MACE_gas) "
                        "+ 0.5*<V_solute,ASC>; CDS=0"
                    ),
                }
            )
            self._write_manifest()

        def _ensure_scaled_supermolecule_input(self) -> Path:
            return self._ensure_pcm_input_variant(
                key="scaled-supermolecule-v2",
                filename="route-a-scaled-supermolecule-v2.pcm",
                tessera_area_angstrom2=float(rule["tessera_area_A2"]),
                minimum_added_sphere_radius_angstrom=float(
                    rule["minimum_added_sphere_radius_A"]
                ),
            )

        def _cavity_attempt_specs(self):
            return (
                (
                    "scaled-supermolecule-v2",
                    self._ensure_scaled_supermolecule_input,
                    float(rule["tessera_area_A2"]),
                    float(rule["minimum_added_sphere_radius_A"]),
                ),
            )

    return _ScaledSupermoleculePCM()


def _main(request_path: Path) -> int:
    request = json.loads(request_path.read_text(encoding="utf-8"))
    output_path = Path(request["output_path"]).resolve()
    output: dict[str, Any] = {
        "schema_version": 1,
        "request_sha256": request["request_sha256"],
        "status": "failed",
        "results": [],
    }
    try:
        _verify_source(request)
        source_root = str(Path(request["source_root"]).resolve())
        sys.path.insert(0, source_root)

        from maple.function.calculator.extra_correction.implicit import (
            smd as route2_smd,
        )
        from maple.function.calculator.mace._macepol_calculator import (
            MACEPolCalculator,
        )
        from mace.calculators.foundations_models import (
            download_mace_polar_checkpoint,
        )

        cavity = request["cavity_profile"]
        route2_smd.PCM_STABILITY_FALLBACK_TESSERA_AREA_ANGSTROM2 = float(
            cavity["area_angstrom2"]
        )
        route2_smd.PCM_STABILITY_FALLBACK_MIN_RADIUS_ANGSTROM = float(
            cavity["minimum_added_sphere_radius_angstrom"]
        )

        config = dict(request["provider_config"])
        model_kwargs = MACEPolCalculator.build_implicit_solvent_kwargs(config)
        resolved_model_checkpoint = Path(
            download_mace_polar_checkpoint("polar-1-m")
        ).resolve()
        requested_model_checkpoint = Path(
            request["runtime"]["model_checkpoint"]
        ).resolve()
        if resolved_model_checkpoint != requested_model_checkpoint:
            raise RuntimeError(
                "The supplied MACE-POLAR provenance path is not the official "
                "upstream polar-1-m cache path."
            )
        load_started = time.perf_counter()
        calculator = MACEPolCalculator(
            device=request["runtime"]["device"],
            model="macepolm",
            implicit="smd",
            solvent="water",
            **model_kwargs,
        )
        model_load_seconds = time.perf_counter() - load_started
        if _sha256(resolved_model_checkpoint) != (
            request["runtime"]["model_sha256"]
        ):
            raise RuntimeError("MACE-POLAR checkpoint changed during the run.")
        if _sha256(Path(request["runtime"]["pcmsolver_library"])) != (
            request["runtime"]["pcmsolver_library_sha256"]
        ):
            raise RuntimeError("PCMSolver library changed during the run.")

        positions = np.load(
            request["positions_path"], allow_pickle=False
        )["positions_angstrom"]
        if positions.shape != (
            len(request["frame_ids"]),
            len(request["topology"]["atomic_numbers"]),
            3,
        ):
            raise ValueError("Frozen Route 2 positions array shape mismatch.")

        results: list[dict[str, Any]] = []
        provider_mode = request.get("provider_mode", "route2-smd-v1")
        if provider_mode not in {
            "route2-smd-v1",
            "route-a-scaled-supermolecule-pcm-electrostatic-v2",
        }:
            raise ValueError(f"Unsupported outer provider mode: {provider_mode}.")
        scaled_mode = provider_mode == (
            "route-a-scaled-supermolecule-pcm-electrostatic-v2"
        )
        if scaled_mode:
            rule = request.get("supermolecule_cavity")
            scaled_radii = np.asarray(
                request.get("scaled_radii_angstrom"),
                dtype=float,
            )
            if (
                not isinstance(rule, dict)
                or rule.get("cds_policy")
                != "excluded-from-v2-core-mandatory-ablation"
                or scaled_radii.shape
                != (len(request["topology"]["atomic_numbers"]),)
            ):
                raise ValueError(
                    "Route A v2 supermolecule cavity request is incomplete."
                )
        else:
            rule = {}
            scaled_radii = np.empty(0)

        for index, (frame_id, coordinates) in enumerate(
            zip(request["frame_ids"], positions, strict=True)
        ):
            frame_started = time.perf_counter()
            audit_dir = (
                Path(request["frame_audit_root"]).resolve()
                / f"{index:06d}-{frame_id[:12]}"
            )
            try:
                atoms = _frame_atoms(request["topology"], coordinates)
                provider = (
                    _scaled_supermolecule_provider(
                        route2_smd,
                        atoms=atoms,
                        config=config,
                        audit_dir=audit_dir,
                        rule=rule,
                        scaled_radii_angstrom=scaled_radii,
                    )
                    if scaled_mode
                    else route2_smd.SMDImplicitSolvation(
                        atoms,
                        config,
                        audit_dir=audit_dir,
                    )
                )
                result = provider.evaluate(
                    atoms,
                    need_forces=False,
                    calculator=calculator,
                )
                diagnostics = result.provenance.get(
                    "pcmsolver_diagnostics", {}
                )
                warnings: list[str] = []
                if diagnostics.get("native_stderr_warning_count", 0):
                    warnings.append("PCMSolver native warning")
                if diagnostics.get("pedra_warning_count", 0):
                    warnings.append("PEDRA warning")
                source_components = result.components_hartree
                components = (
                    {
                        "solute_polarization": float(
                            source_components["solute_polarization"]
                        ),
                        "pcm_polarization": float(
                            source_components["pcm_polarization"]
                        ),
                        "electrostatic": float(
                            source_components["electrostatic"]
                        ),
                        "cds": 0.0,
                        "standard_state": 0.0,
                        "delta_g_solv": float(
                            source_components["electrostatic"]
                        ),
                    }
                    if scaled_mode
                    else source_components
                )
                route_a_cavity_override = (
                    {
                        "diagnostic_constant_override": True,
                        "public_parameter_changed": True,
                        "whole_supermolecule": True,
                        "radius_scale": float(rule["radius_scale"]),
                        "tessera_area_angstrom2": float(
                            rule["tessera_area_A2"]
                        ),
                        "minimum_added_sphere_radius_angstrom": float(
                            rule["minimum_added_sphere_radius_A"]
                        ),
                        "source_constants_are_not_runtime_provenance": True,
                    }
                    if scaled_mode
                    else {
                        "diagnostic_constant_override": True,
                        "public_parameter_changed": False,
                        "tessera_area_angstrom2": float(
                            cavity["area_angstrom2"]
                        ),
                        "minimum_added_sphere_radius_angstrom": float(
                            cavity[
                                "minimum_added_sphere_radius_angstrom"
                            ]
                        ),
                        "source_constants_are_not_runtime_provenance": True,
                    }
                )
                provenance = {
                    **result.provenance,
                    "cavity_stability_policy": (
                        "Route A v2 uses one per-frame whole-supermolecule "
                        "GePol cavity with radius scale "
                        f"{float(rule['radius_scale']):.10g}, "
                        f"AREA={float(rule['tessera_area_A2']):.15g} A^2, "
                        "MINRADIUS="
                        f"{float(rule['minimum_added_sphere_radius_A']):.15g} "
                        "A, and treats native and PEDRA warnings as fatal."
                        if scaled_mode
                        else (
                            "Frozen Route A contract uses a predetermined "
                            f"AREA={cavity['area_angstrom2']:.15g} A^2 and "
                            "MINRADIUS="
                            f"{cavity['minimum_added_sphere_radius_angstrom']:.15g} "
                            "A, with native and PEDRA warnings treated as fatal."
                        )
                    ),
                    "route_a_cavity_override": route_a_cavity_override,
                    "source_commit": request["source_commit"],
                    "source_blob_sha256": request["source_blob_sha256"],
                    "model_sha256": request["runtime"]["model_sha256"],
                    "pcmsolver_library_sha256": request["runtime"][
                        "pcmsolver_library_sha256"
                    ],
                    "profile": config["profile"],
                    "cavity_policy": config["cavity_policy"],
                    "contract_sha256": request["contract_sha256"],
                    "adapter_source_sha256": request["runtime"][
                        "adapter_source_sha256"
                    ],
                    "worker_source_sha256": request["runtime"][
                        "worker_source_sha256"
                    ],
                    "request_sha256": request["request_sha256"],
                    "frame_id": frame_id,
                    "model_load_seconds": model_load_seconds,
                    "frame_seconds": time.perf_counter() - frame_started,
                }
                if scaled_mode:
                    provenance.update(
                        {
                            "profile": (
                                "scaled-supermolecule-pcm-electrostatic-v2"
                            ),
                            "cavity_policy": (
                                "per-frame-whole-supermolecule-gepol"
                            ),
                            "route_a_protocol": request["route_a_protocol"],
                            "supermolecule_cavity": rule,
                            "frame_cavity_preflight": request[
                                "frame_cavity_preflight"
                            ][index],
                            "legacy_cds_diagnostic_hartree": float(
                                source_components["cds"]
                            ),
                            "cds_excluded_from_returned_hamiltonian": True,
                        }
                    )
                results.append(
                    {
                        "frame_id": frame_id,
                        "status": "ok",
                        "value": {
                            "units": "hartree",
                            **{
                                name: float(components[name])
                                for name in (
                                    "solute_polarization",
                                    "pcm_polarization",
                                    "electrostatic",
                                    "cds",
                                    "standard_state",
                                    "delta_g_solv",
                                )
                            },
                            "warnings": warnings,
                            "provenance": provenance,
                        },
                    }
                )
            except Exception as exc:
                results.append(
                    {
                        "frame_id": frame_id,
                        "status": "error",
                        "error": {
                            "type": type(exc).__name__,
                            "message": str(exc),
                            "audit_directory": str(audit_dir),
                        },
                    }
                )
        output.update(
            {
                "status": "complete",
                "results": results,
                "source_commit": request["source_commit"],
                "model_load_seconds": model_load_seconds,
            }
        )
        _atomic_json(output_path, output)
        return 0
    except Exception as exc:
        output["fatal_error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
        _atomic_json(output_path, output)
        return 1


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(
            "usage: _route2_outer_worker.py /absolute/path/request.json"
        )
    raise SystemExit(_main(Path(sys.argv[1]).resolve()))
