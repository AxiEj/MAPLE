"""Frozen-source V4 component smoke across the complete named SMD catalog.

This is deliberately not a public-entry replay, an accuracy benchmark, or a
scientific admission artifact.  One official MACE-POLAR energy/source graph is
constructed per geometry/device and reused because that graph is independent
of solvent identity.  Every solvent still receives its own strict ddPCM and
legacy-SMD-CDS construction and analytic coordinate derivative.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from dataclasses import asdict, is_dataclass
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch
from ase import Atoms

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from maple.function.calculator.extra_correction.implicit.smd_cds import (
    smd_coulomb_radii,
)
from maple.function.route2_solvents import SMD_SOLVENTS
from maple.solvation.continuum.torch_ddpcm import TorchDDPCM
from maple.solvation.continuum.fp32_partition import FP32_PARTITION_POLICY
from maple.solvation.coupling.exact_gto import (
    mace_polar_learned_source_embedding_matrix,
)
from maple.solvation.models.mace_polar_fp32 import (
    build_official_mace_polar_fp32_adapter,
)
from maple.solvation.numerical_precision import source_charge_tolerance
from maple.solvation.nonpolar.legacy_smd_cds import TorchLegacySMDCDS

ARTIFACT_ID = "route2-pure-macepolar-torch-fp32-v4-multisolvent-component-smoke-v1"
_HVP_CANDIDATES = (
    "water",
    "methanol",
    "acetonitrile",
    "acetone",
    "dimethylsulfoxide",
    "benzene",
    "chloroform",
    "hexane",
)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return _sha256_bytes(encoded)


def _source_manifest() -> dict[str, str]:
    paths = sorted((ROOT / "maple").rglob("*.py")) + [Path(__file__).resolve()]
    return {
        str(path.relative_to(ROOT)): _sha256_bytes(path.read_bytes()) for path in paths
    }


def _plain(value: Any) -> Any:
    if is_dataclass(value):
        return _plain(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _water_geometry() -> Atoms:
    return Atoms(
        "OH2",
        positions=[[0.0, 0.0, 0.0], [0.9572, 0.0, 0.0], [-0.239, 0.9266, 0.0]],
        info={"charge": 0, "mult": 1, "multiplicity": 1},
    )


def _representative_hvp_solvents(
    solvent_names: Sequence[str], count: int
) -> tuple[str, ...]:
    if count < 0:
        raise ValueError("HVP count must be non-negative.")
    available = set(solvent_names)
    selected = [name for name in _HVP_CANDIDATES if name in available]
    selected.extend(name for name in solvent_names if name not in selected)
    return tuple(selected[:count])


def _execution_context(device: str):
    if device == "cpu":
        return nullcontext()
    from maple.solvation.experimental.cuda_execution import cuda_model_execution

    return cuda_model_execution()


def _source_identity(
    *, symbols: Sequence[str], positions: torch.Tensor, source: torch.Tensor
) -> str:
    payload = hashlib.sha256(b"route2-fp32-frozen-source-cache-v1")
    payload.update(json.dumps(tuple(symbols), separators=(",", ":")).encode())
    for tensor in (positions, source):
        array = np.ascontiguousarray(tensor.detach().cpu().numpy())
        payload.update(str(array.dtype).encode())
        payload.update(str(array.shape).encode())
        payload.update(array.tobytes(order="C"))
    return payload.hexdigest()


def _case_failure(solvent: str, device: str, error: BaseException) -> dict[str, Any]:
    return {
        "solvent": solvent,
        "device": device,
        "status": "failure",
        "error_type": type(error).__name__,
        "error": str(error),
    }


def _run_device(
    *,
    atoms: Atoms,
    device: str,
    checkpoint_path: str | Path | None,
    solvent_names: Sequence[str],
    hvp_solvents: frozenset[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    normalized_device = "cuda:0" if device == "cuda" else str(device)
    symbols = tuple(atoms.get_chemical_symbols())
    cases: list[dict[str, Any]] = []
    model = None
    source_sha256 = None
    learned_charge_e = None
    learned_charge_atol_e = None
    model_configuration_sha256 = None
    model_forward_count = 0
    started = time.perf_counter()
    try:
        model = build_official_mace_polar_fp32_adapter(
            device=normalized_device, checkpoint_path=checkpoint_path
        )
        model_configuration_sha256 = model.configuration_sha256()
        positions = torch.tensor(
            atoms.positions,
            dtype=torch.float32,
            device=normalized_device,
            requires_grad=True,
        )
        direction = torch.tensor(
            np.arange(1, 3 * len(symbols) + 1, dtype=np.float32).reshape(-1, 3) / 9.0,
            dtype=torch.float32,
            device=normalized_device,
        )
        with _execution_context(normalized_device):
            model_forward_count += 1
            vacuum, learned = model.energy_source_torch(atoms, positions)
            embedding = torch.tensor(
                mace_polar_learned_source_embedding_matrix(),
                dtype=torch.float32,
                device=normalized_device,
            )
            source = learned @ embedding.T
            if not bool(torch.isfinite(vacuum)) or not bool(
                torch.isfinite(source).all()
            ):
                raise RuntimeError("Official MACE-POLAR produced non-finite output.")
            learned_charge_e = float(learned[:, 0].sum().detach().cpu())
            learned_charge_atol_e = source_charge_tolerance(learned[:, 0])
            if abs(learned_charge_e) > learned_charge_atol_e:
                raise RuntimeError(
                    "Official MACE-POLAR source violates the unchanged neutral "
                    f"charge contract ({learned_charge_e:.6e} e exceeds "
                    f"{learned_charge_atol_e:.6e} e)."
                )
            source_sha256 = _source_identity(
                symbols=symbols, positions=positions, source=source
            )

            for solvent in solvent_names:
                case_started = time.perf_counter()
                try:
                    specification = SMD_SOLVENTS[solvent]
                    continuum = TorchDDPCM(
                        symbols,
                        smd_coulomb_radii(symbols, solvent=solvent),
                        dielectric=specification.descriptors.dielectric,
                        lmax=5,
                        n_lebedev=194,
                        dtype=torch.float32,
                        device=normalized_device,
                    )
                    cds = TorchLegacySMDCDS(
                        symbols, solvent, device=normalized_device, dtype=torch.float32
                    )
                    state = continuum._state_torch(
                        positions, source, enforce_checks=True
                    )
                    ddpcm_energy = state.energy
                    cds_energy = cds.energy_torch(positions)
                    total = vacuum + ddpcm_energy + cds_energy
                    need_hvp = solvent in hvp_solvents
                    gradient = torch.autograd.grad(
                        total,
                        positions,
                        create_graph=need_hvp,
                        retain_graph=True,
                    )[0]
                    if not bool(torch.isfinite(total)) or not bool(
                        torch.isfinite(gradient).all()
                    ):
                        raise RuntimeError("Component sweep produced non-finite E/F.")
                    case = {
                        "solvent": solvent,
                        "pyscf_smd_name": specification.pyscf_smd_name,
                        "device": normalized_device,
                        "status": "success",
                        "dielectric": specification.descriptors.dielectric,
                        "descriptors": list(specification.descriptors.as_pyscf_tuple()),
                        "radii_angstrom": continuum.radii_angstrom.tolist(),
                        "vacuum_energy_eV": float(vacuum.detach().cpu()),
                        "ddpcm_energy_eV": float(ddpcm_energy.detach().cpu()),
                        "cds_energy_eV": float(cds_energy.detach().cpu()),
                        "total_energy_eV": float(total.detach().cpu()),
                        "forces_eV_per_A": (-gradient.detach().cpu()).tolist(),
                        "source_sha256": source_sha256,
                        "continuum_configuration_sha256": (
                            continuum.configuration_sha256()
                        ),
                        "cds_configuration_sha256": cds.configuration_sha256(),
                        "continuum_numerical_policy": dict(continuum.numerical_policy),
                        "hvp_eV_per_A2": None,
                        "topology_warning_codes": (
                            list(state.fp32_partition.warnings)
                            if state.fp32_partition is not None
                            else []
                        ),
                        "fp32_partition": (
                            asdict(state.fp32_partition)
                            if state.fp32_partition is not None
                            else None
                        ),
                        "assessment_gate_failures": list(
                            state.assessment_gate_failures
                        ),
                    }
                    if need_hvp:
                        hvp = torch.autograd.grad(
                            torch.sum(gradient * direction),
                            positions,
                            retain_graph=True,
                        )[0]
                        if not bool(torch.isfinite(hvp).all()):
                            raise RuntimeError(
                                "Component sweep produced non-finite HVP."
                            )
                        case["hvp_eV_per_A2"] = hvp.detach().cpu().tolist()
                except Exception as error:  # Preserve every solvent failure in output.
                    case = _case_failure(solvent, normalized_device, error)
                case["elapsed_seconds_descriptive_only"] = (
                    time.perf_counter() - case_started
                )
                cases.append(case)
    except Exception as error:
        cases = [
            _case_failure(solvent, normalized_device, error)
            for solvent in solvent_names
        ]

    device_record = {
        "device": normalized_device,
        "model_constructed_once": model is not None,
        "model_forward_count": model_forward_count,
        "frozen_source_reused_across_solvents": source_sha256 is not None,
        "source_sha256": source_sha256,
        "learned_source_charge_e": (learned_charge_e if model is not None else None),
        "learned_source_charge_atol_e": (
            learned_charge_atol_e if model is not None else None
        ),
        "model_configuration_sha256": model_configuration_sha256,
        "elapsed_seconds_descriptive_only": time.perf_counter() - started,
    }
    return device_record, cases


def _write_record(path: Path, record: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(_plain(record), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--device",
        action="append",
        dest="devices",
        help="Repeat for multiple devices; default: cpu (cuda aliases cuda:0).",
    )
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--hvp-count", type=int, default=8)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Preserve existing evidence: {args.output}")
    if args.hvp_count < 0:
        raise ValueError("--hvp-count must be non-negative.")

    # Required by the official FP32 adapter; this is process-local CLI bootstrap.
    torch.set_default_dtype(torch.float32)
    torch.set_num_threads(1)
    atoms = _water_geometry()
    solvent_names = tuple(sorted(SMD_SOLVENTS))
    requested_devices = args.devices or ["cpu"]
    hvp_names = _representative_hvp_solvents(solvent_names, args.hvp_count)
    source_manifest = _source_manifest()
    geometry = {
        "symbols": atoms.get_chemical_symbols(),
        "positions_angstrom": atoms.positions.tolist(),
        "charge": 0,
        "multiplicity": 1,
    }
    record: dict[str, Any] = {
        "artifact_id": ARTIFACT_ID,
        "schema_version": 1,
        "claim_boundary": (
            "Cached-component full-catalog engineering smoke only; not a public-entry "
            "end-to-end replay, timing benchmark, fitted result, experimental-label "
            "comparison, or scientific qualification."
        ),
        "scientific_release_admitted": False,
        "experimental_solvation_accuracy_admitted": False,
        "dataset_labels_used": False,
        "fit_performed": False,
        "new_fitted_parameters_introduced": False,
        "catalog_cardinality": len(solvent_names),
        "catalog_solvents": list(solvent_names),
        "catalog_identity_sha256": _json_sha256(
            {
                name: {
                    "pyscf_smd_name": specification.pyscf_smd_name,
                    "mnsol_name": specification.mnsol_name,
                    "descriptors": specification.descriptors.as_pyscf_tuple(),
                }
                for name, specification in SMD_SOLVENTS.items()
            }
        ),
        "method": {
            "dtype": "float32",
            "lmax": 5,
            "lebedev_points": 194,
            "partition_policy": FP32_PARTITION_POLICY,
            "source_cache_scope": "one geometry and one device",
            "solvent_dependent_components_rebuilt": ["ddpcm", "legacy-smd-cds"],
        },
        "geometry": geometry,
        "geometry_sha256": _json_sha256(geometry),
        "hvp_solvents": list(hvp_names),
        "hvp_selection_basis": (
            "Named polarity, hydrogen-bonding, aromatic, halogenated, and "
            "hydrocarbon representatives when present, then deterministic fill."
        ),
        "runtime": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "devices_requested": requested_devices,
        },
        "source_manifest": source_manifest,
        "devices": [],
        "cases": [],
        "summary": {
            "expected_cases": len(solvent_names) * len(requested_devices),
            "recorded_cases": 0,
            "successes": 0,
            "failures": 0,
            "complete_population_reported": False,
        },
        "source_unchanged": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _write_record(args.output, record)

    for device in requested_devices:
        device_record, cases = _run_device(
            atoms=atoms,
            device=device,
            checkpoint_path=args.checkpoint_path,
            solvent_names=solvent_names,
            hvp_solvents=frozenset(hvp_names),
        )
        record["devices"].append(device_record)
        record["cases"].extend(cases)
        successes = sum(case["status"] == "success" for case in record["cases"])
        record["summary"] = {
            "expected_cases": len(solvent_names) * len(requested_devices),
            "recorded_cases": len(record["cases"]),
            "successes": successes,
            "failures": len(record["cases"]) - successes,
            "complete_population_reported": (
                len(record["cases"]) == len(solvent_names) * len(requested_devices)
            ),
        }
        _write_record(args.output, record)

    record["source_unchanged"] = source_manifest == _source_manifest()
    _write_record(args.output, record)
    complete = (
        record["summary"]["expected_cases"] == record["summary"]["recorded_cases"]
    )
    if not complete or not record["source_unchanged"] or record["summary"]["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
