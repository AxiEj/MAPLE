#!/usr/bin/env python3
"""Replay the iodine role-separated ADT hybrid without solvation targets.

This canary closes one precise provider-domain gap found by the immutable
canonical-ADT 505 run: the v1 free-atom-density registry did not contain
iodine.  The geometry is a separately frozen, target-free artifact.  The
calculation does not load the MNSol dataset or its experimental target.

The canary is deliberately narrow.  It proves that the role-separated iodine
ADT asset supports a real MACE-MDP + MACE-POLAR + ddPCM solve with deterministic
multi-start root replay.  It does not make an accuracy, force, Hessian,
variational, or public-admission claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any

SCRIPT_SOURCE_ROOT = Path(__file__).resolve().parents[2]
GEOMETRY_REPO_PATH = (
    "docs/route2/evidence/mace-mdp-polar-iodine-adt-pro-20260819/"
    "iodine_operator_geometry.json"
)
IODINE_ASSET_JSON_REPO_PATH = (
    "docs/implicit-solvation/benchmarks/"
    "route2-adt-radial-shape-iodine-ecp-valence-v1.json"
)
IODINE_ASSET_NPZ_REPO_PATH = (
    "docs/implicit-solvation/benchmarks/"
    "route2-adt-radial-shape-iodine-ecp-valence-v1.npz"
)
MDP_CHECKPOINT = Path("/home/axie/.cache/mace/MACE-MDP.model")
POLAR_CHECKPOINT = Path("/home/axie/.cache/mace/MACEPOLAR1Mmodel")
RANDOM_SEED = 20260819
EXPECTED_GEOMETRY_PAYLOAD_SHA256 = (
    "434f36216c6353292ffc69e28f0fbd33da43d6c42a84baa769125792c2502488"
)
EXPECTED_UPSTREAM_GEOMETRY_SHA256 = (
    "21fa6f885e35bdff424db7538fb843888fda8658998a0b382a39a4dc461613a9"
)
EXPECTED_ATOMIC_NUMBERS = (6, 1, 1, 1, 6, 1, 1, 6, 1, 1, 53)
ENERGY_REPLAY_ATOL_EV = 2.0e-10
FIELD_REPLAY_ATOL_EV = 3.0e-9


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _array_sha256(values: object) -> str:
    import numpy as np

    array = np.ascontiguousarray(values)
    header = json.dumps(
        {"dtype": array.dtype.str, "shape": list(array.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(header + b"\0" + array.tobytes(order="C")).hexdigest()


def _load_geometry(source_root: Path) -> dict[str, Any]:
    payload = json.loads((source_root / GEOMETRY_REPO_PATH).read_text())
    expected_keys = {
        "artifact",
        "atomic_numbers",
        "charge",
        "claim_boundary",
        "coordinates_angstrom",
        "formula",
        "multiplicity",
        "payload_sha256",
        "schema_version",
        "solvent",
        "source",
    }
    if set(payload) != expected_keys:
        raise RuntimeError("Frozen iodine geometry schema changed.")
    stored = payload.pop("payload_sha256")
    if stored != _canonical_sha256(payload):
        raise RuntimeError("Frozen iodine geometry payload SHA256 mismatch.")
    payload["payload_sha256"] = stored
    if stored != EXPECTED_GEOMETRY_PAYLOAD_SHA256:
        raise RuntimeError("Frozen iodine geometry identity changed.")
    if tuple(payload["atomic_numbers"]) != EXPECTED_ATOMIC_NUMBERS:
        raise RuntimeError("Frozen iodine geometry composition changed.")
    if payload["source"].get("upstream_geometry_sha256") != (
        EXPECTED_UPSTREAM_GEOMETRY_SHA256
    ):
        raise RuntimeError("Frozen iodine upstream geometry binding changed.")
    if payload["claim_boundary"] != {
        "experimental_solvation_target_embedded": False,
        "geometry_only": True,
        "purpose": "target-free-iodine-adt-operator-comparison",
    }:
        raise RuntimeError("Frozen iodine geometry claim boundary changed.")
    return payload


def _write_json_exclusive(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=SCRIPT_SOURCE_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mdp-checkpoint", type=Path, default=MDP_CHECKPOINT)
    parser.add_argument("--polar-checkpoint", type=Path, default=POLAR_CHECKPOINT)
    parser.add_argument("--polar-device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--cold-replays", type=int, default=2)
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, Any]:
    source_root = args.source_root.expanduser().resolve(strict=True)
    if source_root != SCRIPT_SOURCE_ROOT:
        raise RuntimeError("--source-root must be the checkout containing this script.")
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}.")
    if args.cold_replays < 2:
        raise ValueError("The evidence canary requires at least two cold replays.")
    mdp_checkpoint = args.mdp_checkpoint.expanduser().resolve(strict=True)
    polar_checkpoint = args.polar_checkpoint.expanduser().resolve(strict=True)
    geometry = _load_geometry(source_root)

    sys.path.insert(0, str(source_root))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    import ase
    from ase import Atoms
    import numpy as np
    import pyddx
    import scipy
    import torch

    from maple.function.calculator.extra_correction.implicit.smd_cds import (
        smd_coulomb_radii,
    )
    from maple.function.route2_solvents import route2_solvent_spec
    from maple.solvation.api.profiles import (
        MACE_POLAR_ANALYTIC_GAUSSIAN_MULTIPOLE_EVALUATOR_ID,
    )
    from maple.solvation.continuum import SeparatedSourceDDXBackend
    from maple.solvation.experimental.mace_mdp_polar_adt_ddx import (
        MACE_MDPPolarCanonicalADTDDXEnergy,
        ROLE_SEPARATED_ADT_HYBRID_DDX_PROFILE_ID,
        ROLE_SEPARATED_ADT_HYBRID_DDX_PROVIDER_ID,
    )
    from maple.solvation.models import (
        MACE_MDPPermanentSourceAdapter,
        MACEPolarOriginalSourceNativeFieldAdapter,
        build_mace_mdp_moment_adapter,
        build_official_mace_polar_1_m_radial_gto_adapter,
    )
    from maple.solvation.models.mace_mdp_polar_adt import (
        MDP_POLAR_ROLE_SEPARATED_ADT_CONTRACT,
        build_mdp_polar_role_separated_adt_response,
    )

    if args.polar_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    atoms = Atoms(
        numbers=geometry["atomic_numbers"],
        positions=geometry["coordinates_angstrom"],
        info={
            "charge": int(geometry["charge"]),
            "multiplicity": int(geometry["multiplicity"]),
        },
    )
    solvent = route2_solvent_spec(str(geometry["solvent"]))
    symbols = tuple(atoms.get_chemical_symbols())

    model_started = time.perf_counter()
    mdp = build_mace_mdp_moment_adapter(
        checkpoint_path=mdp_checkpoint,
        device="cpu",
    )
    radial = build_official_mace_polar_1_m_radial_gto_adapter(
        checkpoint_path=polar_checkpoint,
        device=args.polar_device,
        long_range_evaluator_profile=(
            MACE_POLAR_ANALYTIC_GAUSSIAN_MULTIPOLE_EVALUATOR_ID
        ),
    )
    response = build_mdp_polar_role_separated_adt_response(
        mdp=mdp,
        base=MACEPolarOriginalSourceNativeFieldAdapter(radial),
        source_root=source_root,
    )
    permanent = MACE_MDPPermanentSourceAdapter(mdp)
    model_load_seconds = time.perf_counter() - model_started
    if response.contract_id != MDP_POLAR_ROLE_SEPARATED_ADT_CONTRACT:
        raise RuntimeError("The iodine canary did not build the v2 response contract.")

    replay_records: list[dict[str, Any]] = []
    fields: list[np.ndarray] = []
    energies: list[float] = []
    for replay_index in range(args.cold_replays):
        continuum = SeparatedSourceDDXBackend(
            symbols,
            smd_coulomb_radii(symbols, solvent=solvent.name),
            continuum_model="pcm",
            dielectric=solvent.descriptors.dielectric,
            lmax=8,
            n_lebedev=1202,
            solver_tolerance=1.0e-12,
            eta=0.1,
            n_proc=1,
        )
        evaluator = MACE_MDPPolarCanonicalADTDDXEnergy(
            atoms,
            permanent=permanent,
            response=response,
            continuum=continuum,
        )
        if (
            evaluator.profile_id != ROLE_SEPARATED_ADT_HYBRID_DDX_PROFILE_ID
            or evaluator.provider_id != ROLE_SEPARATED_ADT_HYBRID_DDX_PROVIDER_ID
        ):
            raise RuntimeError("The evaluator did not select the v2 provider/profile.")
        started = time.perf_counter()
        state = evaluator.solve(atoms)
        solve_seconds = time.perf_counter() - started
        fields.append(np.array(state.native_field8, copy=True))
        energies.append(float(state.polarization_energy_ev))
        replay_records.append(
            {
                "replay_index": replay_index,
                "evaluator_configuration_sha256": evaluator.configuration_sha256(),
                "root_sha256": state.root_sha256,
                "continuum_state_sha256": state.continuum_state_sha256,
                "primal_residual_ev": state.primal_residual_ev,
                "polarization_energy_ev": state.polarization_energy_ev,
                "vacuum_energy_ev": state.vacuum_energy_ev,
                "total_energy_ev": state.total_energy_ev,
                "native_field8_sha256": _array_sha256(state.native_field8),
                "permanent_source4_sha256": _array_sha256(state.permanent_source4),
                "radial_residual_source4_sha256": _array_sha256(
                    state.radial_residual_source4
                ),
                "adt_atomic_dipoles_sha256": _array_sha256(
                    state.adt_atomic_dipoles_eangstrom
                ),
                "total_permanent_charge_e": float(
                    np.sum(state.permanent_source4[:, 0])
                ),
                "total_radial_residual_charge_e": float(
                    np.sum(state.radial_residual_source4[:, 0])
                ),
                "adt_atomic_dipole_sum_eangstrom": np.sum(
                    state.adt_atomic_dipoles_eangstrom, axis=0
                ).tolist(),
                "root_starts": [
                    {
                        "label": item.label,
                        "iterations": item.iterations,
                        "residual_norm_ev": item.residual_norm_ev,
                        "polarization_energy_ev": item.polarization_energy_ev,
                    }
                    for item in state.root_starts
                ],
                "solve_seconds": solve_seconds,
            }
        )

    maximum_field_difference = max(
        float(np.max(np.abs(field - fields[0]))) for field in fields[1:]
    )
    maximum_energy_difference = max(abs(value - energies[0]) for value in energies[1:])
    if maximum_field_difference > FIELD_REPLAY_ATOL_EV:
        raise RuntimeError("Independent cold replays found different native fields.")
    if maximum_energy_difference > ENERGY_REPLAY_ATOL_EV:
        raise RuntimeError("Independent cold replays found different energies.")

    payload: dict[str, Any] = {
        "artifact": "route2-iodine-role-separated-adt-full-replay-v1",
        "schema_version": 1,
        "status": "pass-iodine-provider-and-root-replay",
        "claim_boundary": {
            "experimental_solvation_target_read": False,
            "experimental_solvation_target_used": False,
            "experimental_solvation_target_emitted": False,
            "accuracy_claim": False,
            "force_claim": False,
            "hessian_claim": False,
            "variational_claim": False,
            "public_admission_claim": False,
            "proved": (
                "real-checkpoint iodine provider coverage and deterministic "
                "role-separated ADT/ddPCM multi-start root replay"
            ),
        },
        "geometry": {
            "repo_path": GEOMETRY_REPO_PATH,
            "file_sha256": _sha256(source_root / GEOMETRY_REPO_PATH),
            "payload_sha256": geometry["payload_sha256"],
            "upstream_geometry_sha256": EXPECTED_UPSTREAM_GEOMETRY_SHA256,
            "formula": geometry["formula"],
            "solvent": solvent.name,
        },
        "configuration": {
            "profile_id": ROLE_SEPARATED_ADT_HYBRID_DDX_PROFILE_ID,
            "provider_id": ROLE_SEPARATED_ADT_HYBRID_DDX_PROVIDER_ID,
            "response_contract": response.contract_id,
            "response_configuration_sha256": response.configuration_sha256(),
            "continuum_model": "pcm",
            "lmax": 8,
            "n_lebedev": 1202,
            "solver_tolerance": 1.0e-12,
            "eta": 0.1,
            "n_proc": 1,
            "mdp_device": "cpu",
            "polar_device": args.polar_device,
            "cold_replay_count": args.cold_replays,
        },
        "source": {
            "execution_git_head": subprocess.check_output(
                ["git", "-C", str(source_root), "rev-parse", "HEAD"],
                text=True,
            ).strip(),
            "script_repo_path": str(Path(__file__).resolve().relative_to(source_root)),
            "script_sha256": _sha256(Path(__file__).resolve()),
            "iodine_asset_json_sha256": _sha256(
                source_root / IODINE_ASSET_JSON_REPO_PATH
            ),
            "iodine_asset_npz_sha256": _sha256(
                source_root / IODINE_ASSET_NPZ_REPO_PATH
            ),
            "mdp_checkpoint_sha256": _sha256(mdp_checkpoint),
            "polar_checkpoint_sha256": _sha256(polar_checkpoint),
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "ase": ase.__version__,
            "torch": torch.__version__,
            "pyddx": getattr(pyddx, "__version__", "unknown"),
            "cuda_available": torch.cuda.is_available(),
            "model_load_seconds": model_load_seconds,
        },
        "replays": replay_records,
        "maximum_cold_replay_field_difference_ev": maximum_field_difference,
        "maximum_cold_replay_energy_difference_ev": maximum_energy_difference,
    }
    payload["payload_sha256"] = _canonical_sha256(payload)
    return payload


def main() -> int:
    args = _parse_args()
    payload = run(args)
    _write_json_exclusive(args.output.expanduser().resolve(), payload)
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
