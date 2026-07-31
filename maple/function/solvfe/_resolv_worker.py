"""Isolated endpoint-BAR worker for the pinned public ReSolv release.

The worker intentionally accepts only the authors' precomputed endpoint
trajectories.  It never generates conformers and never propagates dynamics.
All source and binary identities are checked before the first pickle/dill load.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import importlib.metadata
import io
import json
import math
import os
import pickle
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path, PurePosixPath
from typing import Any, cast

PINNED_SOURCE_REVISION = "1d85bcc065003e083d2e95ab7091cb1762eeb1bf"
OFFICIAL_VACUUM_MODEL_SHA256 = (
    "83618b7b4f6680e80936f68e5e2d02f9c6ff1c62750333ca28fc5733286347c4"
)
OFFICIAL_WATER_MODEL_SHA256 = (
    "56a72b33e7aa3b26e06f7792924f2630a62f1d0a776f2843f5871195e88e3e13"
)
OFFICIAL_DATABASE_SHA256 = (
    "8a1dd006a54f0986f58b967bf7046fb72293dafa66954ec8529cdc5e68b4d405"
)
OFFICIAL_ARTIFACT_HASHES = {
    "vacuum_model": OFFICIAL_VACUUM_MODEL_SHA256,
    "water_model": OFFICIAL_WATER_MODEL_SHA256,
    "database": OFFICIAL_DATABASE_SHA256,
}
OFFICIAL_ARTIFACT_BASENAMES = {
    "vacuum_model": (
        "261023_QM7x_Nequip_Nequip_QM7x_All_8epochs_iL5emin3_"
        "lrdecay1emin3_scaledTargets_LargeTrainingSet_Cutoff4A_"
        "mlp4_ShiftFalse_ScaleFalse_EnergiesAndForces.pkl"
    ),
    "water_model": (
        "080524_t_prod_250ps_t_equil_50ps_iL1e-06_lrd0.1_"
        "epochs500_seed7_train_389mem_0.97_epoch499.pkl"
    ),
    "database": "database.pickle",
}
TRAJECTORY_DIRECTORY = Path(
    "examples/FreeEnergyScripts/QM7x Scripts/TrainFreeEnergy/"
    "precomputed_trajectories"
)
EXPECTED_SNAPSHOTS_PER_ENDPOINT = 40
TEMPERATURE_K = 298.15
KBT_KCAL_MOL = TEMPERATURE_K * 0.00198720426
MAX_SELF_ENERGY_DRIFT_KCAL_MOL = 1.0e-6
ADMITTED_PYTHON_VERSION = "3.10.20"
ADMITTED_RUNTIME_VERSIONS = {
    "blackjax": "0.3.0",
    "chex": "0.1.86",
    "coax": "0.1.13",
    "dill": "0.3.8",
    "dm-haiku": "0.0.12",
    "e3nn-jax": "0.20.6",
    "flax": "0.8.4",
    "jax": "0.4.23",
    "jax-md": "0.2.8",
    "jaxopt": "0.8.3",
    "jraph": "0.0.6.dev0",
    "ml-dtypes": "0.5.4",
    "ml-collections": "0.1.1",
    "networkx": "3.3",
    "numpy": "1.26.4",
    "opt-einsum": "3.4.0",
    "optax": "0.2.2",
    "rdkit": "2023.9.6",
    "scipy": "1.12.0",
}
ADMITTED_JAXLIB_VERSIONS = {
    "cpu": {"0.4.23"},
    "gpu": {"0.4.23+cuda12.cudnn89"},
}


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--response", type=Path, required=True)
    return parser.parse_args()


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise TypeError(f"{label} must be a string-keyed JSON object.")
    return value


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{label} must be a non-empty string.")
    return value


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer.")
    return value


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _git_blob_oid(payload: bytes) -> str:
    header = f"blob {len(payload)}\0".encode("ascii")
    return hashlib.sha1(header + payload).hexdigest()


def _asset(payload: dict[str, Any], name: str) -> tuple[Path, str]:
    """Read either ``name={path,sha256}`` or ``name_path/name_sha256``."""

    raw = payload.get(name)
    if isinstance(raw, dict):
        item = _mapping(raw, name)
        path = Path(_string(item.get("path"), f"{name}.path")).expanduser().resolve()
        expected = _string(item.get("sha256"), f"{name}.sha256").lower()
    else:
        path = Path(
            _string(payload.get(f"{name}_path", raw), f"{name}_path")
        ).expanduser().resolve()
        expected = _string(payload.get(f"{name}_sha256"), f"{name}_sha256").lower()
    if len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected):
        raise ValueError(f"{name} sha256 must be 64 lowercase hexadecimal characters.")
    return path, expected


def _record_asset(record: dict[str, Any], name: str) -> tuple[Path, str]:
    raw = record.get(name)
    if isinstance(raw, dict):
        item = _mapping(raw, name)
        path = Path(_string(item.get("path"), f"{name}.path")).expanduser().resolve()
        expected = _string(item.get("sha256"), f"{name}.sha256").lower()
    else:
        path = Path(
            _string(record.get(f"{name}_path", raw), f"{name}_path")
        ).expanduser().resolve()
        expected = _string(record.get(f"{name}_sha256"), f"{name}_sha256").lower()
    if len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected):
        raise ValueError(f"{name} sha256 must be 64 lowercase hexadecimal characters.")
    return path, expected


def _verified_bytes(path: Path, expected: str, label: str) -> bytes:
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    # The exact bytes hashed here are retained and later deserialized through
    # BytesIO. This closes the verify/re-open race for untrusted pickle inputs.
    payload = path.read_bytes()
    observed = _sha256_bytes(payload)
    if observed != expected:
        raise RuntimeError(
            f"{label} SHA-256 mismatch: expected {expected}, observed {observed}."
        )
    return payload


def _head_blob_oid(upstream_root: Path, relative: Path) -> str:
    completed = subprocess.run(
        [
            "git",
            "-C",
            str(upstream_root),
            "rev-parse",
            f"{PINNED_SOURCE_REVISION}:{relative.as_posix()}",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    oid = completed.stdout.strip()
    if completed.returncode != 0 or len(oid) != 40:
        raise RuntimeError(f"Trajectory is not tracked at pinned HEAD: {relative}")
    return oid


def _git_archive_bytes(upstream_root: Path) -> bytes:
    completed = subprocess.run(
        [
            "git",
            "-C",
            str(upstream_root),
            "archive",
            "--format=tar",
            PINNED_SOURCE_REVISION,
            "--",
            "chemtrain",
            "util",
            "jax_sgmc",
        ],
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0 or not completed.stdout:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"Failed to export pinned ReSolv source: {detail}")
    return completed.stdout


def _extract_safe_source_archive(archive: bytes, destination: Path) -> list[str]:
    """Extract only ordinary files/directories rooted at chemtrain or util."""

    # jax_sgmc is a direct import-time dependency vendored in the same pinned
    # ReSolv commit; it is not installed as a package in the isolated runtime.
    allowed_roots = {"chemtrain", "util", "jax_sgmc"}
    exported_roots: set[str] = set()
    seen: set[PurePosixPath] = set()
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as bundle:
        for member in bundle.getmembers():
            relative = PurePosixPath(member.name)
            if (
                relative.is_absolute()
                or not relative.parts
                or any(part in {"", ".", ".."} for part in relative.parts)
                or relative.parts[0] not in allowed_roots
            ):
                raise RuntimeError(f"Unsafe ReSolv source archive path: {member.name}")
            if relative in seen:
                raise RuntimeError(f"Duplicate ReSolv source archive path: {member.name}")
            seen.add(relative)
            exported_roots.add(relative.parts[0])
            target = destination.joinpath(*relative.parts)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                raise RuntimeError(
                    f"Unsafe ReSolv source archive entry type: {member.name}"
                )
            extracted = bundle.extractfile(member)
            if extracted is None:
                raise RuntimeError(f"Could not read ReSolv source entry: {member.name}")
            payload = extracted.read()
            if len(payload) != member.size:
                raise RuntimeError(f"Truncated ReSolv source entry: {member.name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
    if exported_roots != allowed_roots:
        raise RuntimeError(
            f"Pinned ReSolv source archive roots mismatch: {sorted(exported_roots)}"
        )
    return sorted(exported_roots)


@contextlib.contextmanager
def _pinned_source_snapshot(upstream_root: Path):
    archive = _git_archive_bytes(upstream_root)
    archive_sha256 = _sha256_bytes(archive)
    temporary = tempfile.TemporaryDirectory(prefix="maple-resolv-source-")
    snapshot_root = Path(temporary.name).resolve()
    inserted = False
    try:
        exported_roots = _extract_safe_source_archive(archive, snapshot_root)
        sys.path.insert(0, str(snapshot_root))
        inserted = True
        yield {
            "revision": PINNED_SOURCE_REVISION,
            "archive_sha256": archive_sha256,
            "exported_roots": exported_roots,
            "source": "git-archive",
            "_snapshot_root": str(snapshot_root),
        }
    finally:
        if inserted:
            while str(snapshot_root) in sys.path:
                sys.path.remove(str(snapshot_root))
        for name, module in tuple(sys.modules.items()):
            module_file = getattr(module, "__file__", None)
            if module_file is None:
                continue
            try:
                Path(module_file).resolve().relative_to(snapshot_root)
            except (OSError, ValueError):
                continue
            sys.modules.pop(name, None)
        temporary.cleanup()


def _verify_source(upstream_root: Path, requested_revision: str) -> None:
    if requested_revision != PINNED_SOURCE_REVISION:
        raise RuntimeError(
            "ReSolv request source revision is not the admitted release: "
            f"{requested_revision}."
        )
    required = (
        Path("util/Initialization.py"),
        Path("chemtrain/reweighting.py"),
        Path("chemtrain/traj_util.py"),
    )
    for relative in required:
        source = upstream_root / relative
        if not source.is_file():
            raise FileNotFoundError(f"Pinned ReSolv source is incomplete: {source}")
    completed = subprocess.run(
        ["git", "-C", str(upstream_root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    observed = completed.stdout.strip()
    if completed.returncode != 0 or observed != requested_revision:
        raise RuntimeError(
            "ReSolv checkout revision mismatch: expected "
            f"{requested_revision}, observed {observed or 'unavailable'}."
        )
    clean = subprocess.run(
        ["git", "-C", str(upstream_root), "diff", "--quiet", requested_revision, "--"],
        check=False,
    )
    if clean.returncode != 0:
        raise RuntimeError(
            "ReSolv tracked worktree differs from the pinned source revision."
        )
    for relative in required:
        committed = subprocess.run(
            [
                "git",
                "-C",
                str(upstream_root),
                "rev-parse",
                f"{requested_revision}:{relative}",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        working = subprocess.run(
            ["git", "-C", str(upstream_root), "hash-object", str(relative)],
            check=False,
            capture_output=True,
            text=True,
        )
        if (
            committed.returncode != 0
            or working.returncode != 0
            or committed.stdout.strip() != working.stdout.strip()
        ):
            raise RuntimeError(
                f"Pinned ReSolv source file differs from HEAD: {relative}"
            )


def _normalise_request(raw: object) -> dict[str, Any]:
    request = _mapping(raw, "request")
    upstream_root = Path(
        _string(request.get("upstream_root"), "upstream_root")
    ).expanduser().resolve()
    revision = _string(request.get("source_revision"), "source_revision")
    vacuum_model = _asset(request, "vacuum_model")
    database = _asset(request, "database")
    water_model = _asset(request, "water_model")

    if "records" in request:
        raw_records = request["records"]
        if not isinstance(raw_records, list) or not raw_records:
            raise TypeError("records must be a non-empty list.")
    elif "record" in request:
        raw_records = [request["record"]]
    else:
        raise KeyError("request must contain record or records.")

    records: list[dict[str, Any]] = []
    for index, raw_record in enumerate(raw_records):
        record = _mapping(raw_record, f"records[{index}]")
        k_index = _integer(record.get("k_index"), f"records[{index}].k_index")
        if k_index < 0:
            raise ValueError("k_index must be non-negative.")
        records.append(
            {
                "k_index": k_index,
                "smiles": _string(record.get("smiles"), f"records[{index}].smiles"),
                "vacuum_trajectory": _record_asset(record, "vacuum_trajectory"),
                "water_trajectory": _record_asset(record, "water_trajectory"),
            }
        )

    continue_on_error = request.get("continue_on_record_error", False)
    if not isinstance(continue_on_error, bool):
        raise TypeError("continue_on_record_error must be boolean.")
    temperature = request.get("temperature_kelvin", TEMPERATURE_K)
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not math.isclose(float(temperature), TEMPERATURE_K, abs_tol=1.0e-12)
    ):
        raise ValueError(
            f"ReSolv endpoint artifacts are fixed at {TEMPERATURE_K} K."
        )
    expected_backend = request.get("expected_backend")
    if expected_backend is not None:
        expected_backend = _string(expected_backend, "expected_backend").lower()
        if expected_backend not in {"cpu", "gpu"}:
            raise ValueError("expected_backend must be cpu or gpu.")
    return {
        "upstream_root": upstream_root,
        "source_revision": revision,
        "vacuum_model": vacuum_model,
        "database": database,
        "water_model": water_model,
        "records": records,
        "continue_on_record_error": continue_on_error,
        "expected_backend": expected_backend,
    }


def _verify_everything(request: dict[str, Any]) -> None:
    """Complete the trust gate before importing code that deserializes assets."""

    _verify_source(request["upstream_root"], request["source_revision"])
    verified_payloads: dict[str, bytes] = {}
    for name in ("vacuum_model", "database", "water_model"):
        path, reported = request[name]
        admitted_basename = OFFICIAL_ARTIFACT_BASENAMES[name]
        if path.name != admitted_basename:
            raise RuntimeError(
                f"{name} basename identifies the wrong artifact role: expected "
                f"{admitted_basename}, observed {path.name}."
            )
        official = OFFICIAL_ARTIFACT_HASHES[name]
        if reported != official:
            raise RuntimeError(
                f"{name} request hash is not the admitted official SHA-256."
            )
        verified_payloads[str(path)] = _verified_bytes(path, official, name)
    for record in request["records"]:
        trajectory_id = record["k_index"] + 1
        expected_names = {
            "vacuum_trajectory": (
                f"250_50ps_load_traj_mol_{trajectory_id}_AC"
            ),
            "water_trajectory": (
                f"250_50ps_load_wat_traj_mol_{trajectory_id}_AC"
            ),
        }
        for name, expected_name in expected_names.items():
            path, reported = record[name]
            if path.name != expected_name:
                raise RuntimeError(
                    f"{name} does not match k_index={record['k_index']}: "
                    f"expected {expected_name}, observed {path.name}"
                )
            expected_path = (
                request["upstream_root"] / TRAJECTORY_DIRECTORY / path.name
            ).resolve()
            if path != expected_path:
                raise RuntimeError(
                    f"{name} must be the pinned upstream trajectory: {expected_path}"
                )
            payload = _verified_bytes(
                path, reported, f"{name}[{record['k_index']}]"
            )
            relative = path.relative_to(request["upstream_root"])
            expected_oid = _head_blob_oid(request["upstream_root"], relative)
            observed_oid = _git_blob_oid(payload)
            if observed_oid != expected_oid:
                raise RuntimeError(
                    f"{name} is not the exact blob from pinned HEAD: {relative}"
                )
            verified_payloads[str(path)] = payload
    request["verified_payloads"] = verified_payloads


def _load_verified(payload: bytes, pickle_module: Any) -> Any:
    return pickle_module.load(io.BytesIO(payload))


def _leaf_dtypes(tree: object, jax: Any) -> list[str]:
    names = {
        str(cast(Any, leaf).dtype)
        for leaf in jax.tree_util.tree_leaves(tree)
        if getattr(leaf, "dtype", None) is not None
    }
    return sorted(names)


def _verify_runtime_versions(backend: str) -> dict[str, object]:
    python_version = sys.version.split()[0]
    if python_version != ADMITTED_PYTHON_VERSION:
        raise RuntimeError(
            "ReSolv Python runtime drifted: expected "
            f"{ADMITTED_PYTHON_VERSION}, observed {python_version}."
        )
    observed = {
        distribution: importlib.metadata.version(distribution)
        for distribution in ADMITTED_RUNTIME_VERSIONS
    }
    mismatches: dict[str, dict[str, object]] = {
        distribution: {
            "expected": expected,
            "observed": observed[distribution],
        }
        for distribution, expected in ADMITTED_RUNTIME_VERSIONS.items()
        if observed[distribution] != expected
    }
    jaxlib_version = importlib.metadata.version("jaxlib")
    allowed_jaxlib = ADMITTED_JAXLIB_VERSIONS.get(backend)
    if allowed_jaxlib is None or jaxlib_version not in allowed_jaxlib:
        mismatches["jaxlib"] = {
            "expected": sorted(allowed_jaxlib or ()),
            "observed": jaxlib_version,
        }
    if mismatches:
        raise RuntimeError(
            "ReSolv runtime package versions differ from the admitted "
            f"environment: {json.dumps(mismatches, sort_keys=True)}"
        )
    return {
        "contract": "maple-resolv-runtime-2026-07-31",
        "python": python_version,
        "packages": {**observed, "jaxlib": jaxlib_version},
    }


def _as_float(value: object) -> float:
    np = cast(Any, importlib.import_module("numpy"))

    result = float(np.asarray(value))
    if not math.isfinite(result):
        raise RuntimeError("ReSolv returned a non-finite BAR result.")
    return result


def _run_from_snapshot(
    request: dict[str, Any], source_snapshot: dict[str, Any]
) -> dict[str, Any]:
    # Set x64 before importing any upstream module (upstream imports JAX itself).
    os.environ["JAX_ENABLE_X64"] = "True"
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    jax = cast(Any, importlib.import_module("jax"))

    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")
    if str(jax.config.jax_default_matmul_precision) != "highest":
        raise RuntimeError("ReSolv JAX matmul precision policy was not applied.")
    dill = cast(Any, importlib.import_module("dill"))
    jnp = cast(Any, importlib.import_module("jax.numpy"))
    np = cast(Any, importlib.import_module("numpy"))
    reweighting = cast(Any, importlib.import_module("chemtrain.reweighting"))
    traj_util = cast(Any, importlib.import_module("chemtrain.traj_util"))
    custom_quantity = cast(
        Any, importlib.import_module("chemtrain.jax_md_mod.custom_quantity")
    )
    Chem = cast(Any, importlib.import_module("rdkit.Chem"))
    Initialization = cast(Any, importlib.import_module("util.Initialization"))

    snapshot_root = Path(source_snapshot["_snapshot_root"])
    for module in (reweighting, traj_util, custom_quantity, Initialization):
        module_file = getattr(module, "__file__", None)
        if not isinstance(module_file, str):
            raise TypeError("ReSolv snapshot module has no filesystem origin.")
        module_path = Path(module_file).resolve()
        try:
            module_path.relative_to(snapshot_root)
        except ValueError:
            raise RuntimeError(
                f"ReSolv upstream import escaped pinned snapshot: {module_path}"
            )

    devices = list(jax.devices())
    backend = jax.default_backend()
    device_platforms = sorted({str(device.platform) for device in devices})
    if not devices or device_platforms != [backend]:
        raise RuntimeError(
            "JAX backend/device attestation is inconsistent: "
            f"backend={backend}, device_platforms={device_platforms}."
        )
    if request["expected_backend"] is not None and backend != request["expected_backend"]:
        raise RuntimeError(
            f"Requested JAX backend {request['expected_backend']}, observed {backend}; "
            "backend fallback is forbidden."
        )
    runtime_environment = _verify_runtime_versions(backend)

    vacuum_path = request["vacuum_model"][0]
    water_path = request["water_model"][0]
    verified_payloads = request["verified_payloads"]
    vacuum_params = _load_verified(verified_payloads[str(vacuum_path)], pickle)
    water_params = _load_verified(verified_payloads[str(water_path)], pickle)

    # initialize_simulation normally reloads U_vac.  Return the already verified
    # and loaded object instead so both checkpoint payloads are deserialized once.
    original_pickle_load = Initialization.pickle.load

    def cached_vacuum_load(_handle: object) -> Any:
        return vacuum_params

    Initialization.pickle.load = cached_vacuum_load

    contexts: dict[int, tuple[Any, Any, Any]] = {}

    def context(atom_count: int, init_pos: Any, masses: Any, species: Any) -> tuple[Any, Any, Any]:
        cached = contexts.get(atom_count)
        if cached is not None:
            return cached
        simulation = Initialization.InitializationClass(
            r_init=init_pos,
            box=jnp.eye(3, dtype=jnp.float64) * 1000.0,
            kbt=KBT_KCAL_MOL,
            masses=masses,
            dt=(0.001 * round(10**3 / 48.8882129, 4)),
            species=species,
        )
        neighbor, loaded_vacuum, simulation_fns, _, _ = Initialization.initialize_simulation(
            simulation,
            "Nequip_HFE",
            target_dict={"free_energy_difference": 0.0},
            integrator="Langevin",
            kbt_dependent=False,
            vac_model_path=str(vacuum_path),
            load_trajectories=True,
            loaded_params=water_params,
        )
        _, energy_fn_template, _ = simulation_fns
        cached = (neighbor, loaded_vacuum, energy_fn_template)
        contexts[atom_count] = cached
        return cached

    quantities_cache: dict[int, tuple[Any, Any]] = {}

    def evaluate(record: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        molecule = Chem.MolFromSmiles(record["smiles"])
        if molecule is None:
            raise ValueError(f"RDKit rejected SMILES: {record['smiles']}")
        molecule = Chem.AddHs(molecule)
        species_np = np.asarray(
            [
                cast(Any, atom).GetAtomicNum()
                for atom in cast(Any, molecule).GetAtoms()
            ],
            dtype=np.int32,
        )
        masses_np = np.asarray(
            [
                cast(Any, atom).GetMass()
                for atom in cast(Any, molecule).GetAtoms()
            ],
            dtype=np.float64,
        )

        vacuum_data = _load_verified(
            verified_payloads[str(record["vacuum_trajectory"][0])], dill
        )
        water_data = _load_verified(
            verified_payloads[str(record["water_trajectory"][0])], dill
        )

        atom_count = int(species_np.size)
        for label, data in (("vacuum", vacuum_data), ("water", water_data)):
            position_shape = tuple(np.asarray(data["sim_state"].position).shape)
            if position_shape != (atom_count, 3):
                raise RuntimeError(
                    f"{label} trajectory atom order disagrees with RDKit AddHs: "
                    f"{position_shape} versus {(atom_count, 3)}."
                )
            snapshots = int(np.asarray(data["aux"]["energy"]).size)
            if snapshots != EXPECTED_SNAPSHOTS_PER_ENDPOINT:
                raise RuntimeError(
                    f"{label} trajectory has {snapshots} snapshots; expected exactly 40."
                )
            if bool(np.asarray(data["over_flow"])):
                raise RuntimeError(
                    f"{label} trajectory records a neighbor-list overflow."
                )
            stored_energy = np.asarray(data["aux"]["energy"], dtype=np.float64)
            if not np.all(np.isfinite(stored_energy)):
                raise RuntimeError(
                    f"{label} trajectory contains a non-finite stored energy."
                )

        species = jnp.asarray(species_np, dtype=jnp.int32)
        masses = jnp.asarray(masses_np, dtype=jnp.float64)
        # Stored states are fractional coordinates. initialize_simulation scales
        # its input by the box when fractional=True, so supply the corresponding
        # Cartesian coordinates rather than scaling the ~0.5 values twice.
        init_pos = jnp.asarray(vacuum_data["sim_state"].position) * 1000.0
        neighbor_template, loaded_vacuum, energy_fn_template = context(
            atom_count, init_pos, masses, species
        )

        def trajectory_state(data: dict[str, Any], params: Any) -> Any:
            neighbor = neighbor_template.update(data["sim_state"].position)
            if bool(np.asarray(neighbor.did_buffer_overflow)):
                raise RuntimeError("ReSolv neighbor-list capacity overflowed.")
            trajectory_state_class = cast(Any, traj_util.TrajectoryStateBAR)
            return trajectory_state_class(
                sim_state=(data["sim_state"], neighbor),
                trajectory=data["trajectory"],
                overflow=data["over_flow"],
                thermostat_kbt=data["thermostat_kbt"],
                barostat_press=data["barostat_press"],
                entropy_diff=data["ds"],
                free_energy_diff=data["df"],
                energy_params=params,
                aux=data["aux"],
            )

        vacuum_traj = trajectory_state(vacuum_data, loaded_vacuum)
        water_traj = trajectory_state(water_data, water_params)
        if atom_count not in quantities_cache:
            quantity = {"energy": custom_quantity.energy_wrapper(energy_fn_template)}
            bar = reweighting.init_bar(
                energy_fn_template,
                KBT_KCAL_MOL,
                energy_batch_size=10,
                max_iter_bar=50,
            )
            quantities_cache[atom_count] = (quantity, bar)
        quantity, bar = quantities_cache[atom_count]

        def endpoint_energy(trajectory: Any, params: Any, label: str) -> Any:
            energy = traj_util.quantity_traj_HFE(
                trajectory, quantity, params, species, batch_size=5
            )["energy"]
            jax.block_until_ready(energy)
            if str(getattr(energy, "dtype", "")) != "float64":
                raise RuntimeError(
                    f"ReSolv {label} energy must be float64; observed "
                    f"{getattr(energy, 'dtype', 'unknown')}."
                )
            if not np.all(np.isfinite(np.asarray(energy))):
                raise RuntimeError(f"ReSolv {label} energy contains non-finite values.")
            return energy

        recomputed_energies = {
            "vacuum_on_vacuum": endpoint_energy(
                vacuum_traj, loaded_vacuum, "vacuum_on_vacuum"
            ),
            "water_on_vacuum": endpoint_energy(
                vacuum_traj, water_params, "water_on_vacuum"
            ),
            "vacuum_on_water": endpoint_energy(
                water_traj, loaded_vacuum, "vacuum_on_water"
            ),
            "water_on_water": endpoint_energy(
                water_traj, water_params, "water_on_water"
            ),
        }
        stored_vacuum = jnp.asarray(
            vacuum_traj.aux["energy"], dtype=jnp.float64
        )
        stored_water = jnp.asarray(water_traj.aux["energy"], dtype=jnp.float64)
        # These are the exact four arrays consumed by upstream init_bar:
        # stored self energies V0/Vp and recomputed cross energies rVp/rV0.
        bar_endpoint_energies = {
            "vacuum_on_vacuum": stored_vacuum,
            "water_on_vacuum": recomputed_energies["water_on_vacuum"],
            "vacuum_on_water": recomputed_energies["vacuum_on_water"],
            "water_on_water": stored_water,
        }
        for label, value in bar_endpoint_energies.items():
            if str(getattr(value, "dtype", "")) != "float64":
                raise RuntimeError(
                    f"ReSolv exact BAR input {label} must be float64; observed "
                    f"{getattr(value, 'dtype', 'unknown')}."
                )
        self_diffs = [
            float(
                np.max(
                    np.abs(
                        np.asarray(recomputed_energies["vacuum_on_vacuum"])
                        - np.asarray(stored_vacuum)
                    )
                )
            ),
            float(
                np.max(
                    np.abs(
                        np.asarray(recomputed_energies["water_on_water"])
                        - np.asarray(stored_water)
                    )
                )
            ),
        ]
        if max(self_diffs) > MAX_SELF_ENERGY_DRIFT_KCAL_MOL:
            raise RuntimeError(
                "ReSolv stored endpoint energies do not match the pinned "
                f"checkpoint: max drift {max(self_diffs):.12g} kcal/mol exceeds "
                f"{MAX_SELF_ENERGY_DRIFT_KCAL_MOL:.1e}."
            )

        delta_v_vacuum = (
            bar_endpoint_energies["water_on_vacuum"]
            - bar_endpoint_energies["vacuum_on_vacuum"]
        )
        delta_v_water = (
            bar_endpoint_energies["water_on_water"]
            - bar_endpoint_energies["vacuum_on_water"]
        )
        beta = jnp.asarray(1.0 / KBT_KCAL_MOL, dtype=jnp.float64)
        dimensionless_work_vacuum = beta * delta_v_vacuum
        dimensionless_work_water = beta * delta_v_water
        derived_bar_arrays = {
            "delta_v_vacuum": delta_v_vacuum,
            "delta_v_water": delta_v_water,
            "dimensionless_work_vacuum": dimensionless_work_vacuum,
            "dimensionless_work_water": dimensionless_work_water,
        }
        for label, value in derived_bar_arrays.items():
            if str(getattr(value, "dtype", "")) != "float64":
                raise RuntimeError(
                    f"ReSolv derived BAR array {label} must be float64; "
                    f"observed {getattr(value, 'dtype', 'unknown')}."
                )

        delta_g, upstream_entropy_raw = bar(vacuum_traj, water_traj, species)
        jax.block_until_ready(delta_g)
        jax.block_until_ready(upstream_entropy_raw)
        if str(getattr(delta_g, "dtype", "")) != "float64" or str(
            getattr(upstream_entropy_raw, "dtype", "")
        ) != "float64":
            raise RuntimeError(
                "ReSolv BAR outputs must execute in float64; observed "
                f"delta={getattr(delta_g, 'dtype', 'unknown')}, "
                "raw upstream entropy-like output="
                f"{getattr(upstream_entropy_raw, 'dtype', 'unknown')}."
            )
        record_runtime = time.perf_counter() - started
        return {
            "k_index": record["k_index"],
            "smiles": record["smiles"],
            "delta_g_kcal_mol": _as_float(delta_g),
            "upstream_entropy_like_raw_unverified_units": _as_float(
                upstream_entropy_raw
            ),
            "upstream_entropy_like_interpretation": (
                "not published as entropy or TdS because the pinned upstream "
                "implementation does not establish a verified output unit"
            ),
            "bar_output_dtype": str(getattr(delta_g, "dtype", "unknown")),
            "snapshots_per_endpoint": EXPECTED_SNAPSHOTS_PER_ENDPOINT,
            "self_energy_max_abs_diff_kcal_mol": max(self_diffs),
            "vacuum_self_max_abs_diff_kcal_mol": self_diffs[0],
            "water_self_max_abs_diff_kcal_mol": self_diffs[1],
            "bar_diagnostics": {
                "endpoint_energy_dtypes": {
                    key: str(value.dtype)
                    for key, value in bar_endpoint_energies.items()
                },
                "endpoint_energy_sources": {
                    "vacuum_on_vacuum": "stored_vacuum_traj_aux_energy_V0",
                    "water_on_vacuum": "recomputed_cross_rVp",
                    "vacuum_on_water": "recomputed_cross_rV0",
                    "water_on_water": "stored_water_traj_aux_energy_Vp",
                },
                "derived_array_dtypes": {
                    key: str(value.dtype) for key, value in derived_bar_arrays.items()
                },
                "endpoint_energies_kcal_mol": {
                    key: np.asarray(value).tolist()
                    for key, value in bar_endpoint_energies.items()
                },
                "recomputed_self_energies_kcal_mol": {
                    "vacuum_on_vacuum": np.asarray(
                        recomputed_energies["vacuum_on_vacuum"]
                    ).tolist(),
                    "water_on_water": np.asarray(
                        recomputed_energies["water_on_water"]
                    ).tolist(),
                },
                "delta_v_kcal_mol": {
                    "vacuum_ensemble": np.asarray(delta_v_vacuum).tolist(),
                    "water_ensemble": np.asarray(delta_v_water).tolist(),
                },
                "dimensionless_work": {
                    "vacuum_ensemble": np.asarray(
                        dimensionless_work_vacuum
                    ).tolist(),
                    "water_ensemble": np.asarray(
                        dimensionless_work_water
                    ).tolist(),
                },
                "uncertainty": {
                    "status": "not_computed",
                    "reason": (
                        "The pinned upstream endpoint artifacts and BAR routine do "
                        "not provide a validated uncertainty estimator."
                    ),
                },
            },
            "runtime_s": record_runtime,
            "runtime_seconds": record_runtime,
        }

    results: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    all_started = time.perf_counter()
    try:
        for record in request["records"]:
            try:
                results.append(evaluate(record))
            except Exception as exc:
                if not request["continue_on_record_error"]:
                    raise
                errors.append(
                    {
                        "k_index": record["k_index"],
                        "smiles": record["smiles"],
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
    finally:
        Initialization.pickle.load = original_pickle_load

    total_runtime = time.perf_counter() - all_started
    return {
        "ok": not errors,
        "source_revision": request["source_revision"],
        "temperature_k": TEMPERATURE_K,
        "protocol": "official-precomputed-40-frame-endpoint-BAR",
        "records": results,
        "record_errors": errors,
        "runtime_s": total_runtime,
        "runtime": {
            "total_seconds": total_runtime,
            "records_completed": len(results),
            "records_failed": len(errors),
        },
        "dtype": {
            "jax_enable_x64": bool(jax.config.jax_enable_x64),
            "vacuum_parameter_dtypes": _leaf_dtypes(vacuum_params, jax),
            "water_parameter_dtypes": _leaf_dtypes(water_params, jax),
        },
        "backend": backend,
        "devices": [str(device) for device in devices],
        "platform_attestation": {
            "requested_backend": request["expected_backend"],
            "default_backend": backend,
            "device_platforms": device_platforms,
            "device_count": len(devices),
            "local_device_count": int(jax.local_device_count()),
            "process_index": int(jax.process_index()),
            "process_count": int(jax.process_count()),
            "devices": [
                {
                    "id": int(device.id),
                    "platform": str(device.platform),
                    "device_kind": str(device.device_kind),
                    "process_index": int(device.process_index),
                    "repr": str(device),
                }
                for device in devices
            ],
            "jax_platform_name_environment": os.environ.get("JAX_PLATFORM_NAME"),
            "cuda_visible_devices_environment": os.environ.get(
                "CUDA_VISIBLE_DEVICES"
            ),
            "x64_enabled": bool(jax.config.jax_enable_x64),
            "default_matmul_precision": str(
                jax.config.jax_default_matmul_precision
            ),
            "nvidia_tf32_override_environment": os.environ.get(
                "NVIDIA_TF32_OVERRIDE"
            ),
            "reduced_float32_matmul_disabled": True,
            "fallback_forbidden_when_requested": True,
        },
        "runtime_environment": runtime_environment,
        "source_snapshot": {
            key: value
            for key, value in source_snapshot.items()
            if not key.startswith("_")
        },
    }


def _run_verified(request: dict[str, Any]) -> dict[str, Any]:
    with _pinned_source_snapshot(request["upstream_root"]) as source_snapshot:
        return _run_from_snapshot(request, source_snapshot)


def run(raw_request: object) -> dict[str, Any]:
    request = _normalise_request(raw_request)
    _verify_everything(request)
    return _run_verified(request)


def main() -> int:
    args = _arguments()
    try:
        raw = json.loads(args.request.read_text(encoding="utf-8"))
        response = run(raw)
        status = 0
    except (
        ArithmeticError,
        ImportError,
        KeyError,
        OSError,
        RuntimeError,
        subprocess.SubprocessError,
        tarfile.TarError,
        TypeError,
        ValueError,
        pickle.PickleError,
    ) as exc:
        response = {
            "ok": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        status = 1
    args.response.parent.mkdir(parents=True, exist_ok=True)
    args.response.write_text(
        json.dumps(response, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return status


if __name__ == "__main__":
    raise SystemExit(main())
