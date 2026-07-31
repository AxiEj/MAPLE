"""Fail-closed contracts for the published ReSolv hydration-FE protocol.

ReSolv is a dedicated endpoint-trajectory/BAR workflow.  It is intentionally
not exposed as an ordinary MAPLE calculator and it does not add a continuum
correction to a gas-phase potential.
"""

from __future__ import annotations

import ast
import hashlib
import math
import pickle
import re
import subprocess
import sys
import types
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path


def _code_object_sha256(code: types.CodeType) -> str:
    """Return a deterministic structural digest of executable module code."""

    digest = hashlib.sha256()

    def update(value: object) -> None:
        if isinstance(value, types.CodeType):
            digest.update(b"code:")
            for field_name in (
                "co_argcount",
                "co_posonlyargcount",
                "co_kwonlyargcount",
                "co_nlocals",
                "co_stacksize",
                "co_flags",
                "co_code",
                "co_consts",
                "co_names",
                "co_varnames",
                "co_filename",
                "co_name",
                "co_qualname",
                "co_firstlineno",
                "co_linetable",
                "co_lnotab",
                "co_exceptiontable",
                "co_freevars",
                "co_cellvars",
            ):
                update(getattr(value, field_name, None))
        elif isinstance(value, tuple):
            digest.update(b"tuple:")
            digest.update(str(len(value)).encode("ascii"))
            for item in value:
                update(item)
        elif isinstance(value, frozenset):
            digest.update(b"frozenset:")
            for item_digest in sorted(
                hashlib.sha256(repr(item).encode("utf-8")).hexdigest()
                for item in value
            ):
                digest.update(item_digest.encode("ascii"))
        elif isinstance(value, bytes):
            digest.update(b"bytes:")
            digest.update(str(len(value)).encode("ascii"))
            digest.update(value)
        elif isinstance(value, str):
            encoded = value.encode("utf-8")
            digest.update(b"str:")
            digest.update(str(len(encoded)).encode("ascii"))
            digest.update(encoded)
        elif value is None or value is Ellipsis or isinstance(
            value, (bool, int, float, complex)
        ):
            digest.update(type(value).__name__.encode("ascii"))
            digest.update(b":")
            digest.update(repr(value).encode("ascii"))
        else:
            raise TypeError(
                f"Unsupported code constant type: {type(value).__name__}"
            )
        digest.update(b";")

    update(code)
    return digest.hexdigest()


RESOLV_UPSTREAM_REVISION = "1d85bcc065003e083d2e95ab7091cb1762eeb1bf"
RESOLV_VACUUM_MODEL_SHA256 = (
    "83618b7b4f6680e80936f68e5e2d02f9c6ff1c62750333ca28fc5733286347c4"
)
RESOLV_WATER_MODEL_SHA256 = (
    "56a72b33e7aa3b26e06f7792924f2630a62f1d0a776f2843f5871195e88e3e13"
)
RESOLV_DATABASE_SHA256 = (
    "8a1dd006a54f0986f58b967bf7046fb72293dafa66954ec8529cdc5e68b4d405"
)
RESOLV_FULL_TEST_MANIFEST_SHA256 = (
    "cf529c1ffb8940d54e849e9b13f5391b68a47124823c92675d59aab6815c1418"
)
RESOLV_IMPORTED_MODULE_CODE_SHA256 = _code_object_sha256(
    sys._getframe().f_code
)

RESOLV_VACUUM_MODEL_RELATIVE = Path(
    "examples/FreeEnergyScripts/savedTrainers/"
    "261023_QM7x_Nequip_Nequip_QM7x_All_8epochs_iL5emin3_"
    "lrdecay1emin3_scaledTargets_LargeTrainingSet_Cutoff4A_"
    "mlp4_ShiftFalse_ScaleFalse_EnergiesAndForces.pkl"
)
RESOLV_WATER_MODEL_RELATIVE = Path(
    "examples/FreeEnergyScripts/QM7x Scripts/TrainFreeEnergy/checkpoints/"
    "080524_t_prod_250ps_t_equil_50ps_iL1e-06_lrd0.1_"
    "epochs500_seed7_train_389mem_0.97_epoch499.pkl"
)
RESOLV_DATABASE_RELATIVE = Path(
    "examples/FreeEnergyScripts/QM7x Scripts/FreeSolvDB/database.pickle"
)
RESOLV_TRAJECTORY_ROOT_RELATIVE = Path(
    "examples/FreeEnergyScripts/QM7x Scripts/TrainFreeEnergy/"
    "precomputed_trajectories"
)
RESOLV_TRAINING_UTILS_RELATIVE = Path(
    "examples/FreeEnergyScripts/QM7x Scripts/TrainFreeEnergy/"
    "training_utils_HFE.py"
)
RESOLV_TRAJECTORY_GENERATOR_RELATIVE = Path(
    "examples/FreeEnergyScripts/QM7x Scripts/TrainFreeEnergy/"
    "BAR_HFE_trajectory_generator.py"
)

RESOLV_PROTOCOL_METADATA: dict[str, object] = {
    "protocol_kind": "dedicated_hydration_free_energy",
    "estimator": "BAR",
    "ordinary_calculator": False,
    "additive_continuum": False,
    "fresh_conformer_generation": False,
    "supports_opt_freq_md": False,
    "solvent": "water",
    "temperature_kelvin": 298.15,
    "total_charge": 0,
    "multiplicity": 1,
    "elements": ("H", "C", "N", "O", "S", "Cl"),
    "trajectory_source": "official_precomputed_endpoint_trajectories",
    "snapshots_per_endpoint": 40,
}

_OFFICIAL_ARTIFACT_SHA256 = {
    "vacuum_model": RESOLV_VACUUM_MODEL_SHA256,
    "water_model": RESOLV_WATER_MODEL_SHA256,
    "database": RESOLV_DATABASE_SHA256,
}
_RELEVANT_SMILES_EXCLUSION = re.compile(
    r"[AaBbDdEeFfGgIiJjKkMmPpQqRrTtUuVvWwXxYyZz]"
)


class ReSolvProtocolError(RuntimeError):
    """The pinned ReSolv protocol or one of its artifacts is invalid."""


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise ReSolvProtocolError(f"Missing ReSolv {label} artifact: {path}")
    observed = sha256_file(path)
    if observed != expected:
        raise ReSolvProtocolError(
            f"ReSolv SHA256 mismatch for {label}: expected {expected}, "
            f"observed {observed}."
        )
    return observed


def _finite_database_float(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReSolvProtocolError(f"{label} must be a finite number.")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ReSolvProtocolError(f"{label} must be finite.")
    return parsed


def _git_revision(root: Path) -> str:
    if not (root / ".git").exists():
        raise ReSolvProtocolError(
            "ReSolv upstream root must be a Git checkout so its revision can "
            f"be verified: {root}"
        )
    completed = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=False,
    )
    revision = completed.stdout.strip()
    if completed.returncode != 0 or len(revision) != 40:
        raise ReSolvProtocolError(
            f"Unable to verify ReSolv source revision at {root}: "
            f"{completed.stderr.strip() or 'unknown Git error'}"
        )
    return revision


def _read_verified_head_blob(root: Path, path: Path) -> bytes:
    """Read once and require those exact bytes to match the current HEAD blob."""

    try:
        relative = path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise ReSolvProtocolError(
            f"ReSolv source file lies outside its pinned checkout: {path}"
        ) from exc
    expected_run = subprocess.run(
        ["git", "-C", str(root), "rev-parse", f"HEAD:{relative}"],
        text=True,
        capture_output=True,
        check=False,
    )
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise ReSolvProtocolError(
            f"Unable to read pinned ReSolv source file: {relative}."
        ) from exc
    observed_run = subprocess.run(
        ["git", "-C", str(root), "hash-object", "--stdin"],
        input=payload,
        capture_output=True,
        check=False,
    )
    expected = expected_run.stdout.strip()
    observed = observed_run.stdout.decode("ascii", errors="replace").strip()
    if (
        expected_run.returncode != 0
        or observed_run.returncode != 0
        or not expected
        or expected != observed
    ):
        raise ReSolvProtocolError(
            "ReSolv source file differs from the pinned HEAD blob: "
            f"{relative}; expected {expected or 'unknown'}, "
            f"observed {observed or 'unknown'}."
        )
    return payload


@dataclass(frozen=True)
class ReSolvArtifactSet:
    """Paths forming one auditable ReSolv installation."""

    upstream_root: Path
    vacuum_model_path: Path | None = None
    water_model_path: Path | None = None
    database_path: Path | None = None
    trajectory_root: Path | None = None

    def __post_init__(self) -> None:
        root = Path(self.upstream_root).expanduser().resolve()
        object.__setattr__(self, "upstream_root", root)
        defaults = (
            ("vacuum_model_path", RESOLV_VACUUM_MODEL_RELATIVE),
            ("water_model_path", RESOLV_WATER_MODEL_RELATIVE),
            ("database_path", RESOLV_DATABASE_RELATIVE),
            ("trajectory_root", RESOLV_TRAJECTORY_ROOT_RELATIVE),
        )
        for field_name, relative in defaults:
            supplied = getattr(self, field_name)
            resolved = (
                root / relative
                if supplied is None
                else Path(supplied).expanduser().resolve()
            )
            object.__setattr__(self, field_name, resolved)

    def vacuum_trajectory_path(self, k_index: int) -> Path:
        trajectory_id = _trajectory_id(k_index)
        assert self.trajectory_root is not None
        return self.trajectory_root / (
            f"250_50ps_load_traj_mol_{trajectory_id}_AC"
        )

    def water_trajectory_path(self, k_index: int) -> Path:
        trajectory_id = _trajectory_id(k_index)
        assert self.trajectory_root is not None
        return self.trajectory_root / (
            f"250_50ps_load_wat_traj_mol_{trajectory_id}_AC"
        )

    def validate(
        self,
        *,
        expected_revision: str = RESOLV_UPSTREAM_REVISION,
        expected_sha256: Mapping[str, str] = _OFFICIAL_ARTIFACT_SHA256,
        required_k_indices: Iterable[int] = (),
    ) -> dict[str, object]:
        """Verify revision, model/database hashes, and endpoint-pair presence."""

        required_indices = tuple(required_k_indices)
        revision = _git_revision(self.upstream_root)
        if revision != expected_revision:
            raise ReSolvProtocolError(
                "ReSolv source revision mismatch: expected "
                f"{expected_revision}, observed {revision}."
            )

        required_hash_names = ("vacuum_model", "water_model", "database")
        missing_hashes = [name for name in required_hash_names if name not in expected_sha256]
        if missing_hashes:
            raise ReSolvProtocolError(
                "Missing expected ReSolv artifact SHA256 value(s): "
                + ", ".join(missing_hashes)
            )

        assert self.vacuum_model_path is not None
        assert self.water_model_path is not None
        assert self.database_path is not None
        observed_hashes = {
            "vacuum_model": _require_sha256(
                self.vacuum_model_path,
                expected_sha256["vacuum_model"],
                "U_vac/vacuum model",
            ),
            "water_model": _require_sha256(
                self.water_model_path,
                expected_sha256["water_model"],
                "U_wat/water model",
            ),
            "database": _require_sha256(
                self.database_path,
                expected_sha256["database"],
                "FreeSolv database",
            ),
        }

        missing_pairs: list[int] = []
        for raw_index in required_indices:
            k_index = _validate_k_index(raw_index)
            if not (
                self.vacuum_trajectory_path(k_index).is_file()
                and self.water_trajectory_path(k_index).is_file()
            ):
                missing_pairs.append(k_index)
        if missing_pairs:
            preview = ", ".join(str(value) for value in missing_pairs[:8])
            raise ReSolvProtocolError(
                "Missing ReSolv endpoint trajectory pair(s) for zero-based "
                f"k_index: {preview}."
            )

        return {
            "source_revision": revision,
            "artifact_sha256": observed_hashes,
            "required_endpoint_pair_count": len(required_indices),
        }


def _validate_k_index(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ReSolvProtocolError(
            f"ReSolv k_index must be a non-negative integer, received {value!r}."
        )
    return value


def _trajectory_id(k_index: int) -> int:
    return _validate_k_index(k_index) + 1


def _literal_assignment(
    path: Path,
    variable_name: str,
    *,
    function_name: str | None = None,
    source_bytes: bytes | None = None,
) -> Sequence[int]:
    try:
        source = (
            path.read_text(encoding="utf-8")
            if source_bytes is None
            else source_bytes.decode("utf-8")
        )
        tree = ast.parse(source, filename=str(path))
    except (OSError, UnicodeDecodeError, SyntaxError) as exc:
        raise ReSolvProtocolError(
            f"Unable to parse pinned ReSolv source list from {path}."
        ) from exc

    nodes: Sequence[ast.stmt] = tree.body
    if function_name is not None:
        functions = [
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == function_name
        ]
        if len(functions) != 1:
            raise ReSolvProtocolError(
                f"Pinned ReSolv source lacks unique function {function_name!r}."
            )
        nodes = functions[0].body

    values: list[object] = []
    for node in nodes:
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == variable_name
            for target in node.targets
        ):
            continue
        try:
            values.append(ast.literal_eval(node.value))
        except (TypeError, ValueError) as exc:
            raise ReSolvProtocolError(
                f"ReSolv source assignment {variable_name!r} is not a literal."
            ) from exc

    if not values or not isinstance(values[-1], (list, tuple)):
        raise ReSolvProtocolError(
            f"Pinned ReSolv source lacks literal list {variable_name!r}."
        )
    parsed = tuple(values[-1])
    if any(isinstance(item, bool) or not isinstance(item, int) for item in parsed):
        raise ReSolvProtocolError(
            f"ReSolv source list {variable_name!r} contains a non-integer."
        )
    return parsed


def _load_verified_database(path: Path) -> Mapping[str, Mapping[str, object]]:
    """Verify and deserialize one immutable copy of the official database."""

    try:
        serialized = path.read_bytes()
    except OSError as exc:
        raise ReSolvProtocolError(
            f"Unable to read the ReSolv FreeSolv database: {path}"
        ) from exc
    observed = hashlib.sha256(serialized).hexdigest()
    if observed != RESOLV_DATABASE_SHA256:
        raise ReSolvProtocolError(
            "ReSolv SHA256 mismatch for FreeSolv database: expected "
            f"{RESOLV_DATABASE_SHA256}, observed {observed}."
        )
    try:
        payload = pickle.loads(serialized)
    except Exception as exc:
        raise ReSolvProtocolError(
            f"Unable to deserialize the verified ReSolv database: {path}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise ReSolvProtocolError("Verified ReSolv database is not a mapping.")
    return payload


def _primary_group(value: object) -> tuple[list[str], str]:
    if not isinstance(value, (list, tuple)):
        groups: list[str] = []
    else:
        groups = [str(item).strip() for item in value if str(item).strip()]
    return groups, groups[0] if groups else "unclassified"


def build_resolv_manifest(
    artifacts: ReSolvArtifactSet,
) -> list[dict[str, object]]:
    """Reconstruct the exact 559-row upstream ordering and 375/162/22 split."""

    revision = _git_revision(artifacts.upstream_root)
    if revision != RESOLV_UPSTREAM_REVISION:
        raise ReSolvProtocolError(
            "ReSolv source revision mismatch while building manifest: "
            f"expected {RESOLV_UPSTREAM_REVISION}, observed {revision}."
        )

    training_utils = artifacts.upstream_root / RESOLV_TRAINING_UTILS_RELATIVE
    trajectory_generator = (
        artifacts.upstream_root / RESOLV_TRAJECTORY_GENERATOR_RELATIVE
    )
    training_utils_bytes = _read_verified_head_blob(
        artifacts.upstream_root,
        training_utils,
    )
    trajectory_generator_bytes = _read_verified_head_blob(
        artifacts.upstream_root,
        trajectory_generator,
    )
    train = set(
        _literal_assignment(
            training_utils,
            "train_dataset_389",
            function_name="get_389_train_dataset",
            source_bytes=training_utils_bytes,
        )
    )
    failed_vacuum = set(
        _literal_assignment(
            trajectory_generator,
            "dataset_for_failed_U_vac",
            source_bytes=trajectory_generator_bytes,
        )
    )
    failed_water = set(
        _literal_assignment(
            trajectory_generator,
            "dataset_for_failed_U_wat",
            source_bytes=trajectory_generator_bytes,
        )
    )
    failed = failed_vacuum | failed_water

    assert artifacts.database_path is not None
    database = _load_verified_database(artifacts.database_path)
    relevant = [
        (str(mobley_id), entry)
        for mobley_id, entry in database.items()
        if isinstance(entry, Mapping)
        and not _RELEVANT_SMILES_EXCLUSION.search(str(entry.get("smiles", "")))
    ]

    records: list[dict[str, object]] = []
    for k_index, (mobley_id, entry) in enumerate(relevant):
        if k_index in failed:
            split = "failed"
        elif k_index in train:
            split = "train"
        else:
            split = "test"
        groups, primary_group = _primary_group(entry.get("groups"))
        records.append(
            {
                "k_index": k_index,
                "trajectory_id": k_index + 1,
                "split": split,
                "mobley_id": mobley_id,
                "smiles": str(entry.get("smiles", "")),
                "iupac": str(entry.get("iupac", "")),
                "groups": groups,
                "primary_functional_group": primary_group,
                "experimental_kcal_mol": _finite_database_float(
                    entry["expt"], f"{mobley_id} expt"
                ),
                "experimental_uncertainty_kcal_mol": _finite_database_float(
                    entry["d_expt"], f"{mobley_id} d_expt"
                ),
                "vacuum_trajectory": str(
                    artifacts.vacuum_trajectory_path(k_index)
                ),
                "water_trajectory": str(
                    artifacts.water_trajectory_path(k_index)
                ),
            }
        )

    split_counts = {
        split: sum(record["split"] == split for record in records)
        for split in ("train", "test", "failed")
    }
    test_group_labels = {
        str(record["primary_functional_group"])
        for record in records
        if record["split"] == "test"
    }
    classified_test_groups = test_group_labels - {"unclassified"}
    unclassified_test_records = sum(
        record["split"] == "test"
        and record["primary_functional_group"] == "unclassified"
        for record in records
    )
    expected_counts = {"train": 375, "test": 162, "failed": 22}
    if len(records) != 559 or split_counts != expected_counts:
        raise ReSolvProtocolError(
            "Pinned ReSolv manifest identity drifted: expected 559 rows and "
            f"{expected_counts}, observed {len(records)} and {split_counts}."
        )
    if len(classified_test_groups) != 26 or unclassified_test_records != 9:
        raise ReSolvProtocolError(
            "Pinned ReSolv test functional-group identity drifted: expected "
            "26 classified primary groups plus 9 unclassified records, observed "
            f"{len(classified_test_groups)} classified groups plus "
            f"{unclassified_test_records} unclassified records."
        )
    return records


class ReSolvProtocolAdapter:
    """Metadata-and-artifact adapter for external ReSolv sidecar execution."""

    protocol_metadata = RESOLV_PROTOCOL_METADATA

    def __init__(self, artifacts: ReSolvArtifactSet):
        self.artifacts = artifacts

    def validate(self, required_k_indices: Iterable[int]) -> dict[str, object]:
        return self.artifacts.validate(required_k_indices=required_k_indices)

    def test_manifest(self) -> list[dict[str, object]]:
        return [
            record
            for record in build_resolv_manifest(self.artifacts)
            if record["split"] == "test"
        ]


__all__ = [
    "RESOLV_DATABASE_SHA256",
    "RESOLV_FULL_TEST_MANIFEST_SHA256",
    "RESOLV_IMPORTED_MODULE_CODE_SHA256",
    "RESOLV_PROTOCOL_METADATA",
    "RESOLV_UPSTREAM_REVISION",
    "RESOLV_VACUUM_MODEL_SHA256",
    "RESOLV_WATER_MODEL_SHA256",
    "ReSolvArtifactSet",
    "ReSolvProtocolAdapter",
    "ReSolvProtocolError",
    "build_resolv_manifest",
    "sha256_file",
]
