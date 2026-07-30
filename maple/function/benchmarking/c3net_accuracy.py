"""Formal, label-free C3Net accuracy adapter.

The adapter keeps source paths out of its registered instance state.  The
isolated accuracy worker supplies only the exact, hash-verified implementation
artifacts registered by :mod:`pretrained_hub`.
"""

from __future__ import annotations

import hashlib
import inspect
import math
import subprocess
import tempfile
from pathlib import Path
from types import MappingProxyType

from maple.function.solvfe.c3net_property import (
    C3NET_CHECKPOINT_SHA256,
    C3NET_EMBEDDING_SHA256,
    C3NET_SOURCE_REVISION,
    C3NET_SOURCE_URL,
    C3NetPropertyAdapter,
)

from .pretrained_hub import (
    AccuracyAdapterArtifactReceipt,
    AccuracyAdapterRegistration,
    AccuracyModelAdapter,
    AccuracyPredictionContext,
    BenchmarkQuantity,
)

C3NET_FORMAL_ACCURACY_ADAPTER_ID = "c3net-formal-v1"
C3NET_SOURCE_BUNDLE_SHA256 = (
    "d13990d5ee93c810438b66ac267267eb648babcfd2befbb6c45987288cd455ac"
)
C3NET_DEPENDENCY_LOCK_SHA256 = (
    "a70be56dc19999cc51be9fb20941ac84ee5f0aa463c7c3ab4683b13766c702b3"
)
C3NET_FORMAL_RUNTIME_LOCK_SHA256 = (
    "3db89924dae337c6cb5de7cabc67f87b7c32ef6a05d5a7321927ff443fc342d6"
)
_GIT_EXECUTABLE = Path("/usr/bin/git")
_REQUIRED_ARTIFACT_ROLES = frozenset(
    {
        "adapter_code",
        "featurizer_code",
        "checkpoint",
        "dependency_lock",
        "embedding",
        "property_adapter_code",
        "runner_code",
        "runtime_lock",
    }
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(path: Path, expected_sha256: str, *, label: str) -> Path:
    source = path.expanduser().resolve(strict=True)
    if not source.is_file():
        raise ValueError(f"Registered C3Net {label} must be a regular file.")
    observed_sha256 = _sha256(source)
    if observed_sha256 != expected_sha256:
        raise ValueError(
            f"Registered C3Net {label} SHA256 mismatch: expected "
            f"{expected_sha256}, observed {observed_sha256}."
        )
    return source


def _run_git(*arguments: str) -> None:
    if not _GIT_EXECUTABLE.is_file():
        raise RuntimeError("Formal C3Net execution requires /usr/bin/git.")
    completed = subprocess.run(
        [str(_GIT_EXECUTABLE), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"Unable to materialize pinned C3Net source: {detail}")


def _materialize_source_checkout(bundle_path: Path, destination: Path) -> Path:
    """Clone only the registered local bundle and restore the official identity."""

    _run_git("clone", "--quiet", str(bundle_path), str(destination))
    _run_git(
        "-C",
        str(destination),
        "checkout",
        "--quiet",
        "--detach",
        C3NET_SOURCE_REVISION,
    )
    _run_git(
        "-C",
        str(destination),
        "remote",
        "set-url",
        "origin",
        C3NET_SOURCE_URL,
    )
    return destination


class C3NetFormalAccuracyAdapter(AccuracyModelAdapter):
    """Execute the pinned C3Net property head on one pure-solvent record."""

    required_artifact_roles = tuple(sorted(_REQUIRED_ARTIFACT_ROLES))
    accuracy_precision_policy = "float32"
    _SOLVENT_ALIASES = MappingProxyType(
        {
            "water": "water",
            "octanol": "1-octanol",
            "1-octanol": "1-octanol",
            "hexadecane": "n-hexadecane",
            "n-hexadecane": "n-hexadecane",
            "hexane": "n-hexane",
            "n-hexane": "n-hexane",
            "methanol": "methanol",
            "ethanol": "ethanol",
            "dmf": "dimethylformamide",
            "dimethylformamide": "dimethylformamide",
        }
    )

    def __init__(self, *, timeout_seconds: float = 180.0):
        timeout = float(timeout_seconds)
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError(
                "C3Net formal-adapter timeout must be finite and positive."
            )
        self.timeout_seconds = timeout

    def predict(self, *, context: AccuracyPredictionContext) -> float:
        if context.target_quantity is not BenchmarkQuantity.PROPERTY_PREDICTION:
            raise ValueError(
                "C3Net formal accuracy is restricted to property_prediction."
            )
        if context.molecular_input.molecular_input_format != "sdf_conformers":
            raise ValueError(
                "C3Net formal accuracy requires a one-to-five-record "
                "sdf_conformers payload."
            )
        if (
            len(context.solvent_components) != 1
            or len(context.solvent_mole_fractions) != 1
            or not math.isclose(
                context.solvent_mole_fractions[0],
                1.0,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
        ):
            raise ValueError(
                "C3Net formal accuracy requires exactly one pure solvent at "
                "mole fraction 1."
            )
        solvent_label = context.solvent_components[0].strip().casefold()
        try:
            solvent = self._SOLVENT_ALIASES[solvent_label]
        except KeyError as exc:
            raise ValueError(
                f"Formal C3Net solvent {solvent_label!r} has no frozen alias."
            ) from exc

        artifact_paths = context.implementation_artifact_paths
        missing_roles = sorted(_REQUIRED_ARTIFACT_ROLES.difference(artifact_paths))
        if missing_roles:
            raise ValueError(
                "C3Net formal adapter lacks registered artifact roles: "
                + ", ".join(missing_roles)
                + "."
            )
        bundle_path = _require_sha256(
            Path(artifact_paths["featurizer_code"]),
            C3NET_SOURCE_BUNDLE_SHA256,
            label="source bundle",
        )
        checkpoint_path = _require_sha256(
            Path(artifact_paths["checkpoint"]),
            C3NET_CHECKPOINT_SHA256,
            label="checkpoint",
        )
        embedding_path = _require_sha256(
            Path(artifact_paths["embedding"]),
            C3NET_EMBEDDING_SHA256,
            label="embedding",
        )
        _require_sha256(
            Path(artifact_paths["dependency_lock"]),
            C3NET_DEPENDENCY_LOCK_SHA256,
            label="dependency lock",
        )
        runtime_lock_path = _require_sha256(
            Path(artifact_paths["runtime_lock"]),
            C3NET_FORMAL_RUNTIME_LOCK_SHA256,
            label="formal runtime lock",
        )
        molecular_payload = context.molecular_input.read()

        with tempfile.TemporaryDirectory(prefix="maple-c3net-formal-") as directory:
            temporary_root = Path(directory)
            source_root = _materialize_source_checkout(
                bundle_path,
                temporary_root / "source",
            )
            cloned_checkpoint = _require_sha256(
                source_root / "prediction/model/checkpoint-1.pth.tar",
                C3NET_CHECKPOINT_SHA256,
                label="cloned checkpoint",
            )
            cloned_embedding = _require_sha256(
                source_root / "module/input/embedding.pt",
                C3NET_EMBEDDING_SHA256,
                label="cloned embedding",
            )
            if (
                cloned_checkpoint.read_bytes() != checkpoint_path.read_bytes()
                or cloned_embedding.read_bytes() != embedding_path.read_bytes()
            ):
                raise ValueError(
                    "C3Net bundle artifacts differ from the separately registered "
                    "checkpoint or embedding."
                )
            sdf_path = temporary_root / "solute-conformers.sdf"
            sdf_path.write_bytes(molecular_payload)
            result = C3NetPropertyAdapter(
                source_root,
                timeout_seconds=self.timeout_seconds,
                runtime_manifest_path=runtime_lock_path,
            ).predict_conformers(sdf_path, solvent)
        return float(result.predicted_solvation_free_energy_kcal_mol)


def build_c3net_formal_accuracy_registration(
    *,
    source_root: str | Path,
    source_bundle: str | Path,
    timeout_seconds: float = 180.0,
) -> AccuracyAdapterRegistration:
    """Build one exact registration from the official clean checkout and bundle."""

    root = Path(source_root).expanduser().resolve(strict=True)
    bundle = _require_sha256(
        Path(source_bundle),
        C3NET_SOURCE_BUNDLE_SHA256,
        label="source bundle",
    )
    checkpoint = _require_sha256(
        root / "prediction/model/checkpoint-1.pth.tar",
        C3NET_CHECKPOINT_SHA256,
        label="checkpoint",
    )
    embedding = _require_sha256(
        root / "module/input/embedding.pt",
        C3NET_EMBEDDING_SHA256,
        label="embedding",
    )
    dependency_lock = _require_sha256(
        root / "environment.yaml",
        C3NET_DEPENDENCY_LOCK_SHA256,
        label="dependency lock",
    )
    runtime_lock = _require_sha256(
        Path(__file__).resolve().parents[3] / "docs/pretrained-solvation-hub/"
        "c3net-formal-runtime-lock-2026-07-30.json",
        C3NET_FORMAL_RUNTIME_LOCK_SHA256,
        label="formal runtime lock",
    )
    property_adapter_source = inspect.getsourcefile(C3NetPropertyAdapter)
    if property_adapter_source is None:
        raise ValueError("C3Net property-adapter source file cannot be resolved.")
    runner_source = inspect.getsourcefile(AccuracyModelAdapter)
    if runner_source is None:
        raise ValueError("Formal accuracy-runner source file cannot be resolved.")
    C3NetPropertyAdapter(
        root,
        timeout_seconds=timeout_seconds,
        runtime_manifest_path=runtime_lock,
    ).assert_source_identity()
    adapter = C3NetFormalAccuracyAdapter(timeout_seconds=timeout_seconds)
    return AccuracyAdapterRegistration.from_components(
        adapter_id=C3NET_FORMAL_ACCURACY_ADAPTER_ID,
        adapter=adapter,
        implementation_artifacts=(
            AccuracyAdapterArtifactReceipt.from_file(
                role="adapter_code",
                path=Path(__file__),
            ),
            AccuracyAdapterArtifactReceipt.from_file(
                role="featurizer_code",
                path=bundle,
            ),
            AccuracyAdapterArtifactReceipt.from_file(
                role="checkpoint",
                path=checkpoint,
            ),
            AccuracyAdapterArtifactReceipt.from_file(
                role="embedding",
                path=embedding,
            ),
            AccuracyAdapterArtifactReceipt.from_file(
                role="property_adapter_code",
                path=Path(property_adapter_source),
            ),
            AccuracyAdapterArtifactReceipt.from_file(
                role="runner_code",
                path=Path(runner_source),
            ),
            AccuracyAdapterArtifactReceipt.from_file(
                role="dependency_lock",
                path=dependency_lock,
            ),
            AccuracyAdapterArtifactReceipt.from_file(
                role="runtime_lock",
                path=runtime_lock,
            ),
        ),
        configuration={
            "source_revision": C3NET_SOURCE_REVISION,
            "source_url": C3NET_SOURCE_URL,
            "source_bundle_sha256": C3NET_SOURCE_BUNDLE_SHA256,
            "checkpoint_sha256": C3NET_CHECKPOINT_SHA256,
            "embedding_sha256": C3NET_EMBEDDING_SHA256,
            "dependency_lock_sha256": C3NET_DEPENDENCY_LOCK_SHA256,
            "runtime_lock_sha256": C3NET_FORMAL_RUNTIME_LOCK_SHA256,
            "timeout_seconds": float(timeout_seconds),
            "target_quantity": BenchmarkQuantity.PROPERTY_PREDICTION.value,
            "conformer_policy": (
                "one to five supplied SDF records; arithmetic mean of the "
                "unmodified upstream predictions"
            ),
        },
    )
