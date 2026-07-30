#!/usr/bin/env python3
"""Reproduce AniSolv compact's six-example upstream water sample.

This is deliberately a source-bound development audit, not an independent
benchmark.  It imports the exact upstream sample definitions and official
inference API, runs only the public compact checkpoint on CPU in the reference
``default`` mode, and freezes row-complete correction energies and forces.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import math
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any

AUDITED_ON = "2026-07-30"
UPSTREAM_REPOSITORY = "https://github.com/Ant-on-knee/anisolv"
UPSTREAM_REVISION = "66f36dc7d4134c78b718255bb2abfed14482acb0"
UPSTREAM_TREE = "522a9cfacb6f0fc00935e2d4d140020b8007c295"
CHECKPOINT = {
    "path": "models/model1_compact.pt",
    "size_bytes": 25_522_563,
    "sha256": "b79be343c55a7cc06579aa4778753156bc71f99b3a00f046a270bc3427c1498f",
}
SOURCE_BLOBS = {
    "package_api": {
        "path": "__init__.py",
        "size_bytes": 815,
        "sha256": "df90259889cd313d1520f012f758c30e1ae9873ae7846effb9dad227d7dbd6dd",
    },
    "package_metadata": {
        "path": "pyproject.toml",
        "size_bytes": 1_376,
        "sha256": "c04577d7f8c82374c2abe18a0b28e7e01588d3f9ed1570fdf46c09f279bde9b5",
    },
    "readme": {
        "path": "README.md",
        "size_bytes": 10_895,
        "sha256": "891d59e886d7952f358682ecbbaea461ffc46305733ae8fff603358c4ea3c409",
    },
    "sample": {
        "path": "samples/H2O_single_point.py",
        "size_bytes": 5_218,
        "sha256": "f511925b35298eca868143fa8a98dff18bbf90edef24e276d3421a5d852c5c1b",
    },
    "model": {
        "path": "model.py",
        "size_bytes": 12_476,
        "sha256": "1ccec6adc1d1c61936c1ace6d4dcc42542fe9c33e4263254d0ce46389c0e5a0c",
    },
    "predict": {
        "path": "predict.py",
        "size_bytes": 3_581,
        "sha256": "5194307dbee41f3f112dd3863c65c81009566de8e7031f0483ef0b2a12f0fd05",
    },
    "solvent": {
        "path": "solvent.py",
        "size_bytes": 3_695,
        "sha256": "b69b9ff778a4ec76062121410960c1fe3e6c42025f4f6869fd1e018007e9f0ca",
    },
    "data": {
        "path": "data.py",
        "size_bytes": 6_625,
        "sha256": "9f3fcbb9109c4d3c2cf7bb989aaaec598b5d199a9067844012fe38a9ba7c2fa9",
    },
    "backbone": {
        "path": "_backbone/escn_md.py",
        "size_bytes": 48_766,
        "sha256": "db8c2d04f1a737c793bdccb35980eb7c1485998b590ebc472035e955b5889931",
    },
    "wigner_data": {
        "path": "_backbone/Jd.pt",
        "size_bytes": 21_697,
        "sha256": "b4059c45be246dcb6c49c545670b65c56550eb0c2e7a9c92b4b50a92d370dbe2",
    },
}
PANEL_IDENTITY_SHA256 = (
    "3a1e02417a6b6e47f05b2842f06e080a459771c277fb78343e8f72ece2e91c85"
)
EV_TO_KCAL_MOL = 23.060548
ROUTE4_MAX_ERROR_TARGET_KCAL_MOL = 1.5
ROUTE4_IDEAL_MAX_ERROR_KCAL_MOL = 1.0


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _git_environment() -> dict[str, str]:
    environment = {
        key: value for key, value in os.environ.items() if not key.startswith("GIT_")
    }
    environment.update(
        {
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_LFS_SKIP_SMUDGE": "1",
            "GIT_OPTIONAL_LOCKS": "0",
            "LC_ALL": "C",
        }
    )
    return environment


def _git(root: Path, *args: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            capture_output=True,
            text=True,
            env=_git_environment(),
        )
    except FileNotFoundError as exc:
        raise RuntimeError("The AniSolv audit requires the system `git`.") from exc
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.strip()
        raise ValueError(f"Git inspection failed for {root}: {detail}") from exc
    return completed.stdout.strip()


def _verify_file(root: Path, specification: dict[str, Any], *, label: str) -> None:
    path = root / specification["path"]
    if not path.is_file():
        raise ValueError(f"{label} is not a regular file: {path}")
    if path.stat().st_size != specification["size_bytes"]:
        raise ValueError(f"{label} size does not match the pinned upstream release.")
    if _sha256_file(path) != specification["sha256"]:
        raise ValueError(f"{label} SHA256 does not match the pinned upstream release.")


def _ignored_paths(root: Path) -> list[str]:
    output = _git(
        root,
        "ls-files",
        "--others",
        "--ignored",
        "--exclude-standard",
    )
    return [] if not output else output.splitlines()


def _verify_ignored_paths_are_bytecode_only(paths: list[str]) -> None:
    invalid = []
    for value in paths:
        path = Path(value)
        if (
            path.is_absolute()
            or len(path.parts) < 2
            or ".." in path.parts
            or path.parts[-2] != "__pycache__"
            or path.suffix != ".pyc"
        ):
            invalid.append(value)
    if invalid:
        raise ValueError(
            "AniSolv checkout has ignored non-bytecode artifacts that could shadow "
            f"the pinned runtime: {invalid!r}"
        )


def verify_upstream_identity(upstream_root: Path) -> list[str]:
    """Fail closed unless the complete tracked checkout and pinned inputs match."""

    if _git(upstream_root, "rev-parse", "HEAD") != UPSTREAM_REVISION:
        raise ValueError("AniSolv checkout revision does not match the pinned release.")
    if _git(upstream_root, "rev-parse", "HEAD^{tree}") != UPSTREAM_TREE:
        raise ValueError("AniSolv checkout tree does not match the pinned release.")
    if _git(upstream_root, "status", "--porcelain"):
        raise ValueError("AniSolv checkout has modified or untracked unignored files.")
    ignored_paths = _ignored_paths(upstream_root)
    _verify_ignored_paths_are_bytecode_only(ignored_paths)
    _verify_file(upstream_root, CHECKPOINT, label="AniSolv compact checkpoint")
    for label, specification in SOURCE_BLOBS.items():
        _verify_file(upstream_root, specification, label=f"AniSolv {label}")
    return ignored_paths


def _materialize_pinned_tree(upstream_root: Path, snapshot_root: Path) -> None:
    """Extract only tracked bytes from the verified Git tree into a fresh root."""

    snapshot_root.mkdir()
    archive_path = snapshot_root.parent / "anisolv-pinned-tree.tar"
    try:
        subprocess.run(
            [
                "git",
                "-C",
                str(upstream_root),
                "archive",
                "--format=tar",
                f"--output={archive_path}",
                UPSTREAM_REVISION,
            ],
            check=True,
            capture_output=True,
            env=_git_environment(),
        )
    except FileNotFoundError as exc:
        raise RuntimeError("The AniSolv audit requires the system `git`.") from exc
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(
            f"Could not materialize the pinned AniSolv tree: {detail}"
        ) from exc

    try:
        with tarfile.open(archive_path, mode="r:") as archive:
            for member in archive.getmembers():
                path = Path(member.name)
                if (
                    path.is_absolute()
                    or ".." in path.parts
                    or member.issym()
                    or member.islnk()
                    or not (member.isfile() or member.isdir())
                ):
                    raise ValueError(
                        f"Unsafe entry in pinned AniSolv Git archive: {member.name!r}"
                    )
            archive.extractall(snapshot_root)
    finally:
        archive_path.unlink(missing_ok=True)

    _verify_file(snapshot_root, CHECKPOINT, label="snapshot AniSolv checkpoint")
    for label, specification in SOURCE_BLOBS.items():
        _verify_file(
            snapshot_root,
            specification,
            label=f"snapshot AniSolv {label}",
        )


def _load_upstream_sample(source_root: Path, isolated_parent: Path):
    package_parent = str(isolated_parent.resolve())
    loaded = sorted(
        name for name in sys.modules if name == "anisolv" or name.startswith("anisolv.")
    )
    if loaded:
        raise RuntimeError(
            "AniSolv audit requires a fresh interpreter before source-isolated import; "
            f"already loaded: {loaded!r}"
        )
    if package_parent not in sys.path:
        sys.path.insert(0, package_parent)

    package = importlib.import_module("anisolv")
    package_file = Path(str(getattr(package, "__file__", ""))).resolve()
    expected_package_file = (source_root / "__init__.py").resolve()
    if package_file != expected_package_file:
        raise RuntimeError("Imported AniSolv package is not the isolated pinned tree.")

    sample_path = isolated_parent / "anisolv" / SOURCE_BLOBS["sample"]["path"]
    spec = importlib.util.spec_from_file_location(
        "maple_anisolv_upstream_water_sample", sample_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import the pinned AniSolv sample: {sample_path}")
    sample = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sample)

    source_parent = str(source_root.parent.resolve())
    sys.path[:] = [
        entry for entry in sys.path if str(Path(entry).resolve()) != source_parent
    ]
    if package_parent not in sys.path:
        sys.path.insert(0, package_parent)
    return sample


def _finite_float(value: Any, *, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} is not a finite scalar.") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} is not finite.")
    return result


def _verify_dependency_origin(module: Any, *, label: str) -> str:
    raw_path = getattr(module, "__file__", None)
    if not raw_path:
        raise RuntimeError(f"{label} has no auditable module origin.")
    path = Path(str(raw_path)).resolve()
    prefix = Path(sys.prefix).resolve()
    if prefix != path and prefix not in path.parents:
        raise RuntimeError(
            f"{label} was not imported from the active Python environment: {path}"
        )
    return path.relative_to(prefix).as_posix()


def _verified_anisolv_module_origins(source_root: Path) -> dict[str, str]:
    origins = {}
    root = source_root.resolve()
    for name, module in sorted(sys.modules.items()):
        if name != "anisolv" and not name.startswith("anisolv."):
            continue
        raw_path = getattr(module, "__file__", None)
        if not raw_path:
            raise RuntimeError(f"Loaded AniSolv module {name!r} has no source origin.")
        path = Path(str(raw_path)).resolve()
        if path != root / "__init__.py" and root not in path.parents:
            raise RuntimeError(
                f"Loaded AniSolv module {name!r} escaped the pinned tree: {path}"
            )
        origins[name] = path.relative_to(root).as_posix()
    if "anisolv" not in origins:
        raise RuntimeError("The pinned AniSolv package was not loaded.")
    return origins


def _configure_no_reduced_precision(torch) -> dict[str, Any]:
    """Lock out optional reduced-precision accelerator controls."""

    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    if hasattr(torch.backends.cuda.matmul, "allow_fp16_accumulation"):
        torch.backends.cuda.matmul.allow_fp16_accumulation = False
    if torch.is_autocast_enabled():
        raise RuntimeError(
            "CUDA autocast must be disabled for the CPU reference audit."
        )
    if hasattr(torch, "is_autocast_enabled") and torch.is_autocast_enabled("cpu"):
        raise RuntimeError("CPU autocast must be disabled for the CPU reference audit.")
    return {
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "fp16_reduced_precision_reduction": bool(
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
        ),
        "bf16_reduced_precision_reduction": bool(
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        ),
        "fp16_accumulation": bool(
            getattr(torch.backends.cuda.matmul, "allow_fp16_accumulation", False)
        ),
        "autocast_cpu": bool(torch.is_autocast_enabled("cpu")),
        "autocast_cuda": bool(torch.is_autocast_enabled()),
    }


def _panel_identity(sample) -> tuple[list[dict[str, Any]], str]:
    panel = []
    for source_row in sample.SOLUTES:
        g2_name, label = source_row[:2]
        numbers, positions, formula = sample._resolve(g2_name)
        panel.append(
            {
                "atomic_numbers": numbers,
                "formula": formula,
                "g2_name": g2_name,
                "label": label,
                "positions_angstrom": positions,
            }
        )
    digest = _sha256_bytes(_canonical_bytes(panel))
    if digest != PANEL_IDENTITY_SHA256:
        raise ValueError(
            "AniSolv upstream sample geometry/source-row identity changed; audit aborted."
        )
    return panel, digest


def _loaded_model_fingerprint(source_root: Path) -> dict[str, Any]:
    predict_module = importlib.import_module("anisolv.predict")
    cache = getattr(predict_module, "_MODEL_CACHE", None)
    if not isinstance(cache, dict) or len(cache) != 1:
        raise RuntimeError("Expected exactly one AniSolv model in the official cache.")
    cache_key, model = next(iter(cache.items()))
    expected_checkpoint = str((source_root / CHECKPOINT["path"]).resolve())
    if not isinstance(cache_key, tuple) or cache_key[0] != expected_checkpoint:
        raise RuntimeError(
            "Loaded AniSolv checkpoint path is not the pinned compact file."
        )
    if cache_key[1:3] != ("cpu", "torch.float32"):
        raise RuntimeError("AniSolv reference model did not load as CPU float32.")
    settings = str(cache_key[3])
    for required in (
        "tf32=False",
        "compile=False",
        "merge_mole=False",
        "execution_mode='general'",
    ):
        if required not in settings:
            raise RuntimeError(
                f"AniSolv default inference settings lost required token {required!r}."
            )
    parameter_dtypes = sorted({str(value.dtype) for value in model.parameters()})
    parameter_devices = sorted({str(value.device) for value in model.parameters()})
    buffer_dtypes = sorted({str(value.dtype) for value in model.buffers()})
    buffer_devices = sorted({str(value.device) for value in model.buffers()})
    if parameter_dtypes != ["torch.float32"] or parameter_devices != ["cpu"]:
        raise RuntimeError("AniSolv model parameters are not entirely CPU float32.")
    if buffer_devices != ["cpu"] or any(
        dtype not in {"torch.float32", "torch.int64"} for dtype in buffer_dtypes
    ):
        raise RuntimeError("AniSolv model buffers have an unexpected device or dtype.")
    return {
        "buffer_devices": buffer_devices,
        "buffer_dtypes": buffer_dtypes,
        "cache_key": {
            "checkpoint": CHECKPOINT["path"],
            "device": str(cache_key[1]),
            "dtype": str(cache_key[2]),
            "inference_settings": str(cache_key[3]),
        },
        "parameter_devices": parameter_devices,
        "parameter_dtypes": parameter_dtypes,
    }


def _execute_sample(
    source_root: Path,
    isolated_parent: Path,
) -> dict[str, Any]:
    import ase
    import numpy as np
    import torch

    dependency_origins = {
        "ase": _verify_dependency_origin(ase, label="ASE"),
        "numpy": _verify_dependency_origin(np, label="NumPy"),
        "torch": _verify_dependency_origin(torch, label="PyTorch"),
    }
    sample = _load_upstream_sample(source_root, isolated_parent)
    precision_state = _configure_no_reduced_precision(torch)
    if float(sample.EV_TO_KCAL) != EV_TO_KCAL_MOL:
        raise ValueError("AniSolv sample energy conversion constant changed.")
    panel, panel_sha256 = _panel_identity(sample)
    checkpoint = str((source_root / CHECKPOINT["path"]).resolve())

    water = next(item for item in panel if item["g2_name"] == "H2O")
    vacuum_energy_ev, vacuum_forces = sample.predict_solvation_energy(
        (water["atomic_numbers"], water["positions_angstrom"]),
        charge=0,
        spin=1,
        solvent=None,
        checkpoint=checkpoint,
        device="cpu",
        dtype=torch.float32,
        inference_settings="default",
    )
    vacuum_energy_ev = _finite_float(vacuum_energy_ev, label="vacuum correction energy")
    vacuum_forces_array = np.asarray(vacuum_forces, dtype=np.float64)
    vacuum_force_shape = (len(water["atomic_numbers"]), 3)
    if (
        vacuum_energy_ev != 0.0
        or vacuum_forces_array.shape != vacuum_force_shape
        or not np.isfinite(vacuum_forces_array).all()
        or not np.array_equal(
            vacuum_forces_array, np.zeros(vacuum_force_shape, dtype=np.float64)
        )
    ):
        raise ValueError("AniSolv compact exact vacuum energy/force gate changed.")

    records = []
    for item in panel:
        energy_ev, forces = sample.predict_solvation_energy(
            (item["atomic_numbers"], item["positions_angstrom"]),
            charge=0,
            spin=1,
            solvent="water",
            checkpoint=checkpoint,
            device="cpu",
            dtype=torch.float32,
            inference_settings="default",
        )
        forces_array = np.asarray(forces, dtype=np.float64)
        expected_shape = (len(item["atomic_numbers"]), 3)
        if forces_array.shape != expected_shape or not np.isfinite(forces_array).all():
            raise ValueError(
                f"AniSolv force correction for {item['label']} is not finite {expected_shape}."
            )
        energy_ev = _finite_float(
            energy_ev,
            label=f"{item['label']} correction energy",
        )
        force_rows = forces_array.tolist()
        records.append(
            {
                "atomic_numbers": item["atomic_numbers"],
                "charge": 0,
                "correction_energy_ev": energy_ev,
                "correction_forces_ev_per_angstrom": force_rows,
                "correction_forces_sha256": _sha256_bytes(_canonical_bytes(force_rows)),
                "formula": item["formula"],
                "g2_name": item["g2_name"],
                "label": item["label"],
                "multiplicity": 1,
                "n_atoms": len(item["atomic_numbers"]),
                "positions_angstrom": item["positions_angstrom"],
                "solvent": "water",
            }
        )

    model_fingerprint = _loaded_model_fingerprint(source_root)
    module_origins = _verified_anisolv_module_origins(source_root)
    model_fingerprint["loaded_source_modules"] = module_origins
    model_fingerprint["loaded_source_modules_sha256"] = _sha256_bytes(
        _canonical_bytes(module_origins)
    )

    return {
        "acceptance_eligible": False,
        "audited_on": AUDITED_ON,
        "evaluation_type": (
            "upstream_six_example_water_single_point_correction_development_audit"
        ),
        "gpu_acceleration": {
            "gpu_executed": False,
            "gpu_parity_demonstrated": False,
            "no_loss_gpu_admission_status": (
                "blocked_until_complete_cpu_gpu_scalar_energy_and_scientific_parity"
            ),
            "reduced_precision_controls": precision_state,
        },
        "identity": {
            "checkpoint": {**CHECKPOINT, "verified": True},
            "repository": UPSTREAM_REPOSITORY,
            "revision": UPSTREAM_REVISION,
            "source_blobs": {
                label: {**specification, "verified": True}
                for label, specification in SOURCE_BLOBS.items()
            },
            "tree": UPSTREAM_TREE,
        },
        "metrics": None,
        "model_runtime_fingerprint": model_fingerprint,
        "panel": {
            "geometry_and_source_row_sha256": panel_sha256,
            "experimental_label_values_embedded": False,
            "experimental_label_values_used_for_identity": False,
            "geometry_source": (
                "ASE g2"
                if bool(getattr(sample, "HAVE_ASE", False))
                else "bundled fallback"
            ),
            "record_count": len(records),
            "records": records,
            "recorded_forces_are_identity_diagnostics_only": True,
            "sample_source": SOURCE_BLOBS["sample"]["path"],
            "solvent_count": 1,
            "split_membership": "unknown",
            "training_overlap": "unknown",
            "vacuum_gate": {
                "correction_energy_ev": vacuum_energy_ev,
                "correction_forces_exactly_zero": True,
                "correction_forces_shape": list(vacuum_force_shape),
                "solute": "water",
            },
        },
        "route4_accuracy_gate": {
            "accuracy_metric_reporting_allowed": False,
            "evaluated": False,
            "ideal_max_error_kcal_mol": ROUTE4_IDEAL_MAX_ERROR_KCAL_MOL,
            "ideal_passed": False,
            "required_max_error_kcal_mol": ROUTE4_MAX_ERROR_TARGET_KCAL_MOL,
            "required_passed": False,
            "reason": (
                "Accuracy evaluation is forbidden: this panel has six records and "
                "no predeclared ten-functional-group taxonomy."
            ),
        },
        "functional_group_accuracy_gate": {
            "accuracy_metric_reporting_allowed": False,
            "functional_group_taxonomy": None,
            "minimum_distinct_functional_groups": 10,
            "minimum_record_count": 10,
            "observed_distinct_functional_groups": 0,
            "observed_record_count": len(records),
            "passes": False,
            "record_assignments_predeclared": False,
            "scope": "runtime_or_interface_smoke_only",
        },
        "runtime": {
            "ase_version": ase.__version__,
            "device": "cpu",
            "dependency_origins_within_active_environment": dependency_origins,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "dtype": "torch.float32",
            "inference_settings": "default",
            "interop_threads": torch.get_num_interop_threads(),
            "numpy_version": np.__version__,
            "python_version": ".".join(map(str, sys.version_info[:3])),
            "threads": torch.get_num_threads(),
            "torch_force_no_weights_only_load": (
                os.environ.get("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD") == "1"
            ),
            "torch_version": torch.__version__,
        },
        "scientific_scope": {
            "absolute_thermodynamic_free_energy_certified": False,
            "complete_standard_state_cycle_executed": False,
            "experimental_labels_described_upstream_as": "FreeSolv / MNSol compilations",
            "experimental_labels_used_for_accuracy": False,
            "independent_experimental_holdout": False,
            "literature_accuracy_certification": False,
            "maximum_error_gate_passed": None,
            "multi_solvent_validation": False,
            "no_fitting_or_calibration_performed": True,
            "no_training_or_fine_tuning_performed": True,
            "runtime_exposure": "single_point_scalar_energy_only",
            "upstream_force_consistency_certified": False,
            "upstream_forces_exposed_by_maple": False,
            "warning": (
                "The upstream source describes experimental hydration-free-energy labels, "
                "but this audit intentionally does not use their numeric values to calculate, "
                "compare, report, or fingerprint accuracy. Exact record IDs, training overlap, "
                "split identity, and a complete thermodynamic standard-state cycle are "
                "unavailable, so these six rows are runtime evidence only and cannot support "
                "an accuracy claim. "
                "Force arrays are frozen only as runtime-identity diagnostics and are not "
                "admitted for optimization, frequency, or dynamics."
            ),
        },
        "source_execution_isolation": {
            "git_archive_from_verified_tree": True,
            "ignored_checkout_files_excluded_from_execution": True,
            "isolated_import_parent": True,
            "isolated_pycache_root": True,
            "loaded_anisolv_module_origins_verified": True,
        },
    }


def audit_sample(upstream_root: Path) -> dict[str, Any]:
    """Run the exact six-example source panel with the pinned compact model."""

    upstream_root = upstream_root.expanduser().resolve()
    verify_upstream_identity(upstream_root)
    original_path = list(sys.path)
    original_pycache_prefix = sys.pycache_prefix

    with tempfile.TemporaryDirectory(prefix="maple-anisolv-audit-") as temporary:
        isolated_parent = Path(temporary)
        source_root = isolated_parent / "anisolv"
        _materialize_pinned_tree(upstream_root, source_root)
        pycache_root = isolated_parent / "pycache"
        pycache_root.mkdir()
        sys.pycache_prefix = str(pycache_root)
        try:
            return _execute_sample(source_root, isolated_parent)
        finally:
            sys.path[:] = original_path
            sys.pycache_prefix = original_pycache_prefix
            for name in list(sys.modules):
                if name == "anisolv" or name.startswith("anisolv."):
                    del sys.modules[name]


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(rendered)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--upstream-root",
        type=Path,
        required=True,
        help="Clean checkout of the exact pinned Ant-on-knee/anisolv revision.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload = audit_sample(args.upstream_root)
    _write_json_atomic(args.output, payload)
    print(
        json.dumps(
            {
                "coverage": 1.0,
                "record_count": payload["panel"]["record_count"],
                "scope": "runtime_or_interface_smoke_only",
            },
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
    )
    print("Route 4 accuracy gate: NOT EVALUATED (six-row runtime smoke only)")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
