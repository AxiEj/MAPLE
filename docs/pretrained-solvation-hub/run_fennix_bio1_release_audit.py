#!/usr/bin/env python3
"""Audit the pinned FeNNix-Bio1 release boundary without executing the model.

This audit verifies the two released checkpoints, the exact FeNNol runtime
revision, and Tinker-HP's generic GPU/Lambda-ABF bridge.  It deliberately does
not infer energies, run molecular dynamics, reconstruct the paper protocol, or
claim that GPU execution preserves numerical or scientific accuracy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

AUDITED_ON = "2026-07-30"
PAPER_DOI = "10.26434/chemrxiv-2025-f1hgn-v4"
MODEL_DISTRIBUTION_REVISION = "83f299b81c1d62e2a15c892280559a7c0cc2fac3"
FENNOL_REVISION = "d62b8740343b803a2b864140ec79e347f8ba034e"
TINKER_HP_REVISION = "384bb7d451f85f2ea825dabd888e64ad4448674b"

WEIGHTS = {
    "small": {
        "upstream_path": "FENNIX-BIO1/fennix-bio1S.fnx",
        "size_bytes": 29_772_732,
        "sha256": ("82c570c57e95cf164b1a1b0ac2122133cb435c89b07d495246773091541c2f07"),
    },
    "medium": {
        "upstream_path": "FENNIX-BIO1/fennix-bio1M.fnx",
        "size_bytes": 38_124_300,
        "sha256": ("5aac1aa309a387484b7ee4d61fe0229abba7d4af23845be17e204c1feb34870a"),
    },
}

FENNOL_BLOBS = {
    "ase_calculator": {
        "path": "src/fennol/ase.py",
        "size_bytes": 8_386,
        "sha256": ("6d6a9dbf40fce96b8589a3702de7de190f8301d0827f629fbfad2c1e4a2f97d7"),
    },
    "md_runtime": {
        "path": "src/fennol/md/dynamic.py",
        "size_bytes": 25_690,
        "sha256": ("666dd2ebaf4f459d66961fc575f26c041ec2179e27becc2cafe4392828a0342c"),
    },
    "package_metadata": {
        "path": "pyproject.toml",
        "size_bytes": 1_300,
        "sha256": ("adce74494f77d993bd38402bb0ae2e2c1957e1d8f720438815fa61cadff9d569"),
    },
}

TINKER_HP_BLOBS = {
    "gpu_fennol_lambda_interface": {
        "path": "v1.3/GPU/source/mlinterface.py",
        "size_bytes": 16_624,
        "sha256": ("4437e0b93a55156d554ba13a3d7c974a4c537a0792b15c3ec3cac7d2d8758967"),
    },
    "gpu_fennol_lambda_header": {
        "path": "v1.3/GPU/source/mlinterface.h",
        "size_bytes": 749,
        "sha256": ("17c781e21bb900f1fbd6fc32114e9ebe8f184dc80d1a2b9d6aac236029a113b6"),
    },
    "fortran_gpu_lambda_bridge": {
        "path": "v1.3/GPU/source/mlpot.f90",
        "size_bytes": 9_572,
        "sha256": ("7e18fca4ce1e6c75941cbad8d33d1614838740b1a33125c834c53bd4633a9d84"),
    },
    "deep_hp_documentation": {
        "path": "v1.3/GPU/Deep-HP.md",
        "size_bytes": 4_722,
        "sha256": ("38077bbd469f987efda4b7b389a97bcc615d46b715a60e73baea6c704c9865a8"),
    },
    "lambda_abf_documentation": {
        "path": "html/md_lambda-ABF.html",
        "size_bytes": 16_824,
        "sha256": ("fb2aa2e30b417bb54ce74a49d36025d48d717e9b50c9d6abbad7ae8c6aa783a0"),
    },
    "generic_gpu_example_key": {
        "path": "v1.3/GPU/examples/Deep-HP_example.key",
        "size_bytes": 622,
        "sha256": ("b0d3bcf471b6491bd8058756983b9474864bda91db6d651416ce1c7ab3f24712"),
    },
}

TINKER_INTERFACE_REQUIRED_TOKENS = (
    "model = fennol.load(model_file)",
    'gradient_keys = gradient_keys + ["alch_elambda", "alch_vlambda"]',
    'inputs["alch_group"] = alch_group',
    'inputs["alch_ligand_charge"] = ligand_charge',
    'inputs["species_ligand_count"] = species_ligand_count',
    'dedle = energy_multiplier * de["alch_elambda"]',
    'dedlv = energy_multiplier * de["alch_vlambda"]',
    "writeGPUptr(dedle_ptr, dedle, wait=True)",
    "writeGPUptr(dedlv_ptr, dedlv, wait=True)",
)

TINKER_GPU_NUMERIC_TOKENS = (
    'jax.config.update("jax_default_matmul_precision", "highest")',
    '_ctype2dtype["float"] = np.float32',
    'coordinates = asJaxArray(coord_ptr, "float", [nb_atoms_full, 3])',
    "alch_elambda = jnp.asarray(float(alch_elambda), dtype=jnp.float32)",
    "alch_vlambda = jnp.asarray(float(alch_vlambda), dtype=jnp.float32)",
)

TINKER_FORTRAN_LAMBDA_TOKENS = (
    "use_lambda_int = 1",
    "vlambda_c = real(vlambda,c_float)",
    "elambda_c = real(elambda,c_float)",
    "alch_group(i) = int(mutInt(iglob), c_int32_t)",
    "delambdae = delambdae + dedle_ml(1)",
    "delambdav = delambdav + dedlv_ml(1)",
)

FENNOL_ASE_PRECISION_TOKENS = (
    "use_float64: bool = False",
    'jax.config.update("jax_enable_x64", True)',
    'self.dtype = "float64" if use_float64 else "float32"',
)

FENNOL_MD_PRECISION_TOKENS = (
    'enable_x64 = simulation_parameters.get("double_precision", False)',
    'matmul_precision = simulation_parameters.get("matmul_prec", "highest").lower()',
    "default matmul precision involves float16 operations",
)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_weight(path: Path, *, label: str, expected: dict[str, Any]) -> None:
    if not path.is_file():
        raise ValueError(f"{label} checkpoint is not a regular file: {path}")
    if path.stat().st_size != expected["size_bytes"]:
        raise ValueError(f"{label} checkpoint size does not match the pinned release.")
    if _sha256_file(path) != expected["sha256"]:
        raise ValueError(
            f"{label} checkpoint SHA256 does not match the pinned release."
        )


def _git_bytes(root: Path, *args: str) -> bytes:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            capture_output=True,
            env={**os.environ, "GIT_LFS_SKIP_SMUDGE": "1"},
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "The release audit requires the system `git` executable."
        ) from exc
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(f"Git inspection failed for {root}: {detail}") from exc
    return completed.stdout


def _verify_revision(root: Path, *, label: str, expected: str) -> str:
    actual = _git_bytes(root, "rev-parse", "HEAD").decode("ascii").strip()
    if actual != expected:
        raise ValueError(f"{label} revision {actual!r} does not match {expected!r}.")
    return actual


def _read_verified_blob(
    root: Path, *, label: str, specification: dict[str, Any]
) -> bytes:
    data = _git_bytes(root, "show", f"HEAD:{specification['path']}")
    if len(data) != specification["size_bytes"]:
        raise ValueError(f"{label} size does not match the pinned source blob.")
    if _sha256_bytes(data) != specification["sha256"]:
        raise ValueError(f"{label} SHA256 does not match the pinned source blob.")
    return data


def _require_tokens(text: str, *, label: str, required: tuple[str, ...]) -> None:
    missing = [token for token in required if token not in text]
    if missing:
        raise ValueError(f"{label} is missing required release tokens: {missing!r}")


def _tree_paths(root: Path) -> list[str]:
    text = _git_bytes(root, "ls-tree", "-r", "--name-only", "HEAD").decode(
        "utf-8", errors="strict"
    )
    return [line for line in text.splitlines() if line]


def _named_paper_artifact_matches(paths: list[str]) -> list[str]:
    markers = ("fennix-bio1", "freesolv")
    return sorted(
        path for path in paths if any(marker in path.lower() for marker in markers)
    )


def _verified_blob_record(specification: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": specification["path"],
        "sha256": specification["sha256"],
        "size_bytes": specification["size_bytes"],
        "verified": True,
    }


def audit_release(
    fennol_root: Path,
    tinker_hp_root: Path,
    small_checkpoint: Path,
    medium_checkpoint: Path,
) -> dict[str, Any]:
    """Verify static release evidence and return a fail-closed admission record."""

    _verify_revision(fennol_root, label="FeNNol", expected=FENNOL_REVISION)
    _verify_revision(tinker_hp_root, label="Tinker-HP", expected=TINKER_HP_REVISION)
    _verify_weight(small_checkpoint, label="small", expected=WEIGHTS["small"])
    _verify_weight(medium_checkpoint, label="medium", expected=WEIGHTS["medium"])

    fennol_sources = {
        name: _read_verified_blob(
            fennol_root, label=f"FeNNol {name}", specification=specification
        )
        for name, specification in FENNOL_BLOBS.items()
    }
    tinker_sources = {
        name: _read_verified_blob(
            tinker_hp_root,
            label=f"Tinker-HP {name}",
            specification=specification,
        )
        for name, specification in TINKER_HP_BLOBS.items()
    }

    interface_text = tinker_sources["gpu_fennol_lambda_interface"].decode("utf-8")
    _require_tokens(
        interface_text,
        label="Tinker-HP FeNNol Lambda interface",
        required=TINKER_INTERFACE_REQUIRED_TOKENS,
    )
    _require_tokens(
        interface_text,
        label="Tinker-HP GPU numeric contract",
        required=TINKER_GPU_NUMERIC_TOKENS,
    )
    _require_tokens(
        tinker_sources["fortran_gpu_lambda_bridge"].decode("utf-8"),
        label="Tinker-HP Fortran Lambda bridge",
        required=TINKER_FORTRAN_LAMBDA_TOKENS,
    )
    _require_tokens(
        fennol_sources["ase_calculator"].decode("utf-8"),
        label="FeNNol ASE precision controls",
        required=FENNOL_ASE_PRECISION_TOKENS,
    )
    _require_tokens(
        fennol_sources["md_runtime"].decode("utf-8"),
        label="FeNNol MD precision controls",
        required=FENNOL_MD_PRECISION_TOKENS,
    )

    example_key = tinker_sources["generic_gpu_example_key"].decode("utf-8")
    if "ML-MODEL ../ml_models/ani2x.fnx" not in example_key:
        raise ValueError("The pinned Tinker-HP generic example identity changed.")

    named_matches = _named_paper_artifact_matches(_tree_paths(tinker_hp_root))
    if named_matches:
        raise ValueError(
            "The pinned Tinker-HP tree now has named FeNNix-Bio1/FreeSolv paths; "
            "the negative packaging conclusion requires a fresh scientific audit."
        )

    return {
        "acceptance_eligible": False,
        "audited_on": AUDITED_ON,
        "evaluation_type": (
            "static_release_identity_and_protocol_source_audit_no_inference"
        ),
        "gpu_acceleration": {
            "admission_status": "blocked_until_no_loss_reference_gpu_parity",
            "fennol_generic_precision_controls": {
                "ase_float64_option_present": True,
                "md_double_precision_option_present": True,
                "md_default_matmul_precision": "highest",
                "md_default_scalar_dtype": "float32",
                "reduced_matmul_warning_present": True,
            },
            "no_precision_loss_demonstrated": False,
            "official_gpu_interface_source_present": True,
            "tinker_gpu_bridge_static_contract": {
                "coordinate_dtype": "float32",
                "lambda_dtype": "float32",
                "matmul_precision": "highest",
                "runtime_precision_switch_exposed": False,
            },
            "parity_executed": {
                "coverage": False,
                "energy": False,
                "maximum_error": False,
                "forces": False,
                "lambda_derivatives": False,
                "trajectory_or_sampled_observables": False,
                "final_free_energy_and_uncertainty": False,
                "virial": False,
            },
            "required_before_enablement": [
                "same checkpoint, equations, cutoffs, lambda schedule, estimator, and convergence criteria on reference and GPU paths",
                "no float16, bfloat16, TF32, fast-math, reduced-matmul, or relaxed-convergence substitution",
                "paired energy, force, virial, alch_elambda, and alch_vlambda numerical-equivalence audit on frozen configurations",
                "paired final free-energy and uncertainty audit with no degradation in maximum error or literature-panel metrics",
            ],
        },
        "identity": {
            "model_distribution_revision": MODEL_DISTRIBUTION_REVISION,
            "paper_doi": PAPER_DOI,
            "weights": {
                name: {
                    **specification,
                    "local_file_verified": True,
                }
                for name, specification in WEIGHTS.items()
            },
        },
        "model_id": "fennix-bio1",
        "negative_admission_reasons": [
            "no unchanged installed environment was executed by this static audit",
            "the exact paper system inputs, lambda schedule, sampling controls, convergence evidence, standard-state treatment, and row ledger are not verified as one public bundle",
            "GPU availability is not evidence of no-loss numerical or scientific parity",
            "published solvation evidence is water-only and supplies neither multi-solvent validation nor a maximum-error ledger",
            "training overlap and a never-used independent experimental panel remain unresolved",
        ],
        "paper_protocol_packaging": {
            "exact_fennix_bio1_freesolv_bundle_verified": False,
            "named_artifact_path_matches_at_exact_tinker_revision": named_matches,
            "path_search_scope": (
                "exact Tinker-HP Git tree path names only; no claim about unsearched "
                "external, author-held, or generically named artifacts"
            ),
            "shipped_gpu_example_model": "ani2x.fnx",
        },
        "runtime_sources": {
            "fennol": {
                "license": "LGPL-3.0",
                "revision": FENNOL_REVISION,
                "verified_blobs": {
                    name: _verified_blob_record(specification)
                    for name, specification in FENNOL_BLOBS.items()
                },
            },
            "tinker_hp": {
                "license": "Tinker academic source license",
                "revision": TINKER_HP_REVISION,
                "verified_blobs": {
                    name: _verified_blob_record(specification)
                    for name, specification in TINKER_HP_BLOBS.items()
                },
            },
        },
        "schema_version": 1,
        "source_capabilities": {
            "combined_fennol_tinker_gpu_source_present": True,
            "fennol_model_load_present": True,
            "lambda_energy_and_vdw_inputs_present": True,
            "lambda_group_and_ligand_charge_inputs_present": True,
            "lambda_energy_and_vdw_derivative_outputs_present": True,
            "generic_lambda_abf_documentation_present": True,
        },
        "validation_scope": {
            "checkpoint_inference_performed": False,
            "free_energy_protocol_executed": False,
            "multi_solvent_evidence": False,
            "predicted_rows_in_artifact": False,
            "published_solvent_scope": "water only",
            "strict_never_used_holdout_proven": False,
            "maximum_error_available": False,
        },
    }


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fennol-root", required=True, type=Path)
    parser.add_argument("--tinker-hp-root", required=True, type=Path)
    parser.add_argument("--small-checkpoint", required=True, type=Path)
    parser.add_argument("--medium-checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    payload = audit_release(
        args.fennol_root,
        args.tinker_hp_root,
        args.small_checkpoint,
        args.medium_checkpoint,
    )
    _write_json_atomic(args.output, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
