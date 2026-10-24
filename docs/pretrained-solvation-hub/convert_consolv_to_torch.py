#!/usr/bin/env python3
"""Convert the published ConSolv Haiku parameter trees to a Torch-only file.

The converter never executes or serializes a JAX model.  A pinned extraction
environment is used only to deserialize the public trees and materialize the
fixed reduced tensor-product bases.  The resulting checkpoint loads with
``torch.load(..., weights_only=True)`` and has no JAX runtime dependency.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path
import subprocess
import tempfile

SOURCE_REVISION = "06801dd5f53182ed86c31eee1fa03cc011c05b5a"
ARTIFACTS = {
    "examples/SFE/output/vacuum/best_params_1.pkl": "6ef9f20b5d5d2ef609d492d9a04c6d8af9c2057fe35ab8d574698db0cb089988",
    "examples/SFE/output/solvent/full_init_params.pkl": "1ceebcd6288d08a507b85dc6d062dbfc8531ba61b578101e5f7b2b4240077021",
    "examples/SFE/output/solvent/final_params.pkl": "6b0d6a9125da4cc8d17e506b561b133d74f05cbda510448f5d3d19ffc6bd3044",
    "examples/SFE/output/solvent/config.toml": "a97d2569cda456989bd61ac665b7882356fb3e59170be94b1ab807445eaa8a20",
    "dataset/descriptor.csv": "8c9106ffe0f484e10590141d4a8e5d977ec2ed20154807eb99f37b1cfe7d2051",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_source(source: Path):
    revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != SOURCE_REVISION:
        raise RuntimeError(f"ConSolv revision mismatch: {revision}")
    observed = {name: sha256(source / name) for name in ARTIFACTS}
    if observed != ARTIFACTS:
        raise RuntimeError("ConSolv public artifact identity mismatch")
    return observed


def _merge(template, vacuum):
    if isinstance(template, dict):
        return {
            key: _merge(value, vacuum[key]) if key in vacuum else value
            for key, value in template.items()
        }
    return vacuum


def extract(source: Path, output: Path, state: str):
    """Run inside the isolated JAX environment; output contains numeric arrays only."""
    import e3nn_jax as e3nn
    import jax
    import numpy as np
    import pandas as pd

    root = source / "examples/SFE/output"
    with (root / "vacuum/best_params_1.pkl").open("rb") as handle:
        vacuum = pickle.load(handle)
    vacuum = {
        key.replace("mace", "mace_consol").replace(
            "linear_node_embedding_layer", "linear_node_embedding_block"
        ): value
        for key, value in vacuum.items()
    }
    with (root / "solvent/full_init_params.pkl").open("rb") as handle:
        initial = pickle.load(handle)
    with (root / "solvent/final_params.pkl").open("rb") as handle:
        delta = pickle.load(handle)
    merged = _merge(initial, vacuum)
    if state == "solution":
        params = jax.tree_util.tree_map(
            lambda a, b: np.asarray(a) + np.asarray(b), delta, merged
        )
    elif state == "vacuum":
        params = _merge(jax.tree_util.tree_map(np.zeros_like, merged), vacuum)
    else:
        raise ValueError(f"Unknown ConSolv potential state: {state!r}")

    arrays, names = {}, []
    for module in sorted(params):
        for parameter in sorted(params[module]):
            name = f"{module}::{parameter}"
            names.append(name)
            arrays[f"p{len(names)-1:03d}"] = np.asarray(params[module][parameter])

    descriptor = pd.read_csv(source / "dataset/descriptor.csv")
    descriptor = np.concatenate(
        (np.zeros((1, descriptor.shape[1])), descriptor.to_numpy()), axis=0
    )[:, 1:].astype(np.float32)
    descriptor = (descriptor - descriptor.mean(0)) / descriptor.std(0)
    arrays["descriptor"] = descriptor

    irreps = e3nn.Irreps("1x0e+1x1o+1x2e+1x3o")
    for order in (1, 2, 3):
        basis = e3nn.reduced_symmetric_tensor_product_basis(
            irreps, order, keep_ir={e3nn.Irrep("0e"), e3nn.Irrep("1o")}
        )
        for (_, ir), value in zip(basis.irreps, basis.list):
            arrays[f"basis_order{order}_{ir}"] = np.asarray(value)
    arrays["parameter_names"] = np.asarray(names, dtype=np.str_)
    np.savez(output, **arrays)


def convert(args):
    import numpy as np
    import torch

    source = args.source.resolve()
    observed = verify_source(source)
    with tempfile.TemporaryDirectory(prefix="consolv-convert-") as directory:
        extracted = Path(directory) / "extracted.npz"
        subprocess.run(
            [
                str(args.extract_python),
                str(Path(__file__).resolve()),
                "--extract",
                "--source",
                str(source),
                "--output",
                str(extracted),
                "--state",
                args.state,
            ],
            check=True,
        )
        values = np.load(extracted)
        names = [str(x) for x in values["parameter_names"]]
        state_dict = {}
        parameter_map = {}
        for index, name in enumerate(names):
            key = f"p{index:03d}"
            state_dict[key] = torch.from_numpy(values[key].copy())
            parameter_map[name] = key
        state_dict["descriptor"] = torch.from_numpy(values["descriptor"].copy())
        for order in (1, 2, 3):
            for ir in ("0e", "1o"):
                key = f"basis_order{order}_{ir}"
                state_dict[key] = torch.from_numpy(values[key].copy())

    unused = ["mace_consol/layer_0/linear_readout_layer/linear::w[0,0] 64x0e,1x0e"]
    payload = {
        "format": "consolv-torch-v1",
        "metadata": {
            "source_revision": SOURCE_REVISION,
            "source_artifact_sha256": observed,
            "coordinate_unit": "angstrom",
            "native_energy_unit": "kJ/mol",
            "cutoff_angstrom": 5.0,
            "dtype_origin": "float32",
            "implementation": "independent_pure_torch",
            "potential_state": args.state,
            "parameter_count": int(
                sum(t.numel() for k, t in state_dict.items() if k.startswith("p"))
            ),
        },
        "configuration": {
            "num_species": 55,
            "num_solvents": 148,
            "num_interactions": 2,
            "hidden_irreps": "64x0e+32x1o",
            "max_ell": 3,
            "correlation": 3,
            "heads": 8,
            "key_size": 64,
        },
        "parameter_map": parameter_map,
        "unused_source_parameters": unused,
        "state_dict": state_dict,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    reloaded = torch.load(args.output, map_location="cpu", weights_only=True)
    if reloaded["format"] != "consolv-torch-v1" or len(reloaded["parameter_map"]) != 60:
        raise RuntimeError("Converted checkpoint failed its safe reload audit")
    receipt = {
        "schema_version": 1,
        "checkpoint": str(args.output),
        "checkpoint_sha256": sha256(args.output),
        "checkpoint_size_bytes": args.output.stat().st_size,
        "source_revision": SOURCE_REVISION,
        "potential_state": args.state,
        "source_artifact_sha256": observed,
        "source_tensor_count": len(names),
        "source_parameter_elements": payload["metadata"]["parameter_count"],
        "unused_source_parameters": unused,
        "safe_reload": True,
        "jax_runtime_required_for_inference": False,
    }
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


def verify_parity(args):
    """Recompute Torch-vs-audit-reference E/F errors into a hashed receipt."""
    import torch

    from maple.function.calculator.consolv._torch_model import (
        load_consolv_torch_checkpoint,
    )

    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    numbers = torch.tensor(reference["input"]["atomic_numbers"], dtype=torch.long)
    positions = torch.tensor(reference["input"]["positions_A"], dtype=torch.float64)
    model = load_consolv_torch_checkpoint(args.checkpoint, expected_state="solution")
    comparisons = {}
    for solvent, result in reference["results"].items():
        energy, forces = model.energy_and_forces_ev(
            numbers, positions, result["official_index"]
        )
        reference_forces = torch.tensor(result["forces_eV_per_A"], dtype=torch.float64)
        comparisons[solvent] = {
            "official_index": result["official_index"],
            "reference_energy_eV": result["energy_eV"],
            "torch_energy_eV": energy.item(),
            "absolute_energy_error_eV": abs(energy.item() - result["energy_eV"]),
            "max_force_component_error_eV_per_A": torch.max(
                torch.abs(forces - reference_forces)
            ).item(),
        }
    receipt = {
        "schema_version": 1,
        "scope": "independent Torch reconstruction parity against a pinned offline JAX audit",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "reference": str(args.reference.resolve()),
        "reference_sha256": sha256(args.reference),
        "audit_script": str(args.audit_script.resolve()),
        "audit_script_sha256": sha256(args.audit_script),
        "source_revision": SOURCE_REVISION,
        "reference_is_unmodified_upstream_oracle": False,
        "audit_api_boundary_repairs": [
            "ignore unsupported false return_attn_weights keyword",
            "ignore unsupported initializer-only parameter_initializer keyword",
        ],
        "torch_cpu_dtype": "float64",
        "reference_dtype": "float32",
        "comparisons": comparisons,
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--verify-parity", action="store_true")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--audit-script", type=Path)
    parser.add_argument(
        "--extract-python",
        type=Path,
        default=Path("/home/axie/.cache/maple/consolv/venv/bin/python"),
    )
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--state", choices=("solution", "vacuum"), default="solution")
    args = parser.parse_args()
    if args.verify_parity:
        missing = [
            name
            for name in ("checkpoint", "reference", "audit_script", "receipt")
            if getattr(args, name) is None
        ]
        if missing:
            flags = ", ".join(f"--{name.replace('_', '-')}" for name in missing)
            parser.error("--verify-parity requires " + flags)
        verify_parity(args)
    elif args.extract:
        if args.source is None or args.output is None:
            parser.error("--extract requires --source and --output")
        extract(args.source.resolve(), args.output, args.state)
    else:
        if args.source is None or args.output is None or args.receipt is None:
            parser.error("conversion requires --source, --output, and --receipt")
        convert(args)


if __name__ == "__main__":
    main()
