#!/usr/bin/env python3
"""Read-only P0 diagnostics for the exact MACE-EF-COS checkpoint.

This does NOT repair, re-export, or admit a model. Native/local comparisons
below test the projection layer only, not the unavailable original eager EF
model. The legacy reconstruction mirrors the two archived norm substitutions;
all energy/response measurements execute the unchanged verified checkpoint.
"""

from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
from functools import partial
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _vector(value, name):
    result = np.asarray(value, dtype=float)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be one finite Cartesian vector")
    return result


def response_scan(gradient, center, steps):
    """Difference energy gradients; preserve column ordering and raw matrices."""
    center = _vector(center, "center")
    steps = np.asarray(steps, dtype=float)
    if (
        steps.ndim != 1
        or not steps.size
        or not np.all(np.isfinite(steps) & (steps > 0))
    ):
        raise ValueError("steps must be a nonempty vector of finite positive values")
    central = _vector(gradient(center.copy()), "gradient")
    rows = []
    for step in steps:
        plus, minus = [], []
        for axis in np.eye(3):
            plus.append(_vector(gradient(center + step * axis), "gradient"))
            minus.append(_vector(gradient(center - step * axis), "gradient"))
        plus, minus = np.column_stack(plus), np.column_stack(minus)
        raw = (plus - minus) / (2 * step)
        symmetric, antisymmetric = 0.5 * (raw + raw.T), 0.5 * (raw - raw.T)
        symmetric_norm = float(np.linalg.norm(symmetric))
        antisymmetric_norm = float(np.linalg.norm(antisymmetric))
        rows.append(
            {
                "center_ev_per_e_angstrom": center.tolist(),
                "step_ev_per_e_angstrom": float(step),
                "center_gradient": central.tolist(),
                "plus_gradients_columns": plus.tolist(),
                "minus_gradients_columns": minus.tolist(),
                "axis_derivative_jumps": np.diag(plus - minus).tolist(),
                "raw_hessian": raw.tolist(),
                "symmetric_hessian": symmetric.tolist(),
                "symmetric_eigenvalues": np.linalg.eigvalsh(symmetric).tolist(),
                "antisymmetric_frobenius_norm": antisymmetric_norm,
                "antisymmetric_to_symmetric_ratio": (
                    antisymmetric_norm / symmetric_norm if symmetric_norm else None
                ),
            }
        )
    return rows


def project_local_potential(projector, values):
    """Reuse native displacement/projection code, with each atom its own origin.

    values = [V, grad_x(V), grad_y(V), grad_z(V)]. No local permutation,
    radial normalization, or coordinate-label-derived matrix is invented.
    The 1/2 factor is the model's two-spin-channel split, before feature norms.
    """
    import torch
    from graph_longrange.gto_utils import DisplacedGTOExternalFieldBlock

    if (
        values.ndim != 2
        or values.shape[1] != 4
        or not bool(torch.isfinite(values).all())
    ):
        raise ValueError("values must be a finite atomwise N x 4 potential jet")
    return 0.5 * DisplacedGTOExternalFieldBlock.forward(
        projector,
        torch.arange(values.shape[0], device=values.device),
        values.new_zeros((values.shape[0], 3)),
        values,
    )


def reconstruct_legacy_projection(projector, values):
    """Reconstruct only the verified archive's proxy feature for comparison."""
    import torch

    zero_potential = values.clone()
    zero_potential[:, 0] = 0
    features = project_local_potential(projector, zero_potential)
    scalar_rows = projector.matrix[:, 0].abs() > 0
    features[:, scalar_rows] = (
        0.5
        * torch.linalg.vector_norm(values[:, 1:], dim=-1, keepdim=True)
        * projector.matrix[scalar_rows, 0].abs()
    )
    return features


def projection_audit(projector, positions, g, candidate_layout, archive_half):
    import torch
    from graph_longrange.gto_utils import DisplacedGTOExternalFieldBlock

    centered = positions - positions.mean(0, keepdim=True)
    values = torch.cat(((centered @ g)[:, None], g.expand(len(positions), 3)), dim=1)
    native = 0.5 * DisplacedGTOExternalFieldBlock.forward(
        projector,
        torch.zeros(len(positions), dtype=torch.long, device=positions.device),
        centered,
        torch.cat((g.new_zeros(1), g))[None, :],
    )
    local = project_local_potential(projector, values)
    candidate_linear = archive_half * torch.einsum(
        "pf,nf->np", projector.matrix, values[:, candidate_layout]
    )
    legacy = reconstruct_legacy_projection(projector, values)
    reversed_legacy = reconstruct_legacy_projection(projector, -values)
    return {
        "scope": "installed-forward local-origin algebra; not full-model eager equivalence",
        "normalization": "half external projection per spin, before field_feature_norms",
        "local_jet": values.detach().cpu().tolist(),
        "native_uniform_projection": native.detach().cpu().tolist(),
        "local_affine_projection": local.detach().cpu().tolist(),
        "reconstructed_legacy_projection": legacy.detach().cpu().tolist(),
        "native_vs_local_max_abs": float((native - local).abs().max()),
        "local_vs_archive_layout_candidate_max_abs": float(
            (local - candidate_linear).abs().max()
        ),
        "legacy_vs_native_max_abs": float((legacy - native).abs().max()),
        "legacy_sign_reversal_even_part_max_abs": float(
            (legacy + reversed_legacy).abs().max()
        ),
        "local_sign_reversal_even_part_max_abs": float(
            (local + project_local_potential(projector, -values)).abs().max()
        ),
    }


def write_json_once(path, payload):
    encoded = json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(encoded)


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _source_hashes(paths):
    return {
        str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path): _sha256(
            path
        )
        for path in (Path(p).resolve() for p in paths)
    }


def _archive_evidence(path):
    with zipfile.ZipFile(path) as archive:
        name = next(
            n for n in archive.namelist() if n.endswith("/mace/modules/extensions.py")
        )
        content = archive.read(name)
    source = content.decode()
    tree = ast.parse(source)
    assignments = {
        node.targets[0].id: node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
    }
    gradient_orders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.List) or len(node.elts) != 4:
            continue
        indices = []
        for element in node.elts[1:]:
            if (
                isinstance(element, ast.Call)
                and ast.unparse(element.func) == "torch.select"
                and len(element.args) == 3
                and ast.unparse(element.args[0]) == "external_field"
                and ast.literal_eval(element.args[1]) == 1
            ):
                indices.append(ast.literal_eval(element.args[2]))
        if len(indices) == 3:
            initial = assignments.get(getattr(node.elts[0], "id", None))
            if (
                not isinstance(initial, ast.Call)
                or ast.unparse(initial.func) != "torch.zeros_like"
            ):
                raise ValueError("archive initial scalar slot is not zeros_like")
            gradient_orders.append(indices)
    if gradient_orders != [[2, 0, 1], [2, 0, 1]]:
        raise ValueError("archive potential/gradient column conventions differ")
    return {
        "archived_model_source": name,
        "archived_model_source_sha256": hashlib.sha256(content).hexdigest(),
        "per_atom_field_norm_occurrences": source.count(
            "torch.linalg_vector_norm(external_field"
        ),
        "scalar_projection_override_occurrences": source.count("torch.index_put_"),
        "inspected_gradient_component_indices": gradient_orders[0],
        "inspected_initial_scalar_slot": "zeros_like; not an atomwise V input",
        "candidate_local_jet_column_indices": [0, *[1 + i for i in gradient_orders[0]]],
        "candidate_scalar_semantics": "replace the archived zero/norm scalar with V; projection-only, not applied to checkpoint",
        "original_eager_identity": "unavailable; archive prefix is not an eager provenance certificate",
    }


def _potential_tests(model, positions, g, offsets):
    import torch

    centered = positions - positions.mean(0, keepdim=True)
    base = torch.cat(((centered @ g)[:, None], g.expand(len(positions), 3)), dim=1)

    def evaluate(values):
        energy, source, density = model.conjugate_source_torch(
            positions,
            values.clone().requires_grad_(True),
        )
        return float(energy.detach()), source.detach(), density.detach()

    energy0, source0, density0 = evaluate(base)
    gauges = []
    for offset in offsets:
        values = base.clone()
        values[:, 0] += offset
        energy, source, density = evaluate(values)
        gauges.append(
            {
                "constant_potential_offset_ev_per_e": offset,
                "energy_change_ev": energy - energy0,
                "expected_energy_change_ev": model.config.total_charge * offset,
                "conjugate_total_charge_e": float(source[:, 0].double().sum()),
                "density_max_abs_change": float((density - density0).abs().max()),
                "conjugate_source_max_abs_change": float(
                    (source - source0).abs().max()
                ),
                "energy_ev": energy,
                "conjugate_source_raw": source.cpu().tolist(),
                "density_diagnostic": density.cpu().tolist(),
            }
        )
    perturbation = torch.linspace(
        -0.001, 0.001, len(positions), device=positions.device
    )
    values = base.clone()
    values[:, 0] += perturbation
    energy, source, density = evaluate(values)
    return {
        "baseline": {
            "local_jet": base.cpu().tolist(),
            "energy_ev": energy0,
            "conjugate_source_raw": source0.cpu().tolist(),
            "density_diagnostic": density0.cpu().tolist(),
        },
        "energy_difference_precision": "float32 total-energy subtraction; not a tight FD certificate",
        "constant_potential_tests": gauges,
        "potential_only_nonuniform_test": {
            "energy_ev": energy,
            "conjugate_source_raw": source.cpu().tolist(),
            "density_diagnostic": density.cpu().tolist(),
            "potential_perturbation_ev_per_e": perturbation.cpu().tolist(),
            "energy_change_ev": energy - energy0,
            "linear_q_dot_delta_v_ev": float(
                (source0[:, 0].double() * perturbation.double()).sum()
            ),
            "density_max_abs_change": float((density - density0).abs().max()),
            "conjugate_charge_max_abs_change": float(
                (source[:, 0] - source0[:, 0]).abs().max()
            ),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    preregistration_bytes = args.preregistration.read_bytes()
    preregistration_sha256 = hashlib.sha256(preregistration_bytes).hexdigest()
    prereg = json.loads(preregistration_bytes)

    import torch
    import graph_longrange.gto_utils as gto
    from maple.function.calculator.extra_correction.implicit.mace_polar_ef import (
        MACEPolarEFConfig,
        MACEPolarEFEnergyModel,
    )

    if _sha256(args.checkpoint) != prereg["checkpoint_sha256"]:
        raise ValueError("checkpoint does not match preregistration")
    evidence = _archive_evidence(args.checkpoint)
    if evidence["per_atom_field_norm_occurrences"] != 2:
        raise ValueError(
            "archive does not contain the two expected proxy substitutions"
        )
    sources = [Path(__file__)]
    sources += [
        ROOT / "maple/function/calculator/extra_correction/implicit" / name
        for name in (
            "mace_polar_ef.py",
            "mace_polar_ef_specs.py",
            "electrostatic_pairing.py",
        )
    ]
    sources.append(Path(inspect.getfile(gto)))
    hashes = _source_hashes(sources)
    rows = []
    for molecule in prereg["molecules"]:
        model = MACEPolarEFEnergyModel(
            MACEPolarEFConfig(
                checkpoint_path=str(args.checkpoint),
                atomic_numbers=tuple(molecule["atomic_numbers"]),
                total_charge=molecule["total_charge"],
                spin_multiplicity=molecule["multiplicity"],
            )
        )
        positions = torch.tensor(
            molecule["positions_angstrom"], dtype=torch.float32, device=model.device
        )
        g = torch.tensor(
            prereg["nonzero_center_ev_per_e_angstrom"],
            dtype=torch.float32,
            device=model.device,
        )
        half = float(model.model.model.code_with_constants[1].const_mapping["c2"])
        if half != 0.5:
            raise ValueError("archive spin-channel projection factor is not 1/2")
        evidence["inspected_spin_half_constant_c2"] = half
        gradient = partial(model._affine_field_energy_gradient, positions)
        rows.append(
            {
                "name": molecule["name"],
                "geometry": molecule,
                "model": model.config.as_provenance(),
                "runtime": dict(model.runtime_provenance),
                "zero_center_scan": response_scan(
                    gradient, np.zeros(3), prereg["steps_ev_per_e_angstrom"]
                ),
                "nonzero_center_scan": response_scan(
                    gradient, g.cpu().numpy(), prereg["steps_ev_per_e_angstrom"]
                ),
                "projection": projection_audit(
                    model.model.model.external_field_contribution,
                    positions,
                    g,
                    evidence["candidate_local_jet_column_indices"],
                    half,
                ),
                "potential_tests": _potential_tests(
                    model, positions, g, prereg["constant_offsets_ev_per_e"]
                ),
            }
        )
        del gradient, model
    if hashes != _source_hashes(sources):
        raise RuntimeError("audit source bytes changed during execution")
    if _sha256(args.preregistration) != preregistration_sha256:
        raise RuntimeError("preregistration changed during execution")
    write_json_once(
        args.output,
        {
            "schema_version": 2,
            "artifact": "mace-ef-cos-input-semantics-v2",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "execution_git_head": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "execution_source_binding": "exact file hashes; new audit is not a committed admission replay",
            "source_files_sha256": hashes,
            "preregistration_sha256": preregistration_sha256,
            "python_executable": sys.executable,
            "library_versions": {
                name: importlib.metadata.version(name)
                for name in ("torch", "numpy", "e3nn", "mace-torch")
            },
            "archive_evidence": evidence,
            "records": rows,
            "claim_boundary": {
                "checkpoint_modified": False,
                "full_model_native_vs_local_eager_tested": False,
                "repaired_checkpoint_tested": False,
                "conductor_cosmo_tested": False,
                "chemical_accuracy_tested": False,
                "release_admitted": False,
                "interpretation": "unchanged CUDA checkpoint response diagnostics plus projection-only equivalence",
            },
        },
    )


if __name__ == "__main__":
    main()
