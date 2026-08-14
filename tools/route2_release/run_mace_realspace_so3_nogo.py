#!/usr/bin/env python3
"""Retain the official MACE-POLAR molecular-realspace SO(3) counterexample.

The official checkpoint is evaluated at zero external field before and after
one deterministic proper rotation.  The runner then isolates the two upstream
``graph_longrange==0.4.0`` molecular-realspace primitives with *exactly*
rotated inputs.  This distinguishes a model/long-range defect from the Route-2
continuum and from the eight-channel field transform.

The artifact is negative evidence only.  It admits no Route-2 E/F/H/V/M tier.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
from pathlib import Path
import shlex
import sys
import time

import numpy as np
from ase import Atoms

# Must be set before Torch creates a CUDA context.  The runtime record binds it.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from maple.solvation.continuum import radial_gto_source_rotation_matrix
from maple.solvation.models import (
    MACEPolarVariationalFieldEnergy,
    build_official_mace_polar_1_m_radial_gto_adapter,
)
from maple.solvation.release import (
    RepositorySnapshot,
    canonical_json_sha256,
    checkpoint_record,
    collect_loaded_repository_sources,
    committed_source_hashes,
    runtime_record,
    sha256_file,
    write_external_json_artifact,
)

SCHEMA_VERSION = "route2-mace-molecular-realspace-so3-nogo-v1"
ARTIFACT_KIND = "disabled-real-checkpoint-model-long-range-so3-counterexample"
DEFAULT_CHECKPOINT = Path.home() / ".cache" / "mace" / "MACEPOLAR1Mmodel"
RANDOM_SEED = 20260814
ROTATION_SEED = 20260815
ROUNDOFF_MULTIPLIER = 4096.0
NO_CAPABILITIES = {tier: False for tier in ("E", "F", "H", "V", "M")}
REQUIRED_SOURCE_PATHS = (
    "maple/solvation/continuum/harmonic_coefficients.py",
    "maple/solvation/models/field_energy.py",
    "maple/solvation/models/mace_polar.py",
    "maple/solvation/models/mace_polar_variational.py",
    "maple/solvation/release/evidence.py",
    "maple/function/calculator/mace/_macepol_calculator.py",
    "maple/function/calculator/mace/_macepol_long_range.py",
    "tools/route2_release/run_mace_realspace_so3_nogo.py",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the official-checkpoint molecular-realspace SO(3) negative "
            "canary; this does not admit Route-2 E/F/H/V/M."
        )
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _water() -> Atoms:
    return Atoms(
        "OH2",
        positions=np.asarray(
            [[0.0, 0.0, 0.0], [0.9572, 0.0, 0.0], [-0.2390, 0.9266, 0.0]],
            dtype=float,
        ),
        info={"charge": 0, "mult": 1},
    )


def _rotation() -> np.ndarray:
    matrix = np.random.default_rng(ROTATION_SEED).normal(size=(3, 3))
    rotation, triangular = np.linalg.qr(matrix)
    rotation = rotation @ np.diag(np.where(np.diag(triangular) < 0.0, -1.0, 1.0))
    if np.linalg.det(rotation) < 0.0:
        rotation[:, 0] *= -1.0
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=2.0e-15, rtol=0.0):
        raise RuntimeError("The deterministic rotation lost orthogonality.")
    return rotation


def _roundoff_threshold(scale: float) -> float:
    value = float(scale)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("roundoff scale must be finite and non-negative.")
    return float(ROUNDOFF_MULTIPLIER * np.finfo(float).eps * max(value, 1.0))


def _scalar_rotation_record(base: float, rotated: float) -> dict[str, object]:
    first = float(base)
    second = float(rotated)
    if not math.isfinite(first) or not math.isfinite(second):
        raise ValueError("rotation scalars must be finite.")
    absolute = abs(second - first)
    threshold = _roundoff_threshold(max(abs(first), abs(second)))
    return {
        "base": first,
        "rotated": second,
        "signed_difference": second - first,
        "absolute_difference": absolute,
        "roundoff_threshold": threshold,
        "numerically_equal": bool(absolute <= threshold),
    }


def _covariance_record(actual: object, expected: object) -> dict[str, object]:
    left = np.asarray(actual, dtype=float)
    right = np.asarray(expected, dtype=float)
    if (
        left.shape != right.shape
        or not np.all(np.isfinite(left))
        or not np.all(np.isfinite(right))
    ):
        raise ValueError("covariance arrays must have one common finite shape.")
    difference = left - right
    absolute_l2 = float(np.linalg.norm(difference))
    maximum_absolute = float(np.max(np.abs(difference), initial=0.0))
    left_l2 = float(np.linalg.norm(left))
    right_l2 = float(np.linalg.norm(right))
    relative_scale = max(left_l2, right_l2, 1.0e-15)
    threshold = _roundoff_threshold(max(left_l2, right_l2))
    return {
        "shape": list(left.shape),
        "actual_l2": left_l2,
        "expected_l2": right_l2,
        "absolute_l2": absolute_l2,
        "maximum_absolute": maximum_absolute,
        "relative_l2": absolute_l2 / relative_scale,
        "roundoff_threshold_l2": threshold,
        "numerically_covariant": bool(absolute_l2 <= threshold),
    }


def _tensor_array(value) -> np.ndarray:
    return np.asarray(value.detach().cpu(), dtype=float).copy()


def _single_tensor_value(output: dict[str, object], name: str) -> float:
    import torch

    value = output.get(name)
    if not torch.is_tensor(value) or value.numel() != 1:
        raise RuntimeError(f"MACE-POLAR output {name!r} is not one tensor scalar.")
    result = float(value.detach().cpu().reshape(()))
    if not math.isfinite(result):
        raise RuntimeError(f"MACE-POLAR output {name!r} is non-finite.")
    return result


def _source_record(path_source, *, role: str) -> dict[str, object]:
    source_type = path_source if isinstance(path_source, type) else type(path_source)
    raw_path = inspect.getsourcefile(source_type)
    if raw_path is None:
        raise RuntimeError(f"Unable to bind external source for {role}.")
    path = Path(raw_path).resolve(strict=True)
    _, first_line = inspect.getsourcelines(source_type)
    return {
        "role": role,
        "module": source_type.__module__,
        "qualname": source_type.__qualname__,
        "resolved_path": str(path),
        "first_source_line": int(first_line),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _axis_stencil_record(module, *, expected_offset: float) -> dict[str, object]:
    axes = np.stack([_tensor_array(getattr(module, name)) for name in ("x", "y", "z")])
    offset = float(getattr(module, "offset"))
    expected = expected_offset * np.eye(3)
    # The checkpoint is loaded from float32 and then promoted to float64, so
    # registered buffers retain the original float32 decimal rounding.
    buffer_error = float(np.max(np.abs(axes - expected)))
    if offset != expected_offset or buffer_error > 5.0e-8:
        raise RuntimeError("graph_longrange fixed-axis stencil changed.")
    return {
        "offset_angstrom": offset,
        "registered_axis_displacements_angstrom": axes.tolist(),
        "buffer_max_abs_from_python_offset_angstrom": buffer_error,
        "laboratory_fixed_cartesian_axes": True,
        "continuous_so3_orbit_closed": False,
    }


def _zero_reduced(model, atom_count: int) -> np.ndarray:
    reduced, gauge = model.duality_map.decompose_field(
        np.zeros((atom_count, 8), dtype=float),
        atom_count=atom_count,
        total_charge=0.0,
    )
    if gauge != 0.0:
        raise RuntimeError("The zero-field canary acquired a nonzero gauge.")
    return reduced


def _checkpoint_output(model, atoms: Atoms, reduced: np.ndarray):
    return model._checkpoint_output_torch(  # noqa: SLF001 - evidence probe
        model._geometry(atoms, requires_grad=False),  # noqa: SLF001
        model._reduced_tensor(  # noqa: SLF001
            reduced,
            atom_count=len(atoms),
            total_charge=0.0,
            requires_grad=False,
        ),
    )


def _full_model_record(model, atoms: Atoms, rotated_atoms: Atoms, rotation: np.ndarray):
    count = len(atoms)
    reduced = _zero_reduced(model, count)
    zero_field = np.zeros((count, 8), dtype=float)
    radial_rotation = radial_gto_source_rotation_matrix(rotation, atom_count=count)

    capture: list[dict[str, object]] = []
    feature_module = (
        model._base._calculator.model.electric_potential_descriptor.realspace_features
    )  # noqa: E501,SLF001

    def capture_first(_module, _args, kwargs, output):
        if capture:
            return
        if not isinstance(output, (tuple, list)) or len(output) != 3:
            raise RuntimeError("Unexpected graph_longrange feature output contract.")
        capture.append(
            {
                "source_feats": kwargs["source_feats"].detach().clone(),
                "node_positions": kwargs["node_positions"].detach().clone(),
                "batch": kwargs["batch"].detach().clone(),
                "features": output[0].detach().clone(),
            }
        )

    handle = feature_module.register_forward_hook(capture_first, with_kwargs=True)
    try:
        base_output = _checkpoint_output(model, atoms, reduced)
    finally:
        handle.remove()
    if len(capture) != 1:
        raise RuntimeError(
            "The first molecular-realspace feature call was not captured."
        )
    rotated_output = _checkpoint_output(model, rotated_atoms, reduced)

    base_source = model.evaluate_source(atoms, zero_field)
    rotated_source = model.evaluate_source(rotated_atoms, zero_field)
    expected_source = (radial_rotation @ base_source.reshape(-1)).reshape(
        base_source.shape
    )
    base_gradient = model.fixed_field_coordinate_gradient(
        atoms, reduced, total_charge=0.0
    )
    rotated_gradient = model.fixed_field_coordinate_gradient(
        rotated_atoms, reduced, total_charge=0.0
    )
    expected_gradient = base_gradient @ rotation.T

    base_density = _tensor_array(base_output["density_coefficients"])
    rotated_density = _tensor_array(rotated_output["density_coefficients"])
    raw_rotation = radial_rotation[:8, :8][2:5, 2:5]
    density_rotation = np.zeros((4, 4), dtype=float)
    density_rotation[0, 0] = 1.0
    density_rotation[1:, 1:] = raw_rotation
    expected_density = base_density @ density_rotation.T

    scalar_names = (
        "energy",
        "interaction_energy",
        "electron_energy",
        "electrostatic_energy",
    )
    component_records = {
        name: _scalar_rotation_record(
            _single_tensor_value(base_output, name),
            _single_tensor_value(rotated_output, name),
        )
        for name in scalar_names
    }
    anchored_base = model.energy_eV(
        atoms, reduced, total_charge=0.0, gauge_potential=0.0
    )
    anchored_rotated = model.energy_eV(
        rotated_atoms, reduced, total_charge=0.0, gauge_potential=0.0
    )
    anchored = _scalar_rotation_record(anchored_base, anchored_rotated)
    if not math.isclose(
        anchored_base,
        component_records["energy"]["base"],
        rel_tol=0.0,
        abs_tol=_roundoff_threshold(abs(anchored_base)),
    ):
        raise RuntimeError("Zero-field anchored and checkpoint energies disagree.")

    return {
        "reduced_zero_field_dimension": int(reduced.size),
        "anchored_field_energy_eV": anchored,
        "checkpoint_energy_components_eV": component_records,
        "energy_gradient_source_covariance": _covariance_record(
            rotated_source, expected_source
        ),
        "original_density_covariance": _covariance_record(
            rotated_density, expected_density
        ),
        "fixed_field_coordinate_gradient_covariance_eV_per_A": _covariance_record(
            rotated_gradient, expected_gradient
        ),
        "base_fixed_field_coordinate_gradient_eV_per_A": base_gradient.tolist(),
        "rotated_fixed_field_coordinate_gradient_eV_per_A": (rotated_gradient.tolist()),
        "first_feature_call": capture[0],
        "density_rotation": density_rotation,
        "radial_rotation": radial_rotation,
    }


def _isolated_upstream_record(model, full_record: dict[str, object], rotation):
    import torch

    network = model._base._calculator.model  # noqa: SLF001 - evidence probe
    feature_module = network.electric_potential_descriptor.realspace_features
    energy_module = network.coulomb_energy.realspace_energy
    captured = full_record.pop("first_feature_call")
    density_rotation = np.asarray(full_record.pop("density_rotation"), dtype=float)
    radial_rotation = np.asarray(full_record.pop("radial_rotation"), dtype=float)
    source = captured["source_feats"]
    positions = captured["node_positions"]
    batch = captured["batch"]
    if source.shape != (len(positions), 1, 4):
        raise RuntimeError("Unexpected first graph_longrange source shape.")
    source_block = torch.as_tensor(
        density_rotation.T.copy(), dtype=source.dtype, device=source.device
    )
    spatial_rotation = torch.as_tensor(
        rotation.T, dtype=positions.dtype, device=positions.device
    )
    exactly_rotated_source = source @ source_block
    rotated_positions = positions @ spatial_rotation
    base_features = feature_module(
        source_feats=source,
        node_positions=positions,
        batch=batch,
    )[0]
    rotated_features = feature_module(
        source_feats=exactly_rotated_source,
        node_positions=rotated_positions,
        batch=batch,
    )[0]
    field_block = torch.as_tensor(
        radial_rotation[:8, :8].T.copy(),
        dtype=base_features.dtype,
        device=base_features.device,
    )
    expected_features = base_features @ field_block

    base_density = torch.as_tensor(
        full_record["base_original_density"],
        dtype=source.dtype,
        device=source.device,
    )
    exactly_rotated_density = base_density @ source_block
    base_energy = energy_module(
        source_feats=base_density,
        positions=positions,
        batch=batch,
    )
    rotated_energy = energy_module(
        source_feats=exactly_rotated_density,
        positions=rotated_positions,
        batch=batch,
    )

    return {
        "input_source_sha256": canonical_json_sha256(_tensor_array(source).tolist()),
        "input_source_was_rotated_exactly_by_declared_l0_plus_l1_representation": True,
        "feature_operator_covariance": _covariance_record(
            _tensor_array(rotated_features), _tensor_array(expected_features)
        ),
        "energy_operator_rotation": _scalar_rotation_record(
            float(base_energy.detach().cpu().reshape(())),
            float(rotated_energy.detach().cpu().reshape(())),
        ),
        "feature_stencil": _axis_stencil_record(feature_module, expected_offset=0.1),
        "energy_stencil": _axis_stencil_record(energy_module, expected_offset=0.02),
    }


def main() -> None:
    args = _parse_args()
    repository = RepositorySnapshot.capture(Path(__file__).parents[2])
    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    started = time.perf_counter()
    import torch

    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)

    atoms = _water()
    rotation = _rotation()
    rotated_atoms = atoms.copy()
    rotated_atoms.positions = atoms.positions @ rotation.T
    base = build_official_mace_polar_1_m_radial_gto_adapter(
        device=args.device,
        checkpoint_path=checkpoint,
    )
    model = MACEPolarVariationalFieldEnergy(base)
    if base.release_contract.structural_so3_equivariance_admitted:
        raise RuntimeError("The audited evaluator unexpectedly advertises SO(3).")

    full = _full_model_record(model, atoms, rotated_atoms, rotation)
    # Retain the exact base density for the isolated upstream energy probe, but
    # do not duplicate it in the final full-model record.
    reduced = _zero_reduced(model, len(atoms))
    base_output = _checkpoint_output(model, atoms, reduced)
    full["base_original_density"] = _tensor_array(
        base_output["density_coefficients"]
    ).tolist()
    isolated = _isolated_upstream_record(model, full, rotation)
    full.pop("base_original_density")

    network = base._calculator.model  # noqa: SLF001 - evidence probe
    descriptor = network.electric_potential_descriptor
    coulomb = network.coulomb_energy
    external_sources = {
        record["role"]: record
        for record in (
            _source_record(network, role="mace_polar_model"),
            _source_record(descriptor, role="gto_electrostatic_features"),
            _source_record(
                descriptor.realspace_features,
                role="realspace_finite_difference_features",
            ),
            _source_record(coulomb, role="gto_electrostatic_energy"),
            _source_record(
                coulomb.realspace_energy,
                role="realspace_finite_difference_energy",
            ),
        )
    }
    protocol = {
        "random_seed": RANDOM_SEED,
        "rotation_seed": ROTATION_SEED,
        "geometry": {
            "formula": atoms.get_chemical_formula(),
            "atomic_numbers": atoms.numbers.tolist(),
            "positions_A": atoms.positions.tolist(),
            "charge": 0,
            "multiplicity": 1,
        },
        "rotation_matrix": rotation.tolist(),
        "external_field": "exact-zero-eight-channel-field",
        "roundoff_multiplier": ROUNDOFF_MULTIPLIER,
        "counterexample_logic": (
            "one material proper-rotation failure disproves a global structural "
            "SO(3) claim for the current evaluator"
        ),
    }
    structural = {
        "model_release_contract": base.release_contract.metadata(),
        "external_python_sources": external_sources,
        "upstream_molecular_realspace_definition": (
            "l=1 multipoles and receiver vectors are represented by scalar "
            "charges displaced along fixed laboratory x/y/z axes"
        ),
        "finite_fixed_axis_stencil_is_closed_under_continuous_so3": False,
    }
    feature_counterexample = not isolated["feature_operator_covariance"][
        "numerically_covariant"
    ]
    energy_counterexample = not isolated["energy_operator_rotation"][
        "numerically_equal"
    ]
    full_energy_counterexample = not full["anchored_field_energy_eV"][
        "numerically_equal"
    ]
    decision = {
        "isolated_feature_operator_so3_counterexample_detected": (
            feature_counterexample
        ),
        "isolated_energy_operator_so3_counterexample_detected": (energy_counterexample),
        "full_anchored_model_scalar_so3_counterexample_detected": (
            full_energy_counterexample
        ),
        "first_broken_operator": (
            "graph_longrange.realspace_electrostatics."
            "RealSpaceFiniteDifferenceElectrostaticFeatures"
        ),
        "eight_channel_field_transform_exonerated_by_zero_field": True,
        "route2_continuum_exonerated_by_model_only_counterexample": True,
        "current_molecular_realspace_model_global_so3_admitted": False,
        "tier_v_admitted": False,
        "public_force_admitted": False,
    }
    if not all(
        (
            feature_counterexample,
            energy_counterexample,
            full_energy_counterexample,
        )
    ):
        raise RuntimeError(
            "The frozen MACE molecular-realspace counterexample vanished."
        )

    measured = {
        "protocol": protocol,
        "structural_evidence": structural,
        "full_model_zero_field_rotation": full,
        "isolated_upstream_realspace_operators": isolated,
        "decision": decision,
    }
    repository.assert_unchanged()
    source_paths = collect_loaded_repository_sources(
        repository.root, required_paths=REQUIRED_SOURCE_PATHS
    )
    source_hashes = committed_source_hashes(repository, source_paths)
    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": ARTIFACT_KIND,
        "status": "model-long-range-so3-counterexample-detected-not-admitted",
        "claim_boundary": (
            "This one-water, one-rotation result is a counterexample to global "
            "SO(3) covariance of the exact pinned molecular-realspace inference "
            "path. It identifies the first broken upstream operator and keeps "
            "all capabilities closed. It does not validate a replacement "
            "evaluator, continuum accuracy, or any Route-2 public force."
        ),
        "capabilities": NO_CAPABILITIES,
        "exact_command": shlex.join(sys.argv),
        "argv": list(sys.argv),
        "execution_git_head": repository.head,
        "execution_git_tree": repository.tree,
        "working_tree_clean": repository.clean,
        "source_files_sha256": source_hashes,
        "checkpoint": checkpoint_record(checkpoint),
        "runtime": runtime_record(),
        "device": args.device,
        "dtype": model.dtype,
        **measured,
        "measurement_sha256": canonical_json_sha256(measured),
        "runtime_seconds": time.perf_counter() - started,
    }
    repository.assert_unchanged()
    artifact = write_external_json_artifact(repository, args.output, payload)
    repository.assert_unchanged()
    print(
        "ROUTE2_MACE_REALSPACE_SO3_NOGO="
        + json.dumps(
            {
                "artifact": artifact,
                "measurement_sha256": payload["measurement_sha256"],
                "decision": decision,
                "capabilities": NO_CAPABILITIES,
                "runtime_seconds": payload["runtime_seconds"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
