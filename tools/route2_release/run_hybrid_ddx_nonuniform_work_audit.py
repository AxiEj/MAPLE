#!/usr/bin/env python3
"""Audit hybrid ddPCM half-work and nonuniform checkpoint work semantics.

This target-independent canary answers the terminal question left by the
uniform-field audit and the 2026-08-17 Pro reviews:

* does the heterogeneous point-permanent/GTO-induced continuum obey its own
  exact scalar-gradient half-work identity;
* can the MLIP-driving external-MEP field be substituted into a naive endpoint
  source pairing (it must not be unless independently proved);
* is the converged reaction field outside the official three-dimensional
  uniform-electric-field subspace; and
* does the installed upstream checkpoint expose an official arbitrary
  nonuniform explicit-work input, rather than only the uniform ``E dot mu``
  branch already audited?

No experimental hydration/solvation target is read.  A passing artifact closes
an identifiability question; it admits no public capability and does not select
an energy ledger by chemical accuracy.
"""

from __future__ import annotations

import argparse
import hashlib
from importlib import metadata
import inspect
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import time

from ase import Atoms
import numpy as np

from maple.function.calculator.extra_correction.implicit.electrostatic_pairing import (
    MACE_POLAR_MODEL_FEATURE_FIELD_INDICES,
)
from maple.function.calculator.extra_correction.implicit.smd_cds import (
    smd_water_coulomb_radii,
)
from maple.solvation.api.profiles import (
    MACE_POLAR_ANALYTIC_GAUSSIAN_MULTIPOLE_EVALUATOR_ID,
)
from maple.solvation.continuum import (
    SeparatedSourceDDXBackend,
    embed_atomic_l1_in_first_radial_channel,
)
from maple.solvation.coupling.metrics import MACE_POLAR_RADIAL_GTO_PAIRING
from maple.solvation.experimental import MACE_MDPPolarHybridDDXEnergy
from maple.solvation.models import (
    MACEPolarOriginalSourceNativeFieldAdapter,
    build_mace_mdp_anchored_mace_polar_hybrid,
    build_mace_mdp_moment_adapter,
    build_official_mace_polar_1_m_radial_gto_adapter,
)
from maple.solvation.release import (
    audit_charging_path,
    canonical_json_sha256,
    checkpoint_record,
    runtime_record,
)

SCHEMA_VERSION = "route2-hybrid-ddx-nonuniform-work-audit-v1"
DEFAULT_MDP_CHECKPOINT = Path.home() / ".cache/mace/MACE-MDP.model"
DEFAULT_POLAR_CHECKPOINT = Path.home() / ".cache/mace/MACEPOLAR1Mmodel"
RANDOM_SEED = 20260817
HALF_WORK_ATOL_EV = 2.0e-9
NATIVE_ENDPOINT_MIN_MISMATCH_EV = 1.0e-7
FIELD_GRADIENT_MIN_DIFFERENCE = 1.0e-7
NONUNIFORM_SUBSPACE_MIN_RELATIVE_RESIDUAL = 1.0e-3
NO_CAPABILITIES = {tier: False for tier in ("E", "F", "H", "V", "M")}
SOURCE_FILES = (
    "maple/solvation/continuum/radial_gto_ddx.py",
    "maple/solvation/continuum/separated_source_ddx.py",
    "maple/solvation/experimental/mace_mdp_polar_ddx.py",
    "maple/solvation/models/mace_mdp.py",
    "maple/solvation/models/mace_mdp_polar_hybrid.py",
    "maple/solvation/models/mace_polar_separated.py",
    "maple/solvation/release/field_semantics.py",
    "tools/route2_release/run_hybrid_ddx_nonuniform_work_audit.py",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ("git", *args), cwd=repo, text=True, stderr=subprocess.DEVNULL
    ).strip()


def _version(package: str) -> str:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return "unavailable"


def _water() -> Atoms:
    return Atoms(
        "OH2",
        positions=np.asarray(
            [[0.0, 0.0, 0.0], [0.9572, 0.0, 0.0], [-0.2399872, 0.927297, 0.0]],
            dtype=float,
        ),
        info={"charge": 0, "multiplicity": 1},
    )


def _configure_torch() -> object:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch = __import__("torch")
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    return torch


def _tensor_numpy(value: object) -> np.ndarray:
    return np.asarray(value.detach().cpu(), dtype=float)


def _uniform_native_basis(radial: object, atoms: Atoms) -> np.ndarray:
    """Return the three official uniform-field directions in native U8."""

    calculator = radial._calculator
    centered = atoms.positions - np.mean(atoms.positions, axis=0, keepdims=True)
    projector = _tensor_numpy(calculator._reaction_projector.upstream.matrix)
    columns: list[np.ndarray] = []
    for axis in range(3):
        gradient = np.zeros(3, dtype=float)
        gradient[axis] = 1.0
        potential = centered @ gradient
        node_values = np.concatenate(
            (potential[:, None], np.broadcast_to(gradient, (len(atoms), 3))),
            axis=1,
        )
        model_input = node_values[:, list(MACE_POLAR_MODEL_FEATURE_FIELD_INDICES)]
        features = np.einsum("pf,nf->np", projector, model_input)
        native = np.linalg.solve(radial.field_transform.matrix, features.T).T
        columns.append(native.reshape(-1))
    return np.column_stack(columns)


def _upstream_uniform_work_surface(radial: object) -> dict[str, object]:
    """Content-address the installed upstream work API and its exact branches."""

    model = radial._calculator.model
    model_type = type(model)
    raw_path = inspect.getsourcefile(model_type)
    if raw_path is None:
        raise RuntimeError("Unable to locate the installed upstream MACE source.")
    path = Path(raw_path).resolve(strict=True)
    forward = model_type.forward
    signature = inspect.signature(forward)
    source_lines, first_line = inspect.getsourcelines(forward)
    source = "".join(source_lines)
    parameter_names = tuple(signature.parameters)
    candidate_nonuniform_parameters = tuple(
        name
        for name in parameter_names
        if name
        in {
            "external_potential",
            "external_potential_features",
            "node_potential",
            "node_potentials",
            "nonuniform_field",
            "potential_features",
        }
    )
    exact_markers = {
        "uniform_external_field_parameter": "external_field" in parameter_names,
        "uniform_external_potential_construction": (
            "external_potential = torch.hstack" in source and "external_field" in source
        ),
        "uniform_dipole_explicit_work": (
            "external_potential[:, 1:] * total_dipole" in source
        ),
        "half_field_added_to_both_spin_channels": (
            "half_external_field = 0.5 * self.external_field_contribution" in source
            and "field_feats_alpha + half_external_field" in source
            and "field_feats_beta + half_external_field" in source
        ),
    }
    matching_lines = []
    needles = (
        "external_field:",
        "external_potential = torch.hstack",
        "half_external_field = 0.5",
        "external_potential[:, 1:] * total_dipole",
    )
    for offset, line in enumerate(source_lines):
        if any(needle in line for needle in needles):
            matching_lines.append({"line": first_line + offset, "text": line.strip()})
    arbitrary_api = bool(candidate_nonuniform_parameters)
    uniform_branch_verified = all(exact_markers.values())
    return {
        "module": model_type.__module__,
        "qualname": model_type.__qualname__,
        "resolved_path": str(path),
        "sha256": _sha256(path),
        "forward_signature": str(signature),
        "forward_parameter_names": list(parameter_names),
        "candidate_arbitrary_nonuniform_parameters": list(
            candidate_nonuniform_parameters
        ),
        "source_markers": exact_markers,
        "matching_source_lines": matching_lines,
        "official_arbitrary_nonuniform_explicit_work_api_detected": arbitrary_api,
        "official_uniform_E_dot_mu_branch_verified": uniform_branch_verified,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mdp-checkpoint", type=Path, default=DEFAULT_MDP_CHECKPOINT)
    parser.add_argument(
        "--polar-checkpoint", type=Path, default=DEFAULT_POLAR_CHECKPOINT
    )
    parser.add_argument(
        "--polar-device",
        choices=("cpu", "cuda"),
        default=os.environ.get("MAPLE_ROUTE2_MACE_DEVICE", "cuda"),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, object]:
    started = time.perf_counter()
    _configure_torch()
    repo = Path(__file__).resolve().parents[2]
    atoms = _water()
    mdp_checkpoint = args.mdp_checkpoint.expanduser().resolve(strict=True)
    polar_checkpoint = args.polar_checkpoint.expanduser().resolve(strict=True)

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
    electronic = MACEPolarOriginalSourceNativeFieldAdapter(radial)
    hybrid = build_mace_mdp_anchored_mace_polar_hybrid(
        permanent=mdp,
        response=electronic,
    )
    continuum = SeparatedSourceDDXBackend(
        tuple(atoms.get_chemical_symbols()),
        smd_water_coulomb_radii(atoms.get_chemical_symbols()),
        continuum_model="pcm",
        dielectric=78.39,
        lmax=8,
        n_lebedev=194,
        solver_tolerance=1.0e-12,
        eta=0.1,
        n_proc=1,
    )
    evaluator = MACE_MDPPolarHybridDDXEnergy(
        atoms,
        hybrid=hybrid,
        continuum=continuum,
    )
    root = evaluator.solve(atoms)
    radial_source = embed_atomic_l1_in_first_radial_channel(root.induced_source4)
    continuum_state = continuum.prepare(
        atoms, evaluator.anchor.permanent_source4
    ).solve(radial_source)
    continuum_replay = continuum_state.state_sha256 == root.continuum_state_sha256

    energy_dual_work = continuum_state.energy_dual_work_ev
    twice_energy = 2.0 * continuum_state.polarization_energy_ev
    half_work_error = abs(energy_dual_work - twice_energy)
    naive_native_endpoint_work = MACE_POLAR_RADIAL_GTO_PAIRING.pair(
        embed_atomic_l1_in_first_radial_channel(continuum_state.permanent_source)
        + continuum_state.radial_source,
        continuum_state.model_field,
    )
    native_endpoint_mismatch = abs(naive_native_endpoint_work - energy_dual_work)
    field_gradient_difference = float(
        np.linalg.norm(
            continuum_state.model_field - continuum_state.energy_source_gradient
        )
    )

    uniform_basis = _uniform_native_basis(radial, atoms)
    field_vector = root.native_field_ev.reshape(-1)
    uniform_coefficients, *_ = np.linalg.lstsq(uniform_basis, field_vector, rcond=None)
    uniform_projection = uniform_basis @ uniform_coefficients
    nonuniform_residual = float(np.linalg.norm(field_vector - uniform_projection))
    field_norm = float(np.linalg.norm(field_vector))
    nonuniform_relative_residual = nonuniform_residual / max(
        field_norm, np.finfo(float).tiny
    )

    zero = np.zeros_like(root.native_field_ev)
    raw_charging = audit_charging_path(
        energy=lambda field: hybrid.conditioned_raw_energy_ev(atoms, field),
        gradient=lambda field: electronic.conditioned_raw_energy_field_gradient(
            atoms, field
        ),
        endpoint_field=root.native_field_ev,
    )
    raw_endpoint_gradient = electronic.conditioned_raw_energy_field_gradient(
        atoms, root.native_field_ev
    )
    raw_endpoint_directional = float(
        np.vdot(raw_endpoint_gradient, root.native_field_ev)
    )
    raw_delta = hybrid.conditioned_raw_energy_ev(
        atoms, root.native_field_ev
    ) - hybrid.conditioned_raw_energy_ev(atoms, zero)

    upstream = _upstream_uniform_work_surface(radial)
    gates = {
        "root_residual": root.primal_residual_ev < 1.0e-10,
        "continuum_state_replay": continuum_replay,
        "continuum_energy_dual_half_work": half_work_error <= HALF_WORK_ATOL_EV,
        "naive_native_endpoint_rejected": (
            native_endpoint_mismatch >= NATIVE_ENDPOINT_MIN_MISMATCH_EV
        ),
        "model_field_distinct_from_energy_gradient": (
            field_gradient_difference >= FIELD_GRADIENT_MIN_DIFFERENCE
        ),
        "reaction_field_materially_nonuniform": (
            nonuniform_relative_residual >= NONUNIFORM_SUBSPACE_MIN_RELATIVE_RESIDUAL
        ),
        "native_raw_charging_identity": raw_charging.passed,
        "upstream_uniform_work_branch_verified": upstream[
            "official_uniform_E_dot_mu_branch_verified"
        ],
        "upstream_arbitrary_nonuniform_work_api_absent": not upstream[
            "official_arbitrary_nonuniform_explicit_work_api_detected"
        ],
    }
    status = (
        "pass-outcome-c-no-upstream-nonuniform-work-endpoint"
        if all(gates.values())
        else "fail-unresolved-nonuniform-work-audit"
    )

    source_hashes = {
        name: _sha256((repo / name).resolve(strict=True)) for name in SOURCE_FILES
    }
    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "target-independent-nonuniform-work-terminal-audit",
        "status": status,
        "claim_boundary": (
            "This artifact establishes the current continuum half-work and the "
            "absence of an official arbitrary-nonuniform upstream explicit-work "
            "endpoint. It reads no solvation target, selects neither Phi0 nor "
            "Phi_raw by accuracy, and admits no E/F/H/V/M capability."
        ),
        "capabilities": NO_CAPABILITIES,
        "exact_command": shlex.join(sys.argv),
        "execution": {
            "git_head": _git(repo, "rev-parse", "HEAD"),
            "git_head_tree": _git(repo, "rev-parse", "HEAD^{tree}"),
            "working_tree_status": _git(repo, "status", "--porcelain=v1"),
            "source_files_sha256": source_hashes,
        },
        "configuration": {
            "hybrid_configuration_sha256": hybrid.configuration_sha256(),
            "hybrid_provenance_sha256": hybrid.provenance_sha256,
            "continuum_configuration_sha256": continuum.configuration_sha256(),
            "continuum_provenance_sha256": continuum.provenance_sha256,
            "evaluator_configuration_sha256": evaluator.configuration_sha256(),
            "mdp_checkpoint": checkpoint_record(mdp_checkpoint),
            "polar_checkpoint": checkpoint_record(polar_checkpoint),
            "polar_device": args.polar_device,
            "continuum": "ddPCM",
            "dielectric": 78.39,
            "lmax": 8,
            "n_lebedev": 194,
            "permanent_source": "MACE-MDP point q/p",
            "induced_source": "MACE-POLAR zero-anchored 1.5-A Gaussian q/p",
            "model_drive": "external-MEP 1.5/3.0-A phi-side receiver",
        },
        "root": {
            "root_sha256": root.root_sha256,
            "continuum_state_sha256": root.continuum_state_sha256,
            "primal_residual_eV": root.primal_residual_ev,
            "cold_iterations": root.cold_iterations,
            "wide_iterations": root.wide_iterations,
            "native_field_l2": field_norm,
            "uniform_subspace_coefficients": uniform_coefficients.tolist(),
            "uniform_subspace_projection_residual_l2": nonuniform_residual,
            "uniform_subspace_projection_relative_residual": (
                nonuniform_relative_residual
            ),
        },
        "continuum_half_work": {
            "polarization_energy_eV": continuum_state.polarization_energy_ev,
            "twice_polarization_energy_eV": twice_energy,
            "energy_dual_work_eV": energy_dual_work,
            "absolute_error_eV": half_work_error,
            "naive_native_endpoint_work_eV": naive_native_endpoint_work,
            "naive_native_endpoint_absolute_mismatch_eV": (native_endpoint_mismatch),
            "model_field_minus_energy_gradient_l2": field_gradient_difference,
            "interpretation": (
                "The source couples to the continuum scalar through its own "
                "direct-sum energy cotangents. The MLIP model field is a "
                "different receiver and is not an admitted endpoint work."
            ),
        },
        "conditioned_raw_energy": {
            **raw_charging.as_dict(),
            "endpoint_gradient_dot_field_eV": raw_endpoint_directional,
            "endpoint_gradient_dot_field_minus_energy_difference_eV": (
                raw_endpoint_directional - raw_delta
            ),
            "interpretation": (
                "The same-graph charging line integral reproduces the raw energy "
                "difference. A single endpoint gradient contraction need not."
            ),
        },
        "upstream_work_surface": upstream,
        "decision": {
            "pro_outcome": "C",
            "uniform_work_extrapolates_to_nonuniform": False,
            "explicit_nonuniform_endpoint_correction_admitted": False,
            "phi0_operational_scalar_remains_available": True,
            "phi_raw_operational_scalar_remains_available": True,
            "physical_ledger_selected": False,
            "tier_v_admitted": False,
            "next_scientific_gate": (
                "matched target-independent electronic-distortion and same-equation "
                "continuum-component comparison of preregistered operational ledgers"
            ),
        },
        "gates": gates,
        "runtime": {
            **runtime_record(),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "packages": {
                "ase": _version("ase"),
                "mace-torch": _version("mace-torch"),
                "numpy": _version("numpy"),
                "pyddx": _version("pyddx"),
                "scipy": _version("scipy"),
                "torch": _version("torch"),
            },
        },
        "runtime_seconds": time.perf_counter() - started,
    }
    payload["measurement_sha256"] = canonical_json_sha256(
        {
            "configuration": payload["configuration"],
            "root": payload["root"],
            "continuum_half_work": payload["continuum_half_work"],
            "conditioned_raw_energy": payload["conditioned_raw_energy"],
            "upstream_work_surface": payload["upstream_work_surface"],
            "decision": payload["decision"],
            "gates": payload["gates"],
        }
    )
    payload["artifact_sha256"] = canonical_json_sha256(payload)
    return payload


def main() -> None:
    args = _parse_args()
    payload = run(args)
    output = args.output.expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    try:
        with output.open("x", encoding="utf-8") as handle:
            handle.write(encoded)
    except FileExistsError as exc:
        raise RuntimeError(f"Refusing to overwrite audit artifact: {output}") from exc
    print(
        "ROUTE2_HYBRID_DDX_NONUNIFORM_WORK_AUDIT="
        + json.dumps(
            {
                "output": str(output.resolve()),
                "status": payload["status"],
                "measurement_sha256": payload["measurement_sha256"],
                "artifact_sha256": payload["artifact_sha256"],
                "gates": payload["gates"],
            },
            sort_keys=True,
        )
    )
    if not all(payload["gates"].values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
