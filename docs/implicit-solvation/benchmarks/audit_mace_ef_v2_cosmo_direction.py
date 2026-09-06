#!/usr/bin/env python3
"""A fixed, physically generated COSMO reaction-potential ray for V2 inputs."""

from __future__ import annotations

import argparse
from functools import partial
import hashlib
import io
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for path in (ROOT, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from audit_mace_ef_v2_trace_inputs import (
    TraceProbe,
    _builder,
    verified_bytes,
    state_digest,
)
from audit_mace_ef_cos_input_semantics import _source_hashes, write_json_once


def directional_curvature(derivative, center, steps):
    rows = []
    for step in steps:
        if not np.isfinite(step) or step <= 0:
            raise ValueError("directional steps must be finite and positive")
        plus, minus = float(derivative(center + step)), float(derivative(center - step))
        if not np.isfinite([plus, minus]).all():
            raise ValueError("nonfinite directional derivative")
        rows.append(
            {
                "center_t": center,
                "step_t": step,
                "derivative_plus_ev": plus,
                "derivative_minus_ev": minus,
                "curvature_ev": (plus - minus) / (2 * step),
            }
        )
    return rows


def _directional_derivative(samples, jet, t):
    return float(np.sum(np.asarray(samples[str(t)]["source_cartesian"]) * jet))


def ray_states(centers, steps):
    return sorted(
        set(centers)
        | {
            center + sign * step
            for center in centers
            for step in steps
            for sign in (-1, 1)
        }
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--input-result", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    protocol_bytes = args.preregistration.read_bytes()
    protocol = json.loads(protocol_bytes)
    evidence_bytes = verified_bytes(
        args.input_result, protocol["input_ablation_result_sha256"]
    )
    evidence = json.loads(evidence_bytes)
    primary = [r for r in evidence["records"] if r["mode"] == "centered_v"]
    if not primary or not all(
        r["gate_assessment"]["all_executed_checks_passed"] for r in primary
    ):
        raise RuntimeError("primary input canaries have not passed")
    selected = next(v for v in evidence["variants"] if v["mode"] == "centered_v")
    if selected["checkpoint_sha256"] != protocol["candidate_checkpoint_sha256"]:
        raise ValueError("candidate is not the frozen primary arm")
    candidate_bytes = verified_bytes(
        args.candidate, protocol["candidate_checkpoint_sha256"]
    )
    paths = [
        Path(__file__),
        HERE / "audit_mace_ef_v2_trace_inputs.py",
        HERE / "audit_mace_ef_cos_input_semantics.py",
        ROOT / "maple/function/cosmors_torch/segment_cosmo.py",
        ROOT / "maple/function/cosmors_torch/surface.py",
        ROOT / "maple/function/mlip_cosmo_rs.py",
    ]
    paths += [
        ROOT / "maple/function/calculator/extra_correction/implicit" / n
        for n in (
            "mace_polar_ef.py",
            "mace_polar_ef_specs.py",
            "electrostatic_pairing.py",
        )
    ]
    sources = _source_hashes(paths)
    import torch
    from maple.function.calculator.extra_correction.implicit.mace_polar_ef import (
        MACEPolarEFEnergyModel,
    )
    from maple.function.cosmors_torch.segment_cosmo import (
        TorchSegmentCOSMO,
        TorchSegmentCOSMOConfig,
    )
    from maple.function.mlip_cosmo_rs import open_cosmors_24a_cavity_radii
    from ase.data import chemical_symbols

    records = []
    for molecule in protocol["molecules"]:
        builder = _builder(args.checkpoint, molecule)
        positions = np.asarray(molecule["positions_angstrom"], dtype=float)
        original = TraceProbe(builder.model, builder, positions)
        reference = original.sample(original.jet(np.zeros(3)))
        source_cartesian = np.asarray(reference["source_cartesian"])
        raw = MACEPolarEFEnergyModel._raw_source_from_cartesian_gradient(
            torch.tensor(source_cartesian, dtype=torch.float64)
        ).numpy()
        continuum = TorchSegmentCOSMO(
            TorchSegmentCOSMOConfig(
                atomic_numbers=tuple(molecule["atomic_numbers"]),
                angular_degree=protocol["angular_degree"],
                radii_angstrom=tuple(
                    open_cosmors_24a_cavity_radii(
                        [chemical_symbols[z] for z in molecule["atomic_numbers"]]
                    )
                ),
            )
        )
        if (
            continuum.config.configuration_sha256
            != protocol["continuum_configuration_sha256"][molecule["name"]]
        ):
            raise RuntimeError("conductor configuration differs from frozen comparator")
        jet = continuum.drive_cartesian(positions, raw)
        boundary_energy = continuum.energy(positions, raw)
        half_error = abs(boundary_energy - 0.5 * float(np.sum(source_cartesian * jet)))
        if half_error > protocol["half_identity_tolerance_ev"]:
            raise RuntimeError("conductor half-coupling identity failed")
        record = {
            "molecule": molecule["name"],
            "geometry": molecule,
            "continuum": continuum.config.as_dict(),
            "reference_source_raw": raw.tolist(),
            "reference_reaction_jet": jet.tolist(),
            "reference_boundary_energy_ev": boundary_energy,
            "half_identity_error_ev": half_error,
            "arms": [],
        }
        for mode in protocol["arms"]:
            module = (
                builder.model
                if mode == "original"
                else torch.jit.load(
                    io.BytesIO(candidate_bytes), map_location=builder.device
                ).eval()
            )
            if state_digest(module) != evidence["state_dict_sha256"]:
                raise RuntimeError("candidate state differs from frozen input audit")
            probe = TraceProbe(module, builder, positions)

            samples = {
                str(t): probe.sample(t * jet)
                for t in ray_states(protocol["centers_t"], protocol["steps_t"])
            }
            derivative = partial(_directional_derivative, samples, jet)
            scans = [
                directional_curvature(derivative, center, protocol["steps_t"])
                for center in protocol["centers_t"]
            ]
            scan_charge_error = max(
                abs(s["conjugate_total_charge_e"] - molecule["total_charge"])
                for s in samples.values()
            )
            record["arms"].append(
                {
                    "mode": mode,
                    "samples": samples,
                    "directional_scans": scans,
                    "max_scan_state_charge_error_e": scan_charge_error,
                    "scan_state_charge_gate_passed": scan_charge_error
                    <= protocol["charge_tolerance_e"],
                }
            )
            if state_digest(module) != evidence["state_dict_sha256"]:
                raise RuntimeError("directional execution changed a tensor")
            del derivative, probe, module
        records.append(record)
        del original, builder
    if (
        sources != _source_hashes(paths)
        or args.preregistration.read_bytes() != protocol_bytes
    ):
        raise RuntimeError("source/preregistration drift")
    write_json_once(
        args.output,
        {
            "schema_version": 1,
            "artifact": "mace-ef-v2-cosmo-direction-v1",
            "preregistration_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
            "input_result_sha256": hashlib.sha256(evidence_bytes).hexdigest(),
            "candidate_checkpoint_sha256": protocol["candidate_checkpoint_sha256"],
            "source_files_sha256": sources,
            "runtime": {
                "python": sys.executable,
                "torch": str(torch.__version__),
                "numpy": str(np.__version__),
                "cuda": torch.version.cuda,
                "device": torch.cuda.get_device_name(0),
            },
            "records": records,
            "claim_boundary": protocol["claim_boundary"],
            "release_admitted": False,
        },
    )


if __name__ == "__main__":
    main()
