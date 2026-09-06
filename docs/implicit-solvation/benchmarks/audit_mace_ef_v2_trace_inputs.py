#!/usr/bin/env python3
"""Exact-tensor, source-only input ablations of the released V2 traced program.

Not a generic TorchScript editor, eager restoration, production adapter, or
admitted checkpoint exporter. Only the two scalar projection sites change;
every other ZIP member, including tensor/constant payloads, must match exactly.
Inherited trace debug ranges are not an original-eager source certificate.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for path in (ROOT, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from audit_mace_ef_cos_input_semantics import (
    response_scan,
    write_json_once,
    _source_hashes,
)

CHECKPOINT_SHA256 = "4f820d381d06bbb37b02574c38da7203e5d429407fa2a512231fc08cdbb69b6b"
SOURCE_SHA256 = "596226178515c9ada00caa559d18a5d630d877075db7993fdd7665083bfed0f6"
SOURCE_MEMBER = "best_ep9-traced/code/__torch__/mace/modules/extensions.py"
MODES = ("roundtrip", "zero_scalar", "raw_v", "centered_v")


def _replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError("expected source site changed or is ambiguous")
    return source.replace(old, new, 1)


def rewrite_source(source, mode):
    if mode not in MODES:
        raise ValueError("unknown trace-input ablation")
    if mode == "roundtrip":
        return source
    # Retain raw external_potential_values0 for the final qV energy term.
    assignment = {
        "zero_scalar": "    input_potential_for_projection = torch.zeros_like(external_potential_values0)\n",
        "raw_v": "    input_potential_for_projection = external_potential_values0\n",
        "centered_v": (
            "    gauge_sums = torch.scatter_add(torch.zeros_like(total_charge), 0, batch, external_potential_values0)\n"
            "    gauge_counts = torch.scatter_add(torch.zeros_like(total_charge), 0, batch, torch.ones_like(external_potential_values0))\n"
            "    input_potential_for_projection = torch.sub(external_potential_values0, torch.index_select(torch.div(gauge_sums, gauge_counts), 0, batch))\n"
        ),
    }[mode]
    anchor = "    external_potential_values0 = torch.squeeze(external_potential_values, -1)\n"
    source = _replace_once(source, anchor, anchor + assignment)
    for norm, scalar, start, end in (
        ("field_norm", "_74", "_77", "_84 = torch.index_put_(_82, _83, _81)"),
        ("field_norm0", "_136", "_139", "_147 = torch.index_put_(_145, _146, _144)"),
    ):
        source = _replace_once(
            source,
            f"    {norm} = torch.linalg_vector_norm(external_field, 2, [-1], True)\n",
            "",
        )
        old = f"    {scalar} = torch.zeros_like(torch.select({norm}, 1, 0), dtype=None, layout=None, device=None, pin_memory=False)\n"
        source = _replace_once(
            source, old, f"    {scalar} = input_potential_for_projection\n"
        )
        begin = f"    {start} = torch.slice(matrix, 0, 0, 9223372036854775807)\n"
        finish = f"    {end}\n"
        if source.count(begin) != 1 or source.count(finish) != 1:
            raise ValueError("scalar override block changed")
        first, last = source.index(begin), source.index(finish) + len(finish)
        if last <= first:
            raise ValueError("scalar override block order changed")
        source = source[:first] + source[last:]
    return source


def _members(data):
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("duplicate archive members")
        return {name: hashlib.sha256(archive.read(name)).hexdigest() for name in names}


def repack_source(original, member, replacement):
    output = io.BytesIO()
    with (
        zipfile.ZipFile(io.BytesIO(original)) as old,
        zipfile.ZipFile(output, "w") as new,
    ):
        if member not in old.namelist():
            raise ValueError("model source member missing")
        for info in old.infolist():
            new.writestr(
                info,
                replacement if info.filename == member else old.read(info.filename),
            )
    result = output.getvalue()
    before, after = _members(original), _members(result)
    if before.keys() != after.keys() or any(
        before[name] != after[name] for name in before if name != member
    ):
        raise RuntimeError("repack changed a non-source payload")
    return result


def build_variant(original, mode):
    if hashlib.sha256(original).hexdigest() != CHECKPOINT_SHA256:
        raise ValueError("only the exact released V2 checkpoint is accepted")
    with zipfile.ZipFile(io.BytesIO(original)) as archive:
        source = archive.read(SOURCE_MEMBER)
    if hashlib.sha256(source).hexdigest() != SOURCE_SHA256:
        raise ValueError("V2 model source bytes do not match")
    changed = rewrite_source(source.decode(), mode).encode()
    result = repack_source(original, SOURCE_MEMBER, changed)
    before, after = _members(original), _members(result)
    differing = [name for name in before if before[name] != after[name]]
    expected = [] if mode == "roundtrip" else [SOURCE_MEMBER]
    if differing != expected:
        raise RuntimeError("unexpected archive payload difference")
    return result, {
        "mode": mode,
        "checkpoint_sha256": hashlib.sha256(result).hexdigest(),
        "parent_checkpoint_sha256": CHECKPOINT_SHA256,
        "model_source_sha256": hashlib.sha256(changed).hexdigest(),
        "changed_members": differing,
        "unchanged_member_count": len(before) - len(differing),
        "original_eager_restored": False,
        "production_registered": False,
    }


def state_digest(model):
    import torch

    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        tensor = tensor.detach().cpu().contiguous()
        digest.update(
            json.dumps([name, str(tensor.dtype), list(tensor.shape)]).encode()
        )
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def verified_bytes(path, expected_sha256):
    payload = Path(path).read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError("variant checkpoint bytes changed before load")
    return payload


def assess_record(record, conditions, gates):
    """Evaluate the preregistered gates without changing any measured response."""
    samples = record["samples"]
    charge = record["geometry"]["total_charge"]
    auxiliary = max(
        abs(np.asarray(s["density"])[:, 0].sum() - charge) for s in samples.values()
    )
    conjugate = max(
        abs(np.asarray(s["source_cartesian"])[:, 0].sum() - charge)
        for s in samples.values()
    )
    gauge_energy, gauge_source = [], []
    for offset in conditions["constant_offsets_ev_per_e"]:
        sample = samples[f"gauge_{offset}"]
        gauge_energy.append(
            abs(sample["energy_ev"] - samples["affine"]["energy_ev"] - charge * offset)
        )
        gauge_source.append(
            sample_differences(sample, samples["affine"])["source_max_abs"]
        )
    curvature = None
    if "zero_center_scan" in record:
        curvature = max(
            max(row["symmetric_eigenvalues"])
            for group in ("zero_center_scan", "nonzero_center_scan")
            for row in record[group]
        )
    passed = {
        "auxiliary_charge": bool(auxiliary <= gates["charge_drift_tolerance_e"]),
        "conjugate_charge": bool(conjugate <= gates["charge_drift_tolerance_e"]),
        "gauge_energy": max(gauge_energy) <= gates["gauge_energy_abs_tolerance_ev"],
        "gauge_source": max(gauge_source) <= gates["gauge_source_abs_tolerance"],
        "sampled_affine_passivity": (
            None
            if curvature is None
            else curvature <= gates["maximum_positive_uniform_curvature"]
        ),
    }
    return {
        "auxiliary_charge_max_error_e": auxiliary,
        "conjugate_charge_max_error_e": conjugate,
        "gauge_energy_max_error_ev": max(gauge_energy),
        "gauge_source_max_abs_change": max(gauge_source),
        "sampled_maximum_curvature": curvature,
        "passed": passed,
        "all_executed_checks_passed": all(
            value for value in passed.values() if value is not None
        ),
        "full_electrostatic_domain_admitted": False,
    }


def sample_differences(left, right):
    return {
        "energy_abs_ev": abs(left["energy_ev"] - right["energy_ev"]),
        "source_max_abs": float(
            np.max(
                np.abs(np.asarray(left["source_cartesian"]) - right["source_cartesian"])
            )
        ),
        "density_max_abs": float(
            np.max(np.abs(np.asarray(left["density"]) - right["density"]))
        ),
    }


class TraceProbe:
    """Fixed-geometry graph input reuse, with a separately identified program.

    The released adapter builds inputs only; no altered module is installed in
    that adapter or represented by its old checkpoint spec/provenance.
    """

    def __init__(self, module, input_builder, positions):
        import torch

        self.module = module
        self.positions = torch.tensor(
            positions, dtype=torch.float32, device=input_builder.device
        )
        self.graph = input_builder._graph(self.positions)
        self.centered = np.asarray(positions, dtype=np.float32)
        self.centered = self.centered - self.centered.mean(0, keepdims=True)

    def jet(self, gradient):
        g = np.asarray(gradient, dtype=np.float32)
        return np.column_stack((self.centered @ g, np.tile(g, (len(self.centered), 1))))

    def sample(self, values):
        import torch

        field = torch.tensor(
            values,
            dtype=torch.float32,
            device=self.positions.device,
            requires_grad=True,
        )
        if field.shape != (len(self.positions), 4) or not bool(
            torch.isfinite(field).all()
        ):
            raise ValueError("expected a finite fixed-geometry N x 4 potential jet")
        energy, nodes, density = self.module(
            self.positions, *self.graph[:9], -field[:, 1:], field[:, 0], self.graph[9]
        )
        if (
            energy.shape != (1,)
            or nodes.shape != (len(self.positions),)
            or density.shape != field.shape
        ):
            raise RuntimeError("trace output schema differs")
        source = torch.autograd.grad(energy.sum(), field)[0]
        if not all(
            bool(torch.isfinite(x).all()) for x in (energy, nodes, density, source)
        ):
            raise FloatingPointError("trace returned nonfinite values")
        return {
            "energy_ev": float(energy.detach()),
            "source_cartesian": source.detach().cpu().tolist(),
            "density": density.detach().cpu().tolist(),
            "jet": field.detach().cpu().tolist(),
            "auxiliary_total_charge_e": float(density[:, 0].detach().double().sum()),
            "conjugate_total_charge_e": float(source[:, 0].detach().double().sum()),
        }

    def affine_gradient(self, gradient):
        source = np.asarray(self.sample(self.jet(gradient))["source_cartesian"])
        return np.sum(source[:, 0, None] * self.centered + source[:, 1:], axis=0)


def anchors(probe, conditions):
    zero = probe.jet(np.zeros(3))
    finite = probe.jet(conditions["nonzero_center_ev_per_e_angstrom"])
    nonuniform = finite.copy()
    amplitude = conditions["nonuniform_potential_amplitude_ev_per_e"]
    nonuniform[:, 0] += np.linspace(-amplitude, amplitude, len(zero))
    values = {"zero": zero, "affine": finite, "nonuniform_v": nonuniform}
    for offset in conditions["constant_offsets_ev_per_e"]:
        shifted = finite.copy()
        shifted[:, 0] += offset
        values[f"gauge_{offset}"] = shifted
    return values


def _builder(checkpoint, molecule):
    from maple.function.calculator.extra_correction.implicit.mace_polar_ef import (
        MACEPolarEFConfig,
        MACEPolarEFEnergyModel,
    )

    return MACEPolarEFEnergyModel(
        MACEPolarEFConfig(
            checkpoint_path=str(checkpoint),
            atomic_numbers=tuple(molecule["atomic_numbers"]),
            total_charge=molecule["total_charge"],
            spin_multiplicity=molecule["multiplicity"],
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    protocol_bytes = args.preregistration.read_bytes()
    protocol = json.loads(protocol_bytes)
    conditions, gates = protocol["fixed_conditions"], protocol["gates"]
    if (
        protocol["arms"] != ["original", *MODES]
        or protocol["checkpoint_sha256"] != CHECKPOINT_SHA256
    ):
        raise ValueError("preregistration arm or checkpoint identity differs")
    original = args.checkpoint.read_bytes()
    if hashlib.sha256(original).hexdigest() != CHECKPOINT_SHA256:
        raise ValueError("only the exact released V2 checkpoint may be loaded")
    args.workdir.mkdir(parents=True, exist_ok=False)
    source_paths = [Path(__file__), HERE / "audit_mace_ef_cos_input_semantics.py"]
    source_paths += [
        ROOT / "maple/function/calculator/extra_correction/implicit" / name
        for name in (
            "mace_polar_ef.py",
            "mace_polar_ef_specs.py",
            "electrostatic_pairing.py",
        )
    ]
    sources = _source_hashes(source_paths)
    import torch

    variants, manifest = {}, []
    original_state = state_digest(
        torch.jit.load(io.BytesIO(original), map_location="cpu")
    )
    for mode in MODES:
        payload, receipt = build_variant(original, mode)
        model = torch.jit.load(io.BytesIO(payload), map_location="cpu")
        receipt["state_dict_sha256"] = state_digest(model)
        if receipt["state_dict_sha256"] != original_state:
            raise RuntimeError("loaded weights/buffers differ")
        receipt["loaded_main_field_norm_occurrences"] = model.model.code.count(
            "linalg_vector_norm(external_field"
        )
        if receipt["loaded_main_field_norm_occurrences"] != (
            2 if mode == "roundtrip" else 0
        ):
            raise RuntimeError(
                "loaded program does not reflect the declared transformation"
            )
        path = args.workdir / f"{mode}.pt"
        with path.open("xb") as handle:
            handle.write(payload)
        variants[mode] = path
        manifest.append(receipt)
        del model, payload
    write_json_once(args.workdir / "archive-manifest.json", manifest)
    variant_hashes = {
        receipt["mode"]: receipt["checkpoint_sha256"] for receipt in manifest
    }
    molecules = conditions["geometries"] + conditions["gauge_only_geometries"]
    controls = []
    for molecule in molecules:
        builder = _builder(args.checkpoint, molecule)
        baseline = TraceProbe(builder.model, builder, molecule["positions_angstrom"])
        module = torch.jit.load(
            io.BytesIO(
                verified_bytes(variants["roundtrip"], variant_hashes["roundtrip"])
            ),
            map_location=builder.device,
        ).eval()
        control = TraceProbe(module, builder, molecule["positions_angstrom"])
        for label, field in anchors(baseline, conditions).items():
            left, right = baseline.sample(field), control.sample(field)
            delta = sample_differences(left, right)
            if (
                delta["energy_abs_ev"] > gates["roundtrip_energy_abs_tolerance_ev"]
                or delta["source_max_abs"] > gates["roundtrip_source_abs_tolerance"]
                or delta["density_max_abs"] > gates["roundtrip_density_abs_tolerance"]
            ):
                raise RuntimeError(
                    "no-op repack failed output/source parity; modified evaluation blocked"
                )
            controls.append(
                {"molecule": molecule["name"], "label": label, "differences": delta}
            )
        del baseline, control, module, builder
    write_json_once(args.workdir / "roundtrip-controls.json", controls)
    records = []
    for molecule in molecules:
        builder = _builder(args.checkpoint, molecule)
        for mode in ["original", *MODES]:
            module = (
                builder.model
                if mode == "original"
                else torch.jit.load(
                    io.BytesIO(verified_bytes(variants[mode], variant_hashes[mode])),
                    map_location=builder.device,
                ).eval()
            )
            probe = TraceProbe(module, builder, molecule["positions_angstrom"])
            samples = {
                label: probe.sample(field)
                for label, field in anchors(probe, conditions).items()
            }
            row = {
                "molecule": molecule["name"],
                "geometry": molecule,
                "mode": mode,
                "samples": samples,
            }
            if molecule in conditions["geometries"]:
                row["zero_center_scan"] = response_scan(
                    probe.affine_gradient,
                    np.zeros(3),
                    conditions["steps_ev_per_e_angstrom"],
                )
                row["nonzero_center_scan"] = response_scan(
                    probe.affine_gradient,
                    conditions["nonzero_center_ev_per_e_angstrom"],
                    conditions["steps_ev_per_e_angstrom"],
                )
            if state_digest(module) != original_state:
                raise RuntimeError("evaluation mutated a weight or buffer")
            row["gate_assessment"] = assess_record(row, conditions, gates)
            records.append(row)
            del probe, module
        del builder
    if (
        sources != _source_hashes(source_paths)
        or args.preregistration.read_bytes() != protocol_bytes
    ):
        raise RuntimeError("execution source/protocol drift")
    write_json_once(
        args.output,
        {
            "schema_version": 1,
            "artifact": "mace-ef-v2-trace-input-ablation-v1",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "preregistration_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
            "source_files_sha256": sources,
            "released_checkpoint_sha256": CHECKPOINT_SHA256,
            "state_dict_sha256": original_state,
            "variants": manifest,
            "roundtrip_controls": controls,
            "runtime": {
                "python": sys.executable,
                "torch": str(torch.__version__),
                "cuda": torch.version.cuda,
                "device": torch.cuda.get_device_name(0),
            },
            "records": records,
            "claim_boundary": {
                "same_serialized_tensors": True,
                "original_eager_restored": False,
                "production_adapter_modified": False,
                "conductor_cosmo_tested": False,
                "chemical_accuracy_tested": False,
                "release_admitted": False,
                "interpretation": "explicitly modified traced programs, not original-export parity or production admission",
            },
        },
    )


if __name__ == "__main__":
    main()
