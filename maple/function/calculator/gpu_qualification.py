"""Supplementary bounded-FP32 evidence, embedded in the schema-2 parity artifact.

Card ``gpu_acceleration.numerical_qualification`` declares ``protocol_id``,
``implementation_source_sha256s`` (repository-relative .py paths),
``required_contracts`` and ``performance_contracts``. Artifact
``numerical_qualification`` repeats protocol/source identities and supplies
``comparisons`` and ``performance``. A comparison repeats its exact card
contract in ``contract`` and supplies ``paired_outputs``, ``comparison_count``,
``observed_maximum`` and ``passed``. Output hashes use the existing parity hash
encoding with observable=comparison ID. Cases are {record_id: configuration_sha256};
contracts also bind explicit reference/candidate {device, backend, dtype, mode},
kind, unit, comparison, threshold, expected_count and configuration (e.g. FD step).

``required_numerical_comparisons`` exposes canonical comparison identities and
units for evidence producers. Protocol v2 explicitly identifies CPU64 independent
FD; it also retains the failed same-FP32 FD diagnostic without using that
non-gating diagnostic to admit a mode. Supporting a protocol does not approve or
publish a scientific card. No model class names or literature metrics occur here.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from statistics import median
from typing import Mapping

PERFORMANCE_CONTROLS = {
    "warmups": 3,
    "repetitions": 10,
    "cuda_synchronized": True,
    "fresh_inference": True,
    "unit": "s",
    "minimum_headroom_bytes": 512 * 1024**2,
    "reference": {
        "device": "cuda",
        "backend": "torch",
        "dtype": "float32",
        "mode": "reference",
    },
    "candidate": {
        "device": "cuda",
        "backend": "torch",
        "dtype": "float32",
        "mode": "optimized",
    },
}
ARCHETYPES = {"scalar_correction", "native_energy_forces", "scalar_composition"}


def _identity(device, dtype, mode="reference", backend="torch"):
    return dict(device=device, backend=backend, dtype=dtype, mode=mode)


def required_numerical_comparisons(protocol_id: str, modes) -> dict:
    """Return mandatory ID -> immutable kind/unit/execution identity fields.

    IDs end in ``_cpu32`` or ``_cuda32_<mode>``. Producer adds comparison,
    threshold, expected_count, cases and configuration before freezing the card.
    ``native_energy_forces/v2`` explicitly uses CPU64 independent FD and retains
    a same-FP32 diagnostic; v1 uses CUDA32 FD. No gate is relaxed by v2.
    """
    archetype, separator, version = protocol_id.partition("/")
    if separator != "/" or archetype not in ARCHETYPES or version not in {"v1", "v2"}:
        raise ValueError("Unsupported numerical qualification protocol_id/archetype.")
    if version == "v2" and archetype != "native_energy_forces":
        raise ValueError("Protocol v2 is defined only for native_energy_forces.")
    modes = tuple(modes)
    if not modes or set(modes) - {"reference", "optimized"}:
        raise ValueError("Numerical qualification requires reference/optimized modes.")
    result = {}

    def add(name, kind, unit, reference, candidate, gating=True):
        result[name] = dict(
            kind=kind,
            unit=unit,
            reference=reference,
            candidate=candidate,
            gating=gating,
        )

    targets = [("cpu32", _identity("cpu", "float32"))] + [
        (f"cuda32_{mode}", _identity("cuda", "float32", mode)) for mode in modes
    ]
    for suffix, candidate in targets:
        add(
            f"energy_drift_{suffix}",
            "energy_drift",
            (
                "eV/atom"
                if archetype in {"native_energy_forces", "scalar_composition"}
                else "eV"
            ),
            _identity("cpu", "float64"),
            candidate,
        )
        if archetype in {"scalar_correction", "scalar_composition"}:
            add(
                f"vacuum_zero_{suffix}",
                "vacuum_zero",
                "eV",
                _identity("cpu", "float64"),
                candidate,
            )
        if archetype == "scalar_composition":
            add(
                f"correction_drift_{suffix}",
                "correction_drift",
                "eV",
                _identity("cpu", "float64"),
                candidate,
            )
        if archetype == "native_energy_forces":
            for kind, unit in (
                ("solution_minus_vacuum_drift", "eV"),
                ("force_drift", "eV/A"),
            ):
                add(
                    f"{kind}_{suffix}",
                    kind,
                    unit,
                    _identity("cpu", "float64"),
                    candidate,
                )
    # Explicit CPU64 vacuum anchor, independent of the FP32 drift panels.
    if archetype in {"scalar_correction", "scalar_composition"}:
        add(
            "vacuum_zero_cpu64",
            "vacuum_zero",
            "eV",
            _identity("cpu", "float64"),
            _identity("cpu", "float64"),
        )
    if archetype == "native_energy_forces" and version == "v2":
        for name, dtype in (
            ("reference_fd_cpu64", "float64"),
            ("independent_fd_cpu32", "float32"),
        ):
            add(
                name,
                "independent_fd",
                "eV/A",
                _identity("cpu", "float64", backend="central_difference"),
                _identity("cpu", dtype),
            )
    for mode in modes:
        candidate = _identity("cuda", "float32", mode)
        if archetype == "scalar_composition":
            add(
                f"composite_scalar_cuda32_{mode}",
                "composite_scalar",
                "eV/atom",
                _identity("cpu", "float32"),
                candidate,
            )
            add(
                f"correction_parity_cuda32_{mode}",
                "correction_parity",
                "eV",
                _identity("cpu", "float32"),
                candidate,
            )
        if archetype == "native_energy_forces":
            fd_reference = (
                _identity("cpu", "float64", backend="central_difference")
                if version == "v2"
                else _identity("cuda", "float32", mode, "central_difference")
            )
            add(
                f"independent_fd_cuda32_{mode}",
                "independent_fd",
                "eV/A",
                fd_reference,
                candidate,
            )
            if version == "v2":
                add(
                    f"same_precision_fd_diagnostic_cuda32_{mode}",
                    "same_precision_fd_diagnostic",
                    "eV/A",
                    _identity("cuda", "float32", mode, "central_difference"),
                    candidate,
                    False,
                )
            for kind, unit in (
                ("rigid_motion_energy", "eV"),
                ("rigid_motion_force", "eV/A"),
                ("sampled_coordinates", "A"),
            ):
                add(f"{kind}_cuda32_{mode}", kind, unit, candidate, candidate)
    return result


def _source_path(root, name):
    path = Path(name)
    if (
        path.is_absolute()
        or ".." in path.parts
        or path.suffix != ".py"
        or path.as_posix() != name
    ):
        raise ValueError(
            "Implementation source paths must be safe repository-relative .py paths."
        )
    current = root
    for part in path.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("Implementation source paths must not contain symlinks.")
    current.resolve().relative_to(root.resolve())
    return current


def _positive_count(value):
    return type(value) is int and value > 0


def _number(value):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError("Numerical qualification requires finite numeric values.")
    return float(value)


def validate_numerical_contract(raw, *, modes, forces, expected_archetype):
    """Validate a card contract before artifact admission; return canonical JSON.

    Card contracts freeze exact counts/cases/thresholds before measurement.
    Values may be tighter than protocol ceilings, never looser.
    """
    from .model_capabilities import _validated_evidence_sha256

    if not isinstance(raw, Mapping) or set(raw) != {
        "protocol_id",
        "implementation_source_sha256s",
        "required_contracts",
        "performance_contracts",
    }:
        raise ValueError(
            "Numerical qualification requires explicit protocol, implementation sources and required contracts/performance contracts."
        )
    protocol = raw["protocol_id"]
    if not isinstance(protocol, str):
        raise ValueError("Numerical qualification protocol_id must be a string.")
    archetype = protocol.split("/")[0]
    if archetype != expected_archetype or forces != (
        archetype == "native_energy_forces"
    ):
        raise ValueError(
            "Numerical qualification protocol/capability archetype mismatch."
        )
    specs = required_numerical_comparisons(protocol, modes)
    contracts = raw["required_contracts"]
    if not isinstance(contracts, Mapping) or set(contracts) != set(specs):
        raise ValueError(
            "Numerical qualification required contracts must exactly cover protocol comparisons."
        )
    sources = raw["implementation_source_sha256s"]
    if not isinstance(sources, Mapping) or not sources:
        raise ValueError(
            "Numerical qualification implementation source hashes must be nonempty."
        )
    for name, digest in sources.items():
        if not isinstance(name, str):
            raise ValueError("Implementation source paths must be strings.")
        _source_path(Path("."), name)
        _validated_evidence_sha256(digest, name)
    for name, spec in specs.items():
        contract = contracts[name]
        keys = set(spec) | {
            "comparison",
            "threshold",
            "expected_count",
            "cases",
            "configuration",
        }
        if (
            not isinstance(contract, Mapping)
            or set(contract) != keys
            or any(contract[k] != v for k, v in spec.items())
        ):
            raise ValueError(
                f"Numerical comparison {name} has invalid kind/unit/dtype/mode contract."
            )
        kind = spec["kind"]
        comparison = contract["comparison"]
        ceiling = {
            "energy_drift": (
                2e-4
                if archetype in {"native_energy_forces", "scalar_composition"}
                else 1e-4
            ),
            "composite_scalar": 2e-4,
            "independent_fd": 2e-3,
            "same_precision_fd_diagnostic": 2e-3,
            "vacuum_zero": 0.0,
        }.get(kind, 1e-4)
        threshold = _number(contract["threshold"])
        if not 0 <= threshold <= ceiling or comparison not in {
            "absolute",
            "equal",
            "non_degradation",
        }:
            raise ValueError(
                f"Numerical comparison {name} has invalid/loosened threshold."
            )
        if kind == "vacuum_zero" and (comparison != "equal" or threshold != 0):
            raise ValueError("Vacuum zero requires exact equality and zero threshold.")
        if kind != "vacuum_zero" and comparison != "absolute":
            raise ValueError("Numerical drift requires absolute comparison.")
        if not _positive_count(contract["expected_count"]):
            raise ValueError(
                "Numerical comparison expected_count must be a positive integer."
            )
        cases = contract["cases"]
        if not isinstance(cases, Mapping) or not cases:
            raise ValueError("Numerical comparison cases must be nonempty.")
        for case, digest in cases.items():
            if not isinstance(case, str) or not case.strip():
                raise ValueError(
                    "Numerical comparison case IDs must be nonempty strings."
                )
            _validated_evidence_sha256(digest, case)
        config = contract["configuration"]
        if not isinstance(config, Mapping) or not config:
            raise ValueError(
                "Numerical comparison configuration must be explicit/nonempty."
            )
        if "fd" in kind and config.get("step_A") != 0.002:
            raise ValueError("FD configuration must freeze step_A=0.002.")
    if protocol == "native_energy_forces/v2":
        for mode in modes:
            independent = contracts[f"independent_fd_cuda32_{mode}"]
            diagnostic = contracts[f"same_precision_fd_diagnostic_cuda32_{mode}"]
            for field in ("expected_count", "cases", "configuration", "threshold"):
                if independent[field] != diagnostic[field]:
                    raise ValueError(
                        "Independent FD and same-FP32 diagnostic must retain identical cases/counts/step/threshold."
                    )
                for reference in ("reference_fd_cpu64", "independent_fd_cpu32"):
                    if independent[field] != contracts[reference][field]:
                        raise ValueError(
                            "CPU and CUDA high-precision-reference FD audits must retain identical cases/counts/step/threshold."
                        )
    performance = raw["performance_contracts"]
    expected = {"optimized"} if "optimized" in modes else set()
    if not isinstance(performance, Mapping) or set(performance) != expected:
        raise ValueError(
            "Optimized modes require exact performance contracts; reference-only has no optimized claim."
        )
    for contract in performance.values():
        if (
            not isinstance(contract, Mapping)
            or set(contract) != set(PERFORMANCE_CONTROLS) | {"cases"}
            or any(
                type(contract[k]) is not type(v) or contract[k] != v
                for k, v in PERFORMANCE_CONTROLS.items()
            )
        ):
            raise ValueError(
                "Performance controls require warmup3/repetitions10, synchronized fresh inference and 512MiB headroom."
            )
        cases = contract["cases"]
        if not isinstance(cases, Mapping) or not cases:
            raise ValueError("Performance requires frozen input cases.")
        for digest in cases.values():
            _validated_evidence_sha256(digest, "performance case")
    return json.dumps(raw, sort_keys=True, allow_nan=False)


def _validate_record(name, record, contract):
    from .model_capabilities import _validate_gpu_paired_outputs

    if not isinstance(record, Mapping) or record.get("contract") != contract:
        raise ValueError(f"Numerical comparison {name} contract differs from card.")
    pairs = record.get("paired_outputs")
    if not isinstance(pairs, list) or len(pairs) != len(contract["cases"]):
        raise ValueError(f"Numerical comparison {name} has missing/extra cases.")
    hashes = {}
    for pair in pairs:
        if not isinstance(pair, Mapping):
            raise ValueError("Numerical paired output must be a mapping.")
        case = pair.get("record_id")
        if (
            not isinstance(case, str)
            or case not in contract["cases"]
            or pair.get("configuration_sha256") != contract["cases"][case]
        ):
            raise ValueError(
                f"Numerical comparison {name} case/configuration identity mismatch."
            )
        hashes[case] = {
            role: pair.get(f"{role}_sha256") for role in ("reference", "accelerator")
        }
    count, maximum = _validate_gpu_paired_outputs(
        record,
        observable=name,
        comparison=contract["comparison"],
        manifest_output_sha256s=hashes,
    )
    if (
        count != contract["expected_count"]
        or type(record.get("comparison_count")) is not int
        or record["comparison_count"] != count
    ):
        raise ValueError(
            f"Numerical comparison {name} count differs from frozen contract."
        )
    if _number(record.get("observed_maximum")) != maximum:
        raise ValueError(
            f"Numerical comparison {name} aggregate does not match raw outputs."
        )
    passed = maximum <= contract["threshold"]
    if record.get("passed") is not passed or (contract["gating"] and not passed):
        raise ValueError(
            f"Numerical comparison {name} failed frozen threshold/verdict."
        )
    if contract["kind"] == "vacuum_zero":
        from .model_capabilities import _finite_numeric_tensor

        for pair in pairs:
            for role in ("reference", "accelerator"):
                _, values = _finite_numeric_tensor(pair[f"{role}_values"], label=name)
                if any(value != 0 for value in values):
                    raise ValueError(
                        "Vacuum evidence must contain actual exact zero outputs."
                    )


def validate_numerical_qualification(contract_json: str, raw, *, evidence_root: Path):
    """Replay raw comparisons/source identities/performance in SHA-bound evidence."""
    from .model_capabilities import _finite_numeric_tensor, _validate_gpu_paired_outputs

    card = json.loads(contract_json)
    if not isinstance(raw, Mapping) or set(raw) != {
        "protocol_id",
        "implementation_source_sha256s",
        "comparisons",
        "performance",
    }:
        raise ValueError("Missing/incomplete numerical_qualification matrix.")
    for field in ("protocol_id", "implementation_source_sha256s"):
        if raw[field] != card[field]:
            raise ValueError(f"Numerical qualification {field} differs from card.")
    for name, digest in card["implementation_source_sha256s"].items():
        try:
            actual = hashlib.sha256(
                _source_path(evidence_root, name).read_bytes()
            ).hexdigest()
        except OSError as exc:
            raise ValueError(f"Implementation source is unavailable: {name}") from exc
        if actual != digest:
            raise ValueError(f"Implementation source SHA256 mismatch: {name}")
    records = raw["comparisons"]
    if not isinstance(records, Mapping) or set(records) != set(
        card["required_contracts"]
    ):
        raise ValueError("Numerical comparisons must exactly match required contracts.")
    for name, contract in card["required_contracts"].items():
        for key in (
            "implementation_addendum",
            "preserved_failure_index",
            "composite_production_performance",
        ):
            reference = contract["configuration"].get(key)
            if reference is not None:
                if not isinstance(reference, Mapping) or set(reference) != {
                    "artifact",
                    "sha256",
                }:
                    raise ValueError("Invalid methodology artifact reference.")
                relative = Path(reference["artifact"])
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError("Unsafe methodology artifact path.")
                path = evidence_root / relative
                path.resolve().relative_to(evidence_root.resolve())
                try:
                    actual = hashlib.sha256(path.read_bytes()).hexdigest()
                except OSError as exc:
                    raise ValueError("Methodology artifact is unavailable.") from exc
                if actual != reference["sha256"]:
                    raise ValueError("Methodology artifact SHA256 mismatch.")
        _validate_record(name, records[name], contract)
    performance = raw["performance"]
    if not isinstance(performance, Mapping) or set(performance) != set(
        card["performance_contracts"]
    ):
        raise ValueError("Missing/extra optimized performance evidence.")
    for mode, contract in card["performance_contracts"].items():
        record = performance[mode]
        if not isinstance(record, Mapping) or record.get("contract") != contract:
            raise ValueError("Optimized performance controls differ from card.")
        pairs = record.get("paired_outputs")
        if not isinstance(pairs, list) or len(pairs) != len(contract["cases"]):
            raise ValueError(
                "Optimized speed requires raw fresh eager/candidate samples."
            )
        hashes = {}
        for pair in pairs:
            if not isinstance(pair, Mapping):
                raise ValueError("Performance paired output must be a mapping.")
            case = pair.get("record_id")
            if (
                not isinstance(case, str)
                or case not in contract["cases"]
                or pair.get("configuration_sha256") != contract["cases"][case]
            ):
                raise ValueError("Performance case/configuration mismatch.")
            hashes[case] = {
                role: pair.get(f"{role}_sha256")
                for role in ("reference", "accelerator")
            }
            samples = []
            for role in ("reference", "accelerator"):
                shape, values = _finite_numeric_tensor(
                    pair.get(f"{role}_values"), label="latency"
                )
                if shape != (10,) or any(value <= 0 for value in values):
                    raise ValueError(
                        "Performance requires ten positive CUDA-synchronized fresh samples."
                    )
                samples.append(median(values))
            if (
                pair.get("eager_median_s") != samples[0]
                or pair.get("candidate_median_s") != samples[1]
                or not samples[1] < samples[0]
            ):
                raise ValueError("Optimized speed failed recomputed median gate.")
            total, peak = pair.get("device_total_bytes"), pair.get(
                "peak_reserved_bytes"
            )
            if (
                type(total) is not int
                or type(peak) is not int
                or not 0 < peak <= total
                or total - peak < contract["minimum_headroom_bytes"]
            ):
                raise ValueError("Optimized memory requires measured 512MiB headroom.")
            free = pair.get("free_bytes_at_peak")
            if type(free) is not int or not 0 <= free <= total:
                raise ValueError(
                    "Optimized memory requires measured device free memory at peak."
                )
            headroom = min(free, total - peak)
            if (
                type(pair.get("measured_headroom_bytes")) is not int
                or pair["measured_headroom_bytes"] != headroom
                or headroom < contract["minimum_headroom_bytes"]
            ):
                raise ValueError(
                    "Optimized measured memory headroom failed recomputed gate."
                )
        _validate_gpu_paired_outputs(
            record,
            observable=f"performance_{mode}",
            comparison="absolute",
            manifest_output_sha256s=hashes,
        )
