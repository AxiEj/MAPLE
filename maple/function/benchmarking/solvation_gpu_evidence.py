"""Build hash-bound GPU cards from completed, label-free diagnostic receipts.

No model inference occurs here. Every numerical/performance gate is replayed by
MAPLE's admission validator before any model card is replaced.
"""

from __future__ import annotations
import argparse, copy, hashlib, json, math, shutil
from pathlib import Path
from statistics import median
import numpy as np
from maple.function.calculator.gpu_qualification import (
    required_numerical_comparisons,
    PERFORMANCE_CONTROLS,
)
from maple.function.calculator.model_capabilities import (
    ModelProvenanceCard,
    _gpu_output_sha256,
    GPU_FORBIDDEN_PRECISION_CONTROLS,
)

ROOT = Path(__file__).resolve().parents[3]
CARD_ROOT = ROOT / "maple/function/calculator/model_cards"
DEST = CARD_ROOT / "evidence/gpu-fp32-2026-10-25"


def encoded(value):
    return (
        json.dumps(value, sort_keys=True, allow_nan=False, indent=2) + "\n"
    ).encode()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return hashlib.sha256(
        (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode()
    ).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def plain(value):
    return np.asarray(value, dtype=float).tolist()


def flat(value):
    return np.asarray(value, dtype=float).reshape(-1).tolist()


def pair(identifier, observable, reference, candidate, configuration=None):
    r, a = plain(reference), plain(candidate)
    return {
        "record_id": identifier,
        "reference_values": r,
        "accelerator_values": a,
        "reference_sha256": _gpu_output_sha256(
            record_id=identifier, observable=observable, role="reference", values=r
        ),
        "accelerator_sha256": _gpu_output_sha256(
            record_id=identifier, observable=observable, role="accelerator", values=a
        ),
        **({"configuration_sha256": configuration} if configuration else {}),
    }


def maximum(pairs, comparison="absolute"):
    differences = []
    for p in pairs:
        r, a = np.asarray(p["reference_values"]), np.asarray(p["accelerator_values"])
        differences.extend(
            flat(np.abs(a - r) if comparison in {"absolute", "equal"} else a - r)
        )
    return max(differences)


def count(pairs):
    return sum(np.asarray(p["reference_values"]).size for p in pairs)


def by_case(rows):
    return {r["case_id"]: r for r in rows}


def fd_vectors(report, panel):
    require(
        len(report["finite_difference_records"])
        == len(panel["finite_difference_cases"]),
        "FD denominator mismatch",
    )
    values = {
        c["case_id"]: np.zeros((len(c["numbers"]), 3))
        for c in panel["scalar_cases"]
        if c["route"] == "consolv"
    }
    for case, row in zip(
        panel["finite_difference_cases"], report["finite_difference_records"]
    ):
        require(
            row["case_id"] == case["case_id"] and row["status"] == "success",
            "FD case/status mismatch",
        )
        values[case["base_case_id"]][case["atom_index"], case["component_index"]] = row[
            "finite_difference_force_ev_per_angstrom"
        ]
    return values


def sample_positions(report):
    return {
        r["case_id"]: [s["next_positions_angstrom"] for s in r["steps"]]
        for r in report["sampled_force_records"]
    }


def require(condition, message):
    if not condition:
        raise ValueError(message)


def expected_models(route):
    """Read trusted pins from implementation constants and current cards."""
    from maple.function.calculator.extra_correction.implicit.anysolv import (
        ANYSOLV_SOURCE_REVISION,
        ANYSOLV_SOURCE_ARTIFACTS,
    )
    from maple.function.calculator.consolv._consolv_calculator import (
        CONSOLV_SOURCE_REVISION,
    )

    if route == "anysolv":
        correction = read(CARD_ROOT / "anysolv-moe.yaml")
        composite = read(CARD_ROOT / "aimnet2-anysolv.yaml")
        return dict(
            source_revision=ANYSOLV_SOURCE_REVISION,
            checkpoints={
                "correction": correction["checkpoint_sha256"],
                "gas": composite["checkpoints"]["gas"]["checkpoint_sha256"],
            },
            external_sources=ANYSOLV_SOURCE_ARTIFACTS,
        )
    card = read(CARD_ROOT / "consolv.yaml")
    return dict(
        source_revision=CONSOLV_SOURCE_REVISION,
        checkpoints={
            key: card["checkpoints"][name]["checkpoint_sha256"]
            for key, name in (
                ("solution", "consolv-torch-v1"),
                ("vacuum", "consolv-vacuum-torch-v1"),
            )
        },
    )


def validate_sources(report, required_sources):
    before, after = report["source_sha256_before"], report["source_sha256_after"]
    require(
        bool(before) and before == after and report["source_drift"] is False,
        "Source drift or missing source identities",
    )
    require(set(required_sources) <= set(after), "Missing critical source identity")
    for name, sha in after.items():
        require(Path(name).is_absolute(), "Raw source paths must be absolute")
        require(digest(name) == sha, f"Current source mismatch: {name}")
    for name, sha in required_sources.items():
        require(after[name] == sha, f"Pinned source mismatch: {name}")


def validate_precision(report, device, dtype, mode):
    execution = report["execution"]
    for key, expected in dict(
        device=device,
        compute_dtype=dtype,
        inference_mode=mode,
        matmul_precision="highest",
        tf32=False,
        autocast=False,
    ).items():
        require(execution.get(key) == expected, f"Execution {key} mismatch")
    require(report["inference_mode"] == mode, "Top-level inference_mode mismatch")
    precision = report["effective_precision"]
    for key, expected in dict(
        compute_dtype=dtype,
        matmul_precision="highest",
        matmul_tf32=False,
        cudnn_tf32=False,
        autocast=False,
    ).items():
        require(precision.get(key) == expected, f"Effective precision {key} mismatch")
    require(
        precision.get("compiler_options")
        == (
            {
                "use_fast_math": False,
                "cuda.use_fast_math": False,
                "cutlass.use_fast_math": False,
                "force_same_precision": True,
            }
            if mode == "optimized"
            else {}
        ),
        "Unexpected compiler precision options",
    )
    if device.startswith("cuda"):
        runtime = report["runtime"]
        require(runtime["accelerator_backend"] == "cuda", "Runtime backend mismatch")
        require(
            runtime["precision_state"] == precision, "Runtime precision state mismatch"
        )
        controls = runtime["precision_controls"]
        require(
            controls["compute_dtype"] == dtype
            and controls["matmul_precision"] == "highest",
            "Runtime dtype/precision mismatch",
        )
        require(
            all(controls.get(k) is False for k in GPU_FORBIDDEN_PRECISION_CONTROLS),
            "Runtime reduced precision controls",
        )
    else:
        require(report["runtime"] is None, "CPU reference has accelerator runtime")


def exact_rows(rows, cases, label):
    ids = [r["case_id"] for r in rows]
    expected = [c["case_id"] for c in cases]
    require(
        ids == expected and len(set(ids)) == len(ids),
        f"{label}: case identity/order mismatch",
    )
    for row, case in zip(rows, cases):
        require(row.get("status") == "success", f"{label}: failed status")
        for key in ("geometry_sha256", "atom_count"):
            if key in row:
                value = len(case["numbers"]) if key == "atom_count" else case[key]
                require(row[key] == value, f"{label}: configuration mismatch")


def validate_report(report, route, panel, expectation, device, dtype, mode):
    require(
        report["route"] == route and report["panel_sha256"] == canonical(panel),
        "Route/panel identity mismatch",
    )
    require(
        report.get("status") != "failed" and not report.get("initialization_error"),
        "Unusable diagnostic receipt",
    )
    require(
        report["experimental_labels_opened"] is False
        and report["confirmation_partition_opened"] is False,
        "Opened scientific labels",
    )
    validate_precision(report, device, dtype, mode)
    validate_sources(report, expectation["required_sources"])
    cases = [c for c in panel["scalar_cases"] if c["route"] == route]
    groups = {
        "scalar": ("scalar_records", cases),
        "vacuum": (
            "vacuum_records",
            [c for c in panel["vacuum_cases"] if c["route"] == route],
        ),
        "finite_difference_components": (
            "finite_difference_records",
            panel["finite_difference_cases"] if route == "consolv" else [],
        ),
        "motion_comparisons": (
            "motion_records",
            panel["motion_cases"] if route == "consolv" else [],
        ),
        "sampled_force_replays": (
            "sampled_force_records",
            (
                [dict(case_id=i) for i in panel["sampled_force_probe"]["case_ids"]]
                if route == "consolv"
                else []
            ),
        ),
    }
    expected_counts = {key: len(c) for key, (_, c) in groups.items()}
    require(
        report["expected_counts"] == expected_counts
        and report["counts"] == {**expected_counts, "failed_records": 0},
        "Incomplete frozen denominator",
    )
    for key, (field, expected) in groups.items():
        exact_rows(report[field], expected, field)
    for row in report["finite_difference_records"]:
        for side in ("plus", "minus"):
            require(
                row[side]["status"] == "success"
                and row[side]["case_id"] == row["case_id"] + ":" + side,
                "Failed/mismatched FD configuration",
            )
    for row in report["motion_records"]:
        require(
            row["evaluation"]["status"] == "success"
            and row["evaluation"]["case_id"] == row["case_id"],
            "Failed motion evaluation",
        )
    for row in report["sampled_force_records"]:
        exact_rows(
            row["steps"],
            [
                dict(case_id=f"{row['case_id']}:step:{i}")
                for i in range(panel["sampled_force_probe"]["steps"])
            ],
            "sampled steps",
        )
    adapter = report["adapter"]
    require(
        adapter["source_revision"] == expectation["source_revision"],
        "Source revision mismatch",
    )
    if route == "anysolv":
        identity = adapter["source_identity"]
        require(
            identity["source_revision"] == expectation["source_revision"],
            "Source identity mismatch",
        )
        external = {
            str(Path(identity["source_root"]) / name): sha
            for name, sha in expectation["external_sources"].items()
        }
        validate_sources(report, {**expectation["required_sources"], **external})
        require(
            adapter["checkpoint"]["sha256"] == expectation["checkpoints"]["correction"],
            "Correction checkpoint mismatch",
        )
        gas = report["gas_base"]
        require(
            gas["checkpoint"]["sha256"] == expectation["checkpoints"]["gas"],
            "Gas checkpoint mismatch",
        )
        require(gas["gas_reference_compute_dtype"] == dtype, "Gas dtype mismatch")
        unique = list({c["geometry_sha256"]: c for c in reversed(cases)}.values())[::-1]
        exact_rows(gas["gas_records"], unique, "gas")
        exact_rows(gas["combined_records"], cases, "combined")
        gas_map = {r["geometry_sha256"]: r["gas_ev"] for r in gas["gas_records"]}
        correction_map = by_case(report["scalar_records"])
        for row in gas["combined_records"]:
            require(
                row["gas_ev"] == gas_map[row["geometry_sha256"]]
                and row["correction_ev"] == correction_map[row["case_id"]]["energy_ev"]
                and row["combined_ev"] == row["gas_ev"] + row["correction_ev"],
                "Combined scalar identity mismatch",
            )
    else:
        require(
            adapter["source_state_unchanged"] is True
            and adapter["state_sha256_before"] == adapter["state_sha256_after"],
            "ConSolv state mutation",
        )
        require(
            {k: v["sha256"] for k, v in adapter["checkpoints"].items()}
            == expectation["checkpoints"],
            "ConSolv checkpoint mismatch",
        )
    timing = report["timing"]
    require(
        timing["status"] == "success"
        and timing["case_id"] in {c["case_id"] for c in cases},
        "Timing identity/status mismatch",
    )
    require(
        timing["warmup_count"] == panel["timing"]["warmup"], "Timing warmup mismatch"
    )
    samples = timing["fresh_inference"]["samples_seconds"]
    require(
        len(samples) == panel["timing"]["repetitions"]
        and all(
            type(v) in (int, float) and math.isfinite(v) and v > 0 for v in samples
        ),
        "Timing denominator/values mismatch",
    )


def load_reports(folder, route, panel, expectation):
    keys = {
        "cpu64": f"{route}.cpu.float64.reference",
        "cpu32": f"{route}.cpu.float32.reference",
        **{m: f"{route}.cuda.float32.{m}" for m in ("reference", "optimized")},
    }
    result = {k: read(folder / f"{name}.json") for k, name in keys.items()}
    for key, report in result.items():
        device, dtype, mode = (
            ("cpu", "float64" if key == "cpu64" else "float32", "reference")
            if key.startswith("cpu")
            else ("cuda:0", "float32", key)
        )
        validate_report(report, route, panel, expectation, device, dtype, mode)
    require(
        result["reference"]["runtime"]["hardware"]
        == result["optimized"]["runtime"]["hardware"]
        and result["reference"]["runtime"]["software_versions"]
        == result["optimized"]["runtime"]["software_versions"],
        "Inconsistent runtime identity",
    )
    return result, keys


def critical_sources(route):
    names = [
        "maple/function/benchmarking/pretrained_solvation_gpu.py",
        "maple/function/benchmarking/solvation_gpu.py",
        "maple/function/calculator/solvation_execution.py",
    ]
    names += (
        [
            "maple/function/calculator/extra_correction/implicit/anysolv.py",
            "maple/function/calculator/aimnet/_aimnet2_calculator.py",
        ]
        if route == "anysolv"
        else [
            f"maple/function/calculator/consolv/{name}.py"
            for name in ("_torch_model", "_compiled", "_consolv_calculator")
        ]
    )
    return {str(ROOT / name): digest(ROOT / name) for name in names}


def load_composite_performance(path, panel, expectation):
    """Required JSON schema (all identities are mandatory):

    schema_version=1, model_id='aimnet2-anysolv', route='anysolv',
    scope='public_production_calculate_gas_plus_correction', panel_sha256,
    checkpoints={gas: SHA, correction: SHA}, source_revision,
    source_identity={source_revision, source_root}, source_sha256_before/after,
    source_drift=false. source maps include ALL critical AnySolv receipt sources
    plus calculator_base.py and set_calculator.py. reference/optimized each have:
      execution={device:'cuda:0',compute_dtype:'float32',inference_mode:MODE,
                 matmul_precision:'highest',tf32:false,autocast:false},
      inference_mode=MODE, effective_precision, runtime (same as raw receipts),
      timing={status:'success',case_id, warmup_count:3,cuda_synchronized:true,
              fresh_inference:{samples_seconds:[10 positive finite seconds],
                               geometry_sha256s:[10 fresh geometry SHAs]}},
      memory={device_total_bytes,peak_reserved_bytes,free_bytes_at_peak,
              measured_headroom_bytes}, resident_models=['gas','correction'].
    Memory covers both resident models throughout the synchronized sequence.
    """
    data = read(path)
    require(
        data["schema_version"] == 1
        and data["model_id"] == "aimnet2-anysolv"
        and data["route"] == "anysolv"
        and data["scope"] == "public_production_calculate_gas_plus_correction",
        "Composite production scope mismatch",
    )
    require(
        data["panel_sha256"] == canonical(panel)
        and data["checkpoints"] == expectation["checkpoints"],
        "Composite panel/checkpoint mismatch",
    )
    sources = {
        **expectation["required_sources"],
        **{
            str(ROOT / f"maple/function/calculator/{name}.py"): digest(
                ROOT / f"maple/function/calculator/{name}.py"
            )
            for name in ("calculator_base", "set_calculator")
        },
    }
    require(
        data["source_revision"] == expectation["source_revision"]
        and data["source_identity"]["source_revision"]
        == expectation["source_revision"],
        "Composite source revision mismatch",
    )
    sources.update(
        {
            str(Path(data["source_identity"]["source_root"]) / name): sha
            for name, sha in expectation["external_sources"].items()
        }
    )
    validate_sources(data, sources)
    cases = {c["case_id"] for c in panel["scalar_cases"] if c["route"] == "anysolv"}
    for mode in ("reference", "optimized"):
        report = data[mode]
        validate_precision(report, "cuda:0", "float32", mode)
        require(
            report["resident_models"] == ["gas", "correction"],
            "Composite model footprint missing",
        )
        timing = report["timing"]
        require(
            timing["status"] == "success"
            and timing["case_id"] in cases
            and timing["warmup_count"] == 3
            and timing["cuda_synchronized"] is True,
            "Composite timing controls mismatch",
        )
        samples = timing["fresh_inference"]["samples_seconds"]
        hashes = timing["fresh_inference"]["geometry_sha256s"]
        require(
            len(samples) == 10
            and len(hashes) == 10
            and len(set(hashes)) == 10
            and all(isinstance(h, str) and len(h) == 64 for h in hashes)
            and all(
                type(v) in (int, float) and math.isfinite(v) and v > 0 for v in samples
            ),
            "Composite fresh timing denominator mismatch",
        )
        mem = report["memory"]
        require(
            set(mem)
            == {
                "device_total_bytes",
                "peak_reserved_bytes",
                "free_bytes_at_peak",
                "measured_headroom_bytes",
            }
            and all(type(v) is int and v >= 0 for v in mem.values())
            and mem["measured_headroom_bytes"]
            == min(
                mem["free_bytes_at_peak"],
                mem["device_total_bytes"] - mem["peak_reserved_bytes"],
            ),
            "Composite memory accounting mismatch",
        )
    r, a = data["reference"], data["optimized"]
    require(
        r["timing"]["case_id"] == a["timing"]["case_id"]
        and r["timing"]["fresh_inference"]["geometry_sha256s"]
        == a["timing"]["fresh_inference"]["geometry_sha256s"]
        and r["runtime"]["hardware"] == a["runtime"]["hardware"]
        and r["runtime"]["software_versions"] == a["runtime"]["software_versions"],
        "Composite reference/candidate controls mismatch",
    )
    return data


def package_methodology(force_protocol, protocol_sha):
    """Keep v3 immutable; bind the implementation addendum and full failure archive."""
    addendum = DEST / "implementation-addendum.json"
    require(
        read(addendum)["force_methodology_protocol_sha256"] == protocol_sha,
        "Implementation addendum force protocol mismatch",
    )
    index_path = force_protocol.parent / "preserved-failure-index.json"
    index = read(index_path)
    frozen_protocol = read(force_protocol)
    require(
        frozen_protocol["complete_failure_index"]["sha256"] == digest(index_path),
        "Force protocol failure index mismatch",
    )
    files = {
        "implementation-addendum.json": addendum,
        "force-audit-protocol-v3.json": force_protocol,
        "preserved-failure-index.json": index_path,
    }
    for name, sha in index["files"].items():
        require(Path(name).name == name, "Unsafe historical failure path")
        source = force_protocol.parent / name
        require(digest(source) == sha, f"Historical failure SHA mismatch: {name}")
        files[name] = source
    return files, {
        "implementation_addendum": {
            "artifact": str((DEST / "implementation-addendum.json").relative_to(ROOT)),
            "sha256": digest(addendum),
        },
        "implementation_description_precedence": "implementation addendum overrides prototype_dynamic implementation text only; v3 science, cases, h and tolerances remain immutable",
        "preserved_failure_index": {
            "artifact": str((DEST / "preserved-failure-index.json").relative_to(ROOT)),
            "sha256": digest(index_path),
        },
    }


def memory(report):
    snapshots = [report["memory"]]
    for owner in report["adapter"]["owners"]:
        snapshots += [
            owner[k]
            for k in (
                "memory_after_initialization",
                "memory_before_release",
                "memory_after_release",
            )
            if k in owner
        ]
        snapshots += [r["memory"] for r in owner["first_calls"]]
    snapshots = [r for r in snapshots if "total_bytes" in r]
    total = snapshots[0]["total_bytes"]
    peak = max(r["peak_reserved_bytes"] for r in snapshots)
    free = min(r["free_bytes"] for r in snapshots)
    return {
        "device_total_bytes": total,
        "peak_reserved_bytes": peak,
        "free_bytes_at_peak": free,
        "measured_headroom_bytes": min(free, total - peak),
    }


def supplemental(
    model_id,
    reports,
    panel,
    modes,
    protocol,
    source_map,
    force_protocol_sha,
    methodology,
    composite_performance=None,
):
    archetype = protocol.split("/")[0]
    if archetype == "scalar_composition" and "optimized" in modes:
        require(
            composite_performance is not None,
            "Composite performance receipt is required",
        )
    cases = [
        c
        for c in panel["scalar_cases"]
        if c["route"] == ("consolv" if model_id == "consolv" else "anysolv")
    ]
    case_map = {c["case_id"]: c for c in cases}
    source = {key: by_case(report["scalar_records"]) for key, report in reports.items()}
    vac = {
        key: {r["case_id"].split(":")[1]: r for r in report["vacuum_records"]}
        for key, report in reports.items()
    }
    combined = (
        {
            key: by_case(report["gas_base"]["combined_records"])
            for key, report in reports.items()
        }
        if archetype == "scalar_composition"
        else None
    )
    fd = (
        {key: fd_vectors(report, panel) for key, report in reports.items()}
        if archetype == "native_energy_forces"
        else None
    )
    motion = (
        {key: by_case(report["motion_records"]) for key, report in reports.items()}
        if fd
        else None
    )
    sampled = (
        {key: sample_positions(report) for key, report in reports.items()}
        if fd
        else None
    )
    contracts, comparisons = {}, {}

    def target(name, spec):
        candidate = spec["candidate"]
        return (
            "cpu64"
            if candidate["dtype"] == "float64"
            else ("cpu32" if candidate["device"] == "cpu" else candidate["mode"])
        )

    def add_values(name, spec):
        key = target(name, spec)
        kind = spec["kind"]
        pairs = []
        if kind in {"vacuum_zero"}:
            for molecule, row in vac[key].items():
                ident = f"vacuum:{molecule}"
                geom = panel["geometries"][molecule]
                pairs.append((ident, 0.0, row["energy_ev"], geom["geometry_sha256"]))
        elif kind in {
            "energy_drift",
            "solution_minus_vacuum_drift",
            "force_drift",
            "composite_scalar",
            "correction_drift",
            "correction_parity",
        }:
            ref = (
                "cpu32"
                if kind in {"composite_scalar", "correction_parity"}
                else "cpu64"
            )
            for case in cases:
                ident = case["case_id"]
                r, a = source[ref][ident], source[key][ident]
                if kind in {"composite_scalar"} or (
                    kind == "energy_drift" and combined is not None
                ):
                    rv, av = combined[ref][ident]["combined_ev"] / len(
                        case["numbers"]
                    ), combined[key][ident]["combined_ev"] / len(case["numbers"])
                elif kind == "force_drift":
                    rv, av = r["forces_ev_per_angstrom"], a["forces_ev_per_angstrom"]
                elif kind == "solution_minus_vacuum_drift":
                    molecule = case["molecule"]
                    rv = r["energy_ev"] - vac[ref][molecule]["energy_ev"]
                    av = a["energy_ev"] - vac[key][molecule]["energy_ev"]
                else:
                    divisor = (
                        len(case["numbers"])
                        if kind == "energy_drift" and model_id == "consolv"
                        else 1
                    )
                    rv, av = r["energy_ev"] / divisor, a["energy_ev"] / divisor
                pairs.append((ident, rv, av, case["geometry_sha256"]))
        elif kind in {"independent_fd", "same_precision_fd_diagnostic"}:
            ref = "cpu64" if kind == "independent_fd" else key
            for case in cases:
                ident = case["case_id"]
                pairs.append(
                    (
                        ident,
                        fd[ref][ident],
                        source[key][ident]["forces_ev_per_angstrom"],
                        case["geometry_sha256"],
                    )
                )
        elif kind in {"rigid_motion_energy", "rigid_motion_force"}:
            for case in panel["motion_cases"]:
                ident = case["case_id"]
                base = f"consolv:{case['molecule']}:{case['solvent']}"
                base_result = source[key][base]
                row = motion[key][ident]["evaluation"]
                if kind == "rigid_motion_energy":
                    rv, av = base_result["energy_ev"], row["energy_ev"]
                else:
                    rv = np.asarray(base_result["forces_ev_per_angstrom"])
                    av = row["forces_ev_per_angstrom"]
                    if case["transform"] == "rotation":
                        rv = rv @ np.asarray(panel["rotation_matrix"]).T
                    elif case["transform"] == "permutation":
                        rv = rv[::-1]
                pairs.append((ident, rv, av, case["geometry_sha256"]))
        elif kind == "sampled_coordinates":
            for ident, values in sampled["cpu32"].items():
                pairs.append(
                    (
                        ident,
                        values,
                        sampled[key][ident],
                        canonical(
                            {"case": ident, "protocol": panel["sampled_force_probe"]}
                        ),
                    )
                )
        else:
            raise ValueError(kind)
        return pairs

    specs = required_numerical_comparisons(protocol, modes)
    for name, spec in specs.items():
        rows = add_values(name, spec)
        config = {
            "panel_sha256": canonical(panel),
            "raw_receipt_content_sha256s": {
                key: canonical(report) for key, report in reports.items()
            },
            "force_audit_protocol_sha256": force_protocol_sha,
            **methodology,
        }
        if archetype != "scalar_composition":
            config.pop("composite_production_performance", None)
        if archetype == "scalar_composition":
            require(
                composite_performance is not None,
                "Composite performance receipt is required",
            )
            config["composite_production_performance_sha256"] = canonical(
                composite_performance
            )
        if "fd" in spec["kind"]:
            config["step_A"] = 0.002
        threshold = (
            0.0
            if spec["kind"] == "vacuum_zero"
            else (
                0.002
                if "fd" in spec["kind"]
                else (0.0002 if spec["unit"] == "eV/atom" else 0.0001)
            )
        )
        comparison = "equal" if spec["kind"] == "vacuum_zero" else "absolute"
        case_ids = {
            ident: canonical({"geometry": geometry, "configuration": config})
            for ident, _, _, geometry in rows
        }
        paired = [pair(ident, name, r, a, case_ids[ident]) for ident, r, a, _ in rows]
        contract = {
            **spec,
            "comparison": comparison,
            "threshold": threshold,
            "expected_count": count(paired),
            "cases": case_ids,
            "configuration": config,
        }
        observed = maximum(paired, comparison)
        contracts[name] = contract
        comparisons[name] = {
            "contract": copy.deepcopy(contract),
            "paired_outputs": paired,
            "comparison_count": count(paired),
            "observed_maximum": observed,
            "passed": observed <= threshold,
        }

    perf_contracts, performance = {}, {}
    if "optimized" in modes:
        performance_reports = (
            composite_performance if archetype == "scalar_composition" else reports
        )
        require(
            performance_reports is not None, "Composite performance receipt is required"
        )
        report = performance_reports["optimized"]
        timing = report["timing"]
        ident = timing["case_id"]
        configuration = canonical({"timing": panel["timing"], "case": ident})
        contract = {**PERFORMANCE_CONTROLS, "cases": {ident: configuration}}
        r = performance_reports["reference"]["timing"]["fresh_inference"][
            "samples_seconds"
        ]
        a = timing["fresh_inference"]["samples_seconds"]
        p = pair(ident, "performance_optimized", r, a, configuration)
        p.update(
            eager_median_s=median(r),
            candidate_median_s=median(a),
            **(
                report["memory"]
                if archetype == "scalar_composition"
                else memory(report)
            ),
        )
        perf_contracts["optimized"] = contract
        performance["optimized"] = {
            "contract": copy.deepcopy(contract),
            "paired_outputs": [p],
        }
    card = {
        "protocol_id": protocol,
        "implementation_source_sha256s": source_map,
        "required_contracts": contracts,
        "performance_contracts": perf_contracts,
    }
    artifact = {
        "protocol_id": protocol,
        "implementation_source_sha256s": source_map,
        "comparisons": comparisons,
        "performance": performance,
    }
    return card, artifact


def build_card(
    model_id,
    reports,
    panel,
    modes,
    force_protocol_sha,
    methodology,
    composite_performance=None,
):
    card_path = CARD_ROOT / f"{model_id}.yaml"
    payload = read(card_path)
    payload["quantity_kind"] = {
        "anysolv-moe": "geometry_level_energy_correction",
        "aimnet2-anysolv": "combined_solution_energy",
        "consolv": "native_solution_energy",
    }[model_id]
    # Parse provenance with qualification pending while constructing its replacement.
    provenance = copy.deepcopy(payload)
    provenance.pop("gpu_acceleration", None)
    base = ModelProvenanceCard.from_payload(provenance, model_name=model_id)
    is_composite = model_id == "aimnet2-anysolv"
    is_native = model_id == "consolv"
    protocol = (
        "native_energy_forces/v2"
        if is_native
        else ("scalar_composition/v1" if is_composite else "scalar_correction/v1")
    )
    source_map = {}
    for path, sha in reports["reference"]["source_sha256_after"].items():
        p = Path(path)
        try:
            rel = p.relative_to(ROOT)
        except ValueError:
            continue
        if p.suffix == ".py":
            if digest(p) != sha:
                raise ValueError(
                    f"Current source differs from measured implementation: {p}"
                )
            source_map[str(rel)] = sha
    for rel in [
        "maple/function/benchmarking/solvation_gpu_evidence.py",
        "maple/function/calculator/model_capabilities.py",
        "maple/function/calculator/gpu_qualification.py",
        "maple/function/calculator/calculator_base.py",
        "maple/function/calculator/set_calculator.py",
    ]:
        source_map[rel] = digest(ROOT / rel)
    numerical_card, numerical_artifact = supplemental(
        model_id,
        reports,
        panel,
        modes,
        protocol,
        source_map,
        force_protocol_sha,
        methodology,
        composite_performance,
    )
    capabilities = base.capabilities
    observables = capabilities.gpu_required_observables("bounded_numerical_fp32")
    contracts = {
        name: {
            "unit": (
                "count"
                if name == "coverage"
                else (
                    "eV/atom"
                    if name == "energy" and (is_native or is_composite)
                    else (
                        "eV"
                        if name == "energy"
                        else "eV/A" if name == "forces" else "A"
                    )
                )
            ),
            "comparison": "equal" if name == "coverage" else "absolute",
            "threshold": (
                0.0
                if name == "coverage"
                else (
                    0.0002
                    if name == "energy" and (is_native or is_composite)
                    else 0.0001
                )
            ),
        }
        for name in observables
    }
    values = {name: [] for name in observables}
    manifest_records = []
    for mode in modes:
        ident = f"{mode}:{model_id}:frozen-public-panel-v2"
        ref = reports["cpu32"]
        cand = reports[mode]

        def vector(report, name):
            if name == "coverage":
                return [
                    report["counts"][key]
                    for key in (
                        "scalar",
                        "vacuum",
                        "finite_difference_components",
                        "motion_comparisons",
                        "sampled_force_replays",
                    )
                ]
            if name == "energy":
                rows = (
                    report["gas_base"]["combined_records"]
                    if is_composite
                    else report["scalar_records"]
                )
                cases = [
                    c
                    for c in panel["scalar_cases"]
                    if c["route"] == ("consolv" if is_native else "anysolv")
                ]
                return [
                    (r["combined_ev"] if is_composite else r["energy_ev"])
                    / (len(case["numbers"]) if is_native or is_composite else 1)
                    for r, case in zip(rows, cases)
                ]
            if name == "forces":
                return [
                    v
                    for row in report["scalar_records"]
                    for v in flat(row["forces_ev_per_angstrom"])
                ]
            return [
                v
                for row in report["sampled_force_records"]
                for step in row["steps"]
                for v in flat(step["next_positions_angstrom"])
            ]

        hashes = {}
        for name in observables:
            p = pair(ident, name, vector(ref, name), vector(cand, name))
            values[name].append(p)
            hashes[name] = {
                "reference": p["reference_sha256"],
                "accelerator": p["accelerator_sha256"],
            }
        manifest_records.append(
            {
                "record_id": ident,
                "panel_id": "frozen-ase-six-v2",
                "configuration_sha256": canonical(
                    {"panel": canonical(panel), "mode": mode, "model": model_id}
                ),
                "reference_sha256": canonical(ref),
                "observable_output_sha256s": hashes,
            }
        )
    manifest_id = f"{model_id}-cuda-fp32-public-ase-v1"
    tasks = capabilities.gpu_exposed_tasks()
    task_modes = sorted(f"{task}:{mode}" for task in tasks for mode in modes)
    manifest = {
        "schema_version": 1,
        "comparison_manifest_id": manifest_id,
        "model_id": model_id,
        "model_version": base.version,
        "checkpoint_sha256s": list(base.checkpoint_sha256s),
        "task_modes": task_modes,
        "records": manifest_records,
        "observable_counts": {name: count(v) for name, v in values.items()},
    }
    runtime = reports["reference"]["runtime"]
    precision = runtime["precision_controls"]
    artifact = {
        "schema_version": 2,
        "verdict": "pass_bounded_numerical_fp32",
        "qualification_kind": "bounded_numerical_fp32",
        "model_id": model_id,
        "model_version": base.version,
        "checkpoint_sha256s": list(base.checkpoint_sha256s),
        "comparison_manifest_id": manifest_id,
        "comparison_manifest_sha256": hashlib.sha256(encoded(manifest)).hexdigest(),
        "allowed_tasks": list(tasks),
        "allowed_inference_modes": list(modes),
        "validated_task_modes": task_modes,
        "precision_controls": {
            "reference_dtype": "float32",
            "accelerator_dtype": "float32",
            "same_scalar_precision": True,
            **precision,
        },
        "runtime": {
            "reference_backend": "cpu",
            "accelerator_backend": "cuda",
            "hardware": runtime["hardware"],
            "software_versions": runtime["software_versions"],
        },
        "observables": {
            name: {
                **contracts[name],
                "comparison_count": count(pairs),
                "observed_maximum": maximum(pairs, contracts[name]["comparison"]),
                "passed": maximum(pairs, contracts[name]["comparison"])
                <= contracts[name]["threshold"],
                "paired_outputs": pairs,
            }
            for name, pairs in values.items()
        },
        "numerical_qualification": numerical_artifact,
        "scientific_scope": "bounded numerical execution only; no strict-no-loss, chemical accuracy, OPT/MD or thermodynamic admission",
        "force_reference": "cpu64_central_fd" if is_native else None,
        "same_fp32_absolute_energy_fd": (
            "failed_resolution_limited" if is_native else "not_applicable"
        ),
    }
    manifest_path = DEST / f"{model_id}-manifest.json"
    artifact_path = DEST / f"{model_id}-evidence.json"
    policy = {
        "qualification_kind": "bounded_numerical_fp32",
        "no_loss_parity_verified": False,
        "numerical_parity_verified": True,
        "status": "passed_frozen_bounded_cuda_fp32_numerical_protocol",
        "compute_dtype": "float32",
        "matmul_precision": "highest",
        "allowed_tasks": list(tasks),
        "allowed_inference_modes": list(modes),
        "allowed_hardware": [runtime["hardware"]],
        "required_runtime_packages": ["torch", "cuda", "e3nn", "triton"]
        + (["fairchem_core"] if not is_native else []),
        "observable_contracts": contracts,
        "comparison_manifest_id": manifest_id,
        "comparison_manifest_artifact": str(manifest_path.relative_to(ROOT)),
        "comparison_manifest_sha256": hashlib.sha256(encoded(manifest)).hexdigest(),
        "evidence_artifact": str(artifact_path.relative_to(ROOT)),
        "evidence_sha256": hashlib.sha256(encoded(artifact)).hexdigest(),
        "numerical_qualification": numerical_card,
    }
    payload["gpu_acceleration"] = policy
    if is_native:
        payload.update(
            production_compute_dtype="float32",
            energy_accumulation_dtype="float32",
            force_reference="cpu64_central_fd",
            same_fp32_absolute_energy_fd="failed_resolution_limited",
            optimized_profile="stateless_static_export_fullgraph_inductor",
        )
    payload["execution_defaults"] = {
        "device": "cuda:0",
        "dtype": "float32",
        "inference": "reference",
    }
    if not is_native:
        payload["forbidden_tasks"] = [
            x for x in payload.get("forbidden_tasks", []) if x != "gpu_execution"
        ]
    return card_path, payload, manifest_path, manifest, artifact_path, artifact


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--receipts", type=Path, required=True)
    p.add_argument("--panel", type=Path, required=True)
    p.add_argument("--force-protocol", type=Path, required=True)
    p.add_argument(
        "--composite-performance",
        type=Path,
        required=True,
        help="Independent public combined-production timing JSON; see load_composite_performance docstring",
    )
    p.add_argument("--publish", action="store_true")
    args = p.parse_args()
    panel = read(args.panel)
    protocol_sha = digest(args.force_protocol)
    require(
        read(args.force_protocol)["frozen_panel_sha256"] == canonical(panel),
        "Force protocol frozen panel mismatch",
    )
    modes = ("optimized", "reference")
    bundles = []
    methodology_files, methodology = package_methodology(
        args.force_protocol, protocol_sha
    )
    expectations = {
        r: {**expected_models(r), "required_sources": critical_sources(r)}
        for r in ("anysolv", "consolv")
    }
    composite_performance = load_composite_performance(
        args.composite_performance, panel, expectations["anysolv"]
    )
    composite_receipt_sha = digest(args.composite_performance)
    methodology_files["composite-production-timing.json"] = args.composite_performance
    methodology["composite_production_performance"] = {
        "artifact": str((DEST / "composite-production-timing.json").relative_to(ROOT)),
        "sha256": composite_receipt_sha,
    }
    reports_by_route = {}
    for route, ids in [
        ("anysolv", ["anysolv-moe", "aimnet2-anysolv"]),
        ("consolv", ["consolv"]),
    ]:
        reports, _ = load_reports(args.receipts, route, panel, expectations[route])
        reports_by_route[route] = reports
        for model in ids:
            bundles.append(
                build_card(
                    model,
                    reports,
                    panel,
                    modes,
                    protocol_sha,
                    methodology,
                    composite_performance,
                )
            )
    DEST.mkdir(parents=True, exist_ok=True)
    for name, source in methodology_files.items():
        target = DEST / name
        if source.resolve() != target.resolve():
            shutil.copyfile(source, target)
    # Data are untrusted until the same parser/admission code recomputes all gates.
    for card_path, payload, mp, manifest, ap, artifact in bundles:
        mp.write_bytes(encoded(manifest))
        ap.write_bytes(encoded(artifact))
        card = ModelProvenanceCard.from_payload(payload, model_name=payload["model_id"])
        route = "consolv" if card.model_id == "consolv" else "anysolv"
        reports = reports_by_route[route]
        for mode in modes:
            card.validate_device(
                "cuda:0",
                task="sp",
                inference_mode=mode,
                evidence_root=ROOT,
                runtime_context=reports[mode]["runtime"],
            )
        print("VALIDATED", card.model_id, "all bounded gates", flush=True)
    # Re-read all eight inputs and live sources after validation, before card writes.
    for route, original in reports_by_route.items():
        current, _ = load_reports(args.receipts, route, panel, expectations[route])
        require(current == original, "Receipt changed during publication")
    require(
        load_composite_performance(
            args.composite_performance, panel, expectations["anysolv"]
        )
        == composite_performance,
        "Composite receipt changed during publication",
    )
    require(
        digest(args.composite_performance) == composite_receipt_sha,
        "Composite input bytes changed during publication",
    )
    _, final_methodology = package_methodology(
        args.force_protocol, digest(args.force_protocol)
    )
    require(
        final_methodology
        == {
            k: v
            for k, v in methodology.items()
            if k != "composite_production_performance"
        },
        "Methodology changed during publication",
    )
    if args.publish:
        for card_path, payload, *_ in bundles:
            card_path.write_bytes(encoded(payload))
        shutil.copyfile(args.panel, DEST / "frozen-panel-v2.json")
        print("PUBLISHED cards after all three validations")
    else:
        print(
            "Dry card publication: validated evidence files written, model cards unchanged"
        )


if __name__ == "__main__":
    main()
