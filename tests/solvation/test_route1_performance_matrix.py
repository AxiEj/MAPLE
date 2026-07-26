from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = REPOSITORY_ROOT / "docs/implicit-solvation/benchmarks"
PROTOCOL_PATH = BENCHMARK_DIR / "route1_performance_matrix_protocol.json"
FORMAL_EVIDENCE_DIR = BENCHMARK_DIR / "route1-performance-matrix-evidence-2026-07-26"
FORMAL_MATRIX_PATH = FORMAL_EVIDENCE_DIR / "route1-performance-matrix-2026-07-26.json"
BENCHMARK_README_PATH = BENCHMARK_DIR / "README.md"
PRODUCT_SPEC_PATH = REPOSITORY_ROOT / "docs/implicit-solvation/ROUTE1_PRODUCT_SPEC.md"
SPEC = importlib.util.spec_from_file_location(
    "run_route1_performance_matrix",
    BENCHMARK_DIR / "run_route1_performance_matrix.py",
)
assert SPEC is not None and SPEC.loader is not None
performance_matrix = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(performance_matrix)

AMBER_FIXTURE = (
    REPOSITORY_ROOT
    / "tests/solvation/data/amber_gb_reference"
    / "audit/methyl-hexanoate"
)
FAKE_TOPOLOGY_PATHS = {
    "molecule.mol2": AMBER_FIXTURE / "normalized.mol2",
    "molecule.frcmod": PROTOCOL_PATH,
    "molecule.prmtop": AMBER_FIXTURE / "obc2/system.prmtop",
    "molecule.inpcrd": AMBER_FIXTURE / "obc2/system.inpcrd",
}
EXECUTABLE_FIXTURE = REPOSITORY_ROOT / "maple/function/engine.py"


def _sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _timing(samples):
    return performance_matrix._timing_summary_with_samples([2.0] * samples)


def _host_load(stage):
    return {
        "stage": stage,
        "logical_cpu_count": 20,
        "load_average_1m": 1.0,
        "load_average_5m": 1.0,
        "load_average_15m": 1.0,
        "load_per_logical_cpu": 0.05,
        "maximum_load_per_logical_cpu": 0.25,
        "passed": True,
    }


def _native_pool(stage):
    return {
        "stage": stage,
        "maximum_threads": 1,
        "pools": [
            {
                "user_api": "blas",
                "internal_api": "openblas",
                "num_threads": 1,
                "prefix": "libopenblas",
                "filepath": "/opt/libopenblas.so",
                "version": "test",
                "threading_layer": "pthreads",
                "architecture": "test",
            }
        ],
        "passed": True,
    }


def _file_record(path):
    return {
        "path": performance_matrix._portable_artifact_path(path),
        "sha256": performance_matrix.sha256_file(path),
        "semantic_sha256": performance_matrix._topology_semantic_sha256(path),
    }


def test_repository_artifact_paths_are_portable_and_resolvable(tmp_path):
    recorded = performance_matrix._portable_artifact_path(PROTOCOL_PATH)

    assert recorded == (
        "docs/implicit-solvation/benchmarks/route1_performance_matrix_protocol.json"
    )
    assert performance_matrix._resolve_artifact_path(recorded) == PROTOCOL_PATH

    external = tmp_path / "external.prmtop"
    external.write_text("external", encoding="utf-8")
    external_record = performance_matrix._portable_artifact_path(external)
    assert Path(external_record).is_absolute()
    assert performance_matrix._resolve_artifact_path(external_record) == external


def _route1_provenance(policy):
    return {
        "method": "gb",
        "model": "obc2",
        "amber_igb": 5,
        "profile": "obc2-mbondi2",
        "provider": "openmm",
        "provider_version": "test",
        "radii": "mbondi2",
        "nonpolar": "ace",
        "platform": "CPU",
        "platform_properties": copy.deepcopy(policy["openmm_platform_properties"]),
        "solvent": "water",
        "solute_dielectric": 1.0,
        "solvent_dielectric": 78.5,
        "energy_force_evaluations_per_call": 1,
        "radius_provider": {
            "name": "openmm-amber-gb-radii",
            "provider": "openmm",
            "profile": "obc2-mbondi2",
            "radii": "mbondi2",
        },
        "nonpolar_provider": {
            "name": "openmm-ace",
            "provider": "openmm",
            "profile": "ace",
            "component_properties": ["energy", "forces"],
        },
    }


def _fake_cell(protocol, protocol_hash, case, model):
    policy = protocol["resource_policy"]
    warm_samples = protocol["timing"]["warm"]["samples"]
    cold_samples = protocol["timing"]["construction_cold"]["samples"]
    timings = {
        "warm": {
            name: _timing(warm_samples) for name in performance_matrix.WARM_TIMING_KEYS
        },
        "construction_cold": {
            name: _timing(cold_samples)
            for name in performance_matrix.CONSTRUCTION_COLD_TIMING_KEYS
        },
    }
    paired_comparisons = performance_matrix._derive_paired_comparisons(timings)
    ratios = performance_matrix._derive_ratios(timings, paired_comparisons)
    expected_conformance = {
        "device": "cpu",
        "torch_threads": 1,
        "torch_interop_threads": 1,
        "openmm_platform": "CPU",
        "warm_samples": warm_samples,
        "warmups": protocol["timing"]["warm"]["warmups"],
        "cold_samples": cold_samples,
    }
    thread_environment = copy.deepcopy(policy["required_environment"])
    native_preflight = _native_pool("timing_preflight")
    eligibility = performance_matrix.named_mm_comparison_eligibility(
        device="cpu",
        torch_threads=1,
        torch_interop_threads=1,
        native_threadpools=native_preflight["pools"],
        maximum_native_threadpool_threads=1,
        required_environment=thread_environment,
        observed_environment=thread_environment,
        route1_platform="CPU",
        route1_properties=policy["openmm_platform_properties"],
        mm_platform="CPU",
        mm_properties=policy["openmm_platform_properties"],
    )
    command_arguments = {
        "command": "run-cell",
        "case_id": case["case_id"],
        "model": model,
        "protocol": str(PROTOCOL_PATH),
        "device": None,
        "torch_threads": None,
        "openmm_platform": None,
        "warm_samples": None,
        "warmups": None,
        "cold_samples": None,
    }
    topology_files = {
        name: _file_record(path) for name, path in FAKE_TOPOLOGY_PATHS.items()
    }
    executable_record = {
        "path": performance_matrix._portable_artifact_path(EXECUTABLE_FIXTURE),
        "sha256": performance_matrix.sha256_file(EXECUTABLE_FIXTURE),
    }
    input_root = (
        REPOSITORY_ROOT / protocol["inputs"]["input_root_relative_path"]
    ).resolve()
    mol2_path = (input_root / case["mol2_relative_path"]).resolve()
    charge_manifest = (
        REPOSITORY_ROOT / protocol["inputs"]["charge_manifest_relative_path"]
    ).resolve()
    cell = {
        "schema_version": 1,
        "artifact_type": "route1-performance-matrix-cell",
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_hash,
        "command_provenance": {
            "script": (
                "docs/implicit-solvation/benchmarks/run_route1_performance_matrix.py"
            ),
            "script_sha256": performance_matrix.sha256_file(
                performance_matrix.__file__
            ),
            "python_executable": "/usr/bin/python",
            "arguments": command_arguments,
            "arguments_sha256": performance_matrix.sha256_bytes(
                performance_matrix.canonical_json_bytes(command_arguments)
            ),
            "environment_variables": {
                **thread_environment,
                "CUDA_VISIBLE_DEVICES": None,
            },
        },
        "claim_scope": protocol["claim_scope"],
        "case": copy.deepcopy(case),
        "model": model,
        "checkpoint": performance_matrix._checkpoint_record(model),
        "protocol_conformance": {
            "conformant": True,
            "reasons": [],
            "expected": expected_conformance,
            "observed": copy.deepcopy(expected_conformance),
        },
        "route1": {
            **protocol["route1"],
            "platform": "CPU",
            "platform_properties": copy.deepcopy(policy["openmm_platform_properties"]),
            "provenance": _route1_provenance(policy),
        },
        "named_mm_comparator": {
            **protocol["mm_comparator"],
            "platform": "CPU",
            "platform_properties": copy.deepcopy(policy["openmm_platform_properties"]),
            "implicit_model_validation": {
                "implicit_solvent": "OBC2",
                "sasa_method": "ACE",
                "force_class": "GBSAOBCForce",
                "force_count": 1,
                "surface_area_energy_kj_mol_nm2": 2.25936,
            },
            "topology": {
                "files": topology_files,
                "executables": {
                    "parmchk2": copy.deepcopy(executable_record),
                    "tleap": copy.deepcopy(executable_record),
                },
                "validation": {
                    "atom_count": case["atom_count"],
                    "atom_names_preserved": True,
                    "coordinate_max_abs_angstrom": 0.0,
                    "charge_max_abs_e": 0.0,
                    "frcmod_nonbon_overrides": 0,
                },
            },
        },
        "comparison_eligibility": eligibility,
        "composition_checks": {
            "energy_closure_hartree": 0.0,
            "structured_energy_closure_hartree": 0.0,
            "force_closure_hartree_per_angstrom_max": 0.0,
            "independent_gas_energy_difference_hartree": 0.0,
            "independent_gas_force_difference_hartree_per_angstrom_max": 0.0,
        },
        "timings": timings,
        "paired_comparisons": paired_comparisons,
        "ratios": ratios,
        "environment": {
            "host_platform": "Linux-test",
            "host_fingerprint": {
                "node": "test-host",
                "machine": "x86_64",
                "processor": "test",
                "cpu_model": "test CPU",
                "logical_cpu_count": 20,
                "cpu_affinity": list(range(20)),
            },
            "python": "3.11",
            "torch": "test",
            "torch_num_threads": 1,
            "torch_num_interop_threads": 1,
            "device": "cpu",
            "openmm": "test",
            "numpy": "test",
            "available_openmm_platforms": ["Reference", "CPU"],
            "required_thread_environment": thread_environment,
            "timing_isolation": {
                "evidence_scope": "entry_and_timing_endpoint_snapshots_only",
                "continuous_host_isolation_monitored": False,
                "whole_run_host_isolation_proven": False,
                "entry": _host_load("entry"),
                "timing_preflight": _host_load("timing_preflight"),
                "postflight": _host_load("postflight"),
                "endpoint_snapshots_passed": True,
            },
            "native_threadpools": {
                "evidence_scope": "timing_endpoint_snapshots_only",
                "timing_preflight": native_preflight,
                "postflight": _native_pool("postflight"),
                "endpoint_snapshots_passed": True,
            },
        },
        "inputs": {
            "mol2": performance_matrix._portable_artifact_path(mol2_path),
            "mol2_sha256": performance_matrix.sha256_file(mol2_path),
            "charge_manifest": performance_matrix._portable_artifact_path(
                charge_manifest
            ),
            "charge_manifest_sha256": performance_matrix.sha256_file(charge_manifest),
        },
        "limitations": copy.deepcopy(protocol["limitations"]),
    }
    return performance_matrix.seal_artifact(cell)


def test_protocol_freezes_exact_three_by_three_cpu_matrix():
    protocol, protocol_hash = performance_matrix.load_and_validate_protocol(
        PROTOCOL_PATH
    )

    assert len(protocol_hash) == 64
    assert protocol["matrix"]["models"] == ["maceoff23m", "aimnet2", "ani2x"]
    assert [case["size_bin"] for case in protocol["matrix"]["cases"]] == [
        "small",
        "medium",
        "large",
    ]
    assert protocol["resource_policy"]["device"] == "cpu"
    assert protocol["resource_policy"]["openmm_platform"] == "CPU"
    assert protocol["resource_policy"]["torch_interop_threads"] == 1
    assert protocol["timing"]["construction_cold"]["samples"] == 6
    assert protocol["mm_comparator"]["implicit_solvent"] == "OBC2"
    assert protocol["mm_comparator"]["sasa_method"] == "ACE"
    assert (
        protocol["mm_comparator"]["expected_surface_area_energy_kj_mol_nm2"] == 2.25936
    )
    assert protocol["acceptance"]["required_cell_count"] == 9
    assert "not a universal faster-than-MM" in protocol["claim_scope"]
    assert protocol["matrix"]["cases"][-1]["case_id"] == "mobley_2078467"
    assert protocol["case_selection"]["rejected_large_case"].startswith(
        "mobley_8124669"
    )
    for case in protocol["matrix"]["cases"]:
        mol2 = (
            REPOSITORY_ROOT / protocol["inputs"]["input_root_relative_path"]
        ) / case["mol2_relative_path"]
        assert performance_matrix.sha256_file(mol2) == case["mol2_sha256"]


def test_protocol_rejects_duplicate_size_bin(tmp_path):
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    protocol["matrix"]["cases"][1]["size_bin"] = "small"
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(protocol), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate molecule-size bin"):
        performance_matrix.load_and_validate_protocol(path)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda protocol: protocol["matrix"]["cases"][0].__setitem__("atom_count", 10.0),
        lambda protocol: protocol["timing"]["warm"].__setitem__("samples", 30.0),
        lambda protocol: protocol["resource_policy"].__setitem__("torch_threads", 1.0),
        lambda protocol: protocol["acceptance"].__setitem__("required_cell_count", 9.0),
    ],
)
def test_protocol_rejects_fractional_or_float_integer_fields(tmp_path, mutation):
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    mutation(protocol)
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(protocol), encoding="utf-8")

    with pytest.raises(TypeError, match="JSON integer"):
        performance_matrix.load_and_validate_protocol(path)


def test_named_mm_eligibility_requires_identical_single_thread_cpu_policy():
    environment = {
        "MKL_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
    }
    pools = [{"prefix": "openblas", "num_threads": 1}]
    eligible = performance_matrix.named_mm_comparison_eligibility(
        device="cpu",
        torch_threads=1,
        torch_interop_threads=1,
        native_threadpools=pools,
        maximum_native_threadpool_threads=1,
        required_environment=environment,
        observed_environment=environment,
        route1_platform="CPU",
        route1_properties={"DeterministicForces": "true", "Threads": "1"},
        mm_platform="CPU",
        mm_properties={"DeterministicForces": "true", "Threads": "1"},
    )
    assert eligible["eligible"] is True
    assert eligible["reasons"] == []

    ineligible = performance_matrix.named_mm_comparison_eligibility(
        device="cuda",
        torch_threads=1,
        torch_interop_threads=2,
        native_threadpools=[{"prefix": "openblas", "num_threads": 8}],
        maximum_native_threadpool_threads=1,
        required_environment=environment,
        observed_environment=environment,
        route1_platform="CPU",
        route1_properties={"DeterministicForces": "true", "Threads": "1"},
        mm_platform="CPU",
        mm_properties={"Threads": "8"},
    )
    assert ineligible["eligible"] is False
    assert "gas MLIP device is not CPU" in ineligible["reasons"]
    assert "PyTorch inter-op pool is not single-threaded" in ineligible["reasons"]
    assert "Route 1 and MM OpenMM platform properties differ" in ineligible["reasons"]


def test_named_mm_system_explicitly_contains_expected_obc2_ace_term():
    from openmm.app import AmberPrmtopFile

    prmtop = AmberPrmtopFile(
        REPOSITORY_ROOT
        / "tests/solvation/data/amber_gb_reference"
        / "audit/methyl-hexanoate/obc2/system.prmtop"
    )
    context, integrator, properties, system = performance_matrix._create_mm_context(
        prmtop,
        platform_name="Reference",
        properties={},
        sasa_method="ACE",
    )
    validation = performance_matrix._validate_obc2_ace_system(
        system,
        expected_force_class="GBSAOBCForce",
        expected_surface_area_energy_kj_mol_nm2=2.25936,
    )

    assert properties == {}
    assert validation == {
        "implicit_solvent": "OBC2",
        "sasa_method": "ACE",
        "force_class": "GBSAOBCForce",
        "force_count": 1,
        "surface_area_energy_kj_mol_nm2": 2.25936,
    }
    del context, integrator, system


def test_construction_cold_helpers_count_calls_and_interleave():
    calls = []

    first, second = performance_matrix.benchmark_paired_construction_cold_calls(
        lambda index: calls.append(("first", index)),
        lambda index: calls.append(("second", index)),
        samples=4,
        synchronize=lambda: None,
    )
    standalone = performance_matrix.benchmark_construction_cold_calls(
        lambda index: calls.append(("standalone", index)),
        samples=3,
        synchronize=lambda: None,
    )

    assert first["n"] == second["n"] == 4
    assert standalone["n"] == 3
    assert calls[:4] == [
        ("first", 0),
        ("second", 0),
        ("second", 1),
        ("first", 1),
    ]


def test_assembly_requires_exact_self_hashed_nine_cell_cross_product():
    protocol, protocol_hash = performance_matrix.load_and_validate_protocol(
        PROTOCOL_PATH
    )
    cells = [
        _fake_cell(protocol, protocol_hash, case, model)
        for case in protocol["matrix"]["cases"]
        for model in protocol["matrix"]["models"]
    ]

    result = performance_matrix.assemble_matrix(protocol, protocol_hash, cells)

    assert result["artifact_type"] == "route1-fair-sp-performance-matrix"
    assert result["summary"]["cell_count"] == 9
    assert result["summary"]["universal_faster_than_mm_claim_admitted"] is False
    assert (
        result["summary"]["universal_negligible_solvent_overhead_claim_admitted"]
        is False
    )
    assert result["content_sha256"] == performance_matrix.artifact_content_sha256(
        result
    )

    with pytest.raises(ValueError, match="coverage mismatch"):
        performance_matrix.assemble_matrix(protocol, protocol_hash, cells[:-1])

    tampered = copy.deepcopy(cells)
    tampered[0]["timings"]["warm"]["gas_mlip_for_overhead"]["median_ms"] = 999.0
    with pytest.raises(ValueError, match="content hash is invalid"):
        performance_matrix.assemble_matrix(protocol, protocol_hash, tampered)


def _load_formal_cells(protocol):
    return [
        json.loads(
            (
                FORMAL_EVIDENCE_DIR / "cells" / f"{case['case_id']}__{model}.json"
            ).read_text(encoding="utf-8")
        )
        for case in protocol["matrix"]["cases"]
        for model in protocol["matrix"]["models"]
    ]


def test_frozen_formal_matrix_reassembles_offline_from_committed_evidence():
    protocol, protocol_hash = performance_matrix.load_and_validate_protocol(
        PROTOCOL_PATH
    )
    cells = _load_formal_cells(protocol)
    frozen = json.loads(FORMAL_MATRIX_PATH.read_text(encoding="utf-8"))

    assert (
        performance_matrix.assemble_matrix(
            protocol,
            protocol_hash,
            cells,
            verify_external_dependencies=False,
        )
        == frozen
    )
    assert frozen["content_sha256"] == (
        "3076d3da83a05b73835c2978259e68bc0204ba8aa88d9d9c47c4c7428d1f457d"
    )
    assert frozen["summary"] == {
        "cell_count": 9,
        "warm_route1_faster_than_named_mm_count": 0,
        "construction_cold_route1_faster_than_named_mm_count": 0,
        "all_cells_protocol_conformant": True,
        "all_named_mm_comparisons_eligible": True,
        "all_cells_same_host_software_resource_policy": True,
        "all_case_mm_topologies_model_invariant": True,
        "universal_faster_than_mm_claim_admitted": False,
        "universal_negligible_solvent_overhead_claim_admitted": False,
    }


def test_frozen_formal_matrix_live_reassembly_when_dependencies_exist():
    protocol, protocol_hash = performance_matrix.load_and_validate_protocol(
        PROTOCOL_PATH
    )
    cells = _load_formal_cells(protocol)
    frozen = json.loads(FORMAL_MATRIX_PATH.read_text(encoding="utf-8"))
    external_paths = {Path(cell["checkpoint"]["path"]) for cell in cells} | {
        Path(record["path"])
        for cell in cells
        for record in cell["named_mm_comparator"]["topology"]["executables"].values()
    }
    if not all(path.is_file() for path in external_paths):
        pytest.skip("Frozen external checkpoints/AmberTools executables unavailable.")

    assert performance_matrix.assemble_matrix(protocol, protocol_hash, cells) == frozen


def test_documentation_binds_formal_matrix_values_and_no_go_boundary():
    benchmark_readme = BENCHMARK_README_PATH.read_text(encoding="utf-8")
    product_spec = PRODUCT_SPEC_PATH.read_text(encoding="utf-8")

    for document in (benchmark_readme, product_spec):
        assert "-0.90%-12.91%" in document
        assert "14.34x-708.23x" in document
        assert "32.26x-387.92x" in document
        assert "0/9" in document
        assert "route1-performance-matrix-2026-07-26.json" in document
    assert "3076d3da83a05b73835c2978259e68bc" in benchmark_readme
    assert "MLIMC" in product_spec
    assert "Distilled multiple-time-step" in product_spec


def test_assembly_resolves_repository_relative_evidence_from_any_cwd(
    tmp_path, monkeypatch
):
    protocol, protocol_hash = performance_matrix.load_and_validate_protocol(
        PROTOCOL_PATH
    )
    cells = [
        _fake_cell(protocol, protocol_hash, case, model)
        for case in protocol["matrix"]["cases"]
        for model in protocol["matrix"]["models"]
    ]
    assert not Path(
        cells[0]["named_mm_comparator"]["topology"]["executables"]["parmchk2"]["path"]
    ).is_absolute()
    assert not Path(cells[0]["inputs"]["mol2"]).is_absolute()

    monkeypatch.chdir(tmp_path)
    result = performance_matrix.assemble_matrix(protocol, protocol_hash, cells)

    assert result["summary"]["cell_count"] == 9


def test_assembly_rejects_nonconformant_or_ineligible_cell():
    protocol, protocol_hash = performance_matrix.load_and_validate_protocol(
        PROTOCOL_PATH
    )
    cells = [
        _fake_cell(protocol, protocol_hash, case, model)
        for case in protocol["matrix"]["cases"]
        for model in protocol["matrix"]["models"]
    ]
    cells[0]["protocol_conformance"]["conformant"] = False
    cells[0] = performance_matrix.seal_artifact(
        {key: value for key, value in cells[0].items() if key != "content_sha256"}
    )
    with pytest.raises(ValueError, match="Nonconformant"):
        performance_matrix.assemble_matrix(protocol, protocol_hash, cells)

    cells[0]["protocol_conformance"]["conformant"] = True
    cells[0]["comparison_eligibility"]["eligible"] = False
    cells[0] = performance_matrix.seal_artifact(
        {key: value for key, value in cells[0].items() if key != "content_sha256"}
    )
    with pytest.raises(ValueError, match="Ineligible"):
        performance_matrix.assemble_matrix(protocol, protocol_hash, cells)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda cell: cell["composition_checks"].__setitem__(
                "energy_closure_hartree", 999.0
            ),
            "energy composition check",
        ),
        (
            lambda cell: cell["case"].__setitem__("atom_count", 999),
            "case/model metadata",
        ),
        (
            lambda cell: cell.pop("command_provenance"),
            "current frozen runner",
        ),
        (
            lambda cell: cell["timings"]["warm"]["gas_mlip_for_overhead"].pop(
                "samples_ms"
            ),
            "raw samples",
        ),
        (
            lambda cell: cell["environment"]["timing_isolation"]["entry"].__setitem__(
                "load_average_1m", 1000.0
            ),
            "host-load entry evidence",
        ),
        (
            lambda cell: cell["environment"]["timing_isolation"]["entry"].__setitem__(
                "logical_cpu_count", 10
            ),
            "host-load entry evidence",
        ),
        (
            lambda cell: cell["named_mm_comparator"]["topology"][
                "validation"
            ].__setitem__("charge_max_abs_e", -1.0),
            "topology validation failed",
        ),
        (
            lambda cell: cell["route1"].__setitem__("provenance", {}),
            "provider provenance",
        ),
        (
            lambda cell: cell["named_mm_comparator"]["topology"]["files"][
                "molecule.mol2"
            ].__setitem__("path", "/tmp/does-not-exist-route1.mol2"),
            "unavailable or does not match",
        ),
        (
            lambda cell: cell["inputs"].__setitem__(
                "mol2", "/tmp/does-not-exist-route1-input.mol2"
            ),
            "source mol2 is unavailable or does not match",
        ),
        (
            lambda cell: cell["inputs"].__setitem__(
                "charge_manifest", "/tmp/does-not-exist-route1-charges.json"
            ),
            "source charge_manifest is unavailable or does not match",
        ),
    ],
)
def test_assembly_rejects_resealed_semantic_tampering(mutate, message):
    protocol, protocol_hash = performance_matrix.load_and_validate_protocol(
        PROTOCOL_PATH
    )
    cells = [
        _fake_cell(protocol, protocol_hash, case, model)
        for case in protocol["matrix"]["cases"]
        for model in protocol["matrix"]["models"]
    ]
    mutate(cells[0])
    cells[0] = performance_matrix.seal_artifact(
        {key: value for key, value in cells[0].items() if key != "content_sha256"}
    )

    with pytest.raises(ValueError, match=message):
        performance_matrix.assemble_matrix(protocol, protocol_hash, cells)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda cell: cell["checkpoint"].__setitem__(
            "size_bytes", float(cell["checkpoint"]["size_bytes"])
        ),
        lambda cell: cell["environment"].__setitem__("torch_num_threads", 1.9),
        lambda cell: cell["environment"]["host_fingerprint"].__setitem__(
            "logical_cpu_count", 20.9
        ),
        lambda cell: cell["environment"]["native_threadpools"][
            "timing_preflight"
        ].__setitem__("maximum_threads", 1.9),
        lambda cell: cell["environment"]["native_threadpools"]["timing_preflight"][
            "pools"
        ][0].__setitem__("num_threads", 1.9),
        lambda cell: cell["named_mm_comparator"]["topology"]["validation"].__setitem__(
            "atom_count", 10.9
        ),
    ],
)
def test_assembly_rejects_resealed_fractional_integer_evidence(mutate):
    protocol, protocol_hash = performance_matrix.load_and_validate_protocol(
        PROTOCOL_PATH
    )
    cells = [
        _fake_cell(protocol, protocol_hash, case, model)
        for case in protocol["matrix"]["cases"]
        for model in protocol["matrix"]["models"]
    ]
    mutate(cells[0])
    cells[0] = performance_matrix.seal_artifact(
        {key: value for key, value in cells[0].items() if key != "content_sha256"}
    )

    with pytest.raises(TypeError, match="JSON integer"):
        performance_matrix.assemble_matrix(protocol, protocol_hash, cells)


def test_assembly_rejects_cross_cell_host_or_topology_drift():
    protocol, protocol_hash = performance_matrix.load_and_validate_protocol(
        PROTOCOL_PATH
    )
    cells = [
        _fake_cell(protocol, protocol_hash, case, model)
        for case in protocol["matrix"]["cases"]
        for model in protocol["matrix"]["models"]
    ]
    cells[0]["environment"]["host_fingerprint"]["node"] = "different-host"
    cells[0] = performance_matrix.seal_artifact(
        {key: value for key, value in cells[0].items() if key != "content_sha256"}
    )
    with pytest.raises(ValueError, match="one host/software/resource policy"):
        performance_matrix.assemble_matrix(protocol, protocol_hash, cells)

    cells = [
        _fake_cell(protocol, protocol_hash, case, model)
        for case in protocol["matrix"]["cases"]
        for model in protocol["matrix"]["models"]
    ]
    alternative_prmtop = (
        REPOSITORY_ROOT
        / "tests/solvation/data/amber_gb_reference"
        / "audit/benzene/obc2/system.prmtop"
    )
    cells[0]["named_mm_comparator"]["topology"]["files"]["molecule.prmtop"] = (
        _file_record(alternative_prmtop)
    )
    cells[0] = performance_matrix.seal_artifact(
        {key: value for key, value in cells[0].items() if key != "content_sha256"}
    )
    with pytest.raises(ValueError, match="model-dependent MM topology"):
        performance_matrix.assemble_matrix(protocol, protocol_hash, cells)


def test_assembly_rejects_common_invented_topology_hash_for_all_case_models():
    protocol, protocol_hash = performance_matrix.load_and_validate_protocol(
        PROTOCOL_PATH
    )
    cells = [
        _fake_cell(protocol, protocol_hash, case, model)
        for case in protocol["matrix"]["cases"]
        for model in protocol["matrix"]["models"]
    ]
    case_id = protocol["matrix"]["cases"][0]["case_id"]
    for index, cell in enumerate(cells):
        if cell["case"]["case_id"] != case_id:
            continue
        cell["named_mm_comparator"]["topology"]["files"]["molecule.prmtop"][
            "semantic_sha256"
        ] = _sha("common-invented-topology")
        cells[index] = performance_matrix.seal_artifact(
            {key: value for key, value in cell.items() if key != "content_sha256"}
        )

    with pytest.raises(ValueError, match="unavailable or does not match"):
        performance_matrix.assemble_matrix(protocol, protocol_hash, cells)
