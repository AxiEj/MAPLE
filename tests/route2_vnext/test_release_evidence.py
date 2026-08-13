from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from maple.solvation.release.evidence import (
    RepositorySnapshot,
    canonical_json_sha256,
    collect_loaded_repository_sources,
    committed_source_hashes,
    write_external_json_artifact,
)

ROOT = Path(__file__).parents[2]
RUNNER = ROOT / "tools" / "route2_release" / "run_fixedbox590_water_pes_diagnostic.py"
COMMON = ROOT / "tools" / "route2_release" / "fixedbox590_water_common.py"
PATH_RUNNER = (
    ROOT / "tools" / "route2_release" / "run_fixedbox590_water_path_diagnostic.py"
)
BOX_RUNNER = (
    ROOT / "tools" / "route2_release" / "run_fixedbox590_water_box_convergence.py"
)
PES_PANEL_RUNNER = ROOT / "tools" / "route2_release" / "run_fixedbox590_pes_panel.py"
PES_PANEL_AGGREGATOR = (
    ROOT / "tools" / "route2_release" / "aggregate_fixedbox590_pes_panel.py"
)
PES_CARTESIAN_RUNNER = (
    ROOT / "tools" / "route2_release" / "run_fixedbox590_cartesian_panel.py"
)
PES_CARTESIAN_AGGREGATOR = (
    ROOT / "tools" / "route2_release" / "aggregate_fixedbox590_cartesian_panel.py"
)
RESIDUAL_FORCE_RUNNER = (
    ROOT / "tools" / "route2_release" / "run_fixedbox590_residual_force_panel.py"
)
RESIDUAL_FORCE_AGGREGATOR = (
    ROOT
    / "tools"
    / "route2_release"
    / "aggregate_fixedbox590_residual_force_panel.py"
)
SYMMETRY_PANEL_RUNNER = (
    ROOT / "tools" / "route2_release" / "run_fixedbox590_symmetry_panel.py"
)
SYMMETRY_PANEL_AGGREGATOR = (
    ROOT / "tools" / "route2_release" / "aggregate_fixedbox590_symmetry_panel.py"
)
PES_PANEL_DOC = ROOT / "docs" / "route2" / "PES_PANEL.md"
PES_CARTESIAN_DOC = ROOT / "docs" / "route2" / "CARTESIAN_PANEL.md"
RESIDUAL_FORCE_DOC = ROOT / "docs" / "route2" / "RESIDUAL_FORCE_GATE.md"
SYMMETRY_PANEL_DOC = ROOT / "docs" / "route2" / "SYMMETRY_PANEL.md"
PATH_EVIDENCE = (
    ROOT / "docs" / "route2" / "evidence" / "fixedbox590-water-path-241e98b7"
)
BOX_EVIDENCE = (
    ROOT / "docs" / "route2" / "evidence" / "fixedbox590-water-box-convergence-a7fdf2fa"
)
PES_PANEL_EVIDENCE = (
    ROOT / "docs" / "route2" / "evidence" / "fixedbox590-pes-panel-f7f68165"
)
PES_CARTESIAN_EVIDENCE = (
    ROOT
    / "docs"
    / "route2"
    / "evidence"
    / "fixedbox590-cartesian-panel-abb34a05"
)
RESIDUAL_FORCE_EVIDENCE = (
    ROOT
    / "docs"
    / "route2"
    / "evidence"
    / "fixedbox590-residual-force-9918dea6"
)


def _git(*arguments: str, root: Path = ROOT) -> str:
    return subprocess.check_output(("git", *arguments), cwd=root, text=True).strip()


def test_repository_snapshot_requires_clean_tree_and_binds_git_blobs(tmp_path):
    repository = tmp_path / "repo"
    repository.mkdir()
    subprocess.run(("git", "init", "-q"), cwd=repository, check=True)
    subprocess.run(
        ("git", "config", "user.email", "route2@example.invalid"),
        cwd=repository,
        check=True,
    )
    subprocess.run(
        ("git", "config", "user.name", "Route 2 Evidence"),
        cwd=repository,
        check=True,
    )
    source = repository / "kernel.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(("git", "add", "kernel.py"), cwd=repository, check=True)
    subprocess.run(
        ("git", "commit", "-q", "-m", "baseline"),
        cwd=repository,
        check=True,
    )

    snapshot = RepositorySnapshot.capture(repository)
    hashes = committed_source_hashes(snapshot, ("kernel.py",))
    assert hashes == {"kernel.py": hashlib.sha256(b"VALUE = 1\n").hexdigest()}
    snapshot.assert_unchanged()

    source.write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="clean working tree"):
        RepositorySnapshot.capture(repository)
    with pytest.raises(RuntimeError, match="does not match Git"):
        committed_source_hashes(snapshot, ("kernel.py",))
    with pytest.raises(RuntimeError, match="working-tree state changed"):
        snapshot.assert_unchanged()


def test_external_writer_refuses_to_dirty_checkout(tmp_path):
    snapshot = RepositorySnapshot.capture(ROOT, require_clean=False)
    with pytest.raises(ValueError, match="outside the source checkout"):
        write_external_json_artifact(
            snapshot, ROOT / "forbidden-evidence.json", {"status": "no"}
        )

    output = tmp_path / "evidence.json"
    record = write_external_json_artifact(snapshot, output, {"status": "diagnostic"})
    assert json.loads(output.read_text()) == {"status": "diagnostic"}
    assert record["sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()


def test_source_collection_rejects_untracked_required_source(tmp_path):
    with pytest.raises(RuntimeError, match="not tracked"):
        collect_loaded_repository_sources(
            ROOT,
            modules={},
            required_paths=(tmp_path / "outside.py",),
        )


def test_canonical_json_hash_is_order_independent_and_rejects_nan():
    assert canonical_json_sha256({"b": 2, "a": 1}) == canonical_json_sha256(
        {"a": 1, "b": 2}
    )
    with pytest.raises(ValueError):
        canonical_json_sha256({"bad": float("nan")})


def test_fixedbox590_runner_help_is_dependency_and_checkpoint_free():
    result = subprocess.run(
        (sys.executable, str(RUNNER), "--help"),
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--mode" in result.stdout
    assert "--output" in result.stdout
    assert "not Tier E/F/H/V/M" in RUNNER.read_text(encoding="utf-8")


def test_fixedbox590_path_runner_preregisters_panel_loop_and_stays_disabled():
    result = subprocess.run(
        (sys.executable, str(PATH_RUNNER), "--help"),
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--loop-subdivisions" in result.stdout
    helper = ROOT / "tools" / "route2_release" / "fixedbox590_water_path.py"
    text = PATH_RUNNER.read_text(encoding="utf-8") + helper.read_text(encoding="utf-8")
    for requirement in (
        "PANEL_COEFFICIENTS_A",
        "closed_loop_work",
        "reverse_closed_path",
        "cold_warm_record",
        '"multi_molecule_pes_panel": False',
        '"box_convergence": False',
        '"capabilities": {tier: False',
    ):
        assert requirement in text


def test_fixedbox590_box_runner_preregisters_distinct_sizes_and_stays_disabled():
    result = subprocess.run(
        (sys.executable, str(BOX_RUNNER), "--help"),
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    text = BOX_RUNNER.read_text(encoding="utf-8")
    for requirement in (
        "BOX_LENGTHS_A",
        "summarize_box_convergence",
        "box_operator_convergence",
        '"multi_geometry_box_convergence": False',
        '"capabilities": {tier: False',
    ):
        assert requirement in text


def test_fixedbox590_pes_panel_runner_is_sharded_source_bound_and_stays_disabled():
    result = subprocess.run(
        (sys.executable, str(PES_PANEL_RUNNER), "--help"),
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--molecule-start" in result.stdout
    assert "--molecule-stop" in result.stdout
    text = PES_PANEL_RUNNER.read_text(encoding="utf-8")
    for requirement in (
        "PES_PANEL_CONTRACT_VERSION",
        "PES_PANEL_ASSET_SHA256",
        "PANEL_ASSET_PATH",
        "_configure_numerical_determinism",
        "torch.use_deterministic_algorithms(True)",
        "CUBLAS_WORKSPACE_CONFIG",
        "PES_PANEL_DIRECTIONAL_STEPS_A",
        "panel_geometries",
        "panel_directions",
        'aggregate_multi_molecule_pes_panel": False',
        '"capabilities": {tier: False',
        "plus_energy = self.scalar.evaluate_energy(plus, plus_state.y)",
        "minus_energy = self.scalar.evaluate_energy(minus, minus_state.y)",
    ):
        assert requirement in text
    assert "(*source_paths, PANEL_ASSET_PATH)" in text
    assert (
        text.index("plus_state = self.solve(")
        < text.index("plus_energy = self.scalar.evaluate_energy(plus, plus_state.y)")
        < text.index("minus_state = self.solve(")
        < text.index("minus_energy = self.scalar.evaluate_energy(minus, minus_state.y)")
    )
    document = PES_PANEL_DOC.read_text(encoding="utf-8")
    assert PES_PANEL_RUNNER.name in document
    assert "status=pass" in document
    assert "fixedbox590-pes-panel-f7f68165" in document
    assert "component-resolved Cartesian panel" in document
    assert "not" in document and "optimized transition state" in document
    assert "normalized coordinate tangent" in document


def test_pes_panel_aggregator_recomputes_raw_values_and_cannot_admit_capabilities():
    result = subprocess.run(
        (sys.executable, str(PES_PANEL_AGGREGATOR), "--help"),
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    text = PES_PANEL_AGGREGATOR.read_text(encoding="utf-8")
    for requirement in (
        "summarize_directional_derivatives",
        "geometry_sha256(plus)",
        "geometry_sha256(minus)",
        "source_relative <= 1.0e-8",
        "energy_difference <= 1.0e-8",
        "CAPABILITIES = {tier: False",
        "sys.exit(2)",
        '"numerical_determinism": payload.get("numerical_determinism")',
    ):
        assert requirement in text


def test_cartesian_panel_runner_and_aggregator_are_raw_source_bound_and_disabled():
    for command in (PES_CARTESIAN_RUNNER, PES_CARTESIAN_AGGREGATOR):
        result = subprocess.run(
            (sys.executable, str(command), "--help"),
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert "--output" in result.stdout

    runner = PES_CARTESIAN_RUNNER.read_text(encoding="utf-8")
    for requirement in (
        "PES_CARTESIAN_PANEL_CONTRACT_VERSION",
        "PES_CARTESIAN_PANEL_STEPS_A",
        "summarize_cartesian_force_differences",
        "raw_displaced_components",
        "plus_energy = self.scalar.evaluate_energy(plus, plus_state.y)",
        "minus_energy = self.scalar.evaluate_energy(minus, minus_state.y)",
        'aggregate_cartesian_pes_panel": False',
        '"capabilities": {tier: False',
    ):
        assert requirement in runner
    assert "(*source_paths, PANEL_ASSET_PATH)" in runner
    assert (
        runner.index("plus_state = self.solve(")
        < runner.index("plus_energy = self.scalar.evaluate_energy(plus, plus_state.y)")
        < runner.index("minus_state = self.solve(")
        < runner.index("minus_energy = self.scalar.evaluate_energy(minus, minus_state.y)")
    )

    aggregator = PES_CARTESIAN_AGGREGATOR.read_text(encoding="utf-8")
    for requirement in (
        "geometry_sha256(plus)",
        "geometry_sha256(minus)",
        "summarize_cartesian_force_differences",
        "summarize_cartesian_pes_panel",
        "CAPABILITIES = {tier: False",
        "sys.exit(2)",
        '"numerical_determinism": payload.get("numerical_determinism")',
    ):
        assert requirement in aggregator

    document = PES_CARTESIAN_DOC.read_text(encoding="utf-8")
    assert PES_CARTESIAN_RUNNER.name in document
    assert PES_CARTESIAN_AGGREGATOR.name in document
    assert "status=pass" in document
    assert "fixedbox590-cartesian-panel-abb34a05" in document
    assert "6e64d1afaa0201dd1ecfd2950e1ea62a" in document
    assert "465 Cartesian components" in document
    assert "central-order-or-ten-percent-error-plateau-v1" in document
    assert "E/F/H/V/M therefore remain false" in document


def test_residual_force_runner_and_aggregator_are_preregistered_and_disabled():
    for command in (RESIDUAL_FORCE_RUNNER, RESIDUAL_FORCE_AGGREGATOR):
        result = subprocess.run(
            (sys.executable, str(command), "--help"),
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert "--output" in result.stdout

    runner = RESIDUAL_FORCE_RUNNER.read_text(encoding="utf-8")
    for requirement in (
        "RESIDUAL_FORCE_CONTRACT_VERSION",
        "PRIMAL_REFINEMENT_TOLERANCES",
        "ADJOINT_REFINEMENT_RELATIVE_TOLERANCES",
        "ADJOINT_REFINEMENT_ABSOLUTE_TOLERANCES",
        "summarize_residual_force_refinement",
        "primal_levels[0] = dict(adjoint_levels[-1])",
        'CAPABILITIES = {tier: False',
        "repository.assert_unchanged()",
    ):
        assert requirement in runner

    aggregator = RESIDUAL_FORCE_AGGREGATOR.read_text(encoding="utf-8")
    for requirement in (
        "summarize_residual_force_refinement",
        "summarize_residual_force_panel",
        "geometry_sha256(atoms)",
        "committed_source_hashes(repository, source_hashes)",
        'CAPABILITIES = {tier: False',
        "sys.exit(2)",
    ):
        assert requirement in aggregator

    document = RESIDUAL_FORCE_DOC.read_text(encoding="utf-8")
    assert RESIDUAL_FORCE_RUNNER.name in document
    assert RESIDUAL_FORCE_AGGREGATOR.name in document
    assert "not a rigorous analytic upper bound" in document
    assert "5e-5 eV/A" in document
    assert "E/F/H/V/M remain false" in document
    assert "status=pass" in document
    assert "fixedbox590-residual-force-9918dea6" in document
    assert "b4c964f17ccac1230ecf00d21709c8a68a39e31735f3cdc80a8c31d7a3291150" in document


def test_symmetry_panel_runner_and_aggregator_are_preregistered_and_disabled():
    for command in (SYMMETRY_PANEL_RUNNER, SYMMETRY_PANEL_AGGREGATOR):
        result = subprocess.run(
            (sys.executable, str(command), "--help"),
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert "--output" in result.stdout
    runner = SYMMETRY_PANEL_RUNNER.read_text(encoding="utf-8")
    for requirement in (
        "SYMMETRY_PANEL_CONTRACT_VERSION",
        "symmetry_panel_rotations",
        "symmetry_panel_permutation",
        "summarize_rigid_symmetry",
        "closed_loop_work",
        "cold_warm_record",
        "summarize_bidirectional_loop_record",
        'CAPABILITIES = {tier: False',
        "repository.assert_unchanged()",
    ):
        assert requirement in runner
    aggregator = SYMMETRY_PANEL_AGGREGATOR.read_text(encoding="utf-8")
    for requirement in (
        "summarize_rigid_symmetry",
        "summarize_bidirectional_loop_record",
        "summarize_symmetry_panel",
        "geometry_sha256(atoms)",
        "committed_source_hashes(repository, source_hashes)",
        'CAPABILITIES = {tier: False',
        "sys.exit(2)",
    ):
        assert requirement in aggregator
    document = SYMMETRY_PANEL_DOC.read_text(encoding="utf-8")
    assert SYMMETRY_PANEL_RUNNER.name in document
    assert SYMMETRY_PANEL_AGGREGATOR.name in document
    assert "20 reference geometries" in document
    assert "E/F/H/V/M" in document
    assert "not matched" in document


def test_runner_source_binding_list_contains_unique_scalar_and_derivative_kernel():
    text = RUNNER.read_text(encoding="utf-8") + COMMON.read_text(encoding="utf-8")
    for relative in (
        "maple/solvation/api/scalar_registry.py",
        "maple/solvation/coupling/energy.py",
        "maple/solvation/coupling/fixed_point.py",
        "maple/solvation/coupling/adjoint.py",
        "maple/solvation/models/mace_polar.py",
        "maple/solvation/continuum/conjugate_fixed_topology_cpcm.py",
    ):
        assert relative in text


def test_fixedbox590_water_path_artifact_is_source_bound_and_capability_closed():
    measurements = json.loads((PATH_EVIDENCE / "measurements.json").read_text())
    manifest = json.loads((PATH_EVIDENCE / "manifest.json").read_text())
    assert measurements["execution_git_head"] == manifest["execution_git_head"]
    assert measurements["working_tree_clean"] is True
    assert measurements["capabilities"] == {
        "E": False,
        "F": False,
        "H": False,
        "M": False,
        "V": False,
    }
    assert all(measurements["gates"].values())
    assert manifest["evidence_status"].endswith("not release admission")
    assert manifest["scope_limits"]["multi_molecule_pes_panel"] is False
    assert manifest["scope_limits"]["box_convergence"] is False
    assert manifest["raw_summary"]["directional_fd_measurement_count"] == 63

    source_hashes = measurements["source_files_sha256"]
    assert source_hashes
    for relative, expected in source_hashes.items():
        blob = subprocess.run(
            ("git", "show", f"{measurements['execution_git_head']}:{relative}"),
            cwd=ROOT,
            check=True,
            capture_output=True,
        ).stdout
        assert hashlib.sha256(blob).hexdigest() == expected


def test_fixedbox590_water_path_artifact_file_hashes_and_raw_gates_close():
    manifest = json.loads((PATH_EVIDENCE / "manifest.json").read_text())
    for name, expected in manifest["artifact_sha256"].items():
        assert (
            hashlib.sha256((PATH_EVIDENCE / name).read_bytes()).hexdigest() == expected
        )
    measurements = json.loads((PATH_EVIDENCE / "measurements.json").read_text())
    assert len(measurements["measurements"]["topology_hashes"]) == 1
    assert measurements["measurements"]["maximum_primal_residual"] <= 1.0e-12
    loop = measurements["measurements"]["closed_loop"]
    for name in ("cold_forward", "cold_reverse", "warm_forward", "warm_reverse"):
        assert abs(loop[name]["simpson_work_eV"]) <= loop[name]["gate_threshold_eV"]
    for geometry in measurements["measurements"]["panel"]:
        assert geometry["cold_warm"]["numerically_equivalent"] is True
        for direction in geometry["directional_force_fd"].values():
            assert direction["all_gates_passed"] is True


def test_fixedbox590_box_artifact_is_source_bound_and_capability_closed():
    measurements = json.loads((BOX_EVIDENCE / "measurements.json").read_text())
    manifest = json.loads((BOX_EVIDENCE / "manifest.json").read_text())
    assert measurements["execution_git_head"] == manifest["execution_git_head"]
    assert measurements["working_tree_clean"] is True
    assert measurements["capabilities"] == {
        "E": False,
        "F": False,
        "H": False,
        "M": False,
        "V": False,
    }
    assert all(measurements["measurements"]["gates"].values())
    assert manifest["evidence_status"].endswith("not release admission")
    assert manifest["scope_limits"]["single_geometry"] is True
    assert manifest["scope_limits"]["multi_geometry_box_convergence"] is False
    assert manifest["box_lengths_A"] == [32, 40, 48, 56]
    assert manifest["tail_pair_A"] == [48, 56]

    source_hashes = measurements["source_files_sha256"]
    assert source_hashes
    for relative, expected in source_hashes.items():
        blob = subprocess.run(
            ("git", "show", f"{measurements['execution_git_head']}:{relative}"),
            cwd=ROOT,
            check=True,
            capture_output=True,
        ).stdout
        assert hashlib.sha256(blob).hexdigest() == expected


def test_fixedbox590_box_artifact_file_hashes_and_raw_gates_close():
    manifest = json.loads((BOX_EVIDENCE / "manifest.json").read_text())
    sums = {}
    for line in (BOX_EVIDENCE / "SHA256SUMS").read_text().splitlines():
        expected, name = line.split("  ", 1)
        sums[name] = expected
    assert (
        sums["manifest.json"]
        == hashlib.sha256((BOX_EVIDENCE / "manifest.json").read_bytes()).hexdigest()
    )
    for name, expected in manifest["artifact_sha256"].items():
        assert (
            hashlib.sha256((BOX_EVIDENCE / name).read_bytes()).hexdigest() == expected
        )
        assert sums[name] == expected
    raw = manifest["raw_summary"]
    assert raw["maximum_primal_residual"] <= 1.0e-12
    assert raw["maximum_adjoint_residual"] <= 1.0e-10
    assert len(raw["surface_topology_hashes"]) == 1
    assert raw["tail"]["total_energy_abs_eV"] == pytest.approx(2.871227934519993e-06)
    assert raw["tail"]["continuum_energy_abs_eV"] == pytest.approx(
        2.3831270675594984e-08
    )
    assert raw["tail"]["force_rms_eV_per_A"] == pytest.approx(6.785481887261842e-07)


def test_fixedbox590_pes_panel_artifact_is_source_bound_and_capability_closed():
    manifest = json.loads((PES_PANEL_EVIDENCE / "manifest.json").read_text())
    aggregate = json.loads((PES_PANEL_EVIDENCE / "aggregate.json").read_text())
    shards = [
        json.loads(path.read_text())
        for path in sorted(PES_PANEL_EVIDENCE.glob("shard-*.json"))
    ]

    assert len(shards) == 20
    assert [
        shard["panel_contract"]["shard_start"] for shard in shards
    ] == list(range(20))
    assert [
        shard["panel_contract"]["shard_stop"] for shard in shards
    ] == list(range(1, 21))
    assert aggregate["status"] == "pass"
    assert aggregate["capabilities"] == {
        "E": False,
        "F": False,
        "H": False,
        "M": False,
        "V": False,
    }
    assert manifest["capabilities"] == aggregate["capabilities"]
    assert manifest["evidence_status"].endswith("not release admission")
    assert manifest["execution_git_head"] == aggregate["execution_git_head"]
    assert manifest["execution_git_tree"] == aggregate["execution_git_tree"]
    assert manifest["tested_working_tree_clean"] is True
    assert all(shard["working_tree_clean"] is True for shard in shards)
    assert {
        shard["execution_git_head"] for shard in shards
    } == {manifest["execution_git_head"]}
    assert {
        shard["checkpoint"]["sha256"] for shard in shards
    } == {manifest["checkpoint_sha256"]}

    source_hashes = aggregate["source_files_sha256"]
    assert source_hashes
    assert all(shard["source_files_sha256"] == source_hashes for shard in shards)
    for relative, expected in source_hashes.items():
        blob = subprocess.run(
            ("git", "show", f"{manifest['execution_git_head']}:{relative}"),
            cwd=ROOT,
            check=True,
            capture_output=True,
        ).stdout
        assert hashlib.sha256(blob).hexdigest() == expected


def test_fixedbox590_pes_panel_artifact_hashes_and_raw_gates_close():
    manifest = json.loads((PES_PANEL_EVIDENCE / "manifest.json").read_text())
    aggregate = json.loads((PES_PANEL_EVIDENCE / "aggregate.json").read_text())
    sums = {}
    for line in (PES_PANEL_EVIDENCE / "SHA256SUMS").read_text().splitlines():
        expected, name = line.split("  ", 1)
        sums[name] = expected

    for name, expected in sums.items():
        assert hashlib.sha256((PES_PANEL_EVIDENCE / name).read_bytes()).hexdigest() == expected
    assert manifest["aggregate_artifact_sha256"] == sums["aggregate.json"]
    assert manifest["artifact_sha256"]["aggregate.json"] == sums["aggregate.json"]
    assert manifest["aggregate_measurement_sha256"] == aggregate[
        "aggregate_measurement_sha256"
    ]

    panel = aggregate["panel_summary"]
    paths = aggregate["path_summary"]
    assert panel["all_gates_passed"] is True
    assert panel["molecule_count"] == 20
    assert panel["geometry_count"] == 60
    assert panel["directional_record_count"] == 180
    assert panel["directional_sample_count"] == 540
    assert paths["all_gates_passed"] is True
    assert paths["path_count"] == 2
    assert paths["path_geometry_count"] == 11
    assert all(panel["gates"].values())
    assert all(paths["gates"].values())

    raw = manifest["raw_summary"]
    assert raw["maximum_base_absolute_error_eV_per_A"] == pytest.approx(
        2.9825647950509904e-05
    )
    assert raw["maximum_base_relative_error_where_applicable"] == pytest.approx(
        0.001257946545683231
    )
    assert raw["maximum_path_absolute_error_eV_per_A"] == pytest.approx(
        2.4073161528193054e-05
    )
    assert raw["maximum_primal_residual"] <= 1.0e-12
    assert raw["maximum_adjoint_residual"] <= 1.0e-10
    assert raw["maximum_cold_warm_energy_abs_difference_eV"] <= 1.0e-8
    assert raw["maximum_cold_warm_source_relative_difference"] <= 1.0e-8
    assert manifest["scope_limits"]["directional_same_scalar_force_fd"] is True
    assert manifest["scope_limits"]["component_resolved_cartesian_force_fd"] is False
    assert manifest["scope_limits"]["hessian_frequency_ts_nve"] is False


def test_cartesian_panel_artifact_is_source_bound_and_capability_closed():
    manifest = json.loads((PES_CARTESIAN_EVIDENCE / "manifest.json").read_text())
    aggregate = json.loads((PES_CARTESIAN_EVIDENCE / "aggregate.json").read_text())
    shards = [
        json.loads(path.read_text())
        for path in sorted(PES_CARTESIAN_EVIDENCE.glob("shard-*.json"))
    ]

    assert len(shards) == 20
    assert [
        shard["panel_contract"]["shard_start"] for shard in shards
    ] == list(range(20))
    assert [
        shard["panel_contract"]["shard_stop"] for shard in shards
    ] == list(range(1, 21))
    assert aggregate["status"] == "pass"
    assert aggregate["capabilities"] == {
        "E": False,
        "F": False,
        "H": False,
        "M": False,
        "V": False,
    }
    assert manifest["capabilities"] == aggregate["capabilities"]
    assert manifest["evidence_status"].endswith("not release admission")
    assert manifest["execution_git_head"] == aggregate["execution_git_head"]
    assert manifest["execution_git_tree"] == aggregate["execution_git_tree"]
    assert manifest["tested_working_tree_clean"] is True
    assert all(shard["working_tree_clean"] is True for shard in shards)
    assert {
        shard["execution_git_head"] for shard in shards
    } == {manifest["execution_git_head"]}
    assert {
        shard["checkpoint"]["sha256"] for shard in shards
    } == {manifest["checkpoint_sha256"]}

    source_hashes = aggregate["source_files_sha256"]
    assert source_hashes
    assert all(shard["source_files_sha256"] == source_hashes for shard in shards)
    for relative, expected in source_hashes.items():
        blob = subprocess.run(
            ("git", "show", f"{manifest['execution_git_head']}:{relative}"),
            cwd=ROOT,
            check=True,
            capture_output=True,
        ).stdout
        assert hashlib.sha256(blob).hexdigest() == expected


def test_cartesian_panel_artifact_hashes_and_raw_gates_close():
    manifest = json.loads((PES_CARTESIAN_EVIDENCE / "manifest.json").read_text())
    aggregate = json.loads((PES_CARTESIAN_EVIDENCE / "aggregate.json").read_text())
    sums = {}
    for line in (PES_CARTESIAN_EVIDENCE / "SHA256SUMS").read_text().splitlines():
        expected, name = line.split("  ", 1)
        sums[name] = expected

    for name, expected in sums.items():
        assert (
            hashlib.sha256((PES_CARTESIAN_EVIDENCE / name).read_bytes()).hexdigest()
            == expected
        )
    assert manifest["aggregate_artifact_sha256"] == sums["aggregate.json"]
    assert manifest["artifact_sha256"]["aggregate.json"] == sums["aggregate.json"]
    assert manifest["aggregate_measurement_sha256"] == aggregate[
        "aggregate_measurement_sha256"
    ]

    panel = aggregate["cartesian_panel_summary"]
    assert panel["all_gates_passed"] is True
    assert panel["molecule_count"] == 20
    assert panel["geometry_count"] == 20
    assert panel["component_count"] == 465
    assert panel["component_sample_count"] == 1395
    assert panel["step_record_count"] == 60
    assert panel["low_error_plateau_count"] == 0
    assert all(panel["gates"].values())

    raw = manifest["raw_summary"]
    assert raw["maximum_rms_error_eV_per_A"] == pytest.approx(
        5.893341195090511e-06
    )
    assert raw["maximum_component_error_eV_per_A"] == pytest.approx(
        1.8557801318763723e-05
    )
    assert raw["minimum_observed_first_to_last_order"] == pytest.approx(
        1.9428853284937788
    )
    assert raw["maximum_primal_residual"] <= 1.0e-12
    assert raw["maximum_adjoint_residual"] <= 1.0e-10
    assert raw["maximum_cold_warm_energy_abs_difference_eV"] <= 1.0e-8
    assert raw["maximum_cold_warm_source_relative_difference"] <= 1.0e-8
    assert manifest["scope_limits"]["directional_same_scalar_force_fd"] is True
    assert (
        manifest["scope_limits"]["component_resolved_cartesian_force_fd"] is True
    )
    assert manifest["scope_limits"]["residual_based_force_error_bound"] is False
    assert manifest["scope_limits"]["hessian_frequency_ts_nve"] is False


def test_residual_force_artifact_is_source_bound_and_capability_closed():
    manifest = json.loads((RESIDUAL_FORCE_EVIDENCE / "manifest.json").read_text())
    aggregate = json.loads((RESIDUAL_FORCE_EVIDENCE / "aggregate.json").read_text())
    shards = [
        json.loads(path.read_text())
        for path in sorted(RESIDUAL_FORCE_EVIDENCE.glob("shard-*.json"))
    ]
    assert len(shards) == 20
    assert aggregate["status"] == "pass"
    assert aggregate["capabilities"] == {
        "E": False,
        "F": False,
        "H": False,
        "M": False,
        "V": False,
    }
    assert manifest["capabilities"] == aggregate["capabilities"]
    assert manifest["evidence_status"].endswith("not release admission")
    assert manifest["execution_git_head"] == aggregate["execution_git_head"]
    assert manifest["execution_git_tree"] == aggregate["execution_git_tree"]
    assert manifest["tested_working_tree_clean"] is True
    assert all(shard["working_tree_clean"] is True for shard in shards)
    assert {
        shard["execution_git_head"] for shard in shards
    } == {manifest["execution_git_head"]}
    assert {
        shard["checkpoint"]["sha256"] for shard in shards
    } == {manifest["checkpoint_sha256"]}
    assert all(
        shard["source_files_sha256"] == aggregate["source_files_sha256"]
        for shard in shards
    )
    for relative, expected in aggregate["source_files_sha256"].items():
        blob = subprocess.run(
            ("git", "show", f"{manifest['execution_git_head']}:{relative}"),
            cwd=ROOT,
            check=True,
            capture_output=True,
        ).stdout
        assert hashlib.sha256(blob).hexdigest() == expected


def test_residual_force_artifact_hashes_and_raw_gates_close():
    manifest = json.loads((RESIDUAL_FORCE_EVIDENCE / "manifest.json").read_text())
    aggregate = json.loads((RESIDUAL_FORCE_EVIDENCE / "aggregate.json").read_text())
    sums = {}
    for line in (RESIDUAL_FORCE_EVIDENCE / "SHA256SUMS").read_text().splitlines():
        expected, name = line.split("  ", 1)
        sums[name] = expected
    for name, expected in sums.items():
        assert (
            hashlib.sha256((RESIDUAL_FORCE_EVIDENCE / name).read_bytes()).hexdigest()
            == expected
        )
    assert manifest["aggregate_artifact_sha256"] == sums["aggregate.json"]
    assert manifest["aggregate_measurement_sha256"] == aggregate[
        "aggregate_measurement_sha256"
    ]
    panel = aggregate["residual_force_panel_summary"]
    assert panel["all_gates_passed"] is True
    assert panel["molecule_count"] == 20
    assert all(panel["gates"].values())
    assert panel["maximum_estimated_rms_error_eV_per_A"] == pytest.approx(
        3.702427949407424e-13
    )
    assert panel["maximum_estimated_component_error_eV_per_A"] == pytest.approx(
        1.2656542480726785e-12
    )
    assert panel[
        "maximum_observed_release_to_reference_error_eV_per_A"
    ] == pytest.approx(3.6060043839825084e-13)
    assert panel["maximum_actual_primal_residual_by_level"] == pytest.approx(
        [9.14172390425434e-13, 9.562721835217735e-14, 9.822859083946273e-15]
    )
    assert panel["maximum_actual_adjoint_residual_by_level"] == pytest.approx(
        [2.464164043685235e-12, 3.6005901866951495e-13, 4.0331160252416295e-14]
    )
    assert manifest["scope_limits"][
        "residual_refinement_force_error_estimate"
    ] is True
    assert manifest["scope_limits"]["rigorous_analytic_upper_bound"] is False
    assert manifest["scope_limits"]["all_panel_symmetry"] is False
    assert manifest["scope_limits"]["complete_nonpolar_free_energy"] is False
