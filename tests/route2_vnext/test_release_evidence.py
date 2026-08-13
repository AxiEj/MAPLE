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
PES_PANEL_DOC = ROOT / "docs" / "route2" / "PES_PANEL.md"
PATH_EVIDENCE = (
    ROOT / "docs" / "route2" / "evidence" / "fixedbox590-water-path-241e98b7"
)
BOX_EVIDENCE = (
    ROOT / "docs" / "route2" / "evidence" / "fixedbox590-water-box-convergence-a7fdf2fa"
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
        "PES_PANEL_DIRECTIONAL_STEPS_A",
        "panel_geometries",
        "panel_directions",
        'aggregate_multi_molecule_pes_panel": False',
        '"capabilities": {tier: False',
    ):
        assert requirement in text
    document = PES_PANEL_DOC.read_text(encoding="utf-8")
    assert PES_PANEL_RUNNER.name in document
    assert "has not yet been" in document
    assert "not" in document and "optimized transition state" in document


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
