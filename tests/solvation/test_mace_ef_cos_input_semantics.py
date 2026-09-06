"""Regression tests for the research-only MACE-EF-COS input audit."""

from __future__ import annotations

import importlib.util
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

PATH = (
    Path(__file__).resolve().parents[2]
    / "docs/implicit-solvation/benchmarks/audit_mace_ef_cos_input_semantics.py"
)


@pytest.fixture(scope="module")
def audit():
    spec = importlib.util.spec_from_file_location("ef_cos_input_audit", PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_smooth_quadratic_has_step_independent_raw_hessian(audit):
    hessian = np.array([[-2.0, 0.2, 0.0], [0.2, -3.0, 0.1], [0.0, 0.1, -4.0]])
    center = np.array([0.3, -0.2, 0.1])
    rows = audit.response_scan(
        lambda g: np.ones(3) + hessian @ g, center, [0.01, 0.005]
    )
    for row in rows:
        np.testing.assert_allclose(row["raw_hessian"], hessian, atol=1e-12)
        assert row["antisymmetric_frobenius_norm"] < 1e-12
        np.testing.assert_allclose(row["center_gradient"], 1.0 + hessian @ center)


def test_cusp_is_reported_as_derivative_jump_not_repaired(audit):
    rows = audit.response_scan(
        lambda g: 0.03 * np.sign(g) - 0.2 * g, np.zeros(3), [0.002, 0.001]
    )
    for row, step in zip(rows, [0.002, 0.001]):
        np.testing.assert_allclose(row["raw_hessian"], np.eye(3) * (0.03 / step - 0.2))
        np.testing.assert_allclose(row["axis_derivative_jumps"], 0.06 - 0.4 * step)
    assert rows[1]["symmetric_eigenvalues"][0] > rows[0]["symmetric_eigenvalues"][0]


def test_antisymmetry_is_not_discarded(audit):
    raw = np.array([[1.0, 2.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -2.0]])
    row = audit.response_scan(lambda g: raw @ g, np.zeros(3), [0.001])[0]
    np.testing.assert_allclose(row["raw_hessian"], raw)
    np.testing.assert_allclose(row["symmetric_hessian"], 0.5 * (raw + raw.T))
    assert row["antisymmetric_frobenius_norm"] == pytest.approx(np.sqrt(2.0))


@pytest.mark.parametrize("steps", [[], [0.0], [-0.1], [float("nan")]])
def test_invalid_steps_fail_before_model_execution(audit, steps):
    def forbidden(_):
        raise AssertionError("invalid input reached evaluator")

    with pytest.raises(ValueError):
        audit.response_scan(forbidden, np.zeros(3), steps)


def test_invalid_gradient_fails_closed(audit):
    with pytest.raises(ValueError, match="gradient"):
        audit.response_scan(lambda _: np.full(3, np.nan), np.zeros(3), [0.001])


@pytest.fixture
def projector():
    torch = pytest.importorskip("torch")
    gto = pytest.importorskip("graph_longrange.gto_utils")
    return gto.DisplacedGTOExternalFieldBlock(1, [0.5, 1.0], "receiver").to(
        torch.float64
    )


def test_native_and_local_affine_projections_match(audit, projector):
    torch = pytest.importorskip("torch")
    from graph_longrange.gto_utils import DisplacedGTOExternalFieldBlock

    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float64
    )
    positions -= positions.mean(0)
    g = torch.tensor([0.013, -0.007, 0.009], dtype=torch.float64)
    local = torch.cat(((positions @ g)[:, None], g.expand(3, 3)), dim=1)
    native = 0.5 * DisplacedGTOExternalFieldBlock.forward(
        projector,
        torch.zeros(3, dtype=torch.long),
        positions,
        torch.cat((torch.zeros(1, dtype=g.dtype), g))[None, :],
    )
    actual = audit.project_local_potential(projector, local)
    torch.testing.assert_close(actual, native, atol=1e-14, rtol=0.0)
    torch.testing.assert_close(
        audit.project_local_potential(projector, -local), -actual
    )


def test_local_projection_is_linear_and_has_potential_derivative(audit, projector):
    torch = pytest.importorskip("torch")
    values = torch.arange(12, dtype=torch.float64).reshape(3, 4).requires_grad_()
    offset = torch.ones_like(values) * 0.3
    project = lambda x: audit.project_local_potential(projector, x)
    torch.testing.assert_close(
        project(values + offset), project(values) + project(offset)
    )
    derivative = torch.autograd.grad(project(values).sum(), values)[0]
    torch.testing.assert_close(
        derivative[:, 0], (0.5 * projector.matrix[:, 0].sum()).expand(3)
    )


def test_legacy_projection_is_even_in_scalar_rows_and_ignores_potential(
    audit, projector
):
    torch = pytest.importorskip("torch")
    values = torch.tensor([[0.4, 0.2, -0.3, 0.1]], dtype=torch.float64)
    scalar = projector.matrix[:, 0] != 0
    legacy = audit.reconstruct_legacy_projection(projector, values)
    reverse = audit.reconstruct_legacy_projection(projector, -values)
    torch.testing.assert_close(legacy[:, scalar], reverse[:, scalar])
    assert torch.max(torch.abs(legacy[:, scalar])) > 0
    changed = values.clone()
    changed[:, 0] += 0.8
    torch.testing.assert_close(
        audit.reconstruct_legacy_projection(projector, changed), legacy
    )


def test_projection_rejects_invalid_shape_or_nonfinite(audit, projector):
    torch = pytest.importorskip("torch")
    for values in [torch.zeros(3, 3), torch.full((3, 4), float("nan"))]:
        with pytest.raises(ValueError):
            audit.project_local_potential(projector, values)


def test_json_output_never_overwrites(audit, tmp_path):
    path = tmp_path / "evidence.json"
    audit.write_json_once(path, {"original": True})
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        audit.write_json_once(path, {"original": False})
    assert path.read_bytes() == before


@pytest.mark.parametrize("indices,valid", [([2, 0, 1], True), ([0, 1, 2], False)])
def test_archive_column_inspection_is_independent_and_fail_closed(
    audit, tmp_path, indices, valid
):
    selectors = ", ".join(f"torch.select(external_field, 1, {i})" for i in indices)
    source = f"zero = torch.zeros_like(dummy)\nfirst = [zero, {selectors}]\nsecond = [zero, {selectors}]\n"
    path = tmp_path / "archive.pt"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("model/code/mace/modules/extensions.py", source)
    if valid:
        inspected = audit._archive_evidence(path)
        assert inspected["inspected_gradient_component_indices"] == [2, 0, 1]
        assert inspected["candidate_local_jet_column_indices"] == [0, 3, 1, 2]
    else:
        with pytest.raises(ValueError, match="column conventions"):
            audit._archive_evidence(path)


def test_frozen_cuda_evidence_is_bound_and_preserves_raw_arithmetic():
    directory = PATH.parent
    protocol = directory / "mace-ef-cos-input-semantics-prereg-v1.json"
    evidence = json.loads(
        (directory / "mace-ef-cos-input-semantics-v2.json").read_text()
    )
    preregistration = json.loads(protocol.read_text())
    assert (
        evidence["preregistration_sha256"]
        == hashlib.sha256(protocol.read_bytes()).hexdigest()
    )
    key = "docs/implicit-solvation/benchmarks/audit_mace_ef_cos_input_semantics.py"
    assert (
        evidence["source_files_sha256"][key]
        == hashlib.sha256(PATH.read_bytes()).hexdigest()
    )
    assert [r["geometry"] for r in evidence["records"]] == preregistration["molecules"]
    assert evidence["claim_boundary"]["checkpoint_modified"] is False
    assert (
        evidence["claim_boundary"]["full_model_native_vs_local_eager_tested"] is False
    )
    assert evidence["claim_boundary"]["repaired_checkpoint_tested"] is False
    assert evidence["claim_boundary"]["release_admitted"] is False
    assert evidence["archive_evidence"]["inspected_spin_half_constant_c2"] == 0.5
    assert evidence["archive_evidence"]["inspected_gradient_component_indices"] == [
        2,
        0,
        1,
    ]
    assert evidence["archive_evidence"]["candidate_local_jet_column_indices"] == [
        0,
        3,
        1,
        2,
    ]
    for record in evidence["records"]:
        assert record["projection"]["native_vs_local_max_abs"] == 0.0
        assert record["projection"]["local_vs_archive_layout_candidate_max_abs"] == 0.0
        for group in ("zero_center_scan", "nonzero_center_scan"):
            assert [
                r["step_ev_per_e_angstrom"] for r in record[group]
            ] == preregistration["steps_ev_per_e_angstrom"]
            for row in record[group]:
                plus = np.asarray(row["plus_gradients_columns"])
                minus = np.asarray(row["minus_gradients_columns"])
                raw = (plus - minus) / (2 * row["step_ev_per_e_angstrom"])
                np.testing.assert_allclose(
                    row["raw_hessian"], raw, rtol=0.0, atol=1e-12
                )
                np.testing.assert_allclose(
                    row["symmetric_hessian"], 0.5 * (raw + raw.T), rtol=0.0, atol=1e-12
                )
                np.testing.assert_allclose(
                    row["axis_derivative_jumps"],
                    np.diag(plus - minus),
                    rtol=0.0,
                    atol=1e-12,
                )
