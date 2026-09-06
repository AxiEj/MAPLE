"""Same-tensor V2 trace ablation, not production model registration."""

from __future__ import annotations

import importlib.util
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import pytest

DIRECTORY = Path(__file__).resolve().parents[2] / "docs/implicit-solvation/benchmarks"


@pytest.fixture(scope="module")
def audit():
    sys.path.insert(0, str(DIRECTORY))
    spec = importlib.util.spec_from_file_location(
        "trace_inputs_audit", DIRECTORY / "audit_mace_ef_v2_trace_inputs.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def source():
    # Minimal transformation fixture; actual entry point also pins full archive/source hashes.
    return (
        "\n".join(
            [
                "    external_potential_values0 = torch.squeeze(external_potential_values, -1)",
                "    field_norm = torch.linalg_vector_norm(external_field, 2, [-1], True)",
                "    _74 = torch.zeros_like(torch.select(field_norm, 1, 0), dtype=None, layout=None, device=None, pin_memory=False)",
                "    _75 = [_74, torch.select(external_field, 1, 2), torch.select(external_field, 1, 0), torch.select(external_field, 1, 1)]",
                "    field_irreps = torch.mul(_76, CONSTANTS.c2)",
                "    _77 = torch.slice(matrix, 0, 0, 9223372036854775807)",
                "    _84 = torch.index_put_(_82, _83, _81)",
                "    half_external_field = torch.div(field_irreps, field_feature_norms0)",
                "    field_norm0 = torch.linalg_vector_norm(external_field, 2, [-1], True)",
                "    _136 = torch.zeros_like(torch.select(field_norm0, 1, 0), dtype=None, layout=None, device=None, pin_memory=False)",
                "    _137 = [_136, torch.select(external_field, 1, 2), torch.select(external_field, 1, 0), torch.select(external_field, 1, 1)]",
                "    field_irreps0 = torch.mul(_138, CONSTANTS.c2)",
                "    _139 = torch.slice(matrix, 0, 0, 9223372036854775807)",
                "    _147 = torch.index_put_(_145, _146, _144)",
                "    half_external_field0 = torch.div(field_irreps0, field_feature_norms1)",
                "    q_term_per_atom = torch.mul(q_atom, external_potential_values0)",
            ]
        )
        + "\n"
    )


@pytest.mark.parametrize("mode", ["zero_scalar", "raw_v", "centered_v"])
def test_both_sites_change_but_qv_and_projection_convention_do_not(audit, source, mode):
    changed = audit.rewrite_source(source, mode)
    assert "linalg_vector_norm(external_field" not in changed
    assert "torch.index_put_" not in changed
    assert "_74 = input_potential_for_projection" in changed
    assert "_136 = input_potential_for_projection" in changed
    for line in source.splitlines():
        if (
            "torch.select(external_field" in line
            or "CONSTANTS.c2" in line
            or "half_external_field" in line
            or "q_term_per_atom" in line
        ):
            assert line in changed
    assert changed.count("input_potential_for_projection =") == 1
    if mode == "centered_v":
        assert "torch.scatter_add" in changed
        assert "torch.index_select" in changed


def test_roundtrip_preserves_source_and_unknown_modes_fail(audit, source):
    assert audit.rewrite_source(source, "roundtrip") == source
    with pytest.raises(ValueError):
        audit.rewrite_source(source, "fit_alpha")


def test_source_drift_fails_instead_of_partial_rewrite(audit, source):
    with pytest.raises(ValueError):
        audit.rewrite_source(source.replace("_136 =", "different ="), "raw_v")


def test_repack_changes_only_explicit_source_member(audit):
    original = io.BytesIO()
    with zipfile.ZipFile(original, "w") as archive:
        archive.writestr("model/code/source.py", b"old")
        archive.writestr("model/data.pkl", b"identity")
        archive.writestr("model/data/0", b"tensor-storage")
        archive.writestr("model/constants/0", b"half-factor")
    changed = audit.repack_source(original.getvalue(), "model/code/source.py", b"new")
    with zipfile.ZipFile(io.BytesIO(changed)) as archive:
        assert archive.read("model/code/source.py") == b"new"
        for name, data in [
            ("model/data.pkl", b"identity"),
            ("model/data/0", b"tensor-storage"),
            ("model/constants/0", b"half-factor"),
        ]:
            assert archive.read(name) == data


def test_unknown_archive_is_rejected_before_loading(audit):
    with pytest.raises(ValueError, match="checkpoint"):
        audit.build_variant(b"not the known V2 checkpoint", "raw_v")


def test_verified_byte_loader_detects_source_only_replacement(audit, tmp_path):
    path = tmp_path / "variant.pt"
    path.write_bytes(b"verified-program")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    verified = audit.verified_bytes(path, digest)
    path.write_bytes(b"same-weights-different-program")
    assert verified == b"verified-program"
    with pytest.raises(ValueError, match="before load"):
        audit.verified_bytes(path, digest)


@pytest.mark.parametrize(
    "failure", [None, "charge", "gauge_energy", "gauge_source", "curvature"]
)
def test_declared_gates_are_applied_to_raw_results(audit, failure):
    protocol = json.loads(
        (DIRECTORY / "mace-ef-v2-trace-input-ablation-prereg-v1.json").read_text()
    )
    conditions, gates = protocol["fixed_conditions"], protocol["gates"]
    sample = {
        "energy_ev": 2.0,
        "source_cartesian": [[1.0, 0.0, 0.0, 0.0]],
        "density": [[1.0, 0.0, 0.0, 0.0]],
    }
    row = {
        "geometry": {"total_charge": 1},
        "samples": {"affine": sample},
        "zero_center_scan": [{"symmetric_eigenvalues": [-0.2, -0.1, -0.05]}],
        "nonzero_center_scan": [{"symmetric_eigenvalues": [-0.2, -0.1, -0.05]}],
    }
    for offset in conditions["constant_offsets_ev_per_e"]:
        shifted = copy.deepcopy(sample)
        shifted["energy_ev"] += offset
        row["samples"][f"gauge_{offset}"] = shifted
    if failure == "charge":
        sample["source_cartesian"][0][0] += 0.001
    elif failure == "gauge_energy":
        row["samples"]["gauge_0.01"]["energy_ev"] += 0.01
    elif failure == "gauge_source":
        row["samples"]["gauge_0.01"]["source_cartesian"][0][1] += 0.001
    elif failure == "curvature":
        row["zero_center_scan"][0]["symmetric_eigenvalues"][2] = 0.02
    result = audit.assess_record(row, conditions, gates)
    json.dumps(result, allow_nan=False)
    assert result["all_executed_checks_passed"] == (failure is None)
    assert result["full_electrostatic_domain_admitted"] is False


def test_centered_scalar_preserves_charge_gauge_without_source_projection():
    torch = pytest.importorskip("torch")
    v = torch.tensor([0.2, -0.1, 0.4], dtype=torch.float64, requires_grad=True)

    def energy(value):
        u = value - value.mean()
        q = torch.softmax(u, 0)  # deliberately charged Q=1
        return u.square().sum() + (q * value).sum()

    torch.testing.assert_close(energy(v + 0.3) - energy(v), v.new_tensor(0.3))
    grad = torch.autograd.grad(energy(v), v)[0]
    torch.testing.assert_close(grad.sum(), v.new_tensor(1.0))


def test_state_digest_detects_weights_and_buffers(audit):
    torch = pytest.importorskip("torch")
    model = torch.nn.Linear(2, 1)
    model.register_buffer("extra", torch.tensor([2.0]))
    before = audit.state_digest(model)
    with torch.no_grad():
        model.extra.add_(1)
    assert before != audit.state_digest(model)
    with torch.no_grad():
        model.extra.sub_(1)
    assert before == audit.state_digest(model)


def test_parity_includes_source_and_density_not_only_energy(audit):
    sample = {
        "energy_ev": 1.0,
        "source_cartesian": np.zeros((2, 4)).tolist(),
        "density": np.zeros((2, 4)).tolist(),
    }
    other = dict(sample, source_cartesian=np.ones((2, 4)).tolist())
    assert audit.sample_differences(sample, other)["source_max_abs"] == 1.0
    assert audit.sample_differences(sample, other)["energy_abs_ev"] == 0.0


def test_real_field_direction_curvature_is_in_dimensionless_ray_units(audit):
    import audit_mace_ef_v2_cosmo_direction as direction

    rows = direction.directional_curvature(
        lambda t: 0.3 - 0.4 * t, 1.0, [0.02, 0.01, 0.005]
    )
    for row in rows:
        assert row["curvature_ev"] == pytest.approx(-0.4)
    with pytest.raises(ValueError):
        direction.directional_curvature(lambda t: t, 0.0, [0.0])


def test_every_direction_difference_state_is_in_the_sample_ledger(audit):
    import audit_mace_ef_v2_cosmo_direction as direction

    centers, steps = [0.0, 1.0], [0.02, 0.01, 0.005]
    ledger = direction.ray_states(centers, steps)
    assert len(ledger) == 14
    for center in centers:
        assert center in ledger
        for step in steps:
            assert center + step in ledger
            assert center - step in ledger


def test_frozen_trace_ablation_gates_recompute(audit):
    path = DIRECTORY / "mace-ef-v2-trace-input-ablation-v1.json"
    result = json.loads(path.read_text())
    protocol_path = DIRECTORY / "mace-ef-v2-trace-input-ablation-prereg-v1.json"
    protocol = json.loads(protocol_path.read_text())
    assert (
        result["preregistration_sha256"]
        == hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    )
    key = "docs/implicit-solvation/benchmarks/audit_mace_ef_v2_trace_inputs.py"
    assert (
        result["source_files_sha256"][key]
        == hashlib.sha256(
            (DIRECTORY / "audit_mace_ef_v2_trace_inputs.py").read_bytes()
        ).hexdigest()
    )
    assert len(result["records"]) == 15
    for variant in result["variants"]:
        assert variant["state_dict_sha256"] == result["state_dict_sha256"]
        assert variant["changed_members"] == (
            [] if variant["mode"] == "roundtrip" else [audit.SOURCE_MEMBER]
        )
    for record in result["records"]:
        assert record["gate_assessment"] == audit.assess_record(
            record, protocol["fixed_conditions"], protocol["gates"]
        )
    assert result["claim_boundary"]["release_admitted"] is False


def test_frozen_cosmo_direction_uses_all_stored_difference_states(audit):
    import audit_mace_ef_v2_cosmo_direction as direction

    path = DIRECTORY / "mace-ef-v2-cosmo-direction-v1.json"
    result = json.loads(path.read_text())
    protocol_path = DIRECTORY / "mace-ef-v2-cosmo-direction-prereg-v1.json"
    protocol = json.loads(protocol_path.read_text())
    assert (
        result["preregistration_sha256"]
        == hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    )
    key = "docs/implicit-solvation/benchmarks/audit_mace_ef_v2_cosmo_direction.py"
    assert (
        result["source_files_sha256"][key]
        == hashlib.sha256(
            (DIRECTORY / "audit_mace_ef_v2_cosmo_direction.py").read_bytes()
        ).hexdigest()
    )
    for record in result["records"]:
        jet = np.asarray(record["reference_reaction_jet"])
        for arm in record["arms"]:
            samples = arm["samples"]
            assert set(samples) == {
                str(t)
                for t in direction.ray_states(
                    protocol["centers_t"], protocol["steps_t"]
                )
            }
            charge_error = max(
                abs(
                    np.asarray(s["source_cartesian"])[:, 0].sum()
                    - record["geometry"]["total_charge"]
                )
                for s in samples.values()
            )
            assert arm["max_scan_state_charge_error_e"] == pytest.approx(
                charge_error, abs=1e-12
            )
            for scan in arm["directional_scans"]:
                for row in scan:
                    plus = direction._directional_derivative(
                        samples, jet, row["center_t"] + row["step_t"]
                    )
                    minus = direction._directional_derivative(
                        samples, jet, row["center_t"] - row["step_t"]
                    )
                    assert row["curvature_ev"] == pytest.approx(
                        (plus - minus) / (2 * row["step_t"]), abs=1e-12
                    )
    assert result["release_admitted"] is False
