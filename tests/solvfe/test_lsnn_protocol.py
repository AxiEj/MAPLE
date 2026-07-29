from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from maple.function.solvfe import (
    LSNNConfigError,
    LSNNMbarResult,
    LSNNMbarRequest,
    LSNNProtocolAdapter,
    LSNNProtocolRequest,
    LSNNProtocolResult,
    LSNNRuntimeMissingError,
    LSNNRuntimeResponse,
    LSNNTiResult,
    load_lsnn_model_card,
)


UPSTREAM_REVISION = "1768d068dcb1ea65e8585af3f4a0cbf4047d9125"
WEIGHT_DICT_SHA = "5b7f9ec224f9264c0220e072ed917201e84e702bab62b331bade37c1ede3a83b"
WEIGHT_PT_SHA = "82e05c684be961eea59c6deb0a8d989ba84809419004ddc16864c3e45fbb8cf5"


def _water_request() -> LSNNProtocolRequest:
    return LSNNProtocolRequest(
        symbols=("O", "H", "H"),
        positions_angstrom=(0.0, 0.0, 0.0, 0.0, 0.75, 0.58, 0.0, -0.75, 0.58),
        lambda_electrostatics=1.0,
        lambda_sterics=1.0,
        solvent="water",
        temperature_kelvin=298.15,
        total_charge=0,
        multiplicity=1,
        progress=0.0,
    )


def _json_card_with_checksum(
    tmp_path: Path,
    weight_name: str,
    dict_sha: str,
    pt_sha: str | None = None,
    audited_domain_enabled: bool = True,
) -> Path:
    payload = {
        "model_id": "lsnn-v1",
        "version": "v1",
        "source_revision": UPSTREAM_REVISION,
        "source_url": "https://arxiv.org/abs/2510.20103",
        "model_checksum_sha256": None,
        "weight_artifacts": [
            {
                "filename": weight_name,
                "source_path": f"Best_Trained_Models/{weight_name}",
                "sha256": dict_sha,
            },
        ],
        "license": "MIT",
        "elements": ["H", "C", "N", "O"],
        "charge_range": "-3 to +3",
        "spin_support": False,
        "audited_domain": {
            "enabled": audited_domain_enabled,
            "elements": ["H", "C", "N", "O"],
            "charge_range": "-1 to +1",
            "multiplicities": [1],
        },
        "solvent": "water",
        "temperature_kelvin": 298.15,
        "energy_reference": "relative",
        "solvent_scope": "water-only",
        "supports_absolute_solvation": False,
        "supports_alchemical_lambda": True,
        "supports_ti": True,
        "supports_mbar": True,
        "supports_water_only": True,
        "scientific_status": "experimental-prototype",
        "default_eligible": False,
    }
    if pt_sha is not None:
        payload["weight_artifacts"].append(
            {
                "filename": f"{weight_name}-companion.pt",
                "source_path": f"Best_Trained_Models/{weight_name}-companion.pt",
                "sha256": pt_sha,
            }
        )
    path = tmp_path / "lsnn-v1.yaml"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _runtime_with_constant_response(request: LSNNProtocolRequest) -> LSNNRuntimeResponse:
    return LSNNRuntimeResponse(
        energy_hartree=1.0 + 0.1 * (request.lambda_electrostatics + request.lambda_sterics),
        dU_dlambda_electrostatics=request.lambda_electrostatics,
        dU_dlambda_sterics=request.lambda_sterics,
    )


def _build_runtime_with_mbar(include_companion: bool = False):
    class _Runtime:
        def __call__(self, request: LSNNProtocolRequest) -> LSNNRuntimeResponse:
            return _runtime_with_constant_response(request)

        def mbar(self, requests):
            diagnostics = {
                "delta_g_hartree": -0.2,
                "overlap": 0.88,
                "effective_sample_size": 32.0,
                "uncertainty": 0.02,
                "uncertainty_units": "hartree",
                "sampling": {
                    "n_frames": sum(req.n_frames for req in requests),
                    "n_states": len(requests),
                },
                "provenance": {
                    "estimator": "mock-mbar",
                    "source": "lsnn-protocol-scaffold",
                },
            }
            if include_companion:
                diagnostics["n_states"] = len(requests)
                diagnostics["effective_frames"] = sum(
                    req.n_frames for req in requests
                )
            return diagnostics

    return _Runtime()


def test_default_card_is_json_and_pinned():
    card = load_lsnn_model_card()
    assert card.source_revision == UPSTREAM_REVISION
    assert card.model_id == "lsnn-v1"
    assert card.license == "MIT"
    assert card.weight_artifacts[0]["filename"] == "280KDATASET2Kv3model.dict"
    assert (
        card.weight_artifacts[0]["source_path"]
        == "Best_Trained_Models/280KDATASET2Kv3model.dict"
    )
    assert card.weight_artifacts[1]["filename"] == "280KDATASET2Kv3.pt"
    assert (
        card.weight_artifacts[1]["source_path"]
        == "Best_Trained_Models/280KDATASET2Kv3.pt"
    )
    assert card.weight_artifacts[0]["sha256"] == WEIGHT_DICT_SHA
    assert card.weight_artifacts[1]["sha256"] == WEIGHT_PT_SHA
    assert card.audited_domain_enabled is False
    assert card.audited_elements == ()
    assert card.audited_charge_range is None
    assert card.audited_multiplicities == ()


def test_water_only_gate_and_lam_contract(tmp_path):
    weight = tmp_path / "weights.ckpt"
    weight.write_text("lsnn")
    card = _json_card_with_checksum(tmp_path, weight.name, hashlib.sha256(weight.read_bytes()).hexdigest())
    adapter = LSNNProtocolAdapter(
        model_path=weight,
        card_path=card,
        runtime=_runtime_with_constant_response,
    )

    request = LSNNProtocolRequest(
        symbols=("H",),
        positions_angstrom=(0.0, 0.0, 0.0),
        lambda_electrostatics=0.2,
        lambda_sterics=0.8,
        solvent="water",
        total_charge=0,
        multiplicity=1,
        progress=0.0,
    )
    result: LSNNProtocolResult = adapter.evaluate(request)
    assert result.lambda_electrostatics == 0.2
    assert result.lambda_sterics == 0.8
    assert result.dU_dlambda == pytest.approx(1.0)
    assert result.dU_dlambda_electrostatics == pytest.approx(0.2)
    assert result.dU_dlambda_sterics == pytest.approx(0.8)

    bad = LSNNProtocolRequest(
        symbols=("H",),
        positions_angstrom=(0.0, 0.0, 0.0),
        lambda_electrostatics=0.2,
        lambda_sterics=0.8,
        solvent="methanol",
        total_charge=0,
        multiplicity=1,
        progress=1.0,
    )
    with pytest.raises(LSNNConfigError, match="water-only"):
        adapter.evaluate(bad)


def test_default_card_execution_is_fail_closed(tmp_path):
    weight = tmp_path / "weights.ckpt"
    weight.write_text("lsnn")
    card = _json_card_with_checksum(
        tmp_path,
        weight.name,
        hashlib.sha256(weight.read_bytes()).hexdigest(),
        audited_domain_enabled=True,
    )
    adapter = LSNNProtocolAdapter(
        model_path=weight,
        card_path=card,
        runtime=_runtime_with_constant_response,
    )
    request = LSNNProtocolRequest(
        symbols=("H",),
        positions_angstrom=(0.0, 0.0, 0.0),
        lambda_electrostatics=0.2,
        lambda_sterics=0.2,
        solvent="water",
        total_charge=10,
        multiplicity=1,
        progress=0.0,
    )
    with pytest.raises(LSNNConfigError, match="audited domain"):
        adapter.evaluate(request)


def test_audited_domain_block_blocks_unsigned_card(tmp_path):
    weight = tmp_path / "weights.ckpt"
    weight.write_text("lsnn")
    card = _json_card_with_checksum(
        tmp_path,
        weight.name,
        hashlib.sha256(weight.read_bytes()).hexdigest(),
        audited_domain_enabled=False,
    )
    adapter = LSNNProtocolAdapter(
        model_path=weight,
        card_path=card,
        runtime=_runtime_with_constant_response,
    )
    request = LSNNProtocolRequest(
        symbols=("H",),
        positions_angstrom=(0.0, 0.0, 0.0),
        lambda_electrostatics=0.2,
        lambda_sterics=0.2,
        solvent="water",
        total_charge=0,
        multiplicity=1,
        progress=0.0,
    )
    with pytest.raises(LSNNConfigError, match="lacks an explicitly audited domain"):
        adapter.evaluate(request)


def test_checksum_gate_is_fail_closed(tmp_path):
    weight = tmp_path / "weights.ckpt"
    weight.write_text("payload")
    real = hashlib.sha256(weight.read_bytes()).hexdigest()
    bad_card = _json_card_with_checksum(tmp_path, weight.name, "0" * 64)
    with pytest.raises(LSNNConfigError, match="checksum mismatch"):
        LSNNProtocolAdapter(model_path=weight, card_path=bad_card, runtime=_runtime_with_constant_response)

    good_card = _json_card_with_checksum(tmp_path, weight.name, real)
    adapter = LSNNProtocolAdapter(model_path=weight, card_path=good_card, runtime=_runtime_with_constant_response)
    assert adapter.model_path == weight


def test_runtime_artifact_must_be_pinned(tmp_path):
    weight = tmp_path / "weights.ckpt"
    weight.write_text("weights")
    card = _json_card_with_checksum(
        tmp_path,
        "other.ckpt",
        hashlib.sha256(weight.read_bytes()).hexdigest(),
    )
    with pytest.raises(LSNNConfigError, match="not a pinned artifact"):
        LSNNProtocolAdapter(model_path=weight, card_path=card, runtime=_runtime_with_constant_response)


def test_runtime_required_for_execution(tmp_path):
    weight = tmp_path / "weights.ckpt"
    weight.write_text("weights")
    card = _json_card_with_checksum(tmp_path, weight.name, hashlib.sha256(weight.read_bytes()).hexdigest())

    with pytest.raises(LSNNRuntimeMissingError, match="runtime"):
        LSNNProtocolAdapter(model_path=weight, card_path=card)


def test_ti_accepts_forward_and_reverse_paths(tmp_path):
    weight = tmp_path / "weights.ckpt"
    weight.write_text("weights")
    card = _json_card_with_checksum(
        tmp_path,
        weight.name,
        hashlib.sha256(weight.read_bytes()).hexdigest(),
    )
    runtime = _build_runtime_with_mbar()
    adapter = LSNNProtocolAdapter(
        model_path=weight,
        card_path=card,
        runtime=runtime,
    )

    forward = (
        LSNNProtocolRequest(
            symbols=("H", "O", "H"),
            positions_angstrom=(0.0, 0.0, 0.0, 0.0, 0.75, 0.58, 0.0, -0.75, 0.58),
            lambda_electrostatics=0.0,
            lambda_sterics=0.0,
            total_charge=0,
            multiplicity=1,
            progress=0.0,
        ),
        LSNNProtocolRequest(
            symbols=("H", "O", "H"),
            positions_angstrom=(0.0, 0.0, 0.0, 0.0, 0.75, 0.58, 0.0, -0.75, 0.58),
            lambda_electrostatics=1.0,
            lambda_sterics=1.0,
            total_charge=0,
            multiplicity=1,
            progress=1.0,
        ),
    )
    reverse = (
        LSNNProtocolRequest(
            symbols=("H", "O", "H"),
            positions_angstrom=(0.0, 0.0, 0.0, 0.0, 0.75, 0.58, 0.0, -0.75, 0.58),
            lambda_electrostatics=1.0,
            lambda_sterics=1.0,
            total_charge=0,
            multiplicity=1,
            progress=1.0,
        ),
        LSNNProtocolRequest(
            symbols=("H", "O", "H"),
            positions_angstrom=(0.0, 0.0, 0.0, 0.0, 0.75, 0.58, 0.0, -0.75, 0.58),
            lambda_electrostatics=0.0,
            lambda_sterics=0.0,
            total_charge=0,
            multiplicity=1,
            progress=0.0,
        ),
    )

    ti = adapter.evaluate_ti(forward, reverse_states=reverse)
    assert isinstance(ti, LSNNTiResult)
    assert ti.protocol_status == "scaffold"
    assert ti.protocol_type == "single-geometry-ti"
    assert ti.points == 2
    assert ti.lambda_start == 0.0
    assert ti.lambda_end == 1.0
    assert ti.reverse_delta_g_hartree == pytest.approx(-1.0)
    assert ti.delta_g_hartree == pytest.approx(1.0)
    assert ti.hysteresis_hartree == pytest.approx(0.0)

    ti_forward_only = adapter.evaluate_ti(forward)
    assert ti_forward_only.hysteresis_hartree is None


def test_ti_validates_reverse_ordering(tmp_path):
    weight = tmp_path / "weights.ckpt"
    weight.write_text("weights")
    card = _json_card_with_checksum(tmp_path, weight.name, hashlib.sha256(weight.read_bytes()).hexdigest())

    adapter = LSNNProtocolAdapter(
        model_path=weight,
        card_path=card,
        runtime=_build_runtime_with_mbar(),
    )

    forward = (
        LSNNProtocolRequest(
            symbols=("H",),
            positions_angstrom=(0.0, 0.0, 0.0),
            lambda_electrostatics=0.0,
            lambda_sterics=0.0,
            total_charge=0,
            multiplicity=1,
            progress=0.0,
        ),
        LSNNProtocolRequest(
            symbols=("H",),
            positions_angstrom=(0.0, 0.0, 0.0),
            lambda_electrostatics=0.5,
            lambda_sterics=0.5,
            total_charge=0,
            multiplicity=1,
            progress=0.5,
        ),
        LSNNProtocolRequest(
            symbols=("H",),
            positions_angstrom=(0.0, 0.0, 0.0),
            lambda_electrostatics=1.0,
            lambda_sterics=1.0,
            total_charge=0,
            multiplicity=1,
            progress=1.0,
        ),
    )
    invalid_reverse = (
        forward[-1],
        forward[1],
        LSNNProtocolRequest(
            symbols=("H",),
            positions_angstrom=(0.0, 0.0, 0.0),
            lambda_electrostatics=0.75,
            lambda_sterics=0.75,
            total_charge=0,
            multiplicity=1,
            progress=0.75,
        ),
        forward[0],
    )
    with pytest.raises(LSNNConfigError, match="reverse states"):
        adapter.evaluate_ti(forward, reverse_states=invalid_reverse)

    wrong_endpoints = (
        forward[-1],
        LSNNProtocolRequest(
            symbols=("H",),
            positions_angstrom=(0.0, 0.0, 0.0),
            lambda_electrostatics=0.1,
            lambda_sterics=0.0,
            total_charge=0,
            multiplicity=1,
            progress=0.0,
        ),
    )
    with pytest.raises(LSNNConfigError, match="endpoints"):
        adapter.evaluate_ti(forward, reverse_states=wrong_endpoints)


def test_mbar_collects_payload_and_diagnostics(tmp_path):
    weight = tmp_path / "weights.ckpt"
    weight.write_text("weights")
    card = _json_card_with_checksum(
        tmp_path,
        weight.name,
        hashlib.sha256(weight.read_bytes()).hexdigest(),
        WEIGHT_PT_SHA,
    )

    runtime = _build_runtime_with_mbar(include_companion=True)
    adapter = LSNNProtocolAdapter(
        model_path=weight,
        card_path=card,
        runtime=runtime,
    )

    mbar = adapter.evaluate_mbar(
        (
            LSNNMbarRequest(
                request=LSNNProtocolRequest(
                    symbols=("H", "O", "H"),
                    positions_angstrom=(0.0, 0.0, 0.0, 0.0, 0.75, 0.58, 0.0, -0.75, 0.58),
                    lambda_electrostatics=0.0,
                    lambda_sterics=0.0,
                    total_charge=0,
                    multiplicity=1,
                    progress=0.0,
                ),
                n_frames=4,
            ),
            LSNNMbarRequest(
                request=LSNNProtocolRequest(
                    symbols=("H", "O", "H"),
                    positions_angstrom=(0.0, 0.0, 0.0, 0.0, 0.75, 0.58, 0.0, -0.75, 0.58),
                    lambda_electrostatics=1.0,
                    lambda_sterics=1.0,
                    total_charge=0,
                    multiplicity=1,
                    progress=1.0,
                ),
                n_frames=4,
            ),
        )
    )
    assert isinstance(mbar, LSNNMbarResult)
    assert mbar.points == 2
    assert mbar.protocol_status == "scaffold"
    assert mbar.protocol_type == "scaffold-mbar"
    assert mbar.overlap == pytest.approx(0.88)
    assert mbar.effective_sample_size == pytest.approx(32.0)
    assert mbar.uncertainty == pytest.approx(0.02)
    assert mbar.uncertainty_units == "hartree"
    assert mbar.sampling["n_states"] == 2
    assert mbar.sampling["n_frames"] == 8
    assert mbar.provenance["estimator"] == "mock-mbar"
    assert mbar.diagnostics is not None
    assert mbar.diagnostics["n_states"] == 2
    assert mbar.diagnostics["effective_frames"] == 8
