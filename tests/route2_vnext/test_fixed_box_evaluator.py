from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from maple.function.calculator.mace._macepol_long_range import (
    MACEPolarLongRangeEvaluator,
)
from maple.function.route2_smd_profiles import (
    MACEPOL_FORCED_RECIPROCAL_FIXED_BOX_PROFILES,
)
from maple.solvation.models.fixed_box_evaluator import (
    MACEPolarFixedBoxDiagnosticEvaluator,
)


def _batch(positions):
    positions = torch.as_tensor(positions, dtype=torch.float64)
    return {
        "positions": positions,
        "batch": torch.zeros(len(positions), dtype=torch.long),
        "cell": torch.zeros((3, 3), dtype=torch.float64),
        "rcell": torch.zeros((3, 3), dtype=torch.float64),
        "volume": torch.zeros(1, dtype=torch.float64),
        "pbc": torch.zeros((1, 3), dtype=torch.bool),
    }


@pytest.mark.parametrize("box_length", (32, 48, 56))
def test_preregistered_non40_boxes_have_distinct_immutable_geometry(box_length):
    profile = MACEPOL_FORCED_RECIPROCAL_FIXED_BOX_PROFILES[box_length]
    evaluator = MACEPolarFixedBoxDiagnosticEvaluator.from_profile(profile)
    prepared = evaluator.prepare_batch(
        _batch([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]), r_max=5.0
    )
    assert evaluator.box_length_angstrom == float(box_length)
    torch.testing.assert_close(
        prepared["cell"], box_length * torch.eye(3, dtype=torch.float64)
    )
    torch.testing.assert_close(
        prepared["volume"],
        torch.tensor([float(box_length**3)], dtype=torch.float64),
    )
    assert evaluator.provenance["profile"] == profile
    assert evaluator.provenance["box_length_angstrom"] == float(box_length)
    assert evaluator.provenance["implementation_scope"].startswith("vNext disabled")


def test_historical_evaluator_does_not_silently_accept_new_box_profiles():
    with pytest.raises(ValueError, match="Unsupported MACE-POLAR"):
        MACEPolarLongRangeEvaluator.from_profile(
            MACEPOL_FORCED_RECIPROCAL_FIXED_BOX_PROFILES[48]
        )


def test_diagnostic_evaluator_rejects_legacy40_and_inconsistent_construction():
    profile = MACEPOL_FORCED_RECIPROCAL_FIXED_BOX_PROFILES[48]
    with pytest.raises(ValueError, match="Unsupported vNext"):
        MACEPolarFixedBoxDiagnosticEvaluator.from_profile(
            MACEPOL_FORCED_RECIPROCAL_FIXED_BOX_PROFILES[40]
        )
    with pytest.raises(ValueError, match="inconsistent"):
        MACEPolarFixedBoxDiagnosticEvaluator(profile, True, 40.0)
