from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from maple.function.calculator.extra_correction.implicit.route2_root_study import (
    NOMINAL_ROUTE2_SCF_REASON,
    Route2MultiStartRootPolicy,
    evaluate_route2_multi_start_root_agreement,
)


@dataclass(frozen=True)
class _State:
    atomic_numbers: np.ndarray
    positions_angstrom: np.ndarray
    density_coefficients: np.ndarray
    initial_density_label: str
    initial_density_sha256: str
    history: tuple[object, ...]
    scf_convergence: dict[str, object]


def _state(
    *,
    seed: str,
    density_shift: float = 0.0,
    positions: np.ndarray | None = None,
    nominal: bool = True,
) -> _State:
    density = np.asarray(
        [[-0.2, 0.1, -0.3, 0.4], [0.2, -0.5, 0.6, -0.7]],
        dtype=float,
    )
    density[0, 1] += density_shift
    return _State(
        atomic_numbers=np.asarray([6, 8]),
        positions_angstrom=(
            np.asarray([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]])
            if positions is None
            else np.asarray(positions, dtype=float)
        ),
        density_coefficients=density,
        initial_density_label=f"seed-{seed}",
        initial_density_sha256=seed * 64,
        history=(object(), object()),
        scf_convergence={
            "reason": NOMINAL_ROUTE2_SCF_REASON if nominal else "not-nominal"
        },
    )


def test_multistart_root_agreement_requires_three_distinct_nominal_seeds():
    states = {
        "gas": _state(seed="a"),
        "zero": _state(seed="b"),
        "warm": _state(seed="c"),
    }
    agreement = evaluate_route2_multi_start_root_agreement(
        states,
        declared_scalar_energies_ev={"gas": -0.2, "zero": -0.2 + 3e-12, "warm": -0.2},
    )

    assert agreement.agreed is True
    assert agreement.failure_reasons == ()
    assert agreement.maximum_pairwise_density_difference == pytest.approx(0.0)
    assert agreement.maximum_pairwise_energy_difference_ev == pytest.approx(3e-12)
    assert [member.initial_density_label for member in agreement.members] == [
        "seed-a",
        "seed-b",
        "seed-c",
    ]


def test_multistart_root_agreement_rejects_same_seed_and_root_disagreement():
    agreement = evaluate_route2_multi_start_root_agreement(
        {
            "gas": _state(seed="a"),
            "copy": _state(seed="a", density_shift=2e-8),
            "zero": _state(seed="b", nominal=False),
        },
        declared_scalar_energies_ev={"gas": -0.2, "copy": -0.2 + 3e-9, "zero": -0.2},
    )

    assert agreement.agreed is False
    assert set(agreement.failure_reasons) == {
        "multi-start-seeds-are-not-distinct",
        "zero:root-is-not-nominal",
        "multi-start-root-density-disagreement",
        "multi-start-root-energy-disagreement",
    }


def test_multistart_root_agreement_rejects_missing_start_and_geometry_drift():
    agreement = evaluate_route2_multi_start_root_agreement(
        {
            "gas": _state(seed="a"),
            "zero": _state(
                seed="b",
                positions=np.asarray([[0.0, 0.0, 0.0], [1.21, 0.0, 0.0]]),
            ),
        },
        declared_scalar_energies_ev={"gas": -0.2, "zero": -0.2},
        policy=Route2MultiStartRootPolicy(required_start_count=3),
    )

    assert agreement.agreed is False
    assert set(agreement.failure_reasons) == {
        "multi-start-states-do-not-share-one-geometry",
        "insufficient-independent-root-starts",
    }


def test_multistart_root_agreement_requires_energy_for_every_state():
    with pytest.raises(ValueError, match="states and declared scalar energies differ"):
        evaluate_route2_multi_start_root_agreement(
            {"gas": _state(seed="a"), "zero": _state(seed="b")},
            declared_scalar_energies_ev={"gas": -0.2},
        )
