from __future__ import annotations

import numpy as np

from maple.function.calculator.extra_correction.implicit.route2_fixed_point import (
    DAMPED_PICARD_SOLVER,
    SAFEGUARDED_ANDERSON_SOLVER,
    FixedPointSample,
    next_fixed_point_density,
)


def _iterate_linear_map(
    matrix: np.ndarray,
    offset: np.ndarray,
    *,
    solver: str,
    maximum_iterations: int,
) -> tuple[np.ndarray, float, list[str]]:
    density = np.zeros_like(offset)
    samples: list[FixedPointSample] = []
    methods: list[str] = []
    residual_inf = float("inf")
    for _iteration in range(maximum_iterations):
        response = matrix @ density + offset
        residual = response - density
        residual_inf = float(np.max(np.abs(residual)))
        if residual_inf <= 1.0e-12:
            break
        samples.append(
            FixedPointSample(
                density=density.reshape(1, 4),
                residual=residual.reshape(1, 4),
            )
        )
        step = next_fixed_point_density(
            samples,
            solver=solver,
            mixing=1.0,
            anderson_depth=4,
            anderson_regularization=1.0e-12,
            anderson_coefficient_l1_limit=100.0,
            anderson_step_ratio_limit=100.0,
        )
        density = step.density.reshape(4)
        methods.append(step.method)
    return density, residual_inf, methods


def test_safeguarded_anderson_converges_a_slow_contracting_map():
    matrix = np.diag([0.0, 0.9, 0.8, 0.7])
    offset = np.asarray([0.0, 0.2, -0.1, 0.3])

    _picard_density, picard_residual, _picard_methods = _iterate_linear_map(
        matrix,
        offset,
        solver=DAMPED_PICARD_SOLVER,
        maximum_iterations=100,
    )
    anderson_density, anderson_residual, anderson_methods = _iterate_linear_map(
        matrix,
        offset,
        solver=SAFEGUARDED_ANDERSON_SOLVER,
        maximum_iterations=30,
    )

    assert picard_residual > 1.0e-12
    assert anderson_residual <= 1.0e-12
    assert SAFEGUARDED_ANDERSON_SOLVER in anderson_methods
    np.testing.assert_allclose(
        anderson_density,
        np.linalg.solve(np.eye(4) - matrix, offset),
        rtol=0.0,
        atol=1.0e-10,
    )


def test_anderson_step_preserves_the_affine_total_charge_constraint():
    samples = [
        FixedPointSample(
            density=np.asarray([[-0.3, 0.1, 0.0, 0.0], [0.3, 0.0, 0.2, 0.0]]),
            residual=np.asarray(
                [[0.02, -0.01, 0.00, 0.01], [-0.02, 0.01, 0.00, -0.01]]
            ),
        ),
        FixedPointSample(
            density=np.asarray([[-0.28, 0.09, 0.0, 0.01], [0.28, 0.01, 0.2, -0.01]]),
            residual=np.asarray(
                [[0.01, -0.005, 0.00, 0.005], [-0.01, 0.005, 0.00, -0.005]]
            ),
        ),
    ]

    step = next_fixed_point_density(
        samples,
        solver=SAFEGUARDED_ANDERSON_SOLVER,
        mixing=1.0,
        anderson_depth=4,
        anderson_regularization=1.0e-12,
        anderson_coefficient_l1_limit=100.0,
        anderson_step_ratio_limit=100.0,
    )

    assert step.method == SAFEGUARDED_ANDERSON_SOLVER
    assert float(np.sum(step.density[:, 0])) == 0.0


def test_unsafe_anderson_coefficients_fall_back_to_picard():
    samples = [
        FixedPointSample(
            density=np.zeros((1, 4)),
            residual=np.asarray([[1.0, 0.0, 0.0, 0.0]]),
        ),
        FixedPointSample(
            density=np.asarray([[1.0, 0.0, 0.0, 0.0]]),
            residual=np.asarray([[0.9, 0.0, 0.0, 0.0]]),
        ),
    ]

    step = next_fixed_point_density(
        samples,
        solver=SAFEGUARDED_ANDERSON_SOLVER,
        mixing=0.5,
        anderson_depth=4,
        anderson_regularization=1.0e-12,
        anderson_coefficient_l1_limit=1.0e-6,
        anderson_step_ratio_limit=20.0,
    )

    assert step.method == DAMPED_PICARD_SOLVER
    assert step.fallback_reason == "anderson-coefficient-limit"
    np.testing.assert_allclose(
        step.density,
        samples[-1].density + 0.5 * samples[-1].residual,
    )
