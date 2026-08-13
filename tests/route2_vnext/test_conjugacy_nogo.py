from __future__ import annotations

import numpy as np
import pytest

from maple.solvation.release.conjugacy import (
    analyze_mace_polar_conjugacy,
    tolerance_contract,
)


def _embedding() -> np.ndarray:
    result = np.zeros((4, 2))
    result[(0, 2), (0, 1)] = 1.0
    return result


def _inputs(*, missing_gradient: float = 0.0, antisymmetric: float = 0.0):
    atom_count = 2
    component_count = 4
    embedding = _embedding()
    pairing = np.eye(component_count)
    charge_weights = np.asarray([1.0, 1.0, 0.0, 0.0])
    source = np.asarray([[0.3, 0.0, -0.2, 0.0], [-0.3, 0.0, 0.1, 0.0]])
    field = np.asarray([[0.01, -0.02, 0.03, -0.01], [-0.02, 0.01, 0.02, 0.04]])
    gradient = source.copy()
    gradient[0, 3] = missing_gradient
    jacobian = np.zeros((atom_count * component_count,) * 2)
    learned_rows = (0, 2, 4, 6)
    jacobian[learned_rows, learned_rows] = 0.2
    if antisymmetric:
        jacobian[2, 3] += antisymmetric
    return {
        "intrinsic_energy_field_gradient": gradient,
        "original_embedded_source": source,
        "physical_field": field,
        "source_jacobian": jacobian,
        "source_embedding": embedding,
        "pairing_block": pairing,
        "charge_weights": charge_weights,
    }


def test_missing_source_subspace_is_a_decisive_gauge_reduced_no_go_witness():
    result = analyze_mace_polar_conjugacy(**_inputs(missing_gradient=2.0e-4))
    assert result.no_go_witness_detected is True
    assert result.missing_subspace.full_dimension == 4
    assert result.missing_subspace.gauge_reduced_dimension == 3
    assert result.missing_subspace.gauge_reduced.absolute_l2 >= 1.0e-4


def test_zero_missing_projection_does_not_overclaim_global_variationality():
    result = analyze_mace_polar_conjugacy(**_inputs())
    assert result.no_go_witness_detected is False
    assert result.missing_subspace.gauge_reduced.numerically_zero is True
    assert "not a global proof" in result.as_dict()["claim_boundary"]
    direct = {item.sign: item for item in result.direct}
    assert direct[1].gauge_reduced.numerically_zero is True
    assert direct[-1].gauge_reduced.numerically_zero is False


def test_reciprocity_uses_the_raw_reduced_jacobian_without_symmetrization():
    reciprocal = analyze_mace_polar_conjugacy(**_inputs())
    broken = analyze_mace_polar_conjugacy(**_inputs(antisymmetric=0.04))
    assert reciprocal.gauge_reduced_reciprocity.reciprocal is True
    assert broken.gauge_reduced_reciprocity.reciprocal is False
    assert broken.gauge_reduced_reciprocity.absolute_frobenius > 0.0


def test_tolerances_are_frozen():
    contract = tolerance_contract()
    assert contract == {
        "contract_version": "route2-mace-conjugacy-nogo-v1",
        "vector_zero_absolute": 1.0e-10,
        "vector_zero_relative": 1.0e-9,
        "gauge_reduced_reciprocity_relative": 1.0e-9,
        "energy_directional_fd_absolute_eV": 2.0e-7,
        "energy_directional_fd_relative": 2.0e-6,
        "jvp_vjp_dot_absolute": 1.0e-10,
        "jvp_vjp_dot_relative": 1.0e-9,
        "decision": (
            "no-go iff the gauge-reduced projection of the intrinsic-energy "
            "field gradient onto ker(S.T Q) is not numerically zero"
        ),
    }


@pytest.mark.parametrize(
    "name,value",
    (
        ("source_jacobian", np.zeros((7, 7))),
        ("source_embedding", np.zeros((4, 2))),
        ("pairing_block", np.zeros((4, 4))),
        ("charge_weights", np.zeros(4)),
    ),
)
def test_malformed_algebra_fails_closed(name, value):
    inputs = _inputs()
    inputs[name] = value
    with pytest.raises(ValueError):
        analyze_mace_polar_conjugacy(**inputs)


def test_source_and_jacobian_must_obey_the_advertised_embedding_range():
    inputs = _inputs()
    bad_source = inputs["original_embedded_source"].copy()
    bad_source[0, 3] = 1.0e-3
    inputs["original_embedded_source"] = bad_source
    with pytest.raises(ValueError, match="source_embedding"):
        analyze_mace_polar_conjugacy(**inputs)

    inputs = _inputs()
    bad_jacobian = inputs["source_jacobian"].copy()
    bad_jacobian[3, 2] = 1.0e-3
    inputs["source_jacobian"] = bad_jacobian
    with pytest.raises(ValueError, match="source_jacobian"):
        analyze_mace_polar_conjugacy(**inputs)
