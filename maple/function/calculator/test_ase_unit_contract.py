import numpy as np
import pytest

from maple.function.calculator._ase_unit_contract import (
    ase_properties_for_maple_request,
    copy_ase_results_to_maple_units,
)


def test_official_ase_property_request_is_minimal_and_stable():
    assert ase_properties_for_maple_request(None) == ["energy"]
    assert ase_properties_for_maple_request(["forces", "stress"]) == ["energy", "forces", "stress"]


def test_official_ase_results_follow_maple_unit_contract():
    ev_per_hartree = 27.211386245988
    target = {}

    copy_ase_results_to_maple_units(
        {
            "energy": ev_per_hartree,
            "free_energy": 2.0 * ev_per_hartree,
            "forces": np.ones((1, 3)) * ev_per_hartree,
            "stress": np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3]),
        },
        target,
    )

    assert target["energy"] == pytest.approx(1.0)
    assert target["free_energy"] == pytest.approx(2.0)
    assert np.allclose(target["forces"], np.ones((1, 3)))
    assert np.allclose(target["stress"], [1.0, 2.0, 3.0, 0.1, 0.2, 0.3])
