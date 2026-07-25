from __future__ import annotations

import pytest
from ase import Atoms

from maple.function.calculator.extra_correction.implicit.outer_base import (
    OuterDeltaResult,
    SMDPolarDeltaProvider,
    atom_list_sha256,
)


def _backend(_atoms):
    return {
        "units": "hartree",
        "solute_polarization": 0.001,
        "pcm_polarization": -0.010,
        "electrostatic": -0.009,
        "cds": 0.004,
        "standard_state": 0.0,
        "delta_g_solv": -0.005,
        "warnings": [],
        "provenance": {
            "source_commit": "7fea1061e9def05914ebf82c2e2eaba8a8f54365",
            "source_blob_sha256": "a" * 64,
            "model_sha256": "b" * 64,
            "pcmsolver_library_sha256": "c" * 64,
            "profile": "smd-iefpcm-gaff2-o",
            "cavity_policy": "fixed-stability-branch",
        },
    }


def test_outer_adapter_validates_route2_components_and_atom_identity():
    atoms = Atoms("OH2", positions=[[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]])
    provider = SMDPolarDeltaProvider(_backend)
    result = provider.evaluate_delta(atoms)

    assert isinstance(result, OuterDeltaResult)
    assert result.atom_list_hash == atom_list_sha256(atoms)
    assert result.delta_g_solv_hartree == pytest.approx(-0.005)
    assert result.electrostatic_hartree == pytest.approx(-0.009)
    assert result.whole_cluster is True
    assert result.force_support is False


def test_outer_adapter_batches_without_weakening_frame_validation():
    atoms = [
        Atoms("He", positions=[[0.0, 0.0, float(index)]])
        for index in range(2)
    ]

    class BatchBackend:
        def __init__(self):
            self.calls = 0

        def __call__(self, _atoms):
            raise AssertionError("single-frame path must not be used")

        def evaluate_many(self, frames):
            self.calls += 1
            assert len(frames) == 2
            return [_backend(frame) for frame in frames]

    backend = BatchBackend()
    results = SMDPolarDeltaProvider(backend).evaluate_many(atoms)

    assert backend.calls == 1
    assert len(results) == 2
    assert results[0].atom_list_hash == results[1].atom_list_hash
    assert all(result.delta_g_solv_hartree == -0.005 for result in results)


def test_outer_adapter_batch_rejects_wrong_frame_count():
    atoms = [Atoms("He"), Atoms("He")]

    class ShortBackend:
        def __call__(self, frame):
            return _backend(frame)

        def evaluate_many(self, frames):
            return [_backend(frames[0])]

    with pytest.raises(ValueError, match="wrong frame count"):
        SMDPolarDeltaProvider(ShortBackend()).evaluate_many(atoms)


@pytest.mark.parametrize(
    "mutation",
    [
        {"warnings": ["PEDRA warning"]},
        {"electrostatic": -0.008},
        {"delta_g_solv": -0.004},
        {"units": "eV"},
    ],
)
def test_outer_adapter_fails_closed_on_warning_identity_or_unit_mismatch(
    mutation,
):
    atoms = Atoms("He")

    def bad_backend(value):
        result = _backend(value)
        result.update(mutation)
        return result

    with pytest.raises(ValueError):
        SMDPolarDeltaProvider(bad_backend).evaluate_delta(atoms)
