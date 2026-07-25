from __future__ import annotations

from types import SimpleNamespace

import pytest
from ase import Atoms


torch = pytest.importorskip("torch")

from maple.function.calculator.mace._macepol_calculator import MACEPolCalculator
from maple.function.calculator.mace._macepol_long_range import (
    MACEPolarLongRangeEvaluator,
)
from maple.function.route2_smd_profiles import (
    DDPCM_GAFF2_CARBONYL_O_MACE_KSPACE40_PROFILE,
    MACEPOL_FORCED_RECIPROCAL_FIXED_BOX40_PROFILE,
    MACEPOL_MOLECULAR_REALSPACE_PROFILE,
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


class _Projection(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer(
            "matrix",
            torch.eye(4, dtype=torch.float64),
        )
        self.projections_dim = 4

    def forward(self, *, batch, positions, node_fields):
        del batch, positions
        assert node_fields.dtype == self.matrix.dtype
        return node_fields


def _compatible_model(projection):
    return SimpleNamespace(
        electric_potential_descriptor=SimpleNamespace(
            non_periodic_correction_terms=SimpleNamespace(
                displaced_interactions=projection,
            )
        )
    )


class _FieldRecorder:
    def __init__(self):
        self.values = None

    def set_node_potential_gradient(self, values):
        self.values = values


class _ForwardRecorder:
    def __init__(self):
        self.use_pbc_evaluator = None

    def __call__(
        self,
        _batch,
        *,
        compute_force,
        compute_stress,
        compute_hessian,
        use_pbc_evaluator,
    ):
        assert compute_force is False
        assert compute_stress is False
        assert compute_hessian is False
        self.use_pbc_evaluator = use_pbc_evaluator
        return {"energy": torch.zeros(1)}


def test_default_evaluator_is_an_exact_no_op():
    evaluator = MACEPolarLongRangeEvaluator.from_profile(
        MACEPOL_MOLECULAR_REALSPACE_PROFILE
    )
    batch = _batch([[0.0, 0.0, 0.0]])
    projection = _Projection()
    model = _compatible_model(projection)

    evaluator.configure_model(model)

    assert evaluator.prepare_batch(batch, r_max=5.0) is batch
    assert evaluator.model_forward_kwargs == {}
    assert (
        model.electric_potential_descriptor.non_periodic_correction_terms
        .displaced_interactions
        is projection
    )


def test_route2_profile_dispatches_one_closed_evaluator_contract():
    default_kwargs = MACEPolCalculator.build_implicit_solvent_kwargs(
        {
            "provider": "pyddx",
            "profile": "smd-ddpcm-l15-n1202-v1",
        }
    )
    reciprocal_kwargs = MACEPolCalculator.build_implicit_solvent_kwargs(
        {
            "provider": "pyddx",
            "profile": DDPCM_GAFF2_CARBONYL_O_MACE_KSPACE40_PROFILE,
        }
    )

    assert default_kwargs == {
        "long_range_evaluator_profile": (
            MACEPOL_MOLECULAR_REALSPACE_PROFILE
        )
    }
    assert reciprocal_kwargs == {
        "long_range_evaluator_profile": (
            MACEPOL_FORCED_RECIPROCAL_FIXED_BOX40_PROFILE
        )
    }


def test_evaluator_rejects_inconsistent_direct_construction():
    with pytest.raises(ValueError, match="inconsistent"):
        MACEPolarLongRangeEvaluator(
            profile=MACEPOL_MOLECULAR_REALSPACE_PROFILE,
            use_pbc_evaluator=True,
            box_length_angstrom=40.0,
        )


def test_forced_reciprocal_evaluator_centres_one_graph_in_fixed_40a_box():
    evaluator = MACEPolarLongRangeEvaluator.from_profile(
        MACEPOL_FORCED_RECIPROCAL_FIXED_BOX40_PROFILE
    )
    base = _batch(
        [[-1.0, 0.5, 0.0], [1.0, -0.5, 0.25]]
    )
    translated = _batch(
        [[8.0, -3.5, 2.0], [10.0, -4.5, 2.25]]
    )

    base_prepared = evaluator.prepare_batch(base, r_max=5.0)
    translated_prepared = evaluator.prepare_batch(
        translated,
        r_max=5.0,
    )

    torch.testing.assert_close(
        base_prepared["positions"],
        translated_prepared["positions"],
    )
    torch.testing.assert_close(
        torch.mean(base_prepared["positions"], dim=0),
        torch.zeros(3, dtype=torch.float64),
    )
    torch.testing.assert_close(
        base_prepared["cell"],
        40.0 * torch.eye(3, dtype=torch.float64),
    )
    torch.testing.assert_close(
        base_prepared["volume"],
        torch.tensor([64000.0], dtype=torch.float64),
    )
    assert evaluator.model_forward_kwargs == {
        "use_pbc_evaluator": True
    }


def test_forced_reciprocal_evaluator_fails_closed_outside_fixed_box_domain():
    evaluator = MACEPolarLongRangeEvaluator.from_profile(
        MACEPOL_FORCED_RECIPROCAL_FIXED_BOX40_PROFILE
    )
    with pytest.raises(RuntimeError, match="40.*fixed box"):
        evaluator.prepare_batch(
            _batch([[-16.0, 0.0, 0.0], [16.0, 0.0, 0.0]]),
            r_max=5.0,
        )

    multi_graph = _batch([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    multi_graph["batch"][1] = 1
    with pytest.raises(RuntimeError, match="one molecular graph"):
        evaluator.prepare_batch(multi_graph, r_max=5.0)

    periodic = _batch([[0.0, 0.0, 0.0]])
    periodic["pbc"][0, 0] = True
    with pytest.raises(RuntimeError, match="non-periodic FFF"):
        evaluator.prepare_batch(periodic, r_max=5.0)


def test_forced_reciprocal_evaluator_installs_only_the_dtype_bridge():
    evaluator = MACEPolarLongRangeEvaluator.from_profile(
        MACEPOL_FORCED_RECIPROCAL_FIXED_BOX40_PROFILE
    )
    projection = _Projection()
    model = _compatible_model(projection)

    evaluator.configure_model(model)
    installed = (
        model.electric_potential_descriptor.non_periodic_correction_terms
        .displaced_interactions
    )
    result = installed(
        batch=torch.zeros(1, dtype=torch.long),
        positions=torch.zeros((1, 3), dtype=torch.float64),
        node_fields=torch.ones((1, 4), dtype=torch.float32),
    )

    assert installed is not projection
    assert installed.upstream is projection
    assert installed.projections_dim == projection.projections_dim
    assert result.dtype == torch.float64
    assert evaluator.provenance["dtype_bridge"] is True
    assert evaluator.provenance["equivalent_to_default_evaluator"] is False


def test_macepol_central_forward_dispatches_the_selected_evaluator():
    calculator = object.__new__(MACEPolCalculator)
    calculator.device = torch.device("cpu")
    calculator.dtype = torch.float64
    calculator._reaction_projector = _FieldRecorder()
    calculator._long_range_evaluator = (
        MACEPolarLongRangeEvaluator.from_profile(
            MACEPOL_FORCED_RECIPROCAL_FIXED_BOX40_PROFILE
        )
    )
    calculator.model = _ForwardRecorder()
    calculator._batch_dict = lambda _atoms: {
        "positions": torch.zeros((1, 3), dtype=torch.float64)
    }

    calculator.polar_output_torch(Atoms("H"))

    assert calculator.model.use_pbc_evaluator is True
