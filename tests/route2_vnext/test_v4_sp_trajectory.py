from __future__ import annotations

from dataclasses import dataclass, FrozenInstanceError
import json
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from maple.function.calculator.calculator_base import EV2HARTREE
from maple.function.dispatcher.sp.sp import SinglePoint
from maple.function.dispatcher.sp.result import SPBatchResult, SPFrameResult
from maple.function.dispatcher.dispatcher import Dispatcher
from maple.function.read.command_control import CommandControl
from maple.function.read.input_reader import InputReader
from maple.function.route2_smd_profiles import (
    PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_TORCH_CPU_V4_PROFILE,
    route2_smd_profile_spec,
    route2_workflow_capability_for_params,
)
from maple.solvation.models.base import model_input_sha256


class _CanonicalFrameCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(
        self,
        *,
        configuration: str = "a" * 64,
        profile: str = "v4-test-profile",
        scalar: str = "v4-test-scalar",
        device: str = "cpu",
        domain: str = "neutral-hydrogen-v1",
    ) -> None:
        super().__init__()
        self.configuration = configuration
        self.profile_spec: Any = SimpleNamespace(
            name=profile,
            scalar_contract_id=scalar,
        )
        self.device = device
        self.domain_identity = domain
        self.is_v4 = False
        self.calculate_calls = 0
        self.requested_properties: list[tuple[str, ...]] = []

    def _validate_atoms(self, atoms):
        return (
            model_input_sha256(atoms),
            self.configuration,
            self.profile_spec.name,
            self.profile_spec.scalar_contract_id,
            self.device,
        )

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        assert atoms is not None
        super().calculate(atoms, properties, system_changes)
        requested = tuple(properties or ())
        self.calculate_calls += 1
        self.requested_properties.append(requested)
        x = float(atoms.positions[0, 0])
        self.results = {"energy": x * x + 1.0}
        if "forces" in requested:
            self.results["forces"] = np.array(  # type: ignore[assignment]
                [[-2.0 * x, 0.0, 0.0]]
            )


@dataclass
class _Control:
    params: dict[str, Any]


def _frame(x: float, calculator: Calculator, *, charge=0, mult=1) -> Atoms:
    atoms = Atoms("H", positions=[[x, 0.0, 0.0]])
    atoms.info.update(charge=charge, mult=mult, spin=(mult - 1) / 2)
    atoms.calc = calculator
    return atoms


def _official_v4_calculator() -> _CanonicalFrameCalculator:
    calculator = _CanonicalFrameCalculator()
    calculator.profile_spec = route2_smd_profile_spec(
        PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_TORCH_CPU_V4_PROFILE
    )
    return calculator


@pytest.fixture
def direct_v4(monkeypatch):
    monkeypatch.setattr(
        "maple.function.dispatcher.pure_nonmd_status.is_pure_torch_v4",
        lambda atoms: True,
    )


def test_direct_a_b_a_keeps_independent_raw_e_v_and_f_records(direct_v4, tmp_path):
    calculator = _CanonicalFrameCalculator()
    frames = [_frame(x, calculator) for x in (0.0, 0.5, 0.0)]

    result = SinglePoint(
        str(tmp_path / "aba.out"),
        frames,
        {"sp": {"mode": "trajectory", "verbose": 1}},
    ).run()

    assert isinstance(result, SPBatchResult)
    assert [frame.frame_index for frame in result.frames] == [0, 1, 2]
    assert [frame.energy_eV for frame in result.frames] == [1.0, 1.25, 1.0]
    first_record = result.frames[0].to_record()
    repeated_record = result.frames[2].to_record()
    first_record.pop("frame_index")
    repeated_record.pop("frame_index")
    assert first_record == repeated_record
    assert result.frames[0].forces_eV_per_A == ((-0.0, 0.0, 0.0),)
    assert result.frames[1].forces_eV_per_A == ((-1.0, 0.0, 0.0),)
    assert all(
        frame.force_scope == "raw_cartesian_total_pes_force_negative_energy_gradient"
        for frame in result.frames
    )
    assert calculator.calculate_calls == 3
    assert calculator.requested_properties == [("forces",)] * 3
    assert result.frames[0].symbols == ("H",)
    assert result.frames[0].positions_angstrom == ((0.0, 0.0, 0.0),)
    assert result.frames[1].positions_angstrom == ((0.5, 0.0, 0.0),)
    assert result.frames[0].masses_amu == result.frames[1].masses_amu
    assert result.frames[0].charge == 0
    assert result.frames[0].multiplicity == 1
    assert result.frames[0].configuration_sha256 == "a" * 64
    assert result.frames[0].profile == "v4-test-profile"
    assert result.frames[0].scalar_contract_id == "v4-test-scalar"
    assert result.frames[0].actual_device == "cpu"
    assert result.frames[0].model_input_sha256 == result.frames[2].model_input_sha256
    assert result.frames[0].model_input_sha256 != result.frames[1].model_input_sha256
    with pytest.raises(FrozenInstanceError):
        setattr(result.frames[0], "energy_eV", 7.0)
    assert isinstance(result.frames[0].positions_angstrom, tuple)
    assert isinstance(result.frames[0].positions_angstrom[0], tuple)


@pytest.mark.parametrize(
    "mismatch",
    [
        "symbols",
        "masses",
        "charge",
        "multiplicity",
        "configuration",
        "profile",
        "scalar",
        "device",
        "domain",
    ],
)
def test_direct_preflight_rejects_every_identity_mismatch_before_evaluation(
    direct_v4, tmp_path, mismatch
):
    first_calculator = _CanonicalFrameCalculator()
    second_kwargs = {}
    attr_values = {
        "configuration": "b" * 64,
        "profile": "other-profile",
        "scalar": "other-scalar",
        "device": "cuda:0",
        "domain": "other-domain",
    }
    if mismatch in attr_values:
        second_kwargs[mismatch] = attr_values[mismatch]
    second_calculator = _CanonicalFrameCalculator(**second_kwargs)
    first = _frame(0.0, first_calculator)
    second = _frame(
        0.5,
        second_calculator,
        charge=1 if mismatch == "charge" else 0,
        mult=2 if mismatch == "multiplicity" else 1,
    )
    if mismatch == "symbols":
        second = Atoms("He", positions=second.positions, info=second.info)
        second.calc = second_calculator
    if mismatch == "masses":
        second.arrays["masses"] = np.array([2.0])

    with pytest.raises(ValueError, match="identity|share|same"):
        SinglePoint(
            str(tmp_path / f"bad-{mismatch}.out"),
            [first, second],
            {"sp": {"mode": "trajectory", "verbose": 1}},
        ).run()

    assert first_calculator.calculate_calls == 0
    assert second_calculator.calculate_calls == 0


def test_direct_api_rejects_mixed_v4_and_non_v4_before_evaluation(
    monkeypatch, tmp_path
):
    v4_calculator = _CanonicalFrameCalculator()
    legacy_calculator = _CanonicalFrameCalculator()
    v4_calculator.is_v4 = True
    legacy_calculator.is_v4 = False
    first = _frame(0.0, v4_calculator)
    second = _frame(0.5, legacy_calculator)
    monkeypatch.setattr(
        "maple.function.dispatcher.pure_nonmd_status.is_pure_torch_v4",
        lambda atoms: bool(getattr(atoms.calc, "is_v4", False)),
    )

    with pytest.raises(ValueError, match="every frame"):
        SinglePoint(
            str(tmp_path / "mixed.out"),
            [first, second],
            {"sp": {"mode": "trajectory", "verbose": 1}},
        ).run()

    assert v4_calculator.calculate_calls == 0
    assert legacy_calculator.calculate_calls == 0


def test_future_frame_geometry_drift_fails_before_its_physical_evaluation(
    direct_v4, tmp_path
):
    calculator = _CanonicalFrameCalculator()
    first = _frame(0.0, calculator)
    future = _frame(0.5, calculator)
    original_calculate = calculator.calculate

    def mutate_future(atoms=None, properties=None, system_changes=all_changes):
        original_calculate(atoms, properties, system_changes)
        if calculator.calculate_calls == 1:
            future.positions[0, 0] = 0.8

    calculator.calculate = mutate_future

    with pytest.raises(RuntimeError, match="drift"):
        SinglePoint(
            str(tmp_path / "future-drift.out"),
            [first, future],
            {"sp": {"mode": "trajectory", "verbose": 1}},
        ).run()

    assert calculator.calculate_calls == 1


def test_current_frame_geometry_drift_fails_after_its_evaluation(direct_v4, tmp_path):
    calculator = _CanonicalFrameCalculator()
    atom = _frame(0.5, calculator)
    original_calculate = calculator.calculate

    def mutate_current(atoms=None, properties=None, system_changes=all_changes):
        assert atoms is not None
        original_calculate(atoms, properties, system_changes)
        atoms.positions[0, 0] = 0.8

    calculator.calculate = mutate_current

    with pytest.raises(RuntimeError, match="drift"):
        SinglePoint(
            str(tmp_path / "current-drift.out"),
            atom,
            {"sp": {"mode": "single", "verbose": 1}},
        ).run()

    assert calculator.calculate_calls == 1


def test_v4_mode_owns_result_shape_and_closes_invalid_requests(direct_v4, tmp_path):
    single_calculator = _CanonicalFrameCalculator()
    single = SinglePoint(
        str(tmp_path / "single-list.out"),
        [_frame(0.0, single_calculator)],
        {"sp": {"mode": "single", "verbose": 0}},
    ).run()
    assert isinstance(single, SPFrameResult)

    trajectory_calculator = _CanonicalFrameCalculator()
    trajectory = SinglePoint(
        str(tmp_path / "trajectory-atom.out"),
        _frame(0.0, trajectory_calculator),
        {"sp": {"mode": "trajectory", "verbose": 0}},
    ).run()
    assert isinstance(trajectory, SPBatchResult)
    assert len(trajectory.frames) == 1

    multi_calculator = _CanonicalFrameCalculator()
    with pytest.raises(ValueError, match="single"):
        SinglePoint(
            str(tmp_path / "single-multi.out"),
            [_frame(0.0, multi_calculator), _frame(0.5, multi_calculator)],
            {"sp": {"mode": "single", "verbose": 1}},
        ).run()
    assert multi_calculator.calculate_calls == 0

    closed_calculator = _CanonicalFrameCalculator()
    with pytest.raises(ValueError, match="mode"):
        SinglePoint(
            str(tmp_path / "closed-mode.out"),
            _frame(0.0, closed_calculator),
            {"sp": {"mode": "collection", "verbose": 1}},
        ).run()
    assert closed_calculator.calculate_calls == 0


def test_direct_v4_rejects_tuple_before_forward_and_list_keeps_unit_boundary(
    direct_v4, tmp_path
):
    tuple_calculator = _CanonicalFrameCalculator()
    tuple_input: Any = (_frame(1.0, tuple_calculator),)
    with pytest.raises(ValueError, match="tuple"):
        SinglePoint(
            str(tmp_path / "tuple.out"),
            tuple_input,
            {"sp": {"mode": "trajectory", "verbose": 0}},
        ).run()
    assert tuple_calculator.calculate_calls == 0

    list_calculator = _CanonicalFrameCalculator()
    output = tmp_path / "list.out"
    result = SinglePoint(
        str(output),
        [_frame(1.0, list_calculator)],
        {"sp": {"mode": "trajectory", "verbose": 0}},
    ).run()
    assert isinstance(result, SPBatchResult)
    assert result.frames[0].energy_eV == 2.0
    assert f"Energy: {2.0 * EV2HARTREE:.10f} Hartree" in output.read_text()


def test_energy_only_records_explicitly_unrequested_forces_without_force_call(
    direct_v4, tmp_path
):
    calculator = _CanonicalFrameCalculator()
    result = SinglePoint(
        str(tmp_path / "energy-only.out"),
        [_frame(0.0, calculator), _frame(0.5, calculator)],
        {"sp": {"mode": "trajectory", "verbose": 0}},
    ).run()

    assert isinstance(result, SPBatchResult)
    assert calculator.requested_properties == [("energy",), ("energy",)]
    assert [frame.energy_eV for frame in result.frames] == [1.0, 1.25]
    assert all(frame.forces_eV_per_A is None for frame in result.frames)
    assert all(frame.force_scope is None for frame in result.frames)


def test_identical_consecutive_frames_reuse_valid_ase_cache(direct_v4, tmp_path):
    calculator = _CanonicalFrameCalculator()
    result = SinglePoint(
        str(tmp_path / "cache-hit.out"),
        [_frame(0.0, calculator), _frame(0.0, calculator)],
        {"sp": {"mode": "trajectory", "verbose": 1}},
    ).run()

    assert isinstance(result, SPBatchResult)
    assert calculator.calculate_calls == 1
    assert len(result.frames) == 2
    assert result.frames[0].energy_eV == result.frames[1].energy_eV


def test_v4_sp_trajectory_mode_selects_multi_image_outcome_v2_policy():
    profile = route2_smd_profile_spec(
        PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_TORCH_CPU_V4_PROFILE
    )
    single = route2_workflow_capability_for_params(profile, "sp", {})
    trajectory = route2_workflow_capability_for_params(
        profile, "sp", {"mode": "trajectory"}
    )

    assert single is not None
    assert trajectory is not None
    assert single.image_policy == "single"
    assert single.outcome_policy == "task_success"
    assert trajectory.image_policy == "multi"
    assert trajectory.outcome_policy == "single_point_frames"


def test_sp_mode_parser_keeps_collection_closed(tmp_path):
    parsed = CommandControl.from_settings(
        ["#sp(mode=trajectory,verbose=1)"], str(tmp_path / "ok.out")
    )
    assert parsed.params["mode"] == "trajectory"
    legacy = CommandControl.from_settings(
        ["#sp(mode=collection)"], str(tmp_path / "legacy.out")
    )
    assert legacy.params["mode"] == "collection"
    with pytest.raises(ValueError, match="SP mode"):
        CommandControl.from_settings(
            [
                "#model=macepolm",
                "#device=cpu",
                "#sp(mode=collection)",
                "#solv(implicit=water,method=smd,provider=torch,experimental=true)",
            ],
            str(tmp_path / "closed.out"),
        )


def _trajectory_input(*, mode: str, second_symbol: str = "O") -> str:
    return "\n".join(
        [
            "#model=macepolm",
            "#device=cpu",
            f"#sp{mode}",
            "#solv(implicit=water,method=smd,provider=torch,experimental=true)",
            "",
            "0 1",
            "O 0 0 0",
            "H 0.9572 0 0",
            "H -0.239 0.9266 0",
            "",
            "0 1",
            f"{second_symbol} 0 0 0.1",
            "H 0.9572 0 0.1",
            "H -0.239 0.9266 0.1",
        ]
    )


def test_input_reader_opens_only_same_identity_v4_trajectory_mode(tmp_path):
    accepted = tmp_path / "accepted.inp"
    accepted.write_text(_trajectory_input(mode="(mode=trajectory)"))
    atoms = InputReader()(str(accepted), str(tmp_path / "accepted.out"))
    assert len(cast(Any, atoms).multiatoms) == 2

    default = tmp_path / "default.inp"
    default.write_text(_trajectory_input(mode=""))
    with pytest.raises(ValueError, match="multiple structures"):
        InputReader()(str(default), str(tmp_path / "default.out"))

    heterogeneous = tmp_path / "heterogeneous.inp"
    heterogeneous.write_text(
        _trajectory_input(mode="(mode=trajectory)", second_symbol="N")
    )
    with pytest.raises(ValueError, match="ordered symbols"):
        InputReader()(str(heterogeneous), str(tmp_path / "heterogeneous.out"))


def test_existing_single_sp_direct_api_and_v2_v3_inputs_stay_compatible(tmp_path):
    calculator = _CanonicalFrameCalculator()
    atom = _frame(0.0, calculator)
    result = SinglePoint(
        str(tmp_path / "single.out"), atom, {"sp": {"verbose": 0}}
    ).run()
    assert result is None
    assert calculator.requested_properties == [("energy",)]


def test_dispatcher_v4_trajectory_serializes_all_frames_not_last_cache(
    monkeypatch, tmp_path
):
    calculator = _official_v4_calculator()
    frames = [_frame(x, calculator) for x in (0.0, 0.5, 0.0)]
    monkeypatch.setattr(
        "maple.function.dispatcher.pure_nonmd_status.is_pure_torch_calculator",
        lambda atoms: True,
    )
    monkeypatch.setattr(
        "maple.function.dispatcher.pure_nonmd_status.is_pure_torch_v4",
        lambda atoms: True,
    )
    output = tmp_path / "dispatcher-trajectory.out"

    record = Dispatcher()(
        _Control(params={"mode": "trajectory", "verbose": 1}),
        "sp",
        frames,
        str(output),
    )

    assert isinstance(record, dict)
    assert record["schema"] == "maple-route2-workflow-outcome-v2"
    assert record["final_metrics"]["frame_count"] == 3
    stored = record["final_metrics"]["frames"]
    assert [frame["energy_eV"] for frame in stored] == [1.0, 1.25, 1.0]
    assert stored[0]["forces_eV_per_A"] == stored[2]["forces_eV_per_A"]
    assert (
        json.loads((tmp_path / "dispatcher-trajectory_status.json").read_text())
        == record
    )


def test_dispatcher_v4_rejects_missing_expected_sp_result(monkeypatch, tmp_path):
    calculator = _official_v4_calculator()
    atom = _frame(0.0, calculator)
    monkeypatch.setattr(
        "maple.function.dispatcher.pure_nonmd_status.is_pure_torch_calculator",
        lambda atoms: True,
    )
    monkeypatch.setattr(
        "maple.function.dispatcher.pure_nonmd_status.is_pure_torch_v4",
        lambda atoms: True,
    )
    monkeypatch.setattr(
        "maple.function.dispatcher.sp.SinglePoint.run", lambda self: None
    )

    with pytest.raises(RuntimeError, match="expected"):
        Dispatcher()(
            {"mode": "trajectory", "verbose": 0},
            "sp",
            atom,
            str(tmp_path / "wrong-result.out"),
        )


def test_dispatcher_status_geometry_comes_from_immutable_frame_result(
    monkeypatch, tmp_path
):
    calculator = _official_v4_calculator()
    atom = _frame(0.5, calculator)
    monkeypatch.setattr(
        "maple.function.dispatcher.pure_nonmd_status.is_pure_torch_calculator",
        lambda atoms: True,
    )
    monkeypatch.setattr(
        "maple.function.dispatcher.pure_nonmd_status.is_pure_torch_v4",
        lambda atoms: True,
    )
    original_run = SinglePoint.run

    def mutate_live_atoms(job):
        result = original_run(job)
        job.atoms.positions[0, 0] = 9.0
        return result

    monkeypatch.setattr(SinglePoint, "run", mutate_live_atoms)
    record = Dispatcher()(
        _Control(params={"verbose": 0}),
        "sp",
        atom,
        str(tmp_path / "immutable-geometry.out"),
    )

    assert isinstance(record, dict)
    assert record["geometry"][0]["positions_angstrom"] == [[0.5, 0.0, 0.0]]


@pytest.mark.parametrize(
    "control",
    [{"verbose": 0}, _Control(params={"verbose": 0})],
)
def test_dispatcher_accepts_mapping_and_params_object(control, monkeypatch, tmp_path):
    calculator = _official_v4_calculator()
    atom = _frame(0.0, calculator)
    monkeypatch.setattr(
        "maple.function.dispatcher.pure_nonmd_status.is_pure_torch_calculator",
        lambda atoms: True,
    )
    monkeypatch.setattr(
        "maple.function.dispatcher.pure_nonmd_status.is_pure_torch_v4",
        lambda atoms: True,
    )
    record = Dispatcher()(control, "sp", atom, str(tmp_path / "control.out"))
    assert isinstance(record, dict)
    assert record["task_success"] is True


def test_dispatcher_single_v4_keeps_compatible_energy_only_fields(
    monkeypatch, tmp_path
):
    calculator = _official_v4_calculator()
    atom = _frame(0.5, calculator)
    monkeypatch.setattr(
        "maple.function.dispatcher.pure_nonmd_status.is_pure_torch_calculator",
        lambda atoms: True,
    )
    monkeypatch.setattr(
        "maple.function.dispatcher.pure_nonmd_status.is_pure_torch_v4",
        lambda atoms: True,
    )

    record = Dispatcher()(
        _Control(params={"verbose": 0}),
        "sp",
        atom,
        str(tmp_path / "dispatcher-single.out"),
    )

    assert isinstance(record, dict)
    assert record["schema"] == "maple-route2-workflow-outcome-v1"
    assert record["final_metrics"]["energy_eV"] == 1.25
    assert record["final_metrics"]["forces_eV_per_A"] is None
    assert record["final_metrics"]["force_scope"] is None
    assert calculator.requested_properties == [("energy",)]


def test_dispatcher_single_v4_keeps_compatible_verbose_force_fields(
    monkeypatch, tmp_path
):
    calculator = _official_v4_calculator()
    atom = _frame(0.5, calculator)
    monkeypatch.setattr(
        "maple.function.dispatcher.pure_nonmd_status.is_pure_torch_calculator",
        lambda atoms: True,
    )
    monkeypatch.setattr(
        "maple.function.dispatcher.pure_nonmd_status.is_pure_torch_v4",
        lambda atoms: True,
    )

    record = Dispatcher()(
        _Control(params={"verbose": 1}),
        "sp",
        atom,
        str(tmp_path / "dispatcher-single-force.out"),
    )

    assert isinstance(record, dict)
    assert record["final_metrics"]["energy_eV"] == 1.25
    assert record["final_metrics"]["forces_eV_per_A"] == [[-1.0, 0.0, 0.0]]
    assert record["final_metrics"]["force_scope"].startswith(
        "raw_cartesian_total_pes_force"
    )
    assert calculator.requested_properties == [("forces",)]
