from __future__ import annotations

import hashlib
import importlib.util
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.io import write

from maple.function.dispatcher.solvfe.bulk_water import (
    BulkWaterValidationError,
    load_water_box,
)
from maple.function.dispatcher.solvfe.bulk_water_npt import (
    IAPWS_DENSITY_298K_G_PER_ML,
    BulkWaterNPTConfig,
    _NPTStepwiseMonitor,
    run_bulk_water_npt,
)
from maple.function.dispatcher.solvfe.bulk_water_campaign import (
    load_bulk_water_npt_artifact,
)
from maple.function.dispatcher.solvfe.provenance import (
    collect_implementation_provenance,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
IMPLEMENTATION_PATHS = (
    "maple/function/dispatcher/solvfe/bulk_water.py",
    "maple/function/dispatcher/solvfe/bulk_water_evidence.py",
    "maple/function/dispatcher/solvfe/bulk_water_npt.py",
    "maple/function/dispatcher/solvfe/protocol.py",
    "maple/function/dispatcher/solvfe/provenance.py",
    "maple/function/calculator/mace/_mace_upstream_calculator.py",
    "examples/solvation/route_a/validate_bulk_water_npt.py",
)


class _VolumeStableCalculator(Calculator):
    implemented_properties = ["energy", "forces", "stress"]

    def __init__(
        self,
        reference_positions: np.ndarray,
        *,
        reference_volume: float,
        provenance: dict,
    ) -> None:
        super().__init__()
        self.reference_positions = np.asarray(
            reference_positions,
            dtype=float,
        ).copy()
        self.reference_volume = float(reference_volume)
        self._provenance = dict(provenance)

    @property
    def provenance(self) -> dict:
        return dict(self._provenance)

    def calculate(
        self,
        atoms=None,
        properties=("energy", "forces", "stress"),
        system_changes=all_changes,
    ) -> None:
        super().calculate(atoms, properties, system_changes)
        displacement = np.asarray(atoms.positions) - self.reference_positions
        relative_volume = atoms.get_volume() / self.reference_volume - 1.0
        self.results = {
            "energy": 0.01 * float(np.sum(displacement**2)),
            "forces": -0.02 * displacement,
            "stress": np.array(
                [0.02 * relative_volume] * 3 + [0.0] * 3,
                dtype=float,
            ),
        }


def _water_box() -> Atoms:
    waters = (
        (1.0, 1.0, 1.0),
        (5.0, 1.0, 1.0),
        (1.0, 5.0, 5.0),
        (5.0, 5.0, 5.0),
    )
    symbols = []
    positions = []
    for ox, oy, oz in waters:
        symbols.extend(("O", "H", "H"))
        positions.extend(
            (
                (ox, oy, oz),
                (ox + 0.96, oy, oz),
                (ox - 0.24, oy + 0.93, oz),
            )
        )
    return Atoms(
        symbols,
        positions=positions,
        cell=[8.0, 8.0, 8.0],
        pbc=True,
    )


def _run_evidence(
    tmp_path: Path,
    atoms: Atoms,
) -> tuple[Atoms, dict, dict, dict]:
    source_path = tmp_path / "waterbox.xyz"
    write(source_path, atoms, format="extxyz")
    source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    loaded_atoms, source = load_water_box(
        source_path,
        expected_sha256=source_hash,
        expected_waters=len(atoms) // 3,
    )

    checkpoint = tmp_path / "toy-checkpoint.model"
    checkpoint.write_bytes(b"deterministic toy calculator checkpoint\n")
    calculator_identity = {
        "checkpoint": checkpoint.as_posix(),
        "checkpoint_sha256": hashlib.sha256(
            checkpoint.read_bytes()
        ).hexdigest(),
        "default_dtype": "float64",
        "interaction_cutoff_angstrom": 3.0,
        "result_units": {
            "energy": "eV",
            "forces": "eV/angstrom",
            "stress": "eV/angstrom^3",
        },
    }
    calculator = {
        "schema": "maple-route-a-bulk-water-calculator-v1",
        "calculator": calculator_identity,
        "runtime": {"backend": "deterministic-test-double"},
    }
    implementation = collect_implementation_provenance(
        PROJECT_ROOT,
        IMPLEMENTATION_PATHS,
    )
    return loaded_atoms, source, calculator, implementation


def _short_config() -> BulkWaterNPTConfig:
    return BulkWaterNPTConfig(
        temperature_k=50.0,
        pressure_bar=1.01325,
        timestep_fs=0.01,
        precondition_steps=0,
        equilibration_steps=4,
        production_steps=8,
        sample_interval_steps=2,
        thermostat_damping_fs=1.0,
        barostat_damping_fs=10.0,
        seed=23,
        rdf_bin_width_angstrom=0.25,
        rdf_max_angstrom=3.75,
        rdf_block_count=2,
        interaction_cutoff_angstrom=3.0,
        cutoff_margin_angstrom=0.1,
        minimum_diagnostic_duration_ps=0.00001,
        minimum_diagnostic_frames=4,
        temperature_relative_tolerance=10.0,
        density_reference_g_per_ml=0.117,
        density_relative_error_limit=10.0,
        density_half_drift_limit=10.0,
        pressure_mean_tolerance_bar=10_000.0,
    )


def test_default_npt_protocol_binds_paper_duration_and_iapws_density():
    config = BulkWaterNPTConfig()

    assert config.temperature_k == pytest.approx(298.15)
    assert config.pressure_bar == pytest.approx(1.01325)
    assert config.timestep_fs == pytest.approx(1.0)
    assert config.precondition_timestep_fs == pytest.approx(0.1)
    assert config.precondition_steps == 1_000
    assert config.equilibration_steps == 100_000
    assert config.production_steps == 400_000
    assert config.sample_interval_steps == 100
    assert config.production_duration_ps == pytest.approx(400.0)
    assert config.production_frame_count == 4_000
    assert config.density_reference_g_per_ml == pytest.approx(
        IAPWS_DENSITY_298K_G_PER_ML
    )
    assert config.paper_duration_fidelity is True
    assert config.paper_integrator_fidelity is False


def test_npt_config_rejects_unsafe_or_unblockable_protocols():
    with pytest.raises(BulkWaterValidationError, match="divisible"):
        BulkWaterNPTConfig(
            production_steps=12,
            sample_interval_steps=5,
        )
    with pytest.raises(BulkWaterValidationError, match="RDF blocks"):
        BulkWaterNPTConfig(
            production_steps=20,
            sample_interval_steps=2,
            rdf_block_count=6,
        )
    with pytest.raises(BulkWaterValidationError, match="cutoff_margin"):
        BulkWaterNPTConfig(cutoff_margin_angstrom=0.0)


def test_short_mtk_npt_run_records_cell_density_and_fail_closed_gates(tmp_path):
    atoms, source, calculator_record, implementation = _run_evidence(
        tmp_path,
        _water_box(),
    )
    calculator = _VolumeStableCalculator(
        atoms.positions,
        reference_volume=atoms.get_volume(),
        provenance=calculator_record["calculator"],
    )
    observations = []

    result = run_bulk_water_npt(
        atoms,
        calculator=calculator,
        config=_short_config(),
        source_provenance=source,
        calculator_provenance=calculator_record,
        implementation_provenance=implementation,
        progress_callback=lambda row: observations.append(dict(row)),
    )

    assert result.summary["schema"].endswith("npt-summary-v1")
    assert result.summary["trajectory"]["ensemble"] == "NPT"
    assert result.summary["trajectory"]["integrator"].endswith(
        "IsotropicMTKNPT"
    )
    assert result.summary["diagnostics"]["stepwise_stability"][
        "all_steps_checked"
    ] is True
    assert result.summary["diagnostics"]["minimum_cell_height_angstrom"] > 6.2
    assert result.summary["diagnostics"]["production_density_block_count"] == 2
    assert result.summary["gates"]["engineering_stability_passed"] is True
    assert result.summary["gates"]["minimum_npt_diagnostic_eligible"] is True
    assert result.summary["gates"]["paper_protocol_reproduced"] is False
    assert result.summary["gates"]["npt_density_validation_passed"] is False
    assert result.summary["gates"]["hamiltonian_freeze_eligible"] is False
    assert result.summary["gates"]["claim_eligible"] is False
    assert result.arrays["production_cells_angstrom"].shape == (4, 3, 3)
    assert result.arrays["production_density_g_per_ml"].shape == (4,)
    assert len(observations) == 7

    output = result.write(tmp_path / "npt-result")
    assert (output / "summary.json").is_file()
    assert (output / "manifest.json").is_file()
    assert load_bulk_water_npt_artifact(output)["result_hash"] == (
        result.result_hash
    )


def test_npt_rejects_initial_or_evolving_cells_that_violate_cutoff(tmp_path):
    atoms = _water_box()
    atoms.set_cell([6.0, 6.0, 6.0], scale_atoms=True)
    loaded, source, calculator_record, implementation = _run_evidence(
        tmp_path,
        atoms,
    )
    calculator = _VolumeStableCalculator(
        loaded.positions,
        reference_volume=loaded.get_volume(),
        provenance=calculator_record["calculator"],
    )

    with pytest.raises(BulkWaterValidationError, match="CELL_CUTOFF_UNSAFE"):
        run_bulk_water_npt(
            loaded,
            calculator=calculator,
            config=replace(
                _short_config(),
                interaction_cutoff_angstrom=3.0,
                cutoff_margin_angstrom=0.1,
            ),
            source_provenance=source,
            calculator_provenance=calculator_record,
            implementation_provenance=implementation,
        )


def test_npt_rejects_config_cutoff_that_differs_from_calculator(tmp_path):
    atoms, source, calculator_record, implementation = _run_evidence(
        tmp_path,
        _water_box(),
    )
    calculator = _VolumeStableCalculator(
        atoms.positions,
        reference_volume=atoms.get_volume(),
        provenance=calculator_record["calculator"],
    )

    with pytest.raises(
        BulkWaterValidationError,
        match="cutoff.*calculator|calculator.*cutoff",
    ):
        run_bulk_water_npt(
            atoms,
            calculator=calculator,
            config=replace(
                _short_config(),
                interaction_cutoff_angstrom=2.9,
            ),
            source_provenance=source,
            calculator_provenance=calculator_record,
            implementation_provenance=implementation,
        )


def test_npt_stepwise_cell_safety_includes_rdf_radius():
    atoms = _water_box()
    calculator_provenance = {
        "checkpoint": "/synthetic/checkpoint",
        "checkpoint_sha256": "a" * 64,
        "interaction_cutoff_angstrom": 3.0,
        "result_units": {
            "energy": "eV",
            "forces": "eV/angstrom",
            "stress": "eV/angstrom^3",
        },
    }
    atoms.calc = _VolumeStableCalculator(
        atoms.positions,
        reference_volume=atoms.get_volume(),
        provenance=calculator_provenance,
    )
    monitor = _NPTStepwiseMonitor(
        replace(
            _short_config(),
            interaction_cutoff_angstrom=3.0,
            rdf_max_angstrom=3.75,
        )
    )
    monitor.inspect(atoms, stage="initial", step=0)
    atoms.set_cell([7.4, 7.4, 7.4], scale_atoms=True)

    with pytest.raises(BulkWaterValidationError, match="CELL_CUTOFF_UNSAFE"):
        monitor.inspect(atoms, stage="equilibration", step=1)


def test_npt_density_diagnostic_requires_pressure_centering(tmp_path):
    atoms, source, calculator_record, implementation = _run_evidence(
        tmp_path,
        _water_box(),
    )
    calculator = _VolumeStableCalculator(
        atoms.positions,
        reference_volume=atoms.get_volume(),
        provenance=calculator_record["calculator"],
    )

    result = run_bulk_water_npt(
        atoms,
        calculator=calculator,
        config=replace(
            _short_config(),
            pressure_mean_tolerance_bar=1.0e-12,
        ),
        source_provenance=source,
        calculator_provenance=calculator_record,
        implementation_provenance=implementation,
    )

    checks = result.summary["gates"]["minimum_npt_diagnostic_checks"]
    assert checks["production_pressure_centered"] is False
    assert result.summary["gates"]["minimum_npt_diagnostic_eligible"] is False


def test_npt_cli_binds_source_checkpoint_and_implementation_hashes():
    script = (
        PROJECT_ROOT
        / "examples/solvation/route_a/validate_bulk_water_npt.py"
    )
    spec = importlib.util.spec_from_file_location(
        "route_a_validate_bulk_water_npt",
        script,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    option_strings = {
        option
        for action in module._parser()._actions
        for option in action.option_strings
    }
    assert {
        "--waterbox-sha256",
        "--expected-waters",
        "--checkpoint-sha256",
        "--default-dtype",
    } <= option_strings
    dtype_action = next(
        action
        for action in module._parser()._actions
        if "--default-dtype" in action.option_strings
    )
    assert dtype_action.default == "float64"
    assert tuple(dtype_action.choices) == ("float32", "float64")
    provenance = module._maple_source_provenance()
    assert provenance["project_root"] == PROJECT_ROOT.as_posix()
    assert len(provenance["git_head"]) in {40, 64}
    assert type(provenance["git_dirty"]) is bool
    assert len(provenance["git_status_sha256"]) == 64
    assert set(provenance["implementation_file_sha256"]) == set(
        IMPLEMENTATION_PATHS
    )
    for relative_path, declared_hash in provenance[
        "implementation_file_sha256"
    ].items():
        assert declared_hash == hashlib.sha256(
            (PROJECT_ROOT / relative_path).read_bytes()
        ).hexdigest()
