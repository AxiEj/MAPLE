from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms, units
from ase.calculators.calculator import Calculator, all_changes
from ase.constraints import FixAtoms
from ase.io import write
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

from maple.function.dispatcher.solvfe.bulk_water import (
    MACE_MD_WATERBOX_COMMIT,
    MACE_MD_WATERBOX_SHA256,
    MACE_MD_WATERBOX_URL,
    BulkWaterNVTConfig,
    BulkWaterValidationError,
    _blocked_rdf,
    _observe,
    _rdf_histograms,
    _StepwiseStabilityMonitor,
    load_water_box,
    replicate_water_box,
    run_bulk_water_nvt,
    validate_water_box,
    water_density_g_per_ml,
)
from maple.function.dispatcher.solvfe import provenance as provenance_module
from maple.function.dispatcher.solvfe.provenance import (
    collect_implementation_provenance,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
IMPLEMENTATION_PATHS = (
    "maple/function/dispatcher/solvfe/bulk_water.py",
    "maple/function/dispatcher/solvfe/protocol.py",
    "maple/function/dispatcher/solvfe/provenance.py",
    "maple/function/calculator/mace/_mace_upstream_calculator.py",
    "examples/solvation/route_a/validate_bulk_water.py",
)


class _BoundedHarmonicCalculator(Calculator):
    implemented_properties = ["energy", "forces", "stress"]

    def __init__(
        self,
        reference: np.ndarray,
        force_constant: float = 0.02,
        stress_ev_per_angstrom3: float = 0.0,
        provenance: dict | None = None,
    ):
        super().__init__()
        self.reference = np.asarray(reference, dtype=float).copy()
        self.force_constant = float(force_constant)
        self.stress_ev_per_angstrom3 = float(stress_ev_per_angstrom3)
        self._provenance = provenance

    @property
    def provenance(self):
        return self._provenance

    def calculate(
        self,
        atoms=None,
        properties=("energy", "forces", "stress"),
        system_changes=all_changes,
    ):
        super().calculate(atoms, properties, system_changes)
        displacement = np.asarray(atoms.positions) - self.reference
        self.results = {
            "energy": 0.5
            * self.force_constant
            * float(np.sum(displacement**2)),
            "forces": -self.force_constant * displacement,
            "stress": np.array(
                [self.stress_ev_per_angstrom3] * 3 + [0.0] * 3,
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


def _short_config() -> BulkWaterNVTConfig:
    return BulkWaterNVTConfig(
        temperature_k=50.0,
        timestep_fs=0.1,
        thermalization_steps=2,
        thermalization_friction_per_fs=0.1,
        equilibration_steps=2,
        equilibration_friction_per_fs=0.1,
        production_steps=4,
        production_friction_per_fs=0.1,
        sample_interval_steps=2,
        seed=17,
        rdf_bin_width_angstrom=0.25,
        rdf_max_angstrom=3.75,
        rdf_block_count=2,
        minimum_diagnostic_duration_ps=0.0001,
        minimum_diagnostic_frames=2,
        temperature_relative_tolerance=10.0,
    )


def _run_evidence(
    tmp_path,
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


def test_official_waterbox_source_is_commit_and_hash_pinned():
    assert MACE_MD_WATERBOX_COMMIT == "e19729524fc91920169d4e193e4edd55bc4c5707"
    assert f"/{MACE_MD_WATERBOX_COMMIT}/" in MACE_MD_WATERBOX_URL
    assert MACE_MD_WATERBOX_SHA256 == (
        "a052257f5f9c068884ec7527d6dd41d05a7c3705b729e7d9703543890931ec61"
    )


def test_waterbox_loader_binds_bytes_topology_and_density(tmp_path):
    source = tmp_path / "waterbox.xyz"
    write(source, _water_box(), format="extxyz")
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()

    atoms, provenance = load_water_box(
        source,
        expected_sha256=source_hash,
        expected_waters=4,
        source_url="https://example.invalid/waterbox.xyz",
        source_commit="test",
    )

    assert provenance["schema"] == "maple-route-a-bulk-water-source-v1"
    assert provenance["sha256"] == source_hash
    assert provenance["topology"]["water_count"] == 4
    assert provenance["topology"]["safe_rdf_radius_angstrom"] == pytest.approx(
        4.0
    )
    assert provenance["topology"]["density_g_per_ml"] == pytest.approx(
        water_density_g_per_ml(atoms)
    )


def test_waterbox_loader_fails_closed_on_byte_or_topology_drift(tmp_path):
    source = tmp_path / "waterbox.xyz"
    atoms = _water_box()
    write(source, atoms, format="extxyz")
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()

    with pytest.raises(BulkWaterValidationError, match="HASH_MISMATCH"):
        load_water_box(
            source,
            expected_sha256="0" * 64,
            expected_waters=4,
        )

    reordered = atoms[[1, 0, *range(2, len(atoms))]]
    with pytest.raises(BulkWaterValidationError, match="atom order"):
        validate_water_box(reordered, expected_waters=4)
    constrained = atoms.copy()
    constrained.set_constraint(FixAtoms(indices=[0]))
    with pytest.raises(BulkWaterValidationError, match="CONSTRAINTS"):
        validate_water_box(constrained, expected_waters=4)
    altered_mass = atoms.copy()
    altered_mass.set_masses(altered_mass.get_masses() + 0.01)
    with pytest.raises(BulkWaterValidationError, match="MASSES"):
        validate_water_box(altered_mass, expected_waters=4)
    assert source_hash != "0" * 64


def test_config_requires_two_production_frames():
    with pytest.raises(BulkWaterValidationError, match="two sampled frames"):
        BulkWaterNVTConfig(
            production_steps=2,
            sample_interval_steps=2,
        )


def test_config_requires_integer_steps_and_equal_rdf_blocks():
    with pytest.raises(BulkWaterValidationError, match="must be an integer"):
        BulkWaterNVTConfig(thermalization_steps=1.5)
    with pytest.raises(BulkWaterValidationError, match="divisible by rdf"):
        BulkWaterNVTConfig(
            production_steps=12,
            sample_interval_steps=2,
            rdf_block_count=4,
        )
    with pytest.raises(BulkWaterValidationError, match="must be positive"):
        BulkWaterNVTConfig(equilibration_steps=0)


def test_short_nvt_run_produces_hash_bound_observations_and_rdfs(tmp_path):
    atoms, source, calculator_record, implementation = _run_evidence(
        tmp_path,
        _water_box(),
    )
    calculator = _BoundedHarmonicCalculator(
        atoms.positions,
        provenance=calculator_record["calculator"],
    )
    observations = []

    result = run_bulk_water_nvt(
        atoms,
        calculator=calculator,
        config=_short_config(),
        source_provenance=source,
        calculator_provenance=calculator_record,
        implementation_provenance=implementation,
        progress_callback=lambda row: observations.append(dict(row)),
    )

    assert len(observations) == 5
    assert result.summary["schema"].endswith("summary-v2")
    assert result.summary["config"]["schema"].endswith("config-v2")
    assert result.summary["gates"]["engineering_stability_passed"] is True
    assert result.summary["gates"]["minimum_diagnostic_eligible"] is True
    assert result.summary["gates"]["hamiltonian_freeze_eligible"] is False
    assert result.summary["gates"]["claim_eligible"] is False
    assert result.summary["diagnostics"]["stepwise_stability"][
        "checked_unique_steps"
    ] == 9
    assert result.summary["diagnostics"]["stepwise_stability"][
        "all_steps_checked"
    ] is True
    assert result.summary["trajectory"]["production_frame_count"] == 2
    assert result.arrays["production_positions_angstrom"].shape == (2, 12, 3)
    assert result.arrays["rdf_r_angstrom"].shape == result.arrays["rdf_oo_g"].shape
    assert np.all(np.isfinite(result.arrays["rdf_oo_g"]))
    assert np.all(np.isfinite(result.arrays["rdf_oh_g"]))
    assert np.all(np.isfinite(result.arrays["rdf_hh_g"]))
    assert len(result.result_hash) == 64

    output = result.write(tmp_path / "result")
    manifest = json.loads((output / "manifest.json").read_text())
    archive = np.load(output / "arrays.npz", allow_pickle=False)
    assert manifest["result_hash"] == result.result_hash
    assert manifest["schema"].endswith("artifact-v2")
    assert set(archive.files) == set(result.arrays)
    with pytest.raises(BulkWaterValidationError, match="OUTPUT_EXISTS"):
        result.write(output)
    with pytest.raises(BulkWaterValidationError, match="summary changed"):
        replace(result, result_hash="0" * 64).write(tmp_path / "bad-summary")
    tampered_arrays = dict(result.arrays)
    tampered_arrays["atomic_numbers"] = np.zeros_like(
        result.arrays["atomic_numbers"]
    )
    with pytest.raises(BulkWaterValidationError, match="arrays changed"):
        replace(result, arrays=tampered_arrays).write(tmp_path / "bad-arrays")


def test_rdf_pair_normalization_and_minimum_image_are_explicit():
    water_count = 32
    numbers = np.tile(np.array([8, 1, 1]), water_count)
    molecule_ids = np.repeat(np.arange(water_count), 3)
    rng = np.random.default_rng(20260726)
    positions = rng.uniform(0.0, 20.0, size=(400, len(numbers), 3))
    cells = np.repeat((np.eye(3) * 20.0)[None, :, :], 400, axis=0)
    edges = np.linspace(0.0, 8.0, 17)

    histograms, normalizations = _rdf_histograms(
        positions,
        cells,
        numbers=numbers,
        molecule_ids=molecule_ids,
        first_atomic_number=8,
        second_atomic_number=8,
        edges=edges,
    )
    shell_volumes = (4.0 * np.pi / 3.0) * (
        edges[1:] ** 3 - edges[:-1] ** 3
    )
    expected_pairs = water_count * (water_count - 1) / 2
    assert normalizations[0] == pytest.approx(
        expected_pairs * shell_volumes / 20.0**3
    )
    rdf, _, blocks = _blocked_rdf(
        histograms,
        normalizations,
        block_count=5,
    )
    assert blocks == 5
    assert np.mean(rdf[4:]) == pytest.approx(1.0, abs=0.08)

    mic_positions = np.zeros((1, 6, 3))
    mic_positions[0, 0] = [0.1, 1.0, 1.0]
    mic_positions[0, 3] = [9.9, 1.0, 1.0]
    mic_histograms, _ = _rdf_histograms(
        mic_positions,
        np.array([np.eye(3) * 10.0]),
        numbers=np.array([8, 1, 1, 8, 1, 1]),
        molecule_ids=np.array([0, 0, 0, 1, 1, 1]),
        first_atomic_number=8,
        second_atomic_number=8,
        edges=np.array([0.0, 0.5, 1.0]),
    )
    assert mic_histograms.tolist() == [[1.0, 0.0]]


def test_observation_pressure_includes_ideal_gas_and_ase_sign():
    atoms = _water_box()
    rng = np.random.default_rng(4)
    MaxwellBoltzmannDistribution(
        atoms,
        temperature_K=300.0,
        force_temp=True,
        rng=rng,
    )
    zero_virial = _BoundedHarmonicCalculator(atoms.positions)
    atoms.calc = zero_virial

    observation = _observe(
        atoms,
        zero_virial,
        stage="test",
        step=0,
        timestep_fs=0.5,
    )

    expected_pressure = (
        2.0 * atoms.get_kinetic_energy() / (3.0 * atoms.get_volume()) / units.bar
    )
    assert observation["temperature_k"] == pytest.approx(300.0)
    assert observation["pressure_bar"] == pytest.approx(expected_pressure)

    atoms.set_velocities(np.zeros((len(atoms), 3)))
    positive_stress = _BoundedHarmonicCalculator(
        atoms.positions,
        stress_ev_per_angstrom3=0.01,
    )
    atoms.calc = positive_stress
    observation = _observe(
        atoms,
        positive_stress,
        stage="test",
        step=0,
        timestep_fs=0.5,
    )
    assert observation["pressure_bar"] == pytest.approx(-0.01 / units.bar)

    invalid_stress = _BoundedHarmonicCalculator(
        atoms.positions,
        stress_ev_per_angstrom3=float("nan"),
    )
    atoms.calc = invalid_stress
    with pytest.raises(BulkWaterValidationError, match="OBSERVATION_INVALID"):
        _observe(
            atoms,
            invalid_stress,
            stage="test",
            step=0,
            timestep_fs=0.5,
        )


def test_same_seed_is_reproducible_and_nonintegral_rdf_edge_is_exact(tmp_path):
    atoms, source, calculator_record, implementation = _run_evidence(
        tmp_path,
        _water_box(),
    )
    config = _short_config()

    first = run_bulk_water_nvt(
        atoms,
        calculator=_BoundedHarmonicCalculator(
            atoms.positions,
            provenance=calculator_record["calculator"],
        ),
        config=config,
        source_provenance=source,
        calculator_provenance=calculator_record,
        implementation_provenance=implementation,
    )
    second = run_bulk_water_nvt(
        atoms,
        calculator=_BoundedHarmonicCalculator(
            atoms.positions,
            provenance=calculator_record["calculator"],
        ),
        config=config,
        source_provenance=source,
        calculator_provenance=calculator_record,
        implementation_provenance=implementation,
    )

    assert first.result_hash == second.result_hash
    for name in first.arrays:
        assert np.array_equal(first.arrays[name], second.arrays[name])

    nonintegral = run_bulk_water_nvt(
        atoms,
        calculator=_BoundedHarmonicCalculator(
            atoms.positions,
            provenance=calculator_record["calculator"],
        ),
        config=replace(
            config,
            rdf_max_angstrom=3.7,
            rdf_bin_width_angstrom=0.25,
        ),
        source_provenance=source,
        calculator_provenance=calculator_record,
        implementation_provenance=implementation,
    )
    assert len(nonintegral.arrays["rdf_r_angstrom"]) == 15
    assert nonintegral.arrays["rdf_r_angstrom"][-1] == pytest.approx(3.6)


def test_provenance_contract_rejects_semantic_drift(tmp_path):
    atoms, source, calculator_record, implementation = _run_evidence(
        tmp_path,
        _water_box(),
    )

    def run_with(source_record, calculator_evidence, implementation_evidence):
        return run_bulk_water_nvt(
            atoms,
            calculator=_BoundedHarmonicCalculator(
                atoms.positions,
                provenance=calculator_record["calculator"],
            ),
            config=_short_config(),
            source_provenance=source_record,
            calculator_provenance=calculator_evidence,
            implementation_provenance=implementation_evidence,
        )

    bad_source = json.loads(json.dumps(source))
    bad_source["topology"]["water_count"] = 99
    with pytest.raises(BulkWaterValidationError, match="source topology"):
        run_with(bad_source, calculator_record, implementation)

    bad_calculator = json.loads(json.dumps(calculator_record))
    bad_calculator["calculator"]["result_units"]["forces"] = "hartree/bohr"
    with pytest.raises(BulkWaterValidationError, match="result units"):
        run_with(source, bad_calculator, implementation)
    missing_cutoff = json.loads(json.dumps(calculator_record))
    missing_cutoff["calculator"].pop("interaction_cutoff_angstrom")
    with pytest.raises(BulkWaterValidationError, match="interaction cutoff"):
        run_with(source, missing_cutoff, implementation)
    with pytest.raises(BulkWaterValidationError, match="provenance mapping"):
        run_bulk_water_nvt(
            atoms,
            calculator=_BoundedHarmonicCalculator(atoms.positions),
            config=_short_config(),
            source_provenance=source,
            calculator_provenance=calculator_record,
            implementation_provenance=implementation,
        )

    bad_implementation = json.loads(json.dumps(implementation))
    bad_implementation["implementation_file_sha256"].pop(
        "maple/function/dispatcher/solvfe/protocol.py"
    )
    with pytest.raises(BulkWaterValidationError, match="hashes are missing"):
        run_with(source, calculator_record, bad_implementation)

    wrong_root = json.loads(json.dumps(implementation))
    wrong_root["project_root"] = tmp_path.as_posix()
    with pytest.raises(BulkWaterValidationError, match="executing MAPLE"):
        run_with(source, calculator_record, wrong_root)

    bad_hash = json.loads(json.dumps(implementation))
    bad_hash["implementation_file_sha256"][
        "maple/function/dispatcher/solvfe/protocol.py"
    ] = "0" * 64
    with pytest.raises(BulkWaterValidationError, match="does not match"):
        run_with(source, calculator_record, bad_hash)

    missing_git = json.loads(json.dumps(implementation))
    missing_git["git_head"] = None
    with pytest.raises(BulkWaterValidationError, match="Git"):
        run_with(source, calculator_record, missing_git)

    invalid_dirty = json.loads(json.dumps(implementation))
    invalid_dirty["git_dirty"] = None
    with pytest.raises(BulkWaterValidationError, match="dirty"):
        run_with(source, calculator_record, invalid_dirty)

    invalid_status_hash = json.loads(json.dumps(implementation))
    invalid_status_hash["git_status_sha256"] = "not-a-sha256"
    with pytest.raises(BulkWaterValidationError, match="status SHA256"):
        run_with(source, calculator_record, invalid_status_hash)

    clean_with_changes = json.loads(json.dumps(implementation))
    clean_with_changes["git_dirty"] = False
    clean_with_changes["git_status_sha256"] = "a" * 64
    with pytest.raises(BulkWaterValidationError, match="Git status"):
        run_with(source, calculator_record, clean_with_changes)

    dirty_without_changes = json.loads(json.dumps(implementation))
    dirty_without_changes["git_dirty"] = True
    dirty_without_changes["git_status_sha256"] = hashlib.sha256(b"").hexdigest()
    with pytest.raises(BulkWaterValidationError, match="Git status"):
        run_with(source, calculator_record, dirty_without_changes)


def test_implementation_provenance_fails_closed_when_git_is_unavailable(
    monkeypatch,
):
    def fail_git(*args, **kwargs):
        raise subprocess.CalledProcessError(128, args[0])

    monkeypatch.setattr(provenance_module.subprocess, "run", fail_git)

    with pytest.raises(RuntimeError, match="GIT_PROVENANCE_UNAVAILABLE"):
        provenance_module.collect_implementation_provenance(
            PROJECT_ROOT,
            IMPLEMENTATION_PATHS,
        )


def test_stepwise_monitor_fails_on_force_or_water_identity():
    atoms = _water_box()
    atoms.calc = _BoundedHarmonicCalculator(
        np.zeros_like(atoms.positions),
    )
    monitor = _StepwiseStabilityMonitor(
        replace(_short_config(), maximum_force_ev_per_angstrom=0.001)
    )
    with pytest.raises(BulkWaterValidationError, match="FORCE_LIMIT"):
        monitor.inspect(atoms, stage="test", step=0)

    broken = _water_box()
    broken.positions[1] += [2.0, 0.0, 0.0]
    broken.calc = _BoundedHarmonicCalculator(broken.positions)
    monitor = _StepwiseStabilityMonitor(_short_config())
    with pytest.raises(BulkWaterValidationError, match="TOPOLOGY_FAILURE"):
        monitor.inspect(broken, stage="test", step=0)


def test_replicate_water_box_unwraps_boundary_crossing_molecules():
    atoms = _water_box()
    atoms.positions[1] += atoms.cell[0]
    validate_water_box(atoms, expected_waters=4)

    replicated = replicate_water_box(atoms, (2, 2, 2))
    topology = validate_water_box(replicated, expected_waters=32)

    assert len(replicated) == 96
    assert np.allclose(
        replicated.cell.lengths(),
        2.0 * atoms.cell.lengths(),
    )
    assert topology["density_g_per_ml"] == pytest.approx(
        water_density_g_per_ml(atoms)
    )
    assert topology["oh_distance_range_angstrom"][1] < 1.3


def test_cli_locks_official_source_and_hashes_its_implementation():
    script = (
        PROJECT_ROOT
        / "examples/solvation/route_a/validate_bulk_water.py"
    )
    spec = importlib.util.spec_from_file_location(
        "route_a_validate_bulk_water",
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
    assert "--waterbox-sha256" not in option_strings
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
