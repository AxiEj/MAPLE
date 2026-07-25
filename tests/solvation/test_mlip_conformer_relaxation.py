from __future__ import annotations

from pathlib import Path
import sys

from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.units import Hartree
import numpy as np
import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = REPOSITORY_ROOT / "docs/implicit-solvation/benchmarks"
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

import benchmark_core as core
import run_mlip_conformer_relaxation as relaxation


class _HartreeCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self):
        super().__init__()
        self.requested_properties: list[tuple[str, ...]] = []

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        requested = tuple(properties or ["energy"])
        self.requested_properties.append(requested)
        self.results = {"energy": 2.0, "free_energy": 2.0}
        if "forces" in requested:
            self.results["forces"] = np.full((len(atoms), 3), 0.5)


def test_hartree_adapter_supports_energy_only_then_energy_consistent_forces():
    source = _HartreeCalculator()
    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.75, 0.0, 0.0]])
    atoms.calc = relaxation.HartreeToEVCalculator(source)

    assert atoms.get_potential_energy() == pytest.approx(2.0 * Hartree)
    assert source.requested_properties == [("energy",)]

    forces = atoms.get_forces()

    assert forces == pytest.approx(np.full((2, 3), 0.5 * Hartree))
    assert source.requested_properties[-1] == ("energy", "forces")


def test_start_selection_deduplicates_roles_in_frozen_priority_order():
    weighting_record = {
        "reference_geometry_union": {"reference_index": 7},
        "result": {
            "gas_dominant_conformer_index": 2,
            "solution_dominant_conformer_index": 2,
        },
    }

    selected = relaxation.select_start_states(weighting_record)

    assert selected == [
        {"index": 7, "roles": ["reference"]},
        {"index": 2, "roles": ["gas-dominant", "solution-dominant"]},
    ]


def test_relaxed_transfer_decomposition_closes_exactly():
    result = relaxation.relaxed_transfer_decomposition(
        gas_min_energy_hartree=-100.0,
        solution_min_gas_energy_hartree=-99.995,
        solution_min_solvent_energy_hartree=-0.010,
    )

    assert result["gas_reorganization_cost_kcal_mol"] == pytest.approx(
        0.005 * relaxation.KCAL_PER_HARTREE
    )
    assert result["solvent_correction_at_solution_min_kcal_mol"] == pytest.approx(
        -0.010 * relaxation.KCAL_PER_HARTREE
    )
    assert result["relaxed_transfer_energy_kcal_mol"] == pytest.approx(
        -0.005 * relaxation.KCAL_PER_HARTREE
    )
    assert result["decomposition_closure_kcal_mol"] == pytest.approx(0.0, abs=1.0e-12)


def test_protocol_pins_weighting_evidence_cases_and_optimizer():
    protocol, fingerprint = relaxation.load_relaxation_protocol(
        BENCHMARK_DIR / "mlip_conformer_relaxation_protocol.json"
    )

    assert len(fingerprint) == 64
    assert protocol["source_partition"] == "development"
    assert protocol["model"]["name"] == "maceoff23m"
    assert [case["compound_id"] for case in protocol["cases"]] == [
        "mobley_8124669",
        "mobley_1770205",
        "mobley_3525176",
        "mobley_5759258",
        "mobley_2881590",
        "mobley_1952272",
    ]
    assert protocol["optimizer"] == {
        "algorithm": "ASE-LBFGS",
        "fmax_eV_per_angstrom": 0.03,
        "max_steps": 300,
        "maxstep_angstrom": 0.1,
    }
    for path_key, hash_key in (
        ("weighting_protocol", "weighting_protocol_sha256"),
        ("weighting_summary", "weighting_summary_sha256"),
    ):
        source = BENCHMARK_DIR / protocol["source_evidence"][path_key]
        assert core.sha256_file(source) == protocol["source_evidence"][hash_key]


def test_forward_relaxation_protocol_uses_am1bcc_default_explicitly():
    protocol, fingerprint = relaxation.load_relaxation_protocol(
        BENCHMARK_DIR / "mlip_conformer_relaxation_am1bcc_protocol.json"
    )

    assert len(fingerprint) == 64
    assert protocol["protocol_id"] == "maple-route1-am1bcc-mlip-conformer-relaxation-v1"
    assert protocol["solvation"]["charge_method"] == "am1bcc"
    assert protocol["solvation"]["model"] == "obc2"
    assert protocol["solvation"]["nonpolar"] == "ace"


def test_frozen_relaxation_summary_reconciles_the_stratified_diagnostic():
    _protocol, fingerprint = relaxation.load_relaxation_protocol(
        BENCHMARK_DIR / "mlip_conformer_relaxation_protocol.json"
    )
    summary = core.load_json(
        BENCHMARK_DIR / "freesolv-mlip-conformer-relaxation-2026-07-24.json"
    )

    assert summary["protocol_fingerprint"] == fingerprint
    assert summary["source_partition"] == "development"
    assert summary["case_count"] == 6
    assert summary["optimization"]["gas"]["attempted"] == 13
    assert summary["optimization"]["gas"]["converged"] == 13
    assert summary["optimization"]["solution"]["attempted"] == 13
    assert summary["optimization"]["solution"]["converged"] == 13
    assert summary["metrics"]["first_stage_mlip_weighted"][
        "mae_kcal_mol"
    ] == pytest.approx(1.999149564984477)
    assert summary["metrics"]["relaxed_transfer_diagnostic"][
        "mae_kcal_mol"
    ] == pytest.approx(1.8740110590595764)
    assert summary["metrics"]["case_outcomes_vs_first_stage_mlip_weighted"] == {
        "improved": 3,
        "unchanged": 0,
        "worsened": 3,
    }
    assert summary["metrics"]["influence_analysis"][
        "first_stage_weighting_to_relaxed_mae_improvement_without_alachlor_kcal_mol"
    ] == pytest.approx(0.16372951826633853)
    assert all(
        case["decomposition_closure_kcal_mol"] == 0.0
        for case in summary["cases"].values()
    )
