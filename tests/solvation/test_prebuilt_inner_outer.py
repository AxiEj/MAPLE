from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest

from maple.function.calculator.extra_correction.implicit.correction import (
    ImplicitSolvationCorrection,
)
from maple.function.calculator.extra_correction.implicit.openmm_gb import OpenMMGB
from maple.function.calculator.calculator_base import CalcABC
from maple.function.dispatcher.sp.sp import SinglePoint
from maple.function.read.command_control import CommandControl
from maple.function.read.filereader.mol2_reader import MOL2Reader
from maple.function.read.input_reader import InputReader

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = REPOSITORY_ROOT / "docs" / "implicit-solvation" / "benchmarks"
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

import benchmark_core as core


def _parse(*lines):
    return CommandControl.from_settings(list(lines)).as_dict()


def test_prebuilt_inner_contract_accepts_only_fixed_mol2_charges():
    params = _parse(
        "#model=ani2x",
        "#opt",
        "#charge(source=mol2,label=prebuilt-cluster)",
        (
            "#solv(implicit=water,inner=prebuilt,method=gb,model=obc2,"
            "nonpolar=ace,experimental=true)"
        ),
    )

    assert params["solv"]["inner"] == "prebuilt"
    assert params["charge"]["source"] == "mol2"
    assert params["charge"]["mode"] == "fixed"

    with pytest.raises(ValueError, match="prebuilt.*source=mol2"):
        _parse(
            "#model=ani2x",
            "#sp",
            "#charge(source=maple)",
            "#solv(implicit=water,inner=prebuilt,method=gb,experimental=true)",
        )


@pytest.mark.parametrize(
    "solvation_line",
    [
        (
            "#solv(implicit=water,inner=prebuilt,method=pb,provider=apbs,"
            "model=lpb,nonpolar=apbs,experimental=true)"
        ),
        (
            "#solv(implicit=water,inner=prebuilt,method=gb,provider=ambertools,"
            "model=chagb,nonpolar=cavity-dispersion,experimental=true)"
        ),
        (
            "#solv(implicit=water,inner=prebuilt,method=gb,provider=openmm,"
            "model=obc2,nonpolar=none,experimental=true)"
        ),
    ],
)
def test_prebuilt_parser_rejects_unvalidated_endpoint_combinations(
    solvation_line,
):
    with pytest.raises(ValueError, match="prebuilt.*OpenMM GB.*ACE or LCPO"):
        _parse(
            "#model=ani2x",
            "#sp",
            "#charge(source=mol2,label=prebuilt-cluster)",
            solvation_line,
        )


def test_prebuilt_direct_api_rejects_pb_endpoint(
    methanol_water_cluster_mol2,
    tmp_path,
):
    atoms = MOL2Reader(
        str(methanol_water_cluster_mol2),
        charge=0,
        mult=1,
        validate_charge=True,
        allow_disconnected=True,
    )

    with pytest.raises(ValueError, match="prebuilt.*OpenMM GB.*ACE or LCPO"):
        ImplicitSolvationCorrection(
            atoms,
            {
                "source": "mol2",
                "label": "prebuilt-cluster",
                "mode": "fixed",
                "geometry": "keep",
            },
            {
                "implicit": "water",
                "inner": "prebuilt",
                "method": "pb",
                "provider": "apbs",
                "model": "lpb",
                "nonpolar": "apbs",
                "experimental": True,
            },
            output=tmp_path / "cluster-pb.out",
        )


def test_existing_explicit_generator_is_not_silently_used_as_a_prebuilt_inner():
    with pytest.raises(ValueError, match="either explicit=.*or implicit="):
        _parse(
            "#model=ani2x",
            "#sp",
            "#charge(source=mol2)",
            (
                "#solv(explicit=water,implicit=water,inner=prebuilt,method=gb,"
                "experimental=true)"
            ),
        )


def test_disconnected_mol2_requires_explicit_prebuilt_inner_opt_in(
    methanol_water_cluster_mol2,
):
    with pytest.raises(ValueError, match="connected molecule"):
        MOL2Reader(
            str(methanol_water_cluster_mol2),
            charge=0,
            mult=1,
            validate_charge=True,
        )

    atoms = MOL2Reader(
        str(methanol_water_cluster_mol2),
        charge=0,
        mult=1,
        validate_charge=True,
        allow_disconnected=True,
    )
    metadata = atoms.info["mol2"]
    assert metadata["component_count"] == 2
    assert metadata["component_ids"] == [0, 0, 0, 0, 0, 0, 1, 1, 1]
    assert metadata["component_charge_sums_e"] == pytest.approx([0.0, 0.0])


def test_input_reader_accepts_only_a_multicomponent_prebuilt_cluster(
    methanol_water_cluster_mol2,
    water_mol2,
    tmp_path,
):
    def write_input(path, mol2):
        path.write_text(
            "\n".join(
                [
                    "#model=ani2x",
                    "#sp",
                    "#charge(source=mol2,label=prebuilt-cluster)",
                    (
                        "#solv(implicit=water,inner=prebuilt,method=gb,"
                        "model=obc2,nonpolar=ace,experimental=true)"
                    ),
                    "",
                    "0 1",
                    f"MOL2 {mol2}",
                ]
            ),
            encoding="utf-8",
        )

    cluster_input = tmp_path / "cluster.inp"
    write_input(cluster_input, methanol_water_cluster_mol2)
    atoms = InputReader()(str(cluster_input), str(tmp_path / "cluster.out"))
    assert atoms.info["mol2"]["component_count"] == 2
    assert atoms.info["_maple_solvation_options"]["inner"] == "prebuilt"

    single_input = tmp_path / "single.inp"
    write_input(single_input, water_mol2)
    with pytest.raises(ValueError, match="at least two connected components"):
        InputReader()(str(single_input), str(tmp_path / "single.out"))


def test_openmm_topology_has_one_synthetic_residue_per_cluster_component(
    methanol_water_cluster_mol2,
):
    from maple.function.calculator.extra_correction.implicit.common import (
        build_openmm_topology,
    )

    atoms = MOL2Reader(
        str(methanol_water_cluster_mol2),
        charge=0,
        mult=1,
        validate_charge=True,
        allow_disconnected=True,
    )
    topology = build_openmm_topology(atoms)

    assert [residue.name for residue in topology.residues()] == ["MOL1", "MOL2"]
    assert topology.getNumAtoms() == len(atoms)
    assert topology.getNumBonds() == 7


def test_prebuilt_inner_manifest_names_the_quantity_without_free_energy_claim(
    methanol_water_cluster_mol2,
    tmp_path,
):
    atoms = MOL2Reader(
        str(methanol_water_cluster_mol2),
        charge=0,
        mult=1,
        validate_charge=True,
        allow_disconnected=True,
    )
    correction = ImplicitSolvationCorrection(
        atoms,
        {
            "source": "mol2",
            "label": "prebuilt-cluster",
            "mode": "fixed",
            "geometry": "keep",
        },
        {
            "implicit": "water",
            "inner": "prebuilt",
            "method": "gb",
            "model": "obc2",
            "nonpolar": "ace",
            "experimental": True,
        },
        output=tmp_path / "cluster.out",
    )
    manifest = json.loads(
        (correction.audit_dir / "manifest.json").read_text(encoding="utf-8")
    )

    assert manifest["route"]["composition_mode"] == (
        "prebuilt-explicit-inner/implicit-outer"
    )
    assert manifest["route"]["coordinate_scope"] == "entire-prebuilt-cluster"
    assert manifest["route"]["thermodynamic_quantity"] == (
        "fixed-shell cluster-continuum configurational potential"
    )
    assert manifest["route"]["absolute_solvation_free_energy_claim"] is False
    assert manifest["route"]["cluster_component_count"] == 2
    assert manifest["charge"]["component_charge_sums_e"] == pytest.approx([0.0, 0.0])
    assert (
        "cluster_formation_or_occupancy_free_energy"
        in manifest["route"]["terms_not_computed"]
    )

    calculator = CalcABC()
    calculator.solvent_correction = correction
    atoms.calc = calculator
    calculator._finalize_results(
        atoms,
        energy=1.0,
        forces=np.zeros((len(atoms), 3)),
        unit="hartree",
    )
    structured = calculator.results["solvation"]
    rendered = "".join(SinglePoint._solvation_lines(atoms))
    assert "delta_g_solv_hartree" not in structured
    assert "cluster_continuum_correction_hartree" in structured
    assert "Delta G_solv" not in rendered
    assert "fixed-shell cluster-continuum configurational potential" in rendered
    assert "not an absolute solvation free energy" in rendered


def test_prebuilt_cluster_gb_force_matches_finite_difference(
    methanol_water_cluster_mol2,
):
    atoms = MOL2Reader(
        str(methanol_water_cluster_mol2),
        charge=0,
        mult=1,
        validate_charge=True,
        allow_disconnected=True,
    )
    provider = OpenMMGB(
        atoms,
        atoms.get_initial_charges(),
        model="obc2",
        nonpolar="ace",
        platform="Reference",
    )
    result = provider.evaluate(atoms, need_forces=True)

    atom_index, axis, step = 7, 0, 0.003
    displaced_plus = atoms.copy()
    displaced_minus = atoms.copy()
    plus = displaced_plus.get_positions()
    minus = displaced_minus.get_positions()
    plus[atom_index, axis] += step
    minus[atom_index, axis] -= step
    displaced_plus.set_positions(plus)
    displaced_minus.set_positions(minus)
    energy_plus = provider.evaluate(displaced_plus).energy_hartree
    energy_minus = provider.evaluate(displaced_minus).energy_hartree
    finite_difference_force = -(energy_plus - energy_minus) / (2.0 * step)

    assert np.isfinite(result.energy_hartree)
    assert result.forces_hartree_per_angstrom is not None
    assert result.forces_hartree_per_angstrom[atom_index, axis] == pytest.approx(
        finite_difference_force,
        abs=2.0e-6,
    )


def test_prebuilt_inner_outer_protocol_and_multimlip_smoke_are_frozen():
    protocol_path = BENCHMARK_DIR / "prebuilt_inner_outer_protocol.json"
    artifact_path = BENCHMARK_DIR / "route1-prebuilt-inner-outer-smoke-2026-07-24.json"
    runner_path = BENCHMARK_DIR / "run_prebuilt_inner_outer_smoke.py"
    mol2_path = (
        REPOSITORY_ROOT
        / "tests"
        / "solvation"
        / "data"
        / "prebuilt_inner_outer"
        / "methanol-water.mol2"
    )
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))

    assert protocol["route"] == {
        "name": "Additive fixed-charge PB/GB implicit solvation",
        "role": "Baseline/Product Route",
        "gas_phase_mm_energy": False,
        "retraining": False,
        "hydration_label_residual": False,
    }
    assert protocol["runtime_contract"]["absolute_solvation_free_energy_claim"] is False
    assert len(protocol["absolute_free_energy_boundary"]["terms_not_computed"]) == 4
    assert protocol["smoke"]["combined_force_check"] == {
        "atom_index_one_based": 8,
        "axis": "x",
        "centered_step_angstrom": 0.003,
        "absolute_tolerance_hartree_per_angstrom": 0.00005,
    }
    assert {reference.get("doi") for reference in protocol["literature"]} >= {
        "10.1021/jp802665d",
        "10.1039/D0CP02768E",
    }

    assert artifact["content_sha256"] == core.artifact_content_sha256(artifact)
    assert artifact["protocol_sha256"] == core.sha256_file(protocol_path)
    assert artifact["input"]["mol2_sha256"] == core.sha256_file(mol2_path)
    assert artifact["command_provenance"]["script_sha256"] == core.sha256_file(
        runner_path
    )
    assert artifact["all_checks_pass"] is True
    assert artifact["absolute_solvation_free_energy_claim"] is False
    assert artifact["input"]["component_count"] == 2
    assert artifact["cross_model"]["sp_cluster_correction_spread_hartree"] == 0.0
    assert {record["model"] for record in artifact["models"]} == {
        "maceoff23m",
        "aimnet2",
        "ani2x",
    }
    for model in artifact["models"]:
        assert model["all_checks_pass"] is True
        assert set(model["tasks"]) == {"sp", "opt", "scan"}
        force_check = model["tasks"]["sp"]["task_files"]["combined_force_check"]
        assert (
            force_check["absolute_error_hartree_per_angstrom"]
            <= force_check["absolute_tolerance_hartree_per_angstrom"]
        )
        for task in model["tasks"].values():
            assert task["all_checks_pass"] is True
            assert task["provenance"]["absolute_solvation_free_energy_claim"] is False
            assert task["provenance"]["composition_mode"] == (
                "prebuilt-explicit-inner/implicit-outer"
            )
