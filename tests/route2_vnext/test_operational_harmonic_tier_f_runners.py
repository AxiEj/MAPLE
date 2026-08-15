from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path
import subprocess
import sys

from ase import Atoms
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools" / "route2_release"
RUNNERS = (
    TOOLS / "run_operational_analytic_harmonic_pes_panel.py",
    TOOLS / "run_operational_analytic_harmonic_cartesian_panel.py",
    TOOLS / "run_operational_analytic_harmonic_symmetry_panel.py",
)


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(TOOLS))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(TOOLS))
    return module


def test_new_tier_f_runners_parse_help_without_importing_torch():
    for runner in RUNNERS:
        script = f"""
import builtins
import runpy
import sys
real_import = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] == 'torch':
        raise AssertionError('runner imported torch before parsing --help')
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded
sys.path.insert(0, {str(TOOLS)!r})
sys.argv = [{str(runner)!r}, '--help']
runpy.run_path({str(runner)!r}, run_name='__main__')
"""
        result = subprocess.run(
            (sys.executable, "-c", script),
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, f"{runner.name}: {result.stderr}"
        assert "--molecule-start" in result.stdout
        assert "--molecule-stop" in result.stdout


def test_new_runners_freeze_distinct_contracts_and_never_reuse_old_numbers():
    pes = _load("operational_harmonic_pes_runner", RUNNERS[0])
    cartesian = _load("operational_harmonic_cartesian_runner", RUNNERS[1])
    symmetry = _load("operational_harmonic_symmetry_runner", RUNNERS[2])
    assert (
        len(
            {
                pes.CONTRACT_VERSION,
                cartesian.CONTRACT_VERSION,
                symmetry.CONTRACT_VERSION,
            }
        )
        == 3
    )
    assert pes.ALTERNATE_ROOT_SEED_VALUES == (1.0e-3, -1.0e-3)
    assert cartesian.ALTERNATE_ROOT_SEED_VALUES == (1.0e-3, -1.0e-3)
    for runner in RUNNERS:
        text = runner.read_text(encoding="utf-8")
        assert "no-legacy-numerical-values" in text
        assert "MACE_POLAR_ANALYTIC_GAUSSIAN_MULTIPOLE_EVALUATOR_ID" in text
        assert "build_system_with_model" in text


def test_legacy_runner_defaults_preserve_their_original_schema_surface():
    pes = _load(
        "fixedbox590_pes_runner_defaults", TOOLS / "run_fixedbox590_pes_panel.py"
    )
    cartesian = _load(
        "fixedbox590_cartesian_runner_defaults",
        TOOLS / "run_fixedbox590_cartesian_panel.py",
    )
    symmetry = _load(
        "fixedbox590_symmetry_runner_defaults",
        TOOLS / "run_fixedbox590_symmetry_panel.py",
    )
    for function in (pes.run_pes_panel, cartesian.run_cartesian_panel):
        parameters = inspect.signature(function).parameters
        assert parameters["record_root_multistart"].default is False
        assert parameters["scalar_identity_builder"].default is None
        assert parameters["domain_record_builder"].default is None
        assert parameters["contract_metadata"].default is None
    symmetry_parameters = inspect.signature(symmetry.run_symmetry_panel).parameters
    assert symmetry_parameters["domain_record_builder"].default is None
    assert symmetry_parameters["contract_metadata"].default is None


class _Continuum:
    @staticmethod
    def debug_geometry_matrices(_atoms):
        return {
            "weighted_basis": np.diag([2.0, 1.0]),
            "surface_operator": np.diag([3.0, 4.0]),
        }


def test_harmonic_domain_record_recomputes_rank_and_condition_margins():
    common = _load(
        "operational_harmonic_common",
        TOOLS / "operational_analytic_harmonic_common.py",
    )
    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.7, 0.0, 0.0]])
    record = common.harmonic_domain_record(_Continuum(), atoms)
    assert record["all_gates_passed"] is True
    assert record["weighted_basis_rank"] == 2
    assert record["minimum_relative_basis_singular_value"] == pytest.approx(0.5)
    assert record["surface_condition_number"] == pytest.approx(4.0 / 3.0)
    assert record["minimum_center_distance_A"] == pytest.approx(0.7)


def _root_leaf(source=1.0, field=2.0, energy=3.0, residual=1.0e-13):
    return {
        "source": [[source, 0.0]],
        "field": [[field, 0.0]],
        "total_energy_eV": energy,
        "actual_unmixed_residual_norm": residual,
    }


def _root_record():
    return {"cold": _root_leaf(), "warm": _root_leaf()}


def test_extended_aggregator_recomputes_root_scalar_and_domain_gates():
    aggregate = _load(
        "aggregate_operational_harmonic_pes_base",
        TOOLS / "aggregate_fixedbox590_pes_panel.py",
    )
    raw = {
        "root_multistart": [_root_record(), _root_record()],
        "scalar_identity": {
            "absolute_error_eV": 0.0,
            "missing_radial_block_max_abs": 0.0,
            "gate_passed": False,
        },
        "domain": {
            "minimum_center_distance_A": 0.7,
            "minimum_relative_basis_singular_value": 0.5,
            "surface_minimum_eigenvalue": 3.0,
            "surface_condition_number": 4.0 / 3.0,
            "all_gates_passed": False,
        },
    }
    recomputed = aggregate._extended_record(raw)
    assert recomputed["all_root_multistart_gates_passed"] is True
    assert recomputed["scalar_identity"]["gate_passed"] is True
    assert recomputed["domain"]["all_gates_passed"] is True

    raw["root_multistart"][1]["warm"]["source"] = [[1.1, 0.0]]
    assert aggregate._extended_record(raw)["all_root_multistart_gates_passed"] is False


def test_extended_loop_aggregator_recomputes_raw_work_roots_and_domain():
    aggregate = _load(
        "aggregate_operational_harmonic_symmetry_base",
        TOOLS / "aggregate_fixedbox590_symmetry_panel.py",
    )
    from maple.solvation.release import (
        SYMMETRY_PANEL_LOOP_SUBDIVISIONS,
        SYMMETRY_PANEL_TRANSLATION_A,
        closed_rectangular_loop,
        load_pes_panel,
        panel_directions,
        panel_geometries,
        reverse_closed_path,
        symmetry_panel_permutation,
        symmetry_panel_rotations,
    )
    from maple.solvation.release.pes_panel import PES_CARTESIAN_PANEL_VARIANT
    from maple.solvation.coupling.state_equation import geometry_sha256

    molecule = load_pes_panel()[0]
    atoms = panel_geometries(molecule)[PES_CARTESIAN_PANEL_VARIANT]
    topology = "a" * 64
    zeros = np.zeros((len(atoms), 3)).tolist()
    zero_source = np.zeros((len(atoms), 8)).tolist()
    base = {
        "energy_eV": 0.0,
        "forces_eV_per_A": zeros,
        "source": zero_source,
        "primal_residual": 0.0,
        "adjoint_residual": 0.0,
        "topology_hash": topology,
    }
    translation = {
        **base,
        "translation_A": list(SYMMETRY_PANEL_TRANSLATION_A),
    }
    permutation = {
        **base,
        "permutation": symmetry_panel_permutation(atoms.numbers).tolist(),
    }
    rotations = [
        {
            **base,
            "index": index,
            "rotation_matrix": rotation.tolist(),
        }
        for index, rotation in enumerate(symmetry_panel_rotations(molecule.molecule_id))
    ]

    directions = panel_directions(atoms, molecule.molecule_id)
    first = directions["seeded-internal"]
    raw_second = directions["radial-internal"]
    second = raw_second - float(np.vdot(first, raw_second)) * first
    second /= float(np.linalg.norm(second))
    forward = closed_rectangular_loop(
        subdivisions_per_edge=SYMMETRY_PANEL_LOOP_SUBDIVISIONS
    )
    reverse = reverse_closed_path(forward)

    def geometry(coefficient):
        value = atoms.copy()
        value.positions += (
            0.02 * coefficient[0] * first + 0.02 * coefficient[1] * second
        )
        return value

    def traversal(coefficients):
        return [
            {
                "geometry_sha256": geometry_sha256(geometry(coefficient)),
                "positions_A": geometry(coefficient).positions.tolist(),
                "forces_eV_per_A": zeros,
                "primal_residual": 0.0,
                "adjoint_residual": 0.0,
                "topology_hash": topology,
            }
            for coefficient in coefficients
        ]

    def root(coefficient):
        return {
            "geometry_sha256": geometry_sha256(geometry(coefficient)),
            "cold": {
                "source": zero_source,
                "field": zero_source,
                "total_energy_eV": 0.0,
                "actual_unmixed_residual_norm": 0.0,
            },
            "warm": {
                "source": zero_source,
                "field": zero_source,
                "total_energy_eV": 0.0,
                "actual_unmixed_residual_norm": 0.0,
            },
        }

    failing_cached_work = {
        "simpson_work_eV": 1.0,
        "gate_threshold_eV": 0.0,
    }
    loop = {
        "cold_forward": failing_cached_work,
        "cold_reverse": failing_cached_work,
        "warm_forward": failing_cached_work,
        "warm_reverse": failing_cached_work,
        "all_cold_warm_roots": False,
        "warm_forward_reverse_repeat": False,
        "maximum_primal_residual": 1.0,
        "maximum_adjoint_residual": 1.0,
        "topology_hashes": ["b" * 64, "c" * 64],
        "raw_traversals": {
            "cold_forward": traversal(forward),
            "cold_reverse": traversal(reverse),
            "warm_forward": traversal(forward),
            "warm_reverse": traversal(reverse),
        },
        "cold_warm_roots": {
            "forward": [root(coefficient) for coefficient in forward],
            "reverse": [root(coefficient) for coefficient in reverse],
        },
        "raw_warm_repeat_records": [
            {
                "geometry_sha256": geometry_sha256(geometry(coefficient)),
                "forward_source": zero_source,
                "reverse_source": zero_source,
                "forward_energy_eV": 0.0,
                "reverse_energy_eV": 0.0,
            }
            for coefficient in forward
        ],
        "domain_records": [
            {
                "geometry_sha256": geometry_sha256(geometry(coefficient)),
                "minimum_center_distance_A": 0.7,
                "minimum_relative_basis_singular_value": 0.5,
                "surface_minimum_eigenvalue": 3.0,
                "surface_condition_number": 4.0 / 3.0,
            }
            for coefficient in forward
        ],
    }
    raw = {
        "molecule_id": molecule.molecule_id,
        "atomic_numbers": atoms.numbers.tolist(),
        "positions_A": atoms.positions.tolist(),
        "geometry_sha256": geometry_sha256(atoms),
        "base": base,
        "rigid_symmetry": {"rotations": [{}, {}, {}]},
        "_raw_rigid_records": {
            "translation": translation,
            "permutation": permutation,
            "rotations": rotations,
        },
        "closed_loop": loop,
    }
    recomputed = aggregate._raw_record(raw, molecule, require_domain_gates=True)
    assert recomputed["closed_loop"]["all_gates_passed"] is True
    assert recomputed["closed_loop"]["maximum_absolute_loop_work_eV"] == 0.0
    assert recomputed["closed_loop"]["all_domain_gates_passed"] is True
