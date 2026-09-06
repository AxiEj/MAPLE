from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = REPOSITORY_ROOT / "docs/implicit-solvation/benchmarks"
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

from run_chagb_continuous_nonpolar_probe import (  # pyright: ignore[reportMissingImports]
    CAVITY_OFFSET,
    CAVITY_SURFACE_TENSION,
    OXYGEN_EPSILON,
    OXYGEN_RMIN,
    RHO_WATER,
    evaluate_nonpolar,
    load_parameters,
    main,
)


def _single_sphere_parameters() -> dict[str, object]:
    return {
        "positions_angstrom": [[0.2, -0.7, 1.1]],
        "rmin_angstrom": [1.5],
        "epsilon_kcal_mol": [0.1],
        "prmtop_sha256": "a" * 64,
        "parmed_version": "test",
    }


def test_single_sphere_matches_analytic_cavity_and_dispersion():
    result = evaluate_nonpolar(_single_sphere_parameters())
    cavity_radius = 1.5 + 1.3
    volume = 4.0 * math.pi * cavity_radius**3 / 3.0
    sigma = (1.5 + OXYGEN_RMIN) * 2.0 ** (-1.0 / 6.0)
    mixed_epsilon = math.sqrt(0.1 * OXYGEN_EPSILON)
    expected_dispersion = -32.0 * math.pi * RHO_WATER * mixed_epsilon * sigma**3 / 9.0

    assert result["components_kcal_mol"]["cavity"] == pytest.approx(
        CAVITY_SURFACE_TENSION * volume + CAVITY_OFFSET, rel=2e-9
    )
    assert result["components_kcal_mol"]["dispersion"] == pytest.approx(
        expected_dispersion, rel=2e-7
    )
    assert result["coordinate_gradient_kcal_mol_angstrom"] == pytest.approx(
        np.zeros((1, 3)), abs=2e-9
    )
    assert result["forces_kcal_mol_angstrom"] == pytest.approx(
        -np.asarray(result["coordinate_gradient_kcal_mol_angstrom"]), abs=0.0
    )


def test_reported_force_is_negative_total_gradient_and_full_cha_stays_closed():
    parameters = _single_sphere_parameters()
    parameters["positions_angstrom"] = [[0, 0, 0], [1.4, 0.3, -0.2]]
    parameters["rmin_angstrom"] = [1.5, 1.2]
    parameters["epsilon_kcal_mol"] = [0.1, 0.08]
    result = evaluate_nonpolar(parameters)
    gradient = np.asarray(result["coordinate_gradient_kcal_mol_angstrom"])
    forces = np.asarray(result["forces_kcal_mol_angstrom"])

    assert forces == pytest.approx(-gradient, abs=0.0)
    assert result["capability_boundary"] == {
        "experimental_same_functional_continuum_quadrature_not_legacy_discrete_endpoint": True,
        "nonpolar_energy": True,
        "nonpolar_coordinate_gradient": True,
        "full_chagb_energy": False,
        "full_chagb_force": False,
        "runtime_provider_promoted": False,
    }


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.update(positions_angstrom=[[0.0, 0.0]]),
        lambda value: value.update(positions_angstrom=[[True, 0.0, 0.0]]),
        lambda value: value.update(rmin_angstrom=[0.0]),
        lambda value: value.update(epsilon_kcal_mol=[-0.1]),
        lambda value: value.update(prmtop_sha256="not-a-hash"),
        lambda value: value.update(parmed_version=""),
    ],
)
def test_invalid_parameter_input_fails_closed(tmp_path, mutation):
    payload = _single_sphere_parameters()
    mutation(payload)
    path = tmp_path / "parameters.json"
    path.write_text(json.dumps(payload))

    with pytest.raises((TypeError, ValueError)):
        load_parameters(path)


def test_cli_requires_explicit_output_and_refuses_distinct_overwrite(tmp_path):
    parameters = tmp_path / "parameters.json"
    parameters.write_text(json.dumps(_single_sphere_parameters()))
    output = tmp_path / "result.json"

    assert main(["--parameters", str(parameters), "--output", str(output)]) == 0
    artifact = json.loads(output.read_text())
    assert artifact["status"] == "completed"
    assert artifact["input"]["sha256"]
    assert artifact["implementation"]["script_sha256"]
    assert artifact["implementation"]["volume_kernel_sha256"]
    assert artifact["implementation"]["dispersion_kernel_sha256"]
    assert main(["--parameters", str(parameters), "--output", str(output)]) == 0

    changed = _single_sphere_parameters()
    changed["epsilon_kcal_mol"] = [0.2]
    parameters.write_text(json.dumps(changed))
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        main(["--parameters", str(parameters), "--output", str(output)])


def test_cli_retains_failure_artifact(tmp_path):
    parameters = tmp_path / "invalid-parameters.json"
    payload = _single_sphere_parameters()
    payload["rmin_angstrom"] = [0.0]
    parameters.write_text(json.dumps(payload))
    output = tmp_path / "failed.json"

    with pytest.raises(ValueError, match="zero-radius"):
        main(["--parameters", str(parameters), "--output", str(output)])

    artifact = json.loads(output.read_text())
    assert artifact["status"] == "failed"
    assert artifact["failure"]["type"] == "ValueError"
    assert artifact["capability_boundary"]["full_chagb_force"] is False
    assert artifact["content_sha256"]


def test_capability_boundaries_are_fresh_copies():
    first = evaluate_nonpolar(_single_sphere_parameters())
    first_boundary = first["capability_boundary"]
    assert isinstance(first_boundary, dict)
    first_boundary["full_chagb_force"] = True

    second = evaluate_nonpolar(_single_sphere_parameters())
    assert second["capability_boundary"]["full_chagb_force"] is False
