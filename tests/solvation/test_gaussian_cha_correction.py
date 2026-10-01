"""Thin MAPLE composition adapter contracts for Gaussian-sign CHA."""

import hashlib
import json
from typing import Any, cast

import numpy as np
import pytest
from ase import Atoms

torch = pytest.importorskip("torch")

from maple.function.calculator.calculator_base import CalcABC
from maple.function.calculator.extra_correction.implicit.continuum_chagb_inputs import (
    ContinuumChaTopology,
)
from maple.function.calculator.extra_correction.implicit.gaussian_cha_correction import (
    GaussianChaCorrection,
)
from maple.function.calculator.extra_correction.implicit.result import SolvationResult
from maple.function.calculator.extra_correction.implicit.torch_continuum_chagb_gaussian import (
    continuum_gaussian_cha_scalar,
)


def _sha(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _charge_sha(values):
    return hashlib.sha256(
        (json.dumps(values, separators=(",", ":")) + "\n").encode()
    ).hexdigest()


def topology():
    charges = [-0.8, 0.4, 0.4]
    payload = {
        "schema_version": 1,
        "profile": "chagb-r6-pbsa-continuum-v1",
        "atom_ids": [1, 2, 3],
        "atom_names": ["O", "H1", "H2"],
        "elements": ["O", "H", "H"],
        "atomic_numbers": [8, 1, 1],
        "gaff2_types": ["oh", "ho", "ho"],
        "bonds": [[0, 1, "1"], [0, 2, "1"]],
        "declared_charge_e": 0,
        "source_charges_e": charges,
        "effective_charges_e": charges,
        "source_charges_sha256": _charge_sha(charges),
        "effective_charges_sha256": _charge_sha(charges),
        "source_mol2_sha256": "0" * 64,
        "prepared_prmtop_sha256": "1" * 64,
        "parameter_source_sha256": "2" * 64,
        "serialization_profile": "direct-fixed-charge-v1",
        "bondi_radii_angstrom": [1.5, 1.2, 1.2],
        "cha_radii_angstrom": [1.88, 1.04, 1.04],
        "lj_rmin_angstrom": [1.7683, 1.2, 1.2],
        "lj_epsilon_kcal_mol": [0.152, 0.02, 0.02],
    }
    payload["content_sha256"] = _sha(payload)
    return ContinuumChaTopology.from_mapping(
        payload, expected_content_sha256=payload["content_sha256"]
    )


def atoms():
    value = Atoms("OHH", positions=[[0, 0, 0], [4.2, 0, 0], [0, 10, 0]])
    value.new_array(
        "_maple_implicit_atom_identity", np.asarray([1, 2, 3], dtype=np.int64)
    )
    return value


def test_evaluate_returns_exact_units_components_and_same_scalar_forces():
    topo = topology()
    target = atoms()
    correction = GaussianChaCorrection(
        target,
        topo,
        expected_topology_sha256=topo.content_sha256,
        sigma_e=0.01,
        order=8,
    )
    result = correction.evaluate(target, need_forces=True)
    assert isinstance(result, SolvationResult)
    forces = result.forces_hartree_per_angstrom
    assert forces is not None
    assert forces.shape == (3, 3)
    assert np.isfinite(forces).all()
    direct = continuum_gaussian_cha_scalar(
        torch.tensor(target.positions, dtype=torch.float64),
        topo,
        expected_topology_sha256=topo.content_sha256,
        sigma_e=0.01,
        order=8,
    )
    kcal_per_hartree = 2625.4996394799 / 4.184
    assert result.energy_hartree == float(direct.total_kcal_mol) / kcal_per_hartree
    assert result.energy_hartree == pytest.approx(
        sum(result.components_hartree.values()), abs=1e-15
    )
    assert set(result.components_hartree) == {"polar", "cavity", "dispersion"}
    assert result.provenance["model_identity"] == "chagb-r6-pbsa-gaussian-sign-v1"
    assert (
        result.provenance["derivatives"]
        == "torch automatic differentiation of complete scalar"
    )
    assert correction.supported_properties == ("energy", "forces")


def test_correction_binds_atom_order_elements_topology_and_rejects_pbc(monkeypatch):
    topo = topology()
    target = atoms()
    correction = GaussianChaCorrection(
        target,
        topo,
        expected_topology_sha256=topo.content_sha256,
        sigma_e=0.01,
        order=8,
    )
    reordered = target[[1, 0, 2]]
    with pytest.raises(ValueError, match="atom order|atomic numbers"):
        correction.evaluate(reordered)
    changed_id = target.copy()
    changed_id.arrays["_maple_implicit_atom_identity"][:] = [3, 2, 1]
    with pytest.raises(ValueError, match="atom identity"):
        correction.evaluate(changed_id)
    float_ids = target.copy()
    del float_ids.arrays["_maple_implicit_atom_identity"]
    float_ids.new_array("_maple_implicit_atom_identity", np.asarray([1.9, 2.9, 3.9]))
    with pytest.raises(ValueError, match="atom identity"):
        correction.evaluate(float_ids)
    float_numbers = target.copy()
    monkeypatch.setattr(
        float_numbers,
        "get_atomic_numbers",
        lambda: np.asarray([8.9, 1.9, 1.9]),
    )
    with pytest.raises(ValueError, match="atomic numbers"):
        correction.evaluate(float_numbers)
    periodic = target.copy()
    periodic.set_cell([20, 20, 20])
    periodic.set_pbc(True)
    with pytest.raises(ValueError, match="periodic"):
        correction.evaluate(periodic)


def test_parameters_are_immutable_and_provider_provenance_drives_calcabc_execution():
    topo = topology()
    target = atoms()
    correction = GaussianChaCorrection(
        target,
        topo,
        expected_topology_sha256=topo.content_sha256,
        sigma_e=0.01,
        order=8,
    )
    assert correction.provider is correction
    with pytest.raises((AttributeError, TypeError)):
        setattr(cast(Any, correction), "sigma_e", 0.1)
    calc = CalcABC()
    calc.solvent_correction = correction
    calc._finalize_results(target, energy=1.0, forces=np.zeros((3, 3)), unit="hartree")
    assert calc.execution_provenance["solvent"]["provider"] == "gaussian-cha-continuum"
    assert calc.execution_provenance["solvent"]["platform"] == "CPU"
    assert calc.results["solvation"]["provenance"]["placement"] == "CPU float64 torch"
    assert "delta_g_solv_hartree" in calc.results["solvation"]
    assert "reference_correction_hartree" not in calc.results["solvation"]
    assert "cluster_continuum_correction_hartree" not in calc.results["solvation"]
    provenance = calc.results["solvation"]["provenance"]
    assert provenance["production_admitted"] is False
    assert provenance["accuracy_certified"] is False
    assert provenance["physical_accuracy_claim"] is False
    assert provenance["thermodynamic_quantity"] == (
        "model-predicted Gaussian-CHA solvation correction"
    )
    assert calc.results["solvation"]["ase_free_energy_is_thermochemical_gibbs"] is False


def test_each_coordinate_call_builds_a_fresh_scalar_and_energy_only_has_no_force():
    topo = topology()
    target = atoms()
    correction = GaussianChaCorrection(
        target,
        topo,
        expected_topology_sha256=topo.content_sha256,
        sigma_e=0.01,
        order=8,
    )
    first = correction.evaluate(target, need_forces=True)
    moved = target.copy()
    moved.positions[1, 0] += 0.01
    second = correction.evaluate(moved, need_forces=True)
    energy_only = correction.evaluate(moved, need_forces=False)

    assert second.energy_hartree != first.energy_hartree
    assert not np.array_equal(
        second.provenance["polar_diagnostics"]["born_radii_angstrom"],
        first.provenance["polar_diagnostics"]["born_radii_angstrom"],
    )
    assert energy_only.forces_hartree_per_angstrom is None
    assert energy_only.energy_hartree == second.energy_hartree


def test_smallest_positive_sigma_is_rejected_before_nonfinite_runtime_force():
    topo = topology()
    target = atoms()
    with pytest.raises(ValueError, match="sigma_e"):
        GaussianChaCorrection(
            target,
            topo,
            expected_topology_sha256=topo.content_sha256,
            sigma_e=5e-324,
            order=8,
        )


@pytest.mark.parametrize(
    "invalid_gradient",
    (
        torch.full((3, 3), float("nan"), dtype=torch.float64),
        torch.zeros((9,), dtype=torch.float64),
    ),
)
def test_adapter_rejects_nonfinite_or_wrong_shape_autograd_force(
    monkeypatch, invalid_gradient
):
    topo = topology()
    target = atoms()
    correction = GaussianChaCorrection(
        target,
        topo,
        expected_topology_sha256=topo.content_sha256,
        sigma_e=0.01,
        order=8,
    )
    monkeypatch.setattr(
        torch.autograd,
        "grad",
        lambda *_args, **_kwargs: (invalid_gradient,),
    )
    with pytest.raises(RuntimeError, match=r"gradient must be finite \[N,3\]"):
        correction.evaluate(target, need_forces=True)
