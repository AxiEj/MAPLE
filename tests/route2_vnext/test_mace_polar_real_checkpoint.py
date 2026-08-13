from __future__ import annotations

import json
import os

import numpy as np
import pytest
from ase import Atoms

from maple.function.calculator.extra_correction.implicit.gto_field_projection import (
    ExactGTOFieldProjector,
)
from maple.solvation.models import (
    MACEPolarVariationalFieldEnergy,
    build_official_mace_polar_1_m_adapter,
    build_official_mace_polar_1_m_radial_gto_adapter,
    validate_response_linearization,
    validate_source_evaluation,
    validate_vacuum_evaluation,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("MAPLE_ROUTE2_REAL_MACEPOL") != "1",
    reason="set MAPLE_ROUTE2_REAL_MACEPOL=1 in the pinned real-checkpoint job",
)


def _checkpoint_path() -> str | None:
    return os.environ.get("ROUTE2_MACE_CHECKPOINT")


def _water() -> Atoms:
    return Atoms(
        "H2O",
        positions=np.asarray(
            [
                [0.0000, 0.0000, 0.0000],
                [0.9572, 0.0000, 0.0000],
                [-0.2390, 0.9266, 0.0000],
            ]
        ),
        info={"charge": 0, "mult": 1},
    )


def test_official_mace_polar_checkpoint_sign_nonuniform_response_and_derivatives():
    adapter = build_official_mace_polar_1_m_adapter(
        device=os.environ.get("MAPLE_ROUTE2_MACE_DEVICE", "cpu"),
        checkpoint_path=_checkpoint_path(),
    )
    atoms = _water()
    vacuum = validate_vacuum_evaluation(adapter, atoms, need_forces=True)
    field = np.asarray(
        [
            [0.012, 0.004, -0.003, 0.002],
            [-0.007, -0.002, 0.005, -0.001],
            [0.003, 0.001, 0.002, -0.004],
        ]
    )
    source = validate_source_evaluation(
        adapter, atoms, field, need_fixed_field_forces=True
    )
    zero = validate_source_evaluation(
        adapter, atoms, np.zeros_like(field), need_fixed_field_forces=False
    )
    assert np.isfinite(vacuum.energy_eV)
    assert vacuum.forces_eV_per_A.shape == (3, 3)
    assert np.linalg.norm(source.source - zero.source) > 1e-10
    assert adapter.exact_gto_audit.source_sigmas_angstrom == (1.5,)
    assert adapter.exact_gto_audit.receiver_sigmas_angstrom == (1.5, 3.0)
    assert adapter.exact_gto_audit.exact_gto_operational_available is False

    rng = np.random.default_rng(20260813)
    direction = rng.normal(scale=0.01, size=(3, 4))
    cotangent = rng.normal(scale=0.1, size=(3, 4))
    jvp, vjp, position_vjp = validate_response_linearization(
        adapter,
        atoms,
        field,
        field_direction=direction,
        source_cotangent=cotangent,
        transpose_atol=2e-9,
        transpose_rtol=2e-8,
    )
    jvp_dot = float(np.vdot(jvp, cotangent))
    vjp_dot = float(np.vdot(direction, vjp))
    assert jvp_dot == pytest.approx(vjp_dot, rel=2e-8, abs=2e-9)

    coordinate_direction = rng.normal(size=(3, 3))
    coordinate_direction /= np.linalg.norm(coordinate_direction)
    step = 2e-4
    plus = atoms.copy()
    minus = atoms.copy()
    plus.positions += step * coordinate_direction
    minus.positions -= step * coordinate_direction
    plus_source = validate_source_evaluation(
        adapter, plus, field, need_fixed_field_forces=False
    ).source
    minus_source = validate_source_evaluation(
        adapter, minus, field, need_fixed_field_forces=False
    ).source
    finite_difference = float(
        np.vdot(cotangent, (plus_source - minus_source) / (2 * step))
    )
    analytic = float(np.vdot(position_vjp, coordinate_direction))
    assert analytic == pytest.approx(finite_difference, rel=3e-4, abs=3e-6)

    # Check the local node-field sign/convention against the checkpoint's own
    # upstream uniform-field path.  MACE centres positions before adding r.E.
    calculator = adapter._calculator
    import torch

    uniform_gradient = np.asarray([0.006, -0.004, 0.003])
    batch = calculator._batch_dict(atoms)
    batch["external_field"] = torch.tensor(
        uniform_gradient.reshape(1, 3),
        dtype=calculator.dtype,
        device=calculator.device,
    )
    upstream = calculator._model_forward(
        batch,
        compute_force=False,
        compute_stress=False,
        compute_hessian=False,
    )
    centred = atoms.positions - np.mean(atoms.positions, axis=0, keepdims=True)
    local_potential = centred @ uniform_gradient
    local = calculator.polar_output_torch(
        atoms,
        node_potential_ev=torch.tensor(
            local_potential, dtype=calculator.dtype, device=calculator.device
        ),
        node_gradient_ev_per_angstrom=torch.tensor(
            np.broadcast_to(uniform_gradient, (len(atoms), 3)).copy(),
            dtype=calculator.dtype,
            device=calculator.device,
        ),
    )
    torch.testing.assert_close(
        local["density_coefficients"],
        upstream["density_coefficients"],
        rtol=2e-11,
        atol=2e-11,
    )
    # The learned source/sign convention matches exactly, but the two energy
    # branches intentionally do not: the local-field hook omits the explicit
    # external coupling while upstream's uniform-field energy includes it.
    energy_branch_difference = float(
        (local["energy"] - upstream["energy"]).detach().cpu()
    )
    assert abs(energy_branch_difference) > 1e-6
    print(
        "ROUTE2_MACEPOL_CANARY="
        + json.dumps(
            {
                "checkpoint_sha256": adapter.provenance.checkpoint_sha256,
                "vacuum_energy_eV": vacuum.energy_eV,
                "nonuniform_source_delta_l2": float(
                    np.linalg.norm(source.source - zero.source)
                ),
                "source_total_charge_e": float(np.sum(source.source[:, 0])),
                "jvp_vjp_dot_abs_error": abs(jvp_dot - vjp_dot),
                "position_vjp_directional_abs_error": abs(analytic - finite_difference),
                "uniform_local_upstream_density_max_abs_error": float(
                    torch.max(
                        torch.abs(
                            local["density_coefficients"]
                            - upstream["density_coefficients"]
                        )
                    )
                    .detach()
                    .cpu()
                ),
                "uniform_energy_branch_difference_eV": energy_branch_difference,
                "exact_gto_operational_available": False,
            },
            sort_keys=True,
        )
    )


def test_official_checkpoint_radial_gto_transform_and_response_derivatives():
    adapter = build_official_mace_polar_1_m_radial_gto_adapter(
        device=os.environ.get("MAPLE_ROUTE2_MACE_DEVICE", "cpu"),
        checkpoint_path=_checkpoint_path(),
    )

    atoms = _water()
    rng = np.random.default_rng(20260813)
    field = rng.normal(scale=0.005, size=(len(atoms), 8))
    direction = rng.normal(scale=0.002, size=field.shape)
    cotangent = rng.normal(scale=0.1, size=field.shape)

    state = validate_source_evaluation(
        adapter, atoms, field, need_fixed_field_forces=False
    )
    assert state.source.shape == field.shape
    np.testing.assert_array_equal(state.source[:, (1, 5, 6, 7)], 0.0)
    assert adapter.exact_gto_coupling_available is True
    assert adapter.exact_gto_operational_available is False
    assert adapter.variational_functional_admitted is False

    spec = adapter._calculator.route2_gto_field_projection_spec()
    projector = ExactGTOFieldProjector(spec)
    potentials = np.stack((field[:, 0], field[:, 1]))
    gradients = np.stack((field[:, (4, 2, 3)], field[:, (7, 5, 6)]), axis=0)
    direct_features = projector.project_smoothed_fields(
        potentials,
        gradients,
        scalar_potential_gauge_reference_ev=0.0,
    )
    transformed_features = adapter.field_transform.to_model_features(field)
    transform_error = float(np.max(np.abs(direct_features - transformed_features)))
    assert transform_error <= 2.0e-14

    jvp, vjp, position_vjp = validate_response_linearization(
        adapter,
        atoms,
        field,
        field_direction=direction,
        source_cotangent=cotangent,
        transpose_atol=2.0e-9,
        transpose_rtol=2.0e-8,
    )
    jvp_dot = float(np.vdot(jvp, cotangent))
    vjp_dot = float(np.vdot(direction, vjp))
    assert jvp_dot == pytest.approx(vjp_dot, rel=2.0e-8, abs=2.0e-9)

    coordinate_direction = rng.normal(size=(len(atoms), 3))
    coordinate_direction /= np.linalg.norm(coordinate_direction)
    step = 2.0e-4
    plus = atoms.copy()
    minus = atoms.copy()
    plus.positions += step * coordinate_direction
    minus.positions -= step * coordinate_direction
    plus_source = validate_source_evaluation(
        adapter, plus, field, need_fixed_field_forces=False
    ).source
    minus_source = validate_source_evaluation(
        adapter, minus, field, need_fixed_field_forces=False
    ).source
    finite_difference = float(
        np.vdot(cotangent, (plus_source - minus_source) / (2.0 * step))
    )
    analytic = float(np.vdot(position_vjp, coordinate_direction))
    assert analytic == pytest.approx(finite_difference, rel=3.0e-4, abs=3.0e-6)

    zero = np.zeros_like(field)
    constant_potential = np.zeros_like(field)
    constant_potential[:, :2] = 0.01
    zero_source = validate_source_evaluation(
        adapter, atoms, zero, need_fixed_field_forces=False
    ).source
    shifted_source = validate_source_evaluation(
        adapter, atoms, constant_potential, need_fixed_field_forces=False
    ).source
    gauge_response_delta = float(np.linalg.norm(shifted_source - zero_source))
    # This is deliberately retained as negative evidence.  The operational
    # continuum fixes the zero-at-infinity gauge, but the checkpoint is not
    # gauge-invariant and cannot be called a common variational field functional.
    assert gauge_response_delta > 1.0e-10
    assert float(np.sum(zero_source[:, 0])) == pytest.approx(0.0, abs=1.0e-12)
    assert float(np.sum(shifted_source[:, 0])) == pytest.approx(0.0, abs=1.0e-12)

    print(
        "ROUTE2_MACEPOL_RADIAL_CANARY="
        + json.dumps(
            {
                "checkpoint_sha256": adapter.provenance.checkpoint_sha256,
                "adapter_configuration_sha256": adapter.configuration_sha256(),
                "field_transform_direct_max_abs_error": transform_error,
                "jvp_vjp_dot_abs_error": abs(jvp_dot - vjp_dot),
                "position_vjp_directional_abs_error": abs(analytic - finite_difference),
                "constant_potential_source_delta_l2": gauge_response_delta,
                "constant_potential_gauge_invariance": False,
                "nonlearned_radial_source_max_abs": float(
                    np.max(np.abs(state.source[:, (1, 5, 6, 7)]))
                ),
                "exact_gto_coupling_available": True,
                "operational_profile_admitted": False,
                "variational_functional_admitted": False,
            },
            sort_keys=True,
        )
    )


def test_official_checkpoint_scalar_first_candidate_is_conjugate_and_disabled():
    base = build_official_mace_polar_1_m_radial_gto_adapter(
        device=os.environ.get("MAPLE_ROUTE2_MACE_DEVICE", "cpu"),
        checkpoint_path=_checkpoint_path(),
    )
    model = MACEPolarVariationalFieldEnergy(base)
    atoms = _water()
    coordinates = model.duality_map.coordinates(atom_count=len(atoms), total_charge=0.0)
    zero = np.zeros(coordinates.reduced_dimension)
    zero_source = model.source_from_energy(atoms, zero, total_charge=0.0)
    original_source = validate_source_evaluation(
        base,
        atoms,
        np.zeros((len(atoms), 8)),
        need_fixed_field_forces=False,
    ).source
    zero_anchor_error = float(np.max(np.abs(zero_source - original_source)))
    assert zero_anchor_error <= 3.0e-12

    rng = np.random.default_rng(20260814)
    reduced = rng.normal(scale=5.0e-4, size=coordinates.reduced_dimension)
    direction = rng.normal(scale=2.0e-3, size=coordinates.reduced_dimension)
    source = model.source_from_energy(atoms, reduced, total_charge=0.0)
    field_direction = model.duality_map.lift_field(
        direction,
        atom_count=len(atoms),
        total_charge=0.0,
    )
    reverse = model.field_space.pair(
        source,
        field_direction,
        atom_count=len(atoms),
    )
    forward = model.energy_directional_derivative(
        atoms,
        reduced,
        direction,
        total_charge=0.0,
    )
    assert reverse == pytest.approx(forward, abs=1.0e-9, rel=1.0e-8)

    cotangent = rng.normal(scale=0.1, size=(len(atoms), 8))
    jvp = model.source_jvp(
        atoms,
        reduced,
        direction,
        total_charge=0.0,
    )
    vjp = model.source_vjp(
        atoms,
        reduced,
        cotangent,
        total_charge=0.0,
    )
    jvp_vjp_error = abs(float(np.vdot(jvp, cotangent)) - float(np.vdot(direction, vjp)))
    assert jvp_vjp_error <= 1.0e-9
    assert np.linalg.norm(source[:, (1, 5, 6, 7)]) > 1.0e-8
    assert model.source_space.total_charge(
        source, atom_count=len(atoms)
    ) == pytest.approx(0.0, abs=2.0e-12)
    assert model.capabilities.enabled_tiers == ()
    assert model.variational_functional_admitted is False

    print(
        "ROUTE2_MACEPOL_VARIATIONAL_CANDIDATE_CANARY="
        + json.dumps(
            {
                "checkpoint_sha256": model.provenance.checkpoint_sha256,
                "model_configuration_sha256": model.configuration_sha256(),
                "field_graph_configuration_sha256": (
                    model._field_graph.configuration_sha256()
                ),
                "zero_anchor_max_abs_error": zero_anchor_error,
                "forward_reverse_ad_abs_error": abs(reverse - forward),
                "jvp_vjp_dot_abs_error": jvp_vjp_error,
                "second_radial_source_l2": float(
                    np.linalg.norm(source[:, (1, 5, 6, 7)])
                ),
                "total_charge_e": float(
                    model.source_space.total_charge(source, atom_count=len(atoms))
                ),
                "enabled_tiers": list(model.capabilities.enabled_tiers),
                "variational_functional_admitted": False,
            },
            sort_keys=True,
        )
    )
