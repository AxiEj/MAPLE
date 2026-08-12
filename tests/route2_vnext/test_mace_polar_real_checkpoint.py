from __future__ import annotations

import json
import os

import numpy as np
import pytest
from ase import Atoms

from maple.solvation.models import (
    build_official_mace_polar_1_m_adapter,
    validate_response_linearization,
    validate_source_evaluation,
    validate_vacuum_evaluation,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("MAPLE_ROUTE2_REAL_MACEPOL") != "1",
    reason="set MAPLE_ROUTE2_REAL_MACEPOL=1 in the pinned real-checkpoint job",
)


def test_official_mace_polar_checkpoint_sign_nonuniform_response_and_derivatives():
    adapter = build_official_mace_polar_1_m_adapter(
        device=os.environ.get("MAPLE_ROUTE2_MACE_DEVICE", "cpu")
    )
    atoms = Atoms(
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
    # This is negative evidence against treating the field-conditioned energy
    # as the operational scalar, not a reason to patch or fit either branch.
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
