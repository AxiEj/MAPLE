from __future__ import annotations

import os
from pathlib import Path

from ase import Atoms
import numpy as np
import pytest

_ENABLED = os.environ.get("MAPLE_ROUTE2_REAL_AIMNET2") == "1"


def _cartesian_samples(model, continuum, scalar, atoms, steps):
    samples = []
    for step in steps:
        components = []
        for atom in range(len(atoms)):
            for axis in range(3):
                plus = atoms.copy()
                minus = atoms.copy()
                plus.positions[atom, axis] += step
                minus.positions[atom, axis] -= step
                components.append(
                    {
                        "atom": atom,
                        "axis": axis,
                        "plus_energy_eV": scalar.evaluate_energy(plus),
                        "minus_energy_eV": scalar.evaluate_energy(minus),
                        "plus_model_topology": model.neighbor_topology(plus).as_dict(),
                        "minus_model_topology": model.neighbor_topology(
                            minus
                        ).as_dict(),
                        "plus_continuum_topology": continuum.topology_state(plus),
                        "minus_continuum_topology": continuum.topology_state(minus),
                    }
                )
        samples.append({"step_A": step, "components": components})
    return samples


@pytest.mark.skipif(
    not _ENABLED,
    reason="set MAPLE_ROUTE2_REAL_AIMNET2=1 for the real checkpoint canary",
)
def test_reconstructed_float64_first_order_energy_uses_smooth_parity_gated_dftd3():
    checkpoint_raw = os.environ.get("MAPLE_ROUTE2_AIMNET2_CHECKPOINT")
    assert checkpoint_raw, (
        "MAPLE_ROUTE2_REAL_AIMNET2=1 requires " "MAPLE_ROUTE2_AIMNET2_CHECKPOINT"
    )
    checkpoint = Path(checkpoint_raw).resolve()
    assert checkpoint.is_file(), f"missing AIMNet2 checkpoint: {checkpoint}"

    torch = pytest.importorskip("torch")
    pytest.importorskip("aimnet")
    from maple.function.calculator.aimnet._aimnet2_float64_source import (
        AIMNET_FIRST_ORDER_ENERGY_PARITY_ABSOLUTE_TOLERANCE_EV,
        AIMNET_FLOAT64_RUNTIME_VERSION,
        AIMNET_SECOND_ORDER_CHARGE_PARITY_ABSOLUTE_TOLERANCE_E,
        AIMNET_SECOND_ORDER_CHARGE_VJP_PARITY_ABSOLUTE_TOLERANCE_EV_PER_A,
        AIMNET_SECOND_ORDER_GRADIENT_PARITY_ABSOLUTE_TOLERANCE_EV_PER_A,
        AIMNet2ReconstructedFloat64SourceCalculator,
    )
    from maple.solvation.release import (
        aimnet2_geometry_mediated_pes_molecule,
        panel_directions,
        panel_geometries,
    )

    torch.set_num_threads(1)
    molecule = aimnet2_geometry_mediated_pes_molecule(5)
    atoms = panel_geometries(molecule)["reference"]
    direction = panel_directions(atoms, molecule.molecule_id)["radial-internal"]
    calculator = AIMNet2ReconstructedFloat64SourceCalculator(
        model_path=checkpoint,
        device="cpu",
    )
    provenance = calculator.runtime_provenance()
    assert AIMNET_FLOAT64_RUNTIME_VERSION == "aimnet-reconstructed-float64-runtime-v3"
    assert provenance["first_order_coordinate_graph"] == (
        "frozen-deep-copy-with-embedded-dftd3-replaced-by-identity; "
        "source-bound-upstream-dftd3-reapplied-with-hessian-true"
    )
    assert provenance["ordinary_forward_role"] == (
        "per-geometry energy-charge-gradient-vjp parity oracle only"
    )

    response = calculator.charge_position_response(atoms, np.zeros(len(atoms)))
    analytic = float(
        np.sum(response.intrinsic_energy_gradient_ev_per_angstrom * direction)
    )
    errors = []
    for step in (4.0e-4, 2.0e-4, 1.0e-4):
        plus = atoms.copy()
        minus = atoms.copy()
        plus.positions += step * direction
        minus.positions -= step * direction
        finite_difference = (
            calculator.charge_state(plus).energy_ev
            - calculator.charge_state(minus).energy_ev
        ) / (2.0 * step)
        errors.append(abs(finite_difference - analytic))
    assert errors[1] <= 0.35 * errors[0]
    assert errors[2] <= 0.35 * errors[1]
    assert errors[2] <= 1.0e-6

    parity = calculator.last_ordinary_decomposed_parity()
    assert parity["energy_absolute_error_eV"] <= (
        AIMNET_FIRST_ORDER_ENERGY_PARITY_ABSOLUTE_TOLERANCE_EV
    )
    assert parity["charge_max_absolute_error_e"] <= (
        AIMNET_SECOND_ORDER_CHARGE_PARITY_ABSOLUTE_TOLERANCE_E
    )
    assert parity["intrinsic_gradient_max_absolute_error_eV_per_A"] <= (
        AIMNET_SECOND_ORDER_GRADIENT_PARITY_ABSOLUTE_TOLERANCE_EV_PER_A
    )
    assert parity["charge_vjp_max_absolute_error_eV_per_A"] <= (
        AIMNET_SECOND_ORDER_CHARGE_VJP_PARITY_ABSOLUTE_TOLERANCE_EV_PER_A
    )


@pytest.mark.skipif(
    not _ENABLED,
    reason="set MAPLE_ROUTE2_REAL_AIMNET2=1 for the real checkpoint canary",
)
def test_real_aimnet2_pyddx_geometry_mediated_directional_derivative():
    checkpoint_raw = os.environ.get("MAPLE_ROUTE2_AIMNET2_CHECKPOINT")
    assert checkpoint_raw, (
        "MAPLE_ROUTE2_REAL_AIMNET2=1 requires " "MAPLE_ROUTE2_AIMNET2_CHECKPOINT"
    )
    checkpoint = Path(checkpoint_raw).resolve()
    assert checkpoint.is_file(), f"missing AIMNet2 checkpoint: {checkpoint}"

    torch = pytest.importorskip("torch")
    pytest.importorskip("pyddx")
    from maple.function.calculator.aimnet._aimnet2_calculator import (
        AIMNet2Calculator,
    )
    from maple.function.calculator.extra_correction.implicit.smd_cds import (
        route2_coulomb_radii,
    )
    from maple.function.route2_smd_profiles import DDPCM_MULTISOLVENT_SMD_PROFILE
    from maple.function.route2_solvents import route2_solvent_spec
    from maple.solvation.continuum.atomic_l1_pyddx import (
        AtomicL1PyDDXPCMBackend,
    )
    from maple.solvation.coupling.geometry_mediated import (
        GeometryMediatedElectrostaticScalar,
    )
    from maple.solvation.models.aimnet2 import (
        AIMNET2_WB97M_D3_CHECKPOINT_SHA256,
        AIMNet2GeometryMediatedModelAdapter,
    )
    from maple.solvation.release.geometry_mediated import (
        GEOMETRY_MEDIATED_COORDINATE_STEPS_A,
        geometry_mediated_admission_decision,
        geometry_mediated_coordinate_direction,
        geometry_mediated_rotations,
        summarize_geometry_mediated_cartesian_audit,
        summarize_geometry_mediated_directional_audit,
        summarize_geometry_mediated_rotation_audit,
    )

    torch.set_num_threads(1)
    atoms = Atoms(
        "OHH",
        positions=[
            [0.0, 0.0, 0.0],
            [0.9572, 0.0, 0.0],
            [-0.2399872, 0.927297, 0.0],
        ],
        info={"charge": 0, "mult": 1},
    )
    calculator = AIMNet2Calculator(
        device=torch.device("cpu"),
        model="aimnet2",
        model_path=str(checkpoint),
        coulomb_method="simple",
    )
    model = AIMNet2GeometryMediatedModelAdapter(calculator)
    assert model.provenance.checkpoint_sha256 == AIMNET2_WB97M_D3_CHECKPOINT_SHA256
    radii = route2_coulomb_radii(
        atoms.get_chemical_symbols(),
        solvent="water",
        profile=DDPCM_MULTISOLVENT_SMD_PROFILE,
    )
    dielectric = float(route2_solvent_spec("water").descriptors.dielectric)
    continuum = AtomicL1PyDDXPCMBackend(
        atoms,
        radii,
        dielectric=dielectric,
        lmax=7,
        n_lebedev=302,
        solver_tolerance=1.0e-12,
    )
    scalar = GeometryMediatedElectrostaticScalar(model, continuum)
    result = scalar.evaluate(atoms)

    assert result.reciprocity_audit.gate_passed is True
    assert result.reciprocity_audit.charge_gauge_vjp_norm_eV_per_A <= 1.0e-7

    direction = geometry_mediated_coordinate_direction(len(atoms))
    samples = []
    for step in GEOMETRY_MEDIATED_COORDINATE_STEPS_A:
        plus = atoms.copy()
        minus = atoms.copy()
        plus.positions += step * direction
        minus.positions -= step * direction
        samples.append(
            {
                "step_A": step,
                "plus_energy_eV": scalar.evaluate_energy(plus),
                "minus_energy_eV": scalar.evaluate_energy(minus),
                "plus_model_topology": model.neighbor_topology(plus).as_dict(),
                "minus_model_topology": model.neighbor_topology(minus).as_dict(),
                "plus_continuum_topology": continuum.topology_state(plus),
                "minus_continuum_topology": continuum.topology_state(minus),
            }
        )
    directional = summarize_geometry_mediated_directional_audit(
        analytic_gradient_eV_per_A=result.total_gradient_eV_per_A,
        direction=direction,
        center_model_topology=model.neighbor_topology(atoms).as_dict(),
        center_continuum_topology=continuum.topology_state(atoms),
        samples=samples,
        reciprocity_audit=result.reciprocity_audit.as_dict(),
    )
    assert directional["topology"]["all_stencils_same_stratum"] is True
    assert directional["reciprocity_metric_charge_gauge_gate_passed"] is True
    assert directional["gate_passed"] is False
    assert len(directional["records"]) == 3
    assert result.reciprocity_gradient_error_eV_per_source_unit <= 1.0e-10
    assert np.max(np.abs(result.continuum_fixed_source_gradient_eV_per_A)) > 0.0
    assert np.max(np.abs(result.source_response_gradient_eV_per_A)) > 0.0
    assert scalar.continuum.fixed_topology is False
    assert scalar.continuum.capabilities.enabled_tiers == ()

    cartesian = summarize_geometry_mediated_cartesian_audit(
        analytic_gradient_eV_per_A=result.total_gradient_eV_per_A,
        center_model_topology=model.neighbor_topology(atoms).as_dict(),
        center_continuum_topology=continuum.topology_state(atoms),
        samples=_cartesian_samples(
            model,
            continuum,
            scalar,
            atoms,
            GEOMETRY_MEDIATED_COORDINATE_STEPS_A,
        ),
        reciprocity_audit=result.reciprocity_audit.as_dict(),
    )
    assert cartesian["gate_passed"] is False

    rotations = []
    for rotation in geometry_mediated_rotations():
        rotated = atoms.copy()
        rotated.positions = atoms.positions @ rotation.T
        rotated_result = scalar.evaluate(rotated)
        rotations.append(
            {
                "rotation_matrix": rotation.tolist(),
                "energy_eV": rotated_result.energy.total_energy_eV,
                "forces_eV_per_A": rotated_result.forces_eV_per_A.tolist(),
                "source": rotated_result.source.tolist(),
                "model_topology": model.neighbor_topology(rotated).as_dict(),
                "continuum_topology": continuum.topology_state(rotated),
            }
        )
    rotation_audit = summarize_geometry_mediated_rotation_audit(
        positions_A=atoms.positions,
        base_energy_eV=result.energy.total_energy_eV,
        base_forces_eV_per_A=result.forces_eV_per_A,
        base_source=result.source,
        base_model_topology=model.neighbor_topology(atoms).as_dict(),
        base_continuum_topology=continuum.topology_state(atoms),
        rotation_records=rotations,
    )
    assert rotation_audit["all_rotation_topologies_match"] is False
    assert rotation_audit["gate_passed"] is False
    decision = geometry_mediated_admission_decision(
        deterministic_replay_passed=True,
        directional_audit=directional,
        cartesian_audit=cartesian,
        rotation_audit=rotation_audit,
        post_solve_residual_available=False,
    )
    assert decision["local_diagnostic_gates_passed"] is False
    assert decision["public_energy_admitted"] is False
    assert decision["public_force_admitted"] is False
    assert decision["opt_admitted"] is False


@pytest.mark.skipif(
    not _ENABLED,
    reason="set MAPLE_ROUTE2_REAL_AIMNET2=1 for the real checkpoint canary",
)
@pytest.mark.parametrize(
    ("runtime_kind", "expected_force_gate"),
    (
        ("legacy-jit-float32", False),
        ("reconstructed-python-float64", True),
    ),
)
def test_real_aimnet2_point_harmonic_precision_and_rotation_gates(
    runtime_kind,
    expected_force_gate,
):
    checkpoint_raw = os.environ.get("MAPLE_ROUTE2_AIMNET2_CHECKPOINT")
    assert checkpoint_raw, (
        "MAPLE_ROUTE2_REAL_AIMNET2=1 requires " "MAPLE_ROUTE2_AIMNET2_CHECKPOINT"
    )
    checkpoint = Path(checkpoint_raw).resolve()
    assert checkpoint.is_file(), f"missing AIMNet2 checkpoint: {checkpoint}"

    torch = pytest.importorskip("torch")
    from maple.function.calculator.aimnet._aimnet2_calculator import (
        AIMNet2Calculator,
    )
    from maple.function.calculator.extra_correction.implicit.smd_cds import (
        route2_coulomb_radii,
    )
    from maple.function.route2_smd_profiles import DDPCM_MULTISOLVENT_SMD_PROFILE
    from maple.solvation.api import (
        DIAGNOSTIC_AIMNET2_GEOMETRY_MEDIATED_SMOOTH_HARMONIC_CPCM_ELECTROSTATIC_V1,
        DIAGNOSTIC_AIMNET2_GEOMETRY_MEDIATED_SMOOTH_HARMONIC_CPCM_PROFILE_V1,
    )
    from maple.solvation.continuum import (
        SmoothPointChargeHarmonicGalerkinFunctionalCandidate,
    )
    from maple.solvation.coupling.geometry_mediated import (
        GeometryMediatedElectrostaticScalar,
    )
    from maple.solvation.models.aimnet2 import (
        AIMNET2_WB97M_D3_RECONSTRUCTED_FLOAT64_CONTRACT,
        AIMNet2GeometryMediatedModelAdapter,
    )
    from maple.solvation.release.geometry_mediated import (
        GEOMETRY_MEDIATED_COORDINATE_STEPS_A,
        geometry_mediated_admission_decision,
        geometry_mediated_coordinate_direction,
        geometry_mediated_rotations,
        summarize_geometry_mediated_cartesian_audit,
        summarize_geometry_mediated_directional_audit,
        summarize_geometry_mediated_rotation_audit,
    )

    torch.set_num_threads(1)
    atoms = Atoms(
        "OHH",
        positions=[
            [0.0, 0.0, 0.0],
            [0.9572, 0.0, 0.0],
            [-0.2399872, 0.927297, 0.0],
        ],
        info={"charge": 0, "mult": 1},
    )
    if runtime_kind == "legacy-jit-float32":
        calculator = AIMNet2Calculator(
            device=torch.device("cpu"),
            model="aimnet2",
            model_path=str(checkpoint),
            coulomb_method="simple",
        )
        model = AIMNet2GeometryMediatedModelAdapter(calculator)
    else:
        pytest.importorskip("aimnet")
        from maple.function.calculator.aimnet._aimnet2_float64_source import (
            AIMNET_REQUIRED_FILE_SHA256,
            AIMNet2ReconstructedFloat64SourceCalculator,
        )

        calculator = AIMNet2ReconstructedFloat64SourceCalculator(
            model_path=checkpoint,
            device="cpu",
        )
        model = AIMNet2GeometryMediatedModelAdapter(
            calculator,
            AIMNET2_WB97M_D3_RECONSTRUCTED_FLOAT64_CONTRACT,
        )
        runtime_provenance = calculator.runtime_provenance()
        assert runtime_provenance["coordinate_dtype"] == "torch.float64"
        assert runtime_provenance["aimnet_runtime_files_sha256"] == dict(
            sorted(AIMNET_REQUIRED_FILE_SHA256.items())
        )
        with pytest.raises(NotImplementedError, match="not a public ASE calculator"):
            calculator.calculate(atoms)

        # Loading the state dictionary is exact; this cross-runtime comparison
        # separately bounds the expected float32/float64 numerical difference.
        legacy = AIMNet2Calculator(
            device=torch.device("cpu"),
            model="aimnet2",
            model_path=str(checkpoint),
            coulomb_method="simple",
        )
        cotangent = np.asarray([-0.3, 0.1, 0.2])
        legacy_response = legacy.charge_position_response(atoms, cotangent)
        reconstructed_response = calculator.charge_position_response(atoms, cotangent)
        assert reconstructed_response.charge_state.energy_ev == pytest.approx(
            legacy_response.charge_state.energy_ev, abs=2.0e-6
        )
        np.testing.assert_allclose(
            reconstructed_response.charge_state.charges_e,
            legacy_response.charge_state.charges_e,
            rtol=0.0,
            atol=2.0e-7,
        )
        np.testing.assert_allclose(
            reconstructed_response.intrinsic_energy_gradient_ev_per_angstrom,
            legacy_response.intrinsic_energy_gradient_ev_per_angstrom,
            rtol=0.0,
            atol=3.0e-6,
        )
        np.testing.assert_allclose(
            reconstructed_response.charge_position_vjp_ev_per_angstrom,
            legacy_response.charge_position_vjp_ev_per_angstrom,
            rtol=0.0,
            atol=5.0e-7,
        )
    radii = route2_coulomb_radii(
        atoms.get_chemical_symbols(),
        solvent="water",
        profile=DDPCM_MULTISOLVENT_SMD_PROFILE,
    )
    continuum = SmoothPointChargeHarmonicGalerkinFunctionalCandidate(
        atomic_numbers=tuple(int(value) for value in atoms.numbers),
        radii_angstrom=tuple(float(value) for value in radii),
        transition_width_angstrom2=0.18,
        surface_lmax=1,
        exposure_lmax=2,
        exposure_radial_quadrature_order=32,
        green_radial_quadrature_order=32,
        dtype=torch.float64,
        device="cpu",
    )
    scalar = GeometryMediatedElectrostaticScalar(
        model,
        continuum,
        scalar_id=(
            DIAGNOSTIC_AIMNET2_GEOMETRY_MEDIATED_SMOOTH_HARMONIC_CPCM_ELECTROSTATIC_V1
        ),
        profile_id=(
            DIAGNOSTIC_AIMNET2_GEOMETRY_MEDIATED_SMOOTH_HARMONIC_CPCM_PROFILE_V1
        ),
    )
    result = scalar.evaluate(atoms)
    assert result.reciprocity_audit.gate_passed is True

    # The continuum itself is structurally covariant for an invariant point-l0
    # source.  The full audit below additionally measures the selected AIMNet2
    # numerical graph rather than hiding its finite-precision frame drift.
    continuum_energy = continuum.energy_eV(atoms, result.source)
    continuum_gradient = continuum.coordinate_partial(atoms, result.source)
    rotation_records = []
    for rotation in geometry_mediated_rotations():
        rotated = atoms.copy()
        rotated.positions = atoms.positions @ rotation.T
        assert continuum.energy_eV(rotated, result.source) == pytest.approx(
            continuum_energy, abs=2.0e-13
        )
        np.testing.assert_allclose(
            continuum.coordinate_partial(rotated, result.source),
            continuum_gradient @ rotation.T,
            atol=5.0e-13,
            rtol=0.0,
        )
        rotated_result = scalar.evaluate(rotated)
        rotation_records.append(
            {
                "rotation_matrix": rotation.tolist(),
                "energy_eV": rotated_result.energy.total_energy_eV,
                "forces_eV_per_A": rotated_result.forces_eV_per_A.tolist(),
                "source": rotated_result.source.tolist(),
                "model_topology": model.neighbor_topology(rotated).as_dict(),
                "continuum_topology": continuum.topology_state(rotated),
            }
        )
    rotation = summarize_geometry_mediated_rotation_audit(
        positions_A=atoms.positions,
        base_energy_eV=result.energy.total_energy_eV,
        base_forces_eV_per_A=result.forces_eV_per_A,
        base_source=result.source,
        base_model_topology=model.neighbor_topology(atoms).as_dict(),
        base_continuum_topology=continuum.topology_state(atoms),
        rotation_records=rotation_records,
    )
    assert rotation["all_rotation_topologies_match"] is True
    assert rotation["sphere_tangency_guard_applicable"] is True
    assert rotation["all_sphere_tangency_margins_available"] is True
    assert rotation["gate_passed"] is True

    direction = geometry_mediated_coordinate_direction(len(atoms))
    samples = []
    for step in GEOMETRY_MEDIATED_COORDINATE_STEPS_A:
        plus = atoms.copy()
        minus = atoms.copy()
        plus.positions += step * direction
        minus.positions -= step * direction
        samples.append(
            {
                "step_A": step,
                "plus_energy_eV": scalar.evaluate_energy(plus),
                "minus_energy_eV": scalar.evaluate_energy(minus),
                "plus_model_topology": model.neighbor_topology(plus).as_dict(),
                "minus_model_topology": model.neighbor_topology(minus).as_dict(),
                "plus_continuum_topology": continuum.topology_state(plus),
                "minus_continuum_topology": continuum.topology_state(minus),
            }
        )
    directional = summarize_geometry_mediated_directional_audit(
        analytic_gradient_eV_per_A=result.total_gradient_eV_per_A,
        direction=direction,
        center_model_topology=model.neighbor_topology(atoms).as_dict(),
        center_continuum_topology=continuum.topology_state(atoms),
        samples=samples,
        reciprocity_audit=result.reciprocity_audit.as_dict(),
    )
    assert directional["topology"]["all_stencils_same_stratum"] is True
    assert directional["topology"]["continuum_event_guard_applicable"] is True
    assert directional["topology"]["all_continuum_event_margins_available"] is True
    assert directional["topology"]["sphere_tangency_guard_applicable"] is True
    assert directional["topology"]["all_sphere_tangency_margins_available"] is True
    assert directional["topology"]["all_sphere_tangency_guards_passed"] is True
    assert directional["gate_passed"] is expected_force_gate

    cartesian = summarize_geometry_mediated_cartesian_audit(
        analytic_gradient_eV_per_A=result.total_gradient_eV_per_A,
        center_model_topology=model.neighbor_topology(atoms).as_dict(),
        center_continuum_topology=continuum.topology_state(atoms),
        samples=_cartesian_samples(
            model,
            continuum,
            scalar,
            atoms,
            GEOMETRY_MEDIATED_COORDINATE_STEPS_A,
        ),
        reciprocity_audit=result.reciprocity_audit.as_dict(),
    )
    assert cartesian["topology"]["continuum_event_guard_applicable"] is True
    assert cartesian["topology"]["all_continuum_event_margins_available"] is True
    assert cartesian["topology"]["sphere_tangency_guard_applicable"] is True
    assert cartesian["topology"]["all_sphere_tangency_margins_available"] is True
    assert cartesian["topology"]["all_sphere_tangency_guards_passed"] is True
    assert cartesian["gate_passed"] is expected_force_gate

    decision = geometry_mediated_admission_decision(
        deterministic_replay_passed=True,
        directional_audit=directional,
        cartesian_audit=cartesian,
        rotation_audit=rotation,
        post_solve_residual_available=True,
    )
    assert decision["rotation_topology_gate_passed"] is True
    assert decision["local_diagnostic_gates_passed"] is expected_force_gate
    assert decision["tier_f_prerequisites_passed"] is expected_force_gate
    assert decision["public_energy_admitted"] is False
    assert decision["public_force_admitted"] is False
    assert decision["opt_admitted"] is False
    if runtime_kind != "reconstructed-python-float64":
        return

    from maple.solvation.release.geometry_mediated_path import (
        aimnet2_geometry_mediated_water_loop_atoms,
        aimnet2_geometry_mediated_water_loop_directions,
    )

    hvp_atoms = aimnet2_geometry_mediated_water_loop_atoms((0.0, 0.0))
    first_direction, second_direction = (
        aimnet2_geometry_mediated_water_loop_directions()
    )
    first_hvp = scalar.hessian_vector_product(hvp_atoms, first_direction)
    assert first_hvp.diagnostic_only is True
    assert first_hvp.tier_h_admitted is False
    assert first_hvp.model_standard_decomposed_energy_absolute_error_eV <= 1.0e-8
    assert first_hvp.model_standard_decomposed_charge_max_absolute_error_e <= 1.0e-10
    assert (
        first_hvp.model_standard_decomposed_intrinsic_gradient_max_absolute_error_eV_per_A
        <= 1.0e-7
    )
    assert (
        first_hvp.model_standard_decomposed_charge_vjp_max_absolute_error_eV_per_A
        <= 1.0e-10
    )
    assert first_hvp.model_charge_tangent_residual_e_per_A <= 1.0e-10
    assert abs(float(np.sum(first_hvp.source_position_jvp[:, 0]))) <= 1.0e-10
    np.testing.assert_array_equal(first_hvp.source_position_jvp[:, 1:], 0.0)
    assert np.linalg.norm(first_hvp.intrinsic_energy_hvp_eV_per_A2) > 0.0
    assert np.linalg.norm(first_hvp.continuum_joint_position_hvp_eV_per_A2) > 0.0
    assert np.linalg.norm(first_hvp.continuum_joint_source_hvp) > 0.0
    assert np.linalg.norm(first_hvp.continuum_source_response_pullback_eV_per_A2) > 0.0
    assert np.linalg.norm(first_hvp.contracted_source_hessian_eV_per_A2) > 0.0

    fixed_charge_cotangent = np.asarray(
        first_hvp.source_gradient_cotangent[:, 0], dtype=float
    )
    model_second_order = calculator.charge_position_second_order(
        hvp_atoms,
        fixed_charge_cotangent,
        first_direction,
    )
    charge_jvp_errors = []
    contracted_charge_hessian_errors = []
    full_hvp_errors = []
    hvp_steps = (4.0e-4, 2.0e-4, 1.0e-4)
    center_model_topology = model.neighbor_topology(hvp_atoms)
    center_continuum_topology = continuum.topology_state(hvp_atoms)
    for step in hvp_steps:
        plus = hvp_atoms.copy()
        minus = hvp_atoms.copy()
        plus.positions += step * first_direction
        minus.positions -= step * first_direction
        assert model.neighbor_topology(plus).topology_sha256 == (
            center_model_topology.topology_sha256
        )
        assert model.neighbor_topology(minus).topology_sha256 == (
            center_model_topology.topology_sha256
        )
        assert continuum.topology_state(plus)["cavity_topology_sha256"] == (
            center_continuum_topology["cavity_topology_sha256"]
        )
        assert continuum.topology_state(minus)["cavity_topology_sha256"] == (
            center_continuum_topology["cavity_topology_sha256"]
        )

        charge_jvp_fd = (
            calculator.charge_state(plus).charges_e
            - calculator.charge_state(minus).charges_e
        ) / (2.0 * step)
        charge_jvp_errors.append(
            float(
                np.linalg.norm(
                    charge_jvp_fd
                    - model_second_order.charge_position_jvp_e_per_angstrom
                )
            )
        )
        contracted_hessian_fd = (
            calculator.charge_position_response(
                plus, fixed_charge_cotangent
            ).charge_position_vjp_ev_per_angstrom
            - calculator.charge_position_response(
                minus, fixed_charge_cotangent
            ).charge_position_vjp_ev_per_angstrom
        ) / (2.0 * step)
        contracted_charge_hessian_errors.append(
            float(
                np.linalg.norm(
                    contracted_hessian_fd
                    - model_second_order.contracted_charge_hessian_ev_per_angstrom2
                )
            )
        )
        full_hvp_fd = (
            scalar.evaluate(plus).total_gradient_eV_per_A
            - scalar.evaluate(minus).total_gradient_eV_per_A
        ) / (2.0 * step)
        full_hvp_errors.append(
            float(np.linalg.norm(full_hvp_fd - first_hvp.total_hvp_eV_per_A2))
        )

    for errors in (
        charge_jvp_errors,
        contracted_charge_hessian_errors,
        full_hvp_errors,
    ):
        assert errors[1] < 0.45 * errors[0]
        assert errors[2] < 0.45 * errors[1]
    assert full_hvp_errors[-1] <= 5.0e-5

    second_hvp = scalar.hessian_vector_product(hvp_atoms, second_direction)
    left = float(np.vdot(first_direction, second_hvp.total_hvp_eV_per_A2))
    right = float(np.vdot(second_direction, first_hvp.total_hvp_eV_per_A2))
    assert left == pytest.approx(right, abs=2.0e-8)

    translation = np.ones_like(first_direction)
    translation /= np.linalg.norm(translation)
    translation_hvp = scalar.hessian_vector_product(hvp_atoms, translation)
    assert np.linalg.norm(translation_hvp.source_position_jvp) <= 1.0e-10
    assert np.linalg.norm(translation_hvp.total_hvp_eV_per_A2) <= 2.0e-8
    assert scalar.continuum.capabilities.hessian is False
    assert model.variational_functional_admitted is False
    with pytest.raises(NotImplementedError, match="does not expose public HVPs"):
        calculator.get_hvp(hvp_atoms, first_direction.reshape(-1))
