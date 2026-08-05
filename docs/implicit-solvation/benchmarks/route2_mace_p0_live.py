"""Live, label-free diagnostics for the MACE-POLAR P0 audit runner.

This helper keeps the executable focused on provenance and artifact assembly.
It uses the established MAPLE calculator, canonical-frame, pairing, and
response-linearization implementations rather than reproducing them.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import ase
from mace import data as mace_data
from mace.tools import torch_geometric, torch_tools
import numpy as np
from scipy.spatial.transform import Rotation
import torch

from maple.function.calculator.extra_correction.implicit.gto_density import (
    cartesian_multipoles,
)
from maple.function.calculator.extra_correction.implicit.route2_mace_p0 import (
    select_uniform_field_convention,
)
from maple.function.calculator.extra_correction.implicit.route2_thermodynamic_diagnostics import (
    response_reciprocity_diagnostic,
    response_stability_diagnostic,
)
from maple.function.calculator.mace._macepol_calculator import (
    MACEPolCalculator,
)
from maple.function.route2_smd_profiles import (
    MACEPOL_MOLECULAR_REALSPACE_PROFILE,
)

MappingLike = dict[str, Any]

COORDINATE_FRAME_POLICY = "jgp94-d2-canonical-v1"
MODEL_FIELD_EVALUATOR = MACEPOL_MOLECULAR_REALSPACE_PROFILE
RANDOM_SEED = 20260805
UNIFORM_FIELD_VECTORS_EV_PER_E_ANGSTROM = (
    (1.0e-3, 0.0, 0.0),
    (0.0, 1.0e-3, 0.0),
    (0.0, 0.0, 1.0e-3),
    (7.0e-4, -1.1e-3, 9.0e-4),
)
TRANSLATION_ANGSTROM = (2.2, -1.3, 0.7)
ROTATION_VECTOR_RADIANS = (0.37, -0.21, 0.44)
SECOND_BATCH_ROTATION_VECTOR_RADIANS = (0.23, 0.31, -0.17)
SECOND_BATCH_FIELD_EV_PER_E_ANGSTROM = (-2.0e-3, 1.0e-3, 7.0e-4)
CONSTANT_POTENTIAL_GAUGE_PROBE_EV_PER_E = 1.0e-3

INTERFACE_ABSOLUTE_TOLERANCE = 5.0e-10
INTERFACE_MINIMUM_DISCRIMINATION_RATIO = 100.0
ENERGY_IDENTITY_TOLERANCE_EV = 5.0e-10
CANONICAL_ENERGY_TOLERANCE_EV = 2.0e-9
CANONICAL_STATE_TOLERANCE = 2.0e-9
BATCH_ISOLATION_TOLERANCE = 2.0e-9
AUTOGRAD_FINITE_DIFFERENCE_STEP = 1.0e-2
AUTOGRAD_FINITE_DIFFERENCE_TOLERANCE_EV_PER_FIELD = 2.0e-6
ENERGY_DIPOLE_RELATIVE_TOLERANCE = 1.0e-4
INDUCED_DIPOLE_FINITE_DIFFERENCE_STEP = 1.0e-3
INDUCED_DIPOLE_FINITE_DIFFERENCE_TOLERANCE = 2.0e-5
INDUCED_DIPOLE_RECIPROCITY_RELATIVE_TOLERANCE = 1.0e-6
INDUCED_DIPOLE_PASSIVITY_TOLERANCE = 1.0e-8
INDUCED_DIPOLE_MINIMUM_RANK_EIGENVALUE = 1.0e-8
SOURCE_RESPONSE_RECIPROCITY_RELATIVE_TOLERANCE = 1.0e-6
SOURCE_RESPONSE_PASSIVITY_TOLERANCE_EV = 1.0e-8
SOURCE_RESPONSE_ADJOINT_TOLERANCE = 1.0e-10


def _sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def _as_numpy(value: Any) -> np.ndarray:
    if not torch.is_tensor(value):
        raise RuntimeError("MACE-POLAR audit expected a tensor output.")
    result = np.asarray(value.detach().cpu(), dtype=float)
    if not np.all(np.isfinite(result)):
        raise RuntimeError("MACE-POLAR audit received a non-finite tensor.")
    return result


def _uniform_node_field(
    atoms: ase.Atoms,
    upstream_field: np.ndarray,
    *,
    sign: float = 1.0,
) -> np.ndarray:
    positions = np.asarray(atoms.positions, dtype=float)
    gradient = sign * np.asarray(upstream_field, dtype=float)
    centered = positions - np.mean(positions, axis=0, keepdims=True)
    return np.column_stack(
        (centered @ gradient, np.broadcast_to(gradient, (len(atoms), 3)))
    )


def _raw_batch(
    calculator: MACEPolCalculator,
    atoms: ase.Atoms,
    *,
    upstream_field: np.ndarray | None = None,
) -> dict[str, torch.Tensor]:
    model_atoms = calculator._atoms_for_mace(atoms)
    if upstream_field is not None:
        model_atoms.info["external_field"] = np.asarray(
            upstream_field, dtype=float
        ).tolist()
    batch = calculator._mace._atoms_to_batch(model_atoms).to_dict()
    for key, value in tuple(batch.items()):
        if torch.is_tensor(value) and torch.is_floating_point(value):
            batch[key] = value.to(dtype=calculator.dtype)
    return calculator._long_range_evaluator.prepare_batch(batch, r_max=calculator.r_max)


def _raw_upstream_output(
    calculator: MACEPolCalculator,
    atoms: ase.Atoms,
    field: np.ndarray,
) -> dict[str, Any]:
    batch = _raw_batch(calculator, atoms, upstream_field=field)
    with calculator._reaction_projector.use_node_potential_gradient(None):
        return calculator._model_forward(
            batch,
            compute_force=False,
            compute_stress=False,
            compute_hessian=False,
        )


def _raw_local_output(
    calculator: MACEPolCalculator,
    atoms: ase.Atoms,
    node_field: np.ndarray,
) -> dict[str, Any]:
    values = torch.as_tensor(
        node_field, dtype=calculator.dtype, device=calculator.device
    )
    batch = _raw_batch(calculator, atoms)
    with calculator._reaction_projector.use_node_potential_gradient(values):
        return calculator._model_forward(
            batch,
            compute_force=False,
            compute_stress=False,
            compute_hessian=False,
        )


def _intrinsic_output_error(left: MappingLike, right: MappingLike) -> dict[str, float]:
    keys = (
        "density_coefficients",
        "dipole",
        "interaction_energy",
        "electron_energy",
        "electrostatic_energy",
        "total_charge",
    )
    errors = {
        key: float(np.max(np.abs(_as_numpy(left[key]) - _as_numpy(right[key]))))
        for key in keys
    }
    errors["maximum"] = max(errors.values())
    return errors


def _uniform_field_interface(
    calculator: MACEPolCalculator,
    atoms: ase.Atoms,
) -> dict[str, Any]:
    records = []
    positive_errors = []
    negative_errors = []
    energy_identity_errors = []
    for base in UNIFORM_FIELD_VECTORS_EV_PER_E_ANGSTROM:
        for sign in (-1.0, 1.0):
            field = sign * np.asarray(base, dtype=float)
            upstream = _raw_upstream_output(calculator, atoms, field)
            positive = _raw_local_output(
                calculator, atoms, _uniform_node_field(atoms, field, sign=1.0)
            )
            negative = _raw_local_output(
                calculator, atoms, _uniform_node_field(atoms, field, sign=-1.0)
            )
            positive_error = _intrinsic_output_error(upstream, positive)
            negative_error = _intrinsic_output_error(upstream, negative)
            upstream_energy = float(np.sum(_as_numpy(upstream["energy"])))
            local_energy = float(np.sum(_as_numpy(positive["energy"])))
            dipole = _as_numpy(upstream["dipole"]).reshape(3)
            explicit_pairing = float(np.vdot(field, dipole))
            identity_error = abs(upstream_energy - local_energy - explicit_pairing)
            positive_errors.append(positive_error["maximum"])
            negative_errors.append(negative_error["maximum"])
            energy_identity_errors.append(identity_error)
            records.append(
                {
                    "upstream_field": field.tolist(),
                    "positive_grad_phi_mapping_error": positive_error,
                    "negative_grad_phi_mapping_error": negative_error,
                    "upstream_total_energy_ev": upstream_energy,
                    "local_intrinsic_energy_ev": local_energy,
                    "upstream_field_dot_dipole_ev": explicit_pairing,
                    "energy_identity_absolute_error_ev": identity_error,
                }
            )
    convention = select_uniform_field_convention(
        positive_mapping_errors=positive_errors,
        negative_mapping_errors=negative_errors,
        absolute_tolerance=INTERFACE_ABSOLUTE_TOLERANCE,
        minimum_discrimination_ratio=INTERFACE_MINIMUM_DISCRIMINATION_RATIO,
    )
    maximum_selected_error = max(
        positive_errors if convention == "potential-gradient" else negative_errors
    )
    maximum_energy_identity_error = max(energy_identity_errors)
    return {
        "passed": bool(
            maximum_selected_error <= INTERFACE_ABSOLUTE_TOLERANCE
            and maximum_energy_identity_error <= ENERGY_IDENTITY_TOLERANCE_EV
        ),
        "selected_field_convention": convention,
        "physical_electric_field_relation": (
            "E_physical=-grad(phi)=-f_upstream"
            if convention == "potential-gradient"
            else "E_physical=f_upstream=-grad(phi)"
        ),
        "maximum_selected_mapping_error": maximum_selected_error,
        "maximum_opposite_mapping_error": max(
            negative_errors if convention == "potential-gradient" else positive_errors
        ),
        "maximum_energy_identity_absolute_error_ev": (maximum_energy_identity_error),
        "thresholds": {
            "mapping_absolute_max": INTERFACE_ABSOLUTE_TOLERANCE,
            "minimum_discrimination_ratio": (INTERFACE_MINIMUM_DISCRIMINATION_RATIO),
            "energy_identity_absolute_max_ev": ENERGY_IDENTITY_TOLERANCE_EV,
        },
        "records": records,
    }


def _canonical_state(
    calculator: MACEPolCalculator,
    atoms: ase.Atoms,
    field: np.ndarray,
    *,
    constant_potential: float = 0.0,
):
    node_field = _uniform_node_field(atoms, field)
    node_field[:, 0] += float(constant_potential)
    state, _ = calculator.polar_state(
        atoms,
        node_potential_ev=node_field[:, 0],
        node_gradient_ev_per_angstrom=node_field[:, 1:],
    )
    return state


def _canonical_translation_gauge(
    calculator: MACEPolCalculator,
    atoms: ase.Atoms,
) -> dict[str, Any]:
    field = np.asarray(UNIFORM_FIELD_VECTORS_EV_PER_E_ANGSTROM[-1])
    reference = _canonical_state(calculator, atoms, field)
    translated = atoms.copy()
    translated.positions += np.asarray(TRANSLATION_ANGSTROM)
    translated_state = _canonical_state(calculator, translated, field)
    shifted_gauge_state = _canonical_state(
        calculator,
        atoms,
        field,
        constant_potential=CONSTANT_POTENTIAL_GAUGE_PROBE_EV_PER_E,
    )
    energy_error = abs(translated_state.energy_ev - reference.energy_ev)
    source_error = float(
        np.max(
            np.abs(
                translated_state.density_coefficients - reference.density_coefficients
            )
        )
    )
    dipole_error = float(
        np.max(np.abs(translated_state.dipole_e_angstrom - reference.dipole_e_angstrom))
    )
    return {
        "passed": bool(
            energy_error <= CANONICAL_ENERGY_TOLERANCE_EV
            and max(source_error, dipole_error) <= CANONICAL_STATE_TOLERANCE
        ),
        "uniform_field_gauge": "phi(r_i)=(r_i-mean(r))*grad(phi)",
        "translation_angstrom": list(TRANSLATION_ANGSTROM),
        "energy_absolute_error_ev": energy_error,
        "source_maximum_absolute_error": source_error,
        "dipole_maximum_absolute_error_e_angstrom": dipole_error,
        "constant_potential_sensitivity_probe": {
            "shift_ev_per_e": CONSTANT_POTENTIAL_GAUGE_PROBE_EV_PER_E,
            "energy_change_ev": (shifted_gauge_state.energy_ev - reference.energy_ev),
            "source_maximum_change": float(
                np.max(
                    np.abs(
                        shifted_gauge_state.density_coefficients
                        - reference.density_coefficients
                    )
                )
            ),
            "interpretation": (
                "Diagnostic only. P-minus-1 fixes the continuum-zero-at-infinity "
                "gauge and does not quotient arbitrary constant potentials."
            ),
        },
        "thresholds": {
            "energy_absolute_max_ev": CANONICAL_ENERGY_TOLERANCE_EV,
            "state_maximum_absolute_max": CANONICAL_STATE_TOLERANCE,
        },
    }


def _canonical_rotation_covariance(
    calculator: MACEPolCalculator,
    atoms: ase.Atoms,
) -> dict[str, Any]:
    field = np.asarray(UNIFORM_FIELD_VECTORS_EV_PER_E_ANGSTROM[-1])
    reference = _canonical_state(calculator, atoms, field)
    rotation = Rotation.from_rotvec(np.asarray(ROTATION_VECTOR_RADIANS)).as_matrix()
    rotated = atoms.copy()
    rotated.positions = np.asarray(atoms.positions) @ rotation.T
    rotated_field = field @ rotation.T
    rotated_state = _canonical_state(calculator, rotated, rotated_field)
    charges, dipoles = cartesian_multipoles(reference.density_coefficients)
    rotated_charges, rotated_dipoles = cartesian_multipoles(
        rotated_state.density_coefficients
    )
    energy_error = abs(rotated_state.energy_ev - reference.energy_ev)
    charge_error = float(np.max(np.abs(rotated_charges - charges)))
    atomic_dipole_error = float(np.max(np.abs(rotated_dipoles - dipoles @ rotation.T)))
    molecular_dipole_error = float(
        np.max(
            np.abs(
                rotated_state.dipole_e_angstrom
                - reference.dipole_e_angstrom @ rotation.T
            )
        )
    )
    state_error = max(charge_error, atomic_dipole_error, molecular_dipole_error)
    return {
        "passed": bool(
            energy_error <= CANONICAL_ENERGY_TOLERANCE_EV
            and state_error <= CANONICAL_STATE_TOLERANCE
        ),
        "rotation_vector_radians": list(ROTATION_VECTOR_RADIANS),
        "rotation_matrix": rotation.tolist(),
        "energy_absolute_error_ev": energy_error,
        "atomic_charge_maximum_absolute_error_e": charge_error,
        "atomic_dipole_maximum_absolute_error_e_angstrom": atomic_dipole_error,
        "molecular_dipole_maximum_absolute_error_e_angstrom": (molecular_dipole_error),
        "thresholds": {
            "energy_absolute_max_ev": CANONICAL_ENERGY_TOLERANCE_EV,
            "state_maximum_absolute_max": CANONICAL_STATE_TOLERANCE,
        },
    }


def _atomic_graph(calculator: MACEPolCalculator, atoms: ase.Atoms):
    model_atoms = calculator._atoms_for_mace(atoms)
    calculator._mace.arrays_keys.update({calculator._mace.charges_key: "charges"})
    key_specification = mace_data.KeySpecification(
        info_keys=calculator._mace.info_keys,
        arrays_keys=calculator._mace.arrays_keys,
    )
    with torch_tools.default_dtype(calculator._mace.default_dtype):
        config = mace_data.config_from_atoms(
            model_atoms,
            key_specification=key_specification,
            head_name=calculator._mace.head,
        )
        return mace_data.AtomicData.from_config(
            config,
            z_table=calculator._mace.z_table,
            cutoff=calculator._mace.r_max,
            heads=calculator._mace.available_heads,
        )


def _graph_batch(
    calculator: MACEPolCalculator, atoms_list: list[ase.Atoms]
) -> dict[str, torch.Tensor]:
    batch = torch_geometric.Batch.from_data_list(
        [_atomic_graph(calculator, atoms) for atoms in atoms_list]
    ).to(calculator.device)
    result = batch.to_dict()
    for key, value in tuple(result.items()):
        if torch.is_tensor(value) and torch.is_floating_point(value):
            result[key] = value.to(dtype=calculator.dtype)
    return calculator._long_range_evaluator.prepare_batch(
        result, r_max=calculator.r_max
    )


def _batch_isolation(
    calculator: MACEPolCalculator,
    atoms: ase.Atoms,
) -> dict[str, Any]:
    second = atoms.copy()
    second_rotation = Rotation.from_rotvec(
        np.asarray(SECOND_BATCH_ROTATION_VECTOR_RADIANS)
    ).as_matrix()
    second.positions = np.asarray(second.positions) @ second_rotation.T
    fields = (
        np.asarray(UNIFORM_FIELD_VECTORS_EV_PER_E_ANGSTROM[-1]),
        np.asarray(SECOND_BATCH_FIELD_EV_PER_E_ANGSTROM),
    )
    atoms_list = [atoms, second]
    node_fields = [
        _uniform_node_field(current_atoms, field)
        for current_atoms, field in zip(atoms_list, fields, strict=True)
    ]
    isolated = [
        _raw_local_output(calculator, current_atoms, node_field)
        for current_atoms, node_field in zip(atoms_list, node_fields, strict=True)
    ]
    batch = _graph_batch(calculator, atoms_list)
    batched_node_field = torch.as_tensor(
        np.concatenate(node_fields, axis=0),
        dtype=calculator.dtype,
        device=calculator.device,
    )
    with calculator._reaction_projector.use_node_potential_gradient(batched_node_field):
        batched = calculator._model_forward(
            batch,
            compute_force=False,
            compute_stress=False,
            compute_hessian=False,
        )
    batched_energy = _as_numpy(batched["energy"]).reshape(2)
    batched_dipole = _as_numpy(batched["dipole"]).reshape(2, 3)
    batched_source = _as_numpy(batched["density_coefficients"])
    records = []
    start = 0
    for index, output in enumerate(isolated):
        stop = start + len(atoms_list[index])
        energy_error = abs(
            batched_energy[index] - float(np.sum(_as_numpy(output["energy"])))
        )
        dipole_error = float(
            np.max(
                np.abs(batched_dipole[index] - _as_numpy(output["dipole"]).reshape(3))
            )
        )
        source_error = float(
            np.max(
                np.abs(
                    batched_source[start:stop]
                    - _as_numpy(output["density_coefficients"])
                )
            )
        )
        records.append(
            {
                "graph_index": index,
                "energy_absolute_error_ev": energy_error,
                "dipole_maximum_absolute_error_e_angstrom": dipole_error,
                "source_maximum_absolute_error": source_error,
            }
        )
        start = stop
    maximum_error = max(
        value
        for record in records
        for key, value in record.items()
        if key != "graph_index"
    )
    return {
        "passed": bool(maximum_error <= BATCH_ISOLATION_TOLERANCE),
        "graph_count": 2,
        "maximum_absolute_error": maximum_error,
        "absolute_tolerance": BATCH_ISOLATION_TOLERANCE,
        "records": records,
    }


def _external_field_output(
    calculator: MACEPolCalculator,
    batch: dict[str, torch.Tensor],
    field: torch.Tensor,
) -> dict[str, Any]:
    with calculator._reaction_projector.use_node_potential_gradient(None):
        return calculator._model_forward(
            batch,
            compute_force=False,
            compute_stress=False,
            compute_hessian=False,
            external_field=field,
        )


def _autograd_energy_dipole_conjugacy(
    calculator: MACEPolCalculator,
    atoms: ase.Atoms,
) -> dict[str, Any]:
    batch = _raw_batch(calculator, atoms)
    field = torch.zeros(
        (1, 3),
        dtype=calculator.dtype,
        device=calculator.device,
        requires_grad=True,
    )
    output = _external_field_output(calculator, batch, field)
    (gradient,) = torch.autograd.grad(output["energy"].sum(), (field,))
    autograd_gradient = _as_numpy(gradient).reshape(3)
    reported_dipole = _as_numpy(output["dipole"]).reshape(3)
    step = AUTOGRAD_FINITE_DIFFERENCE_STEP
    finite_difference = np.empty(3, dtype=float)
    for axis in range(3):
        energies = []
        for sign in (-1.0, 1.0):
            displaced = torch.zeros(
                (1, 3), dtype=calculator.dtype, device=calculator.device
            )
            displaced[0, axis] = sign * step
            displaced_output = _external_field_output(calculator, batch, displaced)
            energies.append(float(np.sum(_as_numpy(displaced_output["energy"]))))
        finite_difference[axis] = (energies[1] - energies[0]) / (2.0 * step)
    implementation_error = float(np.max(np.abs(autograd_gradient - finite_difference)))
    conjugacy_error = float(np.linalg.norm(autograd_gradient - reported_dipole))
    conjugacy_scale = float(
        np.linalg.norm(autograd_gradient) + np.linalg.norm(reported_dipole)
    )
    relative_error = (
        0.0 if conjugacy_scale <= 1.0e-30 else (conjugacy_error / conjugacy_scale)
    )
    implementation_passed = (
        implementation_error <= AUTOGRAD_FINITE_DIFFERENCE_TOLERANCE_EV_PER_FIELD
    )
    conjugacy_passed = relative_error <= ENERGY_DIPOLE_RELATIVE_TOLERANCE
    return {
        "passed": bool(implementation_passed and conjugacy_passed),
        "autograd_implementation_passed": bool(implementation_passed),
        "energy_dipole_conjugacy_passed": bool(conjugacy_passed),
        "autograd_dE_df_ev_per_field": autograd_gradient.tolist(),
        "finite_difference_dE_df_ev_per_field": finite_difference.tolist(),
        "reported_dipole_e_angstrom": reported_dipole.tolist(),
        "autograd_finite_difference_maximum_error": implementation_error,
        "energy_dipole_conjugacy_absolute_l2": conjugacy_error,
        "energy_dipole_conjugacy_relative_l2": relative_error,
        "thresholds": {
            "finite_difference_step": step,
            "autograd_finite_difference_maximum_error": (
                AUTOGRAD_FINITE_DIFFERENCE_TOLERANCE_EV_PER_FIELD
            ),
            "energy_dipole_relative_l2_max": (ENERGY_DIPOLE_RELATIVE_TOLERANCE),
        },
    }


def _induced_dipole_response(
    calculator: MACEPolCalculator,
    atoms: ase.Atoms,
) -> dict[str, Any]:
    batch = _raw_batch(calculator, atoms)

    def dipole_from_field(field: torch.Tensor) -> torch.Tensor:
        return _external_field_output(calculator, batch, field)["dipole"].reshape(3)

    field = torch.zeros(
        (1, 3),
        dtype=calculator.dtype,
        device=calculator.device,
        requires_grad=True,
    )
    jacobian = (
        torch.autograd.functional.jacobian(dipole_from_field, field)
        .reshape(3, 3)
        .detach()
        .cpu()
        .numpy()
    )
    step = INDUCED_DIPOLE_FINITE_DIFFERENCE_STEP
    finite_difference = np.empty((3, 3), dtype=float)
    for axis in range(3):
        dipoles = []
        for sign in (-1.0, 1.0):
            displaced = torch.zeros(
                (1, 3), dtype=calculator.dtype, device=calculator.device
            )
            displaced[0, axis] = sign * step
            dipoles.append(_as_numpy(dipole_from_field(displaced)).reshape(3))
        finite_difference[:, axis] = (dipoles[1] - dipoles[0]) / (2.0 * step)
    finite_difference_error = float(np.max(np.abs(jacobian - finite_difference)))
    physical_polarizability = -jacobian
    symmetric = 0.5 * (physical_polarizability + physical_polarizability.T)
    antisymmetric = 0.5 * (physical_polarizability - physical_polarizability.T)
    eigenvalues = np.linalg.eigvalsh(symmetric)
    reciprocity_relative = float(
        np.linalg.norm(antisymmetric, ord="fro")
        / max(np.linalg.norm(symmetric, ord="fro"), 1.0e-30)
    )
    passivity_passed = bool(
        float(np.min(eigenvalues)) >= -INDUCED_DIPOLE_PASSIVITY_TOLERANCE
    )
    full_rank = bool(
        float(np.min(eigenvalues)) >= INDUCED_DIPOLE_MINIMUM_RANK_EIGENVALUE
    )
    reciprocity_passed = bool(
        reciprocity_relative <= INDUCED_DIPOLE_RECIPROCITY_RELATIVE_TOLERANCE
    )
    finite_difference_passed = bool(
        finite_difference_error <= INDUCED_DIPOLE_FINITE_DIFFERENCE_TOLERANCE
    )
    return {
        "passed": bool(
            passivity_passed
            and full_rank
            and reciprocity_passed
            and finite_difference_passed
        ),
        "upstream_response_jacobian_dmu_df": jacobian.tolist(),
        "physical_polarizability_minus_dmu_df": physical_polarizability.tolist(),
        "finite_difference_dmu_df": finite_difference.tolist(),
        "finite_difference_maximum_error": finite_difference_error,
        "symmetric_physical_polarizability_eigenvalues": eigenvalues.tolist(),
        "reciprocity_relative_frobenius_error": reciprocity_relative,
        "passivity_passed": passivity_passed,
        "full_rank_passed": full_rank,
        "reciprocity_passed": reciprocity_passed,
        "finite_difference_passed": finite_difference_passed,
        "thresholds": {
            "finite_difference_step": step,
            "finite_difference_maximum_error": (
                INDUCED_DIPOLE_FINITE_DIFFERENCE_TOLERANCE
            ),
            "reciprocity_relative_frobenius_max": (
                INDUCED_DIPOLE_RECIPROCITY_RELATIVE_TOLERANCE
            ),
            "passivity_minimum_eigenvalue": (-INDUCED_DIPOLE_PASSIVITY_TOLERANCE),
            "full_rank_minimum_eigenvalue": (INDUCED_DIPOLE_MINIMUM_RANK_EIGENVALUE),
        },
    }


def _source_response(
    calculator: MACEPolCalculator,
    atoms: ase.Atoms,
) -> dict[str, Any]:
    zero_potential = np.zeros(len(atoms), dtype=float)
    zero_gradient = np.zeros((len(atoms), 3), dtype=float)
    response = calculator.linearize_density_response(
        atoms,
        node_potential_ev=zero_potential,
        node_gradient_ev_per_angstrom=zero_gradient,
    )
    rng = np.random.default_rng(RANDOM_SEED)
    reciprocity_records = []
    implementation_adjoint_records = []
    for index in range(3):
        first = rng.normal(size=(len(atoms), 4))
        second = rng.normal(size=(len(atoms), 4))
        first /= np.linalg.norm(first)
        second /= np.linalg.norm(second)
        reciprocity_records.append(
            {
                "pair_index": index,
                **asdict(response_reciprocity_diagnostic(response, first, second)),
            }
        )
        density_cotangent = rng.normal(size=(len(atoms), 4))
        forward = float(np.vdot(density_cotangent, response.jvp(first)))
        reverse = float(np.vdot(response.vjp(density_cotangent), first))
        absolute = abs(forward - reverse)
        implementation_adjoint_records.append(
            {
                "pair_index": index,
                "forward_dot": forward,
                "reverse_dot": reverse,
                "absolute_error": absolute,
                "passed": bool(absolute <= SOURCE_RESPONSE_ADJOINT_TOLERANCE),
            }
        )
    stability = response_stability_diagnostic(
        response,
        atom_count=len(atoms),
        potential_direction_scale_ev=1.0,
        gradient_direction_scale_ev_per_angstrom=1.0,
        eigenvalue_tolerance_ev=SOURCE_RESPONSE_PASSIVITY_TOLERANCE_EV,
    )
    reciprocity_passed = all(
        record["relative_error"] <= SOURCE_RESPONSE_RECIPROCITY_RELATIVE_TOLERANCE
        for record in reciprocity_records
    )
    adjoint_passed = all(record["passed"] for record in implementation_adjoint_records)
    passivity_passed = (
        stability.passivity_violation_ev <= SOURCE_RESPONSE_PASSIVITY_TOLERANCE_EV
    )
    return {
        "passed": bool(reciprocity_passed and adjoint_passed and passivity_passed),
        "implementation_adjoint_passed": bool(adjoint_passed),
        "electrostatic_reciprocity_passed": bool(reciprocity_passed),
        "passivity_passed": bool(passivity_passed),
        "implementation_adjoint_records": implementation_adjoint_records,
        "electrostatic_reciprocity_records": reciprocity_records,
        "stability": asdict(stability),
        "thresholds": {
            "implementation_adjoint_absolute_max": (SOURCE_RESPONSE_ADJOINT_TOLERANCE),
            "electrostatic_reciprocity_relative_max": (
                SOURCE_RESPONSE_RECIPROCITY_RELATIVE_TOLERANCE
            ),
            "passivity_violation_max_ev": (SOURCE_RESPONSE_PASSIVITY_TOLERANCE_EV),
        },
    }
