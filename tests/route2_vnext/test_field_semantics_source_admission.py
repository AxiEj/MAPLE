from __future__ import annotations

import hashlib

import numpy as np
import pytest

from maple.solvation.release.field_semantics import (
    FieldSemanticsManifest,
    audit_charging_path,
    audit_directional_derivative,
    audit_field_work_sign,
    audit_uniform_field_replay,
    audit_zero_field_baseline,
)
from maple.solvation.release.source_mep import (
    ContinuumReferenceValidationBinding,
    compare_continuum_active_source_to_reference,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _manifest(**overrides) -> FieldSemanticsManifest:
    values = {
        "checkpoint_sha256": _digest("checkpoint"),
        "adapter_configuration_sha256": _digest("adapter-config"),
        "adapter_provenance_sha256": _digest("adapter-provenance"),
        "model_provider_id": "test.mace-polar.provider.v1",
        "model_profile_id": "test.mace-polar.profile.v1",
        "source_space_sha256": _digest("source-space"),
        "native_field_space_sha256": _digest("native-field-space"),
        "pairing_metric_sha256": _digest("pairing"),
        "field_channel_order": (
            "potential_sigma_1p5",
            "potential_sigma_3p0",
            "gradient_y_sigma_1p5",
            "gradient_z_sigma_1p5",
            "gradient_x_sigma_1p5",
            "gradient_y_sigma_3p0",
            "gradient_z_sigma_3p0",
            "gradient_x_sigma_3p0",
        ),
        "field_radial_widths_angstrom": (1.5, 3.0),
        "real_ylm_convention": "raw l1 columns [m0,m1,m-1] map to [y,z,x]",
        "cartesian_spherical_l1_transform": "[q,l10,l11,l1-1] -> [q,x,y,z]=[q,l1-1,l10,l11]",
        "field_units": (
            "eV/e",
            "eV/e",
            "eV/(e*angstrom)",
            "eV/(e*angstrom)",
            "eV/(e*angstrom)",
            "eV/(e*angstrom)",
            "eV/(e*angstrom)",
            "eV/(e*angstrom)",
        ),
        "energy_unit": "eV",
        "external_potential_sign": None,
        "uniform_field_sign": None,
        "spin_channel_factor": None,
        "native_injection_explicit_work_included": None,
        "upstream_uniform_explicit_work_included": None,
        "origin_convention": "unverified",
        "evidence_measurement_sha256s": (),
    }
    values.update(overrides)
    return FieldSemanticsManifest(**values)


def test_field_semantics_manifest_is_content_addressed_and_fail_closed():
    manifest = _manifest()
    assert len(manifest.configuration_sha256()) == 64
    assert manifest.phi1_semantics_complete is False
    assert manifest.as_dict()["capability_admitted"] is False
    changed = _manifest(field_radial_widths_angstrom=(1.5, 2.0))
    assert changed.configuration_sha256() != manifest.configuration_sha256()

    verified = _manifest(
        external_potential_sign=-1,
        uniform_field_sign=-1,
        spin_channel_factor=0.5,
        native_injection_explicit_work_included=True,
        upstream_uniform_explicit_work_included=True,
        origin_convention="atomic-position mean removed before uniform potential",
        evidence_measurement_sha256s=(_digest("field-semantics-water-v1"),),
    )
    assert verified.phi1_semantics_complete is True
    with pytest.raises(ValueError, match="sign"):
        _manifest(external_potential_sign=0)
    with pytest.raises(ValueError, match="SHA256"):
        _manifest(evidence_measurement_sha256s=("not-content-addressed",))


def test_field_semantics_canaries_distinguish_graph_consistency_from_upstream_semantics():
    hessian = np.asarray([[0.4, 0.1], [0.1, 0.2]])
    linear = np.asarray([0.3, -0.2])

    def energy(field):
        value = np.asarray(field, dtype=float)
        return 1.7 + float(linear @ value + 0.5 * value @ hessian @ value)

    def gradient(field):
        value = np.asarray(field, dtype=float)
        return linear + hessian @ value

    field = np.asarray([0.2, -0.1])
    direction = np.asarray([0.4, 0.3])
    zero = audit_zero_field_baseline(
        conditioned_zero_energy_eV=energy(np.zeros(2)),
        vacuum_energy_eV=1.7,
        conditioned_zero_forces_eV_per_A=np.zeros((1, 3)),
        vacuum_forces_eV_per_A=np.zeros((1, 3)),
    )
    directional = audit_directional_derivative(
        energy=energy,
        gradient=gradient,
        field=field,
        direction=direction,
    )
    charging = audit_charging_path(
        energy=energy,
        gradient=gradient,
        endpoint_field=field,
    )
    sign = audit_field_work_sign(
        energy_gradient=gradient(field),
        reference_source=-gradient(field),
        sign=-1,
    )
    replay = audit_uniform_field_replay(
        source_max_abs_error=2.0e-13,
        energy_absolute_error_eV=1.0e-3,
        force_max_abs_error_eV_per_A=0.0,
        dipole_max_abs_error_e_angstrom=0.0,
        energy_derivative_absolute_error_e_angstrom=0.0,
    )
    assert zero.passed is True
    assert directional.passed is True
    assert charging.passed is True
    assert sign.passed is True
    assert replay.source_path_matched is True
    assert replay.passed is False
    assert replay.branch_semantics == "hybrid-branch-dependent"


class _Continuum:
    atom_count = 1
    geometry_sha256 = _digest("geometry")
    continuum_configuration_sha256 = _digest("continuum-config")
    provenance_sha256 = _digest("continuum-provenance")
    topology_sha256 = _digest("topology")
    cavity_profile_id = "test.cavity.v1"

    surface_operator = np.asarray([[2.0, 0.2], [0.2, 1.4]])
    source_to_boundary = np.asarray([[0.4, 0.1, 0.0, -0.2], [0.2, -0.1, 0.3, 0.1]])

    def source_rhs(self, source):
        return self.source_to_boundary @ np.asarray(source).reshape(-1)


def _continuum_validation(
    continuum: _Continuum, *, validated: bool = True
) -> ContinuumReferenceValidationBinding:
    return ContinuumReferenceValidationBinding(
        validation_identity="test-independent-continuum-reference-v1",
        validation_evidence_sha256=_digest("continuum-reference-evidence"),
        geometry_sha256=continuum.geometry_sha256,
        continuum_configuration_sha256=continuum.continuum_configuration_sha256,
        continuum_provenance_sha256=continuum.provenance_sha256,
        topology_sha256=continuum.topology_sha256,
        cavity_profile_id=continuum.cavity_profile_id,
        validated_for_source_gate=validated,
    )


def test_continuum_active_source_gate_uses_A_inverse_energy_norm_and_bound():
    continuum = _Continuum()
    source = np.asarray([[0.2, -0.1, 0.05, 0.03]])
    reference_rhs = continuum.source_rhs(source)
    exact = compare_continuum_active_source_to_reference(
        continuum=continuum,
        predicted_source=source,
        reference_boundary_rhs=reference_rhs,
        reference_identity="test-qm-boundary-projection-v1",
        reference_artifact_sha256=_digest("qm-reference"),
        continuum_reference_validation=_continuum_validation(continuum),
        fixed_source_energy_error_budget_eV=1.0e-4,
    )
    assert exact.a_inverse_rhs_distance_sqrt_eV == 0.0
    assert exact.fixed_source_energy_absolute_error_eV == 0.0
    assert exact.case_passed is True
    assert exact.as_dict()["capability_admitted"] is False

    perturbed = compare_continuum_active_source_to_reference(
        continuum=continuum,
        predicted_source=source + np.asarray([[0.1, 0.0, 0.0, 0.0]]),
        reference_boundary_rhs=reference_rhs,
        reference_identity="test-qm-boundary-projection-v1",
        reference_artifact_sha256=_digest("qm-reference"),
        continuum_reference_validation=_continuum_validation(continuum),
        fixed_source_energy_error_budget_eV=1.0e-8,
    )
    assert (
        perturbed.fixed_source_energy_absolute_error_eV
        <= perturbed.fixed_source_energy_error_upper_bound_eV + 1.0e-15
    )
    assert perturbed.case_passed is False


def test_continuum_active_source_gate_requires_independent_continuum_validation():
    continuum = _Continuum()
    source = np.asarray([[0.2, -0.1, 0.05, 0.03]])
    reference_rhs = continuum.source_rhs(source)
    unvalidated = compare_continuum_active_source_to_reference(
        continuum=continuum,
        predicted_source=source,
        reference_boundary_rhs=reference_rhs,
        reference_identity="test-qm-boundary-projection-v1",
        reference_artifact_sha256=_digest("qm-reference"),
        continuum_reference_validation=_continuum_validation(
            continuum, validated=False
        ),
        fixed_source_energy_error_budget_eV=1.0e-4,
    )
    assert unvalidated.metric_within_budget is True
    assert unvalidated.continuum_reference_validated_for_source_gate is False
    assert unvalidated.case_passed is False

    mismatched = _continuum_validation(continuum)
    object.__setattr__(mismatched, "topology_sha256", _digest("other-topology"))
    with pytest.raises(ValueError, match="continuum.topology_sha256"):
        compare_continuum_active_source_to_reference(
            continuum=continuum,
            predicted_source=source,
            reference_boundary_rhs=reference_rhs,
            reference_identity="test-qm-boundary-projection-v1",
            reference_artifact_sha256=_digest("qm-reference"),
            continuum_reference_validation=mismatched,
            fixed_source_energy_error_budget_eV=1.0e-4,
        )
