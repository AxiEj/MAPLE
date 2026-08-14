from __future__ import annotations

from dataclasses import FrozenInstanceError
import math
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from maple.solvation.api.profiles import (  # noqa: E402
    MACE_POLAR_ANALYTIC_GAUSSIAN_MULTIPOLE_EVALUATOR_ID,
    MACE_POLAR_ANALYTIC_GAUSSIAN_MULTIPOLE_MODEL_PROFILE_ID,
    MACE_POLAR_MOLECULAR_REALSPACE_EVALUATOR_ID,
)
from maple.solvation.models.runtime.analytic_gaussian_multipole import (  # noqa: E402
    AnalyticGaussianMultipoleElectrostaticEnergy,
    AnalyticGaussianMultipoleElectrostaticFeatures,
    MACEPolarAnalyticGaussianMultipoleEvaluator,
)
from maple.solvation.models.mace_polar import (  # noqa: E402
    MACE_POLAR_ANALYTIC_GAUSSIAN_MULTIPOLE_RELEASE_CONTRACT,
    OFFICIAL_MACE_POLAR_1_M_CONTRACT,
)

FIELD_CONSTANT = 1.0 / (5.526349406e-3)
COULOMB_PREFACTOR = FIELD_CONSTANT / (4.0 * math.pi)


class _FeatureSelfInteraction(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.num_radial = 2
        self.non_zero_terms = 8
        self.register_buffer("matrix", torch.zeros((8, 4), dtype=torch.float64))

    def forward(self, source):
        return source @ self.matrix.T


class _EnergySelfInteraction(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.num_radial = 1
        self.non_zero_terms = 4
        self.register_buffer("matrix", torch.zeros((4, 4), dtype=torch.float64))

    def forward(self, source):
        return source @ self.matrix.T


class _UpstreamFeatureStencil(torch.nn.Module):
    def __init__(self, *, receiver_sigmas=(1.5, 3.0), offset=0.1):
        super().__init__()
        self.density_max_l = 1
        self.projection_max_l = 1
        self.density_smearing_width = 1.5
        self.projection_smearing_widths = list(receiver_sigmas)
        self.num_radial = 2
        self.include_self_interaction = False
        self.offset = float(offset)
        self.self_interaction = _FeatureSelfInteraction()
        widths = [math.sqrt((1.5**2 + value**2) / 2.0) for value in receiver_sigmas]
        self.register_buffer(
            "total_width_factors", torch.tensor(widths, dtype=torch.float64)
        )
        self.register_buffer(
            "l0_factors", torch.tensor([1.2, 0.7], dtype=torch.float64)
        )
        self.register_buffer(
            "l1_factors",
            torch.tensor([0.8 / offset, 1.4 / offset], dtype=torch.float64),
        )


class _UpstreamEnergyStencil(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.density_max_l = 1
        self.density_smearing_width = 1.5
        self.include_self_interaction = True
        self.offset = 0.02
        self.self_interaction = _EnergySelfInteraction()


def _modules(*, include_energy_self=False):
    features = AnalyticGaussianMultipoleElectrostaticFeatures(
        density_smearing_width=1.5,
        projection_smearing_widths=(1.5, 3.0),
        include_self_interaction=False,
        total_width_factors=torch.tensor([1.5, math.sqrt(5.625)], dtype=torch.float64),
        l0_factors=torch.tensor([1.0, 0.7], dtype=torch.float64),
        l1_limit_factors=torch.tensor([1.0, 1.4], dtype=torch.float64),
        self_interaction=_FeatureSelfInteraction(),
        minimum_separation_angstrom=1.0e-10,
    )
    energy = AnalyticGaussianMultipoleElectrostaticEnergy(
        density_smearing_width=1.5,
        include_self_interaction=include_energy_self,
        self_interaction=_EnergySelfInteraction(),
        minimum_separation_angstrom=1.0e-10,
    ).to(dtype=torch.float64)
    return features, energy


def _geometry_and_source(*, requires_grad=False):
    positions = torch.tensor(
        [[0.1, -0.3, 0.2], [1.2, 0.4, -0.5], [-0.7, 1.3, 0.8]],
        dtype=torch.float64,
        requires_grad=requires_grad,
    )
    source = torch.tensor(
        [[0.2, 0.1, -0.2, 0.3], [-0.4, 0.2, 0.15, -0.1], [0.2, -0.3, 0.05, 0.07]],
        dtype=torch.float64,
    )
    batch = torch.zeros(3, dtype=torch.long)
    return positions, source, batch


def _rotation():
    return torch.tensor(
        [[0.36, -0.48, 0.8], [0.8, 0.60, 0.0], [-0.48, 0.64, 0.60]],
        dtype=torch.float64,
    )


def _rotate_raw_l1(values, rotation):
    result = values.clone()
    cartesian = values[:, (3, 1, 2)] @ rotation.T
    result[:, 1] = cartesian[:, 1]
    result[:, 2] = cartesian[:, 2]
    result[:, 3] = cartesian[:, 0]
    return result


def _rotate_feature_l1(features, rotation):
    result = features.clone()
    for radial in range(2):
        block = slice(2 + 3 * radial, 2 + 3 * (radial + 1))
        raw = features[:, block]
        cartesian = raw[:, (2, 0, 1)] @ rotation.T
        result[:, block] = cartesian[:, (1, 2, 0)]
    return result


def _kernel(displacement, width):
    distance = torch.linalg.vector_norm(displacement, dim=-1)
    return COULOMB_PREFACTOR * torch.erf(0.5 * distance / width) / distance


def _stencil_charges(source, positions, step):
    count = source.shape[0]
    shifts = positions.new_tensor(
        [[0.0, 0.0, 0.0], [step, 0.0, 0.0], [0.0, step, 0.0], [0.0, 0.0, step]]
    )
    extended_positions = positions[:, None, :] + shifts[None, :, :]
    extended_charges = source.new_zeros((count, 4))
    extended_charges[:, 1] = source[:, 3] / step
    extended_charges[:, 2] = source[:, 1] / step
    extended_charges[:, 3] = source[:, 2] / step
    extended_charges[:, 0] = source[:, 0] - torch.sum(extended_charges[:, 1:], dim=-1)
    return extended_positions, extended_charges, shifts


def _one_sided_feature_stencil(features, source, positions, step):
    source_positions, charges, shifts = _stencil_charges(source, positions, step)
    receiver_positions = positions[:, None, :] + shifts[None, :, :]
    scalar = source.new_zeros((source.shape[0], 4, 2))
    for receiver in range(source.shape[0]):
        for receiver_shift in range(4):
            for sender in range(source.shape[0]):
                if sender == receiver:
                    continue
                displacement = (
                    source_positions[sender]
                    - receiver_positions[receiver, receiver_shift]
                )
                for radial, width in enumerate(features.total_width_factors):
                    scalar[receiver, receiver_shift, radial] += torch.sum(
                        charges[sender] * _kernel(displacement, width)
                    )
    l0 = scalar[:, 0, :] * features.l0_factors
    raw = torch.stack(
        (
            scalar[:, 2, :] - scalar[:, 0, :],
            scalar[:, 3, :] - scalar[:, 0, :],
            scalar[:, 1, :] - scalar[:, 0, :],
        ),
        dim=-1,
    )
    l1 = raw * features.l1_limit_factors[None, :, None] / step
    return torch.cat((l0, l1.reshape(source.shape[0], -1)), dim=-1)


def _one_sided_energy_stencil(source, positions, step):
    extended_positions, charges, _ = _stencil_charges(source, positions, step)
    result = source.new_zeros(())
    for left in range(source.shape[0]):
        for right in range(left + 1, source.shape[0]):
            displacement = (
                extended_positions[left, :, None, :]
                - extended_positions[right, None, :, :]
            )
            result = result + torch.sum(
                charges[left, :, None]
                * charges[right, None, :]
                * _kernel(displacement, source.new_tensor(1.5))
            )
    return result


def test_release_identity_is_distinct_and_remains_unadmitted():
    contract = MACE_POLAR_ANALYTIC_GAUSSIAN_MULTIPOLE_RELEASE_CONTRACT
    assert contract.model_profile_id == (
        MACE_POLAR_ANALYTIC_GAUSSIAN_MULTIPOLE_MODEL_PROFILE_ID
    )
    assert contract.long_range_evaluator_profile == (
        MACE_POLAR_ANALYTIC_GAUSSIAN_MULTIPOLE_EVALUATOR_ID
    )
    assert (
        contract.model_profile_id != OFFICIAL_MACE_POLAR_1_M_CONTRACT.model_profile_id
    )
    assert contract.long_range_evaluator_profile != (
        MACE_POLAR_MOLECULAR_REALSPACE_EVALUATOR_ID
    )
    assert (
        contract.checkpoint_sha256 == OFFICIAL_MACE_POLAR_1_M_CONTRACT.checkpoint_sha256
    )
    assert contract.structural_so3_equivariance_admitted is True
    assert "E/F/H/V/M unadmitted" in contract.release_status


def test_same_width_feature_is_exact_energy_source_gradient():
    features, energy = _modules()
    positions, source, batch = _geometry_and_source()
    source.requires_grad_(True)
    feature_values = features(source[:, None, :], positions, batch)[0]
    energy_value = energy(source, positions, batch).sum()
    (source_gradient,) = torch.autograd.grad(energy_value, source)
    same_width_feature = torch.cat(
        (feature_values[:, 0:1], feature_values[:, 2:5]), dim=-1
    )
    torch.testing.assert_close(
        same_width_feature, source_gradient, atol=2.0e-15, rtol=2.0e-15
    )


def test_energy_feature_and_coordinate_gradient_rotate_at_roundoff():
    features, energy = _modules()
    positions, source, batch = _geometry_and_source(requires_grad=True)
    feature_values = features(source[:, None, :], positions, batch)[0]
    energy_value = energy(source, positions, batch).sum()
    (coordinate_gradient,) = torch.autograd.grad(energy_value, positions)

    rotation = _rotation()
    torch.testing.assert_close(
        rotation @ rotation.T, torch.eye(3, dtype=torch.float64), atol=3.0e-16, rtol=0.0
    )
    assert float(torch.linalg.det(rotation)) == pytest.approx(1.0, abs=3.0e-16)
    rotated_positions = (positions.detach() @ rotation.T).requires_grad_(True)
    rotated_source = _rotate_raw_l1(source, rotation)
    rotated_features = features(rotated_source[:, None, :], rotated_positions, batch)[0]
    rotated_energy = energy(rotated_source, rotated_positions, batch).sum()
    (rotated_gradient,) = torch.autograd.grad(rotated_energy, rotated_positions)

    torch.testing.assert_close(rotated_energy, energy_value, atol=2.0e-15, rtol=0.0)
    torch.testing.assert_close(
        rotated_features,
        _rotate_feature_l1(feature_values, rotation),
        atol=8.0e-15,
        rtol=3.0e-15,
    )
    torch.testing.assert_close(
        rotated_gradient,
        coordinate_gradient @ rotation.T,
        atol=3.0e-15,
        rtol=3.0e-15,
    )


def test_coordinate_gradient_matches_central_finite_difference():
    _, energy = _modules()
    positions, source, batch = _geometry_and_source(requires_grad=True)
    value = energy(source, positions, batch).sum()
    (gradient,) = torch.autograd.grad(value, positions)
    direction = torch.tensor(
        [[0.2, -0.4, 0.1], [-0.3, 0.1, 0.5], [0.1, 0.3, -0.6]],
        dtype=torch.float64,
    )
    direction = direction / torch.linalg.vector_norm(direction)
    analytic = torch.sum(gradient * direction)
    errors = []
    for step in (2.0e-4, 1.0e-4, 5.0e-5):
        plus = energy(source, positions.detach() + step * direction, batch).sum()
        minus = energy(source, positions.detach() - step * direction, batch).sum()
        finite_difference = (plus - minus) / (2.0 * step)
        errors.append(abs(float(finite_difference - analytic)))
    assert errors[-1] < 2.0e-9
    assert errors[-1] < errors[0]


def test_analytic_operator_is_limit_of_axis_stencil_not_a_smaller_offset_patch():
    features, energy = _modules()
    positions, source, batch = _geometry_and_source()
    exact_features = features(source[:, None, :], positions, batch)[0]
    exact_energy = energy(source, positions, batch).sum()
    feature_errors = []
    energy_errors = []
    for step in (0.08, 0.04, 0.02):
        feature_errors.append(
            float(
                torch.max(
                    torch.abs(
                        _one_sided_feature_stencil(features, source, positions, step)
                        - exact_features
                    )
                )
            )
        )
        energy_errors.append(
            abs(
                float(_one_sided_energy_stencil(source, positions, step) - exact_energy)
            )
        )
    assert feature_errors[2] < feature_errors[1] < feature_errors[0]
    assert energy_errors[2] < energy_errors[1] < energy_errors[0]
    assert feature_errors[2] < 0.35 * feature_errors[0]
    assert energy_errors[2] < 0.35 * energy_errors[0]


def test_batch_separation_translation_and_guards():
    features, energy = _modules()
    positions, source, _ = _geometry_and_source()
    batch = torch.tensor([0, 0, 1], dtype=torch.long)
    base_features = features(source[:, None, :], positions, batch)[0]
    base_energy = energy(source, positions, batch)
    translated = positions.clone()
    translated[batch == 0] += torch.tensor([2.0, -1.0, 0.4])
    translated[batch == 1] += torch.tensor([-4.0, 3.0, 1.2])
    torch.testing.assert_close(
        features(source[:, None, :], translated, batch)[0],
        base_features,
        atol=3.0e-15,
        rtol=2.0e-15,
    )
    torch.testing.assert_close(
        energy(source, translated, batch), base_energy, atol=3.0e-15, rtol=2.0e-15
    )

    coincident = positions.clone()
    coincident[1] = coincident[0]
    with pytest.raises(ValueError, match="coincident"):
        features(source[:, None, :], coincident, torch.zeros(3, dtype=torch.long))
    with pytest.raises(ValueError, match="coincident"):
        energy(source, coincident, torch.zeros(3, dtype=torch.long))
    nonfinite = source.clone()
    nonfinite[0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        energy(nonfinite, positions, torch.zeros(3, dtype=torch.long))


def test_evaluator_replaces_only_realspace_modules_and_detects_drift(monkeypatch):
    import maple.solvation.models.runtime.analytic_gaussian_multipole as module

    monkeypatch.setattr(module, "version", lambda name: "0.4.0")
    old_features = _UpstreamFeatureStencil()
    old_energy = _UpstreamEnergyStencil()
    model = SimpleNamespace(
        electric_potential_descriptor=SimpleNamespace(
            realspace_features=old_features,
            sentinel="descriptor-unchanged",
        ),
        coulomb_energy=SimpleNamespace(
            realspace_energy=old_energy,
            sentinel="energy-unchanged",
        ),
    )
    evaluator = MACEPolarAnalyticGaussianMultipoleEvaluator.from_profile(
        MACE_POLAR_ANALYTIC_GAUSSIAN_MULTIPOLE_EVALUATOR_ID
    )
    assert evaluator.is_default is True
    assert evaluator.model_forward_kwargs == {}
    evaluator.configure_model(model)
    assert isinstance(
        model.electric_potential_descriptor.realspace_features,
        AnalyticGaussianMultipoleElectrostaticFeatures,
    )
    assert isinstance(
        model.coulomb_energy.realspace_energy,
        AnalyticGaussianMultipoleElectrostaticEnergy,
    )
    assert model.electric_potential_descriptor.sentinel == "descriptor-unchanged"
    assert model.coulomb_energy.sentinel == "energy-unchanged"
    configuration = evaluator.configuration_sha256(model)
    assert len(configuration) == 64
    assert evaluator.provenance["checkpoint_weights_changed"] is False
    assert evaluator.provenance["inference_operator_changed"] is True
    assert not any(evaluator.provenance["capabilities"].values())
    evaluator.configure_model(model)
    assert evaluator.configuration_sha256(model) == configuration

    with torch.no_grad():
        model.electric_potential_descriptor.realspace_features.l0_factors[0] += 0.01
    with pytest.raises(RuntimeError, match="configuration drifted"):
        evaluator.configuration_sha256(model)
    with pytest.raises(FrozenInstanceError):
        evaluator.minimum_separation_angstrom = 0.2


def test_evaluator_rejects_wrong_checkpoint_basis_before_install(monkeypatch):
    import maple.solvation.models.runtime.analytic_gaussian_multipole as module

    monkeypatch.setattr(module, "version", lambda name: "0.4.0")
    old_features = _UpstreamFeatureStencil(receiver_sigmas=(1.5, 2.9))
    old_energy = _UpstreamEnergyStencil()
    model = SimpleNamespace(
        electric_potential_descriptor=SimpleNamespace(realspace_features=old_features),
        coulomb_energy=SimpleNamespace(realspace_energy=old_energy),
    )
    evaluator = MACEPolarAnalyticGaussianMultipoleEvaluator.from_profile(
        MACE_POLAR_ANALYTIC_GAUSSIAN_MULTIPOLE_EVALUATOR_ID
    )
    with pytest.raises(ValueError, match="receiver smearing widths"):
        evaluator.configure_model(model)
    assert model.electric_potential_descriptor.realspace_features is old_features
    assert model.coulomb_energy.realspace_energy is old_energy


def test_configuration_hashes_are_stable_and_content_sensitive():
    features, energy = _modules()
    feature_hash = features.configuration_sha256()
    energy_hash = energy.configuration_sha256()
    assert len(feature_hash) == len(energy_hash) == 64
    assert features.configuration_sha256() == feature_hash
    assert energy.configuration_sha256() == energy_hash
    with torch.no_grad():
        features.l1_limit_factors[1] *= 1.01
    assert features.configuration_sha256() != feature_hash
