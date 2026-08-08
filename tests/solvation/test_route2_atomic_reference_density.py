from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from maple.function.calculator.extra_correction.implicit.route2_atomic_reference_density import (
    ATOMIC_REFERENCE_DENSITY_ARTIFACT,
    ATOMIC_REFERENCE_DENSITY_CONTENT_SHA256,
    ATOMIC_REFERENCE_DENSITY_MANIFEST_SHA256,
    ATOMIC_REFERENCE_DENSITY_TABLE_SHA256,
    AtomicReferenceDensity,
    AtomicReferenceDensityAsset,
    GaussianMixtureAtom,
    load_atomic_reference_density_asset,
)

ROOT = Path(__file__).resolve().parents[2]
TABLE = (
    ROOT / "docs/implicit-solvation/benchmarks/"
    "route2-rhodrop-atomic-reference-gaussian-mixture-v1.npz"
)
MANIFEST = TABLE.with_suffix(".json")
GENERATOR = (
    ROOT / "docs/implicit-solvation/benchmarks/"
    "generate_route2_rhodrop_atomic_reference_density.py"
)


def _asset():
    return load_atomic_reference_density_asset(
        table_path=TABLE,
        manifest_path=MANIFEST,
    )


def _central_jacobian(function, point, *, step):
    point = np.asarray(point, dtype=float)
    reference = np.asarray(function(point), dtype=float)
    result = np.empty(reference.shape + point.shape, dtype=float)
    for coordinate in range(point.size):
        plus = point.copy()
        minus = point.copy()
        plus[coordinate] += step
        minus[coordinate] -= step
        result[..., coordinate] = (
            np.asarray(function(plus)) - np.asarray(function(minus))
        ) / (2.0 * step)
    return result


def _synthetic_reference_density():
    mixture = GaussianMixtureAtom(
        electron_counts=np.asarray([0.35, 0.65]),
        gaussian_exponents_bohr2=np.asarray([0.42, 1.7]),
    )
    asset = AtomicReferenceDensityAsset(
        mixtures_by_atomic_number={1: mixture},
        table_sha256="0" * 64,
        manifest_sha256="1" * 64,
        provenance={"purpose": "unit-test"},
    )
    position = np.asarray([[-0.4, 0.2, 0.1]])
    return (
        AtomicReferenceDensity(asset, np.asarray([1]), position),
        mixture,
        position[0],
    )


def test_reference_density_asset_has_the_declared_identity_and_element_coverage():
    asset = _asset()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    assert manifest["artifact"] == ATOMIC_REFERENCE_DENSITY_ARTIFACT
    assert hashlib.sha256(TABLE.read_bytes()).hexdigest() == (
        ATOMIC_REFERENCE_DENSITY_TABLE_SHA256
    )
    assert hashlib.sha256(MANIFEST.read_bytes()).hexdigest() == (
        ATOMIC_REFERENCE_DENSITY_MANIFEST_SHA256
    )
    assert manifest["table"]["sha256"] == ATOMIC_REFERENCE_DENSITY_TABLE_SHA256
    assert asset.content_sha256 == ATOMIC_REFERENCE_DENSITY_CONTENT_SHA256
    asset.require_frozen_v1_identity()
    assert asset.supported_atomic_numbers == (1, 6, 7, 8, 16, 17)


def test_reference_density_asset_rejects_a_tampered_table(tmp_path):
    tampered = tmp_path / TABLE.name
    tampered.write_bytes(TABLE.read_bytes() + b"tampered")

    with pytest.raises(ValueError, match="SHA256|hash"):
        load_atomic_reference_density_asset(
            table_path=tampered,
            manifest_path=MANIFEST,
        )


def test_reference_density_asset_rejects_coordinated_table_and_manifest_tampering(
    tmp_path,
):
    tampered_table = tmp_path / TABLE.name
    with np.load(TABLE, allow_pickle=False) as archive:
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    arrays["gaussian_exponents_bohr2"][0] *= 1.01
    np.savez(tampered_table, **arrays)

    tampered_manifest = tmp_path / MANIFEST.name
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    payload["table"]["sha256"] = hashlib.sha256(tampered_table.read_bytes()).hexdigest()
    payload["source"]["table_sha256"] = "a" * 64
    payload["generator"]["sha256"] = "b" * 64
    tampered_manifest.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="manifest hash|frozen v1"):
        load_atomic_reference_density_asset(
            table_path=tampered_table,
            manifest_path=tampered_manifest,
        )


def test_reference_density_asset_rejects_forged_official_hash_claims() -> None:
    official = _asset()
    mixtures = dict(official.mixtures_by_atomic_number)
    hydrogen = mixtures[1]
    exponents = np.array(hydrogen.gaussian_exponents_bohr2, copy=True)
    exponents[0] *= 2.0
    mixtures[1] = GaussianMixtureAtom(
        electron_counts=hydrogen.electron_counts,
        gaussian_exponents_bohr2=exponents,
    )

    with pytest.raises(ValueError, match="content digest"):
        AtomicReferenceDensityAsset(
            mixtures_by_atomic_number=mixtures,
            table_sha256=ATOMIC_REFERENCE_DENSITY_TABLE_SHA256,
            manifest_sha256=ATOMIC_REFERENCE_DENSITY_MANIFEST_SHA256,
            provenance=official.provenance,
        )


def test_reference_density_generator_cold_reproduces_frozen_hashes(
    tmp_path,
    monkeypatch,
):
    spec = importlib.util.spec_from_file_location(
        "route2_rhodrop_reference_density_generator",
        GENERATOR,
    )
    assert spec is not None and spec.loader is not None
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)
    regenerated_table = tmp_path / TABLE.name
    regenerated_manifest = tmp_path / MANIFEST.name
    monkeypatch.setattr(generator, "OUTPUT_TABLE", regenerated_table)
    monkeypatch.setattr(generator, "OUTPUT_MANIFEST", regenerated_manifest)

    generator.main()

    assert hashlib.sha256(regenerated_table.read_bytes()).hexdigest() == (
        ATOMIC_REFERENCE_DENSITY_TABLE_SHA256
    )
    assert hashlib.sha256(regenerated_manifest.read_bytes()).hexdigest() == (
        ATOMIC_REFERENCE_DENSITY_MANIFEST_SHA256
    )


def test_reference_density_asset_mixtures_are_normalized_and_immutable():
    asset = _asset()

    for atomic_number in asset.supported_atomic_numbers:
        mixture = asset.mixture(atomic_number)
        assert mixture.electron_count == pytest.approx(
            float(atomic_number), rel=0.0, abs=5.0e-12
        )
        assert mixture.electron_counts.flags.writeable is False
        assert mixture.gaussian_exponents_bohr2.flags.writeable is False


@pytest.mark.parametrize(
    ("electron_counts", "gaussian_exponents_bohr2"),
    [
        ([-0.1, 1.1], [0.4, 1.2]),
        ([0.4, 0.6], [0.4, 0.0]),
        ([0.4], [0.4, 1.2]),
        ([np.nan], [0.4]),
    ],
)
def test_gaussian_mixture_atom_rejects_nonphysical_components(
    electron_counts,
    gaussian_exponents_bohr2,
):
    with pytest.raises(ValueError):
        GaussianMixtureAtom(
            electron_counts=np.asarray(electron_counts),
            gaussian_exponents_bohr2=np.asarray(gaussian_exponents_bohr2),
        )


def test_reference_density_spatial_jet_matches_central_differences_through_third_order():
    density, mixture, center = _synthetic_reference_density()
    point = np.asarray([0.37, -0.28, 0.61])
    jet = density.evaluate_spatial(point[None, :])
    displacement = point - center
    radius_squared = float(np.dot(displacement, displacement))
    expected_value = sum(
        count * (exponent / np.pi) ** 1.5 * np.exp(-exponent * radius_squared)
        for count, exponent in zip(
            mixture.electron_counts,
            mixture.gaussian_exponents_bohr2,
            strict=True,
        )
    )

    assert jet.value[0] == pytest.approx(expected_value, rel=2.0e-15, abs=2.0e-16)

    def value_at(candidate):
        return density.evaluate_spatial(candidate[None, :]).value[0]

    def gradient_at(candidate):
        return density.evaluate_spatial(candidate[None, :]).gradient[0]

    def hessian_at(candidate):
        return density.evaluate_spatial(candidate[None, :]).hessian[0]

    np.testing.assert_allclose(
        jet.gradient[0],
        _central_jacobian(value_at, point, step=2.0e-6),
        rtol=3.0e-8,
        atol=3.0e-10,
    )
    np.testing.assert_allclose(
        jet.hessian[0],
        _central_jacobian(gradient_at, point, step=2.0e-6),
        rtol=3.0e-7,
        atol=3.0e-9,
    )
    np.testing.assert_allclose(
        jet.third[0],
        _central_jacobian(hessian_at, point, step=8.0e-6),
        rtol=4.0e-6,
        atol=4.0e-8,
    )


def test_reference_density_is_covariant_under_rigid_translation():
    asset = _asset()
    positions = np.asarray([[-0.4, 0.2, 0.1], [1.1, -0.3, 0.5]])
    points = np.asarray([[0.37, -0.28, 0.61], [1.8, 0.2, -0.4]])
    shift = np.asarray([0.71, -0.33, 0.24])
    original = AtomicReferenceDensity(asset, np.asarray([1, 8]), positions)
    translated = AtomicReferenceDensity(
        asset,
        np.asarray([1, 8]),
        positions + shift,
    )

    original_jet = original.evaluate_spatial(points)
    translated_jet = translated.evaluate_spatial(points + shift)

    for name in ("value", "gradient", "hessian", "third"):
        np.testing.assert_allclose(
            getattr(translated_jet, name),
            getattr(original_jet, name),
            rtol=0.0,
            atol=3.0e-14,
        )


def test_reference_density_strict_underflow_screening_returns_a_finite_zero_jet():
    density, _, _ = _synthetic_reference_density()

    jet = density.evaluate_spatial(np.asarray([[1.0e6, -1.0e6, 1.0e6]]))

    for name in ("value", "gradient", "hessian", "third"):
        values = getattr(jet, name)
        assert np.all(np.isfinite(values))
        np.testing.assert_array_equal(values, np.zeros_like(values))
