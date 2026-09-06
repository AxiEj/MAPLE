from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from maple.function.calculator.extra_correction.implicit.gto_density import (
    cartesian_multipoles,
)
from maple.solvation.models.mace_mdp_mbis import (
    MACE_MDP_MBIS_SOURCE_HEAD_EXPECTED_SHA256,
    MACE_MDPMBISSourceHeadAsset,
    build_mace_mdp_mbis_source_adapter,
)

CHECKPOINT = Path.home() / ".cache/mace/MACE-MDP.model"
HEAD_ROOT = Path.home() / ".local/share/maple/route2/mdp-mbis-source-head-prototype-v1"


def _rotation() -> np.ndarray:
    axis = np.asarray((0.31, -0.72, 0.61), dtype=float)
    axis /= np.linalg.norm(axis)
    angle = 0.731
    cross = np.asarray(
        (
            (0.0, -axis[2], axis[1]),
            (axis[2], 0.0, -axis[0]),
            (-axis[1], axis[0], 0.0),
        )
    )
    return (
        np.cos(angle) * np.eye(3)
        + (1.0 - np.cos(angle)) * np.outer(axis, axis)
        + np.sin(angle) * cross
    )


def _head_path() -> Path:
    candidates = sorted(HEAD_ROOT.glob("execution-*/source_head.npz"))
    if not candidates:
        pytest.skip("No local MDP-MBIS source-head artifact is available.")
    return candidates[-1]


def _write_head(path: Path, **overrides: np.ndarray) -> str:
    arrays: dict[str, np.ndarray] = {
        "scalar_mean": np.zeros(256),
        "scalar_scale": np.ones(256),
        "vector_scale": np.ones(256),
        "charge_weights": np.zeros(266),
        "dipole_weights": np.zeros(256),
        "charge_sigma": np.asarray(0.2),
        "dipole_sigma": np.asarray(0.1),
        "atomic_numbers": np.asarray((1, 6, 7, 8, 9, 15, 16, 17, 35, 53)),
    }
    arrays.update(overrides)
    np.savez_compressed(path, **arrays)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_source_head_asset_is_exact_schema_content_addressed_and_readonly(
    tmp_path: Path,
) -> None:
    path = tmp_path / "head.npz"
    digest = _write_head(path)
    asset = MACE_MDPMBISSourceHeadAsset.from_npz(path, expected_sha256=digest)

    assert asset.sha256 == digest
    assert asset.scalar_mean.shape == (256,)
    assert not asset.scalar_mean.flags.writeable
    with pytest.raises(ValueError, match="SHA256"):
        MACE_MDPMBISSourceHeadAsset.from_npz(path, expected_sha256="0" * 64)

    bad = tmp_path / "bad.npz"
    bad_digest = _write_head(bad, extra=np.asarray(1.0))
    with pytest.raises(ValueError, match="schema"):
        MACE_MDPMBISSourceHeadAsset.from_npz(bad, expected_sha256=bad_digest)


@pytest.mark.skipif(not CHECKPOINT.is_file(), reason="MACE-MDP checkpoint absent")
def test_real_source_head_rotation_translation_and_coordinate_vjp() -> None:
    torch = pytest.importorskip("torch")
    from ase import Atoms

    device = "cuda" if torch.cuda.is_available() else "cpu"
    adapter = build_mace_mdp_mbis_source_adapter(
        checkpoint_path=CHECKPOINT,
        source_head_path=_head_path(),
        device=device,
        expected_source_head_sha256=MACE_MDP_MBIS_SOURCE_HEAD_EXPECTED_SHA256,
    )
    positions = np.asarray(
        (
            (0.0000, 0.0000, 0.1173),
            (0.0000, 0.7572, -0.4692),
            (0.0000, -0.7572, -0.4692),
        ),
        dtype=float,
    )
    atoms = Atoms("OH2", positions=positions, info={"charge": 0, "mult": 1})
    rotation = _rotation()
    rotated = atoms.copy()
    rotated.positions = positions @ rotation.T
    translated = atoms.copy()
    translated.positions = positions + np.asarray((1.2, -0.7, 0.4))

    state = adapter.evaluate_state(atoms)
    source = state.source4_raw_l1
    assert state.latent_mdp_charges_e.shape == (3,)
    assert state.latent_mdp_dipoles_eangstrom.shape == (3, 3)
    np.testing.assert_allclose(
        np.sum(
            state.latent_mdp_charges_e[:, None] * positions
            + state.latent_mdp_dipoles_eangstrom,
            axis=0,
        ),
        state.public_molecular_dipole_eangstrom,
        rtol=0.0,
        atol=2.0e-11,
    )
    rotated_source = adapter.evaluate_source(rotated)
    translated_source = adapter.evaluate_source(translated)
    charges, dipoles = cartesian_multipoles(source)
    rotated_charges, rotated_dipoles = cartesian_multipoles(rotated_source)
    translated_charges, translated_dipoles = cartesian_multipoles(translated_source)
    np.testing.assert_allclose(rotated_charges, charges, rtol=0.0, atol=2.0e-9)
    np.testing.assert_allclose(
        rotated_dipoles, dipoles @ rotation.T, rtol=0.0, atol=2.0e-9
    )
    np.testing.assert_allclose(translated_charges, charges, rtol=0.0, atol=2.0e-9)
    np.testing.assert_allclose(translated_dipoles, dipoles, rtol=0.0, atol=2.0e-9)

    rng = np.random.default_rng(48211)
    cotangent = rng.normal(size=source.shape)
    direction = rng.normal(size=positions.shape)
    direction -= np.mean(direction, axis=0, keepdims=True)
    direction /= np.linalg.norm(direction)
    analytic = float(np.sum(adapter.source_position_vjp(atoms, cotangent) * direction))

    def contraction(displacement: float) -> float:
        moved = atoms.copy()
        moved.positions = positions + displacement * direction
        return float(np.sum(adapter.evaluate_source(moved) * cotangent))

    errors = []
    for step in (2.0e-4, 1.0e-4, 5.0e-5):
        finite_difference = (contraction(step) - contraction(-step)) / (2.0 * step)
        errors.append(abs(finite_difference - analytic))
    assert errors[-1] < 2.0e-7
    assert errors[-1] <= 0.45 * errors[0] + 2.0e-10
