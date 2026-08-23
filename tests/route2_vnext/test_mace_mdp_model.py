from __future__ import annotations

import fcntl
import hashlib
import numpy as np
import os
from pathlib import Path
import pytest
import sys
from types import ModuleType

from maple.function.calculator.extra_correction.implicit.gto_density import (
    cartesian_multipoles,
)
from maple.solvation.models.mace_mdp import (
    POLARIZABILITY_BOHR3_PER_EANGSTROM2_PER_VOLT,
    build_mace_mdp_moment_adapter,
    build_mace_mdp_moment_state,
)
from maple.solvation.models import checkpoint_bytes as checkpoint_bytes_module


def _state():
    positions = np.array([[0.0, 0.0, 0.0], [1.2, -0.1, 0.3]])
    charges = np.array([-0.2, 0.2])
    dipoles = np.array([[0.1, 0.2, -0.1], [-0.05, 0.03, 0.02]])
    atomic_alpha = np.array(
        [
            [[0.20, 0.01, 0.00], [0.01, 0.25, 0.02], [0.00, 0.02, 0.30]],
            [[0.30, -0.01, 0.02], [-0.01, 0.35, 0.00], [0.02, 0.00, 0.40]],
        ]
    )
    total_alpha = np.sum(atomic_alpha, axis=0)
    total_dipole = np.sum(charges[:, None] * positions + dipoles, axis=0)
    return build_mace_mdp_moment_state(
        configuration_sha256="1" * 64,
        model_input_sha256_value="2" * 64,
        atomic_numbers=np.array([6, 8]),
        positions_angstrom=positions,
        charges_e=charges,
        atomic_dipoles_eangstrom=dipoles,
        atomic_polarizabilities_eangstrom2_per_volt=atomic_alpha,
        public_dipole_eangstrom=total_dipole,
        public_polarizability_eangstrom2_per_volt=total_alpha,
    )


def test_mace_mdp_state_preserves_moments_pairing_and_polarizability():
    state = _state()
    charges, dipoles = cartesian_multipoles(state.source4_raw_l1)
    assert np.array_equal(charges, state.charges_e)
    assert np.array_equal(dipoles, state.atomic_dipoles_eangstrom)
    assert state.total_charge_e == pytest.approx(0.0, abs=1.0e-15)
    assert np.allclose(np.sum(state.atomic_dipole_weights, axis=0), np.eye(3))
    assert np.allclose(
        state.polarizability_bohr3,
        state.public_polarizability_eangstrom2_per_volt
        * POLARIZABILITY_BOHR3_PER_EANGSTROM2_PER_VOLT,
    )
    assert len(state.state_sha256) == 64


def test_mace_mdp_state_owns_immutable_constructor_arrays():
    positions = np.array([[0.0, 0.0, 0.0], [1.2, -0.1, 0.3]])
    state = _state()
    positions[:] = 999.0
    assert not np.any(state.positions_angstrom == 999.0)
    with pytest.raises(ValueError):
        state.charges_e.setflags(write=True)


def test_mace_mdp_state_rejects_nonreciprocal_or_nonpositive_response():
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    charges = np.array([-0.1, 0.1])
    dipoles = np.zeros((2, 3))
    bad = np.array([[1.0, 0.5, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    with pytest.raises(ValueError, match="reciprocal"):
        build_mace_mdp_moment_state(
            configuration_sha256="1" * 64,
            model_input_sha256_value="2" * 64,
            atomic_numbers=[6, 8],
            positions_angstrom=positions,
            charges_e=charges,
            atomic_dipoles_eangstrom=dipoles,
            atomic_polarizabilities_eangstrom2_per_volt=np.stack(
                (0.5 * bad, 0.5 * bad)
            ),
            public_dipole_eangstrom=np.array([0.1, 0.0, 0.0]),
            public_polarizability_eangstrom2_per_volt=bad,
        )


def test_mace_mdp_state_rejects_atomic_moment_mismatch():
    state = _state()
    with pytest.raises(ValueError, match="public dipole"):
        build_mace_mdp_moment_state(
            configuration_sha256=state.configuration_sha256,
            model_input_sha256_value=state.model_input_sha256,
            atomic_numbers=state.atomic_numbers,
            positions_angstrom=state.positions_angstrom,
            charges_e=state.charges_e,
            atomic_dipoles_eangstrom=state.atomic_dipoles_eangstrom,
            atomic_polarizabilities_eangstrom2_per_volt=(
                state.atomic_polarizabilities_eangstrom2_per_volt
            ),
            public_dipole_eangstrom=state.public_dipole_eangstrom + 1.0,
            public_polarizability_eangstrom2_per_volt=(
                state.public_polarizability_eangstrom2_per_volt
            ),
        )


class _FakeModel:
    pass


class _FakeCalculator:
    consumed_bytes: bytes | None = None
    model_path: str | None = None
    descriptor: int | None = None
    descriptor_seals: int | None = None
    swap_path: Path | None = None

    def __init__(self, *, model_paths: str, **kwargs: object) -> None:
        del kwargs
        type(self).model_path = model_paths
        if type(self).swap_path is not None:
            type(self).swap_path.write_bytes(b"attacker replacement")
        if model_paths.startswith("/proc/self/fd/"):
            descriptor = int(model_paths.rsplit("/", 1)[1])
            type(self).descriptor = descriptor
            type(self).descriptor_seals = fcntl.fcntl(
                descriptor, checkpoint_bytes_module._F_GET_SEALS
            )
        type(self).consumed_bytes = Path(model_paths).read_bytes()
        self.models = [_FakeModel()]

    def get_property(self, *args: object, **kwargs: object) -> None:
        del args, kwargs

    def _atoms_to_batch(self, *args: object, **kwargs: object) -> None:
        del args, kwargs

    def _clone_batch(self, *args: object, **kwargs: object) -> None:
        del args, kwargs


def _install_fake_mace(monkeypatch: pytest.MonkeyPatch) -> None:
    _FakeCalculator.consumed_bytes = None
    _FakeCalculator.model_path = None
    _FakeCalculator.descriptor = None
    _FakeCalculator.descriptor_seals = None
    _FakeCalculator.swap_path = None
    mace = ModuleType("mace")
    calculators = ModuleType("mace.calculators")
    calculators.MACECalculator = _FakeCalculator
    mace.calculators = calculators
    monkeypatch.setitem(sys.modules, "mace", mace)
    monkeypatch.setitem(sys.modules, "mace.calculators", calculators)


def test_captured_checkpoint_bytes_use_one_sealed_procfd_and_close_after_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_mace(monkeypatch)
    captured = b"captured MACE-MDP checkpoint bytes"
    digest = hashlib.sha256(captured).hexdigest()

    adapter = build_mace_mdp_moment_adapter(
        checkpoint_bytes=captured,
        expected_checkpoint_sha256=digest,
    )

    assert adapter.checkpoint_sha256 == digest
    assert _FakeCalculator.consumed_bytes == captured
    assert _FakeCalculator.model_path.startswith("/proc/self/fd/")
    required = (
        checkpoint_bytes_module._F_SEAL_SEAL
        | checkpoint_bytes_module._F_SEAL_SHRINK
        | checkpoint_bytes_module._F_SEAL_GROW
        | checkpoint_bytes_module._F_SEAL_WRITE
    )
    assert _FakeCalculator.descriptor_seals & required == required
    with pytest.raises(OSError):
        os.fstat(_FakeCalculator.descriptor)


def test_captured_bytes_ignore_checkpoint_path_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_mace(monkeypatch)
    path = tmp_path / "checkpoint.model"
    path.write_bytes(b"original path bytes")
    captured = b"already captured authoritative bytes"
    _FakeCalculator.swap_path = path

    build_mace_mdp_moment_adapter(
        checkpoint_path=path,
        checkpoint_bytes=captured,
        expected_checkpoint_sha256=hashlib.sha256(captured).hexdigest(),
    )

    assert path.read_bytes() == b"attacker replacement"
    assert _FakeCalculator.consumed_bytes == captured


def test_captured_bytes_reject_wrong_digest_without_constructing_calculator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_mace(monkeypatch)
    with pytest.raises(ValueError, match="does not match"):
        build_mace_mdp_moment_adapter(
            checkpoint_bytes=b"wrong bytes",
            expected_checkpoint_sha256="0" * 64,
        )
    assert _FakeCalculator.consumed_bytes is None


def test_captured_bytes_create_no_named_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_mace(monkeypatch)
    before = tuple(tmp_path.iterdir())
    captured = b"no named temporary artifact"
    build_mace_mdp_moment_adapter(
        checkpoint_bytes=captured,
        expected_checkpoint_sha256=hashlib.sha256(captured).hexdigest(),
    )
    assert tuple(tmp_path.iterdir()) == before


def test_captured_bytes_fail_closed_without_sealing_or_procfs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_mace(monkeypatch)
    captured = b"secure bytes"
    digest = hashlib.sha256(captured).hexdigest()
    real_fcntl = fcntl.fcntl

    def reject_seal(fd: int, operation: int, *args: object):
        if operation == checkpoint_bytes_module._F_ADD_SEALS:
            raise OSError("sealing unavailable")
        return real_fcntl(fd, operation, *args)

    monkeypatch.setattr(fcntl, "fcntl", reject_seal)
    with pytest.raises(RuntimeError, match="sealing is unavailable"):
        build_mace_mdp_moment_adapter(
            checkpoint_bytes=captured,
            expected_checkpoint_sha256=digest,
        )

    monkeypatch.setattr(fcntl, "fcntl", real_fcntl)
    real_stat = os.stat

    def reject_proc(path: object, *args: object, **kwargs: object):
        if str(path).startswith("/proc/self/fd/"):
            raise FileNotFoundError("procfs unavailable")
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(os, "stat", reject_proc)
    with pytest.raises(RuntimeError, match="proc/self/fd"):
        build_mace_mdp_moment_adapter(
            checkpoint_bytes=captured,
            expected_checkpoint_sha256=digest,
        )


def test_captured_bytes_fail_closed_without_memfd(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_mace(monkeypatch)

    class _NoMemfdLibc:
        pass

    monkeypatch.setattr(
        checkpoint_bytes_module.ctypes,
        "CDLL",
        lambda *args, **kwargs: _NoMemfdLibc(),
    )
    with pytest.raises(RuntimeError, match="sealable Linux memfd"):
        build_mace_mdp_moment_adapter(
            checkpoint_bytes=b"captured bytes",
            expected_checkpoint_sha256=hashlib.sha256(b"captured bytes").hexdigest(),
        )


def test_legacy_checkpoint_path_loading_is_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_mace(monkeypatch)
    path = tmp_path / "legacy.model"
    content = b"legacy checkpoint path bytes"
    path.write_bytes(content)
    build_mace_mdp_moment_adapter(
        checkpoint_path=path,
        expected_checkpoint_sha256=hashlib.sha256(content).hexdigest(),
    )
    assert _FakeCalculator.model_path == str(path.resolve())
    assert _FakeCalculator.consumed_bytes == content
