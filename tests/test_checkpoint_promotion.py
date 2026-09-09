from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator

from maple.function.dispatcher.md.logger import MDLogger
from maple.function.dispatcher.md.rst_io import (
    promote_rst_checkpoint,
    read_rst,
    write_rst,
)
from maple.function.dispatcher.md.state import (
    PreparedMDState,
    prepare_restart_candidate,
)

IDENTITY = {
    "backend": "checkpoint-promotion-test",
    "model_fingerprint": {
        "algorithm": "sha256",
        "digest": "a" * 64,
        "source": "test",
    },
    "relevant_settings": {},
}
DYNAMICS = {"remove_com": False, "remove_angular": False}


class IdentityCalculator(Calculator):
    SUPPORTS_CHARGE_MULT = True

    def __init__(self) -> None:
        super().__init__()
        self.maple_pes_identity = IDENTITY


def _atoms() -> Atoms:
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    atoms.info.update(charge=0, mult=1)
    atoms.calc = IdentityCalculator()
    return atoms


def _write_checkpoint(path: Path, *, step: int) -> bytes:
    atoms = _atoms()
    write_rst(
        path,
        atoms,
        np.zeros((1, 3)),
        step=step,
        timestep=0.1,
        ensemble="nve",
        energy=0.0,
        velocity_representation="standard",
        pes_identity=IDENTITY,
        dynamics_parameters=DYNAMICS,
    )
    return path.read_bytes()


def _fail_canonical_publish(monkeypatch, canonical: Path) -> None:
    from maple.function.dispatcher.md import rst_io

    original_replace = rst_io.os.replace

    def fail_replace(source, destination):
        if Path(destination) == canonical:
            raise OSError("injected canonical publication failure")
        return original_replace(source, destination)

    monkeypatch.setattr(rst_io.os, "replace", fail_replace)


def test_prev_fallback_remains_default_recoverable_if_promotion_fails(
    tmp_path, monkeypatch
):
    canonical = tmp_path / "run_md.rst"
    previous = tmp_path / "run_md_prev.rst"
    canonical.write_bytes(b"damaged checkpoint")
    valid_bytes = _write_checkpoint(previous, step=4)
    _fail_canonical_publish(monkeypatch, canonical)

    with pytest.raises(OSError, match="canonical publication failure"):
        promote_rst_checkpoint(
            previous,
            canonical,
            previous,
            source_bytes=valid_bytes,
        )

    assert previous.read_bytes() == valid_bytes
    prepared = prepare_restart_candidate(
        _atoms(), [canonical, previous], load_state=False
    )
    assert prepared.source == previous
    assert prepared.source_bytes == valid_bytes
    assert prepared.checkpoint["step"] == 4


def test_completed_restart_failure_preserves_valid_prev_candidate(
    tmp_path, monkeypatch
):
    logger = MDLogger(str(tmp_path / "completed.out"), verbose=0)
    logger.rst_path.write_bytes(b"damaged checkpoint")
    valid_bytes = _write_checkpoint(logger.rst_prev_path, step=4)
    checkpoint = read_rst(logger.rst_prev_path)
    prepared = PreparedMDState(
        atoms=_atoms(),
        velocities=checkpoint["velocities"].copy(),
        checkpoint=checkpoint,
        source=logger.rst_prev_path,
        source_sha256="unused-by-promotion",
        source_bytes=valid_bytes,
        load_state=False,
    )
    _fail_canonical_publish(monkeypatch, logger.rst_path)

    with pytest.raises(OSError, match="canonical publication failure"):
        logger.consume_prepared_restart(prepared, completed=True)

    assert logger.rst_prev_path.read_bytes() == valid_bytes
    assert read_rst(logger.rst_prev_path)["step"] == 4
    assert not list(tmp_path.glob("completed_md_seg*"))


def test_successful_prev_fallback_keeps_prev_as_valid_backup(tmp_path):
    canonical = tmp_path / "run_md.rst"
    previous = tmp_path / "run_md_prev.rst"
    canonical.write_bytes(b"damaged checkpoint")
    valid_bytes = _write_checkpoint(previous, step=4)

    promote_rst_checkpoint(
        previous,
        canonical,
        previous,
        source_bytes=valid_bytes,
    )

    assert canonical.read_bytes() == valid_bytes
    assert previous.read_bytes() == valid_bytes


def test_successful_promotion_rotates_old_main_after_publishing_captured_bytes(
    tmp_path,
):
    canonical = tmp_path / "run_md.rst"
    previous = tmp_path / "run_md_prev.rst"
    source = tmp_path / "external.rst"
    old_main = _write_checkpoint(canonical, step=2)
    captured = _write_checkpoint(source, step=4)
    source.write_bytes(b"changed after validation")

    promote_rst_checkpoint(
        source,
        canonical,
        previous,
        source_bytes=captured,
    )

    assert canonical.read_bytes() == captured
    assert previous.read_bytes() == old_main
