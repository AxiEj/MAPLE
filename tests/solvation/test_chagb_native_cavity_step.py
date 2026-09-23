"""Label-free native canary: the locked PBSA SAV term has a discrete step."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from maple.function.calculator.extra_correction.implicit.amber_chagb import (
    AmberToolsChaGB,
    KCAL_PER_MOL_PER_HARTREE,
    _sha256_file,
)
from maple.function.read.filereader.mol2_reader import MOL2Reader

ROOT = Path(__file__).resolve().parents[2]
SOURCE = (
    ROOT / ".omx/benchmarks/route1-numerical-repair-20260905/reference/molecule.mol2"
)
AMBER_BIN = Path("/home/axie/miniconda3/envs/maple-ambertools/bin")
SOURCE_SHA = "7e241f1e218fb909bfa52b2a3e400b7382e9de41fb55d8bbc66e1f3704e01205"
BUNDLE_SHA256 = {
    "gbnsr6": "6f14c7a0418b8f03be1b0273367717335822d2f7719f137ad56f2d49e0489e42",
    "pbsa": "1cb42f0464e031b5f4331b8ae335cb0c130ddad50305956695efb62a19b2defd",
    "parmchk2": "c2ef0906b71c64e5277673bcba943512daf75ef334ad9421c0045b6b80b6d7cc",
    "tleap": "f1c526e0a24ad1fefc6c82be1279aa33eff5f9557d577c2236aa81437dfb52ac",
}


def test_native_pbsa_cavity_energy_has_a_small_coordinate_jump(tmp_path):
    if not SOURCE.is_file() or any(
        not (AMBER_BIN / name).is_file() for name in BUNDLE_SHA256
    ):
        pytest.skip("Pinned local AmberTools/reference assets are unavailable.")
    assert _sha256_file(SOURCE) == SOURCE_SHA
    for name, expected in BUNDLE_SHA256.items():
        assert _sha256_file(AMBER_BIN / name) == expected

    atoms = MOL2Reader(str(SOURCE), charge=0, mult=1)
    provider = AmberToolsChaGB(
        atoms,
        np.asarray(atoms.get_initial_charges(), dtype=np.float64),
        executable=str(AMBER_BIN / "gbnsr6"),
        audit_dir=tmp_path / "native-amber-audit",
    )
    center = atoms.get_positions()
    cavity = {}
    for displacement in (-0.005, -0.001, 0.0, 0.001):
        positions = center.copy()
        positions[0, 0] += displacement
        cavity[displacement] = (
            provider.evaluate_coordinates(positions).components_hartree["cavity"]
            * KCAL_PER_MOL_PER_HARTREE
        )

    assert cavity[-0.005] == pytest.approx(20.9437, abs=5e-5)
    for displacement in (-0.001, 0.0, 0.001):
        assert cavity[displacement] == pytest.approx(20.9485, abs=5e-5)
    # Native output has four decimal places; this demonstrates a real SAV
    # occupancy change, not the exact location of the discontinuity.
    assert cavity[-0.001] - cavity[-0.005] > 0.004
