"""Composition identity tests for the native-FP64 response v2 PES."""

from __future__ import annotations

import importlib

import pytest

torch = pytest.importorskip("torch")


def test_v2_import_has_no_default_dtype_side_effect():
    old = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float32)
        importlib.import_module(
            "maple.solvation.experimental.mace_polar_response_native_fp64_v2"
        )
        assert torch.get_default_dtype() is torch.float32
    finally:
        torch.set_default_dtype(old)


def test_v2_ids_are_explicit_and_distinct_from_v1():
    from maple.solvation.experimental.mace_polar_response import _PROVIDER_ID as v1
    from maple.solvation.experimental.mace_polar_response_native_fp64_v2 import (
        NATIVE_FP64_RESPONSE_V2_PROVIDER_ID,
        native_fp64_response_v2_ids,
    )

    ids = native_fp64_response_v2_ids("cpu")
    assert NATIVE_FP64_RESPONSE_V2_PROVIDER_ID != v1
    assert all("native-fp64" in value and "v2" in value for value in ids.values())
    assert len(set(ids.values())) == 3


def test_v2_builder_rejects_scientific_overrides():
    from maple.solvation.experimental.mace_polar_response_native_fp64_v2 import (
        build_smd_mace_polar_response_native_fp64_v2_pes,
    )

    with pytest.raises(TypeError, match="unexpected keyword"):
        build_smd_mace_polar_response_native_fp64_v2_pes(
            ("O", "H", "H"), "water", lmax=9
        )
