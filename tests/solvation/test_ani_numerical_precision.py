from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

MODEL_PATH = (
    Path(__file__).resolve().parents[2]
    / "maple/function/calculator/model/ani2x.pt"
)


def _ani_calculator(**kwargs):
    torch = pytest.importorskip("torch")
    if not MODEL_PATH.is_file():
        pytest.skip(f"Local checkpoint is unavailable: {MODEL_PATH}")
    from maple.function.calculator.ani._ani_calculator import ANICalculator

    return ANICalculator(
        torch.device("cpu"),
        model="ani2x",
        model_path=str(MODEL_PATH),
        implicit="none",
        **kwargs,
    )


def _water() -> Atoms:
    return Atoms(
        "OH2",
        positions=[
            [0.000000, 0.000000, 0.000000],
            [0.957200, 0.000000, 0.000000],
            [-0.239987, 0.927297, 0.000000],
        ],
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_ani_build_kwargs_preserves_default_and_explicit_precision_options():
    from maple.function.calculator.ani._ani_calculator import ANICalculator

    assert ANICalculator.OPTION_KEYS == ("d4", "dtype")
    assert ANICalculator.build_kwargs_from_options("ani2x", {}) == {
        "d4": False,
        "dtype": "auto",
        "hessian": "analytic",
    }
    assert ANICalculator.build_kwargs_from_options(
        "ani2x",
        {"d4": "true", "dtype": "float64", "hessian": "numerical"},
    ) == {"d4": True, "dtype": "float64", "hessian": "numerical"}
    assert ANICalculator.build_kwargs_from_options(
        model="ani2x",
        options={"dtype": "float64"},
    ) == {"d4": False, "dtype": "float64", "hessian": "analytic"}


@pytest.mark.parametrize("dtype", ["half", "FLOAT64", 64])
def test_ani_rejects_invalid_dtype_before_loading_checkpoint(dtype, tmp_path):
    torch = pytest.importorskip("torch")
    from maple.function.calculator.ani._ani_calculator import ANICalculator

    with pytest.raises(ValueError, match="dtype.*auto.*float32.*float64"):
        ANICalculator(
            torch.device("cpu"),
            model_path=str(tmp_path / "missing.pt"),
            dtype=dtype,
        )


def test_ani_rejects_explicit_float32_numerical_curvature(tmp_path):
    torch = pytest.importorskip("torch")
    from maple.function.calculator.ani._ani_calculator import ANICalculator

    with pytest.raises(ValueError, match="float32.*numerical curvature.*float64"):
        ANICalculator(
            torch.device("cpu"),
            model_path=str(tmp_path / "missing.pt"),
            dtype="float32",
            hessian="numerical",
        )


def test_ani_rejects_invalid_hessian_mode_before_loading_checkpoint(tmp_path):
    torch = pytest.importorskip("torch")
    from maple.function.calculator.ani._ani_calculator import ANICalculator

    with pytest.raises(ValueError, match="hessian.*analytic.*numerical"):
        ANICalculator(
            torch.device("cpu"),
            model_path=str(tmp_path / "missing.pt"),
            hessian="finite-difference",
        )


def test_ani_numerical_hessian_selects_float64_during_initialization():
    torch = pytest.importorskip("torch")
    calculator = _ani_calculator(hessian="numerical")

    assert calculator.dtype is torch.float64
    assert calculator.hessian == "numerical"
    provenance = calculator.inference_precision_provenance
    assert provenance["requested_dtype"] == "auto"
    assert provenance["effective_dtype"] == "float64"
    assert provenance["numerical_curvature_prepared"] is True
    assert provenance["recommended_step_angstrom"] == pytest.approx(0.0005)
    assert provenance["in_memory_cast_only"] is True
    assert provenance["reason"] == "numerical_curvature_requires_float64"
    assert provenance["original_checkpoint_sha256"] == _sha256(MODEL_PATH)


def test_ani_legacy_positional_constructor_order_is_preserved():
    torch = pytest.importorskip("torch")
    if not MODEL_PATH.is_file():
        pytest.skip(f"Local checkpoint is unavailable: {MODEL_PATH}")
    from maple.function.calculator.ani._ani_calculator import ANICalculator

    calculator = ANICalculator(
        torch.device("cpu"),
        "ani2x",
        str(MODEL_PATH),
        False,
        False,
        "none",
        "none",
    )

    assert calculator.dtype is torch.float32
    assert calculator.hessian == "analytic"
    assert calculator.inference_precision_provenance[
        "original_checkpoint_sha256"
    ] is None


def test_ani_lazy_sha_refuses_replaced_checkpoint_identity(tmp_path):
    torch = pytest.importorskip("torch")
    if not MODEL_PATH.is_file():
        pytest.skip(f"Local checkpoint is unavailable: {MODEL_PATH}")
    from maple.function.calculator.ani._ani_calculator import ANICalculator

    checkpoint = tmp_path / "ani2x.pt"
    shutil.copyfile(MODEL_PATH, checkpoint)
    calculator = ANICalculator(torch.device("cpu"), model_path=str(checkpoint))
    replacement = tmp_path / "replacement.pt"
    replacement.write_bytes(b"replacement checkpoint bytes")
    replacement.replace(checkpoint)

    with pytest.raises(RuntimeError, match="identity changed after.*loaded"):
        calculator.prepare_numerical_derivatives()


def test_ani_precision_hook_promotes_once_clears_cache_and_never_demotes():
    torch = pytest.importorskip("torch")
    calculator = _ani_calculator()
    original_sha = _sha256(MODEL_PATH)

    assert calculator.dtype is torch.float32
    assert any(parameter.dtype is torch.float32 for parameter in calculator.model.parameters())
    calculator.results = {"energy": -1.0}

    assert calculator.prepare_numerical_derivatives() == pytest.approx(0.0005)
    assert calculator.dtype is torch.float64
    assert calculator.results == {}
    assert all(parameter.dtype is torch.float64 for parameter in calculator.model.parameters())

    calculator.results = {"energy": -2.0}
    assert calculator.prepare_numerical_derivatives() == pytest.approx(0.0005)
    assert calculator.dtype is torch.float64
    assert calculator.results == {"energy": -2.0}
    assert calculator.inference_precision_provenance["original_checkpoint_sha256"] == original_sha
    assert _sha256(MODEL_PATH) == original_sha


def test_ani_numerical_hessian_uses_recommended_step_and_records_diagnostics():
    calculator = _ani_calculator(hessian="numerical")

    hessian = calculator.get_hessian(_water())

    assert hessian.shape == (9, 9)
    assert np.isfinite(hessian).all()
    assert calculator.last_numerical_hessian_diagnostics is not None
    assert calculator.last_numerical_hessian_diagnostics[
        "cartesian_displacement_angstrom"
    ] == pytest.approx(0.0005)


def test_ani_double_conversion_preserves_numeric_values_and_integer_buffers():
    torch = pytest.importorskip("torch")
    calculator = _ani_calculator()
    before = {
        name: tensor.detach().cpu().clone()
        for name, tensor in (
            list(calculator.model.named_parameters())
            + list(calculator.model.named_buffers())
        )
    }

    calculator.prepare_numerical_derivatives()
    after = dict(calculator.model.named_parameters()) | dict(
        calculator.model.named_buffers()
    )

    assert before.keys() == after.keys()
    for name, old in before.items():
        new = after[name].detach().cpu()
        if old.is_floating_point():
            assert new.dtype is torch.float64
            torch.testing.assert_close(new, old.to(torch.float64), rtol=0.0, atol=0.0)
        else:
            assert new.dtype == old.dtype
            torch.testing.assert_close(new, old, rtol=0.0, atol=0.0)


def test_ani_default_remains_native_float32_and_float64_has_loose_ef_parity():
    torch = pytest.importorskip("torch")
    native = _ani_calculator()
    promoted = _ani_calculator(dtype="float64")
    atoms = _water()

    native.calculate(atoms, properties=["energy", "forces"])
    promoted.calculate(atoms, properties=["energy", "forces"])

    assert native.dtype is torch.float32
    assert native.inference_precision_provenance["in_memory_cast_only"] is False
    assert native.inference_precision_provenance["original_checkpoint_sha256"] is None
    assert promoted.dtype is torch.float64
    assert promoted.inference_precision_provenance["reason"] == "explicit_dtype_float64"
    # Precision-only parity budgets: these do not assert physical accuracy.
    assert abs(native.results["energy"] - promoted.results["energy"]) < 1.0e-7
    np.testing.assert_allclose(
        native.results["forces"],
        promoted.results["forces"],
        rtol=0.0,
        atol=1.0e-7,
    )


def test_ani_precision_provenance_is_read_only_copy():
    calculator = _ani_calculator()

    provenance = calculator.inference_precision_provenance
    provenance["effective_dtype"] = "tampered"

    assert calculator.inference_precision_provenance["effective_dtype"] == "float32"
