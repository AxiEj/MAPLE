"""Intentional API retirement, not a repair or an alias to AnySolv-MoE."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import subprocess
import sys
from zipfile import ZipFile

import pytest

from maple.function.calculator.extra_correction import implicit
from maple.function.calculator.set_calculator import SetCalculator
from maple.function.read.command_control import CommandControl

ROOT = Path(__file__).resolve().parents[2]
CALCULATOR = ROOT / "maple/function/calculator"
RETIRED_MODULES = (
    "extra_correction.implicit.anisolv",
    "uma._anisolv_uma_calculator",
)
CARD_HASHES = {
    "anisolv-compact.yaml": "be9bb6994e1b2e8d313fdd1896d492c33d629435e7091b70692ebd9f5b20fdf8",
    "anisolv-uma.yaml": "098181a96a2685af888806d8d9c5be53bc60d45630b0df9330dfc1c223cc997f",
}


@pytest.mark.parametrize("module", RETIRED_MODULES)
def test_retired_runtime_is_not_importable(module):
    assert importlib.util.find_spec("maple.function.calculator." + module) is None
    assert not (CALCULATOR / (module.replace(".", "/") + ".py")).exists()


def test_retired_lazy_export_is_absent():
    assert "AniSolvCompactBackend" not in implicit.__all__
    with pytest.raises(AttributeError):
        getattr(implicit, "AniSolvCompactBackend")


@pytest.mark.parametrize("name, expected", CARD_HASHES.items())
def test_retired_cards_are_historical_bytes_only(name, expected):
    assert not (CALCULATOR / "model_cards" / name).exists()
    archived = ROOT / "docs/pretrained-solvation-hub/history/anisolv-retired" / name
    assert hashlib.sha256(archived.read_bytes()).hexdigest() == expected


@pytest.mark.parametrize(
    "headers",
    [
        ["#model=aimnet2", "#solv(implicit=water,method=anisolv,experimental=true)"],
        ["#model=anisolv-uma"],
        ["#model=anisolv_uma(base_model_path=/old,solvation_model_path=/compact)"],
        [
            "#model=anisolv-uma",
            "#solv(implicit=water,method=anysolv,experimental=true,source_root=/official)",
        ],
    ],
)
def test_legacy_selectors_fail_without_mapping(headers):
    with pytest.raises(ValueError, match="retired"):
        CommandControl.from_settings([*headers, "#sp(verbose=0)"])


@pytest.mark.parametrize(
    "model, method", [("anisolv-uma", "none"), ("aimnet2", "anisolv")]
)
def test_factory_rejects_legacy_identity_without_loading_moe(
    tmp_path, monkeypatch, model, method
):
    def forbidden_mapping(**kwargs):
        pytest.fail("retired identity was mapped to AnySolv-MoE")

    monkeypatch.setattr(implicit, "AnySolvBackend", forbidden_mapping)
    setter = SetCalculator(
        device="cpu",
        model=model,
        implicit=method,
        solvent="none" if method == "none" else "water",
        output=str(tmp_path / "retired.out"),
        task="sp",
    )
    with pytest.raises(ValueError, match="Unsupported model|Unsupported implicit"):
        setter.set_calculator()


def test_modern_anysolv_stays_an_additive_backend():
    from maple.function.calculator.backends import SolvationBackend

    assert issubclass(implicit.AnySolvBackend, SolvationBackend)
    assert (CALCULATOR / "model_cards/anysolv-moe.yaml").is_file()
    control = CommandControl.from_settings(
        [
            "#model=aimnet2",
            "#sp(verbose=0)",
            "#solv(implicit=water,method=anysolv,experimental=true,source_root=/official)",
        ]
    )
    assert control.get("model") == "aimnet2"
    assert control.get("solv")["method"] == "anysolv"


def test_wheel_omits_retired_runtime_and_all_weights(tmp_path):
    # Keep build/lib and egg-info out of the checkout: incremental setuptools
    # builds otherwise retain modules removed after an earlier wheel build.
    source = tmp_path / "source"
    source.mkdir()
    (source / "maple").symlink_to(ROOT / "maple", target_is_directory=True)
    (source / "pyproject.toml").write_bytes((ROOT / "pyproject.toml").read_bytes())
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            ".",
            "--no-deps",
            "--no-build-isolation",
            "--wheel-dir",
            str(tmp_path),
        ],
        cwd=source,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    with ZipFile(next(tmp_path.glob("*.whl"))) as wheel:
        names = wheel.namelist()
    assert "maple/function/calculator/model_cards/anysolv-moe.yaml" in names
    assert not any("anisolv" in name.lower() for name in names)
    assert not any(name.endswith((".pt", ".pth", ".ckpt", ".model")) for name in names)
    assert not any("/model/" in name or "/trained_models/" in name for name in names)
