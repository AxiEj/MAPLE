from __future__ import annotations

from pathlib import Path

from ase import Atoms
import pytest

from maple.function.calculator.set_calculator import SetCalculator

MODEL_DIR = (
    Path(__file__).resolve().parents[2] / "maple/function/calculator/model"
)


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("maceoff23m", [1, 6, 7, 8, 9, 15, 16, 17, 35, 53]),
        (
            "aimnet2",
            [1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 33, 34, 35, 53],
        ),
        ("ani2x", [1, 6, 7, 8, 9, 16, 17]),
    ],
)
def test_shipped_molecular_checkpoint_exposes_element_domain(
    model, expected, tmp_path
):
    pytest.importorskip("torch")
    checkpoint = MODEL_DIR / f"{model}.pt"
    if not checkpoint.is_file():
        pytest.skip(f"Local checkpoint is unavailable: {checkpoint}")
    atoms = Atoms(
        numbers=[6, 1, 1, 1, 1],
        positions=[
            [0.0, 0.0, 0.0],
            [0.6, 0.6, 0.6],
            [-0.6, -0.6, 0.6],
            [-0.6, 0.6, -0.6],
            [0.6, -0.6, -0.6],
        ],
    )

    calculator = SetCalculator(
        "cpu",
        model,
        str(tmp_path / f"{model}.log"),
        atoms=atoms,
    ).set_calculator()

    assert calculator.atomic_numbers == expected


def test_set_calculator_rejects_unsupported_element_before_evaluation(tmp_path):
    pytest.importorskip("torch")
    checkpoint = MODEL_DIR / "ani2x.pt"
    if not checkpoint.is_file():
        pytest.skip(f"Local checkpoint is unavailable: {checkpoint}")
    phosphorus = Atoms(numbers=[15], positions=[[0.0, 0.0, 0.0]])

    with pytest.raises(
        ValueError,
        match=r"ani2x.*does not support atomic number\(s\) \[15\]",
    ):
        SetCalculator(
            "cpu",
            "ani2x",
            str(tmp_path / "ani2x-unsupported.log"),
            atoms=phosphorus,
        ).set_calculator()


def test_external_calculator_without_declared_domain_remains_compatible(tmp_path):
    atoms = Atoms(numbers=[53], positions=[[0.0, 0.0, 0.0]])
    setter = SetCalculator(
        "cpu",
        "external",
        str(tmp_path / "external.log"),
        atoms=atoms,
    )

    setter._validate_calculator_element_domain(object())
