"""Entry-layer charged admission owned exclusively by registered Torch V4 profiles."""

from __future__ import annotations

import pytest
from ase import Atoms

from maple.function.calculator.set_calculator import SetCalculator
from maple.function.read.input_reader import InputReader
from maple.function.route2_smd_profiles import (
    PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_TORCH_CPU_V3_PROFILE,
    PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_TORCH_CPU_V4_PROFILE,
    PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_TORCH_CUDA_V4_PROFILE,
    PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_WORKFLOW_PROFILE,
    pure_macepolar_allows_charged,
    route2_smd_profile_spec,
)


def _charged_atoms(formula: str, *, charge: int, multiplicity: int = 1) -> Atoms:
    if formula == "NH4":
        positions = [
            [0.0, 0.0, 0.0],
            [0.63, 0.63, 0.63],
            [-0.63, -0.63, 0.63],
            [-0.63, 0.63, -0.63],
            [0.63, -0.63, -0.63],
        ]
    elif formula == "OH":
        positions = [[0.0, 0.0, 0.0], [0.97, 0.0, 0.0]]
    else:  # pragma: no cover - test helper misuse
        raise AssertionError(formula)
    return Atoms(
        formula,
        positions=positions,
        info={"charge": charge, "mult": multiplicity},
    )


def _solvation_options(
    profile: str | None, *, provider: str = "torch"
) -> dict[str, object]:
    options: dict[str, object] = {
        "method": "smd",
        "implicit": "water",
        "experimental": True,
        "provider": provider,
        "response": "frozen",
        "standard_state": "1m",
    }
    if profile is not None:
        options["profile"] = profile
    return options


def _builder(
    atoms: Atoms,
    *,
    profile: str | None,
    device: str = "cpu",
    provider: str = "torch",
    output: str = "entry-contract.out",
) -> SetCalculator:
    return SetCalculator(
        device=device,
        model="macepolm",
        output=output,
        atoms=atoms,
        implicit="smd",
        solvent="water",
        solvation_options=_solvation_options(profile, provider=provider),
    )


@pytest.mark.parametrize(
    "profile",
    [
        PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_TORCH_CPU_V4_PROFILE,
        PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_TORCH_CUDA_V4_PROFILE,
    ],
)
def test_only_exact_registered_v4_profiles_allow_charged(profile):
    assert pure_macepolar_allows_charged(profile) is True
    assert pure_macepolar_allows_charged(route2_smd_profile_spec(profile)) is True

    assert (
        pure_macepolar_allows_charged(
            PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_TORCH_CPU_V3_PROFILE
        )
        is False
    )
    assert pure_macepolar_allows_charged("unregistered-v4") is False


@pytest.mark.parametrize(
    "formula, charge, profile, device",
    [
        (
            "NH4",
            1,
            PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_TORCH_CPU_V4_PROFILE,
            "cpu",
        ),
        (
            "OH",
            -1,
            PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_TORCH_CPU_V4_PROFILE,
            "cpu",
        ),
        (
            "NH4",
            1,
            PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_TORCH_CUDA_V4_PROFILE,
            "cuda:0",
        ),
        (
            "OH",
            -1,
            PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_TORCH_CUDA_V4_PROFILE,
            "cuda:0",
        ),
    ],
)
def test_set_calculator_v4_validation_accepts_closed_shell_ions(
    formula, charge, profile, device, tmp_path
):
    builder = _builder(
        _charged_atoms(formula, charge=charge),
        profile=profile,
        device=device,
        output=str(tmp_path / "job.out"),
    )

    builder._validate_solvent_config()

    assert (
        builder.solvation_options["profile"]
        == profile
    )


def test_set_calculator_default_torch_v4_validation_accepts_charged(tmp_path):
    builder = _builder(
        _charged_atoms("NH4", charge=1),
        profile=None,
        output=str(tmp_path / "job.out"),
    )

    builder._validate_solvent_config()

    assert (
        builder.solvation_options["profile"]
        == PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_TORCH_CPU_V4_PROFILE
    )


@pytest.mark.parametrize(
    "profile, provider, atoms",
    [
        (
            PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_TORCH_CPU_V3_PROFILE,
            "torch",
            _charged_atoms("NH4", charge=1),
        ),
        (
            PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_WORKFLOW_PROFILE,
            "pyddx",
            _charged_atoms("OH", charge=-1),
        ),
        (
            PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_TORCH_CPU_V4_PROFILE,
            "torch",
            _charged_atoms("NH4", charge=1, multiplicity=2),
        ),
    ],
)
def test_entry_validation_does_not_expand_legacy_or_spin_domain(
    profile, provider, atoms, tmp_path
):
    builder = _builder(
        atoms,
        profile=profile,
        provider=provider,
        output=str(tmp_path / "job.out"),
    )

    with pytest.raises(ValueError, match="closed-shell"):
        builder._validate_solvent_config()


@pytest.mark.parametrize(
    "formula, charge, coordinate_lines",
    [
        (
            "NH4",
            1,
            (
                "N 0.0 0.0 0.0",
                "H 0.63 0.63 0.63",
                "H -0.63 -0.63 0.63",
                "H -0.63 0.63 -0.63",
                "H 0.63 -0.63 -0.63",
            ),
        ),
        ("OH", -1, ("O 0.0 0.0 0.0", "H 0.97 0.0 0.0")),
    ],
)
def test_input_reader_v4_accepts_closed_shell_ions(
    formula, charge, coordinate_lines, tmp_path
):
    del formula
    input_path = tmp_path / "charged.inp"
    input_path.write_text(
        "\n".join(
            (
                "#model=macepolm",
                "#device=cpu",
                "#sp",
                "#solv(implicit=water,method=smd,provider=torch,"
                f"profile={PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_TORCH_CPU_V4_PROFILE},"
                "response=frozen,experimental=true)",
                "",
                f"{charge} 1",
                *coordinate_lines,
                "",
            )
        ),
        encoding="utf-8",
    )

    atoms = InputReader()(str(input_path))

    assert atoms.info["charge"] == charge
    assert atoms.info["mult"] == 1


def test_input_reader_v4_still_rejects_open_shell(tmp_path):
    input_path = tmp_path / "open-shell.inp"
    input_path.write_text(
        "\n".join(
            (
                "#model=macepolm",
                "#device=cpu",
                "#sp",
                "#solv(implicit=water,method=smd,provider=torch,"
                f"profile={PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_TORCH_CPU_V4_PROFILE},"
                "response=frozen,experimental=true)",
                "",
                "1 2",
                "N 0.0 0.0 0.0",
                "H 0.63 0.63 0.63",
                "H -0.63 -0.63 0.63",
                "H -0.63 0.63 -0.63",
                "H 0.63 -0.63 -0.63",
                "",
            )
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="closed-shell"):
        InputReader()(str(input_path))


def test_input_reader_v3_still_rejects_charged(tmp_path):
    input_path = tmp_path / "v3-charged.inp"
    input_path.write_text(
        "\n".join(
            (
                "#model=macepolm",
                "#device=cpu",
                "#sp",
                "#solv(implicit=water,method=smd,provider=torch,"
                f"profile={PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_TORCH_CPU_V3_PROFILE},"
                "response=frozen,experimental=true)",
                "",
                "-1 1",
                "O 0.0 0.0 0.0",
                "H 0.97 0.0 0.0",
                "",
            )
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="neutral closed-shell"):
        InputReader()(str(input_path))
