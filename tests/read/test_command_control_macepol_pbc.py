import pytest

from maple.function.read.command_control import CommandControl


def test_macepol_pbc_model_name_keeps_hyphens():
    control = CommandControl.from_settings([
        "#model=macepol-pbc-small",
        "#sp",
    ])

    assert control.params["model"] == "macepol-pbc-small"


def test_macepol_pbc_default_dtype_round_trips_to_validator():
    control = CommandControl.from_settings([
        "#model=macepol-pbc-small(default_dtype=float64)",
        "#sp",
    ])

    assert control.params["model_options"]["default_dtype"] == "float64"

    with pytest.raises(ValueError, match="bfloat16.*Supported values"):
        CommandControl.from_settings([
            "#model=macepol-pbc-small(default_dtype=bfloat16)",
            "#sp",
        ])


def test_aimnet_coulomb_method_alias_is_parser_supported():
    control = CommandControl.from_settings([
        "#model=aimnet2(coulomb_method=dsf)",
        "#sp",
    ])

    assert control.params["model_options"] == {"coulomb": "dsf"}


def test_aimnet_common_model_options_survive_parser_canonicalization():
    control = CommandControl.from_settings([
        "#model=aimnet2(coulomb_method=dsf,model_path=/tmp/aimnet2.pt)",
        "#sp",
    ])

    assert control.params["model_options"] == {
        "coulomb": "dsf",
        "model_path": "/tmp/aimnet2.pt",
    }


@pytest.mark.parametrize(
    "line",
    [
        "#model=aimnet2(coulomb=ewald)",
        "#model=aimnet2(coulomb_method=ewald)",
    ],
)
def test_legacy_aimnet_ewald_is_rejected_by_parser(line):
    with pytest.raises(ValueError, match="Unsupported AIMNet2 Coulomb method: 'ewald'"):
        CommandControl.from_settings([line, "#sp"])


@pytest.mark.parametrize(
    "line",
    [
        "#model=aimnet2-pbc(hessian=numerical)",
        "#model=mace-mp-pbc-small(hessian=numerical)",
        "#model=macepol-pbc-small(hessian=numerical)",
    ],
)
def test_official_pbc_hessian_is_rejected_by_parser(line):
    with pytest.raises(ValueError, match="hessian.*Supported options"):
        CommandControl.from_settings([line, "#sp"])
