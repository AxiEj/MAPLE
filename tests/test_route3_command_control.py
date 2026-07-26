from pathlib import Path

import pytest

from maple.function.read.command_control import CommandControl


def parse(*lines):
    return CommandControl.from_settings(list(lines))


def test_experimental_gb_polar_cluster_continuum_accepts_first_shell():
    control = parse(
        "#model=ani2x",
        "#sp(verbose=0)",
        "#solv(explicit=water,number=4,implicit=water,"
        "method=experimental-gb-polar,experimental=true)",
    )

    assert control.params["solv"] == {
        "explicit": "water",
        "number": 4,
        "implicit": "water",
        "method": "experimental-gb-polar",
        "experimental": True,
        "shape": "sphere",
        "clash_method": "vdw",
        "radius": 10.0,
    }


def test_experimental_gb_polar_cluster_continuum_remains_energy_only():
    with pytest.raises(ValueError, match="energy-only"):
        parse(
            "#model=ani2x",
            "#opt(lbfgs)",
            "#solv(explicit=water,number=4,implicit=water,"
            "method=experimental-gb-polar,experimental=true)",
        )


def test_tblite_alpb_cluster_continuum_accepts_force_tasks():
    control = parse(
        "#model=ani2x",
        "#opt(lbfgs)",
        "#solv(explicit=water,number=4,implicit=water,method=alpb,provider=tblite,experimental=true)",
    )

    assert control.params["solv"]["method"] == "alpb"
    assert control.params["solv"]["provider"] == "tblite"


def test_tblite_alpb_requires_explicit_provider():
    with pytest.raises(ValueError, match="provider=tblite"):
        parse(
            "#model=ani2x",
            "#sp",
            "#solv(implicit=water,method=alpb,experimental=true)",
        )


@pytest.mark.parametrize(
    "settings",
    [
        (
            "#sp",
            "#solv(implicit=water,method=alpb,provider=tblite)",
        ),
        (
            "#sp",
            "#pbc(10,10,10)",
            "#solv(implicit=water,method=alpb,provider=tblite,experimental=true)",
        ),
    ],
)
def test_tblite_alpb_fails_closed(settings):
    with pytest.raises(ValueError):
        parse("#model=ani2x", *settings)


def test_provider_is_rejected_for_experimental_gb_polar():
    with pytest.raises(ValueError, match="provider"):
        parse(
            "#model=ani2x",
            "#sp",
            "#solv(implicit=water,method=experimental-gb-polar,"
            "provider=tblite,experimental=true)",
        )


@pytest.mark.parametrize("provider", ["maple-qeq-heuristic", "none"])
def test_experimental_gb_polar_accepts_named_or_default_provider(provider):
    control = parse(
        "#model=ani2x",
        "#sp",
        "#solv(implicit=water,method=experimental-gb-polar,"
        f"provider={provider},experimental=true)",
    )

    expected = "maple-qeq-heuristic" if provider != "none" else None
    assert control.params["solv"].get("provider") == expected


def test_legacy_gbsa_name_fails_with_canonical_replacement():
    with pytest.raises(ValueError, match="experimental-gb-polar"):
        parse(
            "#model=ani2x",
            "#sp",
            "#solv(implicit=water,method=gbsa,experimental=true)",
        )


@pytest.mark.parametrize(
    "relative_path",
    [
        "examples/solvation/route3/experimental_gb_polar_first_shell.inp",
        "examples/solvation/route3/tblite_alpb_prebuilt_cluster.inp",
    ],
)
def test_documented_route3_examples_parse(relative_path):
    lines = Path(relative_path).read_text().splitlines()
    control = CommandControl.from_settings([line for line in lines if line.startswith("#")])

    assert control.params["solv"]["implicit"] == "water"
