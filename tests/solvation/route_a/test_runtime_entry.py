from __future__ import annotations

from pathlib import Path

import pytest

from maple.function.read.command_control import CommandControl


PROTOCOL = "docs/solvation/route-a/protocol-v1.json"
SAMPLER_SHA = "e5ccf5837f685899811a68754e7c994393bfd1a81720393b03c643b46c70bc69"
SCORER_SHA = "9b64b4fd5153ca578c694abc57806d8111050de6ff652e695c9b525bc4d36469"


def _valid_lines(*, dry_run: bool = True) -> list[str]:
    return [
        (
            "#model=maceoff24m(checkpoint=/tmp/MACE-OFF24_medium.model,"
            f"sha256={SAMPLER_SHA},license_ack=true)"
        ),
        (
            "#scorer=maceomol(checkpoint=/tmp/MACE-omol.model,"
            f"sha256={SCORER_SHA},license_ack=true)"
        ),
        (
            "#outer=smd(provider=mace-polar-pcmsolver,model=mace-polar-1-m,"
            "profile=smd-iefpcm-gaff2-o,response=scf,standard_state=1m,"
            "license_ack=true)"
        ),
        (
            "#solvfe(method=qct,solvent=water,temperature=298.15,"
            f"pressure_bar=1.0,standard_state=1m,protocol={PROTOCOL},"
            f"experimental=true,dry_run={str(dry_run).lower()})"
        ),
    ]


def test_valid_route_a_header_parses_to_separate_provider_specs():
    control = CommandControl.from_settings(_valid_lines())

    assert control.task == "solvfe"
    assert control.params["method"] == "qct"
    assert control.params["solvent"] == "water"
    assert control.params["standard_state"] == "1m"
    assert control.params["experimental"] is True
    assert control.params["dry_run"] is True
    assert control.params["scorer"] == "maceomol"
    assert control.params["scorer_options"]["sha256"] == SCORER_SHA
    assert control.params["outer"] == "smd"
    assert control.params["outer_options"]["provider"] == (
        "mace-polar-pcmsolver"
    )
    assert Path(control.params["protocol"]).as_posix() == PROTOCOL


@pytest.mark.parametrize(
    ("needle", "replacement", "message"),
    [
        ("experimental=true", "experimental=false", "experimental=true"),
        ("solvent=water", "solvent=methanol", "solvent=water"),
        ("standard_state=1m", "standard_state=1atm", "standard_state=1m"),
        ("method=qct", "method=resolv", "method=qct"),
        ("dry_run=true", "dry_run=maybe", "dry_run"),
    ],
)
def test_route_a_domain_fails_closed_in_parser(needle, replacement, message):
    lines = [line.replace(needle, replacement) for line in _valid_lines()]
    with pytest.raises(ValueError, match=message):
        CommandControl.from_settings(lines)


def test_route_a_rejects_nested_solvation_and_other_tasks():
    with pytest.raises(ValueError, match="MUTUALLY_EXCLUSIVE_SOLVATION"):
        CommandControl.from_settings(
            _valid_lines()
            + ["#solv(implicit=water,method=gbsa,experimental=true)"]
        )

    with pytest.raises(ValueError, match="Multiple tasks"):
        CommandControl.from_settings(_valid_lines() + ["#sp"])


def test_route_a_rejects_missing_provider_specs_and_unknown_fields():
    with pytest.raises(ValueError, match="#scorer"):
        CommandControl.from_settings(
            [line for line in _valid_lines() if not line.startswith("#scorer")]
        )

    bad = [
        line.replace("dry_run=true", "dry_run=true,after_results_q=0.2")
        for line in _valid_lines()
    ]
    with pytest.raises(ValueError, match="Unknown SOLVFE parameter"):
        CommandControl.from_settings(bad)

