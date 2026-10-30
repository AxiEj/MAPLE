from __future__ import annotations

from pathlib import Path

import pytest
from ase import Atoms
import numpy as np

from maple.function.calculator.extra_correction.implicit.correction import (
    _validate_direct_contract,
)
from maple.function.calculator.extra_correction.implicit.charges import prepare_charges
from maple.function.calculator.extra_correction.implicit.selector import (
    resolve_implicit_provider,
)
from maple.function.read.command_control import CommandControl
from maple.function.read.charge_options import normalize_charge_options


def _parse(*lines: str) -> dict:
    return CommandControl.from_settings(list(lines)).as_dict()


def _private_options(topology: Path | str = "prepared-topology.json") -> dict:
    return {
        "implicit": "water",
        "method": "gb",
        "model": "chagb",
        "profile": "chagb-bondi-pbsa-inp2",
        "nonpolar": "cavity-dispersion",
        "dtype": "float32",
        "prepared_topology": str(topology),
        "prepared_topology_sha256": "a" * 64,
        "experimental": True,
    }


def test_selector_resolves_only_explicit_chagb_model_to_private_backend():
    assert resolve_implicit_provider({"method": "gb"}) == "torch"
    assert resolve_implicit_provider({"method": "gb", "model": "chagb"}) == (
        "private-torch-cha"
    )
    assert (
        resolve_implicit_provider(
            {"method": "gb", "model": "chagb", "provider": "ambertools"}
        )
        == "ambertools"
    )


def test_missing_prepared_topology_still_resolves_private_then_fails_contract():
    options = _private_options()
    del options["prepared_topology"]
    assert resolve_implicit_provider(options) == "private-torch-cha"
    with pytest.raises(ValueError, match="prepared_topology"):
        _validate_direct_contract(options, "sp")


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("dtype", "float64", "dtype=float32"),
        ("mode", "native", "energy-only"),
        ("inner", "prebuilt", "no inner"),
        ("platform", "cuda", "model device"),
        ("unknown_private_knob", 1, "does not support options"),
    ],
)
def test_direct_private_contract_rejects_invalid_selectors_without_io(
    key, value, message, monkeypatch
):
    options = _private_options()
    options[key] = value
    monkeypatch.setattr(
        Path, "read_bytes", lambda self: pytest.fail("filesystem touched")
    )
    with pytest.raises(ValueError, match=message):
        _validate_direct_contract(options, "sp")


@pytest.mark.parametrize("task", ["opt", "scan", "freq", "ts", "md"])
def test_direct_private_contract_is_sp_only(task):
    with pytest.raises(ValueError, match="single-point energy-only"):
        _validate_direct_contract(_private_options(), task)


def test_invalid_private_contract_has_no_audit_charge_or_atom_side_effects(monkeypatch):
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    before = atoms.copy()
    options = _private_options()
    options["dtype"] = "float64"
    monkeypatch.setattr(
        Path, "mkdir", lambda *args, **kwargs: pytest.fail("audit mkdir")
    )
    monkeypatch.setattr(
        "maple.function.calculator.extra_correction.implicit.correction.prepare_charges",
        lambda *args, **kwargs: pytest.fail("charge preparation"),
    )
    from maple.function.calculator.extra_correction.implicit.correction import (
        ImplicitSolvationCorrection,
    )

    with pytest.raises(ValueError, match="dtype=float32"):
        ImplicitSolvationCorrection(
            atoms,
            {"source": "mol2", "mode": "fixed", "geometry": "keep"},
            options,
            task_context="sp",
        )
    assert np.array_equal(atoms.numbers, before.numbers)
    assert np.array_equal(atoms.positions, before.positions)
    assert atoms.info == before.info


def test_command_contract_normalizes_private_profile_and_paths(tmp_path):
    topology = tmp_path / "prepared.json"
    params = _parse(
        "#model=ani2x",
        "#sp",
        "#charge(source=mol2,mode=fixed,geometry=keep)",
        (
            "#solv(implicit=water,method=gb,model=chagb,dtype=float32,"
            f"prepared_topology={topology},prepared_topology_sha256={'b' * 64},"
            "experimental=true)"
        ),
    )
    assert params["solv"] == {
        "implicit": "water",
        "method": "gb",
        "model": "chagb",
        "dtype": "float32",
        "prepared_topology": str(topology),
        "prepared_topology_sha256": "b" * 64,
        "experimental": True,
        "provider": "private-torch-cha",
        "profile": "chagb-bondi-pbsa-inp2",
        "nonpolar": "cavity-dispersion",
    }
    assert params.get("device", "cpu") == "cpu"


def test_command_contract_accepts_explicit_cpu_for_private_chagb():
    params = _parse(
        "#model=ani2x",
        "#device=cpu",
        "#sp",
        "#charge(source=prepared)",
        (
            "#solv(implicit=water,method=gb,model=chagb,"
            "prepared_topology=x.json,"
            f"prepared_topology_sha256={'c' * 64},experimental=true)"
        ),
    )
    assert params["device"] == "cpu"


def test_command_contract_preserves_explicit_cuda_index_for_private_chagb():
    params = _parse(
        "#model=ani2x",
        "#device=cuda:3",
        "#sp",
        "#charge(source=prepared)",
        (
            "#solv(implicit=water,method=gb,model=chagb,"
            "prepared_topology=x.json,"
            f"prepared_topology_sha256={'c' * 64},experimental=true)"
        ),
    )
    assert params["device"] == "cuda:3"


def test_command_contract_preserves_leading_zero_topology_sha256():
    digest = "0" + "1" * 63
    params = _parse(
        "#model=ani2x",
        "#sp",
        "#charge(source=mol2)",
        (
            "#solv(implicit=water,method=gb,model=chagb,"
            f"prepared_topology=x.json,prepared_topology_sha256={digest},"
            "experimental=true)"
        ),
    )
    assert params["solv"]["prepared_topology_sha256"] == digest


def test_prepared_charge_normalizer_is_fixed_keep_without_tool_options():
    assert normalize_charge_options({"source": "prepared"}) == {
        "source": "prepared",
        "mode": "fixed",
        "geometry": "keep",
    }
    with pytest.raises(ValueError, match="validated private topology"):
        normalize_charge_options({"source": "prepared", "method": "am1bcc"})


def test_generic_charge_preparation_rejects_prepared_source(tmp_path):
    with pytest.raises(ValueError, match="only through.*private-torch-cha"):
        prepare_charges(
            Atoms("H", positions=[[0.0, 0.0, 0.0]]),
            {"source": "prepared"},
            tmp_path,
        )


def test_command_contract_accepts_prepared_charge_only_for_private_chagb():
    params = _parse(
        "#model=ani2x",
        "#sp",
        "#charge(source=prepared)",
        (
            "#solv(implicit=water,method=gb,model=chagb,"
            f"prepared_topology=x.json,prepared_topology_sha256={'d' * 64},"
            "experimental=true)"
        ),
    )
    assert params["charge"] == {
        "source": "prepared",
        "mode": "fixed",
        "geometry": "keep",
    }
    with pytest.raises(ValueError, match="only with.*private-torch-cha"):
        _parse(
            "#model=ani2x",
            "#sp",
            "#charge(source=prepared)",
            "#solv(implicit=water,method=gb,model=obc2,experimental=true)",
        )


@pytest.mark.parametrize(
    "solvation_line",
    [None, "#solv(explicit=water)"],
)
def test_command_contract_rejects_prepared_charge_outside_private_implicit_sp(
    solvation_line,
):
    lines = ["#model=ani2x", "#sp", "#charge(source=prepared)"]
    if solvation_line is not None:
        lines.append(solvation_line)
    with pytest.raises(ValueError, match="only with.*private-torch-cha"):
        _parse(*lines)


@pytest.mark.parametrize("selector", ["prepared_topology", "prepared_topology=123"])
def test_command_contract_requires_topology_path_string(selector):
    with pytest.raises(ValueError, match="non-empty filesystem path string"):
        _parse(
            "#model=ani2x",
            "#sp",
            "#charge(source=mol2)",
            (
                "#solv(implicit=water,method=gb,model=chagb,"
                f"{selector},prepared_topology_sha256={'c' * 64},"
                "experimental=true)"
            ),
        )


@pytest.mark.parametrize(
    ("task_line", "charge_line", "extra", "message"),
    [
        ("#opt", "#charge(source=mol2)", "", "single-point energy-only"),
        ("#sp", "#charge(source=maple)", "", "source=mol2"),
        ("#sp", "#charge(source=mol2,mode=dynamic)", "", "fixed"),
        ("#sp", "#charge(source=mol2,geometry=provider)", "", "geometry=keep"),
        ("#sp", "#charge(source=mol2)", ",platform=cuda", "CUDA|OpenMM"),
        ("#sp", "#charge(source=mol2)", ",dtype=float64", "dtype=float32"),
    ],
)
def test_command_contract_rejects_private_out_of_scope_use(
    task_line, charge_line, extra, message
):
    with pytest.raises(ValueError, match=message):
        _parse(
            "#model=ani2x",
            task_line,
            charge_line,
            (
                "#solv(implicit=water,method=gb,model=chagb,"
                "prepared_topology=x.json,"
                f"prepared_topology_sha256={'c' * 64}{extra},experimental=true)"
            ),
        )
