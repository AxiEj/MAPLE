"""WS1 — MD ensemble dataclasses are the single source of truth for defaults.

DEFAULTS["md"] only selects the ensemble; every physics default comes from the
NVE/NVT/NPT dataclasses, resolved at _init_params time so that the precedence
explicit-inline > MDP > dataclass holds and no physics default is duplicated in
command_control.
"""

from dataclasses import fields

import pytest

from maple.function.read.command_control import CommandControl
from maple.function.dispatcher.md.ensemble.nve import NVEParams
from maple.function.dispatcher.md.ensemble.nvt import NVTParams
from maple.function.dispatcher.md.ensemble.npt import NPTParams


def _md_params(lines):
    return CommandControl.from_settings(lines).params


# ──────────────────────────────────────────────────────────────────────────
# Single source of truth + whitelist lock
# ──────────────────────────────────────────────────────────────────────────

def test_defaults_md_contains_only_ensemble():
    # Reintroducing any physics default here (e.g. timestep) must fail CI.
    assert CommandControl.DEFAULTS["md"] == {"ensemble": "nve"}


def test_md_param_keys_equals_dataclass_field_union():
    expected = {"ensemble", "mdp"}
    for dataclass_type in (NVEParams, NVTParams, NPTParams):
        expected.update(f.name for f in fields(dataclass_type))
    assert CommandControl._md_param_keys() == expected


def test_bare_md_does_not_hardcode_physics_defaults():
    params = _md_params(["#md"])
    assert params["ensemble"] == "nve"
    # No physics default leaks into the parsed params; the dataclass owns them.
    for key in ("timestep", "steps", "temperature", "thermostat", "remove_com_every"):
        assert key not in params


def test_bare_md_resolves_to_nve_dataclass_defaults():
    params = _md_params(["#md"])
    resolved = NVEParams()
    for field in fields(NVEParams):
        if field.name in params:
            assert getattr(resolved, field.name) == params[field.name]
    # Spot-check the values that used to be overridden by DEFAULTS["md"].
    assert resolved.timestep == 0.1
    assert resolved.steps == 100000
    assert resolved.remove_com_every == 0
    assert resolved.remove_angular is True


# ──────────────────────────────────────────────────────────────────────────
# Precedence: explicit inline > MDP > dataclass
# ──────────────────────────────────────────────────────────────────────────

def test_inline_beats_mdp_beats_dataclass(tmp_path):
    mdp = tmp_path / "run.mdp"
    mdp.write_text(
        "timestep = 0.5\n"
        "steps = 5000\n"
        "temperature = 250\n"
        "friction = 0.005\n"
    )
    params = _md_params([f"#md(ensemble=nvt, timestep=0.3, mdp={mdp})"])
    assert params["timestep"] == 0.3      # inline wins over MDP (0.5) and dataclass (0.1)
    assert params["steps"] == 5000        # MDP wins over dataclass (100000)
    assert params["temperature"] == 250   # MDP wins over dataclass (300)
    assert params["friction"] == 0.005    # MDP physics key still imports


def test_mdp_physics_keys_still_import_after_defaults_shrink(tmp_path):
    # Regression guard: filtering MDP keys on DEFAULTS["md"] (now {ensemble}) would
    # silently drop every physics key.  The filter must use _md_param_keys().
    mdp = tmp_path / "run.mdp"
    mdp.write_text("tau_t = 321\nremove_com_every = 7\n")
    params = _md_params([f"#md(ensemble=nvt, mdp={mdp})"])
    assert params["tau_t"] == 321
    assert params["remove_com_every"] == 7


def test_mdp_unknown_key_is_not_imported(tmp_path):
    mdp = tmp_path / "run.mdp"
    mdp.write_text("not_a_real_md_key = 5\ntimestep = 0.2\n")
    params = _md_params([f"#md(mdp={mdp})"])
    assert "not_a_real_md_key" not in params
    assert params["timestep"] == 0.2


# ──────────────────────────────────────────────────────────────────────────
# Whitelist now accepts every dataclass field (e.g. verbose), rejects unknowns
# ──────────────────────────────────────────────────────────────────────────

def test_verbose_is_accepted_via_cli():
    params = _md_params(["#md(verbose=2)"])
    assert params["verbose"] == 2


def test_allow_partial_pbc_is_accepted_via_cli():
    params = _md_params(["#md(ensemble=nvt, allow_partial_pbc=true)"])
    assert params["allow_partial_pbc"] is True


def test_unknown_md_key_is_rejected():
    with pytest.raises(ValueError, match="Unknown MD parameter"):
        _md_params(["#md(bogus_key=1)"])
