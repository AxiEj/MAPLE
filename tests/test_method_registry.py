from dataclasses import fields

import pytest

from maple.function.dispatcher.method_registry import (
    CapabilityKey,
    METHOD_SPECS,
    PublicMethodKey,
    implemented_methods,
    normalize_public_method,
    required_capabilities,
    validate_input_arity,
    validate_method_parameters,
)


def test_registry_covers_public_methods_and_exact_dataclass_fields():
    assert implemented_methods("irc") == {"gs", "hpc", "lqa", "eulerpc"}
    assert implemented_methods("ts") == {"prfo", "dimer", "neb", "string", "autoneb"}
    for spec in METHOD_SPECS.values():
        assert set(spec.parameter_rules) == {
            field.name for field in fields(spec.parameter_dataclass())
        }


@pytest.mark.parametrize(
    "key,count",
    [
        (PublicMethodKey.IRC_HPC, 1),
        (PublicMethodKey.TS_PRFO, 1),
        (PublicMethodKey.TS_NEB, 2),
        (PublicMethodKey.TS_STRING, 2),
        (PublicMethodKey.TS_AUTONEB, 2),
    ],
)
def test_input_arities_accept_the_public_domain(key, count):
    validate_input_arity(key, count)


@pytest.mark.parametrize(
    "key,count",
    [
        (PublicMethodKey.IRC_HPC, 2),
        (PublicMethodKey.TS_PRFO, 2),
        (PublicMethodKey.TS_NEB, 1),
        (PublicMethodKey.TS_STRING, 1),
        (PublicMethodKey.TS_STRING, 3),
        (PublicMethodKey.TS_AUTONEB, 1),
    ],
)
def test_input_arities_reject_invalid_shapes(key, count):
    with pytest.raises(ValueError):
        validate_input_arity(key, count)


def test_alias_collisions_and_mixed_parameter_sources_fail_before_normalization():
    with pytest.raises(ValueError, match="duplicate"):
        validate_method_parameters(PublicMethodKey.IRC_GS, {"MAX_STEPS": 1, "max_steps": 2})
    with pytest.raises(ValueError, match="resolves"):
        validate_method_parameters(
            PublicMethodKey.IRC_GS,
            {"max_points": 1, "max_steps": 2},
        )
    with pytest.raises(ValueError, match="mixes flat"):
        validate_method_parameters(
            PublicMethodKey.TS_NEB,
            {"neb": {"max_iter": 1}, "step0": 0.1},
        )


def test_refinement_owns_active_fields_and_capabilities():
    with pytest.raises(ValueError, match="inactive"):
        validate_method_parameters(
            PublicMethodKey.TS_NEB,
            {"refine": None, "cineb_max_iter": 4},
        )
    neb = validate_method_parameters(
        PublicMethodKey.TS_NEB,
        {"refine": "nebts", "cineb_max_iter": 4, "prfo_max_iter": 5},
    )
    assert required_capabilities(PublicMethodKey.TS_NEB, neb) == {
        CapabilityKey.TS_NEB_BASE,
        CapabilityKey.TS_NEB_CI,
        CapabilityKey.TS_NEB_TS,
    }
    string = validate_method_parameters(
        PublicMethodKey.TS_STRING,
        {"refine": "stringts", "max_iter_cistring": 4},
    )
    assert required_capabilities(PublicMethodKey.TS_STRING, string) == {
        CapabilityKey.TS_STRING_BASE,
        CapabilityKey.TS_STRING_CI,
        CapabilityKey.TS_STRING_TS,
    }


@pytest.mark.parametrize(
    "prfo_params,error",
    [
        ({"max_iter": True}, TypeError),
        ({"trust_radius": -1.0}, ValueError),
        ({"unknown_key": 1}, ValueError),
        ({"prfo": {"max_iter": 1}}, ValueError),
        ({"MAX_ITER": 1, "max_iter": 2}, ValueError),
        ({"project_rigid_modes": True}, ValueError),
        ({"recalc": None}, TypeError),
        ({"recalc": 0}, ValueError),
    ],
)
def test_string_prfo_params_use_the_canonical_prfo_contract(prfo_params, error):
    with pytest.raises(error):
        validate_method_parameters(
            PublicMethodKey.TS_STRING,
            {"refine": "stringts", "prfo_params": prfo_params},
        )


def test_string_prfo_params_are_normalized_before_dispatch():
    normalized = validate_method_parameters(
        PublicMethodKey.TS_STRING,
        {"refine": "stringts", "prfo_params": {"MAX_ITER": 3}},
    )
    assert normalized["prfo_params"] == {"max_iter": 3}


@pytest.mark.parametrize(
    "params",
    [
        {"trust_min": 0.3, "trust_radius": 0.2},
        {"trust_radius": 0.2, "trust_max": 0.1},
    ],
)
def test_prfo_cross_field_invariants_fail_at_the_registry_boundary(params):
    with pytest.raises(ValueError):
        validate_method_parameters(PublicMethodKey.TS_PRFO, params)


def test_dimer_preserves_existing_derivative_admission_without_advanced_key():
    key = normalize_public_method("ts", "dimer")
    params = validate_method_parameters(key, {"use_hvp": True})
    assert required_capabilities(key, params) == frozenset()


def test_autoneb_domain_and_capability_selection_are_explicit():
    with pytest.raises(ValueError):
        validate_method_parameters(PublicMethodKey.TS_AUTONEB, {"n_max": 2, "climb": False})
    base = validate_method_parameters(
        PublicMethodKey.TS_AUTONEB,
        {"n_max": 3, "climb": False},
    )
    assert required_capabilities(PublicMethodKey.TS_AUTONEB, base) == {
        CapabilityKey.TS_AUTONEB_BASE,
    }


@pytest.mark.parametrize("key", [PublicMethodKey.IRC_HPC, PublicMethodKey.IRC_EULERPC])
def test_dwi_grid_size_must_be_even_at_the_registry_boundary(key):
    with pytest.raises(ValueError, match="positive even"):
        validate_method_parameters(key, {"dwi_n": 3})
    assert validate_method_parameters(key, {"dwi_n": 4}) == {"dwi_n": 4}


def test_unknown_public_method_is_rejected():
    with pytest.raises(ValueError):
        normalize_public_method("ts", "recursive-autoneb")
