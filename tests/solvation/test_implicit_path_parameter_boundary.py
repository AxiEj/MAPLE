from types import SimpleNamespace

import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator

from maple.function.dispatcher.convergence import (
    CONVERGENCE_CONTEXT_KEYS,
    CONVERGENCE_THRESHOLDS,
    convergence_thresholds,
)
from maple.function.dispatcher.dispatcher import Dispatcher
from maple.function.dispatcher.implicit_path import validate_path_parameters
from maple.function.dispatcher.irc.algorithm.gs import GSParams
from maple.function.dispatcher.ts.algorithm.neb import NEBParams
from maple.function.dispatcher.ts.algorithm.string import GSMParams
from maple.function.read.command_control import CommandControl


class _ImplicitCalculator(Calculator):
    implemented_properties = []

    def __init__(self):
        super().__init__()
        self.solvent_correction = SimpleNamespace()


def _control(task_line):
    return CommandControl.from_settings([
        "#model=aimnet2025(hessian=analytic,d4=false)",
        task_line,
        "#charge(source=mol2,mode=fixed,geometry=keep)",
        "#solv(implicit=water,method=gb,provider=torch,model=obc2,"
        "profile=obc2-mbondi2,nonpolar=ace,dtype=float32,experimental=true)",
    ])


def _implicit_atom(x=0.0):
    atoms = Atoms("H", positions=[[x, 0.0, 0.0]])
    atoms.calc = _ImplicitCalculator()
    return atoms


@pytest.mark.parametrize(
    "method,task_line,algorithm_name,expected",
    [
        ("gs", "#irc(method=gs,max_steps=7)", "GS", {"max_steps": 7}),
        (
            "neb",
            "#ts(method=neb,n_images=4,max_iter=8,refine=None)",
            "NEB",
            {"n_images": 6, "max_iter": 8},
        ),
        (
            "string",
            "#ts(method=string,n_images=5,max_iter_grow=9,refine=None)",
            "GSM",
            {"n_images": 5, "max_iter_grow": 9},
        ),
    ],
)
@pytest.mark.parametrize(
    "level,expected_thresholds",
    [
        (None, dict(CONVERGENCE_THRESHOLDS["medium"])),
        ("tight", dict(CONVERGENCE_THRESHOLDS["tight"])),
    ],
)
def test_real_dispatcher_and_wrappers_extract_path_parameters_from_full_config(
    tmp_path,
    monkeypatch,
    method,
    task_line,
    algorithm_name,
    expected,
    level,
    expected_thresholds,
):
    import maple.function.dispatcher.irc.irc as irc_module
    import maple.function.dispatcher.ts.ts as ts_module
    import maple.function.dispatcher.irc.algorithm as irc_algorithms
    import maple.function.dispatcher.ts.algorithm as ts_algorithms

    captured = {}
    algorithm = (
        getattr(irc_algorithms, algorithm_name)
        if method == "gs"
        else getattr(ts_algorithms, algorithm_name)
    )

    def capture(instance):
        values = instance.p if method == "gs" else instance.params
        captured.update(vars(values))
        return {"status": "captured"}

    monkeypatch.setattr(algorithm, "run", capture)
    monkeypatch.setattr(irc_module, "bind_admitted_implicit_path", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(ts_module, "bind_admitted_implicit_path", lambda *_args, **_kwargs: True)

    control = _control(task_line)
    command = SimpleNamespace(params=control.as_dict())
    if level is not None:
        command.params["level"] = level
    atoms = _implicit_atom() if method == "gs" else [_implicit_atom(), _implicit_atom(1.0)]
    if method != "gs":
        atoms[1].calc = atoms[0].calc

    Dispatcher()(command, control.task, atoms, str(tmp_path / f"{method}.out"))

    assert {key: captured[key] for key in expected} == expected
    assert {
        key: command.params[key] for key in CONVERGENCE_CONTEXT_KEYS
    } == expected_thresholds
    for image in atoms if isinstance(atoms, list) else [atoms]:
        assert {
            key: getattr(image, key) for key in CONVERGENCE_CONTEXT_KEYS
        } == expected_thresholds
    if method == "gs":
        assert captured["f_max_th"] == expected_thresholds["f_max_th"]
        assert captured["f_rms_th"] == expected_thresholds["f_rms_th"]


def test_dispatcher_and_path_boundary_share_one_convergence_context_schema():
    expected = {
        "extratight": (0.00030, 0.00020, 0.00030, 0.00020),
        "tight": (0.00085, 0.00055, 0.00110, 0.00075),
        "medium": (0.00285, 0.00190, 0.00315, 0.00210),
        "loose": (0.00380, 0.00250, 0.00600, 0.00400),
        "extraloose": (0.00755, 0.00500, 0.01200, 0.00800),
        "superloose": (0.08500, 0.05500, 0.14500, 0.09500),
    }
    names = ("f_max_th", "f_rms_th", "dp_max_th", "dp_rms_th")
    assert {
        level: tuple(thresholds[name] for name in names)
        for level, thresholds in CONVERGENCE_THRESHOLDS.items()
    } == expected
    assert all(
        frozenset(thresholds) == CONVERGENCE_CONTEXT_KEYS
        for thresholds in CONVERGENCE_THRESHOLDS.values()
    )
    assert dict(convergence_thresholds(None)) == dict(CONVERGENCE_THRESHOLDS["medium"])
    assert dict(convergence_thresholds("unknown")) == dict(CONVERGENCE_THRESHOLDS["medium"])
    assert dict(convergence_thresholds("TIGHT")) == dict(CONVERGENCE_THRESHOLDS["medium"])


def test_nested_gs_override_wins_over_dispatcher_convergence_default():
    configuration = {
        "level": "tight",
        **dict(CONVERGENCE_THRESHOLDS["tight"]),
        "gs": {"max_steps": 2, "f_max_th": 0.0012},
    }
    assert validate_path_parameters("gs", configuration, GSParams) == {
        "f_rms_th": CONVERGENCE_THRESHOLDS["tight"]["f_rms_th"],
        "max_steps": 2,
        "f_max_th": 0.0012,
    }


@pytest.mark.parametrize(
    "method,params_class,key",
    [
        ("gs", GSParams, "dp_max_th"),
        ("neb", NEBParams, "f_max_th"),
        ("string", GSMParams, "f_max_th"),
    ],
)
def test_partial_dispatcher_convergence_context_is_not_silently_ignored(
    method, params_class, key
):
    with pytest.raises(ValueError, match=f"Unknown {method} parameter: {key}"):
        validate_path_parameters(method, {key: 0.001}, params_class)


@pytest.mark.parametrize(
    "method,params_class,key,value",
    [
        ("gs", GSParams, "max_steps", 3),
        ("neb", NEBParams, "max_iter", 4),
        ("string", GSMParams, "max_iter_grow", 5),
    ],
)
@pytest.mark.parametrize("nested", [False, True])
def test_direct_path_parameter_forms_remain_valid(
    method, params_class, key, value, nested
):
    request = {method: {key: value}} if nested else {key: value}
    assert validate_path_parameters(method, request, params_class) == {key: value}


@pytest.mark.parametrize("nested", [False, True])
def test_real_irc_wrapper_preserves_direct_parameter_forms(tmp_path, monkeypatch, nested):
    import maple.function.dispatcher.irc.irc as irc_module
    import maple.function.dispatcher.irc.algorithm as algorithms

    captured = {}

    def capture(instance):
        captured.update(vars(instance.p))
        return {"status": "captured"}

    monkeypatch.setattr(algorithms.GS, "run", capture)
    monkeypatch.setattr(irc_module, "bind_admitted_implicit_path", lambda *_args, **_kwargs: True)
    params = {"gs": {"max_steps": 11}} if nested else {"max_steps": 11}
    irc_module.IRC(params, str(tmp_path / "gs.out"), _implicit_atom(), "gs").run()
    assert captured["max_steps"] == 11


@pytest.mark.parametrize("method", ["neb", "string"])
@pytest.mark.parametrize("nested", [False, True])
def test_real_ts_wrapper_preserves_direct_parameter_forms(
    tmp_path, monkeypatch, method, nested
):
    import maple.function.dispatcher.ts.ts as ts_module
    import maple.function.dispatcher.ts.algorithm as algorithms

    key = "max_iter" if method == "neb" else "max_iter_grow"
    value = 12
    fields = {"n_images": 3, key: value, "refine": None}
    params = {method: fields} if nested else fields
    captured = {}
    algorithm = algorithms.NEB if method == "neb" else algorithms.GSM

    def capture(instance):
        captured.update(vars(instance.params))
        return {"status": "captured"}

    monkeypatch.setattr(algorithm, "run", capture)
    monkeypatch.setattr(ts_module, "bind_admitted_implicit_path", lambda *_args, **_kwargs: True)
    left, right = _implicit_atom(), _implicit_atom(1.0)
    right.calc = left.calc
    ts_module.TransitionState(
        str(tmp_path / f"{method}.out"), [left, right], params, method
    ).run()
    assert captured[key] == value


def test_nested_full_configuration_keeps_only_method_parameters():
    full = _control("#ts(method=neb,n_images=4,max_iter=8,refine=None)").as_dict()
    assert validate_path_parameters("neb", {"neb": full}, NEBParams) == {
        "n_images": 4,
        "max_iter": 8,
        "refine": None,
    }


@pytest.mark.parametrize(
    "configuration,error",
    [
        ({"max_steps": 1, "gs": {"f_max_th": 0.01}}, "mixes flat"),
        ({"neb": {"max_iter": 1}, "string": {}}, "multiple method"),
        ({"MAX_STEPS": 1, "max_steps": 1}, "duplicate"),
        ({"step_length_bohr": 0.1, "sd_len_bohr": 0.1}, "duplicate"),
        ({"unknown": 1}, "Unknown gs parameter"),
    ],
)
def test_path_parameter_normalization_rejects_ambiguous_requests(configuration, error):
    with pytest.raises(ValueError, match=error):
        validate_path_parameters("gs", configuration, GSParams)


@pytest.mark.parametrize(
    "configuration",
    [
        {"max_steps": 1, "gs": {"f_max_th": 0.01}},
        {"MAX_STEPS": 1, "max_steps": 1},
        {"step_length_bohr": 0.1, "sd_len_bohr": 0.1},
        {"unknown": 1},
    ],
)
def test_real_wrapper_rejects_ambiguity_before_binding_or_algorithm_work(
    tmp_path, monkeypatch, configuration
):
    import maple.function.dispatcher.irc.irc as irc_module
    import maple.function.dispatcher.irc.algorithm as algorithms

    side_effects = []
    monkeypatch.setattr(
        irc_module,
        "bind_admitted_implicit_path",
        lambda *_args, **_kwargs: side_effects.append("bind"),
    )
    monkeypatch.setattr(
        algorithms.GS,
        "run",
        lambda _self: side_effects.append("algorithm"),
    )
    with pytest.raises(ValueError):
        irc_module.IRC(
            configuration, str(tmp_path / "rejected.out"), _implicit_atom(), "gs"
        ).run()
    assert side_effects == []


@pytest.mark.parametrize("configuration", [[], {1: "value"}, {"gs": []}])
def test_path_parameter_normalization_rejects_non_mapping_shapes(configuration):
    with pytest.raises(TypeError):
        validate_path_parameters("gs", configuration, GSParams)


def test_none_path_parameters_preserve_default_semantics():
    assert validate_path_parameters("gs", None, GSParams) == {}


@pytest.mark.parametrize(
    "task_line",
    [
        "#irc(method=gs,max_steps=1,MAX_STEPS=2)",
        "#irc(method=gs,step_length_bohr=0.1,sd_len_bohr=0.2)",
    ],
)
def test_raw_nested_cli_rejects_duplicate_canonical_parameters(task_line):
    with pytest.raises(ValueError, match="[Dd]uplicate"):
        _control(task_line)


def test_raw_nested_parser_preserves_openmm_device_index_commas():
    parsed = {}
    CommandControl._parse_nested(parsed, "platform=CUDA,device_index=0,1,precision=mixed")
    assert parsed == {
        "platform": "CUDA",
        "device_index": "0,1",
        "precision": "mixed",
    }
