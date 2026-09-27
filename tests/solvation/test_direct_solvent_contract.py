"""Scientific identity must be enforced before any direct-API side effect."""

from types import SimpleNamespace

import pytest
from ase import Atoms

from maple.function.calculator.extra_correction.implicit import correction as module


@pytest.mark.parametrize(
    "change,match",
    [
        ({"implicit": "methanol"}, "water"),
        ({"profile": "not-obc2"}, "profile"),
        ({"banana": "ignored"}, "options"),
        ({"pressure": 0.0}, "options"),
        ({"explicit": "water"}, "options"),
        ({"model": "bad-model"}, "model"),
        ({"provider": "bogus"}, "provider"),
        ({"inner": "generated"}, "inner"),
        *[
            ({key: None}, key)
            for key in (
                "method",
                "provider",
                "model",
                "profile",
                "implicit",
                "nonpolar",
            )
        ],
    ],
)
def test_direct_invalid_identity_is_rejected_before_side_effects(
    tmp_path, monkeypatch, change, match
):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid direct input reached a charge/provider side effect")

    monkeypatch.setattr(module, "prepare_charges", forbidden)
    monkeypatch.setattr(module, "OpenMMGB", forbidden)
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.8]])
    original_arrays = set(atoms.arrays)
    options = {
        "method": "gb",
        "provider": "openmm",
        "model": "obc2",
        "experimental": True,
        **change,
    }
    with pytest.raises(ValueError, match=match):
        module.ImplicitSolvationCorrection(
            atoms, {"source": "mol2"}, options, output=tmp_path / "bad.out"
        )
    assert not list(tmp_path.iterdir())
    assert set(atoms.arrays) == original_arrays


@pytest.mark.parametrize(
    "model,profile",
    [
        ("hct", "hct-mbondi"),
        ("obc1", "obc1-mbondi2"),
        ("obc2", "obc2-mbondi2"),
        ("gbn", "gbn-bondi"),
        ("gbn2", "gbn2-mbondi3"),
    ],
)
def test_direct_openmm_profiles_match_shared_radius_registry(model, profile):
    options = {"method": "gb", "model": model, "profile": profile, "experimental": True}
    assert module._validate_direct_contract(options, "sp")[:2] == (
        "gb",
        "openmm",
    )


@pytest.mark.parametrize("provider", ["apbs", "ddx", "amber-pbsa"])
def test_direct_pb_does_not_silently_relabel_an_invalid_model_or_solvent(provider):
    for change in (
        {"implicit": "methanol"},
        {"model": "nonlinear"},
        {"profile": "other"},
        {"banana": 1},
    ):
        with pytest.raises(ValueError):
            module._validate_direct_contract(
                {"method": "pb", "provider": provider, "experimental": True, **change},
                "sp",
            )


@pytest.mark.parametrize(
    "provider,error,match",
    [
        ("ddx", ValueError, "kappa"),
        ("amber-pbsa", NotImplementedError, "evidence-gated"),
    ],
)
def test_unavailable_or_incomplete_provider_leaves_atoms_unchanged(
    tmp_path, provider, error, match
):
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.8]])
    arrays = set(atoms.arrays)
    with pytest.raises(error, match=match):
        module.ImplicitSolvationCorrection(
            atoms,
            {"source": "mol2"},
            {"method": "pb", "provider": provider, "experimental": True},
            output=tmp_path / "bad.out",
        )
    assert set(atoms.arrays) == arrays
    assert not list(tmp_path.iterdir())


def test_unqualified_amber_charge_profile_leaves_atoms_unchanged(tmp_path):
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.8]])
    arrays = set(atoms.arrays)
    with pytest.raises(ValueError, match="validated AmberTools CHA-GB profile"):
        module.ImplicitSolvationCorrection(
            atoms,
            {"source": "mol2"},
            {"method": "gb", "provider": "ambertools", "experimental": True},
            output=tmp_path / "bad.out",
        )
    assert set(atoms.arrays) == arrays
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "device,d4", [("cuda", False), ("cuda:0", False), ("mps", False), ("cpu", True)]
)
def test_analytic_composite_gate_rejects_unqualified_device_and_d4_before_model_build(
    device, d4
):
    from maple.function.calculator.set_calculator import SetCalculator

    settings = SimpleNamespace(
        implicit="gb",
        atoms=None,
        model="ani2x",
        device=device,
        d4=d4,
        model_options={"hessian": "analytic"},
    )
    backend = SimpleNamespace(
        SUPPORTS_IMPLICIT_SOLVATION=True,
        SUPPORTED_HESSIAN_MODES=("analytic", "numerical"),
        ANALYTIC_IMPLICIT_HESSIAN_MODELS=("ani2x",),
    )
    with pytest.raises(ValueError, match="CPU.*D4|qualified.*cell"):
        SetCalculator._validate_against_class(settings, backend)


def test_qualified_cpu_no_d4_analytic_cell_remains_accepted():
    from maple.function.calculator.set_calculator import SetCalculator

    settings = SimpleNamespace(
        implicit="gb",
        atoms=None,
        model="ani2x",
        device="cpu",
        d4=False,
        model_options={"hessian": "analytic"},
    )
    backend = SimpleNamespace(
        SUPPORTS_IMPLICIT_SOLVATION=True,
        SUPPORTED_HESSIAN_MODES=("analytic", "numerical"),
        ANALYTIC_IMPLICIT_HESSIAN_MODELS=("ani2x",),
    )
    SetCalculator._validate_against_class(settings, backend)


@pytest.mark.parametrize("device,admitted", [("cpu", True), ("cuda:0", False)])
def test_reference_openmm_does_not_imply_qualified_torch_device(
    water_mol2, tmp_path, device, admitted
):
    pytest.importorskip("openmm")
    from maple.function.read.filereader.mol2_reader import MOL2Reader

    atoms = MOL2Reader(str(water_mol2), charge=0, mult=1)
    correction = module.ImplicitSolvationCorrection(
        atoms,
        {"source": "mol2", "mode": "fixed", "geometry": "keep"},
        {
            "method": "gb",
            "provider": "openmm",
            "model": "obc2",
            "platform": "Reference",
            "experimental": True,
        },
        output=tmp_path / "device.out",
        model_device=device,
    )
    assert correction.provider.platform == "Reference"
    assert correction.provider._derivative_backend is None
    assert correction.analytic_task_derivatives_admitted is admitted
    assert "forces" in correction.supported_properties


@pytest.mark.parametrize(
    "wrapper_d4,option_d4,rejected",
    [
        (False, True, True),
        (False, "on", True),
        (True, False, False),
        (False, "false", False),
    ],
)
def test_analytic_gate_uses_effective_backend_d4_option(
    wrapper_d4, option_d4, rejected
):
    from maple.function.calculator.ani._ani_calculator import ANICalculator
    from maple.function.calculator.set_calculator import SetCalculator

    settings = SimpleNamespace(
        implicit="gb",
        atoms=None,
        model="ani2x",
        device="cpu",
        d4=wrapper_d4,
        model_options={"hessian": "analytic", "d4": option_d4},
    )
    if rejected:
        with pytest.raises(ValueError, match="CPU.*D4"):
            SetCalculator._validate_against_class(settings, ANICalculator)
    else:
        SetCalculator._validate_against_class(settings, ANICalculator)
        assert (
            ANICalculator.build_kwargs_from_options("ani2x", settings.model_options)[
                "d4"
            ]
            is False
        )
