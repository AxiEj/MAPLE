from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms
from ase.units import Hartree

from maple.function.calculator.calculator_base import (
    CalcABC,
    ROUTE2_SMD_CALCULATOR_PROFILE,
)
from maple.function.calculator.extra_correction.implicit.correction import (
    ImplicitSolvationCorrection,
)
from maple.function.calculator.extra_correction.implicit.ddpcm_smd import (
    DDPCM_GAFF2_CARBONYL_O_PROFILE,
    DDPCM_SMD_PROFILE,
    DDPCMSMDImplicitSolvation,
)
from maple.function.calculator.extra_correction.implicit.route2_derivative import (
    FULL_REACTION_FIELD_POSITION_DERIVATIVE_CONTRACT_VERSION,
)
from maple.function.calculator.set_calculator import SetCalculator
from maple.function.read.command_control import CommandControl
from maple.function.route2_smd_profiles import (
    DDPCM_GAFF2_CARBONYL_O_MACE_KSPACE40_PROFILE,
    MACEPOL_FORCED_RECIPROCAL_FIXED_BOX40_PROFILE,
    MACEPOL_MOLECULAR_REALSPACE_PROFILE,
)


def _parse(*lines: str) -> dict[str, object]:
    return CommandControl.from_settings(list(lines)).as_dict()


def _atoms() -> Atoms:
    atoms = Atoms(
        "CO",
        positions=[[0.0, 0.0, 0.0], [1.13, 0.0, 0.0]],
    )
    atoms.info.update(
        charge=0,
        mult=1,
        mol2={"atom_types": ["c", "o"]},
    )
    return atoms


def _options() -> dict[str, object]:
    return {
        "method": "smd",
        "implicit": "water",
        "provider": "pyddx",
        "profile": DDPCM_SMD_PROFILE,
        "response": "scf",
        "standard_state": "1m",
        "experimental": True,
    }


def test_public_parser_accepts_explicit_ddpcm_force_candidate():
    params = _parse(
        "#model=macepol-m",
        "#sp(verbose=1)",
        (
            "#solv(implicit=water,method=smd,provider=pyddx,"
            f"profile={DDPCM_SMD_PROFILE},response=scf,"
            "standard_state=1m,experimental=true)"
        ),
    )

    assert params["solv"] == _options()


def test_public_parser_accepts_only_the_versioned_reciprocal_profile():
    options = {
        **_options(),
        "profile": DDPCM_GAFF2_CARBONYL_O_MACE_KSPACE40_PROFILE,
    }
    params = _parse(
        "#model=macepol-m",
        "#sp(verbose=1)",
        (
            "#solv(implicit=water,method=smd,provider=pyddx,"
            f"profile={DDPCM_GAFF2_CARBONYL_O_MACE_KSPACE40_PROFILE},"
            "response=scf,standard_state=1m,experimental=true)"
        ),
    )

    assert params["solv"] == options

    builder = SetCalculator(
        "cpu",
        "macepol-m",
        "maple.out",
        atoms=_atoms(),
        implicit="smd",
        solvent="water",
        solvation_options=options,
    )
    builder._validate_solvent_config()

    with pytest.raises(ValueError, match="Unknown solvation parameter.*evaluator"):
        _parse(
            "#model=macepol-m",
            "#sp(verbose=1)",
            (
                "#solv(implicit=water,method=smd,provider=pyddx,"
                f"profile={DDPCM_GAFF2_CARBONYL_O_MACE_KSPACE40_PROFILE},"
                "evaluator=reciprocal,response=scf,standard_state=1m,"
                "experimental=true)"
            ),
        )


def test_pyddx_requires_an_explicit_versioned_profile_at_every_boundary(
    tmp_path,
):
    with pytest.raises(ValueError, match="explicit versioned profile"):
        _parse(
            "#model=macepol-m",
            "#sp(verbose=1)",
            (
                "#solv(implicit=water,method=smd,provider=pyddx,"
                "response=scf,standard_state=1m,experimental=true)"
            ),
        )

    options = _options()
    options.pop("profile")
    with pytest.raises(ValueError, match="explicit versioned profile"):
        DDPCMSMDImplicitSolvation(
            _atoms(),
            options,
            audit_dir=tmp_path / "provider",
        )

    builder = SetCalculator(
        "cpu",
        "macepol-m",
        str(tmp_path / "maple.out"),
        atoms=_atoms(),
        implicit="smd",
        solvent="water",
        solvation_options=options,
    )
    with pytest.raises(ValueError, match="explicit versioned profile"):
        builder._validate_solvent_config()


def test_ddpcm_gaff2_profile_changes_only_typed_carbonyl_oxygen(tmp_path):
    options = {
        **_options(),
        "profile": DDPCM_GAFF2_CARBONYL_O_PROFILE,
    }
    provider = DDPCMSMDImplicitSolvation(
        _atoms(),
        options,
        audit_dir=tmp_path,
    )
    canonical = DDPCMSMDImplicitSolvation(
        _atoms(),
        _options(),
        audit_dir=tmp_path / "canonical",
    )

    assert provider.coulomb_radii_angstrom[0] == pytest.approx(
        canonical.coulomb_radii_angstrom[0]
    )
    assert provider.coulomb_radii_angstrom[1] == pytest.approx(1.70)
    assert provider.coulomb_radii_angstrom[1] != pytest.approx(
        canonical.coulomb_radii_angstrom[1]
    )


def test_public_parser_keeps_pcmsolver_energy_only():
    with pytest.raises(ValueError, match="does not provide forces"):
        _parse(
            "#model=macepol-m",
            "#sp(verbose=1)",
            "#solv(implicit=water,method=smd,experimental=true)",
        )


def test_public_parser_rejects_gepol_cavity_policy_for_ddpcm():
    with pytest.raises(ValueError, match="cavity_policy.*PCMSolver"):
        _parse(
            "#model=macepol-m",
            "#sp(verbose=1)",
            (
                "#solv(implicit=water,method=smd,provider=pyddx,"
                f"profile={DDPCM_SMD_PROFILE},response=scf,"
                "cavity_policy=fixed-stability-branch,experimental=true)"
            ),
        )


def test_correction_dispatches_ddpcm_without_changing_pcmsolver_default(tmp_path):
    atoms = _atoms()
    ddpcm = ImplicitSolvationCorrection(
        atoms,
        {},
        _options(),
        output=tmp_path / "ddpcm.out",
    )
    pcmsolver_options = {
        **_options(),
        "provider": "pcmsolver",
        "profile": "smd-iefpcm",
    }
    pcmsolver = ImplicitSolvationCorrection(
        atoms,
        {},
        pcmsolver_options,
        output=tmp_path / "pcmsolver.out",
    )

    assert isinstance(ddpcm.provider, DDPCMSMDImplicitSolvation)
    assert ddpcm.supported_properties == {"energy", "forces"}
    assert type(pcmsolver.provider).__name__ == "SMDImplicitSolvation"
    assert pcmsolver.supported_properties == {"energy"}


def test_setcalculator_accepts_the_same_ddpcm_contract_before_model_load(
    tmp_path,
):
    builder = SetCalculator(
        "cpu",
        "macepol-m",
        str(tmp_path / "maple.out"),
        atoms=_atoms(),
        implicit="smd",
        solvent="water",
        solvation_options=_options(),
    )

    builder._validate_solvent_config()


class _ZeroDensityResponse:
    def __init__(self, atom_count: int):
        self.shape = (atom_count, 4)

    def jvp(self, field_direction):
        return np.zeros(self.shape)

    def vjp(self, density_cotangent):
        return np.zeros(self.shape)


class _ZeroReactionField:
    reciprocal_energy_pairing = True
    full_position_derivative_contract_version = (
        FULL_REACTION_FIELD_POSITION_DERIVATIVE_CONTRACT_VERSION
    )

    def __init__(
        self,
        positions_angstrom,
        radii_angstrom,
        **_kwargs,
    ):
        self.atom_count = len(positions_angstrom)
        self.runtime_provenance = {
            "backend": "fake-pyddx",
            "pyddx_version": "0.8.0",
        }
        assert np.asarray(radii_angstrom).shape == (self.atom_count,)

    def apply(self, density_direction):
        return np.zeros_like(density_direction)

    def adjoint(self, field_cotangent):
        return np.zeros_like(field_cotangent)

    def polarization_energy_hartree(self, density_coefficients):
        return 0.0

    def full_position_vjp(self, density, field_cotangent):
        return np.zeros((self.atom_count, 3))


class _FakeMACEPolarCalculator(CalcABC):
    MODEL_ENERGY_UNIT = "eV"
    SUPPORTS_PBC = False

    def __init__(self, atoms):
        super().__init__()
        density = np.asarray(
            [[-0.1, 0.0, 0.0, 0.0], [0.1, 0.0, 0.0, 0.0]]
        )
        self.gas_forces = np.asarray(
            [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]
        )
        self.solvent_forces = np.asarray(
            [[0.8, 0.0, 0.0], [-0.8, 0.0, 0.0]]
        )
        self.gas_state = SimpleNamespace(
            energy_ev=-20.0,
            density_coefficients=density,
            fixed_field_forces_ev_per_angstrom=self.gas_forces,
        )
        self.solvent_state = SimpleNamespace(
            energy_ev=-19.8,
            density_coefficients=density,
            fixed_field_forces_ev_per_angstrom=self.solvent_forces,
        )
        self._last_polar_state = self.gas_state
        self.route2_smd_profile = ROUTE2_SMD_CALCULATOR_PROFILE
        self.long_range_evaluator_profile = (
            MACEPOL_MOLECULAR_REALSPACE_PROFILE
        )
        self.long_range_evaluator_provenance = {
            "profile": MACEPOL_MOLECULAR_REALSPACE_PROFILE,
        }
        self.mace_torch_version = "0.3.16"
        self.dtype = "torch.float64"
        self.atoms = atoms.copy()

    def polar_state(
        self,
        atoms,
        *,
        node_potential_ev=None,
        node_gradient_ev_per_angstrom=None,
        compute_forces=False,
        compute_hessian=False,
    ):
        del atoms, compute_forces, compute_hessian
        if node_potential_ev is None and node_gradient_ev_per_angstrom is None:
            return self.gas_state, {}
        return self.solvent_state, {}

    def intrinsic_energy_field_gradient(
        self,
        atoms,
        *,
        node_potential_ev,
        node_gradient_ev_per_angstrom,
    ):
        del node_potential_ev, node_gradient_ev_per_angstrom
        return np.zeros((len(atoms), 4))

    def linearize_density_response(
        self,
        atoms,
        *,
        node_potential_ev,
        node_gradient_ev_per_angstrom,
    ):
        del node_potential_ev, node_gradient_ev_per_angstrom
        return _ZeroDensityResponse(len(atoms))

    def density_position_vjp(
        self,
        atoms,
        *,
        node_potential_ev,
        node_gradient_ev_per_angstrom,
        density_cotangent,
    ):
        del node_potential_ev, node_gradient_ev_per_angstrom, density_cotangent
        return np.zeros((len(atoms), 3))


def test_reciprocal_profile_requires_matching_calculator_evaluator(tmp_path):
    atoms = _atoms()
    options = {
        **_options(),
        "profile": DDPCM_GAFF2_CARBONYL_O_MACE_KSPACE40_PROFILE,
    }
    provider = DDPCMSMDImplicitSolvation(
        atoms,
        options,
        audit_dir=tmp_path,
    )
    calculator = _FakeMACEPolarCalculator(atoms)

    with pytest.raises(TypeError, match="long-range evaluator"):
        provider._validate_calculator(
            calculator,
            need_forces=False,
        )

    calculator.long_range_evaluator_profile = (
        MACEPOL_FORCED_RECIPROCAL_FIXED_BOX40_PROFILE
    )
    calculator.long_range_evaluator_provenance = {
        "profile": MACEPOL_FORCED_RECIPROCAL_FIXED_BOX40_PROFILE,
        "box_length_angstrom": 40.0,
        "use_pbc_evaluator": True,
        "equivalent_to_default_evaluator": False,
    }
    provider._validate_calculator(
        calculator,
        need_forces=True,
    )

    assert provider.provenance["mace_long_range_evaluator"] == (
        MACEPOL_FORCED_RECIPROCAL_FIXED_BOX40_PROFILE
    )
    assert provider.provenance["default_eligible"] is False


def _fake_cds(atoms):
    gradient = np.asarray(
        [[0.001, 0.0, 0.0], [-0.001, 0.0, 0.0]]
    )
    return SimpleNamespace(
        energy_hartree=0.003,
        position_gradient_hartree_per_angstrom=gradient,
        runtime_provenance={
            "provider": "fake-pyscf-smd-cds",
            "pyscf_version": "2.13.1",
        },
    )


def test_ddpcm_provider_returns_same_profile_energy_and_correction_force(
    monkeypatch,
    tmp_path,
):
    import maple.function.calculator.extra_correction.implicit.ddpcm_smd as module

    atoms = _atoms()
    calculator = _FakeMACEPolarCalculator(atoms)
    monkeypatch.setattr(
        module,
        "PyDDXPCMReactionFieldLinearMap",
        _ZeroReactionField,
    )
    monkeypatch.setattr(
        module,
        "pyscf_smd_water_cds",
        lambda symbols, positions: _fake_cds(atoms),
    )
    provider = DDPCMSMDImplicitSolvation(
        atoms,
        _options(),
        audit_dir=tmp_path,
    )

    result = provider.evaluate(
        atoms,
        need_forces=True,
        calculator=calculator,
    )

    expected_continuum_gradient = (
        calculator.gas_forces - calculator.solvent_forces
    ) / Hartree
    expected_correction_force = -(
        expected_continuum_gradient
        + _fake_cds(atoms).position_gradient_hartree_per_angstrom
    )
    assert result.energy_hartree == pytest.approx(0.2 / Hartree + 0.003)
    np.testing.assert_allclose(
        result.forces_hartree_per_angstrom,
        expected_correction_force,
    )
    assert result.components_hartree["solute_polarization"] == pytest.approx(
        0.2 / Hartree
    )
    assert result.components_hartree["pcm_polarization"] == 0.0
    assert result.provenance["provider"] == "pyddx"
    assert result.provenance["forces_available"] is True
    assert result.provenance["solution_phase_pes"] is False
    assert (tmp_path / "route2-ddpcm-result.json").is_file()
    assert (tmp_path / "route2-ddpcm-state.npz").is_file()


def test_ddpcm_provider_reuses_root_only_for_the_same_geometry(
    monkeypatch,
    tmp_path,
):
    import maple.function.calculator.extra_correction.implicit.ddpcm_smd as module

    atoms = _atoms()
    calculator = _FakeMACEPolarCalculator(atoms)
    builds = 0

    def build_reaction_field(*args, **kwargs):
        nonlocal builds
        builds += 1
        return _ZeroReactionField(*args, **kwargs)

    monkeypatch.setattr(
        module,
        "PyDDXPCMReactionFieldLinearMap",
        build_reaction_field,
    )
    monkeypatch.setattr(
        module,
        "pyscf_smd_water_cds",
        lambda symbols, positions: _fake_cds(atoms),
    )
    provider = DDPCMSMDImplicitSolvation(
        atoms,
        _options(),
        audit_dir=tmp_path,
    )

    provider.evaluate(atoms, calculator=calculator)
    provider.evaluate(
        atoms,
        need_forces=True,
        calculator=calculator,
    )
    moved = atoms.copy()
    moved.positions[0, 1] += 1.0e-4
    provider.evaluate(moved, calculator=calculator)

    assert builds == 2


def test_ddpcm_provider_rejects_force_state_energy_drift(
    monkeypatch,
    tmp_path,
):
    import maple.function.calculator.extra_correction.implicit.ddpcm_smd as module

    class _EnergyDriftCalculator(_FakeMACEPolarCalculator):
        def polar_state(self, atoms, **kwargs):
            state, auxiliary = super().polar_state(atoms, **kwargs)
            if kwargs.get("compute_forces") and kwargs.get(
                "node_potential_ev"
            ) is not None:
                state = SimpleNamespace(
                    energy_ev=state.energy_ev + 1.0e-4,
                    density_coefficients=state.density_coefficients,
                    fixed_field_forces_ev_per_angstrom=(
                        state.fixed_field_forces_ev_per_angstrom
                    ),
                )
            return state, auxiliary

    atoms = _atoms()
    calculator = _EnergyDriftCalculator(atoms)
    monkeypatch.setattr(
        module,
        "PyDDXPCMReactionFieldLinearMap",
        _ZeroReactionField,
    )
    monkeypatch.setattr(
        module,
        "pyscf_smd_water_cds",
        lambda symbols, positions: _fake_cds(atoms),
    )
    provider = DDPCMSMDImplicitSolvation(
        atoms,
        _options(),
        audit_dir=tmp_path,
    )

    with pytest.raises(RuntimeError, match="does not reproduce"):
        provider.evaluate(
            atoms,
            need_forces=True,
            calculator=calculator,
        )


def test_ddpcm_provider_fails_closed_without_its_selected_runtime(
    monkeypatch,
    tmp_path,
):
    import maple.function.calculator.extra_correction.implicit.ddpcm_smd as module

    atoms = _atoms()
    calculator = _FakeMACEPolarCalculator(atoms)

    def unavailable(*_args, **_kwargs):
        raise RuntimeError("synthetic pyddx runtime unavailable")

    monkeypatch.setattr(
        module,
        "PyDDXPCMReactionFieldLinearMap",
        unavailable,
    )
    provider = DDPCMSMDImplicitSolvation(
        atoms,
        _options(),
        audit_dir=tmp_path,
    )

    with pytest.raises(RuntimeError, match="pyddx runtime unavailable"):
        provider.evaluate(atoms, calculator=calculator)

    assert not (tmp_path / "route2-ddpcm-result.json").exists()
    assert not (tmp_path / "route2-ddpcm-state.npz").exists()


def test_calcabc_adds_gas_force_and_ddpcm_correction_exactly_once(
    monkeypatch,
    tmp_path,
):
    import maple.function.calculator.extra_correction.implicit.ddpcm_smd as module

    atoms = _atoms()
    calculator = _FakeMACEPolarCalculator(atoms)
    monkeypatch.setattr(
        module,
        "PyDDXPCMReactionFieldLinearMap",
        _ZeroReactionField,
    )
    monkeypatch.setattr(
        module,
        "pyscf_smd_water_cds",
        lambda symbols, positions: _fake_cds(atoms),
    )
    calculator.solvent_correction = ImplicitSolvationCorrection(
        atoms,
        {},
        _options(),
        output=tmp_path / "maple.out",
    )

    calculator._finalize_results(
        atoms,
        energy=calculator.gas_state.energy_ev,
        forces=calculator.gas_forces,
        unit="eV",
    )

    expected_total_force = (
        calculator.solvent_forces / Hartree
        - _fake_cds(atoms).position_gradient_hartree_per_angstrom
    )
    np.testing.assert_allclose(
        calculator.results["forces"],
        expected_total_force,
    )
    assert calculator.results["energy"] == pytest.approx(
        calculator.solvent_state.energy_ev / Hartree + 0.003
    )
    assert calculator.results["solvation"]["provenance"]["provider"] == "pyddx"
