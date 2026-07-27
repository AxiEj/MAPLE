from __future__ import annotations

import json
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
    PyDDXSMDImplicitSolvation,
)
from maple.function.calculator.extra_correction.implicit.route2_derivative import (
    FULL_REACTION_FIELD_POSITION_DERIVATIVE_CONTRACT_VERSION,
)
from maple.function.calculator.extra_correction.implicit.route2_engine import (
    Route2SCFConvergenceError,
)
from maple.function.calculator.set_calculator import SetCalculator
from maple.function.read.command_control import CommandControl
from maple.function.route2_smd_profiles import (
    DDPCM_GAFF2_CARBONYL_O_MACE_KSPACE40_PROFILE,
    DDPCM_GAFF2_CARBONYL_O_MACE_KSPACE40_OMP4_PROFILE,
    DDPCM_MULTISOLVENT_SMD_PROFILE,
    DDCOSMO_MULTISOLVENT_SMD_PROFILE,
    MACEPOL_FORCED_RECIPROCAL_FIXED_BOX40_PROFILE,
    MACEPOL_MOLECULAR_REALSPACE_PROFILE,
)
from maple.function.route2_solvents import SUPPORTED_ROUTE2_SMD_SOLVENTS


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


def _multisolvent_options(solvent: str) -> dict[str, object]:
    return {
        **_options(),
        "implicit": solvent,
        "profile": DDPCM_MULTISOLVENT_SMD_PROFILE,
    }


def _multisolvent_cosmo_options(solvent: str) -> dict[str, object]:
    return {
        **_options(),
        "implicit": solvent,
        "profile": DDCOSMO_MULTISOLVENT_SMD_PROFILE,
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


@pytest.mark.parametrize("solvent", sorted(SUPPORTED_ROUTE2_SMD_SOLVENTS))
def test_public_parser_accepts_only_multisolvent_profile_for_each_solvent(
    solvent,
):
    params = _parse(
        "#model=macepol-m",
        "#sp(verbose=1)",
        (
            f"#solv(implicit={solvent},method=smd,provider=pyddx,"
            f"profile={DDPCM_MULTISOLVENT_SMD_PROFILE},response=scf,"
            "standard_state=1m,experimental=true)"
        ),
    )

    assert params["solv"] == _multisolvent_options(solvent)


@pytest.mark.parametrize("solvent", sorted(SUPPORTED_ROUTE2_SMD_SOLVENTS))
def test_public_parser_accepts_ddcosmo_as_a_separate_equation_profile(
    solvent,
):
    params = _parse(
        "#model=macepol-m",
        "#sp(verbose=1)",
        (
            f"#solv(implicit={solvent},method=smd,provider=pyddx,"
            f"profile={DDCOSMO_MULTISOLVENT_SMD_PROFILE},response=scf,"
            "standard_state=1m,experimental=true)"
        ),
    )

    assert params["solv"] == _multisolvent_cosmo_options(solvent)


def test_public_parser_canonicalizes_multisolvent_alias():
    params = _parse(
        "#model=macepol-m",
        "#sp(verbose=1)",
        (
            "#solv(implicit=MeCN,method=smd,provider=pyddx,"
            f"profile={DDPCM_MULTISOLVENT_SMD_PROFILE},response=scf,"
            "standard_state=1m,experimental=true)"
        ),
    )

    assert params["solv"] == _multisolvent_options("acetonitrile")


def test_legacy_ddpcm_profile_remains_water_only():
    with pytest.raises(ValueError, match="does not support solvent=acetonitrile"):
        _parse(
            "#model=macepol-m",
            "#sp(verbose=1)",
            (
                "#solv(implicit=acetonitrile,method=smd,provider=pyddx,"
                f"profile={DDPCM_SMD_PROFILE},response=scf,"
                "standard_state=1m,experimental=true)"
            ),
        )


def test_set_calculator_rejects_mismatched_direct_solvent_configuration():
    builder = SetCalculator(
        "cpu",
        "macepol-m",
        "maple.out",
        atoms=_atoms(),
        implicit="smd",
        solvent="water",
        solvation_options=_multisolvent_options("acetonitrile"),
    )

    with pytest.raises(
        ValueError,
        match="Implicit-solvent selector mismatch",
    ):
        builder._validate_solvent_config()


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


def test_public_parser_accepts_versioned_reciprocal_omp4_profile():
    options = {
        **_options(),
        "profile": DDPCM_GAFF2_CARBONYL_O_MACE_KSPACE40_OMP4_PROFILE,
    }
    params = _parse(
        "#model=macepol-m",
        "#sp(verbose=1)",
        (
            "#solv(implicit=water,method=smd,provider=pyddx,"
            f"profile={DDPCM_GAFF2_CARBONYL_O_MACE_KSPACE40_OMP4_PROFILE},"
            "response=scf,standard_state=1m,experimental=true)"
        ),
    )

    assert params["solv"] == options


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


def test_ddpcm_provider_rejects_mol2_atom_type_changes_after_initialization(
    tmp_path,
):
    provider = DDPCMSMDImplicitSolvation(
        _atoms(),
        _options(),
        audit_dir=tmp_path,
    )
    changed = _atoms()
    changed.info["mol2"] = {"atom_types": ["c.3", "o"]}

    with pytest.raises(ValueError, match="MOL2 atom type/order changes"):
        provider.evaluate(changed, calculator=None)


def test_ddpcm_provider_persists_fail_closed_scf_history(tmp_path):
    atoms = _atoms()
    provider = DDPCMSMDImplicitSolvation(
        atoms,
        _options(),
        audit_dir=tmp_path,
    )
    history = [
        {
            "iteration": 1,
            "density_residual_e": 1.0e-6,
            "energy_residual_ev": None,
            "next_density_update": "damped-picard-v1",
        }
    ]
    error = Route2SCFConvergenceError(
        "synthetic nonconvergence",
        history=history,
    )

    provider._write_scf_failure_audit(
        atoms=atoms,
        gas_state=SimpleNamespace(energy_ev=-10.0),
        error=error,
    )

    payload = json.loads(
        (tmp_path / "route2-ddpcm-failure.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["converged"] is False
    assert payload["error"] == "synthetic nonconvergence"
    assert payload["scf"]["history"] == history
    assert payload["scf"]["residual_definition"] == (
        "unmixed neutral-tangent Pi0[M(P(c))-c]"
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


@pytest.mark.parametrize(
    "options",
    [
        _multisolvent_options("acetonitrile"),
        _multisolvent_cosmo_options("acetonitrile"),
    ],
)
def test_canonical_pyddx_profile_accepts_xyz_geometry_without_fake_mol2(
    options,
):
    atoms = _atoms()
    atoms.info.pop("mol2")
    builder = SetCalculator(
        "cpu",
        "macepol-m",
        "maple.out",
        atoms=atoms,
        implicit="smd",
        solvent="acetonitrile",
        solvation_options=options,
    )

    builder._validate_solvent_config()


def test_gaff2_profile_still_requires_mol2_atom_types():
    atoms = _atoms()
    atoms.info.pop("mol2")
    builder = SetCalculator(
        "cpu",
        "macepol-m",
        "maple.out",
        atoms=atoms,
        implicit="smd",
        solvent="water",
        solvation_options={
            **_options(),
            "profile": DDPCM_GAFF2_CARBONYL_O_PROFILE,
        },
    )

    with pytest.raises(ValueError, match="requires one MOL2 molecule"):
        builder._validate_solvent_config()


def test_topology_free_pyddx_geometry_still_requires_one_connected_molecule(
    tmp_path,
):
    atoms = _atoms()
    atoms.info.pop("mol2")
    atoms.positions[1, 0] += 10.0

    with pytest.raises(ValueError, match="one connected molecule"):
        PyDDXSMDImplicitSolvation(
            atoms,
            _multisolvent_options("acetonitrile"),
            audit_dir=tmp_path,
        )


class _ZeroDensityResponse:
    def __init__(self, atom_count: int):
        self.shape = (atom_count, 4)

    def jvp(self, field_direction):
        return np.zeros(self.shape)

    def vjp(self, density_cotangent):
        return np.zeros(self.shape)


class _ZeroReactionField:
    instances = []
    reciprocal_energy_pairing = True
    full_position_derivative_contract_version = (
        FULL_REACTION_FIELD_POSITION_DERIVATIVE_CONTRACT_VERSION
    )

    def __init__(
        self,
        positions_angstrom,
        radii_angstrom,
        **kwargs,
    ):
        self.atom_count = len(positions_angstrom)
        self.n_proc = kwargs["n_proc"]
        self.dielectric = kwargs["dielectric"]
        self.runtime_provenance = {
            "backend": "fake-pyddx",
            "pyddx_version": "0.8.0",
            "n_proc": self.n_proc,
        }
        self.cold_apply_calls = 0
        self.scf_apply_calls = 0
        self.cold_energy_calls = 0
        self.scf_energy_calls = 0
        self.instances.append(self)
        assert np.asarray(radii_angstrom).shape == (self.atom_count,)

    def apply(self, density_direction):
        self.cold_apply_calls += 1
        return np.zeros_like(density_direction)

    def apply_scf(self, density_direction):
        self.scf_apply_calls += 1
        return np.zeros_like(density_direction)

    def adjoint(self, field_cotangent):
        return np.zeros_like(field_cotangent)

    def polarization_energy_hartree(self, density_coefficients):
        self.cold_energy_calls += 1
        return 0.0

    def scf_polarization_energy_hartree(self, density_coefficients):
        self.scf_energy_calls += 1
        return 0.0

    def full_position_vjp(self, density, field_cotangent):
        return np.zeros((self.atom_count, 3))


class _ScalarDensityResponse:
    def __init__(self, field_response: float):
        self.field_response = float(field_response)
        self.shape = (2, 4)
        self.vjp_calls = 0

    def jvp(self, field_direction):
        direction = np.asarray(field_direction, dtype=float)
        response = np.zeros(self.shape)
        response[0, 0] = self.field_response * direction[0, 0]
        response[1, 0] = -response[0, 0]
        return response

    def vjp(self, density_cotangent):
        self.vjp_calls += 1
        cotangent = np.asarray(density_cotangent, dtype=float)
        response = np.zeros(self.shape)
        response[0, 0] = self.field_response * (
            cotangent[0, 0] - cotangent[1, 0]
        )
        return response


class _ScalarCoordinateReactionField:
    instances = []
    reciprocal_energy_pairing = True
    full_position_derivative_contract_version = (
        FULL_REACTION_FIELD_POSITION_DERIVATIVE_CONTRACT_VERSION
    )
    field_scale_intercept = 0.12
    field_scale_slope = 0.01

    def __init__(
        self,
        positions_angstrom,
        radii_angstrom,
        **kwargs,
    ):
        del kwargs
        self.atom_count = len(positions_angstrom)
        self.coordinate = float(positions_angstrom[0][0])
        self.field_scale = (
            self.field_scale_intercept
            + self.field_scale_slope * self.coordinate
        )
        self.runtime_provenance = {
            "backend": "synthetic-coordinate-pyddx",
            "pyddx_version": "0.8.0",
            "n_proc": 1,
        }
        self.full_position_vjp_value = None
        self.instances.append(self)
        assert self.atom_count == 2
        assert np.asarray(radii_angstrom).shape == (self.atom_count,)

    def apply(self, density_direction):
        density = np.asarray(density_direction, dtype=float)
        field = np.zeros_like(density)
        field[:, 0] = self.field_scale * density[:, 0]
        return field

    def apply_scf(self, density_direction):
        return self.apply(density_direction)

    def adjoint(self, field_cotangent):
        cotangent = np.asarray(field_cotangent, dtype=float)
        density_cotangent = np.zeros_like(cotangent)
        density_cotangent[:, 0] = (
            self.field_scale * cotangent[:, 0]
        )
        return density_cotangent

    def polarization_energy_hartree(self, density_coefficients):
        density = np.asarray(density_coefficients, dtype=float)
        return float(
            0.5 * np.vdot(density, self.apply(density)) / Hartree
        )

    def scf_polarization_energy_hartree(self, density_coefficients):
        return self.polarization_energy_hartree(density_coefficients)

    def full_position_vjp(self, density, field_cotangent):
        density = np.asarray(density, dtype=float)
        cotangent = np.asarray(field_cotangent, dtype=float)
        result = np.zeros((self.atom_count, 3))
        result[0, 0] = self.field_scale_slope * np.vdot(
            cotangent[:, 0],
            density[:, 0],
        )
        self.full_position_vjp_value = float(result[0, 0])
        return result


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


class _CoordinateMACEPolarCalculator(_FakeMACEPolarCalculator):
    density_intercept = 0.08
    density_coordinate_slope = 0.015
    density_field_response_intercept = 0.18
    density_field_response_coordinate_slope = 0.02
    field_linear_intercept = 0.40
    field_linear_coordinate_slope = -0.07
    field_quadratic = 0.30
    intrinsic_coordinate_linear = 0.11
    intrinsic_coordinate_quadratic = 0.04
    gas_coordinate_linear = 0.05
    gas_coordinate_quadratic = 0.03

    def __init__(self, atoms):
        super().__init__(atoms)
        self.coordinate = float(atoms.positions[0, 0])
        density = self._density(node_potential_ev=None)
        self.gas_state = SimpleNamespace(
            energy_ev=self._gas_energy(),
            density_coefficients=density,
            fixed_field_forces_ev_per_angstrom=self._gas_forces(),
        )
        self._last_polar_state = self.gas_state
        self.last_density_position_vjp = None
        self.last_density_response = None
        self.force_state_calls = 0

    def _density(self, *, node_potential_ev):
        base = (
            self.density_intercept
            + self.density_coordinate_slope * self.coordinate
        )
        field_response = (
            self.density_field_response_intercept
            + self.density_field_response_coordinate_slope
            * self.coordinate
        )
        potential = (
            0.0
            if node_potential_ev is None
            else float(np.asarray(node_potential_ev)[0])
        )
        charge = base + field_response * potential
        density = np.zeros((2, 4))
        density[:, 0] = [charge, -charge]
        return density

    def _intrinsic_energy(self, node_potential_ev):
        potential = float(np.asarray(node_potential_ev)[0])
        field_linear = (
            self.field_linear_intercept
            + self.field_linear_coordinate_slope * self.coordinate
        )
        return (
            field_linear * potential
            + 0.5 * self.field_quadratic * potential**2
            + self.intrinsic_coordinate_linear * self.coordinate
            + 0.5
            * self.intrinsic_coordinate_quadratic
            * self.coordinate**2
        )

    def _gas_energy(self):
        return (
            self.gas_coordinate_linear * self.coordinate
            + 0.5 * self.gas_coordinate_quadratic * self.coordinate**2
        )

    def _gas_forces(self):
        forces = np.zeros((2, 3))
        forces[0, 0] = -(
            self.gas_coordinate_linear
            + self.gas_coordinate_quadratic * self.coordinate
        )
        return forces

    def polar_state(
        self,
        atoms,
        *,
        node_potential_ev=None,
        node_gradient_ev_per_angstrom=None,
        compute_forces=False,
        compute_hessian=False,
    ):
        del atoms, node_gradient_ev_per_angstrom, compute_hessian
        if node_potential_ev is None:
            return self.gas_state, {}
        if compute_forces:
            self.force_state_calls += 1
        potential = float(np.asarray(node_potential_ev)[0])
        forces = np.zeros((2, 3))
        forces[0, 0] = -(
            self.field_linear_coordinate_slope * potential
            + self.intrinsic_coordinate_linear
            + self.intrinsic_coordinate_quadratic * self.coordinate
        )
        state = SimpleNamespace(
            energy_ev=self._intrinsic_energy(node_potential_ev),
            density_coefficients=self._density(
                node_potential_ev=node_potential_ev
            ),
            fixed_field_forces_ev_per_angstrom=(
                forces if compute_forces else None
            ),
        )
        return state, {}

    def intrinsic_energy_field_gradient(
        self,
        atoms,
        *,
        node_potential_ev,
        node_gradient_ev_per_angstrom,
    ):
        del atoms, node_gradient_ev_per_angstrom
        potential = float(np.asarray(node_potential_ev)[0])
        gradient = np.zeros((2, 4))
        gradient[0, 0] = (
            self.field_linear_intercept
            + self.field_linear_coordinate_slope * self.coordinate
            + self.field_quadratic * potential
        )
        return gradient

    def linearize_density_response(
        self,
        atoms,
        *,
        node_potential_ev,
        node_gradient_ev_per_angstrom,
    ):
        del atoms, node_potential_ev, node_gradient_ev_per_angstrom
        self.last_density_response = _ScalarDensityResponse(
            self.density_field_response_intercept
            + self.density_field_response_coordinate_slope
            * self.coordinate
        )
        return self.last_density_response

    def density_position_vjp(
        self,
        atoms,
        *,
        node_potential_ev,
        node_gradient_ev_per_angstrom,
        density_cotangent,
    ):
        del atoms, node_gradient_ev_per_angstrom
        potential = float(np.asarray(node_potential_ev)[0])
        density_coordinate_derivative = (
            self.density_coordinate_slope
            + self.density_field_response_coordinate_slope * potential
        )
        cotangent = np.asarray(density_cotangent, dtype=float)
        result = np.zeros((2, 3))
        result[0, 0] = density_coordinate_derivative * (
            cotangent[0, 0] - cotangent[1, 0]
        )
        self.last_density_position_vjp = float(result[0, 0])
        return result


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


def test_reciprocal_omp4_profile_passes_thread_count_to_pyddx(
    monkeypatch,
    tmp_path,
):
    import maple.function.calculator.extra_correction.implicit.ddpcm_smd as module

    atoms = _atoms()
    calculator = _FakeMACEPolarCalculator(atoms)
    calculator.long_range_evaluator_profile = (
        MACEPOL_FORCED_RECIPROCAL_FIXED_BOX40_PROFILE
    )
    _ZeroReactionField.instances.clear()
    monkeypatch.setattr(
        module,
        "PyDDXPCMReactionFieldLinearMap",
        _ZeroReactionField,
    )
    monkeypatch.setattr(
        module,
        "pyscf_smd_cds",
        lambda symbols, positions, *, solvent: _fake_cds(atoms),
    )
    provider = DDPCMSMDImplicitSolvation(
        atoms,
        {
            **_options(),
            "profile": (
                DDPCM_GAFF2_CARBONYL_O_MACE_KSPACE40_OMP4_PROFILE
            ),
        },
        audit_dir=tmp_path,
    )

    result = provider.evaluate(atoms, calculator=calculator)

    assert provider.provenance["numerics"]["ddpcm_n_proc"] == 4
    assert _ZeroReactionField.instances[0].n_proc == 4
    assert result.provenance["numerics"]["ddpcm_n_proc"] == 4


def test_multisolvent_provider_routes_dielectric_radii_and_cds_together(
    monkeypatch,
    tmp_path,
):
    import maple.function.calculator.extra_correction.implicit.ddpcm_smd as module

    atoms = _atoms()
    calculator = _FakeMACEPolarCalculator(atoms)
    seen = {}
    _ZeroReactionField.instances.clear()
    monkeypatch.setattr(
        module,
        "PyDDXPCMReactionFieldLinearMap",
        _ZeroReactionField,
    )

    def fake_cds(symbols, positions, *, solvent):
        seen["solvent"] = solvent
        return _fake_cds(Atoms(symbols, positions=positions))

    monkeypatch.setattr(module, "pyscf_smd_cds", fake_cds)
    provider = DDPCMSMDImplicitSolvation(
        atoms,
        _multisolvent_options("acetonitrile"),
        audit_dir=tmp_path,
    )

    result = provider.evaluate(atoms, calculator=calculator)

    assert seen["solvent"] == "acetonitrile"
    assert _ZeroReactionField.instances[0].dielectric == pytest.approx(35.688)
    assert provider.coulomb_radii_angstrom == pytest.approx([1.85, 2.168])
    assert provider.provenance["solvent"] == "acetonitrile"
    assert provider.provenance["strict_original_smd_equivalence"] is False
    assert result.provenance["numerics"]["dielectric"] == pytest.approx(35.688)
    assert (
        result.provenance["numerics"]["coulomb_radii_policy"]
        == "pyscf-smd-2.13.1"
    )


def test_pyddx_provider_dispatches_ddcosmo_without_reusing_ddpcm_label(
    monkeypatch,
    tmp_path,
):
    import maple.function.calculator.extra_correction.implicit.ddpcm_smd as module

    atoms = _atoms()
    calculator = _FakeMACEPolarCalculator(atoms)
    _ZeroReactionField.instances.clear()
    monkeypatch.setattr(
        module,
        "PyDDXCOSMOReactionFieldLinearMap",
        _ZeroReactionField,
    )
    monkeypatch.setattr(
        module,
        "pyscf_smd_cds",
        lambda symbols, positions, *, solvent: _fake_cds(atoms),
    )
    provider = PyDDXSMDImplicitSolvation(
        atoms,
        _multisolvent_cosmo_options("acetonitrile"),
        audit_dir=tmp_path,
    )

    result = provider.evaluate(atoms, calculator=calculator)

    assert isinstance(provider, DDPCMSMDImplicitSolvation)
    assert provider.continuum_label == "ddCOSMO"
    assert provider.provenance["electrostatics"] == "ddCOSMO"
    assert provider.provenance["electrostatics_model"] == "ddcosmo"
    assert provider.provenance["scientific_identity"] == (
        "MACE-POLAR/(l<=1)-point-multipole + ddCOSMO + SMD-CDS"
    )
    assert "E_ddCOSMO" in provider.provenance["energy_composition"]
    assert "ddpcm_n_proc" not in provider.provenance["numerics"]
    assert provider.provenance["numerics"]["pyddx_n_proc"] == 1
    assert result.provenance["electrostatics"] == "ddCOSMO"
    assert (tmp_path / "route2-ddcosmo-result.json").is_file()
    assert (tmp_path / "route2-ddcosmo-state.npz").is_file()
    assert not (tmp_path / "route2-ddpcm-result.json").exists()


def test_multisolvent_water_dielectric_is_versioned_from_legacy_water(
    tmp_path,
):
    legacy = DDPCMSMDImplicitSolvation(
        _atoms(),
        _options(),
        audit_dir=tmp_path / "legacy",
    )
    multisolv = DDPCMSMDImplicitSolvation(
        _atoms(),
        _multisolvent_options("water"),
        audit_dir=tmp_path / "multisolv",
    )

    assert legacy.continuum_dielectric == pytest.approx(78.39)
    assert multisolv.continuum_dielectric == pytest.approx(78.355)
    assert (
        legacy.profile_spec.coulomb_radii_policy
        == "smd-water-reference-smd18-v1"
    )
    assert legacy.provenance["cavity_radii"] == (
        "SMD/SMD18 atomic-number-indexed water Coulomb radii "
        "(P=2.12 A, S=2.49 A, Cl=2.38 A)"
    )
    assert (
        multisolv.profile_spec.coulomb_radii_policy
        == "pyscf-smd-2.13.1"
    )


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


def _coordinate_cds(atoms):
    coordinate = float(atoms.positions[0, 0])
    gradient = np.zeros((2, 3))
    gradient[0, 0] = 0.004 + 0.003 * coordinate
    return SimpleNamespace(
        energy_hartree=(
            0.002 + 0.004 * coordinate + 0.5 * 0.003 * coordinate**2
        ),
        position_gradient_hartree_per_angstrom=gradient,
        runtime_provenance={
            "provider": "synthetic-coordinate-cds",
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
    _ZeroReactionField.instances.clear()
    monkeypatch.setattr(
        module,
        "PyDDXPCMReactionFieldLinearMap",
        _ZeroReactionField,
    )
    monkeypatch.setattr(
        module,
        "pyscf_smd_cds",
        lambda symbols, positions, *, solvent: _fake_cds(atoms),
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
    reaction_field = _ZeroReactionField.instances[0]
    assert reaction_field.n_proc == 1
    assert reaction_field.scf_apply_calls == 1
    assert reaction_field.scf_energy_calls == 1
    assert reaction_field.cold_energy_calls == 0
    assert reaction_field.cold_apply_calls == 0
    assert (tmp_path / "route2-ddpcm-result.json").is_file()
    assert (tmp_path / "route2-ddpcm-state.npz").is_file()


def test_ddpcm_provider_nonzero_response_force_matches_complete_correction_energy_fd(
    monkeypatch,
    tmp_path,
):
    import maple.function.calculator.extra_correction.implicit.ddpcm_smd as module

    _ScalarCoordinateReactionField.instances.clear()
    monkeypatch.setattr(
        module,
        "PyDDXPCMReactionFieldLinearMap",
        _ScalarCoordinateReactionField,
    )
    monkeypatch.setattr(
        module,
        "pyscf_smd_cds",
        lambda symbols, positions, *, solvent: _coordinate_cds(
            Atoms(symbols, positions=positions)
        ),
    )

    def evaluate(coordinate, *, need_forces, label):
        atoms = _atoms()
        atoms.positions[0, 0] = coordinate
        calculator = _CoordinateMACEPolarCalculator(atoms)
        provider = DDPCMSMDImplicitSolvation(
            atoms,
            _options(),
            audit_dir=tmp_path / label,
        )
        return (
            provider.evaluate(
                atoms,
                need_forces=need_forces,
                calculator=calculator,
            ),
            calculator,
        )

    coordinate = 0.23
    center, center_calculator = evaluate(
        coordinate,
        need_forces=True,
        label="center",
    )
    analytic_force = float(
        center.forces_hartree_per_angstrom[0, 0]
    )
    finite_differences = []
    for index, step in enumerate((1.0e-3, 3.0e-4, 1.0e-4)):
        plus, _ = evaluate(
            coordinate + step,
            need_forces=False,
            label=f"plus-{index}",
        )
        minus, _ = evaluate(
            coordinate - step,
            need_forces=False,
            label=f"minus-{index}",
        )
        finite_difference_force = -(
            plus.energy_hartree - minus.energy_hartree
        ) / (2.0 * step)
        finite_differences.append(finite_difference_force)
        assert analytic_force == pytest.approx(
            finite_difference_force,
            rel=2.0e-8,
            abs=2.0e-10,
        )

    assert abs(analytic_force) > 1.0e-4
    assert abs(finite_differences[-1] - finite_differences[0]) < 1.0e-8
    assert (
        abs(center_calculator.last_density_position_vjp) > 1.0e-8
    )
    assert center_calculator.force_state_calls == 1
    assert center_calculator.last_density_response.vjp_calls > 0
    center_reaction_field = _ScalarCoordinateReactionField.instances[0]
    assert abs(center_reaction_field.full_position_vjp_value) > 1.0e-8
    assert center.provenance["forces_available"] is True


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
        "pyscf_smd_cds",
        lambda symbols, positions, *, solvent: _fake_cds(atoms),
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
        "pyscf_smd_cds",
        lambda symbols, positions, *, solvent: _fake_cds(atoms),
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
        "pyscf_smd_cds",
        lambda symbols, positions, *, solvent: _fake_cds(atoms),
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
