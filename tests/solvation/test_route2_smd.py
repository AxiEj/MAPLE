from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms
from ase.units import Bohr, Hartree

from maple.function.calculator.calculator_base import (
    ROUTE2_SMD_CALCULATOR_PROFILE,
)
import maple.function.calculator.extra_correction.implicit.smd as smd_module
from maple.function.calculator.extra_correction.implicit import (
    route2_pcm_response as pcm_response_module,
)
from maple.function.calculator.extra_correction.implicit.continuum_response import (
    EXTERNAL_MEP_RESPONSE_CONTRACT_VERSION,
    PCMSolverExternalMEPCavityResponse,
    SurfaceChargeState,
)
from maple.function.calculator.extra_correction.implicit.gto_density import (
    asc_reaction_potential_gradient,
    density_reaction_coupling,
    gaussian_multipole_potential,
    point_asc_reaction_potential_gradient,
    point_multipole_potential,
)
from maple.function.calculator.extra_correction.implicit.gto_field_projection import (
    MACEPolarGTOFieldProjectionSpec,
)
from maple.function.calculator.extra_correction.implicit.route2_pcm_response import (
    FixedCavityPCMReactionFieldLinearMap,
)
from maple.function.calculator.extra_correction.implicit.smd import (
    PCM_WARNING_MARKER,
    SMDImplicitSolvation,
    _pcm_input_text,
)
from maple.function.calculator.extra_correction.implicit.smd_cds import (
    GAFF2_CARBONYL_O_PROFILE,
    SMD_WATER_COULOMB_RADII_ANGSTROM,
    route2_water_coulomb_radii,
    smd_sasa_radii,
    smd_water_cds,
)
from maple.function.read.command_control import CommandControl
from maple.function.route2_smd_profiles import (
    PCMSOLVER_CENTERED_LOCAL_JET_FIELD_PROFILE,
    PCMSOLVER_EXACT_GTO_FIELD_PROFILE,
)


def parse(*lines):
    return CommandControl.from_settings(list(lines)).as_dict()


def _route2_options(response="scf"):
    return {
        "method": "smd",
        "implicit": "water",
        "provider": "pcmsolver",
        "profile": "smd-iefpcm",
        "response": response,
        "standard_state": "1m",
        "experimental": True,
    }


def _co_atoms():
    atoms = Atoms("CO", positions=[[0.0, 0.0, 0.0], [1.13, 0.0, 0.0]])
    atoms.info.update(charge=0, mult=1)
    return atoms


def test_route2_public_contract_accepts_only_macepolar_m_and_defaults():
    params = parse(
        "#model=macepol-m",
        "#sp",
        "#solv(implicit=water,method=smd,response=scf,standard_state=1m,experimental=true)",
    )

    assert params["solv"] == {
        "implicit": "water",
        "method": "smd",
        "response": "scf",
        "standard_state": "1m",
        "experimental": True,
        "provider": "pcmsolver",
        "profile": "smd-iefpcm",
    }


def test_route2_public_contract_accepts_gaff2_carbonyl_oxygen_profile():
    params = parse(
        "#model=macepol-m",
        "#sp",
        (
            "#solv(implicit=water,method=smd,"
            "profile=smd-iefpcm-gaff2-o,experimental=true)"
        ),
    )

    assert params["solv"]["profile"] == GAFF2_CARBONYL_O_PROFILE


def test_route2_public_contract_accepts_explicit_exact_gto_profile():
    params = parse(
        "#model=macepol-m",
        "#sp",
        (
            "#solv(implicit=water,method=smd,"
            f"profile={PCMSOLVER_EXACT_GTO_FIELD_PROFILE},"
            "experimental=true)"
        ),
    )

    assert (
        params["solv"]["profile"]
        == PCMSOLVER_EXACT_GTO_FIELD_PROFILE
    )


def test_route2_public_contract_accepts_centered_local_jet_profile():
    params = parse(
        "#model=macepol-m",
        "#sp",
        (
            "#solv(implicit=water,method=smd,"
            f"profile={PCMSOLVER_CENTERED_LOCAL_JET_FIELD_PROFILE},"
            "experimental=true)"
        ),
    )

    assert (
        params["solv"]["profile"]
        == PCMSOLVER_CENTERED_LOCAL_JET_FIELD_PROFILE
    )


def test_exact_gto_profile_rejects_frozen_response(tmp_path):
    atoms = _co_atoms()
    options = _route2_options("frozen")
    options["profile"] = PCMSOLVER_EXACT_GTO_FIELD_PROFILE

    with pytest.raises(
        ValueError,
        match="requires response=scf",
    ):
        SMDImplicitSolvation(
            atoms,
            options,
            audit_dir=tmp_path,
        )


def test_centered_local_jet_profile_rejects_frozen_response(tmp_path):
    atoms = _co_atoms()
    options = _route2_options("frozen")
    options["profile"] = PCMSOLVER_CENTERED_LOCAL_JET_FIELD_PROFILE

    with pytest.raises(
        ValueError,
        match="requires response=scf",
    ):
        SMDImplicitSolvation(
            atoms,
            options,
            audit_dir=tmp_path,
        )


def test_route2_public_contract_accepts_fixed_stability_branch_cavity_policy():
    params = parse(
        "#model=macepol-m",
        "#sp",
        (
            "#solv(implicit=water,method=smd,"
            "cavity_policy=fixed-stability-branch,experimental=true)"
        ),
    )

    assert params["solv"]["cavity_policy"] == "fixed-stability-branch"


@pytest.mark.parametrize(
    ("extra_line", "solv", "message"),
    [
        (
            "#charge(source=mol2)",
            "#solv(implicit=water,method=smd,experimental=true)",
            "remove #charge",
        ),
        (
            None,
            "#solv(implicit=water,method=smd,provider=ddx,experimental=true)",
            "provider=pcmsolver",
        ),
        (
            None,
            "#solv(implicit=water,method=smd,standard_state=1atm,experimental=true)",
            "standard_state must be 1m",
        ),
        (
            None,
            "#solv(implicit=water,method=smd,backend=mock,experimental=true)",
            "Unknown solvation parameter",
        ),
    ],
)
def test_route2_rejects_route1_or_development_options(extra_line, solv, message):
    lines = ["#model=macepol-m", "#sp"]
    if extra_line:
        lines.append(extra_line)
    lines.append(solv)
    with pytest.raises(ValueError, match=message):
        parse(*lines)


def test_route2_rejects_unknown_cavity_policy():
    with pytest.raises(ValueError, match="cavity_policy must be"):
        parse(
            "#model=macepol-m",
            "#sp",
            (
                "#solv(implicit=water,method=smd,"
                "cavity_policy=geometry-dependent,experimental=true)"
            ),
        )


@pytest.mark.parametrize("model", ["ani2x", "macepol-s", "macepol-l"])
def test_route2_rejects_any_model_except_official_macepolar_m(model):
    with pytest.raises(ValueError, match="MACE-POLAR-1-M"):
        parse(
            f"#model={model}",
            "#sp",
            "#solv(implicit=water,method=smd,experimental=true)",
        )


def test_route2_rejects_custom_checkpoint_or_plugin_options():
    with pytest.raises(ValueError, match="unmodified official MACE-POLAR-1-M"):
        parse(
            "#model=macepol-m(model_path=/tmp/custom.model)",
            "#sp",
            "#solv(implicit=water,method=smd,experimental=true)",
        )


def test_route2_rejects_gradient_request():
    with pytest.raises(ValueError, match="does not provide forces"):
        parse(
            "#model=macepol-m",
            "#sp(verbose=1)",
            "#solv(implicit=water,method=smd,experimental=true)",
        )


def test_generated_pcmsolver_input_uses_host_geometry_and_all_smd_radii():
    radii = np.asarray([1.85, 1.52, 1.20])
    text = _pcm_input_text(3, radii)

    assert "MODE = ATOMS" in text
    assert "ATOMS = [1, 2, 3]" in text
    assert "RADII = [1.8500000000, 1.5200000000, 1.2000000000]" in text
    assert "SCALING = FALSE" in text
    assert "AREA = 0.2000000000" in text
    assert "MINRADIUS" not in text
    assert "SOLVERTYPE = IEFPCM" in text
    assert "MOLECULE" not in text


def test_generated_pcmsolver_fallback_input_locks_stable_gepol_settings():
    text = _pcm_input_text(
        2,
        np.asarray([1.85, 1.52]),
        tessera_area_angstrom2=0.28,
        minimum_added_sphere_radius_angstrom=0.30,
    )

    assert "AREA = 0.2800000000" in text
    assert "MINRADIUS = 0.3000000000" in text


def test_smd_revised_halogen_coulomb_radii_are_locked():
    assert SMD_WATER_COULOMB_RADII_ANGSTROM["Br"] == 2.60
    assert SMD_WATER_COULOMB_RADII_ANGSTROM["I"] == 2.74


def test_gaff2_carbonyl_oxygen_profile_changes_only_o_atom_type():
    symbols = ["O", "O", "O", "C"]
    atom_types = ["o", "os", "oh", "c"]

    assert route2_water_coulomb_radii(
        symbols,
        atom_types=atom_types,
        profile=GAFF2_CARBONYL_O_PROFILE,
    ) == pytest.approx([1.70, 1.52, 1.52, 1.85])
    assert route2_water_coulomb_radii(
        symbols,
        atom_types=atom_types,
        profile="smd-iefpcm",
    ) == pytest.approx([1.52, 1.52, 1.52, 1.85])


def test_gaff2_carbonyl_oxygen_profile_requires_one_atom_type_per_atom():
    with pytest.raises(ValueError, match="one GAFF/GAFF2 atom type per atom"):
        route2_water_coulomb_radii(
            ["C", "O"],
            atom_types=["c"],
            profile=GAFF2_CARBONYL_O_PROFILE,
        )


def test_route2_provider_applies_gaff2_carbonyl_oxygen_profile():
    atoms = Atoms(
        "COO",
        positions=[[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [-1.3, 0.0, 0.0]],
    )
    atoms.info.update(
        charge=0,
        mult=1,
        mol2={"atom_types": ["c", "o", "os"]},
    )
    options = _route2_options("frozen")
    options["profile"] = GAFF2_CARBONYL_O_PROFILE

    provider = SMDImplicitSolvation(atoms, options, audit_dir=None)

    assert provider.coulomb_radii_angstrom == pytest.approx([1.85, 1.70, 1.52])
    assert provider.provenance["profile"] == GAFF2_CARBONYL_O_PROFILE
    assert "gaff2_pbsa_radii" in provider.provenance["citations"]


def test_route2_provider_rejects_unknown_cavity_policy():
    options = _route2_options("frozen")
    options["cavity_policy"] = "geometry-dependent"

    with pytest.raises(ValueError, match="cavity_policy must be"):
        SMDImplicitSolvation(_co_atoms(), options, audit_dir=None)


def test_smd_sasa_radii_lock_vdw_table_plus_point_four_angstrom_probe():
    symbols = ["H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I"]
    assert smd_sasa_radii(symbols) == pytest.approx(
        [1.60, 2.10, 1.95, 1.92, 1.87, 2.20, 2.20, 2.15, 2.25, 2.38]
    )


def test_gto_surface_and_reaction_couplings_are_reciprocal():
    rng = np.random.default_rng(42)
    positions = rng.normal(size=(4, 3))
    surface = rng.normal(size=(80, 3)) * 4.0
    coefficients = rng.normal(size=(4, 4))
    asc = rng.normal(size=80) * 0.01

    mep = gaussian_multipole_potential(surface, positions, coefficients)
    potential, gradient = asc_reaction_potential_gradient(
        positions, surface, asc
    )

    assert density_reaction_coupling(
        coefficients, potential, gradient
    ) == pytest.approx(float(np.dot(mep, asc)), abs=1.0e-12)


def test_point_multipole_surface_potential_has_expected_coulomb_limit():
    points = np.asarray([[2.0, 0.0, 0.0]])
    positions = np.asarray([[0.0, 0.0, 0.0]])
    coefficients = np.asarray([[1.0, 0.0, 0.0, 0.0]])

    assert point_multipole_potential(
        points, positions, coefficients
    ) == pytest.approx([0.5])


def test_point_surface_and_reaction_couplings_are_reciprocal():
    rng = np.random.default_rng(2026)
    positions = rng.normal(size=(4, 3))
    surface = rng.normal(size=(80, 3)) * 4.0
    coefficients = rng.normal(size=(4, 4))
    asc = rng.normal(size=80) * 0.01

    mep = point_multipole_potential(surface, positions, coefficients)
    potential, gradient = point_asc_reaction_potential_gradient(
        positions, surface, asc
    )

    assert density_reaction_coupling(
        coefficients, potential, gradient
    ) == pytest.approx(float(np.dot(mep, asc)), abs=1.0e-12)


def test_route2_pcm_uses_cavity_exterior_point_multipoles(monkeypatch):
    atoms = _co_atoms()
    provider = SMDImplicitSolvation(
        atoms, _route2_options("frozen"), audit_dir=None
    )
    coefficients = np.asarray(
        [[-0.1, 0.0, 0.0, 0.0], [0.1, 0.0, 0.0, 0.0]]
    )
    session = _FakePCMSolverSession(
        atoms.numbers,
        atoms.get_positions() / Bohr,
        None,
    ).open()
    calls = 0
    implementation = pcm_response_module.point_multipole_potential

    def tracked_point_multipole_potential(*args, **kwargs):
        nonlocal calls
        calls += 1
        return implementation(*args, **kwargs)

    monkeypatch.setattr(
        pcm_response_module,
        "point_multipole_potential",
        tracked_point_multipole_potential,
    )

    response = PCMSolverExternalMEPCavityResponse(
        session,
        cavity_radii_angstrom=provider.coulomb_radii_angstrom,
    )
    reaction_field = FixedCavityPCMReactionFieldLinearMap(
        response,
        atoms.get_positions(),
    )
    state = reaction_field.scf_snapshot(coefficients)

    assert calls == 1
    assert state.polarization_energy_hartree < 0.0


def test_route2_pcm_rejects_response_state_for_a_different_surface_potential():
    atoms = _co_atoms()
    coefficients = np.asarray(
        [[-0.1, 0.0, 0.0, 0.0], [0.1, 0.0, 0.0, 0.0]]
    )

    class _MismatchedResponse:
        contract_version = EXTERNAL_MEP_RESPONSE_CONTRACT_VERSION
        energy_response_is_reciprocal = True
        atomic_numbers = np.asarray(atoms.numbers, dtype=float)
        reference_positions_bohr = atoms.get_positions() / Bohr
        surface_points_bohr = np.asarray(
            [
                [4.0, 0.0, 0.0],
                [-4.0, 0.0, 0.0],
                [0.0, 4.0, 0.0],
                [0.0, -4.0, 0.0],
            ]
        )

        def solve(self, surface_potential_hartree_per_e):
            potential = np.asarray(
                surface_potential_hartree_per_e,
                dtype=float,
            ).copy()
            potential[0] += 1.0e-6
            charge = -0.1 * potential
            return SurfaceChargeState(
                surface_potential_hartree_per_e=potential,
                direct_surface_charge_e=charge,
                adjoint_surface_charge_e=charge,
                energy_conjugate_surface_charge_e=charge,
                polarization_energy_hartree=0.5
                * float(np.dot(potential, charge)),
            )

    with pytest.raises(RuntimeError, match="different surface potential"):
        FixedCavityPCMReactionFieldLinearMap(
            _MismatchedResponse(),
            atoms.get_positions(),
        ).scf_snapshot(
            coefficients,
        )


@pytest.mark.parametrize(
    ("symbols", "positions", "reference_kcal_mol"),
    [
        (
            ["O", "H", "H"],
            [[0, 0, 0], [0.958, 0, 0], [-0.239, 0.927, 0]],
            1.442140966301685,
        ),
        (
            ["C", "H", "H", "H", "H"],
            [
                [0, 0, 0],
                [0.629, 0.629, 0.629],
                [-0.629, -0.629, 0.629],
                [-0.629, 0.629, -0.629],
                [0.629, -0.629, -0.629],
            ],
            2.756661165552430,
        ),
        (
            ["C", "O", "H", "H", "H", "H"],
            [
                [0, 0, 0],
                [1.43, 0, 0],
                [-0.63, 0.9, 0],
                [-0.63, -0.45, 0.78],
                [-0.63, -0.45, -0.78],
                [1.8, 0.8, 0],
            ],
            2.592480030180290,
        ),
    ],
)
def test_native_cds_matches_static_nwchem_reference(
    symbols, positions, reference_kcal_mol
):
    # Static references were generated with NWChem's published mnsol.F SMD
    # implementation; no NWChem/PySCF code is required at runtime.
    result = smd_water_cds(symbols, np.asarray(positions, dtype=float))
    assert result.energy_kcal_mol == pytest.approx(
        reference_kcal_mol, abs=0.015
    )


class _FakePCMSolverSession:
    def __init__(self, atomic_numbers, coordinates_bohr, parsed_input_path):
        self.atomic_numbers = np.asarray(atomic_numbers)
        self.coordinates_bohr = np.asarray(coordinates_bohr)
        self.parsed_input_path = parsed_input_path
        self._is_open = False
        self._centers = np.asarray(
            [[4.0, 0.0, 0.0], [-4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, -4.0, 0.0]]
        )
        self._areas = np.ones(4)

    def __enter__(self):
        return self.open()

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return None

    def open(self):
        self._is_open = True
        return self

    def close(self):
        self._is_open = False

    @property
    def cavity_size(self):
        assert self._is_open
        return int(self._areas.size)

    @property
    def library_source(self):
        assert self._is_open
        return "fake-pcmsolver"

    @property
    def response_operator_is_symmetric(self):
        return True

    @property
    def cavity_centers_bohr(self):
        assert self._is_open
        return self._centers.copy()

    @property
    def cavity_areas_bohr2(self):
        assert self._is_open
        return self._areas.copy()

    def solve(self, mep):
        assert self._is_open
        mep = np.asarray(mep, dtype=float)
        asc = -0.1 * mep
        return {
            "asc": asc,
            "polarization_energy": 0.5 * float(np.dot(mep, asc)),
        }


class _FakePolarCalculator:
    def __init__(self, gas_state, response_state):
        self._last_polar_state = gas_state
        self.response_state = response_state
        self.route2_smd_profile = ROUTE2_SMD_CALCULATOR_PROFILE
        self.mace_polar_checkpoint_provenance = {
            "identifier": "polar-1-m",
            "release_url": "https://example.invalid/MACE-POLAR-1-M.model",
            "resolved_path": "/cache/MACEPOLAR1Mmodel",
            "size_bytes": 123,
            "sha256": "a" * 64,
        }
        self.calls = 0

    def polar_state(
        self,
        atoms,
        *,
        node_potential_ev=None,
        node_gradient_ev_per_angstrom=None,
    ):
        self.calls += 1
        assert np.asarray(node_potential_ev).shape == (len(atoms),)
        assert np.asarray(node_gradient_ev_per_angstrom).shape == (len(atoms), 3)
        return self.response_state, {}


class _FakeExactGTOCalculator(_FakePolarCalculator):
    def __init__(self, gas_state, response_state):
        super().__init__(gas_state, response_state)
        self.model_field_features = []
        self.graph_longrange_version = "0.4.0"

    @staticmethod
    def route2_gto_field_projection_spec():
        return MACEPolarGTOFieldProjectionSpec(
            receiver_sigmas_angstrom=(1.5, 3.0),
            receiver_max_l=1,
            receiver_normalization="receiver",
            upstream_matrix=np.asarray(
                [
                    [3.544907701811032, 0.0, 0.0, 0.0],
                    [3.544907701811032, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 5.771474235728387],
                    [0.0, 5.771474235728387, 0.0, 0.0],
                    [0.0, 0.0, 5.771474235728387, 0.0],
                    [0.0, 0.0, 0.0, 11.542948471456774],
                    [0.0, 11.542948471456774, 0.0, 0.0],
                    [0.0, 0.0, 11.542948471456774, 0.0],
                ]
            ),
        )

    def polar_state(
        self,
        atoms,
        *,
        model_field_features=None,
    ):
        self.calls += 1
        features = np.asarray(model_field_features, dtype=float)
        assert features.shape == (len(atoms), 8)
        self.model_field_features.append(features.copy())
        return self.response_state, {}


def _state(energy_ev, density):
    return SimpleNamespace(
        energy_ev=float(energy_ev),
        density_coefficients=np.asarray(density, dtype=float),
        dipole_e_angstrom=np.zeros(3),
    )


def _install_fake_pcm(monkeypatch, provider, tmp_path):
    dummy = tmp_path / "@route2-smd.pcm"
    dummy.write_text("parsed", encoding="utf-8")
    monkeypatch.setattr(smd_module, "PCMSolverSession", _FakePCMSolverSession)
    monkeypatch.setattr(provider, "_ensure_pcm_input", lambda: dummy)


class _WarningThenStablePCMSolverSession(_FakePCMSolverSession):
    def open(self):
        if "primary" in str(self.parsed_input_path):
            os.write(2, b"PCMSolver warning. synthetic primary instability\n")
        return super().open()

    def solve(self, mep):
        if "primary" in str(self.parsed_input_path):
            raise AssertionError("warned primary cavity must not evaluate PCM")
        return super().solve(mep)


class _AlwaysWarningPCMSolverSession(_FakePCMSolverSession):
    def open(self):
        os.write(2, b"PCMSolver warning. synthetic persistent instability\n")
        return super().open()

    def solve(self, mep):
        raise AssertionError("warned cavity must not evaluate PCM")


class _ResponseWarningThenStablePCMSolverSession(_FakePCMSolverSession):
    def solve(self, mep):
        result = super().solve(mep)
        if "primary" in str(self.parsed_input_path):
            os.write(2, b"PCMSolver warning. synthetic response instability\n")
        return result


class _CloseWarningThenStablePCMSolverSession(_FakePCMSolverSession):
    def close(self):
        if self._is_open and "primary" in str(self.parsed_input_path):
            os.write(2, b"PCMSolver warning. synthetic close instability\n")
        super().close()


class _PEDRAWarningPCMSolverSession(_FakePCMSolverSession):
    def open(self):
        Path("PEDRA.OUT__test").write_text(
            "** WARNING  ** A very poor tesselation has been chosen. "
            "It is valuable almost only for testing.\n",
            encoding="utf-8",
        )
        return super().open()


def _install_fake_pcm_pair(monkeypatch, provider, tmp_path, session_type):
    primary = tmp_path / "@primary.pcm"
    fallback = tmp_path / "@stability-fallback.pcm"
    primary.write_text("parsed", encoding="utf-8")
    fallback.write_text("parsed", encoding="utf-8")
    monkeypatch.setattr(smd_module, "PCMSolverSession", session_type)
    monkeypatch.setattr(provider, "_ensure_pcm_input", lambda: primary)
    monkeypatch.setattr(
        provider,
        "_ensure_pcm_fallback_input",
        lambda: fallback,
    )


def test_frozen_route2_composes_pcm_and_native_cds(monkeypatch, tmp_path):
    atoms = _co_atoms()
    gas = _state(-20.0, [[-0.1, 0.0, 0.0, 0.0], [0.1, 0.0, 0.0, 0.0]])
    calc = _FakePolarCalculator(gas, gas)
    provider = SMDImplicitSolvation(
        atoms, _route2_options("frozen"), audit_dir=tmp_path
    )
    _install_fake_pcm(monkeypatch, provider, tmp_path)

    result = provider.evaluate(atoms, calculator=calc)

    assert calc.calls == 0
    assert result.components_hartree["solute_polarization"] == 0.0
    assert result.components_hartree["pcm_polarization"] < 0.0
    assert result.energy_hartree == pytest.approx(
        result.components_hartree["electrostatic"]
        + result.components_hartree["cds"]
    )
    assert result.provenance["response"] == "frozen"
    assert result.provenance["route_role"] == "research-innovation"
    assert result.provenance["scientific_status"] == "energy-proof-of-concept"
    assert result.provenance["solution_phase_pes"] is False
    assert result.provenance["forces_available"] is False
    assert result.provenance["cavity_stability_policy_force_compatible"] is False
    assert result.provenance["pcm_mep_projection"].startswith(
        "cavity-exterior point monopoles and dipoles"
    )
    audit = (tmp_path / "route2-result.json").read_text(encoding="utf-8")
    assert '"schema_version": 9' in audit
    assert '"pcm_mep_projection": "cavity-exterior-point-multipole-l<=1"' in audit


def test_route2_retries_warning_cavity_with_deterministic_fallback(
    monkeypatch, tmp_path, capfd
):
    atoms = _co_atoms()
    gas = _state(-20.0, [[-0.1, 0.0, 0.0, 0.0], [0.1, 0.0, 0.0, 0.0]])
    calc = _FakePolarCalculator(gas, gas)
    provider = SMDImplicitSolvation(
        atoms, _route2_options("frozen"), audit_dir=tmp_path
    )
    _install_fake_pcm_pair(
        monkeypatch,
        provider,
        tmp_path,
        _WarningThenStablePCMSolverSession,
    )

    result = provider.evaluate(atoms, calculator=calc)

    assert PCM_WARNING_MARKER not in capfd.readouterr().err
    stability = result.provenance["cavity_stability"]
    assert stability == {
        "policy": "warning-fallback",
        "selected": "stability-fallback",
        "force_compatible": False,
        "geometry_selection_branch_free": False,
        "tessera_area_angstrom2": 0.28,
        "minimum_added_sphere_radius_angstrom": 0.30,
        "fallback_used": True,
        "attempt_count": 2,
    }
    audit = json.loads(
        (tmp_path / "route2-result.json").read_text(encoding="utf-8")
    )
    assert audit["cavity_stability"]["selected"] == "stability-fallback"
    assert [
        attempt["warning_detected"]
        for attempt in audit["cavity_stability"]["attempts"]
    ] == [True, False]
    primary, fallback = audit["cavity_stability"]["attempts"]
    assert primary["warning_stage"] == "pcm-initialization"
    assert primary["response_evaluated"] is False
    assert primary["response_evaluation_seconds"] == 0.0
    assert primary["pcm_initialization_seconds"] >= 0.0
    assert fallback["warning_stage"] is None
    assert fallback["response_evaluated"] is True
    assert fallback["response_evaluation_seconds"] >= 0.0
    assert fallback["pcm_initialization_seconds"] >= 0.0


def test_route2_fixed_stability_branch_never_attempts_primary(
    monkeypatch, tmp_path, capfd
):
    atoms = _co_atoms()
    gas = _state(-20.0, [[-0.1, 0.0, 0.0, 0.0], [0.1, 0.0, 0.0, 0.0]])
    calc = _FakePolarCalculator(gas, gas)
    options = _route2_options("frozen")
    options["cavity_policy"] = "fixed-stability-branch"
    provider = SMDImplicitSolvation(atoms, options, audit_dir=tmp_path)
    stable = tmp_path / "@stability.pcm"
    stable.write_text("parsed", encoding="utf-8")
    monkeypatch.setattr(
        smd_module,
        "PCMSolverSession",
        _WarningThenStablePCMSolverSession,
    )
    monkeypatch.setattr(
        provider,
        "_ensure_pcm_input",
        lambda: (_ for _ in ()).throw(
            AssertionError("fixed branch must not attempt primary")
        ),
    )
    monkeypatch.setattr(
        provider,
        "_ensure_pcm_stability_input",
        lambda: stable,
        raising=False,
    )

    result = provider.evaluate(atoms, calculator=calc)

    assert PCM_WARNING_MARKER not in capfd.readouterr().err
    assert result.provenance["cavity_stability"] == {
        "policy": "fixed-stability-branch",
        "selected": "fixed-stability-branch",
        "force_compatible": False,
        "geometry_selection_branch_free": True,
        "tessera_area_angstrom2": 0.28,
        "minimum_added_sphere_radius_angstrom": 0.30,
        "fallback_used": False,
        "attempt_count": 1,
    }
    assert result.provenance["cavity_stability_policy_force_compatible"] is False
    assert (
        result.provenance["cavity_policy_geometry_selection_branch_free"]
        is True
    )
    audit = json.loads(
        (tmp_path / "route2-result.json").read_text(encoding="utf-8")
    )
    assert audit["cavity_stability"]["policy"] == "fixed-stability-branch"
    assert audit["cavity_stability"]["selected"] == "fixed-stability-branch"
    assert audit["cavity_stability"]["force_compatible"] is False
    assert audit["cavity_stability"]["geometry_selection_branch_free"] is True
    assert len(audit["cavity_stability"]["attempts"]) == 1
    assert audit["cavity_stability"]["attempts"][0]["warning_detected"] is False


def test_route2_audits_pedra_warnings_separately_from_solver_gate(
    monkeypatch, tmp_path
):
    (tmp_path / "PEDRA.OUT__stale").write_text(
        "** WARNING  ** This warning belongs to an earlier run.\n",
        encoding="utf-8",
    )
    atoms = _co_atoms()
    gas = _state(-20.0, [[-0.1, 0.0, 0.0, 0.0], [0.1, 0.0, 0.0, 0.0]])
    calc = _FakePolarCalculator(gas, gas)
    options = _route2_options("frozen")
    options["cavity_policy"] = "fixed-stability-branch"
    provider = SMDImplicitSolvation(atoms, options, audit_dir=tmp_path)
    stable = tmp_path / "@stability.pcm"
    stable.write_text("parsed", encoding="utf-8")
    monkeypatch.setattr(
        smd_module,
        "PCMSolverSession",
        _PEDRAWarningPCMSolverSession,
    )
    monkeypatch.setattr(
        provider,
        "_ensure_pcm_stability_input",
        lambda: stable,
    )

    result = provider.evaluate(atoms, calculator=calc)

    assert result.provenance["cavity_stability"]["selected"] == (
        "fixed-stability-branch"
    )
    assert result.provenance["pcmsolver_diagnostics"] == {
        "native_stderr_warning_count": 0,
        "pedra_warning_count": 1,
        "pedra_warnings_are_selection_fatal": False,
    }
    audit = json.loads(
        (tmp_path / "route2-result.json").read_text(encoding="utf-8")
    )
    assert audit["cavity_stability"]["attempts"][0]["warning_detected"] is False
    assert audit["pcmsolver_diagnostics"] == {
        "native_stderr_warning_marker": PCM_WARNING_MARKER,
        "native_stderr_warning_count": 0,
        "pedra_warning_count": 1,
        "pedra_warnings": [
            {
                "file": str(tmp_path / "PEDRA.OUT__test"),
                "line": 1,
                "message": (
                    "** WARNING  ** A very poor tesselation has been chosen. "
                    "It is valuable almost only for testing."
                ),
            }
        ],
        "pedra_warnings_are_selection_fatal": False,
    }


def test_route2_fixed_stability_branch_fails_closed_on_warning(
    monkeypatch, tmp_path
):
    atoms = _co_atoms()
    gas = _state(-20.0, [[-0.1, 0.0, 0.0, 0.0], [0.1, 0.0, 0.0, 0.0]])
    calc = _FakePolarCalculator(gas, gas)
    options = _route2_options("frozen")
    options["cavity_policy"] = "fixed-stability-branch"
    provider = SMDImplicitSolvation(atoms, options, audit_dir=tmp_path)
    stable = tmp_path / "@stability.pcm"
    stable.write_text("parsed", encoding="utf-8")
    monkeypatch.setattr(
        smd_module,
        "PCMSolverSession",
        _AlwaysWarningPCMSolverSession,
    )
    monkeypatch.setattr(
        provider,
        "_ensure_pcm_stability_input",
        lambda: stable,
    )

    with pytest.raises(RuntimeError, match="refuses to publish"):
        provider.evaluate(atoms, calculator=calc)

    assert not (tmp_path / "route2-result.json").exists()
    assert not (tmp_path / "route2-state.npz").exists()
    failure = json.loads(
        (tmp_path / "route2-failure.json").read_text(encoding="utf-8")
    )
    assert failure["cavity_stability"]["policy"] == "fixed-stability-branch"
    assert failure["cavity_stability"]["geometry_selection_branch_free"] is True
    assert failure["cavity_stability"]["selected"] is None
    assert len(failure["cavity_stability"]["attempts"]) == 1
    assert failure["cavity_stability"]["attempts"][0]["warning_detected"] is True
    assert failure["pcmsolver_diagnostics"] == {
        "native_stderr_warning_marker": PCM_WARNING_MARKER,
        "native_stderr_warning_count": 1,
        "pedra_warning_count": 0,
        "pedra_warnings": [],
        "pedra_warnings_are_selection_fatal": False,
    }


def test_route2_retries_warning_detected_during_response(monkeypatch, tmp_path):
    atoms = _co_atoms()
    gas = _state(-20.0, [[-0.1, 0.0, 0.0, 0.0], [0.1, 0.0, 0.0, 0.0]])
    calc = _FakePolarCalculator(gas, gas)
    provider = SMDImplicitSolvation(
        atoms, _route2_options("frozen"), audit_dir=tmp_path
    )
    _install_fake_pcm_pair(
        monkeypatch,
        provider,
        tmp_path,
        _ResponseWarningThenStablePCMSolverSession,
    )

    result = provider.evaluate(atoms, calculator=calc)

    assert result.provenance["cavity_stability"]["selected"] == (
        "stability-fallback"
    )
    audit = json.loads(
        (tmp_path / "route2-result.json").read_text(encoding="utf-8")
    )
    primary, fallback = audit["cavity_stability"]["attempts"]
    assert primary["warning_detected"] is True
    assert primary["warning_stage"] == "response-evaluation"
    assert primary["response_evaluated"] is True
    assert fallback["warning_detected"] is False
    assert fallback["response_evaluated"] is True


def test_route2_captures_close_warning_and_retries(monkeypatch, tmp_path, capfd):
    atoms = _co_atoms()
    gas = _state(-20.0, [[-0.1, 0.0, 0.0, 0.0], [0.1, 0.0, 0.0, 0.0]])
    calc = _FakePolarCalculator(gas, gas)
    provider = SMDImplicitSolvation(
        atoms, _route2_options("frozen"), audit_dir=tmp_path
    )
    _install_fake_pcm_pair(
        monkeypatch,
        provider,
        tmp_path,
        _CloseWarningThenStablePCMSolverSession,
    )

    result = provider.evaluate(atoms, calculator=calc)

    assert PCM_WARNING_MARKER not in capfd.readouterr().err
    assert result.provenance["cavity_stability"]["selected"] == (
        "stability-fallback"
    )
    audit = json.loads(
        (tmp_path / "route2-result.json").read_text(encoding="utf-8")
    )
    primary, fallback = audit["cavity_stability"]["attempts"]
    assert primary["warning_detected"] is True
    assert primary["warning_stage"] == "pcm-close"
    assert fallback["warning_detected"] is False


def test_scf_route2_skips_warned_primary_before_ml_scf(monkeypatch, tmp_path):
    atoms = _co_atoms()
    gas_density = np.asarray(
        [[-0.1, 0.0, 0.0, 0.0], [0.1, 0.0, 0.0, 0.0]]
    )
    response_density = np.asarray(
        [[-0.12, 0.0, 0.0, 0.0], [0.12, 0.0, 0.0, 0.0]]
    )
    gas = _state(-20.0, gas_density)
    response = _state(-19.9, response_density)
    calc = _FakePolarCalculator(gas, response)
    provider = SMDImplicitSolvation(
        atoms, _route2_options("scf"), audit_dir=tmp_path
    )
    _install_fake_pcm_pair(
        monkeypatch,
        provider,
        tmp_path,
        _WarningThenStablePCMSolverSession,
    )

    result = provider.evaluate(atoms, calculator=calc)

    assert result.provenance["cavity_stability"]["selected"] == (
        "stability-fallback"
    )
    assert calc.calls > 1
    audit = json.loads(
        (tmp_path / "route2-result.json").read_text(encoding="utf-8")
    )
    primary, fallback = audit["cavity_stability"]["attempts"]
    assert primary["response_evaluated"] is False
    assert fallback["response_evaluated"] is True


def test_route2_fails_closed_when_fallback_still_warns(monkeypatch, tmp_path):
    atoms = _co_atoms()
    gas = _state(-20.0, [[-0.1, 0.0, 0.0, 0.0], [0.1, 0.0, 0.0, 0.0]])
    calc = _FakePolarCalculator(gas, gas)
    provider = SMDImplicitSolvation(
        atoms, _route2_options("frozen"), audit_dir=tmp_path
    )
    _install_fake_pcm_pair(
        monkeypatch,
        provider,
        tmp_path,
        _AlwaysWarningPCMSolverSession,
    )

    with pytest.raises(RuntimeError, match="refuses to publish"):
        provider.evaluate(atoms, calculator=calc)

    assert not (tmp_path / "route2-result.json").exists()
    assert not (tmp_path / "route2-state.npz").exists()
    failure = json.loads(
        (tmp_path / "route2-failure.json").read_text(encoding="utf-8")
    )
    assert failure["cavity_stability"]["selected"] is None
    assert all(
        attempt["warning_detected"]
        for attempt in failure["cavity_stability"]["attempts"]
    )


def test_scf_route2_iterates_density_and_adds_solute_polarization(
    monkeypatch, tmp_path
):
    atoms = _co_atoms()
    gas_density = np.asarray(
        [[-0.1, 0.0, 0.0, 0.0], [0.1, 0.0, 0.0, 0.0]]
    )
    response_density = np.asarray(
        [[-0.12, 0.0, 0.0, 0.0], [0.12, 0.0, 0.0, 0.0]]
    )
    gas = _state(-20.0, gas_density)
    response = _state(-19.9, response_density)
    calc = _FakePolarCalculator(gas, response)
    provider = SMDImplicitSolvation(
        atoms, _route2_options("scf"), audit_dir=tmp_path
    )
    _install_fake_pcm(monkeypatch, provider, tmp_path)

    result = provider.evaluate(atoms, calculator=calc)

    assert calc.calls > 1
    assert result.components_hartree["solute_polarization"] == pytest.approx(
        0.1 / Hartree
    )
    assert result.provenance["converged"] is True
    assert result.provenance["iterations"] > 1
    assert (tmp_path / "route2-state.npz").is_file()
    assert (tmp_path / "route2-result.json").is_file()


def test_exact_gto_profile_keeps_point_source_and_audits_model_features(
    monkeypatch,
    tmp_path,
):
    atoms = _co_atoms()
    density = np.asarray(
        [[-0.1, 0.02, -0.03, 0.04], [0.1, -0.02, 0.03, -0.04]]
    )
    gas = _state(-20.0, density)
    response = _state(-19.95, density)
    calculator = _FakeExactGTOCalculator(gas, response)
    options = _route2_options("scf")
    options["profile"] = PCMSOLVER_EXACT_GTO_FIELD_PROFILE
    provider = SMDImplicitSolvation(
        atoms,
        options,
        audit_dir=tmp_path,
    )
    _install_fake_pcm(monkeypatch, provider, tmp_path)

    result = provider.evaluate(atoms, calculator=calculator)

    assert calculator.calls == 2
    assert result.provenance["solute_source"] == "point-multipole-l1"
    assert result.provenance["reaction_field_projector"] == "exact-gto-v1"
    assert result.provenance["density_dual_field_gauge"] == (
        "continuum-zero-at-infinity"
    )
    assert result.provenance["model_field_gauge"] == (
        "atomic-center-mean-zero-v1"
    )
    assert result.provenance["strict_original_smd_equivalence"] is False
    with np.load(tmp_path / "route2-state.npz") as state:
        assert state["model_field_features"].shape == (2, 8)
        expected_mep = point_multipole_potential(
            state["cavity_centers_bohr"],
            atoms.get_positions(),
            state["root_density_coefficients"],
        )
        np.testing.assert_allclose(
            state["mep_hartree_per_e"],
            expected_mep,
        )
    audit = json.loads(
        (tmp_path / "route2-result.json").read_text(encoding="utf-8")
    )
    assert audit["pcm_mep_projection"] == (
        "cavity-exterior-point-multipole-l<=1"
    )
    assert audit["reaction_field_projector"] == "exact-gto-v1"
    assert audit["density_dual_field_gauge"] == (
        "continuum-zero-at-infinity"
    )
    assert audit["model_field_gauge"] == "atomic-center-mean-zero-v1"
    assert np.isfinite(audit["model_field_gauge_reference_ev"])
    assert audit["model_field_feature_shape"] == [2, 8]
    assert audit["model_field_projection_contract"][
        "graph_longrange_version"
    ] == "0.4.0"
    assert len(
        audit["model_field_projection_contract"][
            "upstream_matrix_sha256"
        ]
    ) == 64
    assert result.provenance["graph_longrange_version"] == "0.4.0"
    assert result.provenance["model_field_projection_contract"] == (
        audit["model_field_projection_contract"]
    )
    assert audit["polarization_energy"][
        "absolute_identity_error_ev"
    ] == pytest.approx(0.0, abs=1.0e-12)
    assert audit["mace_polar_checkpoint"] == (
        calculator.mace_polar_checkpoint_provenance
    )
    assert result.provenance["mace_polar_checkpoint"] == (
        calculator.mace_polar_checkpoint_provenance
    )


def test_centered_local_jet_profile_audits_dual_and_model_fields(
    monkeypatch,
    tmp_path,
):
    atoms = _co_atoms()
    density = np.asarray(
        [[-0.1, 0.02, -0.03, 0.04], [0.1, -0.02, 0.03, -0.04]]
    )
    gas = _state(-20.0, density)
    response = _state(-19.95, density)
    calculator = _FakePolarCalculator(gas, response)
    options = _route2_options("scf")
    options["profile"] = PCMSOLVER_CENTERED_LOCAL_JET_FIELD_PROFILE
    provider = SMDImplicitSolvation(
        atoms,
        options,
        audit_dir=tmp_path,
    )
    _install_fake_pcm(monkeypatch, provider, tmp_path)

    result = provider.evaluate(atoms, calculator=calculator)

    assert result.provenance["reaction_field_projector"] == "local-jet"
    assert result.provenance["density_dual_field_gauge"] == (
        "continuum-zero-at-infinity"
    )
    assert result.provenance["model_field_gauge"] == (
        "atomic-center-mean-zero-v1"
    )
    with np.load(tmp_path / "route2-state.npz") as state:
        dual_field = state["reaction_field_values_ev"]
        model_field = state["model_local_field_values_ev"]
        assert dual_field.shape == model_field.shape == (2, 4)
        assert float(np.mean(model_field[:, 0])) == pytest.approx(
            0.0,
            abs=1.0e-15,
        )
        np.testing.assert_allclose(model_field[:, 1:], dual_field[:, 1:])
    audit = json.loads(
        (tmp_path / "route2-result.json").read_text(encoding="utf-8")
    )
    assert audit["density_dual_field_gauge"] == (
        "continuum-zero-at-infinity"
    )
    assert audit["model_field_gauge"] == "atomic-center-mean-zero-v1"
    assert audit["model_local_field_shape"] == [2, 4]
    assert audit["model_field_feature_shape"] is None


def test_scf_route2_closes_energy_and_field_on_the_same_finite_tolerance_root(
    monkeypatch, tmp_path
):
    atoms = _co_atoms()
    gas_density = np.asarray(
        [[-0.1, 0.0, 0.0, 0.0], [0.1, 0.0, 0.0, 0.0]]
    )
    response_density = np.asarray(
        [[-0.12, 0.0, 0.0, 0.0], [0.12, 0.0, 0.0, 0.0]]
    )
    gas = _state(-20.0, gas_density)
    response = _state(-19.9, response_density)
    calc = _FakePolarCalculator(gas, response)
    monkeypatch.setattr(smd_module, "SCF_DENSITY_TOLERANCE", 1.1e-2)
    provider = SMDImplicitSolvation(
        atoms, _route2_options("scf"), audit_dir=tmp_path
    )
    _install_fake_pcm(monkeypatch, provider, tmp_path)

    result = provider.evaluate(atoms, calculator=calc)

    expected_root_density = 0.5 * (gas_density + response_density)
    expected_session = _FakePCMSolverSession(
        atoms.numbers,
        atoms.get_positions() / Bohr,
        tmp_path / "@unused.pcm",
    ).open()
    expected_mep = point_multipole_potential(
        expected_session.cavity_centers_bohr,
        atoms.get_positions(),
        expected_root_density,
    )
    expected_pcm_energy = -0.05 * float(np.dot(expected_mep, expected_mep))
    assert result.components_hartree["pcm_polarization"] == pytest.approx(
        expected_pcm_energy
    )
    assert result.provenance["same_root_energy_ledger"] is True
    assert result.provenance[
        "fixed_point_density_residual_inf_e"
    ] == pytest.approx(1.0e-2)

    with np.load(tmp_path / "route2-state.npz") as state:
        np.testing.assert_allclose(
            state["root_density_coefficients"],
            expected_root_density,
        )
        np.testing.assert_allclose(
            state["response_density_coefficients"],
            response_density,
        )
        np.testing.assert_allclose(
            state["mep_hartree_per_e"],
            expected_mep,
        )
    audit = json.loads(
        (tmp_path / "route2-result.json").read_text(encoding="utf-8")
    )
    assert audit["fixed_point"]["density_residual_inf_e"] == pytest.approx(
        1.0e-2
    )
    assert audit["fixed_point"]["same_root_energy_ledger"] is True


def test_scf_route2_requires_two_energy_samples_when_density_starts_converged(
    monkeypatch, tmp_path
):
    atoms = _co_atoms()
    gas_density = np.asarray(
        [[-0.1, 0.0, 0.0, 0.0], [0.1, 0.0, 0.0, 0.0]]
    )
    response_density = np.asarray(
        [[-0.105, 0.0, 0.0, 0.0], [0.105, 0.0, 0.0, 0.0]]
    )
    gas = _state(-20.0, gas_density)
    response = _state(-19.9, response_density)
    calc = _FakePolarCalculator(gas, response)
    monkeypatch.setattr(smd_module, "SCF_DENSITY_TOLERANCE", 1.1e-2)
    provider = SMDImplicitSolvation(
        atoms, _route2_options("scf"), audit_dir=tmp_path
    )
    _install_fake_pcm(monkeypatch, provider, tmp_path)

    result = provider.evaluate(atoms, calculator=calc)

    assert calc.calls == 2
    assert result.provenance["iterations"] == 2
    audit = json.loads(
        (tmp_path / "route2-result.json").read_text(encoding="utf-8")
    )
    assert audit["scf"]["require_two_energy_samples"] is True
    assert audit["scf"]["history"][0]["energy_residual_ev"] is None
    assert audit["scf"]["history"][1]["energy_residual_ev"] == pytest.approx(
        0.0
    )


def test_route2_rejects_geometry_changes(tmp_path):
    atoms = _co_atoms()
    provider = SMDImplicitSolvation(
        atoms, _route2_options("frozen"), audit_dir=tmp_path
    )
    moved = atoms.copy()
    moved.positions[0, 0] += 0.01

    with pytest.raises(ValueError, match="fixed-conformer"):
        provider.evaluate(moved, calculator=object())


def test_route2_rejects_formally_charged_tripos_types_as_zwitterionic_domain():
    atoms = _co_atoms()
    atoms.info["mol2"] = {"atom_types": ["C.cat", "O.co2"]}

    with pytest.raises(ValueError, match="excludes salts and zwitterions"):
        SMDImplicitSolvation(atoms, _route2_options(), audit_dir=None)


def test_route2_python_api_rejects_unmarked_polar_state_provider(tmp_path):
    atoms = _co_atoms()
    provider = SMDImplicitSolvation(
        atoms, _route2_options(), audit_dir=tmp_path
    )
    calculator = SimpleNamespace(polar_state=lambda *_args, **_kwargs: None)

    with pytest.raises(TypeError, match="official MACE-POLAR-1-M float64"):
        provider.evaluate(atoms, calculator=calculator)


def test_pcmsolver_parser_and_explicit_library_must_share_install_prefix(
    monkeypatch, tmp_path
):
    library = tmp_path / "install-a" / "lib" / "libpcm.so"
    library.parent.mkdir(parents=True)
    library.write_bytes(b"fixture")
    python_root = tmp_path / "install-b" / "lib" / "python"
    package = python_root / "pcmsolver"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(
        "def parse_pcm_input(*args, **kwargs):\n    return None\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("PCMSOLVER_LIBRARY", str(library))
    monkeypatch.setenv("PCMSOLVER_PYTHON_PATH", str(python_root))
    previous = sys.modules.pop("pcmsolver", None)
    try:
        with pytest.raises(ImportError, match="same PCMSolver installation prefix"):
            smd_module._load_pcmsolver_parser()
    finally:
        sys.modules.pop("pcmsolver", None)
        if previous is not None:
            sys.modules["pcmsolver"] = previous
