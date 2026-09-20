from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = REPOSITORY_ROOT / "docs/implicit-solvation/benchmarks"
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

import run_ddlpb_reference_validation as validation
from benchmark_core import load_json


@dataclass(frozen=True)
class _FakeSettings:
    lmax: int = 9
    n_lebedev: int = 302
    eta: float = 0.1
    shift: float = 0.0
    solver_tolerance: float = 1.0e-10
    max_iterations: int = 200
    jacobi_n_diis: int = 20
    n_proc: int = 1
    solvent_epsilon: float = 78.5
    solute_epsilon: float = 1.0
    enable_fmm: bool = False


class _FakeDDXLPB:
    def __init__(
        self,
        atoms,
        charges,
        *,
        solvent_kappa_inverse_angstrom,
        settings=None,
        audit_dir=None,
    ):
        del audit_dir
        self.atoms = atoms.copy()
        self.charges = np.asarray(charges, dtype=float).copy()
        self.radii = np.array(
            [1.2 if symbol == "H" else 1.5 for symbol in atoms.get_chemical_symbols()]
        )
        self.kappa = float(solvent_kappa_inverse_angstrom)
        self.settings = settings or _FakeSettings()
        self.provenance = {
            "provider": "fake-ddx",
            "provider_version": "0.8.0-test-double",
            "settings": self.settings.__dict__,
        }

    def evaluate(self, atoms, need_forces=False, calculator=None):
        del calculator
        xyz = np.asarray(atoms.positions, dtype=float)
        if len(atoms) == 1:
            energy = validation.charged_sphere_exact_energy_hartree(
                charge_e=float(self.charges[0]),
                radius_angstrom=float(self.radii[0]),
                solvent_kappa_inverse_angstrom=self.kappa,
                solute_epsilon=self.settings.solute_epsilon,
                solvent_epsilon=self.settings.solvent_epsilon,
            )
            forces = np.zeros_like(xyz)
        else:
            energy = 0.0
            forces = np.zeros_like(xyz)
            for i in range(len(xyz)):
                for j in range(i):
                    delta = xyz[i] - xyz[j]
                    energy += 0.5 * float(delta @ delta) * 1.0e-3
                    forces[i] -= delta * 1.0e-3
                    forces[j] += delta * 1.0e-3
        return SimpleNamespace(
            energy_hartree=energy,
            forces_hartree_per_angstrom=forces if need_forces else None,
            components_hartree={"polar": energy},
            provenance={"converged": True},
        )


def test_ionic_strength_mapping_roundtrips_through_pyddx():
    record = validation.kappa_mapping_record(0.125)

    assert record["ionic_strength_molar"] > 0.0
    assert record["pyddx_version"] == "0.8.0"
    assert record["roundtrip_kappa_inverse_angstrom"] == pytest.approx(0.125)
    assert record["relative_roundtrip_error"] < 1.0e-8
    assert record["roundtrip_within_constants_tolerance"] is True
    assert set(record["scipy_constants"]) == {
        "Boltzmann_constant_J_K",
        "Avogadro_constant_mol_inverse",
        "elementary_charge_C",
        "vacuum_electric_permittivity_F_m",
    }


def test_apbs_input_matches_sphere_bvp_and_keeps_ions_out_of_reference():
    rendered = validation.render_apbs_sphere_input(
        pqr_basename="sphere.pqr",
        ionic_strength_molar=0.123456789012,
        spacing_angstrom=0.25,
        points_per_axis=97,
    )
    solv, reference = rendered.split("elec name ref", maxsplit=1)

    assert "mol pqr sphere.pqr" in rendered
    assert "srfm mol" in rendered
    assert "srad 0.000000" in rendered
    assert "ion charge +1 conc 0.123456789012 radius 0.000000" in solv
    assert "ion charge -1 conc 0.123456789012 radius 0.000000" in solv
    assert "ion charge" not in reference
    assert "apolar" not in rendered.lower()
    assert "print elecEnergy solv - ref end" in rendered


def test_apbs_report_parser_preserves_printed_sign_units_and_diagnostics():
    stdout = """
    Temperature: 298.15 K
    Solvent dielectric: 78.500
    Ionic strength: 0.12346 M
    Debye length: 8.765 A
    Vpbe_ctor2: xkappa = 0.1142
    Global net ELEC energy = -2.500000000000E+01 kJ/mol
    """
    parsed = validation.parse_apbs_report(stdout)

    assert parsed["polar_energy"]["printed"] == "-2.500000000000E+01"
    assert parsed["polar_energy"]["unit"] == "kJ/mol"
    assert parsed["polar_energy"]["value_kj_mol"] == -25.0
    assert parsed["reported"]["ionic_strength"][0]["printed"] == "0.12346"
    assert parsed["reported"]["temperature"][0]["raw_line"].strip().startswith(
        "Temperature:"
    )
    assert parsed["reported"]["xkappa"][0]["printed"] == "0.1142"


def test_agreement_gate_never_passes_without_required_grid_and_domain_evidence():
    one_grid = [{"spacing_angstrom": 0.25, "domain_length_angstrom": 24.0}]
    failed_six = [
        {
            "spacing_angstrom": spacing,
            "domain_length_angstrom": domain,
            "status": "failed",
        }
        for spacing in (0.5, 0.33, 0.25)
        for domain in (24.0, 32.0)
    ]

    assert validation.assess_independent_crosscheck(one_grid)[
        "independent_crosscheck_complete"
    ] is False
    failed = validation.assess_independent_crosscheck(failed_six)
    assert failed["independent_crosscheck_complete"] is False
    assert "successful" in failed["reason"]


def test_validation_retains_full_inputs_force_errors_and_nonblocking_apbs_failure(
    water_mol2,
    tmp_path,
):
    output_dir = tmp_path / "new-output"

    artifact = validation.run_validation(
        mol2_path=water_mol2,
        output_dir=output_dir,
        solvent_kappa_inverse_angstrom=0.1,
        provider_class=_FakeDDXLPB,
        settings_class=_FakeSettings,
        apbs_executable=tmp_path / "missing-apbs",
    )

    assert artifact["core_validation"]["passed"] is True
    assert artifact["independent_apbs_crosscheck"][
        "independent_crosscheck_complete"
    ] is False
    assert artifact["independent_apbs_crosscheck"]["status"] == "failed"
    assert artifact["overall_release_gate"]["passed"] is True
    assert artifact["inputs"]["molecular_mol2"]["bytes"] == water_mol2.stat().st_size
    assert artifact["inputs"]["provider_input_match"]["charges_exact"] is True
    assert len(artifact["molecular_force_validation"]["steps"]) == 2
    for step in artifact["molecular_force_validation"]["steps"]:
        finite_difference = np.asarray(
            step["finite_difference_forces_hartree_per_angstrom"]
        )
        assert finite_difference.shape == (3, 3)
        assert np.asarray(step["signed_errors_hartree_per_angstrom"]).shape == (3, 3)
        assert step["denominators_angstrom"] == [2.0 * step["step_angstrom"]] * 9
    assert artifact["refinement"]["force_delta_hartree_per_angstrom"]
    assert (output_dir / "molecular-input.mol2").read_bytes() == water_mol2.read_bytes()
    assert (output_dir / "validation.json").is_file()


def test_validation_refuses_nonpositive_kappa(water_mol2, tmp_path):
    with pytest.raises(ValueError, match="strictly positive"):
        validation.run_validation(
            mol2_path=water_mol2,
            output_dir=tmp_path / "zero",
            solvent_kappa_inverse_angstrom=0.0,
            provider_class=_FakeDDXLPB,
            settings_class=_FakeSettings,
        )


def test_validation_refuses_existing_output_directory(water_mol2, tmp_path):
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    with pytest.raises(FileExistsError, match="must not already exist"):
        validation.run_validation(
            mol2_path=water_mol2,
            output_dir=occupied,
            solvent_kappa_inverse_angstrom=0.1,
            provider_class=_FakeDDXLPB,
            settings_class=_FakeSettings,
        )


def test_core_failure_retains_completed_fd_side_and_exact_failure_location(
    water_mol2,
    tmp_path,
):
    class FailsOnFirstMinus(_FakeDDXLPB):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.energy_only_calls = 0

        def evaluate(self, atoms, need_forces=False, calculator=None):
            if len(atoms) > 1 and not need_forces:
                self.energy_only_calls += 1
                if self.energy_only_calls == 2:
                    raise RuntimeError("deliberate minus-side failure")
            return super().evaluate(
                atoms,
                need_forces=need_forces,
                calculator=calculator,
            )

    output_dir = tmp_path / "failed-output"
    with pytest.raises(RuntimeError, match="partial evidence retained"):
        validation.run_validation(
            mol2_path=water_mol2,
            output_dir=output_dir,
            solvent_kappa_inverse_angstrom=0.1,
            provider_class=FailsOnFirstMinus,
            settings_class=_FakeSettings,
        )

    artifact = load_json(output_dir / "validation.json")
    progress = artifact["partial_core_evidence"]
    assert len(progress["completed_sphere_records"]) == 2
    completed = progress["molecular_fd_component_evaluations"]
    assert progress["completion_counts"] == {
        "charged_sphere_records": 2,
        "finite_difference_component_sides": 1,
        "complete_finite_difference_steps": 0,
    }
    assert len(completed) == 1
    assert completed[0]["name"] == "molecular_finite_difference"
    assert completed[0]["step_angstrom"] == 1.0e-4
    assert completed[0]["atom_index"] == 0
    assert completed[0]["axis"] == 0
    assert completed[0]["side"] == "plus"
    assert completed[0]["status"] == "success"
    assert np.isfinite(completed[0]["energy_hartree"])
    assert completed[0]["timing_seconds"] >= 0.0
    assert progress["failed_stage"] == {
        "name": "molecular_finite_difference",
        "step_angstrom": 1.0e-4,
        "atom_index": 0,
        "axis": 0,
        "side": "minus",
        "status": "failed",
        "error_type": "RuntimeError",
        "error": "deliberate minus-side failure",
    }


def test_validation_fails_instead_of_claiming_fallback_settings(water_mol2, tmp_path):
    class MissingSettings(_FakeDDXLPB):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            del self.settings

    output_dir = tmp_path / "missing-settings"
    with pytest.raises(RuntimeError, match="partial evidence retained"):
        validation.run_validation(
            mol2_path=water_mol2,
            output_dir=output_dir,
            solvent_kappa_inverse_angstrom=0.1,
            provider_class=MissingSettings,
            settings_class=_FakeSettings,
        )

    artifact = load_json(output_dir / "validation.json")
    assert artifact["core_validation"]["passed"] is False
    assert artifact["partial_core_evidence"]["failed_stage"] == {
        "name": "charged_sphere",
        "solvent_kappa_inverse_angstrom": 0.1,
        "status": "failed",
        "error_type": "ValueError",
        "error": "ddLPB provider did not expose its mandatory settings.",
    }
