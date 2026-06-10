"""WS-A — calculator unit/cutoff admission contract for MD.

Covers the admission rules added in WS-A:
  * a raw ASE calculator (no MAPLE unit contract) is rejected by MD;
  * ``wrap_ase_calculator`` performs a real eV->Ha conversion and is then accepted,
    while a Ha-native source is passed through unchanged (no double conversion);
  * a non-whitelisted source unit is rejected by the adapter (no free-form labels);
  * an unknown neighbor cutoff is rejected for PBC MD unless ``allow_unknown_cutoff``;
  * the run context / calculator provenance record the unit contract and cutoff policy.
"""

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.lj import LennardJones

from maple.function.calculator import wrap_ase_calculator
from maple.function.calculator._ase_unit_contract import EV2HARTREE
from maple.function.dispatcher.md.capabilities import MDUnitContractError
from maple.function.dispatcher.md.ensemble.nve import NVE, NVEParams
from maple.function.dispatcher.md.provenance import (
    build_run_context,
    collect_calculator_provenance,
)
from maple.function.dispatcher.md.semantics import resolve_md_dof_policy


def _lj(rc=6.5):
    return LennardJones(epsilon=0.0103, sigma=3.4, rc=rc, smooth=True)


def _pair():
    # Two Ar atoms ~1.1 sigma apart in a large box: a real, nonzero LJ pair energy.
    return Atoms("Ar2", positions=[[0.0, 0.0, 0.0], [3.8, 0.0, 0.0]],
                 cell=[20.0, 20.0, 20.0], pbc=False)


def _pbc_atoms(calc):
    atoms = Atoms("Ar", positions=[[0.0, 0.0, 0.0]], cell=[10.0, 10.0, 10.0], pbc=True)
    atoms.calc = calc
    return atoms


# ── adapter: real conversion, whitelist ─────────────────────────────────────

def test_adapter_converts_ev_to_hartree():
    base = _pair()
    raw = base.copy()
    raw.calc = _lj()
    e_ev = raw.get_potential_energy()
    f_ev = raw.get_forces()
    assert e_ev != 0.0

    wrapped = base.copy()
    wrapped.calc = wrap_ase_calculator(_lj(), energy_unit="eV", force_unit="eV/A")
    assert np.isclose(wrapped.get_potential_energy(), e_ev * EV2HARTREE)
    assert np.allclose(wrapped.get_forces(), f_ev * EV2HARTREE)


def test_adapter_ha_source_is_passed_through():
    base = _pair()
    raw = base.copy()
    raw.calc = _lj()
    e_native = raw.get_potential_energy()  # the LJ number, taken as already-Hartree

    wrapped = base.copy()
    wrapped.calc = wrap_ase_calculator(_lj(), energy_unit="Ha", force_unit="Ha/A")
    assert np.isclose(wrapped.get_potential_energy(), e_native)  # factor 1.0, no convert


def test_adapter_rejects_non_whitelisted_units():
    with pytest.raises(ValueError, match="Unsupported energy_unit"):
        wrap_ase_calculator(_lj(), energy_unit="kcal/mol")
    with pytest.raises(ValueError, match="Unsupported force_unit"):
        wrap_ase_calculator(_lj(), force_unit="kcal/mol/A")
    with pytest.raises(ValueError, match="Unsupported stress_unit"):
        wrap_ase_calculator(_lj(), stress_unit="GPa")


# ── MD admission: unit contract ─────────────────────────────────────────────

def test_raw_ase_calculator_rejected_by_md(tmp_path):
    atoms = _pbc_atoms(_lj(rc=2.0))
    with pytest.raises(MDUnitContractError, match="unit contract"):
        NVE(output=str(tmp_path / "x.out"), atoms=atoms, paras={"steps": 0, "verbose": 0})


def test_wrapped_ase_calculator_accepted_by_md(tmp_path):
    calc = wrap_ase_calculator(_lj(rc=2.0), pbc_md_supported=True, neighbor_cutoff_A=2.0)
    sim = NVE(output=str(tmp_path / "x.out"), atoms=_pbc_atoms(calc),
              paras={"steps": 0, "verbose": 0})
    assert sim is not None


# ── MD admission: unknown cutoff ────────────────────────────────────────────

def _no_cutoff_calc():
    # Declares the unit contract + PBC support but no neighbor cutoff.
    return wrap_ase_calculator(_lj(rc=2.0), pbc_md_supported=True)


def test_unknown_cutoff_rejected_for_pbc_md(tmp_path):
    with pytest.raises(ValueError, match="does not expose a neighbor cutoff"):
        NVE(output=str(tmp_path / "x.out"), atoms=_pbc_atoms(_no_cutoff_calc()),
            paras={"steps": 0, "verbose": 0})


def test_unknown_cutoff_allowed_with_explicit_override(tmp_path):
    sim = NVE(output=str(tmp_path / "y.out"), atoms=_pbc_atoms(_no_cutoff_calc()),
              paras={"steps": 0, "verbose": 0, "allow_unknown_cutoff": True})
    assert sim is not None


# ── MD admission: equality boundary (cutoff == minimum-image radius) ────────

def test_cutoff_equal_to_mic_radius_is_rejected(tmp_path):
    """Equality is rejected deliberately: a cell with L = 2 * cutoff places an
    atom exactly at its periodic image's interaction surface, where float
    rounding (and any barostat shrinkage) flips the comparison silently. The
    safer rule is strict ``cutoff < radius``; this test pins that boundary so a
    future relaxation to ``<=`` cannot land without breaking this contract.
    """
    # 10 Å orthorhombic cell -> MIC radius 5.0 Å; declare cutoff 5.0 Å.
    calc = wrap_ase_calculator(_lj(rc=2.0), pbc_md_supported=True, neighbor_cutoff_A=5.0)
    with pytest.raises(ValueError, match=">= minimum-image radius"):
        NVE(output=str(tmp_path / "boundary.out"), atoms=_pbc_atoms(calc),
            paras={"steps": 0, "verbose": 0})


def test_multi_image_safe_calculator_is_not_forced_into_mic_supercell_scope(tmp_path):
    # Official periodic neighbor-list backends can evaluate replicated images in
    # primitive cells.  They still advertise a cutoff for provenance, but MAPLE
    # must not reject them solely because that radius exceeds half the shortest
    # lattice vector.
    calc = wrap_ase_calculator(_lj(rc=2.0), pbc_md_supported=True, neighbor_cutoff_A=6.0)
    calc.maple_requires_single_image_mic = False
    calc.maple_periodic_neighborlist_multi_image_safe = True

    sim = NVE(output=str(tmp_path / "multi_image.out"), atoms=_pbc_atoms(calc),
              paras={"steps": 0, "verbose": 0})

    assert sim is not None


# ── provenance: contract + policy recorded ──────────────────────────────────

def test_run_context_records_unit_contract_and_cutoff_policy():
    atoms = _pbc_atoms(
        wrap_ase_calculator(_lj(rc=2.0), pbc_md_supported=True, neighbor_cutoff_A=2.0)
    )
    dof = resolve_md_dof_policy(atoms, NVEParams(), "nve")

    ctx = build_run_context(params=NVEParams(), dof_policy=dof, ensemble="nve")
    assert ctx["unit_contract"] == {"energy": "Ha", "force": "Ha/A", "stress": "eV/A^3"}
    assert ctx["cutoff_policy"]["allow_unknown_cutoff"] is False
    assert "single-image MIC" in ctx["cutoff_policy"]["scope"]

    ctx_override = build_run_context(
        params=NVEParams(allow_unknown_cutoff=True), dof_policy=dof, ensemble="nve"
    )
    assert ctx_override["cutoff_policy"]["allow_unknown_cutoff"] is True


def test_calculator_provenance_records_declared_units():
    calc = wrap_ase_calculator(_lj(rc=2.0), pbc_md_supported=True, neighbor_cutoff_A=2.0)
    caps = collect_calculator_provenance(calc)["capabilities"]
    assert caps["energy_unit"] == "Ha"
    assert caps["force_unit"] == "Ha/A"
    assert caps["neighbor_cutoff_A"] == 2.0
    assert caps["requires_single_image_mic"] is True
    assert caps["periodic_neighborlist_multi_image_safe"] is False
