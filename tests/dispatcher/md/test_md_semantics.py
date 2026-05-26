"""WS0 — MD physics-semantics audit.

Covers the operator-aware DOF resolver (WS0-A), the MD constraints gate
(WS0-B), and the partial-PBC production policy (WS0-C).  The resolver is
unit-tested directly against the operator-aware degree-of-freedom table; the
gates are tested through real ensemble construction so the two-phase
validation wiring (G2) is exercised end to end.
"""

from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.constraints import FixAtoms, FixInternals

from maple.function.calculator._ase_unit_contract import (
    ASE_STRESS_UNIT,
    MAPLE_ENERGY_UNIT,
    MAPLE_FORCE_UNIT,
)

from maple.function.dispatcher.md.ensemble.nve import NVE
from maple.function.dispatcher.md.ensemble.npt import NPT
from maple.function.dispatcher.md.ensemble.nvt import NVT
from maple.function.dispatcher.md.semantics import (
    MDDOFPolicy,
    resolve_md_dof_policy,
    validate_md_semantics,
)
from maple.function.dispatcher.md.utils import (
    calculate_angular_momentum,
    calculate_momentum,
    calculate_temperature,
)


# ──────────────────────────────────────────────────────────────────────────
# Fixtures / helpers
# ──────────────────────────────────────────────────────────────────────────

class _FakeCalc(Calculator):
    """Minimal energy/forces calculator with declarable PBC capability."""

    implemented_properties = ["energy", "forces"]

    def __init__(self, pbc_capable: bool = True):
        super().__init__()
        self.maple_model_name = "fake-pbc" if pbc_capable else "fake-cluster"
        self.maple_pbc_md_supported = pbc_capable
        self.maple_stress_supported = False
        self.maple_energy_unit = MAPLE_ENERGY_UNIT
        self.maple_force_unit = MAPLE_FORCE_UNIT
        self.maple_neighbor_cutoff = 2.0  # < min-image radius of the test cells

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results["energy"] = 0.0
        self.results["forces"] = np.zeros((len(atoms), 3))


class _FakeStressCalc(_FakeCalc):
    implemented_properties = ["energy", "forces", "stress"]

    def __init__(self):
        super().__init__(pbc_capable=True)
        self.maple_stress_supported = True
        self.maple_stress_unit = ASE_STRESS_UNIT

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results["stress"] = np.zeros(6)


def _water(pbc: bool = False) -> Atoms:
    """A bent (non-linear) water molecule; periodic when requested."""
    atoms = Atoms(
        "OH2",
        positions=[[0.0, 0.0, 0.0], [0.757, 0.586, 0.0], [-0.757, 0.586, 0.0]],
    )
    if pbc:
        atoms.set_cell([12.0, 12.0, 12.0])
        atoms.set_pbc(True)
    return atoms


def _diatomic() -> Atoms:
    """A linear (2-atom) isolated molecule."""
    return Atoms("N2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.10]])


def _params(**kw) -> SimpleNamespace:
    base = dict(
        remove_com=True,
        remove_rotation=False,
        remove_angular=False,
        remove_com_every=0,
        remove_angular_every=0,
        thermostat="",
        barostat="",
        allow_partial_pbc=False,
    )
    base.update(kw)
    return SimpleNamespace(**base)


# ──────────────────────────────────────────────────────────────────────────
# WS0-A — operator-aware DOF resolution
# ──────────────────────────────────────────────────────────────────────────

def test_resolver_returns_frozen_policy():
    policy = resolve_md_dof_policy(_water(), _params(remove_angular=True), "nve")
    assert isinstance(policy, MDDOFPolicy)
    with pytest.raises(Exception):
        policy.runtime_n_dof = 999  # frozen dataclass


def test_nve_isolated_water_uses_3n_minus_6():
    # NVE deterministic VV conserves total momentum and angular momentum, so
    # an init projection of COM + rotation is permanent: 3N - 6 = 3.
    policy = resolve_md_dof_policy(_water(), _params(remove_angular=True), "nve")
    assert policy.init_n_dof == 3
    assert policy.runtime_n_dof == 3


def test_vrescale_nvt_water_matches_nve_3n_minus_6():
    # v-rescale is a global scalar (alpha * v): it cannot create new velocity
    # directions, so an init-projected COM/rotation mode stays at zero.
    policy = resolve_md_dof_policy(
        _water(), _params(remove_angular=True, thermostat="v-rescale"), "nvt"
    )
    assert policy.runtime_n_dof == 3


def test_langevin_nvt_water_keeps_all_3n_dof():
    # Per-atom OU noise re-excites COM and rotation, so init-only projection is
    # not a permanent constraint: runtime DOF = 3N = 9.  init basis is still
    # 3N - 6 because the initial draw was projected onto it.
    policy = resolve_md_dof_policy(
        _water(), _params(remove_angular=True, thermostat="langevin"), "nvt"
    )
    assert policy.init_n_dof == 3
    assert policy.runtime_n_dof == 9


def test_linear_diatomic_nve_uses_3n_minus_5():
    policy = resolve_md_dof_policy(_diatomic(), _params(remove_angular=True), "nve")
    assert policy.init_n_dof == 1   # 3*2 - 3 - 2
    assert policy.runtime_n_dof == 1


def test_intermittent_com_removal_does_not_subtract():
    # remove_com_every = 100 is drift control, not a permanent constraint.
    policy = resolve_md_dof_policy(
        _water(pbc=True),
        _params(remove_com_every=100, thermostat="langevin"),
        "nvt",
    )
    assert policy.runtime_n_dof == 9


def test_every_step_com_removal_subtracts_three():
    # remove_com_every = 1 zeroes COM every step ≈ permanent constraint subspace.
    policy = resolve_md_dof_policy(
        _water(pbc=True),
        _params(remove_com_every=1, thermostat="langevin"),
        "nvt",
    )
    assert policy.runtime_n_dof == 6


def test_vrescale_intermittent_com_still_subtracts_operator_aware():
    # Operator-aware (NOT operator-blind): under v-rescale (global scalar) an
    # init-projected COM stays at zero regardless of the intermittent removal
    # cadence, so it is subtracted.  This is the bare-NPT default case.
    policy = resolve_md_dof_policy(
        _water(pbc=True),
        _params(remove_com=True, remove_com_every=100, thermostat="v-rescale"),
        "npt",
    )
    assert policy.runtime_n_dof == 6   # 9 - 3 (COM stays at zero under v-rescale)


def test_langevin_vs_vrescale_intermittent_com_differ():
    # The same intermittent cadence yields different runtime DOF depending on the
    # operator: Langevin repopulates COM (no subtract), v-rescale does not.
    langevin = resolve_md_dof_policy(
        _water(pbc=True), _params(remove_com_every=100, thermostat="langevin"), "nvt"
    )
    vrescale = resolve_md_dof_policy(
        _water(pbc=True), _params(remove_com_every=100, thermostat="v-rescale"), "nvt"
    )
    assert langevin.runtime_n_dof == 9
    assert vrescale.runtime_n_dof == 6


def test_pbc_never_subtracts_rotation():
    # Even with remove_angular / remove_angular_every set, rotation is undefined
    # under PBC and must never be subtracted; only COM may be.
    policy = resolve_md_dof_policy(
        _water(pbc=True),
        _params(remove_angular=True, remove_angular_every=1, thermostat="v-rescale"),
        "nvt",
    )
    assert policy.runtime_n_dof == 6   # 9 - 3 (COM only), rotation untouched


def test_unprojected_isolated_nve_keeps_all_dof():
    # No init projection, no runtime removal, no re-exciting operator: every mode
    # carries fixed initial kinetic energy → 3N.
    policy = resolve_md_dof_policy(
        _water(), _params(remove_com=False, remove_angular=False), "nve"
    )
    assert policy.runtime_n_dof == 9


# ──────────────────────────────────────────────────────────────────────────
# WS0-B — constraints gate (hard reject, no escape hatch)
# ──────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "constraint",
    [
        FixAtoms(indices=[0]),
        FixInternals(bonds=[[1.10, [0, 1]]]),
    ],
    ids=["FixAtoms", "FixInternals"],
)
def test_validate_md_semantics_rejects_constraints(constraint):
    atoms = _diatomic()
    atoms.set_constraint(constraint)
    with pytest.raises(ValueError, match="constraint"):
        validate_md_semantics(atoms, _params(), "nve")


def test_nve_construction_rejects_constrained_atoms(tmp_path):
    atoms = _water()
    atoms.calc = _FakeCalc(pbc_capable=False)
    atoms.set_constraint(FixAtoms(indices=[0]))
    with pytest.raises(ValueError, match="constraint"):
        NVE(output=str(tmp_path / "nve.out"), atoms=atoms, paras={"steps": 0, "verbose": 0})


def test_constraint_rejection_message_points_at_roadmap(tmp_path):
    atoms = _water()
    atoms.calc = _FakeCalc(pbc_capable=False)
    atoms.set_constraint(FixInternals(bonds=[[1.0, [0, 1]]]))
    with pytest.raises(ValueError) as exc:
        NVT(output=str(tmp_path / "nvt.out"), atoms=atoms, paras={"steps": 0, "verbose": 0})
    message = str(exc.value).lower()
    assert "constraint" in message
    assert "not" in message and ("support" in message or "implement" in message)


def test_unconstrained_md_is_unaffected(tmp_path):
    atoms = _water()
    atoms.calc = _FakeCalc(pbc_capable=False)
    NVE(output=str(tmp_path / "nve.out"), atoms=atoms, paras={"steps": 0, "verbose": 0})


# ──────────────────────────────────────────────────────────────────────────
# WS0-C — partial-PBC production policy
# ──────────────────────────────────────────────────────────────────────────

def _slab(calc_pbc_capable: bool = True) -> Atoms:
    atoms = Atoms("He", positions=[[0.0, 0.0, 0.0]], cell=[10.0, 10.0, 30.0], pbc=[True, True, False])
    atoms.calc = _FakeCalc(pbc_capable=calc_pbc_capable)
    return atoms


@pytest.mark.parametrize("ensemble_cls", [NVE, NVT])
def test_partial_pbc_rejected_by_default(ensemble_cls, tmp_path):
    atoms = _slab()
    with pytest.raises(ValueError, match="partial"):
        ensemble_cls(
            output=str(tmp_path / "md.out"),
            atoms=atoms,
            paras={"steps": 0, "verbose": 0, "remove_com_every": 0},
        )


def test_partial_pbc_allowed_with_flag_emits_experimental_banner(tmp_path):
    atoms = _slab()
    out = str(tmp_path / "nve.out")
    NVE(
        output=out,
        atoms=atoms,
        paras={"steps": 0, "verbose": 0, "remove_com_every": 0, "allow_partial_pbc": True},
    )
    text = open(out).read()
    assert "EXPERIMENTAL" in text
    assert "NOT PRODUCTION VALIDATED" in text


def test_full_pbc_is_not_flagged_partial(tmp_path):
    atoms = Atoms("He", positions=[[0.0, 0.0, 0.0]], cell=[10.0, 10.0, 10.0], pbc=True)
    atoms.calc = _FakeCalc(pbc_capable=True)
    out = tmp_path / "nve.out"
    NVE(output=str(out), atoms=atoms, paras={"steps": 0, "verbose": 0, "remove_com_every": 0})
    # No advisory is logged for full 3-D PBC; the .out may not be created at all.
    assert "EXPERIMENTAL" not in (out.read_text() if out.exists() else "")


# ──────────────────────────────────────────────────────────────────────────
# WS0-A consumed by WS1 — ensemble-level DOF behavior (the headline fix)
# ──────────────────────────────────────────────────────────────────────────

def test_nve_bare_water_init_temperature_uses_active_subspace(tmp_path):
    # Bare NVE on isolated water now projects COM + rotation (dataclass default
    # remove_angular=True, remove_com_every=0) and rescales the initial draw to
    # the *active* 3N-6 = 3 DOF, not the 3N = 9 runtime count.  The latter is the
    # latent 3x over-heating bug the defaults flip would otherwise activate.
    atoms = _water()
    atoms.calc = _FakeCalc(pbc_capable=False)
    nve = NVE(
        output=str(tmp_path / "nve.out"),
        atoms=atoms,
        paras={"steps": 0, "verbose": 0, "random_seed": 1234},
    )
    assert nve._dof_policy.init_n_dof == 3
    assert nve._dof_policy.runtime_n_dof == 3
    v = nve._initialize_velocities()
    # Energy distributed over the 3 active DOF == target; over 9 it would be 100 K.
    assert calculate_temperature(atoms, v, n_dof=3) == pytest.approx(300.0, rel=1e-6)
    assert calculate_temperature(atoms, v, n_dof=9) == pytest.approx(100.0, rel=1e-6)
    # COM and rotation were actually projected out.
    np.testing.assert_allclose(calculate_momentum(atoms, v), 0.0, atol=1e-10)
    np.testing.assert_allclose(calculate_angular_momentum(atoms, v), 0.0, atol=1e-8)


def test_vrescale_nvt_water_runtime_equals_init_basis(tmp_path):
    atoms = _water()
    atoms.calc = _FakeCalc(pbc_capable=False)
    nvt = NVT(
        output=str(tmp_path / "nvt.out"),
        atoms=atoms,
        paras={
            "steps": 0, "verbose": 0, "thermostat": "v-rescale", "tau_t": 100.0,
            "remove_angular": True, "remove_com_every": 0, "random_seed": 3,
        },
    )
    assert nvt._dof_policy.init_n_dof == 3
    assert nvt._dof_policy.runtime_n_dof == 3   # v-rescale keeps projected modes at zero
    v = nvt._initialize_velocities()
    assert calculate_temperature(atoms, v, n_dof=3) == pytest.approx(300.0, rel=1e-6)


def test_langevin_nvt_water_init_runtime_basis_differ(tmp_path):
    # Langevin re-excites the init-projected COM/rotation: init basis 3, runtime 9.
    # The initial draw is at target in the init basis; over the runtime basis it
    # reads (3/9)*target at t=0 (a basis difference, not over-heating).
    atoms = _water()
    atoms.calc = _FakeCalc(pbc_capable=False)
    nvt = NVT(
        output=str(tmp_path / "nvt.out"),
        atoms=atoms,
        paras={
            "steps": 0, "verbose": 0, "thermostat": "langevin",
            "remove_angular": True, "remove_com_every": 0, "random_seed": 7,
        },
    )
    assert nvt._dof_policy.init_n_dof == 3
    assert nvt._dof_policy.runtime_n_dof == 9
    v = nvt._initialize_velocities()
    assert calculate_temperature(atoms, v, n_dof=3) == pytest.approx(300.0, rel=1e-6)
    assert calculate_temperature(atoms, v, n_dof=9) == pytest.approx(100.0, rel=1e-6)


def test_init_true_input_velocities_are_conditioned_for_pbc_nvt(tmp_path):
    atoms = _water(pbc=True)
    atoms.calc = _FakeCalc(pbc_capable=True)
    raw = np.array([
        [0.012, -0.004, 0.003],
        [0.006, 0.009, -0.002],
        [-0.005, 0.003, 0.007],
    ]) + np.array([0.02, -0.01, 0.004])
    atoms.arrays["velocities"] = raw.copy()

    nvt = NVT(
        output=str(tmp_path / "nvt.out"),
        atoms=atoms,
        paras={
            "steps": 0,
            "verbose": 0,
            "thermostat": "v-rescale",
            "remove_com_every": 0,
        },
    )
    nvt._run_simulation = lambda velocities, velocity_representation, **kwargs: (
        velocities,
        velocity_representation,
    )
    nvt.run()
    v = atoms.arrays["velocities"]

    assert nvt._dof_policy.init_n_dof == 6
    np.testing.assert_allclose(calculate_momentum(atoms, v), 0.0, atol=1e-10)
    assert calculate_temperature(atoms, v, n_dof=6) == pytest.approx(300.0, rel=1e-6)
    assert "conditioned as the initialization state" in (tmp_path / "nvt.out").read_text()


def test_init_true_input_velocities_are_conditioned_for_pbc_npt(tmp_path):
    atoms = _water(pbc=True)
    atoms.calc = _FakeStressCalc()
    raw = np.array([
        [0.010, 0.006, -0.001],
        [-0.002, 0.005, 0.004],
        [0.004, -0.007, 0.003],
    ]) + np.array([-0.015, 0.012, 0.002])
    atoms.arrays["velocities"] = raw.copy()

    npt = NPT(
        output=str(tmp_path / "npt.out"),
        atoms=atoms,
        paras={
            "steps": 0,
            "verbose": 0,
            "thermostat": "v-rescale",
            "barostat": "c-rescale",
            "remove_com_every": 0,
        },
    )
    npt._run_simulation = lambda velocities, velocity_representation, **kwargs: (
        velocities,
        velocity_representation,
    )
    npt.run()
    v = atoms.arrays["velocities"]

    assert npt._dof_policy.init_n_dof == 6
    np.testing.assert_allclose(calculate_momentum(atoms, v), 0.0, atol=1e-10)
    assert calculate_temperature(atoms, v, n_dof=6) == pytest.approx(300.0, rel=1e-6)


def test_init_true_input_velocities_are_conditioned_for_isolated_angular_nve(tmp_path):
    atoms = _water()
    atoms.calc = _FakeCalc(pbc_capable=False)
    atoms.arrays["velocities"] = np.array([
        [0.015, -0.020, 0.010],
        [0.040, 0.010, -0.030],
        [-0.010, 0.030, 0.020],
    ])

    nve = NVE(
        output=str(tmp_path / "nve.out"),
        atoms=atoms,
        paras={"steps": 0, "verbose": 0},
    )
    nve._run_simulation = lambda velocities, **kwargs: velocities
    nve.run()
    v = atoms.arrays["velocities"]

    assert nve._dof_policy.init_n_dof == 3
    np.testing.assert_allclose(calculate_momentum(atoms, v), 0.0, atol=1e-10)
    np.testing.assert_allclose(calculate_angular_momentum(atoms, v), 0.0, atol=1e-8)
    assert calculate_temperature(atoms, v, n_dof=3) == pytest.approx(300.0, rel=1e-6)


def test_init_false_input_velocities_keep_unprojected_dof_semantics(tmp_path):
    atoms = _water(pbc=True)
    atoms.calc = _FakeCalc(pbc_capable=True)
    raw = np.tile(np.array([0.01, -0.02, 0.03]), (len(atoms), 1))
    atoms.arrays["velocities"] = raw.copy()

    nvt = NVT(
        output=str(tmp_path / "nvt.out"),
        atoms=atoms,
        paras={
            "steps": 0,
            "verbose": 0,
            "thermostat": "v-rescale",
            "init_velocities": False,
            "remove_com_every": 0,
        },
    )
    assert nvt._dof_policy.init_n_dof == 9
    assert nvt._dof_policy.runtime_n_dof == 9

    nvt._run_simulation = lambda velocities, velocity_representation, **kwargs: (
        velocities,
        velocity_representation,
    )
    nvt.run()

    np.testing.assert_allclose(atoms.arrays["velocities"], raw)
    assert np.linalg.norm(calculate_momentum(atoms, atoms.arrays["velocities"])) > 0.0
    text = (tmp_path / "nvt.out").read_text()
    assert "init_velocities=False consumes input velocities as provided" in text
