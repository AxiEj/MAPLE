import numpy as np
import pytest
from ase.io import read
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

import maple.function.dispatcher.md.ensemble.npt as npt_module
import maple.function.dispatcher.md.evaluator as evaluator_module
from maple.function.calculator._ase_unit_contract import (
    ASE_STRESS_UNIT,
    MAPLE_ENERGY_UNIT,
    MAPLE_FORCE_UNIT,
)
from maple.function.dispatcher.md.barostat.berendsen import BerendsenBarostat
from maple.function.dispatcher.md.barostat.crescale import CRescaleBarostat
from maple.function.dispatcher.md.ensemble.npt import NPT
from maple.function.dispatcher.md.ensemble.nve import NVE
from maple.function.dispatcher.md.ensemble.nvt import NVT
from maple.function.dispatcher.md.integrator.velocity_verlet import VelocityVerlet
from maple.function.dispatcher.md.logger import MDLogger
from maple.function.dispatcher.md.rst_io import read_rst
from maple.function.dispatcher.md.utils import (
    AMU_TO_AU,
    BOHR_TO_ANGSTROM,
    EV_PER_ANG3_TO_BAR,
    FS_TO_AU,
    HA_PER_ANG_TO_AU,
    HARTREE_TO_EV,
    IMAGE_FLAGS_ARRAY,
    KELVIN_TO_HARTREE,
    VELOCITY_REPR_LFMIDDLE_CARRIED,
    VELOCITY_REPR_STANDARD,
    compute_instantaneous_pressure,
    ensure_image_flags,
    get_atoms_velocity_representation,
    get_unwrapped_positions,
)


class EnergyForcesCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, *, pbc_capable: bool, stress_capable: bool = False, forces=None):
        super().__init__()
        self.maple_model_name = "fake-pbc" if pbc_capable else "fake-cluster"
        self.maple_pbc_md_supported = pbc_capable
        self.maple_stress_supported = stress_capable
        self.maple_energy_unit = MAPLE_ENERGY_UNIT
        self.maple_force_unit = MAPLE_FORCE_UNIT
        # Below the min-image radius of the smallest test cell here: the image-flag /
        # restart tests use a 2 Å micro-cell (radius 1.0 Å) to force boundary crossings.
        self.maple_neighbor_cutoff = 0.5
        if stress_capable:
            self.maple_stress_unit = ASE_STRESS_UNIT
        self._forces = forces
        self.force_call_volumes = []

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        volume = atoms.get_volume() if atoms is not None and any(atoms.pbc) else None
        self.force_call_volumes.append(volume)
        self.results["energy"] = 0.0
        if self._forces is None:
            self.results["forces"] = np.zeros((len(atoms), 3))
        else:
            self.results["forces"] = np.asarray(self._forces, dtype=float).copy()


class StressCalculator(EnergyForcesCalculator):
    implemented_properties = ["energy", "forces", "stress"]

    def __init__(self, stress, forces=None):
        super().__init__(pbc_capable=True, stress_capable=True, forces=forces)
        self._stress = np.asarray(stress, dtype=float)
        self.maple_stress_unit = ASE_STRESS_UNIT

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results["stress"] = self._stress.copy()


class MissingStressUnitCalculator(StressCalculator):
    def __init__(self, stress, forces=None):
        super().__init__(stress, forces=forces)
        del self.maple_stress_unit


def _periodic_atoms(calc: Calculator) -> Atoms:
    atoms = Atoms(
        "He",
        positions=[[0.0, 0.0, 0.0]],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )
    atoms.calc = calc
    return atoms


def _velocity_for_displacement(displacement_angstrom, dt_fs: float) -> np.ndarray:
    displacement = np.asarray(displacement_angstrom, dtype=float)
    return displacement / (dt_fs * FS_TO_AU * BOHR_TO_ANGSTROM)


class UnitNormalRNG:
    def standard_normal(self):
        return 1.0


def test_npt_rejects_missing_stress_at_startup(tmp_path):
    atoms = _periodic_atoms(EnergyForcesCalculator(pbc_capable=True, stress_capable=True))

    with pytest.raises(ValueError, match="stress tensor"):
        NPT(output=str(tmp_path / "npt.out"), atoms=atoms, paras={"steps": 0, "verbose": 0})


@pytest.mark.parametrize(
    "stress",
    [
        [np.nan, 0.0, 0.0, 0.0, 0.0, 0.0],
        [np.inf, 0.0, 0.0, 0.0, 0.0, 0.0],
    ],
)
def test_npt_rejects_nonfinite_stress_at_startup(stress, tmp_path):
    atoms = _periodic_atoms(StressCalculator(stress))

    with pytest.raises(ValueError, match="non-finite stress"):
        NPT(output=str(tmp_path / "npt.out"), atoms=atoms, paras={"steps": 0, "verbose": 0})


def test_npt_accepts_finite_stress_at_startup(tmp_path):
    atoms = _periodic_atoms(StressCalculator(np.zeros(6)))

    NPT(output=str(tmp_path / "npt.out"), atoms=atoms, paras={"steps": 0, "verbose": 0})


def test_npt_rejects_partial_pbc_even_with_stress(tmp_path):
    atoms = Atoms(
        "He",
        positions=[[0.0, 0.0, 0.0]],
        cell=[10.0, 10.0, 30.0],
        pbc=[True, True, False],
    )
    atoms.calc = StressCalculator(np.zeros(6))

    with pytest.raises(ValueError, match="full three-dimensional PBC"):
        NPT(output=str(tmp_path / "npt.out"), atoms=atoms, paras={"steps": 0, "verbose": 0})


def test_pressure_sign_follows_ase_stress_convention():
    velocities = np.zeros((1, 3))

    compressed = _periodic_atoms(StressCalculator([-1.0, -1.0, -1.0, 0.0, 0.0, 0.0]))
    stretched = _periodic_atoms(StressCalculator([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]))

    assert compute_instantaneous_pressure(compressed, velocities) == pytest.approx(EV_PER_ANG3_TO_BAR)
    assert compute_instantaneous_pressure(stretched, velocities) == pytest.approx(-EV_PER_ANG3_TO_BAR)


def test_berendsen_barostat_returns_pressure_and_original_velocity():
    atoms = _periodic_atoms(StressCalculator(np.zeros(6)))
    velocities = np.array([[0.01, -0.02, 0.03]])
    original = velocities.copy()
    barostat = BerendsenBarostat(
        atoms,
        pressure=0.0,
        tau_p=100.0,
        timestep=1.0,
        compressibility=0.0,
    )

    pressure, returned = barostat.apply(velocities)

    assert pressure == pytest.approx(compute_instantaneous_pressure(atoms, velocities))
    assert returned is velocities
    np.testing.assert_allclose(velocities, original)


def test_kinetic_pressure_can_exclude_com_motion_for_projected_dof_policy():
    """Full kinetic pressure remains available, but NPT production paths can
    switch to K-K_cm when the resolved DOF policy treats COM translation as a
    projected/constrained mode.  A pure COM boost must not change that active
    pressure subspace."""
    atoms = Atoms("Ar2", positions=[[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
                  cell=np.diag([6.0, 6.0, 6.0]), pbc=True)
    atoms.calc = StressCalculator(np.zeros(6))   # zero virial -> purely kinetic pressure

    v_internal = np.array([[0.01, 0.0, 0.0], [-0.01, 0.0, 0.0]])  # equal mass -> zero COM
    v_com = np.array([0.004, -0.002, 0.001])
    v_boosted = v_internal + v_com

    p_internal = compute_instantaneous_pressure(atoms, v_internal)
    p_boosted = compute_instantaneous_pressure(atoms, v_boosted)
    p_com_only = compute_instantaneous_pressure(atoms, np.tile(v_com, (len(atoms), 1)))
    p_internal_active = compute_instantaneous_pressure(
        atoms, v_internal, exclude_com_kinetic=True
    )
    p_boosted_active = compute_instantaneous_pressure(
        atoms, v_boosted, exclude_com_kinetic=True
    )
    p_com_only_active = compute_instantaneous_pressure(
        atoms, np.tile(v_com, (len(atoms), 1)), exclude_com_kinetic=True
    )

    # The COM drift adds exactly its own 2*KE_com/(3V) to the kinetic pressure;
    # the v_internal·v_com cross term vanishes because v_internal is zero-COM.
    assert p_com_only > 0.0
    assert (p_boosted - p_internal) == pytest.approx(p_com_only, rel=1e-9, abs=1e-9)
    assert p_boosted_active == pytest.approx(p_internal_active, rel=1e-9, abs=1e-9)
    assert p_com_only_active == pytest.approx(0.0, abs=1e-12)


def test_crescale_barostat_volume_response_ignores_imposed_com_when_policy_excludes_it():
    atoms = Atoms("Ar2", positions=[[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
                  cell=np.diag([6.0, 6.0, 6.0]), pbc=True)
    atoms.calc = StressCalculator(np.zeros(6))
    boosted = atoms.copy()
    boosted.calc = StressCalculator(np.zeros(6))

    v_internal = np.array([[0.01, 0.0, 0.0], [-0.01, 0.0, 0.0]])
    v_boosted = v_internal + np.array([0.004, -0.002, 0.001])
    initial_volume = atoms.get_volume()

    kwargs = dict(
        pressure=0.0,
        temperature=0.0,
        tau_p=10.0,
        timestep=1.0,
        compressibility=0.5,
        exclude_com_kinetic=True,
    )
    barostat = CRescaleBarostat(atoms, rng=np.random.default_rng(1), **kwargs)
    boosted_barostat = CRescaleBarostat(boosted, rng=np.random.default_rng(1), **kwargs)

    pressure, returned = barostat.apply(v_internal)
    pressure_boosted, returned_boosted = boosted_barostat.apply(v_boosted)

    assert pressure_boosted == pytest.approx(pressure, rel=1e-9, abs=1e-9)
    assert boosted.get_volume() == pytest.approx(atoms.get_volume(), rel=1e-12, abs=1e-12)
    mu = (atoms.get_volume() / initial_volume) ** (1.0 / 3.0)
    np.testing.assert_allclose(
        returned_boosted - returned,
        (v_boosted - v_internal) / mu,
        rtol=1e-12,
        atol=1e-12,
    )


def test_crescale_barostat_returns_velocity_scaled_by_mu_without_mutating_input():
    atoms = _periodic_atoms(StressCalculator(np.zeros(6)))
    initial_volume = atoms.get_volume()
    velocities = np.array([[0.03, -0.06, 0.09]])
    original = velocities.copy()
    barostat = CRescaleBarostat(
        atoms,
        pressure=1.0,
        temperature=0.0,
        tau_p=10.0,
        timestep=1.0,
        compressibility=0.5,
        rng=np.random.default_rng(7),
    )
    barostat.get_pressure = lambda _velocities: 2.0

    pressure, returned = barostat.apply(velocities)

    # Reversible λ=√V step at T=0 (no noise, no Itô correction):
    #   λ_new/λ = 1 + (β·dt/2τ_P)(P_int − P_0);  V_new = V0·(λ_new/λ)²
    lam_ratio = 1.0 + (0.5 * 1.0 / (2.0 * 10.0)) * (2.0 - 1.0)
    vol_ratio = lam_ratio ** 2
    mu = vol_ratio ** (1.0 / 3.0)
    assert pressure == pytest.approx(2.0)
    assert atoms.get_volume() == pytest.approx(initial_volume * vol_ratio)
    np.testing.assert_allclose(velocities, original)
    np.testing.assert_allclose(returned, original / mu)


def test_crescale_stochastic_term_uses_inverse_pressure_units():
    atoms = _periodic_atoms(StressCalculator(np.zeros(6)))
    initial_volume = atoms.get_volume()
    velocities = np.zeros((1, 3))
    temperature = 300.0
    compressibility = 4.5e-5
    timestep = 1.0
    tau_p = 1000.0
    barostat = CRescaleBarostat(
        atoms,
        pressure=1.0,
        temperature=temperature,
        tau_p=tau_p,
        timestep=timestep,
        compressibility=compressibility,
        rng=UnitNormalRNG(),
    )
    barostat.get_pressure = lambda _velocities: 1.0

    _, returned = barostat.apply(velocities)

    # Reversible λ=√V step with P_int == P_0 and W == 1 (UnitNormalRNG): only the
    # Itô-correction drift k_BT/(2V) and the CONSTANT-amplitude noise survive.
    # The noise prefactor sqrt(k_BT·β·dt/2τ_P) carries no 1/√V — the property
    # that distinguishes the reversible √V form from the ε (log-volume) form.
    kT_ev = temperature * KELVIN_TO_HARTREE * HARTREE_TO_EV
    beta_ang3_per_ev = compressibility * EV_PER_ANG3_TO_BAR
    lam = np.sqrt(initial_volume)
    kT_over_2v_bar = (0.5 * kT_ev / initial_volume) * EV_PER_ANG3_TO_BAR
    d_lam_det = (compressibility * timestep / (2.0 * tau_p)) * lam * kT_over_2v_bar
    d_lam_stoch = np.sqrt(kT_ev * beta_ang3_per_ev * timestep / (2.0 * tau_p))
    vol_ratio = ((lam + d_lam_det + d_lam_stoch) / lam) ** 2
    mu = vol_ratio ** (1.0 / 3.0)
    assert atoms.get_volume() == pytest.approx(initial_volume * vol_ratio)
    np.testing.assert_allclose(returned, velocities / mu)


def test_crescale_np_stride_advances_full_barostat_interval():
    atoms = _periodic_atoms(StressCalculator(np.zeros(6)))
    initial_volume = atoms.get_volume()
    velocities = np.array([[0.03, 0.0, 0.0]])
    barostat = CRescaleBarostat(
        atoms,
        pressure=1.0,
        temperature=0.0,
        tau_p=10.0,
        timestep=1.0,
        compressibility=0.5,
        rng=np.random.default_rng(7),
    )
    barostat.get_pressure = lambda _velocities: 2.0

    _, returned = barostat.apply(velocities, timestep_multiplier=4)

    # Literature N_P propagation: the λ SDE advances over N_P·dt.  This must be
    # a four-step interval update, not a one-step update after simply skipping
    # three pressure-control steps.
    lam_ratio = 1.0 + (0.5 * 4.0 / (2.0 * 10.0)) * (2.0 - 1.0)
    vol_ratio = lam_ratio ** 2
    mu = vol_ratio ** (1.0 / 3.0)
    assert atoms.get_volume() == pytest.approx(initial_volume * vol_ratio)
    np.testing.assert_allclose(returned, velocities / mu)


@pytest.mark.parametrize(
    ("ensemble_class", "paras", "message"),
    [
        (NVE, {"timestep": 0.0}, "timestep"),
        (NVE, {"log_every": 0}, "log_every"),
        (NVT, {"temperature": 0.0}, "temperature"),
        (NVT, {"thermostat": "v-rescale", "tau_t": 0.0}, "tau_t"),
        (NVT, {"thermostat": "langevin", "friction": -1.0}, "friction"),
    ],
)
def test_md_parameter_validation_rejects_invalid_common_ranges(
    ensemble_class,
    paras,
    message,
    tmp_path,
):
    atoms = Atoms("He", positions=[[0.0, 0.0, 0.0]])
    atoms.calc = EnergyForcesCalculator(pbc_capable=False)
    full_paras = {"steps": 0, "verbose": 0, **paras}

    with pytest.raises(ValueError, match=message):
        ensemble_class(output=str(tmp_path / "md.out"), atoms=atoms, paras=full_paras)


@pytest.mark.parametrize(
    ("paras", "message"),
    [
        ({"tau_p": 0.0}, "tau_p"),
        ({"compressibility": -1.0}, "compressibility"),
        ({"compressibility": 0.0}, "compressibility"),
        ({"barostat_stride": 0}, "barostat_stride"),
    ],
)
def test_npt_parameter_validation_rejects_invalid_barostat_ranges(
    paras,
    message,
    tmp_path,
):
    atoms = _periodic_atoms(StressCalculator(np.zeros(6)))
    full_paras = {"steps": 0, "verbose": 0, **paras}

    with pytest.raises(ValueError, match=message):
        NPT(output=str(tmp_path / "npt.out"), atoms=atoms, paras=full_paras)


def test_npt_accepts_literature_np_alias_for_barostat_stride(tmp_path):
    atoms = _periodic_atoms(StressCalculator(np.zeros(6)))

    sim = NPT(
        output=str(tmp_path / "npt.out"),
        atoms=atoms,
        paras={"steps": 0, "verbose": 0, "barostat_np": 4},
    )

    assert sim.params.barostat_stride == 4


def test_npt_rejects_conflicting_barostat_stride_aliases(tmp_path):
    atoms = _periodic_atoms(StressCalculator(np.zeros(6)))

    with pytest.raises(ValueError, match="Conflicting C-rescale N_P aliases"):
        NPT(
            output=str(tmp_path / "npt.out"),
            atoms=atoms,
            paras={"steps": 0, "verbose": 0, "barostat_stride": 2, "barostat_np": 3},
        )


def test_pbc_image_flags_reconstruct_unwrapped_boundary_crossing(tmp_path):
    atoms = _periodic_atoms(StressCalculator(np.zeros(6)))
    atoms.set_cell([2.0, 2.0, 2.0])
    atoms.set_positions([[1.9, 0.0, 0.0]])
    displacement_a = 0.3
    dt_fs = 1.0
    velocities = np.array([[displacement_a / (dt_fs * FS_TO_AU * BOHR_TO_ANGSTROM), 0.0, 0.0]])

    VelocityVerlet(atoms, timestep=dt_fs).full_step_r(velocities)

    assert atoms.positions[0, 0] == pytest.approx(0.2)
    np.testing.assert_array_equal(atoms.arrays[IMAGE_FLAGS_ARRAY], np.array([[1, 0, 0]]))
    assert get_unwrapped_positions(atoms)[0, 0] == pytest.approx(2.2)

    logger = MDLogger(
        output_path=str(tmp_path / "md.out"),
        log_every=999,
        traj_every=1,
        verbose=0,
    )
    logger.start_simulation(
        ensemble="nve",
        timestep=dt_fs,
        n_steps=1,
        temperature=300.0,
        atoms=atoms,
    )
    logger.log_step(
        step=1,
        time=dt_fs,
        temperature=0.0,
        kinetic_energy=0.0,
        potential_energy=0.0,
        total_energy=0.0,
        atoms=atoms,
        velocities=np.zeros((1, 3)),
        rst_every=1,
    )
    logger.thermo_file.close()
    logger.traj_file.close()
    logger.unwrapped_traj_file.close()

    wrapped_text = (tmp_path / "md_md_traj.xyz").read_text()
    unwrapped_text = (tmp_path / "md_md_traj_unwrapped.xyz").read_text()
    assert "CoordinateMode=wrapped" in wrapped_text
    assert "CoordinateMode=unwrapped" in unwrapped_text
    assert "Lattice=" in wrapped_text
    assert "Properties=species:S:1:pos:R:3" in wrapped_text
    assert 'pbc="T T T"' in wrapped_text
    assert "He      0.20000000" in wrapped_text
    assert "He      2.20000000" in unwrapped_text
    state = read_rst(tmp_path / "md_md.rst")
    np.testing.assert_array_equal(state["image_flags"], np.array([[1, 0, 0]]))
    reread = read(str(tmp_path / "md_md_traj.xyz"))
    assert list(reread.pbc) == [True, True, True]
    np.testing.assert_allclose(reread.cell.lengths(), [2.0, 2.0, 2.0])


def test_pbc_image_flags_track_negative_boundary_crossing():
    atoms = _periodic_atoms(StressCalculator(np.zeros(6)))
    atoms.set_cell([2.0, 2.0, 2.0])
    atoms.set_positions([[0.1, 0.0, 0.0]])
    dt_fs = 1.0
    velocities = np.array([_velocity_for_displacement([-0.3, 0.0, 0.0], dt_fs)])

    VelocityVerlet(atoms, timestep=dt_fs).full_step_r(velocities)

    assert atoms.positions[0, 0] == pytest.approx(1.8)
    np.testing.assert_array_equal(atoms.arrays[IMAGE_FLAGS_ARRAY], np.array([[-1, 0, 0]]))
    assert get_unwrapped_positions(atoms)[0, 0] == pytest.approx(-0.2)


def test_pbc_image_flags_accumulate_multi_step_crossings():
    atoms = _periodic_atoms(StressCalculator(np.zeros(6)))
    atoms.set_cell([2.0, 2.0, 2.0])
    atoms.set_positions([[1.9, 0.0, 0.0]])
    dt_fs = 1.0
    velocities = np.array([_velocity_for_displacement([2.3, 0.0, 0.0], dt_fs)])
    integrator = VelocityVerlet(atoms, timestep=dt_fs)

    integrator.full_step_r(velocities)
    np.testing.assert_array_equal(atoms.arrays[IMAGE_FLAGS_ARRAY], np.array([[2, 0, 0]]))
    assert get_unwrapped_positions(atoms)[0, 0] == pytest.approx(4.2)

    integrator.full_step_r(velocities)
    np.testing.assert_array_equal(atoms.arrays[IMAGE_FLAGS_ARRAY], np.array([[3, 0, 0]]))
    assert atoms.positions[0, 0] == pytest.approx(0.5)
    assert get_unwrapped_positions(atoms)[0, 0] == pytest.approx(6.5)


def test_unwrapped_reconstruction_works_for_triclinic_cells():
    atoms = _periodic_atoms(StressCalculator(np.zeros(6)))
    cell = np.array(
        [
            [2.0, 0.0, 0.0],
            [0.4, 2.1, 0.0],
            [0.2, 0.3, 2.2],
        ]
    )
    start_scaled = np.array([0.9, 0.1, 0.2])
    delta_scaled = np.array([0.4, -0.3, 1.2])
    atoms.set_cell(cell)
    atoms.set_scaled_positions([start_scaled])
    dt_fs = 1.0
    displacement = delta_scaled @ cell
    velocities = np.array([_velocity_for_displacement(displacement, dt_fs)])

    VelocityVerlet(atoms, timestep=dt_fs).full_step_r(velocities)

    expected_scaled = start_scaled + delta_scaled
    expected_flags = np.floor(expected_scaled).astype(np.int64)
    expected_wrapped = expected_scaled - expected_flags
    np.testing.assert_array_equal(atoms.arrays[IMAGE_FLAGS_ARRAY], np.array([expected_flags]))
    np.testing.assert_allclose(atoms.positions[0], expected_wrapped @ cell, atol=1e-12)
    np.testing.assert_allclose(get_unwrapped_positions(atoms)[0], expected_scaled @ cell, atol=1e-12)


def test_barostat_cell_scaling_preserves_image_flags_and_affine_unwrapped_position():
    atoms = _periodic_atoms(StressCalculator(np.zeros(6)))
    atoms.set_cell([2.0, 2.0, 2.0])
    atoms.set_positions([[0.2, 0.0, 0.0]])
    ensure_image_flags(atoms)[:] = np.array([[1, 0, 0]])
    initial_unwrapped = get_unwrapped_positions(atoms).copy()
    barostat = CRescaleBarostat(
        atoms,
        pressure=1.0,
        temperature=0.0,
        tau_p=10.0,
        timestep=1.0,
        compressibility=0.5,
        rng=np.random.default_rng(7),
    )
    barostat.get_pressure = lambda _velocities: 2.0

    _, returned = barostat.apply(np.zeros((1, 3)))

    # Reversible λ=√V step at T=0: length scale μ = (λ_new/λ)^{2/3}, and the
    # affine unwrapped position scales by the same μ as the cell.
    lam_ratio = 1.0 + (0.5 * 1.0 / (2.0 * 10.0)) * (2.0 - 1.0)
    mu = lam_ratio ** (2.0 / 3.0)
    np.testing.assert_array_equal(atoms.arrays[IMAGE_FLAGS_ARRAY], np.array([[1, 0, 0]]))
    np.testing.assert_allclose(get_unwrapped_positions(atoms), initial_unwrapped * mu)
    np.testing.assert_allclose(returned, np.zeros((1, 3)))


def test_restart_continuation_preserves_image_flags_for_next_crossing(tmp_path):
    atoms = _periodic_atoms(EnergyForcesCalculator(pbc_capable=True))
    atoms.set_cell([2.0, 2.0, 2.0])
    atoms.set_positions([[1.9, 0.0, 0.0]])
    dt_fs = 1.0
    atoms.arrays["velocities"] = np.array([_velocity_for_displacement([1.3, 0.0, 0.0], dt_fs)])

    first = NVE(
        output=str(tmp_path / "first.out"),
        atoms=atoms,
        paras={
            "steps": 1,
            "timestep": dt_fs,
            "init_velocities": False,
            "remove_com_every": 0,
            "verbose": 0,
            "log_every": 999,
            "traj_every": 1,
            "rst_every": 1,
        },
    )
    first.run()
    np.testing.assert_array_equal(read_rst(tmp_path / "first_md.rst")["image_flags"], np.array([[1, 0, 0]]))

    restarted_atoms = _periodic_atoms(EnergyForcesCalculator(pbc_capable=True))
    second = NVE(
        output=str(tmp_path / "second.out"),
        atoms=restarted_atoms,
        paras={
            "steps": 1,
            "timestep": dt_fs,
            "load_state": True,
            "rst_file": str(tmp_path / "first_md.rst"),
            "init_velocities": False,
            "remove_com_every": 0,
            "verbose": 0,
            "log_every": 999,
            "traj_every": 1,
            "rst_every": 1,
        },
    )
    second.run()

    state = read_rst(tmp_path / "second_md.rst")
    np.testing.assert_array_equal(state["image_flags"], np.array([[2, 0, 0]]))
    assert state["positions"][0, 0] == pytest.approx(0.5)
    assert "He      4.50000000" in (tmp_path / "second_md_traj_unwrapped.xyz").read_text()


def test_pbc_dcd_restart_keeps_unwrapped_xyz_sidecar_open(tmp_path):
    atoms = _periodic_atoms(EnergyForcesCalculator(pbc_capable=True))
    atoms.set_cell([2.0, 2.0, 2.0])
    atoms.set_positions([[1.9, 0.0, 0.0]])
    dt_fs = 1.0
    atoms.arrays["velocities"] = np.array([_velocity_for_displacement([1.3, 0.0, 0.0], dt_fs)])
    output = str(tmp_path / "dcd.out")
    paras = {
        "steps": 1,
        "timestep": dt_fs,
        "traj_format": "dcd",
        "init_velocities": False,
        "remove_com_every": 0,
        "verbose": 0,
        "log_every": 999,
        "traj_every": 1,
        "rst_every": 1,
    }

    NVE(output=output, atoms=atoms, paras=paras).run()
    NVE(
        output=output,
        atoms=_periodic_atoms(EnergyForcesCalculator(pbc_capable=True)),
        paras={**paras, "steps": 2, "restart": True},
    ).run()

    assert (tmp_path / "dcd_md_traj.dcd").stat().st_size > 0
    unwrapped_text = (tmp_path / "dcd_md_traj_unwrapped.xyz").read_text()
    assert unwrapped_text.count("CoordinateMode=unwrapped") == 2
    assert "He      4.50000000" in unwrapped_text


def test_npt_rejects_calculator_with_wrong_stress_unit(tmp_path):
    atoms = _periodic_atoms(StressCalculator(np.zeros(6)))
    atoms.calc.maple_stress_unit = "GPa"

    with pytest.raises(ValueError, match="eV/A"):
        NPT(output=str(tmp_path / "npt.out"), atoms=atoms, paras={"steps": 0, "verbose": 0})


def test_npt_rejects_stress_capable_calculator_missing_stress_unit(tmp_path):
    atoms = _periodic_atoms(MissingStressUnitCalculator(np.zeros(6)))

    with pytest.raises(ValueError, match="maple_stress_unit"):
        NPT(output=str(tmp_path / "npt.out"), atoms=atoms, paras={"steps": 0, "verbose": 0})


def test_npt_berendsen_logs_equilibration_only_warning(tmp_path):
    atoms = _periodic_atoms(StressCalculator(np.zeros(6)))

    NPT(
        output=str(tmp_path / "npt.out"),
        atoms=atoms,
        paras={
            "steps": 0,
            "barostat": "berendsen",
            "allow_equilibration_only_barostat": True,
            "remove_com_every": 0,
            "verbose": 0,
        },
    )

    assert "barostat=berendsen is equilibration-only" in (tmp_path / "npt.out").read_text()


def test_pbc_runtime_com_warning_mentions_transport_analysis(tmp_path):
    atoms = _periodic_atoms(EnergyForcesCalculator(pbc_capable=True))

    NVT(
        output=str(tmp_path / "nvt.out"),
        atoms=atoms,
        paras={"steps": 0, "remove_com_every": 100, "verbose": 0},
    )

    text = (tmp_path / "nvt.out").read_text()
    assert "remove_com_every under PBC removes total momentum" in text
    assert "transport" in text


def test_npt_vrescale_thermostat_is_split_around_verlet(tmp_path):
    force = np.array([[1.0, 0.0, 0.0]])
    atoms = _periodic_atoms(StressCalculator(np.zeros(6), forces=force))
    initial_velocity = np.array([[1.0e-4, 0.0, 0.0]])
    atoms.arrays["velocities"] = initial_velocity.copy()
    npt = NPT(
        output=str(tmp_path / "npt.out"),
        atoms=atoms,
        paras={
            "steps": 1,
            "timestep": 0.1,
            "thermostat": "v-rescale",
            "barostat": "c-rescale",
            "init_velocities": False,
            "remove_com_every": 0,
            "verbose": 0,
            "log_every": 999,
            "traj_every": 999,
            "rst_every": 0,
        },
    )
    seen = []

    def thermostat_apply(velocities, timestep_fraction=1.0):
        seen.append((timestep_fraction, atoms.get_positions().copy(), velocities.copy()))
        return velocities, 0.0

    npt.thermostat.apply = thermostat_apply
    npt.barostat.apply = lambda velocities, pressure_velocities=None, timestep_multiplier=1.0: (0.0, velocities)

    npt.run()

    mass = atoms.get_masses()[0] * AMU_TO_AU
    expected = initial_velocity + force * HA_PER_ANG_TO_AU / mass * (0.1 * FS_TO_AU)
    assert [call[0] for call in seen] == [0.5, 0.5]
    np.testing.assert_allclose(seen[0][2], initial_velocity)
    np.testing.assert_allclose(seen[1][2], expected)


def test_npt_vrescale_crescale_barostat_runs_before_verlet(tmp_path):
    # Bernetti-Bussi reversible-Euler ordering propagates sqrt(V), scales the
    # cell/coordinates, refreshes forces, then performs split thermostat +
    # full Velocity Verlet + split thermostat.
    # The barostat must therefore see the pre-Verlet coordinates, while the
    # first thermostat half-step still sees the pre-Verlet coordinates/velocity
    # and the second sees the post-Verlet coordinates/velocity.
    force = np.array([[1.0, 0.0, 0.0]])
    atoms = _periodic_atoms(StressCalculator(np.zeros(6), forces=force))
    initial_position = atoms.get_positions().copy()
    initial_velocity = np.array([[1.0e-4, 0.0, 0.0]])
    atoms.arrays["velocities"] = initial_velocity.copy()
    npt = NPT(
        output=str(tmp_path / "npt.out"),
        atoms=atoms,
        paras={
            "steps": 1,
            "timestep": 0.1,
            "thermostat": "v-rescale",
            "barostat": "c-rescale",
            "init_velocities": False,
            "remove_com_every": 0,
            "verbose": 0,
            "log_every": 999,
            "traj_every": 999,
            "rst_every": 0,
        },
    )
    seen = {"thermostat": []}

    def barostat_apply(velocities, pressure_velocities=None, timestep_multiplier=1.0):
        seen["barostat_position"] = atoms.get_positions().copy()
        seen["barostat_velocity"] = velocities.copy()
        seen["timestep_multiplier"] = timestep_multiplier
        return 0.0, velocities

    def thermostat_apply(velocities, timestep_fraction=1.0):
        seen["thermostat"].append(
            (timestep_fraction, atoms.get_positions().copy(), velocities.copy())
        )
        return velocities, 0.0

    npt.barostat.apply = barostat_apply
    npt.thermostat.apply = thermostat_apply

    npt.run()

    mass = atoms.get_masses()[0] * AMU_TO_AU
    expected_velocity = initial_velocity + force * HA_PER_ANG_TO_AU / mass * (0.1 * FS_TO_AU)
    np.testing.assert_allclose(seen["barostat_position"], initial_position)
    np.testing.assert_allclose(seen["barostat_velocity"], initial_velocity)
    assert seen["timestep_multiplier"] == pytest.approx(1.0)
    assert [call[0] for call in seen["thermostat"]] == [0.5, 0.5]
    np.testing.assert_allclose(seen["thermostat"][0][1], initial_position)
    np.testing.assert_allclose(seen["thermostat"][0][2], initial_velocity)
    assert not np.allclose(seen["thermostat"][1][1], initial_position)
    np.testing.assert_allclose(seen["thermostat"][1][2], expected_velocity)


def test_npt_crescale_np_stride_is_full_interval_not_single_step_skip(tmp_path):
    atoms = _periodic_atoms(StressCalculator(np.zeros(6)))
    atoms.arrays["velocities"] = np.array([[1.0e-4, 0.0, 0.0]])
    npt = NPT(
        output=str(tmp_path / "npt.out"),
        atoms=atoms,
        paras={
            "steps": 5,
            "timestep": 0.1,
            "thermostat": "v-rescale",
            "barostat": "c-rescale",
            "barostat_stride": 3,
            "init_velocities": False,
            "remove_com_every": 0,
            "verbose": 0,
            "log_every": 999,
            "traj_every": 999,
            "rst_every": 0,
        },
    )
    calls = []

    def barostat_apply(velocities, pressure_velocities=None, timestep_multiplier=1.0):
        calls.append(timestep_multiplier)
        return 123.0, velocities

    npt.barostat.apply = barostat_apply
    npt.thermostat.apply = lambda velocities, timestep_fraction=1.0: (velocities, 0.0)

    npt.run()

    assert calls == [pytest.approx(3.0)]
    thermo = np.loadtxt(tmp_path / "npt_md_thermo.dat", comments="#")
    # Step 3 is the only scheduled N_P barostat step in a five-step run; all
    # other rows intentionally carry NaN pre-rescale diagnostics rather than a
    # fake one-step pressure update.
    assert thermo[:, 0].tolist() == [1, 2, 3, 4, 5]
    assert np.isnan(thermo[0, 8]) and np.isnan(thermo[1, 8])
    assert thermo[2, 8] == pytest.approx(123.0)
    assert np.isnan(thermo[3, 8]) and np.isnan(thermo[4, 8])


def test_npt_logs_post_rescale_primary_with_pre_rescale_diagnostic(
    monkeypatch,
    tmp_path,
):
    # WS2 correctness-first (HR-2/G6): the primary logged pressure/volume are the
    # post-rescale state (paired with the post-rescale T/KE/PE), evaluated with a
    # fresh stress at the post-rescale cell; the barostat's pre-rescale pair is
    # kept only as a labeled diagnostic.  We record (not forbid) the fresh
    # post-rescale pressure evaluation and the volume it sees.
    calc = StressCalculator(np.zeros(6))
    atoms = _periodic_atoms(calc)
    atoms.arrays["velocities"] = np.array([[1.0e-4, 0.0, 0.0]])
    npt = NPT(
        output=str(tmp_path / "npt.out"),
        atoms=atoms,
        paras={
            "steps": 1,
            "thermostat": "v-rescale",
            "barostat": "c-rescale",
            "init_velocities": False,
            "remove_com_every": 0,
            "verbose": 0,
            "log_every": 999,
            "traj_every": 999,
            "rst_every": 0,
        },
    )
    calls = []

    def apply_barostat(velocities, pressure_velocities=None, timestep_multiplier=1.0):
        calls.append(velocities.copy())
        atoms.set_cell(atoms.get_cell() * 2.0, scale_atoms=True)  # volume 1000 -> 8000
        return 123.0, velocities

    post_eval_volumes = []
    # WS2: NPT's post-rescale pressure now flows through evaluate_md_properties,
    # which calls compute_instantaneous_pressure; spy at the evaluator boundary.
    real_pressure = evaluator_module.compute_instantaneous_pressure

    def record_pressure(a, v, **kwargs):
        post_eval_volumes.append(a.get_volume())
        return real_pressure(a, v, **kwargs)

    records = {}
    original_log_step = npt.logger.log_step

    def log_step_spy(**kwargs):
        records.update(kwargs)
        return original_log_step(**kwargs)

    npt.barostat.apply = apply_barostat
    npt.thermostat.apply = lambda velocities, timestep_fraction=1.0: (np.zeros_like(velocities), 0.0)
    npt.logger.log_step = log_step_spy
    monkeypatch.setattr(evaluator_module, "compute_instantaneous_pressure", record_pressure)

    npt.run()

    assert len(calls) == 1
    # primary = post-rescale state
    assert records["volume"] == pytest.approx(8000.0)
    assert records["pressure"] == pytest.approx(0.0)   # zero velocity + zero stress
    # diagnostic = pre-rescale pair that drove the barostat decision
    assert records["pressure_pre"] == pytest.approx(123.0)
    assert records["volume_pre"] == pytest.approx(1000.0)
    # the fresh pressure was evaluated at the post-rescale cell, not the pre cell
    assert post_eval_volumes and post_eval_volumes[-1] == pytest.approx(8000.0)
    assert calc.force_call_volumes[-1] == pytest.approx(8000.0)
    thermo_text = (tmp_path / "npt_md_thermo.dat").read_text()
    assert "Press(bar)" in thermo_text and "Vol(A^3)" in thermo_text
    assert "Press_pre(bar)" in thermo_text and "Vol_pre(A^3)" in thermo_text


def test_npt_runtime_mic_guard_after_barostat_shrink(tmp_path):
    calc = StressCalculator(np.zeros(6))
    calc.maple_neighbor_cutoff = 6.0
    atoms = _periodic_atoms(calc)
    atoms.set_cell([14.0, 14.0, 14.0], scale_atoms=False)  # MIC radius 7 Å: startup passes.
    atoms.arrays["velocities"] = np.array([[1.0e-4, 0.0, 0.0]])
    npt = NPT(
        output=str(tmp_path / "npt.out"),
        atoms=atoms,
        paras={
            "steps": 1,
            "thermostat": "v-rescale",
            "barostat": "c-rescale",
            "init_velocities": False,
            "remove_com_every": 0,
            "verbose": 0,
            "log_every": 999,
            "traj_every": 999,
            "rst_every": 0,
        },
    )

    def shrink_below_minimum_image(velocities, pressure_velocities=None, timestep_multiplier=1.0):
        atoms.set_cell([10.0, 10.0, 10.0], scale_atoms=True)  # MIC radius 5 Å < cutoff 6 Å.
        return 0.0, velocities

    npt.barostat.apply = shrink_below_minimum_image
    npt.thermostat.apply = lambda velocities, timestep_fraction=1.0: (velocities, 0.0)

    with pytest.raises(ValueError, match="runtime cutoff/MIC guard.*step 1"):
        npt.run()

    text = (tmp_path / "npt.out").read_text()
    assert "ABORTED" in text
    assert not (tmp_path / "npt_md_manifest.json").exists()
    assert not any(
        volume is not None and np.isclose(volume, 1000.0)
        for volume in calc.force_call_volumes
    )


def test_npt_langevin_post_rescale_pressure_uses_synchronized_velocity(monkeypatch, tmp_path):
    # WS2 refinement: for LF-Middle Langevin NPT the post-rescale pressure kinetic
    # term must use the synchronized standard velocity (full increment), not the
    # half-step carried velocity, matching the pre-rescale decision and the
    # sync-corrected T/KE.
    force = np.array([[1.0, 0.0, 0.0]])
    atoms = _periodic_atoms(StressCalculator(np.zeros(6), forces=force))
    atoms.arrays["velocities"] = np.zeros((1, 3))
    timestep_fs = 0.1
    npt = NPT(
        output=str(tmp_path / "npt.out"),
        atoms=atoms,
        paras={
            "steps": 1,
            "timestep": timestep_fs,
            "thermostat": "langevin",
            "barostat": "c-rescale",
            "init_velocities": False,
            "remove_com_every": 0,
            "verbose": 0,
            "log_every": 999,
            "traj_every": 999,
            "rst_every": 0,
        },
    )
    npt.thermostat.apply = lambda velocities: velocities
    npt.barostat.apply = lambda velocities, pressure_velocities=None: (0.0, velocities)

    seen = {}
    # WS2: NPT's post-rescale pressure now flows through evaluate_md_properties,
    # which calls compute_instantaneous_pressure; spy at the evaluator boundary.
    real_pressure = evaluator_module.compute_instantaneous_pressure

    def record_pressure(a, v, **kwargs):
        seen["pressure_velocity"] = v.copy()
        return real_pressure(a, v, **kwargs)

    monkeypatch.setattr(evaluator_module, "compute_instantaneous_pressure", record_pressure)

    npt.run()

    mass = atoms.get_masses()[0] * AMU_TO_AU
    velocity_increment = force * HA_PER_ANG_TO_AU / mass * (timestep_fs * FS_TO_AU)
    # Synchronized standard velocity == full increment (carried half-step == 0.5x).
    np.testing.assert_allclose(seen["pressure_velocity"], velocity_increment)


def test_npt_langevin_barostat_pressure_uses_synchronized_standard_velocity(tmp_path):
    force = np.array([[1.0, 0.0, 0.0]])
    atoms = _periodic_atoms(StressCalculator(np.zeros(6), forces=force))
    atoms.arrays["velocities"] = np.zeros((1, 3))
    timestep_fs = 0.1
    npt = NPT(
        output=str(tmp_path / "npt.out"),
        atoms=atoms,
        paras={
            "steps": 1,
            "timestep": timestep_fs,
            "thermostat": "langevin",
            "barostat": "c-rescale",
            "init_velocities": False,
            "remove_com_every": 0,
            "verbose": 0,
            "log_every": 999,
            "traj_every": 999,
            "rst_every": 0,
        },
    )
    seen = {}

    npt.thermostat.apply = lambda velocities: velocities

    def barostat_apply(velocities, pressure_velocities=None):
        seen["state_velocities"] = velocities.copy()
        seen["pressure_velocities"] = pressure_velocities.copy()
        return 0.0, velocities

    npt.barostat.apply = barostat_apply

    npt.run()

    mass = atoms.get_masses()[0] * AMU_TO_AU
    velocity_increment = force * HA_PER_ANG_TO_AU / mass * (timestep_fs * FS_TO_AU)
    np.testing.assert_allclose(seen["state_velocities"], 0.5 * velocity_increment)
    np.testing.assert_allclose(seen["pressure_velocities"], velocity_increment)
    assert get_atoms_velocity_representation(npt.atoms) == VELOCITY_REPR_LFMIDDLE_CARRIED


def test_npt_vrescale_load_state_keeps_standard_velocity_representation(tmp_path):
    atoms = _periodic_atoms(StressCalculator(np.zeros(6)))
    atoms.arrays["velocities"] = np.array([[1.0e-4, 0.0, 0.0]])
    first = NPT(
        output=str(tmp_path / "first.out"),
        atoms=atoms,
        paras={
            "steps": 1,
            "thermostat": "v-rescale",
            "barostat": "c-rescale",
            "init_velocities": False,
            "remove_com_every": 0,
            "verbose": 0,
            "log_every": 999,
            "traj_every": 999,
            "rst_every": 1,
        },
    )
    first.run()
    assert get_atoms_velocity_representation(first.atoms) == VELOCITY_REPR_STANDARD

    loaded_atoms = _periodic_atoms(StressCalculator(np.zeros(6)))
    second = NPT(
        output=str(tmp_path / "second.out"),
        atoms=loaded_atoms,
        paras={
            "steps": 1,
            "load_state": True,
            "rst_file": str(tmp_path / "first_md.rst"),
            "thermostat": "v-rescale",
            "barostat": "c-rescale",
            "remove_com_every": 0,
            "verbose": 0,
            "log_every": 999,
            "traj_every": 999,
            "rst_every": 0,
        },
    )
    second.run()

    assert get_atoms_velocity_representation(second.atoms) == VELOCITY_REPR_STANDARD


@pytest.mark.parametrize("ensemble_cls", [NVE, NVT])
def test_nve_nvt_accept_periodic_pbc_capable_calculator(ensemble_cls, tmp_path):
    atoms = _periodic_atoms(EnergyForcesCalculator(pbc_capable=True))

    ensemble_cls(
        output=str(tmp_path / f"{ensemble_cls.__name__.lower()}.out"),
        atoms=atoms,
        paras={"steps": 0, "verbose": 0},
    )


@pytest.mark.parametrize("ensemble_cls", [NVE, NVT])
def test_nve_nvt_reject_periodic_non_pbc_calculator(ensemble_cls, tmp_path):
    atoms = _periodic_atoms(EnergyForcesCalculator(pbc_capable=False))

    with pytest.raises(ValueError, match=f"{ensemble_cls.__name__.upper()} with PBC"):
        ensemble_cls(
            output=str(tmp_path / f"{ensemble_cls.__name__.lower()}.out"),
            atoms=atoms,
            paras={"steps": 0, "verbose": 0},
        )
