import numpy as np
import pytest
from ase.io import read
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

import maple.function.dispatcher.md.ensemble.npt as npt_module
from maple.function.calculator._ase_unit_contract import ASE_STRESS_UNIT
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

    d_epsilon = 0.5 * 1.0 / 10.0 * (2.0 - 1.0)
    mu = np.exp(d_epsilon / 3.0)
    assert pressure == pytest.approx(2.0)
    assert atoms.get_volume() == pytest.approx(initial_volume * np.exp(d_epsilon))
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

    kT_ev = temperature * KELVIN_TO_HARTREE * HARTREE_TO_EV
    beta_ang3_per_ev = compressibility * EV_PER_ANG3_TO_BAR
    d_epsilon = np.sqrt(
        2.0 * kT_ev * beta_ang3_per_ev * (timestep / tau_p) / initial_volume
    )
    mu = np.exp(d_epsilon / 3.0)
    assert atoms.get_volume() == pytest.approx(initial_volume * np.exp(d_epsilon))
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

    mu = np.exp((0.5 * 1.0 / 10.0 * (2.0 - 1.0)) / 3.0)
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


def test_npt_vrescale_thermostat_receives_full_step_velocity(tmp_path):
    force = np.array([[1.0, 0.0, 0.0]])
    atoms = _periodic_atoms(StressCalculator(np.zeros(6), forces=force))
    atoms.arrays["velocities"] = np.zeros((1, 3))
    npt = NPT(
        output=str(tmp_path / "npt.out"),
        atoms=atoms,
        paras={
            "steps": 1,
            "timestep": 0.1,
            "thermostat": "v-rescale",
            "barostat": "berendsen",
            "init_velocities": False,
            "remove_com_every": 0,
            "verbose": 0,
            "log_every": 999,
            "traj_every": 999,
            "rst_every": 0,
        },
    )
    seen = {}

    def thermostat_apply(velocities):
        seen["velocities"] = velocities.copy()
        return velocities, 0.0

    npt.thermostat.apply = thermostat_apply
    npt.barostat.apply = lambda velocities, pressure_velocities=None: (0.0, velocities)

    npt.run()

    mass = atoms.get_masses()[0] * AMU_TO_AU
    expected = force * HA_PER_ANG_TO_AU / mass * (0.1 * FS_TO_AU)
    np.testing.assert_allclose(seen["velocities"], expected)


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
    atoms.arrays["velocities"] = np.zeros((1, 3))
    npt = NPT(
        output=str(tmp_path / "npt.out"),
        atoms=atoms,
        paras={
            "steps": 1,
            "thermostat": "v-rescale",
            "barostat": "berendsen",
            "init_velocities": False,
            "remove_com_every": 0,
            "verbose": 0,
            "log_every": 999,
            "traj_every": 999,
            "rst_every": 0,
        },
    )
    calls = []

    def apply_barostat(velocities, pressure_velocities=None):
        calls.append(velocities.copy())
        atoms.set_cell(atoms.get_cell() * 2.0, scale_atoms=True)  # volume 1000 -> 8000
        return 123.0, velocities

    post_eval_volumes = []
    real_pressure = npt_module.compute_instantaneous_pressure

    def record_pressure(a, v):
        post_eval_volumes.append(a.get_volume())
        return real_pressure(a, v)

    records = {}
    original_log_step = npt.logger.log_step

    def log_step_spy(**kwargs):
        records.update(kwargs)
        return original_log_step(**kwargs)

    npt.barostat.apply = apply_barostat
    npt.logger.log_step = log_step_spy
    monkeypatch.setattr(npt_module, "compute_instantaneous_pressure", record_pressure)

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
    real_pressure = npt_module.compute_instantaneous_pressure

    def record_pressure(a, v):
        seen["pressure_velocity"] = v.copy()
        return real_pressure(a, v)

    monkeypatch.setattr(npt_module, "compute_instantaneous_pressure", record_pressure)

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
    atoms.arrays["velocities"] = np.zeros((1, 3))
    first = NPT(
        output=str(tmp_path / "first.out"),
        atoms=atoms,
        paras={
            "steps": 1,
            "thermostat": "v-rescale",
            "barostat": "berendsen",
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
            "barostat": "berendsen",
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
