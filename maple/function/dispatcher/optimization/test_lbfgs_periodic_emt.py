import numpy as np
import pytest

ase_emt = pytest.importorskip("ase.calculators.emt")

from ase.build import bulk
from ase.calculators.emt import EMT
from ase.optimize import LBFGS as ASELBFGS

from maple.function.dispatcher.optimization.algorithm.LBFGS import LBFGS as MapleLBFGS


def _periodic_cu(rng_seed=0):
    atoms = bulk("Cu", "fcc", a=3.61).repeat((2, 2, 2))  # 32 atoms, periodic
    rng = np.random.default_rng(rng_seed)
    atoms.set_positions(
        atoms.get_positions() + 0.05 * rng.standard_normal((len(atoms), 3))
    )
    return atoms


def test_maple_lbfgs_matches_ase_lbfgs_on_periodic_emt(tmp_path):
    ref = _periodic_cu()
    ref.calc = EMT()
    ASELBFGS(ref, logfile=None).run(fmax=0.01, steps=200)

    trial = _periodic_cu()
    trial.calc = EMT()
    # Stamp convergence thresholds the way Dispatcher.set_throshould does for level=medium
    for attr, val in [
        ("f_max_th", 0.00285),
        ("f_rms_th", 0.00190),
        ("dp_max_th", 0.00315),
        ("dp_rms_th", 0.00210),
    ]:
        setattr(trial, attr, val)

    job = MapleLBFGS(
        atoms=trial,
        output=str(tmp_path / "opt.out"),
        paras={"max_iter": 200, "verbose": 0},
    )
    job.run()

    e_ref = ref.get_potential_energy()
    e_maple = trial.get_potential_energy()
    assert abs(e_ref - e_maple) < 1e-3, f"energy mismatch: ref={e_ref}, maple={e_maple}"

    max_pos_diff = float(np.abs(ref.get_positions() - trial.get_positions()).max())
    assert max_pos_diff < 5e-2, f"position mismatch: max_diff={max_pos_diff}"

    assert all(trial.pbc), "pbc must survive optimization"
    assert np.allclose(trial.get_cell(), ref.get_cell()), "cell must round-trip"
