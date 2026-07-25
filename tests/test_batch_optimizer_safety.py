import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from ase import Atoms

from maple.function.dispatcher.optimization.algorithm.blbfgs import BatchLBFGS


class _Molecules:
    def __init__(self, atoms, calc):
        self.multiatoms = atoms
        self.calc = calc


class _SyntheticBatchCalculator:
    def __init__(self, *, nonfinite=False, stationary=False):
        self.device = torch.device("cpu")
        self.nonfinite = nonfinite
        self.stationary = stationary
        self.coord = torch.zeros((0, 3), dtype=torch.float64)
        self._backup = None
        self._atoms = []

    def prepare(self, atoms_list, fixed_nmax=None):
        self._atoms = list(atoms_list)
        self.coord = torch.from_numpy(
            np.concatenate(
                [atoms.get_positions() for atoms in atoms_list],
                axis=0,
            )
        ).to(
            dtype=torch.float64,
        )
        self.nmax = fixed_nmax or max(3 * len(atoms) for atoms in atoms_list)

    def get_ef_gpu(self):
        batch_size = len(self._atoms)
        energy = torch.zeros(batch_size, dtype=torch.float64)
        force = torch.zeros((batch_size, self.nmax), dtype=torch.float64)
        start = 0
        for index, atoms in enumerate(self._atoms):
            stop = start + len(atoms)
            if not self.stationary:
                energy[index] = -self.coord[start:stop].sum()
                force[index, : 3 * len(atoms)] = 1.0
            start = stop
        if self.nonfinite:
            energy[0] = torch.nan
        return energy, force

    def backup_coords(self):
        self._backup = self.coord.clone()

    def restore_coords(self):
        if self._backup is not None:
            self.coord.copy_(self._backup)
            self._backup = None

    def step_cart_(self, step):
        start = 0
        for index, atoms in enumerate(self._atoms):
            stop = start + len(atoms)
            self.coord[start:stop].add_(
                step[index, : 3 * len(atoms)].reshape(-1, 3)
            )
            start = stop


class _QuadraticBatchCalculator(_SyntheticBatchCalculator):
    def get_ef_gpu(self):
        batch_size = len(self._atoms)
        energy = torch.zeros(batch_size, dtype=torch.float64)
        force = torch.zeros((batch_size, self.nmax), dtype=torch.float64)
        start = 0
        for index, atoms in enumerate(self._atoms):
            stop = start + len(atoms)
            positions = self.coord[start:stop]
            energy[index] = 500.0 * (positions * positions).sum()
            force[index, : 3 * len(atoms)] = (
                -1000.0 * positions
            ).reshape(-1)
            start = stop
        return energy, force


class _RaisingBatchCalculator(_SyntheticBatchCalculator):
    def get_ef_gpu(self):
        raise RuntimeError("synthetic backend failure")


class _RaisingNonfiniteCalculator(_SyntheticBatchCalculator):
    def get_ef_gpu(self):
        raise FloatingPointError("synthetic backend non-finite state")


class _TrialNonfiniteCalculator(_SyntheticBatchCalculator):
    def __init__(self):
        super().__init__()
        self.evaluations = 0

    def get_ef_gpu(self):
        self.evaluations += 1
        if self.evaluations > 1:
            raise FloatingPointError("synthetic trial non-finite state")
        return super().get_ef_gpu()


class BatchOptimizerSafetyTests(unittest.TestCase):
    def test_constructor_rejects_invalid_parameters(self):
        invalid = (
            {"memory": 0},
            {"memory": True},
            {"curvature": float("nan")},
            {"curvature": 0.0},
            {"maxstep": float("inf")},
            {"maxiter": 0},
            {"traj_every": 0},
        )
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                BatchLBFGS(output="unused.out", device="cpu", **kwargs)

    def test_run_rejects_invalid_structure_thresholds_before_output(self):
        atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
        atoms.f_max_th = float("nan")
        optimizer = BatchLBFGS(output="unused.out", device="cpu")
        with self.assertRaisesRegex(ValueError, "f_max_th"):
            optimizer.run(
                _Molecules([atoms], _SyntheticBatchCalculator())
            )
        self.assertIsNone(optimizer.log_fp)

    def test_nonfinite_initial_state_never_writes_production_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "job.out"
            optimizer = BatchLBFGS(
                output=str(output),
                device="cpu",
                maxiter=1,
            )
            molecules = _Molecules(
                [Atoms("H", positions=[[0.0, 0.0, 0.0]])],
                _SyntheticBatchCalculator(nonfinite=True),
            )

            with self.assertRaisesRegex(
                FloatingPointError, "initial evaluation"
            ):
                optimizer.run(molecules)

            self.assertFalse((Path(tmpdir) / "job_opt.xyz").exists())
            self.assertEqual(optimizer.statuses, ["failed_nonfinite"])
            self.assertIsNone(optimizer.log_fp)

    def test_nonfinite_status_is_limited_to_the_bad_structure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            optimizer = BatchLBFGS(
                output=str(Path(tmpdir) / "job.out"),
                device="cpu",
                maxiter=1,
            )
            molecules = _Molecules(
                [
                    Atoms("H", positions=[[0.0, 0.0, 0.0]]),
                    Atoms("H", positions=[[1.0, 0.0, 0.0]]),
                ],
                _SyntheticBatchCalculator(nonfinite=True),
            )
            with self.assertRaises(FloatingPointError):
                optimizer.run(molecules)
            self.assertEqual(
                optimizer.statuses,
                ["failed_nonfinite", "running"],
            )
            self.assertEqual(optimizer.failure_details[0]["index"], 0)
            self.assertIn("energies", optimizer.failure_details[0]["fields"])

    def test_backend_exception_sets_explicit_status(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            optimizer = BatchLBFGS(
                output=str(Path(tmpdir) / "job.out"),
                device="cpu",
            )
            molecules = _Molecules(
                [Atoms("H", positions=[[0.0, 0.0, 0.0]])],
                _RaisingBatchCalculator(),
            )
            with self.assertRaisesRegex(RuntimeError, "backend failure"):
                optimizer.run(molecules)
            self.assertEqual(optimizer.statuses, ["failed_backend"])
            self.assertEqual(
                optimizer.failure_details[0]["stage"],
                "initial evaluation",
            )

    def test_backend_nonfinite_exception_sets_terminal_status(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            optimizer = BatchLBFGS(
                output=str(Path(tmpdir) / "job.out"),
                device="cpu",
            )
            molecules = _Molecules(
                [Atoms("H", positions=[[0.0, 0.0, 0.0]])],
                _RaisingNonfiniteCalculator(),
            )
            with self.assertRaisesRegex(
                FloatingPointError, "backend non-finite"
            ):
                optimizer.run(molecules)
            self.assertEqual(optimizer.statuses, ["failed_nonfinite"])
            self.assertEqual(
                optimizer.failure_details[0]["fields"],
                ("backend",),
            )

    def test_trial_nonfinite_exception_restores_coordinates_and_status(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            optimizer = BatchLBFGS(
                output=str(Path(tmpdir) / "job.out"),
                device="cpu",
                maxiter=1,
            )
            atoms = Atoms("H", positions=[[0.25, 0.0, 0.0]])
            calc = _TrialNonfiniteCalculator()
            with self.assertRaisesRegex(
                FloatingPointError, "trial non-finite"
            ):
                optimizer.run(_Molecules([atoms], calc))
            np.testing.assert_allclose(
                calc.coord,
                [[0.25, 0.0, 0.0]],
            )
            self.assertEqual(optimizer.statuses, ["failed_nonfinite"])
            self.assertIn(
                "trial 0",
                optimizer.failure_details[0]["stage"],
            )

    def test_non_descent_direction_falls_back_per_structure(self):
        optimizer = BatchLBFGS(output="unused.out", device="cpu")
        gradient = torch.tensor([[2.0, -1.0], [1.0, 3.0]])
        proposed = torch.tensor([[2.0, -1.0], [-1.0, -3.0]])
        safeguarded = optimizer._safeguard_search_direction(
            proposed,
            gradient,
        )
        torch.testing.assert_close(
            safeguarded[0],
            -gradient[0] / optimizer.curvature,
        )
        torch.testing.assert_close(safeguarded[1], proposed[1])

    def test_nonfinite_search_direction_fails_instead_of_falling_back(self):
        optimizer = BatchLBFGS(output="unused.out", device="cpu")
        optimizer._orig_index = torch.tensor([0])
        optimizer.statuses = ["running"]
        with self.assertRaisesRegex(
            FloatingPointError, "raw L-BFGS search direction"
        ):
            optimizer._require_finite(
                "raw L-BFGS search direction",
                search_direction=torch.tensor([[float("nan"), 1.0]]),
            )
        self.assertEqual(optimizer.statuses, ["failed_nonfinite"])

    def test_armijo_backtracking_rejects_uphill_clipped_step(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "job.out"
            atoms = Atoms("H", positions=[[0.05, 0.0, 0.0]])
            initial_energy = 500.0 * 0.05**2
            optimizer = BatchLBFGS(
                output=str(output),
                device="cpu",
                curvature=1.0,
                maxstep=0.2,
                maxiter=1,
            )
            with self.assertRaisesRegex(RuntimeError, "unconverged"):
                optimizer.run(
                    _Molecules([atoms], _QuadraticBatchCalculator())
                )
            final_energy = 500.0 * float(np.sum(atoms.positions**2))
            self.assertLessEqual(final_energy, initial_energy)
            self.assertAlmostEqual(atoms.positions[0, 0], 0.0, places=12)

    def test_maxiter_writes_diagnostic_and_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "job.out"
            optimizer = BatchLBFGS(
                output=str(output),
                device="cpu",
                maxiter=1,
            )
            molecules = _Molecules(
                [Atoms("H", positions=[[0.0, 0.0, 0.0]])],
                _SyntheticBatchCalculator(),
            )

            with self.assertRaisesRegex(RuntimeError, "unconverged"):
                optimizer.run(molecules)

            self.assertFalse((Path(tmpdir) / "job_opt.xyz").exists())
            self.assertTrue(
                (Path(tmpdir) / "job_opt_unconverged.xyz").exists()
            )
            self.assertEqual(optimizer.statuses, ["maxiter"])
            self.assertIsNone(optimizer.log_fp)

    def test_converged_run_writes_production_output_and_status(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "job.out"
            optimizer = BatchLBFGS(
                output=str(output),
                device="cpu",
                maxiter=2,
            )
            molecules = _Molecules(
                [Atoms("H", positions=[[0.0, 0.0, 0.0]])],
                _SyntheticBatchCalculator(stationary=True),
            )

            statuses = optimizer.run(molecules)

            self.assertEqual(statuses, ("converged",))
            self.assertTrue((Path(tmpdir) / "job_opt.xyz").exists())
            self.assertFalse(
                (Path(tmpdir) / "job_opt_unconverged.xyz").exists()
            )


if __name__ == "__main__":
    unittest.main()
