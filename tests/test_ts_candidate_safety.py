import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from maple.function.dispatcher.ts.algorithm.BPRFO import (
    BatchPRFO,
    _regularize_signed,
)
from maple.function.dispatcher.ts.algorithm.PRFO import (
    PRFO,
    PRFOConvergenceError,
    PRFOResult,
    PRFOStatus,
)
from maple.function.dispatcher.ts.algorithm.neb import NEB
from maple.function.dispatcher.ts.algorithm.string import GSM


class _AnalyticSaddleCalculator(Calculator):
    implemented_properties = ["energy", "free_energy", "forces"]

    def calculate(
        self,
        atoms=None,
        properties=("energy",),
        system_changes=all_changes,
    ):
        super().calculate(atoms, properties, system_changes)
        x, y, z = atoms.positions[0]
        energy = -0.5 * x * x + 0.5 * y * y + 0.5 * z * z
        self.results = {
            "energy": float(energy),
            "free_energy": float(energy),
            "forces": np.asarray([[x, -y, -z]], dtype=np.float64),
        }

    def get_hessian(self, atoms):
        return np.diag([-1.0, 1.0, 1.0])


class PRFOCandidateSafetyTests(unittest.TestCase):
    def test_geometry_convergence_writes_candidate_not_certified_ts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "job.out"
            atoms = Atoms(
                "H",
                positions=[[0.05, 0.0, 0.0]],
                calculator=_AnalyticSaddleCalculator(),
            )
            prfo = PRFO(
                output=str(output),
                atoms=atoms,
                paras={
                    "max_iter": 2,
                    "f_max_th": 0.1,
                    "f_rms_th": 0.1,
                    "dp_max_th": 0.2,
                    "dp_rms_th": 0.2,
                },
            )

            result = prfo.run_result()

            self.assertEqual(result.status, PRFOStatus.GEOMETRY_CONVERGED)
            self.assertTrue(result.geometry_converged)
            self.assertTrue(
                (Path(tmpdir) / "job_prfo_ts_candidate.xyz").exists()
            )
            self.assertFalse((Path(tmpdir) / "job_prfo_ts.xyz").exists())
            self.assertIn(
                "not a frequency/IRC-verified transition state",
                output.read_text(),
            )

    def test_maxiter_is_structured_failure_and_run_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "job.out"
            atoms = Atoms(
                "H",
                positions=[[0.05, 0.0, 0.0]],
                calculator=_AnalyticSaddleCalculator(),
            )
            prfo = PRFO(
                output=str(output),
                atoms=atoms,
                paras={"max_iter": 0},
            )

            result = prfo.run_result()

            self.assertEqual(result.status, PRFOStatus.FAILED_MAXITER)
            self.assertFalse(result.geometry_converged)
            self.assertTrue(
                (Path(tmpdir) / "job_prfo_unconverged.xyz").exists()
            )
            self.assertFalse((Path(tmpdir) / "job_prfo_ts.xyz").exists())
            self.assertFalse(
                (Path(tmpdir) / "job_prfo_ts_candidate.xyz").exists()
            )

            retry = PRFO(
                output=str(Path(tmpdir) / "retry.out"),
                atoms=atoms,
                paras={"max_iter": 0},
            )
            with self.assertRaises(PRFOConvergenceError):
                retry.run()


class TSCallerSafetyTests(unittest.TestCase):
    @staticmethod
    def _failed_prfo_result(atoms, path):
        return PRFOResult(
            atoms=atoms,
            status=PRFOStatus.FAILED_MAXITER,
            iterations=0,
            structure_path=str(path),
        )

    def test_string_rejects_unconverged_prfo_result(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "string.out"
            images = [
                Atoms("H", positions=[[float(i), 0.0, 0.0]])
                for i in range(3)
            ]
            gsm = GSM.__new__(GSM)
            gsm.output = str(output)
            gsm.raw_paras = {}
            failed = self._failed_prfo_result(
                images[1],
                Path(tmpdir) / "string_prfo_unconverged.xyz",
            )

            with patch.object(PRFO, "run_result", return_value=failed):
                with self.assertRaises(PRFOConvergenceError):
                    gsm.restart_run(images, 1, str(Path(tmpdir) / "string"))

            self.assertFalse(
                (Path(tmpdir) / "string_stringts_ts_candidate.xyz").exists()
            )

    def test_neb_rejects_unconverged_prfo_result(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "neb.out"
            images = [
                Atoms("H", positions=[[float(i), 0.0, 0.0]])
                for i in range(3)
            ]
            for name in ("f_max_th", "f_rms_th", "dp_max_th", "dp_rms_th"):
                setattr(images[0], name, 1.0)

            neb = NEB.__new__(NEB)
            neb.output = str(output)
            neb.atoms_R = images[0]
            neb.params = SimpleNamespace(
                lbfgs_m=2,
                cistep0=0.01,
                neb_f_max_th=1.0,
                neb_f_rms_th=1.0,
                cineb_f_max_th=1.0,
                cineb_f_rms_th=1.0,
                k_max=0.3,
                max_iter=1,
                refine="nebts",
            )
            neb._pack_internal = lambda current: np.zeros(3)
            neb._unpack_internal = lambda values, current: None
            neb._path_energy_forces = lambda current: (
                [0.0, 1.0, 0.0],
                [np.zeros((1, 3)) for _ in current],
            )
            neb.cineb_forces = lambda current, energies, spring, raw_forces: (
                [np.zeros((1, 3)) for _ in current],
                0.0,
                1,
            )
            neb.atoms_to_xyz = lambda atoms: ""
            failed = self._failed_prfo_result(
                images[1],
                Path(tmpdir) / "neb_prfo_unconverged.xyz",
            )

            with patch.object(PRFO, "run_result", return_value=failed):
                with self.assertRaises(PRFOConvergenceError):
                    neb.restart_run(images, energies=[0.0, 1.0, 0.0])

            self.assertFalse(
                (Path(tmpdir) / "neb_nebts_ts_candidate.xyz").exists()
            )


class BatchPRFOIsolationTests(unittest.TestCase):
    def test_exact_zero_regularization_is_nonzero(self):
        values = torch.tensor(
            [-1.0e-20, 0.0, 1.0e-20, 2.0],
            dtype=torch.float64,
        )
        actual = _regularize_signed(values, 1.0e-10)
        torch.testing.assert_close(
            actual,
            torch.tensor(
                [-1.0e-10, 1.0e-10, 1.0e-10, 2.0],
                dtype=torch.float64,
            ),
        )
        self.assertTrue(torch.all(actual != 0.0))

    def test_batch_prfo_is_runtime_disabled(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            optimizer = BatchPRFO(
                output=str(Path(tmpdir) / "batch-prfo.out"),
                device="cpu",
            )
            with self.assertRaisesRegex(
                NotImplementedError,
                "experimental and runtime-disabled",
            ):
                optimizer.run(object())


if __name__ == "__main__":
    unittest.main()
