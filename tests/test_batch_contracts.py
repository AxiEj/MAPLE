import tempfile
import unittest

import numpy as np
import torch
from ase import Atoms
from ase.constraints import FixAtoms

from maple.function.calculator._batch_eval import (
    FDHessianEvaluator,
    HVPEvaluator,
    _copy_with_positions,
    _estimate_auto_batch_size_from_item_bytes,
)
from maple.function.calculator._batch_types import BatchResult
from maple.function.calculator._batch_utils import (
    normalize_energy_forces_request,
    sequential_calculate_many,
    split_atomwise_array,
)
from maple.function.calculator.aimnet._aimnet2_calculator import (
    AIMNET2_BATCH_LAYOUT_BY_SHA256,
    AIMNET2_PADDED_PER_MOLECULE_LAYOUT,
    AIMNet2Calculator,
    identify_aimnet2_batch_layout,
)
from maple.function.calculator.ani._ani_calculator import ANICalculator


class _CacheCalculator:
    def __init__(self, fail_at=None):
        self.results = {
            "energy": 7.0,
            "free_energy": 7.0,
            "forces": np.full((1, 3), 4.0),
        }
        self.atoms = Atoms("He", positions=[[9.0, 8.0, 7.0]])
        self.fail_at = fail_at
        self.calls = 0

    def calculate(self, atoms, properties, system_changes):
        self.calls += 1
        self.atoms = atoms.copy()
        self.results = {
            "energy": float(self.calls),
            "free_energy": float(self.calls),
            "forces": np.full((len(atoms), 3), float(self.calls)),
        }
        if self.calls == self.fail_at:
            raise RuntimeError("synthetic backend failure")


class _QuadraticCalculator:
    supports_hvp = False

    def calculate_many(self, atoms_list, properties=("energy", "forces")):
        energies = []
        forces = []
        for atoms in atoms_list:
            positions = atoms.get_positions()
            energies.append(0.5 * float(np.sum(positions**2)))
            forces.append(-positions)
        return BatchResult(
            energies=np.asarray(energies, dtype=np.float64),
            forces=forces,
        )


class _BadAnalyticHVP:
    supports_hvp = True

    def get_hvp(self, atoms, n):
        size = 3 * len(atoms)
        return (
            torch.full((size,), torch.nan),
            torch.zeros(size),
            torch.tensor(0.0),
        )


class _ShortANIModel:
    def __call__(self, species, coordinates):
        return (torch.zeros(1, dtype=coordinates.dtype, device=coordinates.device),)


class _NaNANIModel:
    def __call__(self, species, coordinates):
        energy = coordinates.sum(dim=(1, 2)) * float("nan")
        return (energy,)


class BatchResultContractTests(unittest.TestCase):
    def test_rejects_nonfinite_numeric_fields(self):
        with self.assertRaisesRegex(ValueError, "energies.*finite"):
            BatchResult(energies=np.array([np.nan]))
        with self.assertRaisesRegex(ValueError, r"forces\[0\].*finite"):
            BatchResult(forces=[np.array([[0.0, np.inf, 0.0]])])
        with self.assertRaisesRegex(ValueError, r"hessians\[0\].*finite"):
            BatchResult(hessians=[np.array([[np.nan]])])

    def test_rejects_fractional_or_negative_padding(self):
        with self.assertRaisesRegex(ValueError, "integer"):
            BatchResult(padding_counts=np.array([1.5]))
        with self.assertRaisesRegex(ValueError, "non-negative"):
            BatchResult(padding_counts=np.array([-1]))

    def test_validate_against_checks_requested_fields_and_atom_counts(self):
        atoms = [Atoms("H"), Atoms("H2")]
        result = BatchResult(
            energies=np.array([1.0, 2.0]),
            forces=[np.zeros((1, 3)), np.zeros((1, 3))],
        )
        with self.assertRaisesRegex(ValueError, r"forces\[1\].*expected"):
            result.validate_against(atoms, ("energy", "forces"))

        with self.assertRaisesRegex(ValueError, "requested.*forces"):
            BatchResult(energies=np.array([1.0, 2.0])).validate_against(
                atoms, ("energy", "forces")
            )

    def test_nested_arrays_are_read_only_and_revalidated(self):
        atoms = [Atoms("H")]
        result = BatchResult(
            energies=np.array([1.0]),
            forces=[np.zeros((1, 3))],
        )
        self.assertIsInstance(result.forces, tuple)
        with self.assertRaises(ValueError):
            result.energies[0] = np.nan
        with self.assertRaises(ValueError):
            result.forces[0][0, 0] = np.inf

        result.energies.setflags(write=True)
        result.energies[0] = np.nan
        with self.assertRaisesRegex(ValueError, "energies.*finite"):
            result.validate_against(atoms, ("energy", "forces"))


class BatchUtilityContractTests(unittest.TestCase):
    def test_unknown_properties_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "Unsupported calculate_many"):
            normalize_energy_forces_request(("energy", "stress"))

    def test_split_atomwise_array_rejects_wrong_shape_or_incomplete_counts(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            split_atomwise_array(np.zeros((2, 2)), [1, 1])
        with self.assertRaisesRegex(ValueError, "sum"):
            split_atomwise_array(np.zeros((3, 3)), [1, 1])
        with self.assertRaisesRegex(ValueError, "non-negative integers"):
            split_atomwise_array(np.zeros((2, 3)), [1.5, 0.5])

    def test_sequential_fallback_restores_cache_after_success(self):
        calc = _CacheCalculator()
        old_results = calc.results
        old_atoms = calc.atoms
        old_results_copy = {
            key: np.array(value, copy=True) if isinstance(value, np.ndarray) else value
            for key, value in old_results.items()
        }

        result = sequential_calculate_many(
            calc,
            [Atoms("H"), Atoms("H")],
            ("energy", "forces"),
            True,
            True,
        )

        np.testing.assert_allclose(result.energies, [1.0, 2.0])
        self.assertIs(calc.results, old_results)
        self.assertIs(calc.atoms, old_atoms)
        for key, value in old_results_copy.items():
            np.testing.assert_allclose(calc.results[key], value)

    def test_sequential_fallback_restores_cache_after_exception(self):
        calc = _CacheCalculator(fail_at=2)
        old_results = calc.results
        old_atoms = calc.atoms

        with self.assertRaisesRegex(RuntimeError, "synthetic backend failure"):
            sequential_calculate_many(
                calc,
                [Atoms("H"), Atoms("H")],
                ("energy", "forces"),
                True,
                True,
            )

        self.assertIs(calc.results, old_results)
        self.assertIs(calc.atoms, old_atoms)
        self.assertEqual(calc.results["energy"], 7.0)

    def test_displaced_atoms_do_not_share_constraint_objects(self):
        atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.7, 0.0, 0.0]])
        atoms.set_constraint(FixAtoms(indices=[0]))
        displaced = _copy_with_positions(
            atoms,
            atoms.get_positions() + 0.01,
            apply_constraints=False,
        )
        self.assertIsNot(displaced.constraints[0], atoms.constraints[0])


class BackendContractTests(unittest.TestCase):
    def test_aimnet_batch_schema_is_bound_to_checkpoint_identity(self):
        self.assertEqual(
            set(AIMNET2_BATCH_LAYOUT_BY_SHA256.values()),
            {AIMNET2_PADDED_PER_MOLECULE_LAYOUT},
        )
        self.assertEqual(len(AIMNET2_BATCH_LAYOUT_BY_SHA256), 2)
        with tempfile.NamedTemporaryFile() as custom:
            custom.write(b"unrecognized checkpoint")
            custom.flush()
            self.assertIsNone(
                identify_aimnet2_batch_layout(custom.name)
            )

    def test_ani_rejects_short_energy_vector(self):
        calc = ANICalculator.__new__(ANICalculator)
        calc.device = torch.device("cpu")
        calc.dtype = torch.float64
        calc.d4 = False
        calc.solvent_correction = None
        calc.model = _ShortANIModel()
        atoms = [
            Atoms("H", positions=[[0.0, 0.0, 0.0]]),
            Atoms("H", positions=[[1.0, 0.0, 0.0]]),
        ]
        with self.assertRaisesRegex(RuntimeError, "ANI energy output"):
            calc.calculate_many(atoms, properties=("energy",))

    def test_aimnet_requires_padded_per_molecule_energy_layout(self):
        output = torch.tensor([1.0, 2.0, 0.0])
        actual = AIMNet2Calculator._energy_vector_from_output(
            output,
            batch_size=2,
            layout=AIMNET2_PADDED_PER_MOLECULE_LAYOUT,
        )
        torch.testing.assert_close(actual, torch.tensor([1.0, 2.0]))

        with self.assertRaisesRegex(RuntimeError, "padded per-molecule"):
            AIMNet2Calculator._energy_vector_from_output(
                torch.tensor([1.0, 2.0]),
                batch_size=2,
                layout=AIMNET2_PADDED_PER_MOLECULE_LAYOUT,
            )
        with self.assertRaisesRegex(RuntimeError, "padded per-molecule"):
            AIMNet2Calculator._energy_vector_from_output(
                torch.arange(5.0),
                batch_size=2,
                layout=AIMNET2_PADDED_PER_MOLECULE_LAYOUT,
            )
        with self.assertRaisesRegex(RuntimeError, "validated"):
            AIMNet2Calculator._energy_vector_from_output(
                torch.tensor([1.0, 2.0, 3.0]),
                batch_size=2,
                layout=None,
            )

    def test_ani_direct_hvp_rejects_nonfinite_outputs(self):
        calc = ANICalculator.__new__(ANICalculator)
        calc.device = torch.device("cpu")
        calc.dtype = torch.float64
        calc.d4 = False
        calc.solvent_correction = None
        calc.model = _NaNANIModel()
        atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])

        with self.assertRaisesRegex(FloatingPointError, "finite scalar energy"):
            calc.get_hvp(atoms, np.ones(3))
        with self.assertRaisesRegex(ValueError, "non-zero"):
            calc.get_hvp(atoms, np.zeros(3))


class EvaluatorSafetyTests(unittest.TestCase):
    def test_auto_batch_sizing_never_rounds_above_memory_budget(self):
        self.assertEqual(
            _estimate_auto_batch_size_from_item_bytes(
                n_total=10,
                item_bytes=1,
                free_bytes=12,
                target_fraction=0.75,
            ),
            9,
        )
        self.assertEqual(
            _estimate_auto_batch_size_from_item_bytes(
                n_total=10,
                item_bytes=0,
                free_bytes=12,
                target_fraction=0.75,
            ),
            1,
        )

    def test_fd_hessian_rejects_nonfinite_delta_and_forces(self):
        calc = _QuadraticCalculator()
        atoms = Atoms("H", positions=[[0.1, 0.0, 0.0]])
        with self.assertRaisesRegex(ValueError, "finite and positive"):
            FDHessianEvaluator(calc).hessian(atoms, delta=np.nan)
        with self.assertRaisesRegex(ValueError, "finite"):
            FDHessianEvaluator._validated_force(
                np.array([[np.nan, 0.0, 0.0]]), 1, 0
            )

    def test_fd_hessian_large_antisymmetry_fails_closed(self):
        evaluator = FDHessianEvaluator(_QuadraticCalculator())
        hessian = np.array([[0.0, 1.0], [0.0, 0.0]])
        with self.assertRaisesRegex(RuntimeError, "antisymmetric residual"):
            evaluator._symmetrize(hessian)

    def test_hvp_validates_direction_and_backend_outputs(self):
        atoms = Atoms("H", positions=[[0.1, 0.0, 0.0]])
        with self.assertRaisesRegex(ValueError, "exactly 3"):
            HVPEvaluator(_QuadraticCalculator()).hn(
                atoms, np.zeros(2), delta=0.005
            )
        with self.assertRaisesRegex(ValueError, "non-zero"):
            HVPEvaluator(_QuadraticCalculator()).hn(
                atoms, np.zeros(3), delta=0.005
            )
        with self.assertRaisesRegex(ValueError, "finite and positive"):
            HVPEvaluator(_QuadraticCalculator()).hn(
                atoms, np.ones(3), delta=np.nan
            )
        with self.assertRaisesRegex(FloatingPointError, "HVP"):
            HVPEvaluator(_BadAnalyticHVP()).hn(
                atoms, np.ones(3), delta=0.005
            )


if __name__ == "__main__":
    unittest.main()
