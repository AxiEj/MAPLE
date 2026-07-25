from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import all_changes
from ase.constraints import FixAtoms

from maple.function.calculator import BatchResult
from maple.function.calculator.calculator_base import CalcABC
from maple.function.free_energy import evaluate_gas_conformer_energies

MODEL_DIR = Path(__file__).resolve().parents[2] / "maple/function/calculator/model"


class HarmonicCalculator(CalcABC):
    implemented_properties = ["energy", "forces", "free_energy"]
    MODEL_ENERGY_UNIT = "hartree"

    def __init__(self, reference):
        super().__init__()
        self.reference = np.asarray(reference, dtype=np.float64)

    def calculate(
        self,
        atoms=None,
        properties=("energy",),
        system_changes=all_changes,
    ):
        atoms = super().calculate(atoms, properties, system_changes)
        displacement = atoms.get_positions() - self.reference
        energy = 0.5 * float(np.sum(displacement**2))
        forces = -displacement if "forces" in properties else None
        self._finalize_results(atoms, energy=energy, forces=forces)


class RecordingHarmonicCalculator(HarmonicCalculator):
    supports_batch_energy_forces = True

    def __init__(self, reference):
        super().__init__(reference)
        self.batch_lengths = []

    def calculate_many(self, atoms_list, properties=("energy", "forces")):
        atoms_list = list(atoms_list)
        self.batch_lengths.append(len(atoms_list))
        return super().calculate_many(atoms_list, properties)


class DivergentEnergyCalculator(CalcABC):
    implemented_properties = ["energy", "free_energy"]

    def calculate(
        self,
        atoms=None,
        properties=("energy",),
        system_changes=all_changes,
    ):
        super().calculate(atoms, properties, system_changes)
        self.results = {"energy": 1.0, "free_energy": 2.0}


def _hydrogen_dimer_states():
    return [
        Atoms(
            "H2",
            positions=[[0.0, 0.0, 0.0], [0.72 + 0.01 * index, 0.0, 0.0]],
        )
        for index in range(3)
    ]


def _sequential_results(calculator, atoms_list):
    energies = []
    forces = []
    for atoms in atoms_list:
        calculator.calculate(
            atoms,
            properties=["energy", "forces"],
            system_changes=all_changes,
        )
        energies.append(float(calculator.results["energy"]))
        forces.append(np.asarray(calculator.results["forces"], dtype=np.float64))
    return np.asarray(energies), forces


def _assert_matches_sequential(batched, expected_energies, expected_forces):
    np.testing.assert_allclose(
        batched.energies,
        expected_energies,
        rtol=1.0e-6,
        atol=1.0e-7,
    )
    assert batched.forces is not None
    for got, expected in zip(batched.forces, expected_forces):
        np.testing.assert_allclose(got, expected, rtol=1.0e-6, atol=1.0e-7)


def test_batch_result_normalizes_and_validates_fields():
    result = BatchResult(
        energies=[1, 2],
        forces=[[[0, 1, 2]], np.asarray([[3, 4, 5]], dtype=np.float32)],
    )

    assert result.energies.dtype == np.float64
    assert all(force.dtype == np.float64 for force in result.forces)

    with pytest.raises(ValueError, match="lengths"):
        BatchResult(energies=np.zeros(2), forces=[np.zeros((1, 3))])
    with pytest.raises(ValueError, match="shape"):
        BatchResult(forces=[np.zeros(3)])


def test_base_calculate_many_is_a_result_driven_sequential_fallback():
    reference = np.zeros((2, 3), dtype=np.float64)
    atoms_list = [
        Atoms("H2", positions=reference + 0.05 * (index + 1)) for index in range(3)
    ]
    calculator = HarmonicCalculator(reference)

    result = calculator.calculate_many(
        atoms_list,
        properties=("energy", "forces"),
    )

    assert isinstance(result, BatchResult)
    np.testing.assert_allclose(result.energies, [0.0075, 0.03, 0.0675])
    assert len(result.forces) == 3
    assert calculator.supports_batch_energy_forces is False


def test_sequential_fallback_restores_single_structure_cache():
    reference = np.zeros((2, 3), dtype=np.float64)
    calculator = HarmonicCalculator(reference)
    cached_atoms = Atoms("H2", positions=reference + 0.25)
    calculator.atoms = cached_atoms
    calculator.results = {"energy": 123.0, "free_energy": 123.0}

    calculator.calculate_many(
        [Atoms("H2", positions=reference + 0.05)],
        properties=("energy", "forces"),
    )

    assert calculator.atoms is cached_atoms
    assert calculator.results == {"energy": 123.0, "free_energy": 123.0}


def test_sequential_fallback_returns_requested_energy_not_free_energy():
    calculator = DivergentEnergyCalculator()

    result = calculator.calculate_many(
        [Atoms("H", positions=np.zeros((1, 3)))],
        properties=("energy",),
    )

    np.testing.assert_allclose(result.energies, [1.0])


def test_calculate_many_rejects_hessian_assembly():
    calculator = HarmonicCalculator(np.zeros((2, 3), dtype=np.float64))

    with pytest.raises(NotImplementedError, match="Hessian"):
        calculator.calculate_many(
            [Atoms("H2", positions=np.zeros((2, 3)))],
            properties=("hessian",),
        )


def test_calculate_many_rejects_unknown_properties():
    calculator = HarmonicCalculator(np.zeros((2, 3), dtype=np.float64))

    with pytest.raises(NotImplementedError, match="energy and forces"):
        calculator.calculate_many(
            [Atoms("H2", positions=np.zeros((2, 3)))],
            properties=("stress",),
        )


def test_conformer_evaluator_chunks_without_mutating_template():
    reference = np.zeros((2, 3), dtype=np.float64)
    positions = np.asarray([reference + 0.01 * (index + 1) for index in range(5)])
    template = Atoms("H2", positions=reference)
    calculator = RecordingHarmonicCalculator(reference)

    energies = evaluate_gas_conformer_energies(
        calculator,
        template,
        positions,
        batch_size=2,
    )

    assert calculator.batch_lengths == [2, 2, 1]
    np.testing.assert_allclose(
        energies,
        [0.0003, 0.0012, 0.0027, 0.0048, 0.0075],
    )
    np.testing.assert_array_equal(template.get_positions(), reference)


def test_conformer_evaluator_requires_a_gas_phase_calculator():
    template = Atoms("H2", positions=np.zeros((2, 3)))
    calculator = RecordingHarmonicCalculator(np.zeros((2, 3)))
    calculator.solvent_correction = object()

    with pytest.raises(ValueError, match="gas-phase"):
        evaluate_gas_conformer_energies(
            calculator,
            template,
            np.zeros((1, 2, 3)),
            batch_size=1,
        )


def test_conformer_evaluator_uses_submitted_positions_despite_constraints():
    reference = np.zeros((2, 3), dtype=np.float64)
    template = Atoms("H2", positions=reference)
    template.set_constraint(FixAtoms(indices=[0]))
    calculator = RecordingHarmonicCalculator(reference)

    energies = evaluate_gas_conformer_energies(
        calculator,
        template,
        np.full((1, 2, 3), 0.1),
        batch_size=1,
    )

    np.testing.assert_allclose(energies, [0.03])


def test_non_calcabc_uma_exposes_the_common_sequential_fallback():
    pytest.importorskip("fairchem")
    from maple.function.calculator.uma._uma_calculator import UMACalculator

    assert UMACalculator.supports_batch_energy_forces is False
    assert UMACalculator.calculate_many is not CalcABC.calculate_many


def test_ani_calculate_many_uses_one_torch_batch():
    torch = pytest.importorskip("torch")
    from maple.function.calculator.ani._ani_calculator import ANICalculator

    model_path = MODEL_DIR / "ani2x.pt"
    if not model_path.is_file():
        pytest.skip(f"Local checkpoint is unavailable: {model_path}")

    atoms_list = _hydrogen_dimer_states()
    batch_calculator = ANICalculator(
        torch.device("cpu"),
        model="ani2x",
        model_path=str(model_path),
        implicit="none",
    )
    sequential_calculator = ANICalculator(
        torch.device("cpu"),
        model="ani2x",
        model_path=str(model_path),
        implicit="none",
    )

    class CountingModel:
        def __init__(self, model):
            self.model = model
            self.calls = []

        def __call__(self, species, coordinates):
            self.calls.append(tuple(coordinates.shape))
            return self.model(species, coordinates)

    counter = CountingModel(batch_calculator.model)
    batch_calculator.model = counter
    batched = batch_calculator.calculate_many(
        atoms_list,
        properties=("energy", "forces"),
    )
    expected = _sequential_results(sequential_calculator, atoms_list)

    assert counter.calls == [(len(atoms_list), 2, 3)]
    assert batch_calculator.supports_batch_energy_forces is True
    _assert_matches_sequential(batched, *expected)


def test_ani_calculate_many_rejects_short_model_output():
    torch = pytest.importorskip("torch")
    from maple.function.calculator.ani._ani_calculator import ANICalculator

    class ShortBatchModel:
        def __call__(self, species, coordinates):
            energies = coordinates.sum(dim=(1, 2))[:-1]
            return (energies,)

    calculator = ANICalculator.__new__(ANICalculator)
    calculator.device = torch.device("cpu")
    calculator.dtype = torch.float32
    calculator.d4 = False
    calculator.solvent_correction = None
    calculator.model = ShortBatchModel()

    with pytest.raises(
        RuntimeError,
        match="returned 2 energies for 3 structures",
    ):
        calculator.calculate_many(
            _hydrogen_dimer_states(),
            properties=("energy",),
        )


def test_aimnet2_calculate_many_uses_one_molecule_index_batch():
    torch = pytest.importorskip("torch")
    from maple.function.calculator.aimnet._aimnet2_calculator import (
        AIMNet2Calculator,
    )

    model_path = MODEL_DIR / "aimnet2.pt"
    if not model_path.is_file():
        pytest.skip(f"Local checkpoint is unavailable: {model_path}")

    atoms_list = _hydrogen_dimer_states()
    batch_calculator = AIMNet2Calculator(
        torch.device("cpu"),
        model_path=str(model_path),
        implicit="none",
    )
    sequential_calculator = AIMNet2Calculator(
        torch.device("cpu"),
        model_path=str(model_path),
        implicit="none",
    )

    class CountingModel:
        def __init__(self, model):
            self.model = model
            self.calls = []

        def __call__(self, data):
            self.calls.append(int(data["mol_idx"].max().item()))
            return self.model(data)

    counter = CountingModel(batch_calculator.model)
    batch_calculator.model = counter
    batched = batch_calculator.calculate_many(
        atoms_list,
        properties=("energy", "forces"),
    )
    expected = _sequential_results(sequential_calculator, atoms_list)

    assert counter.calls == [len(atoms_list)]
    assert batch_calculator.supports_batch_energy_forces is True
    _assert_matches_sequential(batched, *expected)


def test_aimnet2_batch_energy_contract_is_padded_per_molecule_only():
    torch = pytest.importorskip("torch")
    from maple.function.calculator.aimnet._aimnet2_calculator import (
        AIMNet2Calculator,
    )

    energies = AIMNet2Calculator._energy_vector_from_output(
        torch.tensor([1.0, 2.0, 99.0]),
        batch_size=2,
    )
    torch.testing.assert_close(energies, torch.tensor([1.0, 2.0]))

    with pytest.raises(RuntimeError, match="plus the padded molecule"):
        AIMNet2Calculator._energy_vector_from_output(
            torch.tensor([1.0, 2.0]),
            batch_size=2,
        )


def test_mace_calculate_many_uses_one_disconnected_graph_batch():
    torch = pytest.importorskip("torch")
    from maple.function.calculator.mace._mace_calculator import MACECalculator

    model_path = MODEL_DIR / "maceoff23m.pt"
    if not model_path.is_file():
        pytest.skip(f"Local checkpoint is unavailable: {model_path}")

    atoms_list = _hydrogen_dimer_states()
    batch_calculator = MACECalculator(
        torch.device("cpu"),
        model="maceoff23m",
        model_path=str(model_path),
        implicit="none",
    )
    sequential_calculator = MACECalculator(
        torch.device("cpu"),
        model="maceoff23m",
        model_path=str(model_path),
        implicit="none",
    )

    class CountingModel:
        def __init__(self, model):
            self.model = model
            self.calls = []

        def forward(self, data, local_or_ghost, compute_virials=False):
            self.calls.append(int(data["ptr"].numel() - 1))
            return self.model.forward(
                data=data,
                local_or_ghost=local_or_ghost,
                compute_virials=compute_virials,
            )

    counter = CountingModel(batch_calculator.model)
    batch_calculator.model = counter
    batched = batch_calculator.calculate_many(
        atoms_list,
        properties=("energy", "forces"),
    )
    expected = _sequential_results(sequential_calculator, atoms_list)

    assert counter.calls == [len(atoms_list)]
    assert batch_calculator.supports_batch_energy_forces is True
    _assert_matches_sequential(batched, *expected)
