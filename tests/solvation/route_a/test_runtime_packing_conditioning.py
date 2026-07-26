from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from maple.function.dispatcher.solvfe.membership import (
    SoftCutoffMembership,
    soft_occupancy_weights,
)
from maple.function.dispatcher.solvfe.packing_conditioning import (
    GeometryConditionedCavity,
    ProductMeasurePackingCalculator,
    atom_index_map_hash,
    periodic_cell_hash,
)
from maple.function.dispatcher.solvfe.packing_contract import (
    PackingBiasState,
    PackingSchedule,
)


class _ConstantCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(
        self,
        *,
        energy_ev: float,
        force_value: float,
        hamiltonian_hash: str = "3" * 64,
    ) -> None:
        super().__init__()
        self.energy_ev = float(energy_ev)
        self.force_value = float(force_value)
        self.maple_hamiltonian_hash = hamiltonian_hash

    def calculate(
        self,
        atoms=None,
        properties=("energy", "forces"),
        system_changes=all_changes,
    ):
        super().calculate(atoms, properties, system_changes)
        self.results = {
            "energy": self.energy_ev,
            "forces": np.full(
                (len(atoms), 3),
                self.force_value,
                dtype=float,
            ),
        }


class _DistanceCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def calculate(
        self,
        atoms=None,
        properties=("energy", "forces"),
        system_changes=all_changes,
    ):
        super().calculate(atoms, properties, system_changes)
        assert not np.any(atoms.get_pbc())
        distance = float(
            np.linalg.norm(atoms.positions[1] - atoms.positions[0])
        )
        self.results = {
            "energy": distance,
            "forces": np.zeros((len(atoms), 3), dtype=float),
        }


def _membership(
    *,
    carbon_radius: float = 1.5,
) -> SoftCutoffMembership:
    return SoftCutoffMembership(
        vdw_radii_angstrom={6: carbon_radius},
        lambda_s_angstrom=1.0,
        softness_angstrom=0.1,
        shell_boundary_id="nearest-solute-vdw-surface-v3",
    )


def _cavity(
    *,
    solute_indices: tuple[int, ...] = (0,),
    membership: SoftCutoffMembership | None = None,
    active_occupancy_max: int = 1,
) -> GeometryConditionedCavity:
    return GeometryConditionedCavity(
        solute_indices=solute_indices,
        membership=_membership() if membership is None else membership,
        temperature_k=298.15,
        conditioning_measure_id="vacuum-solute-times-pure-water-v3",
        solute_measure_hash="2" * 64,
        solute_atom_map_hash="6" * 64,
        active_occupancy_max=active_occupancy_max,
    )


def _schedule(
    cavity: GeometryConditionedCavity,
    *,
    oxygen_indices: tuple[int, ...] = (1,),
    cell_length_angstrom: float = 20.0,
) -> PackingSchedule:
    return PackingSchedule(
        states=(
            PackingBiasState("unbiased", 0.0),
            PackingBiasState("full-field", 1.0),
        ),
        target_state_index=0,
        full_field_state_index=1,
        membership_definition_hash=cavity.membership_definition_hash,
        observation_volume_hash=cavity.observation_volume_hash,
        boundary_adapter_hash=cavity.boundary_adapter_hash,
        conditioning_measure_id=cavity.conditioning_measure_id,
        solute_measure_hash=cavity.solute_measure_hash,
        water_hamiltonian_hash="3" * 64,
        oxygen_atom_map_hash=atom_index_map_hash(
            role="water-oxygen",
            indices=oxygen_indices,
        ),
        solute_atom_map_hash=cavity.solute_atom_map_hash,
        cell_hash=periodic_cell_hash(
            np.eye(3) * cell_length_angstrom,
            np.ones(3, dtype=bool),
        ),
        active_occupancy_max=cavity.active_occupancy_max,
        temperature_k=cavity.temperature_k,
        ensemble="NVT",
        pressure_bar=None,
        boundary_conditions="periodic-3d",
    )


def _boundary_system() -> Atoms:
    # d_surface = 2.5 - 1.5 = lambda_s, hence b = 1/2.
    return Atoms(
        "CO",
        positions=[
            [0.0, 0.0, 0.0],
            [2.5, 0.0, 0.0],
        ],
        cell=np.eye(3) * 20.0,
        pbc=True,
    )


def test_geometry_conditioned_cavity_uses_complementary_soft_empty_field():
    cavity = _cavity()
    evaluation = cavity.evaluate(_boundary_system(), oxygen_indices=(1,))
    expected = cavity.membership.evaluate_signed_distances(
        np.asarray([1.0]),
        temperature_k=cavity.temperature_k,
    )

    assert evaluation.hard_empty is False
    assert evaluation.memberships == pytest.approx([0.5])
    assert evaluation.nonmemberships == pytest.approx([0.5])
    assert evaluation.soft_occupancy_weights == pytest.approx([0.5, 0.5, 0.0])
    assert evaluation.log_empty_weight == pytest.approx(-np.log(2.0))
    assert evaluation.energy_ev == pytest.approx(
        expected.empty_potential_ev[0]
    )
    np.testing.assert_allclose(
        evaluation.forces_ev_per_angstrom.sum(axis=0),
        0.0,
        atol=1.0e-12,
    )
    assert evaluation.forces_ev_per_angstrom[0, 0] == pytest.approx(
        expected.empty_derivative_ev_per_angstrom[0]
    )
    assert evaluation.forces_ev_per_angstrom[1, 0] == pytest.approx(
        -expected.empty_derivative_ev_per_angstrom[0]
    )


def test_geometry_conditioned_cavity_uses_preregistered_active_support():
    atoms = Atoms(
        "COOO",
        positions=[
            [0.0, 0.0, 0.0],
            [2.3, 0.0, 0.0],
            [2.7, 0.0, 0.0],
            [3.1, 0.0, 0.0],
        ],
        cell=np.eye(3) * 20.0,
        pbc=True,
    )
    cavity = _cavity(active_occupancy_max=1)
    evaluation = cavity.evaluate(atoms, oxygen_indices=(1, 2, 3))

    assert evaluation.soft_occupancy_weights.shape == (3,)
    assert evaluation.soft_occupancy_weights == pytest.approx(
        soft_occupancy_weights(
            evaluation.memberships,
            active_occupancy_max=1,
        )
    )
    assert np.sum(evaluation.soft_occupancy_weights) == pytest.approx(1.0)


def test_geometry_conditioned_cavity_rejects_packing_only_n0_support():
    with pytest.raises(ValueError, match="include n=0 and at least n=1"):
        _cavity(active_occupancy_max=0)


@pytest.mark.parametrize("atom_index", [0, 1])
def test_geometry_conditioned_cavity_force_matches_finite_difference(
    atom_index,
):
    atoms = _boundary_system()
    cavity = _cavity()
    step = 1.0e-6
    plus = atoms.copy()
    minus = atoms.copy()
    plus.positions[atom_index, 0] += step
    minus.positions[atom_index, 0] -= step
    finite_difference_force = -(
        cavity.evaluate(plus, oxygen_indices=(1,)).energy_ev
        - cavity.evaluate(minus, oxygen_indices=(1,)).energy_ev
    ) / (2.0 * step)

    analytic = cavity.evaluate(
        atoms,
        oxygen_indices=(1,),
    ).forces_ev_per_angstrom[atom_index, 0]

    assert analytic == pytest.approx(finite_difference_force, abs=1.0e-8)


def test_geometry_conditioned_cavity_uses_minimum_image():
    atoms = Atoms(
        "CO",
        positions=[
            [0.2, 0.0, 0.0],
            [9.8, 0.0, 0.0],
        ],
        cell=np.eye(3) * 10.0,
        pbc=True,
    )

    evaluation = _cavity().evaluate(atoms, oxygen_indices=(1,))

    assert evaluation.signed_distances_angstrom[0] == pytest.approx(-1.1)
    assert evaluation.forces_ev_per_angstrom[1, 0] < 0.0
    np.testing.assert_allclose(
        evaluation.forces_ev_per_angstrom.sum(axis=0),
        0.0,
        atol=1.0e-12,
    )


def test_product_measure_calculator_keeps_hamiltonians_decoupled():
    atoms = Atoms(
        "COHH",
        positions=[
            [0.0, 0.0, 0.0],
            [2.5, 0.0, 0.0],
            [3.3, 0.0, 0.0],
            [2.5, 0.8, 0.0],
        ],
        cell=np.eye(3) * 20.0,
        pbc=True,
    )
    cavity = _cavity()
    bias = cavity.evaluate(atoms, oxygen_indices=(1,))
    atoms.calc = ProductMeasurePackingCalculator(
        solute_calculator=_ConstantCalculator(
            energy_ev=2.0,
            force_value=0.1,
        ),
        solvent_calculator=_ConstantCalculator(
            energy_ev=3.0,
            force_value=0.2,
        ),
        cavity=cavity,
        schedule=_schedule(cavity),
        state_index=1,
        solvent_indices=(1, 2, 3),
        oxygen_indices=(1,),
    )

    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()

    assert energy == pytest.approx(2.0 + 3.0 + bias.energy_ev)
    np.testing.assert_allclose(
        forces[0],
        0.1 + bias.forces_ev_per_angstrom[0],
    )
    np.testing.assert_allclose(
        forces[1],
        0.2 + bias.forces_ev_per_angstrom[1],
    )
    np.testing.assert_allclose(forces[2:], 0.2)
    components = atoms.calc.results["components"]
    assert components["conditioning_measure_id"] == (
        cavity.conditioning_measure_id
    )
    assert components["membership_definition_hash"] == (
        cavity.membership_definition_hash
    )
    assert components["schedule_hash"] == _schedule(cavity).content_hash
    assert components["cross_solute_solvent_energy_ev"] == 0.0
    assert components["log_empty_weight"] == pytest.approx(-np.log(2.0))


def test_product_measure_calculator_unwraps_solute_before_vacuum_energy():
    atoms = Atoms(
        "CCO",
        positions=[
            [0.2, 0.0, 0.0],
            [9.8, 0.0, 0.0],
            [5.0, 0.0, 0.0],
        ],
        cell=np.eye(3) * 10.0,
        pbc=True,
    )
    cavity = _cavity(solute_indices=(0, 1))
    atoms.calc = ProductMeasurePackingCalculator(
        solute_calculator=_DistanceCalculator(),
        solvent_calculator=_ConstantCalculator(
            energy_ev=0.0,
            force_value=0.0,
        ),
        cavity=cavity,
        schedule=_schedule(cavity, oxygen_indices=(2,), cell_length_angstrom=10.0),
        state_index=0,
        solvent_indices=(2,),
        oxygen_indices=(2,),
    )

    assert atoms.get_potential_energy() == pytest.approx(0.4)


def test_product_measure_calculator_rejects_unassigned_atoms():
    atoms = Atoms(
        "COH",
        positions=np.zeros((3, 3)),
        cell=np.eye(3) * 20.0,
        pbc=True,
    )
    cavity = _cavity()
    atoms.calc = ProductMeasurePackingCalculator(
        solute_calculator=_ConstantCalculator(
            energy_ev=0.0,
            force_value=0.0,
        ),
        solvent_calculator=_ConstantCalculator(
            energy_ev=0.0,
            force_value=0.0,
        ),
        cavity=cavity,
        schedule=_schedule(cavity),
        state_index=0,
        solvent_indices=(1,),
        oxygen_indices=(1,),
    )

    with pytest.raises(RuntimeError, match="PACKING_PARTITION_INVALID"):
        atoms.get_potential_energy()


def test_product_measure_calculator_rejects_schedule_cavity_drift():
    cavity = _cavity()
    changed = _cavity(membership=_membership(carbon_radius=1.6))

    with pytest.raises(ValueError, match="membership-definition"):
        ProductMeasurePackingCalculator(
            solute_calculator=_ConstantCalculator(
                energy_ev=0.0,
                force_value=0.0,
            ),
            solvent_calculator=_ConstantCalculator(
                energy_ev=0.0,
                force_value=0.0,
            ),
            cavity=changed,
            schedule=_schedule(cavity),
            state_index=0,
            solvent_indices=(1,),
            oxygen_indices=(1,),
        )


def test_product_measure_calculator_rejects_hamiltonian_or_atom_map_drift():
    cavity = _cavity()
    with pytest.raises(ValueError, match="Hamiltonian identity"):
        ProductMeasurePackingCalculator(
            solute_calculator=_ConstantCalculator(
                energy_ev=0.0,
                force_value=0.0,
            ),
            solvent_calculator=_ConstantCalculator(
                energy_ev=0.0,
                force_value=0.0,
                hamiltonian_hash="9" * 64,
            ),
            cavity=cavity,
            schedule=_schedule(cavity),
            state_index=0,
            solvent_indices=(1,),
            oxygen_indices=(1,),
        )

    with pytest.raises(ValueError, match="oxygen atom-map"):
        ProductMeasurePackingCalculator(
            solute_calculator=_ConstantCalculator(
                energy_ev=0.0,
                force_value=0.0,
            ),
            solvent_calculator=_ConstantCalculator(
                energy_ev=0.0,
                force_value=0.0,
            ),
            cavity=cavity,
            schedule=_schedule(cavity),
            state_index=0,
            solvent_indices=(1, 2),
            oxygen_indices=(2,),
        )


def test_product_measure_calculator_rejects_runtime_cell_drift():
    atoms = _boundary_system()
    cavity = _cavity()
    atoms.calc = ProductMeasurePackingCalculator(
        solute_calculator=_ConstantCalculator(
            energy_ev=0.0,
            force_value=0.0,
        ),
        solvent_calculator=_ConstantCalculator(
            energy_ev=0.0,
            force_value=0.0,
        ),
        cavity=cavity,
        schedule=_schedule(cavity, cell_length_angstrom=19.0),
        state_index=0,
        solvent_indices=(1,),
        oxygen_indices=(1,),
    )

    with pytest.raises(RuntimeError, match="CELL_IDENTITY"):
        atoms.get_potential_energy()


def test_geometry_conditioned_cavity_hashes_measure_and_membership():
    first = _cavity()
    changed_radius = _cavity(
        membership=_membership(carbon_radius=1.6)
    )
    changed_index = _cavity(solute_indices=(1,))

    assert first.observation_volume_hash != (
        changed_radius.observation_volume_hash
    )
    assert first.observation_volume_hash != changed_index.observation_volume_hash
    assert first.content_hash != changed_radius.content_hash
