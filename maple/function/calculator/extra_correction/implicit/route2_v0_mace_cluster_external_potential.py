"""Zero-field MACE cluster interaction for the Route-2 V0 liquid branch.

The learned field-response map of the current MACE-POLAR checkpoint is not a
gradient field and therefore must not be used as the electronic stationary
condition of a liquid functional.  Its *zero-field energy*, in contrast, is a
scalar.  For a neutral singlet solute ``A`` and one rigid neutral-singlet
solvent molecule ``B_Gamma`` at a declared configuration ``Gamma``, this
module defines the external molecular potential

``u_MACE(A, B_Gamma) = E0_MACE(A union B_Gamma) - E0_MACE(A) - E0_MACE(B_Gamma)``.

Every term is evaluated with the same unmodified, zero-field MACE-POLAR-1-M
checkpoint.  Thus intramolecular reference energies cancel exactly, no
atom-charge/Lennard-Jones surrogate is introduced, and the interaction force
is the corresponding difference of zero-field MACE forces.  This is the
appropriate scalar input to a later stationary molecular HNC/MDFT functional:
the liquid degrees of freedom are variational, while the MACE cluster energy
is a fixed external potential.

The class deliberately makes no claim that the resulting hybrid liquid is
already physical.  In particular, a real endpoint still needs a frozen
solvent-side correlation/thermodynamic asset and an independently justified
cross-model reference convention.  It also does not use a field-conditioned
MACE energy, density response, density mixing, post-training, fine-tuning, or
target-solvation labels.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
from ase import Atoms
from ase.units import Bohr, Hartree

from ....route2_smd_profiles import MACEPOL_MOLECULAR_REALSPACE_PROFILE
from ...calculator_base import ROUTE2_SMD_CALCULATOR_PROFILE
from .route2_v0_molecular_external_potential import (
    Route2V0MolecularConfigurations,
    Route2V0MolecularSolventReference,
)
from .route2_v0_molecular_external_potential_contract import (
    Route2V0MolecularExternalPotentialContract,
)
from .route2_v0_structured_solvent import RegularCartesianGrid

V0_MACE_CLUSTER_EXTERNAL_POTENTIAL_CONSTRUCTION = (
    "route2-v0-mace-zero-field-cluster-external-potential-v1"
)
V0_MACE_CLUSTER_EXTERNAL_POTENTIAL_SCOPE = (
    "zero-field-mace-cluster-interaction-control-only-v1"
)
V0_MACE_CLUSTER_MOLECULAR_EXTERNAL_POTENTIAL_CONSTRUCTION = (
    "route2-v0-mace-zero-field-molecular-external-potential-v1"
)
V0_MACE_CLUSTER_MOLECULAR_EXTERNAL_POTENTIAL_SCOPE = (
    "zero-field-mace-molecular-external-potential-control-only-v1"
)
V0_MACE_CLUSTER_MOLECULAR_EXTERNAL_POTENTIAL_COORDINATE_POLICY = (
    "zero-field-mace-force-per-configuration-no-liquid-force-v1"
)

# Keep this content-addressed V0 source on the exact Route-2 checkpoint rather
# than accepting an arbitrary checkpoint that happens to expose polar_state().
_MACE_POLAR_1_M_CHECKPOINT = {
    "identifier": "polar-1-m",
    "release_url": (
        "https://github.com/ACEsuit/mace-foundations/releases/download/"
        "mace_polar_1/MACE-POLAR-1-M.model"
    ),
    "sha256": "fab8b8713c832f31a2a853aaa22fd638be8a369cbf5095e6b3e982a18d10e93a",
    "size_bytes": 68_133_235,
}
_MACE_TORCH_VERSION = "0.3.16"


def _immutable_array(
    values: np.ndarray,
    *,
    name: str,
    shape: tuple[int, ...],
) -> np.ndarray:
    """Return one immutable finite floating-point array of an exact shape."""

    array = np.asarray(values, dtype=float)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite with shape {shape}.")
    result = np.array(array, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _atomic_numbers(values: np.ndarray, *, name: str) -> np.ndarray:
    """Validate one nonempty, finite, positive-integer atomic-number vector."""

    raw = np.asarray(values)
    if raw.ndim != 1 or raw.size == 0:
        raise ValueError(f"{name} must be a nonempty one-dimensional array.")
    try:
        numeric = np.asarray(raw, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} must contain positive integer atomic numbers."
        ) from exc
    rounded = np.rint(numeric)
    if (
        not np.all(np.isfinite(numeric))
        or not np.array_equal(numeric, rounded)
        or np.any(rounded < 1.0)
    ):
        raise ValueError(f"{name} must contain positive integer atomic numbers.")
    result = rounded.astype(np.int64, copy=True)
    result.setflags(write=False)
    return result


def _finite_scalar(value: object, *, name: str) -> float:
    """Return one finite scalar without accepting a nonnumeric shortcut."""

    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be finite.")
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite.") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _nonnegative_integer(value: object, *, name: str) -> int:
    """Validate one nonnegative integer result field."""

    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a nonnegative integer.")
    try:
        result = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a nonnegative integer.") from exc
    if result != value or result < 0:
        raise ValueError(f"{name} must be a nonnegative integer.")
    return result


def _neutral_singlet_fragment(atoms: Any, *, name: str) -> Atoms:
    """Copy one neutral, nonperiodic singlet fragment into the MACE domain."""

    try:
        numbers = _atomic_numbers(np.asarray(atoms.numbers), name=f"{name} numbers")
        positions = np.asarray(atoms.get_positions(), dtype=float)
        pbc = np.asarray(atoms.pbc, dtype=bool)
        info = dict(atoms.info)
    except AttributeError as exc:
        raise TypeError(
            f"{name} must be an ASE-like atoms object with numbers, positions, "
            "PBC, and info."
        ) from exc
    if positions.shape != (numbers.size, 3) or not np.all(np.isfinite(positions)):
        raise ValueError(
            f"{name} positions must be finite with shape ({numbers.size}, 3)."
        )
    if pbc.shape != (3,) or np.any(pbc):
        raise ValueError(f"{name} must be a nonperiodic molecular fragment.")
    charge = _finite_scalar(info.get("charge", 0.0), name=f"{name} charge")
    if abs(charge) > 1.0e-12:
        raise ValueError(f"{name} must be neutral for the V0 cluster control.")
    multiplicity = _nonnegative_integer(
        info.get("mult", 1), name=f"{name} multiplicity"
    )
    if multiplicity != 1:
        raise ValueError(f"{name} must be a singlet for the V0 cluster control.")
    return Atoms(
        numbers=numbers,
        positions=positions,
        pbc=False,
        info={"charge": 0.0, "mult": 1},
    )


def _combined_neutral_singlet(solute: Atoms, solvent: Atoms) -> Atoms:
    """Build the exact nonperiodic union used in the cluster-energy difference."""

    return Atoms(
        numbers=np.concatenate((solute.numbers, solvent.numbers)),
        positions=np.concatenate((solute.get_positions(), solvent.get_positions())),
        pbc=False,
        info={"charge": 0.0, "mult": 1},
    )


def _validate_route2_mace_calculator(calculator: object) -> None:
    """Require the immutable Route-2 MACE-POLAR-1-M zero-field source."""

    if not callable(getattr(calculator, "polar_state", None)):
        raise TypeError(
            "The V0 MACE cluster source requires MACEPolCalculator.polar_state()."
        )
    checkpoint = getattr(calculator, "mace_polar_checkpoint_provenance", None)
    if not isinstance(checkpoint, Mapping) or any(
        checkpoint.get(key) != value
        for key, value in _MACE_POLAR_1_M_CHECKPOINT.items()
    ):
        raise ValueError(
            "The V0 MACE cluster source requires the exact content-addressed "
            "official MACE-POLAR-1-M checkpoint."
        )
    if getattr(calculator, "mace_torch_version", None) != _MACE_TORCH_VERSION:
        raise ValueError(
            "The V0 MACE cluster source requires mace-torch " f"{_MACE_TORCH_VERSION}."
        )
    if getattr(calculator, "route2_smd_profile", None) != ROUTE2_SMD_CALCULATOR_PROFILE:
        raise ValueError(
            "The V0 MACE cluster source requires the Route-2 float64 "
            "MACE-POLAR calculator profile."
        )
    if (
        getattr(calculator, "long_range_evaluator_profile", None)
        != MACEPOL_MOLECULAR_REALSPACE_PROFILE
    ):
        raise ValueError(
            "The V0 MACE cluster source requires the default molecular "
            "real-space long-range evaluator."
        )


def _evaluate_zero_field_state(
    calculator: object,
    atoms: Atoms,
    *,
    compute_forces: bool,
    name: str,
) -> SimpleNamespace:
    """Evaluate and validate one zero-field MACE scalar state."""

    try:
        result = calculator.polar_state(atoms, compute_forces=compute_forces)  # type: ignore[attr-defined]
    except Exception as exc:
        raise RuntimeError(f"Zero-field MACE evaluation failed for {name}.") from exc
    if not isinstance(result, tuple) or len(result) != 2:
        raise RuntimeError(
            "MACE polar_state() must return the documented (state, output) pair."
        )
    state = result[0]
    energy = _finite_scalar(getattr(state, "energy_ev", None), name=f"{name} energy")
    forces = None
    if compute_forces:
        forces = _immutable_array(
            getattr(state, "fixed_field_forces_ev_per_angstrom", None),
            name=f"{name} zero-field forces",
            shape=(len(atoms), 3),
        )
    return SimpleNamespace(energy_ev=energy, forces_ev_per_angstrom=forces)


@dataclass(frozen=True)
class Route2V0MaceZeroFieldClusterInteraction:
    """One exact zero-field MACE solute--molecule interaction scalar.

    The optional force arrays are the negative Cartesian gradients of the
    interaction energy on each fragment.  They are present only when all three
    zero-field MACE evaluations requested forces.
    """

    solute_atomic_numbers: np.ndarray
    solute_positions_angstrom: np.ndarray
    solvent_atomic_numbers: np.ndarray
    solvent_positions_angstrom: np.ndarray
    solute_energy_ev: float
    solvent_energy_ev: float
    cluster_energy_ev: float
    interaction_energy_ev: float
    solute_interaction_forces_ev_per_angstrom: np.ndarray | None
    solvent_interaction_forces_ev_per_angstrom: np.ndarray | None
    construction: str = V0_MACE_CLUSTER_EXTERNAL_POTENTIAL_CONSTRUCTION
    interaction_scope: str = V0_MACE_CLUSTER_EXTERNAL_POTENTIAL_SCOPE

    def __post_init__(self) -> None:
        solute_numbers = _atomic_numbers(
            self.solute_atomic_numbers,
            name="MACE cluster solute atomic numbers",
        )
        solvent_numbers = _atomic_numbers(
            self.solvent_atomic_numbers,
            name="MACE cluster solvent atomic numbers",
        )
        solute_positions = _immutable_array(
            self.solute_positions_angstrom,
            name="MACE cluster solute positions",
            shape=(solute_numbers.size, 3),
        )
        solvent_positions = _immutable_array(
            self.solvent_positions_angstrom,
            name="MACE cluster solvent positions",
            shape=(solvent_numbers.size, 3),
        )
        energies = {
            "solute_energy_ev": self.solute_energy_ev,
            "solvent_energy_ev": self.solvent_energy_ev,
            "cluster_energy_ev": self.cluster_energy_ev,
            "interaction_energy_ev": self.interaction_energy_ev,
        }
        for name, value in energies.items():
            object.__setattr__(self, name, _finite_scalar(value, name=name))
        expected = (
            self.cluster_energy_ev - self.solute_energy_ev - self.solvent_energy_ev
        )
        tolerance = 1.0e-12 * max(1.0, abs(expected))
        if abs(self.interaction_energy_ev - expected) > tolerance:
            raise ValueError(
                "MACE cluster interaction energy must equal cluster minus both "
                "zero-field fragment energies."
            )
        solute_forces = self.solute_interaction_forces_ev_per_angstrom
        solvent_forces = self.solvent_interaction_forces_ev_per_angstrom
        if (solute_forces is None) != (solvent_forces is None):
            raise ValueError(
                "MACE cluster interaction forces must be supplied together."
            )
        if solute_forces is not None:
            solute_forces = _immutable_array(
                solute_forces,
                name="MACE cluster solute interaction forces",
                shape=(solute_numbers.size, 3),
            )
            solvent_forces = _immutable_array(
                solvent_forces,
                name="MACE cluster solvent interaction forces",
                shape=(solvent_numbers.size, 3),
            )
        if self.construction != V0_MACE_CLUSTER_EXTERNAL_POTENTIAL_CONSTRUCTION:
            raise ValueError("Unsupported Route-2 V0 MACE cluster construction.")
        if self.interaction_scope != V0_MACE_CLUSTER_EXTERNAL_POTENTIAL_SCOPE:
            raise ValueError("Unsupported Route-2 V0 MACE cluster interaction scope.")
        object.__setattr__(self, "solute_atomic_numbers", solute_numbers)
        object.__setattr__(self, "solute_positions_angstrom", solute_positions)
        object.__setattr__(self, "solvent_atomic_numbers", solvent_numbers)
        object.__setattr__(self, "solvent_positions_angstrom", solvent_positions)
        object.__setattr__(
            self,
            "solute_interaction_forces_ev_per_angstrom",
            solute_forces,
        )
        object.__setattr__(
            self,
            "solvent_interaction_forces_ev_per_angstrom",
            solvent_forces,
        )

    @property
    def interaction_energy_hartree(self) -> float:
        """Return the exact external-potential value in Hartree."""

        return self.interaction_energy_ev / Hartree

    @property
    def forces_available(self) -> bool:
        """Return whether this state carries the exact interaction force split."""

        return self.solute_interaction_forces_ev_per_angstrom is not None


def evaluate_route2_v0_mace_zero_field_cluster_interaction(
    *,
    calculator: object,
    solute: Any,
    solvent: Any,
    compute_forces: bool = False,
) -> Route2V0MaceZeroFieldClusterInteraction:
    """Return the no-training MACE scalar for one solvent configuration.

    ``solute`` and ``solvent`` must each be neutral, nonperiodic singlets.  The
    MACE calls intentionally have no local potential, local gradient, or
    feature-field argument, so this function never evaluates the nonvariational
    response branch.  A later liquid functional must use the returned
    interaction energy as one whole external potential; it must not add the
    old MACE Gaussian electrostatics or Thomas--Fermi control on top of it.
    """

    _validate_route2_mace_calculator(calculator)
    solute_atoms = _neutral_singlet_fragment(solute, name="Solute")
    solvent_atoms = _neutral_singlet_fragment(solvent, name="Solvent")
    cluster_atoms = _combined_neutral_singlet(solute_atoms, solvent_atoms)
    solute_state = _evaluate_zero_field_state(
        calculator,
        solute_atoms,
        compute_forces=compute_forces,
        name="solute fragment",
    )
    solvent_state = _evaluate_zero_field_state(
        calculator,
        solvent_atoms,
        compute_forces=compute_forces,
        name="solvent fragment",
    )
    cluster_state = _evaluate_zero_field_state(
        calculator,
        cluster_atoms,
        compute_forces=compute_forces,
        name="solute--solvent cluster",
    )
    solute_forces = None
    solvent_forces = None
    if compute_forces:
        if (
            solute_state.forces_ev_per_angstrom is None
            or solvent_state.forces_ev_per_angstrom is None
            or cluster_state.forces_ev_per_angstrom is None
        ):
            raise RuntimeError("Zero-field MACE force evaluations are incomplete.")
        solute_size = len(solute_atoms)
        solute_forces = (
            cluster_state.forces_ev_per_angstrom[:solute_size]
            - solute_state.forces_ev_per_angstrom
        )
        solvent_forces = (
            cluster_state.forces_ev_per_angstrom[solute_size:]
            - solvent_state.forces_ev_per_angstrom
        )
    return Route2V0MaceZeroFieldClusterInteraction(
        solute_atomic_numbers=solute_atoms.numbers,
        solute_positions_angstrom=solute_atoms.get_positions(),
        solvent_atomic_numbers=solvent_atoms.numbers,
        solvent_positions_angstrom=solvent_atoms.get_positions(),
        solute_energy_ev=solute_state.energy_ev,
        solvent_energy_ev=solvent_state.energy_ev,
        cluster_energy_ev=cluster_state.energy_ev,
        interaction_energy_ev=(
            cluster_state.energy_ev - solute_state.energy_ev - solvent_state.energy_ev
        ),
        solute_interaction_forces_ev_per_angstrom=solute_forces,
        solvent_interaction_forces_ev_per_angstrom=solvent_forces,
    )


@dataclass(frozen=True)
class Route2V0MaceClusterMolecularExternalPotential(
    Route2V0MolecularExternalPotentialContract
):
    """One configuration grid of whole MACE cluster interaction energies.

    This is intentionally a different source definition from
    :class:`Route2V0MolecularExternalPotential`: the latter contains the
    promolecular Thomas--Fermi plus Gaussian-electrostatic control, while this
    class contains only the exact zero-field MACE cluster difference.  The
    common marker contract lets either named scalar enter the molecular ideal
    and HNC equations, but never permits their implicit addition.
    """

    integration_grid: RegularCartesianGrid
    solute_atomic_numbers: np.ndarray
    solute_atom_positions_angstrom: np.ndarray
    solvent: Route2V0MolecularSolventReference
    configurations: Route2V0MolecularConfigurations
    interactions: tuple[Route2V0MaceZeroFieldClusterInteraction, ...]
    external_potential_hartree: np.ndarray
    solute_interaction_forces_ev_per_angstrom: np.ndarray | None
    construction: str = V0_MACE_CLUSTER_MOLECULAR_EXTERNAL_POTENTIAL_CONSTRUCTION
    interaction_scope: str = V0_MACE_CLUSTER_MOLECULAR_EXTERNAL_POTENTIAL_SCOPE
    coordinate_derivative_policy: str = (
        V0_MACE_CLUSTER_MOLECULAR_EXTERNAL_POTENTIAL_COORDINATE_POLICY
    )

    def __post_init__(self) -> None:
        if not isinstance(self.integration_grid, RegularCartesianGrid):
            raise TypeError(
                "MACE molecular external potential requires a regular grid."
            )
        if not isinstance(self.solvent, Route2V0MolecularSolventReference):
            raise TypeError(
                "MACE molecular external potential requires a solvent reference."
            )
        if not isinstance(self.configurations, Route2V0MolecularConfigurations):
            raise TypeError(
                "MACE molecular external potential requires configurations."
            )
        solute_numbers = _atomic_numbers(
            self.solute_atomic_numbers,
            name="MACE molecular external-potential solute atomic numbers",
        )
        solute_positions = _immutable_array(
            self.solute_atom_positions_angstrom,
            name="MACE molecular external-potential solute positions",
            shape=(solute_numbers.size, 3),
        )
        interaction_values = tuple(self.interactions)
        count = self.configurations.configuration_count
        if len(interaction_values) != count or any(
            not isinstance(value, Route2V0MaceZeroFieldClusterInteraction)
            for value in interaction_values
        ):
            raise ValueError(
                "MACE molecular external potential requires one cluster interaction "
                "for every declared configuration."
            )
        expected_solvent_positions = (
            self.configurations.site_positions_bohr(self.solvent) * Bohr
        )
        for index, interaction in enumerate(interaction_values):
            if not np.array_equal(interaction.solute_atomic_numbers, solute_numbers):
                raise ValueError(
                    "MACE molecular cluster interaction solute atomic numbers differ "
                    "from the declared source."
                )
            if not np.array_equal(
                interaction.solute_positions_angstrom,
                solute_positions,
            ):
                raise ValueError(
                    "MACE molecular cluster interaction solute geometry differs from "
                    "the declared source."
                )
            if not np.array_equal(
                interaction.solvent_atomic_numbers,
                self.solvent.atomic_numbers,
            ):
                raise ValueError(
                    "MACE molecular cluster interaction solvent atomic numbers differ "
                    "from the declared solvent reference."
                )
            if not np.array_equal(
                interaction.solvent_positions_angstrom,
                expected_solvent_positions[index],
            ):
                raise ValueError(
                    "MACE molecular cluster interaction solvent geometry differs from "
                    "its declared configuration."
                )
        external = _immutable_array(
            self.external_potential_hartree,
            name="MACE molecular external potential",
            shape=(count,),
        )
        expected_external = np.asarray(
            [value.interaction_energy_hartree for value in interaction_values],
            dtype=float,
        )
        scale = max(1.0, float(np.max(np.abs(expected_external))))
        if not np.allclose(
            external,
            expected_external,
            rtol=0.0,
            atol=1.0e-12 * scale,
        ):
            raise ValueError(
                "MACE molecular external potential must equal the exact sequence "
                "of cluster interaction energies."
            )
        forces = self.solute_interaction_forces_ev_per_angstrom
        all_forces_available = all(
            interaction.forces_available for interaction in interaction_values
        )
        if forces is None:
            if all_forces_available:
                raise ValueError(
                    "MACE molecular external-potential forces cannot be discarded "
                    "when every cluster interaction provides them."
                )
        else:
            if not all_forces_available:
                raise ValueError(
                    "MACE molecular external-potential forces require every cluster "
                    "interaction force."
                )
            forces = _immutable_array(
                forces,
                name="MACE molecular external-potential solute forces",
                shape=(count, solute_numbers.size, 3),
            )
            expected_forces = np.stack(
                [
                    interaction.solute_interaction_forces_ev_per_angstrom
                    for interaction in interaction_values
                ]
            )
            force_scale = max(1.0, float(np.max(np.abs(expected_forces))))
            if not np.allclose(
                forces,
                expected_forces,
                rtol=0.0,
                atol=1.0e-12 * force_scale,
            ):
                raise ValueError(
                    "MACE molecular external-potential forces must equal the exact "
                    "per-configuration cluster interaction forces."
                )
        if (
            self.construction
            != V0_MACE_CLUSTER_MOLECULAR_EXTERNAL_POTENTIAL_CONSTRUCTION
        ):
            raise ValueError(
                "Unsupported Route-2 V0 MACE molecular external-potential construction."
            )
        if self.interaction_scope != V0_MACE_CLUSTER_MOLECULAR_EXTERNAL_POTENTIAL_SCOPE:
            raise ValueError(
                "Unsupported Route-2 V0 MACE molecular external-potential scope."
            )
        if (
            self.coordinate_derivative_policy
            != V0_MACE_CLUSTER_MOLECULAR_EXTERNAL_POTENTIAL_COORDINATE_POLICY
        ):
            raise ValueError(
                "Unsupported Route-2 V0 MACE molecular external-potential "
                "coordinate-derivative policy."
            )
        object.__setattr__(self, "solute_atomic_numbers", solute_numbers)
        object.__setattr__(self, "solute_atom_positions_angstrom", solute_positions)
        object.__setattr__(self, "interactions", interaction_values)
        object.__setattr__(self, "external_potential_hartree", external)
        object.__setattr__(self, "solute_interaction_forces_ev_per_angstrom", forces)

    @property
    def forces_available(self) -> bool:
        """Return whether every configuration carries a MACE source force."""

        return self.solute_interaction_forces_ev_per_angstrom is not None


def evaluate_route2_v0_mace_cluster_molecular_external_potential(
    *,
    calculator: object,
    integration_grid: RegularCartesianGrid,
    solute_atomic_numbers: np.ndarray,
    solute_atom_positions_angstrom: np.ndarray,
    solvent: Route2V0MolecularSolventReference,
    configurations: Route2V0MolecularConfigurations,
    compute_forces: bool = False,
) -> Route2V0MaceClusterMolecularExternalPotential:
    """Evaluate one whole MACE external-potential vector on a configuration grid.

    Re-evaluating the isolated solvent at each translated/oriented configuration
    is intentional: it makes the three-energy ledger exact for the precise
    MACE inputs used at that quadrature point rather than presuming a numerical
    rotation/translation invariance when caching a fragment energy.
    """

    if not isinstance(integration_grid, RegularCartesianGrid):
        raise TypeError(
            "MACE molecular external potential requires a regular integration grid."
        )
    if not isinstance(solvent, Route2V0MolecularSolventReference):
        raise TypeError(
            "MACE molecular external potential requires a solvent reference."
        )
    if not isinstance(configurations, Route2V0MolecularConfigurations):
        raise TypeError("MACE molecular external potential requires configurations.")
    solute_numbers = _atomic_numbers(
        solute_atomic_numbers,
        name="MACE molecular external-potential solute atomic numbers",
    )
    solute_positions = _immutable_array(
        solute_atom_positions_angstrom,
        name="MACE molecular external-potential solute positions",
        shape=(solute_numbers.size, 3),
    )
    solute = Atoms(
        numbers=solute_numbers,
        positions=solute_positions,
        pbc=False,
        info={"charge": 0.0, "mult": 1},
    )
    interactions: list[Route2V0MaceZeroFieldClusterInteraction] = []
    for solvent_positions_bohr in configurations.site_positions_bohr(solvent):
        solvent_atoms = Atoms(
            numbers=solvent.atomic_numbers,
            positions=solvent_positions_bohr * Bohr,
            pbc=False,
            info={"charge": 0.0, "mult": 1},
        )
        interactions.append(
            evaluate_route2_v0_mace_zero_field_cluster_interaction(
                calculator=calculator,
                solute=solute,
                solvent=solvent_atoms,
                compute_forces=compute_forces,
            )
        )
    external = np.asarray(
        [interaction.interaction_energy_hartree for interaction in interactions],
        dtype=float,
    )
    forces = None
    if compute_forces:
        forces = np.stack(
            [
                interaction.solute_interaction_forces_ev_per_angstrom
                for interaction in interactions
            ]
        )
    return Route2V0MaceClusterMolecularExternalPotential(
        integration_grid=integration_grid,
        solute_atomic_numbers=solute_numbers,
        solute_atom_positions_angstrom=solute_positions,
        solvent=solvent,
        configurations=configurations,
        interactions=tuple(interactions),
        external_potential_hartree=external,
        solute_interaction_forces_ev_per_angstrom=forces,
    )


__all__ = [
    "Route2V0MaceClusterMolecularExternalPotential",
    "Route2V0MaceZeroFieldClusterInteraction",
    "V0_MACE_CLUSTER_EXTERNAL_POTENTIAL_CONSTRUCTION",
    "V0_MACE_CLUSTER_EXTERNAL_POTENTIAL_SCOPE",
    "V0_MACE_CLUSTER_MOLECULAR_EXTERNAL_POTENTIAL_CONSTRUCTION",
    "V0_MACE_CLUSTER_MOLECULAR_EXTERNAL_POTENTIAL_COORDINATE_POLICY",
    "V0_MACE_CLUSTER_MOLECULAR_EXTERNAL_POTENTIAL_SCOPE",
    "evaluate_route2_v0_mace_cluster_molecular_external_potential",
    "evaluate_route2_v0_mace_zero_field_cluster_interaction",
]
