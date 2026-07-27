"""Geometry-mediated AIMNet2 point-charge/ddPCM research objective.

This module implements the scalar

``E_solution(R) = E_AIMNet2(R) + U_ddPCM(R, q(R)) + G_SMD-CDS(R)``

and its complete first derivative.  AIMNet2 charges are recomputed at every
geometry, so geometry relaxation closes an outer geometry--charge--continuum
loop.  This is not fixed-geometry electronic mutual polarization: AIMNet2 has
no reaction-field input and responds only through ``R -> q(R)``.

The implementation remains an explicit research primitive rather than a
public MAPLE solution-phase PES.  It is restricted to neutral closed-shell
single molecules and does not certify smooth optimization, conformer
thermochemistry, or experimental accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase.units import Hartree

from ....route2_smd_profiles import DDPCM_MULTISOLVENT_SMD_PROFILE
from ....route2_solvents import (
    normalize_route2_solvent_name,
    route2_solvent_spec,
)
from .ddpcm_smd import (
    DDPCM_ETA,
    DDPCM_LMAX,
    DDPCM_N_LEBEDEV,
    DDPCM_SOLVER_TOLERANCE,
)
from .pyddx_pcm_response import PyDDXPCMReactionFieldLinearMap
from .pyscf_smd_cds import pyscf_smd_cds
from .route2_domain import validate_route2_domain
from .smd_cds import HARTREE_TO_KCAL_MOL, route2_coulomb_radii
from .source import PointChargeL0Source, solve_fixed_charge_continuum


@dataclass(frozen=True)
class AIMNet2GeometryCoupledState:
    """One same-geometry AIMNet2/ddPCM/SMD-CDS scalar and derivative."""

    positions_angstrom: np.ndarray
    charges_e: np.ndarray
    reaction_potential_ev_per_e: np.ndarray
    solute_energy_ev: float
    polarization_energy_hartree: float
    cds_energy_hartree: float
    solution_energy_ev: float
    intrinsic_gradient_ev_per_angstrom: np.ndarray
    continuum_fixed_source_gradient_ev_per_angstrom: np.ndarray
    charge_response_gradient_ev_per_angstrom: np.ndarray
    cds_gradient_ev_per_angstrom: np.ndarray
    total_gradient_ev_per_angstrom: np.ndarray
    half_coupling_identity_error_ev: float
    continuum_provenance: Mapping[str, object]

    def __post_init__(self) -> None:
        positions = np.asarray(self.positions_angstrom, dtype=float).copy()
        charges = np.asarray(self.charges_e, dtype=float).copy()
        potential = np.asarray(
            self.reaction_potential_ev_per_e,
            dtype=float,
        ).copy()
        if (
            positions.ndim != 2
            or positions.shape[0] == 0
            or positions.shape[1] != 3
            or not np.all(np.isfinite(positions))
        ):
            raise ValueError(
                "Geometry-coupled positions must be finite with shape "
                "(n_atoms, 3)."
            )
        atom_count = positions.shape[0]
        for name, values in (
            ("charges", charges),
            ("reaction potential", potential),
        ):
            if values.shape != (atom_count,) or not np.all(np.isfinite(values)):
                raise ValueError(
                    f"Geometry-coupled {name} must be finite with shape "
                    f"{(atom_count,)}."
                )

        gradient_names = (
            "intrinsic_gradient_ev_per_angstrom",
            "continuum_fixed_source_gradient_ev_per_angstrom",
            "charge_response_gradient_ev_per_angstrom",
            "cds_gradient_ev_per_angstrom",
            "total_gradient_ev_per_angstrom",
        )
        gradients: dict[str, np.ndarray] = {}
        for name in gradient_names:
            values = np.asarray(getattr(self, name), dtype=float).copy()
            if values.shape != positions.shape or not np.all(np.isfinite(values)):
                raise ValueError(
                    f"Geometry-coupled {name} must be finite with shape "
                    f"{positions.shape}."
                )
            values.setflags(write=False)
            gradients[name] = values

        scalars = (
            self.solute_energy_ev,
            self.polarization_energy_hartree,
            self.cds_energy_hartree,
            self.solution_energy_ev,
            self.half_coupling_identity_error_ev,
        )
        if not all(math.isfinite(float(value)) for value in scalars):
            raise ValueError("Geometry-coupled scalar values must be finite.")
        if self.half_coupling_identity_error_ev < 0.0:
            raise ValueError(
                "Half-coupling identity error must be nonnegative."
            )
        expected_total = (
            self.solute_energy_ev
            + self.polarization_energy_hartree * Hartree
            + self.cds_energy_hartree * Hartree
        )
        if abs(expected_total - self.solution_energy_ev) > 1.0e-9:
            raise ValueError(
                "Geometry-coupled solution energy does not close its scalar "
                "ledger."
            )
        expected_gradient = sum(
            (
                gradients["intrinsic_gradient_ev_per_angstrom"],
                gradients[
                    "continuum_fixed_source_gradient_ev_per_angstrom"
                ],
                gradients["charge_response_gradient_ev_per_angstrom"],
                gradients["cds_gradient_ev_per_angstrom"],
            ),
            start=np.zeros_like(positions),
        )
        if not np.allclose(
            expected_gradient,
            gradients["total_gradient_ev_per_angstrom"],
            rtol=0.0,
            atol=2.0e-10,
        ):
            raise ValueError(
                "Geometry-coupled total gradient does not close its component "
                "ledger."
            )

        positions.setflags(write=False)
        charges.setflags(write=False)
        potential.setflags(write=False)
        object.__setattr__(self, "positions_angstrom", positions)
        object.__setattr__(self, "charges_e", charges)
        object.__setattr__(
            self,
            "reaction_potential_ev_per_e",
            potential,
        )
        for name, values in gradients.items():
            object.__setattr__(self, name, values)
        object.__setattr__(
            self,
            "continuum_provenance",
            MappingProxyType(dict(self.continuum_provenance)),
        )

    @property
    def polarization_energy_kcal_mol(self) -> float:
        return self.polarization_energy_hartree * HARTREE_TO_KCAL_MOL

    @property
    def cds_energy_kcal_mol(self) -> float:
        return self.cds_energy_hartree * HARTREE_TO_KCAL_MOL

    def solvation_energy_kcal_mol(
        self,
        *,
        gas_reference_energy_ev: float,
    ) -> float:
        """Return the geometry-relaxed ledger against one frozen gas reference."""

        reference = float(gas_reference_energy_ev)
        if not math.isfinite(reference):
            raise ValueError("Gas reference energy must be finite.")
        return (
            (self.solution_energy_ev - reference)
            / Hartree
            * HARTREE_TO_KCAL_MOL
        )


class AIMNet2GeometryCoupledObjective:
    """Evaluate the neutral AIMNet2 geometry--charge--ddPCM scalar."""

    def __init__(
        self,
        reference_atoms,
        aimnet2_calculator,
        *,
        solvent: str,
        profile: str = DDPCM_MULTISOLVENT_SMD_PROFILE,
        lmax: int = DDPCM_LMAX,
        n_lebedev: int = DDPCM_N_LEBEDEV,
        n_proc: int = 1,
        solver_tolerance: float = DDPCM_SOLVER_TOLERANCE,
        eta: float = DDPCM_ETA,
        energy_identity_tolerance_ev: float = 1.0e-8,
    ) -> None:
        validate_route2_domain(reference_atoms)
        charge = float(reference_atoms.info.get("charge", 0.0))
        multiplicity = float(reference_atoms.info.get("mult", 1.0))
        if charge != 0.0 or multiplicity != 1.0:
            raise NotImplementedError(
                "Geometry-mediated AIMNet2/ddPCM is restricted to neutral "
                "closed-shell molecules."
            )
        required = ("charge_state", "charge_position_response")
        missing = [
            name
            for name in required
            if not callable(getattr(aimnet2_calculator, name, None))
        ]
        if missing:
            raise TypeError(
                "Geometry-mediated AIMNet2/ddPCM requires AIMNet2 charge "
                "and coordinate-response APIs; missing: "
                + ", ".join(missing)
                + "."
            )
        solvent_name = normalize_route2_solvent_name(solvent)
        solvent_spec = route2_solvent_spec(solvent_name)
        self._numbers = np.asarray(reference_atoms.numbers, dtype=int).copy()
        self._symbols = tuple(reference_atoms.get_chemical_symbols())
        self._charge = charge
        self._multiplicity = multiplicity
        self._calculator = aimnet2_calculator
        self.solvent = solvent_name
        self.profile = str(profile).strip().lower()
        self.dielectric = float(solvent_spec.descriptors.dielectric)
        self.radii_angstrom = route2_coulomb_radii(
            self._symbols,
            solvent=self.solvent,
            profile=self.profile,
        )
        self.lmax = self._positive_integer(lmax, name="lmax")
        self.n_lebedev = self._positive_integer(
            n_lebedev,
            name="n_lebedev",
        )
        self.n_proc = self._positive_integer(n_proc, name="n_proc")
        self.solver_tolerance = float(solver_tolerance)
        self.eta = float(eta)
        self.energy_identity_tolerance_ev = float(
            energy_identity_tolerance_ev
        )

    @staticmethod
    def _positive_integer(value: int, *, name: str) -> int:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, np.integer))
            or int(value) < 1
        ):
            raise ValueError(f"{name} must be a positive integer.")
        return int(value)

    def _validate_atoms(self, atoms) -> None:
        validate_route2_domain(atoms)
        if not np.array_equal(
            np.asarray(atoms.numbers, dtype=int),
            self._numbers,
        ):
            raise ValueError(
                "Geometry-mediated AIMNet2/ddPCM does not permit atom "
                "identity or order changes."
            )
        if float(atoms.info.get("charge", 0.0)) != self._charge:
            raise ValueError(
                "Geometry-mediated AIMNet2/ddPCM does not permit charge "
                "changes."
            )
        if float(atoms.info.get("mult", 1.0)) != self._multiplicity:
            raise ValueError(
                "Geometry-mediated AIMNet2/ddPCM does not permit "
                "multiplicity changes."
            )

    def evaluate(self, atoms) -> AIMNet2GeometryCoupledState:
        self._validate_atoms(atoms)
        positions = np.asarray(atoms.get_positions(), dtype=float)
        charge_state = self._calculator.charge_state(atoms)
        source = PointChargeL0Source(
            charges_e=charge_state.charges_e,
            declared_total_charge_e=self._charge,
            source_model="aimnet2-nqe-geometry-adaptive",
        )
        reaction_field = PyDDXPCMReactionFieldLinearMap(
            positions,
            self.radii_angstrom,
            dielectric=self.dielectric,
            lmax=self.lmax,
            n_lebedev=self.n_lebedev,
            n_proc=self.n_proc,
            solver_tolerance=self.solver_tolerance,
            eta=self.eta,
        )
        continuum = solve_fixed_charge_continuum(
            reaction_field,
            source,
            energy_identity_tolerance_ev=(
                self.energy_identity_tolerance_ev
            ),
        )
        reaction_potential = np.asarray(
            continuum.reaction_field_values_ev[:, 0],
            dtype=float,
        )
        response = self._calculator.charge_position_response(
            atoms,
            reaction_potential,
        )
        if not np.allclose(
            response.charge_state.charges_e,
            source.charges_e,
            rtol=0.0,
            atol=5.0e-7,
        ):
            raise RuntimeError(
                "AIMNet2 repeated charge evaluation changed at fixed geometry."
            )
        if (
            abs(response.charge_state.energy_ev - charge_state.energy_ev)
            > 1.0e-6
        ):
            raise RuntimeError(
                "AIMNet2 repeated energy evaluation changed at fixed geometry."
            )

        continuum_fixed_gradient = (
            reaction_field.polarization_position_gradient_ev_per_angstrom(
                source.continuum_multipole_coefficients
            )
        )
        cds = pyscf_smd_cds(
            self._symbols,
            positions,
            solvent=self.solvent,
        )
        cds_gradient_ev = (
            cds.position_gradient_hartree_per_angstrom * Hartree
        )
        intrinsic_gradient = (
            response.intrinsic_energy_gradient_ev_per_angstrom
        )
        charge_response_gradient = (
            response.charge_position_vjp_ev_per_angstrom
        )
        total_gradient = (
            intrinsic_gradient
            + continuum_fixed_gradient
            + charge_response_gradient
            + cds_gradient_ev
        )
        solution_energy_ev = (
            response.charge_state.energy_ev
            + continuum.polarization_energy_hartree * Hartree
            + cds.energy_hartree * Hartree
        )
        return AIMNet2GeometryCoupledState(
            positions_angstrom=positions,
            charges_e=source.charges_e,
            reaction_potential_ev_per_e=reaction_potential,
            solute_energy_ev=response.charge_state.energy_ev,
            polarization_energy_hartree=(
                continuum.polarization_energy_hartree
            ),
            cds_energy_hartree=cds.energy_hartree,
            solution_energy_ev=solution_energy_ev,
            intrinsic_gradient_ev_per_angstrom=intrinsic_gradient,
            continuum_fixed_source_gradient_ev_per_angstrom=(
                continuum_fixed_gradient
            ),
            charge_response_gradient_ev_per_angstrom=(
                charge_response_gradient
            ),
            cds_gradient_ev_per_angstrom=cds_gradient_ev,
            total_gradient_ev_per_angstrom=total_gradient,
            half_coupling_identity_error_ev=(
                continuum.energy_identity_error_ev
            ),
            continuum_provenance=reaction_field.runtime_provenance,
        )


class AIMNet2GeometryCoupledASECalculator(Calculator):
    """ASE bridge for bounded research optimization of the coupled scalar."""

    implemented_properties = ("energy", "forces")

    def __init__(
        self,
        objective: AIMNet2GeometryCoupledObjective,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.objective = objective
        self.last_state: AIMNet2GeometryCoupledState | None = None

    def calculate(
        self,
        atoms=None,
        properties=("energy", "forces"),
        system_changes=all_changes,
    ) -> None:
        super().calculate(atoms, properties, system_changes)
        state = self.objective.evaluate(self.atoms)
        self.last_state = state
        self.results = {
            "energy": state.solution_energy_ev,
            "forces": -state.total_gradient_ev_per_angstrom.copy(),
        }


__all__ = [
    "AIMNet2GeometryCoupledASECalculator",
    "AIMNet2GeometryCoupledObjective",
    "AIMNet2GeometryCoupledState",
]
