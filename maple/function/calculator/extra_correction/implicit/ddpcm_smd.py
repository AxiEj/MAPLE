"""Self-consistent MACE-POLAR/ddPCM/SMD research force candidate.

This provider is deliberately separate from the public PCMSolver/GePol energy
proof of concept.  One pyddx ddPCM object owns the scalar polarization energy,
reaction-field forward/adjoint maps, and complete coordinate derivative.  The
official PySCF SMD CDS entrypoint supplies its scalar energy and matching
analytic gradient.  Components from the two continuum providers are never
mixed.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from ase.units import Hartree

from ....route2_smd_profiles import (
    DDPCM_GAFF2_CARBONYL_O_MACE_KSPACE40_PROFILE,
    DDPCM_GAFF2_CARBONYL_O_PROFILE,
    DDPCM_SMD_PROFILE,
    MACEPOL_FORCED_RECIPROCAL_FIXED_BOX40_PROFILE,
    SUPPORTED_DDPCM_SMD_PROFILES,
    route2_smd_profile_spec,
)
from ...calculator_base import ROUTE2_SMD_CALCULATOR_PROFILE
from .gto_density import external_field_to_density_order
from .pyddx_pcm_response import PyDDXPCMReactionFieldLinearMap
from .pyscf_smd_cds import pyscf_smd_water_cds
from .result import SolvationResult
from .route2_derivative import (
    assemble_total_solvation_coordinate_gradient,
    continuum_coupled_solvation_coordinate_gradient,
    fixed_cavity_energy_density_gradient,
)
from .route2_domain import validate_route2_domain
from .route2_response import (
    UnmixedDensityResidualLinearization,
    solve_adjoint,
)
from .smd_cds import route2_water_coulomb_radii


WATER_STATIC_DIELECTRIC = 78.39
DDPCM_LMAX = 15
DDPCM_N_LEBEDEV = 1202
DDPCM_SOLVER_TOLERANCE = 1.0e-12
DDPCM_ETA = 0.1
SCF_MIXING = 1.0
SCF_DENSITY_TOLERANCE = 2.0e-12
SCF_ENERGY_TOLERANCE_EV = 1.0e-10
SCF_MAX_ITERATIONS = 100
ADJOINT_RELATIVE_TOLERANCE = 1.0e-10
ADJOINT_ABSOLUTE_TOLERANCE = 1.0e-13
ADJOINT_MAX_ITERATIONS = 100
ENERGY_IDENTITY_TOLERANCE_EV = 2.0e-10
FORCE_STATE_ENERGY_TOLERANCE_EV = 1.0e-9
NEUTRAL_DENSITY_TOLERANCE = 1.0e-8


@dataclass(frozen=True)
class _CoupledState:
    calculator_identity: int
    atomic_numbers: np.ndarray
    positions_angstrom: np.ndarray
    reaction_field: Any
    density_coefficients: np.ndarray
    reaction_field_values_ev: np.ndarray
    solvent_state: Any
    polarization_energy_hartree: float
    energy_identity_error_ev: float
    cds_result: Any
    history: tuple[dict[str, float | int | None], ...]

    def matches(self, calculator, atoms) -> bool:
        return (
            self.calculator_identity == id(calculator)
            and np.array_equal(
                self.atomic_numbers,
                np.asarray(atoms.numbers, dtype=int),
            )
            and np.array_equal(
                self.positions_angstrom,
                np.asarray(atoms.get_positions(), dtype=float),
            )
        )


@dataclass
class DDPCMSMDImplicitSolvation:
    """Aqueous SMD correction with a complete ddPCM response derivative."""

    atoms: Any
    solvation_options: dict[str, Any]
    audit_dir: Path | None = None

    supported_properties = frozenset({"energy", "forces"})

    def __post_init__(self) -> None:
        self.solvation_options = dict(self.solvation_options)
        if "profile" not in self.solvation_options:
            raise ValueError(
                "Route 2 provider=pyddx requires an explicit "
                "versioned profile."
            )
        self.provider = str(
            self.solvation_options.get("provider", "")
        ).lower()
        self.profile = str(
            self.solvation_options["profile"]
        ).lower()
        self.response = str(
            self.solvation_options.get("response", "scf")
        ).lower()
        self.standard_state = str(
            self.solvation_options.get("standard_state", "1m")
        ).lower()
        self._validate_options()
        self.profile_spec = route2_smd_profile_spec(self.profile)
        validate_route2_domain(self.atoms)
        self._reference_numbers = np.asarray(
            self.atoms.numbers,
            dtype=int,
        ).copy()

        mol2 = self.atoms.info.get("mol2")
        atom_types = (
            mol2.get("atom_types")
            if isinstance(mol2, dict)
            else None
        )
        self.coulomb_radii_angstrom = route2_water_coulomb_radii(
            self.atoms.get_chemical_symbols(),
            atom_types=atom_types,
            profile=self.profile,
        )
        self._cached_state: _CoupledState | None = None

        if self.audit_dir is not None:
            self.audit_dir = Path(self.audit_dir).resolve()
            self.audit_dir.mkdir(parents=True, exist_ok=True)

        self.provenance = {
            "provider": "pyddx",
            "method": "smd",
            "profile": self.profile,
            "solvent": "water",
            "response": "scf",
            "standard_state": "1M(gas)->1M(solution)",
            "standard_state_correction_hartree": 0.0,
            "density_source": (
                "official MACE-POLAR-1-M l<=1 residual charge density"
            ),
            "density_interpretation": (
                "coarse-grained net charge density, not a QM electron density"
            ),
            "electrostatics": "ddPCM",
            "pcm_projection": "atom-centred l<=1 real spherical multipoles",
            "cavity_radii": (
                "SMD Coulomb radii with revised Br=2.60 A and I=2.74 A"
                if not self.profile_spec.uses_gaff2_carbonyl_oxygen
                else (
                    "SMD Coulomb radii with GAFF/GAFF2 carbonyl oxygen "
                    "(atom type o) overridden to 1.70 A"
                )
            ),
            "mace_long_range_evaluator": (
                self.profile_spec.mace_long_range_evaluator
            ),
            "mace_long_range_evaluator_status": (
                "experimental fixed-box operator variant; not equivalent "
                "to the default molecular real-space evaluator"
                if self.profile_spec.mace_long_range_evaluator
                == MACEPOL_FORCED_RECIPROCAL_FIXED_BOX40_PROFILE
                else "official default molecular real-space evaluator"
            ),
            "cds": "official PySCF water-SMD libsolvent energy and gradient",
            "route_role": "research-innovation",
            "scientific_status": "single-point-force-candidate",
            "solution_phase_pes": False,
            "forces_available": True,
            "accuracy_certified": False,
            "default_eligible": False,
            "energy_composition": (
                "delta_G_solv = (E_MACE_intrinsic[V_reac]-E_MACE_gas) "
                "+ E_ddPCM + G_CDS"
            ),
            "force_composition": (
                "F_solution = F_MACE_gas "
                "- d(delta_G_solv)/dR, with the converged-density "
                "response eliminated by one adjoint solve"
            ),
            "numerics": {
                "dielectric": WATER_STATIC_DIELECTRIC,
                "lmax": DDPCM_LMAX,
                "n_lebedev": DDPCM_N_LEBEDEV,
                "ddpcm_n_proc": self.profile_spec.ddpcm_n_proc,
                "ddpcm_solver_tolerance": DDPCM_SOLVER_TOLERANCE,
                "ddpcm_eta": DDPCM_ETA,
                "scf_mixing": SCF_MIXING,
                "scf_density_tolerance_e": SCF_DENSITY_TOLERANCE,
                "scf_energy_tolerance_ev": SCF_ENERGY_TOLERANCE_EV,
                "scf_maximum_iterations": SCF_MAX_ITERATIONS,
                "adjoint_relative_tolerance": (
                    ADJOINT_RELATIVE_TOLERANCE
                ),
                "adjoint_absolute_tolerance": (
                    ADJOINT_ABSOLUTE_TOLERANCE
                ),
                "adjoint_maximum_iterations": ADJOINT_MAX_ITERATIONS,
            },
        }
        self._write_manifest()

    def _validate_options(self) -> None:
        if self.solvation_options.get("experimental") is not True:
            raise ValueError(
                "The ddPCM Route-2 force candidate requires "
                "experimental=true explicitly."
            )
        if str(self.solvation_options.get("method", "")).lower() != "smd":
            raise ValueError("DDPCMSMDImplicitSolvation requires method=smd.")
        if str(self.solvation_options.get("implicit", "")).lower() != "water":
            raise ValueError(
                "The ddPCM Route-2 force candidate supports "
                "implicit=water only."
            )
        if self.provider != "pyddx":
            raise ValueError(
                "DDPCMSMDImplicitSolvation requires provider=pyddx."
            )
        if self.profile not in SUPPORTED_DDPCM_SMD_PROFILES:
            supported = ", ".join(sorted(SUPPORTED_DDPCM_SMD_PROFILES))
            raise ValueError(
                "The pyddx Route-2 profile must be one of: "
                f"{supported}."
            )
        if self.response != "scf":
            raise ValueError(
                "The ddPCM Route-2 force candidate requires response=scf."
            )
        if self.standard_state != "1m":
            raise ValueError(
                "Route 2 uses the 1 M gas -> 1 M solution convention only; "
                "standard_state must be 1m."
            )
        if "cavity_policy" in self.solvation_options:
            raise ValueError(
                "cavity_policy is specific to the PCMSolver/GePol provider "
                "and is not valid for provider=pyddx."
            )

    def _write_manifest(self) -> None:
        if self.audit_dir is None:
            return
        manifest = {
            "schema_version": 1,
            "solvation_options": self.solvation_options,
            "provenance": self.provenance,
            "elements": self.atoms.get_chemical_symbols(),
            "positions_angstrom": np.asarray(
                self.atoms.get_positions(),
                dtype=float,
            ).tolist(),
        }
        (self.audit_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def _validate_atoms(self, atoms) -> None:
        validate_route2_domain(atoms)
        numbers = np.asarray(atoms.numbers, dtype=int)
        if not np.array_equal(numbers, self._reference_numbers):
            raise ValueError(
                "Route 2 does not permit atom identity/order changes."
            )

    def _validate_calculator(self, calculator, *, need_forces: bool) -> None:
        if calculator is None or not callable(
            getattr(calculator, "polar_state", None)
        ):
            raise TypeError(
                "The ddPCM Route-2 provider requires the "
                "MACEPolCalculator polar_state() API."
            )
        if (
            getattr(calculator, "route2_smd_profile", None)
            != ROUTE2_SMD_CALCULATOR_PROFILE
        ):
            raise TypeError(
                "Route 2 requires the official MACE-POLAR-1-M float64 "
                "local-field calculator profile."
            )
        expected_evaluator = self.profile_spec.mace_long_range_evaluator
        actual_evaluator = getattr(
            calculator,
            "long_range_evaluator_profile",
            None,
        )
        if actual_evaluator != expected_evaluator:
            raise TypeError(
                "The selected Route-2 profile requires MACE-POLAR "
                f"long-range evaluator {expected_evaluator!r}; received "
                f"{actual_evaluator!r}."
            )
        if not need_forces:
            return
        required = (
            "intrinsic_energy_field_gradient",
            "linearize_density_response",
            "density_position_vjp",
        )
        missing = [
            name
            for name in required
            if not callable(getattr(calculator, name, None))
        ]
        if missing:
            raise TypeError(
                "The ddPCM Route-2 force candidate requires the complete "
                "MACE response API; missing: "
                + ", ".join(missing)
                + "."
            )

    @staticmethod
    def _validate_density(
        values: np.ndarray,
        atom_count: int,
        *,
        name: str,
    ) -> np.ndarray:
        density = np.asarray(values, dtype=float)
        expected_shape = (atom_count, 4)
        if density.shape != expected_shape or not np.all(np.isfinite(density)):
            raise RuntimeError(
                f"{name} must be finite with shape {expected_shape}; "
                f"received {density.shape}."
            )
        monopole_sum = float(np.sum(density[:, 0]))
        if abs(monopole_sum) > NEUTRAL_DENSITY_TOLERANCE:
            raise RuntimeError(
                f"{name} violates the neutral charge constraint "
                f"(sum={monopole_sum:.6e} e)."
            )
        return density.copy()

    @staticmethod
    def _validate_field(
        values: np.ndarray,
        atom_count: int,
    ) -> np.ndarray:
        field = np.asarray(values, dtype=float)
        expected_shape = (atom_count, 4)
        if field.shape != expected_shape or not np.all(np.isfinite(field)):
            raise RuntimeError(
                "The ddPCM reaction field must be finite with shape "
                f"{expected_shape}; received {field.shape}."
            )
        return field.copy()

    @staticmethod
    def _gas_state(calculator, atoms, *, need_forces: bool):
        cached = getattr(calculator, "cached_polar_state", None)
        if callable(cached):
            state = cached(atoms, require_forces=need_forces)
        else:
            state = getattr(calculator, "_last_polar_state", None)
        if state is None or (
            need_forces
            and getattr(
                state,
                "fixed_field_forces_ev_per_angstrom",
                None,
            )
            is None
        ):
            state, _ = calculator.polar_state(
                atoms,
                compute_forces=need_forces,
            )
        return state

    def _build_reaction_field(self, atoms):
        return PyDDXPCMReactionFieldLinearMap(
            np.asarray(atoms.get_positions(), dtype=float),
            self.coulomb_radii_angstrom,
            dielectric=WATER_STATIC_DIELECTRIC,
            lmax=DDPCM_LMAX,
            n_lebedev=DDPCM_N_LEBEDEV,
            n_proc=self.profile_spec.ddpcm_n_proc,
            solver_tolerance=DDPCM_SOLVER_TOLERANCE,
            eta=DDPCM_ETA,
        )

    def _solve_coupled_state(self, atoms, calculator, gas_state) -> _CoupledState:
        reaction_field = self._build_reaction_field(atoms)
        density = self._validate_density(
            gas_state.density_coefficients,
            len(atoms),
            name="Gas MACE-POLAR density",
        )
        previous_energy_ev: float | None = None
        history: list[dict[str, float | int | None]] = []

        for iteration in range(1, SCF_MAX_ITERATIONS + 1):
            field = self._validate_field(
                reaction_field.apply_scf(density),
                len(atoms),
            )
            solvent_state, _ = calculator.polar_state(
                atoms,
                node_potential_ev=field[:, 0],
                node_gradient_ev_per_angstrom=field[:, 1:],
            )
            response_density = self._validate_density(
                solvent_state.density_coefficients,
                len(atoms),
                name="Field-polarized MACE-POLAR density",
            )
            density_residual = float(
                np.max(np.abs(response_density - density))
            )
            current_energy_ev = float(solvent_state.energy_ev)
            if not math.isfinite(current_energy_ev):
                raise RuntimeError(
                    "Field-polarized MACE-POLAR energy is non-finite."
                )
            energy_residual = (
                None
                if previous_energy_ev is None
                else abs(current_energy_ev - previous_energy_ev)
            )
            history.append(
                {
                    "iteration": iteration,
                    "density_residual_e": density_residual,
                    "energy_residual_ev": energy_residual,
                    "intrinsic_energy_ev": current_energy_ev,
                }
            )
            energy_converged = (
                energy_residual is None
                or energy_residual <= SCF_ENERGY_TOLERANCE_EV
            )
            if (
                density_residual <= SCF_DENSITY_TOLERANCE
                and energy_converged
            ):
                break
            density = (
                (1.0 - SCF_MIXING) * density
                + SCF_MIXING * response_density
            )
            previous_energy_ev = current_energy_ev
        else:
            last = history[-1]
            raise RuntimeError(
                "MACE-POLAR/ddPCM reaction-field SCF did not converge in "
                f"{SCF_MAX_ITERATIONS} iterations "
                f"(density residual={last['density_residual_e']:.3e} e, "
                f"energy residual={last['energy_residual_ev']!r} eV)."
            )

        polarization_energy_hartree = float(
            reaction_field.scf_polarization_energy_hartree(density)
        )
        if not math.isfinite(polarization_energy_hartree):
            raise RuntimeError("ddPCM polarization energy is non-finite.")
        paired_energy_ev = 0.5 * float(
            np.vdot(
                density,
                external_field_to_density_order(field),
            )
        )
        provider_energy_ev = polarization_energy_hartree * Hartree
        identity_error_ev = abs(paired_energy_ev - provider_energy_ev)
        if identity_error_ev > ENERGY_IDENTITY_TOLERANCE_EV:
            raise RuntimeError(
                "ddPCM reaction field failed the polarization-energy "
                f"identity (absolute error={identity_error_ev:.3e} eV)."
            )

        cds_result = pyscf_smd_water_cds(
            atoms.get_chemical_symbols(),
            np.asarray(atoms.get_positions(), dtype=float),
        )
        return _CoupledState(
            calculator_identity=id(calculator),
            atomic_numbers=np.asarray(atoms.numbers, dtype=int).copy(),
            positions_angstrom=np.asarray(
                atoms.get_positions(),
                dtype=float,
            ).copy(),
            reaction_field=reaction_field,
            density_coefficients=density,
            reaction_field_values_ev=field,
            solvent_state=solvent_state,
            polarization_energy_hartree=polarization_energy_hartree,
            energy_identity_error_ev=identity_error_ev,
            cds_result=cds_result,
            history=tuple(history),
        )

    def _coupled_state(self, atoms, calculator, gas_state) -> _CoupledState:
        cached = self._cached_state
        if cached is not None and cached.matches(calculator, atoms):
            return cached
        state = self._solve_coupled_state(
            atoms,
            calculator,
            gas_state,
        )
        self._cached_state = state
        return state

    def _solvent_correction_force(
        self,
        atoms,
        calculator,
        gas_state,
        coupled: _CoupledState,
    ):
        field = coupled.reaction_field_values_ev
        density = coupled.density_coefficients
        solvent_state, _ = calculator.polar_state(
            atoms,
            node_potential_ev=field[:, 0],
            node_gradient_ev_per_angstrom=field[:, 1:],
            compute_forces=True,
        )
        response_density = self._validate_density(
            solvent_state.density_coefficients,
            len(atoms),
            name="Force-evaluation MACE-POLAR density",
        )
        force_state_residual = float(
            np.max(np.abs(response_density - density))
        )
        if force_state_residual > max(
            10.0 * SCF_DENSITY_TOLERANCE,
            1.0e-10,
        ):
            raise RuntimeError(
                "The MACE-POLAR force state does not match the converged "
                f"density root (residual={force_state_residual:.3e} e)."
            )
        force_state_energy_error_ev = abs(
            float(solvent_state.energy_ev)
            - float(coupled.solvent_state.energy_ev)
        )
        if (
            not math.isfinite(force_state_energy_error_ev)
            or force_state_energy_error_ev
            > FORCE_STATE_ENERGY_TOLERANCE_EV
        ):
            raise RuntimeError(
                "The MACE-POLAR force evaluation does not reproduce the "
                "converged intrinsic energy "
                f"(absolute error={force_state_energy_error_ev:.3e} eV)."
            )

        gas_forces = getattr(
            gas_state,
            "fixed_field_forces_ev_per_angstrom",
            None,
        )
        solvent_forces = getattr(
            solvent_state,
            "fixed_field_forces_ev_per_angstrom",
            None,
        )
        if gas_forces is None or solvent_forces is None:
            raise RuntimeError(
                "MACE-POLAR omitted the gas or fixed-field force partial."
            )

        intrinsic_gradient = (
            calculator.intrinsic_energy_field_gradient(
                atoms,
                node_potential_ev=field[:, 0],
                node_gradient_ev_per_angstrom=field[:, 1:],
            )
        )
        density_response = calculator.linearize_density_response(
            atoms,
            node_potential_ev=field[:, 0],
            node_gradient_ev_per_angstrom=field[:, 1:],
        )
        physical_rhs = fixed_cavity_energy_density_gradient(
            coupled.reaction_field,
            reaction_field_values=field,
            intrinsic_energy_field_gradient=intrinsic_gradient,
        )
        residual = UnmixedDensityResidualLinearization(
            atom_count=len(atoms),
            reaction_field=coupled.reaction_field,
            density_response=density_response,
        )
        adjoint = solve_adjoint(
            residual,
            physical_rhs,
            relative_tolerance=ADJOINT_RELATIVE_TOLERANCE,
            absolute_tolerance=ADJOINT_ABSOLUTE_TOLERANCE,
            max_iterations=ADJOINT_MAX_ITERATIONS,
        )
        density_position_vjp = calculator.density_position_vjp(
            atoms,
            node_potential_ev=field[:, 0],
            node_gradient_ev_per_angstrom=field[:, 1:],
            density_cotangent=adjoint.solution,
        )
        continuum_gradient = (
            continuum_coupled_solvation_coordinate_gradient(
                coupled.reaction_field,
                density_response,
                density_coefficients=density,
                intrinsic_energy_field_gradient=intrinsic_gradient,
                adjoint_solution=adjoint.solution,
                adjoint_density_position_vjp=density_position_vjp,
                solvent_fixed_field_forces_ev_per_angstrom=(
                    solvent_forces
                ),
                gas_forces_ev_per_angstrom=gas_forces,
            )
        )
        total = assemble_total_solvation_coordinate_gradient(
            continuum_gradient,
            coupled.cds_result.position_gradient_hartree_per_angstrom,
        )
        derivative = {
            "continuum_position_gradient_ev_per_angstrom": (
                continuum_gradient
            ),
            "cds_position_gradient_hartree_per_angstrom": (
                coupled.cds_result.position_gradient_hartree_per_angstrom
            ),
            "total_position_gradient_hartree_per_angstrom": (
                total.total_position_gradient_hartree_per_angstrom
            ),
            "solvent_correction_forces_hartree_per_angstrom": (
                total.solvent_correction_forces_hartree_per_angstrom
            ),
            "force_state_density_residual_e": force_state_residual,
            "force_state_energy_error_ev": force_state_energy_error_ev,
            "adjoint": {
                "method": adjoint.method,
                "relative_tolerance": ADJOINT_RELATIVE_TOLERANCE,
                "absolute_tolerance": ADJOINT_ABSOLUTE_TOLERANCE,
                "restart_size": adjoint.restart_size,
                "maximum_inner_iterations": (
                    adjoint.maximum_inner_iterations
                ),
                "operator_applications": adjoint.operator_applications,
                "residual_callback_count": (
                    adjoint.residual_callback_count
                ),
                "residual_norm": adjoint.residual_norm,
                "relative_residual": adjoint.relative_residual,
            },
        }
        return (
            total.solvent_correction_forces_hartree_per_angstrom,
            derivative,
        )

    def _write_result_audit(
        self,
        *,
        atoms,
        gas_state,
        coupled: _CoupledState,
        components: dict[str, float],
        derivative: dict[str, Any] | None,
    ) -> None:
        if self.audit_dir is None:
            return
        arrays: dict[str, np.ndarray] = {
            "positions_angstrom": np.asarray(
                atoms.get_positions(),
                dtype=float,
            ),
            "cavity_radii_angstrom": np.asarray(
                self.coulomb_radii_angstrom,
                dtype=float,
            ),
            "density_coefficients": coupled.density_coefficients,
            "reaction_field_values_ev": (
                coupled.reaction_field_values_ev
            ),
        }
        if derivative is not None:
            arrays.update(
                {
                    key: np.asarray(value, dtype=float)
                    for key, value in derivative.items()
                    if key != "adjoint"
                }
            )
        state_path = self.audit_dir / "route2-ddpcm-state.npz"
        np.savez_compressed(state_path, **arrays)

        payload = {
            "schema_version": 1,
            "converged": True,
            "forces_evaluated": derivative is not None,
            "profile": self.profile,
            "energies_hartree": components,
            "gas_mace_energy_ev": float(gas_state.energy_ev),
            "solvent_intrinsic_mace_energy_ev": float(
                coupled.solvent_state.energy_ev
            ),
            "polarization_energy_identity_error_ev": (
                coupled.energy_identity_error_ev
            ),
            "scf": {
                "mixing": SCF_MIXING,
                "density_tolerance_e": SCF_DENSITY_TOLERANCE,
                "energy_tolerance_ev": SCF_ENERGY_TOLERANCE_EV,
                "maximum_iterations": SCF_MAX_ITERATIONS,
                "iterations": len(coupled.history),
                "history": list(coupled.history),
            },
            "providers": {
                "continuum": dict(
                    coupled.reaction_field.runtime_provenance
                ),
                "cds": dict(coupled.cds_result.runtime_provenance),
            },
            "adjoint": (
                None if derivative is None else derivative["adjoint"]
            ),
            "array_archive": str(state_path),
        }
        (self.audit_dir / "route2-ddpcm-result.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def evaluate(
        self,
        atoms,
        need_forces: bool = False,
        calculator=None,
    ) -> SolvationResult:
        self._validate_atoms(atoms)
        self._validate_calculator(
            calculator,
            need_forces=need_forces,
        )
        gas_state = self._gas_state(
            calculator,
            atoms,
            need_forces=need_forces,
        )
        self._validate_density(
            gas_state.density_coefficients,
            len(atoms),
            name="Gas MACE-POLAR density",
        )
        coupled = self._coupled_state(
            atoms,
            calculator,
            gas_state,
        )

        delta_e_solute = (
            float(coupled.solvent_state.energy_ev)
            - float(gas_state.energy_ev)
        ) / Hartree
        pcm_polarization = coupled.polarization_energy_hartree
        electrostatic = delta_e_solute + pcm_polarization
        cds_energy = float(coupled.cds_result.energy_hartree)
        total_energy = electrostatic + cds_energy
        components = {
            "solute_polarization": delta_e_solute,
            "pcm_polarization": pcm_polarization,
            "electrostatic": electrostatic,
            "cds": cds_energy,
            "standard_state": 0.0,
            "delta_g_solv": total_energy,
        }

        correction_forces = None
        derivative = None
        if need_forces:
            correction_forces, derivative = (
                self._solvent_correction_force(
                    atoms,
                    calculator,
                    gas_state,
                    coupled,
                )
            )
        self._write_result_audit(
            atoms=atoms,
            gas_state=gas_state,
            coupled=coupled,
            components=components,
            derivative=derivative,
        )

        runtime_provenance = {
            **self.provenance,
            "converged": True,
            "iterations": len(coupled.history),
            "continuum_provider": dict(
                coupled.reaction_field.runtime_provenance
            ),
            "cds_provider": dict(
                coupled.cds_result.runtime_provenance
            ),
            "calculator_profile": getattr(
                calculator,
                "route2_smd_profile",
                None,
            ),
            "mace_torch_version": getattr(
                calculator,
                "mace_torch_version",
                None,
            ),
            "mace_dtype": str(getattr(calculator, "dtype", None)),
            "mace_long_range_evaluator": dict(
                getattr(
                    calculator,
                    "long_range_evaluator_provenance",
                    {},
                )
            ),
            "audit_directory": (
                None
                if self.audit_dir is None
                else str(self.audit_dir)
            ),
        }
        return SolvationResult(
            energy_hartree=total_energy,
            forces_hartree_per_angstrom=correction_forces,
            components_hartree=components,
            provenance=runtime_provenance,
        )


__all__ = [
    "ADJOINT_ABSOLUTE_TOLERANCE",
    "ADJOINT_MAX_ITERATIONS",
    "ADJOINT_RELATIVE_TOLERANCE",
    "DDPCM_ETA",
    "DDPCM_GAFF2_CARBONYL_O_MACE_KSPACE40_PROFILE",
    "DDPCM_GAFF2_CARBONYL_O_PROFILE",
    "DDPCM_LMAX",
    "DDPCM_N_LEBEDEV",
    "DDPCM_SMD_PROFILE",
    "DDPCM_SOLVER_TOLERANCE",
    "DDPCMSMDImplicitSolvation",
    "FORCE_STATE_ENERGY_TOLERANCE_EV",
    "SCF_DENSITY_TOLERANCE",
    "SCF_ENERGY_TOLERANCE_EV",
    "SCF_MAX_ITERATIONS",
    "SCF_MIXING",
    "WATER_STATIC_DIELECTRIC",
]
