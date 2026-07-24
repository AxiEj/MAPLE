"""Route 2: MACE-POLAR density coupled to aqueous SMD/IEFPCM.

The implementation is deliberately narrow:

* official MACE-POLAR-1-M density coefficients;
* neutral, closed-shell H/C/N/O/F/P/S/Cl/Br/I molecules;
* fixed-conformer single-point energies in water;
* PCMSolver IEFPCM electrostatics with SMD Coulomb radii;
* the published aqueous SMD CDS term implemented locally.

It does not invoke a quantum-chemistry executable and it does not train,
fine-tune, bundle, or modify an ML model.
"""

from __future__ import annotations

from contextlib import contextmanager
import ctypes
from dataclasses import dataclass
import importlib
import json
import os
from pathlib import Path
import sys
import threading
from typing import Any

import numpy as np
from ase.units import Bohr, Hartree

from ...calculator_base import ROUTE2_SMD_CALCULATOR_PROFILE
from .gto_density import (
    density_reaction_coupling,
    point_asc_reaction_potential_gradient,
    point_multipole_potential,
)
from .pcmsolver import PCMSolverSession
from .result import SolvationResult
from .smd_cds import (
    CANONICAL_SMD_PROFILE,
    GAFF2_CARBONYL_O_PROFILE,
    SASA_GRID_POINTS,
    SUPPORTED_ROUTE2_SMD_PROFILES,
    route2_water_coulomb_radii,
    smd_water_cds,
)


SUPPORTED_ELEMENTS = frozenset({"H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I"})
FORMALLY_CHARGED_TRIPOS_TYPES = frozenset({"n.4", "c.cat", "o.co2"})
MIN_MOLECULAR_MASS_DA = 16.0
MAX_MOLECULAR_MASS_DA = 500.0
PCM_TESSERA_AREA_ANGSTROM2 = 0.2
PCM_STABILITY_FALLBACK_TESSERA_AREA_ANGSTROM2 = 0.28
PCM_STABILITY_FALLBACK_MIN_RADIUS_ANGSTROM = 0.30
PCM_WARNING_MARKER = "PCMSolver warning."
SCF_MAX_ITERATIONS = 50
SCF_MIXING = 0.5
SCF_DENSITY_TOLERANCE = 1.0e-5
SCF_ENERGY_TOLERANCE_EV = 1.0e-5
_PCMSOLVER_CWD_LOCK = threading.RLock()


@dataclass(frozen=True)
class _PCMState:
    mep_hartree_per_e: np.ndarray
    asc_e: np.ndarray
    polarization_energy_hartree: float
    reaction_potential_hartree_per_e: np.ndarray
    reaction_gradient_hartree_per_e_bohr: np.ndarray
    density_reaction_coupling_hartree: float


@contextmanager
def _pcmsolver_audit_working_directory(path: Path):
    """Contain legacy PCMSolver/PEDRA side files inside the audit directory.

    PCMSolver v1.1 writes ``PEDRA.OUT_*``, ``cavity.off_*``, and
    ``cavity.npz`` relative to the process working directory. MAPLE is
    single-process here, so a lock serializes the short process-global cwd
    change and restores it even when the provider raises.
    """

    destination = Path(path).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with _PCMSOLVER_CWD_LOCK:
        previous = Path.cwd()
        os.chdir(destination)
        try:
            yield
        finally:
            os.chdir(previous)


@contextmanager
def _capture_process_stderr(path: Path):
    """Capture native-library stderr while preserving the caller's descriptor."""

    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    saved_stderr = os.dup(2)
    capture_fd = os.open(
        destination,
        os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
        0o600,
    )
    try:
        sys.stderr.flush()
        os.dup2(capture_fd, 2)
        yield
    finally:
        ctypes.CDLL(None).fflush(None)
        sys.stderr.flush()
        os.dup2(saved_stderr, 2)
        os.close(capture_fd)
        os.close(saved_stderr)


def _load_pcmsolver_parser():
    """Load the official parser and bind explicit paths to one installation."""

    explicit_python = os.environ.get("PCMSOLVER_PYTHON_PATH")
    explicit_library = os.environ.get("PCMSOLVER_LIBRARY")
    if explicit_python and not explicit_library:
        raise ImportError(
            "PCMSOLVER_PYTHON_PATH requires PCMSOLVER_LIBRARY so Route 2 can "
            "verify that the parser and shared library share one installation."
        )

    candidates: list[Path] = []
    library_path: Path | None = None
    if explicit_library:
        library_path = Path(explicit_library).expanduser().resolve()
        if not library_path.is_file():
            raise ImportError(
                f"PCMSOLVER_LIBRARY is not a file: {library_path}."
            )
        if explicit_python:
            candidates.append(Path(explicit_python).expanduser().resolve())
        candidates.extend(
            (
                library_path.parent / "python",
                library_path.parent.parent / "lib" / "python",
                library_path.parent.parent / "python",
            )
        )

    module = None
    existing = sys.modules.get("pcmsolver")
    for candidate in candidates:
        candidate = candidate.resolve()
        if not (candidate / "pcmsolver").is_dir():
            continue
        if existing is not None:
            existing_file = Path(getattr(existing, "__file__", "")).resolve()
            try:
                existing_file.relative_to(candidate)
            except ValueError:
                raise ImportError(
                    "A different pcmsolver Python package is already imported; "
                    "restart Python with the parser matching PCMSOLVER_LIBRARY."
                ) from None
            module = existing
            break
        sys.path.insert(0, str(candidate))
        try:
            module = importlib.import_module("pcmsolver")
            break
        except ImportError:
            continue
        finally:
            try:
                sys.path.remove(str(candidate))
            except ValueError:
                pass

    if module is None:
        try:
            module = importlib.import_module("pcmsolver")
        except ImportError:
            raise ImportError(
                "Route 2 requires PCMSolver's official Python input parser in addition "
                "to libpcm.so. Install one coherent PCMSolver distribution or set both "
                "PCMSOLVER_LIBRARY and PCMSOLVER_PYTHON_PATH."
            ) from None

    if library_path is not None:
        library_dir = library_path.parent
        installation_prefix = (
            library_dir.parent
            if library_dir.name.lower() in {"lib", "lib64"}
            else library_dir
        ).resolve()
        module_path = Path(getattr(module, "__file__", "")).resolve()
        try:
            module_path.relative_to(installation_prefix)
        except ValueError:
            raise ImportError(
                "PCMSOLVER_PYTHON_PATH and PCMSOLVER_LIBRARY must come from "
                f"the same PCMSolver installation prefix ({installation_prefix}); "
                f"loaded parser: {module_path}."
            ) from None

    parser = getattr(module, "parse_pcm_input", None)
    if not callable(parser):
        raise ImportError(
            "The imported pcmsolver package does not expose parse_pcm_input(); "
            "use the Python package installed with the same PCMSolver release as libpcm.so."
        )
    return parser


def _pcm_input_text(
    atom_count: int,
    radii_angstrom: np.ndarray,
    *,
    tessera_area_angstrom2: float = PCM_TESSERA_AREA_ANGSTROM2,
    minimum_added_sphere_radius_angstrom: float | None = None,
) -> str:
    """Return a human-readable PCMSolver input with explicit per-atom radii.

    ``MODE=ATOMS`` is intentional: it lets the C API initialize the molecular
    geometry from the host arrays while replacing every built-in radius with
    the SMD radius.  ``MODE=EXPLICIT`` instead turns the sphere list into dummy
    unit-charge atoms in the v1.1 C API.
    """

    radii = np.asarray(radii_angstrom, dtype=float)
    if radii.shape != (atom_count,) or not np.all(np.isfinite(radii)):
        raise ValueError("PCMSolver radii must be one finite value per atom.")
    area = float(tessera_area_angstrom2)
    if not np.isfinite(area) or area <= 0.0:
        raise ValueError("PCMSolver tessera area must be finite and positive.")
    minimum_radius = (
        None
        if minimum_added_sphere_radius_angstrom is None
        else float(minimum_added_sphere_radius_angstrom)
    )
    if minimum_radius is not None and (
        not np.isfinite(minimum_radius) or minimum_radius <= 0.0
    ):
        raise ValueError(
            "PCMSolver minimum added-sphere radius must be finite and positive."
        )
    atom_indices = ", ".join(str(index) for index in range(1, atom_count + 1))
    radius_values = ", ".join(f"{radius:.10f}" for radius in radii)
    minimum_radius_line = (
        ""
        if minimum_radius is None
        else f"  MINRADIUS = {minimum_radius:.10f}\n"
    )
    return (
        "UNITS = ANGSTROM\n"
        "CODATA = 2010\n"
        "CAVITY\n"
        "{\n"
        "  TYPE = GEPOL\n"
        f"  AREA = {area:.10f}\n"
        "  SCALING = FALSE\n"
        f"{minimum_radius_line}"
        "  MODE = ATOMS\n"
        f"  ATOMS = [{atom_indices}]\n"
        f"  RADII = [{radius_values}]\n"
        "}\n"
        "MEDIUM\n"
        "{\n"
        "  SOLVERTYPE = IEFPCM\n"
        "  SOLVENT = WATER\n"
        "  NONEQUILIBRIUM = FALSE\n"
        "  MATRIXSYMM = TRUE\n"
        "  DIAGONALINTEGRATOR = COLLOCATION\n"
        "  DIAGONALSCALING = 1.07\n"
        "}\n"
    )


@dataclass
class SMDImplicitSolvation:
    """Aqueous SMD correction driven by the MACE-POLAR residual density."""

    atoms: Any
    solvation_options: dict[str, Any]
    audit_dir: Path | None = None

    supported_properties = frozenset({"energy"})

    def __post_init__(self) -> None:
        self.solvation_options = dict(self.solvation_options)
        self.response = str(self.solvation_options.get("response", "scf")).lower()
        self.standard_state = str(
            self.solvation_options.get("standard_state", "1m")
        ).lower()
        self.provider = str(
            self.solvation_options.get("provider", "pcmsolver")
        ).lower()
        self.profile = str(
            self.solvation_options.get("profile", "smd-iefpcm")
        ).lower()

        self._validate_options()
        self._validate_domain(self.atoms)
        self._reference_numbers = np.asarray(self.atoms.numbers, dtype=int).copy()
        self._reference_positions = np.asarray(
            self.atoms.get_positions(), dtype=float
        ).copy()
        mol2 = self.atoms.info.get("mol2")
        atom_types = mol2.get("atom_types") if isinstance(mol2, dict) else None
        self.coulomb_radii_angstrom = route2_water_coulomb_radii(
            self.atoms.get_chemical_symbols(),
            atom_types=atom_types,
            profile=self.profile,
        )
        self._parsed_pcm_input_path: Path | None = None
        self._parsed_pcm_input_paths: dict[str, Path] = {}
        self._pcmsolver_parser_path: Path | None = None

        if self.audit_dir is not None:
            self.audit_dir = Path(self.audit_dir).resolve()
            self.audit_dir.mkdir(parents=True, exist_ok=True)

        self.provenance = {
            "provider": "pcmsolver",
            "method": "smd",
            "profile": self.profile,
            "solvent": "water",
            "response": self.response,
            "standard_state": "1M(gas)->1M(solution)",
            "standard_state_correction_hartree": 0.0,
            "density_source": "official MACE-POLAR-1-M l<=1 residual charge density",
            "density_interpretation": "coarse-grained net charge density, not a QM electron density",
            "pcm_mep_projection": (
                "cavity-exterior point monopoles and dipoles; the model-internal "
                "1.5 A GTO smearing is not extended across the dielectric boundary"
            ),
            "electrostatics": "IEFPCM",
            "cavity_stability_policy": (
                "Start with AREA=0.20 A^2 and no added spheres. If PCMSolver "
                "emits a cavity warning, retry deterministically with "
                "AREA=0.28 A^2 and MINRADIUS=0.30 A; fail closed if a warning "
                "persists. This warning-triggered branch is fixed-conformer "
                "energy infrastructure and is not a force/PES policy."
            ),
            "cavity_stability_policy_force_compatible": False,
            "cavity_radii": (
                "SMD Coulomb radii with revised Br=2.60 A and I=2.74 A"
                if self.profile == CANONICAL_SMD_PROFILE
                else (
                    "SMD Coulomb radii with GAFF/GAFF2 carbonyl oxygen "
                    "(atom type o) overridden to 1.70 A"
                )
            ),
            "cds": "aqueous SMD atomic surface tensions with native NumPy SASA",
            "cds_grid_points_per_atom": SASA_GRID_POINTS,
            "route_role": "research-innovation",
            "scientific_status": "energy-proof-of-concept",
            "solution_phase_pes": False,
            "forces_available": False,
            "next_primary_milestone": (
                "energy-consistent force derivative for the converged "
                "MACE-POLAR/PCM/SMD state"
            ),
            "accuracy_certified": False,
            "default_eligible": False,
            "energy_composition": (
                "delta_G_solv = (E_MACE_intrinsic[V_reac]-E_MACE_gas) "
                "+ 0.5*<V_solute,ASC> + G_CDS"
            ),
            "citations": {
                "smd": "Marenich, Cramer, Truhlar, JPCB 2009, DOI:10.1021/jp810292n",
                "pcmsolver": "Di Remigio et al., JOSS 2019, DOI:10.21105/joss.01190",
                "mace_polar": "MACE-POLAR-1, arXiv:2602.19411",
                **(
                    {
                        "gaff2_pbsa_radii": (
                            "Sun et al., J. Comput. Chem. 2023, "
                            "DOI:10.1002/jcc.27089"
                        )
                    }
                    if self.profile == GAFF2_CARBONYL_O_PROFILE
                    else {}
                ),
            },
        }
        self._write_manifest()

    def _validate_options(self) -> None:
        if self.solvation_options.get("experimental") is not True:
            raise ValueError(
                "Route 2 is an energy-only research proof-of-concept; "
                "set experimental=true explicitly."
            )
        if str(self.solvation_options.get("method", "")).lower() != "smd":
            raise ValueError("SMDImplicitSolvation requires method=smd.")
        if str(self.solvation_options.get("implicit", "")).lower() != "water":
            raise ValueError("Route 2 currently supports implicit=water only.")
        if self.provider != "pcmsolver":
            raise ValueError("Route 2 research contract uses provider=pcmsolver only.")
        if self.profile not in SUPPORTED_ROUTE2_SMD_PROFILES:
            raise ValueError(
                "Route 2 profile must be smd-iefpcm or "
                "smd-iefpcm-gaff2-o."
            )
        if self.response not in {"frozen", "scf"}:
            raise ValueError("SMD response must be frozen or scf.")
        if self.standard_state != "1m":
            raise ValueError(
                "Route 2 uses the 1 M gas -> 1 M solution convention only; "
                "standard_state must be 1m."
            )

    @staticmethod
    def _validate_domain(atoms) -> None:
        if atoms is None or len(atoms) == 0:
            raise ValueError("Route 2 requires one non-empty molecule.")
        symbols = tuple(atoms.get_chemical_symbols())
        unsupported = sorted(set(symbols).difference(SUPPORTED_ELEMENTS))
        if unsupported:
            raise ValueError(
                "Route 2 supports H/C/N/O/F/P/S/Cl/Br/I only; unsupported elements: "
                + ", ".join(unsupported)
                + "."
            )
        molecular_mass = float(np.sum(atoms.get_masses()))
        if not MIN_MOLECULAR_MASS_DA <= molecular_mass <= MAX_MOLECULAR_MASS_DA:
            raise ValueError(
                "Route 2 v1 is validated for neutral organics from 16 to 500 Da; "
                f"received {molecular_mass:.6f} Da."
            )
        try:
            charge = float(atoms.info.get("charge", 0))
            multiplicity_value = float(atoms.info.get("mult", 1))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Route 2 requires numeric charge=0 and multiplicity=1 metadata."
            ) from exc
        if not multiplicity_value.is_integer():
            raise ValueError("Route 2 multiplicity metadata must be an integer.")
        multiplicity = int(multiplicity_value)
        if charge != 0.0 or multiplicity != 1:
            raise ValueError(
                "Route 2 v1 supports neutral closed-shell molecules only "
                f"(received charge={charge:g}, multiplicity={multiplicity})."
            )
        mol2 = atoms.info.get("mol2")
        if isinstance(mol2, dict):
            atom_types = {
                str(atom_type).strip().lower()
                for atom_type in mol2.get("atom_types", ())
            }
            charged_markers = sorted(
                atom_types.intersection(FORMALLY_CHARGED_TRIPOS_TYPES)
            )
            if charged_markers:
                raise ValueError(
                    "Route 2 v1 excludes salts and zwitterions; the MOL2 uses "
                    "formally charged Tripos atom type(s): "
                    + ", ".join(charged_markers)
                    + "."
                )
        if np.any(atoms.get_pbc()):
            raise ValueError("Route 2 SMD is non-periodic.")

    def _validate_fixed_geometry(self, atoms) -> None:
        numbers = np.asarray(atoms.numbers, dtype=int)
        positions = np.asarray(atoms.get_positions(), dtype=float)
        if not np.array_equal(numbers, self._reference_numbers):
            raise ValueError("Route 2 does not permit atom identity/order changes.")
        if not np.array_equal(positions, self._reference_positions):
            raise ValueError(
                "Route 2 v1 is fixed-conformer single-point only; the geometry changed "
                "after the solvent cavity was defined."
            )

    def _write_manifest(self) -> None:
        if self.audit_dir is None:
            return
        manifest = {
            "schema_version": 2,
            "solvation_options": self.solvation_options,
            "provenance": self.provenance,
            "elements": self.atoms.get_chemical_symbols(),
            "positions_angstrom": np.asarray(
                self.atoms.get_positions(), dtype=float
            ).tolist(),
        }
        (self.audit_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def _ensure_pcm_input(self) -> Path:
        return self._ensure_pcm_input_variant(
            key="primary",
            filename="route2-smd.pcm",
            tessera_area_angstrom2=PCM_TESSERA_AREA_ANGSTROM2,
            minimum_added_sphere_radius_angstrom=None,
        )

    def _ensure_pcm_fallback_input(self) -> Path:
        return self._ensure_pcm_input_variant(
            key="stability-fallback",
            filename="route2-smd-stability-fallback.pcm",
            tessera_area_angstrom2=(
                PCM_STABILITY_FALLBACK_TESSERA_AREA_ANGSTROM2
            ),
            minimum_added_sphere_radius_angstrom=(
                PCM_STABILITY_FALLBACK_MIN_RADIUS_ANGSTROM
            ),
        )

    def _ensure_pcm_input_variant(
        self,
        *,
        key: str,
        filename: str,
        tessera_area_angstrom2: float,
        minimum_added_sphere_radius_angstrom: float | None,
    ) -> Path:
        cached = self._parsed_pcm_input_paths.get(key)
        if cached is not None:
            return cached
        if self.audit_dir is None:
            raise RuntimeError(
                "Route 2 requires an audit directory so the exact PCMSolver input "
                "and parsed machine file can be retained."
            )

        raw_path = self.audit_dir / filename
        raw_path.write_text(
            _pcm_input_text(
                len(self.atoms),
                self.coulomb_radii_angstrom,
                tessera_area_angstrom2=tessera_area_angstrom2,
                minimum_added_sphere_radius_angstrom=(
                    minimum_added_sphere_radius_angstrom
                ),
            ),
            encoding="utf-8",
        )
        parser = _load_pcmsolver_parser()
        parser_module = importlib.import_module(parser.__module__)
        parser_file = getattr(parser_module, "__file__", None)
        if parser_file:
            self._pcmsolver_parser_path = Path(parser_file).resolve()
        try:
            parser(str(raw_path), write_out=True)
        except Exception as exc:
            raise RuntimeError(
                f"PCMSolver failed to parse the generated SMD input {raw_path}: {exc}"
            ) from exc
        parsed_path = raw_path.with_name("@" + raw_path.name)
        if not parsed_path.is_file():
            raise RuntimeError(
                "PCMSolver parse_pcm_input(..., write_out=True) did not create the "
                f"expected machine input: {parsed_path}."
            )
        self._parsed_pcm_input_paths[key] = parsed_path
        return parsed_path

    @staticmethod
    def _gas_state(calculator, atoms):
        state = getattr(calculator, "_last_polar_state", None)
        if state is None:
            state, _ = calculator.polar_state(atoms)
        return state

    @staticmethod
    def _validate_density(density_coefficients: np.ndarray, atom_count: int) -> np.ndarray:
        density = np.asarray(density_coefficients, dtype=float)
        if density.shape != (atom_count, 4) or not np.all(np.isfinite(density)):
            raise RuntimeError(
                "Route 2 requires finite MACE-POLAR density coefficients with "
                f"shape ({atom_count}, 4); received {density.shape}."
            )
        net_charge = float(np.sum(density[:, 0]))
        if abs(net_charge) > 1.0e-4:
            raise RuntimeError(
                "The MACE-POLAR residual density violates the neutral charge "
                f"constraint (sum={net_charge:.6e} e)."
            )
        return density.copy()

    def _solve_pcm(
        self,
        session: PCMSolverSession,
        atoms,
        density_coefficients: np.ndarray,
    ) -> _PCMState:
        mep = point_multipole_potential(
            session.cavity_centers_bohr,
            atoms.get_positions(),
            density_coefficients,
        )
        solved = session.solve(mep)
        asc = np.asarray(solved["asc"], dtype=float)
        polarization_energy = float(solved["polarization_energy"])
        if not np.all(np.isfinite(asc)) or not np.isfinite(polarization_energy):
            raise RuntimeError("PCMSolver returned non-finite ASC/energy values.")

        surface_coupling = float(np.dot(mep, asc))
        expected_coupling = 2.0 * polarization_energy
        tolerance = max(1.0e-10, 1.0e-8 * abs(expected_coupling))
        if abs(surface_coupling - expected_coupling) > tolerance:
            raise RuntimeError(
                "PCMSolver polarization-energy convention check failed: "
                "compute_polarization_energy must equal 0.5*dot(MEP,ASC)."
            )

        reaction_potential, reaction_gradient = point_asc_reaction_potential_gradient(
            atoms.get_positions(),
            session.cavity_centers_bohr,
            asc,
        )
        multipole_coupling = density_reaction_coupling(
            density_coefficients,
            reaction_potential,
            reaction_gradient,
        )
        if abs(multipole_coupling - surface_coupling) > max(
            1.0e-10, 1.0e-8 * abs(surface_coupling)
        ):
            raise RuntimeError(
                "MACE-POLAR point-multipole/ASC reciprocity check failed; the reaction-field "
                "projection is inconsistent with the cavity MEP."
            )

        return _PCMState(
            mep_hartree_per_e=mep,
            asc_e=asc,
            polarization_energy_hartree=polarization_energy,
            reaction_potential_hartree_per_e=reaction_potential,
            reaction_gradient_hartree_per_e_bohr=reaction_gradient,
            density_reaction_coupling_hartree=surface_coupling,
        )

    @staticmethod
    def _polarize(calculator, atoms, pcm_state: _PCMState):
        node_potential_ev = pcm_state.reaction_potential_hartree_per_e * Hartree
        node_gradient_ev_per_angstrom = (
            pcm_state.reaction_gradient_hartree_per_e_bohr * Hartree / Bohr
        )
        state, _ = calculator.polar_state(
            atoms,
            node_potential_ev=node_potential_ev,
            node_gradient_ev_per_angstrom=node_gradient_ev_per_angstrom,
        )
        return state

    def _self_consistent_state(
        self,
        session: PCMSolverSession,
        atoms,
        calculator,
        gas_state,
    ):
        density = self._validate_density(
            gas_state.density_coefficients, len(atoms)
        )
        previous_energy_ev: float | None = None
        history: list[dict[str, float | int | None]] = []
        final_state = None

        for iteration in range(1, SCF_MAX_ITERATIONS + 1):
            pcm_state = self._solve_pcm(session, atoms, density)
            response_state = self._polarize(calculator, atoms, pcm_state)
            response_density = self._validate_density(
                response_state.density_coefficients, len(atoms)
            )
            density_residual = float(np.max(np.abs(response_density - density)))
            energy_residual = (
                None
                if previous_energy_ev is None
                else abs(float(response_state.energy_ev) - previous_energy_ev)
            )
            history.append(
                {
                    "iteration": iteration,
                    "density_residual_e": density_residual,
                    "energy_residual_ev": energy_residual,
                    "intrinsic_energy_ev": float(response_state.energy_ev),
                    "pcm_polarization_energy_hartree": (
                        pcm_state.polarization_energy_hartree
                    ),
                }
            )

            if (
                previous_energy_ev is not None
                and density_residual <= SCF_DENSITY_TOLERANCE
                and energy_residual is not None
                and energy_residual <= SCF_ENERGY_TOLERANCE_EV
            ):
                final_state = response_state
                break

            density = (
                (1.0 - SCF_MIXING) * density
                + SCF_MIXING * response_density
            )
            previous_energy_ev = float(response_state.energy_ev)

        if final_state is None:
            last = history[-1]
            raise RuntimeError(
                "MACE-POLAR/IEFPCM reaction-field SCF did not converge in "
                f"{SCF_MAX_ITERATIONS} iterations "
                f"(density residual={last['density_residual_e']:.3e} e, "
                f"energy residual={last['energy_residual_ev']:.3e} eV)."
            )

        final_density = self._validate_density(
            final_state.density_coefficients, len(atoms)
        )
        final_pcm = self._solve_pcm(session, atoms, final_density)
        return final_state, final_pcm, history

    def _write_result_audit(
        self,
        *,
        gas_state,
        solvent_state,
        pcm_state: _PCMState,
        session: PCMSolverSession,
        cds_result,
        components: dict[str, float],
        history: list[dict[str, float | int | None]],
    ) -> None:
        if self.audit_dir is None:
            return
        np.savez_compressed(
            self.audit_dir / "route2-state.npz",
            gas_density_coefficients=np.asarray(gas_state.density_coefficients),
            solvent_density_coefficients=np.asarray(
                solvent_state.density_coefficients
            ),
            cavity_centers_bohr=session.cavity_centers_bohr,
            cavity_areas_bohr2=session.cavity_areas_bohr2,
            mep_hartree_per_e=pcm_state.mep_hartree_per_e,
            asc_e=pcm_state.asc_e,
            reaction_potential_hartree_per_e=(
                pcm_state.reaction_potential_hartree_per_e
            ),
            reaction_gradient_hartree_per_e_bohr=(
                pcm_state.reaction_gradient_hartree_per_e_bohr
            ),
            cds_atom_areas_angstrom2=cds_result.atom_areas_angstrom2,
            cds_atom_tensions_cal_mol_angstrom2=(
                cds_result.atom_tensions_cal_mol_angstrom2
            ),
        )
        payload = {
            "schema_version": 4,
            "response": self.response,
            "pcm_mep_projection": "cavity-exterior-point-multipole-l<=1",
            "converged": True,
            "iterations": len(history),
            "scf": {
                "maximum_iterations": SCF_MAX_ITERATIONS,
                "mixing": SCF_MIXING,
                "density_tolerance_e": SCF_DENSITY_TOLERANCE,
                "energy_tolerance_ev": SCF_ENERGY_TOLERANCE_EV,
                "history": history,
            },
            "energies_hartree": components,
            "gas_mace_energy_ev": float(gas_state.energy_ev),
            "solvent_intrinsic_mace_energy_ev": float(solvent_state.energy_ev),
            "density_reaction_coupling_hartree": (
                pcm_state.density_reaction_coupling_hartree
            ),
            "pcmsolver_input": str(self._parsed_pcm_input_path),
            "pcmsolver_python_parser": (
                None
                if self._pcmsolver_parser_path is None
                else str(self._pcmsolver_parser_path)
            ),
            "pcmsolver_library": getattr(session, "library_source", None),
            "pcmsolver_side_files": sorted(
                str(path)
                for pattern in ("PEDRA.OUT*", "cavity.off*", "cavity.npz")
                for path in self.audit_dir.glob(pattern)
            ),
            "array_archive": str(self.audit_dir / "route2-state.npz"),
        }
        (self.audit_dir / "route2-result.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def evaluate(self, atoms, need_forces: bool = False, calculator=None) -> SolvationResult:
        if need_forces:
            raise NotImplementedError("Route 2 SMD v1 is energy-only.")
        self._validate_fixed_geometry(atoms)
        if calculator is None or not callable(getattr(calculator, "polar_state", None)):
            raise TypeError(
                "Route 2 requires the MACEPolCalculator polar_state() density/response API."
            )
        if (
            getattr(calculator, "route2_smd_profile", None)
            != ROUTE2_SMD_CALCULATOR_PROFILE
        ):
            raise TypeError(
                "Route 2 requires the official MACE-POLAR-1-M float64 local-field "
                "calculator profile; alternate or gas-only polar_state() providers "
                "are not accepted."
            )

        gas_state = self._gas_state(calculator, atoms)
        self._validate_density(gas_state.density_coefficients, len(atoms))
        assert self.audit_dir is not None
        cds_result = smd_water_cds(
            atoms.get_chemical_symbols(),
            atoms.get_positions(),
        )
        cavity_attempts: list[dict[str, Any]] = []
        selected_cavity: dict[str, Any] | None = None
        with _pcmsolver_audit_working_directory(self.audit_dir):
            for (
                attempt_name,
                input_factory,
                tessera_area,
                minimum_radius,
            ) in (
                (
                    "primary",
                    self._ensure_pcm_input,
                    PCM_TESSERA_AREA_ANGSTROM2,
                    None,
                ),
                (
                    "stability-fallback",
                    self._ensure_pcm_fallback_input,
                    PCM_STABILITY_FALLBACK_TESSERA_AREA_ANGSTROM2,
                    PCM_STABILITY_FALLBACK_MIN_RADIUS_ANGSTROM,
                ),
            ):
                parsed_input = input_factory()
                native_stderr_path = (
                    self.audit_dir / f"pcmsolver-{attempt_name}-stderr.log"
                )
                with _capture_process_stderr(native_stderr_path):
                    with PCMSolverSession(
                        np.asarray(atoms.numbers, dtype=float),
                        np.asarray(atoms.get_positions(), dtype=float) / Bohr,
                        parsed_input,
                    ) as session:
                        if self.response == "frozen":
                            solvent_state = gas_state
                            pcm_state = self._solve_pcm(
                                session, atoms, gas_state.density_coefficients
                            )
                            history = []
                        else:
                            (
                                solvent_state,
                                pcm_state,
                                history,
                            ) = self._self_consistent_state(
                                session, atoms, calculator, gas_state
                            )

                        delta_e_solute = (
                            float(solvent_state.energy_ev)
                            - float(gas_state.energy_ev)
                        ) / Hartree
                        pcm_polarization = (
                            pcm_state.polarization_energy_hartree
                        )
                        electrostatic = delta_e_solute + pcm_polarization
                        total = electrostatic + cds_result.energy_hartree
                        components = {
                            "solute_polarization": delta_e_solute,
                            "pcm_polarization": pcm_polarization,
                            "electrostatic": electrostatic,
                            "cds": cds_result.energy_hartree,
                            "standard_state": 0.0,
                            "delta_g_solv": total,
                        }
                        self._parsed_pcm_input_path = parsed_input
                        self._write_result_audit(
                            gas_state=gas_state,
                            solvent_state=solvent_state,
                            pcm_state=pcm_state,
                            session=session,
                            cds_result=cds_result,
                            components=components,
                            history=history,
                        )
                        pcmsolver_library = getattr(
                            session, "library_source", None
                        )

                native_stderr = native_stderr_path.read_text(
                    encoding="utf-8", errors="replace"
                )
                warning_detected = PCM_WARNING_MARKER in native_stderr
                attempt = {
                    "name": attempt_name,
                    "tessera_area_angstrom2": tessera_area,
                    "minimum_added_sphere_radius_angstrom": minimum_radius,
                    "warning_detected": warning_detected,
                    "stderr_log": str(native_stderr_path),
                    "cavity_tesserae": int(pcm_state.asc_e.size),
                }
                cavity_attempts.append(attempt)
                if warning_detected:
                    continue
                selected_cavity = attempt
                break

        if selected_cavity is None:
            for filename in ("route2-result.json", "route2-state.npz"):
                (self.audit_dir / filename).unlink(missing_ok=True)
            failure_payload = {
                "schema_version": 1,
                "reason": "pcmsolver-cavity-warning-after-deterministic-fallback",
                "cavity_stability": {
                    "policy": "warning-triggered deterministic fallback",
                    "force_compatible": False,
                    "selected": None,
                    "attempts": cavity_attempts,
                },
            }
            (self.audit_dir / "route2-failure.json").write_text(
                json.dumps(failure_payload, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            raise RuntimeError(
                "PCMSolver emitted warnings for both the primary and "
                "deterministic fallback cavities; Route 2 refuses to publish "
                "a numerically suspect result. See "
                f"{self.audit_dir / 'pcmsolver-primary-stderr.log'} and "
                f"{self.audit_dir / 'pcmsolver-stability-fallback-stderr.log'}."
            )

        audit_path = self.audit_dir / "route2-result.json"
        audit_payload = json.loads(audit_path.read_text(encoding="utf-8"))
        audit_payload["cavity_stability"] = {
            "policy": "warning-triggered deterministic fallback",
            "force_compatible": False,
            "selected": selected_cavity["name"],
            "attempts": cavity_attempts,
        }
        audit_path.write_text(
            json.dumps(audit_payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )

        provenance = {
            **self.provenance,
            "converged": True,
            "iterations": len(history),
            "density_reaction_coupling_hartree": (
                pcm_state.density_reaction_coupling_hartree
            ),
            "cavity_tesserae": int(pcm_state.asc_e.size),
            "cavity_stability": {
                "selected": selected_cavity["name"],
                "force_compatible": False,
                "tessera_area_angstrom2": selected_cavity[
                    "tessera_area_angstrom2"
                ],
                "minimum_added_sphere_radius_angstrom": selected_cavity[
                    "minimum_added_sphere_radius_angstrom"
                ],
                "fallback_used": (
                    selected_cavity["name"] == "stability-fallback"
                ),
                "attempt_count": len(cavity_attempts),
            },
            "audit_directory": (
                None if self.audit_dir is None else str(self.audit_dir)
            ),
            "calculator_profile": getattr(
                calculator, "route2_smd_profile", None
            ),
            "mace_torch_version": getattr(
                calculator, "mace_torch_version", None
            ),
            "mace_dtype": str(getattr(calculator, "dtype", None)),
            "pcmsolver_library": pcmsolver_library,
            "pcmsolver_python_parser": (
                None
                if self._pcmsolver_parser_path is None
                else str(self._pcmsolver_parser_path)
            ),
        }
        return SolvationResult(
            energy_hartree=total,
            components_hartree=components,
            provenance=provenance,
        )


__all__ = [
    "MAX_MOLECULAR_MASS_DA",
    "MIN_MOLECULAR_MASS_DA",
    "PCM_STABILITY_FALLBACK_MIN_RADIUS_ANGSTROM",
    "PCM_STABILITY_FALLBACK_TESSERA_AREA_ANGSTROM2",
    "PCM_TESSERA_AREA_ANGSTROM2",
    "SCF_DENSITY_TOLERANCE",
    "SCF_ENERGY_TOLERANCE_EV",
    "SCF_MAX_ITERATIONS",
    "SCF_MIXING",
    "SMDImplicitSolvation",
    "SUPPORTED_ELEMENTS",
]
