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

from collections.abc import Callable
from contextlib import contextmanager
import ctypes
from dataclasses import dataclass
import importlib
import json
import os
from pathlib import Path
import sys
import threading
import time
from typing import Any

import numpy as np
from ase.units import Bohr, Hartree

from ...calculator_base import ROUTE2_SMD_CALCULATOR_PROFILE
from ....route2_smd_profiles import (
    route2_smd_profile_spec,
)
from .continuum_response import (
    PCMSolverExternalMEPCavityResponse,
)
from .pcmsolver import PCMSolverSession
from .result import SolvationResult
from .route2_engine import (
    Route2ContinuumEngine,
    Route2CoupledState,
    Route2EngineSettings,
)
from .route2_pcm_response import (
    FixedCavityPCMSnapshot,
    FixedCavityPCMReactionFieldLinearMap,
)
from .gto_field_projection import ExactGTOFieldProjector
from .route2_domain import (
    FORMALLY_CHARGED_TRIPOS_TYPES,
    MAX_MOLECULAR_MASS_DA,
    MIN_MOLECULAR_MASS_DA,
    SUPPORTED_ELEMENTS,
    validate_route2_domain,
)
from .smd_cds import (
    CANONICAL_SMD_PROFILE,
    GAFF2_CARBONYL_O_PROFILE,
    SASA_GRID_POINTS,
    SUPPORTED_PCMSOLVER_SMD_PROFILES,
    route2_water_coulomb_radii,
    smd_water_cds,
)


PCM_TESSERA_AREA_ANGSTROM2 = 0.2
PCM_STABILITY_FALLBACK_TESSERA_AREA_ANGSTROM2 = 0.28
PCM_STABILITY_FALLBACK_MIN_RADIUS_ANGSTROM = 0.30
PCM_WARNING_MARKER = "PCMSolver warning."
CAVITY_POLICY_WARNING_FALLBACK = "warning-fallback"
CAVITY_POLICY_FIXED_STABILITY_BRANCH = "fixed-stability-branch"
SUPPORTED_CAVITY_POLICIES = frozenset(
    {
        CAVITY_POLICY_WARNING_FALLBACK,
        CAVITY_POLICY_FIXED_STABILITY_BRANCH,
    }
)
SCF_MAX_ITERATIONS = 50
SCF_MIXING = 0.5
SCF_DENSITY_TOLERANCE = 1.0e-5
SCF_ENERGY_TOLERANCE_EV = 1.0e-5
_PCMSOLVER_CWD_LOCK = threading.RLock()


def _pcmsolver_engine_settings() -> Route2EngineSettings:
    """Build the PCMSolver policy from the public numerical knobs."""

    return Route2EngineSettings(
        continuum_label="IEFPCM",
        scf_mixing=SCF_MIXING,
        scf_density_tolerance=SCF_DENSITY_TOLERANCE,
        scf_energy_tolerance_ev=SCF_ENERGY_TOLERANCE_EV,
        scf_max_iterations=SCF_MAX_ITERATIONS,
        # PCMSolver remains energy-only; these positive derivative settings
        # satisfy the shared policy contract but are not consumed here.
        adjoint_relative_tolerance=1.0e-10,
        adjoint_absolute_tolerance=1.0e-13,
        adjoint_max_iterations=100,
        energy_identity_tolerance_ev=1.0e-8,
        force_state_energy_tolerance_ev=1.0e-8,
        neutral_density_tolerance=1.0e-4,
        scf_require_two_energy_samples=True,
    )


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


def _flush_process_stderr() -> None:
    ctypes.CDLL(None).fflush(None)
    sys.stderr.flush()


def _pedra_file_fingerprints(audit_dir: Path) -> dict[Path, tuple[int, int]]:
    """Record enough file state to exclude stale PEDRA outputs from a run."""

    return {
        path: (path.stat().st_mtime_ns, path.stat().st_size)
        for path in audit_dir.glob("PEDRA.OUT*")
    }


def _pedra_warning_records(
    audit_dir: Path,
    baseline: dict[Path, tuple[int, int]],
) -> list[dict[str, Any]]:
    """Return warning lines from PEDRA side files created or changed this run."""

    warnings: list[dict[str, Any]] = []
    for path in sorted(audit_dir.glob("PEDRA.OUT*")):
        stat = path.stat()
        if baseline.get(path) == (stat.st_mtime_ns, stat.st_size):
            continue
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8", errors="replace").splitlines(),
            start=1,
        ):
            message = line.strip()
            if "WARNING" not in message.upper():
                continue
            warnings.append(
                {
                    "file": str(path),
                    "line": line_number,
                    "message": message,
                }
            )
    return warnings


def _pcmsolver_diagnostics(
    audit_dir: Path,
    cavity_attempts: list[dict[str, Any]],
    pedra_baseline: dict[Path, tuple[int, int]],
) -> dict[str, Any]:
    pedra_warnings = _pedra_warning_records(audit_dir, pedra_baseline)
    return {
        "native_stderr_warning_marker": PCM_WARNING_MARKER,
        "native_stderr_warning_count": sum(
            bool(attempt["warning_detected"]) for attempt in cavity_attempts
        ),
        "pedra_warning_count": len(pedra_warnings),
        "pedra_warnings": pedra_warnings,
        "pedra_warnings_are_selection_fatal": False,
    }


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
        _flush_process_stderr()
        os.dup2(capture_fd, 2)
        yield
    finally:
        _flush_process_stderr()
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
        self.cavity_policy = str(
            self.solvation_options.get(
                "cavity_policy",
                CAVITY_POLICY_WARNING_FALLBACK,
            )
        ).lower()
        self._cavity_policy_geometry_selection_branch_free = (
            self.cavity_policy
            == CAVITY_POLICY_FIXED_STABILITY_BRANCH
        )

        self.profile_spec = route2_smd_profile_spec(self.profile)
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
        self._engine_settings = _pcmsolver_engine_settings()

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
            "solute_source": self.profile_spec.solute_source,
            "reaction_field_projector": (
                self.profile_spec.reaction_field_projector
            ),
            "density_dual_field_gauge": "continuum-zero-at-infinity",
            "model_field_gauge": self.profile_spec.model_field_gauge,
            "nonpolar_model": self.profile_spec.nonpolar_model,
            "strict_original_smd_equivalence": (
                self.profile_spec.strict_original_smd_equivalence
            ),
            "pcm_mep_projection": (
                "cavity-exterior point monopoles and dipoles; the "
                "model-internal 1.5 A GTO smearing is not extended "
                "across the dielectric boundary"
            ),
            "electrostatics": "IEFPCM",
            "cavity_policy": self.cavity_policy,
            "cavity_stability_policy": self._cavity_policy_description(),
            "cavity_policy_geometry_selection_branch_free": (
                self._cavity_policy_geometry_selection_branch_free
            ),
            "cavity_stability_policy_force_compatible": False,
            "cavity_radii": (
                "SMD Coulomb radii with revised Br=2.60 A and I=2.74 A"
                if not self.profile_spec.uses_gaff2_carbonyl_oxygen
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
                    if self.profile_spec.uses_gaff2_carbonyl_oxygen
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
        if self.profile not in SUPPORTED_PCMSOLVER_SMD_PROFILES:
            raise ValueError(
                "Route 2 profile must be one of the registered PCMSolver "
                "SMD profiles."
            )
        if self.cavity_policy not in SUPPORTED_CAVITY_POLICIES:
            raise ValueError(
                "Route 2 cavity_policy must be warning-fallback or "
                "fixed-stability-branch."
            )
        if self.response not in {"frozen", "scf"}:
            raise ValueError("SMD response must be frozen or scf.")
        if (
            (
                self.profile_spec.reaction_field_projector != "local-jet"
                or self.profile_spec.model_field_gauge
                != "continuum-zero-at-infinity"
            )
            and self.response != "scf"
        ):
            raise ValueError(
                "A non-default reaction-field projector or model-field gauge "
                "requires response=scf; a frozen response would configure but "
                "never apply that model drive."
            )
        if self.standard_state != "1m":
            raise ValueError(
                "Route 2 uses the 1 M gas -> 1 M solution convention only; "
                "standard_state must be 1m."
            )

    def _cavity_policy_description(self) -> str:
        if self.cavity_policy == CAVITY_POLICY_FIXED_STABILITY_BRANCH:
            return (
                "Use AREA=0.28 A^2 and MINRADIUS=0.30 A from the first "
                "evaluation and fail closed on the native "
                "'PCMSolver warning.' stderr marker. PEDRA.OUT warnings are "
                "recorded separately and do not select a cavity branch. The "
                "discretization parameters are predetermined rather than "
                "selected from each geometry, removing policy-level branch "
                "switching only; GePol surface/operator derivatives and "
                "topology continuity remain unproven."
            )
        return (
            "Start with AREA=0.20 A^2 and no added spheres. If PCMSolver "
            "emits the native 'PCMSolver warning.' stderr marker, retry "
            "deterministically with "
            "AREA=0.28 A^2 and MINRADIUS=0.30 A; fail closed if a warning "
            "persists. PEDRA.OUT warnings are recorded separately and do not "
            "select a cavity branch. This warning-triggered branch is "
            "fixed-conformer energy infrastructure and is not a force/PES "
            "policy."
        )

    @staticmethod
    def _validate_domain(atoms) -> None:
        validate_route2_domain(atoms)

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

    def _ensure_pcm_stability_input(self) -> Path:
        """Reuse the established stability cavity without changing its audit file."""
        return self._ensure_pcm_fallback_input()

    def _cavity_attempt_specs(
        self,
    ) -> tuple[tuple[str, Callable[[], Path], float, float | None], ...]:
        stability = (
            CAVITY_POLICY_FIXED_STABILITY_BRANCH,
            self._ensure_pcm_stability_input,
            PCM_STABILITY_FALLBACK_TESSERA_AREA_ANGSTROM2,
            PCM_STABILITY_FALLBACK_MIN_RADIUS_ANGSTROM,
        )
        if self.cavity_policy == CAVITY_POLICY_FIXED_STABILITY_BRANCH:
            return (stability,)
        return (
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
        return Route2ContinuumEngine.gas_state(
            calculator,
            atoms,
            need_forces=False,
        )

    def _engine_for_reaction_field(
        self,
        reaction_field: FixedCavityPCMReactionFieldLinearMap,
        cds_result,
    ) -> Route2ContinuumEngine:
        return Route2ContinuumEngine(
            reaction_field_factory=lambda _atoms: reaction_field,
            cds_evaluator=lambda _atoms: cds_result,
            settings=self._engine_settings,
        )

    def _write_result_audit(
        self,
        *,
        gas_state,
        solvent_state,
        root_density_coefficients: np.ndarray,
        response_density_coefficients: np.ndarray,
        reaction_field_values_ev: np.ndarray,
        model_local_field_values_ev: np.ndarray | None,
        model_field_features: np.ndarray | None,
        reaction_field_projector: str,
        model_field_gauge: str,
        model_field_gauge_reference_ev: float,
        model_field_projection_contract: dict[str, object] | None,
        pcm_state: FixedCavityPCMSnapshot,
        session: PCMSolverSession,
        cds_result,
        components: dict[str, float],
        history: tuple[dict[str, float | int | None], ...],
    ) -> None:
        if self.audit_dir is None:
            return
        root_density = np.asarray(root_density_coefficients, dtype=float)
        response_density = np.asarray(
            response_density_coefficients,
            dtype=float,
        )
        density_residual_inf = float(
            np.max(np.abs(response_density - root_density))
        )
        archive_arrays = {
            "gas_density_coefficients": np.asarray(
                gas_state.density_coefficients
            ),
            # Legacy name retained for readers of schema <=5.
            "solvent_density_coefficients": response_density,
            "root_density_coefficients": root_density,
            "response_density_coefficients": response_density,
            "reaction_field_values_ev": np.asarray(
                reaction_field_values_ev,
                dtype=float,
            ),
            "cavity_centers_bohr": session.cavity_centers_bohr,
            "cavity_areas_bohr2": session.cavity_areas_bohr2,
            "mep_hartree_per_e": pcm_state.mep_hartree_per_e,
            "asc_e": pcm_state.asc_e,
            "reaction_potential_hartree_per_e": (
                pcm_state.reaction_potential_hartree_per_e
            ),
            "reaction_gradient_hartree_per_e_bohr": (
                pcm_state.reaction_gradient_hartree_per_e_bohr
            ),
            "cds_atom_areas_angstrom2": cds_result.atom_areas_angstrom2,
            "cds_atom_tensions_cal_mol_angstrom2": (
                cds_result.atom_tensions_cal_mol_angstrom2
            ),
        }
        if model_field_features is not None:
            archive_arrays["model_field_features"] = np.asarray(
                model_field_features,
                dtype=float,
            )
        if model_local_field_values_ev is not None:
            archive_arrays["model_local_field_values_ev"] = np.asarray(
                model_local_field_values_ev,
                dtype=float,
            )
        np.savez_compressed(
            self.audit_dir / "route2-state.npz",
            **archive_arrays,
        )
        payload = {
            "schema_version": 9,
            "response": self.response,
            "pcm_mep_projection": "cavity-exterior-point-multipole-l<=1",
            "reaction_field_projector": reaction_field_projector,
            "density_dual_field_gauge": "continuum-zero-at-infinity",
            "model_field_gauge": model_field_gauge,
            "model_field_gauge_reference_ev": (
                model_field_gauge_reference_ev
            ),
            "model_local_field_shape": (
                None
                if model_local_field_values_ev is None
                else list(np.asarray(model_local_field_values_ev).shape)
            ),
            "model_field_feature_shape": (
                None
                if model_field_features is None
                else list(np.asarray(model_field_features).shape)
            ),
            "model_field_projection_contract": (
                model_field_projection_contract
            ),
            "converged": True,
            "iterations": len(history),
            "fixed_point": {
                "applies": self.response == "scf",
                "same_root_energy_ledger": True,
                "root_density_definition": (
                    "density c supplied to the continuum operator"
                ),
                "response_density_definition": (
                    "unmixed MACE-POLAR response M(P(c))"
                ),
                "density_residual_inf_e": (
                    density_residual_inf
                    if self.response == "scf"
                    else None
                ),
            },
            "scf": {
                "maximum_iterations": (
                    self._engine_settings.scf_max_iterations
                ),
                "mixing": self._engine_settings.scf_mixing,
                "density_tolerance_e": (
                    self._engine_settings.scf_density_tolerance
                ),
                "energy_tolerance_ev": (
                    self._engine_settings.scf_energy_tolerance_ev
                ),
                "require_two_energy_samples": (
                    self._engine_settings.scf_require_two_energy_samples
                ),
                "history": list(history),
            },
            "energies_hartree": components,
            "gas_mace_energy_ev": float(gas_state.energy_ev),
            "solvent_intrinsic_mace_energy_ev": float(solvent_state.energy_ev),
            "density_reaction_coupling_hartree": (
                pcm_state.density_reaction_coupling_hartree
            ),
            "polarization_energy": {
                "provider_hartree": (
                    pcm_state.polarization_energy_hartree
                ),
                "paired_half_coupling_hartree": (
                    0.5
                    * pcm_state.density_reaction_coupling_hartree
                ),
                "absolute_identity_error_ev": (
                    abs(
                        pcm_state.polarization_energy_hartree
                        - 0.5
                        * pcm_state.density_reaction_coupling_hartree
                    )
                    * Hartree
                ),
            },
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
        assert self.audit_dir is not None
        cds_result = smd_water_cds(
            atoms.get_chemical_symbols(),
            atoms.get_positions(),
        )
        cavity_attempts: list[dict[str, Any]] = []
        selected_cavity: dict[str, Any] | None = None
        pedra_baseline = _pedra_file_fingerprints(self.audit_dir)
        with _pcmsolver_audit_working_directory(self.audit_dir):
            for (
                attempt_name,
                input_factory,
                tessera_area,
                minimum_radius,
            ) in self._cavity_attempt_specs():
                parsed_input = input_factory()
                native_stderr_path = (
                    self.audit_dir / f"pcmsolver-{attempt_name}-stderr.log"
                )
                session = PCMSolverSession(
                    np.asarray(atoms.numbers, dtype=float),
                    np.asarray(atoms.get_positions(), dtype=float) / Bohr,
                    parsed_input,
                )
                candidate_ready = False
                with _capture_process_stderr(native_stderr_path):
                    try:
                        initialization_started = time.perf_counter()
                        session.open()
                        initialization_seconds = (
                            time.perf_counter() - initialization_started
                        )
                        continuum_response = (
                            PCMSolverExternalMEPCavityResponse(
                                session,
                                cavity_radii_angstrom=(
                                    self.coulomb_radii_angstrom
                                ),
                            )
                        )
                        _flush_process_stderr()
                        native_stderr = native_stderr_path.read_text(
                            encoding="utf-8", errors="replace"
                        )
                        initialization_warning = (
                            PCM_WARNING_MARKER in native_stderr
                        )
                        attempt = {
                            "name": attempt_name,
                            "tessera_area_angstrom2": tessera_area,
                            "minimum_added_sphere_radius_angstrom": minimum_radius,
                            "warning_detected": initialization_warning,
                            "warning_stage": (
                                "pcm-initialization"
                                if initialization_warning
                                else None
                            ),
                            "stderr_log": str(native_stderr_path),
                            "cavity_tesserae": session.cavity_size,
                            "pcm_initialization_seconds": initialization_seconds,
                            "response_evaluated": False,
                            "response_evaluation_seconds": 0.0,
                        }
                        cavity_attempts.append(attempt)
                        if not initialization_warning:
                            response_started = time.perf_counter()
                            reaction_field = (
                                FixedCavityPCMReactionFieldLinearMap(
                                    continuum_response,
                                    np.asarray(
                                        atoms.get_positions(),
                                        dtype=float,
                                    ),
                                    model_field_gauge=(
                                        self.profile_spec.model_field_gauge
                                    ),
                                    **(
                                        {
                                            "model_field_projector": (
                                                ExactGTOFieldProjector(
                                                    calculator
                                                    .route2_gto_field_projection_spec()
                                                )
                                            )
                                        }
                                        if self.profile_spec
                                        .reaction_field_projector
                                        == "exact-gto-v1"
                                        else {}
                                    ),
                                )
                            )
                            engine = self._engine_for_reaction_field(
                                reaction_field,
                                cds_result,
                            )
                            if self.response == "frozen":
                                solvent_state = gas_state
                                root_density = engine.validate_density(
                                    gas_state.density_coefficients,
                                    len(atoms),
                                    name="Gas MACE-POLAR density",
                                )
                                response_density = root_density
                                reaction_field_values = (
                                    reaction_field.apply_scf(root_density)
                                )
                                model_local_field_values = None
                                model_field_features = None
                                reaction_field_projector = (
                                    self.profile_spec
                                    .reaction_field_projector
                                )
                                model_field_gauge = (
                                    self.profile_spec.model_field_gauge
                                )
                                model_field_gauge_reference_ev = 0.0
                                pcm_state = reaction_field.scf_snapshot(
                                    root_density
                                )
                                history = ()
                                components = (
                                    engine.compose_energy_components(
                                        gas_energy_ev=float(
                                            gas_state.energy_ev
                                        ),
                                        solvent_energy_ev=float(
                                            gas_state.energy_ev
                                        ),
                                        polarization_energy_hartree=(
                                            pcm_state
                                            .polarization_energy_hartree
                                        ),
                                        cds_energy_hartree=float(
                                            cds_result.energy_hartree
                                        ),
                                    )
                                )
                            else:
                                coupled: Route2CoupledState = (
                                    engine.solve_coupled_state(
                                        atoms,
                                        calculator,
                                        gas_state,
                                        provider_cache_signature=(
                                            self.profile,
                                            self.cavity_policy,
                                            attempt_name,
                                        ),
                                    )
                                )
                                solvent_state = coupled.solvent_state
                                root_density = (
                                    coupled.root_density_coefficients
                                )
                                response_density = (
                                    coupled.response_density_coefficients
                                )
                                reaction_field_values = (
                                    coupled.reaction_field_values_ev
                                )
                                model_field_features = (
                                    coupled.model_field_features
                                )
                                model_local_field_values = (
                                    coupled.model_local_field_values_ev
                                )
                                reaction_field_projector = (
                                    coupled.reaction_field_projector
                                )
                                model_field_gauge = (
                                    coupled.model_field_gauge
                                )
                                model_field_gauge_reference_ev = (
                                    coupled
                                    .model_field_gauge_reference_ev
                                )
                                pcm_state = reaction_field.scf_snapshot(
                                    root_density
                                )
                                history = coupled.history
                                components = engine.energy_components(
                                    gas_state,
                                    coupled,
                                )
                            attempt["response_evaluated"] = True
                            attempt["response_evaluation_seconds"] = (
                                time.perf_counter() - response_started
                            )
                            _flush_process_stderr()
                            native_stderr = native_stderr_path.read_text(
                                encoding="utf-8", errors="replace"
                            )
                            if PCM_WARNING_MARKER in native_stderr:
                                attempt["warning_detected"] = True
                                attempt["warning_stage"] = "response-evaluation"
                            else:
                                total = components["delta_g_solv"]
                                self._parsed_pcm_input_path = parsed_input
                                self._write_result_audit(
                                    gas_state=gas_state,
                                    solvent_state=solvent_state,
                                    root_density_coefficients=(
                                        root_density
                                    ),
                                    response_density_coefficients=(
                                        response_density
                                    ),
                                    reaction_field_values_ev=(
                                        reaction_field_values
                                    ),
                                    model_local_field_values_ev=(
                                        model_local_field_values
                                    ),
                                    model_field_features=(
                                        model_field_features
                                    ),
                                    reaction_field_projector=(
                                        reaction_field_projector
                                    ),
                                    model_field_gauge=(
                                        model_field_gauge
                                    ),
                                    model_field_gauge_reference_ev=(
                                        model_field_gauge_reference_ev
                                    ),
                                    model_field_projection_contract=(
                                        reaction_field
                                        .model_field_projection_provenance
                                    ),
                                    pcm_state=pcm_state,
                                    session=session,
                                    cds_result=cds_result,
                                    components=components,
                                    history=history,
                                )
                                pcmsolver_library = getattr(
                                    session,
                                    "library_source",
                                    None,
                                )
                                candidate_ready = True
                    finally:
                        session.close()

                native_stderr = native_stderr_path.read_text(
                    encoding="utf-8", errors="replace"
                )
                if (
                    PCM_WARNING_MARKER in native_stderr
                    and not attempt["warning_detected"]
                ):
                    attempt["warning_detected"] = True
                    attempt["warning_stage"] = "pcm-close"
                    candidate_ready = False
                    for filename in ("route2-result.json", "route2-state.npz"):
                        (self.audit_dir / filename).unlink(missing_ok=True)
                if attempt["warning_detected"]:
                    continue
                if not candidate_ready:
                    raise RuntimeError(
                        f"Route 2 cavity attempt {attempt_name!r} produced "
                        "neither a warning nor a publishable result."
                    )
                selected_cavity = attempt
                break

        if selected_cavity is None:
            for filename in ("route2-result.json", "route2-state.npz"):
                (self.audit_dir / filename).unlink(missing_ok=True)
            failure_payload = {
                "schema_version": 1,
                "reason": "pcmsolver-cavity-warning-for-all-policy-attempts",
                "pcmsolver_diagnostics": _pcmsolver_diagnostics(
                    self.audit_dir,
                    cavity_attempts,
                    pedra_baseline,
                ),
                "cavity_stability": {
                    "policy": self.cavity_policy,
                    "force_compatible": False,
                    "geometry_selection_branch_free": (
                        self._cavity_policy_geometry_selection_branch_free
                    ),
                    "selected": None,
                    "attempts": cavity_attempts,
                },
            }
            (self.audit_dir / "route2-failure.json").write_text(
                json.dumps(failure_payload, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            stderr_logs = ", ".join(
                attempt["stderr_log"] for attempt in cavity_attempts
            )
            raise RuntimeError(
                "PCMSolver emitted a warning for every cavity attempt under "
                f"policy {self.cavity_policy!r}; Route 2 refuses to publish "
                f"a numerically suspect result. See {stderr_logs}."
            )

        audit_path = self.audit_dir / "route2-result.json"
        audit_payload = json.loads(audit_path.read_text(encoding="utf-8"))
        pcmsolver_diagnostics = _pcmsolver_diagnostics(
            self.audit_dir,
            cavity_attempts,
            pedra_baseline,
        )
        audit_payload["cavity_stability"] = {
            "policy": self.cavity_policy,
            "force_compatible": False,
            "geometry_selection_branch_free": (
                self._cavity_policy_geometry_selection_branch_free
            ),
            "selected": selected_cavity["name"],
            "attempts": cavity_attempts,
        }
        audit_payload["pcmsolver_diagnostics"] = pcmsolver_diagnostics
        audit_path.write_text(
            json.dumps(audit_payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )

        provenance = {
            **self.provenance,
            "converged": True,
            "iterations": len(history),
            "same_root_energy_ledger": True,
            "fixed_point_density_residual_inf_e": (
                None
                if self.response == "frozen"
                else float(
                    np.max(
                        np.abs(response_density - root_density)
                    )
                )
            ),
            "density_reaction_coupling_hartree": (
                pcm_state.density_reaction_coupling_hartree
            ),
            "cavity_tesserae": int(pcm_state.asc_e.size),
            "cavity_stability": {
                "policy": self.cavity_policy,
                "selected": selected_cavity["name"],
                "force_compatible": False,
                "geometry_selection_branch_free": (
                    self._cavity_policy_geometry_selection_branch_free
                ),
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
            "pcmsolver_diagnostics": {
                "native_stderr_warning_count": (
                    pcmsolver_diagnostics["native_stderr_warning_count"]
                ),
                "pedra_warning_count": pcmsolver_diagnostics[
                    "pedra_warning_count"
                ],
                "pedra_warnings_are_selection_fatal": False,
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
            "graph_longrange_version": getattr(
                calculator, "graph_longrange_version", None
            ),
            "model_field_projection_contract": (
                reaction_field.model_field_projection_provenance
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
    "CAVITY_POLICY_FIXED_STABILITY_BRANCH",
    "CAVITY_POLICY_WARNING_FALLBACK",
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
    "SUPPORTED_CAVITY_POLICIES",
    "SUPPORTED_ELEMENTS",
]
