"""Strict fixed-charge ddX linearized Poisson--Boltzmann provider."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from maple.function.read.filereader.mol2_reader import (
    MOL2_ATOM_ID_ARRAY,
    mol2_identity_sha256,
)

from .common import build_openmm_topology
from .radii import OpenMMMbondi2RadiusProvider
from .result import SolvationResult

BOHR_ANGSTROM = 0.529177210903
PROFILE_NAME = "ddlpb-union-mbondi2-v1"
_PYDDX_VERSION = "0.8.0"
_AUDITED_BINARY_SHA256 = (
    "697bafe818a749bb70963ef13a36496bcba52fc427046e8eefe6e769a0d68845"
)
_LEBEDEV_ORDERS = frozenset(
    {
        6,
        14,
        26,
        38,
        50,
        74,
        86,
        110,
        146,
        170,
        194,
        230,
        266,
        302,
        350,
        434,
        590,
        770,
        974,
        1202,
        1454,
        1730,
        2030,
        2354,
        2702,
        3074,
        3470,
        3890,
        4334,
        4802,
        5294,
        5810,
    }
)


@dataclass(frozen=True)
class DDXLPBSettings:
    """Numerical controls for the direct validation API.

    Normal MAPLE input uses the zero-argument, locked profile.  Explicit
    instances allow controlled numerical-refinement studies without widening
    the command-language surface.
    """

    lmax: int = 9
    n_lebedev: int = 302
    eta: float = 0.1
    shift: float = 0.0
    solver_tolerance: float = 1.0e-10
    max_iterations: int = 200
    jacobi_n_diis: int = 20
    n_proc: int = 1
    solvent_epsilon: float = 78.5
    solute_epsilon: float = 1.0
    enable_fmm: bool = False


def _integer(name: str, value: Any, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(  # noqa: TRY004
            f"ddX LPB {name} must be an integer."
        )
    result = int(value)
    if result < minimum:
        raise ValueError(f"ddX LPB {name} must be at least {minimum}.")
    return result


def _finite_float(name: str, value: Any) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(  # noqa: TRY004
            f"ddX LPB {name} must be a finite real number, not a boolean."
        )
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"ddX LPB {name} must be finite.") from exc
    if not np.isfinite(result):
        raise ValueError(f"ddX LPB {name} must be finite.")
    return result


def _validate_settings(settings: DDXLPBSettings) -> None:
    if not isinstance(settings, DDXLPBSettings):
        raise ValueError(  # noqa: TRY004
            "ddX LPB settings must be a DDXLPBSettings instance."
        )
    _integer("lmax", settings.lmax, minimum=0)
    n_lebedev = _integer("n_lebedev", settings.n_lebedev, minimum=1)
    if n_lebedev not in _LEBEDEV_ORDERS:
        raise ValueError(
            f"ddX LPB n_lebedev={n_lebedev} is not a supported Lebedev grid size."
        )
    eta = _finite_float("eta", settings.eta)
    shift = _finite_float("shift", settings.shift)
    tolerance = _finite_float("solver_tolerance", settings.solver_tolerance)
    solvent_epsilon = _finite_float("solvent_epsilon", settings.solvent_epsilon)
    solute_epsilon = _finite_float("solute_epsilon", settings.solute_epsilon)
    _integer("max_iterations", settings.max_iterations, minimum=1)
    _integer("jacobi_n_diis", settings.jacobi_n_diis, minimum=1)
    _integer("n_proc", settings.n_proc, minimum=1)
    if not 0.0 <= eta <= 1.0:
        raise ValueError("ddX LPB eta must be in the closed interval [0, 1].")
    if not -1.0 <= shift <= 1.0:
        raise ValueError("ddX LPB shift must be in the closed interval [-1, 1].")
    if tolerance <= 0.0:
        raise ValueError("ddX LPB solver_tolerance must be strictly positive.")
    if solvent_epsilon <= 0.0:
        raise ValueError("ddX LPB solvent_epsilon must be strictly positive.")
    if solute_epsilon != 1.0:
        raise ValueError("ddX LPB solute_epsilon is fixed to 1.0.")
    if settings.enable_fmm is not False:
        raise ValueError("ddX LPB enable_fmm is fixed to False.")


def _sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ImportError(
            "ddX LPB native import failed: could not hash the loaded pyddx binary."
        ) from exc
    return digest.hexdigest()


def _require_pyddx():
    """Load and identity-check the optional native dependency lazily."""
    try:
        pyddx = importlib.import_module("pyddx")
    except ImportError as exc:
        raise ImportError(
            "ddX LPB native import failed: install the optional "
            "`maple[implicit-ddlpb]` dependencies."
        ) from exc
    version = getattr(pyddx, "__version__", None)
    if version != _PYDDX_VERSION:
        raise ImportError(
            f"ddX LPB requires pyddx=={_PYDDX_VERSION}; loaded version {version!r}."
        )
    binary = getattr(pyddx, "__file__", None)
    if not binary:
        raise ImportError(
            "ddX LPB native import failed: loaded pyddx has no binary path."
        )
    return pyddx, _sha256_file(binary)


def _array_sha256(array: np.ndarray) -> str:
    canonical = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(canonical.dtype).encode("ascii"))
    digest.update(np.asarray(canonical.shape, dtype=np.int64).tobytes())
    digest.update(canonical.tobytes())
    return digest.hexdigest()


def _readonly_copy(array: np.ndarray) -> np.ndarray:
    result = np.asarray(array).copy()
    result.setflags(write=False)
    return result


def _mol2_identity(atoms) -> str | None:
    metadata = atoms.info.get("mol2")
    if not metadata:
        return None
    return mol2_identity_sha256(metadata)


def _bind_mol2_atom_identity(atoms) -> tuple[int, ...]:
    metadata = atoms.info.get("mol2") or {}
    atom_ids = tuple(int(value) for value in metadata.get("atom_ids") or [])
    if len(atom_ids) != len(atoms) or len(set(atom_ids)) != len(atoms):
        raise ValueError(
            "ddX LPB requires one unique MOL2 atom ID per atom for frozen identity."
        )
    identity = atoms.arrays.get(MOL2_ATOM_ID_ARRAY)
    if identity is None:
        atoms.new_array(MOL2_ATOM_ID_ARRAY, np.asarray(atom_ids, dtype=np.int64))
        return atom_ids
    identity_array = np.asarray(identity)
    if (
        identity_array.shape != (len(atoms),)
        or not np.issubdtype(identity_array.dtype, np.integer)
        or tuple(int(value) for value in identity_array) != atom_ids
    ):
        raise ValueError(
            "ddX LPB current atom order does not match the frozen MOL2 atom IDs."
        )
    return atom_ids


class DDXLPB:
    """Polar-only ddX LPB energy and complete analytical coordinate force."""

    supported_properties = frozenset({"energy", "forces"})

    def __init__(
        self,
        atoms,
        charges,
        *,
        solvent_kappa_inverse_angstrom: float,
        settings: DDXLPBSettings | None = None,
        audit_dir: str | os.PathLike[str] | None = None,
    ):
        self.settings = DDXLPBSettings() if settings is None else settings
        _validate_settings(self.settings)
        kappa = _finite_float(
            "solvent_kappa_inverse_angstrom", solvent_kappa_inverse_angstrom
        )
        if kappa <= 0.0:
            raise ValueError(
                "ddX LPB solvent_kappa_inverse_angstrom must be strictly positive."
            )
        if np.any(np.asarray(atoms.get_pbc(), dtype=bool)):
            raise ValueError("ddX LPB supports nonperiodic structures only.")
        positions = np.asarray(atoms.get_positions(), dtype=np.float64)
        if positions.shape != (len(atoms), 3) or not np.isfinite(positions).all():
            raise ValueError("ddX LPB requires finite coordinates with shape (N, 3).")
        fixed_charges = np.asarray(charges, dtype=np.float64)
        if fixed_charges.shape != (len(atoms),) or not np.isfinite(fixed_charges).all():
            raise ValueError("ddX LPB requires one finite partial charge per atom.")

        atom_ids = _bind_mol2_atom_identity(atoms)
        topology = build_openmm_topology(atoms)
        self.radius_provider = OpenMMMbondi2RadiusProvider()
        self.radius_result = self.radius_provider.assign(topology)
        radii = np.asarray(self.radius_result.radii_angstrom, dtype=np.float64)
        if (
            radii.shape != (len(atoms),)
            or not np.isfinite(radii).all()
            or np.any(radii <= 0.0)
        ):
            raise ValueError("ddX LPB requires one finite positive radius per atom.")

        self.atoms = atoms.copy()
        self._charges = fixed_charges.copy()
        self._radii = radii.copy()
        self.solvent_kappa_inverse_angstrom = kappa
        self.audit_dir = Path(audit_dir) if audit_dir is not None else None
        self._atomic_numbers = tuple(int(value) for value in atoms.get_atomic_numbers())
        self._atom_ids = atom_ids
        self._mol2_identity_sha256 = _mol2_identity(atoms)

    @property
    def charges(self) -> np.ndarray:
        return _readonly_copy(self._charges)

    @property
    def radii(self) -> np.ndarray:
        return _readonly_copy(self._radii)

    @property
    def provenance(self) -> dict[str, Any]:
        """Static setup provenance; solve diagnostics are result-local."""
        return {
            "provider": "ddx",
            "method": "lpb",
            "profile": PROFILE_NAME,
            "role": "Reference",
            "reference_only": True,
            "absolute_solvation_free_energy_claim": False,
            "model": "lpb",
            "nonpolar": "none",
            "polar_only": True,
            "fixed_charge": True,
            "fixed_radius": True,
            "dynamic_charge_api": False,
            "higher_multipoles": False,
            "radius_response": False,
            "energy_unit": "hartree",
            "force_unit": "hartree/angstrom",
            "coordinate_input_unit": "angstrom",
            "native_length_unit": "bohr",
            "kappa_input_unit": "angstrom^-1",
            "native_kappa_unit": "bohr^-1",
            "solvent_kappa_inverse_angstrom": self.solvent_kappa_inverse_angstrom,
            "solute_dielectric": self.settings.solute_epsilon,
            "solvent_dielectric": self.settings.solvent_epsilon,
            "cavity": "union-of-atomic-balls-no-probe-no-radius-shift",
            "radii": "mbondi2",
            "radius_provider": self.radius_result.provenance,
            "settings": asdict(self.settings),
            "gradient_convention": (
                "native positive dE/dR is the sum of ddrun solvation terms and "
                "multipole_force_terms; MAPLE force is -dE/dR"
            ),
            "charges_sha256": _array_sha256(self._charges),
            "radii_sha256": _array_sha256(self._radii),
            "atomic_numbers": list(self._atomic_numbers),
            "mol2_atom_ids": list(self._atom_ids),
            "mol2_identity_sha256": self._mol2_identity_sha256,
            "required_provider_version": _PYDDX_VERSION,
            "audited_binary_sha256": _AUDITED_BINARY_SHA256,
        }

    def _validate_atoms(self, atoms) -> np.ndarray:
        if np.any(np.asarray(atoms.get_pbc(), dtype=bool)):
            raise ValueError("ddX LPB supports nonperiodic structures only.")
        atomic_numbers = tuple(int(value) for value in atoms.get_atomic_numbers())
        identity = atoms.arrays.get(MOL2_ATOM_ID_ARRAY)
        current_atom_ids = (
            None
            if identity is None
            else tuple(int(value) for value in np.asarray(identity))
        )
        if (
            len(atoms) != len(self._charges)
            or atomic_numbers != self._atomic_numbers
            or current_atom_ids != self._atom_ids
            or _mol2_identity(atoms) != self._mol2_identity_sha256
        ):
            raise ValueError(
                "ddX LPB is bound to the frozen atom identity, order, and MOL2 topology."
            )
        positions = np.asarray(atoms.get_positions(), dtype=np.float64)
        if (
            positions.shape != (len(self._charges), 3)
            or not np.isfinite(positions).all()
        ):
            raise ValueError("ddX LPB requires finite coordinates with shape (N, 3).")
        return positions

    def evaluate(
        self, atoms, need_forces: bool = False, calculator=None
    ) -> SolvationResult:
        del calculator
        positions = self._validate_atoms(atoms)
        pyddx, binary_sha256 = _require_pyddx()
        centers_bohr = np.asfortranarray(positions.T / BOHR_ANGSTROM)
        radii_bohr = self._radii / BOHR_ANGSTROM
        kappa_bohr_inverse = self.solvent_kappa_inverse_angstrom * BOHR_ANGSTROM

        try:
            model = pyddx.Model(
                "lpb",
                centers_bohr,
                radii_bohr,
                solvent_epsilon=float(self.settings.solvent_epsilon),
                solvent_kappa=kappa_bohr_inverse,
                eta=float(self.settings.eta),
                shift=float(self.settings.shift),
                lmax=int(self.settings.lmax),
                n_lebedev=int(self.settings.n_lebedev),
                maxiter=int(self.settings.max_iterations),
                jacobi_n_diis=int(self.settings.jacobi_n_diis),
                n_proc=int(self.settings.n_proc),
                enable_fmm=False,
                enable_force=bool(need_forces),
            )
        except Exception as exc:
            raise RuntimeError("ddX LPB model construction failed.") from exc

        multipoles = np.asfortranarray(
            self._charges.reshape(1, -1) / np.sqrt(4.0 * np.pi)
        )
        try:
            field = model.multipole_electrostatics(multipoles)
            psi = model.multipole_psi(multipoles)
        except Exception as exc:
            raise RuntimeError("ddX LPB electrostatics construction failed.") from exc
        if (
            not isinstance(field, dict)
            or "phi" not in field
            or "e" not in field
            or (need_forces and "g" not in field)
        ):
            raise ValueError(
                "ddX LPB electrostatics returned incomplete phi/e/g fields."
            )
        phi = np.asarray(field["phi"], dtype=np.float64)
        electric_field = np.asarray(field["e"], dtype=np.float64)
        psi_array = np.asarray(psi, dtype=np.float64)
        if (
            phi.ndim != 1
            or phi.size == 0
            or not np.isfinite(phi).all()
            or electric_field.shape != (3, phi.size)
            or not np.isfinite(electric_field).all()
            or psi_array.ndim != 2
            or psi_array.shape[1] != len(self._charges)
            or not np.isfinite(psi_array).all()
        ):
            raise ValueError(
                "ddX LPB electrostatics returned invalid phi/e/psi arrays."
            )
        if need_forces:
            field_gradient = np.asarray(field["g"], dtype=np.float64)
            if (
                field_gradient.shape != (3, 3, phi.size)
                or not np.isfinite(field_gradient).all()
            ):
                raise ValueError(
                    "ddX LPB electrostatics returned an invalid field-gradient array."
                )

        try:
            state = pyddx.State(model, psi, field["phi"], field["e"])
            native_result = state.ddrun(
                field, tol=float(self.settings.solver_tolerance)
            )
        except Exception as exc:
            raise RuntimeError("ddX LPB native solve failed.") from exc
        if need_forces:
            if not isinstance(native_result, (tuple, list)) or len(native_result) != 2:
                raise ValueError(
                    "ddX LPB native force solve returned an invalid result tuple."
                )
            energy, solvation_gradient = native_result
        else:
            if isinstance(native_result, (tuple, list)):
                raise ValueError(
                    "ddX LPB native energy solve returned a non-scalar result."
                )
            energy = native_result
            solvation_gradient = None
        if not bool(state.is_solved):
            raise RuntimeError("ddX LPB forward solve did not converge.")
        if need_forces and not bool(state.is_solved_adjoint):
            raise RuntimeError("ddX LPB adjoint solve did not converge.")

        try:
            energy_hartree = float(energy)
        except (TypeError, ValueError) as exc:
            raise ValueError("ddX LPB returned an invalid polar energy.") from exc
        if not np.isfinite(energy_hartree):
            raise ValueError("ddX LPB returned a non-finite polar energy.")

        forces: np.ndarray | None = None
        if need_forces:
            solvation = np.asarray(solvation_gradient, dtype=np.float64)
            if (
                solvation.shape != (3, len(self._charges))
                or not np.isfinite(solvation).all()
            ):
                raise ValueError(
                    "ddX LPB returned an invalid solvation gradient with expected shape (3, N)."
                )
            try:
                multipole = np.asarray(
                    state.multipole_force_terms(multipoles), dtype=np.float64
                )
            except Exception as exc:
                raise RuntimeError(
                    "ddX LPB multipole force evaluation failed."
                ) from exc
            if (
                multipole.shape != (3, len(self._charges))
                or not np.isfinite(multipole).all()
            ):
                raise ValueError(
                    "ddX LPB returned invalid multipole force terms with expected shape (3, N)."
                )
            forces = -(solvation + multipole).T / BOHR_ANGSTROM
            if forces.shape != positions.shape or not np.isfinite(forces).all():
                raise ValueError("ddX LPB returned a non-finite force array.")

        try:
            x_n_iter = int(state.x_n_iter)
            s_n_iter = int(state.s_n_iter) if need_forces else None
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "ddX LPB returned invalid solver iteration counts."
            ) from exc
        if x_n_iter < 0 or (s_n_iter is not None and s_n_iter < 0):
            raise ValueError("ddX LPB returned negative solver iteration counts.")

        binary_path = getattr(pyddx, "__file__", None)
        if not isinstance(binary_path, (str, os.PathLike)):
            raise ImportError(  # noqa: TRY004
                "ddX LPB native import failed: loaded pyddx has no binary path."
            )
        runtime = {
            "provider_version": str(pyddx.__version__),
            "binary_path": str(Path(binary_path).resolve()),
            "binary_sha256": binary_sha256,
            "matches_audited_binary": binary_sha256 == _AUDITED_BINARY_SHA256,
        }
        provenance = {
            **self.provenance,
            "provider_version": str(pyddx.__version__),
            "provider_binary_sha256": binary_sha256,
            "native_runtime": runtime,
            "coordinate_sha256": _array_sha256(positions),
            "solver": {
                "requested_tolerance": float(self.settings.solver_tolerance),
                "maximum_iterations": int(self.settings.max_iterations),
                "is_solved": True,
                "is_solved_adjoint": True if need_forces else None,
                "x_n_iter": x_n_iter,
                "s_n_iter": s_n_iter,
                "residual_not_exposed_by_pyddx_0_8": True,
            },
        }
        result = SolvationResult(
            energy_hartree=energy_hartree,
            forces_hartree_per_angstrom=forces,
            components_hartree={"polar": energy_hartree},
            provenance=provenance,
        )
        if self.audit_dir is not None:
            self._write_result_audit(result)
        return result

    def _write_result_audit(self, result: SolvationResult) -> None:
        assert self.audit_dir is not None
        payload = {
            "energy_hartree": result.energy_hartree,
            "forces_hartree_per_angstrom": (
                None
                if result.forces_hartree_per_angstrom is None
                else result.forces_hartree_per_angstrom.tolist()
            ),
            "components_hartree": result.components_hartree,
            "provenance": result.provenance,
        }
        self.audit_dir.mkdir(parents=True, exist_ok=True)
        target = self.audit_dir / "ddx-lpb.result.json"
        temporary_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.audit_dir,
                prefix=".ddx-lpb.result.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary_name = handle.name
                json.dump(payload, handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, target)
        except OSError as exc:
            if temporary_name is not None:
                try:
                    Path(temporary_name).unlink(missing_ok=True)
                except OSError:
                    pass
            raise RuntimeError("ddX LPB result audit write failed.") from exc
