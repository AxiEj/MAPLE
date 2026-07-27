"""Optional ddPCM/ddCOSMO maps for atom-centred MACE-POLAR multipoles.

This research adapter keeps one mathematical object responsible for the
fixed-geometry reaction map, its discrete adjoint, polarization energy, and
complete coordinate derivative.  It is intentionally independent of the
surface-MEP/ASC provider contract because ddX natively accepts atom-centred
multipoles.

``pyddx`` is imported lazily and version-gated.  The adapter is wired only into
explicit, non-default experimental Route-2 research paths.  pyddx 0.8.0
returns unscaled COSMO quantities, so the host applies ``(epsilon-1)/epsilon``
to the energy, reaction map, and coordinate derivative as one inseparable
physical factor.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import math
from typing import Any, Literal

import numpy as np
from ase.units import Bohr, Hartree

from .gto_density import (
    density_to_external_field_order,
    external_field_to_density_order,
)
from .route2_derivative import (
    FULL_REACTION_FIELD_POSITION_DERIVATIVE_CONTRACT_VERSION,
)

TESTED_PYDDX_VERSION = "0.8.0"
_SUPPORTED_CONTINUUM_MODELS = frozenset({"pcm", "cosmo"})


@dataclass(frozen=True)
class _PyDDXRuntime:
    """Versioned optional runtime bundle, injectable for contract tests."""

    version: str
    module: Any


def _load_pyddx_runtime() -> _PyDDXRuntime:
    try:
        module = importlib.import_module("pyddx")
    except ImportError as exc:
        raise RuntimeError(
            "The optional pyddx Route-2 research adapter requires "
            f"pyddx {TESTED_PYDDX_VERSION}; MAPLE does not install it "
            "as a core dependency."
        ) from exc
    return _PyDDXRuntime(
        version=str(getattr(module, "__version__", "unknown")),
        module=module,
    )


def _require_tested_pyddx_version(version: object) -> str:
    normalized = str(version)
    if normalized != TESTED_PYDDX_VERSION:
        raise RuntimeError(
            "The optional pyddx Route-2 research adapter is tested only "
            f"with pyddx {TESTED_PYDDX_VERSION}; received {normalized}."
        )
    return normalized


def _validated_density_block(
    values: np.ndarray,
    *,
    atom_count: int | None,
    name: str,
) -> np.ndarray:
    block = np.asarray(values, dtype=float)
    expected_shape = "(n_atoms, 4)" if atom_count is None else str((atom_count, 4))
    valid_shape = (
        block.ndim == 2
        and block.shape[0] > 0
        and block.shape[1] == 4
        and (atom_count is None or block.shape[0] == atom_count)
    )
    if not valid_shape or not np.all(np.isfinite(block)):
        raise ValueError(
            f"{name} must be finite with shape {expected_shape}; "
            f"received {block.shape}."
        )
    return block


def mace_polar_density_to_pyddx_multipoles(
    density_coefficients: np.ndarray,
) -> np.ndarray:
    """Convert raw MACE-POLAR ``l<=1`` coefficients to ddX multipoles.

    MACE-POLAR stores ``[q, y, z, x]`` with charge in elementary-charge units
    and dipoles in elementary-charge angstrom.  ddX uses orthonormal real
    spherical multipoles in atomic units, so the corresponding factors are
    ``sqrt(4*pi)`` for ``l=0`` and ``sqrt(4*pi/3)`` for ``l=1``.  The direct
    component order is retained; Cartesian reordering belongs only at the
    external node-field boundary.
    """

    density = _validated_density_block(
        density_coefficients,
        atom_count=None,
        name="density_coefficients",
    )
    multipoles = np.empty((4, density.shape[0]), dtype=float)
    multipoles[0] = density[:, 0] / math.sqrt(4.0 * math.pi)
    multipoles[1:] = density[:, 1:].T / (Bohr * math.sqrt(4.0 * math.pi / 3.0))
    return multipoles


class PyDDXReactionFieldLinearMap:
    """Reciprocal pyddx map and complete position VJP for MACE multipoles."""

    reciprocal_energy_pairing = True
    full_position_derivative_contract_version = (
        FULL_REACTION_FIELD_POSITION_DERIVATIVE_CONTRACT_VERSION
    )

    def __init__(
        self,
        positions_angstrom: np.ndarray,
        radii_angstrom: np.ndarray,
        *,
        continuum_model: Literal["pcm", "cosmo"],
        dielectric: float,
        lmax: int,
        n_lebedev: int,
        n_proc: int = 1,
        solver_tolerance: float = 1.0e-10,
        eta: float = 0.1,
        _runtime: _PyDDXRuntime | None = None,
    ) -> None:
        normalized_model = str(continuum_model).strip().lower()
        if normalized_model not in _SUPPORTED_CONTINUUM_MODELS:
            raise ValueError(
                "continuum_model must be either 'pcm' or 'cosmo'; "
                f"received {continuum_model!r}."
            )
        method_label = "ddPCM" if normalized_model == "pcm" else "ddCOSMO"
        positions = np.asarray(positions_angstrom, dtype=float)
        if (
            positions.ndim != 2
            or positions.shape[0] == 0
            or positions.shape[1] != 3
            or not np.all(np.isfinite(positions))
        ):
            raise ValueError(
                "positions_angstrom must be finite with shape (n_atoms, 3)."
            )
        radii = np.asarray(radii_angstrom, dtype=float)
        if (
            radii.shape != (positions.shape[0],)
            or not np.all(np.isfinite(radii))
            or np.any(radii <= 0.0)
        ):
            raise ValueError(
                "radii_angstrom must be finite and positive with shape "
                f"{(positions.shape[0],)}."
            )
        dielectric_value = float(dielectric)
        if not math.isfinite(dielectric_value) or dielectric_value <= 1.0:
            raise ValueError(
                f"{method_label} dielectric must be finite and greater than 1."
            )
        if (
            isinstance(lmax, bool)
            or not isinstance(lmax, (int, np.integer))
            or int(lmax) < 1
        ):
            raise ValueError(
                f"{method_label} lmax must be an integer of at least 1."
            )
        if (
            isinstance(n_lebedev, bool)
            or not isinstance(n_lebedev, (int, np.integer))
            or int(n_lebedev) <= 0
        ):
            raise ValueError(
                f"{method_label} n_lebedev must be a positive integer."
            )
        if (
            isinstance(n_proc, bool)
            or not isinstance(n_proc, (int, np.integer))
            or int(n_proc) <= 0
        ):
            raise ValueError(
                f"{method_label} n_proc must be a positive integer."
            )
        tolerance = float(solver_tolerance)
        if not math.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError(
                f"{method_label} solver_tolerance must be finite and positive."
            )
        eta_value = float(eta)
        if not math.isfinite(eta_value) or not 0.0 <= eta_value <= 1.0:
            raise ValueError(
                f"{method_label} eta must be finite and lie in [0, 1]."
            )

        runtime = _load_pyddx_runtime() if _runtime is None else _runtime
        version = _require_tested_pyddx_version(runtime.version)
        model = runtime.module.Model(
            normalized_model,
            (positions / Bohr).T,
            radii / Bohr,
            dielectric_value,
            eta=eta_value,
            shift=0.0,
            lmax=int(lmax),
            n_lebedev=int(n_lebedev),
            enable_fmm=False,
            n_proc=int(n_proc),
            enable_force=True,
        )
        if int(getattr(model, "n_spheres", -1)) != positions.shape[0]:
            raise RuntimeError(
                f"pyddx {method_label} model returned an invalid sphere count."
            )
        if getattr(model, "has_force_enabled", False) is not True:
            raise RuntimeError(
                f"pyddx {method_label} model did not enable analytic "
                "coordinate derivatives."
            )
        backend_n_proc = getattr(model, "n_proc", None)
        if (
            isinstance(backend_n_proc, bool)
            or not isinstance(backend_n_proc, (int, np.integer))
            or int(backend_n_proc) != int(n_proc)
        ):
            raise RuntimeError(
                f"pyddx {method_label} model did not retain the requested "
                "thread count."
            )

        self._runtime = runtime
        self._version = version
        self._model = model
        self._continuum_model = normalized_model
        self._method_label = method_label
        self._positions_angstrom = positions.copy()
        self._radii_angstrom = radii.copy()
        self._dielectric = dielectric_value
        self._dielectric_scaling = (
            1.0
            if normalized_model == "pcm"
            else (dielectric_value - 1.0) / dielectric_value
        )
        self._dielectric_scaling_source = (
            "native-pyddx-pcm"
            if normalized_model == "pcm"
            else "pyddx-0.8.0-host-applied-(epsilon-1)/epsilon"
        )
        self._lmax = int(lmax)
        self._n_lebedev = int(n_lebedev)
        self._n_proc = int(backend_n_proc)
        self._solver_tolerance = tolerance
        self._eta = eta_value
        self.atom_count = positions.shape[0]
        self._scf_state = None
        self._scf_state_has_adjoint_solution = False
        self._scf_state_creations = 0
        self._scf_state_updates = 0
        self._scf_forward_warm_starts = 0
        self._scf_adjoint_warm_starts = 0

    @property
    def runtime_provenance(self) -> dict[str, object]:
        return {
            "backend": "pyddx",
            "pyddx_version": self._version,
            "model": self._continuum_model,
            "method": self._method_label,
            "dielectric": self._dielectric,
            "dielectric_scaling": self._dielectric_scaling,
            "dielectric_scaling_source": self._dielectric_scaling_source,
            "lmax": self._lmax,
            "n_lebedev": self._n_lebedev,
            "solver_tolerance": self._solver_tolerance,
            "eta": self._eta,
            "shift": 0.0,
            "enable_fmm": False,
            "n_proc": self._n_proc,
            "scf_state_reuse": "pyddx.State.update_problem warm start",
            "scf_state_creations": self._scf_state_creations,
            "scf_state_updates": self._scf_state_updates,
            "scf_forward_warm_starts": self._scf_forward_warm_starts,
            "scf_adjoint_warm_starts": self._scf_adjoint_warm_starts,
        }

    @property
    def reference_positions_angstrom(self) -> np.ndarray:
        return self._positions_angstrom.copy()

    @property
    def cavity_radii_angstrom(self) -> np.ndarray:
        return self._radii_angstrom.copy()

    def _validated_density(self, values: np.ndarray, *, name: str) -> np.ndarray:
        return _validated_density_block(
            values,
            atom_count=self.atom_count,
            name=name,
        )

    def _solve_state(
        self,
        density: np.ndarray,
        *,
        coordinate_derivative: bool,
        solve_adjoint: bool,
        warm_start: bool = False,
    ) -> tuple[np.ndarray, dict[str, np.ndarray], Any]:
        multipoles = mace_polar_density_to_pyddx_multipoles(density)
        derivative_order = -1 if coordinate_derivative else 0
        electrostatics = self._model.multipole_electrostatics(
            multipoles,
            derivative_order=derivative_order,
        )
        phi = np.asarray(electrostatics.get("phi"), dtype=float)
        psi = np.asarray(self._model.multipole_psi(multipoles), dtype=float)
        if not np.all(np.isfinite(phi)) or not np.all(np.isfinite(psi)):
            raise RuntimeError(
                f"pyddx {self._method_label} source construction returned "
                "non-finite data."
            )

        state = self._scf_state if warm_start else None
        try:
            if state is None:
                state = self._runtime.module.State(self._model, psi, phi)
                state.fill_guess(self._solver_tolerance)
                if warm_start:
                    if not callable(getattr(state, "update_problem", None)):
                        raise RuntimeError(
                            "pyddx State does not expose the tested "
                            "update_problem warm-start API."
                        )
                    self._scf_state = state
                    self._scf_state_creations += 1
            else:
                state.update_problem(psi, phi)
                self._scf_state_updates += 1
                self._scf_forward_warm_starts += 1
            state.solve(self._solver_tolerance)
            if not np.all(np.isfinite(np.asarray(state.x, dtype=float))):
                raise RuntimeError(
                    f"pyddx {self._method_label} forward solution contains "
                    "non-finite data."
                )
            if solve_adjoint:
                if warm_start and self._scf_state_has_adjoint_solution:
                    self._scf_adjoint_warm_starts += 1
                else:
                    state.fill_guess_adjoint(self._solver_tolerance)
                state.solve_adjoint(self._solver_tolerance)
                if warm_start:
                    self._scf_state_has_adjoint_solution = True
                if not np.all(np.isfinite(np.asarray(state.xi, dtype=float))):
                    raise RuntimeError(
                        f"pyddx {self._method_label} adjoint solution contains "
                        "non-finite data."
                    )
        except Exception:
            if warm_start:
                self._scf_state = None
                self._scf_state_has_adjoint_solution = False
            raise
        return multipoles, electrostatics, state

    def _polarization_energy_hartree(
        self,
        density_coefficients: np.ndarray,
        *,
        warm_start: bool,
    ) -> float:
        density = self._validated_density(
            density_coefficients,
            name="density_coefficients",
        )
        _, _, state = self._solve_state(
            density,
            coordinate_derivative=False,
            solve_adjoint=False,
            warm_start=warm_start,
        )
        energy = self._dielectric_scaling * float(state.energy())
        if not math.isfinite(energy):
            raise RuntimeError(
                f"pyddx {self._method_label} polarization energy is non-finite."
            )
        return energy

    def polarization_energy_hartree(
        self,
        density_coefficients: np.ndarray,
    ) -> float:
        return self._polarization_energy_hartree(
            density_coefficients,
            warm_start=False,
        )

    def scf_polarization_energy_hartree(
        self,
        density_coefficients: np.ndarray,
    ) -> float:
        """Evaluate final SCF energy using the sequential-state warm start."""

        return self._polarization_energy_hartree(
            density_coefficients,
            warm_start=True,
        )

    def _raw_reaction_gradient_hartree(
        self,
        density_coefficients: np.ndarray,
        *,
        warm_start: bool = False,
    ) -> np.ndarray:
        density = self._validated_density(
            density_coefficients,
            name="density_coefficients",
        )
        _, _, state = self._solve_state(
            density,
            coordinate_derivative=False,
            solve_adjoint=True,
            warm_start=warm_start,
        )
        forward_solution = np.asarray(state.x, dtype=float)
        adjoint_solution = np.asarray(state.xi, dtype=float)
        gradient = np.empty(density.size, dtype=float)
        basis_density = np.zeros_like(density)
        # These basis calls differentiate the two source maps after exactly one
        # forward and one adjoint continuum solve; they are not 4N additional
        # continuum solves. Recomputing the cheap source projections avoids
        # retaining O(4N * (n_cav + N*n_basis)) geometry-sized arrays.
        for flat_index in range(density.size):
            basis_density.fill(0.0)
            basis_density.reshape(-1)[flat_index] = 1.0
            basis_multipoles = mace_polar_density_to_pyddx_multipoles(basis_density)
            basis_psi = np.asarray(
                self._model.multipole_psi(basis_multipoles),
                dtype=float,
            )
            basis_phi = np.asarray(
                self._model.multipole_electrostatics(
                    basis_multipoles,
                    derivative_order=0,
                )["phi"],
                dtype=float,
            )
            gradient[flat_index] = 0.5 * (
                float(np.vdot(basis_psi, forward_solution))
                - float(np.vdot(basis_phi, adjoint_solution))
            )
        gradient = self._dielectric_scaling * gradient.reshape(density.shape)
        if not np.all(np.isfinite(gradient)):
            raise RuntimeError(
                f"pyddx {self._method_label} reaction field contains "
                "non-finite data."
            )

        energy = self._dielectric_scaling * float(state.energy())
        paired_energy = 0.5 * float(np.vdot(density, gradient))
        scale = max(1.0, abs(energy), abs(paired_energy))
        relative_error = abs(paired_energy - energy) / scale
        allowed_error = max(100.0 * self._solver_tolerance, 1.0e-10)
        if not math.isfinite(energy) or relative_error > allowed_error:
            raise RuntimeError(
                f"pyddx {self._method_label} reaction field failed the "
                "polarization-energy "
                f"identity (relative error {relative_error:.3e})."
            )
        return gradient

    def apply(self, density_direction: np.ndarray) -> np.ndarray:
        """Map raw MACE density coefficients to external field order."""

        density = self._validated_density(
            density_direction,
            name="density_direction",
        )
        raw_gradient = self._raw_reaction_gradient_hartree(density)
        return density_to_external_field_order(raw_gradient) * Hartree

    def apply_scf(self, density_direction: np.ndarray) -> np.ndarray:
        """Apply the map with a warm start for sequential ML-SCF densities."""

        density = self._validated_density(
            density_direction,
            name="density_direction",
        )
        raw_gradient = self._raw_reaction_gradient_hartree(
            density,
            warm_start=True,
        )
        return density_to_external_field_order(raw_gradient) * Hartree

    def adjoint(self, field_cotangent: np.ndarray) -> np.ndarray:
        """Apply the reciprocal reaction map in the raw density-dual order."""

        external = self._validated_density(
            field_cotangent,
            name="field_cotangent",
        )
        raw_cotangent = external_field_to_density_order(external)
        return self._raw_reaction_gradient_hartree(raw_cotangent) * Hartree

    def _coordinate_energy_gradient_hartree_per_bohr(
        self,
        density_coefficients: np.ndarray,
    ) -> np.ndarray:
        density = self._validated_density(
            density_coefficients,
            name="density_coefficients",
        )
        multipoles, electrostatics, state = self._solve_state(
            density,
            coordinate_derivative=True,
            solve_adjoint=True,
        )
        coordinate_gradient = np.asarray(
            state.solvation_force_terms(electrostatics),
            dtype=float,
        )
        coordinate_gradient += np.asarray(
            state.multipole_force_terms(multipoles),
            dtype=float,
        )
        # pyddx 0.8.0 constructs this Fortran-layout result as
        # (3, n_spheres).  The strict version gate above makes a layout change
        # an incompatibility rather than silently guessing a transpose.
        expected_shape = (3, self.atom_count)
        if coordinate_gradient.shape != expected_shape or not np.all(
            np.isfinite(coordinate_gradient)
        ):
            raise RuntimeError(
                f"pyddx {self._method_label} coordinate derivative must be "
                "finite with shape "
                f"{expected_shape}; received {coordinate_gradient.shape}."
            )

        # Despite the upstream "force" method names, the pyddx 0.8.0 test
        # suite defines this returned array by central dE/dR finite differences.
        return self._dielectric_scaling * coordinate_gradient.T

    def polarization_position_gradient_ev_per_angstrom(
        self,
        density_coefficients: np.ndarray,
    ) -> np.ndarray:
        """Return ``dE_pol/dR`` at fixed source multipole coefficients.

        Geometry-adaptive source models must add their own source-response
        chain rule separately.  Keeping this term explicit prevents double
        counting the charge/multipole response.
        """

        return (
            self._coordinate_energy_gradient_hartree_per_bohr(
                density_coefficients
            )
            * Hartree
            / Bohr
        )

    def full_position_vjp(
        self,
        density: np.ndarray,
        field_cotangent: np.ndarray,
    ) -> np.ndarray:
        """Differentiate ``<field_cotangent, P_R density>`` completely.

        For the reciprocal reaction map ``P_R`` and
        ``E_R(c)=0.5*c.T@P_R@c``, the bilinear coordinate derivative follows
        from the exact polarization identity

        ``d(a.T@P_R@b)/dR = 0.5*(grad E_R(a+b)-grad E_R(a-b))``.
        """

        right = self._validated_density(density, name="density")
        external_left = self._validated_density(
            field_cotangent,
            name="field_cotangent",
        )
        left = external_field_to_density_order(external_left)
        gradient_plus = self._coordinate_energy_gradient_hartree_per_bohr(left + right)
        gradient_minus = self._coordinate_energy_gradient_hartree_per_bohr(left - right)
        return 0.5 * (gradient_plus - gradient_minus) * Hartree / Bohr


class PyDDXPCMReactionFieldLinearMap(PyDDXReactionFieldLinearMap):
    """Compatibility wrapper selecting the pyddx ddPCM equation."""

    def __init__(
        self,
        positions_angstrom: np.ndarray,
        radii_angstrom: np.ndarray,
        *,
        dielectric: float,
        lmax: int,
        n_lebedev: int,
        n_proc: int = 1,
        solver_tolerance: float = 1.0e-10,
        eta: float = 0.1,
        _runtime: _PyDDXRuntime | None = None,
    ) -> None:
        super().__init__(
            positions_angstrom,
            radii_angstrom,
            continuum_model="pcm",
            dielectric=dielectric,
            lmax=lmax,
            n_lebedev=n_lebedev,
            n_proc=n_proc,
            solver_tolerance=solver_tolerance,
            eta=eta,
            _runtime=_runtime,
        )


class PyDDXCOSMOReactionFieldLinearMap(PyDDXReactionFieldLinearMap):
    """pyddx ddCOSMO equation with host-applied finite-dielectric scaling."""

    def __init__(
        self,
        positions_angstrom: np.ndarray,
        radii_angstrom: np.ndarray,
        *,
        dielectric: float,
        lmax: int,
        n_lebedev: int,
        n_proc: int = 1,
        solver_tolerance: float = 1.0e-10,
        eta: float = 0.1,
        _runtime: _PyDDXRuntime | None = None,
    ) -> None:
        super().__init__(
            positions_angstrom,
            radii_angstrom,
            continuum_model="cosmo",
            dielectric=dielectric,
            lmax=lmax,
            n_lebedev=n_lebedev,
            n_proc=n_proc,
            solver_tolerance=solver_tolerance,
            eta=eta,
            _runtime=_runtime,
        )


__all__ = [
    "TESTED_PYDDX_VERSION",
    "PyDDXCOSMOReactionFieldLinearMap",
    "PyDDXPCMReactionFieldLinearMap",
    "PyDDXReactionFieldLinearMap",
    "mace_polar_density_to_pyddx_multipoles",
]
