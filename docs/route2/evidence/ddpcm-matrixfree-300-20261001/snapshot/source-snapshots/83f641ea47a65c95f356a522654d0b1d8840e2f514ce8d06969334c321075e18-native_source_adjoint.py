"""Independent pyddx-0.8 source-adjoint probe for learned point multipoles.

This module is deliberately research-only.  It contracts the analytic source
Jacobians with an already solved native ddX state without making the legacy
``4*N`` calls to ``multipole_psi`` and ``multipole_electrostatics``.  It does
not implement the Torch continuum path, MACE, CDS, Hessians, or admission.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
from ase.units import Bohr

from maple.solvation.api.units import HARTREE_TO_EV


_SQRT_4PI = math.sqrt(4.0 * math.pi)
_SQRT_4PI_OVER_3 = math.sqrt(4.0 * math.pi / 3.0)


def raw_density_to_pyddx_multipoles(density: np.ndarray) -> np.ndarray:
    """Map raw ``[q, y, z, x]`` coefficients to pyddx atomic units."""

    values = np.asarray(density, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 4 or not np.all(np.isfinite(values)):
        raise ValueError("density must be finite with shape (n_atoms, 4)")
    multipoles = np.empty((4, values.shape[0]), dtype=np.float64)
    multipoles[0] = values[:, 0] / _SQRT_4PI
    multipoles[1:] = values[:, 1:].T / (Bohr * _SQRT_4PI_OVER_3)
    return multipoles


def tiled_raw_source_vjp_hartree(
    model: Any,
    state: Any,
    *,
    source_tile: int = 16,
) -> np.ndarray:
    """Return ``dE/d[q,y,z,x]`` in Hartree by exact tiled contraction.

    The native state energy derivative is

    ``0.5 * ((dpsi/dc).T @ state.x - (dphi/dc).T @ state.xi)``.

    ``psi`` is local for atom-centred l<=1 multipoles.  ``phi`` is evaluated
    from the exact point source expression at the exposed native cavity points.
    Source tiling bounds temporary storage by ``O(n_cavity * source_tile)``.
    """

    if (
        isinstance(source_tile, bool)
        or not isinstance(source_tile, (int, np.integer))
        or int(source_tile) < 1
    ):
        raise ValueError("source_tile must be a positive integer")
    source_tile = int(source_tile)
    centres = np.asarray(model.sphere_centres, dtype=np.float64).T
    radii = np.asarray(model.sphere_radii, dtype=np.float64)
    cavity = np.asarray(model.cavity, dtype=np.float64).T
    x = np.asarray(state.x, dtype=np.float64)
    xi = np.asarray(state.xi, dtype=np.float64)
    atom_count = centres.shape[0]
    if centres.shape != (atom_count, 3) or radii.shape != (atom_count,):
        raise ValueError("invalid pyddx sphere geometry")
    if x.ndim != 2 or x.shape[1] != atom_count or x.shape[0] < 4:
        raise ValueError("state.x is incompatible with l<=1 source contraction")
    if cavity.ndim != 2 or cavity.shape[1] != 3 or xi.shape != (len(cavity),):
        raise ValueError("state.xi is incompatible with the native cavity")
    if not all(np.all(np.isfinite(value)) for value in (centres, radii, cavity, x, xi)):
        raise RuntimeError("native source-adjoint inputs contain non-finite data")
    if np.any(radii <= 0.0):
        raise RuntimeError("native cavity radii must be positive")
    if getattr(state, "is_solved", False) is not True:
        raise RuntimeError("native forward state is not solved")
    if getattr(state, "is_solved_adjoint", False) is not True:
        raise RuntimeError("native adjoint state is not solved")

    # dpsi/d(raw c): pyddx uses 4*pi/(2*l+1)/radius**l times
    # orthonormal multipoles.  Raw dipoles are supplied in Angstrom, so their
    # conversion to atomic units contributes 1/Bohr.
    gradient = np.empty((atom_count, 4), dtype=np.float64)
    gradient[:, 0] = _SQRT_4PI * x[0]
    gradient[:, 1:] = (
        _SQRT_4PI_OVER_3 * x[1:4].T / (radii[:, None] * Bohr)
    )

    # dphi/d(raw c), with raw dipole component order y,z,x.  All geometry is
    # native pyddx Bohr; raw dipoles therefore carry the same 1/Bohr factor as
    # raw_density_to_pyddx_multipoles.
    cartesian_for_raw = np.array((1, 2, 0), dtype=np.intp)
    for start in range(0, atom_count, source_tile):
        stop = min(start + source_tile, atom_count)
        delta = cavity[:, None, :] - centres[None, start:stop, :]
        inverse_r = 1.0 / np.linalg.norm(delta, axis=2)
        charge_vjp = np.einsum("p,pt->t", xi, inverse_r, optimize=True)
        dipole_cartesian = np.einsum(
            "p,pt,ptk->tk",
            xi,
            inverse_r**3,
            delta,
            optimize=True,
        ) / Bohr
        gradient[start:stop, 0] -= charge_vjp
        gradient[start:stop, 1:] -= dipole_cartesian[:, cartesian_for_raw]
    return 0.5 * gradient


def tiled_raw_source_vjp_ev(
    model: Any,
    state: Any,
    *,
    source_tile: int = 16,
) -> np.ndarray:
    """Return the raw learned-source VJP in eV per raw coefficient."""

    return HARTREE_TO_EV * tiled_raw_source_vjp_hartree(
        model, state, source_tile=source_tile
    )


@dataclass(frozen=True)
class NativeSolveResult:
    energy_ev: float
    source_gradient_ev: np.ndarray
    position_gradient_ev_per_angstrom: np.ndarray
    forward_iterations: int
    adjoint_iterations: int
    n_cavity: int
    n_basis: int
    requested_tolerance: float
    original_residual_measured_independently: bool

    @property
    def fixed_source_force_ev_per_angstrom(self) -> np.ndarray:
        """Physical force ``-dE/dR`` at fixed raw learned-source coefficients."""

        return -self.position_gradient_ev_per_angstrom


def solve_native_reference(
    positions_angstrom: np.ndarray,
    radii_angstrom: np.ndarray,
    density: np.ndarray,
    *,
    dielectric: float = 78.39,
    lmax: int = 15,
    n_lebedev: int = 1202,
    eta: float = 0.1,
    tolerance: float = 1.0e-12,
    source_tile: int = 16,
    n_proc: int = 1,
) -> NativeSolveResult:
    """Run an unmodified native pyddx solve and analytic reference gradients."""

    import pyddx

    if str(pyddx.__version__) != "0.8.0":
        raise RuntimeError(f"probe requires pyddx 0.8.0, found {pyddx.__version__}")
    positions = np.asarray(positions_angstrom, dtype=np.float64)
    radii = np.asarray(radii_angstrom, dtype=np.float64)
    raw = np.asarray(density, dtype=np.float64)
    if (
        positions.ndim != 2
        or positions.shape[0] == 0
        or positions.shape[1] != 3
        or not np.all(np.isfinite(positions))
    ):
        raise ValueError("positions_angstrom must be finite with shape (n_atoms, 3)")
    if radii.shape != (len(positions),) or raw.shape != (len(positions), 4):
        raise ValueError("radii and density must match the atom count")
    if not np.all(np.isfinite(radii)) or np.any(radii <= 0.0):
        raise ValueError("radii_angstrom must be finite and positive")
    if not np.all(np.isfinite(raw)):
        raise ValueError("density must contain only finite values")
    tolerance_value = float(tolerance)
    if (
        not math.isfinite(tolerance_value)
        or tolerance_value <= 0.0
        or tolerance_value > 1.0e-12
    ):
        raise ValueError("tolerance must be finite, positive, and at most 1e-12")
    if (
        isinstance(source_tile, bool)
        or not isinstance(source_tile, (int, np.integer))
        or int(source_tile) < 1
    ):
        raise ValueError("source_tile must be a positive integer")
    multipoles = raw_density_to_pyddx_multipoles(raw)
    model = pyddx.Model(
        "pcm",
        (positions / Bohr).T,
        radii / Bohr,
        float(dielectric),
        eta=float(eta),
        shift=0.0,
        lmax=int(lmax),
        n_lebedev=int(n_lebedev),
        enable_fmm=False,
        incore=False,
        n_proc=int(n_proc),
        enable_force=True,
    )
    electrostatics = model.multipole_electrostatics(multipoles, derivative_order=-1)
    state = pyddx.State(model, model.multipole_psi(multipoles), electrostatics["phi"])
    state.fill_guess(tolerance_value)
    state.solve(tolerance_value)
    state.fill_guess_adjoint(tolerance_value)
    state.solve_adjoint(tolerance_value)
    if state.is_solved is not True or state.is_solved_adjoint is not True:
        raise RuntimeError("native pyddx solve did not complete both state equations")
    position_gradient = np.asarray(
        state.solvation_force_terms(electrostatics), dtype=np.float64
    ) + np.asarray(state.multipole_force_terms(multipoles), dtype=np.float64)
    energy_ev = float(state.energy()) * HARTREE_TO_EV
    source_gradient = tiled_raw_source_vjp_ev(
            model, state, source_tile=source_tile
        )
    position_gradient_ev = position_gradient.T * HARTREE_TO_EV / Bohr
    if not math.isfinite(energy_ev):
        raise RuntimeError("native polarization energy is non-finite")
    if not np.all(np.isfinite(source_gradient)):
        raise RuntimeError("native source gradient is non-finite")
    if not np.all(np.isfinite(position_gradient_ev)):
        raise RuntimeError("native fixed-source position gradient is non-finite")
    return NativeSolveResult(
        energy_ev=energy_ev,
        source_gradient_ev=source_gradient,
        position_gradient_ev_per_angstrom=position_gradient_ev,
        forward_iterations=int(state.x_n_iter),
        adjoint_iterations=int(state.s_n_iter),
        n_cavity=int(model.n_cav),
        n_basis=int(model.n_basis),
        requested_tolerance=tolerance_value,
        original_residual_measured_independently=False,
    )
