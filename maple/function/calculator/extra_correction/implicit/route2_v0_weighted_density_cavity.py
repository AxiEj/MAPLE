r"""No-label weighted-density cavitation scalar for Route-2 V0-AQ-C.

This module implements only the *cavity-formation* part of the
weighted-density construction of Sundararaman, Gunceler, and Arias,
J. Chem. Phys. 141, 134105 (2014).  It is deliberately separate from the
dielectric reaction block and from any dispersion approximation:

.. math::

   \bar{s}=W_s s,\qquad
   G_{\rm cav}[s]=\int f_s(\bar{s}(\mathbf r))\,d\mathbf r.

``s`` is a solvent-*centre* occupancy: zero in an excluded cavity and one in
bulk liquid.  ``W_s`` is a normalized, self-adjoint nearest-neighbour shell
average.  The local function is fixed by five *pure-liquid* conditions--the
small-cavity and small-droplet limits plus the planar surface tension--rather
than by solvation labels:

.. math::

   f_s(x)=p_s(1-x)+N_sT_sx(1-x)\left[
       x+(1-x)\Gamma_s+15x(1-x)A_s\right],

   \Gamma_s=\log(N_sT_s/p_{\rm vap,s})-1,\qquad
   A_s=\frac{\sigma_s}{N_sT_sR_s}-\frac{1+\Gamma_s}{6}.

All internal quantities are atomic units.  :meth:`from_si` is merely an
explicit unit conversion helper; it does not turn arbitrary user input into a
source-provenanced solvent asset.  A physical caller must bind every input to
independent pure-liquid/equation-of-state evidence and must separately prove
the map from an auxiliary electron density to this *solvent-centre* occupancy.

The analytical discrete derivative is retained exactly.  This matters because
a future auxiliary electronic Euler equation must receive the derivative of
the same nonpolar scalar, not an area correction appended after a PCM solve.

This is not a total solvation model: it contains no solute--solvent dispersion
or Pauli functional, no standard-state term, no auxiliary electronic
functional, and no identification of ``s`` with the electrostatic cavity.
In particular it never exposes a fitted ``s6``, SMD/CDS surface tension, or a
solvation-error-selected radius.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field

import numpy as np
from ase.units import Bohr, Hartree, kB

from .route2_v0_structured_solvent import RegularCartesianGrid

V0_WEIGHTED_DENSITY_CAVITY_CONSTRUCTION = "route2-v0-weighted-density-cavity-v1"
V0_WEIGHTED_DENSITY_CAVITY_SCOPE = "cavitation-scalar-only-not-total-v1"

_ELECTRONIC_CHARGE_JOULE = 1.602176634e-19
_HARTREE_PER_BOHR3_TO_PASCAL = (
    Hartree * _ELECTRONIC_CHARGE_JOULE / (Bohr * 1.0e-10) ** 3
)
_HARTREE_PER_BOHR2_TO_NEWTON_PER_METER = (
    Hartree * _ELECTRONIC_CHARGE_JOULE / (Bohr * 1.0e-10) ** 2
)


def _finite_positive(value: object, *, name: str) -> float:
    """Return one finite positive scalar without accepting booleans."""

    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, float, np.integer, np.floating),
    ):
        raise TypeError(f"{name} must be a finite positive real number.")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return result


def _immutable_grid(
    values: np.ndarray,
    *,
    shape: tuple[int, int, int],
    name: str,
    nonnegative: bool = False,
) -> np.ndarray:
    """Validate and freeze one finite real grid field."""

    if np.iscomplexobj(values):
        raise ValueError(f"{name} must be real-valued.")
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite real grid field.") from exc
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite with shape {shape}.")
    if nonnegative and np.any(array < 0.0):
        raise ValueError(f"{name} must be nonnegative.")
    result = np.array(array, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _periodic_reverse(values: np.ndarray) -> np.ndarray:
    """Return the periodic ``r -> -r`` partner in the grid convention."""

    indices = tuple((-np.arange(length)) % length for length in values.shape)
    return values[np.ix_(*indices)]


def _real_fft_result(values: np.ndarray, *, name: str) -> np.ndarray:
    """Reject a material imaginary residual from a real periodic FFT map."""

    real = np.real(values)
    scale = max(1.0, float(np.max(np.abs(real))))
    imaginary = float(np.max(np.abs(np.imag(values))))
    if imaginary > 128.0 * np.finfo(float).eps * scale:
        raise RuntimeError(f"{name} produced a non-real FFT result.")
    return np.array(real, dtype=float, copy=True)


def _operator_fingerprint(
    *,
    grid: RegularCartesianGrid,
    thermodynamics: Route2V0WeightedDensityCavityThermodynamics,
    kernel: np.ndarray,
) -> str:
    """Hash all numerical data that define one discrete cavity functional."""

    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(grid.origin_bohr, dtype=np.float64).tobytes())
    digest.update(np.ascontiguousarray(grid.spacing_bohr, dtype=np.float64).tobytes())
    digest.update(np.asarray(grid.shape, dtype=np.int64).tobytes())
    digest.update(np.ascontiguousarray(kernel, dtype=np.float64).tobytes())
    digest.update(
        np.asarray(
            (
                thermodynamics.pressure_hartree_per_bohr3,
                thermodynamics.molecular_number_density_per_bohr3,
                thermodynamics.thermal_energy_hartree,
                thermodynamics.vapor_pressure_hartree_per_bohr3,
                thermodynamics.surface_tension_hartree_per_bohr2,
                thermodynamics.solvent_vdw_radius_bohr,
            ),
            dtype=np.float64,
        ).tobytes()
    )
    return digest.hexdigest()


@dataclass(frozen=True)
class Route2V0WeightedDensityCavityThermodynamics:
    """Pure-liquid inputs to the no-label weighted-density cavitation scalar.

    Inputs are in atomic units and describe the *same* temperature/pressure
    state.  ``solvent_vdw_radius_bohr`` is the solvent centre's thermodynamic
    van der Waals radius used by the nearest-neighbour shell, not a fitted
    solute atom radius or a PCM radius selected from solvation errors.
    """

    pressure_hartree_per_bohr3: float
    molecular_number_density_per_bohr3: float
    thermal_energy_hartree: float
    vapor_pressure_hartree_per_bohr3: float
    surface_tension_hartree_per_bohr2: float
    solvent_vdw_radius_bohr: float

    def __post_init__(self) -> None:
        for name in (
            "pressure_hartree_per_bohr3",
            "molecular_number_density_per_bohr3",
            "thermal_energy_hartree",
            "vapor_pressure_hartree_per_bohr3",
            "surface_tension_hartree_per_bohr2",
            "solvent_vdw_radius_bohr",
        ):
            object.__setattr__(
                self,
                name,
                _finite_positive(getattr(self, name), name=name),
            )

    @classmethod
    def from_si(
        cls,
        *,
        temperature_kelvin: float,
        pressure_pascal: float,
        molecular_number_density_angstrom3: float,
        vapor_pressure_pascal: float,
        surface_tension_newton_per_meter: float,
        solvent_vdw_radius_angstrom: float,
    ) -> Route2V0WeightedDensityCavityThermodynamics:
        """Convert explicit independent pure-liquid properties to atomic units.

        The method intentionally accepts no fitted solvation coefficient and
        stores no implicit defaults.  Provenance and state/model binding are a
        higher-level source-record responsibility.
        """

        temperature = _finite_positive(temperature_kelvin, name="temperature_kelvin")
        pressure = _finite_positive(pressure_pascal, name="pressure_pascal")
        density = _finite_positive(
            molecular_number_density_angstrom3,
            name="molecular_number_density_angstrom3",
        )
        vapor_pressure = _finite_positive(
            vapor_pressure_pascal,
            name="vapor_pressure_pascal",
        )
        tension = _finite_positive(
            surface_tension_newton_per_meter,
            name="surface_tension_newton_per_meter",
        )
        radius = _finite_positive(
            solvent_vdw_radius_angstrom,
            name="solvent_vdw_radius_angstrom",
        )
        return cls(
            pressure_hartree_per_bohr3=pressure / _HARTREE_PER_BOHR3_TO_PASCAL,
            molecular_number_density_per_bohr3=density * Bohr**3,
            thermal_energy_hartree=(kB * temperature) / Hartree,
            vapor_pressure_hartree_per_bohr3=(
                vapor_pressure / _HARTREE_PER_BOHR3_TO_PASCAL
            ),
            surface_tension_hartree_per_bohr2=(
                tension / _HARTREE_PER_BOHR2_TO_NEWTON_PER_METER
            ),
            solvent_vdw_radius_bohr=radius / Bohr,
        )

    @property
    def small_droplet_log_term(self) -> float:
        """Return ``Gamma = log(N*T/p_vap) - 1`` from the droplet limit."""

        ratio = (
            self.molecular_number_density_per_bohr3 * self.thermal_energy_hartree
        ) / self.vapor_pressure_hartree_per_bohr3
        if not math.isfinite(ratio) or ratio <= 0.0:
            raise RuntimeError("Weighted-density droplet ratio is invalid.")
        result = math.log(ratio) - 1.0
        if not math.isfinite(result):
            raise RuntimeError("Weighted-density droplet log term is invalid.")
        return result

    @property
    def surface_coefficient(self) -> float:
        """Return the fourth-order coefficient fixed by planar tension."""

        denominator = (
            self.molecular_number_density_per_bohr3
            * self.thermal_energy_hartree
            * self.solvent_vdw_radius_bohr
        )
        result = (
            self.surface_tension_hartree_per_bohr2 / denominator
            - (1.0 + self.small_droplet_log_term) / 6.0
        )
        if not math.isfinite(result):
            raise RuntimeError("Weighted-density surface coefficient is invalid.")
        return result


@dataclass(frozen=True)
class Route2V0WeightedDensityCavityState:
    """One immutable weighted-occupancy evaluation of the cavity scalar."""

    solvent_center_occupancy: np.ndarray
    weighted_occupancy: np.ndarray
    cavity_energy_hartree: float
    operator_fingerprint: str
    construction: str = V0_WEIGHTED_DENSITY_CAVITY_CONSTRUCTION
    response_scope: str = V0_WEIGHTED_DENSITY_CAVITY_SCOPE

    def __post_init__(self) -> None:
        shape = np.asarray(self.solvent_center_occupancy).shape
        if len(shape) != 3:
            raise ValueError("Weighted-density cavity state occupancy must be 3D.")
        grid_shape = (int(shape[0]), int(shape[1]), int(shape[2]))
        occupancy = _immutable_grid(
            self.solvent_center_occupancy,
            shape=grid_shape,
            name="Weighted-density cavity state occupancy",
        )
        weighted = _immutable_grid(
            self.weighted_occupancy,
            shape=grid_shape,
            name="Weighted-density cavity state weighted occupancy",
        )
        if np.any(occupancy < 0.0) or np.any(occupancy > 1.0):
            raise ValueError("Weighted-density cavity occupancy must lie in [0, 1].")
        if np.any(weighted < 0.0) or np.any(weighted > 1.0):
            raise ValueError(
                "Weighted-density cavity weighted occupancy must lie in [0, 1]."
            )
        energy = float(self.cavity_energy_hartree)
        if not math.isfinite(energy):
            raise ValueError("Weighted-density cavity energy must be finite.")
        fingerprint = self.operator_fingerprint
        if not isinstance(fingerprint, str) or len(fingerprint) != 64:
            raise ValueError("Weighted-density cavity state fingerprint is invalid.")
        if self.construction != V0_WEIGHTED_DENSITY_CAVITY_CONSTRUCTION:
            raise ValueError("Unsupported weighted-density cavity state construction.")
        if self.response_scope != V0_WEIGHTED_DENSITY_CAVITY_SCOPE:
            raise ValueError("Unsupported weighted-density cavity state scope.")
        object.__setattr__(self, "solvent_center_occupancy", occupancy)
        object.__setattr__(self, "weighted_occupancy", weighted)
        object.__setattr__(self, "cavity_energy_hartree", energy)


@dataclass(frozen=True)
class Route2V0WeightedDensityCavityFunctional:
    """Discrete no-label weighted-density cavity-formation functional.

    The supplied shell kernel must be a nonnegative normalized even function
    sampled with its origin at index ``(0, 0, 0)``.  These conditions make the
    periodic convolution both a convex occupancy average and its own discrete
    adjoint.  The implementation leaves construction of that physical shell
    kernel to a source-bound solvent asset instead of burying a grid resolution
    or shell quadrature in the model.
    """

    grid: RegularCartesianGrid
    thermodynamics: Route2V0WeightedDensityCavityThermodynamics
    nearest_neighbor_shell_kernel_per_bohr3: np.ndarray
    construction: str = V0_WEIGHTED_DENSITY_CAVITY_CONSTRUCTION
    response_scope: str = V0_WEIGHTED_DENSITY_CAVITY_SCOPE
    _operator_fingerprint: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.grid, RegularCartesianGrid):
            raise TypeError(
                "Weighted-density cavity requires a regular Cartesian grid."
            )
        if not isinstance(
            self.thermodynamics,
            Route2V0WeightedDensityCavityThermodynamics,
        ):
            raise TypeError(
                "Weighted-density cavity requires pure-liquid thermodynamics."
            )
        if self.construction != V0_WEIGHTED_DENSITY_CAVITY_CONSTRUCTION:
            raise ValueError("Unsupported Route-2 V0 weighted-density construction.")
        if self.response_scope != V0_WEIGHTED_DENSITY_CAVITY_SCOPE:
            raise ValueError("Unsupported Route-2 V0 weighted-density scope.")
        kernel = _immutable_grid(
            self.nearest_neighbor_shell_kernel_per_bohr3,
            shape=self.grid.shape,
            name="Weighted-density nearest-neighbour shell kernel",
            nonnegative=True,
        )
        normalization = self.grid.volume_element_bohr3 * float(np.sum(kernel))
        if not math.isclose(normalization, 1.0, rel_tol=0.0, abs_tol=2.0e-13):
            raise ValueError(
                "Weighted-density shell kernel must integrate to one under the grid pairing."
            )
        reversed_kernel = _periodic_reverse(kernel)
        scale = max(1.0, float(np.max(np.abs(kernel))))
        if float(np.max(np.abs(kernel - reversed_kernel))) > (
            128.0 * np.finfo(float).eps * scale
        ):
            raise ValueError(
                "Weighted-density shell kernel must be even for an exact self-adjoint map."
            )
        fingerprint = _operator_fingerprint(
            grid=self.grid,
            thermodynamics=self.thermodynamics,
            kernel=kernel,
        )
        object.__setattr__(self, "nearest_neighbor_shell_kernel_per_bohr3", kernel)
        object.__setattr__(self, "_operator_fingerprint", fingerprint)

    @property
    def operator_fingerprint(self) -> str:
        """Return the immutable identity of this discrete scalar."""

        return self._operator_fingerprint

    @property
    def is_total_solvation_asset(self) -> bool:
        """Return false because cavity formation alone is not total solvation."""

        return False

    def _convolution(self, values: np.ndarray) -> np.ndarray:
        """Apply ``W_s`` under the periodic volume-pairing convention."""

        raw = self.grid.volume_element_bohr3 * np.fft.ifftn(
            np.fft.fftn(self.nearest_neighbor_shell_kernel_per_bohr3)
            * np.fft.fftn(values)
        )
        return _real_fft_result(raw, name="Weighted-density shell convolution")

    def _adjoint_convolution(self, values: np.ndarray) -> np.ndarray:
        """Apply the exact discrete adjoint of :meth:`_convolution`."""

        raw = self.grid.volume_element_bohr3 * np.fft.ifftn(
            np.conj(np.fft.fftn(self.nearest_neighbor_shell_kernel_per_bohr3))
            * np.fft.fftn(values)
        )
        return _real_fft_result(raw, name="Weighted-density shell adjoint")

    def _local_energy_density_hartree_per_bohr3(self, values: np.ndarray) -> np.ndarray:
        """Return ``f_s(x)`` from the fixed pure-liquid constraints."""

        x = values
        thermodynamics = self.thermodynamics
        gamma = thermodynamics.small_droplet_log_term
        coefficient = thermodynamics.surface_coefficient
        pair = x * (1.0 - x)
        bracket = x + (1.0 - x) * gamma + 15.0 * pair * coefficient
        return (
            thermodynamics.pressure_hartree_per_bohr3 * (1.0 - x)
            + thermodynamics.molecular_number_density_per_bohr3
            * thermodynamics.thermal_energy_hartree
            * pair
            * bracket
        )

    def _local_energy_derivative_hartree_per_bohr3(
        self,
        values: np.ndarray,
    ) -> np.ndarray:
        """Return ``df_s/dx`` for the exact weighted-density VJP."""

        x = values
        thermodynamics = self.thermodynamics
        gamma = thermodynamics.small_droplet_log_term
        coefficient = thermodynamics.surface_coefficient
        pair = x * (1.0 - x)
        pair_derivative = 1.0 - 2.0 * x
        bracket = x + (1.0 - x) * gamma + 15.0 * pair * coefficient
        bracket_derivative = 1.0 - gamma + 15.0 * pair_derivative * coefficient
        return (
            -thermodynamics.pressure_hartree_per_bohr3
            + thermodynamics.molecular_number_density_per_bohr3
            * thermodynamics.thermal_energy_hartree
            * (pair_derivative * bracket + pair * bracket_derivative)
        )

    def _validate_state(self, state: Route2V0WeightedDensityCavityState) -> None:
        if not isinstance(state, Route2V0WeightedDensityCavityState):
            raise TypeError(
                "Weighted-density cavity derivative requires a cavity state."
            )
        if state.operator_fingerprint != self._operator_fingerprint:
            raise ValueError(
                "Weighted-density cavity state does not match this functional."
            )

    def evaluate(
        self,
        solvent_center_occupancy: np.ndarray,
    ) -> Route2V0WeightedDensityCavityState:
        """Evaluate the cavity scalar for one smooth solvent-centre occupancy."""

        occupancy = _immutable_grid(
            solvent_center_occupancy,
            shape=self.grid.shape,
            name="Weighted-density solvent-centre occupancy",
        )
        if np.any(occupancy < 0.0) or np.any(occupancy > 1.0):
            raise ValueError(
                "Weighted-density solvent-centre occupancy must lie in [0, 1]."
            )
        weighted = self._convolution(occupancy)
        tolerance = 512.0 * np.finfo(float).eps
        if (
            float(np.min(weighted)) < -tolerance
            or float(np.max(weighted)) > 1.0 + tolerance
        ):
            raise RuntimeError(
                "Weighted-density shell average escaped [0, 1]; the supplied kernel "
                "is not a valid positive occupancy average on this grid."
            )
        # The clipping affects only round-off at the mathematically exact bounds.
        weighted = np.clip(weighted, 0.0, 1.0)
        energy = float(
            self.grid.volume_element_bohr3
            * np.sum(self._local_energy_density_hartree_per_bohr3(weighted))
        )
        if not math.isfinite(energy):
            raise RuntimeError("Weighted-density cavity scalar is non-finite.")
        weighted.setflags(write=False)
        return Route2V0WeightedDensityCavityState(
            solvent_center_occupancy=occupancy,
            weighted_occupancy=weighted,
            cavity_energy_hartree=energy,
            operator_fingerprint=self._operator_fingerprint,
        )

    def cavity_energy_hartree(self, state: Route2V0WeightedDensityCavityState) -> float:
        """Return the scalar after checking that its state has not been crossed."""

        self._validate_state(state)
        return state.cavity_energy_hartree

    def occupancy_functional_derivative_hartree_per_bohr3(
        self,
        state: Route2V0WeightedDensityCavityState,
    ) -> np.ndarray:
        """Return ``delta G_cav / delta s`` under ``dV * sum`` pairing."""

        self._validate_state(state)
        derivative = self._adjoint_convolution(
            self._local_energy_derivative_hartree_per_bohr3(state.weighted_occupancy)
        )
        derivative.setflags(write=False)
        return derivative

    def occupancy_coordinate_gradient_hartree(
        self,
        state: Route2V0WeightedDensityCavityState,
    ) -> np.ndarray:
        """Return coordinate derivatives ``dG_cav/ds_i`` in Hartree.

        This is the form expected by
        :meth:`Route2V0IsoDensityProductCavity.occupancy_coordinate_vjp` when
        the future source contract proves that its occupancy is the same
        solvent-centre field required by this functional.
        """

        result = np.array(
            self.grid.volume_element_bohr3
            * self.occupancy_functional_derivative_hartree_per_bohr3(state),
            dtype=float,
            copy=True,
        )
        result.setflags(write=False)
        return result


__all__ = [
    "V0_WEIGHTED_DENSITY_CAVITY_CONSTRUCTION",
    "V0_WEIGHTED_DENSITY_CAVITY_SCOPE",
    "Route2V0WeightedDensityCavityFunctional",
    "Route2V0WeightedDensityCavityState",
    "Route2V0WeightedDensityCavityThermodynamics",
]
