"""Profile-bound aqueous linear CDS on the smooth harmonic cavity.

This module composes two independently testable objects:

* the non-negative, differentiable atom areas from the harmonic exposure
  fraction; and
* the published aqueous SMD atomic-tension feature basis with one explicit,
  frozen coefficient vector.

The result is one additive scalar.  Its gradient contains both the area and
environment-dependent tension derivatives.  No coefficient fitting, solvent
benchmark lookup, or continuum electrostatic calculation occurs here.
"""

from __future__ import annotations

from pathlib import Path
from ase.data import atomic_numbers
import numpy as np

from maple.function.calculator.extra_correction.implicit.smd_cds import (
    HARTREE_TO_KCAL_MOL,
    SMD_WATER_STOCK_TENSION_COEFFICIENTS_CAL_MOL_ANGSTROM2,
    aqueous_atomic_surface_tension_position_vjp_from_coefficients,
    aqueous_atomic_surface_tensions_from_coefficients,
    aqueous_cds_tension_design_row,
    smd_sasa_radii,
    validate_smd_symbols,
)
from maple.solvation.api.units import HARTREE_TO_EV
from maple.solvation.continuum.harmonic_cds_area import (
    SmoothHarmonicExposureArea,
)
from maple.solvation.coupling.operator import (
    canonical_metadata_sha256,
    source_files_sha256,
)
from maple.solvation.coupling.state_equation import geometry_sha256
from maple.solvation.solvent_terms import SolventEnergyState

MAPLE_CDS_W1_HARMONIC_LINEAR_PROFILE_ID = (
    "maple-cds-w1-smooth-harmonic-linear-water-candidate-v1"
)
SMOOTH_HARMONIC_AQUEOUS_LINEAR_CDS_PROVIDER_ID = (
    "maple.route2.solvent-term.smooth-harmonic-aqueous-linear-cds.impl.v1"
)
_KCAL_MOL_TO_EV = HARTREE_TO_EV / HARTREE_TO_KCAL_MOL
_MODULE_PATH = Path(__file__).resolve()
_MAPLE_ROOT = _MODULE_PATH.parent.parent
_SMD_CDS_PATH = (
    _MAPLE_ROOT
    / "function"
    / "calculator"
    / "extra_correction"
    / "implicit"
    / "smd_cds.py"
)
_HARMONIC_AREA_PATH = _MODULE_PATH.parent / "continuum" / "harmonic_cds_area.py"


def _text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")
    return value.strip()


def _coefficients(value: object) -> tuple[float, ...]:
    result = np.asarray(value, dtype=float)
    expected = SMD_WATER_STOCK_TENSION_COEFFICIENTS_CAL_MOL_ANGSTROM2.shape
    if result.shape != expected or not np.all(np.isfinite(result)):
        raise ValueError(f"tension coefficients must be finite with shape {expected}.")
    return tuple(float(item) for item in result)


def _positions(geometry: object, *, atom_count: int) -> np.ndarray:
    getter = getattr(geometry, "get_positions", None)
    values = getter() if callable(getter) else getattr(geometry, "positions", None)
    result = np.asarray(values, dtype=float)
    expected = (atom_count, 3)
    if result.shape != expected or not np.all(np.isfinite(result)):
        raise ValueError(f"geometry positions must be finite with shape {expected}.")
    return np.array(result, copy=True)


class SmoothHarmonicAqueousLinearCDSTerm:
    """One immutable water CDS scalar with complete first derivatives."""

    __slots__ = (
        "_area",
        "_coefficients",
        "_configuration_sha256",
        "_profile_id",
        "_sealed",
        "_symbols",
    )

    provider_id = SMOOTH_HARMONIC_AQUEOUS_LINEAR_CDS_PROVIDER_ID
    solvent = "water"

    def __init__(
        self,
        *,
        symbols: tuple[str, ...],
        area: SmoothHarmonicExposureArea,
        coefficients_cal_mol_angstrom2: object,
        profile_id: str = MAPLE_CDS_W1_HARMONIC_LINEAR_PROFILE_ID,
    ) -> None:
        normalized_symbols = validate_smd_symbols(tuple(symbols))
        if not isinstance(area, SmoothHarmonicExposureArea):
            raise TypeError("area must be SmoothHarmonicExposureArea.")
        expected_numbers = tuple(
            atomic_numbers[symbol] for symbol in normalized_symbols
        )
        if area.atomic_numbers != expected_numbers:
            raise ValueError("area atomic numbers do not match CDS symbols.")
        coefficients = _coefficients(coefficients_cal_mol_angstrom2)
        normalized_profile = _text(profile_id, name="profile_id")
        configuration = canonical_metadata_sha256(
            {
                "contract": "route2-smooth-harmonic-aqueous-linear-cds-v1",
                "provider_id": self.provider_id,
                "profile_id": normalized_profile,
                "solvent": self.solvent,
                "symbols": list(normalized_symbols),
                "area_provider_id": area.provider_id,
                "area_configuration_sha256": area.configuration_sha256(),
                "area_measure": area.area_measure,
                "tension_basis": "published-aqueous-smd-linear-18-column-v1",
                "coefficients_cal_mol_angstrom2": list(coefficients),
                "energy_unit": "eV",
                "gradient_unit": "eV/A",
                "kcal_mol_to_eV": _KCAL_MOL_TO_EV,
                "derivative": "area-vjp-plus-atomic-tension-vjp",
                "implementation_files_sha256": dict(
                    source_files_sha256(
                        {
                            "solvation/harmonic_cds.py": _MODULE_PATH,
                            "continuum/harmonic_cds_area.py": _HARMONIC_AREA_PATH,
                            "implicit/smd_cds.py": _SMD_CDS_PATH,
                        }
                    )
                ),
                "capabilities": "none",
            }
        )
        object.__setattr__(self, "_symbols", normalized_symbols)
        object.__setattr__(self, "_area", area)
        object.__setattr__(self, "_coefficients", coefficients)
        object.__setattr__(self, "_profile_id", normalized_profile)
        object.__setattr__(self, "_configuration_sha256", configuration)
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError(
                "SmoothHarmonicAqueousLinearCDSTerm is immutable after construction."
            )
        object.__setattr__(self, name, value)

    @property
    def symbols(self) -> tuple[str, ...]:
        return self._symbols

    @property
    def profile_id(self) -> str:
        return self._profile_id

    @property
    def area(self) -> SmoothHarmonicExposureArea:
        return self._area

    @property
    def coefficients_cal_mol_angstrom2(self) -> np.ndarray:
        result = np.asarray(self._coefficients, dtype=float)
        result.setflags(write=False)
        return result

    def configuration_sha256(self) -> str:
        current = canonical_metadata_sha256(
            {
                "contract": "route2-smooth-harmonic-aqueous-linear-cds-v1",
                "provider_id": self.provider_id,
                "profile_id": self._profile_id,
                "solvent": self.solvent,
                "symbols": list(self._symbols),
                "area_provider_id": self._area.provider_id,
                "area_configuration_sha256": self._area.configuration_sha256(),
                "area_measure": self._area.area_measure,
                "tension_basis": "published-aqueous-smd-linear-18-column-v1",
                "coefficients_cal_mol_angstrom2": list(self._coefficients),
                "energy_unit": "eV",
                "gradient_unit": "eV/A",
                "kcal_mol_to_eV": _KCAL_MOL_TO_EV,
                "derivative": "area-vjp-plus-atomic-tension-vjp",
                "implementation_files_sha256": dict(
                    source_files_sha256(
                        {
                            "solvation/harmonic_cds.py": _MODULE_PATH,
                            "continuum/harmonic_cds_area.py": _HARMONIC_AREA_PATH,
                            "implicit/smd_cds.py": _SMD_CDS_PATH,
                        }
                    )
                ),
                "capabilities": "none",
            }
        )
        if current != self._configuration_sha256:
            raise RuntimeError("smooth harmonic aqueous CDS configuration drifted.")
        return current

    def design_row_kcal_mol(self, geometry: object) -> np.ndarray:
        """Return the frozen linear feature row before coefficient fitting."""

        positions = _positions(geometry, atom_count=len(self._symbols))
        areas = self._area.atom_areas_angstrom2(geometry)
        result = aqueous_cds_tension_design_row(self._symbols, positions, areas)
        if result.shape != (len(self._coefficients),) or not np.all(
            np.isfinite(result)
        ):
            raise RuntimeError("aqueous CDS design row is invalid.")
        return np.array(result, copy=True)

    def validate_same_cavity_as(self, continuum: object) -> str:
        """Prove that the CDS area and one harmonic continuum share a cavity."""

        self.configuration_sha256()
        return self._area.validate_same_cavity_as(continuum)

    def evaluate(
        self,
        geometry: object,
        *,
        need_gradient: bool,
    ) -> SolventEnergyState:
        if type(need_gradient) is not bool:
            raise TypeError("need_gradient must be bool.")
        self.configuration_sha256()
        positions = _positions(geometry, atom_count=len(self._symbols))
        symbol_getter = getattr(geometry, "get_chemical_symbols", None)
        if callable(symbol_getter):
            actual_symbols = tuple(str(value) for value in symbol_getter())
            if actual_symbols != self._symbols:
                raise ValueError("geometry symbols do not match the CDS term.")
        areas = self._area.atom_areas_angstrom2(geometry)
        coefficients = self.coefficients_cal_mol_angstrom2
        tensions = aqueous_atomic_surface_tensions_from_coefficients(
            self._symbols,
            positions,
            coefficients,
        )
        energy_eV = float(np.dot(areas, tensions) / 1000.0 * _KCAL_MOL_TO_EV)
        gradient = None
        if need_gradient:
            area_gradient = self._area.position_vjp(
                geometry,
                tensions / 1000.0 * _KCAL_MOL_TO_EV,
            )
            tension_gradient = (
                aqueous_atomic_surface_tension_position_vjp_from_coefficients(
                    self._symbols,
                    positions,
                    areas / 1000.0 * _KCAL_MOL_TO_EV,
                    coefficients,
                )
            )
            gradient = area_gradient + tension_gradient
        topology = canonical_metadata_sha256(
            {
                "contract": "smooth-harmonic-fixed-coefficient-topology-v1",
                "area_configuration_sha256": self._area.configuration_sha256(),
                "atom_count": len(self._symbols),
                "symbols": list(self._symbols),
                "active_set_changes": False,
            }
        )
        return SolventEnergyState(
            provider_id=self.provider_id,
            configuration_sha256=self.configuration_sha256(),
            geometry_sha256=geometry_sha256(geometry),
            topology_id=topology,
            topology_observation_coverage="complete",
            unobservable_topology_components=(),
            atom_count=len(self._symbols),
            energy_eV=energy_eV,
            gradient_eV_per_A=gradient,
        )


def build_stock_smd_water_harmonic_cds(
    *,
    symbols: tuple[str, ...],
    area: SmoothHarmonicExposureArea,
) -> SmoothHarmonicAqueousLinearCDSTerm:
    """Return the stock-coefficient M1 control on the new area definition."""

    normalized_symbols = validate_smd_symbols(tuple(symbols))
    expected_numbers = tuple(atomic_numbers[symbol] for symbol in normalized_symbols)
    if area.atomic_numbers != expected_numbers:
        raise ValueError("area atomic numbers do not match CDS symbols.")
    expected_radii = tuple(float(value) for value in smd_sasa_radii(normalized_symbols))
    if not np.allclose(
        area.radii_angstrom,
        expected_radii,
        rtol=0.0,
        atol=2.0e-15,
    ):
        raise ValueError(
            "the stock SMD-water control requires published SMD SASA radii "
            "including the 0.4 A probe."
        )

    return SmoothHarmonicAqueousLinearCDSTerm(
        symbols=normalized_symbols,
        area=area,
        coefficients_cal_mol_angstrom2=(
            SMD_WATER_STOCK_TENSION_COEFFICIENTS_CAL_MOL_ANGSTROM2
        ),
        profile_id="stock-smd-water-tensions-on-smooth-harmonic-area-control-v1",
    )


__all__ = [
    "MAPLE_CDS_W1_HARMONIC_LINEAR_PROFILE_ID",
    "SMOOTH_HARMONIC_AQUEOUS_LINEAR_CDS_PROVIDER_ID",
    "SmoothHarmonicAqueousLinearCDSTerm",
    "build_stock_smd_water_harmonic_cds",
]
