"""Model-independent additive solvent-energy terms.

Electrostatic continuum response and non-electrostatic solvent contributions
are separate axes of a Route-2 scalar.  This module provides the small shared
contract needed by both pure and hybrid MLIP profiles without importing a
particular electronic model or fixed-point solver.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from maple.function.calculator.extra_correction.implicit.pyscf_runtime import (
    TESTED_PYSCF_VERSION,
)
from maple.function.calculator.extra_correction.implicit.pyscf_smd_cds import (
    pyscf_smd_cds,
)
from maple.function.calculator.extra_correction.implicit.smd_cds import (
    validate_smd_symbols,
)
from maple.function.route2_solvents import route2_solvent_spec
from maple.solvation.api.units import HARTREE_TO_EV
from maple.solvation.coupling.operator import (
    canonical_metadata_sha256,
    source_files_sha256,
)
from maple.solvation.coupling.state_equation import geometry_sha256
from maple.solvation.derivatives import normalize_topology_observation
from maple.solvation.models.base import atom_count, model_charge_and_multiplicity

PYSCF_SMD_CDS_PROVIDER_ID = "maple.route2.solvent-term.pyscf-smd-cds.impl.v1"
_MODULE_PATH = Path(__file__).resolve()
_MAPLE_ROOT = _MODULE_PATH.parent.parent
_PYSCF_SMD_CDS_PATH = (
    _MAPLE_ROOT
    / "function"
    / "calculator"
    / "extra_correction"
    / "implicit"
    / "pyscf_smd_cds.py"
)
_ROUTE2_SOLVENTS_PATH = _MAPLE_ROOT / "function" / "route2_solvents.py"


def _digest(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA256 digest.")
    return value


def _text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")
    return value.strip()


def _finite(value: object, *, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a finite real number.")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _readonly(values: object, *, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite with shape {shape}.")
    return np.frombuffer(
        np.ascontiguousarray(array, dtype=np.float64).tobytes(), dtype=np.float64
    ).reshape(shape)


def _symbols(geometry: object) -> tuple[str, ...]:
    getter = getattr(geometry, "get_chemical_symbols", None)
    if not callable(getter):
        raise TypeError("geometry must expose callable get_chemical_symbols().")
    values = tuple(str(value) for value in getter())
    if not values or len(values) != atom_count(geometry):
        raise ValueError("geometry chemical symbols are invalid.")
    return values


def _positions(geometry: object) -> np.ndarray:
    getter = getattr(geometry, "get_positions", None)
    values = getter() if callable(getter) else getattr(geometry, "positions", None)
    result = np.asarray(values, dtype=float)
    expected = (atom_count(geometry), 3)
    if result.shape != expected or not np.all(np.isfinite(result)):
        raise ValueError(f"geometry positions must be finite with shape {expected}.")
    return np.array(result, copy=True)


@dataclass(frozen=True, slots=True)
class SolventEnergyState:
    """One content-addressed additive solvent scalar and optional gradient."""

    provider_id: str
    configuration_sha256: str
    geometry_sha256: str
    topology_id: str
    atom_count: int
    energy_eV: float
    gradient_eV_per_A: np.ndarray | None
    topology_observation_coverage: str = "unobservable"
    unobservable_topology_components: tuple[str, ...] = (
        "legacy-unspecified-topology-observation",
    )
    state_sha256: str = ""

    def __post_init__(self) -> None:
        provider_id = _text(self.provider_id, name="provider_id")
        configuration = _digest(self.configuration_sha256, name="configuration_sha256")
        geometry = _digest(self.geometry_sha256, name="geometry_sha256")
        topology = _text(self.topology_id, name="topology_id")
        coverage, components = normalize_topology_observation(
            self.topology_observation_coverage,
            self.unobservable_topology_components,
        )
        if type(self.atom_count) is not int or self.atom_count < 1:
            raise ValueError("atom_count must be a positive integer.")
        energy = _finite(self.energy_eV, name="energy_eV")
        gradient = None
        if self.gradient_eV_per_A is not None:
            gradient = _readonly(
                self.gradient_eV_per_A,
                shape=(self.atom_count, 3),
                name="gradient_eV_per_A",
            )
        payload = {
            "contract": "route2-additive-solvent-energy-state-v2",
            "provider_id": provider_id,
            "configuration_sha256": configuration,
            "geometry_sha256": geometry,
            "topology_id": topology,
            "topology_observation_coverage": coverage,
            "unobservable_topology_components": list(components),
            "atom_count": self.atom_count,
            "energy_eV": energy,
            "gradient_eV_per_A": None if gradient is None else gradient.tolist(),
        }
        expected = canonical_metadata_sha256(payload)
        if self.state_sha256 and self.state_sha256 != expected:
            raise ValueError("state_sha256 does not match solvent energy contents.")
        object.__setattr__(self, "provider_id", provider_id)
        object.__setattr__(self, "configuration_sha256", configuration)
        object.__setattr__(self, "geometry_sha256", geometry)
        object.__setattr__(self, "topology_id", topology)
        object.__setattr__(self, "topology_observation_coverage", coverage)
        object.__setattr__(self, "unobservable_topology_components", components)
        object.__setattr__(self, "energy_eV", energy)
        object.__setattr__(self, "gradient_eV_per_A", gradient)
        object.__setattr__(self, "state_sha256", expected)


@runtime_checkable
class SolventEnergyTerm(Protocol):
    """Additive geometry scalar, independent of the electronic provider."""

    provider_id: str

    def configuration_sha256(self) -> str: ...

    def evaluate(
        self, geometry: object, *, need_gradient: bool
    ) -> SolventEnergyState: ...


@dataclass(frozen=True, slots=True)
class PySCFSMDCDSTerm:
    """Official PySCF SMD CDS energy and analytic gradient for one solvent.

    The legacy libsolvent entrypoint does not expose its internal surface
    active set.  ``topology_id`` therefore records that topology as
    unobservable instead of incorrectly hashing the continuously changing
    geometry.  Richardson error estimates remain the numerical smoothness
    guard for this additive term.
    """

    symbols: tuple[str, ...]
    solvent: str
    provider_id: str = PYSCF_SMD_CDS_PROVIDER_ID

    def __post_init__(self) -> None:
        symbols = validate_smd_symbols(tuple(self.symbols))
        solvent = route2_solvent_spec(self.solvent).name
        if self.provider_id != PYSCF_SMD_CDS_PROVIDER_ID:
            raise ValueError("PySCF SMD CDS provider identity is fixed.")
        object.__setattr__(self, "symbols", symbols)
        object.__setattr__(self, "solvent", solvent)

    def configuration_sha256(self) -> str:
        specification = route2_solvent_spec(self.solvent)
        return canonical_metadata_sha256(
            {
                "contract": "route2-pyscf-smd-cds-solvent-term-v1",
                "provider_id": self.provider_id,
                "symbols": list(self.symbols),
                "solvent": specification.name,
                "pyscf_smd_name": specification.pyscf_smd_name,
                "smd_descriptors": list(specification.descriptors.as_pyscf_tuple()),
                "pyscf_version": TESTED_PYSCF_VERSION,
                "upstream_entrypoint": "pyscf.solvent.smd.get_cds_legacy",
                "internal_surface_topology_observable": False,
                "topology_id_semantics": (
                    "configuration-stable-unobservable-internal-surface"
                ),
                "unit_conversion_hartree_to_eV": HARTREE_TO_EV,
                "implementation_files_sha256": dict(
                    source_files_sha256(
                        {
                            "solvation/solvent_terms.py": _MODULE_PATH,
                            "implicit/pyscf_smd_cds.py": _PYSCF_SMD_CDS_PATH,
                            "function/route2_solvents.py": _ROUTE2_SOLVENTS_PATH,
                        }
                    )
                ),
            }
        )

    def evaluate(self, geometry: object, *, need_gradient: bool) -> SolventEnergyState:
        if type(need_gradient) is not bool:
            raise TypeError("need_gradient must be a bool.")
        if _symbols(geometry) != self.symbols:
            raise ValueError("geometry symbols differ from the CDS configuration.")
        if model_charge_and_multiplicity(geometry) != (0, 1):
            raise ValueError("PySCF SMD CDS term currently supports neutral singlets.")
        result = pyscf_smd_cds(
            self.symbols,
            _positions(geometry),
            solvent=self.solvent,
        )
        gradient = None
        if need_gradient:
            gradient = HARTREE_TO_EV * result.position_gradient_hartree_per_angstrom
        geometry_digest = geometry_sha256(geometry)
        configuration = self.configuration_sha256()
        topology = canonical_metadata_sha256(
            {
                "contract": "pyscf-smd-cds-internal-topology-unobservable-v1",
                "configuration_sha256": configuration,
                "internal_surface_topology_observable": False,
            }
        )
        return SolventEnergyState(
            provider_id=self.provider_id,
            configuration_sha256=configuration,
            geometry_sha256=geometry_digest,
            topology_id=topology,
            atom_count=len(self.symbols),
            energy_eV=result.energy_hartree * HARTREE_TO_EV,
            gradient_eV_per_A=gradient,
            topology_observation_coverage="unobservable",
            unobservable_topology_components=("pyscf-smd-libsolvent-internal-surface",),
        )


__all__ = [
    "PYSCF_SMD_CDS_PROVIDER_ID",
    "PySCFSMDCDSTerm",
    "SolventEnergyState",
    "SolventEnergyTerm",
]
