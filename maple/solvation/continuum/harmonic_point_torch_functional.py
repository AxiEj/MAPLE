"""SO(3)-structured smooth harmonic conductor for AIMNet2 point charges.

This disabled provider reuses the common weighted-harmonic ``E(R)`` and
Coulomb ``K(R)`` assembly, but supplies the physically distinct analytic
point-monopole boundary map ``V_point(R)``.  It is not a zero-width Gaussian
channel and it does not alter AIMNet2's inputs, graph, state, or checkpoint.

The reduced scalar and every derivative come from the inherited sealed Torch
graph.  The extra protocol aliases expose that same scalar as a linear
reaction-field provider for the geometry-mediated composition; they do not
provide independent formulas or admit a public capability.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np

from maple.solvation.api.profiles import (
    AIMNET2_POINT_L0_GEOMETRY_MEDIATED_COUPLING_ID,
)
from maple.solvation.api.scalar_registry import (
    DIAGNOSTIC_AIMNET2_GEOMETRY_MEDIATED_SMOOTH_HARMONIC_CPCM_ELECTROSTATIC_V1,
)
from maple.solvation.coupling.metrics import ATOMIC_L1_PAIRING
from maple.solvation.coupling.spaces import (
    ATOMIC_L1_FIELD_DUAL_SPACE,
    ATOMIC_L1_SOURCE_SPACE,
)

from .harmonic_point_source import point_source_topology
from .harmonic_torch_functional import (
    SmoothWeightedHarmonicGalerkinFunctionalCandidate,
)
from .harmonic_torch_primitives import _assemble_point_l0_source

SMOOTH_POINT_HARMONIC_GALERKIN_PROVIDER_ID = (
    "maple.route2.continuum.smooth-harmonic-point-l0-functional.impl.v1"
)
SMOOTH_POINT_HARMONIC_GALERKIN_FUNCTIONAL_CONTRACT_ID = (
    "maple.route2.continuum.smooth-harmonic-point-l0-same-scalar.v1"
)


def _sha(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


class SmoothPointChargeHarmonicGalerkinFunctionalCandidate(
    SmoothWeightedHarmonicGalerkinFunctionalCandidate
):
    """Fixed-dimensional harmonic conductor driven by atomic point monopoles."""

    __slots__ = ()

    provider_id = SMOOTH_POINT_HARMONIC_GALERKIN_PROVIDER_ID
    functional_contract_id = SMOOTH_POINT_HARMONIC_GALERKIN_FUNCTIONAL_CONTRACT_ID
    coupling_id = AIMNET2_POINT_L0_GEOMETRY_MEDIATED_COUPLING_ID
    _source_space_contract = ATOMIC_L1_SOURCE_SPACE
    _field_space_contract = ATOMIC_L1_FIELD_DUAL_SPACE
    _pairing_contract = ATOMIC_L1_PAIRING
    _accepted_scalar_ids = frozenset(
        {DIAGNOSTIC_AIMNET2_GEOMETRY_MEDIATED_SMOOTH_HARMONIC_CPCM_ELECTROSTATIC_V1}
    )
    _source_implementation_files = (
        "harmonic_point_source.py",
        "harmonic_point_torch_functional.py",
    )
    _source_radial_quadrature_required = False
    _geometry_quadrature = "finite-band-exact-and-analytic-point-addition-theorem"
    fixed_topology = False
    point_source_shell_events_fail_closed = True
    fixed_geometry_electronic_mutual_polarization = False

    def __init__(
        self,
        *,
        atomic_numbers: tuple[int, ...],
        radii_angstrom: tuple[float, ...],
        transition_width_angstrom2: float,
        surface_lmax: int,
        exposure_lmax: int,
        exposure_radial_quadrature_order: int = 96,
        green_radial_quadrature_order: int = 128,
        dtype: object,
        device: object,
        scalar_id: str = (
            DIAGNOSTIC_AIMNET2_GEOMETRY_MEDIATED_SMOOTH_HARMONIC_CPCM_ELECTROSTATIC_V1
        ),
    ) -> None:
        super().__init__(
            atomic_numbers=atomic_numbers,
            radii_angstrom=radii_angstrom,
            transition_width_angstrom2=transition_width_angstrom2,
            surface_lmax=surface_lmax,
            exposure_lmax=exposure_lmax,
            exposure_radial_quadrature_order=exposure_radial_quadrature_order,
            source_radial_quadrature_order=None,
            green_radial_quadrature_order=green_radial_quadrature_order,
            dtype=dtype,
            device=device,
            scalar_id=scalar_id,
        )

    def _assemble_raw_source_torch(self, positions: Any):
        return _assemble_point_l0_source(
            positions,
            radii=self._radii_angstrom,
            lmax=self.physical_lmax,
        )

    # ContinuumEnergyFunctional requires each concrete subclass to name its
    # scalar implementation explicitly.  The implementation remains the one
    # inherited stationary scalar; no derivative method is overridden.
    def _energy_torch(self, positions: Any, source: Any):
        return super()._energy_torch(positions, source)

    def evaluate_field(self, geometry: object, source: object) -> np.ndarray:
        return self.drive(geometry, source)

    field = evaluate_field

    def coordinate_vjp(
        self, geometry: object, source: object, field_cotangent: object
    ) -> np.ndarray:
        return self.mixed_coordinate_source_vjp(
            geometry, source, field_cotangent
        ).reshape(-1)

    def topology_state(self, geometry: object) -> dict[str, object]:
        getter = getattr(geometry, "get_positions", None)
        positions = getter() if callable(getter) else geometry
        point = point_source_topology(positions, self._radii_angstrom)
        topology = _sha(
            {
                "coefficient_topology_sha256": self.topology_sha256(),
                "point_source_topology_sha256": point.topology_sha256,
            }
        )
        return {
            "configuration_sha256": self.configuration_sha256(),
            "cavity_topology_sha256": topology,
            "coefficient_count": len(self._radii_angstrom)
            * (self._surface_lmax + 1) ** 2,
            "coefficient_topology_sha256": self.topology_sha256(),
            "point_source_topology_sha256": point.topology_sha256,
            "point_source_relations": [list(item) for item in point.relations],
            "minimum_point_source_shell_margin_angstrom": (
                point.minimum_shell_margin_angstrom
            ),
        }


__all__ = [
    "SMOOTH_POINT_HARMONIC_GALERKIN_FUNCTIONAL_CONTRACT_ID",
    "SMOOTH_POINT_HARMONIC_GALERKIN_PROVIDER_ID",
    "SmoothPointChargeHarmonicGalerkinFunctionalCandidate",
]
