"""AIMNet2-bound smooth-partition harmonic ddPCM candidate.

This module keeps the model and continuum contracts deliberately separate.
AIMNet2 is evaluated once at each geometry and contributes only its vacuum
energy and NQE point monopoles.  The continuum never supplies a field back to
the model and no electronic fixed-point iteration is introduced.

The continuum uses the fixed, SO(3)-covariant local-harmonic coefficient
topology implemented by :mod:`harmonic_ddpcm_functional`.  Solvent dielectric
and SMD Coulomb radii are bound into the immutable configuration.  The exact
profile is disabled pending full accuracy and daily-task evidence.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from maple.function.calculator.extra_correction.implicit.smd_cds import (
    route2_coulomb_radii,
)
from maple.function.route2_smd_profiles import DDPCM_MULTISOLVENT_SMD_PROFILE
from maple.function.route2_solvents import route2_solvent_spec
from maple.solvation.api.profiles import (
    AIMNET2_POINT_L0_GEOMETRY_MEDIATED_COUPLING_ID,
    SMOOTH_PARTITION_HARMONIC_MULTISOLVENT_CAVITY_PROFILE_ID,
    SMOOTH_PARTITION_HARMONIC_MULTISOLVENT_DDPCM_CONFIGURATION_CONTRACT_ID,
    SMOOTH_PARTITION_HARMONIC_MULTISOLVENT_DDPCM_CONTINUUM_PROFILE_ID,
)
from maple.solvation.api.scalar_registry import (
    CANDIDATE_AIMNET2_FROZEN_CHARGE_MULTISOLVENT_SMOOTH_PARTITION_HARMONIC_DDPCM_ELECTROSTATIC_V1,
)

from .harmonic_ddpcm_functional import (
    SmoothPartitionHarmonicDDPCMFunctionalCandidate,
)

AIMNET2_SMOOTH_PARTITION_HARMONIC_DDPCM_PROVIDER_ID = (
    "maple.route2.continuum.aimnet2-frozen-charge-multisolvent-"
    "smooth-partition-harmonic-ddpcm.impl.v1"
)
AIMNET2_SMOOTH_PARTITION_HARMONIC_DDPCM_CONTRACT_ID = (
    "maple.route2.continuum.aimnet2-frozen-charge-one-shot-"
    "smooth-partition-harmonic-ddpcm-scalar.v1"
)
AIMNET2_SMOOTH_PARTITION_TRANSITION_WIDTH_ANGSTROM2 = 0.18
AIMNET2_SMOOTH_PARTITION_SURFACE_LMAX = 4
AIMNET2_SMOOTH_PARTITION_PARTITION_LMAX = 8
AIMNET2_SMOOTH_PARTITION_PARTITION_RADIAL_ORDER = 96
AIMNET2_SMOOTH_PARTITION_SOURCE_RADIAL_ORDER = 128
AIMNET2_SMOOTH_PARTITION_DOUBLE_LAYER_RADIAL_ORDER = 128


def _sha(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


class AIMNet2FrozenChargeSmoothPartitionDDPCMFunctionalCandidate(
    SmoothPartitionHarmonicDDPCMFunctionalCandidate
):
    """Exact disabled multi-solvent continuum for one-shot AIMNet2 charges."""

    __slots__ = ("_solvent", "_symbols")

    provider_id = AIMNET2_SMOOTH_PARTITION_HARMONIC_DDPCM_PROVIDER_ID
    functional_contract_id = AIMNET2_SMOOTH_PARTITION_HARMONIC_DDPCM_CONTRACT_ID
    continuum_profile_id = (
        SMOOTH_PARTITION_HARMONIC_MULTISOLVENT_DDPCM_CONTINUUM_PROFILE_ID
    )
    cavity_profile_id = SMOOTH_PARTITION_HARMONIC_MULTISOLVENT_CAVITY_PROFILE_ID
    configuration_contract_id = (
        SMOOTH_PARTITION_HARMONIC_MULTISOLVENT_DDPCM_CONFIGURATION_CONTRACT_ID
    )
    coupling_id = AIMNET2_POINT_L0_GEOMETRY_MEDIATED_COUPLING_ID
    scalar_id = CANDIDATE_AIMNET2_FROZEN_CHARGE_MULTISOLVENT_SMOOTH_PARTITION_HARMONIC_DDPCM_ELECTROSTATIC_V1
    reciprocal = True
    registered_scalar = True
    fixed_cavity_descriptor = False
    full_geometry_intertwiner_assembly_available = True
    moving_cavity_coordinate_derivative_available = True
    fixed_geometry_electronic_mutual_polarization = False
    continuum_field_supplied_to_aimnet2 = False
    electronic_scf_iteration = False
    aimnet2_source_evaluation = "one-shot-per-geometry"
    point_monopoles_only = True

    def __init__(
        self,
        *,
        symbols: tuple[str, ...],
        solvent: str,
        dtype: object,
        device: object,
    ) -> None:
        from ase.data import atomic_numbers

        normalized_symbols = tuple(str(symbol).strip() for symbol in symbols)
        if not normalized_symbols or any(not symbol for symbol in normalized_symbols):
            raise ValueError("symbols must contain non-empty element symbols.")
        try:
            numbers = tuple(
                int(atomic_numbers[symbol]) for symbol in normalized_symbols
            )
        except KeyError as exc:
            raise ValueError(f"Unsupported element symbol: {exc.args[0]!r}.") from exc
        spec = route2_solvent_spec(solvent)
        radii = route2_coulomb_radii(
            normalized_symbols,
            solvent=spec.name,
            profile=DDPCM_MULTISOLVENT_SMD_PROFILE,
        )
        object.__setattr__(self, "_symbols", normalized_symbols)
        object.__setattr__(self, "_solvent", spec.name)
        super().__init__(
            atomic_numbers=numbers,
            radii_angstrom=tuple(float(value) for value in radii),
            transition_width_angstrom2=(
                AIMNET2_SMOOTH_PARTITION_TRANSITION_WIDTH_ANGSTROM2
            ),
            surface_lmax=AIMNET2_SMOOTH_PARTITION_SURFACE_LMAX,
            partition_lmax=AIMNET2_SMOOTH_PARTITION_PARTITION_LMAX,
            partition_radial_quadrature_order=(
                AIMNET2_SMOOTH_PARTITION_PARTITION_RADIAL_ORDER
            ),
            source_radial_quadrature_order=(
                AIMNET2_SMOOTH_PARTITION_SOURCE_RADIAL_ORDER
            ),
            double_layer_radial_quadrature_order=(
                AIMNET2_SMOOTH_PARTITION_DOUBLE_LAYER_RADIAL_ORDER
            ),
            dielectric=spec.descriptors.dielectric,
            dtype=dtype,
            device=device,
        )

    @property
    def symbols(self) -> tuple[str, ...]:
        return self._symbols

    @property
    def solvent(self) -> str:
        return self._solvent

    def _implementation_fingerprint(self) -> tuple[tuple[str, str], ...]:
        own = Path(__file__).resolve()
        return super()._implementation_fingerprint() + (
            (own.name, hashlib.sha256(own.read_bytes()).hexdigest()),
        )

    def _configuration_payload(self) -> dict[str, object]:
        payload = super()._configuration_payload()
        spec = route2_solvent_spec(self._solvent)
        payload.update(
            {
                "cavity_profile_id": self.cavity_profile_id,
                "configuration_contract_id": self.configuration_contract_id,
                "coupling_id": self.coupling_id,
                "symbols": self._symbols,
                "solvent": spec.name,
                "pyscf_smd_solvent": spec.pyscf_smd_name,
                "solvent_descriptors": spec.descriptors.as_pyscf_tuple(),
                "cavity_radii_policy": DDPCM_MULTISOLVENT_SMD_PROFILE,
                "aimnet2_source_evaluation": self.aimnet2_source_evaluation,
                "continuum_field_supplied_to_aimnet2": False,
                "electronic_scf_iteration": False,
                "model_source": "AIMNet2 NQE point monopoles [q,0,0,0]",
            }
        )
        return payload

    # The sealed derivative implementation remains entirely inherited.  A
    # concrete subclass must explicitly name its one scalar implementation.
    def _energy_torch(self, positions: Any, source: Any):
        if bool(__import__("torch").any(source[:, 1:] != 0.0).detach().cpu()):
            raise ValueError(
                "AIMNet2 frozen-charge ddPCM accepts point monopoles only; "
                "l=1 source components must be exactly zero."
            )
        return super()._energy_torch(positions, source)

    def evaluate_field(self, geometry: object, source: object):
        return self.drive(geometry, source)

    field = evaluate_field

    def coordinate_vjp(self, geometry: object, source: object, field_cotangent: object):
        return self.mixed_coordinate_source_vjp(
            geometry, source, field_cotangent
        ).reshape(-1)

    def topology_state(self, geometry: object) -> dict[str, object]:
        getter = getattr(geometry, "get_positions", None)
        positions = getter() if callable(getter) else geometry
        count = len(positions)
        if count != len(self._symbols):
            raise ValueError("geometry atom count does not match the candidate.")
        topology = _sha(
            {
                "schema": "fixed-local-harmonic-coefficient-topology-v1",
                "atomic_numbers": self.atomic_numbers,
                "surface_lmax": self.surface_lmax,
                "coefficient_count": count * (self.surface_lmax + 1) ** 2,
                "active_coefficient_deletion": False,
                "laboratory_fixed_surface_grid": False,
            }
        )
        return {
            "configuration_sha256": self.configuration_sha256(),
            "cavity_topology_sha256": topology,
            "coefficient_topology_sha256": topology,
            "coefficient_count": count * (self.surface_lmax + 1) ** 2,
            "active_coefficient_deletion": False,
            "laboratory_fixed_surface_grid": False,
        }

    def runtime_provenance(self) -> tuple[tuple[str, object], ...]:
        payload = dict(super().runtime_provenance())
        spec = route2_solvent_spec(self._solvent)
        payload.update(
            {
                "cavity_profile_id": self.cavity_profile_id,
                "configuration_contract_id": self.configuration_contract_id,
                "coupling_id": self.coupling_id,
                "solvent": spec.name,
                "pyscf_smd_solvent": spec.pyscf_smd_name,
                "cavity_radii_policy": DDPCM_MULTISOLVENT_SMD_PROFILE,
                "aimnet2_source_evaluation": self.aimnet2_source_evaluation,
                "continuum_field_supplied_to_aimnet2": False,
                "electronic_scf_iteration": False,
                "active_coefficient_deletion": False,
            }
        )
        return tuple(sorted(payload.items()))


def build_aimnet2_frozen_charge_smooth_partition_ddpcm_candidate(
    symbols: tuple[str, ...],
    *,
    solvent: str,
    dtype: object,
    device: object,
) -> AIMNet2FrozenChargeSmoothPartitionDDPCMFunctionalCandidate:
    """Build the exact disabled multi-solvent AIMNet2 continuum candidate."""

    return AIMNet2FrozenChargeSmoothPartitionDDPCMFunctionalCandidate(
        symbols=tuple(symbols),
        solvent=solvent,
        dtype=dtype,
        device=device,
    )


__all__ = [
    "AIMNET2_SMOOTH_PARTITION_DOUBLE_LAYER_RADIAL_ORDER",
    "AIMNET2_SMOOTH_PARTITION_HARMONIC_DDPCM_CONTRACT_ID",
    "AIMNET2_SMOOTH_PARTITION_HARMONIC_DDPCM_PROVIDER_ID",
    "AIMNET2_SMOOTH_PARTITION_PARTITION_LMAX",
    "AIMNET2_SMOOTH_PARTITION_PARTITION_RADIAL_ORDER",
    "AIMNET2_SMOOTH_PARTITION_SOURCE_RADIAL_ORDER",
    "AIMNET2_SMOOTH_PARTITION_SURFACE_LMAX",
    "AIMNET2_SMOOTH_PARTITION_TRANSITION_WIDTH_ANGSTROM2",
    "AIMNet2FrozenChargeSmoothPartitionDDPCMFunctionalCandidate",
    "build_aimnet2_frozen_charge_smooth_partition_ddpcm_candidate",
]
