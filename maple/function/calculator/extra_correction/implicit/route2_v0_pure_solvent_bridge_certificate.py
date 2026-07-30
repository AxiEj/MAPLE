"""Hash-bound pure-liquid evidence for the Route-2 V0 weighted-density bridge.

This v1 schema remains useful for source-bound scalar controls, but its old
flat ``B``-to-planar-surface-tension bracket cannot establish physical-liquid
admission.  A finite-density gas branch moves with the quartic coefficient,
so a physical construction must first retain the same-scalar coexistence
continuation for ``B`` and then solve any remaining planar surface-tension
condition on that coexistence-preserving branch.  That nested evidence needs a
future schema together with the constrained planar solver; v1 fails closed for
physical-liquid admission rather than giving its legacy fields a false
thermodynamic meaning.

A v1 certificate is therefore metadata binding frozen solvent sources,
control-planar evidence, and exact discrete ``D``/``K`` operators.  It is not
an energy term, a physical-liquid result, or an accuracy claim.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import SupportsFloat, SupportsIndex, TypedDict, cast

import numpy as np
from ase.units import Bohr, Hartree

from .route2_v0_solvent_asset import (
    V0_REQUIRED_ASSET_FILE_ROLES,
    V0_REQUIRED_EXCLUDED_TARGET_LABEL_SETS,
)

V0_PURE_SOLVENT_BRIDGE_CERTIFICATE_CONSTRUCTION = (
    "route2-v0-pure-solvent-bridge-certificate-v1"
)
V0_PURE_SOLVENT_BRIDGE_CERTIFICATE_SCHEMA_VERSION = 1
V0_PURE_SOLVENT_BRIDGE_CERTIFICATE_V1_PHYSICAL_ADMISSION_REJECTED = (
    "route2-v0-pure-solvent-bridge-certificate-v1-no-physical-admission"
)
V0_PURE_SOLVENT_BRIDGE_CERTIFICATE_EVIDENCE_SCOPE_SYNTHETIC_CONTROL = (
    "synthetic-control"
)
V0_PURE_SOLVENT_BRIDGE_CERTIFICATE_EVIDENCE_SCOPE_PHYSICAL_LIQUID = (
    "physical-pure-liquid-admission"
)
V0_PURE_SOLVENT_BRIDGE_CERTIFICATE_EVIDENCE_SCOPES = frozenset(
    {
        V0_PURE_SOLVENT_BRIDGE_CERTIFICATE_EVIDENCE_SCOPE_SYNTHETIC_CONTROL,
        V0_PURE_SOLVENT_BRIDGE_CERTIFICATE_EVIDENCE_SCOPE_PHYSICAL_LIQUID,
    }
)
V0_CANONICAL_NUMERIC_ARRAY_DIGEST_CONSTRUCTION = (
    "route2-v0-canonical-float64-array-digest-v1"
)
V0_WEIGHTED_DENSITY_OPERATOR_DIGEST_CONSTRUCTION = (
    "route2-v0-weighted-density-operator-digest-v1"
)
V0_MOLECULAR_HOMOGENEOUS_PHASE_CONSTRUCTION = (
    "route2-v0-molecular-homogeneous-phase-gate-v1"
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SOLVENT_ID = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_ELECTRONIC_CHARGE_JOULE = 1.602176634e-19
_HARTREE_PER_BOHR2_TO_N_PER_M = (
    Hartree * _ELECTRONIC_CHARGE_JOULE / (Bohr * 1.0e-10) ** 2
)
_HARTREE_PER_BOHR3_TO_PA = Hartree * _ELECTRONIC_CHARGE_JOULE / (Bohr * 1.0e-10) ** 3


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be an object.")
    return cast(Mapping[str, object], value)


def _text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string.")
    return value.strip()


def _digest(value: object, *, name: str) -> str:
    result = _text(value, name=name).lower()
    if not _SHA256.fullmatch(result):
        raise ValueError(f"{name} must be a lower-case SHA-256 digest.")
    return result


def _number(
    value: object,
    *,
    name: str,
    positive: bool = False,
    nonnegative: bool = False,
) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be finite.")
    try:
        result = float(
            cast(str | bytes | bytearray | SupportsFloat | SupportsIndex, value)
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite.") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    if positive and result <= 0.0:
        raise ValueError(f"{name} must be positive.")
    if nonnegative and result < 0.0:
        raise ValueError(f"{name} must be nonnegative.")
    return result


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a positive integer.")
    numeric = _number(value, name=name)
    result = int(numeric)
    if result != numeric or result < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return result


def _close(left: float, right: float) -> bool:
    return math.isclose(
        left,
        right,
        rel_tol=1.0e-12,
        abs_tol=1.0e-14 * max(1.0, abs(left), abs(right)),
    )


def _file_sha256(path: Path) -> str:
    if not path.is_file():
        raise ValueError(f"Pure-solvent bridge certificate is absent: {path}.")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_hashes(value: object) -> tuple[tuple[str, str], ...]:
    entry = _mapping(value, name="liquid_source.source_file_sha256")
    expected = set(V0_REQUIRED_ASSET_FILE_ROLES)
    if set(entry) != expected:
        missing = sorted(expected.difference(entry))
        extra = sorted(set(entry).difference(expected))
        raise ValueError(
            "Certificate source hashes must contain exactly the frozen solvent "
            f"roles; missing={missing}, extra={extra}."
        )
    return tuple(
        (role, _digest(entry[role], name=f"liquid_source.source_file_sha256.{role}"))
        for role in sorted(expected)
    )


def _excluded_label_sets(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise TypeError("no_target_policy.excluded_target_label_sets must be a list.")
    labels = tuple(
        _text(item, name="Excluded target-label set").lower() for item in value
    )
    if len(set(labels)) != len(labels) or set(labels) != set(
        V0_REQUIRED_EXCLUDED_TARGET_LABEL_SETS
    ):
        raise ValueError(
            "Certificate must contain exactly the V0 excluded target-label sets."
        )
    return labels


def _admission(value: object) -> tuple[str, str]:
    """Read the evidence tier without letting a synthetic control pass as physical."""

    entry = _mapping(value, name="admission")
    expected = {"evidence_scope", "physical_liquid_admitted", "claim_boundary"}
    if set(entry) != expected:
        missing = sorted(expected.difference(entry))
        extra = sorted(set(entry).difference(expected))
        raise ValueError(
            f"Pure-solvent bridge admission keys differ; missing={missing}, extra={extra}."
        )
    scope = _text(entry.get("evidence_scope"), name="admission.evidence_scope")
    if scope not in V0_PURE_SOLVENT_BRIDGE_CERTIFICATE_EVIDENCE_SCOPES:
        raise ValueError("Unsupported pure-solvent bridge evidence scope.")
    physical = entry.get("physical_liquid_admitted")
    if not isinstance(physical, bool):
        raise TypeError("admission.physical_liquid_admitted must be a boolean.")
    expected_physical = (
        scope == V0_PURE_SOLVENT_BRIDGE_CERTIFICATE_EVIDENCE_SCOPE_PHYSICAL_LIQUID
    )
    if physical is not expected_physical:
        raise ValueError(
            "Pure-solvent bridge evidence scope and physical-liquid admission flag disagree."
        )
    return scope, _text(entry.get("claim_boundary"), name="admission.claim_boundary")


def canonical_float64_array_sha256(values: np.ndarray, *, name: str) -> str:
    """Hash finite real values as little-endian, C-order float64 bytes."""

    if np.iscomplexobj(values):
        raise ValueError(f"{name} must be real-valued for a canonical digest.")
    array = np.asarray(values, dtype=np.dtype("<f8"))
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite for a canonical digest.")
    array = np.ascontiguousarray(array)
    header = json.dumps(
        {
            "construction": V0_CANONICAL_NUMERIC_ARRAY_DIGEST_CONSTRUCTION,
            "dtype": "float64-little-endian",
            "shape": list(array.shape),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(header + b"\n" + array.tobytes(order="C")).hexdigest()


def weighted_density_operator_sha256(
    *,
    operator: str,
    construction: str,
    grid: object,
    values: np.ndarray,
) -> str:
    """Hash one bridge operator together with its Cartesian-grid convention."""

    if operator not in {"molecular-centre-projection", "weighted-density-kernel"}:
        raise ValueError("Unsupported Route-2 weighted-density operator kind.")
    shape = getattr(grid, "shape", None)
    layout = getattr(grid, "layout", None)
    origin = np.asarray(getattr(grid, "origin_bohr", None), dtype=float)
    spacing = np.asarray(getattr(grid, "spacing_bohr", None), dtype=float)
    if (
        not isinstance(shape, tuple)
        or len(shape) != 3
        or not isinstance(layout, str)
        or origin.shape != (3,)
        or spacing.shape != (3,)
        or not np.all(np.isfinite(origin))
        or not np.all(np.isfinite(spacing))
        or np.any(spacing <= 0.0)
    ):
        raise TypeError("Weighted-density digest requires a regular Cartesian grid.")
    shape = tuple(int(item) for item in shape)
    if any(item < 1 for item in shape):
        raise ValueError("Weighted-density grid dimensions must be positive.")
    array = np.asarray(values)
    expected_shape = shape if operator == "weighted-density-kernel" else (*shape,)
    if array.shape[: len(expected_shape)] != expected_shape:
        raise ValueError("Weighted-density operator values do not match the grid.")
    descriptor = {
        "array_sha256": canonical_float64_array_sha256(array, name=operator),
        "construction": V0_WEIGHTED_DENSITY_OPERATOR_DIGEST_CONSTRUCTION,
        "grid": {
            "layout": layout,
            "origin_bohr_hex": [float(item).hex() for item in origin],
            "shape": list(shape),
            "spacing_bohr_hex": [float(item).hex() for item in spacing],
        },
        "operator": operator,
        "operator_construction": _text(
            construction,
            name="Weighted-density operator construction",
        ),
    }
    return hashlib.sha256(
        json.dumps(descriptor, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class Route2V0PureSolventHomogeneousPhaseEvidence:
    """Content-addressed coexistence evidence for one physical liquid scalar.

    The vacuum-limit pressure identity alone cannot establish a stationary
    low-density phase.  This record preserves the full-gradient and curvature
    evidence needed before a planar profile is allowed to represent a
    liquid--gas interface.  It is evidence metadata, not an energy term.
    """

    zero_external_potential_residual_hartree: float
    zero_external_potential_tolerance_hartree: float
    stationarity_tolerance: float
    gradient_uniformity_tolerance: float
    curvature_tolerance_hartree_per_bohr3: float
    coexistence_tolerance_hartree_per_bohr3: float
    liquid_density_scale: float
    gas_density_scale: float
    liquid_grand_potential_density_hartree_per_bohr3: float
    gas_grand_potential_density_hartree_per_bohr3: float
    liquid_stationarity_residual: float
    gas_stationarity_residual: float
    liquid_gradient_uniformity_residual: float
    gas_gradient_uniformity_residual: float
    liquid_curvature_hartree_per_bohr3: float
    gas_curvature_hartree_per_bohr3: float
    grand_potential_density_difference_hartree_per_bohr3: float
    construction: str = V0_MOLECULAR_HOMOGENEOUS_PHASE_CONSTRUCTION


@dataclass(frozen=True)
class Route2V0PureSolventBridgeCertificate:
    """A parsed pure-liquid certificate for one HNC-plus-bridge scalar."""

    certificate_path: Path
    content_sha256: str
    evidence_scope: str
    admission_claim_boundary: str
    solvent_id: str
    model_identifier: str
    closure: str
    temperature_kelvin: float
    pressure_bar: float
    molecular_bulk_number_density_bohr3: float
    source_file_sha256: tuple[tuple[str, str], ...]
    center_projection_sha256: str
    weighted_density_kernel_sha256: str
    hnc_bulk_pressure_hartree_per_bohr3: float
    target_bulk_pressure_hartree_per_bohr3: float
    target_surface_tension_n_per_m: float
    target_surface_tension_hartree_per_bohr2: float
    surface_tension_reference: str
    cubic_coefficient_hartree_bohr6: float
    quartic_coefficient_low_hartree_bohr15: float
    quartic_coefficient_high_hartree_bohr15: float
    quartic_coefficient_hartree_bohr15: float
    surface_tension_low_hartree_per_bohr2: float
    surface_tension_high_hartree_per_bohr2: float
    surface_tension_root_hartree_per_bohr2: float
    surface_tension_root_tolerance_hartree_per_bohr2: float
    stationarity_residual_low: float
    stationarity_residual_high: float
    stationarity_residual_root: float
    stationarity_residual_tolerance: float
    transverse_area_bohr2: float
    interface_count: int
    coarse_surface_tension_hartree_per_bohr2: float
    fine_surface_tension_hartree_per_bohr2: float
    grid_refinement_tolerance_hartree_per_bohr2: float
    excluded_target_label_sets: tuple[str, ...]
    homogeneous_phase_coexistence: Route2V0PureSolventHomogeneousPhaseEvidence | None

    @property
    def is_physical_pure_liquid_admission(self) -> bool:
        """Return whether this record claims physical-liquid rather than control.

        Synthetic controls remain useful for scalar and derivative tests only.
        """

        return (
            self.evidence_scope
            == V0_PURE_SOLVENT_BRIDGE_CERTIFICATE_EVIDENCE_SCOPE_PHYSICAL_LIQUID
        )

    @property
    def has_homogeneous_phase_coexistence(self) -> bool:
        """Return whether the certificate carries the required phase evidence."""

        return self.homogeneous_phase_coexistence is not None

    @property
    def source_hashes_by_role(self) -> dict[str, str]:
        """Return a fresh immutable-by-convention source-hash view."""

        return dict(self.source_file_sha256)

    def verify_content_integrity(self) -> None:
        """Reparse the content-addressed certificate before physical admission."""

        path = self.certificate_path.resolve()
        if _file_sha256(path) != self.content_sha256:
            raise ValueError(
                "Pure-solvent bridge certificate content changed after it was loaded."
            )
        if load_route2_v0_pure_solvent_bridge_certificate(path) != self:
            raise ValueError(
                "Pure-solvent bridge certificate object does not match its "
                "content-addressed JSON source."
            )

    def verify_bound_inputs(
        self,
        *,
        solvent_id: object,
        model_identifier: object,
        closure: object,
        temperature_kelvin: object,
        pressure_bar: object,
        source_file_sha256: Mapping[str, str],
        hnc_bulk_pressure_hartree_per_bohr3: object,
        molecular_bulk_number_density_bohr3: object,
        center_projection_sha256: object,
        weighted_density_kernel_sha256: object,
    ) -> None:
        """Reject a live scalar unless every pure-liquid binding edge matches."""

        if _text(solvent_id, name="Bound solvent ID") != self.solvent_id:
            raise ValueError("Pure-solvent certificate solvent ID does not match.")
        if (
            _text(model_identifier, name="Bound model identifier")
            != self.model_identifier
        ):
            raise ValueError(
                "Pure-solvent certificate model identifier does not match."
            )
        if (
            _text(closure, name="Bound liquid closure").casefold()
            != self.closure.casefold()
        ):
            raise ValueError("Pure-solvent certificate closure does not match.")
        for name, actual, expected in (
            ("temperature", temperature_kelvin, self.temperature_kelvin),
            ("pressure", pressure_bar, self.pressure_bar),
            (
                "molecular bulk density",
                molecular_bulk_number_density_bohr3,
                self.molecular_bulk_number_density_bohr3,
            ),
            (
                "molecular HNC pressure",
                hnc_bulk_pressure_hartree_per_bohr3,
                self.hnc_bulk_pressure_hartree_per_bohr3,
            ),
        ):
            if not _close(
                _number(actual, name=f"Bound {name}", positive=True), expected
            ):
                raise ValueError(f"Pure-solvent certificate {name} does not match.")
        observed_hashes = {
            _text(role, name="Bound source role"): _digest(
                digest,
                name="Bound source SHA-256",
            )
            for role, digest in source_file_sha256.items()
        }
        if observed_hashes != self.source_hashes_by_role:
            raise ValueError(
                "Pure-solvent certificate source hashes do not match the frozen "
                "solvent asset."
            )
        if (
            _digest(
                center_projection_sha256,
                name="Bound molecular-centre projection SHA-256",
            )
            != self.center_projection_sha256
        ):
            raise ValueError(
                "Pure-solvent certificate molecular-centre projection does not "
                "match the live bridge."
            )
        if (
            _digest(
                weighted_density_kernel_sha256,
                name="Bound weighted-density kernel SHA-256",
            )
            != self.weighted_density_kernel_sha256
        ):
            raise ValueError(
                "Pure-solvent certificate weighted-density kernel does not match "
                "the live bridge."
            )


class _PlanarInterfaceValues(TypedDict):
    quartic_coefficient_low_hartree_bohr15: float
    quartic_coefficient_high_hartree_bohr15: float
    quartic_coefficient_hartree_bohr15: float
    surface_tension_low_hartree_per_bohr2: float
    surface_tension_high_hartree_per_bohr2: float
    surface_tension_root_hartree_per_bohr2: float
    surface_tension_root_tolerance_hartree_per_bohr2: float
    stationarity_residual_low: float
    stationarity_residual_high: float
    stationarity_residual_root: float
    stationarity_residual_tolerance: float
    transverse_area_bohr2: float
    interface_count: int
    coarse_surface_tension_hartree_per_bohr2: float
    fine_surface_tension_hartree_per_bohr2: float
    grid_refinement_tolerance_hartree_per_bohr2: float


def _read_planar_interface(
    entry: Mapping[str, object],
    *,
    target_surface_tension: float,
) -> _PlanarInterfaceValues:
    planar = _mapping(entry.get("planar_interface"), name="planar_interface")
    bracket = _mapping(planar.get("quartic_bracket"), name="quartic_bracket")
    surface = _mapping(planar.get("surface_tension"), name="planar surface_tension")
    stationary = _mapping(planar.get("stationarity"), name="planar stationarity")
    refinement = _mapping(planar.get("grid_refinement"), name="grid_refinement")
    low_b = _number(
        bracket.get("low_hartree_bohr15"),
        name="Quartic lower bracket",
        nonnegative=True,
    )
    high_b = _number(
        bracket.get("high_hartree_bohr15"), name="Quartic upper bracket", positive=True
    )
    root_b = _number(
        bracket.get("root_hartree_bohr15"), name="Quartic root", nonnegative=True
    )
    if low_b > high_b or not low_b <= root_b <= high_b:
        raise ValueError("Planar-interface quartic root must lie inside its bracket.")
    low_gamma = _number(
        surface.get("low_hartree_per_bohr2"),
        name="Lower surface tension",
        positive=True,
    )
    high_gamma = _number(
        surface.get("high_hartree_per_bohr2"),
        name="Upper surface tension",
        positive=True,
    )
    root_gamma = _number(
        surface.get("root_hartree_per_bohr2"),
        name="Root surface tension",
        positive=True,
    )
    root_tolerance = _number(
        surface.get("root_tolerance_hartree_per_bohr2"),
        name="Surface-tension root tolerance",
        positive=True,
    )
    if (
        not min(low_gamma, high_gamma)
        <= target_surface_tension
        <= max(low_gamma, high_gamma)
    ):
        raise ValueError("Target surface tension must lie inside the frozen bracket.")
    if abs(root_gamma - target_surface_tension) > root_tolerance:
        raise ValueError("Planar-interface root misses the target surface tension.")
    residuals = tuple(
        _number(
            stationary.get(key), name=f"Planar stationarity {key}", nonnegative=True
        )
        for key in ("low_residual", "high_residual", "root_residual")
    )
    residual_tolerance = _number(
        stationary.get("tolerance"), name="Planar stationarity tolerance", positive=True
    )
    if max(residuals) > residual_tolerance:
        raise ValueError("Planar-interface stationarity residual exceeds tolerance.")
    coarse = _number(
        refinement.get("coarse_hartree_per_bohr2"),
        name="Coarse-grid surface tension",
        positive=True,
    )
    fine = _number(
        refinement.get("fine_hartree_per_bohr2"),
        name="Fine-grid surface tension",
        positive=True,
    )
    refinement_tolerance = _number(
        refinement.get("tolerance_hartree_per_bohr2"),
        name="Grid-refinement tolerance",
        positive=True,
    )
    if abs(coarse - fine) > refinement_tolerance:
        raise ValueError("Planar-interface grid refinement exceeds tolerance.")
    if abs(fine - root_gamma) > max(refinement_tolerance, root_tolerance):
        raise ValueError("Refined planar interface does not reproduce the root.")
    if planar.get("same_scalar_all_terms_retained") is not True:
        raise ValueError("Planar-interface certificate must retain every scalar term.")
    values: _PlanarInterfaceValues = {
        "quartic_coefficient_low_hartree_bohr15": low_b,
        "quartic_coefficient_high_hartree_bohr15": high_b,
        "quartic_coefficient_hartree_bohr15": root_b,
        "surface_tension_low_hartree_per_bohr2": low_gamma,
        "surface_tension_high_hartree_per_bohr2": high_gamma,
        "surface_tension_root_hartree_per_bohr2": root_gamma,
        "surface_tension_root_tolerance_hartree_per_bohr2": root_tolerance,
        "stationarity_residual_low": residuals[0],
        "stationarity_residual_high": residuals[1],
        "stationarity_residual_root": residuals[2],
        "stationarity_residual_tolerance": residual_tolerance,
        "transverse_area_bohr2": _number(
            planar.get("transverse_area_bohr2"),
            name="Planar transverse area",
            positive=True,
        ),
        "interface_count": _positive_integer(
            planar.get("interface_count"), name="Planar interface count"
        ),
        "coarse_surface_tension_hartree_per_bohr2": coarse,
        "fine_surface_tension_hartree_per_bohr2": fine,
        "grid_refinement_tolerance_hartree_per_bohr2": refinement_tolerance,
    }
    return values


def _read_homogeneous_phase_coexistence(
    entry: Mapping[str, object],
    *,
    required: bool,
) -> Route2V0PureSolventHomogeneousPhaseEvidence | None:
    """Read full-scalar coexistence evidence before a planar phase is admitted.

    A source-bound synthetic control may omit this record.  A certificate that
    claims physical pure-liquid admission may not: the empty-density pressure
    identity is weaker than a stable, coexisting low-density phase.
    """

    raw = entry.get("homogeneous_phase_coexistence")
    if raw is None:
        if required:
            raise ValueError(
                "Physical pure-liquid admission requires homogeneous phase "
                "coexistence evidence before a planar-interface claim."
            )
        return None
    phase = _mapping(raw, name="homogeneous_phase_coexistence")
    expected = {
        "construction",
        "same_scalar_all_terms_retained",
        "zero_external_potential",
        "tolerances",
        "liquid",
        "gas",
        "coexistence",
    }
    if set(phase) != expected:
        missing = sorted(expected.difference(phase))
        extra = sorted(set(phase).difference(expected))
        raise ValueError(
            "Homogeneous phase-coexistence evidence keys differ; "
            f"missing={missing}, extra={extra}."
        )
    if (
        _text(phase.get("construction"), name="Homogeneous phase construction")
        != V0_MOLECULAR_HOMOGENEOUS_PHASE_CONSTRUCTION
    ):
        raise ValueError("Unsupported homogeneous phase-coexistence construction.")
    if phase.get("same_scalar_all_terms_retained") is not True:
        raise ValueError(
            "Homogeneous phase evidence must retain every liquid-scalar term."
        )
    zero_external = _mapping(
        phase.get("zero_external_potential"),
        name="homogeneous zero_external_potential",
    )
    if set(zero_external) != {"residual_hartree", "tolerance_hartree"}:
        raise ValueError(
            "Homogeneous zero-external-potential evidence has unsupported keys."
        )
    external_residual = _number(
        zero_external.get("residual_hartree"),
        name="Homogeneous zero-external-potential residual",
        nonnegative=True,
    )
    external_tolerance = _number(
        zero_external.get("tolerance_hartree"),
        name="Homogeneous zero-external-potential tolerance",
        positive=True,
    )
    if external_residual > external_tolerance:
        raise ValueError(
            "Homogeneous phase evidence has a nonzero pure-liquid external "
            "potential beyond its tolerance."
        )
    tolerances = _mapping(phase.get("tolerances"), name="homogeneous tolerances")
    if set(tolerances) != {
        "stationarity",
        "gradient_uniformity",
        "curvature_hartree_per_bohr3",
    }:
        raise ValueError("Homogeneous phase tolerances have unsupported keys.")
    stationarity_tolerance = _number(
        tolerances.get("stationarity"),
        name="Homogeneous phase stationarity tolerance",
        positive=True,
    )
    uniformity_tolerance = _number(
        tolerances.get("gradient_uniformity"),
        name="Homogeneous phase gradient-uniformity tolerance",
        positive=True,
    )
    curvature_tolerance = _number(
        tolerances.get("curvature_hartree_per_bohr3"),
        name="Homogeneous phase curvature tolerance",
        positive=True,
    )

    def read_phase(
        key: str,
        *,
        liquid: bool,
    ) -> tuple[float, float, float, float, float]:
        record = _mapping(phase.get(key), name=f"homogeneous {key} phase")
        required_keys = {
            "density_scale",
            "grand_potential_density_hartree_per_bohr3",
            "stationarity_residual",
            "gradient_uniformity_residual",
            "curvature_hartree_per_bohr3",
        }
        if set(record) != required_keys:
            raise ValueError(f"Homogeneous {key} phase has unsupported keys.")
        density_scale = _number(
            record.get("density_scale"),
            name=f"Homogeneous {key} density scale",
            positive=True,
        )
        if liquid:
            if not _close(density_scale, 1.0):
                raise ValueError(
                    "Homogeneous liquid phase must use the declared bulk density "
                    "scale one."
                )
        elif density_scale >= 1.0:
            raise ValueError(
                "Homogeneous gas phase must be below the declared liquid density "
                "scale one."
            )
        omega = _number(
            record.get("grand_potential_density_hartree_per_bohr3"),
            name=f"Homogeneous {key} grand-potential density",
        )
        stationarity = _number(
            record.get("stationarity_residual"),
            name=f"Homogeneous {key} stationarity residual",
            nonnegative=True,
        )
        uniformity = _number(
            record.get("gradient_uniformity_residual"),
            name=f"Homogeneous {key} gradient-uniformity residual",
            nonnegative=True,
        )
        curvature = _number(
            record.get("curvature_hartree_per_bohr3"),
            name=f"Homogeneous {key} curvature",
        )
        if stationarity > stationarity_tolerance:
            raise ValueError(
                f"Homogeneous {key} phase is not stationary within its tolerance."
            )
        if uniformity > uniformity_tolerance:
            raise ValueError(
                f"Homogeneous {key} phase is not configuration-uniform within its "
                "tolerance."
            )
        if curvature <= curvature_tolerance:
            raise ValueError(f"Homogeneous {key} phase is not a strict local minimum.")
        return density_scale, omega, stationarity, uniformity, curvature

    (
        liquid_scale,
        liquid_omega,
        liquid_stationarity,
        liquid_uniformity,
        liquid_curvature,
    ) = read_phase("liquid", liquid=True)
    gas_scale, gas_omega, gas_stationarity, gas_uniformity, gas_curvature = read_phase(
        "gas", liquid=False
    )
    coexistence = _mapping(phase.get("coexistence"), name="homogeneous coexistence")
    if set(coexistence) != {
        "grand_potential_density_difference_hartree_per_bohr3",
        "tolerance_hartree_per_bohr3",
    }:
        raise ValueError("Homogeneous coexistence evidence has unsupported keys.")
    difference = _number(
        coexistence.get("grand_potential_density_difference_hartree_per_bohr3"),
        name="Homogeneous coexistence grand-potential density difference",
    )
    tolerance = _number(
        coexistence.get("tolerance_hartree_per_bohr3"),
        name="Homogeneous coexistence tolerance",
        positive=True,
    )
    expected_difference = gas_omega - liquid_omega
    difference_tolerance = 1.0e-12 * max(1.0, abs(difference), abs(expected_difference))
    if abs(difference - expected_difference) > difference_tolerance:
        raise ValueError(
            "Homogeneous coexistence gap must equal the gas-minus-liquid "
            "grand-potential density."
        )
    if abs(difference) > tolerance:
        raise ValueError(
            "Homogeneous gas and liquid phases are not coexistent within the "
            "declared tolerance."
        )
    return Route2V0PureSolventHomogeneousPhaseEvidence(
        zero_external_potential_residual_hartree=external_residual,
        zero_external_potential_tolerance_hartree=external_tolerance,
        stationarity_tolerance=stationarity_tolerance,
        gradient_uniformity_tolerance=uniformity_tolerance,
        curvature_tolerance_hartree_per_bohr3=curvature_tolerance,
        coexistence_tolerance_hartree_per_bohr3=tolerance,
        liquid_density_scale=liquid_scale,
        gas_density_scale=gas_scale,
        liquid_grand_potential_density_hartree_per_bohr3=liquid_omega,
        gas_grand_potential_density_hartree_per_bohr3=gas_omega,
        liquid_stationarity_residual=liquid_stationarity,
        gas_stationarity_residual=gas_stationarity,
        liquid_gradient_uniformity_residual=liquid_uniformity,
        gas_gradient_uniformity_residual=gas_uniformity,
        liquid_curvature_hartree_per_bohr3=liquid_curvature,
        gas_curvature_hartree_per_bohr3=gas_curvature,
        grand_potential_density_difference_hartree_per_bohr3=difference,
    )


def load_route2_v0_pure_solvent_bridge_certificate(
    path: str | Path,
) -> Route2V0PureSolventBridgeCertificate:
    """Load a strict, content-addressed, no-solvation-label bridge certificate."""

    path = Path(path).resolve()
    content_sha256 = _file_sha256(path)
    try:
        entry = _mapping(
            json.loads(path.read_text(encoding="utf-8")), name="Certificate"
        )
    except json.JSONDecodeError as exc:
        raise ValueError(
            "Pure-solvent bridge certificate must contain valid JSON."
        ) from exc
    if entry.get("construction") != V0_PURE_SOLVENT_BRIDGE_CERTIFICATE_CONSTRUCTION:
        raise ValueError("Unsupported Route-2 pure-solvent bridge certificate.")
    if entry.get("schema_version") != V0_PURE_SOLVENT_BRIDGE_CERTIFICATE_SCHEMA_VERSION:
        raise ValueError("Unsupported pure-solvent bridge certificate schema version.")
    liquid = _mapping(entry.get("liquid_source"), name="liquid_source")
    evidence_scope, admission_claim_boundary = _admission(entry.get("admission"))
    pressure = _mapping(entry.get("pure_liquid_pressure"), name="pure_liquid_pressure")
    surface = _mapping(entry.get("surface_tension"), name="surface_tension")
    bridge = _mapping(entry.get("bridge"), name="bridge")
    operators = _mapping(entry.get("operators"), name="operators")
    policy = _mapping(entry.get("no_target_policy"), name="no_target_policy")
    for key in (
        "target_solvation_labels_used",
        "post_training",
        "fine_tuning",
        "experimental_solvation_fit",
        "map_or_uq_calibration",
        "error_driven_cavity_or_dispersion_adjustment",
    ):
        if policy.get(key) is not False:
            raise ValueError(f"no_target_policy.{key} must be explicitly false.")
    solvent_id = _text(liquid.get("solvent_id"), name="liquid_source.solvent_id")
    if not _SOLVENT_ID.fullmatch(solvent_id):
        raise ValueError("liquid_source.solvent_id must be a lowercase stable slug.")
    closure = _text(liquid.get("closure"), name="liquid_source.closure")
    if closure.casefold() != "hnc":
        raise ValueError("Pure-solvent weighted-density certificate requires HNC.")
    temperature = _number(
        liquid.get("temperature_kelvin"), name="Liquid temperature", positive=True
    )
    pressure_bar = _number(
        liquid.get("pressure_bar"), name="Liquid pressure", positive=True
    )
    density = _number(
        liquid.get("molecular_bulk_number_density_bohr3"),
        name="Molecular bulk density",
        positive=True,
    )
    hnc_pressure = _number(
        bridge.get("hnc_bulk_pressure_hartree_per_bohr3"),
        name="Molecular HNC pressure",
        positive=True,
    )
    target_pressure = _number(
        pressure.get("target_hartree_per_bohr3"),
        name="Target liquid pressure",
        nonnegative=True,
    )
    if not _close(target_pressure, pressure_bar * 1.0e5 / _HARTREE_PER_BOHR3_TO_PA):
        raise ValueError(
            "Target liquid pressure must equal pressure_bar in atomic units."
        )
    if target_pressure >= hnc_pressure:
        raise ValueError("Target liquid pressure must be below the HNC pressure.")
    target_gamma_n_per_m = _number(
        surface.get("target_n_per_m"),
        name="Target surface tension in N/m",
        positive=True,
    )
    target_gamma = _number(
        surface.get("target_hartree_per_bohr2"),
        name="Target surface tension in Hartree/Bohr^2",
        positive=True,
    )
    if not _close(target_gamma, target_gamma_n_per_m / _HARTREE_PER_BOHR2_TO_N_PER_M):
        raise ValueError("Target surface tension must use the exact SI conversion.")
    cubic = _number(
        bridge.get("cubic_coefficient_hartree_bohr6"),
        name="Cubic bridge coefficient",
        positive=True,
    )
    if not _close(cubic, (hnc_pressure - target_pressure) / density**3):
        raise ValueError(
            "Cubic bridge coefficient must satisfy the HNC pressure identity."
        )
    homogeneous_phase_coexistence = _read_homogeneous_phase_coexistence(
        entry,
        required=(
            evidence_scope
            == V0_PURE_SOLVENT_BRIDGE_CERTIFICATE_EVIDENCE_SCOPE_PHYSICAL_LIQUID
        ),
    )
    if (
        evidence_scope
        == V0_PURE_SOLVENT_BRIDGE_CERTIFICATE_EVIDENCE_SCOPE_PHYSICAL_LIQUID
    ):
        raise ValueError(
            "Pure-solvent bridge certificate v1 cannot claim physical-liquid "
            "admission: it lacks the required nested same-scalar quartic "
            "coexistence continuation and constrained planar-interface schema."
        )
    return Route2V0PureSolventBridgeCertificate(
        certificate_path=path,
        content_sha256=content_sha256,
        evidence_scope=evidence_scope,
        admission_claim_boundary=admission_claim_boundary,
        solvent_id=solvent_id,
        model_identifier=_text(
            liquid.get("model_identifier"), name="Liquid model identifier"
        ),
        closure=closure,
        temperature_kelvin=temperature,
        pressure_bar=pressure_bar,
        molecular_bulk_number_density_bohr3=density,
        source_file_sha256=_source_hashes(liquid.get("source_file_sha256")),
        center_projection_sha256=_digest(
            operators.get("center_projection_sha256"),
            name="operators.center_projection_sha256",
        ),
        weighted_density_kernel_sha256=_digest(
            operators.get("weighted_density_kernel_sha256"),
            name="operators.weighted_density_kernel_sha256",
        ),
        hnc_bulk_pressure_hartree_per_bohr3=hnc_pressure,
        target_bulk_pressure_hartree_per_bohr3=target_pressure,
        target_surface_tension_n_per_m=target_gamma_n_per_m,
        target_surface_tension_hartree_per_bohr2=target_gamma,
        surface_tension_reference=_text(
            surface.get("independent_reference"),
            name="Independent surface-tension reference",
        ),
        cubic_coefficient_hartree_bohr6=cubic,
        excluded_target_label_sets=_excluded_label_sets(
            policy.get("excluded_target_label_sets")
        ),
        homogeneous_phase_coexistence=homogeneous_phase_coexistence,
        **_read_planar_interface(entry, target_surface_tension=target_gamma),
    )


__all__ = [
    "V0_CANONICAL_NUMERIC_ARRAY_DIGEST_CONSTRUCTION",
    "V0_MOLECULAR_HOMOGENEOUS_PHASE_CONSTRUCTION",
    "V0_PURE_SOLVENT_BRIDGE_CERTIFICATE_CONSTRUCTION",
    "V0_PURE_SOLVENT_BRIDGE_CERTIFICATE_EVIDENCE_SCOPES",
    "V0_PURE_SOLVENT_BRIDGE_CERTIFICATE_EVIDENCE_SCOPE_PHYSICAL_LIQUID",
    "V0_PURE_SOLVENT_BRIDGE_CERTIFICATE_EVIDENCE_SCOPE_SYNTHETIC_CONTROL",
    "V0_PURE_SOLVENT_BRIDGE_CERTIFICATE_SCHEMA_VERSION",
    "V0_PURE_SOLVENT_BRIDGE_CERTIFICATE_V1_PHYSICAL_ADMISSION_REJECTED",
    "V0_WEIGHTED_DENSITY_OPERATOR_DIGEST_CONSTRUCTION",
    "Route2V0PureSolventBridgeCertificate",
    "Route2V0PureSolventHomogeneousPhaseEvidence",
    "canonical_float64_array_sha256",
    "load_route2_v0_pure_solvent_bridge_certificate",
    "weighted_density_operator_sha256",
]
