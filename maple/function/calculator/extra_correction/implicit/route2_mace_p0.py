"""Evidence binding for the MACE-POLAR Route-2 P0 audit.

The live audit is intentionally kept outside the import-time calculator path.
This module only owns small, deterministic pieces that the run-time plug-in
needs: content hashing, the analytic Born sign canary, field-convention
selection, and validation of a versioned audit certificate.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import inspect
import json
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .route2_plugin_errors import PluginContractError
from .route2_plugin_registry import ArtifactDispositionRegistry

MACE_P0_ARTIFACT_ID = "route2-mace-p0-field-contract-v1"
MACE_P0_SCHEMA_VERSION = 1
MACE_P0_ARTIFACT_RELATIVE_PATH = (
    "docs/implicit-solvation/benchmarks/route2-mace-p0-field-contract-v1.json"
)
MACE_P0_DISPOSITION_REGISTRY_RELATIVE_PATH = (
    "docs/implicit-solvation/benchmarks/" "route2-mace-p0-artifact-dispositions-v1.json"
)

AuditedFieldConvention = Literal[
    "potential-gradient",
    "physical-electric-field",
]


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of one file without loading it all at once."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def composite_sha256(file_hashes: Mapping[str, str]) -> str:
    """Hash a labelled set of file digests with a canonical JSON encoding."""

    normalized: dict[str, str] = {}
    for raw_label, raw_digest in file_hashes.items():
        label = str(raw_label).strip()
        digest = str(raw_digest).strip().lower()
        if not label:
            raise ValueError("inference-source labels must be non-empty.")
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise ValueError(f"inference-source digest for {label!r} is not SHA-256.")
        normalized[label] = digest
    if not normalized:
        raise ValueError("at least one inference-source digest is required.")
    payload = json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def runtime_inference_file_hashes(calculator: object) -> dict[str, str]:
    """Fingerprint the exact local and upstream field-inference code in use."""

    model = getattr(calculator, "model", None)
    reaction_projector = getattr(calculator, "_reaction_projector", None)
    upstream_projector = getattr(reaction_projector, "upstream", None)
    module_directory = Path(__file__).resolve().parent
    candidates = {
        "maple_mace_calculator": inspect.getsourcefile(type(calculator)),
        "maple_mace_plugin": module_directory / "route2_mace_plugin.py",
        "maple_local_reaction_field": module_directory / "route2_field_state.py",
        "maple_electrostatic_pairing": module_directory / "electrostatic_pairing.py",
        "mace_polar_model": inspect.getsourcefile(type(model)),
        "graph_longrange_external_field_projector": inspect.getsourcefile(
            type(upstream_projector)
        ),
    }
    missing = [
        label
        for label, path in candidates.items()
        if path is None or not Path(path).is_file()
    ]
    if missing:
        raise PluginContractError(
            "Unable to locate MACE-POLAR inference source files: "
            + ", ".join(missing)
            + "."
        )
    return {
        label: sha256_file(Path(path).resolve())
        for label, path in candidates.items()
        if path is not None
    }


def born_point_charge_sign_canary(
    *, charge_e: float = 1.0, radius_bohr: float = 3.0, dielectric: float = 78.355
) -> dict[str, float | bool | str]:
    r"""Evaluate the analytic central-charge Born reaction-potential sign.

    In atomic units,

    ``phi_reac = -(1 - 1/epsilon) q / a`` and
    ``G_pol = 0.5 q phi_reac``.

    This is a sign and half-coupling canary only.  It does not exercise a
    cavity discretisation or certify a continuum implementation.
    """

    charge = float(charge_e)
    radius = float(radius_bohr)
    epsilon = float(dielectric)
    if not np.isfinite(charge) or charge == 0.0:
        raise ValueError("Born canary charge must be finite and non-zero.")
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("Born canary radius must be finite and positive.")
    if not np.isfinite(epsilon) or epsilon <= 1.0:
        raise ValueError("Born canary dielectric must be finite and exceed one.")
    reaction_potential = -(1.0 - 1.0 / epsilon) * charge / radius
    full_pairing = charge * reaction_potential
    half_coupling_energy = 0.5 * full_pairing
    passed = (
        np.sign(reaction_potential) == -np.sign(charge)
        and full_pairing < 0.0
        and half_coupling_energy < 0.0
        and abs(full_pairing - 2.0 * half_coupling_energy) <= 1.0e-15
    )
    return {
        "equation": "phi_reac=-(1-1/epsilon)q/a; G=0.5*q*phi_reac",
        "charge_e": charge,
        "radius_bohr": radius,
        "dielectric": epsilon,
        "reaction_potential_hartree_per_e": reaction_potential,
        "full_pairing_hartree": full_pairing,
        "half_coupling_energy_hartree": half_coupling_energy,
        "passed": bool(passed),
    }


def select_uniform_field_convention(
    *,
    positive_mapping_errors: Sequence[float],
    negative_mapping_errors: Sequence[float],
    absolute_tolerance: float,
    minimum_discrimination_ratio: float,
) -> AuditedFieldConvention:
    """Select the unique local-gradient sign matching upstream uniform fields.

    A positive mapping means ``grad(phi) = f_upstream``.  A negative mapping
    means ``grad(phi) = -f_upstream``.  Ambiguous or inaccurate probes fail
    closed instead of selecting a convention by method name.
    """

    positive = np.asarray(tuple(positive_mapping_errors), dtype=float)
    negative = np.asarray(tuple(negative_mapping_errors), dtype=float)
    tolerance = float(absolute_tolerance)
    ratio = float(minimum_discrimination_ratio)
    if (
        positive.ndim != 1
        or negative.ndim != 1
        or positive.size == 0
        or positive.shape != negative.shape
        or not np.all(np.isfinite(positive))
        or not np.all(np.isfinite(negative))
    ):
        raise ValueError("field-convention errors must be matched finite vectors.")
    if tolerance < 0.0 or ratio <= 1.0 or not np.isfinite(ratio):
        raise ValueError("field-convention thresholds are invalid.")

    positive_max = float(np.max(positive))
    negative_max = float(np.max(negative))
    floor = max(tolerance, np.finfo(float).eps)
    positive_pass = positive_max <= tolerance and negative_max >= ratio * floor
    negative_pass = negative_max <= tolerance and positive_max >= ratio * floor
    if positive_pass == negative_pass:
        raise PluginContractError(
            "The MACE-POLAR uniform-field audit did not select a unique field "
            "convention."
        )
    return "potential-gradient" if positive_pass else "physical-electric-field"


def _require_mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PluginContractError(f"P0 artifact {name} must be a mapping.")
    return value


def _require_passed(gates: Mapping[str, Any], name: str) -> None:
    gate = _require_mapping(gates.get(name), name=f"gate {name!r}")
    if gate.get("passed") is not True:
        raise PluginContractError(
            f"P0 artifact gate {name!r} did not pass; certification is blocked."
        )


@dataclass(frozen=True)
class MACEP0AuditCertificate:
    """Validated, content-addressed P0 evidence usable by ``MACEPolarPlugin``."""

    artifact_id: str
    artifact_path: str
    artifact_sha256: str
    checkpoint_sha256: str
    inference_code_sha256: str
    field_convention: AuditedFieldConvention
    coordinate_frame_policy: str
    model_field_evaluator: str
    audited_atomic_numbers: frozenset[int]
    total_charge_domain_e: tuple[float, float]
    optimizer_coverage: Literal["unattestable"]
    blocked_routes: tuple[str, ...]

    @classmethod
    def load(
        cls,
        artifact_path: str | Path,
        *,
        disposition_registry: ArtifactDispositionRegistry,
        repo_root: str | Path | None = None,
        verify_repository_sources: bool = True,
    ) -> "MACEP0AuditCertificate":
        path = Path(artifact_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise PluginContractError("P0 artifact root must be a JSON object.")
        if (
            payload.get("artifact") != MACE_P0_ARTIFACT_ID
            or payload.get("schema_version") != MACE_P0_SCHEMA_VERSION
        ):
            raise PluginContractError("Unsupported MACE-POLAR P0 artifact identity.")

        record = disposition_registry.disposition_for(MACE_P0_ARTIFACT_ID)
        if record is None or record.disposition != "accepted-current":
            raise PluginContractError(
                "The MACE-POLAR P0 artifact is not accepted-current in the "
                "disposition registry."
            )
        artifact_digest = sha256_file(path)
        if artifact_digest != record.artifact_sha256:
            raise PluginContractError(
                "The MACE-POLAR P0 artifact bytes do not match their disposition."
            )
        if repo_root is not None:
            relative = path.relative_to(Path(repo_root).resolve()).as_posix()
            if relative != record.artifact_path:
                raise PluginContractError(
                    "The MACE-POLAR P0 artifact path does not match its disposition."
                )

        gates = _require_mapping(payload.get("gates"), name="gates")
        for gate_name in (
            "uniform_field_interface",
            "canonical_translation_gauge",
            "canonical_rotation_covariance",
            "batch_isolation",
            "born_sign",
        ):
            _require_passed(gates, gate_name)

        decision = _require_mapping(payload.get("decision"), name="decision")
        convention = decision.get("field_convention")
        if convention not in {"potential-gradient", "physical-electric-field"}:
            raise PluginContractError(
                "P0 artifact did not establish a supported field convention."
            )
        if decision.get("admitted_route") != "response-only":
            raise PluginContractError(
                "This P0 certificate admits only the response-only route."
            )
        raw_blocked = decision.get("blocked_routes")
        if not isinstance(raw_blocked, list) or not all(
            isinstance(value, str) and value for value in raw_blocked
        ):
            raise PluginContractError("P0 artifact blocked_routes must be a list.")
        blocked = tuple(raw_blocked)
        if not {"operational-force", "variational"}.issubset(blocked):
            raise PluginContractError(
                "P0 evidence must keep operational-force and variational routes blocked."
            )
        if decision.get("optimizer_coverage") != "unattestable":
            raise PluginContractError(
                "The public MACE-POLAR checkpoint has no auditable optimizer coverage."
            )

        checkpoint = _require_mapping(payload.get("checkpoint"), name="checkpoint")
        checkpoint_sha = str(checkpoint.get("sha256", "")).lower()
        if len(checkpoint_sha) != 64 or any(
            character not in "0123456789abcdef" for character in checkpoint_sha
        ):
            raise PluginContractError("P0 artifact checkpoint SHA-256 is invalid.")
        inference_sha = str(payload.get("inference_code_sha256", "")).lower()
        runtime_hashes = _require_mapping(
            payload.get("runtime_inference_files_sha256"),
            name="runtime inference hashes",
        )
        if composite_sha256(runtime_hashes) != inference_sha:
            raise PluginContractError(
                "P0 artifact inference-code composite hash is inconsistent."
            )

        if verify_repository_sources:
            if repo_root is None:
                raise ValueError(
                    "repo_root is required when repository source verification is enabled."
                )
            source_hashes = _require_mapping(
                payload.get("repository_source_files_sha256"),
                name="repository source hashes",
            )
            root = Path(repo_root).resolve()
            for relative, expected in source_hashes.items():
                source_path = root / str(relative)
                if not source_path.is_file() or sha256_file(source_path) != expected:
                    raise PluginContractError(
                        "P0 artifact repository-source hash mismatch for "
                        f"{relative!r}."
                    )

        protocol = _require_mapping(payload.get("protocol"), name="protocol")
        audited_numbers = frozenset(protocol.get("audited_atomic_numbers", ()))
        if not audited_numbers or any(
            isinstance(number, bool) or not isinstance(number, int) or number < 1
            for number in audited_numbers
        ):
            raise PluginContractError(
                "P0 artifact audited_atomic_numbers must be positive integers."
            )
        raw_charge_domain = protocol.get("audited_total_charge_domain_e")
        if (
            not isinstance(raw_charge_domain, list)
            or len(raw_charge_domain) != 2
            or not all(np.isfinite(float(value)) for value in raw_charge_domain)
        ):
            raise PluginContractError(
                "P0 artifact audited_total_charge_domain_e is invalid."
            )
        charge_domain = (
            float(raw_charge_domain[0]),
            float(raw_charge_domain[1]),
        )
        if charge_domain[0] > charge_domain[1]:
            raise PluginContractError("P0 artifact charge-domain bounds are reversed.")
        return cls(
            artifact_id=MACE_P0_ARTIFACT_ID,
            artifact_path=record.artifact_path,
            artifact_sha256=artifact_digest,
            checkpoint_sha256=checkpoint_sha,
            inference_code_sha256=inference_sha,
            field_convention=convention,
            coordinate_frame_policy=str(protocol.get("coordinate_frame_policy", "")),
            model_field_evaluator=str(protocol.get("model_field_evaluator", "")),
            audited_atomic_numbers=audited_numbers,
            total_charge_domain_e=charge_domain,
            optimizer_coverage="unattestable",
            blocked_routes=blocked,
        )

    def verify_calculator(self, calculator: object) -> None:
        """Reject a certificate replayed with different weights or code."""

        checkpoint = getattr(calculator, "mace_polar_checkpoint_provenance", None)
        if not isinstance(checkpoint, Mapping) or (
            str(checkpoint.get("sha256", "")).lower() != self.checkpoint_sha256
        ):
            raise PluginContractError(
                "MACE-POLAR P0 certificate/checkpoint identity mismatch."
            )
        current_hashes = runtime_inference_file_hashes(calculator)
        if composite_sha256(current_hashes) != self.inference_code_sha256:
            raise PluginContractError(
                "MACE-POLAR P0 certificate/inference-code identity mismatch."
            )
        if (
            str(getattr(calculator, "route2_mace_geometry_frame_policy", ""))
            != self.coordinate_frame_policy
        ):
            raise PluginContractError(
                "MACE-POLAR P0 certificate/coordinate-frame policy mismatch."
            )
        if (
            str(getattr(calculator, "long_range_evaluator_profile", ""))
            != self.model_field_evaluator
        ):
            raise PluginContractError(
                "MACE-POLAR P0 certificate/field-evaluator policy mismatch."
            )


def load_mace_p0_certificate(
    repo_root: str | Path,
    *,
    verify_repository_sources: bool = True,
) -> MACEP0AuditCertificate:
    """Load the repository's accepted-current MACE-POLAR P0 certificate."""

    root = Path(repo_root).resolve()
    registry_payload = json.loads(
        (root / MACE_P0_DISPOSITION_REGISTRY_RELATIVE_PATH).read_text(encoding="utf-8")
    )
    registry = ArtifactDispositionRegistry.from_mapping(registry_payload)
    return MACEP0AuditCertificate.load(
        root / MACE_P0_ARTIFACT_RELATIVE_PATH,
        disposition_registry=registry,
        repo_root=root,
        verify_repository_sources=verify_repository_sources,
    )


__all__ = [
    "AuditedFieldConvention",
    "MACEP0AuditCertificate",
    "MACE_P0_ARTIFACT_ID",
    "MACE_P0_ARTIFACT_RELATIVE_PATH",
    "MACE_P0_DISPOSITION_REGISTRY_RELATIVE_PATH",
    "MACE_P0_SCHEMA_VERSION",
    "born_point_charge_sign_canary",
    "composite_sha256",
    "load_mace_p0_certificate",
    "runtime_inference_file_hashes",
    "select_uniform_field_convention",
    "sha256_file",
]
