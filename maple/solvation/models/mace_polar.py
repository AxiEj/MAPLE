"""Fail-closed MACE-POLAR-1 model adapter for the vNext Route-2 contracts.

The implementation delegates every numerical model operation to the audited
``MACEPolCalculator`` methods.  It does not duplicate MACE inference.  The
currently validated checkpoint can drive the local atom-resolved field
interface, but its one-width density source and two-width receiver basis are
not one conjugate exact-GTO operator.  The exact-GTO production profile and
all public E/F/H/V/M capabilities therefore remain closed.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np

from maple.solvation.api.profiles import LOCAL_JET_DIAGNOSTIC_COUPLING_ID
from maple.solvation.coupling.operator import (
    canonical_metadata_sha256,
    source_files_sha256,
)
from maple.solvation.coupling.spaces import (
    ATOMIC_L1_FIELD_DUAL_SPACE,
    ATOMIC_L1_SOURCE_SPACE,
)
from maple.solvation.coupling.state_equation import provider_behavior_sha256

from .base import (
    ElectronicSourceState,
    ModelDomain,
    ModelProvenance,
    VacuumState,
    array_sha256,
    atom_count,
    model_input_sha256,
)

OFFICIAL_MACE_POLAR_MODEL_PROFILE_ID = "mace-polar-route2-source-field-contract-v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")
    return value.strip()


def _hash(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class MACEPolarReleaseContract:
    """Expected checkpoint/runtime identity for one explicitly named adapter."""

    provider_id: str
    model_profile_id: str
    checkpoint_identifier: str
    checkpoint_release_url: str
    checkpoint_sha256: str
    checkpoint_size_bytes: int
    mace_torch_version: str
    graph_longrange_version: str
    upstream_commit: str
    release_status: str

    def __post_init__(self) -> None:
        for name in (
            "provider_id",
            "model_profile_id",
            "checkpoint_identifier",
            "checkpoint_release_url",
            "mace_torch_version",
            "graph_longrange_version",
            "upstream_commit",
            "release_status",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        digest = str(self.checkpoint_sha256).lower()
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("checkpoint_sha256 must contain 64 hexadecimal digits.")
        object.__setattr__(self, "checkpoint_sha256", digest)
        if (
            isinstance(self.checkpoint_size_bytes, bool)
            or not isinstance(self.checkpoint_size_bytes, int)
            or self.checkpoint_size_bytes < 1
        ):
            raise ValueError("checkpoint_size_bytes must be a positive integer.")

    def metadata(self) -> dict[str, object]:
        return {
            "provider_id": self.provider_id,
            "model_profile_id": self.model_profile_id,
            "checkpoint_identifier": self.checkpoint_identifier,
            "checkpoint_release_url": self.checkpoint_release_url,
            "checkpoint_sha256": self.checkpoint_sha256,
            "checkpoint_size_bytes": self.checkpoint_size_bytes,
            "mace_torch_version": self.mace_torch_version,
            "graph_longrange_version": self.graph_longrange_version,
            "upstream_commit": self.upstream_commit,
            "release_status": self.release_status,
        }


OFFICIAL_MACE_POLAR_1_M_CONTRACT = MACEPolarReleaseContract(
    provider_id="maple.route2.model.mace-polar-1-m-local-field.impl.v1",
    model_profile_id=OFFICIAL_MACE_POLAR_MODEL_PROFILE_ID,
    checkpoint_identifier="polar-1-m",
    checkpoint_release_url=(
        "https://github.com/ACEsuit/mace-foundations/releases/download/"
        "mace_polar_1/MACE-POLAR-1-M.model"
    ),
    checkpoint_sha256=(
        "fab8b8713c832f31a2a853aaa22fd638be8a369cbf5095e6b3e982a18d10e93a"
    ),
    checkpoint_size_bytes=68_133_235,
    mace_torch_version="0.3.16",
    graph_longrange_version="0.4.0",
    upstream_commit="unavailable-in-installed-wheel-release-metadata",
    release_status="operational-candidate; exact-gto-and-E/F/H/V/M-unadmitted",
)


@dataclass(frozen=True, slots=True)
class MACEPolarExactGTOCompatibilityAudit:
    source_sigmas_angstrom: tuple[float, ...]
    source_max_l: int
    source_normalization: str
    receiver_sigmas_angstrom: tuple[float, ...]
    receiver_max_l: int
    receiver_normalization: str
    upstream_matrix_shape: tuple[int, int]
    same_basis_conjugacy: bool
    exact_gto_operational_available: bool
    reason: str

    def metadata(self) -> dict[str, object]:
        return {
            "source_sigmas_angstrom": list(self.source_sigmas_angstrom),
            "source_max_l": self.source_max_l,
            "source_normalization": self.source_normalization,
            "receiver_sigmas_angstrom": list(self.receiver_sigmas_angstrom),
            "receiver_max_l": self.receiver_max_l,
            "receiver_normalization": self.receiver_normalization,
            "upstream_matrix_shape": list(self.upstream_matrix_shape),
            "same_basis_conjugacy": self.same_basis_conjugacy,
            "exact_gto_operational_available": self.exact_gto_operational_available,
            "reason": self.reason,
        }


def _basis_metadata(calculator: object) -> MACEPolarExactGTOCompatibilityAudit:
    model = getattr(calculator, "model", None)
    coulomb = getattr(model, "coulomb_energy", None)
    source_basis = getattr(coulomb, "density_basis", None)
    source_sigmas = tuple(float(v) for v in getattr(source_basis, "sigmas", ()))
    source_max_l = int(getattr(source_basis, "max_l", -1))
    source_normalization = str(getattr(source_basis, "normalize", ""))
    projection_spec = calculator.route2_gto_field_projection_spec()
    receiver_sigmas = tuple(float(v) for v in projection_spec.receiver_sigmas_angstrom)
    receiver_max_l = int(projection_spec.receiver_max_l)
    receiver_normalization = str(projection_spec.receiver_normalization)
    matrix = np.asarray(projection_spec.upstream_matrix, dtype=float)
    if (
        not source_sigmas
        or source_max_l < 0
        or not source_normalization
        or not receiver_sigmas
        or receiver_max_l < 0
        or not receiver_normalization
        or matrix.ndim != 2
        or not np.all(np.isfinite(matrix))
    ):
        raise RuntimeError("MACE-POLAR checkpoint GTO basis metadata is incomplete.")
    same_basis = (
        source_sigmas == receiver_sigmas
        and source_max_l == receiver_max_l
        and source_normalization == receiver_normalization
        and matrix.shape[0] == 4
    )
    if same_basis:
        reason = "source and receiver declare the same four-dimensional basis"
    else:
        reason = (
            "checkpoint density source and field receiver are different radial/"
            "normalization spaces; a 4-component learned source cannot be the "
            "adjoint of the 8-feature receiver without an independently proven map"
        )
    return MACEPolarExactGTOCompatibilityAudit(
        source_sigmas,
        source_max_l,
        source_normalization,
        receiver_sigmas,
        receiver_max_l,
        receiver_normalization,
        tuple(int(v) for v in matrix.shape),
        same_basis,
        False,
        reason,
    )


def _calculator_source_files(calculator: object) -> tuple[tuple[str, str], ...]:
    files: dict[str, str | Path] = {
        "maple.solvation.models.mace_polar": Path(__file__),
    }
    seen: set[Path] = {Path(__file__).resolve()}
    for index, cls in enumerate(type(calculator).__mro__):
        try:
            raw = inspect.getsourcefile(cls)
        except (TypeError, OSError):
            continue
        if raw is None:
            continue
        path = Path(raw).resolve()
        if not path.is_file() or path in seen:
            continue
        seen.add(path)
        files[f"bound_calculator_mro_{index}_{cls.__name__}"] = path
    return source_files_sha256(files)


class MACEPolarLocalFieldModelAdapter:
    """MACE-POLAR vacuum/source/linearization adapter; no PES admission."""

    __slots__ = (
        "_calculator",
        "_release_contract",
        "_checkpoint_path",
        "_checkpoint_stat",
        "_configuration_sha256",
        "_sealed",
        "provider_id",
        "model_profile_id",
        "coupling_id",
        "provenance",
        "provenance_sha256",
        "dtype",
        "device",
        "domain",
        "field_convention",
        "coordinate_frame_policy",
        "exact_gto_audit",
    )
    source_space = ATOMIC_L1_SOURCE_SPACE
    field_space = ATOMIC_L1_FIELD_DUAL_SPACE
    variational_functional_admitted = False
    exact_gto_operational_available = False

    def __init__(
        self,
        calculator: object,
        release_contract: MACEPolarReleaseContract = OFFICIAL_MACE_POLAR_1_M_CONTRACT,
    ) -> None:
        if not isinstance(release_contract, MACEPolarReleaseContract):
            raise TypeError("release_contract must be MACEPolarReleaseContract.")
        required = (
            "polar_state",
            "linearize_density_response",
            "density_position_vjp",
            "route2_gto_field_projection_spec",
        )
        for name in required:
            if not callable(getattr(calculator, name, None)):
                raise TypeError(f"MACE-POLAR calculator must expose callable {name}().")
        checkpoint = getattr(calculator, "mace_polar_checkpoint_provenance", None)
        if not isinstance(checkpoint, Mapping):
            raise ValueError("MACE-POLAR calculator lacks checkpoint provenance.")
        expected_checkpoint = {
            "identifier": release_contract.checkpoint_identifier,
            "release_url": release_contract.checkpoint_release_url,
            "size_bytes": release_contract.checkpoint_size_bytes,
            "sha256": release_contract.checkpoint_sha256,
        }
        for name, expected in expected_checkpoint.items():
            if checkpoint.get(name) != expected:
                raise ValueError(
                    f"MACE-POLAR checkpoint provenance field {name!r} is not release-bound."
                )
        checkpoint_path = Path(str(checkpoint.get("resolved_path", ""))).resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError("MACE-POLAR checkpoint path is unavailable.")
        stat = checkpoint_path.stat()
        if (
            stat.st_size != release_contract.checkpoint_size_bytes
            or _sha256_file(checkpoint_path) != release_contract.checkpoint_sha256
        ):
            raise ValueError(
                "MACE-POLAR checkpoint bytes do not match the release contract."
            )
        if (
            getattr(calculator, "mace_torch_version", None)
            != release_contract.mace_torch_version
        ):
            raise ValueError(
                "MACE-POLAR mace-torch version does not match release contract."
            )
        if (
            getattr(calculator, "graph_longrange_version", None)
            != release_contract.graph_longrange_version
        ):
            raise ValueError(
                "MACE-POLAR graph-longrange version does not match release contract."
            )
        dtype = str(getattr(calculator, "dtype", "")).replace("torch.", "")
        if dtype != "float64":
            raise ValueError("Route-2 MACE-POLAR adapter requires float64 inference.")
        frame = str(getattr(calculator, "route2_mace_geometry_frame_policy", ""))
        if frame != "laboratory-v1":
            raise ValueError(
                "The first vNext MACE-POLAR adapter requires laboratory-v1 coordinates."
            )
        numbers = tuple(
            sorted(set(int(v) for v in getattr(calculator, "atomic_numbers", ())))
        )
        if not numbers or any(value < 1 for value in numbers):
            raise ValueError("MACE-POLAR calculator atomic-number domain is missing.")
        exact_audit = _basis_metadata(calculator)
        domain = ModelDomain(numbers, (0, 0), (1,))
        source_files = _calculator_source_files(calculator)
        inference_sha = canonical_metadata_sha256(dict(source_files))
        provenance = ModelProvenance(
            provider_id=release_contract.provider_id,
            model_profile_id=release_contract.model_profile_id,
            model_family="MACE-POLAR-1",
            checkpoint_sha256=release_contract.checkpoint_sha256,
            upstream_version=(
                f"mace-torch=={release_contract.mace_torch_version};"
                f"graph-longrange=={release_contract.graph_longrange_version}"
            ),
            upstream_commit=release_contract.upstream_commit,
            inference_code_sha256=inference_sha,
            dtype="float64",
            device=str(getattr(calculator, "device", "")),
            domain=domain,
            field_convention=self.field_space.field_convention,
            coordinate_frame_policy="laboratory Cartesian Angstrom; nonperiodic",
            optimizer_parameter_groups_audited=False,
            optimizer_audit_evidence_sha256=None,
        )
        object.__setattr__(self, "_calculator", calculator)
        object.__setattr__(self, "_release_contract", release_contract)
        object.__setattr__(self, "_checkpoint_path", checkpoint_path)
        object.__setattr__(
            self,
            "_checkpoint_stat",
            (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns),
        )
        object.__setattr__(self, "provider_id", provenance.provider_id)
        object.__setattr__(self, "model_profile_id", provenance.model_profile_id)
        object.__setattr__(self, "coupling_id", LOCAL_JET_DIAGNOSTIC_COUPLING_ID)
        object.__setattr__(self, "provenance", provenance)
        object.__setattr__(self, "provenance_sha256", provenance.sha256)
        object.__setattr__(self, "dtype", provenance.dtype)
        object.__setattr__(self, "device", provenance.device)
        object.__setattr__(self, "domain", provenance.domain)
        object.__setattr__(self, "field_convention", provenance.field_convention)
        object.__setattr__(
            self, "coordinate_frame_policy", provenance.coordinate_frame_policy
        )
        object.__setattr__(self, "exact_gto_audit", exact_audit)
        object.__setattr__(
            self, "_configuration_sha256", self._current_configuration_sha256()
        )
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("MACEPolarLocalFieldModelAdapter is immutable.")
        object.__setattr__(self, name, value)

    def _current_configuration_sha256(self) -> str:
        stat = self._checkpoint_path.stat()
        if (
            stat.st_dev,
            stat.st_ino,
            stat.st_size,
            stat.st_mtime_ns,
        ) != self._checkpoint_stat:
            raise RuntimeError(
                "MACE-POLAR checkpoint file identity changed after construction."
            )
        source_files = _calculator_source_files(self._calculator)
        payload = {
            "schema": "route2-mace-polar-local-field-model-adapter-v1",
            "release_contract": self._release_contract.metadata(),
            "source_files_sha256": dict(source_files),
            "calculator_behavior_sha256": provider_behavior_sha256(
                self._calculator,
                (
                    "polar_state",
                    "linearize_density_response",
                    "density_position_vjp",
                    "route2_gto_field_projection_spec",
                ),
                label="mace_polar_calculator",
            ),
            "provider_id": self.provider_id,
            "model_profile_id": self.model_profile_id,
            "coupling_id": self.coupling_id,
            "provenance_sha256": self.provenance_sha256,
            "dtype": str(getattr(self._calculator, "dtype", "")).replace("torch.", ""),
            "device": str(getattr(self._calculator, "device", "")),
            "mace_torch_version": getattr(self._calculator, "mace_torch_version", None),
            "graph_longrange_version": getattr(
                self._calculator, "graph_longrange_version", None
            ),
            "coordinate_frame_policy": getattr(
                self._calculator, "route2_mace_geometry_frame_policy", None
            ),
            "long_range_evaluator_profile": getattr(
                self._calculator, "long_range_evaluator_profile", None
            ),
            "atomic_numbers": list(self.domain.atomic_numbers),
            "source_space_sha256": self.source_space.metadata_hash(),
            "field_space_sha256": self.field_space.metadata_hash(),
            "exact_gto_audit": self.exact_gto_audit.metadata(),
        }
        return _hash(payload)

    def configuration_sha256(self) -> str:
        current = self._current_configuration_sha256()
        if current != self._configuration_sha256:
            raise ValueError(
                "MACE-POLAR adapter configuration drifted after construction."
            )
        return current

    def _state(
        self, atoms: object, field: np.ndarray | None, *, need_forces: bool
    ) -> object:
        self.configuration_sha256()
        self.domain.validate_atoms(atoms)
        kwargs: dict[str, object] = {"compute_forces": need_forces}
        if field is not None:
            values = self.field_space.validate(field, atom_count=atom_count(atoms))
            kwargs.update(
                node_potential_ev=values[:, 0],
                node_gradient_ev_per_angstrom=values[:, 1:],
            )
        result = self._calculator.polar_state(atoms, **kwargs)
        if not isinstance(result, tuple) or len(result) != 2:
            raise TypeError("MACE-POLAR polar_state must return (state, diagnostics).")
        return result[0]

    def evaluate_vacuum(self, atoms: object, *, need_forces: bool) -> VacuumState:
        state = self._state(atoms, None, need_forces=need_forces)
        forces = getattr(state, "fixed_field_forces_ev_per_angstrom", None)
        return VacuumState(
            self.provider_id,
            self.provenance_sha256,
            model_input_sha256(atoms),
            atom_count(atoms),
            float(getattr(state, "energy_ev")),
            need_forces,
            forces,
        )

    def evaluate_source(
        self, atoms: object, field: object, *, need_fixed_field_forces: bool
    ) -> ElectronicSourceState:
        count = atom_count(atoms)
        values = self.field_space.validate(field, atom_count=count)
        state = self._state(atoms, values, need_forces=need_fixed_field_forces)
        source = self.source_space.validate(
            getattr(state, "density_coefficients"), atom_count=count
        )
        return ElectronicSourceState(
            self.provider_id,
            self.provenance_sha256,
            model_input_sha256(atoms),
            array_sha256(values, name="field"),
            self.source_space.metadata_hash(),
            self.field_space.metadata_hash(),
            count,
            source,
            need_fixed_field_forces,
            getattr(state, "fixed_field_forces_ev_per_angstrom", None),
        )

    def _linearization(self, atoms: object, field: object):
        self.configuration_sha256()
        self.domain.validate_atoms(atoms)
        count = atom_count(atoms)
        values = self.field_space.validate(field, atom_count=count)
        return self._calculator.linearize_density_response(
            atoms,
            node_potential_ev=values[:, 0],
            node_gradient_ev_per_angstrom=values[:, 1:],
        )

    def source_jvp(
        self, atoms: object, field: object, field_direction: object
    ) -> np.ndarray:
        count = atom_count(atoms)
        direction = self.field_space.validate(
            field_direction, atom_count=count, name="field_direction"
        )
        return self.source_space.validate(
            self._linearization(atoms, field).jvp(direction),
            atom_count=count,
            name="source_jvp",
        )

    def source_vjp(
        self, atoms: object, field: object, source_cotangent: object
    ) -> np.ndarray:
        count = atom_count(atoms)
        cotangent = self.source_space.validate(
            source_cotangent, atom_count=count, name="source_cotangent"
        )
        return self.field_space.validate(
            self._linearization(atoms, field).vjp(cotangent),
            atom_count=count,
            name="source_vjp",
        )

    def source_position_vjp(
        self, atoms: object, field: object, source_cotangent: object
    ) -> np.ndarray:
        self.configuration_sha256()
        self.domain.validate_atoms(atoms)
        count = atom_count(atoms)
        values = self.field_space.validate(field, atom_count=count)
        cotangent = self.source_space.validate(
            source_cotangent, atom_count=count, name="source_cotangent"
        )
        result = np.asarray(
            self._calculator.density_position_vjp(
                atoms,
                node_potential_ev=values[:, 0],
                node_gradient_ev_per_angstrom=values[:, 1:],
                density_cotangent=cotangent,
            ),
            dtype=float,
        )
        if result.shape != (count, 3) or not np.all(np.isfinite(result)):
            raise RuntimeError(
                "MACE-POLAR source position VJP must be finite with shape (N,3)."
            )
        return result.copy()


def build_official_mace_polar_1_m_adapter(
    *, device: str = "cpu"
) -> MACEPolarLocalFieldModelAdapter:
    """Load the exact official checkpoint in float64 without a solvent sidecar."""

    from maple.function.calculator.calculator_base import (
        _IMPLICIT_SOLVENT_FACTORY_TOKEN,
    )
    from maple.function.calculator.mace._macepol_calculator import (
        MACEPolCalculator,
    )

    class _VNextModelOnlyMACEPolCalculator(MACEPolCalculator):
        def implicit_solv_init(self, implicit: str, solvent: str) -> None:
            del implicit, solvent
            self.solvent_correction = None

    calculator = _VNextModelOnlyMACEPolCalculator(
        device=device,
        model="macepolm",
        implicit="smd",
        solvent="water",
        _implicit_solvent_factory_token=_IMPLICIT_SOLVENT_FACTORY_TOKEN,
    )
    return MACEPolarLocalFieldModelAdapter(calculator, OFFICIAL_MACE_POLAR_1_M_CONTRACT)


__all__ = [
    "MACEPolarExactGTOCompatibilityAudit",
    "MACEPolarLocalFieldModelAdapter",
    "MACEPolarReleaseContract",
    "OFFICIAL_MACE_POLAR_1_M_CONTRACT",
    "OFFICIAL_MACE_POLAR_MODEL_PROFILE_ID",
    "build_official_mace_polar_1_m_adapter",
]
