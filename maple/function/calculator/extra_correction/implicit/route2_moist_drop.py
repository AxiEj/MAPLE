"""Pinned MOIST adapter for source-dependent Route-2 DROP surfaces.

This module is deliberately a thin boundary.  MOIST owns reference-surface
construction, closest-point projection, DROP quadrature, the CPCM ``A``
matrix, and its surface adjoints.  MAPLE owns the reconstructed density level
set, Route-2 source ordering, and all fixed-point/energy logic.

The pinned upstream Python interface is pre-release software.  Runtime
provenance is therefore verified explicitly and the adapter never imports
``moist`` at module import time.  A synthetic backend can be admitted only by
an explicit test-only switch; it is never production evidence.
"""

from __future__ import annotations

import ctypes
from dataclasses import asdict, dataclass, field
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
from ase.units import Bohr

from .route2_density_levelset import (
    DensityLevelSetProvider,
    LSFAdjointWeights,
    SpatialLSFJet,
)

MOIST_DROP_ADAPTER_CONTRACT_VERSION = 2
MOIST_UPSTREAM_REPOSITORY = "https://github.com/lukaswittmann/moist"
MOIST_PINNED_COMMIT = "6e94f4a5841f5d2e1987e1ca496c7b9baad75e59"
MOIST_PINNED_SOURCE_VERSION = "0.6.0-alpha.1"
MOIST_PINNED_PYTHON_VERSION = "0.6.0"
MOIST_PINNED_C_API_VERSION = "0.6.0"
MOIST_PINNED_SHARED_LIBRARY_SONAME = "libmoist.so.0"
MOIST_PINNED_PYTHON_PACKAGE_SHA256 = (
    "a9a907902c42242b9f65df58bf41a11df56899e08bca3eb1fcb3a41589af9720"
)
MOIST_PINNED_IMPORT_PATCH_REPO_PATH = (
    "docs/implicit-solvation/patches/moist-6e94f4a-python-import.patch"
)
MOIST_PINNED_IMPORT_PATCH_SHA256 = (
    "96608ddc4135efd35324e593fdef343f7481a73564bd88d34d2aac7688ba9190"
)
MOIST_CALLBACK_SCALE = 1.0

# The public constructor at the pinned commit does not expose the parameter
# object's ``tolerance`` field.  Its compiled default is 1e-10.  Keeping that
# fact explicit prevents a 1e-10 runtime from being reported as a literature-
# parity 1e-12 calculation.
MOIST_PINNED_COMPILED_DROP_TOLERANCE = 1.0e-10

_MOIST_REDUCTION_TOTAL_RTOL = 1.0e-13
_MOIST_REDUCTION_TOTAL_ATOL = 1.0e-12


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_json(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _canonical_surface_geometry_totals(
    areas_bohr2: np.ndarray,
    points_bohr: np.ndarray,
    normals: np.ndarray,
) -> tuple[float, float]:
    """Return deterministic area and divergence-theorem volume totals."""

    areas = np.asarray(areas_bohr2, dtype=float)
    points = np.asarray(points_bohr, dtype=float)
    unit_normals = np.asarray(normals, dtype=float)
    area = math.fsum(float(value) for value in areas)
    volume = (
        math.fsum(
            float(weight)
            * (
                float(point[0]) * float(normal[0])
                + float(point[1]) * float(normal[1])
                + float(point[2]) * float(normal[2])
            )
            for weight, point, normal in zip(
                areas,
                points,
                unit_normals,
                strict=True,
            )
        )
        / 3.0
    )
    return area, volume


def _require_moist_reduction_total(
    raw_value: object,
    canonical_value: float,
    *,
    name: str,
) -> None:
    raw = float(raw_value)
    if not math.isfinite(raw) or not math.isclose(
        raw,
        canonical_value,
        rel_tol=_MOIST_REDUCTION_TOTAL_RTOL,
        abs_tol=_MOIST_REDUCTION_TOTAL_ATOL,
    ):
        raise RuntimeError(
            f"MOIST {name} is inconsistent with the canonical surface "
            f"quadrature (raw={raw!r}, canonical={canonical_value!r})."
        )


def _sha256_python_package(module_root: Path) -> str:
    """Hash the pinned runtime Python sources in a relocation-stable order."""

    digest = hashlib.sha256()
    for relative_path in ("__init__.py", "interface.py", "library.py"):
        path = module_root / relative_path
        if not path.is_file():
            raise RuntimeError(
                f"Imported MOIST package is missing pinned source {relative_path}."
            )
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _validated_sha256(value: object, *, name: str) -> str:
    text = str(value)
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise ValueError(f"{name} must be a lowercase SHA256 digest.")
    return text


def loaded_moist_shared_library_path() -> Path:
    """Resolve the already-loaded pinned MOIST SONAME through its C symbol."""

    no_load = getattr(os, "RTLD_NOLOAD", None)
    now = getattr(os, "RTLD_NOW", None)
    if no_load is None or now is None:
        raise RuntimeError(
            "This platform cannot attest the loaded MOIST shared-library path."
        )
    try:
        library = ctypes.CDLL(
            MOIST_PINNED_SHARED_LIBRARY_SONAME,
            mode=int(no_load) | int(now),
        )
        symbol = library.moist_get_version
        symbol_address = ctypes.cast(symbol, ctypes.c_void_p).value
    except (AttributeError, OSError) as exc:
        raise RuntimeError(
            "The imported extension is not bound to the pinned MOIST shared-library "
            "SONAME."
        ) from exc
    if symbol_address is None:
        raise RuntimeError("Cannot resolve the loaded MOIST version symbol.")

    class _DlInfo(ctypes.Structure):
        _fields_ = (
            ("dli_fname", ctypes.c_char_p),
            ("dli_fbase", ctypes.c_void_p),
            ("dli_sname", ctypes.c_char_p),
            ("dli_saddr", ctypes.c_void_p),
        )

    process = ctypes.CDLL(None)
    dladdr = getattr(process, "dladdr", None)
    if dladdr is None:
        raise RuntimeError("The platform loader does not expose dladdr().")
    dladdr.argtypes = (ctypes.c_void_p, ctypes.POINTER(_DlInfo))
    dladdr.restype = ctypes.c_int
    info = _DlInfo()
    if dladdr(ctypes.c_void_p(symbol_address), ctypes.byref(info)) == 0 or not (
        info.dli_fname
    ):
        raise RuntimeError("Cannot attest the loaded MOIST shared-library path.")
    return Path(os.fsdecode(info.dli_fname)).resolve()


def _readonly_array(
    values: object,
    *,
    name: str,
    shape: tuple[int, ...] | None = None,
    dtype: Any = float,
) -> np.ndarray:
    array = np.asarray(values, dtype=dtype)
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}; received {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    immutable = np.array(array, dtype=dtype, copy=True)
    immutable.setflags(write=False)
    return immutable


def _hash_arrays(payload: object, arrays: tuple[np.ndarray, ...]) -> str:
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
    )
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(contiguous.dtype.str.encode("ascii"))
        digest.update(json.dumps(contiguous.shape).encode("ascii"))
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def route2_source_state_sha256(values: object) -> str:
    """Content identity for one finite raw MACE-POLAR ``l<=1`` source."""

    source = np.asarray(values, dtype=float)
    if (
        source.ndim != 2
        or source.shape[0] == 0
        or source.shape[1] != 4
        or not np.all(np.isfinite(source))
    ):
        raise ValueError("Route-2 source identity requires shape (n_atoms, 4).")
    return _hash_arrays(
        {"source_order": "raw-MACE-POLAR-lte1", "unit": "e,e-angstrom"},
        (source,),
    )


@dataclass(frozen=True)
class MoistRuntimeProvenance:
    """Verifiable identity of one MOIST build used by the adapter."""

    evidence_kind: str
    upstream_commit: str = MOIST_PINNED_COMMIT
    source_version: str = MOIST_PINNED_SOURCE_VERSION
    python_package_version: str = MOIST_PINNED_PYTHON_VERSION
    c_api_version: str = MOIST_PINNED_C_API_VERSION
    import_patch_sha256: str = MOIST_PINNED_IMPORT_PATCH_SHA256
    python_extension_path: str | None = None
    python_extension_sha256: str | None = None
    shared_library_path: str | None = None
    shared_library_sha256: str | None = None
    build_toolchain: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.evidence_kind not in {
            "real-pinned-build",
            "synthetic-test-double",
        }:
            raise ValueError("Unsupported MOIST evidence kind.")
        if self.upstream_commit != MOIST_PINNED_COMMIT:
            raise ValueError("MOIST runtime does not match the pinned upstream commit.")
        if self.source_version != MOIST_PINNED_SOURCE_VERSION:
            raise ValueError("MOIST source version does not match the pinned profile.")
        if self.python_package_version != MOIST_PINNED_PYTHON_VERSION:
            raise ValueError("MOIST Python version does not match the pinned profile.")
        if self.c_api_version != MOIST_PINNED_C_API_VERSION:
            raise ValueError("MOIST C-API version does not match the pinned profile.")
        if self.import_patch_sha256 != MOIST_PINNED_IMPORT_PATCH_SHA256:
            raise ValueError("MOIST Python-import patch identity is not admitted.")
        if not isinstance(self.build_toolchain, tuple) or not all(
            isinstance(item, str) and item for item in self.build_toolchain
        ):
            raise ValueError("MOIST build toolchain entries must be nonempty strings.")

        path_hash_pairs = (
            (
                self.python_extension_path,
                self.python_extension_sha256,
                "MOIST Python extension",
            ),
            (
                self.shared_library_path,
                self.shared_library_sha256,
                "MOIST shared library",
            ),
        )
        if self.evidence_kind == "real-pinned-build":
            for raw_path, expected_hash, name in path_hash_pairs:
                if raw_path is None or expected_hash is None:
                    raise ValueError(f"{name} path and SHA256 are required.")
                expected = _validated_sha256(expected_hash, name=f"{name} SHA256")
                path = Path(raw_path).expanduser().resolve()
                if not path.is_file():
                    raise ValueError(f"{name} does not exist: {path}")
                observed = _sha256_file(path)
                if observed != expected:
                    raise ValueError(f"{name} SHA256 does not match provenance.")
        else:
            for raw_path, expected_hash, name in path_hash_pairs:
                if raw_path is not None or expected_hash is not None:
                    raise ValueError(
                        f"Synthetic {name} provenance cannot carry real artifact paths."
                    )

    @property
    def identity_sha256(self) -> str:
        payload = asdict(self)
        # Absolute install prefixes are audit metadata, not scientific content.
        # Binary hashes remain in the identity, so relocating byte-identical
        # artifacts does not perturb surface/forward-state fingerprints.
        payload.pop("python_extension_path")
        payload.pop("shared_library_path")
        payload["python_package_sha256"] = MOIST_PINNED_PYTHON_PACKAGE_SHA256
        return _sha256_json(payload)

    def verify_module(self, module: ModuleType | Any) -> None:
        if getattr(module, "__version__", None) != self.python_package_version:
            raise RuntimeError("Imported MOIST Python package version is not pinned.")
        library = getattr(module, "library", None)
        get_api_version = getattr(library, "get_api_version", None)
        if not callable(get_api_version) or get_api_version() != self.c_api_version:
            raise RuntimeError("Imported MOIST C-API version is not pinned.")
        for name in ("Structure", "IsodensityDROPCavity"):
            if not callable(getattr(module, name, None)):
                raise RuntimeError(f"Imported MOIST runtime is missing {name}.")

        if self.evidence_kind == "real-pinned-build":
            canonical_module = importlib.import_module("moist")
            canonical_interface = importlib.import_module("moist.interface")
            canonical_library = importlib.import_module("moist.library")
            if module is not canonical_module:
                raise RuntimeError(
                    "Real MOIST provenance forbids injected Python module objects."
                )
            if module.library is not canonical_library:
                raise RuntimeError("Imported MOIST library module is not canonical.")
            if (
                module.Structure is not canonical_interface.Structure
                or module.IsodensityDROPCavity
                is not canonical_interface.IsodensityDROPCavity
            ):
                raise RuntimeError("Imported MOIST runtime classes are not canonical.")
            extension_path = Path(str(self.python_extension_path)).resolve()
            module_root = Path(str(module.__file__)).resolve().parent
            if extension_path.parent != module_root:
                raise RuntimeError(
                    "MOIST extension provenance does not belong to the imported package."
                )
            extension_spec = importlib.util.find_spec(f"{module.__name__}._libmoist")
            loaded_origin = None if extension_spec is None else extension_spec.origin
            if loaded_origin is None or Path(loaded_origin).resolve() != extension_path:
                raise RuntimeError(
                    "Imported MOIST extension does not match the attested binary path."
                )
            observed_python_hash = _sha256_python_package(module_root)
            if observed_python_hash != MOIST_PINNED_PYTHON_PACKAGE_SHA256:
                raise RuntimeError(
                    "Imported MOIST Python package sources do not match the pinned "
                    "commit/patch."
                )
            shared_library_path = Path(str(self.shared_library_path)).resolve()
            loaded_library_path = loaded_moist_shared_library_path()
            if loaded_library_path != shared_library_path:
                raise RuntimeError(
                    "Loaded MOIST shared library does not match the attested path."
                )


@dataclass(frozen=True)
class MoistDropSettings:
    """Frozen public settings supported by the pinned callback constructor."""

    nleb: int = 194
    wleb_prune_level: int = 0
    callback_scale: float = MOIST_CALLBACK_SCALE
    compiled_drop_tolerance: float = MOIST_PINNED_COMPILED_DROP_TOLERANCE
    debug: bool = False
    verbosity: int = 0
    do_fine: bool = False
    surface_tolerance: float = 1.0e-9
    minimum_gradient_norm_bohr: float = 1.0e-10
    validation_shell_distance_bohr: float = 5.0e-2

    def __post_init__(self) -> None:
        if not isinstance(self.nleb, int) or self.nleb <= 0:
            raise ValueError("MOIST nleb must be a positive integer.")
        if not isinstance(self.wleb_prune_level, int) or not (
            0 <= self.wleb_prune_level <= 6
        ):
            raise ValueError("MOIST wleb_prune_level must be an integer from 0 to 6.")
        if float(self.callback_scale) != MOIST_CALLBACK_SCALE:
            raise ValueError(
                "The reconstructed level set is already scaled; MOIST scale must be 1.0."
            )
        if float(self.compiled_drop_tolerance) != (
            MOIST_PINNED_COMPILED_DROP_TOLERANCE
        ):
            raise ValueError(
                "The pinned public MOIST constructor cannot override its compiled "
                "DROP tolerance."
            )
        if not isinstance(self.debug, bool) or not isinstance(self.do_fine, bool):
            raise TypeError("MOIST debug/do_fine settings must be bool.")
        if not isinstance(self.verbosity, int) or self.verbosity < 0:
            raise ValueError("MOIST verbosity must be a nonnegative integer.")
        for name in (
            "surface_tolerance",
            "minimum_gradient_norm_bohr",
            "validation_shell_distance_bohr",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite.")

    @property
    def identity_sha256(self) -> str:
        return _sha256_json(asdict(self))


@dataclass(frozen=True)
class MoistSurfaceWeights:
    """DROP surface-response weights in MAPLE row-major point layout."""

    xi: np.ndarray
    switching: np.ndarray
    points_bohr: np.ndarray

    def __post_init__(self) -> None:
        xi = _readonly_array(self.xi, name="DROP xi weights")
        if xi.ndim != 1 or xi.size == 0:
            raise ValueError("DROP xi weights must be a nonempty vector.")
        switching = _readonly_array(
            self.switching,
            name="DROP switching weights",
            shape=xi.shape,
        )
        points = _readonly_array(
            self.points_bohr,
            name="DROP point weights",
            shape=(xi.size, 3),
        )
        object.__setattr__(self, "xi", xi)
        object.__setattr__(self, "switching", switching)
        object.__setattr__(self, "points_bohr", points)

    def plus(self, other: "MoistSurfaceWeights") -> "MoistSurfaceWeights":
        if not isinstance(other, MoistSurfaceWeights):
            raise TypeError("DROP surface weights can only be added to each other.")
        if other.xi.shape != self.xi.shape:
            raise ValueError("DROP surface-weight point counts do not match.")
        return MoistSurfaceWeights(
            self.xi + other.xi,
            self.switching + other.switching,
            self.points_bohr + other.points_bohr,
        )


@dataclass(frozen=True)
class MoistDropSurfaceSnapshot:
    """Immutable DROP surface/operator snapshot for one level-set state."""

    atomic_numbers: np.ndarray
    atom_positions_bohr: np.ndarray
    surface_points_bohr: np.ndarray
    surface_normals: np.ndarray
    reference_normals: np.ndarray
    surface_areas_bohr2: np.ndarray
    owner_atom_indices: np.ndarray
    converged: np.ndarray
    xi: np.ndarray
    switching_f: np.ndarray
    wleb: np.ndarray
    reference_displacements_bohr: np.ndarray
    branch_rho: np.ndarray
    amat: np.ndarray
    area_bohr2: float
    volume_bohr3: float
    geometry_sha256: str
    level_set_state_sha256: str
    parameter_sha256: str
    runtime_sha256: str
    bound_source_sha256: str | None = None
    snapshot_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        numbers = np.asarray(self.atomic_numbers)
        if numbers.ndim != 1 or numbers.size == 0:
            raise ValueError("DROP snapshot atomic numbers must be a nonempty vector.")
        if np.any(numbers < 1) or np.any(numbers != np.rint(numbers)):
            raise ValueError("DROP snapshot atomic numbers must be positive integers.")
        numbers = np.array(numbers, dtype=np.int64, copy=True)
        numbers.setflags(write=False)
        atom_count = numbers.size
        positions = _readonly_array(
            self.atom_positions_bohr,
            name="DROP atom positions",
            shape=(atom_count, 3),
        )
        points = _readonly_array(
            self.surface_points_bohr,
            name="DROP surface points",
        )
        if points.ndim != 2 or points.shape[0] == 0 or points.shape[1] != 3:
            raise ValueError("DROP surface points must have shape (n_surface, 3).")
        surface_size = points.shape[0]
        normals = _readonly_array(
            self.surface_normals,
            name="DROP surface normals",
            shape=(surface_size, 3),
        )
        normal_norms = np.linalg.norm(normals, axis=1)
        if not np.allclose(normal_norms, 1.0, rtol=0.0, atol=1.0e-10):
            raise ValueError("DROP surface normals must be unit vectors.")
        reference_normals = _readonly_array(
            self.reference_normals,
            name="DROP reference normals",
            shape=(surface_size, 3),
        )
        areas = _readonly_array(
            self.surface_areas_bohr2,
            name="DROP surface areas",
            shape=(surface_size,),
        )
        if np.any(areas < 0.0):
            raise ValueError("DROP surface areas cannot be negative.")
        owners = np.asarray(self.owner_atom_indices)
        if owners.shape != (surface_size,) or np.any(owners != np.rint(owners)):
            raise ValueError("DROP owner indices must be an integer surface vector.")
        owners = np.array(owners, dtype=np.int64, copy=True)
        if np.any(owners < 0) or np.any(owners >= atom_count):
            raise ValueError("DROP owner indices are outside the atom range.")
        owners.setflags(write=False)
        converged = np.asarray(self.converged, dtype=bool)
        if converged.shape != (surface_size,) or not np.all(converged):
            raise ValueError("Every DROP projection point must converge.")
        converged = np.array(converged, dtype=bool, copy=True)
        converged.setflags(write=False)

        vectors: dict[str, np.ndarray] = {}
        for name in (
            "xi",
            "switching_f",
            "wleb",
            "reference_displacements_bohr",
            "branch_rho",
        ):
            vectors[name] = _readonly_array(
                getattr(self, name),
                name=f"DROP {name}",
                shape=(surface_size,),
            )
        amat = _readonly_array(
            self.amat,
            name="DROP CPCM A-matrix",
            shape=(surface_size, surface_size),
        )
        if not np.allclose(amat, amat.T, rtol=0.0, atol=1.0e-12):
            raise ValueError("DROP CPCM A-matrix must be symmetric.")
        area = float(self.area_bohr2)
        volume = float(self.volume_bohr3)
        if not np.isfinite(area) or area <= 0.0:
            raise ValueError("DROP total area must be positive and finite.")
        if not np.isfinite(volume) or volume <= 0.0:
            raise ValueError("DROP total volume must be positive and finite.")
        identities = {
            "geometry_sha256": _validated_sha256(
                self.geometry_sha256,
                name="DROP geometry SHA256",
            ),
            "level_set_state_sha256": _validated_sha256(
                self.level_set_state_sha256,
                name="DROP level-set state SHA256",
            ),
            "parameter_sha256": _validated_sha256(
                self.parameter_sha256,
                name="DROP parameter SHA256",
            ),
            "runtime_sha256": _validated_sha256(
                self.runtime_sha256,
                name="DROP runtime SHA256",
            ),
        }
        bound_source_hash = self.bound_source_sha256
        if bound_source_hash is not None:
            bound_source_hash = _validated_sha256(
                bound_source_hash,
                name="DROP bound source SHA256",
            )

        object.__setattr__(self, "atomic_numbers", numbers)
        object.__setattr__(self, "atom_positions_bohr", positions)
        object.__setattr__(self, "surface_points_bohr", points)
        object.__setattr__(self, "surface_normals", normals)
        object.__setattr__(self, "reference_normals", reference_normals)
        object.__setattr__(self, "surface_areas_bohr2", areas)
        object.__setattr__(self, "owner_atom_indices", owners)
        object.__setattr__(self, "converged", converged)
        object.__setattr__(self, "amat", amat)
        object.__setattr__(self, "area_bohr2", area)
        object.__setattr__(self, "volume_bohr3", volume)
        for name, vector in vectors.items():
            object.__setattr__(self, name, vector)
        for name, identity in identities.items():
            object.__setattr__(self, name, identity)
        object.__setattr__(self, "bound_source_sha256", bound_source_hash)

        digest = _hash_arrays(
            {
                **identities,
                "bound_source_sha256": bound_source_hash,
                "area_bohr2": area,
                "volume_bohr3": volume,
                "contract_version": MOIST_DROP_ADAPTER_CONTRACT_VERSION,
            },
            (
                numbers,
                positions,
                points,
                normals,
                reference_normals,
                areas,
                owners,
                converged,
                vectors["xi"],
                vectors["switching_f"],
                vectors["wleb"],
                vectors["reference_displacements_bohr"],
                vectors["branch_rho"],
                amat,
            ),
        )
        object.__setattr__(self, "snapshot_sha256", digest)

    @property
    def atom_count(self) -> int:
        return int(self.atomic_numbers.size)

    @property
    def surface_size(self) -> int:
        return int(self.surface_points_bohr.shape[0])


@dataclass(frozen=True)
class MoistDropSurfaceState:
    """One source-bound DROP state plus its upstream adjoint contractions."""

    snapshot: MoistDropSurfaceSnapshot
    _cavity: Any = field(repr=False, compare=False)
    _level_set: DensityLevelSetProvider = field(repr=False, compare=False)
    _canonical_to_raw: np.ndarray | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    _raw_to_canonical: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        surface_size = self.snapshot.surface_size
        if self._canonical_to_raw is None:
            canonical_to_raw = np.arange(surface_size, dtype=np.int64)
        else:
            raw = np.asarray(self._canonical_to_raw)
            if (
                raw.shape != (surface_size,)
                or not np.all(np.isfinite(raw))
                or np.any(raw != np.rint(raw))
            ):
                raise ValueError("DROP canonical-to-raw order must be an index vector.")
            canonical_to_raw = np.asarray(raw, dtype=np.int64)
            if not np.array_equal(
                np.sort(canonical_to_raw),
                np.arange(surface_size, dtype=np.int64),
            ):
                raise ValueError("DROP canonical-to-raw order must be a permutation.")
        canonical_to_raw = np.array(canonical_to_raw, copy=True)
        raw_to_canonical = np.argsort(canonical_to_raw)
        canonical_to_raw.setflags(write=False)
        raw_to_canonical.setflags(write=False)
        object.__setattr__(self, "_canonical_to_raw", canonical_to_raw)
        object.__setattr__(self, "_raw_to_canonical", raw_to_canonical)

    def _surface_vector(self, values: object, *, name: str) -> np.ndarray:
        return _readonly_array(
            values,
            name=name,
            shape=(self.snapshot.surface_size,),
        )

    def contract_amat_surface_weights(
        self,
        left: object,
        right: object,
    ) -> MoistSurfaceWeights:
        q1 = self._surface_vector(left, name="left A-matrix contraction vector")
        q2 = self._surface_vector(right, name="right A-matrix contraction vector")
        raw = self._cavity.contract_amat_surface_weights(
            q1[self._raw_to_canonical],
            q2[self._raw_to_canonical],
        )
        if not isinstance(raw, tuple) or len(raw) != 3:
            raise RuntimeError(
                "MOIST returned an invalid A-matrix surface contraction."
            )
        w_xi, w_f, w_xyz = raw
        return MoistSurfaceWeights(
            xi=np.asarray(w_xi)[self._canonical_to_raw],
            switching=np.asarray(w_f)[self._canonical_to_raw],
            points_bohr=(np.asarray(w_xyz, dtype=float).T[self._canonical_to_raw]),
        )

    def contract_surface_lsf_weights(
        self,
        weights: MoistSurfaceWeights,
    ) -> LSFAdjointWeights:
        if not isinstance(weights, MoistSurfaceWeights):
            raise TypeError("MOIST LSF contraction requires MoistSurfaceWeights.")
        if weights.xi.shape != (self.snapshot.surface_size,):
            raise ValueError("DROP surface-weight point count does not match state.")
        raw = self._cavity.contract_surface_lsf_weights(
            weights.xi[self._raw_to_canonical],
            weights.switching[self._raw_to_canonical],
            weights.points_bohr[self._raw_to_canonical].T,
        )
        if not isinstance(raw, tuple) or len(raw) != 3:
            raise RuntimeError("MOIST returned an invalid surface-LSF contraction.")
        w0, w1, w2 = raw
        return LSFAdjointWeights(
            value=np.asarray(w0)[self._canonical_to_raw],
            gradient=(np.asarray(w1, dtype=float).T[self._canonical_to_raw]),
            hessian=(
                np.asarray(w2, dtype=float).transpose(2, 0, 1)[self._canonical_to_raw]
            ),
        )

    def source_vjp(self, weights: MoistSurfaceWeights) -> np.ndarray:
        lsf_weights = self.contract_surface_lsf_weights(weights)
        result = np.asarray(
            self._level_set.source_vjp(
                self.snapshot.surface_points_bohr,
                lsf_weights,
            ),
            dtype=float,
        )
        expected_shape = (self.snapshot.atom_count, 4)
        if result.shape != expected_shape or not np.all(np.isfinite(result)):
            raise RuntimeError(
                "Density level-set source VJP must be finite with shape "
                f"{expected_shape}."
            )
        return result

    def nuclear_field_vjp(self, weights: MoistSurfaceWeights) -> np.ndarray:
        """Contract only the explicit level-set field nuclear partial.

        The reference-anchor contribution is intentionally absent.  This
        method cannot be used as a complete reaction-map coordinate VJP.
        """

        lsf_weights = self.contract_surface_lsf_weights(weights)
        result = np.asarray(
            self._level_set.nuclear_vjp(
                self.snapshot.surface_points_bohr,
                lsf_weights,
            ),
            dtype=float,
        )
        expected_shape = (self.snapshot.atom_count, 3)
        if result.shape != expected_shape or not np.all(np.isfinite(result)):
            raise RuntimeError(
                "Density level-set nuclear VJP must be finite with shape "
                f"{expected_shape}."
            )
        return result


def reconstructed_level_set_state_sha256(
    level_set: DensityLevelSetProvider,
) -> str:
    """Hash the source/geometry/asset state of the production level set."""

    required = (
        "level_set_identity",
        "density_identity",
        "atomic_numbers",
        "positions_bohr",
        "density_coefficients",
        "n_iso_e_per_bohr3",
        "sigma_bohr",
        "expected_total_charge_e",
        "asset",
    )
    missing = [name for name in required if not hasattr(level_set, name)]
    if missing:
        raise TypeError(
            "Cannot fingerprint a generic density level set without an explicit "
            f"state hash; missing {', '.join(missing)}."
        )
    asset = getattr(level_set, "asset")
    require_content_integrity = getattr(asset, "require_content_integrity", None)
    if not callable(require_content_integrity):
        raise TypeError(
            "Cannot fingerprint a level set whose density asset has no content "
            "integrity contract."
        )
    require_content_integrity()
    payload = {
        "level_set_identity": str(getattr(level_set, "level_set_identity")),
        "density_identity": str(getattr(level_set, "density_identity")),
        "n_iso_e_per_bohr3": float(getattr(level_set, "n_iso_e_per_bohr3")),
        "sigma_bohr": float(getattr(level_set, "sigma_bohr")),
        "expected_total_charge_e": float(getattr(level_set, "expected_total_charge_e")),
        "asset_table_sha256": str(getattr(asset, "table_sha256")),
        "asset_manifest_sha256": str(getattr(asset, "manifest_sha256")),
        "asset_content_sha256": str(getattr(asset, "content_sha256")),
    }
    arrays = tuple(
        np.asarray(getattr(level_set, name))
        for name in ("atomic_numbers", "positions_bohr", "density_coefficients")
    )
    return _hash_arrays(payload, arrays)


class MoistDropAdapter:
    """Build one immutable MOIST DROP state from a MAPLE level-set provider."""

    contract_version = MOIST_DROP_ADAPTER_CONTRACT_VERSION
    source_dependent_geometry = True
    complete_position_derivative_available = False

    def __init__(
        self,
        level_set: DensityLevelSetProvider,
        atomic_numbers: object,
        atom_positions_angstrom: object,
        *,
        runtime: MoistRuntimeProvenance,
        settings: MoistDropSettings = MoistDropSettings(),
        level_set_state_sha256: str | None = None,
        moist_module: ModuleType | Any | None = None,
        _allow_synthetic_runtime: bool = False,
    ) -> None:
        if not isinstance(runtime, MoistRuntimeProvenance):
            raise TypeError("MOIST adapter requires verified runtime provenance.")
        if not isinstance(settings, MoistDropSettings):
            raise TypeError("MOIST adapter settings must be MoistDropSettings.")
        if runtime.evidence_kind == "synthetic-test-double" and not (
            _allow_synthetic_runtime
        ):
            raise RuntimeError(
                "Synthetic MOIST backends are test doubles, not production evidence."
            )
        if not isinstance(level_set, DensityLevelSetProvider):
            raise TypeError("MOIST adapter requires a DensityLevelSetProvider.")
        if not callable(getattr(level_set, "require_valid_boundary", None)):
            raise TypeError(
                "Production rho-DROP requires a level-set boundary admission check."
            )

        positions = np.asarray(atom_positions_angstrom, dtype=float)
        if (
            positions.ndim != 2
            or positions.shape[0] == 0
            or positions.shape[1] != 3
            or not np.all(np.isfinite(positions))
        ):
            raise ValueError("MOIST atom positions must have shape (n_atoms, 3).")
        numbers = np.asarray(atomic_numbers)
        if (
            numbers.shape != (positions.shape[0],)
            or not np.all(np.isfinite(numbers))
            or np.any(numbers != np.rint(numbers))
            or np.any(numbers < 1)
        ):
            raise ValueError("MOIST atomic numbers must be positive integers per atom.")
        if level_set.atom_count != positions.shape[0]:
            raise ValueError("MOIST geometry and level-set atom counts do not match.")

        module = moist_module
        if module is None:
            try:
                module = importlib.import_module("moist")
            except ImportError as exc:
                raise RuntimeError(
                    "Pinned MOIST Python runtime is unavailable; rho-DROP cannot "
                    "silently fall back to another cavity."
                ) from exc
        runtime.verify_module(module)

        if level_set_state_sha256 is None:
            state_hash = reconstructed_level_set_state_sha256(level_set)
        else:
            state_hash = _validated_sha256(
                level_set_state_sha256,
                name="rho-DROP level-set state SHA256",
            )
            try:
                observed_state_hash = reconstructed_level_set_state_sha256(level_set)
            except TypeError:
                # Generic protocol implementations may supply their own explicit
                # content identity.  The production reconstructed-density provider
                # is fingerprintable and must never take this branch.
                observed_state_hash = None
            if observed_state_hash is not None and observed_state_hash != state_hash:
                raise RuntimeError(
                    "Declared rho-DROP level-set state hash does not match the "
                    "callback provider."
                )
        numbers = np.array(numbers, dtype=np.int32, copy=True)
        numbers.setflags(write=False)
        positions = np.array(positions, dtype=float, copy=True)
        positions.setflags(write=False)
        self._level_set = level_set
        self._atomic_numbers = numbers
        self._positions_angstrom = positions
        self._runtime = runtime
        self._settings = settings
        self._level_set_state_sha256 = state_hash
        self._moist = module

    @property
    def runtime_provenance(self) -> MoistRuntimeProvenance:
        return self._runtime

    @property
    def settings(self) -> MoistDropSettings:
        return self._settings

    def build_surface(self) -> MoistDropSurfaceState:
        callback_error: list[Exception] = []

        def callback(point: np.ndarray):
            try:
                jet = self._level_set.evaluate_spatial(
                    np.asarray(point, dtype=float)[None, :]
                )
                if not isinstance(jet, SpatialLSFJet):
                    raise TypeError("Density level set returned a non-SpatialLSFJet.")
                return (
                    float(jet.value[0]),
                    jet.gradient[0],
                    jet.hessian[0],
                    jet.third[0],
                )
            except Exception as exc:  # CFFI callbacks cannot propagate directly.
                if not callback_error:
                    callback_error.append(exc)
                return (
                    float("nan"),
                    np.full(3, np.nan),
                    np.full((3, 3), np.nan),
                    np.full((3, 3, 3), np.nan),
                )

        structure = self._moist.Structure(
            self._atomic_numbers,
            self._positions_angstrom / Bohr,
        )
        cavity = self._moist.IsodensityDROPCavity(
            callback,
            nleb=self._settings.nleb,
            scale=MOIST_CALLBACK_SCALE,
            debug=self._settings.debug,
            verbosity=self._settings.verbosity,
            do_fine=self._settings.do_fine,
            wleb_prune_level=self._settings.wleb_prune_level,
        )
        try:
            cavity.update(structure)
        except Exception as exc:
            if callback_error:
                raise RuntimeError("rho-DROP level-set callback failed.") from (
                    callback_error[0]
                )
            raise RuntimeError("MOIST DROP projection failed.") from exc
        if callback_error:
            raise RuntimeError("rho-DROP level-set callback failed.") from (
                callback_error[0]
            )

        raw = cavity.cavity
        if int(raw.nsph) != self._atomic_numbers.size:
            raise RuntimeError("MOIST DROP sphere count does not match the molecule.")
        try:
            amat, xi = cavity.assemble_amat()
        except Exception as exc:
            raise RuntimeError("MOIST DROP CPCM A-matrix assembly failed.") from exc
        points = np.asarray(raw.xyz, dtype=float).T
        if int(raw.ngrid) != points.shape[0]:
            raise RuntimeError("MOIST DROP grid count does not match its point array.")
        jet = self._level_set.evaluate_spatial(points)
        gradient_norms = np.linalg.norm(jet.gradient, axis=1)
        if np.max(np.abs(jet.value)) > self._settings.surface_tolerance:
            raise RuntimeError(
                "MOIST DROP points do not satisfy the level-set surface."
            )
        if np.min(gradient_norms) < self._settings.minimum_gradient_norm_bohr:
            raise RuntimeError(
                "MOIST DROP surface contains a near-critical level-set point."
            )
        normals = jet.gradient / gradient_norms[:, None]
        shell_distance = self._settings.validation_shell_distance_bohr
        validation_shell = np.concatenate(
            (points - shell_distance * normals, points + shell_distance * normals),
            axis=0,
        )
        self._level_set.require_valid_boundary(points, validation_shell)

        owners_raw = np.asarray(raw.owner)
        if owners_raw.shape != (points.shape[0],):
            raise RuntimeError("MOIST DROP owner array has an invalid shape.")
        # OpenMP scheduling may permute otherwise identical surface points.
        # Canonicalize every surface-indexed quantity and both A axes here;
        # MoistDropSurfaceState retains the inverse mapping for native adjoints.
        canonical_to_raw = np.lexsort(
            (
                points[:, 2],
                points[:, 1],
                points[:, 0],
                owners_raw,
            )
        )
        points = points[canonical_to_raw]
        normals = normals[canonical_to_raw]
        amat = np.asarray(amat, dtype=float)[np.ix_(canonical_to_raw, canonical_to_raw)]

        positions_bohr = self._positions_angstrom / Bohr
        geometry_hash = _hash_arrays(
            {"coordinate_unit": "bohr", "atomic_number_dtype": "int32"},
            (self._atomic_numbers, positions_bohr),
        )
        canonical_areas = np.asarray(raw.a, dtype=float)[canonical_to_raw]
        canonical_area, canonical_volume = _canonical_surface_geometry_totals(
            canonical_areas,
            points,
            normals,
        )
        _require_moist_reduction_total(
            raw.area,
            canonical_area,
            name="total area",
        )
        _require_moist_reduction_total(
            raw.volume,
            canonical_volume,
            name="total volume",
        )
        snapshot = MoistDropSurfaceSnapshot(
            atomic_numbers=self._atomic_numbers,
            atom_positions_bohr=positions_bohr,
            surface_points_bohr=points,
            surface_normals=normals,
            reference_normals=(
                np.asarray(raw.normal0, dtype=float).T[canonical_to_raw]
            ),
            surface_areas_bohr2=canonical_areas,
            owner_atom_indices=owners_raw[canonical_to_raw],
            converged=np.asarray(raw.converged)[canonical_to_raw],
            xi=np.asarray(xi)[canonical_to_raw],
            switching_f=np.asarray(raw.f)[canonical_to_raw],
            wleb=np.asarray(raw.wleb)[canonical_to_raw],
            reference_displacements_bohr=(np.asarray(raw.r_iI0)[canonical_to_raw]),
            branch_rho=np.asarray(raw.rho)[canonical_to_raw],
            amat=amat,
            area_bohr2=canonical_area,
            volume_bohr3=canonical_volume,
            geometry_sha256=geometry_hash,
            level_set_state_sha256=self._level_set_state_sha256,
            parameter_sha256=self._settings.identity_sha256,
            runtime_sha256=self._runtime.identity_sha256,
            bound_source_sha256=(
                None
                if not hasattr(self._level_set, "density_coefficients")
                else route2_source_state_sha256(
                    getattr(self._level_set, "density_coefficients")
                )
            ),
        )
        return MoistDropSurfaceState(
            snapshot=snapshot,
            _cavity=cavity,
            _level_set=self._level_set,
            _canonical_to_raw=canonical_to_raw,
        )


__all__ = [
    "MOIST_CALLBACK_SCALE",
    "MOIST_DROP_ADAPTER_CONTRACT_VERSION",
    "MOIST_PINNED_C_API_VERSION",
    "MOIST_PINNED_COMMIT",
    "MOIST_PINNED_COMPILED_DROP_TOLERANCE",
    "MOIST_PINNED_IMPORT_PATCH_REPO_PATH",
    "MOIST_PINNED_IMPORT_PATCH_SHA256",
    "MOIST_PINNED_PYTHON_VERSION",
    "MOIST_PINNED_PYTHON_PACKAGE_SHA256",
    "MOIST_PINNED_SHARED_LIBRARY_SONAME",
    "MOIST_PINNED_SOURCE_VERSION",
    "MOIST_UPSTREAM_REPOSITORY",
    "MoistDropAdapter",
    "MoistDropSettings",
    "MoistDropSurfaceSnapshot",
    "MoistDropSurfaceState",
    "MoistRuntimeProvenance",
    "MoistSurfaceWeights",
    "loaded_moist_shared_library_path",
    "reconstructed_level_set_state_sha256",
    "route2_source_state_sha256",
]
