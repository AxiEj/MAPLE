from __future__ import annotations

import ctypes
import os
from ctypes.util import find_library
from pathlib import Path
from typing import Callable

import numpy as np


PCMSOLVER_READER_OWN = 0
PCMSOLVER_READER_HOST = 1
_REQUIRED_SYMBOLS = (
    "pcmsolver_is_compatible_library",
    "pcmsolver_new_v1112",
    "pcmsolver_delete",
    "pcmsolver_get_cavity_size",
    "pcmsolver_get_centers",
    "pcmsolver_get_areas",
    "pcmsolver_set_surface_function",
    "pcmsolver_compute_asc",
    "pcmsolver_compute_response_asc",
    "pcmsolver_get_surface_function",
    "pcmsolver_compute_polarization_energy",
    "pcmsolver_print",
)
_DEFAULT_MEP_LABEL = b"MAPLE_MEP"
_DEFAULT_ASC_LABEL = b"MAPLE_ASC"
_STANDARD_LIBRARY_NAMES = ("libpcm.so", "libpcm.so.1", "pcm")


class PCMSolverError(RuntimeError):
    """Base error for PCMSolver binding failures."""


class PCMSolverDiscoveryError(PCMSolverError):
    """Raised when libpcm cannot be located or loaded."""


class PCMSolverABIError(PCMSolverError):
    """Raised when the loaded library does not match the expected API."""


class PCMInput(ctypes.Structure):
    """v1.1.12-style host-input structure from PCMSolver's public C header."""

    _fields_ = [
        ("cavity_type", ctypes.c_char * 8),
        ("patch_level", ctypes.c_int),
        ("coarsity", ctypes.c_double),
        ("area", ctypes.c_double),
        ("radii_set", ctypes.c_char * 9),
        ("min_distance", ctypes.c_double),
        ("der_order", ctypes.c_int),
        ("scaling", ctypes.c_bool),
        ("restart_name", ctypes.c_char * 20),
        ("min_radius", ctypes.c_double),
        ("solver_type", ctypes.c_char * 7),
        ("correction", ctypes.c_double),
        ("solvent", ctypes.c_char * 16),
        ("probe_radius", ctypes.c_double),
        ("equation_type", ctypes.c_char * 11),
        ("inside_type", ctypes.c_char * 7),
        ("outside_epsilon", ctypes.c_double),
        ("outside_type", ctypes.c_char * 22),
    ]

    @classmethod
    def defaults(cls) -> "PCMInput":
        host_input = cls()
        host_input.cavity_type = b"gepol"
        host_input.patch_level = 2
        host_input.coarsity = 0.5
        host_input.area = 0.2
        host_input.radii_set = b"bondi"
        host_input.min_distance = 0.1
        host_input.der_order = 4
        host_input.scaling = True
        host_input.restart_name = b"cavity.npz"
        host_input.min_radius = 100.0
        host_input.solver_type = b"iefpcm"
        host_input.correction = 0.0
        host_input.solvent = b"water"
        host_input.probe_radius = 1.0
        host_input.equation_type = b"secondkind"
        host_input.inside_type = b"vacuum"
        host_input.outside_epsilon = 1.0
        host_input.outside_type = b"uniformdielectric"
        return host_input


_HostWriter = ctypes.CFUNCTYPE(None, ctypes.c_char_p)


def _noop_host_writer(_message: bytes | None) -> None:
    return None


class _PCMSolverLibrary:
    def __init__(self, cdll: object, source: str):
        self.cdll = cdll
        self.source = source
        self._bind_required_symbols()
        self._verify_abi()

    def _bind_required_symbols(self) -> None:
        missing = [name for name in _REQUIRED_SYMBOLS if not hasattr(self.cdll, name)]
        if missing:
            joined = ", ".join(sorted(missing))
            raise PCMSolverABIError(
                f"Loaded PCMSolver library '{self.source}' is missing required symbols: {joined}. "
                "Use a libpcm.so built from the v1.1.12-style public C API."
            )

        self.pcmsolver_is_compatible_library = self.cdll.pcmsolver_is_compatible_library
        self.pcmsolver_is_compatible_library.argtypes = []
        self.pcmsolver_is_compatible_library.restype = ctypes.c_bool

        self.pcmsolver_new_v1112 = self.cdll.pcmsolver_new_v1112
        self.pcmsolver_new_v1112.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_char_p,
            ctypes.POINTER(PCMInput),
            _HostWriter,
        ]
        self.pcmsolver_new_v1112.restype = ctypes.c_void_p

        self.pcmsolver_delete = self.cdll.pcmsolver_delete
        self.pcmsolver_delete.argtypes = [ctypes.c_void_p]
        self.pcmsolver_delete.restype = None

        self.pcmsolver_get_cavity_size = self.cdll.pcmsolver_get_cavity_size
        self.pcmsolver_get_cavity_size.argtypes = [ctypes.c_void_p]
        self.pcmsolver_get_cavity_size.restype = ctypes.c_int

        self.pcmsolver_get_centers = self.cdll.pcmsolver_get_centers
        self.pcmsolver_get_centers.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double)]
        self.pcmsolver_get_centers.restype = None

        self.pcmsolver_get_areas = self.cdll.pcmsolver_get_areas
        self.pcmsolver_get_areas.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double)]
        self.pcmsolver_get_areas.restype = None

        self.pcmsolver_set_surface_function = self.cdll.pcmsolver_set_surface_function
        self.pcmsolver_set_surface_function.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_char_p,
        ]
        self.pcmsolver_set_surface_function.restype = None

        self.pcmsolver_compute_asc = self.cdll.pcmsolver_compute_asc
        self.pcmsolver_compute_asc.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
        self.pcmsolver_compute_asc.restype = None

        self.pcmsolver_compute_response_asc = self.cdll.pcmsolver_compute_response_asc
        self.pcmsolver_compute_response_asc.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_int,
        ]
        self.pcmsolver_compute_response_asc.restype = None

        self.pcmsolver_get_surface_function = self.cdll.pcmsolver_get_surface_function
        self.pcmsolver_get_surface_function.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_char_p,
        ]
        self.pcmsolver_get_surface_function.restype = None

        self.pcmsolver_compute_polarization_energy = self.cdll.pcmsolver_compute_polarization_energy
        self.pcmsolver_compute_polarization_energy.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_char_p,
        ]
        self.pcmsolver_compute_polarization_energy.restype = ctypes.c_double

        self.pcmsolver_print = self.cdll.pcmsolver_print
        self.pcmsolver_print.argtypes = [ctypes.c_void_p]
        self.pcmsolver_print.restype = None

    def _verify_abi(self) -> None:
        if not bool(self.pcmsolver_is_compatible_library()):
            raise PCMSolverABIError(
                f"PCMSolver library '{self.source}' rejected the header compatibility check. "
                "This usually means the runtime libpcm.so and the expected v1.1.12-style C API do not match."
            )


def _candidate_library_names() -> list[str]:
    candidates: list[str] = []
    explicit = os.environ.get("PCMSOLVER_LIBRARY")
    if explicit:
        candidates.append(explicit)
    discovered = find_library("pcm")
    if discovered:
        candidates.append(discovered)
    candidates.extend(_STANDARD_LIBRARY_NAMES)

    unique: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate and candidate not in seen:
            unique.append(candidate)
            seen.add(candidate)
    return unique


def load_pcmsolver_library(library_path: str | os.PathLike[str] | None = None) -> _PCMSolverLibrary:
    explicit_environment = os.environ.get("PCMSOLVER_LIBRARY")
    authoritative = library_path is not None or bool(explicit_environment)
    if library_path is not None:
        candidates = [os.fspath(library_path)]
    elif explicit_environment:
        candidates = [explicit_environment]
    else:
        candidates = _candidate_library_names()
    errors: list[str] = []
    for candidate in candidates:
        try:
            return _PCMSolverLibrary(ctypes.CDLL(candidate), candidate)
        except OSError as exc:
            errors.append(f"{candidate}: {exc}")
        except PCMSolverError as exc:
            if authoritative:
                raise
            errors.append(f"{candidate}: {exc}")

    attempted = ", ".join(candidates) if candidates else "<none>"
    raise PCMSolverDiscoveryError(
        "Unable to load PCMSolver libpcm.so. Checked PCMSOLVER_LIBRARY, ctypes.util.find_library('pcm'), "
        f"and standard sonames. Attempted: {attempted}. Details: {'; '.join(errors) if errors else 'no candidates found'}."
    )


class PCMSolverSession:
    """Minimal ctypes session for libpcm.so using pcmsolver_new_v1112 in READER_OWN mode."""

    def __init__(
        self,
        atomic_numbers: np.ndarray | list[float] | list[int],
        coordinates_bohr: np.ndarray | list[list[float]],
        parsed_input_path: str | os.PathLike[str],
        *,
        symmetry_info: np.ndarray | list[int] | tuple[int, int, int, int] | None = None,
        library_path: str | os.PathLike[str] | None = None,
        host_writer: Callable[[bytes | None], None] | None = None,
    ):
        self.atomic_numbers = self._normalize_atomic_numbers(atomic_numbers)
        self.coordinates_bohr = self._normalize_coordinates(coordinates_bohr, self.atomic_numbers.size)
        self.parsed_input_path = Path(parsed_input_path)
        self.symmetry_info = self._normalize_symmetry(symmetry_info)
        self.library_path = library_path
        self._host_messages: list[str] = []
        downstream_writer = host_writer or _noop_host_writer

        def capture_host_message(message: bytes | None) -> None:
            if message is not None:
                self._host_messages.append(
                    message.decode("utf-8", errors="replace")
                )
            downstream_writer(message)

        self._writer_callback = _HostWriter(capture_host_message)
        self._library: _PCMSolverLibrary | None = None
        self._context: int | None = None
        self._input = PCMInput.defaults()
        self._cavity_centers_bohr: np.ndarray | None = None
        self._cavity_areas_bohr2: np.ndarray | None = None

    @staticmethod
    def _normalize_atomic_numbers(atomic_numbers: np.ndarray | list[float] | list[int]) -> np.ndarray:
        values = np.asarray(atomic_numbers, dtype=np.float64)
        if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
            raise ValueError("atomic_numbers must be a one-dimensional finite array.")
        return np.ascontiguousarray(values)

    @staticmethod
    def _normalize_coordinates(coordinates_bohr: np.ndarray | list[list[float]], natoms: int) -> np.ndarray:
        coords = np.asarray(coordinates_bohr, dtype=np.float64)
        if coords.shape != (natoms, 3) or not np.isfinite(coords).all():
            raise ValueError(
                "coordinates_bohr must be a finite array with shape (n_atoms, 3) in Bohr."
            )
        return np.ascontiguousarray(coords)

    @staticmethod
    def _normalize_symmetry(
        symmetry_info: np.ndarray | list[int] | tuple[int, int, int, int] | None,
    ) -> np.ndarray:
        if symmetry_info is None:
            values = np.zeros(4, dtype=np.int32)
        else:
            values = np.asarray(symmetry_info, dtype=np.int32)
        if values.shape != (4,):
            raise ValueError("symmetry_info must contain exactly four integers.")
        return np.ascontiguousarray(values)

    def __enter__(self) -> "PCMSolverSession":
        return self.open()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    @property
    def cavity_centers_bohr(self) -> np.ndarray:
        self._require_open()
        assert self._cavity_centers_bohr is not None
        return self._cavity_centers_bohr.copy()

    @property
    def cavity_areas_bohr2(self) -> np.ndarray:
        self._require_open()
        assert self._cavity_areas_bohr2 is not None
        return self._cavity_areas_bohr2.copy()

    @property
    def cavity_size(self) -> int:
        self._require_open()
        assert self._cavity_areas_bohr2 is not None
        return int(self._cavity_areas_bohr2.size)

    @property
    def library_source(self) -> str:
        self._require_open()
        assert self._library is not None
        return self._library.source

    @property
    def response_operator_is_symmetric(self) -> bool:
        """Return the parsed ``MATRIXSYMM`` setting used by PCMSolver."""

        if not self.parsed_input_path.is_file():
            return False
        lines = self.parsed_input_path.read_text(
            encoding="utf-8",
            errors="replace",
        ).splitlines()
        for index, line in enumerate(lines):
            fields = line.strip().split()
            if len(fields) < 2 or fields[0] != "BOOL":
                continue
            if fields[1].upper() != "MATRIXSYMM":
                continue
            for value in lines[index + 1 :]:
                stripped = value.strip()
                if stripped:
                    return stripped.lower() == "true"
        return False

    def render_info(self) -> str:
        """Ask the pinned runtime to emit its effective cavity/medium report."""

        self._require_open()
        assert self._library is not None and self._context is not None
        self._host_messages.clear()
        self._library.pcmsolver_print(self._context)
        return "".join(self._host_messages)

    def open(self) -> "PCMSolverSession":
        if self._context is not None:
            return self
        if not self.parsed_input_path.is_file():
            raise FileNotFoundError(
                f"PCMSolver parsed input file not found: {self.parsed_input_path}. "
                "Generate the parsed PCMSolver input first and pass its path here."
            )

        self._library = load_pcmsolver_library(self.library_path)
        flattened_coordinates = np.ascontiguousarray(self.coordinates_bohr.reshape(-1))
        parsed_bytes = os.fsencode(self.parsed_input_path)
        try:
            context = self._library.pcmsolver_new_v1112(
                PCMSOLVER_READER_OWN,
                int(self.atomic_numbers.size),
                self.atomic_numbers.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                flattened_coordinates.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                self.symmetry_info.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                parsed_bytes,
                ctypes.byref(self._input),
                self._writer_callback,
            )
            if not context:
                raise PCMSolverError(
                    "pcmsolver_new_v1112 returned a null context. Check that the parsed input file is valid "
                    "for the installed libpcm.so."
                )
            self._context = int(context)
            self._cache_cavity_geometry()
            return self
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        if self._context is not None and self._library is not None:
            self._library.pcmsolver_delete(self._context)
        self._context = None
        self._cavity_centers_bohr = None
        self._cavity_areas_bohr2 = None

    def _require_open(self) -> None:
        if self._context is None or self._library is None:
            raise PCMSolverError("PCMSolver session is not open. Use it as a context manager or call open().")

    def _cache_cavity_geometry(self) -> None:
        self._require_open()
        assert self._library is not None and self._context is not None
        size = int(self._library.pcmsolver_get_cavity_size(self._context))
        if size <= 0:
            raise PCMSolverError(
                f"PCMSolver returned a non-positive cavity size ({size}). Check the parsed input and cavity settings."
            )
        flat_centers = np.zeros(3 * size, dtype=np.float64)
        areas = np.zeros(size, dtype=np.float64)
        self._library.pcmsolver_get_centers(
            self._context,
            flat_centers.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
        self._library.pcmsolver_get_areas(
            self._context,
            areas.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
        self._cavity_centers_bohr = np.array(flat_centers.reshape((3, size), order="F").T, copy=True)
        self._cavity_areas_bohr2 = np.array(areas, copy=True)

    def compute_asc(
        self,
        mep: np.ndarray | list[float],
        *,
        irrep: int = 0,
        mep_label: bytes | str = _DEFAULT_MEP_LABEL,
        asc_label: bytes | str = _DEFAULT_ASC_LABEL,
    ) -> np.ndarray:
        """Apply the static PCMSolver MEP-to-ASC response operator."""

        self._require_open()
        assert self._library is not None and self._context is not None
        if self._cavity_areas_bohr2 is None:
            raise PCMSolverError("Cavity geometry was not initialized.")

        mep_values = np.asarray(mep, dtype=np.float64)
        if mep_values.shape != (self._cavity_areas_bohr2.size,) or not np.isfinite(mep_values).all():
            raise ValueError(
                f"mep must be a finite one-dimensional array with length {self._cavity_areas_bohr2.size}."
            )
        mep_values = np.ascontiguousarray(mep_values)
        asc = np.zeros_like(mep_values)
        mep_name = mep_label.encode("utf-8") if isinstance(mep_label, str) else mep_label
        asc_name = asc_label.encode("utf-8") if isinstance(asc_label, str) else asc_label

        self._library.pcmsolver_set_surface_function(
            self._context,
            int(mep_values.size),
            mep_values.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            mep_name,
        )
        self._library.pcmsolver_compute_asc(self._context, mep_name, asc_name, int(irrep))
        self._library.pcmsolver_get_surface_function(
            self._context,
            int(asc.size),
            asc.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            asc_name,
        )
        return asc

    def solve(
        self,
        mep: np.ndarray | list[float],
        *,
        irrep: int = 0,
        mep_label: bytes | str = _DEFAULT_MEP_LABEL,
        asc_label: bytes | str = _DEFAULT_ASC_LABEL,
    ) -> dict[str, np.ndarray | float]:
        """Return static ASC plus its polarization energy."""

        mep_values = np.asarray(mep, dtype=np.float64)
        asc = self.compute_asc(
            mep_values,
            irrep=irrep,
            mep_label=mep_label,
            asc_label=asc_label,
        )
        self._require_open()
        assert self._library is not None and self._context is not None
        mep_values = np.ascontiguousarray(mep_values)
        mep_name = mep_label.encode("utf-8") if isinstance(mep_label, str) else mep_label
        asc_name = asc_label.encode("utf-8") if isinstance(asc_label, str) else asc_label
        energy = float(
            self._library.pcmsolver_compute_polarization_energy(
                self._context,
                mep_name,
                asc_name,
            )
        )
        return {"asc": asc, "polarization_energy": energy}


__all__ = [
    "PCMInput",
    "PCMSOLVER_READER_HOST",
    "PCMSOLVER_READER_OWN",
    "PCMSolverABIError",
    "PCMSolverDiscoveryError",
    "PCMSolverError",
    "PCMSolverSession",
    "load_pcmsolver_library",
]
