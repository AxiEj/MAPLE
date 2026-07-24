from __future__ import annotations

import ctypes
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import maple.function.calculator.extra_correction.implicit.pcmsolver as pcmsolver


class FakeFunction:
    def __init__(self, impl):
        self.impl = impl
        self.argtypes = None
        self.restype = None

    def __call__(self, *args):
        return self.impl(*args)


class FakePCMSolverCDLL:
    def __init__(self, *, compatible: bool = True, include_compute_asc: bool = True):
        self.deleted_contexts: list[int] = []
        self.last_new: dict[str, object] | None = None
        self.last_surface: np.ndarray | None = None
        self.raise_after_new = False
        self._context = 101
        self._cavity_size = 2
        self._centers = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float64)
        self._areas = np.array([0.5, 0.75], dtype=np.float64)
        self._asc = np.array([0.125, -0.250], dtype=np.float64)
        self._energy = 7.5
        self.energy_calls = 0

        self.pcmsolver_is_compatible_library = FakeFunction(lambda: compatible)
        self.pcmsolver_new_v1112 = FakeFunction(self._pcmsolver_new_v1112)
        self.pcmsolver_delete = FakeFunction(lambda ctx: self.deleted_contexts.append(int(ctx)))
        self.pcmsolver_get_cavity_size = FakeFunction(self._pcmsolver_get_cavity_size)
        self.pcmsolver_get_centers = FakeFunction(self._pcmsolver_get_centers)
        self.pcmsolver_get_areas = FakeFunction(self._pcmsolver_get_areas)
        self.pcmsolver_set_surface_function = FakeFunction(self._pcmsolver_set_surface_function)
        if include_compute_asc:
            self.pcmsolver_compute_asc = FakeFunction(lambda ctx, mep_name, asc_name, irrep: None)
        self.pcmsolver_compute_response_asc = FakeFunction(lambda ctx, mep_name, asc_name, irrep: None)
        self.pcmsolver_get_surface_function = FakeFunction(self._pcmsolver_get_surface_function)
        self.pcmsolver_compute_polarization_energy = FakeFunction(
            self._pcmsolver_compute_polarization_energy
        )

    def _pcmsolver_new_v1112(
        self,
        input_reading,
        nr_nuclei,
        charges,
        coordinates,
        symmetry_info,
        parsed_fname,
        host_input,
        writer,
    ):
        charges_array = np.ctypeslib.as_array(charges, shape=(nr_nuclei,)).copy()
        coordinates_array = np.ctypeslib.as_array(coordinates, shape=(3 * nr_nuclei,)).copy()
        symmetry_array = np.ctypeslib.as_array(symmetry_info, shape=(4,)).copy()
        writer(b"pcmsolver-started")
        self.last_new = {
            "input_reading": input_reading,
            "nr_nuclei": nr_nuclei,
            "charges": charges_array,
            "coordinates": coordinates_array,
            "symmetry": symmetry_array,
            "parsed_fname": parsed_fname,
        }
        return self._context

    def _pcmsolver_get_cavity_size(self, ctx):
        if self.raise_after_new:
            raise RuntimeError("broken cavity")
        return self._cavity_size

    def _pcmsolver_get_centers(self, ctx, buffer):
        np.ctypeslib.as_array(buffer, shape=(self._centers.size,))[:] = self._centers

    def _pcmsolver_get_areas(self, ctx, buffer):
        np.ctypeslib.as_array(buffer, shape=(self._areas.size,))[:] = self._areas

    def _pcmsolver_set_surface_function(self, ctx, size, values, name):
        self.last_surface = np.ctypeslib.as_array(values, shape=(size,)).copy()

    def _pcmsolver_get_surface_function(self, ctx, size, values, name):
        np.ctypeslib.as_array(values, shape=(size,))[:] = self._asc

    def _pcmsolver_compute_polarization_energy(self, ctx, mep_name, asc_name):
        self.energy_calls += 1
        return self._energy


@pytest.fixture
def parsed_input_file(tmp_path: Path) -> Path:
    path = tmp_path / "pcmsolver.parsed.inp"
    path.write_text("medium {}\n", encoding="utf-8")
    return path


def test_library_discovery_failure_reports_attempted_candidates(monkeypatch):
    monkeypatch.delenv("PCMSOLVER_LIBRARY", raising=False)
    monkeypatch.setattr(pcmsolver, "find_library", lambda name: None)

    def fake_cdll(candidate):
        raise OSError(f"cannot open {candidate}")

    monkeypatch.setattr(pcmsolver.ctypes, "CDLL", fake_cdll)

    with pytest.raises(pcmsolver.PCMSolverDiscoveryError, match="libpcm.so"):
        pcmsolver.load_pcmsolver_library()


def test_explicit_environment_library_is_authoritative_and_never_falls_back(
    monkeypatch,
):
    attempted = []
    monkeypatch.setenv("PCMSOLVER_LIBRARY", "/explicit/broken/libpcm.so")
    monkeypatch.setattr(pcmsolver, "find_library", lambda name: "/fallback/libpcm.so")

    def fake_cdll(candidate):
        attempted.append(candidate)
        raise OSError(f"cannot open {candidate}")

    monkeypatch.setattr(pcmsolver.ctypes, "CDLL", fake_cdll)

    with pytest.raises(
        pcmsolver.PCMSolverDiscoveryError,
        match="/explicit/broken/libpcm.so",
    ):
        pcmsolver.load_pcmsolver_library()
    assert attempted == ["/explicit/broken/libpcm.so"]


def test_library_abi_gate_rejects_missing_symbol(monkeypatch):
    monkeypatch.setattr(pcmsolver.ctypes, "CDLL", lambda candidate: FakePCMSolverCDLL(include_compute_asc=False))

    with pytest.raises(pcmsolver.PCMSolverABIError, match="pcmsolver_compute_asc"):
        pcmsolver.load_pcmsolver_library("/fake/libpcm.so")


def test_library_abi_gate_rejects_incompatible_runtime(monkeypatch):
    monkeypatch.setattr(pcmsolver.ctypes, "CDLL", lambda candidate: FakePCMSolverCDLL(compatible=False))

    with pytest.raises(pcmsolver.PCMSolverABIError, match="compatibility check"):
        pcmsolver.load_pcmsolver_library("/fake/libpcm.so")


def test_session_exposes_centers_areas_and_polarization_energy(monkeypatch, parsed_input_file):
    fake = FakePCMSolverCDLL()
    monkeypatch.setenv("PCMSOLVER_LIBRARY", "/env/libpcm.so")
    monkeypatch.setattr(pcmsolver, "find_library", lambda name: "/ignored/by/env.so")
    monkeypatch.setattr(pcmsolver.ctypes, "CDLL", lambda candidate: fake)

    atomic_numbers = np.array([8, 1], dtype=float)
    coordinates_bohr = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], dtype=float)
    with pcmsolver.PCMSolverSession(
        atomic_numbers,
        coordinates_bohr,
        parsed_input_file,
        symmetry_info=[0, 1, 2, 3],
    ) as session:
        assert fake.last_new is not None
        assert fake.last_new["input_reading"] == pcmsolver.PCMSOLVER_READER_OWN
        assert fake.last_new["nr_nuclei"] == 2
        assert np.allclose(fake.last_new["charges"], atomic_numbers)
        assert np.allclose(fake.last_new["coordinates"], [0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        assert np.array_equal(fake.last_new["symmetry"], np.array([0, 1, 2, 3], dtype=np.int32))
        assert fake.last_new["parsed_fname"] == bytes(parsed_input_file)

        assert np.allclose(session.cavity_centers_bohr, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        assert np.allclose(session.cavity_areas_bohr2, [0.5, 0.75])
        assert session.response_operator_is_symmetric is False

        assert np.allclose(session.compute_asc([10.0, -2.0]), [0.125, -0.250])
        assert fake.energy_calls == 0
        result = session.solve([10.0, -2.0])
        assert np.allclose(fake.last_surface, [10.0, -2.0])
        assert np.allclose(result["asc"], [0.125, -0.250])
        assert result["polarization_energy"] == pytest.approx(7.5)
        assert fake.energy_calls == 1

    assert fake.deleted_contexts == [101]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("True", True),
        ("False", False),
    ],
)
def test_session_reads_matrixsym_from_machine_input(
    tmp_path: Path,
    value: str,
    expected: bool,
):
    parsed = tmp_path / "pcmsolver.parsed.inp"
    parsed.write_text(
        "SECT MEDIUM 1 True\n"
        "TAG F KW 1\n"
        "BOOL MATRIXSYMM 1 True\n"
        f"{value}\n",
        encoding="utf-8",
    )
    session = pcmsolver.PCMSolverSession(
        [1],
        [[0.0, 0.0, 0.0]],
        parsed,
        library_path="/unused/libpcm.so",
    )

    assert session.response_operator_is_symmetric is expected


def test_session_cleans_up_if_open_fails_after_context_creation(monkeypatch, parsed_input_file):
    fake = FakePCMSolverCDLL()
    fake.raise_after_new = True
    monkeypatch.setattr(pcmsolver.ctypes, "CDLL", lambda candidate: fake)

    session = pcmsolver.PCMSolverSession([1], [[0.0, 0.0, 0.0]], parsed_input_file, library_path="/fake/libpcm.so")
    with pytest.raises(RuntimeError, match="broken cavity"):
        session.open()
    assert fake.deleted_contexts == [101]


def test_session_requires_existing_parsed_input(monkeypatch, tmp_path: Path):
    fake = FakePCMSolverCDLL()
    monkeypatch.setattr(pcmsolver.ctypes, "CDLL", lambda candidate: fake)

    missing = tmp_path / "missing.inp"
    session = pcmsolver.PCMSolverSession([1], [[0.0, 0.0, 0.0]], missing, library_path="/fake/libpcm.so")
    with pytest.raises(FileNotFoundError, match="parsed input file"):
        session.open()
    assert fake.deleted_contexts == []
