from __future__ import annotations

from dataclasses import dataclass
import types

import numpy as np
import pytest
from ase.units import Bohr

from maple.function.calculator.extra_correction.implicit.pyscf_swig_response import (
    TESTED_PYSCF_VERSION,
    PySCFSWIGCPCMResponse,
    PySCFSWIGConductorResponse,
    PySCFSWIGCOSMOResponse,
    PySCFSWIGIEFPCMResponse,
    PySCFSWIGPCMResponse,
    _PySCFRuntime,
)


class _FakeMolecule:
    natm = 2
    nao = 2

    @staticmethod
    def atom_coords(unit="B"):
        assert unit == "B"
        return np.asarray([[-1.0, 0.0, 0.2], [1.1, 0.3, -0.2]])


class _FakeGTO:
    _charges = {"H": 1, "C": 6, "O": 8}

    @staticmethod
    def M(**_kwargs):
        return _FakeMolecule()

    @classmethod
    def charge(cls, symbol):
        if isinstance(symbol, int):
            return symbol
        return cls._charges[symbol]


class _FakePCMObject:
    def __init__(self, mol):
        self.mol = mol


class _FakePCM:
    PCM = _FakePCMObject
    XI = {6: 4.84566077868}
    last_surface_elements = None
    last_surface_radii = None

    @classmethod
    def gen_surface(cls, _mol, *, ng, rad, surface_discretization_method):
        assert ng == 6
        assert surface_discretization_method == "SWIG"
        np.testing.assert_allclose(
            _mol.atom_coords(unit="B"),
            _FakeMolecule.atom_coords(unit="B"),
        )
        cls.last_surface_elements = tuple(_mol.elements)
        cls.last_surface_radii = np.asarray(rad, dtype=float).copy()
        return {
            "grid_coords": np.asarray(
                [
                    [-2.0, 0.3, 0.1],
                    [1.8, -0.4, 0.6],
                    [2.3, 0.8, -0.2],
                    [-1.5, -1.2, 0.7],
                ]
            ),
            "gslice_by_atom": [[0, 2], [2, 4]],
        }

    @staticmethod
    def get_F_A(_surface):
        return np.ones(4), np.asarray([0.8, 0.9, 1.1, 1.2])

    @staticmethod
    def get_D_S(_surface, *, with_S, with_D):
        assert with_S is True
        assert with_D is True
        return (
            np.asarray(
                [
                    [0.0, 0.03, -0.01, 0.02],
                    [0.02, 0.0, 0.01, -0.02],
                    [-0.03, 0.02, 0.0, 0.01],
                    [0.01, -0.01, 0.02, 0.0],
                ]
            ),
            np.asarray(
                [
                    [1.4, 0.1, 0.03, -0.02],
                    [0.1, 1.5, -0.04, 0.02],
                    [0.03, -0.04, 1.6, 0.08],
                    [-0.02, 0.02, 0.08, 1.7],
                ]
            ),
        )


class _FakePCMGradient:
    _base = np.asarray([[1.0, -2.0, 0.5], [-0.7, 0.4, 1.3]])
    last_method = None

    @classmethod
    def grad_solver(cls, pcm_object, dm):
        assert dm.shape == (2, 2)
        cls.last_method = pcm_object.method
        potential = np.asarray(
            pcm_object._intermediates["v_grids"],
            dtype=float,
        )
        return float(np.dot(potential, potential)) * cls._base


@dataclass(frozen=True)
class _FakeRuntimeBundle:
    runtime: _PySCFRuntime
    pcm: type[_FakePCM]
    gradient: type[_FakePCMGradient]


@pytest.fixture
def fake_runtime():
    class RecordingPCM(_FakePCM):
        pass

    RecordingPCM.last_surface_elements = None
    RecordingPCM.last_surface_radii = None
    _FakePCMGradient.last_method = None
    return _FakeRuntimeBundle(
        runtime=_PySCFRuntime(
            version=TESTED_PYSCF_VERSION,
            gto=_FakeGTO,
            gen_grid=types.SimpleNamespace(LEBEDEV_ORDER={13: 74, 17: 6}),
            pcm=RecordingPCM,
            pcm_grad=_FakePCMGradient,
        ),
        pcm=RecordingPCM,
        gradient=_FakePCMGradient,
    )


def _response(fake_runtime):
    return PySCFSWIGIEFPCMResponse(
        ("H", "O"),
        np.asarray([[-0.7, 0.0, 0.1], [0.8, 0.2, -0.1]]),
        np.asarray([1.2, 1.5]),
        dielectric=78.39,
        lebedev_order=17,
        _runtime=fake_runtime.runtime,
    )


def test_pyscf_swig_response_uses_energy_conjugate_direct_and_adjoint_solve(
    fake_runtime,
):
    response = _response(fake_runtime)
    potential = np.asarray([0.2, -0.1, 0.3, -0.25])

    state = response.solve(potential)
    f_epsilon = (78.39 - 1.0) / (78.39 + 1.0)
    _, area = fake_runtime.pcm.get_F_A(None)
    D, S = fake_runtime.pcm.get_D_S(None, with_S=True, with_D=True)
    DA = D * area
    K = S - f_epsilon / (2.0 * np.pi) * (DA @ S)
    R = -f_epsilon * (np.eye(4) - DA / (2.0 * np.pi))
    direct = np.linalg.solve(K, R @ potential)
    adjoint = R.T @ np.linalg.solve(K.T, potential)
    conjugate = 0.5 * (direct + adjoint)

    np.testing.assert_allclose(
        state.direct_surface_charge_e,
        direct,
        rtol=2.0e-13,
        atol=2.0e-13,
    )
    np.testing.assert_allclose(
        state.adjoint_surface_charge_e,
        adjoint,
        rtol=2.0e-13,
        atol=2.0e-13,
    )
    np.testing.assert_allclose(
        state.energy_conjugate_surface_charge_e,
        conjugate,
        rtol=2.0e-13,
        atol=2.0e-13,
    )
    assert state.polarization_energy_hartree == pytest.approx(
        0.5 * float(np.dot(potential, conjugate)),
        rel=2.0e-13,
        abs=2.0e-13,
    )
    np.testing.assert_array_equal(
        response.surface_parent_atom_indices,
        np.asarray([0, 0, 1, 1]),
    )
    assert fake_runtime.pcm.last_surface_elements == (0, 1)
    np.testing.assert_allclose(
        fake_runtime.pcm.last_surface_radii,
        np.asarray([1.2, 1.5]) / Bohr,
    )
    assert (
        response.runtime_provenance["cavity_radius_assignment"]
        == "per-atom"
    )
    assert np.issubdtype(response.atomic_numbers.dtype, np.integer)


@pytest.mark.parametrize(
    ("response_type", "method", "f_epsilon"),
    (
        (
            PySCFSWIGCPCMResponse,
            "C-PCM",
            (4.0 - 1.0) / 4.0,
        ),
        (
            PySCFSWIGCOSMOResponse,
            "COSMO",
            (4.0 - 1.0) / (4.0 + 0.5),
        ),
    ),
)
def test_pyscf_swig_conductor_models_use_distinct_upstream_screening(
    fake_runtime,
    response_type,
    method,
    f_epsilon,
):
    response = response_type(
        ("H", "O"),
        np.asarray([[-0.7, 0.0, 0.1], [0.8, 0.2, -0.1]]),
        np.asarray([1.2, 1.5]),
        dielectric=4.0,
        lebedev_order=17,
        _runtime=fake_runtime.runtime,
    )
    potential = np.asarray([0.2, -0.1, 0.3, -0.25])
    _, S = fake_runtime.pcm.get_D_S(None, with_S=True, with_D=True)
    expected = np.linalg.solve(S, -f_epsilon * potential)

    state = response.solve(potential)

    np.testing.assert_allclose(
        state.direct_surface_charge_e,
        expected,
        rtol=2.0e-13,
        atol=2.0e-13,
    )
    np.testing.assert_allclose(
        state.adjoint_surface_charge_e,
        expected,
        rtol=2.0e-13,
        atol=2.0e-13,
    )
    np.testing.assert_allclose(
        state.energy_conjugate_surface_charge_e,
        expected,
        rtol=2.0e-13,
        atol=2.0e-13,
    )
    assert response.runtime_provenance["pyscf_pcm_method"] == method
    assert response.runtime_provenance["dielectric_scaling"] == pytest.approx(
        f_epsilon
    )

    response.operator_position_vjp(potential, -0.5 * potential)
    assert fake_runtime.gradient.last_method == method


def test_pyscf_swig_cpcm_and_cosmo_are_not_aliases(fake_runtime):
    common = dict(
        symbols=("H", "O"),
        atom_positions_angstrom=np.asarray(
            [[-0.7, 0.0, 0.1], [0.8, 0.2, -0.1]]
        ),
        cavity_radii_angstrom=np.asarray([1.2, 1.5]),
        dielectric=4.0,
        lebedev_order=17,
        _runtime=fake_runtime.runtime,
    )
    potential = np.asarray([0.2, -0.1, 0.3, -0.25])

    cpcm = PySCFSWIGCPCMResponse(**common).solve(potential)
    cosmo = PySCFSWIGCOSMOResponse(**common).solve(potential)

    assert cpcm.polarization_energy_hartree != pytest.approx(
        cosmo.polarization_energy_hartree,
        rel=1.0e-8,
        abs=1.0e-12,
    )


def test_pyscf_swig_conductor_response_has_unit_screening(fake_runtime):
    response = PySCFSWIGConductorResponse(
        ("H", "O"),
        np.asarray([[-0.7, 0.0, 0.1], [0.8, 0.2, -0.1]]),
        np.asarray([1.2, 1.5]),
        lebedev_order=17,
        _runtime=fake_runtime.runtime,
    )
    potential = np.asarray([0.2, -0.1, 0.3, -0.25])
    _, S = fake_runtime.pcm.get_D_S(None, with_S=True, with_D=True)

    state = response.solve(potential)

    np.testing.assert_allclose(
        state.direct_surface_charge_e,
        np.linalg.solve(S, -potential),
        rtol=2.0e-13,
        atol=2.0e-13,
    )
    assert response.runtime_provenance["continuum_model"] == (
        "conductor-limit-cpcm"
    )
    assert response.runtime_provenance["conductor_limit"] is True
    assert response.runtime_provenance["dielectric_scaling"] == 1.0


def test_pyscf_swig_generic_response_rejects_unknown_model(fake_runtime):
    with pytest.raises(ValueError, match="iefpcm, cpcm, or cosmo"):
        PySCFSWIGPCMResponse(
            ("H", "O"),
            np.asarray([[-0.7, 0.0, 0.1], [0.8, 0.2, -0.1]]),
            np.asarray([1.2, 1.5]),
            continuum_model="unknown",
            dielectric=78.39,
            lebedev_order=17,
            _runtime=fake_runtime.runtime,
        )


def test_pyscf_swig_response_rejects_grid_missing_upstream_swig_switching_data(
    fake_runtime,
):
    """A Lebedev grid alone is insufficient for PySCF's SWIG construction."""

    with pytest.raises(
        ValueError,
        match=r"Lebedev order 13: it maps to 74 points.*switching table",
    ):
        PySCFSWIGIEFPCMResponse(
            ("H", "O"),
            np.asarray([[-0.7, 0.0, 0.1], [0.8, 0.2, -0.1]]),
            np.asarray([1.2, 1.5]),
            dielectric=78.39,
            lebedev_order=13,
            _runtime=fake_runtime.runtime,
        )


def test_pyscf_swig_operator_bilinear_vjp_uses_polarization_identity(
    fake_runtime,
):
    response = _response(fake_runtime)
    left = np.asarray([0.2, -0.1, 0.3, -0.25])
    right = np.asarray([-0.3, 0.15, 0.05, 0.4])

    analytic = response.operator_position_vjp(left, right)
    expected = (
        2.0
        * float(np.dot(left, right))
        * fake_runtime.gradient._base
        / Bohr
    )

    np.testing.assert_allclose(
        analytic,
        expected,
        rtol=2.0e-13,
        atol=2.0e-13,
    )


def test_pyscf_swig_response_supports_mixed_same_element_radii(fake_runtime):
    response = PySCFSWIGIEFPCMResponse(
        ("O", "O"),
        np.asarray([[-0.7, 0.0, 0.1], [0.8, 0.2, -0.1]]),
        np.asarray([1.52, 1.70]),
        dielectric=78.39,
        lebedev_order=17,
        _runtime=fake_runtime.runtime,
    )

    np.testing.assert_array_equal(response.atomic_numbers, np.asarray([8, 8]))
    np.testing.assert_allclose(response.cavity_radii_angstrom, [1.52, 1.70])
    assert fake_runtime.pcm.last_surface_elements == (0, 1)
    np.testing.assert_allclose(
        fake_runtime.pcm.last_surface_radii,
        np.asarray([1.52, 1.70]) / Bohr,
    )


def test_pyscf_swig_response_rejects_changed_atom_index_lookup(fake_runtime):
    class RemappedIndexGTO(_FakeGTO):
        @classmethod
        def charge(cls, symbol):
            if isinstance(symbol, int):
                return symbol + 1
            return super().charge(symbol)

    runtime = _PySCFRuntime(
        version=fake_runtime.runtime.version,
        gto=RemappedIndexGTO,
        gen_grid=fake_runtime.runtime.gen_grid,
        pcm=fake_runtime.runtime.pcm,
        pcm_grad=fake_runtime.runtime.pcm_grad,
    )
    with pytest.raises(
        RuntimeError,
        match="no longer preserves atom indices",
    ):
        PySCFSWIGIEFPCMResponse(
            ("O", "O"),
            np.asarray([[-0.7, 0.0, 0.1], [0.8, 0.2, -0.1]]),
            np.asarray([1.52, 1.70]),
            dielectric=78.39,
            lebedev_order=17,
            _runtime=runtime,
        )


def test_pyscf_swig_response_rejects_untested_private_gradient_api(
    fake_runtime,
):
    runtime = _PySCFRuntime(
        version="2.14.0",
        gto=fake_runtime.runtime.gto,
        gen_grid=fake_runtime.runtime.gen_grid,
        pcm=fake_runtime.runtime.pcm,
        pcm_grad=fake_runtime.runtime.pcm_grad,
    )
    with pytest.raises(RuntimeError, match="tested only"):
        PySCFSWIGIEFPCMResponse(
            ("H", "O"),
            np.asarray([[-0.7, 0.0, 0.1], [0.8, 0.2, -0.1]]),
            np.asarray([1.2, 1.5]),
            dielectric=78.39,
            lebedev_order=17,
            _runtime=runtime,
        )


def test_pyscf_swig_response_rejects_noninteger_surface_slices(fake_runtime):
    class NonintegerSlicePCM(_FakePCM):
        @staticmethod
        def gen_surface(*args, **kwargs):
            surface = _FakePCM.gen_surface(*args, **kwargs)
            surface["gslice_by_atom"] = [[0.0, 2.0], [2.0, 4.0]]
            return surface

    runtime = _PySCFRuntime(
        version=TESTED_PYSCF_VERSION,
        gto=fake_runtime.runtime.gto,
        gen_grid=fake_runtime.runtime.gen_grid,
        pcm=NonintegerSlicePCM,
        pcm_grad=fake_runtime.runtime.pcm_grad,
    )
    with pytest.raises(RuntimeError, match="bounds must be integers"):
        PySCFSWIGIEFPCMResponse(
            ("H", "O"),
            np.asarray([[-0.7, 0.0, 0.1], [0.8, 0.2, -0.1]]),
            np.asarray([1.2, 1.5]),
            dielectric=78.39,
            lebedev_order=17,
            _runtime=runtime,
        )


def test_pyscf_swig_response_fails_closed_for_singular_operator(fake_runtime):
    class SingularPCM(_FakePCM):
        @staticmethod
        def get_D_S(_surface, *, with_S, with_D):
            assert with_S is True
            assert with_D is True
            return np.zeros((4, 4)), np.zeros((4, 4))

    runtime = _PySCFRuntime(
        version=TESTED_PYSCF_VERSION,
        gto=fake_runtime.runtime.gto,
        gen_grid=fake_runtime.runtime.gen_grid,
        pcm=SingularPCM,
        pcm_grad=fake_runtime.runtime.pcm_grad,
    )
    with pytest.raises(RuntimeError, match="factorization failed"):
        PySCFSWIGIEFPCMResponse(
            ("H", "O"),
            np.asarray([[-0.7, 0.0, 0.1], [0.8, 0.2, -0.1]]),
            np.asarray([1.2, 1.5]),
            dielectric=78.39,
            lebedev_order=17,
            _runtime=runtime,
        )
