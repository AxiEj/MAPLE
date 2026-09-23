from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ase.units import Bohr

from maple.function.calculator.extra_correction.implicit.smd_cds import (
    smd_coulomb_radii,
)
from maple.solvation.api.units import HARTREE_TO_EV
import maple.solvation.continuum.torch_ddpcm as torch_ddpcm_module
from maple.solvation.continuum.torch_ddpcm import TorchDDPCM

POSITIONS = np.array([[0.0, 0.0, 0.0], [2.4, 0.2, -0.1]])
RADII = np.array([1.7, 1.5])
SOURCE = np.zeros((2, 8))
SOURCE[:, (0, 2, 3, 4)] = np.array([[0.4, 0.1, -0.1, 0.2], [-0.4, -0.05, 0.12, -0.18]])


def _functional(**kwargs) -> TorchDDPCM:
    return TorchDDPCM(
        ("C", "O"),
        RADII,
        dielectric=78.39,
        lmax=2,
        n_lebedev=50,
        max_dense_bytes=100_000_000,
        **kwargs,
    )


def _pyddx_oracle(
    positions: np.ndarray,
    source: np.ndarray,
    *,
    lmax: int = 2,
    n_lebedev: int = 50,
    radii: np.ndarray = RADII,
) -> tuple[float, np.ndarray]:
    pyddx = pytest.importorskip("pyddx")
    if pyddx.__version__ != "0.8.0":
        pytest.skip("oracle is pinned to pyddx 0.8.0")
    atom_count = len(source)
    multipoles = np.empty((4, atom_count))
    multipoles[0] = source[:, 0] / math.sqrt(4.0 * math.pi)
    multipoles[1:] = source[:, (2, 3, 4)].T / (Bohr * math.sqrt(4.0 * math.pi / 3.0))
    model = pyddx.Model(
        "pcm",
        (positions / Bohr).T,
        radii / Bohr,
        78.39,
        eta=0.1,
        shift=0.0,
        lmax=lmax,
        n_lebedev=n_lebedev,
        enable_fmm=False,
        n_proc=1,
        enable_force=True,
    )
    electrostatics = model.multipole_electrostatics(multipoles, derivative_order=1)
    state = pyddx.State(
        model,
        model.multipole_psi(multipoles),
        electrostatics["phi"],
        elec_field=electrostatics["e"],
    )
    state.fill_guess(1.0e-12)
    state.solve(1.0e-12)
    state.fill_guess_adjoint(1.0e-12)
    state.solve_adjoint(1.0e-12)
    gradient = (
        (
            np.asarray(state.solvation_force_terms(electrostatics))
            + np.asarray(state.multipole_force_terms(multipoles))
        ).T
        * HARTREE_TO_EV
        / Bohr
    )
    return state.energy() * HARTREE_TO_EV, gradient


def test_dense_ddpcm_energy_and_fixed_source_gradient_match_pyddx_0p8() -> None:
    functional = _functional()
    positions = torch.tensor(POSITIONS, dtype=torch.float64, requires_grad=True)
    source = torch.tensor(SOURCE, dtype=torch.float64, requires_grad=True)
    energy = functional.energy_torch(positions, source)
    (gradient,) = torch.autograd.grad(energy, positions)
    reference_energy, reference_gradient = _pyddx_oracle(POSITIONS, SOURCE)
    assert abs(float(energy.detach()) - reference_energy) <= 1.0e-12
    np.testing.assert_allclose(
        gradient.detach().numpy(), reference_gradient, rtol=0.0, atol=2.0e-11
    )


def test_registered_lmax15_lebedev1202_energy_and_gradient_match_pyddx() -> None:
    functional = TorchDDPCM(
        ("C", "O"),
        RADII,
        dielectric=78.39,
        lmax=15,
        n_lebedev=1202,
        max_dense_bytes=1_000_000_000,
    )
    positions = torch.tensor(POSITIONS, dtype=torch.float64, requires_grad=True)
    source = torch.tensor(SOURCE, dtype=torch.float64)
    energy = functional.energy_torch(positions, source)
    (gradient,) = torch.autograd.grad(energy, positions)
    reference_energy, reference_gradient = _pyddx_oracle(
        POSITIONS, SOURCE, lmax=15, n_lebedev=1202
    )
    assert abs(float(energy.detach()) - reference_energy) <= 1.0e-12
    np.testing.assert_allclose(
        gradient.detach().numpy(), reference_gradient, rtol=0.0, atol=2.0e-11
    )
    diagnostics = functional.diagnostics(positions.detach(), source)
    dimension = diagnostics["resource"]["basis_dimension"]
    for name in ("L_stability", "R_epsilon_stability"):
        stability = diagnostics[name]
        assert stability["numerical_rank"] == dimension
        assert (
            stability["reciprocal_condition"]
            > stability["minimum_reciprocal_condition"]
        )
        assert np.isfinite(stability["condition_number"])


def test_registered_water_matrices_remain_full_rank_and_match_pyddx() -> None:
    symbols = ("O", "H", "H")
    positions_numpy = np.array(
        [[0.0, 0.0, 0.0], [0.9572, 0.0, 0.0], [-0.239987, 0.927297, 0.0]]
    )
    source_numpy = np.zeros((3, 8))
    source_numpy[:, (0, 2, 3, 4)] = np.array(
        [[-0.8, 0.02, -0.01, 0.03], [0.4, -0.01, 0.0, 0.01], [0.4, 0.0, 0.01, -0.02]]
    )
    radii = smd_coulomb_radii(symbols, solvent="water")
    functional = TorchDDPCM(
        symbols,
        radii,
        dielectric=78.39,
        lmax=15,
        n_lebedev=1202,
        max_dense_bytes=1_000_000_000,
    )
    positions = torch.tensor(positions_numpy, dtype=torch.float64, requires_grad=True)
    source = torch.tensor(source_numpy, dtype=torch.float64)
    energy = functional.energy_torch(positions, source)
    (gradient,) = torch.autograd.grad(energy, positions)
    reference_energy, reference_gradient = _pyddx_oracle(
        positions_numpy,
        source_numpy,
        lmax=15,
        n_lebedev=1202,
        radii=radii,
    )
    assert abs(float(energy.detach()) - reference_energy) <= 1.0e-12
    np.testing.assert_allclose(
        gradient.detach().numpy(), reference_gradient, rtol=0.0, atol=2.0e-11
    )
    diagnostics = functional.diagnostics(positions.detach(), source)
    dimension = diagnostics["resource"]["basis_dimension"]
    assert diagnostics["L_stability"]["numerical_rank"] == dimension
    assert diagnostics["R_epsilon_stability"]["numerical_rank"] == dimension


def test_isolated_sphere_point_source_matches_pyddx() -> None:
    source_values = np.zeros((1, 8))
    source_values[0, (0, 2, 3, 4)] = (0.5, 0.1, -0.2, 0.3)
    functional = TorchDDPCM(
        ("C",),
        np.array([1.5]),
        dielectric=78.39,
        lmax=2,
        n_lebedev=50,
        max_dense_bytes=100_000_000,
    )
    positions = torch.zeros((1, 3), dtype=torch.float64)
    source = torch.tensor(source_values, dtype=torch.float64)
    value = float(functional.energy_torch(positions, source))

    pyddx = pytest.importorskip("pyddx")
    multipoles = np.empty((4, 1))
    multipoles[0] = source_values[:, 0] / math.sqrt(4.0 * math.pi)
    multipoles[1:] = source_values[:, (2, 3, 4)].T / (
        Bohr * math.sqrt(4.0 * math.pi / 3.0)
    )
    model = pyddx.Model(
        "pcm",
        np.zeros((3, 1)),
        np.array([1.5 / Bohr]),
        78.39,
        eta=0.1,
        shift=0.0,
        lmax=2,
        n_lebedev=50,
        enable_fmm=False,
    )
    state = pyddx.State(
        model,
        model.multipole_psi(multipoles),
        model.multipole_electrostatics(multipoles, derivative_order=0)["phi"],
    )
    state.fill_guess(1e-12)
    state.solve(1e-12)
    assert abs(value - state.energy() * HARTREE_TO_EV) <= 2e-12


def test_gradgrad_hessian_hvp_and_translation_symmetry() -> None:
    functional = _functional()
    positions = torch.tensor(POSITIONS, dtype=torch.float64, requires_grad=True)
    source = torch.tensor(SOURCE, dtype=torch.float64, requires_grad=True)

    def scalar(candidate):
        return functional.energy_torch(candidate, source)

    hessian = torch.autograd.functional.hessian(scalar, positions, vectorize=True)
    matrix = hessian.detach().numpy().reshape(6, 6)
    assert np.all(np.isfinite(matrix))
    np.testing.assert_allclose(matrix, matrix.T, rtol=0.0, atol=2e-10)
    direction = torch.tensor([[0.2, -0.3, 0.1], [-0.1, 0.4, -0.2]], dtype=torch.float64)
    _, hvp = torch.autograd.functional.hvp(scalar, positions, direction)
    np.testing.assert_allclose(
        hvp.detach().numpy().reshape(-1),
        matrix @ direction.numpy().reshape(-1),
        rtol=0.0,
        atol=2e-10,
    )
    gradient = torch.autograd.grad(scalar(positions), positions)[0]
    np.testing.assert_allclose(
        gradient.detach().numpy().sum(axis=0), np.zeros(3), atol=2e-12
    )


def test_hessian_matches_independent_fixed_topology_pyddx_gradient_difference() -> None:
    functional = _functional()
    positions = torch.tensor(POSITIONS, dtype=torch.float64)
    source = torch.tensor(SOURCE, dtype=torch.float64)
    hessian = (
        torch.autograd.functional.hessian(
            lambda candidate: functional.energy_torch(candidate, source),
            positions,
            vectorize=True,
        )
        .detach()
        .numpy()
        .reshape(6, 6)
    )
    step = 1.0e-4
    finite_difference = np.empty((6, 6))
    topology_hash = functional.diagnostics(positions, source)["topology"][
        "topology_sha256"
    ]
    for coordinate in range(6):
        plus = POSITIONS.copy().reshape(-1)
        minus = POSITIONS.copy().reshape(-1)
        plus[coordinate] += step
        minus[coordinate] -= step
        _, gradient_plus = _pyddx_oracle(plus.reshape(2, 3), SOURCE)
        _, gradient_minus = _pyddx_oracle(minus.reshape(2, 3), SOURCE)
        for displaced in (plus, minus):
            diagnostics = functional.diagnostics(
                torch.tensor(displaced.reshape(2, 3), dtype=torch.float64), source
            )
            assert diagnostics["topology"]["topology_sha256"] == topology_hash
        finite_difference[:, coordinate] = (
            gradient_plus.reshape(-1) - gradient_minus.reshape(-1)
        ) / (2.0 * step)
    np.testing.assert_allclose(hessian, finite_difference, rtol=0.0, atol=1.0e-5)


def test_source_layout_and_dense_resource_guard_fail_closed() -> None:
    positions = torch.tensor(POSITIONS, dtype=torch.float64)
    bad_source = torch.tensor(SOURCE, dtype=torch.float64)
    bad_source[0, 1] = 1e-30
    with pytest.raises(ValueError, match="exact zeros"):
        _functional().energy_torch(positions, bad_source)
    with pytest.raises(MemoryError, match="preflight rejected"):
        TorchDDPCM(
            ("C", "O"),
            RADII,
            dielectric=78.39,
            lmax=15,
            n_lebedev=50,
            max_dense_bytes=1,
        )


def test_exposed_radii_are_bytes_backed_and_cannot_be_reenabled_for_writes() -> None:
    functional = _functional()
    radii = functional.radii_angstrom
    assert not radii.flags.writeable
    assert not radii.flags.owndata
    with pytest.raises(ValueError, match="cannot set WRITEABLE flag"):
        radii.setflags(write=True)
    with pytest.raises(ValueError):
        radii[0] = 999.0
    np.testing.assert_array_equal(functional.radii_angstrom, RADII)


def test_configuration_binds_implementation_files_and_detects_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    functional = _functional()
    assert len(functional.configuration_sha256()) == 64
    original = torch_ddpcm_module._implementation_files_sha256()
    monkeypatch.setattr(
        torch_ddpcm_module,
        "_implementation_files_sha256",
        lambda: original + (("synthetic.drift", "0" * 64),),
    )
    with pytest.raises(RuntimeError, match="implementation configuration changed"):
        functional.configuration_sha256()
    with pytest.raises(RuntimeError, match="implementation configuration changed"):
        _ = functional.provenance_sha256


def test_derivative_order_resource_preflight_is_explicit() -> None:
    functional = TorchDDPCM(
        ("C", "O"),
        RADII,
        dielectric=78.39,
        lmax=15,
        n_lebedev=1202,
        max_dense_bytes=1_000_000_000,
    )
    energy = functional.estimate_resources(derivative_order=0)
    force = functional.preflight_resources(derivative_order=1)
    hessian = functional.preflight_resources(derivative_order=2)
    assert energy.conservative_peak_bytes == energy.forward_peak_bytes
    assert force.conservative_peak_bytes == force.first_order_peak_bytes
    assert hessian.conservative_peak_bytes == hessian.second_order_peak_bytes
    assert hessian.second_order_peak_bytes == 4 * force.first_order_peak_bytes
    assert (
        energy.conservative_peak_bytes
        < force.conservative_peak_bytes
        < hessian.conservative_peak_bytes
    )
    with pytest.raises(MemoryError, match="derivative order 2"):
        functional.preflight_resources(derivative_order=2, limit_bytes=1)
    expanded = functional.preflight_resources(
        derivative_order=2,
        limit_bytes=hessian.second_order_peak_bytes,
    )
    assert expanded.within_limit
    with pytest.raises(ValueError, match="exactly 0, 1, or 2"):
        functional.estimate_resources(derivative_order=3)


def test_tiny_solve_residual_does_not_admit_an_ill_conditioned_operator() -> None:
    matrix = torch.diag(torch.tensor([1.0, 1.0e-20], dtype=torch.float64))
    rhs = torch.tensor([1.0, 1.0e-20], dtype=torch.float64)
    solution = torch.linalg.solve(matrix, rhs)
    assert float(torch.linalg.vector_norm(matrix @ solution - rhs)) == 0.0
    with pytest.raises(RuntimeError, match="rank deficient or ill-conditioned"):
        TorchDDPCM._validate_matrix_stability("injected", matrix)


def test_topology_guard_accepts_buried_plateau_and_rejects_active_crossing() -> None:
    functional = _functional()
    # One fully buried plateau point: chi=1 has zero local derivatives and is
    # not a max(0, 1-fi) crossing despite fi being exactly one.
    ratios = torch.tensor(
        [[[1.0, 0.8], [1.0, 1.2]], [[1.2, 1.0], [1.2, 1.0]]], dtype=torch.float64
    )
    chi = torch.tensor(
        [[[0.0, 1.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]], dtype=torch.float64
    )
    fi = chi.sum(dim=-1)
    certificate = functional._certify_topology(
        ratios, chi, fi, torch.clamp(1 - fi, min=0)
    )
    assert certificate.buried_plateau_count == 1
    assert len(certificate.topology_sha256) == 64

    crossing = TorchDDPCM(
        ("H", "H", "H"),
        np.ones(3),
        dielectric=78.39,
        lmax=2,
        n_lebedev=50,
        max_dense_bytes=100_000_000,
    )
    ratios3 = torch.tensor([[[1.0, 1.0, 1.0]]] * 3, dtype=torch.float64)
    chi3 = torch.zeros((3, 1, 3), dtype=torch.float64)
    chi3[0, 0, 1:] = torch.tensor([0.4, 0.6], dtype=torch.float64)
    fi3 = chi3.sum(dim=-1)
    with pytest.raises(RuntimeError, match="branch-change margin"):
        crossing._certify_topology(ratios3, chi3, fi3, torch.clamp(1 - fi3, min=0))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_small_cpu_cuda_energy_gradient_parity() -> None:
    cpu = _functional(device="cpu")
    cuda = _functional(device="cuda")
    r_cpu = torch.tensor(POSITIONS, dtype=torch.float64, requires_grad=True)
    s_cpu = torch.tensor(SOURCE, dtype=torch.float64, requires_grad=True)
    r_cuda = r_cpu.detach().cuda().requires_grad_(True)
    s_cuda = s_cpu.detach().cuda().requires_grad_(True)
    e_cpu = cpu.energy_torch(r_cpu, s_cpu)
    e_cuda = cuda.energy_torch(r_cuda, s_cuda)
    g_cpu = torch.autograd.grad(e_cpu, (r_cpu, s_cpu))
    g_cuda = torch.autograd.grad(e_cuda, (r_cuda, s_cuda))
    assert abs(float(e_cpu.detach()) - float(e_cuda.detach().cpu())) <= 1e-10
    for left, right in zip(g_cpu, g_cuda, strict=True):
        np.testing.assert_allclose(
            left.detach().numpy(), right.detach().cpu().numpy(), atol=1e-9, rtol=0.0
        )
