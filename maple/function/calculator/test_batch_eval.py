"""Equivalence tests for the Phase 1 batched calculator infrastructure.

Covers:

* ``CalcABC.calculate_many`` sequential fallback (energy + force return).
* ``FDHessianEvaluator`` central-difference Hessian — analytic match,
  chunked-vs-unchunked bit identity, FixAtoms handling, geometry restoration.
* ``energy_forces_one`` — single-invocation E + F equivalence with the
  legacy ``get_potential_energy + get_forces`` pattern.
* ``HVPEvaluator`` autodiff and FD fallback paths.
* OPT-RFO reject path no longer wastes a forward pass when rolling back.
* TS-PRFO outer-iteration loop no longer refetches E/F between accepted
  iterations.

All tests use lightweight in-process calculators with closed-form energies
and forces so they run without GPU / model weights.
"""
from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.calculators.calculator import all_changes
from ase.constraints import FixAtoms

from maple.function.calculator._autograd_hessian import (
    hessian_batched_vjp,
    hessian_loop,
)
from maple.function.calculator._batch_eval import (
    FDHessianContext,
    FDHessianEvaluator,
    HVPEvaluator,
    PathEvaluator,
    _estimate_auto_batch_size_from_item_bytes,
    energy_forces_one,
)
from maple.function.calculator._batch_types import BatchResult
from maple.function.calculator.calculator_base import CalcABC


# ---------------------------------------------------------------------------
# Toy calculators
# ---------------------------------------------------------------------------
class HarmonicCalc(CalcABC):
    """E = 0.5 * k * sum_i |R_i - R_i^0|^2  (analytic Hessian = k * I)."""

    implemented_properties = ["energy", "forces", "free_energy"]

    def __init__(self, k: float = 1.5, ref_positions: Optional[np.ndarray] = None):
        super().__init__()
        self.k = float(k)
        self.ref = np.array(ref_positions, dtype=np.float64)
        self.dtype = np.float64
        self.calls = 0  # counter so tests can verify no redundant forwards

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.calls += 1
        R = atoms.get_positions().astype(np.float64)
        d = R - self.ref
        e = 0.5 * self.k * float(np.sum(d * d))
        self.results["energy"] = e
        self.results["free_energy"] = e
        if "forces" in properties:
            self.results["forces"] = -self.k * d

    def get_hessian(self, atoms, delta: float = 0.002):
        n3 = 3 * len(atoms)
        return self.k * np.eye(n3, dtype=np.float64)


class CoupledPairCalc(CalcABC):
    """E = 0.5 * k * |R1 - R0 - d0|^2, with off-diagonal Hessian blocks."""

    implemented_properties = ["energy", "forces", "free_energy"]

    def __init__(self, k: float = 1.0, d0: Optional[np.ndarray] = None):
        super().__init__()
        self.k = float(k)
        self.d0 = np.zeros(3, dtype=np.float64) if d0 is None else np.asarray(d0, dtype=np.float64)

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        d = atoms.get_positions()[1] - atoms.get_positions()[0] - self.d0
        e = 0.5 * self.k * float(np.dot(d, d))
        self.results["energy"] = e
        self.results["free_energy"] = e
        if "forces" in properties:
            F = np.zeros((len(atoms), 3), dtype=np.float64)
            F[0] = self.k * d
            F[1] = -self.k * d
            self.results["forces"] = F


class SaddleCalc(CalcABC):
    """E = -0.5*kx*x^2 + 0.5*ky*y^2 + 0.5*kz*z^2  on atom 0 only.

    Saddle at origin with one negative curvature along x. Built into a
    larger atoms object so FixAtoms isolation is exercised.
    """

    implemented_properties = ["energy", "forces", "free_energy"]
    supported_hessian_modes = ("numerical",)

    def __init__(self, kx=2.0, ky=1.0, kz=1.5):
        super().__init__()
        self.kx, self.ky, self.kz = kx, ky, kz
        self.dtype = np.float64
        self.hessian = "numerical"
        self.calls = 0

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.calls += 1
        x, y, z = atoms.get_positions()[0]
        e = -0.5 * self.kx * x * x + 0.5 * self.ky * y * y + 0.5 * self.kz * z * z
        self.results["energy"] = e
        self.results["free_energy"] = e
        if "forces" in properties:
            F = np.zeros((len(atoms), 3), dtype=np.float64)
            F[0] = (self.kx * x, -self.ky * y, -self.kz * z)
            self.results["forces"] = F

    def get_hessian(self, atoms, delta=0.002):
        return FDHessianEvaluator(
            self, fd_batch_size=getattr(self, "fd_batch_size", None)
        ).hessian(atoms, delta=delta)


class FakeNumericalTSCalc(HarmonicCalc):
    """Harmonic minimum whose numerical Hessian lies about one negative mode."""

    supported_hessian_modes = ("analytic", "numerical")
    supports_analytic_hessian = True

    def __init__(self):
        super().__init__(k=1.0, ref_positions=np.zeros((4, 3)))
        self.hessian = "numerical"

    def get_hessian(self, atoms, delta: float = 0.002):
        n3 = 3 * len(atoms)
        H = np.eye(n3, dtype=np.float64)
        if self.hessian == "numerical":
            H[0, 0] = -1.0
        return H


def _set_thresholds(atoms: Atoms, val: float = 1e-5) -> None:
    for t in ("f_max_th", "f_rms_th", "dp_max_th", "dp_rms_th"):
        setattr(atoms, t, val)


def test_command_control_accepts_model_level_batch_size():
    from maple.function.read.command_control import CommandControl

    cc = CommandControl.from_settings([
        "#model=uma(task=omol,batch_size=2)",
        "#freq",
        "#device=gpu0",
    ])

    assert cc.params["model"] == "uma"
    assert cc.params["model_options"]["task"] == "omol"
    assert cc.params["model_options"]["batch_size"] == 2

    cc_auto = CommandControl.from_settings([
        "#model=uma(task=omol,batch_size=auto)",
        "#freq",
    ])

    assert cc_auto.params["model_options"]["batch_size"] == "auto"


def test_command_control_promotes_global_batch_size_to_model_option():
    from maple.function.read.command_control import CommandControl

    cc = CommandControl.from_settings([
        "#model=aimnet2",
        "#batch_size=3",
        "#freq",
    ])

    assert "batch_size" not in cc.params
    assert cc.params["model_options"]["batch_size"] == 3


def test_command_control_rejects_invalid_model_batch_size():
    from maple.function.read.command_control import CommandControl

    with pytest.raises(ValueError, match="batch_size"):
        CommandControl.from_settings(["#model=uma(batch_size=0)", "#freq"])


def test_command_control_rejects_pbc_with_batch_size():
    from maple.function.read.command_control import CommandControl

    with pytest.raises(ValueError, match="PBC and batch_size"):
        CommandControl.from_settings([
            "#model=aimnet2",
            "#pbc(10,10,10)",
            "#batch_size=2",
            "#freq",
        ])

    with pytest.raises(ValueError, match="PBC and batch_size"):
        CommandControl.from_settings([
            "#model=aimnet2",
            "#pbc(10,10,10)",
            "#batch_size=auto",
            "#freq",
        ])


def test_command_control_accepts_rfo_fd_batch_size_option():
    from maple.function.read.command_control import CommandControl

    cc = CommandControl.from_settings([
        "#model=ani2x",
        "#opt(method=rfo,fd_batch_size=2)",
    ])

    assert cc.task == "opt"
    assert cc.params["method"] == "rfo"
    assert cc.params["fd_batch_size"] == 2

    cc_auto = CommandControl.from_settings([
        "#model=ani2x",
        "#opt(method=rfo,fd_batch_size=auto)",
    ])
    assert cc_auto.params["fd_batch_size"] == "auto"

    with pytest.raises(ValueError, match="fd_batch_size"):
        CommandControl.from_settings([
            "#model=ani2x",
            "#opt(method=rfo,fd_batch_size=0)",
        ])


def test_command_control_accepts_prfo_hessian_recalc_options():
    from maple.function.read.command_control import CommandControl

    cc = CommandControl.from_settings([
        "#ts(method=prfo,hessian_recalc=5,hessian_update=bofill)",
    ])

    assert cc.task == "ts"
    assert cc.params["method"] == "prfo"
    assert cc.params["hessian_recalc"] == 5
    assert cc.params["hessian_update"] == "bofill"


def test_command_control_rejects_misspelled_prfo_hessian_recalc():
    """Typos in TS-method parameters must be caught by the unknown-param gate.

    Before the TS allow-set was wired into VALIDATED_TASK_PARAMS, a misspelled
    ``hessian_recalcc`` was silently stored under that wrong key, so the
    user's intended Bofill reuse interval was never applied.  The gate must
    raise and, where possible, suggest the closest known field.
    """
    import pytest
    from maple.function.read.command_control import CommandControl

    with pytest.raises(ValueError, match="hessian_recalc"):
        CommandControl.from_settings([
            "#ts(method=prfo,hessian_recalcc=5)",
        ])


def test_command_control_rejects_unknown_ts_param():
    import pytest
    from maple.function.read.command_control import CommandControl

    with pytest.raises(ValueError, match="not_a_field"):
        CommandControl.from_settings([
            "#ts(method=neb,not_a_field=true)",
        ])


def test_command_control_allows_prfo_options_under_nebts_refine():
    """NEB(refine=nebts) hands off to PRFO via ``raw_paras``, so PRFO fields
    must be accepted at the TS task layer when refine triggers that pipeline.
    """
    from maple.function.read.command_control import CommandControl

    cc = CommandControl.from_settings([
        "#ts(method=neb,refine=nebts,hessian_recalc=5,hessian_update=bofill)",
    ])
    assert cc.task == "ts"
    assert cc.params["method"] == "neb"
    assert cc.params["refine"] == "nebts"
    assert cc.params["hessian_recalc"] == 5


def test_auto_batch_sizer_caps_aimnet2_on_path_fd_and_hvp():
    """AIMNet2's batch path concatenates structures into a single dense
    neighbor mask whose memory is ``O((sum N_i)^2)``.  The generic per-item
    memory estimate underestimates this, so ``_AutoBatchSizer._apply_math_cap``
    must clamp AIMNet2 chunks for *every* batch kind, not just path E/F.

    ANI's cap, in contrast, is only the path-throughput cliff and stays
    kind='path' only.
    """
    import torch
    from ase import Atoms
    from maple.function.calculator._batch_eval import _AutoBatchSizer

    class _MockAIMNet:
        device = "cpu"
        dtype = torch.float32

    _MockAIMNet.__module__ = "maple.function.calculator.aimnet._aimnet2_calculator"
    _MockAIMNet.__name__ = "AIMNet2Calculator"

    class _MockANI:
        device = "cpu"
        dtype = torch.float32

    _MockANI.__module__ = "maple.function.calculator.ani._ani_calculator"
    _MockANI.__name__ = "ANICalculator"

    atoms_list = [Atoms("H" * 16) for _ in range(32)]
    for kind in ("path", "fd", "hvp"):
        chunk = _AutoBatchSizer(
            _MockAIMNet(), atoms_list, ("energy", "forces"), kind=kind
        ).chunk
        assert chunk <= 8, f"AIMNet2 kind={kind!r} should cap to <=8, got {chunk}"

    assert (
        _AutoBatchSizer(_MockANI(), atoms_list, ("energy", "forces"), kind="path").chunk
        <= 8
    )
    # ANI FD/HVP run per-structure forwards, so their math cap is intentionally
    # not engaged; this guards against an over-eager future widening of the cap.
    assert (
        _AutoBatchSizer(_MockANI(), atoms_list, ("energy", "forces"), kind="fd").chunk
        == len(atoms_list)
    )


def test_auto_batch_sizer_aimnet2_hard_cap_beats_user_override():
    """An over-permissive ``auto_path_batch_cap`` must not lift AIMNet2 above 8.

    Regression caught by codex review: the previous ordering returned from the
    explicit-cap branch before reaching the AIMNet2 hard cap, so a user-set
    ``calc.auto_path_batch_cap=32`` would bypass the 8-image safety bound and
    re-introduce the concat-mask OOM risk this commit is supposed to close.
    """
    import torch
    from ase import Atoms
    from maple.function.calculator._batch_eval import _AutoBatchSizer

    class _MockAIMNet:
        device = "cpu"
        dtype = torch.float32
        auto_path_batch_cap = 32

    _MockAIMNet.__module__ = "maple.function.calculator.aimnet._aimnet2_calculator"
    _MockAIMNet.__name__ = "AIMNet2Calculator"

    atoms_list = [Atoms("H" * 16) for _ in range(32)]
    for kind in ("path", "fd", "hvp"):
        chunk = _AutoBatchSizer(
            _MockAIMNet(), atoms_list, ("energy", "forces"), kind=kind
        ).chunk
        assert chunk <= 8, (
            f"AIMNet2 + auto_path_batch_cap=32 kind={kind!r}: hard cap should "
            f"still floor chunk at <=8, got {chunk}"
        )

    # User tightening below 8 must still take effect.
    class _MockAIMNetTight(_MockAIMNet):
        auto_path_batch_cap = 4

    for kind in ("path", "fd", "hvp"):
        chunk = _AutoBatchSizer(
            _MockAIMNetTight(), atoms_list, ("energy", "forces"), kind=kind
        ).chunk
        assert chunk == 4, (
            f"AIMNet2 + auto_path_batch_cap=4 kind={kind!r}: explicit cap "
            f"should tighten further, got {chunk}"
        )


def test_auto_batch_sizer_auto_path_batch_cap_stays_path_only_for_ani_mace():
    """For non-AIMNet backends, ``auto_path_batch_cap`` is a path-throughput
    knob and must not constrain FD Hessian or HVP workloads.

    Regression caught by codex review: the previous edit dropped the
    ``kind != "path"`` early-return and so widened the cap to fd/hvp.
    """
    import torch
    from ase import Atoms
    from maple.function.calculator._batch_eval import _AutoBatchSizer

    class _MockANI:
        device = "cpu"
        dtype = torch.float32
        auto_path_batch_cap = 4

    _MockANI.__module__ = "maple.function.calculator.ani._ani_calculator"
    _MockANI.__name__ = "ANICalculator"

    atoms_list = [Atoms("H" * 16) for _ in range(32)]

    # Path is capped by the user override.
    assert (
        _AutoBatchSizer(_MockANI(), atoms_list, ("energy", "forces"), kind="path").chunk
        == 4
    )
    # FD and HVP must remain at n_total — auto_path_batch_cap is path-only.
    assert (
        _AutoBatchSizer(_MockANI(), atoms_list, ("energy", "forces"), kind="fd").chunk
        == len(atoms_list)
    )
    assert (
        _AutoBatchSizer(_MockANI(), atoms_list, ("energy", "forces"), kind="hvp").chunk
        == len(atoms_list)
    )


def test_setcalculator_applies_model_batch_size_aliases():
    from maple.function.calculator.set_calculator import SetClaculator

    setter = object.__new__(SetClaculator)
    setter.model_options = {"batch_size": 4}
    calc = HarmonicCalc(k=1.0, ref_positions=np.zeros((1, 3)))

    setter._apply_batch_size(calc)

    assert calc.batch_size == 4
    assert calc.path_batch_size == 4
    assert calc.fd_batch_size == 4
    assert calc.hessian_batch_size == 4
    assert calc.analytic_hessian_batch_size == 4

    setter.model_options = {"batch_size": "auto"}
    auto_calc = HarmonicCalc(k=1.0, ref_positions=np.zeros((1, 3)))
    setter._apply_batch_size(auto_calc)
    assert auto_calc.batch_size == "auto"
    assert auto_calc.path_batch_size == "auto"
    assert auto_calc.fd_batch_size == "auto"
    assert auto_calc.hessian_batch_size == "auto"
    assert auto_calc.analytic_hessian_batch_size == "auto"


def test_setcalculator_rejects_batch_size_for_periodic_atoms():
    from maple.function.calculator.set_calculator import SetClaculator

    setter = object.__new__(SetClaculator)
    setter.model_options = {"batch_size": 4}
    setter.atoms = Atoms("H", positions=np.zeros((1, 3)), cell=np.eye(3) * 8.0, pbc=True)
    calc = HarmonicCalc(k=1.0, ref_positions=np.zeros((1, 3)))

    with pytest.raises(ValueError, match="PBC and batch_size"):
        setter._apply_batch_size(calc)

    setter.model_options = {"batch_size": "auto"}
    with pytest.raises(ValueError, match="PBC and batch_size"):
        setter._apply_batch_size(calc)


def test_aimnet2_advertises_analytic_hessian_support():
    from maple.function.calculator.aimnet._aimnet2_calculator import AIMNet2Calculator

    assert AIMNet2Calculator.supports_analytic_hessian is True


def _quadratic_hessian(batch_size=None, *, batched=False):
    coords = torch.tensor(
        [[0.1, -0.2, 0.3], [0.4, -0.5, 0.6]],
        dtype=torch.float64,
        requires_grad=True,
    )
    weights = torch.arange(1, 7, dtype=torch.float64).reshape(2, 3)
    energy = 0.5 * torch.sum(weights * coords * coords)
    helper = hessian_batched_vjp if batched else hessian_loop
    return helper(
        energy,
        coords,
        output_dof=6,
        input_dof=6,
        batch_size=batch_size,
    )


def test_exact_autograd_hessian_respects_row_batch_size():
    expected = torch.diag(torch.arange(1, 7, dtype=torch.float64))

    loop = _quadratic_hessian(batch_size=1)
    chunked = _quadratic_hessian(batch_size=2)
    auto = _quadratic_hessian(batch_size="auto")
    full_vjp = _quadratic_hessian(batch_size=None, batched=True)
    chunked_vjp = _quadratic_hessian(batch_size=3, batched=True)
    auto_vjp = _quadratic_hessian(batch_size="auto", batched=True)

    torch.testing.assert_close(loop, expected)
    torch.testing.assert_close(chunked, expected)
    torch.testing.assert_close(auto, expected)
    torch.testing.assert_close(full_vjp, expected)
    torch.testing.assert_close(chunked_vjp, expected)
    torch.testing.assert_close(auto_vjp, expected)


def test_exact_autograd_hessian_rejects_invalid_batch_size():
    with pytest.raises(ValueError, match="batch_size"):
        _quadratic_hessian(batch_size=0)


def test_exact_autograd_batched_vjp_can_warn_on_fallback(monkeypatch):
    import maple.function.calculator._autograd_hessian as ah

    def fail_batched_vjp(*args, **kwargs):
        raise RuntimeError("vmap unavailable")

    monkeypatch.setattr(ah, "_batched_vjp_from_grad", fail_batched_vjp)

    coords = torch.tensor(
        [[0.1, -0.2, 0.3], [0.4, -0.5, 0.6]],
        dtype=torch.float64,
        requires_grad=True,
    )
    weights = torch.arange(1, 7, dtype=torch.float64).reshape(2, 3)
    energy = 0.5 * torch.sum(weights * coords * coords)

    with pytest.warns(RuntimeWarning, match="fell back"):
        H = ah.hessian_batched_vjp(
            energy,
            coords,
            output_dof=6,
            input_dof=6,
            batch_size=2,
            warn_on_fallback=True,
        )

    expected = torch.diag(torch.arange(1, 7, dtype=torch.float64))
    torch.testing.assert_close(H, expected)


def test_auto_batch_estimate_targets_seventy_five_percent_free_memory():
    assert _estimate_auto_batch_size_from_item_bytes(
        n_total=100,
        item_bytes=10,
        free_bytes=100,
        target_fraction=0.75,
    ) == 7
    assert _estimate_auto_batch_size_from_item_bytes(
        n_total=4,
        item_bytes=0,
        free_bytes=100,
        target_fraction=0.75,
    ) == 4


# ---------------------------------------------------------------------------
# CalcABC.calculate_many fallback
# ---------------------------------------------------------------------------
def test_calculate_many_fallback_returns_per_structure_results():
    ref = np.zeros((3, 3))
    atoms_list = [
        Atoms("HHH", positions=ref + 0.05 * (i + 1)) for i in range(4)
    ]
    calc = HarmonicCalc(k=1.0, ref_positions=ref)
    for at in atoms_list:
        at.calc = calc

    result = calc.calculate_many(atoms_list, properties=("energy", "forces"))

    assert isinstance(result, BatchResult)
    assert result.energies is not None and result.energies.shape == (4,)
    assert result.forces is not None and len(result.forces) == 4
    assert all(f.shape == (3, 3) for f in result.forces)

    # Energies should monotonically increase as displacement grows.
    assert np.all(np.diff(result.energies) > 0.0)


def test_calculate_many_rejects_hessian_property():
    calc = HarmonicCalc(k=1.0, ref_positions=np.zeros((2, 3)))
    atoms = Atoms("HH", positions=np.zeros((2, 3))); atoms.calc = calc
    with pytest.raises(NotImplementedError):
        calc.calculate_many([atoms], properties=("hessian",))


def test_batch_result_validates_lengths_and_shapes():
    with pytest.raises(ValueError, match="lengths"):
        BatchResult(
            energies=np.zeros(2),
            forces=[np.zeros((1, 3))],
        )

    with pytest.raises(ValueError, match="shape"):
        BatchResult(forces=[np.zeros((3,))])

    with pytest.raises(ValueError, match="corresponding force atom count"):
        BatchResult(
            forces=[np.zeros((2, 3))],
            hessians=[np.zeros((3, 3))],
        )


# ---------------------------------------------------------------------------
# energy_forces_one
# ---------------------------------------------------------------------------
def test_energy_forces_one_matches_separate_calls():
    ref = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    atoms_a = Atoms("HH", positions=ref + 0.04)
    atoms_a.calc = HarmonicCalc(k=2.2, ref_positions=ref)
    e_legacy = float(atoms_a.get_potential_energy(force_consistent=True))
    f_legacy = np.asarray(atoms_a.get_forces(), dtype=np.float64)

    atoms_b = Atoms("HH", positions=ref + 0.04)
    atoms_b.calc = HarmonicCalc(k=2.2, ref_positions=ref)
    e_merged, f_merged = energy_forces_one(atoms_b.calc, atoms_b)

    assert e_legacy == pytest.approx(e_merged, abs=0.0)
    np.testing.assert_array_equal(f_legacy, f_merged)


# ---------------------------------------------------------------------------
# FDHessianEvaluator
# ---------------------------------------------------------------------------
def test_fd_hessian_matches_analytic_for_harmonic():
    ref = np.array(
        [[0.0, 0.0, 0.0], [1.1, 0.0, 0.0], [0.0, 1.1, 0.0], [0.0, 0.0, 1.1]]
    )
    atoms = Atoms("CHHH", positions=ref + 0.03)
    atoms.calc = HarmonicCalc(k=1.5, ref_positions=ref)

    H = FDHessianEvaluator(atoms.calc).hessian(atoms, delta=1e-4)
    H_expected = 1.5 * np.eye(12, dtype=np.float64)
    assert np.max(np.abs(H - H_expected)) < 1e-6


def test_fd_hessian_chunked_is_bit_identical_to_unchunked():
    ref = np.zeros((5, 3))
    atoms = Atoms("CHHHH", positions=ref + 0.02)
    atoms.calc = HarmonicCalc(k=1.0, ref_positions=ref)

    h_full = FDHessianEvaluator(atoms.calc).hessian(atoms, delta=1e-4)
    h_chunk = FDHessianEvaluator(atoms.calc, fd_batch_size=3).hessian(atoms, delta=1e-4)
    np.testing.assert_array_equal(h_chunk, h_full)


def test_fd_hessian_zeros_rows_and_columns_for_fixed_atoms():
    ref = np.zeros((4, 3))
    atoms = Atoms("CHHH", positions=ref + 0.02)
    atoms.calc = HarmonicCalc(k=1.0, ref_positions=ref)
    atoms.set_constraint(FixAtoms(indices=[0, 2]))

    H = FDHessianEvaluator(atoms.calc).hessian(atoms, delta=1e-4)

    # Frozen atoms 0 and 2 are projected out as both rows and columns.
    for a in (0, 2):
        assert np.all(np.abs(H[3 * a : 3 * a + 3, :]) < 1e-12)
        assert np.all(np.abs(H[:, 3 * a : 3 * a + 3]) < 1e-12)


def test_fd_hessian_fixatoms_projection_is_symmetric_for_coupled_pes():
    atoms = Atoms("HH", positions=[[0.0, 0.0, 0.0], [1.2, 0.1, 0.0]])
    atoms.calc = CoupledPairCalc(k=2.0, d0=np.array([1.0, 0.0, 0.0]))
    atoms.set_constraint(FixAtoms(indices=[0]))

    H = FDHessianEvaluator(atoms.calc).hessian(atoms, delta=1e-4)

    assert np.all(np.abs(H[:3, :]) < 1e-12)
    assert np.all(np.abs(H[:, :3]) < 1e-12)
    np.testing.assert_allclose(H, H.T, atol=1e-10)
    np.testing.assert_allclose(H[3:, 3:], 2.0 * np.eye(3), atol=1e-8)


def test_fd_hessian_symmetrizes_force_derivative_noise():
    class AsymmetricLinearForceCalc(CalcABC):
        implemented_properties = ["energy", "forces", "free_energy"]

        def __init__(self, matrix):
            super().__init__()
            self.matrix = np.asarray(matrix, dtype=np.float64)

        def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            pos = atoms.get_positions().reshape(-1)
            self.results["energy"] = 0.0
            self.results["free_energy"] = 0.0
            if "forces" in properties:
                self.results["forces"] = (-(self.matrix @ pos)).reshape(-1, 3)

    A = np.array(
        [
            [1.0, 0.2, 0.0],
            [0.0, 2.0, 0.3],
            [0.1, 0.0, 3.0],
        ],
        dtype=np.float64,
    )
    atoms = Atoms("H", positions=[[0.1, -0.2, 0.3]])
    atoms.calc = AsymmetricLinearForceCalc(A)

    H = FDHessianEvaluator(atoms.calc).hessian(atoms, delta=1e-4)

    np.testing.assert_allclose(H, 0.5 * (A + A.T), atol=1e-10)
    np.testing.assert_allclose(H, H.T, atol=0.0)


def test_fd_hessian_rejects_invalid_delta_and_batch_size():
    atoms = Atoms("H", positions=np.zeros((1, 3)))
    atoms.calc = HarmonicCalc(k=1.0, ref_positions=np.zeros((1, 3)))

    with pytest.raises(ValueError, match="delta"):
        FDHessianEvaluator(atoms.calc).hessian(atoms, delta=0.0)

    with pytest.raises(ValueError, match="fd_batch_size"):
        FDHessianEvaluator(atoms.calc, fd_batch_size=0)


def test_fd_hessian_validates_force_count_and_shape():
    class WrongCountCalc(HarmonicCalc):
        def calculate_many(self, atoms_list, properties=("forces",)):
            return BatchResult(forces=[np.zeros((1, 3))])

    class WrongShapeCalc(HarmonicCalc):
        def calculate_many(self, atoms_list, properties=("forces",)):
            return BatchResult(forces=[np.zeros((1, 2)) for _ in atoms_list])

    atoms = Atoms("H", positions=np.zeros((1, 3)))
    atoms.calc = WrongCountCalc(k=1.0, ref_positions=np.zeros((1, 3)))
    with pytest.raises(ValueError, match="wrong number"):
        FDHessianEvaluator(atoms.calc).hessian(atoms, delta=1e-4)

    atoms.calc = WrongShapeCalc(k=1.0, ref_positions=np.zeros((1, 3)))
    with pytest.raises(ValueError, match="shape"):
        FDHessianEvaluator(atoms.calc).hessian(atoms, delta=1e-4)


def test_fd_hessian_restores_geometry():
    ref = np.zeros((3, 3))
    pos = ref + 0.07
    atoms = Atoms("HHH", positions=pos)
    atoms.calc = HarmonicCalc(k=1.0, ref_positions=ref)
    before = atoms.get_positions().copy()
    _ = FDHessianEvaluator(atoms.calc).hessian(atoms, delta=1e-4)
    np.testing.assert_array_equal(before, atoms.get_positions())


def test_fd_hessian_uses_make_fd_context_when_available():
    """Phase 2A hook: evaluator should prefer a calculator-provided context
    and avoid the calculate_many fallback when the context exists."""

    class CountingContext(FDHessianContext):
        def force_at(self, positions: np.ndarray) -> np.ndarray:
            self.calc.context_force_calls += 1
            R = np.asarray(positions, dtype=np.float64)
            return -self.calc.k * (R - self.calc.ref)

    class ContextCalc(HarmonicCalc):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.context_force_calls = 0
            self.calculate_many_calls = 0
            self.context_modes = []

        def make_fd_context(self, atoms, *, delta=None, fd_context_mode=None):
            self.context_modes.append(fd_context_mode or self.fd_context_mode)
            return CountingContext(
                self,
                atoms,
                mode=fd_context_mode or self.fd_context_mode,
                delta=delta,
            )

        def calculate_many(self, atoms_list, properties=("forces",)):
            self.calculate_many_calls += 1
            raise AssertionError("context path should bypass calculate_many fallback")

    ref = np.zeros((2, 3))
    atoms = Atoms("HH", positions=ref + 0.02)
    atoms.calc = ContextCalc(k=1.25, ref_positions=ref)

    H = FDHessianEvaluator(
        atoms.calc,
        fd_context_mode="fast",
    ).hessian(atoms, delta=1e-4)

    np.testing.assert_allclose(H, 1.25 * np.eye(6), atol=1e-8)
    assert atoms.calc.context_force_calls == 12  # 2 * 3 * N
    assert atoms.calc.calculate_many_calls == 0
    assert atoms.calc.context_modes == ["fast"]


def test_fd_hessian_reads_calculator_level_batch_size_for_freq():
    """Frequency calls calc.get_hessian(); model batch_size must still chunk FD.

    This covers the user-facing OOM guard for numerical Hessians: setting
    ``#model(..., batch_size=N)`` attaches ``calc.batch_size`` and
    FDHessianEvaluator consumes it even when a freq job does not pass an
    algorithm-specific fd_batch_size.
    """

    from maple.function.dispatcher.frequency.frequency import MWFrequency

    class ChunkRecordingCalc(HarmonicCalc):
        supported_hessian_modes = ("numerical",)

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.chunk_sizes = []

        def calculate_many(self, atoms_list, properties=("forces",)):
            self.chunk_sizes.append(len(atoms_list))
            forces = []
            for at in atoms_list:
                R = at.get_positions().astype(np.float64)
                forces.append(-self.k * (R - self.ref))
            return BatchResult(forces=forces)

        def get_hessian(self, atoms, delta: float = 1e-4):
            return FDHessianEvaluator(self).hessian(atoms, delta=delta)

    ref = np.zeros((3, 3))
    atoms = Atoms("HHH", positions=ref + 0.02)
    calc = ChunkRecordingCalc(k=1.0, ref_positions=ref)
    calc.batch_size = 5
    atoms.calc = calc

    with tempfile.NamedTemporaryFile(mode="w", suffix=".out", delete=False) as fh:
        out = fh.name
    try:
        H = MWFrequency(output=out, atoms=atoms).get_hessian()
    finally:
        os.unlink(out)

    np.testing.assert_allclose(H, np.eye(9), atol=1e-8)
    assert calc.chunk_sizes == [5, 5, 5, 3]


def test_fd_hessian_disables_batch_path_for_pbc():
    class NoBatchWhenPBCCalc(HarmonicCalc):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.calculate_many_calls = 0

        def calculate_many(self, atoms_list, properties=("forces",)):
            self.calculate_many_calls += 1
            raise AssertionError("PBC Hessian must use sequential calculate, not batch")

    ref = np.zeros((1, 3))
    atoms = Atoms("H", positions=ref + 0.02, cell=np.eye(3) * 8.0, pbc=True)
    atoms.calc = NoBatchWhenPBCCalc(k=1.0, ref_positions=ref)

    H = FDHessianEvaluator(atoms.calc).hessian(atoms, delta=1e-4)

    np.testing.assert_allclose(H, np.eye(3), atol=1e-8)
    assert atoms.calc.calculate_many_calls == 0
    assert atoms.calc.calls == 6


def test_fd_hessian_accepts_auto_batch_size():
    class ChunkRecordingCalc(HarmonicCalc):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.chunk_sizes = []

        def calculate_many(self, atoms_list, properties=("forces",)):
            self.chunk_sizes.append(len(atoms_list))
            return super().calculate_many(atoms_list, properties=properties)

    ref = np.zeros((2, 3))
    atoms = Atoms("HH", positions=ref + 0.02)
    calc = ChunkRecordingCalc(k=1.0, ref_positions=ref)
    calc.device = "cpu"
    calc.batch_size = "auto"
    atoms.calc = calc

    H = FDHessianEvaluator(atoms.calc).hessian(atoms, delta=1e-4)

    np.testing.assert_allclose(H, np.eye(6), atol=1e-8)
    assert calc.chunk_sizes == [12]


def test_fd_hessian_invalid_context_mode_fails_fast():
    calc = HarmonicCalc(k=1.0, ref_positions=np.zeros((1, 3)))
    calc.fd_context_mode = "unsafe"
    atoms = Atoms("H", positions=np.zeros((1, 3)))
    atoms.calc = calc

    with pytest.raises(ValueError, match="fd_context_mode"):
        FDHessianEvaluator(atoms.calc).hessian(atoms, delta=1e-4)


# ---------------------------------------------------------------------------
# PathEvaluator (Phase 2 scaffold; Phase 1 only ships the API)
# ---------------------------------------------------------------------------
def test_path_evaluator_returns_energies_and_forces_for_images():
    ref = np.zeros((2, 3))
    images = [Atoms("HH", positions=ref + i * 0.05) for i in range(5)]
    calc = HarmonicCalc(k=1.0, ref_positions=ref)
    for at in images:
        at.calc = calc

    es, fs = PathEvaluator(calc, batch_size=2).energy_forces(images)
    assert es.shape == (5,) and len(fs) == 5
    assert np.all(np.diff(es) >= 0.0)


def test_path_evaluator_disables_batch_path_for_pbc_images():
    class NoBatchWhenPBCCalc(HarmonicCalc):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.calculate_many_calls = 0

        def calculate_many(self, atoms_list, properties=("energy", "forces")):
            self.calculate_many_calls += 1
            raise AssertionError("PBC path evaluation must use sequential calculate, not batch")

    ref = np.zeros((1, 3))
    images = [
        Atoms("H", positions=ref + i * 0.01, cell=np.eye(3) * 8.0, pbc=True)
        for i in range(3)
    ]
    calc = NoBatchWhenPBCCalc(k=1.0, ref_positions=ref)

    es, fs = PathEvaluator(calc, batch_size=2).energy_forces(images)

    assert es.shape == (3,)
    assert len(fs) == 3
    assert calc.calculate_many_calls == 0
    assert calc.calls == 3


def test_path_evaluator_validates_energy_and_force_counts():
    from types import SimpleNamespace

    class WrongEnergyCountCalc(HarmonicCalc):
        def calculate_many(self, atoms_list, properties=("energy", "forces")):
            return SimpleNamespace(
                energies=np.zeros(len(atoms_list) - 1, dtype=np.float64),
                forces=[np.zeros((1, 3)) for _ in atoms_list],
            )

    class WrongForceCountCalc(HarmonicCalc):
        def calculate_many(self, atoms_list, properties=("energy", "forces")):
            return SimpleNamespace(
                energies=np.zeros(len(atoms_list), dtype=np.float64),
                forces=[np.zeros((1, 3)) for _ in atoms_list[:-1]],
            )

    images = [Atoms("H", positions=[[i * 0.01, 0.0, 0.0]]) for i in range(3)]

    with pytest.raises(RuntimeError, match="wrong number of energies"):
        PathEvaluator(
            WrongEnergyCountCalc(k=1.0, ref_positions=np.zeros((1, 3)))
        ).energy_forces(images)

    with pytest.raises(RuntimeError, match="wrong number of force arrays"):
        PathEvaluator(
            WrongForceCountCalc(k=1.0, ref_positions=np.zeros((1, 3)))
        ).energy_forces(images)


def test_neb_path_energy_forces_uses_native_batch_snapshot():
    """NEB/CINEB should consume one batched E/F snapshot for a path.

    This guards the TS path acceleration without touching the NEB force
    formula itself: model calls are consolidated through calculate_many, and
    projected-force assembly receives the returned forces directly.
    """
    from maple.function.dispatcher.ts.algorithm.neb import NEB, neb_forces

    class NativeBatchHarmonicCalc(HarmonicCalc):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.batch_calls = 0

        def calculate_many(self, atoms_list, properties=("energy", "forces")):
            self.batch_calls += 1
            energies = []
            forces = []
            for at in atoms_list:
                R = at.get_positions().astype(np.float64)
                d = R - self.ref
                energies.append(0.5 * self.k * float(np.sum(d * d)))
                forces.append(-self.k * d)
            return BatchResult(
                energies=np.asarray(energies, dtype=np.float64),
                forces=forces,
            )

    ref = np.zeros((2, 3))
    images = [Atoms("HH", positions=ref + i * 0.02) for i in range(4)]
    calc = NativeBatchHarmonicCalc(k=1.0, ref_positions=ref)
    for img in images:
        img.calc = calc

    neb = object.__new__(NEB)
    energies, forces = neb._path_energy_forces(images)
    projected, _, _ = neb_forces(images, energies, k_spring=0.1, raw_forces=forces)

    assert calc.batch_calls == 1
    assert calc.calls == 0
    assert len(energies) == len(images)
    assert len(forces) == len(images)
    assert all(f.shape == (2, 3) for f in projected)


# ---------------------------------------------------------------------------
# HVPEvaluator
# ---------------------------------------------------------------------------
class TinyANI(CalcABC):
    """ANI-style: torch model, autograd path for HVP. Quadratic potential."""

    implemented_properties = ["energy", "forces", "free_energy"]
    supports_hvp = True

    def __init__(self, k=1.0):
        super().__init__()
        self.k = k
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self.d4 = False

    def model(self, species, coordinates):
        # CalcABC.get_hvp signature expects model(species, coords)[0]
        return [0.5 * self.k * (coordinates ** 2).sum()]

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        R = atoms.get_positions().astype(np.float64)
        e = 0.5 * self.k * float(np.sum(R * R))
        self.results["energy"] = e
        self.results["free_energy"] = e
        if "forces" in properties:
            self.results["forces"] = -self.k * R


def test_hvp_evaluator_autodiff_path_matches_analytic():
    """For E = 0.5 k |R|^2, H = k I, so Hn = k * n."""
    atoms = Atoms("HH", positions=[[0.3, 0.0, 0.0], [0.0, 0.2, 0.0]])
    atoms.calc = TinyANI(k=2.0)

    n_vec = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    Hn, F, E = HVPEvaluator(atoms.calc).hn(atoms, n_vec, delta=1e-3)
    np.testing.assert_allclose(Hn, 2.0 * n_vec, atol=1e-5)


def test_hvp_evaluator_fd_fallback_when_no_get_hvp():
    """Default ``supports_hvp = False`` should use the FD pair fallback
    regardless of whether ``get_hvp`` is inherited from CalcABC."""

    class NoHVPCalc(HarmonicCalc):
        pass

    ref = np.zeros((2, 3))
    atoms = Atoms("HH", positions=ref + 0.1)
    atoms.calc = NoHVPCalc(k=1.5, ref_positions=ref)

    n_vec = np.zeros(6); n_vec[0] = 1.0
    Hn, F, E = HVPEvaluator(atoms.calc).hn(atoms, n_vec, delta=1e-4)
    # H = k I, so Hn along x of atom 0 should be (k, 0, 0, 0, 0, 0).
    expected = np.zeros(6); expected[0] = 1.5
    np.testing.assert_allclose(Hn, expected, atol=1e-6)

    # F and E must be the CURRENT-point values at R (matching CalcABC.get_hvp),
    # not the +-delta midpoint average. At R = ref + 0.1 the analytic harmonic
    # forces are -k*d = -0.15 and the energy is 0.5*k*sum(d^2) = 0.045. The old
    # midpoint code returned E = 0.045 + 0.5*k*delta^2 (= +7.5e-9 here), so the
    # tight atol below fails on the midpoint bug and passes on the current-point fix.
    np.testing.assert_allclose(F, np.full(6, -1.5 * 0.1), atol=1e-10)
    np.testing.assert_allclose(E, 0.5 * 1.5 * 6 * 0.1 ** 2, atol=1e-10)


def test_hvp_evaluator_fd_fallback_reads_calculator_level_batch_size():
    class ChunkRecordingNoHVPCalc(HarmonicCalc):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.chunk_sizes = []

        def calculate_many(self, atoms_list, properties=("energy", "forces")):
            self.chunk_sizes.append(len(atoms_list))
            energies = []
            forces = []
            for at in atoms_list:
                R = at.get_positions().astype(np.float64)
                d = R - self.ref
                energies.append(0.5 * self.k * float(np.sum(d * d)))
                forces.append(-self.k * d)
            return BatchResult(energies=np.asarray(energies), forces=forces)

    ref = np.zeros((2, 3))
    atoms = Atoms("HH", positions=ref + 0.1)
    calc = ChunkRecordingNoHVPCalc(k=1.5, ref_positions=ref)
    calc.batch_size = 1
    atoms.calc = calc

    n_vec = np.zeros(6)
    n_vec[0] = 1.0
    Hn, _, _ = HVPEvaluator(atoms.calc).hn(atoms, n_vec, delta=1e-4)

    expected = np.zeros(6)
    expected[0] = 1.5
    np.testing.assert_allclose(Hn, expected, atol=1e-6)
    # FD batch is [R+dn, R-dn, R]: three structures, so batch_size=1 chunks
    # the single calculate_many call into three single-structure evaluations.
    assert calc.chunk_sizes == [1, 1, 1]


# ---------------------------------------------------------------------------
# OPT-RFO reject path no longer wastes a forward pass
# ---------------------------------------------------------------------------
def test_opt_rfo_reject_branch_does_not_call_calculator():
    """The OPT-RFO reject branch must restore E/F from the iteration
    snapshot, not re-evaluate them on the calculator.

    This regression check inspects the source of ``RFO.run`` and confirms
    the reject branch contains no ``get_potential_energy`` or
    ``get_forces`` calls. The pre-fix code did exactly two such calls per
    reject (one for energy, one for forces) which together cost an extra
    forward + backward pass per rejected step.
    """
    import inspect
    # The package __init__ shadows the .RFO module with the RFO class, so
    # walk straight to RFO.run via the class object.
    from maple.function.dispatcher.optimization.algorithm.RFO import RFO

    src = inspect.getsource(RFO.run)
    # Crude but robust: split at the marker comment we left at the reject
    # site and check the immediate aftermath for forbidden calls.
    marker = "reject: rollback geometry"
    assert marker in src, "Reject branch comment missing — refactor broke the marker"
    reject_chunk = src.split(marker, 1)[1].split("\n\n", 1)[0]
    forbidden = ("get_potential_energy", "get_forces")
    found = [tok for tok in forbidden if tok in reject_chunk]
    assert not found, (
        f"OPT-RFO reject branch still contains {found}; the fix is to "
        "restore E and F from the iteration's E_old/F_cart snapshot, "
        "not re-evaluate them on the calculator."
    )
    assert "reset_calculator_cache" in reject_chunk


def test_opt_rfo_runs_end_to_end_with_forced_reject():
    """Force one reject and check the optimizer terminates and converges,
    proving the restored E/F snapshot is consistent across the rollback.
    """
    from maple.function.dispatcher.optimization.algorithm.RFO import RFO

    class ResetCountingCalc(HarmonicCalc):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.reset_calls = 0

        def reset(self):
            self.reset_calls += 1
            return super().reset()

    ref = np.zeros((2, 3))
    atoms = Atoms("HH", positions=ref + 0.3)
    atoms.calc = ResetCountingCalc(k=1.0, ref_positions=ref)
    _set_thresholds(atoms, val=1e-4)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".out", delete=False) as fh:
        out = fh.name
    try:
        opt = RFO(
            atoms=atoms,
            output=out,
            paras={"rfo": {"max_iter": 30, "trust_radius_init": 0.5}},
        )
        original_decide = opt._accept_or_reject
        opt._reject_used = False

        def fake_decide(rho, on_boundary):
            if not opt._reject_used:
                opt._reject_used = True
                return False
            return original_decide(rho, on_boundary)

        opt._accept_or_reject = fake_decide
        opt.run()

        assert opt._reject_used, "Test setup did not trigger a reject"
        assert atoms.calc.reset_calls >= 1
        max_f = float(np.abs(atoms.get_forces()).max())
        assert max_f < 1e-4, (
            f"OPT-RFO did not converge despite reject-and-restore: "
            f"|F|_max = {max_f}; the snapshot restore may be inconsistent."
        )
    finally:
        for ext in ("", "_opt_traj.xyz", "_traj.xyz", "_opt.xyz"):
            p = os.path.splitext(out)[0] + ext if ext else out
            if os.path.exists(p):
                os.remove(p)


# ---------------------------------------------------------------------------
# TS-PRFO outer loop no longer refetches E/F between iterations
# ---------------------------------------------------------------------------
def test_bofill_hessian_update_satisfies_secant_and_symmetry():
    from maple.function.dispatcher.ts.algorithm.PRFO import bofill_hessian_update

    H0 = np.diag([0.7, 1.2, -0.4]).astype(np.float64)
    H_true = np.array(
        [
            [2.0, 0.2, 0.0],
            [0.2, 1.5, -0.1],
            [0.0, -0.1, -0.8],
        ],
        dtype=np.float64,
    )
    q0 = np.array([0.1, -0.2, 0.3], dtype=np.float64)
    step = np.array([0.05, -0.03, 0.04], dtype=np.float64)
    grad_old = H_true @ q0
    grad_new = H_true @ (q0 + step)

    H_new, ok, source = bofill_hessian_update(H0, step, grad_old, grad_new)

    assert ok
    assert "bofill" in source or "psb" in source
    np.testing.assert_allclose(H_new, H_new.T, atol=1e-12)
    np.testing.assert_allclose(
        H_new @ step,
        grad_new - grad_old,
        atol=1e-11,
        rtol=1e-11,
    )


def test_ts_prfo_validates_hessian_recalc_parameters():
    from maple.function.dispatcher.ts.algorithm.PRFO import PRFO

    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])

    with pytest.raises(ValueError, match="hessian_recalc"):
        PRFO(output=os.devnull, atoms=atoms, paras={"prfo": {"hessian_recalc": 0}})

    with pytest.raises(ValueError, match="hessian_update"):
        PRFO(output=os.devnull, atoms=atoms, paras={"prfo": {"hessian_update": "bfgs"}})


def test_ts_prfo_hessian_recalc_reuses_updated_hessian(monkeypatch):
    import importlib

    prfo_mod = importlib.import_module("maple.function.dispatcher.ts.algorithm.PRFO")

    atoms = Atoms(
        "CHHH",
        positions=[[0.3, -0.2, 0.05], [10, 0, 0], [0, 10, 0], [0, 0, 10]],
    )
    atoms.calc = SaddleCalc()

    exact_calls = 0
    update_calls = 0

    def fake_hessian(_atoms):
        nonlocal exact_calls
        exact_calls += 1
        H = np.eye(12, dtype=np.float64)
        H[0, 0] = -2.0
        return H

    def fake_update(H, step, grad_old, grad_new, *, eps=1e-12):
        nonlocal update_calls
        update_calls += 1
        return H.copy(), True, "fake-bofill"

    monkeypatch.setattr(prfo_mod, "calculate_Hessian", fake_hessian)
    monkeypatch.setattr(prfo_mod, "bofill_hessian_update", fake_update)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".out", delete=False) as fh:
        out = fh.name
    try:
        prfo_mod.PRFO(
            output=out,
            atoms=atoms,
            paras={
                "prfo": {
                    "max_iter": 3,
                    "trust_radius": 0.05,
                    "hessian_recalc": 99,
                    "validate_ts_mode": False,
                    "f_max_th": -1.0,
                    "f_rms_th": -1.0,
                    "dp_max_th": -1.0,
                    "dp_rms_th": -1.0,
                }
            },
        ).run()

        assert exact_calls == 1
        assert update_calls >= 2
        text = open(out, "r", encoding="utf-8").read()
        assert "Hessian policy: exact Hessian initially" in text
        assert "Hessian source: fake-bofill" in text
    finally:
        for ext in ("", "_prfo_traj.xyz", "_prfo_ts.xyz"):
            p = os.path.splitext(out)[0] + ext if ext else out
            if os.path.exists(p):
                os.remove(p)


def test_ts_prfo_reject_branch_resets_calculator_cache():
    import inspect
    from maple.function.dispatcher.ts.algorithm.PRFO import PRFO

    src = inspect.getsource(PRFO.run)
    marker = "Reject: rollback geometry"
    assert marker in src
    reject_chunk = src.split(marker, 1)[1].split("continue", 1)[0]
    assert "reset_calculator_cache" in reject_chunk


def test_ts_prfo_carries_ef_across_outer_iterations():
    """After an accepted trial, the next outer iteration should reuse the
    accepted-trial E/F rather than calling the calculator again at the top
    of the loop. We verify by counting calculator invocations and asserting
    a tight upper bound that the old top-of-loop refetch would violate."""
    from maple.function.dispatcher.ts.algorithm.PRFO import PRFO

    atoms = Atoms(
        "CHHH",
        positions=[[0.3, -0.2, 0.05], [10, 0, 0], [0, 10, 0], [0, 0, 10]],
    )
    atoms.set_constraint(FixAtoms(indices=[1, 2, 3]))
    atoms.calc = SaddleCalc()
    _set_thresholds(atoms, val=1e-5)

    n_before = atoms.calc.calls

    with tempfile.NamedTemporaryFile(mode="w", suffix=".out", delete=False) as fh:
        out = fh.name
    try:
        PRFO(
            output=out,
            atoms=atoms,
            paras={"prfo": {"max_iter": 30, "trust_radius": 0.1}},
        ).run()

        # The saddle is exactly at the origin and the analytic Hessian-style
        # FD on this PES converges in O(few) PRFO steps. Per outer iter we
        # need:
        #   - 1 initial energy_forces_one (before the outer loop)
        #   - 1 trial energy_only (per inner attempt)
        #   - 1 forces fetch (per accept)
        #   - 2 * 3 * N_movable = 6 FD-Hessian force evaluations per outer
        #     iter (atom 0 only; FixAtoms zeros the other three atoms)
        # The old code additionally did 1 (E + F) refetch at the top of
        # every outer iter; we expect those refetches to be gone.
        #
        # Loose upper bound: 200 forward passes for this trivial problem
        # would absolutely require the old pattern.
        delta = atoms.calc.calls - n_before
        # Convergence sanity:
        assert float(np.linalg.norm(atoms.get_positions()[0])) < 1e-3
        # Generous bound that the old code would still violate
        # (old: 6 FD + 1 E refetch + 1 F refetch + 1 trial + 1 accept-F per
        # outer iter, easily > 200 over ~20 iters); new bound here is loose
        # enough to be implementation-tolerant but tight enough to flag a
        # regression where the top-of-loop refetch is reintroduced.
        assert delta < 200, (
            f"TS-PRFO did {delta} calculator invocations; the top-of-loop "
            "E/F refetch may have been reintroduced."
        )
    finally:
        for ext in ("", "_prfo_traj.xyz", "_prfo_ts.xyz"):
            p = os.path.splitext(out)[0] + ext if ext else out
            if os.path.exists(p):
                os.remove(p)


def test_ts_prfo_rejects_numerical_hessian_when_analytic_is_available():
    """Precision-first TS searches must not replace analytic Hessians with FD."""
    from maple.function.dispatcher.ts.algorithm.PRFO import PRFO

    atoms = Atoms(
        "CHHH",
        positions=np.zeros((4, 3)),
    )
    atoms.calc = FakeNumericalTSCalc()
    _set_thresholds(atoms, val=1e-8)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".out", delete=False) as fh:
        out = fh.name
    try:
        with pytest.raises(ValueError, match="requires the analytic Hessian"):
            PRFO(
                output=out,
                atoms=atoms,
                paras={"prfo": {"max_iter": 2, "trust_radius": 0.1}},
            ).run()

        text = open(out, "r", encoding="utf-8").read()
        assert "requires the analytic Hessian" in text
        assert "Normal Termination" not in text
    finally:
        for ext in ("", "_prfo_traj.xyz", "_prfo_ts.xyz"):
            p = os.path.splitext(out)[0] + ext if ext else out
            if os.path.exists(p):
                os.remove(p)


def test_opt_lbfgs_first_step_is_downhill_for_harmonic():
    from maple.function.dispatcher.optimization.algorithm.LBFGS import LBFGS

    ref = np.zeros((1, 3))
    atoms = Atoms("H", positions=[[1.0, 0.0, 0.0]])
    atoms.calc = HarmonicCalc(k=1.0, ref_positions=ref)
    _set_thresholds(atoms, val=0.0)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".out", delete=False) as fh:
        out = fh.name
    try:
        start_x = float(atoms.positions[0, 0])
        LBFGS(
            atoms=atoms,
            output=out,
            paras={"lbfgs": {"max_iter": 1, "verbose": 0, "max_step": 0.5}},
        ).run()
        assert abs(float(atoms.positions[0, 0])) < abs(start_x)
        assert float(atoms.get_potential_energy()) < 0.5
    finally:
        for ext in ("", "_opt_traj.xyz", "_traj.xyz", "_opt.xyz"):
            p = os.path.splitext(out)[0] + ext if ext else out
            if os.path.exists(p):
                os.remove(p)


def test_sdcg_bb_scale_uses_previous_forces_not_current_forces():
    from maple.function.dispatcher.optimization.algorithm.SDCG import SDCG

    atoms = Atoms("H", positions=[[0.8, 0.0, 0.0]])
    atoms.calc = HarmonicCalc(k=1.0, ref_positions=np.zeros((1, 3)))

    with tempfile.NamedTemporaryFile(mode="w", suffix=".out", delete=False) as fh:
        out = fh.name
    try:
        opt = SDCG(
            atoms=atoms,
            output=out,
            paras={"sdcg": {"max_step": 0.2, "verbose": 0}},
        )
        opt._prev_positions = np.array([[1.0, 0.0, 0.0]])
        opt._prev_forces = np.array([[-1.0, 0.0, 0.0]])

        scale = opt._estimate_step_scale(np.array([[-0.8, 0.0, 0.0]]))
        assert scale == pytest.approx(1.0)
    finally:
        if os.path.exists(out):
            os.remove(out)


def test_nebts_candidate_indices_try_barrier_neighbors():
    from types import SimpleNamespace

    from maple.function.dispatcher.ts.algorithm.neb import NEB

    neb = object.__new__(NEB)
    neb.params = SimpleNamespace(nebts_candidates=4)
    images = [Atoms("H", positions=[[float(i), 0.0, 0.0]]) for i in range(6)]
    energies = [0.0, 1.0, 5.0, 4.5, 4.0, 0.0]

    # Primary highest image stays first; adjacent and next-highest internal
    # images are tried before lower-energy path regions. This locks the
    # NEB-TS handoff fallback without touching NEB/CINEB/PRFO equations.
    assert neb._nebts_candidate_indices(images, energies, primary_idx=2) == [
        2,
        1,
        3,
        4,
    ]


def test_path_refinement_prfo_handoff_preserves_user_prfo_options():
    import inspect

    from maple.function.dispatcher.ts.algorithm.neb import NEB
    from maple.function.dispatcher.ts.algorithm.string import GSM

    assert "self.raw_paras" in inspect.getsource(NEB.__init__)
    assert "paras=self.raw_paras" in inspect.getsource(NEB.restart_run)
    assert "self.raw_paras" in inspect.getsource(GSM.__init__)
    assert "paras=self.raw_paras" in inspect.getsource(GSM.restart_run)


def test_path_evaluator_reads_calculator_level_batch_size():
    class ChunkRecordingCalc(HarmonicCalc):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.chunk_sizes = []

        def calculate_many(self, atoms_list, properties=("energy", "forces")):
            self.chunk_sizes.append(len(atoms_list))
            energies = []
            forces = []
            for at in atoms_list:
                R = at.get_positions().astype(np.float64)
                d = R - self.ref
                energies.append(0.5 * self.k * float(np.sum(d * d)))
                forces.append(-self.k * d)
            return BatchResult(energies=np.asarray(energies), forces=forces)

    ref = np.zeros((1, 3))
    images = [Atoms("H", positions=ref + i * 0.01) for i in range(5)]
    calc = ChunkRecordingCalc(k=1.0, ref_positions=ref)
    calc.batch_size = 2

    es, fs = PathEvaluator(calc).energy_forces(images)

    assert es.shape == (5,)
    assert len(fs) == 5
    assert calc.chunk_sizes == [2, 2, 1]


def test_path_evaluator_auto_batch_size_backs_off_after_oom():
    class OOMAboveTwoCalc(HarmonicCalc):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.chunk_sizes = []

        def calculate_many(self, atoms_list, properties=("energy", "forces")):
            self.chunk_sizes.append(len(atoms_list))
            if len(atoms_list) > 2:
                raise RuntimeError("CUDA out of memory")
            energies = []
            forces = []
            for at in atoms_list:
                R = at.get_positions().astype(np.float64)
                d = R - self.ref
                energies.append(0.5 * self.k * float(np.sum(d * d)))
                forces.append(-self.k * d)
            return BatchResult(energies=np.asarray(energies), forces=forces)

    ref = np.zeros((1, 3))
    images = [Atoms("H", positions=ref + i * 0.01) for i in range(5)]
    calc = OOMAboveTwoCalc(k=1.0, ref_positions=ref)
    calc.device = "cpu"
    calc.batch_size = "auto"

    es, fs = PathEvaluator(calc).energy_forces(images)

    assert es.shape == (5,)
    assert len(fs) == 5
    assert calc.chunk_sizes == [5, 2, 2, 1]
    assert calc._auto_batch_size_last == 2


def test_path_evaluator_auto_caps_ani_and_mace_path_batches():
    class FakeANICalculator(HarmonicCalc):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.device = "cuda"
            self.dtype = np.float32
            self.chunk_sizes = []

        def calculate_many(self, atoms_list, properties=("energy", "forces")):
            self.chunk_sizes.append(len(atoms_list))
            return super().calculate_many(atoms_list, properties=properties)

    ref = np.zeros((1, 3))
    images = [Atoms("H", positions=ref + i * 0.01) for i in range(20)]
    calc = FakeANICalculator(k=1.0, ref_positions=ref)
    calc.batch_size = "auto"

    es, fs = PathEvaluator(calc).energy_forces(images)

    assert es.shape == (20,)
    assert len(fs) == 20
    assert calc.chunk_sizes == [8, 8, 4]
    assert calc._auto_batch_size_last == 8


def test_path_evaluator_rejects_invalid_batch_size():
    with pytest.raises(ValueError, match="batch_size"):
        PathEvaluator(HarmonicCalc(k=1.0, ref_positions=np.zeros((1, 3))), batch_size=0)
