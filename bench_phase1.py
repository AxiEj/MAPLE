"""Phase 1 wall-time + forward-pass comparison using real ANI2x.

Three calculators wrap the same ANI2x TorchScript model. They differ
only in their batch contract:

  LegacyANI       — pre-Phase-1: per-DOF Python loop for numerical
                    Hessian (each iteration = 1 single-structure forward
                    + autograd backward).
  Phase1ANI       — current main branch: same per-call cost, but
                    FDHessianEvaluator drives the displacement loop via
                    calculate_many's default sequential fallback.
  TrueBatchANI    — Phase-2 promise: overrides calculate_many to stack
                    the entire displacement list into a (B, N, 3) tensor
                    and evaluate them in a single ANI forward + autograd
                    backward.

We measure:
  (A) numerical Hessian on a 16-atom organic molecule (48 DOF -> 96
      force evaluations).
  (B) OPT-RFO with forced rejects (snapshot-restore path).
  (C) TS-PRFO (outer-loop E/F carry).
"""
from __future__ import annotations

import os
import time
import tempfile

import numpy as np
import torch
from ase import Atoms
from ase.calculators.calculator import all_changes
from ase.constraints import FixAtoms

from maple.function.calculator.calculator_base import CalcABC
from maple.function.calculator._batch_eval import FDHessianEvaluator
from maple.function.calculator._batch_types import BatchResult

torch.manual_seed(0)
torch.set_num_threads(4)

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "maple/function/calculator/model/ani2x.pt",
)
_ANI_MODEL = torch.jit.load(MODEL_PATH, map_location=_DEVICE)
_ANI_MODEL.eval()
for p in _ANI_MODEL.parameters():
    p.requires_grad_(False)


def _cuda_sync():
    if _DEVICE.type == "cuda":
        torch.cuda.synchronize()


class _ANIBase(CalcABC):
    """Shared ANI2x backbone — calls _ANI_MODEL directly."""

    implemented_properties = ['energy', 'forces', 'free_energy']

    def __init__(self):
        super().__init__()
        self.device = _DEVICE
        self.dtype = torch.float32
        self.calls = 0
        self.hessian = 'numerical'

    def calculate(self, atoms=None, properties=('energy',), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.calls += 1
        coords = torch.tensor(atoms.get_positions(), dtype=self.dtype,
                              device=self.device,
                              requires_grad='forces' in properties).unsqueeze(0)
        species = torch.tensor(atoms.get_atomic_numbers(),
                               dtype=torch.long, device=self.device).unsqueeze(0)
        e = _ANI_MODEL(species, coords)[0]
        self.results['energy'] = float(e.item())
        self.results['free_energy'] = float(e.item())
        if 'forces' in properties:
            f = -torch.autograd.grad(e, coords)[0].squeeze(0)
            self.results['forces'] = f.detach().cpu().numpy().astype(np.float64)


class LegacyANI(_ANIBase):
    """Pre-Phase-1: numerical Hessian via per-DOF self.calculate loop.

    Mirrors the old MAPLE ANI _get_hessian_numerical implementation
    exactly (including the at = atoms.copy() per displacement).
    """

    supports_batch_energy_forces = False

    def get_hessian(self, atoms, delta: float = 0.002):
        N = len(atoms)
        pos0 = atoms.get_positions().copy()
        fixed = {i for c in getattr(atoms, "constraints", []) or []
                 if isinstance(c, FixAtoms)
                 for i in c.get_indices()}
        movable = [i for i in range(N) if i not in fixed]
        H = np.zeros((3 * N, 3 * N), dtype=np.float64)
        for a in movable:
            for k in range(3):
                row = 3 * a + k
                at_p = atoms.copy()
                pp = pos0.copy(); pp[a, k] += delta
                at_p.set_positions(pp)
                self.calculate(at_p, properties=['forces'])
                Fp = self.results['forces'].copy()

                at_m = atoms.copy()
                pm = pos0.copy(); pm[a, k] -= delta
                at_m.set_positions(pm)
                self.calculate(at_m, properties=['forces'])
                Fm = self.results['forces'].copy()

                H[row, :] = (-(Fp - Fm) / (2.0 * delta)).reshape(-1)
        atoms.set_positions(pos0)
        return H


class Phase1ANI(LegacyANI):
    """Phase-1 main branch: FDHessianEvaluator drives the loop through
    calculate_many's sequential fallback. Per-call cost identical to
    LegacyANI; Python-level overhead reduced."""

    def get_hessian(self, atoms, delta: float = 0.002):
        return FDHessianEvaluator(
            self, fd_batch_size=getattr(self, "fd_batch_size", None)
        ).hessian(atoms, delta=delta)


class TrueBatchANI(Phase1ANI):
    """Phase-2 hypothetical: overrides calculate_many to stack all
    displacement geometries into a (B, N, 3) tensor and evaluate them in
    one batched ANI forward. All N atoms have the same species across
    displacements, so we can just broadcast the species table."""

    supports_batch_energy_forces = True

    def calculate_many(self, atoms_list, properties=('energy', 'forces')):
        self.calls += 1
        B = len(atoms_list)
        coords = torch.stack([
            torch.tensor(at.get_positions(), dtype=self.dtype, device=self.device)
            for at in atoms_list
        ])
        coords.requires_grad_('forces' in properties)
        species = torch.tensor(atoms_list[0].get_atomic_numbers(),
                               dtype=torch.long, device=self.device
                               ).unsqueeze(0).expand(B, -1).contiguous()
        e_batch = _ANI_MODEL(species, coords)[0]  # (B,)
        energies = e_batch.detach().cpu().numpy().astype(np.float64)
        forces_list = None
        if 'forces' in properties:
            grad = -torch.autograd.grad(e_batch.sum(), coords)[0]  # (B, N, 3)
            f_np = grad.detach().cpu().numpy().astype(np.float64)
            forces_list = [f_np[i] for i in range(B)]
        return BatchResult(
            energies=energies if 'energy' in properties else None,
            forces=forces_list,
        )


# ===========================================================================
# Test molecules
# ===========================================================================
def build_ethanol():
    """C2H6O — 9 atoms, 27 DOF."""
    return Atoms(
        'C2H6O',
        positions=[
            [ 1.20,  0.00,  0.00],
            [-1.20,  0.00,  0.00],
            [ 1.55,  1.02,  0.00],
            [ 1.55, -0.51,  0.88],
            [ 1.55, -0.51, -0.88],
            [-1.55,  0.51,  0.88],
            [-1.55,  0.51, -0.88],
            [-1.55, -1.02,  0.00],
            [ 0.00,  0.00,  0.00],
        ],
    )


def _build_alkane(n_C: int) -> Atoms:
    """Linear alkane CnH(2n+2): 3n+2 atoms, 9n+6 DOF.

    n_C=10  → C10H22  (32 atoms,  96 DOF, 192 force evals)
    n_C=20  → C20H42  (62 atoms, 186 DOF, 372 force evals)
    n_C=30  → C30H62  (92 atoms, 276 DOF, 552 force evals)
    """
    rng = np.random.RandomState(11)
    bond_C = 1.54
    bond_CH = 1.09
    a = 109.47 * np.pi / 180.0  # tetrahedral
    positions = []
    symbols = []
    # zig-zag backbone in xy plane
    for i in range(n_C):
        x = i * bond_C * np.sin(a / 2.0)
        y = (i % 2) * bond_C * np.cos(a / 2.0)
        positions.append([x, y, 0.0])
        symbols.append('C')
    # add hydrogens: end carbons get 3, middle carbons get 2
    for i in range(n_C):
        x, y, _ = positions[i]
        if i == 0 or i == n_C - 1:
            # 3 H out of plane + reverse direction
            for hx, hy, hz in [
                (-bond_CH * 0.6, -bond_CH * 0.5,  0.0),
                ( 0.0,            bond_CH * 0.3,  bond_CH * 0.85),
                ( 0.0,            bond_CH * 0.3, -bond_CH * 0.85),
            ]:
                positions.append([x + hx, y + hy, hz])
                symbols.append('H')
        else:
            for hz in (+bond_CH * 0.85, -bond_CH * 0.85):
                positions.append([x, y + (0.4 if i % 2 == 0 else -0.4) * bond_CH, hz])
                symbols.append('H')
    pos = np.array(positions) + 0.02 * rng.randn(len(positions), 3)
    return Atoms(symbols=symbols, positions=pos)


def build_C10H22():
    return _build_alkane(10)


def build_C20H42():
    return _build_alkane(20)


def build_C30H62():
    return _build_alkane(30)


def _set_thresholds(atoms, val=1e-3):
    for t in ('f_max_th', 'f_rms_th', 'dp_max_th', 'dp_rms_th'):
        setattr(atoms, t, val)


# ===========================================================================
# Benchmarks
# ===========================================================================
def time_hessian(calc_cls, build_atoms, n_repeats: int = 1, warmup: bool = True):
    calc = calc_cls()
    atoms = build_atoms()
    atoms.calc = calc
    if warmup:
        # Warmup: single force call. Avoids the full Hessian warmup which
        # dominates wall time for large systems on battery power.
        calc.calculate(atoms, properties=("forces",))
        _cuda_sync()
    calc.calls = 0
    t0 = time.perf_counter()
    for _ in range(n_repeats):
        H = calc.get_hessian(atoms, delta=1e-3)
    _cuda_sync()
    dt = (time.perf_counter() - t0) / n_repeats
    return dt, calc.calls // n_repeats, H


def run_opt_rfo(calc_cls, force_rejects: int = 0):
    from maple.function.dispatcher.optimization.algorithm.RFO import RFO
    calc = calc_cls()
    atoms = build_ethanol()
    # Add tiny displacement so the first step is non-trivial
    atoms.positions += 0.02 * np.random.RandomState(1).randn(*atoms.positions.shape)
    atoms.calc = calc
    _set_thresholds(atoms, val=1e-3)

    with tempfile.NamedTemporaryFile(mode='w', suffix='.out', delete=False) as fh:
        out = fh.name
    try:
        opt = RFO(atoms=atoms, output=out,
                  paras={'rfo': {'max_iter': 8, 'trust_radius_init': 0.2}})
        opt._reject_remaining = force_rejects
        orig_decide = opt._accept_or_reject

        def dec(rho, on_b):
            if opt._reject_remaining > 0:
                opt._reject_remaining -= 1
                return False
            return orig_decide(rho, on_b)
        opt._accept_or_reject = dec

        calc.calls = 0
        t0 = time.perf_counter()
        opt.run()
        dt = time.perf_counter() - t0
        return dt, calc.calls
    finally:
        for ext in ('', '_opt_traj.xyz', '_traj.xyz', '_opt.xyz'):
            p = os.path.splitext(out)[0] + ext if ext else out
            if os.path.exists(p):
                os.remove(p)


def run_ts_prfo(calc_cls):
    from maple.function.dispatcher.ts.algorithm.PRFO import PRFO
    calc = calc_cls()
    atoms = build_ethanol()
    atoms.positions += 0.04 * np.random.RandomState(3).randn(*atoms.positions.shape)
    atoms.calc = calc
    _set_thresholds(atoms, val=1e-2)

    with tempfile.NamedTemporaryFile(mode='w', suffix='.out', delete=False) as fh:
        out = fh.name
    try:
        calc.calls = 0
        t0 = time.perf_counter()
        PRFO(output=out, atoms=atoms,
             paras={'prfo': {'max_iter': 5, 'trust_radius': 0.1}}).run()
        dt = time.perf_counter() - t0
        return dt, calc.calls
    finally:
        for ext in ('', '_prfo_traj.xyz', '_prfo_ts.xyz'):
            p = os.path.splitext(out)[0] + ext if ext else out
            if os.path.exists(p):
                os.remove(p)


def main():
    print(f"Real ANI2x (TorchScript) benchmark — device: {_DEVICE.type.upper()}",
          f"({torch.cuda.get_device_name(0)})" if _DEVICE.type == "cuda" else "")
    print()

    sizes = [
        ("Ethanol  C2H6O",  build_ethanol),
        ("Decane   C10H22", build_C10H22),
        ("C20H42",          build_C20H42),
    ]

    print("=" * 92)
    print("Numerical Hessian wall time — ANI2x autograd backward, n_repeats=1 each")
    print("=" * 92)
    header = f"{'System':<18} {'N':>4} {'DOF':>5} {'#F evals':>9} " \
             f"{'Legacy':>11} {'Phase 1':>11} {'Phase 2':>11} {'P2 speedup':>11}"
    print(header)
    print('-' * 92)

    for sys_name, builder in sizes:
        N = len(builder())
        n_dof = 3 * N
        n_forces = 2 * n_dof
        dt_legacy, _, H_l = time_hessian(LegacyANI,    builder, n_repeats=1)
        dt_phase1, _, _   = time_hessian(Phase1ANI,    builder, n_repeats=1)
        dt_phase2, _, H_b = time_hessian(TrueBatchANI, builder, n_repeats=1)
        sp = dt_legacy / dt_phase2
        print(f"{sys_name:<18} {N:>4} {n_dof:>5} {n_forces:>9} "
              f"{dt_legacy*1000:>9.1f} ms {dt_phase1*1000:>9.1f} ms "
              f"{dt_phase2*1000:>9.1f} ms {sp:>9.2f}x")
        eq = float(np.max(np.abs(H_l - H_b)))
        print(f"   equiv max |dH| = {eq:.3e}")

    print()
    print("Notes:")
    print(" * Phase-1 row should match Legacy (calculate_many sequential fallback;")
    print("   the refactor itself does not change per-call cost).")
    print(" * Phase-2 = a single batched torch forward + autograd backward over the")
    print("   full displacement set, replacing all 2*3*N sequential force calls.")
    print(" * Equivalence is bounded by float32 + central-difference noise (1e-3 step).")


if __name__ == '__main__':
    main()
