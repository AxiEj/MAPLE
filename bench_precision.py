"""Diagnose where the max|dH| ~1e-3 between Legacy and Phase 2 comes from.

Hypothesis: it's fp32 reduction-order non-determinism in CUDA, NOT an
algorithmic difference. To prove:

  (1)  Legacy vs Phase 1 (both sequential, same per-call forward) — if
       both go through the same single-structure forward path, max|dH|
       must be very small or zero.
  (2)  Phase 2 with different batch chunk sizes (B=1, B=2, B=full) —
       same algorithm, just batched differently. If max|dH| varies with
       chunk size, that proves it's fp32 reduction non-determinism.
  (3)  Relative error: |dH_ij| / |H_ij|. Should be at fp32 epsilon
       (~1e-7) for matrix entries that aren't near zero, and absolute
       on near-zero entries.
"""
import os
import numpy as np
import torch
from ase import Atoms

from bench_phase1 import (
    LegacyANI, Phase1ANI, TrueBatchANI, build_C10H22, _cuda_sync
)
from maple.function.calculator._batch_eval import FDHessianEvaluator


torch.manual_seed(0)


def hessian_with(cls, builder, fd_batch_size=None):
    calc = cls()
    if fd_batch_size is not None:
        calc.fd_batch_size = fd_batch_size
    atoms = builder()
    atoms.calc = calc
    return calc.get_hessian(atoms, delta=1e-3)


def diff(A, B, label):
    d = np.abs(A - B)
    max_abs = float(d.max())
    # relative only where |A| > 1e-6 to avoid dividing near-zero entries
    mask = np.abs(A) > 1e-6
    if mask.any():
        rel = (d[mask] / np.abs(A[mask])).max()
    else:
        rel = 0.0
    typ = float(np.median(np.abs(A[mask]))) if mask.any() else 0.0
    print(f"  {label:<42}  max|dH|={max_abs:.3e}  "
          f"max rel err={rel:.2e}  (typical |H|={typ:.2e})")


def main():
    builder = build_C10H22
    print(f"Decane C10H22 (32 atoms, 96 DOF)")
    print(f"ANI2x dtype: float32, central difference delta=1e-3\n")

    H_legacy  = hessian_with(LegacyANI,    builder)
    H_phase1  = hessian_with(Phase1ANI,    builder)
    H_p2_full = hessian_with(TrueBatchANI, builder)
    # Run Phase 2 with smaller chunks to test reduction-order sensitivity
    H_p2_b1   = hessian_with(TrueBatchANI, builder, fd_batch_size=1)
    H_p2_b8   = hessian_with(TrueBatchANI, builder, fd_batch_size=8)

    print("(1) Legacy vs Phase 1 — both sequential, both call ANI one geometry at a time:")
    diff(H_legacy, H_phase1, "Legacy  vs  Phase 1 (seq)")

    print()
    print("(2) Phase 2 with different chunk sizes — same algorithm, different batch:")
    diff(H_p2_full, H_p2_b1,  "Phase 2 (B=full)  vs  Phase 2 (B=1)")
    diff(H_p2_full, H_p2_b8,  "Phase 2 (B=full)  vs  Phase 2 (B=8)")
    diff(H_p2_b1,   H_p2_b8,  "Phase 2 (B=1)     vs  Phase 2 (B=8)")

    print()
    print("(3) Sequential vs batched — the eye-catching number from earlier:")
    diff(H_legacy, H_p2_full, "Legacy  vs  Phase 2 (B=full batched)")

    # How does the *eigenvalue spectrum* differ? That's what matters for
    # downstream RFO/PRFO (which only uses eigendecomposition).
    print()
    w_l, _ = np.linalg.eigh(0.5 * (H_legacy + H_legacy.T))
    w_p2, _ = np.linalg.eigh(0.5 * (H_p2_full + H_p2_full.T))
    print(f"(4) Eigenvalue spectrum diff (what RFO actually consumes):")
    print(f"    Legacy:  λ_min = {w_l[0]:.4e}   λ_max = {w_l[-1]:.4e}")
    print(f"    Phase 2: λ_min = {w_p2[0]:.4e}   λ_max = {w_p2[-1]:.4e}")
    print(f"    max |λ_legacy - λ_phase2| = {float(np.max(np.abs(w_l - w_p2))):.3e}")


if __name__ == '__main__':
    main()
