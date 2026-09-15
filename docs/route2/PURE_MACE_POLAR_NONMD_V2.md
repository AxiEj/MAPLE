# Pure MACE-POLAR: CPU/CUDA non-MD workflows

## What changes, and what does not

The v2 profiles expose the existing pure frozen-source total potential to
additional MAPLE tasks and to CUDA model inference. They do not retrain the
model, change the solvent parameters, or establish new physical accuracy.

```text
E(R) = E_vac(MACE-POLAR) + G_ddPCM[R,c0(R)] + G_PySCF-SMD-CDS(R)
c0(R) = MACE-POLAR(R, external_field=0)
```

The official MACE-POLAR-1-M checkpoint remains byte-bound. Inference uses
float64; ddPCM remains lmax=15, Lebedev=1202, tolerance=1e-12, eta=0.1,
nproc=1, with PySCF 2.13.1 SMD-CDS. CUDA moves the model and its derivatives
to the requested GPU; ddPCM/CDS and the NumPy molecular-mode analysis remain
on the CPU. This is not an all-GPU continuum solver or a speed guarantee.

CUDA-v2 also fixes deterministic Torch reductions, disables JIT executor
optimization during model calls, and uses `CUBLAS_WORKSPACE_CONFIG=:4096:8`.
This prevents both repeated-reduction roundoff and first-call JIT transitions
from invalidating exact source replay. The original replay checks are retained.
The CLI prepares the workspace before CUDA initialization; a programmatic user
with an existing CUDA context must set it before starting that context. Conflicting
workspace settings fail clearly. Torch flags are restored after each model call;
this is a serial inference contract, not a guarantee for concurrent foreign Torch
threads. See [PyTorch 2.12 deterministic algorithms](https://docs.pytorch.org/docs/2.12/generated/torch.use_deterministic_algorithms.html).

The old `pure-macepolar-frozen-point-l1-ddpcm-smd-workflow-v1` remains the
CPU-only SP/OPT/FREQ/P-RFO interface. Its historical evidence is not relabelled
as CUDA evidence. One input bug is intentionally fixed: task defaults must not
overwrite an explicit device, including a device line before `#freq`.

## Input profiles

For CUDA, use both the CUDA profile and an explicit device:

```text
#model=macepolm
#device=cuda:0
#solv(method=smd,provider=pyddx,profile=pure-macepolar-frozen-point-l1-ddpcm-smd-nonmd-cuda-v2,implicit=water,response=frozen,experimental=true)
#sp(verbose=1)
```

For CPU, replace the device with `cpu` and the profile with
`pure-macepolar-frozen-point-l1-ddpcm-smd-nonmd-cpu-v2`. A CUDA request must
fail clearly if CUDA or the requested index is unavailable; it must not silently
become a CPU calculation. Use `cuda:N` rather than a separate `gpuid` option.

The corresponding device-specific scalar IDs are
`route2-experimental-pure-macepolar-frozen-point-l1-ddpcm-smd-nonmd-cpu-v2`
and `route2-experimental-pure-macepolar-frozen-point-l1-ddpcm-smd-nonmd-cuda-v2`.

## Task surface

| Task | Methods | Interpretation |
| --- | --- | --- |
| SP | Energy; optional gradients | Same total potential |
| OPT | LBFGS, SD, CG, SDCG, RFO | Convergence is reported separately from execution |
| SCAN | Rigid; relaxed LBFGS, SD, CG, SDCG | Each requested point and inner optimization has its own outcome |
| FREQ | MW | Signed internal modes with numerical uncertainty |
| TS | P-RFO, Dimer | A claimed saddle requires convergence and a resolved internal index of one |
| Paths | NEB/CINEB, String/CI-String, AutoNEB | Path convergence is not saddle certification |
| Path refinement | NEBTS, StringTS | Requires the requested path stage and saddle postvalidation |
| IRC | GS, HPC, EulerPC, LQA | Starts along a resolved internal negative mode; both directions are reported |

NEB variants use `#ts(method=neb)` with optional `refine=cineb` or
`refine=nebts`. String variants use `#ts(method=string)` with optional
`refine=cistring` or `refine=stringts`. A path image or an iteration cap is not
a converged transition state. An honest bounded nonconvergence does not, by
itself, mean that the method is unavailable.

### Explicit exceptions

- **MD remains unavailable**, as requested.
- **Relaxed RFO SCAN remains unavailable:** the current RFO path has no
  validated constrained-Hessian treatment. Removing `FixInternals` would
  change the requested optimization problem, not fix it.
- **Non-MW/both FREQ remain unavailable:** the legacy unit-mass calculation
  is not a physical vibrational spectrum. Negative modes are never flipped
  into positive ones with `treat_imag_as_real`.
- The model domain is unchanged: neutral singlets, supported elements,
  16–500 Da, one connected covalent graph, nonperiodic geometries, no external
  charge override and no added D4. Path endpoints/images must have the same
  ordered elements and a single profile/scalar/configuration identity.

## Running this checkout

Examples are in [`examples/solvation/pure_nonmd`](../../examples/solvation/pure_nonmd).
They select CUDA explicitly and retain the declared model/domain restrictions.
From the checkout root, with the pinned environment active:

```bash
PYTHONPATH="$PWD" LD_LIBRARY_PATH="${CONDA_PREFIX:+$CONDA_PREFIX/lib:}${LD_LIBRARY_PATH:-}" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m maple.main examples/solvation/pure_nonmd/sp-cuda.inp /new/path/sp.out
```

This avoids accidentally importing a different editable MAPLE checkout.
The official checkpoint comes from the existing shared MACE cache; the shared
`calculator/model` directory is not a substitute for the byte-bound checkpoint.

## Verification and accuracy boundary

The v2 canary records requested/actual device, profile/scalar, checkpoint and
source hashes, and task-specific execution and numerical outcomes. CPU replay
and CPU/CUDA comparison use prospective tolerances of 1e-7 eV for energy,
1e-6 eV/angstrom for forces, and 1e-4 eV/angstrom^2 for the Hessian. Numerical
derivative error guards retain the original workflow policy.

**Validation status:** implementation validation is in progress. Until the
v2 evidence summary is present, the task table describes the intended v2
surface, not a claim of completed real-model validation for every method.

The existing 505-record MNSol development MAE of 1.285 kcal/mol is historical
solvation-energy evidence for the original exact CPU calculation; it is not a
new GPU benchmark or a bound on frequency, geometry, or barrier errors. The
confirmation partition remains sealed. Neither workflow access, numerical
parity, an IRC connection, nor inherited RRHO output establishes full solution
Gibbs free energies or scientific/release admission.
