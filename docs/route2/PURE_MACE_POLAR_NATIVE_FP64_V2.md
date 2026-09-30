# Pure MACE-POLAR native-FP64 response V2

## Scope

This is a separately identified, programmatic research path. It does not replace
Torch V3, register a public calculator, inherit historical qualification, or
open scientific/release capabilities. The original 352-file V3 snapshot and
its positive and negative evidence remain unchanged.

The physical construction remains the official zero-field MACE-POLAR model,
ddPCM at `lmax=15`, Lebedev 1202 and `eta=0.1`, and legacy SMD-CDS. V2 fixes two
numerical-reproduction defects under its own identity:

1. The installed `graph-longrange==0.4.0` real-space feature kernel allocates an
   intermediate using the process default dtype. V2 requires an explicitly
   configured FP64 process before constructing the model and around every
   forward. The model weights and installed dependency are not modified.
2. PySCF's unsuffixed Fortran `DATA` literals are rounded to binary32 before
   being stored as doubles. V2 encodes those exact constants, while preserving
   explicitly double-precision constants such as the `0.4D0` probe. All live CDS
   arithmetic remains FP64, and energy conversion uses MAPLE's shared unit.

Neither correction is a parameter fit, output compensation, radius adjustment,
or tolerance change. Both change the historical numerical result slightly, so
their identities and evidence must remain separate from V1/V3.

## Explicit process contract

Use a dedicated, serial process. The calling application selects FP64 before
importing/building the MACE model; reusable MAPLE library code does not mutate
or restore Torch's global default:

```python
import torch

torch.set_default_dtype(torch.float64)

from maple.solvation.experimental.mace_polar_response_native_fp64_v2 import (
    build_smd_mace_polar_response_native_fp64_v2_pes,
)

pes = build_smd_mace_polar_response_native_fp64_v2_pes(
    symbols=("O", "H", "H"), solvent="water", device="cpu"
)
```

Ambient-dtype drift, mismatched dependency version/source, and mixed returned
tensor precision must fail closed. This is not a general thread-safety claim:
unrelated concurrent Torch calls that mutate process state are outside the
research execution contract. A future upstream dtype-local kernel would need
its own pinned runtime and fresh qualification.

The initial CDS domain is H/C/O with water, ethanol, or hexane. Native-literal
agreement is asserted only on certified regular topology; the code retains
Torch's explicit rejection of singular surface branches rather than copying
PySCF's rare radius-perturbation behavior.

## Verification boundaries

- The response and independent direct-autograd oracle must use the same
  corrected model, CDS, geometry, source, precision, and continuum settings.
- Direct-autograd full-Hessian/HVP and performance comparisons are limited to
  the existing water/methane resource envelope. The earlier one-time 16 GiB
  diagnostic authorization is not reusable.
- Acetone-10 response qualification requires real CPU and CUDA E/F/H/HVP,
  native-pyddx E/F, source witnesses, raw symmetry/translation checks, and the
  unchanged accepted-step total-force FD gates.
- The preserved `2e-5 Å` acetone topology rejection is negative evidence; the
  `1e-5` and `5e-6 Å` comparisons and their agreement remain required.
- Component CDS checks retain raw finite differences. The named asymmetric
  ethanol fixture additionally tests second-order convergence before fixed
  Richardson extrapolation; this does not replace total-PES raw FD gates.
- Performance requires newly frozen sources, one warmup, three repetitions,
  unchanged time/memory gates, and all failures retained. Historical mixed-
  default V1 timings cannot establish a V2 speedup.

Implementation, numerical qualification, performance qualification, independent
review, and scientific admission are separate outcomes. None follows merely
from the existence of this document or a passing unit-test panel.
