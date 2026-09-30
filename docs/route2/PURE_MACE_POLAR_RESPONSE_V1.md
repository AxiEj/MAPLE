# Pure MACE-POLAR structured ddPCM response (prospective V1)

## Status and scope

This document describes a **programmatic, prospective research backend**. It is
not registered as a public calculator, has no admitted capability tier, and is
not a scientific or release qualification. `scientific_admitted` remains
`false`. The frozen MACE-POLAR checkpoint, radial-GTO point source, ddPCM
configuration (`lmax=15`, Lebedev 1202, `eta=0.1`, FP64), and legacy SMD-CDS
term are unchanged. No fit, scale, response tempering, radius change, or
solvent retuning is performed.

The backend is constructed only with
`build_smd_mace_polar_response_pes(...)`. It provides total energy, force,
directional Hessian-vector product, and raw full Hessian methods. The returned
full Hessian is not post-hoc symmetrized.

### Numerical-policy boundary identified on 2026-09-30

V1 retains the historical numerical path; it is not a fully FP64-qualified
backend. The installed `graph-longrange==0.4.0` real-space feature kernel uses
Torch's ambient default dtype for an internal allocation. Float64 model
parameters, coordinates, and returned tensors do not by themselves prove that
this intermediate is float64. The response configuration and benchmark now
bind the ambient default dtype explicitly and reject drift; they do not change
it or repair the old numerical path.

The frozen Torch CDS implementation also differs from the native PySCF scalar:
Fortran unsuffixed `DATA` literals undergo binary32 rounding before promotion
to double, unlike Python double literals. A separate private Hartree/eV
constant adds another difference. These are implementation-identity issues,
not grounds for fitting coefficients, changing physical radii, compensating
outputs, or relaxing qualification thresholds.

A separately identified native-FP64 V2 repair is under development. V1/V3
results and historical failures remain independent evidence and cannot be
relabelled as V2 qualification. The original 352-file V3 source snapshot is
unchanged.

## Scalar and derivatives

At fixed geometry and learned source, the continuum stationary system retains
the same scalar as the frozen direct-autograd implementation. With primal
states `y,z`, adjoints `lambda,mu`, geometry `R`, and embedded active source
`s`, the implementation differentiates the off-shell stationary Lagrangian in
the independent coordinates

```text
theta = (R, s).
```

The total coordinate Hessian is the exact outer composition

```text
H_total = P.T @ H_theta @ P
          + H(E_vac + G_CDS + g_s.detach().T @ c(R)),
P = [I; J_c].
```

The final term retains the learned-source curvature exactly once. The
directional route solves the structured tangent equations directly. The full
route reuses the central factors for batched primal tangents and performs no
adjoint-tangent solve or per-coordinate factorization.

## Identity and tensor boundary

The response result binds the geometry, active-source payload, configuration,
topology policy, derivative policy, implementation sources, factor/table
versions, source witnesses, and original-equation residual records. Public
arrays are copied and read-only. Private operators, factors, and evaluation
state are not exposed.

Ordinary Torch mutation remains possible only while building the private
evaluation graph. A returned result cannot be used to mutate the cached
operator state. The active source occupies columns `[0,2,3,4]`; inactive
columns `[1,5,6,7]` must be bitwise zero. Nonzero source JVP, VJP,
coordinate/source mixed response, and weighted-source-curvature witnesses are
qualification gates rather than inferred properties.

## Response-only topology amendment

The original v3 topology guard and its sealed acetone failure remain unchanged.
The response backend has a separate policy identity. Near `f=1`, it classifies
each switch from the actual distance ratio, rejects endpoint proximity, and
accepts only two local proofs:

1. one stable plateau-one contribution proves `f>=1`; or
2. exactly one interior contribution with all others plateau-zero proves
   `f<=1`.

Computed equality with an active interior derivative is rejected. Pure stable
plateau equality preserves both original inclusive derivative masks. No clamp,
rounding, tolerance relaxation, or discarded interior derivative is permitted.
This local certificate is not a global smoothness claim.

## Frozen qualification protocol

Prepare only after all candidate sources are stable, then use the resulting
fresh directory without overwriting rows:

```bash
python tools/route2_release/run_ddpcm_response_benchmark.py prepare \
  --output-dir .omx/research/ddpcm-response-20260926/<fresh-run>

python tools/route2_release/run_ddpcm_response_benchmark.py run \
  --output-dir .omx/research/ddpcm-response-20260926/<fresh-run> \
  --backend response --device cpu --case water --mode validation

python tools/route2_release/run_ddpcm_response_benchmark.py summarize \
  --output-dir .omx/research/ddpcm-response-20260926/<fresh-run>
```

The full matrix requires CPU and `cuda:0` rows, one warmup and three measured
repetitions for both backends on water and methane HVP/full-Hessian paths,
response validations for water, methane, acetone-10 and the hexane-20 attempt,
and response-only continuum-memory measurements for water, methane and
acetone-10. A host-wide nonblocking lock prevents overlapping benchmark
processes. Every CUDA measured window has before/after foreign-process checks.
All repetitions, failures, medians, MADs, and observed ranges are retained;
there is no outlier removal.

The 50- and 100-atom commands use `--mode resource` only. They execute static,
device-specific preflight arithmetic with named host/device objects; they do
not allocate continuum matrices and do not establish execution readiness,
matrix-free equivalence, or scalability. Continuum-memory rows compare only
the incremental sampled CPU USS or CUDA allocated peak against the continuum
estimate. That is neither an absolute continuum-memory measurement nor an
allocator guarantee.

No completion or performance values are stated here. They may be reported only
from a complete source-bound artifact that passes every accuracy, topology,
residual, translation, source, resource, timing, original-352-path integrity,
and independent-review gate.
