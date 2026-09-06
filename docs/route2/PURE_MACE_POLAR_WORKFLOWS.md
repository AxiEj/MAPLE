# Pure frozen MACE-POLAR: experimental MAPLE workflows

## Exact model and access boundary

The versioned input profile
`pure-macepolar-frozen-point-l1-ddpcm-smd-workflow-v1` routes directly to the
existing pure total PES, not the legacy SCF SMD correction:

```text
E(R) = E_vac(MACE-POLAR) + G_ddPCM[R,c0(R)] + G_PySCF-SMD-CDS(R)
c0(R) = MACE-POLAR(R, external_field=0)
```

It preserves the official MACE-POLAR-1-M CPU/float64 model, point-l<=1 source,
ddPCM lmax=15 / Lebedev=1202 / tolerance=1e-12 / eta=0.1 / nproc=1, and
PySCF 2.13.1 SMD-CDS. The 505-record development result belongs to this energy
identity; adding workflow access does not turn it into independent validation.
No MACE-MDP source, response fitting, or mutual electronic/PCM root is added.

The following tasks are explicitly available for experimental use:

| Task | Supported methods | Meaning of a completed task |
| --- | --- | --- |
| SP | standard SP; optional gradients | One total-PES evaluation |
| OPT | LBFGS, RFO | Optimizer convergence on the declared PES, not reference-geometry accuracy |
| FREQ | MW only | Signed numerical internal modes with reported uncertainty; not automatic minimum certification |
| TS | P-RFO only | Optimizer convergence **and** an uncertainty-resolved index-one internal Hessian |

The scope remains neutral singlets, the existing model element domain, no PBC,
and no D4 or independently supplied charges. MD, IRC, SCAN, other TS methods,
non-mass-weighted FREQ, and `treat_imag_as_real=true` remain unavailable for
this profile. Legacy profiles keep their previous task restrictions. Global
E/F/H/V/M scientific/release admission flags are not enabled by this interface.
The existing 16–500 Da and single connected covalent-graph checks also remain;
this does not newly admit dissociated fragments or arbitrary bond-breaking
domains just because a TS task can be requested.

## Input

Common header, with coordinates supplied normally:

```text
#model=macepolm
#device=cpu
#solv(method=smd,provider=pyddx,profile=pure-macepolar-frozen-point-l1-ddpcm-smd-workflow-v1,implicit=water,response=frozen,experimental=true)
```

Choose exactly one task line:

```text
#sp(verbose=1)
#opt(method=lbfgs,max_iter=128,max_step=0.1)
#freq(method=mw,treat_imag_as_real=false)
#ts(method=prfo,max_iter=40,recalc=8,hessian_update=bofill,trust_radius=0.1)
```

For a minimum-frequency calculation, use the converged OPT geometry rather
than assuming the input geometry is stationary. For TS, supply a plausible
transition-state guess. A step cap, a returned structure, or an optimizer-only
convergence message is not TS success. The pure-profile P-RFO postcheck reports
optimizer convergence, internal Hessian index, and uncertain modes separately.

Use the existing `#level` settings to select optimization stopping criteria.
The practical numerical policy does not silently relax those criteria.

Complete input files are under
[`examples/solvation/pure_frozen`](../../examples/solvation/pure_frozen).
The FREQ example uses the water geometry from the successful OPT canary.
Because an installed `maple` executable may still point to another editable
checkout, launch this working tree explicitly from its root:

```bash
PYTHONPATH="$PWD" LD_LIBRARY_PATH="${CONDA_PREFIX:+$CONDA_PREFIX/lib:}${LD_LIBRARY_PATH:-}" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m maple.main examples/solvation/pure_frozen/sp.inp /new/path/sp.out
```

## Tolerating numerical events without hiding failures

The new profile selects a versioned bounded-noise Hessian policy:

| Quantity | Prospective allowance |
| --- | --- |
| Coarse/fine displacement | 0.004 / 0.002 angstrom |
| Richardson Hessian component discrepancy | 0.05 eV/angstrom^2 |
| Raw Hessian antisymmetry | 0.05 eV/angstrom^2 |
| Signed endpoint energy-work residual divided by displacement | 0.003 eV/angstrom |

These are engineering consistency allowances, not physical force/Hessian error
guarantees. They were selected before new workflow results, not increased until
a failed molecule passed. Existing strict derivative policies remain unchanged.

An observed ddX topology-ID change is recorded rather than rejected solely on
the hash. Every signed endpoint at `+h,-h,+h/2,-h/2` must still satisfy the
energy-work check; Richardson disagreement and raw antisymmetry remain hard
limits. Nonfinite values, replay/configuration drift, missing columns, or
changed topology-observation metadata still fail. PySCF CDS's unobservable
internal surface is explicitly reported as partial topology coverage.

The policy records all displaced topology IDs, change flags, energy-work
residuals, step sizes, and derivative hashes. Raw Hessian asymmetry is checked
**before** symmetrization. It does not claim that all MLIPs are discontinuous
or that every topology change is harmless. MACE's gated radial cutoff can be
smooth despite a changing edge list; output consistency is the relevant test.
[MACE cutoff implementation](https://github.com/ACEsuit/mace/blob/4d2da09413ac1407f37cdbb6b81fa28e4c15655e/mace/modules/radial.py)

## FREQ and TS interpretation

For this profile only, mode analysis mass-weights the Hessian **before**
removing rigid translations/rotations. SVD identifies the rigid rank, including
five rigid modes for linear molecules. The Richardson component-error matrix
provides a non-cancelling numerical uncertainty envelope for internal
eigenvalues. Negative, positive, and uncertain modes remain distinguishable;
small negative signs are not flipped to make the result look stable.

P-RFO reports success only when convergence is achieved and exactly one internal
mode is resolved negative, with every other internal mode resolved positive.
Uncertain or wrong-index outcomes are failed TS validation, not normal
termination. A real reaction-path connection check is still required before
interpreting an arbitrary saddle as the intended chemical transition state.

Inherited ideal-gas RRHO translational/rotational/standard-state corrections in
the frequency report do **not** by themselves establish a full solution Gibbs
free energy. Solvation-energy MAE does not establish vibrational-frequency or
reaction-barrier accuracy.

## Reproducible validation

The label-free prospective protocol is
[`preregistrations/pure-mace-polar-workflows-v1.json`](preregistrations/pure-mace-polar-workflows-v1.json).
The runner is
[`run_pure_mace_polar_workflow_canary.py`](../../tools/route2_release/run_pure_mace_polar_workflow_canary.py).
It checks original-water SP replay, displaced-water OPT/FREQ, and a declared
NH3 inversion P-RFO fixture with initial/final internal-mode checks and two
downhill endpoint relaxations. It never opens experimental benchmark labels.

From the repository, with the existing pinned environment active:

```bash
PYTHONPATH="$PWD" LD_LIBRARY_PATH="${CONDA_PREFIX:+$CONDA_PREFIX/lib:}${LD_LIBRARY_PATH:-}" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python tools/route2_release/run_pure_mace_polar_workflow_canary.py --checkpoint /path/to/MACEPOLAR1Mmodel --output-dir /new/output/directory --tasks sp,opt,freq,ts
```

The output directory must be new. `result.json` records task-specific gates,
all computation-source hashes, runtime, checkpoint, measured errors and
failures. `source_unchanged=false`, an unrequested task, or a failed task cannot
be summarized as a successful full workflow suite. Unit-test quadratic models
are separate engineering evidence, not real-checkpoint scientific validation.

## Executed canaries and remaining verification gaps

The current [summary](evidence/pure-mace-polar-workflows-summary-v1.json) binds
the [water record](evidence/pure-mace-polar-workflows-water-v1.json) and
[NH3 record](evidence/pure-mace-polar-workflows-ammonia-ts-v1.json), each copied
byte-for-byte from its raw execution result. Both runs retained unchanged
computation-source hashes, and all 341 bound files were checked against the
current implementation before publication.

- SP reproduced the pre-interface water energy and force exactly.
- Water LBFGS OPT converged in 8 iterations; MW FREQ resolved all 3 internal
  modes positive. Its 36 stencil samples included 22 observed topology-ID
  changes, with maximum endpoint-work discrepancy `5.11e-4 eV/A`, below the
  prospective `0.003 eV/A` allowance.
- NH3 P-RFO converged in 3 iterations and passed the final exact Hessian check:
  1 resolved-negative and 5 resolved-positive internal modes. Both downhill
  endpoints converged, had opposite nitrogen heights (about `-0.378/+0.378 A`),
  and lay about `0.212 eV` below the model saddle. These are model-internal
  connection checks, not a reference reaction-barrier benchmark.
- RFO OPT is exposed by the shared Hessian interface but was not separately
  exercised by these real canaries.

Targeted regressions passed 171 tests. The full Route-2/solvation run with this
checkout on `PYTHONPATH` passed 2216 and skipped 12; one historical matched-QM
preregistration source-hash test remains failed. That older frozen manifest
already mismatched 6 files at the base commit; this interface work changes
4 more files bound by it. Its hashes were **not** refreshed to reuse old
evidence. The new pure-workflow records use their own current source snapshot.
Known independent-review findings were corrected and tested; final independent
re-review was unavailable after the native-agent usage limit.
