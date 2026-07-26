# Route-2 secondary energy diagnostic

## Route 2: MACE-POLAR-1-M/SMD/IEFPCM

[`route2-protocol.json`](route2-protocol.json) is a reproducible FreeSolv
diagnostic for the Route-2 energy proof-of-concept. It does not define Route 2,
certify analytic forces, or establish a complete solution-phase PES.
It pins the same FreeSolv v0.52 source hashes as route 1 while defining a
separate split seed, model/continuum/CDS composition, domain, standard state,
runtime measurement, and confirmation gates. The ten structures previously
inspected during route-1 development remain development-only; this prevents
prior human review from leaking into Route-2 confirmation.

Canonical commands:

```bash
PROTOCOL=docs/implicit-solvation/benchmarks/route2-protocol.json
WORK=.omx/benchmarks/route2-macepolar-smd

python docs/implicit-solvation/benchmarks/run_route2_freesolv.py \
  prepare --protocol "$PROTOCOL" --work-dir "$WORK"
python docs/implicit-solvation/benchmarks/run_route2_freesolv.py \
  run-supervised --protocol "$PROTOCOL" --work-dir "$WORK" \
  --partition development --device cuda
python docs/implicit-solvation/benchmarks/run_route2_freesolv.py \
  summarize --protocol "$PROTOCOL" --work-dir "$WORK" \
  --partition development --output "$WORK/development-summary.json"
```

`run-supervised` keeps the normal calculator-reuse path inside one worker, but
restarts that worker if PCMSolver terminates the process on a fatal cavity
check. It freezes the affected compound as an audited provider failure and
continues; unidentified worker exits are not relabelled. Long runs can also be
split into deterministic, disjoint shards that share the same work directory:

```bash
for SHARD in 0 1 2 3; do
  python docs/implicit-solvation/benchmarks/run_route2_freesolv.py \
    run-supervised --protocol "$PROTOCOL" --work-dir "$WORK" \
    --partition development --device cuda \
    --shard-count 4 --shard-index "$SHARD" &
done
wait
```

The runner parses the exact public three-line Route-2 input, initializes the
official MACE-POLAR-1-M calculator once through `SetCalculator`, and evaluates
each molecule through `ImplicitSolvationCorrection` plus the common calculator
finalizer rather than calling the SMD provider directly. Model-load time is
excluded. Each atomic record stores the public input, gas/solvation/combined
energy identity, non-thermochemical ASE `free_energy` marker, and the complete
PCMSolver audit directory. The summary reports MSE/MAE/RMSE/max error/failure
rate, deterministic bootstrap intervals, chemistry strata, component energies,
and the median/p90 ratio of integrated Route-2 wall time to gas MACE wall time.
A `--max-compounds` run is a smoke only and cannot be summarized as a complete
partition.

Before confirmation, freeze the exact reviewed proposal and pass rule:

```bash
python docs/implicit-solvation/benchmarks/run_route2_freesolv.py \
  freeze-confirmation --protocol "$PROTOCOL" --work-dir "$WORK" \
  --proposed-default 'macepol-m/smd-iefpcm-water/scf' \
  --pass-rule 'MAE <= 1.5 kcal/mol; failure_rate == 0; median total/gas <= 2.0'
```

The protocol also records separate open Dip146 (`MAE <= 0.25 D`) and HR46
(`MAE <= 2.0 A^3`) response gates. Those datasets are not silently folded into
FreeSolv, because hydration error alone cannot certify the learned
density/polarizability mechanism.

No Route-2 FreeSolv accuracy artifact is frozen yet. The real water and static
SMD-CDS controls in `VALIDATION_STATUS.md` prove execution, units, reciprocity,
and convergence only. Until development and one-shot confirmation pass, the
public input continues to require `experimental=true`; no constant shift,
fine-tuning, or confirmation-set refit is allowed.

[`route2-qm-fidelity-v1.json`](route2-qm-fidelity-v1.json) is a separate frozen
bounded pilot: three same-geometry Route-2/QM/experiment comparisons plus one
locked four-geometry electronic conformer panel. Its tests recompute energy
decomposition, MAE/RMSE, and ensemble statistics. It does not replace the
unrun FreeSolv confirmation partition, and its observed CUDA-versus-CPU timing
ratios are not hardware-normalized speedups. The recorded `.omx` source paths
and SHA256 values are witnesses to retained local raw evidence, not tracked CI
inputs. The JSON records evidence-generating execution heads separately from
the runtime-equivalence reference head; ancestry alone is not treated as
evidence that a result was rerun. All three fixed-conformer energies have clean
canaries at `5746f24`, `f3e9892`, and `e34abc5`, with no tracked
`maple/` runtime-source change among those heads. One
2-acetoxyethyl-acetate source-geometry analytic-force canary at `d72dfba`
reproduces the historical correction force within `2.66e-14 eV/angstrom`.
Its direct one-component `5e-4`-angstrom finite-difference pair was rerun
against the same runtime and differs from the analytic force by
`3.084e-6 eV/angstrom`; both displaced energies converge in 18 iterations
without PCMSolver or legacy `primary` warnings. One pre-registered central
C--C torsion pair at \(\pm0.5^\circ\) was then rerun at `d72dfba`. Its analytic
and central-difference generalized forces are `0.0452064347` and
`0.0451054120 eV/rad`, differing by `1.01023e-4 eV/rad` (`0.2235%`); both
energy-only points converge in 18 iterations without either warning class.
A separately locked \(\pm1.0^\circ\) pair on the same coordinate gives
`0.0450446898 eV/rad`, an absolute error of `1.61745e-4 eV/rad` (`0.3578%`).
Refining to \(0.5^\circ\) reduces the absolute error by
`6.07221e-5 eV/rad`, to `0.62458` of the coarse-step error. Its two roots also
converge in 18 iterations with no PCMSolver or legacy `primary` warnings; the
`151.35 s` wall time is not used as a speed claim. The original
\(0.5^\circ\) runner wrote both immutable point records before a NumPy-Boolean
JSON serialization error, so a read-only finalizer validated those records and
no scientific energy was rerun. The two steps establish a bounded refinement
trend, not an asymptotic convergence order.

The third pre-registered \(\pm0.25^\circ\) pair gives
`0.0451675753 eV/rad`; its analytic discrepancy falls to
`3.88594e-5 eV/rad` (`0.0860%`). Both roots converge in 18 iterations with no
PCMSolver or legacy `primary` warnings. However, the \(0.5^\circ\)-to-
\(0.25^\circ\) drift is `6.21634e-5 eV/rad`, not smaller than the preceding
`6.07221e-5 eV/rad`, and the observed central-difference order is `-0.0338`.
The pre-registered `1.5--2.5` smooth second-order gate therefore fails. The
runner's exit 1 records that scientific failure after both results were
written; it is not an execution failure. The `148.50 s` wall time is not a
speed claim.

Component decomposition gives order `2.008` for gas MACE and `2.001` for CDS,
but only `0.544` for solvent-intrinsic MACE and `0.212` for PCM polarization.
The failing behavior is localized to the self-consistent electrostatic
coupling block.

The completed pre-registered root-cause diagnostic reuses all six immutable
torsion states and performs exactly one independent \(+0.25^\circ\)
energy-only root, with no new geometry, finite-difference step, or force
evaluation. Energy reproduces to `2.22e-16 eV` and density to `2.22e-16 e`;
all 18 locked validity gates pass, and the PCMSolver and legacy `primary`
warning counts are zero.
Formal PySCF-1202-Lebedev mapping finds 19 changed active points on the
minus-side fine interval and 27 on the plus side. Frozen density carries
`73.8%` of the fine-drift L1 norm; fixed-center-density PCM and
reaction-map-through-MACE carry `67.6%` and `31.9%` of that block's
subcomponent L1 norm. The bounded classification is
`active-set-associated-explicit-continuum-geometry-response`.

This is an association, not an operator-only/cavity-only causal split, and it
does not authorize a production change. The four-geometry flexible panel,
closed-loop panel, and second-molecule force evidence remain historical. The
JSON freezes the reviewed numerical ledger, including the failed
pre-registered gate, the subsequent valid diagnostic, and arithmetic checks.

The converged MLIP--PCM/SMD total derivative is now implemented for the
explicit pyddx/PySCF single-point candidate. The next primary milestone is a
separately pre-registered upstream-backed smooth cavity/operator feasibility
profile with one same-energy forward/adjoint/coordinate-VJP/CDS definition,
followed by local smooth-force and translation/rotation canaries. The existing
optional PySCF SWIG control is not automatically promotable because its fixed
laboratory-frame quadrature has already shown nonmonotonic rotation residuals.
Do not add another finer pyddx point, retune the failed gate, rerun the bounded
closed loop, or advance to a second flexible molecule or short-NVE
conservation before the smooth-provider gate passes.
Running more FreeSolv records must not displace those PES gates, while the
bounded QM-fidelity panel must not be promoted into a broad accuracy claim.
