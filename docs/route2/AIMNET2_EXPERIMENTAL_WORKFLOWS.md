# Experimental AIMNet2 smooth-ddPCM molecular workflows

This is an **explicitly experimental workflow**, not formal E/F/H/V/M
admission, a chemical-accuracy certification, or original self-consistent SMD.
The formal scalar/profile registries remain disabled. Gas-phase AIMNet2 and
the existing MACE/legacy SMD selectors are unchanged.

## Exact input

Use the original checkpoint, not a replacement trained model. The required
file is 11,613,188 bytes with SHA256
`85ba59d8c78eb4d3185f6b1614df79706427f7ca72f53f2d90e365a1723d953d`.
The checkpoint is not redistributed or automatically downloaded by this lane.

```text
#model=aimnet2(model_path=/absolute/path/to/aimnet2.pt,hessian=numerical)
#device=cpu
#solv(method=smd,provider=aimnet2-smooth-ddpcm,profile=route2-profile-candidate-aimnet2-frozen-charge-multisolvent-smoothpartitionharmonic-ddpcm-pyscf-smdcds-v1,implicit=water,experimental=true)
#sp(verbose=1)

0 1
O 0 0 0
H 0.9572 0 0
H -0.2399872 0.927297 0
```

Task selectors (FREQ and TS also require an appropriate input geometry):

| Task | Task line | Required outcome |
| --- | --- | --- |
| SP | `#sp(verbose=1)` | Finite same-total-scalar E/F and passing domain guard |
| OPT | `#opt(lbfgs,max_iter=64,max_step=0.02)` | Actual optimizer convergence and fresh final force check |
| FREQ | `#freq(method=mw,verbosity=10)` | Stationary minimum, converged numerical Hessian, rigid-mode and positive-vibration checks |
| TS | `#ts(prfo,max_iter=24,trust_radius=0.02,trust_max=0.02,recalc=6)` | Actual convergence, fresh final numerical Hessian, exactly one robust imaginary vibrational mode |

See [example inputs](../../examples/aimnet2-smooth-ddpcm/). Replace the model
path in each input. `sp.inp` and `opt.inp` use water. `freq.inp` contains a
water geometry optimized in this development checkout; normally run FREQ on
**your own converged OPT geometry**, not the initial geometry. `ts.inp` is a
planar NH3 diagnostic, not a generally useful TS guess for other reactions.

From this checkout, explicitly select its Python sources:

```bash
PYTHONPATH="$PWD" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m maple.main examples/aimnet2-smooth-ddpcm/sp.inp
```

The normal installed `maple input.inp` command uses the same input route, but
an editable installation may select a different MAPLE worktree. No shared
installation is changed by this feature.

## Domain and scalar

- CPU, float64, one non-periodic, unconstrained molecule, H/C/N/O only.
- Explicit integer `0 1` charge/multiplicity; no external charge source, D4,
  alternate model module, analytic Hessian, SCAN/IRC/MD, or unlisted optimizer.
- The existing solvent table is used unchanged; supported syntax is not
  evidence of accuracy or derivative coverage for every solvent/molecule.
- The energy is `E_AIMNet2(R) + G_smooth-ddPCM(R,q_NQE(R)) + G_PySCF-SMD-CDS(R)`.
  AIMNet2 is reevaluated at each geometry but never receives a continuum field.
  There is no electronic SCF, charge scaling, radius fitting, or duplicated
  legacy SMD correction.
- Forces are the full derivative of this total energy, including the
  charge-position chain rule and CDS. Public ASE units are eV, eV/angstrom,
  and eV/angstrom². Only the existing private optimizer view converts to
  Hartree units; FREQ consumes the raw ASE Hessian.

## Outcomes, cost, and negative evidence

Each task writes `<output>.experimental.json` with task status, parameters,
geometry, identity, work counts, and failure details. `completed`/`converged`
means that task's checks passed, **not scientific admission**. Failed
optimization, an invalid Hessian, or a wrong saddle index raises through the
CLI and produces a nonzero exit status. Final structures and negative
diagnostics are retained rather than labeled a successful TS.

The public lane uses workflow `aimnet2-smooth-ddpcm-experimental-workflows-v2`
and schema `aimnet2-experimental-richardson-hessian-v2`. Numerical Hessians
evaluate the **total force** at guarded Cartesian
displacements. Raw matrices, endpoint forces/energies, convergence errors,
antisymmetry, center replay, and failing guards are saved in separate
`<output>.hessian-NNNN.json` files. No Hessian is cached across geometries, and
the center's accepted topology state is restored after stencil evaluation.
The analytic CDS HVP remains unimplemented; numerical support does not change
that fact.

V2 recomputes four raw central-difference Hessians at
`h = 4e-4, 2e-4, 1e-4, 5e-5 angstrom`. Adjacent matrices form
`K = (4 H_fine - H_coarse)/3`, cancelling the leading second-order truncation
term. Three successive K matrices must satisfy a refinement-ratio limit of
0.25 (or a declared `1e-7 eV/angstrom²` plateau). The terminal error estimate
must be at most `5e-5 eV/angstrom²`; both raw finest-H and final-K
antisymmetry must be at most `1e-4 eV/angstrom²`. Center replay must agree
within `1e-10 eV` and `1e-9 eV/angstrom`. Only then is final K symmetrized.
Each complete Hessian costs **24N+2 total E/F evaluations**. It is not an
analytic Hessian and does not reuse old stencil results.

The separate v1 three-step method is retained. Its fixed planar-NH3 initial
Hessian failed the `5e-5 eV/angstrom²` truncation-error gate with an estimate
of `1.18587e-4`; that failure was **not** relabeled a pass. V2 is a separately
preregistered numerical method with an additional, independently evaluated
finest step, not relaxed v1 tolerances or changed physical parameters.

OPT/TS use an effective maximum-force target no looser than `1e-5 eV/angstrom`
and a maximum LBFGS displacement / PRFO trust radius of `0.02`. FREQ retains
its independent stationarity and rigid-mode checks. Negative vibrational
modes are never reinterpreted as positive by this experimental lane.

An OPT with iteration cap `I` uses at most `I+2` E/F evaluations: initial
center, up to `I` iterations, and a forced final replay. A TS additionally
accounts for recalculated Hessians and its mandatory fresh final Hessian;
the existing PRFO algorithm permits up to eight trust-step attempts per
iteration. The calculator enforces the declared work limits before excess
evaluations, rather than merely reporting cost afterwards.

Frequency thermochemistry uses MAPLE's existing ideal-gas RRHO convention
applied to the effective-potential modes. It is not a validated solution-phase
binding free energy or a new standard-state correction.

## Validation status

[Final real-task results and verification details](AIMNET2_EXPERIMENTAL_WORKFLOW_VALIDATION_2026-09-10.md)
record successful SP, OPT, FREQ and TS canaries for the frozen final source.

The retained historical 653-record MNSol result (MAE 2.5817 kcal/mol) remains
a fixed-geometry accuracy result for its recorded source identity, not an
accuracy claim for OPT/FREQ/TS. See the [September 6 progress report](AIMNET2_PROGRESS_2026-09-06.md).
Current numerical protocol and real-task results are recorded separately;
failed development protocols are not overwritten by later numerical repairs.
