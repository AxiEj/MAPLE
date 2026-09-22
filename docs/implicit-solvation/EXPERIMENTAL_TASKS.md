# Experimental Route 1 tasks

Experimental availability is separate from chemical-accuracy certification.
A failed external benchmark or incomplete coverage does not, by itself, block
trying a task on a correctly identified, force-capable potential.

## Currently available paths

Runtime charge choices are fixed MOL2, AM1-BCC and explicit ABCG2.
**QEq/CQEq is disabled at user request (2026-09-08)**, including both fixed
and polarizable QEq and explicit experimental selection.
The polarizable CQEq-GTO/GB code and ddX/ddPCM reference runners were removed
from this branch on 2026-09-09; archived results do not provide runnable paths.
The subsequent user-requested **ddLPB** numerical-reference provider is a new,
explicit positive-kappa path, not restoration of those old runners. See
[numerical reference usage](NUMERICAL_REFERENCE.md).

| Implicit-solvent provider | SP | OPT | FREQ | TS |
|---|---|---|---|---|
| OpenMM GB, fixed charges | Yes | Yes | Numerical composed Hessian; analytic ANI2x/OBC-II/ACE or none on Reference | Experimental P-RFO / dimer, one solute geometry; analytic profile is separately gated |
| AmberTools CHA-GB/PBSA | Energy only | Not implemented | Not implemented | Not implemented |
| ddX/ddLPB reference | Polar energy + analytic force | Experimental | Numerical composed Hessian | Experimental P-RFO / dimer, one solute geometry |
| APBS PB | Energy only | Not implemented | Not implemented | Not implemented |

Existing supported nonperiodic OpenMM GB SCAN/MD paths are unchanged. This
implicit TS entry does not add NEB/string/AutoNEB, multicomponent
`inner=prebuilt`, polarizable charge response, periodic boundaries or analytic
implicit Hessians. Gas-phase TS paths are unchanged.

The analytic profile is an execution-backend rewrite, not a new solvent
model. It shares the existing fixed charges and mbondi2 parameters, reproduces
the pinned OpenMM OBC-II/ACE energy and force, and obtains the solvent Hessian
or HVP by Torch automatic differentiation. OpenMM CPU remains the ordinary
SP/OPT/SCAN/MD default. Analytic FREQ/P-RFO/Dimer currently requires ANI2x,
float64 (automatic promotion is allowed), `model=obc2`, `nonpolar=ace|none`,
and explicit `platform=Reference`; other backend/device/profile cells retain
the complete-force numerical fallback.

## P-RFO input

With an appropriate TS-guess MOL2 and available ANI2x model:

```text
#model=ani2x(hessian=numerical)
#ts(method=prfo,max_iter=100,project_rigid_modes=true)
#device=cpu
#charge(source=maple,method=am1bcc,geometry=keep)
#solv(implicit=water,method=gb,provider=openmm,model=obc2,nonpolar=ace,experimental=true)

0 1
MOL2 ts-guess.mol2
```

ANI2x here is a tested syntax/dispatch example, not a reaction-accuracy
recommendation. Choose a gas model appropriate for the chemistry.

For a free isolated molecule, `project_rigid_modes=true` removes rigid
translations/rotations from the mass-weighted P-RFO search subspace and mode
tracking. **Full Cartesian forces still determine convergence.** This avoids
spurious steps along noisy rigid near-zero modes without changing the energy
model. It is explicit and defaults to false: do not enable it for a potential
that depends on the laboratory frame (for example an external field), periodic
systems or constrained coordinates. The latter two are rejected when enabled.

The [foundation validation](FOUNDATION_VALIDATION.md) now includes genuinely
converged ammonia inversion saddles, rather than only iteration-limit smoke
runs. Numerical qualification and its retained negative results remain
separate from task availability.

Alternatively, use `#charge(source=mol2,label=your-charge-provenance)` for an
already prepared fixed-charge MOL2. The usual topology, charge, element and
external-provider requirements still apply. Optimization uses the selected
MLIP plus the **same** OpenMM GB energy and force; the numerical Hessian
uses that composed force. No OBC-II force is substituted into CHA-GB energy.

The output identifies the experimental candidate-search boundary. A completed
program run or an iteration-limit exit is not a converged transition state.
Inspect convergence and vibrational curvature before interpreting the result
as a saddle, and establish reaction connectivity separately when needed.
These are interpretation checks, not prerequisites for starting the task.

## Force-only dimer input

The same solvent and fixed-charge setup also accepts:

```text
#model=ani2x
#ts(method=dimer,max_iter=100,n_init=force)
#device=cpu
#charge(source=mol2,label=your-charge-provenance)
#solv(implicit=water,method=gb,provider=openmm,model=obc2,nonpolar=ace,experimental=true)

0 1
MOL2 ts-guess.mol2
```

Dimer estimates directional curvature by central differences of the **complete
composed force**, without constructing a full Hessian. For implicit solvent,
omit `use_hvp` or set `use_hvp=false`; `use_hvp=true` is rejected because the
existing analytic HVP omits solvent. The legacy default displacement is
`delta=0.005` angstrom for backends without a recommendation. ANI numerical
curvature now selects float64 and `delta=0.0005`; an explicit delta overrides
the recommendation. Gas-phase automatic mode retains the existing calculator HVP path.
Use an unconstrained, nonperiodic single-solute geometry. Coordinate constraints
are not supported by this force-difference path.

The output reports force, rotation residual and negative-curvature candidate
checks; it does not certify a first-order saddle without final vibrational
analysis. Frame energies correspond to the coordinates written in each frame.

## ddLPB workflow precision

The ddLPB reference now also opens OPT, mass-weighted numerical FREQ and
single-geometry P-RFO/dimer. See [inputs and the ANI precision repair](NUMERICAL_REFERENCE.md).
The same polar-only scalar is used throughout; this does not open CHA/APBS
forces or claim full solvation free energies.

## Progress without over-gating

- Reject structural, unit, non-finite-value and unsupported derivative errors;
  do not silently change the model to avoid them.
- Report limitations, approximation and convergence instead of calling an
  exploratory result certified or hiding a failed calculation.
- Broader accuracy, solvent coverage and speed work can proceed independently.
  There is no mandatory experimental error threshold for new exploratory work.

CHA still needs a usable polar derivative implementation. Continuous nonpolar
kernels and precision-only executable builds do not supply that missing term.
This is a distinct implementation gap, not a requirement to perfect every
benchmark before exposing the already force-capable OpenMM route.

## Verification for this opening (2026-09-08)

The real ANI2x/OBC-II/ACE water smoke runs one P-RFO iteration and records
finite composed energy, forces and a 9x9 numerical Hessian. Each solvent
contribution is nonzero relative to the gas-only control. The output explicitly
says `Maximum Iterations Reached`; this is workflow evidence, not a water TS.
Its input, output and summary are local under
`.omx/benchmarks/route1-experimental-prfo-20260908/`.

A separate real ANI2x/OBC-II/ACE water run executes one dimer iteration with
finite composed energy and forces, without a full Hessian. Regression tests
also make gas-only HVP and full-Hessian calls fail explicitly while running
this task, and check serialized energies against their frame geometries.
Local raw evidence is under
`.omx/benchmarks/route1-experimental-dimer-20260908-final/` (the earlier
pre-typing-cleanup run is retained in the corresponding unsuffixed directory).
This is likewise workflow evidence, not a converged water transition state.

The final targeted parser, derivative, dispatch, documentation and archival
compatibility suite passes **112 tests**, including a gas-mode threshold and
step-metric preservation regression. The full suite on the same implementation
reports **927 passed and 2 unchanged historical reserve-file-hash failures**;
the final additional threshold-preservation test was run in the targeted suite.
Scoped typing is clean. Dimer/dispatcher/tests pass lint; the parser retains
the same 56 pre-existing whole-file lint findings as HEAD.
Historical source compatibility records
are audit-only; they do not authorize old evidence to
be rerun or resealed with new code.
