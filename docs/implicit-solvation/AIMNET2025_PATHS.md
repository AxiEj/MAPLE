# AIMNet2025 plus Torch OBC-II path tasks

## Execution domain

The v7 implementation supports these tasks after source-bound qualification.
An unsealed or stale receipt does not admit them.

| Task | Method and refinement | Result |
|---|---|---|
| IRC | `gs`, `hpc`, `lqa`, `eulerpc` | Ordered path and exact input TS |
| TS | `prfo` | Certified TS or explicit non-success candidate |
| NEB | `neb`, `refine=None` | Base path stage |
| CI-NEB | `neb`, `refine=cineb` | Base and CI stages |
| NEB-TS | `neb`, `refine=nebts` | Base, CI, then PRFO after CI convergence |
| String | `string`, `refine=None` | Candidate path growth |
| CI-String | `string`, `refine=cistring` | Growth, ordinary String, then CI-String |
| String-TS | `string`, `refine=stringts` | String stages, then PRFO after CI convergence |
| AutoNEB | `autoneb`, optional `climb=true` | Adaptive single path with base and optional CI stages |

The exact composed cell uses AIMNet2025 N32 and independent Torch OBC-II
FP32 with ACE or no nonpolar term. Omitted device selects CPU; CUDA must be
explicit. Receipt-bound enrollment requires `hessian=analytic`. Path stages
use energy and forces; IRC and PRFO also consume analytic curvature.

Images retain source MOL2 metadata and share ordered atoms, topology, fixed
charges, positive masses, and one prepared composed calculator. The domain is
one isolated, neutral, closed-shell solute. This extension does not admit PBC,
constraints, inner shells, polarizable charges, PB, another gas model, or BPRFO.
Private Torch CHA remains single-point energy-only. Previously admitted ANI
analytic PRFO, numerical PRFO, and Dimer retain their separate boundaries.

## Parameters and ownership

`dispatcher/method_registry.py` owns method names, input arity, parameter rules,
and required capabilities. CLI parsing and direct dispatch use that definition.
Unknown keys, case collisions, mixed flat and nested parameters, and inactive
refinement controls fail before model work. String's nested `prfo_params` uses
the same PRFO rules as standalone PRFO.

AutoNEB inserts images into one path using spatial and energy resolution. It is
not the former recursive tree. `n_max` is at least three. Base and CI budgets
are separate; budget exhaustion is not convergence.

Ordinary String uses full-force evolution and equal-arc reparameterization.
CI-String freezes its initial CI tangent for the CI stage. Each update moves
the climbing image, then holds its updated position fixed during separate
left and right reparameterizations. CI-String is not an alias for PRFO.

## Units and curvature

Positions use angstrom, energies use hartree, forces use hartree per angstrom,
and Cartesian Hessians use hartree per square angstrom. IRC parameters named
in bohr retain their method-specific conversion. Mass-weighted coordinates
use square-root atomic-mass-unit angstrom.

`dispatcher/curvature.py` checks the raw Hessian before forming its physical
symmetric part. Exact qualified N32 sources retain the raw-skew budget
`2e-6 * max(1, max(abs(H)))`. Unknown sources retain the strict generic rule;
forged or drifted analytic identity is refused. NumPy storage precision does
not establish backend computation precision. Physical equations, fitted
parameters, and raw backend Hessians remain unchanged.

## Results and output lifecycle

IRC returns TS-outward records and the ordered path
`reversed(backward) + [input_ts] + forward`, with TS index `len(backward)`.
Zero steps validates energy, forces, curvature, masses, and the selected
internal negative mode, then returns the input TS once without a half-step.

NEB and String retain dictionary results and add immutable stage records.
AutoNEB returns `AutoNEBResult` with insertion counts and stage records.
Stages distinguish `converged`, `bounded_nonconverged`, `failed`, and `refused`.
Final geometry, energy, forces, and metrics describe the same point. A stage
can converge with zero updates only if its actual initial state passes that
stage's convergence checks.

Low-level PRFO returns `PRFOResult`. High-level PRFO returns `Atoms` only for
a certified result and retains the typed result in `TransitionState.last_result`.
Other outcomes remain typed candidates. Certification requires force and
displacement convergence plus fresh index-one curvature at the final geometry.
A refined candidate stays separate from the MEP. An HEI or climbing image is
not automatically a certified TS.

Certified TS outputs are archived before a new attempt and published atomically
only after certification. Failed reruns cannot present an old TS file as current.
Implicit wrappers restore caller-visible calculator caches and preserve job
convergence thresholds across working copies. Engine results are cleared at
the start of each new attempt.

## Qualification and scientific claims

The v7 receipt is `independent_obc2_analytic_qualification_v7.json`. It binds the
unchanged 11 numeric gates, 40 cutoff refusals, frozen task matrix, and canonical
software-contract rows to one source identity. The sealer independently replays
canonical rows. Runtime admission requires sealed numeric evidence and the
matching software capability. Historical v1 through v6 receipts are retained.

Implementation, source tests, sealed numeric qualification, and installed
CPU/CUDA execution are separate evidence stages. A bounded task check is not
chemical convergence, reaction connectivity, solvation accuracy, or a performance
result. An unqualified pointer remains fail-closed even when code and software
tests are present.
