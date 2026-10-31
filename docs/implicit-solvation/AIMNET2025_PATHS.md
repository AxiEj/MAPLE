# AIMNet2025 plus Torch OBC-II path tasks

## Scope

This experimental boundary opens three path tasks for one exact composed
potential.

- GS-IRC uses AIMNet2025 and independent Torch OBC-II with ACE or no nonpolar
  term. It requires an admitted analytic composed Hessian and rigid-mode
  projection for an isolated molecule.
- Standard NEB uses the same composed calculator on every image. The model
  configuration must explicitly declare `hessian=analytic` for receipt-bound
  enrollment, but NEB requests only energy and forces.
- String generates a Cartesian candidate path only when `refine=None`. The
  model configuration must explicitly declare `hessian=analytic`, but String
  requests only energy and forces.

The boundary does not open other IRC integrators, AutoNEB, climbing-image
refinement, PRFO/BPRFO refinement from a path, another gas model, OpenMM GB,
PB, an inner shell, constraints, periodic systems, charged systems, or
open-shell systems. The private Torch CHA provider remains single-point and
energy-only.

Every image must retain its own source MOL2 metadata. All images must have the
same ordered atoms, topology, fixed charge vector, positive mass vector, and
one exact composed calculator instance. Charge, radius, topology, model, and
provider preparation occurs once for the path.

## Units

- Positions and path displacements use angstrom.
- GS step input uses bohr and is converted with the existing GS conversion.
- Energies use hartree.
- Forces use hartree per angstrom.
- Cartesian Hessians use hartree per square angstrom.
- GS mass-weighted coordinates use square-root atomic-mass-unit angstrom.

String directions are normalized `3N` Cartesian displacements after Kabsch
alignment. They are not centroid translations. NEB and String do not alter the
physical potential or fitted parameters.

## Result contracts

Every opened task returns a small dictionary with `status`, `converged`, and
`termination_reason`.

GS-IRC also returns forward and backward TS-outward records, the exact input TS,
its index, and the ordered path `reversed(backward) + input TS + forward`.
Zero steps validates the input energy, force, Hessian, masses, and selected
internal negative mode. It returns the input TS once and performs no half-step.

NEB returns images, energies, projected-force diagnostics, and the iteration
count. A zero iteration budget is bounded execution. Convergence is true only
when the existing projected-force predicate is met.

String returns images, energies, growth iterations, and a path-generation
reason. `merged_path` means that path growth reached the existing merge
criterion. It does not certify a force-converged MEP, transition state,
reaction connection, or chemical accuracy.

The String writer uses `_gsm_candidate_path.xyz` and
`_gsm_equal_arc_candidate.xyz`. The file names do not claim an MEP.

## Qualification boundary

Source changes require a new immutable
`independent_obc2_analytic_qualification_v6.json` receipt. The v6 pointer is
unqualified until fresh ANI/OBC and AIMNet2025 evidence from one committed
source freeze passes the existing denominators and is sealed. Historical v1
through v5 receipts remain unchanged.

The new GS route retains the existing N32 raw-Hessian symmetry guard of
`2e-6 * max(1, max(abs(H)))`. The older PRFO `1e-8` behavior is outside this
path-only change and remains unchanged.
