# CHA segment preconditions: coordinate-only exact characterization

This unregistered research component characterizes two necessary coordinate
facts along an affine three-site path: exact squared pair-distance quadratics
and the branch of the radius-weighted electrostatic size. It is deliberately
not a whole-segment continuum certificate and does not enable OPT, FREQ, TS,
MD, provider routing, or public admission.

## Exact arithmetic identity

Finite CPU `float64` coordinates and topology CHA radii are decoded with
`as_integer_ratio()`. Expanded radii are formed by the point code's actual
rounded `float64` addition of each CHA radius and `float64(1.4 - 0.52)`, then
decoded. Radius powers, the affine weighted center, inertia entries, and the
degree-at-most-six polynomial for `size(t)^6` use exact rational arithmetic.
The sphere moment uses the exact decoded value of the promoted float literal
`2.0 / 5.0`. This proves the declared real-valued model, not bitwise equality
to Torch reductions, determinants, sixth roots, or branch comparisons.

Positive radius-cubed weights and the positive sphere moment make the inertia
matrix positive definite for every path point. Exact Bernstein coefficients
bound `size(t)^6` on dyadic subintervals. The fixed limits are depth 12, 8191
nodes, and 8192 integer bits; exhaustion is unresolved rather than success.
All pair, guard, polynomial, determinant, interval-composition, and Bernstein
operations pass through a local checked rational tracker. Multiplication uses
cross-cancellation before bounded integer products, and the reported maximum
is the maximum tracked stored numerator, denominator, or explicit arithmetic
intermediate, not an input-size heuristic. Successful results never retain an
over-budget tracked integer. Integer multiplication can form at most a
one-bit-over-budget product before the typed resource rejection records it.
Python `Fraction` ordering is outside that counter: comparing two admitted
rationals may use cross-products bounded by twice `max_integer_bits`, because
each admitted numerator and denominator is at most the frozen bound. Reports
record this `comparison_scratch_bound_bits` separately. Pending Bernstein
stack nodes are reserved before a split, so budget exhaustion still leaves an
exact partition of `[0,1]`.

The `1e-8 Å` distinct-center guard is reported separately from the
probe-expanded inner/outer sphere relations. A nonzero separation can be
inside the distinct-center guard even when unequal radii make the expanded
inner relation appear stable.

## Scientific boundary

The omitted polar dependency is explicit: contact and torus integration gives
the R6 flux `J_i` in Å⁻³; the unshifted inverse Born radius is `J_i^(1/3)` in
Å⁻¹; `B_i=[J_i^(1/3)+shift(size)]^-1` is in Å; and the weighted CHA sign uses
dynamic `B_i B_j` denominators. None is evaluated here. SAV/SAS ownership,
triple/torus/tube guards, chart decisions, adaptive quadrature smoothness,
dispersion sigma regularity, force accuracy, gas-model behavior, and optimizer
rollback/cache semantics are also omitted.

Every report therefore fixes:

- `dynamic_born_sign_status=UNRESOLVED`
- `computed_quadrature_smoothness=UNRESOLVED`
- `optimizer_eligible=false`
- `public_admission=false`
- `segment_certificate=false`

The strongest result is `SEGMENT_PRECONDITIONS_CHARACTERIZED`.

## Reproducible diagnostic

The runner is intentionally two-phase. `preregister` atomically writes the
ordered 283-case inventory, input hashes, live source hashes, constants,
budgets, and status rules before any case is characterized. `run` rechecks the
externally frozen protocol-file SHA256 before parsing, then its internal hash,
approved plan/handoff identities, and all input/source hashes. It refuses
output overwrite and verifies source stability afterward. Protocol,
source-result, topology, and protected-manifest JSON are parsed from the same
single byte buffers that were hashed; the three run inputs are hashed again
after characterization and must remain unchanged.

```bash
python docs/implicit-solvation/benchmarks/characterize_cha_segment_preconditions.py \
  preregister --source-result full-v4/result.json --topology topology.json \
  --output prospective-segment-protocol.json

python docs/implicit-solvation/benchmarks/characterize_cha_segment_preconditions.py \
  run --protocol prospective-segment-protocol.json \
  --expected-protocol-sha256 <sha256-of-protocol-file> \
  --source-result full-v4/result.json --topology topology.json \
  --output segment-preconditions.json
```

The 283 cases are five singleton centers, all 270 center-to-stencil paths, and
eight directed adjacent-center paths. Historical `EVENT` annotations remain
external annotations; a coordinate precondition result never reclassifies a
weighted-sign event as safe.
