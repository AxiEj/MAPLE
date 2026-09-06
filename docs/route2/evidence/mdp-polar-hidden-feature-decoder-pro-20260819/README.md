# Hidden-feature PCM-source identifiability decision

## Question

Could the unreleased `l >= 2` equivariant hidden features of the official
MACE-POLAR/MACE-MDP checkpoints be decoded, without training or independent
electrostatic supervision, into a physically identified density or
cavity-MEP source that improves the hybrid PCM calculation?

## Verified Pro review

The question in `prompt.md` was submitted through the dedicated Chrome window
only after UI Automation verified both:

- composer model pill: `Pro`;
- menu accessible name: `Pro, 5 of 5.`

The submitted text matched the frozen prompt byte-for-byte.  The answer ran
with the `Pro thinking` indicator and was extracted only after both the
thinking and stop controls disappeared.  `complete-uia.tsv` is the complete
post-run UI Automation capture; `answer.md` is the exact answer text range
from `Terminal decision` through the required terminal marker.

The answer selected option B and ended with exactly:

```text
HIDDEN FEATURES DO NOT IDENTIFY A PHYSICAL PCM SOURCE
```

## Independent verification

The local proof in `local-derivation.md` was written while the Pro answer was
still running.  It establishes the same theorem independently.  If the public
observable map is `O`, the continuum boundary map is `B`, and there is an
equivariant map `N` with

```text
O N = 0,       B N h != 0,
```

then `D_t = D_0 + t N` is an infinite family of equivariant decoders with the
same public charge/dipole/polarizability closure but different PCM-visible
sources.  The executable `l=2` counterexample confirms:

- rotation-covariance maximum error: `1.6653345369377348e-16`;
- unchanged total charge, molecular dipole, and uniform-field polarizability;
- surface-MEP maximum difference: `0.040885985178921114`;
- stationary quadratic-scalar difference: `-0.002121830138005043`.

Repeated hidden irreps also admit invertible multiplicity-channel gauge
changes.  Content-addressing one numerical checkpoint plus a hand-written
decoder makes a reproducible *new profile*; it does not make that decoder a
checkpoint-native physical observable.  Minimum norm, Coulomb norm, maximum
entropy, or another chosen variational rule fixes a convention only after a
metric/prior has been added.  It does not resolve physical identifiability.

## Terminal program decision

Do not implement a zero-training hidden-feature density decoder.  Finish the
already frozen canonical atomic-density-translation (ADT) development run.
If ADT fails its preregistered accuracy and physical gates, terminate the
current zero-training source-expansion program and move to a separately named
scalar-first, density, or cavity-MEP head trained only on independent QM
electrostatic data.  Such a successor is a new model and may not use
experimental solvation targets as a residual-fitting shortcut.

The only exceptions would be a checkpoint-bound trained physical decoder, a
proved architecture theorem assigning physical basis semantics to the hidden
block, or sufficiently complete supervised density/boundary-potential data.
None is present in the released checkpoint contract recorded in
`official-contract.md`.
