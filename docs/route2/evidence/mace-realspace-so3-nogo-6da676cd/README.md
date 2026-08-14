# MACE-POLAR molecular-realspace SO(3) negative evidence

This directory retains a clean, official-checkpoint counterexample for the
molecular real-space inference path used by `MACE-POLAR-1-M` with
`graph-longrange==0.4.0`.  It is negative evidence only.  `E/F/H/V/M` all
remain disabled.

## Bound identity

- execution commit: `6da676cdef27c9f1d72b8fa84a49be326e13adb2`
- official checkpoint SHA-256:
  `fab8b8713c832f31a2a853aaa22fd638be8a369cbf5095e6b3e982a18d10e93a`
- measurement SHA-256:
  `8774134e8dfdbf540f62a17b6fafbd2778f61b2ffb42af6d79d951c3265112dd`
- device/dtype: CUDA / float64 model execution
- geometry/field: one water geometry, exact zero eight-channel field
- rotation: deterministic proper rotation, seed `20260815`

The primary and cold-replay JSON files have different wall-clock metadata and
output paths, but their complete `protocol`, `structural_evidence`,
`full_model_zero_field_rotation`, `isolated_upstream_realspace_operators`,
`decision`, and `measurement_sha256` values are exactly identical.

## Decisive measurements

| quantity | result |
| --- | ---: |
| anchored zero-field scalar rotation drift | `-8.888361298886593e-05 eV` |
| local MACE interaction-energy drift | `+1.4868177800053672e-09 eV` |
| electron-energy drift | `-4.337914875582327e-05 eV` |
| electrostatic-energy drift | `-4.5505951146815327e-05 eV` |
| energy-gradient source covariance relative error | `1.573371208574198e-04` |
| original density covariance relative error | `1.5733712085689075e-04` |
| fixed-field coordinate-gradient covariance relative error | `1.2075108741020767e-03` |
| fixed-field coordinate-gradient maximum absolute error | `3.092120585912461e-04 eV/Angstrom` |
| isolated real-space feature covariance relative error | `2.1280691879188778e-02` |
| isolated real-space feature maximum absolute error | `1.5412896871566772e-02` |
| isolated real-space Coulomb-energy rotation drift | `-5.3144358972190275e-05` |

The isolated operator checks take source coefficients emitted by the official
checkpoint at zero field and rotate them *exactly* in the declared `l=0+1`
representation before calling the upstream primitives.  Therefore the
counterexample does not depend on Route-2 continuum assembly or on a nonzero
external-field transform.

## First broken operator

The first isolated failure is:

```text
graph_longrange.realspace_electrostatics.RealSpaceFiniteDifferenceElectrostaticFeatures
```

The pinned upstream implementation represents vector sources and receivers by
scalar charges displaced along fixed laboratory `x/y/z` axes.  Its feature
stencil uses offset `0.1 Angstrom`; the corresponding real-space Coulomb-energy
stencil uses `0.02 Angstrom`.  A finite fixed-axis stencil is not closed under
the continuous `SO(3)` orbit.  The artifact binds the exact installed Python
source paths, class line numbers, byte counts, and SHA-256 digests rather than
inferring this from class names.

## Decision boundary

This counterexample is sufficient to reject a global structural `SO(3)` claim
for the current molecular-realspace model evaluator.  It does **not** show that
the checkpoint weights are unusable with a separately versioned analytic
Gaussian-multipole evaluator, and it does not validate such a replacement.
No principal-axis frame, finite rotation average, tolerance relaxation, or
continuum change is admitted as a repair.
