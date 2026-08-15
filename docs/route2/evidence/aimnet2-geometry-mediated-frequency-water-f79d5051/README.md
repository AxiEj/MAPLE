# AIMNet2 geometry-mediated stationary-water Hessian canary

Status: **diagnostic gates passed; no capability or MAPLE workflow admitted**.

Two independent clean CPU processes reproduced scientific measurement
SHA-256
`126327853eb0caadcf5e98b41992030789bbdf0f7ddf23cec9c4c31ed98b99a4`
from source commit `f79d50513a39cb4fd6e77b8d3a80dc4b65aa12f8` and tree
`c086a9592d6f1b2c98d58cc506aed0992aa20075`. The unchanged local AIMNet2
checkpoint is bound by SHA-256
`85ba59d8c78eb4d3185f6b1614df79706427f7ca72f53f2d90e365a1723d953d`.

## What was executed

1. The existing frozen water geometry was mapped to its two O-H lengths and
   H-O-H angle while preserving center of mass and laboratory-frame embedding.
2. SciPy's MINPACK hybrid root solver searched for zero internal gradient.
   Every consecutive solver trial was required to retain the same AIMNet2
   neighbor and harmonic-cavity topology and to pass the conservative
   point/source-shell and sphere-tangency segment certificate.
3. The complete weak-scalar Cartesian Hessian was assembled from nine canonical
   HVP columns. No public calculator Hessian/HVP entry point was opened.
4. Every Hessian column was independently checked against central differences
   of the total scalar gradient at `8e-4`, `4e-4`, `2e-4`, and `1e-4` angstrom.
5. The Hessian was mass-weighted as
   `M^-1/2 H_cart M^-1/2`; translations and rotations were then removed in that
   same mass-weighted space before the three water vibrational modes were
   diagonalized.

## Retained measurements

- stationary internal coordinates:
  `(1.0226804300596355 A, 1.0226804300596355 A, 2.0118160254551998 rad)`;
- final Cartesian gradient norm:
  `5.375312708523219e-12 eV/A`;
- PCM stationarity residual:
  `7.961471427347596e-15 eV/e` with condition number
  `91.25858490311596`;
- maximum reciprocity absolute error: `1.7763568394002505e-15 eV`;
- Hessian symmetry maximum error:
  `1.2434497875801753e-14 eV/A^2`;
- all-column HVP/gradient-FD Frobenius errors:
  `(1.4897589772074288e-3, 3.69929014813758e-4,
  9.454998028117011e-5, 4.074765566486307e-5) eV/A^2`;
- first two halving ratios are approximately `0.25`; the terminal refinement
  continues downward while approaching the float64 graph's numerical floor;
- translation HVP norms:
  `(7.303296316030796e-16, 1.6249875634344887e-14,
  1.5302438523914105e-14) eV/A^2`;
- stationary rotation HVP norms:
  `(4.0596458768953666e-9, 2.7068386143688735e-9,
  6.719144789488478e-9) eV/A^2`;
- projected vibrational eigenvalues:
  `(9.869399590890731, 25.428929759819567,
  29.47965103286058) eV/(A^2 amu)`;
- implementation frequencies:
  `(1638.2321451446662, 2629.6268042951565,
  2831.3349242547506) cm^-1`.

The frequencies are **not** physical solvent predictions or an accuracy
benchmark. They belong to a water-only, conductor-reference, point-charge,
`surface_lmax=1` research scalar without a finite-dielectric solvent identity
or nonpolar term.

## Scientific boundary

Analytical PCM Hessians and response corrections are established techniques
([Garcia-Rates et al., 2019](https://onlinelibrary.wiley.com/doi/abs/10.1002/jcc.25833)),
and analytical polarizable-embedding/PCM derivatives likewise require a common
response ledger
([Giovannini et al., 2015](https://pubmed.ncbi.nlm.nih.gov/26598266/)). This
artifact tests that local implementation structure; it does not inherit a
production claim from the literature. Recent point-charge PCM Hessian work
also reports discretization-induced instability and derives a Gaussian-charge
alternative
([Hashimoto and Nakai, 2026](https://www.sciencedirect.com/science/article/abs/pii/S0009261426002198));
MAPLE does not invent an AIMNet2 solute width from that result.

The current profile still fails its full-domain v2 panel because bond-stretched
methanol has only `0.015511399564898554 A` point/source-shell margin, below the
frozen `0.02 A` guard. Therefore this local stationary-water result does not
admit Tier H, FREQ, OPT, TS, IRC, MD, public ASE, finite-dielectric solvation,
or any `E/F/H/V/M` capability.

## Files

- `measurements.json`: first clean process, raw operands plus independently
  recomputed summary;
- `cold-replay.json`: second clean process with identical scientific fields;
- `SHA256SUMS`: byte hashes for the retained bundle.
