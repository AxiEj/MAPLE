# Fixed-box40 / C-PCM-590 water diagnostic

## Claim boundary

This is a disabled diagnostic for the unique operational scalar

```text
E_op(R) = E_vac(R) + 0.5 <c*(R), P_R(c*(R))>_Q
G_np = 0
```

It does not admit E, F, H, V, or M.  It is one equilibrium water geometry,
not the required multi-molecule/multi-geometry PES panel; the reciprocal
MACE-POLAR evaluator is fixed-box and has no box-convergence certificate; the
radial electrostatic component still lacks physical/chemical validation and a
compatible nonpolar/free-energy ledger.

## Why a new identity was required

The original 194-point candidate used the upstream molecular real-space
MACE-POLAR evaluator.  A same-scalar angular finite difference demonstrated
that its torque was a real derivative of a weak orientation-dependent scalar,
not a missing force term.  Changing either the MACE long-range evaluator or
the angular grid therefore changes the evaluation operator and must not be
hidden under the old profile.

The new disabled profile binds all of the following:

- model profile `mace-polar-route2-source-field-fixed-box40-contract-v1`;
- official MACE-POLAR-1-M checkpoint SHA256
  `fab8b8713c832f31a2a853aaa22fd638be8a369cbf5095e6b3e982a18d10e93a`;
- forced periodic reciprocal-space evaluation in a centered 40 Angstrom box;
- water dielectric 78.39 and versioned SMD-water Coulomb radii;
- fixed-topology amplitude-SWIG with Lebedev order 41, 590 candidates/atom;
- conjugate radial-GTO `B/B*`, the operational root, and its implicit adjoint.

## Executed evidence

The measurements below were executed locally on 2026-08-13 with Python
3.11.14, NumPy 2.4.6, PySCF 2.13.1, Torch 2.12.0+cu130, mace-torch 0.3.16,
graph-longrange 0.4.0, CPU float64, and one BLAS/OpenMP thread.  Temporary
diagnostic logs were SHA256 bound; they are summarized here rather than
presented as a release artifact because the working tree contained the code
under test.

### Lebedev-order isolation scan

| points/atom | rotation force relative | torque (eV) | total energy (eV) |
| ---: | ---: | ---: | ---: |
| 194 | 1.0371215208e-4 | 8.0903717844e-6 | -2079.881029476352 |
| 302 | 7.4488283728e-5 | 3.9864044538e-5 | -2079.881040770752 |
| 434 | 5.1924380295e-5 | 1.8610244527e-5 | -2079.881046800693 |
| 590 | 3.1858519055e-5 | 3.7249854942e-6 | -2079.881049681293 |

The 590-point total differs from the 194-point total by
`-2.0204940483381506e-5 eV`; this is a discretization change, not an accuracy
claim.  Log SHA256:
`efc788b8c20c9665e10fc8c837e7b2963d82860778b94b41f641d2ef7c416ca9`.

### Six independent rigid orientations

- maximum absolute energy change: `8.595316103310324e-7 eV`;
- maximum relative force-covariance error: `6.714619772947083e-5`;
- maximum relative source-covariance error: `4.5831179448529686e-7`;
- base torque: `3.7249854942383864e-6 eV`;
- base primal residual: `3.5420510710844027e-13`;
- base adjoint residual: `1.3001749579185527e-16`.

These pass the preregistered water symmetry thresholds, but do not establish a
general admitted domain.  Log SHA256:
`19627619921a26b20a44fcb6e1fc54460e74659c0939b8b857d40845f2802a6a`.

### Full Cartesian same-scalar finite difference

All nine Cartesian components were evaluated at three central-difference
steps.  Errors decrease by approximately four under step halving, consistent
with the expected second-order truncation regime.

| step (A) | RMS error (eV/A) | max error (eV/A) | relative Frobenius error |
| ---: | ---: | ---: | ---: |
| 4e-4 | 6.4775919470e-6 | 1.0930184224e-5 | 4.2913389267e-5 |
| 2e-4 | 1.6195112505e-6 | 2.7316579214e-6 | 1.0729415468e-5 |
| 1e-4 | 4.0439435924e-7 | 6.8302110562e-7 | 2.6791707935e-6 |

The largest displaced primal residual was `8.283839033072413e-13`; the base
adjoint residual was `1.3001749579185527e-16`.  Log SHA256:
`302cbb6fb15b77a27d785dc9a1cc01686f1c1fc5793c3f66f892fe971a9a1928`.

## Result

The one-water local force/symmetry blocker is materially improved and the
returned force is numerically consistent with the declared scalar at that
geometry.  Tier F remains false because the required multiple geometries,
distortions, molecules, closed loops in both directions, root-repeat panel,
topology paths, box convergence, physical component validation, and clean
source/model/runtime-bound release artifact have not yet been completed.
