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
`5be65f2f24132952af939384ee22e9f6bd724b082f1e359e60c774b022c2a5ab`.

This second hash is the post-commit clean-tree rerun at Git HEAD
`23075e2cd634e18480f7d4ef3e5b1c10b36d6c0b`.  The six-orientation clean-tree
rerun also reproduced the measurements above; its log SHA256 is
`433f0bfd42a1711ab9a43ef062a40e39b231f2049ad6ad4d750a43271f6f0957`.

## Reproducible diagnostic command

The repository now contains a fail-closed evidence runner.  Run it only from a
clean checkout and write the generated JSON outside that checkout:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python tools/route2_release/run_fixedbox590_water_pes_diagnostic.py \
  --checkpoint "$HOME/.cache/mace/MACEPOLAR1Mmodel" \
  --device cpu \
  --mode full \
  --output /tmp/route2-fixedbox590-water-pes.json
```

The JSON binds Git HEAD/tree, every loaded repository Python source, checkpoint
bytes, provider/configuration identities, runtime/dependency/hardware metadata,
raw cold/warm roots, translation/orientation records, all Cartesian finite
differences, topology hashes, and primal/adjoint residuals.  The command fails
on a dirty tree and refuses to write its artifact inside the repository.  Its
result remains a disabled diagnostic and cannot alter the capability registry.

The preregistered next-stage water path command is:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python tools/route2_release/run_fixedbox590_water_path_diagnostic.py \
  --checkpoint "$HOME/.cache/mace/MACEPOLAR1Mmodel" \
  --device cpu \
  --output /tmp/route2-fixedbox590-water-path.json
```

Its contract was committed before execution: seven distorted/equilibrium
geometries, three internal directions and three central-difference steps per
geometry, independent cold/multi-start roots, one fixed-topology hash, and a
four-subinterval-per-edge symmetric-stretch/bend loop evaluated cold and warm
in both directions.

### Clean-commit distorted-water and closed-loop result

The command above was executed without changing its preregistered panel at Git
HEAD `241e98b742e83277adb87cf4e05e01c735b00b24`. The full raw JSON, captured
warnings, command logs, hashes, and a scope-limited manifest are under
`docs/route2/evidence/fixedbox590-water-path-241e98b7/`.

| measurement | result |
| --- | ---: |
| panel geometries | 7 |
| directional central differences | 63 |
| largest absolute directional error | `1.8373047201e-5 eV/A` |
| largest applicable relative directional error | `3.7464585634e-4` |
| largest primal residual | `9.9280857080e-13` |
| largest adjoint residual | `3.5360817966e-16` |
| distinct topology hashes | 1 |
| cold forward loop work | `7.2197580638e-6 eV` |
| cold reverse loop work | `-7.2197580638e-6 eV` |
| warm forward loop work | `7.2197580415e-6 eV` |
| warm reverse loop work | `-7.2197580850e-6 eV` |

All preregistered directional-FD, panel cold/multi-start root, topology,
forward/reverse loop-work, work-antisymmetry, and warm path-repeat gates passed.
The run used CPU float64 and one OpenMP/MKL/OpenBLAS thread, took
`3195.20 s`, and the external JSON SHA256 was
`0f5cafad687262cb22e4ec202e0ce315add2c8c90fb6c9e954acc6e69f9e02d8`.
Seventy-three upstream/runtime warnings were retained verbatim rather than
suppressed.

This materially closes the **single-water** multi-geometry and loop gate. It
does not close the required multi-molecule PES panel, 40-Angstrom box-size
convergence, component-level QM/PCM validation, compatible nonpolar/free-energy
ledger, Hessian/FREQ/TS/NVE gates, or any public capability. E/F/H/V/M therefore
remain false.

## Result

The one-water force/symmetry/path blocker is materially improved and the
returned force is numerically consistent with the declared scalar over the
executed distorted-water panel. Tier F remains false because the required
multi-molecule PES panel, box convergence, physical component validation,
Hessian/NVE evidence, and public integration/admission have not been completed.
