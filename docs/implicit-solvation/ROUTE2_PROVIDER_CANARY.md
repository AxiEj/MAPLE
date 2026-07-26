# Route-2 rotation-covariant continuum-provider canary

## Decision being tested

Route 2 already uses pyddx 0.8.0 **ddPCM** for its explicit single-point force
candidate.  The failed three-step torsion gate therefore cannot be repaired by
renaming the current provider as “ddPCM/ddCOSMO”.  The bounded diagnosis
associates the fine-interval drift with the explicit continuum geometry
response and changes in the finite cavity active set; it does not establish an
operator-only or cavity-only cause.

The next experiment tests one narrower hypothesis:

> Evaluate the unchanged pyddx ddPCM scalar in a molecule-following standard
> frame derived from the nuclear-charge moment tensor.  This should remove the
> laboratory-frame Lebedev orientation as a source of rigid-rotation variance
> without changing radii, dielectric, ddPCM equations, grid order, or solver
> tolerance.

This is a host-side **research adaptation** of the rotationally invariant
quadrature construction of Johnson, Gill, and Pople (JGP94).  That paper
establishes the frame construction and its complete orientation-matrix
derivatives for atom-centred numerical quadrature; it does not itself validate
this ddPCM application.  The candidate is provisionally called
`ddpcm-ri-jgp94-v1` in this document only.  It is not accepted by the MAPLE
parser or calculator factory.

## Why this candidate, and why not the alternatives

| Candidate | Evidence and useful property | Reason it is not the present repair |
|---|---|---|
| Current pyddx 0.8.0 ddPCM | Open, maintained Python API; same provider owns scalar energy, reaction map, adjoint, and analytic coordinate terms | This is already the Route-2 candidate. Its finite laboratory-frame grid failed the locked smooth-torsion test, and ddX issue 165 independently reports noisy/discontinuous ddCOSMO surface-area gradients. |
| PySCF SWIG/ISWIG | Published smooth switching and an analytic same-energy gradient implementation | It retains atom-centred Lebedev grids fixed in the laboratory frame. Orders 47 and 53 were nonmonotonic across rigid acetone orientations, so changing only the switching function is not fundamental repair. |
| PCMSolver/GePol | Existing default energy path and stable fixed-conformer infrastructure | The public v1.1.12-style interface has no complete boundary/operator coordinate derivative. A missing derivative may not be set to zero. |
| FIXPVA/CPCM | Published continuous electrostatic CPCM surface and exact analytic gradients | The paper still reports finite-grid rotational variance, and no maintained open callable host API was identified. It remains an external oracle, not a MAPLE provider. |
| JGP94 molecule-following frame around pyddx | Rigorous rigid-rotation construction, small \(O(N)\) frame overhead, and published first/second orientation derivatives | Selected only for the one-molecule scalar canary below. A complete frame VJP and a fail-closed high-symmetry domain are still absent; no Route-2 speed claim follows from JGP94. |

Simply increasing the Lebedev order is rejected: the observed acetone
rotation error was nonmonotonic from order 47 to 53 while cost and memory
increased.  Post-hoc zero-torque projection is also rejected because it would
not differentiate the implemented scalar energy.

## Frame definition

For nuclear charges \(Z_A\), positions \(\mathbf R_A\), and center of nuclear
charge

\[
\mathbf T=\frac{\sum_A Z_A\mathbf R_A}{\sum_A Z_A},
\]

JGP94 defines the translationally invariant nuclear-charge moment tensor

\[
\mathbf M
=\sum_A Z_A\left[
\left|\mathbf R_A-\mathbf T\right|^2\mathbf I
-(\mathbf R_A-\mathbf T)(\mathbf R_A-\mathbf T)^\mathsf T
\right].
\]

Let the ordered orthonormal eigenvectors form \(\mathbf O\):

\[
\mathbf O^\mathsf T\mathbf M\mathbf O
=\operatorname{diag}(\lambda_1,\lambda_2,\lambda_3),
\qquad
\mathbf O^\mathsf T\mathbf O=\mathbf I.
\]

The present scalar candidate evaluates the complete fixed-density ddPCM
problem in body coordinates

\[
\mathbf y_A=\mathbf O^\mathsf T(\mathbf R_A-\mathbf T)
\]

(written in the runner's row-vector convention as
`Y = (R - T) @ O`).  Cartesian atomic dipoles are transformed by the same
matrix; monopoles and radii are unchanged.  This is equivalent to making each
atom-centred quadrature follow the molecular orientation,

\[
\mathbf r_g=\mathbf R_A+\mathbf O\,\mathbf s_g.
\]

For an active rigid rotation \(\mathbf Q\), distinct eigenvalues imply
\(\mathbf O'=\mathbf Q\mathbf O\mathbf D\), where \(\mathbf D\) represents
the remaining signed-axis convention.  A proper signed-axis transformation is
an exact symmetry of a Lebedev grid.  The canary nevertheless aligns that
symmetry explicitly when comparing geometry, density, and active
sphere/Lebedev pairs.

## Derivative obligation and fail-closed boundary

Rigid-rotation invariance of the scalar is necessary but not sufficient for a
Route-2 PES.  If \(x\) is a nuclear coordinate, JGP94 gives

\[
\frac{\partial\mathbf r_g}{\partial x}
=\delta_{A,x}\mathbf e_x+\mathbf O^x\mathbf s_g.
\]

Rotating a provider gradient back would retain only the first contribution.
The second term is required.  With
\(\mathbf P^x=\mathbf O^\mathsf T\mathbf O^x\),
\(\mathbf P^x\) is antisymmetric and, for \(i\ne j\),

\[
P^x_{ij}
=\frac{
\left(\mathbf O^\mathsf T\mathbf M^x\mathbf O\right)_{ij}
}{
\lambda_j-\lambda_i
}.
\]

Therefore the future coordinate VJP must differentiate the center
\(\mathbf T\), orientation \(\mathbf O\), body-frame positions, transformed
dipoles/reaction fields, and the already implemented ML--SCF fixed-point
response.  It must contract these terms without forming a dense molecular
Jacobian.

At degenerate or nearly degenerate eigenvalues, the orientation derivative is
singular or ill-conditioned and axes may switch.  JGP94 explicitly identifies
this high-symmetry limitation.  MAPLE must fail closed there unless a later
symmetry-preserving construction is separately derived and validated.  The
canary fixes

\[
\frac{\min_i(\lambda_{i+1}-\lambda_i)}
{\max(1,\max_i|\lambda_i|)}
\ge 0.05.
\]

The value `0.05` is a predeclared engineering conditioning guard, not a fitted
solvation parameter and not a claim of universal sufficiency.

## One-shot acetone scalar canary

The tracked runner is
`benchmarks/run_route2_jgp94_ddpcm_canary.py`.  Its immutable scientific
scope is:

- molecule: one fixed FreeSolv acetone geometry (`mobley_3867265`);
- density: the already converged frozen MACE-POLAR Route-2 density;
- provider constants: water dielectric `78.39`, `eta=0.1`, `shift=0`,
  `lmax=15`, 1202 Lebedev points per sphere, one thread, and ddX tolerance
  `1e-12`;
- rigid orientations: identity; 37 degrees about normalized `(1,2,3)`; and
  113 degrees about normalized `(-2,1,0.5)`;
- exactly three unchanged laboratory-frame controls and three body-frame
  candidates;
- one forward/adjoint scalar response per case, plus one cavity-only
  one-sphere model used to identify Lebedev indices;
- no MACE call, new SCF root, CDS term, force, finite difference, geometry
  optimization, parameter scan, retry, or second molecule.

The laboratory-frame span is informational: an accidental small value for the
three selected orientations neither passes nor fails the hypothesis.

### Frozen gates

| Gate | Required value |
|---|---:|
| Relative moment-tensor eigengap | `>= 0.05` |
| Frame orthogonality and determinant errors | `<= 1e-12` |
| Signed-axis-aligned body geometry/density errors | `<= 2e-12` |
| Surface-owner and Lebedev-index reconstruction errors | `<= 1e-10 bohr` / `<= 1e-10` |
| Surface-owner second-best gap / Lebedev second-neighbour distance | `>= 1e-8 bohr` / `>= 1e-3` |
| Half-coupling energy identity | `<= 1e-10 eV` |
| Body-frame energy span across three rotations | `<= 1e-10 eV` |
| Body-frame versus current identity-orientation energy shift | `<= 0.01 kcal/mol` |
| Candidate cavity size and active `(sphere, Lebedev)` signature | identical; every pair unique |
| Evaluation budget | exactly `3 + 3`, no retry |

Before any solve, `freeze` requires a clean committed checkout and writes an
exclusive lock in the runner's fixed protocol directory
`.omx/benchmarks/route2-ddpcm-ri-jgp94-acetone-20260726`; the CLI exposes no
alternate work directory. The lock contains the Git head and SHA256 values of
the runner, document, MOL2, frozen state, and compiled pyddx extension. `run`
writes an attempt marker before constructing a continuum solve. A crash or
failed gate does not permit a retry with different output location,
orientations, gates, or parameters.

The non-scientific preflight is:

```bash
PYTHONPATH="$PWD:/home/axie/.cache/maple-envs/psi4-ddx-qmref/lib/python3.11/site-packages" \
  /home/axie/miniconda3/envs/maple/bin/python \
  docs/implicit-solvation/benchmarks/run_route2_jgp94_ddpcm_canary.py \
  preflight
```

After review and a clean preregistration commit, the only authorized execution
sequence is:

```bash
PYTHONPATH="$PWD:/home/axie/.cache/maple-envs/psi4-ddx-qmref/lib/python3.11/site-packages" \
  /home/axie/miniconda3/envs/maple/bin/python \
  docs/implicit-solvation/benchmarks/run_route2_jgp94_ddpcm_canary.py \
  freeze

PYTHONPATH="$PWD:/home/axie/.cache/maple-envs/psi4-ddx-qmref/lib/python3.11/site-packages" \
  /home/axie/miniconda3/envs/maple/bin/python \
  docs/implicit-solvation/benchmarks/run_route2_jgp94_ddpcm_canary.py \
  run
```

## Frozen outcome

The protocol was frozen and run exactly once on 2026-07-26 at Git commit
`da89ab8da2796b74409277c606c8d3bfffb3a861`.

- lock SHA256:
  `a2f590b00602527ccf38be4dc4c6f585b93828eb286393f50c582113fbce5308`;
- attempt SHA256:
  `d879acb49543cec0882b8bf3f65db223b21f69c7a4a73f1740d80d329ce5a665`;
- raw/tracked result SHA256:
  `f1b7b01eb80b42489c1306b2006f919e5cc79bc295f73c84824626abccbeeeb7`;
- tracked result:
  `benchmarks/route2-ddpcm-ri-jgp94-acetone-v1.json`.

All frozen gates passed:

| Metric | Observed | Frozen gate |
|---|---:|---:|
| Laboratory-frame rotation span | `7.1672e-6 eV` (`1.6528e-4 kcal/mol`) | informational |
| Molecule-frame rotation span | `2.1094e-15 eV` (`4.8644e-14 kcal/mol`) | `<= 1e-10 eV` |
| Identity-profile shift | `1.4473e-4 kcal/mol` | `<= 0.01 kcal/mol` |
| Maximum half-coupling identity error | `2.2760e-14 eV` | `<= 1e-10 eV` |
| Candidate active pairs | `4993 / 4993` unique in every orientation | identical and unique |
| Maximum owner residual | `8.8818e-16 bohr` | `<= 1e-10 bohr` |
| Minimum owner second-best gap | `5.2961e-5 bohr` | `>= 1e-8 bohr` |
| Maximum Lebedev direction error | `3.3766e-16` | `<= 1e-10` |
| Minimum Lebedev second-neighbour distance | `5.2523e-2` | `>= 1e-3` |
| Provider energy cases | `6` | exactly `3 + 3`, no retry |

The six scalar cases took `11.35 s`; the complete process took approximately
`13 s` on this host. These one-shot observations are not a performance claim.
The result supports the scalar hypothesis only within this fixed-density,
nondegenerate acetone case.

## Interpretation and stop condition

A **failure** rejects this candidate without retuning or adding another
orientation.  A **pass** establishes only fixed-density scalar
rigid-rotation invariance and a negligible one-geometry profile shift.  It
does not establish:

- the analytic frame VJP;
- smooth Cartesian or torsional derivatives;
- self-consistent MACE--ddPCM forces or CDS derivatives;
- solution-phase geometry optimization, energy conservation, or a PES;
- QM/experiment accuracy or broad chemical-space applicability;
- a public MAPLE profile.

A pass authorizes only a separately pre-registered analytic frame-VJP canary.
The failed torsion test, closed loop, second flexible molecule, NVE, and broad
FreeSolv runs remain frozen until that derivative stage and its local
smoothness gates pass.

## Primary sources

- B. G. Johnson, P. M. W. Gill, and J. A. Pople, “A rotationally invariant
  procedure for density functional calculations,” *Chem. Phys. Lett.* **220**,
  377--384 (1994), DOI `10.1016/0009-2614(94)00199-5`.
- M. Nottoli et al., “ddX: Polarizable continuum solvation from small
  molecules to proteins,” *WIREs Comput. Mol. Sci.* **14**, e1726 (2024),
  DOI `10.1002/wcms.1726`.
- A. Mikhalev, M. Nottoli, and B. Stamm, “Linearly scaling computation of
  ddPCM solvation energy and forces using the fast multipole method,”
  *J. Chem. Phys.* **157**, 114103 (2022),
  DOI `10.1063/5.0104536`.
- ddX upstream issue 165, “ddCOSMO Surface-Area (Gradient) Is
  Noisy/Discontinuous,”
  `https://github.com/ddsolvation/ddX/issues/165`.
