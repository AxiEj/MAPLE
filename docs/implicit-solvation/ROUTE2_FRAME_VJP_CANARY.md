# Route-2 JGP94 analytic frame-VJP canary

## Decision being tested

The preceding one-shot acetone scalar canary established that evaluating the
unchanged pyddx 0.8.0 ddPCM scalar in the Johnson--Gill--Pople
nuclear-charge principal-axis frame reduces the selected rigid-orientation
energy span from `1.6528e-4 kcal/mol` to `4.8644e-14 kcal/mol`.  That result did
not establish a derivative.

This stage tests exactly one follow-up question:

> Does the complete reverse derivative of the molecule-following frame match
> central finite differences of that same fixed-density ddPCM scalar, including
> the frame dependence of both atomic positions and atomic dipoles?

It is a benchmark-local research canary.  It does not create a public MAPLE
profile, replace the current PCMSolver path, call MACE, solve an ML--SCF root,
evaluate SMD CDS, or expose forces.

## Differentiated scalar and row-vector convention

For nuclear charges \(Z_A\), laboratory positions \(\mathbf R_A\), and the
center of nuclear charge

\[
\mathbf T
=\frac{\sum_A Z_A\mathbf R_A}{\sum_A Z_A},
\qquad
\mathbf C_A=\mathbf R_A-\mathbf T,
\]

the JGP94 moment tensor and proper ordered eigenvector matrix are

\[
\mathbf M
=\sum_A Z_A
\left[
\lVert\mathbf C_A\rVert^2\mathbf I
-\mathbf C_A\mathbf C_A^\mathsf T
\right],
\qquad
\mathbf O^\mathsf T\mathbf M\mathbf O
=\operatorname{diag}(\lambda_1,\lambda_2,\lambda_3).
\]

The runner uses row vectors:

\[
\mathbf Y=\mathbf C\mathbf O,
\qquad
\mathbf p_b=\mathbf p\mathbf O.
\]

Charges and radii are unchanged.  The accepted candidate energy is therefore

\[
E_{\mathrm{PCM}}^{\mathrm{body}}
=E_{\mathrm{pyddx}}
\left(\mathbf Y,\mathbf q,\mathbf p_b;\mathbf a\right),
\]

where \(\mathbf a\) denotes the unchanged atom radii.  A Cartesian finite
difference changes one laboratory coordinate while holding the original
laboratory-frame density coefficients fixed, rebuilds the local frame,
transforms the dipoles, and evaluates this same scalar.

## Complete reverse derivative

Let

\[
\mathbf G=\frac{\partial L}{\partial\mathbf Y},
\qquad
\mathbf H=\frac{\partial L}{\partial\mathbf p_b}.
\]

The direct contractions are

\[
\overline{\mathbf C}_{\mathrm{direct}}
=\mathbf G\mathbf O^\mathsf T,
\qquad
\overline{\mathbf O}
=\mathbf C^\mathsf T\mathbf G+\mathbf p^\mathsf T\mathbf H,
\qquad
\overline{\mathbf p}
=\mathbf H\mathbf O^\mathsf T.
\]

Define

\[
\mathbf K=\mathbf O^\mathsf T\overline{\mathbf O},
\qquad
\mathbf S=\frac{\mathbf K-\mathbf K^\mathsf T}{2}.
\]

Within one nondegenerate ordered-eigenvector chart, the moment-tensor
cotangent in the eigenbasis is

\[
\widetilde{\overline M}_{ij}
=
\begin{cases}
\dfrac{S_{ij}}{\lambda_j-\lambda_i},&i\ne j,\\[6pt]
0,&i=j,
\end{cases}
\]

and

\[
\overline{\mathbf M}
=\mathbf O\widetilde{\overline M}\mathbf O^\mathsf T.
\]

For each centered nuclear coordinate,

\[
\overline{\mathbf C}_{A,\mathrm{frame}}
=2Z_A\left[
\operatorname{tr}(\overline{\mathbf M})\mathbf C_A
-\mathbf C_A\overline{\mathbf M}
\right].
\]

After adding the direct and frame terms, the center-of-nuclear-charge VJP is

\[
\overline{\mathbf R}_A
=\overline{\mathbf C}_A
-\frac{Z_A}{\sum_B Z_B}
\sum_B\overline{\mathbf C}_B.
\]

This last term guarantees translation closure.  Merely rotating the pyddx
coordinate gradient back to the laboratory frame would omit
\(\overline{\mathbf C}_{\mathrm{frame}}\), including the transformed-dipole
contribution, and would differentiate a different scalar.

For the actual canary:

- \(\mathbf G\) is pyddx's analytic fixed-density coordinate derivative of the
  body-frame scalar;
- \(\mathbf H\) is the Cartesian dipole block of the analytic pyddx density
  cotangent;
- the raw MACE-POLAR order `[q,y,z,x]` is converted to Cartesian dual order
  `[q,x,y,z]` before applying the frame VJP.

The pure transform and VJP live in
`benchmarks/route2_jgp94_frame.py`.  This module has no provider, MACE, parser,
or calculator dependency.

## Fail-closed local chart

The derivative is accepted only while

\[
\frac{\min_i(\lambda_{i+1}-\lambda_i)}
{\max(1,\max_i|\lambda_i|)}
\ge 0.05.
\]

Every displaced frame is sign-aligned to the base proper frame and must retain
minimum corresponding-axis overlap `>= 0.999999`.  Eigenvalue permutations,
improper axes, a small eigengap, or a changed finite cavity active mapping fail
closed.  The `0.05` and overlap values are conditioning guards, not fitted
solvation parameters.

## Immutable acetone protocol

The runner is
`benchmarks/run_route2_jgp94_frame_vjp_canary.py`.

Inputs:

- fixed FreeSolv acetone geometry `mobley_3867265`;
- the previously converged, frozen MACE-POLAR density;
- the passed tracked scalar result
  `benchmarks/route2-ddpcm-ri-jgp94-acetone-v1.json`;
- pyddx 0.8.0 ddPCM with water dielectric `78.39`, `eta=0.1`,
  `lmax=15`, 1202 Lebedev points, one thread, and tolerance `1e-12`;
- the two largest Cartesian continuum-gradient components from the immutable
  pre-existing, translation-equivalent acetone archive, selected before this
  candidate was evaluated:
  zero-based atom 1, \(y\), `-0.806069893702285 eV/angstrom`; and zero-based
  atom 2, \(y\), `+0.5830010743437304 eV/angstrom`.

The component-source archive has SHA256
`997fcc45ee7cb78ccf2b5fb2465461ea77388570b7ca0cf91a4fb7e161be8497`.
It selects components only; none of its derivative values is reused as a
candidate result.

Central-difference steps are fixed at `1e-3` and `5e-4 angstrom`.  The exact
continuum-solve budget is:

| Work | Count |
|---|---:|
| Base analytic coordinate-derivative state | 1 |
| Base analytic density-cotangent state | 1 |
| Two components x two steps x two signs | 8 |
| Total pyddx state solves | **10** |
| MACE calls / ML--SCF roots / CDS calls / retries | **0 / 0 / 0 / 0** |

The same base cotangents are reused for three rigid-orientation covariance
checks; those checks add no continuum solve.

## Zero-solve preflight

Before a lock can be written, the runner:

1. checks all 30 acetone coordinate and 30 dipole components against a
   deterministic synthetic linear loss at steps `1e-4`, `5e-5`, and
   `2.5e-5`;
2. verifies a nonzero frame contribution and translation closure;
3. constructs, but does not solve, each displaced pyddx cavity;
4. requires every displaced cavity to retain the passed base signature
   `f4de8cf...e9d49300` with exactly `4993/4993` unique active pairs.

The synthetic position-error limits are `2e-7`, `6e-8`, and `2e-8`, with each
halving ratio `<=0.4`.  The dipole limit is `5e-10` and translation closure is
`<=1e-12`.  These are algebra/oracle gates, not chemical-accuracy claims.

## Frozen one-shot result gates

| Gate | Required value |
|---|---:|
| Base energy versus passed scalar canary | `<=1e-10 eV` |
| Energies from the two base derivative solves | difference `<=1e-10 eV` |
| Half-coupling identity | `<=1e-10 eV` |
| Translation closure | `<=1e-10 eV/angstrom` |
| Physical position-plus-dipole torque | `<=1e-8 eV` |
| Three-orientation position-gradient covariance | `<=1e-10 eV/angstrom` |
| Three-orientation dipole-gradient covariance | `<=1e-10 eV` |
| Every analytic/FD component at both steps | absolute `<=2e-5 eV/angstrom`; relative `<=1e-4` |
| Fine/coarse absolute-error ratio, each component | `<=0.5` |
| Base and all displaced active mappings | exact passed signature; `4993/4993` unique |
| Evaluation budget | exactly 10 state solves; no retry |

The fixed work directory is
`.omx/benchmarks/route2-ddpcm-ri-jgp94-frame-vjp-acetone-20260726`.
`freeze` requires a clean committed checkout and hashes the runner, pure frame
module, this document, the prior scalar result, all immutable local inputs, and
the compiled pyddx extension.  `run` writes an exclusive attempt marker before
the first continuum solve.  A crash or failed gate cannot be retried in
another directory.

The non-scientific preflight is:

```bash
PYTHONPATH="$PWD:/home/axie/.cache/maple-envs/psi4-ddx-qmref/lib/python3.11/site-packages" \
  /home/axie/miniconda3/envs/maple/bin/python \
  docs/implicit-solvation/benchmarks/run_route2_jgp94_frame_vjp_canary.py \
  preflight
```

Had the preflight passed, the only authorized execution after a clean
preregistration commit and independent review would have been:

```bash
PYTHONPATH="$PWD:/home/axie/.cache/maple-envs/psi4-ddx-qmref/lib/python3.11/site-packages" \
  /home/axie/miniconda3/envs/maple/bin/python \
  docs/implicit-solvation/benchmarks/run_route2_jgp94_frame_vjp_canary.py \
  freeze

PYTHONPATH="$PWD:/home/axie/.cache/maple-envs/psi4-ddx-qmref/lib/python3.11/site-packages" \
  /home/axie/miniconda3/envs/maple/bin/python \
  docs/implicit-solvation/benchmarks/run_route2_jgp94_frame_vjp_canary.py \
  run
```

Because the frozen preflight failed, neither command is authorized or was
executed in this stage.

## Frozen preflight outcome

The zero-solve preflight was executed before any one-shot lock or continuum
solve.  Its exact tracked mirror is
`benchmarks/route2-jgp94-frame-vjp-preflight-v1.json`, SHA256
`eefad3ce832bf8ffc1c2db9d29663e151da83e1558a3489e4193a360eaaeafb2`.

The pure frame algebra passed:

- all 30 coordinate and 30 dipole components were checked;
- coordinate maximum absolute errors at steps `1e-4`, `5e-5`, and `2.5e-5`
  were `9.3481e-8`, `2.3364e-8`, and `5.7252e-9`;
- the corresponding error ratios were `0.24994` and `0.24504`;
- dipole maximum errors were at most `7.4937e-12`;
- translation closure was `1.1103e-15`;
- the orientation contribution norm was `6.9426`, so the test did not reduce
  to a direct-coordinate-only path.

The pyddx cavity-domain preflight failed before a state solve:

- the eight displaced cavities produced four active-set signatures;
- active-pair counts were `4992` or `4993`, rather than the required fixed
  `4993`;
- the carbonyl-carbon `+/-5e-4 angstrom` pair retained the base signature, but
  its `-1e-3 angstrom` point did not;
- all four carbonyl-oxygen displacement points changed the base active set.

Therefore `cavity_active_mapping` and `cavity_pair_count` failed, while
`cavity_pairs_unique` passed because every displaced mapping still contains no
duplicate sphere/Lebedev pairs.  The provider-state solve count was exactly
zero. No lock, attempt marker, finite-difference energy, or result was created.

This outcome separates two facts: the analytic JGP94 frame map is locally
correct, but placing the existing discrete pyddx cavity in that frame does not
remove general geometry-induced active-set changes.  A molecule-following
orientation repairs rigid-grid rotation covariance; it is not by itself a
smooth-cavity construction.

## Interpretation and stop condition

The failed zero-solve preflight rejects this one-shot experiment before
spending the solve budget.  The prescribed components and steps are not
replaced by a more convenient stable subset, and no continuum finite
difference is run.

A pass establishes only a complete local analytic VJP for the accepted
fixed-density molecule-frame scalar, for two acetone components inside one
nondegenerate, unchanged-active-set chart.  It does not establish:

- the converged MACE--ddPCM fixed-point adjoint;
- SMD CDS or total-solution forces;
- general geometry continuity outside the fixed active set;
- optimization, transition states, MD, NVE conservation, or a MAPLE PES;
- QM/experimental energy or force accuracy;
- a public provider or universal chemical domain.

The stage stops after tracking and reviewing the preflight rejection.  A later
stage may authorize only a separately pre-registered smooth-cavity design or a
narrow fixed-density derivative experiment; this rejected one-shot cannot be
retuned or retried.

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
