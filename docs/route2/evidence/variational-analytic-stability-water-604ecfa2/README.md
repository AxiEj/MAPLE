# Analytic Gaussian-multipole local stability and multi-start water canary

This bundle retains two clean-process executions of the disabled, separately
identified MACE-POLAR analytic Gaussian-multipole field-energy candidate coupled
to the smooth weighted harmonic-Galerkin continuum scalar. It tests one local
23-dimensional reduced susceptibility/Hessian pair and five declared initial
states. It is negative Tier-V evidence, not a release artifact.

## Bound identity

- execution commit: `604ecfa2035f140750124ada13df55f5bd3a3164`
- official checkpoint SHA-256:
  `fab8b8713c832f31a2a853aaa22fd638be8a369cbf5095e6b3e982a18d10e93a`
- model evaluator:
  `graph-longrange-analytic-gaussian-multipole-realspace-v1`
- scalar:
  `route2-variational-macepolar-analytic-gaussian-multipole-energygradient-smoothharmonicgalerkin-cpcm-v1`
- measurement SHA-256 (both runs):
  `56e7b23760fbd45aadccc9d1579d3d8647ccba9879a30db72f5fff49d9a5e1ff`
- primary JSON SHA-256:
  `516fb35e14b1e6fa06e1a0b2f692c87f12250576bf53fa287fbf93f44dac6cb6`
- replay JSON SHA-256:
  `d6b3a53d90c2e66892730cb064043704e4bc49f0882085d4e415478694c683f3`

The two complete files differ in execution metadata and wall-clock timings.
Their protocol, identities, geometry, five roots, dense matrices, spectra,
decisions, and `measurement_sha256` are exactly identical.

## Executed command

```bash
LD_LIBRARY_PATH=/home/axie/miniconda3/envs/maple/lib:${LD_LIBRARY_PATH:-} \
python tools/route2_release/run_variational_analytic_stability_water_canary.py \
  --checkpoint /home/axie/.cache/mace/MACEPOLAR1Mmodel \
  --device cuda \
  --output /tmp/route2-variational-analytic-stability-604ecfa2-run1.json
```

The replay used the same command with `run2.json` as the output.

A clean dry run of the preregistered code at `8ad6ff93` stopped before producing
an artifact because the runner incorrectly demanded that the full continuum
field have zero constant-potential gauge. The state equation actually works in
the fixed-charge reduced field returned by `decompose_field()`. Commit
`604ecfa2` removed only that invalid section requirement and records the gauge;
it did not change any numerical threshold, start, scalar, response map, or
stability decision.

## Multi-start and linearization result

Five fixed starts were used: zero; one `0.25` axis start; deterministic random
starts with norms `+0.25`, `-0.25`, and `+0.50`. All converged under the same
Anderson protocol in 31 or 32 iterations. The largest pairwise differences were:

| quantity | maximum | frozen threshold |
| --- | ---: | ---: |
| reduced root relative difference | `3.821619684420472e-10` | `1e-8` |
| source relative difference | `3.821619668205479e-10` | `1e-8` |
| field relative difference | `4.39964064689302e-10` | `1e-8` |
| scalar energy absolute difference | `4.547473508864641e-13 eV` | `1e-8 eV` |

The largest actual unmixed residual was `1.720697910222882e-10`, below the
frozen `2e-10` tolerance. The independently materialized residual Jacobian
satisfied

```text
J_r = I - J_M H_G
```

with relative Frobenius error `4.8093797227417603e-17`, versus the preregistered
`1e-10` gate. The constant-potential component was
`0.11740458051762441 eV/e`; water has total charge zero, so its coupling energy
is exactly zero and all stability matrices use the fixed-charge reduced chart.

These results are local evidence for repeated convergence and derivative
factorization at one geometry. They do not prove global root uniqueness.

## Negative local stability result

The model response `J_M = dc/dxi` is reciprocal to numerical precision
(relative symmetry defect `1.3495196506404244e-15`) but is indefinite and
singular on the declared 23-dimensional reduced space:

- minimum eigenvalue: `-0.021567868753146407`;
- maximum eigenvalue: `+0.044638953966440714`;
- 5 eigenvalues are below `-1e-12`;
- 12 eigenvalues are above `+1e-12`;
- 6 eigenvalues have absolute value at most `1e-12`.

Consequently:

- model passivity gate: **failed**;
- model local-invertibility gate: **failed**;
- feedback nonnegative-spectrum gate: **failed**;
- combined-Hessian positive gate: **failed/not constructible**.

The continuum reduced Hessian is reciprocal and negative-semidefinite within
roundoff. The feedback spectral radius is small (`0.010081162126547305`) and
the state residual remains well conditioned (minimum/maximum singular-value
ratio `0.9726080031165828`), but these operational-root properties do not repair
the failed common-functional stability conditions.

## Claim boundary

This is a decisive counterexample for admitting the current analytic
changed-inference candidate as a strict Tier-V model on its presently declared
reduced source/field space. It does not rule out another retrained or
structurally constrained model, nor does it invalidate the separately defined
operational conservative-PES route.

All public capability tiers remain closed:

```text
E = false
F = false
H = false
V = false
M = false
```
