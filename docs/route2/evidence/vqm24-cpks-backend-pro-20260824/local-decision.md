# Local decision after Pro review and numerical prototypes

The Pro 5/5 derivation is accepted for the CPKS equations, closed-shell factors,
observable signs, use of `gen_response(singlet=None, hermi=1, with_nlc=True)`,
and the distinction between `gen_response(hermi=1)` and
`cphf.solve(hermi=False)`.  Independent local prototypes reproduce the frozen
finite-field oracle:

| case | CPKS vs 1e-3 MEP | CPKS vs 1e-3 dipole | curvature | reciprocity | CPKS solve | 16-SCF oracle |
|---|---:|---:|---:|---:|---:|---:|
| Gate-0 CH6N4 | 1.36e-4 | 1.23e-4 | 3.23e-4 | 2.68e-8 | 902 s | 553 s |
| Gate-A Cl | 1.30e-4 | 7.67e-5 | 4.42e-5 | 1.59e-7 | 890 s | 2882 s |

Thus CPKS computes the same q-to-zero target and is materially faster for the
heavier chlorine record, but it is not unconditionally faster for small
systems.  Speed is measured, not inferred from the method name.

Three qualifications are retained:

1. The finite-field oracle used the PySCF default `small_rho_cutoff` and built
   both level-3 grids once.  A live reconstruction confirmed that the resolved
   PySCF 2.13.1 default is already `0.0`; the CPKS backend records that value
   and hashes both coordinate/weight arrays rather than assuming it from a
   symbolic `level=3` label.
2. PySCF's Krylov stopping condition is not a physical CPKS residual.  The
   unrefined prototypes had direct residuals around 1.6e-5 and 5.0e-5 despite
   observable agreement.  The backend therefore recomputes the equation
   residual and attempts correction with PySCF's existing `lib.krylov(x0=...)`.
   It may be used for batch generation only after the corrected residual and
   all observable gates pass on unopened Gate-A records.
3. The four point-charge directions are observable probes, not a complete
   density or susceptibility label.  AO response densities remain internal and
   are not emitted as training targets.

The finite-field records remain the oracle.  CPKS is an implementation backend,
not a new electronic-structure target, model fit, or accuracy result.

## Gate-B outcome

After this decision was frozen, unopened Br/F/P/S records were executed under
`vqm24-static-cpks-gate-b-v1`.  Br, F, and S passed, but phosphorus exceeded
the preregistered absolute energy-curvature budget:

```text
observed absolute error  3.6092498987e-4 Ha/e^2
frozen absolute budget   3.2610437807e-4 Ha/e^2
```

The aggregate CPKS MEP and dipole discrepancies were nevertheless only
`6.19e-5` and `5.10e-5` symmetric relative, reciprocity/passivity passed, and
the median wall-time ratio was `0.778`.  These favorable secondary results do
not override the failed preregistered gate.  CPKS therefore remains a
diagnostic, not the admitted batch generator.  The training mainline retains
the directly converged `+/-1e-3 e` finite-field energies, MEPs, and dipoles and
does not form a noisy second-energy-difference label.
