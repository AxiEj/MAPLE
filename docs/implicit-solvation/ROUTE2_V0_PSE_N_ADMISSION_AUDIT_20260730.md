# Route-2 V0: native PSE-\(n\)/3D-RISM admission audit (2026-07-30)

## Decision

The checked-in cSPC/E PSE3 source is a useful **bulk-source control**, but it
is **not** a no-training Route-2 V0 replacement for the current
MACE-cluster/molecular-HNC bridge.  It must remain fail-closed.

This is not because PSE-\(n\) lacks a thermodynamic expression.  In its
native site-based 3D-RISM setting, PSE-\(n\) has a path-independent,
closure-matched excess-chemical-potential expression.  The obstruction is
that neither the checked-in bulk `Cvv` nor the whole-molecule MACE cluster
potential supplies the state variables of that expression.  Treating either
one as though it did would again separate the response equation from the
reported scalar.

## What the native PSE-\(n\) result actually says

For a site \(\alpha\) around a fixed solute, native 3D-RISM defines

\[
\Xi_\alpha(\mathbf r)
=-\beta u_\alpha(\mathbf r)+h_\alpha(\mathbf r)-c_\alpha(\mathbf r),
\]

and the PSE-\(n\) closure is

\[
g_\alpha(\mathbf r)=
\begin{cases}
\exp\Xi_\alpha(\mathbf r),&\Xi_\alpha\leq0,\\
\displaystyle\sum_{j=0}^{n}\frac{\Xi_\alpha(\mathbf r)^j}{j!},
  &\Xi_\alpha>0.
\end{cases}
\]

On a **converged native PSE-\(n\) 3D-RISM solution** the matched closed-form
chemical potential is

\[
\Delta\mu_{\mathrm{PSE-}n}
=\Delta\mu_{\mathrm{HNC}}
-k_BT\sum_\alpha\rho_\alpha\int
\Theta(\Xi_\alpha)
\frac{\Xi_\alpha^{n+1}}{(n+1)!}\,d\mathbf r,
\]

with

\[
\Delta\mu_{\mathrm{HNC}}
=k_BT\sum_\alpha\rho_\alpha\int
\left[\tfrac12h_\alpha^2-c_\alpha-
\tfrac12c_\alpha h_\alpha\right]d\mathbf r.
\]

Kast's path-independence construction and the PSE-\(n\) closed-form
expression therefore rule out the opposite mistake: PSE3 is not an arbitrary
numerical closure whose energy may be borrowed from HNC.  The native PSE3
scalar needs the matching \(u_\alpha,h_\alpha,c_\alpha\) fields and the PSE3
closure residual.

## Why it cannot consume the present MACE source

The V0 molecular bridge intentionally has a different state and source:

\[
\Omega_{\mathrm{mHNC}}[\nu]
=\Omega_{\mathrm{id}}[\nu;u_{\mathrm{MACE}}]
-\frac{k_BT}{2}\langle P\nu-\rho_b,
C_{\mathrm{HNC}}(P\nu-\rho_b)\rangle.
\]

Here \(\nu(\mathbf X,\Omega)\) is a rigid-molecule configuration density,
\(P\) is its fixed configuration-to-site-density map, and

\(u_{\mathrm{MACE}}(A,B_{\mathbf X,\Omega})\) is one **whole-molecule**
cluster energy difference.  Native PSE3 instead requires one site-resolved
solute--solvent potential \(u_\alpha(\mathbf r)\) and the resulting
site-resolved 3D-RISM \(h_\alpha,c_\alpha\).  A whole MACE cluster scalar
cannot be decomposed into these site potentials without inventing an energy
partition.  That would violate the existing whole-molecule source boundary.

Likewise, the cSPC/E `Cvv` file is a **bulk site--site direct correlation**
from a 1D DRISM/PSE3 calculation.  It is not a solute PSE3 solution and cannot
reconstruct \(u_\alpha,h_\alpha,c_\alpha\).  It cannot be inserted into the
current molecular-HNC quadratic term, nor can the PSE3 on-shell correction be
added after that HNC minimisation.  Both shortcuts would report a scalar whose
stationarity equation was not solved.

## Accuracy consequence under the V0 contract

Direct PSE3 is not a credible route to the locked `all records < 1.5
kcal/mol` gate by itself.  Johnson *et al.* reported direct cSPC/E
3D-RISM-PSE3 hydration results for a broad 1123-molecule study with MUE
`19.693 kcal/mol` and RMSE `20.473 kcal/mol`; the lower reported values used
linear volume/descriptor corrections.  Those numbers are **not** a Route-2
result and do not prove an error for any member of the frozen FreeSolv10
panel.  They do establish that raw PSE3 literature performance cannot justify
promoting the source asset or bypassing the per-record V0 gate.  The
corrections are excluded here because they are post-hoc and/or label-fitted.

## Admission rule

A future PSE-\(n\) route may be considered only after all of the following
exist before target-solvation results are read:

1. A source-bound native site-PSE solver with the complete site potentials,
   bulk susceptibility, closure residual, matched on-shell scalar, pressure,
   standard-state convention, and coordinate derivatives; **or** a separately
   derived full orientational molecular functional with a frozen angular
   direct-correlation source.  The latter is a new theory, not a relabelling
   of `Cvv`.
2. A proof and finite-difference certificate that the same stationary scalar
   supplies energy, response, and forces.  The PSE correction alone is not a
   force implementation.
3. Frozen source-complete physical assets for all eleven default solvents and
   the custom-solvent admission contract.
4. The unchanged historical FreeSolv10 records (including ethyl acetate) and
   every later preregistered record strictly below `1.5 kcal/mol`, followed by
   a disjoint blind set.  No PC/PC+, UC, NgB, MAP, calibration, radius choice,
   closure choice, post-training, or fine-tuning may be selected from those
   results.

Until then, the PSE3 asset is retained solely as a hash-bound native-source
control and `Route2V0MaceClusterRismMolecularHNCBridge` must continue to
reject it before kernel assembly.

## Sources

1. S. M. Kast and T. Kloss, *Closed-Form Expressions of the Chemical
   Potential for Integral Equation Closures with Certain Bridge Functions*,
   J. Chem. Phys. **129**, 236101 (2008),
   [DOI:10.1063/1.3041709](https://doi.org/10.1063/1.3041709).
2. E. L. Ratkova *et al.*, *Solvation Thermodynamics of Organic Molecules by
   the Molecular Integral Equation Theory: Approaching Chemical Accuracy*,
   Chem. Rev. **115**, 6312--6356 (2015),
   [DOI:10.1021/cr5000283](https://doi.org/10.1021/cr5000283).  Its Eqs.
   47 and 77--79 give the 3D PSE-\(n\) closure and matched free-energy
   expression used above.
3. J. Johnson *et al.*, *Small Molecule Hydration Energy and Entropy from
   3D-RISM*, J. Phys.: Condens. Matter **28**, 344002 (2016),
   [DOI:10.1088/0953-8984/28/34/344002](https://doi.org/10.1088/0953-8984/28/34/344002).
   Its direct PSE3 and corrected-study statistics are external context only,
   never an input to Route-2 model selection or fitting.
