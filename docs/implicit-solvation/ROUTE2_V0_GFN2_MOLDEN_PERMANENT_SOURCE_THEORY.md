# Route-2 V0: zero-field GFN2 MOLDEN permanent-source gate

## Status and strict boundary

This is a **permanent-reference** investigation, not a replacement for the
rejected GFN2-xTB field response.  It is introduced because the current V0
full-response KKT construction has a mathematically complete neutral curvature
but has no identified affine origin for the permanent molecular charge source.

GFN2-xTB is a self-consistent tight-binding method with multipole
 electrostatics ([Bannwarth, Ehlert & Grimme, 2019](https://doi.org/10.1021/acs.jctc.8b01176)).
The official program can emit a MOLDEN orbital export and machine-readable
occupations/dipole for a single point; its documentation also treats external
point charges and implicit-solvent models as separate optional interfaces
([xTB properties](https://xtb-docs.readthedocs.io/en/latest/properties.html),
[xTB external embedding](https://xtb-docs.readthedocs.io/en/latest/pcem.html)).
V0 uses only the unembedded, gas-phase, zero-field state below.

This route does **not** use:

* xTB `--cosmo`, GBSA, ALPB, CPCM-X, or any xTB solvation number;
* the xTB external-point-charge finite-field response already rejected by the
  preregistered acetone screen;
* a fitted charge, radius, response scale, density cutoff, field step, or
  experimental solvation label;
* a new trained/fine-tuned model; or
* a claim about PCM, cavity, force, speed, or solvation accuracy.

The initial closed-shell source domain is the actual GFN2 MOLDEN `s`/`p` AO
export.  Unsupported angular shells, open shells, non-unit MOLDEN shell
scales, incomplete MO matrices, non-metric coefficients, solvent/embedded
runs, and ambiguous effective cores all fail closed.

## 1. Exact zero-field source representation

Let the MOLDEN export give contracted Cartesian AOs

\[
\chi_\mu(\mathbf r)
 = \sum_p d_{\mu p}
   (x-X_\mu)^{l_x}(y-Y_\mu)^{l_y}(z-Z_\mu)^{l_z}
   e^{-\alpha_{\mu p}|\mathbf r-\mathbf R_\mu|^2},
\]

with only \(l_x+l_y+l_z\le 1\).  The adapter evaluates the *printed*
contracted functions; it does not renormalize a primitive or fit a Gaussian
width.  Its overlap is the exact analytic Cartesian integral

\[
S_{\mu\nu}=\int \chi_\mu(\mathbf r)\chi_\nu(\mathbf r)\,d\mathbf r.
\]

For closed-shell MO coefficients \(C\) and occupations \(f_i\in\{0,2\}\),

\[
C^\mathsf TSC=I,\qquad
D_0=C\operatorname{diag}(f)C^\mathsf T,
\qquad
N_{\rm val}=\operatorname{Tr}(D_0S)=\sum_i f_i.
\]

The first two equalities are executable hard gates.  They establish only that
the exported MOLDEN state is represented correctly in the declared AO metric;
they do not establish agreement with QM.

GFN2 is a valence-electron method.  Its effective core charge is not guessed:
the version-bound shipped GFN2 parameter file declares the first modelled shell
through `ao=...`.  If that shell has principal number \(n\), the closed-shell
core is the fixed periodic-shell count \(N_{\rm core}(n)\), so

\[
Z_a^{\rm eff}=Z_a-N_{\rm core}(n_a).
\]

The corresponding permanent electrostatic source is

\[
\rho_0(\mathbf r)=
\sum_a Z_a^{\rm eff}\delta(\mathbf r-\mathbf R_a)
-
\sum_{\mu\nu}D_{0,\mu\nu}\chi_\mu(\mathbf r)\chi_\nu(\mathbf r),
\]

and the permanent dipole is

\[
\boldsymbol\mu_0=
\sum_a Z_a^{\rm eff}\mathbf R_a
-
\operatorname{Tr}(D_0\mathbf r).
\]

The adapter requires the reconstructed \(\boldsymbol\mu_0\) to agree with
the same xTB run's reported dipole before promoting the source.  This catches
a silent full-nuclear-charge/valence-density mismatch: pairing a 24-electron
GFN2 valence density with bare \(Z\) rather than \(Z^{\rm eff}\) produces an
incorrect source even if \(C^\mathsf TSC=I\).

## 2. How this supplies the missing affine origin without reviving xTB response

The existing V0-AIPR/MACE-MDP construction supplies an independently sourced,
positive neutral covariance \(C_{\mathbf R}\) in a coefficient/source-dual
space.  It establishes response but not the permanent source (the
reference-shift theorem in
[`ROUTE2_V0_PERMANENT_REFERENCE_IDENTIFIABILITY.md`](ROUTE2_V0_PERMANENT_REFERENCE_IDENTIFIABILITY.md)).

After separate physics gates have mapped the certified GFN2 source into that
same dual space, its only allowed role is the origin \(c_{0,\mathbf R}\):

\[
\begin{aligned}
\mathcal G_{\mathbf R}(\delta c)
={}&E_{\rm MACE,gas}(\mathbf R)
+\frac12\delta c^\mathsf TC_{\mathbf R}^{+}\delta c\\
&+\frac12\left[B_{\mathbf R}(c_{0,\mathbf R}+\delta c)\right]^\mathsf T
Q_{\mathbf R}
\left[B_{\mathbf R}(c_{0,\mathbf R}+\delta c)\right]
+\mathcal G_{\rm cavity,np,std}.
\end{aligned}
\]

Here \(Q_{\mathbf R}\preceq0\) is the reciprocal continuum reaction operator
and \(B_{\mathbf R}\) is the exact source-to-dual map.  Therefore the
stationary induced state obeys

\[
\left(C_{\mathbf R}^{+}+B_{\mathbf R}^\mathsf TQ_{\mathbf R}B_{\mathbf R}\right)
\delta c^*
=-B_{\mathbf R}^\mathsf TQ_{\mathbf R}B_{\mathbf R}c_{0,\mathbf R},
\]

with the usual support and charge constraints.  The response is symmetric and
passive whenever the support Schur complement is positive.  The zero-field
xTB state is **not differentiated with respect to external field**; its failed
field response cannot enter \(C_{\mathbf R}\), cannot be interpolated, and
cannot be rescaled to improve an error.

This is a defined composite V0 scalar, not a claim that MACE's gas energy and
GFN2's permanent density were originally derivatives of one checkpoint.  Its
scientific admissibility depends on the gates below, not on that algebraic
possibility alone.

## 3. Required gates in order

1. **MOLDEN/source round trip.**  Verify SCC convergence, no embedding/no
   solvent, \(C^\mathsf TSC=I\), \(\operatorname{Tr}D_0S=N_{\rm val}\),
   the effective total charge, and xTB dipole round trip.  The frozen acetone
   protocol is
   [`route2-v0-gfn2-molden-permanent-source-acetone-prereg-v2.json`](benchmarks/route2-v0-gfn2-molden-permanent-source-acetone-prereg-v2.json).
   The v1 runner-import preflight failure is retained separately in
   [`route2-v0-gfn2-molden-permanent-source-acetone-preflight-failure-v1.json`](benchmarks/route2-v0-gfn2-molden-permanent-source-acetone-preflight-failure-v1.json);
   v2 changes only that package-import defect, not the scientific contract.
2. **QM permanent-MEP source gate.**  Compare the total source (effective cores
   plus valence density) against the existing frozen exterior QM-MEP geometry
   set before any continuum calculation.  This must be preregistered before
   reading the result and must be repeated over the broad ten-functional-group
   QM panel.  A dipole round trip is necessary but not sufficient for a
   near-field potential.
3. **All-electron cavity completion.**  The valence density is sufficient for
   its effective-core electrostatic ledger but is not an all-electron
   iso-density cavity field.  A core-density completion must be source-bound,
   charge-preserving, nonnegative, and demonstrated to share the same
   electrostatic/cavity dual calculus.  A free-atom promolecule cannot be
   silently substituted for the molecular permanent source.
4. **Full common-scalar KKT.**  Bind the accepted permanent source, the already
   frozen full response covariance, and a smooth reciprocal continuum; prove
   source duality, Schur-complement positivity, reciprocity, passivity,
   envelope forces, and coordinate-loop work.
5. **Only then, accuracy.**  Freeze all assets and perform the immutable
   historical FreeSolv10/12 all-record test (including ethyl acetate), then the
   ten-plus-solvent and blind disjoint protocols.  Every individual error,
   not a selected MAE, must meet the user's \(<1.5\) then \(<1\) kcal/mol
   gates.

## 4. Current conclusion

The zero-field MOLDEN route is an admissible *candidate way to close the
permanent-reference gap* because it derives a reproducible stationary xTB
valence density and exact effective-core electrostatic ledger.  It is not yet
an accepted V0 electronic functional or a solution to accuracy.  Its next
honest test is static QM MEP—not a small FreeSolv panel and not an xTB solvent
calculation.
