# Route-2 V0: frozen static-MEP gate for the GFN2 MOLDEN permanent source

## Scope

The previous GFN2 MOLDEN gate proves a **representation identity** only: the
zero-field GFN2 valence AO density, the parameter-file-derived effective cores,
and the reported xTB dipole describe the same source.  It does not show that
this source resembles a QM permanent electrostatic source away from the
molecule.

This document defines the next falsification step.  It compares that exact
source against one pre-existing, gas-phase
\(\omega\)B97M-V/def2-TZVPD RKS checkpoint at all 516 immutable exterior
points.  No continuum surface is generated or selected in this step; these
points were copied from the existing exact-GTO canary only as a fixed
source-representation validation set.

The test does **not** use:

- xTB finite-field response, xTB implicit solvation, a PCM solve, cavity, or
  nonpolar model;
- a fitted charge, Gaussian width, response scale, point subset, error-weight,
  experimental solvation label, post-training, or fine tuning;
- a new QM SCF calculation; or
- static-MEP timing as a Route-2-versus-QM speed claim.

A failure rejects this particular GFN2 permanent-source candidate.  It does not
justify retuning the source.  A pass only admits it to all-electron cavity and
common-scalar/KKT gates.

## 1. Candidate potential is fixed before the comparison

The zero-field MOLDEN export gives the previously certified closed-shell
valence density

\[
D_0=C\operatorname{diag}(f)C^\mathsf T,
\qquad C^\mathsf T S C=I,
\]

and the GFN2 parameter file fixes \(Z_a^{\rm eff}\).  At every frozen point
\(\mathbf s_k\), the candidate potential is exactly

\[
V_{\rm GFN2}(\mathbf s_k)=
\sum_a\frac{Z_a^{\rm eff}}{|\mathbf R_a-\mathbf s_k|}
-
\sum_{\mu\nu}D_{0,\mu\nu}
\int\frac{\chi_\mu(\mathbf r)\chi_\nu(\mathbf r)}
{|\mathbf r-\mathbf s_k|}\,d\mathbf r.
\]

There is no atomic-charge fit hidden in this equation: \(Z_a^{\rm eff}\) is
already determined by the first modelled GFN2 shell, and \(D_0\) is the
zero-field MOLDEN density.

## 2. Independent integral carrier and exact representation check

The internal MOLDEN adapter evaluates the printed Cartesian contractions
\(\chi_\mu\) directly.  The validation helper needs one reliable AO Coulomb
integral implementation but must not silently change that representation.
PySCF's molecular object exposes a user basis and the `int1e_rinv` one-electron
operator with a movable \(1/r\) origin
([PySCF GTO developer documentation](https://pyscf.org/contributor/gto_developer.html),
[PySCF GTO API](https://pyscf.org/pyscf_api_docs/pyscf.gto.html)).

For a shell with angular momentum \(l\), primitive exponent \(\alpha_p\), and
printed coefficient \(d_p\), PySCF internally multiplies an input contraction
coefficient by its known primitive radial factor \(n_l(\alpha_p)\).  Therefore
the helper supplies

\[
\tilde d_p=\frac{d_p}{n_l(\alpha_p)}.
\]

This is a change of library convention, not a learned or fitted rescaling.  It
is accepted only when the independently evaluated objects agree:

\[
\frac{\|\widetilde S-S\|_F}{\|S\|_F}\le5\times10^{-8},
\quad
\|\widetilde C^\mathsf T\widetilde S\widetilde C-I\|_2\le5\times10^{-8},
\]

\[
\left\|\operatorname{Tr}(D_0\widetilde{\mathbf r})
-\operatorname{Tr}(D_0\mathbf r)\right\|_2
\le10^{-6}\ e\,a_0.
\]

The same check is performed for the total effective-core dipole.  If this
transfer fails, the MEP is not interpreted; no numerical potential from a
mismatched basis is admissible.

## 3. Frozen QM comparator

For the frozen all-electron checkpoint density \(D_{\rm QM}\), the comparator
is

\[
V_{\rm QM}(\mathbf s_k)=
\sum_a\frac{Z_a}{|\mathbf R_a-\mathbf s_k|}
-
\operatorname{Tr}\left[D_{\rm QM}
I(\mathbf s_k)\right],
\qquad
I_{\mu\nu}(\mathbf s_k)=
\int\frac{\phi_\mu(\mathbf r)\phi_\nu(\mathbf r)}
{|\mathbf r-\mathbf s_k|}\,d\mathbf r.
\]

The checkpoint must retain its closed-shell AO metric, electron count, total
charge, method/grid identity, and geometry identity.  Its AO density hash is
also bound to the pre-existing QM finite-field record, so a static reference
cannot be swapped after inspecting the candidate result.

## 4. Registered source-physics metrics

With \(v\) and \(v_{\rm QM}\) the complete 516-point vectors and
\(\boldsymbol\mu\) the corresponding permanent dipoles, the preregistered
falsification metrics are

\[
\epsilon_2=\frac{\|v-v_{\rm QM}\|_2}{\|v_{\rm QM}\|_2},
\qquad
\epsilon_\infty=
\frac{\|v-v_{\rm QM}\|_\infty}{\|v_{\rm QM}\|_\infty},
\qquad
\epsilon_\mu=
\frac{\|\boldsymbol\mu-\boldsymbol\mu_{\rm QM}\|_2}
{\|\boldsymbol\mu_{\rm QM}\|_2}.
\]

The gates are fixed before execution as

\[
\epsilon_2<0.20,qquad
\epsilon_\infty<0.30,qquad
\epsilon_\mu<0.20.
\]

They deliberately reuse the pre-existing \(0.20/0.30\) source-falsifier
scale; they are a coarse rejection screen, not a chemical-accuracy threshold.
Passing cannot establish an energy functional, a cavity density, response,
force, PES, or experimental solvation error.

## 5. Consequences

A pass provides one necessary fact that the earlier reference-shift theorem
lacked: an independently checked candidate for the **affine permanent source**
\(c_{0,\mathbf R}\).  The all-electron density needed for the iso-density
cavity remains missing, and the frozen response covariance must still be bound
to the same source/dual space and a reciprocal continuum scalar.  Thus the
next mathematical obligations remain

\[
\nabla_{\delta c}\mathcal G_{\mathbf R}=0,
\quad
Q_{\mathbf R}=Q_{\mathbf R}^\mathsf T\preceq0,
\quad
C_{\mathbf R}^{+}+B_{\mathbf R}^\mathsf TQ_{\mathbf R}B_{\mathbf R}\succ0
\ \text{on the constrained support}.
\]

Only after those structural gates, force/PES checks, and frozen total-model
assets are complete can the historical all-record FreeSolv and multi-solvent
accuracy protocols be run.
