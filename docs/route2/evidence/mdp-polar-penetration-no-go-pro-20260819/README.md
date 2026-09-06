# Neutral-atom penetration and zero-training higher-multipole decision

## Claim boundary

This directory records one verified Chrome **Pro, 5 of 5.** advisory review
and an independent local adjudication. It does **not** admit a new source,
continuum backend, energy ledger, force, or accuracy claim. The submitted
prompt was verified exactly before submission, the retained composer reported
`Pro`, the model menu reported `Pro, 5 of 5.`, and `Pro thinking` was observed.
At capture time Chrome had virtualized the prompt out of the accessibility
tree, so `capture-recovery.ps1` accepts the response only on the same retained
window after checking the Pro composer, absence of running controls, the
terminal marker, exactly one response-copy control, and the clipboard answer
hash. The original submission proof remains authoritative for prompt identity.

The copied response is advisory. The mathematical and scientific decision
below was checked independently against the current source definition, the
existing benzene outlying-charge evidence, and primary continuum literature.

## Accepted structural conclusions

1. The analytic neutral-atom primitive

   ```text
   delta rho_A = Z_A delta(r-R_A) - sum_k n_Ak(r-R_A)
   ```

   is a legitimate prescribed three-dimensional charge density. Its
   stationary sharp-dielectric parent problem is the generalized Poisson
   functional, and its electrostatic polarization energy is

   ```text
   G_pol[rho,R] = 1/2 integral rho(r) phi_rxn[rho,R](r) dr.
   ```

   The functional is quadratic and reciprocal for a fixed linear dielectric.
   Herbert's continuum review gives this Poisson equation and energy identity
   directly and distinguishes apparent surface charge from exterior volume
   polarization: <https://arxiv.org/html/2203.06846v1>.

2. A compact, neutral, spherical core-plus-electron density whose entire
   support is inside the cavity contributes exactly zero to the exterior
   potential. Its cross term with any reaction potential harmonic throughout
   that support also vanishes by the spherical mean-value property. Such a
   source cannot repair the cavity MEP or the polarization energy.

3. The repository primitive is an untruncated Gaussian mixture. Every finite
   cavity therefore contains exterior or escaped charge. For the declared
   sharp-interface Poisson model the exact reaction potential contains both a
   surface contribution and a volume-polarization contribution. The exact
   SVPE equation and escaped-charge density are summarized in Eqs. 2.37--2.40
   of the review above, which in turn cites Chipman's primary SVPE series.
   The energy contains the matching volume receiver; changing only the
   boundary MEP while retaining an unchanged standard ddPCM/CPCM half-coupling
   is not the same functional.

4. A boundary-only exact representation cannot cover all allowed exterior
   charge densities. The response gives a valid constructive counterexample:
   choose a smooth potential compactly supported outside and away from the
   boundary, and define its charge by Poisson's equation. Every boundary jet
   is zero while the dielectric polarization energy is nonzero. Therefore no
   boundary-trace-only map can be exact on this source class.

5. The frozen checkpoint observables do not identify a unique atom-local
   `l >= 2` completion. Geometry supplies equivariant tensor covariants but
   not their scalar amplitudes. Infinitely many local quadrupole and octupole
   fields preserve the same atomwise q/p, molecular dipole, and molecular
   uniform-field polarizability. A hidden-feature decoder or geometric frame
   would select a new model rather than recover a uniquely identified source.

6. Translating a spherical free-atom density has a pure `l=1` first tangent.
   Its `l=2` and `l=3` pieces first appear at second and third order in the
   displacement. A finite rigid-translation hierarchy is a possible declared
   density model, but it is not uniquely implied by the checkpoint's q/p or
   molecular polarizability. The canonical linear ADT lift is therefore the
   only source-provenanced zero-training tangent retained by the current v3
   candidate.

## Qualifications imposed by the local review

* The displayed SVPE signs and prefactors assume Gaussian units,
  `epsilon_in = 1`, a sharp interface, and the response's normal convention.
  A production implementation must be derived and tested in the exact pyddx
  basis/metric convention rather than copied symbol-for-symbol.
* IEFPCM/SS(V)PE can simulate volume polarization and may be numerically close
  to SVPE. The primary review reports sub-0.1 kcal/mol examples, but also
  states that exact equivalence follows only when escaped charge vanishes.
  Therefore standard ddPCM is not declared useless; it is simply not an exact
  oracle for this new continuous exterior source.
* A smooth density-dependent dielectric is a coherent stationary alternative,
  but it is a new continuum/cavity model and requires the dielectric-response
  derivative. It cannot be used as an unrecorded repair of the current sharp
  cavity profile.
* The Pro response proves neither that the neutral-atom primitive is chemically
  material nor that it improves the 505-panel error. Those remain numerical
  questions under a prospective, target-independent experiment.
* The earlier benzene QM/pyddx evidence measured about `0.5474028424 e` outside
  the exact-profile cavity and a wrong-sign frozen-density continuum energy.
  That is strong evidence that source support and the volume ledger matter,
  but it is not a direct accuracy result for the present free-atom primitive.

## Frozen continuation and stop rule

The zero-training route now has only two remaining objects:

1. the already preregistered POLAR-zero plus MDP-alpha ADT v3 candidate; and
2. at most one exact escaped-charge penetration experiment.

The v3 Stage A and Stage B mechanism gates run first without modification. A
mechanism failure terminates v3 and cannot be repaired by penetration.

Only if v3 passes its mechanism gates but later fails its already frozen
source or chemical gate may the penetration experiment run. It must use:

* a manufactured compact-neutral zero control;
* the actual centered and one preregistered off-center Gaussian mixture;
* an existing converged volumetric Poisson or exact SVPE oracle;
* the same sharp cavity and dielectric;
* reciprocal energy, charging, PDE/interface, refinement, and coordinate-FD
  identities; and
* one fixed asymmetric molecular density separating q/p, Q, O, free-atom
  radial penetration, continuum-operator, and energy-ledger contributions.

No experimental solvation target enters this experiment. The Gaussian
primitive may be evaluated once on the already frozen prospective source panel
only after the manufactured identities pass. It is rejected if it fails the
frozen source/tail gate, if the remaining error is independently dominated by
Q/O, if it does not reproduce the evaluation-only QM radial `l <= 1`
component, or if an apparent improvement exists only in the incomplete
boundary-only ledger.

There is no third zero-training local-multipole decoder. If both retained
objects fail, the next admissible source is a separately named scalar-first
`l <= 3` electrostatic head trained only on molecule-disjoint QM density/ESP/
multipole/external-field data. It may not train on solvation targets or act as
a residual correction to the final hydration free energy.

## Existing volumetric oracle route; no new Poisson solver

The host already contains `/usr/bin/apbs`, reported as APBS `1.4.1` from the
Ubuntu `1.4-1ubuntu1` package; its executable SHA-256 is
`13b555e8a38041d80ab4a9476126f8df6ca10eb6d140c12619e5794d83b7f080`.
The installed parser contains charge-map and dielectric-map support. Current
official APBS documentation independently specifies OpenDX charge-density maps
in `e / angstrom^3`, three half-grid-shifted dielectric maps, and a finite-
difference calculation mode that replaces atom-discretized charges with the
supplied density map:

* <https://apbs.readthedocs.io/en/latest/using/input/old/read.html>
* <https://apbs.readthedocs.io/en/stable/using/input/new/calculate/index.html>

This gives a non-reinvented evaluation path for the conditional manufactured
Poisson oracle. The old installed executable is not admitted merely because it
exists: its version, binary and libraries must be content-addressed, and the
result must converge under box/grid refinement. A current APBS build is
preferred if the conditional experiment is triggered. No new dependency is
installed and no chemical source is evaluated at this stage.

## Local decision

```text
PRO_ADVICE_ACCEPTED_WITH_IMPLEMENTATION_QUALIFICATIONS
BOUNDARY_ONLY_PENETRATION_PATCH_FORBIDDEN
ZERO_TRAINING_LOCAL_L_GE_2_DECODER_NOT_IDENTIFIABLE
V3_MECHANISM_SCREEN_REMAINS_FIRST
ONE_EXACT_ESCAPED_CHARGE_TEST_REMAINS_CONDITIONAL
```

The terminal marker in the captured response is:

```text
ZERO-TRAINING PENETRATION DECISION
```
