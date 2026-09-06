# Route 2 — MACE-MDP + MACE-POLAR Hybrid

## Ownership

This workspace advances only the MACE-MDP + MACE-POLAR hybrid Route-2 family.
The active zero-training accuracy candidate is the role-separated v3 profile:

```text
MACE-POLAR zero-field point q/p permanent source
+ MACE-POLAR nonuniform response with its zero-field uniform tangent removed
+ canonical ADT lift of the supervised MACE-MDP molecular polarizability
+ continuum electrostatics
+ a separately declared solvent term
```

The original MACE-MDP latent atomwise permanent `q/p` plus MACE-POLAR induced
profile remains a separately identified callable experimental/daily surface and
historical accuracy control.  It is not the active quantitative source
hypothesis: its frozen physical source gate rejected those latent atomwise
moments as a transferable PCM permanent density.  MACE-MDP remains an essential
member of v3 through the checkpoint property it was actually trained to
predict, the molecular polarizability tensor.  This role change must not be
described as pure MACE-POLAR.

Pure MACE-POLAR is developed in a separate workspace. Evidence, scalar IDs,
source definitions, checkpoints, and admission decisions from the two routes
must not be mixed.

## Frozen scientific target

The primary quantitative target is:

```text
mean absolute error <= 1.5 kcal/mol
```

on the preregistered frozen 505-record MNSol development panel, using the
exact profile bound by
`route2-hybrid-smd-development-prereg-v3`. The 148-record confirmation
partition remains sealed. The v3 result is a frozen baseline: it cannot be
retroactively altered by fitting, calibration, checkpoint selection, or
method selection.

The running evidence source is read-only:

```text
source snapshot: ~/.local/share/maple/route2/hybrid-smd-development-v3/source
records:         ~/.local/share/maple/route2/hybrid-smd-development-v3/records
```

This workspace was copied from that snapshot. It may evolve only under new,
explicitly versioned profiles; it must not alter the running snapshot or
retroactively change its preregistered method.

## Hard zero-training mainline

Post-training is a last-resort fallback, not an accuracy-development tool for
the active hybrid route.  Until every preregistered zero-training candidate has
reached a terminal physical decision, the active mainline must not:

- fine-tune either official checkpoint or train a replacement source/scalar
  head;
- fit a residual, per-element/source scale, Gaussian width, dielectric factor,
  cavity radius, solvent factor, or CDS parameter to solvation targets;
- select a source/kernel/ledger by development-set solvation error; or
- use the sealed confirmation partition for diagnosis or method choice.

The required order is adapter/units semantics, exact checkpoint composition,
source topology and analytic kernel physics, continuum/operator fidelity,
energy ledger, and only then prospective target-blind validation.  A lower MAE
obtained by supervised correction before those terminal decisions is not an
improvement of this mainline.  If all such zero-training routes fail, any later
trained component must be a separately named fallback profile with frozen
backbones and independent QM electrostatic supervision; it may not be presented
as the unchanged MACE-MDP + MACE-POLAR hybrid.

## Chrome Pro mathematical advisor

The retained Windows Chrome session is an approved autonomous external-advisor
surface for this hybrid route. When a material scientific decision depends on
advanced mathematics or remains unresolved after a concrete local derivation,
the agent should use that session to ask the genuine **Pro** model rather than
silently guessing or reducing the claim. Appropriate topics include:

- variational and stationary-functional constructions;
- implicit-function, adjoint, HVP, and mixed-derivative derivations;
- reciprocity, gauge, passivity, conditioning, and root-uniqueness arguments;
- equivariant continuum discretization and cavity regularity;
- identifiability of electrostatic and CDS components;
- adversarial review of a proposed scientific admission proof.

Before submitting, the UI must visibly identify the selected model as `Pro`
(the prior validated surface displayed `Pro, 5 of 5`), not `Medium` or an
unverified default. The prompt, selected-model evidence, answer, timestamps,
and SHA256 digests should be retained in a repository evidence note, following
[`docs/route2/evidence/HARMONIC_GREEN_OPERATOR_PRO_AUDIT_2026-08-14.md`](docs/route2/evidence/HARMONIC_GREEN_OPERATOR_PRO_AUDIT_2026-08-14.md).

Pro output is an external mathematical review, not authoritative implementation
or admission evidence. Every accepted result must be independently rederived,
translated into an explicit repository contract, and verified by analytic
identities, finite differences, or other appropriate tests. Routine coding
questions should remain local; Pro should be used promptly when the uncertainty
is genuinely mathematical or could change the scientific model.

Do not allow several material mathematical choices to accumulate before asking.
A new focused Pro review is due as soon as any of the following is true: two
locally defensible physical ledgers remain indistinguishable; a proposed repair
changes the source, continuum equation, scalar, or admission domain; or a proof
depends on nontrivial functional analysis, identifiability, implicit response,
or global well-posedness.  This is a minimum checkpoint cadence, not a quota:
do not spend Pro on routine implementation, and never overlap or duplicate a
still-running question.

For this goal, Pro review is a required second-opinion checkpoint rather than a
one-off escalation. Unless a single completed answer explicitly closes the next
checkpoint too, ask a new, focused Pro question before freezing each of:

1. the target-blind M3 statistical subspace and its grouped validation rule;
2. the final positive-parent, SO(3)-structured smooth CDS area and its force
   regularity contract;
3. the source/continuum/energy-ledger interpretation and matched QM/PCM
   component admission;
4. the complete block/reduced implicit-force derivation and root-well-posedness
   certificate; and
5. the final public scientific admission proof.

Questions should combine tightly related mathematics, but a still-running query
must never be duplicated or overlapped. Poll sparsely, retain the complete
answer and its evidence hashes, then critically rederive and test it before the
next checkpoint is submitted. Passing Pro review never substitutes for local
proof, chemical reference data, or profile-bound validation.

## Frozen fallback evidence: profile-bound CDS lane

This lane is retained as historical falsification evidence and is not the
active accuracy mainline under the hard zero-training rule above.  Stock
SMD-CDS transferability is a hypothesis tested by the frozen v3
baseline, not an invariant. SMD electrostatic and CDS decompositions are
model-dependent, so a separately versioned effective CDS may be developed
after the complete v3 component matrix is frozen.

The first such candidate is water-only:

```text
MAPLE-CDS-W1
= low-dimensional linear aqueous SMD surface-tension basis
+ a frozen differentiable area definition
+ frozen MACE-MDP + MACE-POLAR hybrid electrostatics v3
```

It may fit only the residual between the frozen electrostatic prediction and
the experimental total under the frozen standard-state convention. It must
not alter source scales, Gaussian widths, dielectric factors, cavity radii,
checkpoints, or the electrostatic ledger. It is an effective model-dependent
term, not an experimentally observable standalone CDS component.

The 505 records remain the development pool; no retroactive blind subset may
be carved out after partial execution. The existing 148-record confirmation
manifest remains the only current MNSol confirmation partition. A water-only
candidate and a future multi-solvent candidate require different profile IDs
and admission decisions. See
[`docs/route2/MAPLE_CDS_W1.md`](docs/route2/MAPLE_CDS_W1.md).

### Frozen CDS terminal result

The final positive-parent PH1.0 CDS lane was preregistered before target
joining and then executed exactly once.  All ten numerical minimizer
certificates passed, but the conjunctive development gate failed:

```text
water grouped-OOF exact-minimizer MAE upper bound: 1.6969972 kcal/mol  FAIL
mixed-505 exact-minimizer MAE upper bound:          1.4546582 kcal/mol  PASS
required:                                           both <= 1.5 kcal/mol
```

The mixed-panel mean must not hide the tail.  The same grouped-OOF candidate
has a `18.303746 kcal/mol` maximum absolute error, with 66/505 records at or
above `3 kcal/mol` and 11/505 at or above `5 kcal/mol`.  Therefore neither a
four-molecule canary nor the mixed-505 MAE by itself is an accuracy admission;
water-specific and tail gates remain mandatory.

Consequently, no all-306 deployment fit exists and no further CDS feature,
loss, mode, intercept, regularizer, fold, or threshold variant is permitted.
This closes only the attempt to repair the current electrostatic profile by
increasing CDS capacity; it does not close the hybrid route.

The target-independent heterogeneous-electrostatics audit has now rejected the
first smooth weighted-shell harmonic C-PCM weak form.  It passes exact isolated
sphere identities, but on the frozen acetic-acid component case its point
permanent energy changes from `-12.20` to `-37.36 kcal/mol` as the retained
harmonic degree grows from 1 to 3.  A full-rank QR coordinate change reproduces
each energy to approximately `1e-12 kcal/mol`, while same-cavity PCMSolver C-PCM
converges near `-10.284 kcal/mol`.  This is a trial-space/weak-form fidelity
failure, not a removable coefficient-conditioning issue.  No spectral clipping
or empirical regularization may be used to rescue that operator.

The mainline therefore reuses the repository's existing separated-source
`pyddx` ddPCM implementation rather than inventing another continuum solver:

```text
MDP permanent source = point monopoles/dipoles
POLAR induced source = exact zero-anchored 1.5 A Gaussian increments
continuum             = shared pyddx ddPCM coefficient state
POLAR receiver        = external-MEP 1.5/3.0 A radial field
```

The current tree has now replayed the earlier four-case electrostatic result
(`0.869089 kcal/mol` MAE), a real-checkpoint directional-force canary
(`5.31e-7`, `1.35e-7`, and `3.01e-8 eV/A` errors at successively halved
steps), source-component and `lmax` diagnostics, and a water rotation check.
The four-case number is a mechanism/replay canary only: `N=4` is not an
accuracy panel and that MAE must not be quoted as evidence that the hybrid is
quantitatively accurate.  The current accuracy evidence is the preregistered
505-record development result.  On that panel electrostatics alone has
`2.953989 kcal/mol` MAE and `20.066532 kcal/mol` maximum absolute error; adding
stock SMD-CDS gives `1.696313 kcal/mol` MAE and `14.904481 kcal/mol` maximum
absolute error.  A preregistered one-parameter water CDS rescaling improves the
mixed-505 development MAE to `1.411999 kcal/mol`, but fails its water gate
(`1.626595 kcal/mol` MAE) and worsens the maximum error to
`16.621963 kcal/mol`; it is therefore not an admitted solution.

The largest full-panel error has also been decomposed rather than treated as a
generic tail statistic.  For 1,4,5,8-tetraminoanthraquinone in water, the
converged `-28.966532 kcal/mol` electrostatic energy consists of
`-21.747904` permanent MACE-MDP, `-0.671742` induced-only MACE-POLAR, and
`-6.546886 kcal/mol` permanent-induced cross terms.  The root residual is
`1.632e-11 eV`, so the error is not an unconverged-SCF artifact.  A subsequent
target-independent fixed-cavity gauge witness redistributes only `0.01 e A`
between two learned local atomic dipoles while preserving total charge and all
three components of the molecular dipole exactly.  That invisible molecular-
observable direction changes the PCM energy by `+0.336911` or
`-0.356401 kcal/mol`; direct ddX replay agrees with the exact quadratic
prediction to `4.253e-14 eV`.  This is one development-case identifiability
witness, not a population statistic, but it proves that the checkpoint's
molecular dipole/polarizability supervision does not uniquely determine the
near-field atomwise source used by PCM.  The unchanged MACE-MDP atomwise q/p
partition must therefore remain a latent diagnostic rather than a quantitative
source claim.  Evidence is frozen under
[`docs/route2/evidence/hybrid-mdp-atomic-partition-gauge-20260817/`](docs/route2/evidence/hybrid-mdp-atomic-partition-gauge-20260817/).

Post-training is not the authorized main accuracy route.  The mainline must
first exhaust physically derived combinations of the two unchanged official
checkpoints and analytic electrostatic representations.  In order, it must
test:

1. the exact MACE-POLAR zero-field atomwise source topology as the permanent
   prior, with the field-induced part still defined by the exact zero-anchored
   MACE-POLAR increment;
2. an exact, differentiable projection of that permanent prior to the
   MACE-MDP total charge and public molecular dipole, using a preregistered
   Gaussian-density Coulomb metric rather than an arbitrary Euclidean norm;
3. point, native 1.5 A Gaussian, and physically derived charge-penetration
   kernels under the same source and continuum equation; and
4. only if required by a target-independent source diagnosis, fixed analytic
   neutral-atom/core-density or higher-multipole terms whose parameters come
   from published atomic physics rather than solvation fitting.

The fourth item has an additional compact-support gate.  Standard ddPCM
assumes that the solute charge density is fully supported inside the cavity.
A spherical neutral core-plus-electron density that is fully contained inside
its parent sphere has exactly zero exterior potential by Gauss' theorem and
therefore cannot repair cavity-surface MEP.  If its tail does change the
boundary potential, it is an outlying-charge source and cannot be inserted into
the current ddPCM scalar by supplying only a modified boundary ``phi``.  Such a
candidate must either include the matching density-pairing/volume term of a
well-defined continuum functional or be rejected; enlarging the tail, clipping
it, or treating the boundary potential as a drop-in point-source replacement
is not an allowed zero-training shortcut.

The same frozen 60-molecule SPICE/MBIS source panel may be used only as an
independent falsification gate for Q/mu closure, exterior/cavity MEP, and
fixed-source ddPCM energy.  It must not be used to fit projection weights,
radial widths, per-element scales, solvent parameters, or a residual model.
Experimental solvation targets, CDS residuals, per-solvent scales, and the 505
confirmation partition remain forbidden throughout this mainline.

An earlier training-based fallback exists only as a frozen falsification
prototype; it is not the authorized successor or a basis for iterative
post-training.  On a
molecule-held-out SPICE subset its projected atomwise charge MAE changes from
the original latent MDP partition's `0.294992 e` to `0.021735 e`, and its
atomwise dipole-component MAE changes from `0.052093 e A` to
`0.004415 e A`.  A real 58-atom CUDA replay of the runtime provider passes
rotation, translation, permutation, total-charge closure, molecular-dipole
closure, and a same-graph coordinate-VJP finite-difference audit; the finest
directional VJP error is `1.02e-9`.  These results established that the
source-identifiability defect is repairable without solvation labels, but the
subsequent prospective 60-molecule-held-out PCM source gate failed.  Against
independent SPICE MBIS q/p on the same cavity and ddPCM operator, the original
latent MDP partition has a fixed-source energy MAE of `8.191931 kcal/mol` and
a `17.589141 kcal/mol` maximum error.  The learned head improves 58/60 cases
and reduces those values to `2.534650` and `10.261000 kcal/mol`, respectively,
but reduces the primary radius-1.0 q/p MEP global RMSE by only `7.43%` rather
than the preregistered `50%`, and worsens the larger-radius and q/p/Q/O MEP
diagnostics.  It is therefore not admitted for hybrid integration.  This
independent target-free result strongly localizes the current long tail to
permanent-source physics rather than SCF convergence, while also showing that
the first replacement head is not yet sufficient.  The exact prototype,
label-unit audit, runtime audit, hashes, and claim boundary are frozen under
[`docs/route2/evidence/mdp-mbis-source-head-prototype-20260817/`](docs/route2/evidence/mdp-mbis-source-head-prototype-20260817/).
The 60-case PCM source gate is frozen under
[`docs/route2/evidence/mdp-mbis-pcm-source-gate-20260817/`](docs/route2/evidence/mdp-mbis-pcm-source-gate-20260817/).

A second, prospectively selected and molecule-disjoint 60-case zero-training
gate has now tested whether the official MACE-POLAR checkpoint's own
well-conditioned uniform-field response manifold can close the remaining
permanent-source tail without fitting.  The nonlinear three-dimensional solve
is numerically sound (maximum dipole residual `9.9997e-10 e A`, maximum
Jacobian condition number `3.4619`, minimum singular value `0.8335`, and
maximum hidden field `0.07691 V/A`), but the scientific improvement is not
material.  The zero-field point source gives fixed-source ddPCM
`MAE/q95/max = 1.46096/3.67088/4.53094 kcal/mol`; the response-manifold source
changes these to `1.42833/3.54529/4.49363 kcal/mol`, improves only `32/60`
paired cases, and lowers MAE by only `2.23%`.  Both candidates therefore fail
the frozen tail gates, and the extra response state is rejected rather than
integrated into the hybrid runtime.  This result rules out molecular-dipole
closure and local solver conditioning as the dominant remaining mechanism;
the next zero-training investigation must target local near-field topology,
higher multipoles, and/or a continuum-consistent analytic density treatment.
The aggregate and all 60 immutable records are frozen under
[`docs/route2/evidence/mdp-polar-uniform-response-source-gate-20260817/`](docs/route2/evidence/mdp-polar-uniform-response-source-gate-20260817/).

One further parameter-free hybrid gauge has been terminally falsified rather
than tuned.  It keeps the MACE-POLAR zero-field charges and assigns the
MACE-MDP/POLAR molecular-dipole mismatch as
`delta p_A = alpha_A alpha_total^-1 delta mu`, using only MACE-MDP's exact
atomwise-polarizability sum.  On the seven previously opened worst tail cases,
it improves `5/7` but lowers mean error by only `2.02%` and worsens the global
q/p/Q/O cavity-MEP error.  It therefore fails its prelocked falsification gate;
no scale, clipping, tensor projection, or fitted mixture is allowed.  Evidence
is frozen under
[`docs/route2/evidence/mdp-polarizability-dipole-closure-tail-20260817/`](docs/route2/evidence/mdp-polarizability-dipole-closure-tail-20260817/).

A prospectively locked decomposition now rules out treating either CDS fitting
or another q/p-only readout as the main accuracy repair.  Metric-projecting
exact MBIS q/p to the frozen MACE-MDP molecular dipole changes fixed-source
ddPCM energy by only `0.022134 kcal/mol` MAE (`0.132971` maximum), and changes
the global cavity MEP by only `0.00699--0.00875` relative RMSE.  The MACE-MDP
molecular dipole is therefore retained as a transferable hard closure.  In
contrast, adding exact MBIS quadrupoles removes only `42.54%` of the
q/p-to-q/p/Q/O cavity-MEP RMSE at the physical radius scale 1.0, below the
preregistered `50%` gate, although it removes `59.31%` and `67.36%` at radius
scales 1.25 and 1.5.  The near-cavity error therefore requires l=3 information
and/or a compact analytic density/potential basis; parameter tuning inside the
existing q/p or q/p/Q function space cannot close it.  The next source profile
must first be a zero-training composition of the two official checkpoints with
an analytic, content-addressed electrostatic metric/kernel.  Independent
transferable QM electrostatic labels are evaluation-only evidence for that
profile, with exact charge and MACE-MDP-dipole closure and
molecule/chemistry-heldout validation.  It must not fit those labels, read
solvation targets, or fit a CDS residual.  Evidence is frozen under
[`docs/route2/evidence/mdp-mbis-source-error-decomposition-20260817/`](docs/route2/evidence/mdp-mbis-source-error-decomposition-20260817/).

A parameter-free Ewald-gauge composition has also reached a terminal negative
decision.  It assigns the exact MACE-POLAR zero-field atomwise source to the
point short-range channel and adds the MACE-MDP minus MACE-POLAR charge/dipole
difference through the official 1.5 A Gaussian kernel.  This preserves the
MACE-MDP molecular charge and dipole exactly without fitting.  Against the
previously opened seven-case tail, the q/p-only oracle already rejected it:
the candidate fixed-source ddPCM MAE was `4.488881 kcal/mol` versus
`3.713054 kcal/mol` for the MACE-POLAR zero-field point source, with only
`2/7` paired improvements.  That result is frozen under
[`docs/route2/evidence/mdp-polar-ewald-gauge-tail-20260817/`](docs/route2/evidence/mdp-polar-ewald-gauge-tail-20260817/).

Because a q/p-only reference cannot adjudicate a source intended to restore
near-field information beyond dipoles, the candidate then received one final,
separately preregistered same-equation test against exact SPICE MBIS Cartesian
q/p/Q/O projected into the pyddx real-spherical `l <= 3` basis.  This matched
oracle did not rescue it.  The MACE-POLAR zero-field point source gives
`2.245072 kcal/mol` MAE and `4.290953 kcal/mol` maximum error; the Ewald-gauge
candidate worsens them to `3.856007` and `6.771218 kcal/mol`, improves only
`2/7` cases, and has a `-71.75%` mean-error reduction.  Its global q/p/Q/O
cavity-MEP relative RMSE improves slightly (`0.288136 -> 0.274686`), while its
matched ddPCM energy becomes much worse.  Therefore this exact composition is
terminally rejected and may not be rescued by an interpolation coefficient,
width change, clipping, element scale, or fitted mixture.  The q/p/Q/O oracle,
records, runtime, and hashes are frozen under
[`docs/route2/evidence/mdp-polar-ewald-gauge-l3-tail-20260817/`](docs/route2/evidence/mdp-polar-ewald-gauge-l3-tail-20260817/).

The subsequent arithmetic-span susceptibility repair has also reached a
terminal, target-free mathematical rejection on the general molecular domain.
Writing `B = J_P U`, `C_P = C B`, and restricting the added tangent to
`K = B A G` with `G U = I` gives
`C (J_P + K) U = C_P (I + A)`, so the corrected molecular response can never
have rank greater than `C_P`.  For the official checkpoints on planar water,
`C_P` has singular values `5.9202383e-2, 2.5244887e-2, 0` (rank two), whereas
the MACE-MDP molecular polarizability has rank three.  The exact least-squares
lower bound on the relative Frobenius residual is `0.55974756`; an independent
rotated/translated replay gives the same value and aligns the null direction
with the molecular-plane normal to numerical precision.  A pseudoinverse
cannot enlarge the range, so this candidate is rejected rather than restricted
to a geometry-dependent chart.  The proof, real-checkpoint runner, immutable
record, and reproduction command are frozen under
[`docs/route2/evidence/mdp-polar-planar-rank-obstruction-20260818/`](docs/route2/evidence/mdp-polar-planar-rank-obstruction-20260818/).
This rejection does not reject the complete hybrid: the next zero-training
decision must explicitly leave the original POLAR uniform-source span, using
either a separately justified MACE-MDP atomwise dipolar tangent or the
source-provenanced free-atom density-translation response.  Neither may inherit
accuracy, force, or variational admission from the rejected arithmetic repair.

The source-provenanced option has now passed its target-free real-checkpoint
prequalification.  It uses the canonical free-atom-density
atomic-displacement-translation lift of the official MACE-MDP molecular
polarizability, while retaining the MACE-MDP permanent point `q/p` branch and
the MACE-POLAR nonuniform zero-anchored 1.5 A Gaussian residual as separate
source categories in one pyddx coefficient solve.  No solvation target,
calibration, post-training, source scale, or CDS parameter was used to define
this profile.

On water, all five deterministic starts converge to the same root in at most
21 iterations with a `6.54e-11 eV` residual.  The JVP/VJP relative dot defect is
`1.60e-12`; the finest centered finite-field error is `1.82e-8`; the local
residual-Jacobian minimum singular value is `0.94368`; and the selected uniform
response has three positive eigenvalues (`0.09761`, `0.09997`, `0.10438`).
At `n_lebedev=1202`, combined rotation/translation changes the energy by
`2.79e-6 eV` and the native field by `2.20e-5` relative.  A target-free angular
truncation scan shows that `lmax=8` differs from `lmax=15` by only
`1.38e-4 eV` (`0.00318 kcal/mol`) and `4.89e-4` relative field norm, so the
lower order is frozen for the 505 accuracy run rather than selected after
observing targets.  The complete artifact and exact hashes are frozen under
[`docs/route2/evidence/mace-mdp-polar-canonical-adt-ddx1202-water-20260818/`](docs/route2/evidence/mace-mdp-polar-canonical-adt-ddx1202-water-20260818/).

The immutable v1 505 run subsequently exposed one provider-domain failure:
iodine (`Z=53`) was absent from the original neutral-atom ADT registry. A
separately identified v2 role-separated profile now uses the frozen
25-electron def2-ECP valence pseudo-density only as the iodine ADT response
shape, while continuing to forbid its use as a neutral penetration density.
On the frozen target-free iodine geometry, two independent CPU cold replays
are bitwise identical; each replay's five starts converge in 19--21 iterations,
the largest residual is `8.53e-11 eV`, and residual total charge closes to
`9.80e-17 e`. The runner and output are frozen under
[`docs/route2/evidence/mace-mdp-polar-iodine-role-separated-full-replay-20260819/`](docs/route2/evidence/mace-mdp-polar-iodine-role-separated-full-replay-20260819/).
This closes the iodine provider mechanism only. It does not alter the frozen
v1 run, establish v2 accuracy, or admit any daily or variational capability.

### Final zero-training source hypothesis: POLAR-zero + MDP-alpha ADT

The two molecule-disjoint SPICE/MBIS source gates show that MACE-MDP's latent
atomwise `q/p` is not a quantitative physical permanent density, while the
MACE-POLAR zero-field point source is substantially closer to the same-equation
fixed-source ddPCM reference.  On the prospective second sixty-molecule panel,
the latter has `1.460958 kcal/mol` MAE; its `3.670878 kcal/mol` q95 and
`4.530936 kcal/mol` maximum still fail the frozen tail gate.  Closing only the
public molecular dipole through the checkpoint's uniform response manifold
improves MAE by just `2.23%` and also fails the tail gate.

The final honest no-training hypothesis therefore separates the roles:

```text
permanent source       = MACE-POLAR zero-field point q/p
nonuniform response    = POLAR(u) - POLAR(0) - J_POLAR(0) U G u
uniform response       = canonical ADT[-alpha_MDP G u]
```

Here `G U = I` is enforced exactly, MDP contributes only its supervised
molecular polarizability, and the removed zero-field uniform POLAR tangent is
not double counted.  A genuine Chrome `Pro, 5 of 5.` review conditionally
approved this construction as a role-separated operational density and agreed
that the MDP latent atomwise `q/p` should retire from production.  The complete
prompt, answer, model proof, scripts, and hashes are frozen under
[`docs/route2/evidence/mdp-polar-zero-point-adt-pro-20260819/`](docs/route2/evidence/mdp-polar-zero-point-adt-pro-20260819/).
The answer is advisory; local algebra and real-checkpoint tests remain the
authority.

The `lmax=12`, 1202-point water diagnostic closes root replay, charge,
JVP/VJP, finite-field, passivity, local-root, source/field covariance, and
harmonic-order convergence.  It still records a `1.20877e-5 eV` rigid-energy
drift and a `4.13262e-5` fixed-source field-covariance error from the finite
laboratory grid.  These small chemical magnitudes are retained as an explicit
structural defect rather than silently passed; the full measurements are under
[`docs/route2/evidence/macepolar-zero-mdp-alpha-role-separated-water-20260819/`](docs/route2/evidence/macepolar-zero-mdp-alpha-role-separated-water-20260819/).

Before any new 505 evaluation, a twelve-geometry adversarial coupled screen is
now prospectively frozen at
[`docs/route2/preregistrations/mace-mdp-polar-zero-adt-coupled-screen-v2.json`](docs/route2/preregistrations/mace-mdp-polar-zero-adt-coupled-screen-v2.json).
It reads no MNSol/FreeSolv target: the cases are selected from the already
frozen source/MEP diagnostics by four source-energy tails, two surface-MEP
tails, two largest systems, largest/smallest molecular dipole, worst response
conditioning, and largest response displacement.  Stage A uses one zero start
only as a cheap rejection screen and cannot admit accuracy or E/F/H/V/M.  A
pass requires a later five-start/replay/rotation/local-root screen; a failure
terminates this zero-training source route rather than authorizing a
solvation-target residual patch.

The superseded v1 invocation failed before evaluator construction because its
new geometry loader omitted the frozen source gate's `NUL` separator in the
array-identity hash.  No coupled state or target was evaluated.  The failure,
exact bytes, and narrow remediation are retained under
[`docs/route2/evidence/macepolar-zero-mdp-alpha-coupled-screen-provider-remediation-20260819/`](docs/route2/evidence/macepolar-zero-mdp-alpha-coupled-screen-provider-remediation-20260819/);
v2 changes only that identity-byte contract and binds the identical selection
and scientific method.

This result permits the exact canonical-ADT profile to enter the already-open
505 development energy panel.  It is not itself an accuracy result, a global
single-root proof, a structural SO(3) proof, or an E/F/H/V/M admission.  The
development history must remain explicit: those 505 targets were seen by older
profiles, whereas this new profile identity was frozen only from the bound
target-free canary before its own 505 evaluation.  The independent 148-row
confirmation partition remains sealed.

One further verified Chrome **Pro, 5 of 5.** review has now closed the two
remaining zero-training near-field ambiguities.  The analytic neutral-atom
Gaussian primitive is a legitimate prescribed density only under a continuum
functional that retains its escaped exterior charge.  For the current sharp
cavity this requires the exact Poisson/SVPE surface-plus-volume state and the
matching direct volume energy receiver; adding only its boundary potential to
unchanged ddPCM/CPCM is not the same stationary scalar.  Conversely, a compact
neutral spherical source fully contained in the cavity has exactly zero
exterior potential and cannot repair the cavity MEP.  The review also supplies
an independent non-identifiability proof: frozen q/p, molecular dipole,
molecular polarizability, geometry, and spherical free-atom densities do not
determine unique local quadrupoles or octupoles.  Free-atom translation fixes
only the linear dipolar ADT tangent; its higher-order translation moments are a
chosen nonlinear density model, not a checkpoint-implied decoder.

The advice was independently checked against Herbert's continuum review,
Chipman's SVPE construction, the local Gaussian source, and the existing
benzene outlying-charge evidence.  It is accepted with implementation
qualifications and frozen under
[`docs/route2/evidence/mdp-polar-penetration-no-go-pro-20260819/`](docs/route2/evidence/mdp-polar-penetration-no-go-pro-20260819/).
The current v3 Stage A/B mechanism screen remains first.  Only if v3 passes
those gates but fails the prospective source or chemical criterion may one
evaluation-only exact escaped-charge Poisson/SVPE experiment run.  A
boundary-only penetration patch is forbidden.  If that single test also fails,
the zero-training near-field route is exhausted and the next admissible model
is a separately named scalar-first `l <= 3` electrostatic head trained on
independent QM electrostatic data, never on solvation residuals.

### Hidden-feature decoder loophole is closed

A verified Chrome `Pro, 5 of 5.` review and an independently written local
SO(3)-equivariant counterexample now close the proposal to decode unreleased
`l >= 2` hidden tensors into a zero-training physical PCM source.  Charge,
molecular-dipole, and uniform-polarizability closure leave an infinite family
of equivariant, continuum-visible completions, while repeated hidden irreps
also possess an internal multiplicity-channel gauge.  A hand-written decoder
can therefore be reproducible for one serialized checkpoint without being an
identified checkpoint-native density.  The executable witness preserves
charge, dipole, polarizability, and rotation covariance to `1.67e-16`, yet
changes both the cavity MEP and a legitimate stationary quadratic scalar.

The evidence, exact Pro-mode proof, complete answer, local derivation, and
executable counterexample are frozen under
[`docs/route2/evidence/mdp-polar-hidden-feature-decoder-pro-20260819/`](docs/route2/evidence/mdp-polar-hidden-feature-decoder-pro-20260819/).
Its terminal marker is `HIDDEN FEATURES DO NOT IDENTIFY A PHYSICAL PCM SOURCE`.
No zero-training hidden decoder may be added to the hybrid mainline.  The
POLAR-zero/MDP-alpha role-separated ADT v3 profile remains the final
already-preregistered no-training source candidate; if it fails, the next
admissible source expansion is a separately named
scalar-first, density, or cavity-MEP head trained on independent QM
electrostatic data rather than experimental solvation targets.

Post-training becomes admissible only after every preregistered no-training
candidate above reaches a terminal physical failure and the failure is not an
adapter, normalization, source-kernel, cavity, or continuum-operator defect.
Even then it is a separately named fallback profile, must freeze the existing
backbones, may train only a small scalar/source head on independent QM
electrostatic data, and still may not use experimental solvation targets or
the sealed confirmation partition.  A lower MAE obtained by skipping this
order does not qualify as progress on the hybrid mainline.

At `lmax=15`, increasing the Lebedev rule from `1202` to `5810` lowers the
maximum rotated-force mismatch from `1.32e-4` to `1.15e-5 eV/A`; this is
numerical convergence evidence, not structural SO(3) proof or public force
admission.  Distorted-PES, multi-geometry force, closed-loop, root-domain, and
physical-ledger gates remain open.

### Nonuniform-work and ledger decision

Two genuine Chrome **Pro 5/5** reviews (71,614 answer characters in total)
first examined the source/work/ledger ambiguity.  Their prompts,
model-selection records, complete answers, timestamps, and hashes are frozen in
[`docs/route2/evidence/hybrid-ddx-source-ledger-pro-20260817/`](docs/route2/evidence/hybrid-ddx-source-ledger-pro-20260817/).

The accepted conclusion is narrow and terminal: the upstream `+E dot mu`
identity determines only the uniform-electric-field subspace.  The installed
checkpoint exposes no official arbitrary-nonuniform explicit-work endpoint,
so it must not be extrapolated into an ad hoc pairing between MACE-MDP/MACE-
POLAR sources and the eight-channel ddPCM model field.

An independent real-checkpoint water replay then established:

```text
correct direct-sum work - 2 G_cont:         5.33e-15 eV
naive source/model-field endpoint mismatch: 4.15e-1 eV
nonuniform fraction of reaction field:       5.80e-1
raw charging-line-integral error:             4.59e-13 eV
```

Both CUDA processes reproduced measurement SHA-256
`d91edf0e5a7e29ec20708add487c70583d3ad811447e7ef735e2d5948aca8593`;
see
[`docs/route2/evidence/hybrid-ddx-nonuniform-work-20260817/`](docs/route2/evidence/hybrid-ddx-nonuniform-work-20260817/).

Accordingly, the continuum half-work is internally exact, while its
external-MEP MLIP drive is a different receiver and cannot be used as a
physical endpoint work.  `Phi0 = E_vac + G_cont` and
`Phi_raw = E_conditioned_raw(u*) + G_cont` remain two explicit operational
scalar candidates.  Neither is a uniquely identified physical solvation
ledger or Tier V.

A third genuine Chrome **Pro 5/5** checkpoint has now frozen the exact terminal
comparison.  Its complete 29,589-byte response and UI/model evidence are in
[`docs/route2/evidence/hybrid-ddx-matched-ledger-pro-20260817/`](docs/route2/evidence/hybrid-ddx-matched-ledger-pro-20260817/).
The earlier QM/PCMSolver IEFPCM/GEPOL component panel cannot choose the ledger,
because its electronic density, cavity equation, and discrete operator are not
the hybrid pyddx-ddPCM problem.  The next chemistry gate is therefore one new,
target-independent, same-equation QM/pyddx-ddPCM decomposition on the frozen
benzene, acetone, acetic-acid, and diester geometries.  It must compare only
the two solvation shifts `G_cont` and
`DeltaE_conditioned_raw + G_cont` with the matched QM total electrostatic
shift.  The raw MLIP energy difference must not be labelled as the QM
electronic-distortion component.  No new endpoint factor, sign, empirical
coefficient, experimental hydration target, CDS, or standard-state term may
be used to choose between the ledgers.

The existing PH1 OOF CDS predictions may be carried only as the immutable CDS
term of a comparison; they may not be refitted.  The terminal CDS result,
post-failure Pro reviews, harmonic continuum rejection, and current ddPCM
replay are recorded in
`docs/route2/evidence/POSITIVE_PH1_TERMINAL_FAILURE_2026-08-17.md` and
`docs/route2/evidence/POSITIVE_PH1_POSTFAILURE_PRO_AUDIT_2026-08-17.md`, and
`docs/route2/evidence/HARMONIC_POINT_PERMANENT_PRO_AUDIT_2026-08-17.md`, and
`docs/route2/evidence/HYBRID_DDX_MAINLINE_REPLAY_2026-08-17.md`, and the two
nonuniform-work directories linked above.

## Required MAPLE capability surface

### Parallel experimental daily-workflow lane

Chemical-accuracy research must not make the already derived operational
scalar unusable for routine development.  The explicit model identity
`macemdppolarhybridddx` is therefore allowed to expose an **experimental**
daily surface independently of the `<= 1.5 kcal/mol` accuracy gate:

- normal MAPLE/ASE single-point energy and conservative force;
- first-order `LBFGS`/`SD`/`SDCG`/`CG` optimization;
- experimental mass-weighted vibrational-only `FREQ`, without gas-phase
  translational/rotational thermochemistry;
- experimental `TS(method=neb)` and `TS(method=neb,refine=cineb)` path searches;
- direct same-scalar Richardson Cartesian Hessian;
- a declared-origin nonperiodic molecular virial;
- named-solvent construction through the existing Route-2 solvent registry.

This availability lane must keep its numerical result honest: the frozen
development baseline remains `1.696313 kcal/mol` MAE with a
`14.904481 kcal/mol` maximum absolute error.  It must not relabel that result
as chemical accuracy.  Strict Tier V (`V`) is also not an availability flag:
the original source/energy pair is nonconjugate, so `V` remains false as a
chemical-theory statement. As authorized on 2026-09-05, the `1.5 kcal/mol`
target does not block experimental workflow development. The vibrational-only
driver separates frequency/mode output from gas-phase thermochemistry;
Richardson error and observed-topology guards remain enforced, with unobserved
PySCF-SMD topology explicitly reported. NEB/CINEB results are TS candidates,
not certified first-order saddles. PRFO/NEBTS, Dimer, solution-phase
thermochemistry, IRC/MD, periodic stress, and production admission remain
outside this availability decision. Historical frozen accuracy gates and
results above are not reclassified.

The daily calculator and the accuracy candidate are distinct profiles.  In
particular, the running canonical-ADT 505 evaluation may not silently replace
the daily force until its moving-geometry derivative is complete.

The final hybrid route must expose one content-addressed scalar per solvent and
derive all supported quantities from that same scalar:

- energy;
- conservative force;
- molecular virial/stress where defined;
- Hessian-vector products;
- Cartesian Hessian and frequencies;
- stable OPT/TS/IRC and, after independent validation, MD.

The implementation must remain modular in three independent spaces:

1. permanent and induced MLIP source models;
2. source-to-continuum and continuum-to-native-field operators;
3. additive solvent free-energy terms.

Source/receiver dimensions need not be artificially identical. Every operator,
checkpoint, solvent definition, cavity, unit convention, derivative route, and
runtime must be bound by stable provenance.

## Admission boundaries

Passing the 505 energy target establishes only development-set full-solvation
energy accuracy for the frozen profile. It does not by itself admit forces,
Hessians, optimization, frequencies, or MD.

Those capabilities require, on the same exact profile:

- cold/warm replay and root residual checks;
- analytic-force versus multi-step scalar finite differences on distorted
  geometries;
- translation, rotation, permutation, and torque covariance;
- closed-loop work and branch/topology guards;
- HVP/Hessian symmetry and finite-difference convergence;
- downstream OPT/FREQ/NVE validation appropriate to the claimed domain.

Strict common-functional Tier V is a separate claim. The operational hybrid
route may provide a conservative composite PES through an explicitly frozen
ledger and complete implicit differentiation without claiming that the
original MACE-MDP/MACE-POLAR response equations are stationary equations of a
single electronic functional.

## 2026-08-24 source-representation terminal audit

The canonical ADT 505 development replay is now complete, including the frozen
10-record iodine remediation.  It improves stock hybrid M1 MAE only from
`1.6963134653` to `1.6389621921 kcal/mol`; RMSE is `2.2778372822`, q95 is
`4.9849508792`, and the maximum error is `14.6888779571 kcal/mol`.  ADT is a
small positive result, not the route to the one-kcal/mol objective.

A matched-QM acetone audit then localized the failure.  MACE-MDP reproduces the
molecular polarizability within roughly `0.3--0.4%`, but the spatial ADT
boundary response has `49--69%` continuum-active error.  A separate frozen
12-molecule nonuniform response replay gives `46.5%` mean induced-MEP error for
the combined residual+ADT branch.  The unchanged zero-training source family
is therefore terminal for quantitative PCM.

Four new genuine Chrome `Pro, 5 of 5.` reviews, together with the earlier
matched-QM architecture review and independent local derivations, then examined
the next scalar-first representation.  Their full prompts, answers, UI proof,
and decisions are retained in:

- `docs/route2/evidence/hybrid-matched-qm-next-architecture-pro-20260824/`;
- `docs/route2/evidence/auxiliary-density-low-signal-gate-pro-20260824/`;
- `docs/route2/evidence/auxiliary-density-basis-ladder-pro-20260824/`;
- `docs/route2/evidence/auxiliary-density-conditioning-architecture-pro-20260824/`;
- `docs/route2/evidence/observable-supervised-scalar-head-pro-20260824/`.

The representation results are now terminal and must not be retuned:

1. `df.make_auxbasis` failed the original 12-case relative-MEP maximum gate
   because methane had a low-signal 19.63% ratio.
2. The same basis failed the more physical PCMSolver reaction-metric mean-bound
   gate: `0.330061 > 0.25 kcal/mol`, although its actual mean fixed-source error
   was `0.127611 kcal/mol`.
3. A preregistered cheapest-first standard ladder (ETB beta 2.0, AutoAux, ETB
   beta 1.5) gave excellent reaction bounds, respectively `0.03256`, `0.01914`,
   and `0.01388 kcal/mol` mean, but all three failed the frozen global
   coefficient-identifiability condition gate.
4. The sole allowed fixed per-element/per-l isolated-atom whitening attempt
   also failed unchanged `1e10` global conditioning on benzene and aniline
   (`3.18e10` and `3.85e10`).  The global auxiliary-density coefficient route
   is closed; no cutoff, pivot, intermediate beta, pruning, or alternative
   whitening may rescue it.
5. A smooth promolecular stockholder partition then produced unique and
   well-conditioned local coefficient labels, but their PCMSolver reaction
   bound failed at `0.707210` mean and `1.010777 kcal/mol` maximum.  Partition
   widths, grids, or local labels may not be tuned to rescue that result.

These failures do **not** say that continuous density/MEP resolution is
insufficient.  ETB beta 2.0 already bounds the fixed-source PCM representation
error near `0.03 kcal/mol`.  They say that neither global overlapping
coefficients nor independently fitted partition coefficients are acceptable
observables.  The next mainline is therefore one new scalar-first,
observable-supervised strongly convex latent-source architecture:

```text
frozen MACE-MDP geometry backbone
  + optional frozen zero-field MACE-POLAR latent features
  + fixed internal radial x spherical source/field span
  + strongly convex charge-constrained latent energy A_theta(R,z)
  -> zero-anchored concave external-potential scalar
  -> permanent and induced source from the same derivative
  -> direct QM energy / exterior-MEP / nonuniform-response supervision
  -> reciprocal continuum scalar only at held-out transfer/admission
```

It must not train on any global or partitioned density-fitting coefficient
vector and must not add the old MACE-POLAR residual/ADT as a physical source.
Only zero-field latent MACE-POLAR features may appear as frozen inputs in a
capacity-matched ablation.  The
chemically disjoint target-free confirmation identity was frozen before the
standard-basis ladder results in
`docs/route2/preregistrations/auxiliary-density-confirmation-selection-v1.json`.
Experimental solvation targets remain forbidden for representation or response
training; they remain final full-ledger evaluation only.

The experimental daily MAPLE energy ledger has also been component-replayed on
the tutorial water geometry.  Its reported `-76.4411151822 Hartree` closes to
`4.4e-11 Hartree` as vacuum MACE energy plus ddX polarization plus stock
SMD-CDS.  The associated solvation shift is `-4.7194708677 kcal/mol`.  This is
an internally correct experimental scalar, not a thermochemical Gibbs free
energy or a chemical-accuracy claim.

### Observable-supervised scalar training data

The final 2026-08-24 Pro checkpoint and local algebra close the source-label
question.  All global and partitioned density coefficient labels are excluded.
The retained electronic model is the zero-anchored constrained convex dual

```text
E_theta(R,v) = E_vac^MDP(R)
             + min_(a^T z=Q) [A_theta(R,z) - eta_R(v)^T z]
             - min_(a^T z=Q) A_theta(R,z),
```

with a registered strong-convexity floor.  Permanent and induced sources are
the zero-field value and field-induced increment of the same derivative.  The
NumPy and differentiable Torch quadratic references now pass charge, gauge,
energy/source finite-difference, reciprocity, PSD susceptibility, and concave
field-Hessian tests.  `passive_training.py` now admits the target-free kinds
`exterior_mep`, `nonuniform_field_energy`, and
`nonuniform_exterior_mep_response`; fixed-source PCM energy remains admission
only.

A new external-field-QM identity has been frozen before any MAPLE QM calculation:

```text
dataset:  Vector-QM24 DFT all / ColabFit
revision: baf79fff54725bb9a349f6234f2e23f0de22a147
license:  CC-BY-4.0
records:  60 unique molecular formulas
splits:   train 32 / validation 14 / blind 14
strata:   HCNO, F, P, S, Cl, Br
```

The selection scanned all 784,875 configurations and selected one geometry per
formula solely by a frozen SHA-256 rule.  It used and emitted no VQM24 energy,
model output, experimental solvation value, FreeSolv identity, or MNSol
identity.  The immutable artifact is
`docs/route2/preregistrations/vqm24-nonuniform-qm-geometry-selection-v1.json`.
The next execution must generate independent uniform and localized nonuniform
omegaB97M-V/def2-TZVPD field energies, exterior MEPs, dipoles, and responses on
these exact splits.  PCMSolver remains absent from training and model selection.

The first real VQM24 nonuniform-field Gate-0 datum is now complete on the frozen
CH6N4 training identity.  Its geometry-only probe shell uses the frozen
promolecular `1e-4 e/bohr^3` atomic radii plus `1.0 A` clearance and four
farthest exterior point-charge modes; no PCM cavity or solvent enters.

```text
zero-field RKS omegaB97M-V/def2-TZVPD:
  energy                    -260.7028882944692 Ha
  SCF cycles                16
  checkpoint density error  3.47e-18

16 localized perturbation SCFs:
  cycles                    6--8
  max electron-count error  4.97e-14 e
  two-step induced MEP diff 5.97e-4 relative
  two-step dipole diff      5.57e-4 relative
  energy curvature          negative for all 4 modes and both steps
```

The preregistration, checkpoint, energies, induced MEP/dipole tensors, source
hashes, and numerical seal are retained in
`docs/route2/evidence/vqm24-nonuniform-qm-gate0-20260824/`.  This is the first
actual training datum, not a trained model or capability admission.  Batch QM
execution must preserve the same immutable split and protocol.

The rare-element Gate-A is also complete.  The first SG1 attempt was rejected
before scientific execution because PySCF 2.13.1's SG1 pruning radii cover
only elements through Ar.  The replacement was preregistered uniformly for
all five records as PySCF level 3 for both semilocal and nonlocal grids; no
element-specific grid exception was introduced.  One frozen train record from
each F/P/S/Cl/Br stratum then passed:

```text
stratum  formula     gas cycles  response cycles  MEP step diff  dipole step diff
F        C2H6FNO     17          6--8             1.63e-4        1.30e-4
P        H3FP2S2     16          7--8             3.00e-4        2.79e-4
S        H3FS4       17          6--9             1.75e-4        1.41e-4
Cl       H2ClNP2S    20          8--11            5.72e-4        2.77e-4
Br       CH4BrNOS    17          6--9             1.74e-4        1.37e-4
```

Across all 80 perturbed SCFs, the maximum electron-count error is
`1.42e-13 e`; every registered energy curvature is negative at both field
steps.  The five content-addressed records and their aggregate seal are in
`docs/route2/evidence/vqm24-nonuniform-qm-gate-a-*-20260824/` and
`docs/route2/evidence/vqm24-nonuniform-qm-gate-a-20260824/`.  This proves that
the target-free localized-response data protocol is numerically viable across
the selected element range.  It does not measure a model error, train the
latent-source head, read VQM24 energies or solvation labels, or admit a MAPLE
capability.

The next scaling step must avoid spending 16 independently converged finite-
field SCFs per geometry if an existing static CPKS implementation can produce
the same linear response.  A CPKS backend is acceptable only after its density,
exterior-MEP, dipole, and energy-curvature actions reproduce the complete
Gate-0/Gate-A finite-field oracle within the already observed two-step error
budget.  Finite-field records remain the independent reference and may not be
discarded or relabelled as model accuracy.

That CPKS Gate-B has now been executed without changing its preregistered
thresholds.  Br, F, and S passed, but phosphorus missed the frozen absolute
energy-curvature budget (`3.60925e-4` observed versus `3.26104e-4 Ha/e^2`
allowed).  The aggregate MEP/dipole discrepancies were only `6.19e-5` and
`5.10e-5` symmetric relative, reciprocity/passivity passed, and the median
CPKS/finite-field wall-time ratio was `0.778`; none of those secondary results
overrides the failed gate.  CPKS remains a diagnostic rather than the batch
label generator.  The frozen evidence is
`docs/route2/evidence/vqm24-static-cpks-gate-b-20260824/`, and the genuine
Chrome Pro 5/5 derivation plus local accept/reject record is in
`docs/route2/evidence/vqm24-cpks-backend-pro-20260824/`.

The admitted data generator therefore retains directly converged
`q=+/-1e-3 e` states and stores their raw external enthalpies, full molecular
MEPs, and molecular dipoles.  The external point-charge--nuclear work is
included explicitly so the central energy slope is conjugate to the complete
zero-field MEP.  Numerical energy curvature is **not** a target.  An opened
fluorine pilot reproduces the prior Gate-A large-step MEP and dipole to
`2.06e-9` and `5.80e-10`, while energy-slope/MEP conjugacy closes to
`1.91e-7`; see
`docs/route2/evidence/vqm24-observable-training-pilot-fluorine-20260827/`.

Before the 32-record response batch began, a representation audit found that
the original 26-direction shell had fewer probes than the two-width `8N`
radial-GTO source dimension and could therefore be interpolated exactly.  It
was superseded, before any batch observable solve, by two target-independent
PySCF 50-point Lebedev exterior partitions: a fit frame and one frozen generic
rotation used only for audit.  Every molecule has more fit and audit probes
than `8N`.  On the opened fluorine pilot the same source span gives `1.12%`
fit and `1.47%` audit zero-field MEP error instead of a misleading roundoff
fit.  The sparse-v1 gas checkpoints remain reusable, while all response labels
are bound to
`docs/route2/preregistrations/vqm24-observable-training-batch-dense-v2.json`.

A further genuine Chrome `Pro, 5 of 5.` architecture review and same-chat
follow-up are frozen in
`docs/route2/evidence/mdp-polar-passive-head-architecture-pro-20260827/`.
The accepted response architecture is the direct sparse quadratic scalar
`K=C.T@C`, not an inverse-hardness/QEq or an inner nonlinear source solve.
The variable-N factor uses atom radial-null rows, two atom vector rows, and
local edge rows; its gauge, rotation, edge-orientation, and negative-
semidefinite Hessian tests now pass in
`maple/solvation/models/sparse_passive_factor.py`.

The first sulfur validation observable then changed the permanent-source
conclusion without changing the response conclusion:

```text
radial8 zero-field MEP audit     6.91%   (fails frozen 5%)
radial8 response MEP audit       1.27%   (passes frozen 3%)
radial8 + permanent point l=2    1.64%   zero-field audit
radial8 + induced point l=2      1.15%   response audit
```

This supports a split-order `P13/R8` scalar in theory: permanent radial8 plus
geometry-only `l=2`, with induced response remaining radial8.  The STF point-
quadrupole source/rotation kernel is implemented and tested.  However, the
subsequent preregistered overdetermined point-P13 span gate failed its audit/fit
generalization ratio on 8/32 records, despite every absolute audit error being
below `1.56%`.  That favorable absolute result cannot override the frozen
ratio gate; the point-P13 campaign is closed.  The named next permanent
representation is fixed-width Gaussian `l=2`, not a local energy residual and
not induced `l=2`.  The ongoing QM response batch remains fully reusable because
the response manifold and targets are unchanged.

The complete 32-record dense response batch is now available, but its data seal
is deliberately `fail`: one sulfur record has external-enthalpy-slope/full-MEP
closure `1.337e-5`, above the frozen `1e-5` gate.  Electron count and MEP/dipole
nonlinearity otherwise pass by wide margins (maximum nonlinear ratios
`3.38e-4` and `3.03e-4`).  Raw `E(+/-q)` remains diagnostic data, but numerical
energy slope is removed from the first training objective rather than relaxing
the gate; scalar energy/source conjugacy is structural and tested by AD.

The full radial8 representation gate also fails.  Five records exceed the
zero-field `5%` MEP gate, and one bromine record has response audit `2.48%` but
maximum absolute response error `0.00340 Ha/e`, above the frozen `0.002` limit.
Adding induced Gaussian `l=2` at `sigma=1.5 A` on that record reduces response
audit to `0.630%` and maximum error to `0.000837 Ha/e`.  Therefore the complete
data invalidates the earlier zero-induced-`l=2` premise: the next dynamic model
is full block-passive P13, initially with radial and local `l=2` factor blocks
and no cross factor.

The preregistered one-channel Gaussian-`l=2` coefficient-span ladder
(`1.0/1.5/2.0 A`) has no candidate that passes every three-pilot audit/fit
ratio gate.  This prohibits using any per-molecule fitted coefficient or its
gauge as a label; it does not substitute for the shared geometry head test.
The first 56-parameter element-linear shared head then failed honestly
(`31.9/32.1%` mean fit/audit MEP; `93.9%` maximum).  The bounded local-invariant
Q-A head and final Q-B head with two target-free frozen POLAR-l2 carriers also
failed (`21.1%` and `19.4%` mean audit MEP respectively).  The permanent-head
capacity sequence is closed without opening validation or blind formulas.
Frozen MDP feature extraction also confirmed that its checkpoint has no energy
property; it supplies descriptors only and no energy is invented.

### 2026-08-27 block-passive P13 response disposition

The permanent-head capacity sequence is now terminal.  The final allowed Q-B
398-parameter MDP plus target-free zero-field POLAR-l2 head improved over Q-A
but still gave `19.37%` mean and `32.90%` worst rotated-audit permanent MEP
error.  No validation or blind formula was opened, and MACE-MDP point q/p
remains the operational permanent affine source.

The preregistered scalar-first induced-response pilot then trained two MDP-only
full-P13 factors on the 32 train formulas.  Both generated the induced source
only from `d[-1/2 ||C_theta(R) eta||^2]/d eta` and passed charge, reciprocity,
passivity and fit/audit generalization to numerical precision:

```text
candidate  params  mean audit MEP  worst audit  mean dipole  mean source-point
A2         432     13.15%          21.99%       8.62%        14.01%
A3         634     11.43%          20.08%       7.64%        12.93%
```

A3 improves the frozen score by `12.85%` and reduces the historical
approximately `46.5%` nonuniform-response error by a factor of `4.07`, but it
still fails the frozen `8%` mean, `5%` source-point mean and `0.002 Ha/e`
absolute gates.  It is not selected.  Evidence is frozen in
`docs/route2/evidence/vqm24-mdp-p13-passive-response-pilot-20260827/`.

A fresh genuine Chrome `Pro, 5 of 5.` review selected one optimization-closure
attempt before any descriptor escalation.  The resulting analytic derivative
backend matches the scalar-autograd radial/l2 source to `1.11e-16`, the full
parameter gradient to `3.73e-17` absolute / `2.68e-14` relative, and the
energy/work identity to `1.39e-17`; gradcheck and gradgradcheck also pass.  A
bounded full-batch strong-Wolfe L-BFGS continuation reduced the total training
objective from `0.0169507` to `0.0116846`, but hit the frozen 500-step cap
without the required plateau.  Its last three 50-step blocks still improved by
`1.08%`, `1.44%` and `1.25%`, above the `0.25%` threshold.

Therefore this exact A3 identity terminates as `fail-a3-o2-no-plateau`:

- audit gates were not reopened after the hard-cap termination;
- grouped CV, validation and blind formulas remain sealed;
- A4 latent-descriptor escalation is not authorized by this run;
- radial-l2 cross-factor topology remains unidentified and forbidden;
- no MAPLE capability or solvation-accuracy claim follows.

The next scientific branch must improve response identifiability rather than
raise the failed candidate's post-hoc step/capacity budget.  The present four
point-charge inputs determine susceptibility action on rank at most four per
molecule.  The next preregistration should therefore add target-independent,
sector-balanced nonuniform input modes (with a separately validated finite-
field or response-only CPKS generator) or replace the nonconvex coefficient
parameterization by a globally certifiable response model.  Experimental
solvation labels, PCM/cavity targets and failed-candidate audit information
remain forbidden.  O2 evidence is in
`docs/route2/evidence/vqm24-mdp-p13-a3-o2-lbfgs-20260827/`; the Pro answer and
independent local decision are in
`docs/route2/evidence/mdp-polar-p13-postpilot-pro-20260827/`.

The first response-identifiability input artifact is now complete without
opening any new target.  For each of the 32 train geometries, the existing four
point-charge modes are retained and eight more are chosen from the dense fit
surface by constant-potential-gauge projection, separate radial/l2 RMS
balancing, and deterministic maximum-residual row-volume growth.  All 32
selected 12-mode matrices have full row rank; the maximum balanced condition
number is `4.366`, the minimum greedy residual is `0.879`, and the radial share
of balanced selected norm lies in `[0.486, 0.504]`.  Rotation and atom-order
invariance tests pass.  No QM response, model prediction, PCM/cavity value or
solvation label was read.  The artifact is
`docs/route2/evidence/vqm24-sector-balanced-p13-response-modes-20260827/`.
It authorizes only the next response-only CPKS-versus-finite-field generator
gate, not new training or a capability.

The CPKS generator path has now been narrowed before any expanded-mode QM
execution.  The already opened F/P/S/Br Gate-B records pass every MEP, dipole,
CPKS-residual, checkpoint, electron-count, reciprocity and passivity gate; the
phosphorus curvature failure is retained but is not reclassified as a pass.
Instead, the response-only pilot points to the pre-existing dense-v2 target
contract, which explicitly set numerical energy curvature to `false`.  The
pilot therefore authorizes only a new all-32 CPKS-versus-finite-field response
coverage run and does not yet admit the generator.

That 32-record coverage is now running as two independent 8-thread shards.  The
first completed fluorine and phosphorus records both pass.  Their global
CPKS/finite-field MEP discrepancies are `3.30e-5` and `8.80e-5`, and dipole
discrepancies are `2.67e-5` and `6.38e-5`, respectively, against the frozen
`5.8e-4` limits.  Twelve-mode QM response generation remains blocked until all
32 records pass and the aggregate is sealed.
