# Independent mathematical and scientific review request

## Target

We are improving one fixed Route-2 hybrid only:

- permanent electrostatics: official MACE-MDP checkpoint;
- induced response: official MACE-POLAR checkpoint, zero-field anchored;
- continuum: existing pyddx ddPCM, no new solver;
- target: honest total solvation MAE near 1 kcal/mol, hard current goal <=1.5;
- no experimental-solvation fitting in the electronic source, no benchmark
  target leakage, no per-molecule correction, no spectral clipping, no hidden
  rescaling, and no claim of a strict common MACE/PCM functional.

We need a high-level mathematical decision and a minimal **zero-training**
composition of the two installed official checkpoints, not encouragement.  A new
trained head is a terminal fallback only after every physically derived candidate
has failed a preregistered, target-independent gate.

## Current population evidence

A preregistered 505-record development panel gives:

- hybrid electrostatics only: MAE 2.953989, RMSE 3.665386, max 20.066532 kcal/mol;
- plus stock SMD-CDS: MAE 1.696313, RMSE 2.351159, q95 5.061021,
  max 14.904481 kcal/mol;
- 88/505 errors >=3 and 27/505 >=5 kcal/mol;
- water 306: MAE 2.095807; nonwater 199: MAE 1.082016 kcal/mol.

The old N=4 value near 0.8 kcal/mol has been withdrawn as an accuracy claim; it
is only a mechanism replay.

For the worst record, 1,4,5,8-tetraminoanthraquinone in water:

- experiment: -8.900 kcal/mol;
- MDP permanent-only continuum: -21.747904;
- POLAR induced-only continuum: -0.671742;
- permanent-induced cross: -6.546886;
- total electrostatic: -28.966532;
- stock CDS: +5.162050;
- final: -23.804481, error -14.904481 kcal/mol.

The ddX root residual is 1.632e-11 eV and cold/wide roots both require 17
iterations.  Continuum linearity and energy decomposition close numerically.
This is not an unconverged-SCF explanation.

## Exact MACE-MDP architecture fact

The installed official `DipolePolarizabilityMACE` graph predicts atomwise
charge-like scalars, local dipoles, and polarizability contributions, then sums
them to molecular dipole/polarizability.  Charge is mean-projected to the total
charge.  Its training loss is only molecular dipole plus molecular
polarizability MSE.  We find no atomwise charge/dipole/ESP supervision in that
loss.  Therefore the atomwise partition has a large nullspace even when total
Q, molecular dipole, and molecular polarizability are exact.

An earlier frozen same-cavity QM/PCMSolver source panel accepted only 3/4 cases;
the flexible diester missed the fixed-source polarization energy by
1.823863 kcal/mol under an all-case 1 kcal/mol gate.

## New target-independent nullspace witness

On the worst development molecule, choose the most continuum-sensitive pure
atomic-local-dipole redistribution using only the fixed-cavity continuum energy
gradient.  Add +0.01 e-Angstrom to one atom and -0.01 e-Angstrom to another in
the same Cartesian component.  This preserves exactly:

- total charge;
- molecular dipole x, y, z.

Constraint residual is [0,0,0,0].  Yet permanent continuum energy changes by
+0.336911 or -0.356401 kcal/mol for the two signs.  Direct ddX replay agrees
with the exact quadratic prediction within 4.253e-14 eV.  The experimental
target was used to select the diagnostic tail molecule, but not to select the
gauge direction or perturbation.

## Official checkpoint pieces available without post-training

The installed checkpoints already provide two different kinds of information:

1. MACE-MDP exposes a public molecular dipole and polarizability.  Its latent
   atomwise q/p partition is not uniquely supervised and has failed the
   quantitative PCM source gate, but the public molecular dipole is accurate
   enough to use as a hard observable constraint.
2. MACE-POLAR exposes an atomwise four-channel q/p source at zero native field
   and a field-dependent source.  Its induced response can be written exactly
   as `c_POLAR(R,u)-c_POLAR(R,0)`.  The current coupling represents this source
   with its native 1.5 Angstrom Gaussian density, while the model-field receiver
   is a separate eight-channel object.

The first no-training candidate is therefore

    c_perm(R) = P_G[c_POLAR(R,0); Q_MDP(R), mu_MDP(R)]
    c_total(R,u) = c_perm(R) + c_POLAR(R,u) - c_POLAR(R,0),

where `P_G` is an exact differentiable projection onto total charge and the
MACE-MDP molecular dipole.  The metric `G` must be fixed by analytic
Gaussian-density electrostatics, not fitted residual variances or an arbitrary
Euclidean norm.  Point, native-Gaussian, and physically derived neutral-atom
charge-penetration kernels should be compared without changing either
checkpoint.

SPICE/MBIS labels remain available only as an independent falsification panel.
They may evaluate Q/mu closure, exterior and cavity MEP, and fixed-source ddPCM
energy, but may not fit projection weights, widths, element scales, source
coefficients, CDS, or any solvation residual.  The 505 solvation development
panel and sealed confirmation set must not participate in this source decision.

An earlier frozen-backbone learned q/p prototype is retained only as evidence
that a naive post-trained q/p head does not close the physical source gap.  It
must not become the default next step.

## New prospective 60-molecule source-gate result

After freezing the question above, we implemented the first frozen-backbone
q/p head and prospectively locked a completely target-free source gate before
reading any MEP or fixed-source PCM result.  It uses 60 molecule-held-out SPICE
test-split configurations, identical SMD-water/pyddx ddPCM cavity operators,
and independent MBIS multipoles.  No MNSol/FreeSolv target, CDS value, or
solvation residual is read.

Against independent MBIS q/p through the identical fixed-source ddPCM map:

- original latent MACE-MDP q/p: MAE 8.191931, q95 12.719589, max 17.589141
  kcal/mol;
- first learned MBIS q/p head: MAE 2.534650, q95 6.440642, max 10.261000
  kcal/mol;
- the learned head improves 58/60 configurations and reduces energy MAE by
  69.06%.

However, the preregistered source gate still fails:

- at the physical radius scale 1.0, global q/p cavity-MEP RMSE improves only
  7.43% (required 50%), although 46/60 individual configurations improve;
- global relative q/p MEP RMSE changes 0.29758 -> 0.27548 at scale 1.0, but
  worsens 0.23419 -> 0.26700 at scale 1.25 and 0.19801 -> 0.26050 at 1.5;
- against MBIS q/p/Q/O it worsens at every radius except a small scale-1.0
  comparison;
- even exact MBIS q/p truncated against MBIS q/p/Q/O has global relative MEP
  error 0.1770, 0.1332, and 0.1098 at radius scales 1.0, 1.25, and 1.5.

The prototype enforces the frozen MACE-MDP molecular dipole exactly.  Across
these 60 cases the MACE-MDP versus MBIS molecular-dipole mismatch has mean
0.01998 e-Angstrom, maximum 0.06538, and essentially no correlation with the
original fixed-source energy error (Pearson about 0.022).  This suggests the
original error is mainly atomwise near-field partition, while the hard dipole
anchor may now limit the learned successor.  The current gate is failed and no
hybrid integration is admitted.

## Prospectively locked anchor/order decomposition

We then froze a second target-free decision before computing it.  On the same
60 molecules, same cavity nodes, same ddPCM operator, and same runtime:

1. We metric-projected exact MBIS q/p to total charge zero and the frozen
   MACE-MDP molecular dipole, using the first head's heldout q/p residual
   variances.  This hard anchor changes fixed-source ddPCM energy by only
   0.022134 kcal/mol MAE, 0.076415 q95, and 0.132971 maximum.  Its global
   relative cavity-MEP RMSE is 0.006992, 0.007897, and 0.008754 at radius
   scales 1.0, 1.25, and 1.5.  The preregistered <=1 kcal/mol and <=0.05 gates
   therefore both pass: the original MACE-MDP molecular dipole is a safe hard
   closure even though its latent atomwise partition is not.
2. Exact MBIS q/p has relative q/p/Q/O MEP RMSE 0.177030, 0.133201, and
   0.109847.  Adding exact MBIS quadrupoles lowers the remaining q/p/Q versus
   q/p/Q/O RMSE to 0.101722, 0.054199, and 0.035857.  The removed fractions
   are 42.54%, 59.31%, and 67.36%.  The preregistered rule required >=50% at
   every radius; q/p/Q therefore fails at the physically closest radius 1.0.

This resolves the dipole-anchor ambiguity and rejects a q/p/Q-only successor.
The remaining decision is specifically between an l=3-capable source and a
compact analytic density/potential representation that also treats near-field
penetration.  A recent 2026 preprint on polarizable atomic multipoles uses a
hierarchical q/p/Q representation with an analytic Gaussian-screened Coulomb
kernel; our near-cavity result suggests that radial/penetration information may
be as important as increasing angular order.

## Questions requiring a terminal mathematical answer

1. Critically assess the no-training candidate above.  Is using the original
   MACE-POLAR zero-field source as the atomwise permanent topology, then hard-
   constraining it to MACE-MDP Q and molecular dipole, physically coherent?
   State precisely what information comes from each checkpoint and what model
   identity is changed or preserved.
2. Derive the unique minimum-electrostatic-change projection for atom-centred
   Gaussian monopoles/dipoles.  Starting from the normalized 1.5 Angstrom
   Gaussian basis, derive the Coulomb Gram metric (including q/p blocks,
   inter-atomic blocks, units, signs, charged-system origin convention, and
   constant-potential gauge), then give the KKT solution and its coordinate
   VJP.  Explain how to remain nonsingular for one-atom, linear, symmetric, and
   nearly coincident geometries.  If the full Coulomb metric is not the right
   choice, give the physically correct alternative and prove why.
3. Decide which no-training source kernel should be evaluated first and why:
   (a) current point MDP permanent q/p, (b) projected POLAR-zero q/p with the
   native 1.5 Angstrom Gaussian kernel, or (c) a nuclei-plus-published-neutral-
   atom-density penetration term plus residual Gaussian multipoles.  Identify
   which choices can change near-cavity MEP while preserving Q and molecular
   dipole, and which would merely reparameterize the same insufficient space.
4. The exact MBIS decomposition says q/p/Q still leaves relative cavity-MEP
   error 0.1017 at radius scale 1.0, while q/p/Q/O is the reference.  Determine
   what part can plausibly be recovered without training from analytic charge
   penetration or translation of fixed atomic densities, and what part truly
   requires environment-dependent l=2/l=3 information that neither installed
   checkpoint exposes.  Do not invent unavailable outputs.
5. Give a prospective experiment that compares the finite no-training
   candidates without turning the 60-case panel into a tuning set.  Specify
   immutable candidates, nested molecule/chemistry-heldout use if model choice
   is unavoidable, physically meaningful MEP and A^-1 energy norms, and exact
   terminal stop rules.
6. State the scientific conditions under which post-training finally becomes
   justified.  It must be the last resort, use independent QM electrostatic
   labels only, freeze both official backbones, and never read experimental
   solvation targets or CDS residuals.  Distinguish a proof that the existing
   checkpoints lack required information from a mere failure of one kernel or
   projection.
7. For the best no-training candidate, list every derivative required for the
   operational implicit-adjoint conservative PES: MDP Q/mu position response,
   POLAR zero/field source response, projection metric and constraint
   derivatives, source-kernel coordinate VJP, continuum coordinate VJP, and
   model-field receiver VJP.  Do not mislabel this as Tier V.
8. Provide a finite decision tree with terminal failures.  We will not accept
   an open-ended sequence of widths, radii, per-element scales, response
   tempering, source patches, or solvation-target tuning.

## Requested final format

- `NO-TRAINING CANDIDATE VERDICT`
- `GAUSSIAN COULOMB PROJECTION DERIVATION`
- `FINITE PHYSICAL KERNEL DECISION`
- `TARGET-INDEPENDENT EXPERIMENT AND STOP RULES`
- `FORCE/ADJOINT CONTRACT`
- `POST-TRAINING LAST-RESORT THRESHOLD`
- `EXPECTED ACCURACY IMPACT AND REMAINING RISKS`
- end with the exact marker `HYBRID NO-TRAINING SOURCE DECISION`.
