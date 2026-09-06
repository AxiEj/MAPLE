# Pro audit prompt: terminal zero-training near-field source decision

You are reviewing one narrow theoretical-chemistry decision for a hybrid
MACE-MDP + MACE-POLAR implicit-solvent model. Please reason mathematically and
chemically, not encouragingly. Do not propose fitting to experimental
solvation free energies, changing cavity radii to reduce error, scaling source
coefficients, or training a residual correction.

## Frozen model roles

The current final zero-training candidate uses only two unchanged official
checkpoints:

```text
permanent source    = MACE-POLAR zero-field atomwise point q/p
nonuniform response = POLAR(R,u)-POLAR(R,0)-J_POLAR(R,0) U G u
uniform response    = canonical atomic-density-translation lift[-alpha_MDP G u]
```

Here `u` is the checkpoint-native eight-channel potential feature; `U` embeds
the three uniform potential-gradient coordinates and `G U = I`. MACE-MDP is
used only for its supervised molecular polarizability `alpha_MDP`; its latent
atomwise q/p partition has been retired. The nonuniform MACE-POLAR response and
the ADT response are separate source categories in one pyddx ddPCM solve. No
experimental solvation target or fitted parameter defines this construction.

The candidate has already passed water root, charge, JVP/VJP, finite-field,
local-Jacobian, passivity, and replay checks. The first source-selected 45-atom
adversarial molecule has now also passed all preregistered Stage-A mechanism
gates. The rest of the twelve-case screen is still running. This question does
not ask you to predict its final score or to promote it to a variational
functional.

## Target-independent source evidence

On a prospective 60-molecule SPICE/MBIS panel, the MACE-POLAR zero-field point
q/p source, evaluated with the same cavity and ddPCM equation as the MBIS q/p
reference, gives fixed-source energy

```text
MAE / q95 / max = 1.460958 / 3.670878 / 4.530936 kcal/mol.
```

Closing the public MACE-MDP molecular dipole through the native POLAR uniform
response reduces MAE by only 2.23% and fails the tail gate. In a separate exact
MBIS decomposition, adding atomic quadrupoles removes 42.54% of the q/p-to-
q/p/Q/O cavity-MEP error at the physical cavity radius, but adding octupoles
and quadrupoles is needed for the remaining near field. A recent preprint,
arXiv:2605.05746, likewise finds systematic gains from monopole/dipole/
quadrupole hierarchies, but its multipoles are learned and therefore do not
provide a zero-training completion here.

The old full 505 development panel used the latent MDP permanent source. Its
stock-CDS total MAE was 1.696313 kcal/mol with a 14.904481 kcal/mol maximum.
The currently running canonical-ADT candidate retains that old permanent
source and changes the first 52 paired predictions only weakly: the M1 MAE
changes 1.428517 -> 1.373994 kcal/mol, the prediction correlation is 0.998715,
and the median polarization shift is 0.001046 kcal/mol. This is development-
only interim evidence, not a population conclusion, but it reinforces that
the permanent near-field source rather than the uniform response is the main
remaining mechanism.

## Proposed parameter-free penetration primitive

The repository contains content-addressed analytic neutral-atom Gaussian
mixtures from independently sourced atomic densities. For atom A they define

```text
delta V_A(r) = Z_A/r - V_e,A(r)
             = sum_k N_Ak erfc(sqrt(alpha_Ak) r)/r.
```

This correction is spherical, smooth away from the nucleus, carries exactly
zero total charge and dipole at infinity, and has no fitted solvation
parameter. It represents nuclear/electronic charge penetration omitted by a
pure point-multipole source.

However, standard ddPCM assumes the solute charge is supported inside the
cavity. If a neutral spherical core-plus-electron distribution is fully inside
its parent sphere, Gauss' theorem gives zero exterior correction, so it cannot
change the cavity MEP. If its Gaussian tail reaches outside the cavity, adding
only `delta V` on the boundary may be an outlying-charge modification that is
not the derivative of the same standard ddPCM scalar. The repository has
therefore not inserted this primitive into the production continuum.

## Questions

1. Give a decisive mathematical answer: can the above neutral-atom
   penetration primitive be coupled to ddPCM/CPCM in a parameter-free,
   stationary, energy-conjugate way when some source density lies outside the
   cavity? If yes, derive the parent functional and the exact additional
   volume/boundary terms. Identify the source-to-boundary map, receiver, energy,
   and nuclear-coordinate derivative ownership. If no, state the no-go
   assumptions precisely.

2. Distinguish three cases rigorously:
   (a) all neutral-atom density is inside its own sphere;
   (b) analytic Gaussian tails cross the dielectric boundary;
   (c) a smooth isodensity or weighted dielectric replaces the sharp cavity.
   In which cases can the penetration term materially alter polarization
   energy without violating the declared electrostatic functional?

3. Is there any uniquely determined, SO(3)-equivariant zero-training local
   l>=2 completion obtainable from the frozen outputs available here
   (POLAR atomwise q/p and response; MDP molecular dipole and polarizability;
   element identities; geometry; free-atom radial densities)? Either construct
   it and prove uniqueness/conjugacy, or prove the relevant non-identifiability.
   Do not treat unreleased hidden tensors as physical multipoles.

4. Could derivatives/translations of the free-atom radial densities generate
   a legitimate local quadrupole/octupole response fixed by `alpha_MDP`, or do
   they determine only the dipolar ADT subspace already used? Analyze the
   representation rank and whether any higher multipoles would be arbitrary.

5. Design the smallest target-independent terminal experiment that separates:
   - missing local l>=2 multipoles;
   - radial charge penetration;
   - ddPCM outlying-charge/operator error;
   - incorrect continuum energy ledger.
   Specify matched quantities and identities, not empirical thresholds chosen
   after seeing results. Direct QM density/MEP may be used as evaluation-only
   evidence; experimental solvation targets may not be read.

6. If standard sharp-cavity ddPCM cannot admit this correction, compare the
   scientifically nearest existing alternatives rather than inventing a new
   solver: an outlying-charge-corrected PCM, density-based smooth dielectric,
   IEFPCM/SS(V)PE variant, or a trained scalar-first electrostatic head. Which
   is the minimal defensible continuation and what exact evidence must bind it?

7. Give a finite stop rule. If the current POLAR-zero/MDP-alpha-ADT candidate
   passes mechanism gates but fails source/chemical accuracy, when must the
   zero-training route be declared exhausted and a separately named small head
   trained only on independent QM density/ESP/multipoles be accepted as the
   next scientific model?

Separate structural proof, numerical verification, and chemical calibration.
End your answer with the exact marker:

`ZERO-TRAINING PENETRATION DECISION`
