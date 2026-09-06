# Local matched-QM decision before accepting external review

## Theorem-level statement

Let C map an atom-centred source representation to total charge and molecular
dipole. Matching C s (and, for response, C ds/dE) does not identify s:
every component in ker(C) leaves those molecular moments unchanged while it
can change the potential on a finite molecular boundary. Therefore exact
molecular charge, dipole, and polarizability are insufficient to prove a
quantitative PCM source.

A cavity-specific projection chosen after observing PCM errors would define a
new solvent/cavity-dependent model. It cannot be described as a harmless
coordinate correction of the frozen checkpoint.

## Direct evidence

The source-bound acetone audit uses omegaB97M-V/def2-TZVPD finite-field QM, a
separate def2-TZVP control, and the identical PCMSolver cavity.

- The MACE-MDP molecular polarizability reproduces the QM induced molecular
  dipole derivative within 0.32--0.42 percent.
- The MDP permanent point-q/p boundary MEP has 31.29 percent area-weighted
  relative error and 35.04 percent continuum-active relative error.
- ADT induced boundary potentials have 49.29--68.70 percent
  continuum-active relative error over the three axes.
- Those ADT errors are 7.49--13.85 times the independently frozen basis and
  finite-field uncertainty.
- The complete ADT 505-development MAE is 1.638962 kcal/mol versus 1.696313
  for the parent hybrid; water remains 2.011536 and the maximum error remains
  14.688878 kcal/mol.
- The earlier molecule-held-out MDP-feature MBIS q/p readout failed its
  physical gate at 2.53465 kcal/mol fixed-source ddPCM MAE.
- The tested Ewald/l=3 tail candidate also failed its adversarial energy gate;
  merely adding a fixed high-multipole correction is not an identified repair.

## Evidence-supported inference

The canonical ADT allocation and the latent MDP point-q/p partition must not
advance as quantitative PCM sources. The accurate MDP molecular polarizability
should remain a molecular constraint/diagnostic, not evidence for the current
atomic response distribution.

No evidence yet terminates the original MACE-POLAR nonuniform residual:
the completed matched-QM audit probes uniform fields and compares ADT, not a
continuum-like nonuniform perturbation. Retaining that branch requires one
finite terminal test against QM induced boundary potentials under predeclared
nonuniform external electrostatic perturbations.

## Unknowns

- Whether the MACE-POLAR residual reproduces nonuniform QM density response.
- Whether a frozen-feature source head in an l<=3 or compact density/potential
  basis can pass molecule-held-out permanent boundary-MEP gates.
- Whether one scalar-first field-energy head can reproduce nonuniform response
  while satisfying reciprocity and passivity.
- Whether any source/response model passing component gates reaches 1 kcal/mol
  total-solvation accuracy without changing the CDS model.

## Provisional next experiment, not a frozen architecture

Use predeclared weak external point-charge/dipole perturbations outside several
molecular shells. For each perturbation compare primary/control-basis QM
induced boundary potential against:

1. the full frozen MACE-POLAR induced response;
2. its removed uniform tangent;
3. the retained nonuniform residual;
4. ADT alone and residual+ADT.

The perturbations and cases must be selected before response values are opened.
If the residual error exceeds five times the independent finite-field/basis
uncertainty on the conjunctive gate, retire the original residual. If it
passes, it may remain a separately identified branch while a new permanent
source and uniform/spatial response representation are developed.

## Post-Pro critical adoption

The genuine Pro 5/5 review independently reaches the same terminal decision and
selects one cavity-independent scalar generating-functional head over a
nonuniform potential basis. This is accepted with four local qualifications.

1. Reuse the existing FieldEnergyFunctional and passive-polarization
   infrastructure. A second scalar/derivative framework is unnecessary.
2. Do not globally hard-constrain the MDP molecular dipole or polarizability
   from the single acetone response result. First require a molecule-held-out
   QM panel proving those molecular observables within a frozen tolerance.
   Until then they remain supervised anchors or diagnostics.
3. The Pro numerical potential thresholds are design suggestions, not
   mathematical constants. Final thresholds must also be derived from a
   predeclared continuum-active fixed-source energy budget.
4. A new one-case constrained oracle shows that atom-centred point multipoles
   remain inadequate even through l=3: the middle-shell relative MEP error only
   changes from 25.29 percent at l<=1 to 23.95 percent at l<=3, while the
   coefficient norm grows to 16.58. The next representation must contain
   radial density/potential resolution; no point-l3 head will be trained.

The old MACE-POLAR residual will not be directly added to the new physical
source. It may enter only as a frozen feature and only after the one-use
capacity-matched ablation proposed by Pro. This preserves scalar-generated
reciprocity and passivity.

## Current-profile nonuniform-response terminal replay

After the review, the exact current eight-channel radial-GTO/ADT profile was
replayed on the already preregistered twelve-molecule localized point-charge QM
response data. This closes the previously listed unknown without new QM runs.

Mean weighted induced-MEP errors are:

- original MACE-POLAR response: 66.95 percent;
- retained radial residual: 91.62 percent;
- ADT: 55.69 percent;
- radial residual plus ADT: 46.45 percent.

Every branch passes zero of twelve records under the inherited 20 percent gate.
The old residual is therefore retired as a directly additive physical source.
It may be tested only once as an input feature to a future scalar head, as
specified by the Pro capacity-matched ablation.

## Auxiliary-density representation evidence

A no-fit Coulomb-metric projection into the standard PySCF make_auxbasis
density basis, followed by exact electron-count and first-moment constraints,
gives cavity-MEP errors of 2.68, 1.81, 1.13, and 1.84 percent on the four
previously opened QM cases. This supports a prospectively frozen broader basis
gate. By contrast, atom-centred point multipoles through l=3 remain near 24
percent on acetone and are retired.
