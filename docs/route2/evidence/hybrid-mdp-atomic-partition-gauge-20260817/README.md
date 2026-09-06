# MACE-MDP atomic-partition gauge witness

## Decision boundary

This is a **development-only diagnostic**, not a fitted correction and not a
population accuracy estimate.  Selection index 199 was opened because it is the
largest full-505 M1 error.  The gauge direction itself uses only the frozen
continuum energy gradient; it does not use the experimental hydration target.
The confirmation partition remains sealed.

The official MACE-MDP architecture is supervised by molecular dipole and
molecular polarizability losses.  It constructs those observables by summing
learned atomwise charge-like, local-dipole, and polarizability contributions.
Consequently those aggregate labels do not, by themselves, identify a unique
atomwise source for a cavity-near-field continuum.

## Full-panel context

The copied 505-record component diagnostic is the accuracy evidence; the old
four-case replay is only a mechanism canary.

| ledger | MAE | RMSE | maximum absolute error |
| --- | ---: | ---: | ---: |
| hybrid electrostatics only | 2.953989 | 3.665386 | 20.066532 kcal/mol |
| plus stock SMD-CDS | 1.696313 | 2.351159 | 14.904481 kcal/mol |

For the stock-CDS ledger, 88/505 records exceed 3 kcal/mol and 27/505 exceed
5 kcal/mol.  Removing the worst case alone leaves MAE 1.670107 kcal/mol, so the
tail is not a single corrupt-record explanation.

## Worst-case component attribution

For 1,4,5,8-tetraminoanthraquinone in water, the converged electrostatic energy
is -28.966532 kcal/mol:

- permanent MACE-MDP source: -21.747904 kcal/mol;
- MACE-POLAR induced source alone: -0.671742 kcal/mol;
- permanent-induced cross term: -6.546886 kcal/mol.

Stock SMD-CDS adds +5.162050 kcal/mol, producing -23.804481 kcal/mol against
-8.9 kcal/mol experimental total.  The ddX root residual is 1.632e-11 eV and
cold/wide solves both take 17 iterations.  This is therefore not an
unconverged-SCF explanation.

## Gauge witness

A target-independent direction was chosen by the fixed-cavity permanent-source
energy gradient.  It adds +0.01 e-Angstrom to one learned local atomic dipole
and -0.01 e-Angstrom to another in the same Cartesian component.  It preserves
all four public aggregate constraints exactly:

```text
Delta total charge = 0
Delta molecular dipole x/y/z = 0, 0, 0
```

Nevertheless the permanent continuum energy changes by +0.336911 or
-0.356401 kcal/mol for the two signs.  Direct ddX replay agrees with the exact
quadratic prediction to at worst 4.253e-14 eV.

This proves a material near-field continuum sensitivity inside a null direction
of the checkpoint's public molecular charge/dipole observables for this case.
It does **not** prove a universal error bound, but together with the full-505
tail and the earlier 3/4 fixed-source gate it rejects treating the unchanged
latent atomwise partition as already source-validated.

## Consequence

Do not fit a solvent-energy scale, atom-specific patch, or geometry-dependent
radial correction to this source.  The next hybrid candidate must retain the
useful MACE-MDP molecular dipole/polarizability information while giving the
continuum a separately source-supervised, charge/dipole-constrained atomwise
representation.  A frozen-backbone readout trained on independent QM
MBIS/ESP data is permitted as a **new profile**; experimental solvation targets
remain forbidden for source construction.
