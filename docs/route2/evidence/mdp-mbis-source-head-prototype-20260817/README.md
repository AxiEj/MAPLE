# Frozen-backbone MDP-MBIS permanent-source prototype

This evidence addresses the largest identified error mechanism in the current
MACE-MDP + MACE-POLAR hybrid: the continuum sees an atomwise near-field source,
whereas the published MACE-MDP checkpoint identifies its original atomwise
charge-like/local-dipole partition only through molecular dipole and
polarizability losses.

The replacement is a **new research profile**.  It freezes the official
MACE-MDP backbone, reads invariant/vector node features from the two product
layers, and fits two small ridge readouts to independent SPICE 2.0.1 MBIS
atomic charges and dipoles.  A differentiable metric projection then preserves
the checkpoint molecular total charge and dipole exactly.  No MNSol, FreeSolv,
CDS residual, solvent identity, PCM energy, or experimental solvation target is
read during source construction.

## Label-integrity boundary

The curated SPICE subset contains 313 supported neutral molecules and 770
configurations.  Its source arrays are `float32`.  Reconstructing molecular
charge and dipole from MBIS atomwise labels gives:

| internal closure diagnostic | value |
| --- | ---: |
| maximum total-charge residual | 6.1103e-4 e |
| charge residual RMSE | 1.8082e-4 e |
| maximum dipole-component residual | 2.9766e-3 e Angstrom |
| dipole-component RMSE | 4.7341e-4 e Angstrom |

These residuals validate the dataset unit conversion and finite-precision
internal consistency.  They do not prove that MBIS is the unique optimal PCM
source.

## Molecule-held-out feasibility result

The prototype split is by molecule hash, so conformers never cross train,
validation, and test.  On 60 held-out molecules / 135 configurations / 7497
atoms:

| atomwise target | original MDP latent partition | projected MBIS head |
| --- | ---: | ---: |
| charge MAE | 0.294992 e | **0.021735 e** |
| charge RMSE | 0.373438 e | **0.030464 e** |
| dipole-component MAE | 0.052093 e Angstrom | **0.004415 e Angstrom** |
| dipole-component RMSE | 0.081783 e Angstrom | **0.007010 e Angstrom** |

The projection closes total charge to `1.78e-15 e` and the frozen checkpoint
molecular dipole to `4.15e-15 e Angstrom` in the prototype report.

## Actual runtime graph audit

The production-form research adapter was rerun on a held-out 58-atom SPICE
molecule using the RTX 4060 GPU, not merely tested with a synthetic projection:

| runtime property | maximum error |
| --- | ---: |
| rotation, atomwise charge | 1.82e-14 e |
| rotation, atomwise dipole | 1.61e-15 e Angstrom |
| translation, atomwise charge | 2.02e-14 e |
| translation, atomwise dipole | 6.38e-16 e Angstrom |
| permutation, atomwise charge | 1.19e-14 e |
| permutation, atomwise dipole | 6.47e-16 e Angstrom |
| total-charge closure | 2.08e-16 e |
| molecular-dipole closure | 1.39e-15 e Angstrom |

The same Torch graph supplies the coordinate VJP.  Its directional central-FD
error decreases from `2.23e-8` to `1.02e-9` as the step is halved from
`2e-4` to `5e-5` Angstrom.

## What this does and does not establish

Established:

- the original atomwise partition error is technically repairable without
  changing the MDP molecular dipole or using solvation labels;
- the learned source is rotation/translation/permutation covariant to numerical
  precision in a real checkpoint replay;
- its coordinate VJP is generated from the same differentiable source graph.

Not established:

- cavity-surface or exterior MEP accuracy;
- fixed-source ddPCM polarization accuracy;
- improvement of the 505-record hybrid solvation panel or its 14.9 kcal/mol
  maximum error;
- public energy, force, Hessian, virial, or MD capability.

The next admissible scientific action is therefore a target-independent QM MEP
and same-cavity fixed-source ddPCM gate.  Only a source profile that passes that
gate may replace the original latent MDP partition in a new MDP+POLAR hybrid
development run.

