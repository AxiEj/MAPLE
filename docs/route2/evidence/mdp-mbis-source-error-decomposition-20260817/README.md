# MDP/MBIS permanent-source error decomposition

## Scope

This is a target-free, 60-molecule-held-out architecture diagnostic.  It reuses
the exact selection, SMD-water radii, ddPCM settings, checkpoints, source-head
asset, and runtime contract of the failed v1 source gate.  It reads no MNSol,
FreeSolv, CDS, standard-state, or experimental solvation target.

The old four-molecule replay is not accuracy evidence.  This panel also is not
a solvation-accuracy panel: it decides which permanent-source representation is
scientifically eligible for the next hybrid prototype.

## Prospective questions

The preregistration froze two decisions before these values were generated:

1. A hard MACE-MDP molecular-dipole anchor is allowed only when projecting exact
   MBIS q/p to that dipole changes fixed-source ddPCM energy by at most
   1 kcal/mol MAE and changes the cavity MEP by at most 5% at every radius.
2. A q/p/Q source head is allowed only when quadrupoles remove at least 50% of
   the q/p-to-q/p/Q/O global cavity-MEP RMSE at every radius.

## Results

The hard MACE-MDP molecular-dipole anchor passes cleanly:

| metric | result |
| --- | ---: |
| fixed-source ddPCM MAE versus unprojected MBIS q/p | 0.022134 kcal/mol |
| fixed-source ddPCM q95 | 0.076415 kcal/mol |
| fixed-source ddPCM maximum | 0.132971 kcal/mol |
| global relative cavity-MEP RMSE, radius 1.00 | 0.006992 |
| global relative cavity-MEP RMSE, radius 1.25 | 0.007897 |
| global relative cavity-MEP RMSE, radius 1.50 | 0.008754 |

This establishes that the frozen MACE-MDP molecular dipole is a useful hard
closure for a new permanent source.  It does **not** validate the checkpoint's
latent atomwise q/p partition.

Quadrupoles are useful but fail the frozen near-field criterion:

| radius scale | q/p vs q/p/Q/O relative RMSE | q/p/Q vs q/p/Q/O relative RMSE | removed fraction |
| --- | ---: | ---: | ---: |
| 1.00 | 0.177030 | 0.101722 | 0.425398 |
| 1.25 | 0.133201 | 0.054199 | 0.593103 |
| 1.50 | 0.109847 | 0.035857 | 0.673569 |

The closest cavity is the scientifically decisive case.  Because q/p/Q removes
only 42.54% there, a q/p/Q-only head is rejected under the preregistered 50%
threshold.  The error is therefore not reducible to a molecular-dipole scale or
one additional quadrupole readout; l=3 information and/or a compact analytic
density/potential basis is required to represent the cavity near field.

The v1 learned q/p head remains rejected: its fixed-source energy MAE is
2.534650 kcal/mol and its global relative cavity-MEP RMSE remains
0.275480/0.267003/0.260502 at the three radii.

## Architecture consequence

Do not integrate the v1 q/p head, fit a solvation-energy residual, or tune CDS
to compensate for this source defect.  The next source profile must:

* preserve total charge exactly;
* preserve the frozen MACE-MDP molecular dipole exactly;
* add l=3 or use a compact SO(3)-equivariant analytic density/potential basis;
* be supervised only by transferable QM electrostatic observables such as MBIS
  multipoles and/or cavity-independent ESP/density targets;
* retain molecule-held-out and chemistry-heldout evaluation;
* remain distinct from the MACE-POLAR induced-response source.

No E/F/H/V/M or hybrid integration capability is admitted by this result.
