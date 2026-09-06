# MDP/POLAR Ewald-gauge source: exact q/p/Q/O tail falsification

## Claim boundary

This is a target-independent, zero-training, adversarial-tail falsification of
one parameter-free source construction for the MACE-MDP + MACE-POLAR hybrid.
It is **not** an independent accuracy estimate and admits no capability.
Experimental solvation targets, CDS residuals, and confirmation records were
not read.  The seven molecules were already opened as worst source-error tails.

The candidate is

\[
V_{\rm cand}=V_{\rm point}[c_{\rm POLAR}(0)]
+V_{\rm Gaussian,1.5\AA}[c_{\rm MDP}-c_{\rm POLAR}(0)].
\]

The point term owns the short-range MACE-POLAR topology; the official 1.5 A
Gaussian correction restores the exact MACE-MDP molecular charge and dipole.
No coefficient, radial width, cavity, dielectric, or source scale was fitted.

## Matched oracle

The fixed-source reference is the SPICE MBIS atomic Cartesian q/p/Q/O potential
projected exactly into the pyddx real-spherical `l <= 3` basis.  Candidate and
reference use the same pyddx ddPCM cavity, operator, dielectric, geometry, and
surface projection.  The Cartesian-to-spherical conversion is a numerical
basis transformation obtained from analytic exterior-potential collocation;
it is not target fitting.  This oracle is a q/p/Q/O truncation, not full QM
density.

Preregistration:
`docs/route2/preregistrations/mdp-polar-ewald-gauge-l3-tail-v1.json`

## Frozen result

```text
records:                                    7
POLAR-zero point vs q/p/Q/O MAE:            2.2450718621 kcal/mol
POLAR-zero point vs q/p/Q/O maximum:        4.2909528465 kcal/mol
Ewald-gauge candidate vs q/p/Q/O MAE:       3.8560074021 kcal/mol
Ewald-gauge candidate vs q/p/Q/O maximum:   6.7712176420 kcal/mol
paired improvements:                        2 / 7
mean error reduction:                       -71.7542973703 %
POLAR-zero q/p/Q/O relative MEP RMSE:        0.2881359805
candidate q/p/Q/O relative MEP RMSE:         0.2746862988
far-field Q/mu closure:                     PASS
prelocked scientific decision:              REJECT
```

The candidate improves the aggregate q/p/Q/O cavity-MEP metric by about 4.67%,
but strongly worsens the matched ddPCM energy.  It fails the prelocked paired,
mean-reduction, and maximum-error gates.  This candidate is terminally rejected;
no interpolation weight, width, clipping, or per-element rescue is permitted.

The result also shows that adding high-order information to the *reference*
does not rescue this particular Ewald-gauge composition.  It does not prove
that all analytic-density or higher-multipole source constructions fail.

## Runtime

```text
Python 3.11.14
NumPy 2.4.6
SciPy 1.17.1
Torch 2.12.0+cu130
pyddx 0.8.0
GPU NVIDIA GeForce RTX 4060 Laptop GPU
MACE-POLAR device cuda
MACE-MDP device cpu
wall time 1:57.40
maximum RSS 2,988,708 KiB
```

## Integrity

```text
preregistration SHA256: 5807e62792b48f492200d1f464f3df5a3323fefd947ad09b941c585aadeac859
runner SHA256:          a84e090a7fee371f264e6730efe04d1c98e2e9088f631522f9471c136034201a
oracle SHA256:          54271ab07a123311a51eca9ba73fb4e8c5b6c98d75abc2138b1544d27df14ca9
aggregate payload SHA:  06b9fe30d915fa7ecbc5455588bbee45935db30eea1f012a600e80a92fa48114
aggregate file SHA256:  0c82e2a8e8e9de2cccfd61197b281ccd852a5247c52a2217adc05b06b4be6a11
```
