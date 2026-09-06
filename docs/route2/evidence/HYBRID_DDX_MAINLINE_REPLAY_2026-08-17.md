# Hybrid separated-source ddPCM mainline replay — 2026-08-17

## Scope and decision

The separated-source `pyddx` ddPCM implementation is not a new replacement for
the frozen v3 accuracy calculation: it is the continuum already used by that
505-record baseline.  The failed smooth weighted-shell harmonic experiment
therefore returns the hybrid route to its existing ddPCM mainline rather than
changing the v3 scientific identity.

The replayed state is

```text
permanent source = frozen MACE-MDP point monopoles/dipoles
induced source   = MACE-POLAR 1.5-A Gaussian [M(R,u) - M(R,0)]
continuum        = pyddx ddPCM
receiver         = MACE-POLAR 1.5/3.0-A external-MEP field
```

All calculations used the hybrid checkout explicitly through `PYTHONPATH`.
MACE-MDP remained on its frozen CPU/float64 path; MACE-POLAR used CUDA.  A GPU
attempt for MACE-MDP correctly failed because the checkpoint profile is
currently content-bound to CPU/float64.  This replay does not weaken that
provenance gate for speed.

These are target-independent electrostatic/derivative diagnostics.  The
four-case component comparison uses already frozen QM/PCMSolver component
references, not experimental total-solvation targets.  Nothing here opens a
public capability or authorizes a new CDS fit.

## Current-tree analytic-force canary

One real-checkpoint water calculation reproduced the existing block-adjoint
force path at base HEAD
`0e64f187748bd3aabab9eb2797a7f7affab47c8b`:

| quantity | result |
| --- | ---: |
| root residual | `4.3453e-13 eV` |
| adjoint residual | `2.0385e-11 eV` |
| directional FD error, `h=2e-4 A` | `5.3145e-7 eV/A` |
| directional FD error, `h=1e-4 A` | `1.3468e-7 eV/A` |
| directional FD error, `h=5e-5 A` | `3.0093e-8 eV/A` |
| maximum net-force component | `2.325e-16 eV/A` |

The approximately fourfold decrease under step halving is the expected central
finite-difference regime.  It proves only this local same-scalar derivative
canary, not a global PES, chemistry, or topology domain.

Artifact:

```text
/tmp/mace-mdp-polar-ddx-water-analytic-force-canary-current.json
SHA256 5b4bad1f710eb709406b065b27726278715e6eab120eea98bfefd951fb3167d8
```

## Rotation convergence diagnostic

The same rigid water rotation was evaluated at three numerical resolutions:

| `lmax` | Lebedev | total energy drift (eV) | maximum force covariance error (eV/A) | scaled force error |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 194 | `2.2231e-5` | `2.6979e-3` | `2.9264e-3` |
| 15 | 1202 | `3.4428e-8` | `1.3204e-4` | `1.1209e-4` |
| 15 | 5810 | `5.8864e-8` | `1.1479e-5` | `1.0804e-5` |

The force covariance error improves strongly with angular resolution.  Energy
drift is already chemically negligible at `lmax=15`, but it is not monotone in
Lebedev order and the finite point quadrature is not a structural SO(3)
intertwiner.  Therefore this is convergence evidence only; strict rotational
equivariance is not claimed.

Script/output digests:

```text
l8/n194 script  3914f1a6e27850c1cae7fb024851cb1edc49f0a31fc35ef3d7524fad46ac3021
l8/n194 output  7d7a235d0e910efaad3a92ef955bb7f6a5af7ed863a6200638222661515da53f
l15/n1202 script a9449cb93ccd94a3843015aaf841eed80e1df9469dab60b8b58eeacd1e60f13f
l15/n1202 output cd6ed24b5d685d2314e14250abfb3b02f0e00a4473758582f7e0013c8042f733
l15/n5810 script 7bf3aba2f40effb09e489b60a89c32a6d3e549c05ee1cc4f1150b4bebf996a6f
l15/n5810 output 52ec6c0fca6a80f664996bff6c0b695f65411804f75b9f4c2afea4eb31a622a5
```

## Source-component and harmonic-order convergence

The acetic-acid self-consistent induced source was frozen once.  Re-solving
only the linear continuum decomposition isolates the permanent, induced, and
bilinear cross terms:

| `lmax` | permanent | induced | cross | direct sum |
| ---: | ---: | ---: | ---: | ---: |
| 4 | -10.167094 | -0.019928 | -0.882920 | -11.069942 |
| 8 | -10.214505 | -0.019946 | -0.883570 | -11.118020 |
| 12 | -10.227330 | -0.019952 | -0.883864 | -11.131146 |
| 15 | -10.231267 | -0.019954 | -0.883956 | -11.135178 |

All values are kcal/mol at 1202 Lebedev points.  A higher-order 5810-point
diagnostic gives direct-sum energies `-11.136637`, `-11.139370`, and
`-11.140705 kcal/mol` for `lmax=15`, `18`, and `20`, respectively.  Thus the
frozen v3 choice (`lmax=15`, 1202 points) is within about `0.0055 kcal/mol` of
this `lmax=20`, 5810-point diagnostic for this case.

The superficially smaller four-case MAE at `lmax=8`/194 is not used to select
that cheaper profile: it is error cancellation against a different-equation
reference, not convergence.

Script/output digests:

```text
scan script       8bd9c5bff7687919a7f92c396c5634f1dea30c152326e4e9af52b68d2c01d04c
scan output       b61502eab1d841c431f55ccc378b9135b071a21085337ffdf1019dc9239336bc
high-L script     4f702a58122cb50162861a8e57b2838a4711f15918186b7eafd15ca2a8b049a6
high-L output     4dbc1f9b7a443c4fe2740c5834697ba8ffe6575390a6157c13babdba831ff241
l8 panel script   585c82a59c43c14cb54f2228267d4b050a20c8cd3d60559336b9568a272901e7
l8 panel output   e143991b10ed028773b72585486b22f2abfd2d67e8457f51772d6e6737bf91b1
```

## Same frozen ddX source in PCMSolver

Feeding the identical frozen induced coefficients and permanent point moments
to the fixed 454-point PCMSolver cavity gives:

| backend/equation | direct-sum energy (kcal/mol) |
| --- | ---: |
| ddPCM, `lmax=15`, 1202 | -11.135178 |
| PCMSolver C-PCM | -10.798506 |
| PCMSolver IEFPCM | -10.744120 |

The `0.337 kcal/mol` ddPCM/C-PCM and `0.391 kcal/mol` ddPCM/IEFPCM differences
include continuum equation/discretization/cavity differences.  They are small
enough to keep the ddPCM route scientifically useful, but they forbid treating
the PCMSolver component as an exact same-equation parity oracle.

```text
comparison script 8228873eb9db15ee3f9a84b2ebc50ee73690765d70fe3a5c0a5b113bae634631
comparison output 8e30ef3ec8e58cf1c7be5a2f10078b34bcc8bd5f5836f38cdfada667c678a9d2
```

## Four-case electrostatic component replay

At the frozen v3 numerical resolution (`lmax=15`, 1202 points):

| compound | hybrid ddPCM - QM/PCMSolver reference (kcal/mol) |
| --- | ---: |
| benzene | -1.763092 |
| acetone | -0.236395 |
| acetic acid | -0.151746 |
| 2-acetoxyethyl acetate | +1.325125 |

Aggregate metrics:

```text
MAE  = 0.8690891864 kcal/mol
RMSE = 1.1116830821 kcal/mol
MSE  = -0.2065268149 kcal/mol
max  = 1.7630916952 kcal/mol
```

This is a promising small component panel, not a release gate.  It mixes a
ddPCM prediction with a fixed PCMSolver reference and contains only four
chemistries.  In particular, the opposing benzene and ester errors show why a
small mean must not be used to justify a global scale correction.

```text
panel script 7b89e0fe4d1d12c3998c4a05af73006f8b375cd699852f3d3a72843b63a1db10
panel output a6ec769e010b0b1471d74f98ed078c4d0add2ab211419da33d04226138b0aede
```

## Frozen 505 baseline and remaining scientific question

The completed v3 profile already used this separated-source ddPCM method plus
stock PySCF SMD-CDS:

```text
505 MAE   1.6963134653 kcal/mol
505 RMSE  2.3511594923 kcal/mol
505 MSE  +1.1173060192 kcal/mol
505 max  14.9044813606 kcal/mol
water 306 MAE 2.0958072825 kcal/mol
```

The positive-parent PH1 CDS experiment lowered the mixed-505 grouped-OOF MAE
upper bound to `1.4546582`, but its water grouped-OOF bound remained
`1.6969972`; its conjunctive gate failed and no deployment fit exists.

The immediate scientific question is consequently not whether to invent a new
continuum.  The earlier field-semantics terminal audit already established that
MAPLE's native eight-channel injection returns a differentiable **raw** scalar
which omits the explicit `+E dot mu` work included by the upstream uniform-field
branch.  Its zero-field, AD/FD, and charging identities pass, but the raw
`Phi1Delta` ledger is not a complete external enthalpy.

The remaining question is whether a rigorously reconstructed, non-double-counted
external-work ledger and a matched QM/PCM distortion-versus-continuum
decomposition admit a better operational scalar, or whether the unchanged
MDP/POLAR sources require a separately trained scalar-first successor.  That
choice must be target-independent and was escalated to a fresh genuine Chrome
Pro review before implementation.

## Current admission boundary

The following remain false:

- strict common-functional Tier V;
- quantitative certification of unchanged MDP q/p as a standalone PCM source;
- structural SO(3) continuum equivariance;
- broad distorted-PES and multi-geometry analytic-force admission;
- Hessian/FREQ/OPT/TS/IRC/MD admission;
- any public hybrid solvation capability.

The replay supports retaining ddPCM as the operational continuum while the
source/energy-ledger interpretation is decided.  It does not authorize fitting
another CDS model or choosing a numerical resolution by experimental error.
