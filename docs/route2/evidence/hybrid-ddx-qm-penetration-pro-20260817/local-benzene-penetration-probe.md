# Local benzene QM/pyddx penetration probe

Date: 2026-08-17

## Claim boundary

This is a target-independent diagnostic, not a matched-ledger admission run.
It reads no experimental hydration free energy, CDS term, standard-state term,
or confirmation record.  It tests whether the exact small-cavity hybrid ddPCM
profile admits a stable diffuse-basis QM/ddPCM state suitable for the matched
component calculation requested in Q17.

## Runtime and profile

```text
Python       3.11.15 (conda-forge, GCC 14.3.0)
Psi4         1.11
pyddx        0.8.0
NumPy        2.4.6
SciPy        1.17.1
QM           W97M-V/def2-TZVPD, RKS/DF
solvent      water, epsilon=78.355
ddPCM        lmax=15, nleb=1202, eta=0.1, shift=0
radii        C=1.85 A, H=1.20 A before the diagnostic scale factor
geometry     frozen mobley_3053621 benzene geometry
```

The official Psi4 ddX energy/Fock derivative was separately checked on its
small H2 PCM construction.  At a density perturbation step of `1e-5`, the
analytic contraction and central finite difference differed by about
`3.6e-11 Eh`; therefore the benzene failure was not accepted as an unchecked
Fock-hook result.

## Exact-radius gas-density result

At the converged vacuum density:

```text
G_dd / Eh                  +0.0149328239063841
AO reaction V min/max / Eh -1.96100139697 / +1.46424521390
Frobenius norm(V)           9.94112846902
norm(xi)                    127.248184501
```

Starting PCM-SCF from the converged vacuum orbitals then left the physical
energy scale after the first update and visited total energies of roughly
`-1.4e5` to `-2.6e5 Eh`, with an orbital-commutator norm of order `1e2`.

A small basis sequence showed a real finite-AO stability boundary rather than
a universal failure of the Psi4 hook:

* W97M-V/3-21G converged at the exact radii, with
  `D_QM=+0.001268887692 Eh`, `S_QM=-0.008802350328 Eh`, and
  `T_QM=-0.007533462636 Eh`;
* W97M-V/def2-SVP remained near the physical scale initially, but failed to
  reduce the commutator and by iteration 27 had departed from `-231.97 Eh` to
  `-234.23 Eh`, after which the diagnostic was terminated;
* W97M-V/def2-TZVP (no diffuse suffix) moved from `-232.2213 Eh` to
  `-286.5018`, `-1151.1957`, `-3219.8672`, and `-4881.8570 Eh` within six
  updates;
* W97M-V/def2-TZVPD gave the most severe collapse described above.

Diffuse and flexible valence spaces strongly exacerbate the effect, but the
exact small cavity is not rendered stable merely by deleting the explicit
diffuse basis functions.  Conversely, the converged 3-21G result proves that
the software path is not categorically incapable of QM/ddPCM SCF.

A level-5 PySCF quadrature over the same converged vacuum density integrated
`41.9999999899` electrons and estimated `0.5474028424 e` (`1.30334%`) outside
the union of the exact C/H spheres.

## Radius-scale isolation

Holding the density and all other settings fixed:

| radius scale | G_dd / Eh | norm(V) | outlying electrons / e |
| ---: | ---: | ---: | ---: |
| 1.0 | +0.01493282391 | 9.94113 | 0.547403 |
| 1.1 | -0.00817805061 | 2.05690 | 0.310420 |
| 1.2 | -0.00266376576 | 0.27128 | 0.189533 |
| 1.3 | -0.00199232682 | 0.20391 | 0.112575 |
| 1.5 | -0.00133851997 | 0.11115 | 0.038969 |
| 1.8 | -0.00063655343 | 0.06101 | 0.006162 |
| 2.0 | -0.00040912285 | 0.04254 | 0.001677 |

At scale 1.5, the same diffuse-basis PCM-SCF converged from the vacuum
orbitals:

```text
D_QM / Eh                  +0.000198617301862
S_QM / Eh                  -0.001642223099765
T_QM / Eh                  -0.001443605797903
D_QM + S_QM - T_QM / Eh   -4.60e-17
```

This radius isolation, the direct outlying-charge estimate, and the failed
non-diffuse TZVP isolation support an electron-penetration/polarization-
catastrophe diagnosis rather than an ordinary diffuse-basis-only convergence
problem.  They do not by themselves select a replacement cavity or allow the
scale-1.5 result to be transported back to the exact hybrid profile.

## Content-addressed local inputs

```text
python executable      2c1935c0bce8fc50f5f12d5720d197dc03d3ff36432a2bf62799876b6616b6ca
Psi4 __init__.py       bf1d11f72cb32c5d8a62c309ed3a1a962beb79562a7f761084c942acf406a241
pyddx shared library   697bafe818a749bb70963ef13a36496bcba52fc427046e8eefe6e769a0d68845
Psi4 ddx.py            d436f2365a9131b524e6f438fd816b934534b8aa362d4a54a61d8de822d652f6
vacuum-orbital file    3514d26951e210cdf2413c4d8e2138526d2b165105e2c004788fe01bbfad9f11
```

The temporary exploratory scripts were not promoted to production code.  A
future admitted runner must freeze its own source, environment, geometries,
operators, densities, and replay artifacts before execution.
