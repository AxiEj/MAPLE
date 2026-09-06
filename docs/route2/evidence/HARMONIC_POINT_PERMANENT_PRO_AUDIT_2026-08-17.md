# Harmonic point-permanent terminal audit — 2026-08-17

## Decision

The current smooth weighted-shell harmonic C-PCM candidate is **not** the
hybrid Route-2 production continuum.  It passes isolated-sphere normalization
identities, but it fails a same-cavity, same-source C-PCM fidelity check for a
multi-atom molecule and becomes more overpolarizing as the retained harmonic
degree grows.

This closes only this particular weighted-shell weak form.  It does not reject
the MACE-MDP point permanent source, the zero-anchored MACE-POLAR induced
source, or harmonic coefficient methods in general.  The mainline reuses the
already implemented separated-source `pyddx` ddPCM backend instead of creating
another continuum solver.

No experimental target, CDS coefficient, confirmation record, or FreeSolv/
MNSol total solvation value was read to make this decision.  Acetic acid was
the first compound of the already frozen four-case electrostatic-component
panel.

## Correction retained in the failed candidate

The audit found and fixed a separate finite-dielectric bug.  C-PCM requires

```text
f_eps = (epsilon - 1) / epsilon
field = -f_eps * S.T * sigma
G_pol = -f_eps/2 * b.T * A^-1 * b
```

The factor is now bound into the candidate configuration identity.  Exact
single-sphere point/Gaussian monopole and dipole tests cover conductor limit,
`epsilon=2`, and `epsilon=78.355`; their errors are approximately `1e-14`.
Fixing this bug is necessary, but it does not repair the multi-sphere weak
form.

## Target-independent local measurements

### Same fixed source, same GEPOL cavity

The induced source was solved once and then frozen.  Both PCMSolver equations
used the same 454-point SMD intrinsic cavity, the same MACE-MDP point
monopoles/dipoles, the same 1.5-A Gaussian induced increments, and
`epsilon=78.355`.

| equation | permanent | induced | direct sum |
| --- | ---: | ---: | ---: |
| IEFPCM | -10.21937597 | -0.00820868 | -10.72330605 |
| CPCM | -10.27145306 | -0.00824611 | -10.77759806 |
| CPCM - IEFPCM | -0.05207709 | -0.00003742 | -0.05429200 |

Units are kcal/mol.  Therefore the IEFPCM/CPCM equation distinction is real,
but it is two orders of magnitude too small to explain the harmonic error.

PCMSolver C-PCM also converges stably as the GEPOL target area is reduced:

| target area | cavity points | permanent energy (kcal/mol) |
| ---: | ---: | ---: |
| 0.9998981512 | 454 | -10.27145306 |
| 0.5 | 765 | -10.27908235 |
| 0.25 | 1533 | -10.28363118 |

### Smooth weighted harmonic C-PCM

For the same point-permanent source, transition width `0.18 A^2`, and finite
dielectric factor, the candidate gives:

| physical L | permanent energy (kcal/mol) | `cond(A)` | QR reduced `cond` | direct - QR |
| ---: | ---: | ---: | ---: | ---: |
| 1 | -12.20283373 | 680.80 | 55.03 | `1.07e-14` |
| 2 | -31.13667879 | 4572.84 | 65.00 | `-4.16e-13` |
| 3 | -37.36032136 | 18199.94 | 82.36 | `1.06e-12` |

The full-rank QR coordinate change leaves the stationary energy unchanged to
roundoff.  Consequently, ordinary coefficient conditioning is not the primary
failure, and deleting small modes would change the scientific model rather
than precondition the same equation.

The same earlier fixed-source decomposition showed that the Gaussian induced
component remains near `-0.0083 kcal/mol` while the point-permanent branch
runs away.  Narrowing the exposure transition from `0.18` to `0.05 A^2` did
not restore convergence.

## Mathematical diagnosis

The implemented matrix

```text
A = E.T * K * E
```

is the Coulomb energy of independent, exposure-weighted charges living on the
immersed collection of complete atom-centred sphere sheets.  It is not
automatically a conforming Galerkin discretization on the single physical
union boundary.  A finite band-limited reconstruction of a smooth exposure
cannot remain exactly zero on an open buried patch.  Increasing `L` therefore
admits increasingly localized charge on weakly exposed internal sheets.  A
nearby point multipole couples strongly to these artificial modes, whereas a
finite-width Gaussian source is much more strongly low-pass filtered.

Positive definiteness of every finite `A_L` is not a convergence proof.  A
valid weak form would additionally need a single-interface lifting or a
quotient by representation redundancy, a uniform lower frame/coercivity bound,
source separation from the physical trial support, and density of the lifted
spaces in the physical boundary space.  Those properties are not established
for the current weighted-shell construction, and the same-equation numerical
counterexample is already sufficient to reject it.

## Genuine Pro review

The retained Windows Chrome session was verified before submission:

```text
composer = Pro
power    = Pro, 5 of 5.
conversation = https://chatgpt.com/c/6a82451e-5958-83e8-9a94-9bdf52961f29
conversation control = conversation-options-WEB:d5e43b8a-882e-43e9-81a5-21f521654d1b
answer marker = HARMONIC POINT-PERMANENT DIAGNOSIS
```

Artifact digests:

```text
prompt SHA256:
fcc9ca67cef624e3694649bd479ccf2727dabe0f5961fec40aab14b0b8a2aecd

submission evidence SHA256:
cd2b838caa6e5fae119512e72f727172cacb58e64f25c66aec908fae2316a7fc

complete answer SHA256:
bb76db49f744339318789ef56f67c882f6c61a8ee3fec089fc7bb4365f2a215d
```

The external answer correctly separated two issues: finite-dielectric
IEFPCM/CPCM equation mismatch and an additional point-source-selective weak
form defect.  It also correctly predicted that a full-rank QR/SVD coordinate
change must leave the energy invariant.  These points were accepted only after
the independent local CPCM and QR experiments above.

One recommendation is deliberately not adopted: no positive-mode truncation,
geometry-dependent spectral deletion, or empirical regularizer will be used
to rescue this candidate.  Such operations change the response operator and
can introduce non-smooth mode selection.

## Reproducibility

Execution checkout:

```text
branch = research/route2-mdp-polar-hybrid-v1
base HEAD = 0e64f187748bd3aabab9eb2797a7f7affab47c8b
Python = 3.11.14
ASE = 3.27.0
NumPy = 2.4.6
SciPy = 1.17.1
Torch = 2.12.0+cu130
MACE = 0.3.16
PySCF = 2.13.1
pyddx = 0.8.0
CUDA = 13.0
```

Diagnostic source/output digests:

```text
PCMSolver IEF/CPCM comparison script:
c7d90cebe87d957e0008677aa861d06c7bae0b11e9d0cdec75827bf3a1c2d844
output:
2e3620f4a74778905006a9cca1871a56b8fe7a443f715651d1d4c5a6ed680051

PCMSolver C-PCM area scan script:
ec7bb52167706e8c040a520d0ed1fdbbe14604b38edff2c367ee7745d3b8226e
output:
20c33ded31eacbbfc77242b75a4222817d579f62fdb6c4bd984d26064a7f43e1

harmonic QR diagnostic script:
3797dfa802772af53777acfb8bd96375129eae8af801c661973dc0fd14ee749a
output:
e4d6ac73402492b231560c628041eebfbd547d51d0b54306aa6747ef94768f99

MACE-MDP checkpoint:
126f8d1602549e6fa0df775c701a5119ddeb0e3738202af8e7aa736de6c2b692
MACE-POLAR checkpoint:
fab8b8713c832f31a2a853aaa22fd638be8a369cbf5095e6b3e982a18d10e93a
PCMSolver libpcm.so.1:
296b6f34a03789943ae8823b3790f36357c50c896497f21374ac16fe5a6c43a3
```

The diagnostic scripts were executed with `PYTHONPATH` set to this checkout so
that the sibling pure-MACE-POLAR worktree could not contaminate imports.

## Mainline replacement

The repository already contains the required heterogeneous-source ddPCM route:

```text
MACE-MDP point permanent q/p
+ MACE-POLAR zero-anchored 1.5-A Gaussian induced q/p
+ shared pyddx ddPCM state
+ 1.5/3.0-A external-MEP receiver
```

It has prior four-case electrostatic evidence (`0.869089 kcal/mol` MAE) and a
real-checkpoint analytic-force canary (largest directional error
`5.31e-7 eV/A`).  Those are encouraging historical results, not automatic
admission for the current tree.  The next work is to replay component and
`lmax` convergence, rotation/covariance, distorted-PES, and force gates on one
new content-addressed ddPCM profile.  The failed harmonic result cannot be
combined with old evidence or used to open capabilities.
