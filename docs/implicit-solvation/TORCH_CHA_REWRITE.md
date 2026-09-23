# Source-faithful Torch CHA-GB rewrite: staged evidence

This is a **separate research implementation**, not a new MAPLE solvent
provider. The existing `AmberToolsChaGB` scalar remains the only exposed
CHA-GB/PBSA endpoint, supports SP only, and continues to return exactly
`EGB + ECAVITY + EDISPER`. Nothing here changes fixed charges, GAFF2
typing, molecular inputs, fitted parameters, hydration predictions, or the
OBC-II force-capable runtime.

## Current milestone and API

`implicit/torch_chagb.py` currently implements only the zero-salt CHA-GB/ALPB
**polar algebra downstream of NSR6 inverse Born radii**. It consumes four
explicit same-device float64 tensors: Cartesian positions [Å], fixed charges
[electron], `effective_cha_radii_angstrom`, and
`unshifted_inverse_born_per_angstrom` [Å⁻¹]. Its result is in kcal/mol and
retains tensor graphs for those inputs. The inverse Born input is **before**
the CHA size shift; a printed post-shift `rinv` is not a valid substitute.
The supplied radii are **after** the native `cha_rad` mapping and `+Rs`, not
the Bondi values in the original topology.

Actual native input processing also reduces the requested 1.4-Å GBNSR6
probe by `Rs=0.52 Å`, giving 0.88 Å within the CHA factor. Charges entering
the native energy equation are the supplied electron charges multiplied by
18.2223. These conventions are part of the target endpoint and not
tunable Torch options.

**No complete coordinate force exists in this milestone.** Supplying Born
radii from an external executable severs their geometry derivative; the
algebra's autograd output is then only a partial derivative. The module is
intentionally unregistered, has no `SolvationResult`, no `get_forces`, and
is not imported by `ImplicitSolvationCorrection`.

## Critical fidelity boundaries

1. CHA effective-charge sign changes and the electrostatic-size threshold at
   10 Å are hard branches. This implementation preserves them and rejects
   differentiation near those surfaces. It never replaces `sign` with `tanh`.
2. Native GBNSR6 constructs a discrete SES/grid and numerically integrates
   R6 radii. The requested `space=0.3 Å, arcres=0.2 Å` results in **effective**
   `arcres=0.15 Å`. A supplied Born tensor does not implement or validate this
   coordinate chain.
3. The selected PBSA `inp=2, use_sav=1` cavity uses a 0.5-Å integer voxel
   occupancy count, not a continuous sphere-union volume. The existing
   continuous sphere-union prototype is a distinct numerical model and
   cannot be silently substituted. A label-free live canary on pinned
   methyl hexanoate displaced atom-1 x by −0.005 Å and −0.001 Å from its
   reference position and obtained
   cavity values of 20.9437 and 20.9485 kcal/mol, respectively, while
   displacements −0.001/0/+0.001 Å shared the latter rounded value. This
   corroborates the occupancy step; it is not a derivative or experimental-
   accuracy result.
   PBSA dispersion likewise uses its own finite surface sampling and GAFF2
   Lennard-Jones mixing, not ACE.
4. The full native source trace and extracted-equation oracle are built only
   from the verified local AmberTools26 RC7 source in two clean scratch trees.
   The exact 64-file GBNSR6 source inventory is checked against the tracked
   SHA256 manifest, and all six included header files are pinned separately;
   extra files or symlinks are rejected before a forced source build. Object,
   binary, static-link input and resolved dynamic-library hashes are retained.
   The full trace reproduces the previously recorded reference EGB of
   −6.282267527560081 kcal/mol. The extracted equation differs from that
   full trace by only 4.44×10⁻¹⁵ kcal/mol; Torch differs from the nine
   frozen oracle cases by at most 2.85×10⁻¹⁴ kcal/mol in any polar energy
   component and 1.78×10⁻¹⁴ in recorded algebra intermediates. These are
   algebra results, not full-endpoint parity, derivative qualification, or
   experimental accuracy. The committed numeric fixture contains **no
   upstream source or binary**.
5. The exact selected cavity scalar has integer-occupancy steps. At those
   crossings a finite classical force does not exist. Torch automatic
   differentiation cannot turn an exactly matching discontinuous energy
   into a globally smooth force. Any smooth substitute would define a new
   endpoint and requires an explicit scientific decision and fresh accuracy
   validation, not a hidden implementation detail.

The relevant numerical definitions are source-pinned in the local verified
AT26 RC7 `egb.F90`, `gb_read.F90`, `pb_init.F90`, `sa_driver.F90`, and
`NSR6routines.F90`. The independent public AmberClassic snapshot is
[`0b35bfeb96026ffa4e5876391a0828f39b3cfc8d`](https://github.com/Amber-MD/AmberClassic/tree/0b35bfeb96026ffa4e5876391a0828f39b3cfc8d).
GBNSR6 carries GPL-2.0-or-later terms; the locally inspected PBSA tree has
an LGPL-3.0 notice. Formula-level independent implementation is not a
declaration that translating and redistributing either native code body
under MAPLE's existing license is automatically permitted. Licence review
is required before any such source translation is distributed.

## Open gates (not completed by the polar algebra)

| Gate | Required evidence | Status |
| --- | --- | --- |
| M1 polar algebra | Native full-run bridge, independently extracted equation oracle, fixed synthetic cases, exact units and branches, two clean-scratch replays | Implemented and independently approved; no force claim |
| M2 geometry | Same-source surface/grid owners, point positions, R6 Born radii, electrostatic-size derivative, E/F local convergence and branch control | Not implemented |
| M3 nonpolar | Matching PBSA SAV/cavity and sigma-dispersion **energy and complete derivative**, with actual GAFF2/topology parameters | Not implemented |
| M4 full force | Paired complete EGB/ECAVITY/EDISPER parity over diverse molecules and displacements; full E→F closure and an honest valid-domain statement | Not admitted |

The historic FreeSolv 526/116 MAE does not transfer merely because a tensor
routine has been written. Until M2–M4 close, no CHA OPT, TS, FREQ, or MD path
is opened. The previous scalar-only force rejection remains in force.
