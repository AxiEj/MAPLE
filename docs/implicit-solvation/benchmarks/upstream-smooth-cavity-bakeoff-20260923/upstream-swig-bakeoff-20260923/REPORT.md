# Upstream smooth-cavity feasibility bake-off — 2026-09-23

## Identity and claim boundary

The released Torch v3 source tree was not edited. Git HEAD is
`6ce902c5aafb61299934f75ebe51563fc9bf67c8`; its 352-file source manifest
matches the frozen ten-molecule v3 panel exactly (SHA256
`527b7aebf3623c6cc0e7ada61fa1aca508d29b2c772c2dd908f3fdffadfa6d99`).
MACE-POLAR checkpoint SHA256 is
`fab8b8713c832f31a2a853aaa22fd638be8a369cbf5095e6b3e982a18d10e93a`.
ASE fixed neutral water geometries are the ten previously frozen species, not
experimental solution geometries or free energies.

The candidate electrostatics is **upstream PySCF 2.13.1 C-PCM** with SWIG or
ISWIG, SMD Coulomb radii, and the unchanged MACE point-multipole source. It is
not ddPCM, and no equation-change difference is called an implementation error.
The candidate CDS is **MOIST SvdW-DROP** per-sphere area with the original SMD
CDS radii and unchanged legacy SMD area tensions. It is not the original
DAREAL/CDS scalar and was not refitted. MOIST was built in an isolated `/tmp`
tree at `a0116e8ee369c546d48558e5e389e6f20bdff2e9`, not installed into
MAPLE. The tagged `v0.6.0-alpha.1` build did not resolve its `jonquil.wrap`
`recursion` ref, so no result is attributed to that tag.

These tests measure *model drift and numerical consistency*, **not physical
solvation-free-energy accuracy**, total-PES global C2, or an admitted analytic
Hessian. The nine comparable v3 records exclude HCN because its v3 analytic
branch is intentionally fail-closed; HCN stays in the candidate panel.

## Polar C-PCM results

Frozen protocol/results: `protocol.json` / `results.json`; both methods ran
for all ten species at 110, 194, and 302 Lebedev points per atom. Linear-solve
relative residuals were at most `1.16e-15`.

| Metric | SWIG | ISWIG |
|---|---:|---:|
| Largest absolute polar-energy change from v3 ddPCM, 302 grid, 9 comparable species | 0.00169085 eV (0.03899 kcal/mol), water | 0.00147186 eV (0.03394 kcal/mol), H2O2 |
| Largest 110-to-302 polar-energy shift (also changes upstream switching width) | 0.00284136 eV, NH3 | 0.00198885 eV, water |
| Largest absolute polar-energy rotation shift, 302 grid, 4 fixed rotations of water/CH4/HCN | 0.00014019 eV | 0.00028375 eV |

SWIG's existing analytic first-derivative adapter produced complete-polar
forces on water, methane, and HCN. Against v3 *total* forces at the 302 grid,
water differed by a maximum `0.00106794 eV/A` and methane by
`0.00015203 eV/A` (different electrostatic equations, not a parity failure).
HCN has no certified v3 force comparison. The preregistered SWIG-110 polar
force directional audit against centered differences of its own energy gave
maximum discrepancies `1.77e-6 eV/A` for water and `3.34e-6 eV/A` for HCN
across the two frozen steps. Finite differences are audits, not returned
production derivatives. ISWIG was energy-only here; no force/Hessian claim.

## MOIST SvdW-DROP area/CDS results

The first frozen MOIST wrapper passed transposed coordinates to the high-level
API. Its water shape was accidentally still `(3,3)`, so even apparent water
numbers were invalid. The full invalid attempt, including failures, is
preserved in `MOIST_V1_INVALID.md` and the v1 result. A separately frozen v2
wrapper passes `(natoms,3)` and reads back the native sphere centres before
using any area. Its protocol/results are in sibling
`../upstream-moist-area-v2-20260923/`.

All ten v2 species completed at 110, 302, and 590 points per atom using the
same SMD CDS radii and unmodified legacy tensions. At 302, the largest absolute
CDS energy change against DAREAL was `0.00167701 eV` (`0.03868 kcal/mol`,
methane); maximum atom-area change was `0.45561 A^2` (NH3). At 590, the
largest energy change *increased* to `0.00446253 eV` (`0.10292 kcal/mol`,
methane), so the 110/302/590 ladder is not monotone. Grid level also changes
parts of the DROP discretization, not merely a fixed-scalar quadrature order.

At fixed 302 grid, the four preregistered rigid rotations of all ten species
produced maximum CDS-energy change `0.00191030 eV` (`0.04406 kcal/mol`,
methane) and maximum per-atom area change `0.34018 A^2` (methane). This is a
measurable fixed-grid orientation dependence, not an exact rotationally
invariant CDS scalar. No acceptance threshold was preregistered for this
exploratory bake-off, and no favorable orientation was selected for a result.

An additional, explicitly exploratory two-species smoke check called MOIST's
native `moist_compute_cavity_gradient` / `moist_get_cavity_gradient` C API at
302 grid. Its per-sphere area gradient, contracted with one seeded displacement
direction, agreed with centered area finite differences at `1e-4 A` to maximum
`3.72e-6 A^2/A` (water) and `1.85e-6 A^2/A` (HCN). This confirms a usable
first-derivative route for those cases, **not** a second-derivative API or
whole-total-PES force/Hessian admission.

## Joint descriptive comparison (not an accuracy verdict)

Adding the already measured SWIG-302 polar change and MOIST-302 CDS change to
the nine comparable v3 total-solvation energies gives a maximum absolute shift
of `0.00154558 eV` (`0.03564 kcal/mol`, H2O2). This number can be small through
component cancellation and is smaller than some observed grid/rotation shifts;
it is **not** experimental accuracy and is not evidence of non-inferiority.

## Outcome and next gate

- No upstream drop-in passed the full requirement. PySCF supplies a credible
  polar E/F route; MOIST supplies atomwise area and a native first derivative,
  but the observed fixed-grid orientation dependence still needs explicit
  force/Hessian covariance limits and verification before total-PES admission.
- Neither this PySCF/MOIST combination nor either upstream Python API provides
  an already-integrated analytic Hessian for the exact MACE-POLAR + polar +
  SMD-CDS total scalar. Do not finite-difference the production Hessian.
- The five named experimental datasets are not available locally; no 3-fit/
  2-sealed-test protocol, source de-duplication, parameter fit, experimental
  accuracy verdict, or final Astra max approval was performed.
- Targeted existing tests: `30 passed in 2.49s`. Both private benchmark scripts
  pass Black (Python 3.11), Pyflakes, and bytecode compile. Tracked Git state
  remains clean.
