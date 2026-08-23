# Smooth-partition harmonic ddPCM candidate

## Purpose

This candidate replaces uniform-COSMO dielectric scaling by the published
finite-dielectric ddPCM two-stage equation while preserving the fixed,
SO(3)-covariant coefficient topology of the harmonic ddCOSMO candidate.  It is
the finite-dielectric continuum intended for the Route 2 mainline; it does not
use a laboratory-fixed surface grid or an active tessera set.

For the localized solute potential

\[
F=-Bc,
\]

the implementation solves

\[
A_\epsilon G=A_\infty F,
\qquad
LX=G,
\]

and evaluates the on-shell scalar

\[
G_{\rm ddPCM}(R,c)=\frac12 c^T C X.
\]

The finite-dielectric block uses

\[
A_\epsilon=
2\pi\frac{\epsilon+1}{\epsilon-1}I-D,
\qquad
A_\infty=2\pi I-D,
\]

with the signs and normal convention frozen against ddX commit
`4d79e3d9caeae5e602683572a71cb550414f9b09`.

## Smooth double-layer assembly

The ddPCM double layer follows the ddX centered characteristic convention:

- distinct-sphere blocks use the exterior multipole continuation throughout
  the centered switching layer;
- the centered pair characteristic is applied before coefficient projection,
  so a singular bare exterior continuation is never formed on buried target
  points;
- remaining pair factors are combined in harmonic coefficient algebra;
- the self principal-value block is multiplied by the full centered exposure;
- every atom retains the same \((L+1)^2\)-dimensional coefficient block.

Pair directions enter only through complete harmonic representations and
invariant one-dimensional quadrature.  Consequently changing `surface_lmax`
changes continuum accuracy, but does not introduce a preferred laboratory
orientation.

## Scalar derivatives

The dielectric and Schwarz operators are generally nonsymmetric.  The code
therefore does not invent a symmetric quadratic in either intermediate state.
Torch differentiates the physical on-shell scalar through both linear solves,
which is equivalent to the composite primal-adjoint Lagrangian.  Source
gradients, coordinate gradients, mixed derivatives, and coordinate HVP actions
come from that same scalar graph.

Current structural evidence includes:

- exact finite-dielectric single-sphere monopole and dipole limits;
- systematic convergence to real pyddx ddPCM for water, reaching
  `1.211143e-5 eV` at `surface_lmax=5`;
- finite, quadratically convergent derivatives at internal and external sphere
  tangencies for the smooth profile;
- energy and force SO(3) covariance at floating-point error;
- translation covariance, source JVP/VJP duality, coordinate finite
  differences, and coordinate-HVP bilinear symmetry.

## Frozen MNSol-10 accuracy result

The preregistered ten-record, ten-solvent profile uses:

- official unmodified MACE-POLAR-1-M zero-field point-\(l\le1\) source;
- PySCF 2.13.1 SMD solvent-dependent Coulomb radii;
- `surface_lmax=5`, `partition_lmax=10`;
- transition width `0.18 A^2`;
- partition/source/double-layer radial orders `96/128/128`;
- unchanged PySCF SMD-CDS;
- no fit, calibration, panel-dependent parameter change, or profile mixing.

Result:

| metric | value (kcal/mol) |
|---|---:|
| MAE | **0.8834415187** |
| RMSE | 1.0050963427 |
| maximum absolute error | 1.6482383685 |
| target | MAE <= 1.5 |

The profile therefore passes the frozen mainline MAE target.  This is evidence
for a **frozen zero-field source + harmonic finite-dielectric ddPCM + SMD-CDS**
profile only.  It does not yet admit mutual MACE-POLAR self-consistency,
MACE-MDP+MACE-POLAR, public E/F/H/V/M, OPT, FREQ, TS/IRC, or MD.

## Evidence

- preregistration:
  `docs/implicit-solvation/benchmarks/route2-mnsol10-harmonic-ddpcm-preregistration-v1.json`
- public aggregate:
  `docs/route2/evidence/mace-polar-point-l1-harmonic-ddpcm-mnsol10-accuracy-v1.json`
- private row-level replay:
  `.omx/route2/mnsol10-point-l1-harmonic-ddpcm-private-v1.json`
- implementation:
  `maple/solvation/continuum/harmonic_ddpcm_primitives.py` and
  `maple/solvation/continuum/harmonic_ddpcm_functional.py`
- tests:
  `tests/route2_vnext/test_harmonic_ddpcm_functional.py` and
  `tests/route2_vnext/test_harmonic_ddpcm_accuracy_artifact.py`
