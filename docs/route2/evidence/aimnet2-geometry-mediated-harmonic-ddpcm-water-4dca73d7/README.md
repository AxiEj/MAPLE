# AIMNet2 geometry-mediated harmonic-ddPCM water canary

This directory retains two independent clean-tree executions of commit
`4dca73d795f92cab3ee9d07b7c31e4b247d4e758` with the unchanged SHA256-bound
AIMNet2 wB97M-D3 checkpoint and the source-bound reconstructed Python float64
runtime.  The electronic source is evaluated once per geometry and is exactly
field independent; there is no AIMNet2--continuum electronic SCF iteration.

Both processes reproduce scientific measurement SHA256
`7ec7732e79aacfd1d9502d7c19de8fd49b0b1975b764ef1cf59a413eaaf5cbbf`.
Runtime duration, timestamp, command path, and output path are intentionally
outside that measurement hash.

## Frozen physical and numerical identity

- solvent dielectric: water, `epsilon=78.355`;
- source: AIMNet2 NQE point monopoles embedded as `[q,0,0,0]`;
- continuum: smooth weighted harmonic finite-dielectric ddPCM, not a uniform
  COSMO scaling of the conductor energy;
- cavity radii: the existing water SMD Coulomb-radius profile;
- harmonic configuration: `surface_lmax=1`, `exposure_lmax=2`, 32-point
  exposure and Green radial quadratures, transition width `0.18 A^2`;
- scalar: `E_AIMNet2(R) - 1/2 b(R)^T x(R)`, including the complete AIMNet2
  charge-coordinate chain rule and the transpose/KKT energy cotangent;
- omitted: SMD-CDS/nonpolar and standard-state terms.

## Result

- deterministic same-process center replay: exact;
- two clean processes: identical scientific measurement SHA256;
- continuum energy: `-0.49914801549225324 eV`;
- adaptive directional analytic-vs-numerical gradient error:
  `1.8257224045914455e-6 eV/A` with seven scalar evaluations;
- terminal frozen-step directional error at `2.5e-4 A`:
  `1.7625409445276574e-6 eV/A`;
- terminal complete Cartesian maximum/RMS errors:
  `2.826476747208595e-5` / `1.5826934760464932e-5 eV/A`;
- the complete Cartesian refinement order is `1.9945636040785466`;
- three rigid rotations have zero recorded energy drift and maximum relative
  force-covariance error `5.5951040041648464e-11`;
- base net-force and torque norms are `4.658887789198649e-11 eV/A` and
  `3.0449087707327465e-11 eV`;
- the largest primal/adjoint relative residual is
  `1.0735508334481661e-15`, the largest condition number is
  `109.61213248462516`, and the half-coupling identity error is
  `4.440892098500626e-16 eV`;
- model-neighbor, point/source-shell, sphere-tangency, reciprocity,
  charge-tangent, gauge, stationarity, directional, Cartesian, and rotation
  gates all pass locally.

This is the first retained real-checkpoint positive force diagnostic for the
finite-dielectric harmonic ddPCM implementation.  It is still one water
geometry under a parameterized diagnostic identity.  It does not establish a
solvent-bound admission profile, a broad smooth PES, CDS/nonpolar forces,
chemical accuracy, Hessian/FREQ/TS/IRC, NVE/MD, or any public E/F/H/V/M
capability.

The finite-dielectric equation follows the ddPCM formulation documented by
[ddX](https://ddsolvation.github.io/ddX/md_docs_theory.html), the systematic
ddPCM discretization of
[Stamm et al.](https://doi.org/10.1063/1.4940136), and the analytic-force
derivation of [Gatto et al.](https://doi.org/10.1063/1.5008329).  AIMNet2 model
scope and NQE charge semantics follow the
[AIMNet2 paper](https://doi.org/10.1039/D4SC08572H).
