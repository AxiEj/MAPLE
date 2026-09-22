# Pure MACE-POLAR operational Route 2 trial

Date: 2026-08-15

Base commit: `80a94e6257718a9cc98ab98cbc1a4c3ffd7f0f46`

Status: research candidate; `E/F/H/V/M = false`

## Decision frozen by this trial

The scientific target is a modular operational composite potential built from
pure MACE-POLAR plus continuum, nonpolar, and standard-state modules. It is not
a common or unified variational MACE--PCM functional.

The total-solvation-free-energy admission threshold is frozen at
`MAE <= 1.5 kcal/mol`. A release panel must contain at least 150 independent
blind neutral-water molecules, exactly cover its content-addressed case list,
use only real backends, and also satisfy a one-sided 95% molecule-bootstrap
MAE upper bound of at most `1.5 kcal/mol`. Electrostatic-only values cannot be
scored as experimental total solvation free energies.

Energy admission does not admit force. Force, OPT, FREQ/TS, and MD remain
independent fail-closed steps.

## Current pure-checkpoint evidence

The existing original four-channel MACE-POLAR source failed the frozen matched
QM/PCMSolver four-case pilot:

| Candidate | MAE (kcal/mol) | Maximum error (kcal/mol) | Result |
|---|---:|---:|---|
| Original source4, sigma 1.5 A | 7.461 | 11.828 | fail |
| Fixed radial repair, sigma 0.75 A | 3.104 | 4.888 | fail |

These four cases are negative pilot evidence, not an admission-sized panel.
They are already sufficient to prohibit enabling the current profiles, but
they are not sufficient to certify a replacement.

## Applied-potential completion candidate

The official PolarMACE uniform-field forward adds explicit applied work to the
field-conditioned energy. MAPLE's native arbitrary-field feature injection
previously exposed only the raw field-conditioned scalar. This trial adds a
separate, immutable completion:

\[
E_{\mathrm{complete}}(R,u)
= E_{\mathrm{raw}}(R,u)
+ (S c(R,u))^T Q u,
\]

where `c` is the original four-channel MACE-POLAR density response, `S` is the
frozen source4-to-native-source8 embedding, `u` is the native eight-channel
receiver field, and `Q` is the registered energy pairing. The positive work
sign matches the official uniform-field implementation.

The implementation differentiates this exact scalar:

\[
\frac{\partial E_{\mathrm{complete}}}{\partial u}
= \frac{\partial E_{\mathrm{raw}}}{\partial u}
+ Q^T S c
+ J_u(c)^T S^T Q u,
\]

\[
\left.\frac{\partial E_{\mathrm{complete}}}{\partial R}\right|_u
= \left.\frac{\partial E_{\mathrm{raw}}}{\partial R}\right|_u
+ J_R(c)^T S^T Q u.
\]

This is an operational MLIP-side scalar candidate. It does not select a final
PCM ledger, prove passivity or reciprocity, or justify an envelope-theorem
force. If it is used at a nonstationary continuum fixed point, the outer total
force must still include the full fixed-point implicit derivative.

## Modular boundary

The completion consumes a provider-neutral response protocol:

- source and receiver spaces, units, gauge, and content-addressed identity;
- source evaluation;
- source field VJP and coordinate VJP;
- raw energy, raw field gradient, and raw fixed-field coordinate gradient.

The source/receiver work bridge is a separate immutable object. A future MLIP
can replace MACE-POLAR without modifying continuum, root, adjoint, or release
gate code.

## Verification in this branch

Dependency-light tests cover:

- the exact source4/field8 work pairing;
- all-coordinate central finite differences of the completed field gradient;
- all-coordinate central finite differences of the fixed-field geometry
  gradient;
- immutable provenance and fail-closed capability metadata;
- the 150-case, real-backend, total-free-energy-only 1.5 kcal/mol blind gate.

These are algebra and contract tests. They are not chemical-accuracy evidence.

One real official MACE-POLAR-1-M CPU canary was also executed with checkpoint
SHA256 `fab8b8713c832f31a2a853aaa22fd638be8a369cbf5095e6b3e982a18d10e93a`:

- explicit applied work: `0.003852832572334272 eV`;
- completed uniform-field energy versus upstream: `0.0 eV` absolute error;
- completed fixed-field coordinate directional derivative versus central FD:
  `4.60e-6 eV/A` absolute error;
- completed physical uniform-field directional force versus upstream:
  `8.11e-6 eV/A` absolute error.

This canary verifies the work sign, scalar identity, and one directional force
path. It is one water geometry and therefore cannot admit E or F.

## Next kill tests

1. Replay the official checkpoint's uniform-field energy and forces against the
   upstream branch, including the explicit work term.
2. Freeze candidate identities before seeing matched PCM errors. Compare the
   original density source, raw intrinsic-energy-gradient source, and the
   applied-work completion without tuning to the blind panel.
3. Reuse a maintained continuum implementation (ddPCM/ddCOSMO from ddX) with an
   exact Gaussian q+dipole MEP/VJP adapter. Add a separately versioned CDS and
   standard-state module before comparing to experimental total free energy.
4. Run the >=150-molecule blind energy gate first. If its MAE or one-sided 95%
   upper bound exceeds 1.5 kcal/mol, stop that profile before spending cloud
   compute on force, Hessian, or MD admission.
5. Only after energy passes, run fully reconverged finite-difference force,
   rigid covariance, closed-loop work, OPT, FREQ/TS, and NVE MD gates in that
   order.

## Primary sources

- [MACE-POLAR-1 paper](https://arxiv.org/html/2602.19411v1)
- [Official PolarMACE documentation](https://mace-docs.readthedocs.io/en/latest/guide/polar_mace.html)
- [Official v0.3.16 PolarMACE forward implementation](https://github.com/ACEsuit/mace/blob/4d2da09413ac1407f37cdbb6b81fa28e4c15655e/mace/modules/extensions.py)
- [ddX reference implementation](https://github.com/ddsolvation/ddX)
- [FreeSolv reference data](https://github.com/MobleyLab/FreeSolv)
- [SMD model paper](https://pubs.acs.org/doi/10.1021/jp810292n)
