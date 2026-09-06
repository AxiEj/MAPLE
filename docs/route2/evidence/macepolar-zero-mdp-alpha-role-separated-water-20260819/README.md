# MACE-POLAR-zero permanent + MDP-alpha ADT water canary

Date: 2026-08-19

This is a target-free real-checkpoint structural canary for the distinct research
profile `route2-research-macepolar-zero-point-mdp-alpha-polar-residual-role-separated-adt-ddx-operational-v3`.  It reads no experimental
solvation target and admits no E/F/H/V/M or Tier-V capability.

## What passed

- exact two-cold replay: `True`
- five-start residual and energy agreement: `True` / `True`
- JVP/VJP dot identity: `True`
- centred finite-field linearization: `True`
- positive supervised molecular-alpha response and local root certificate: `True` / `True`
- zero-field point permanent equals the exact response-chart source by construction (also covered by unit tests).

The selected water root has polarization energy
`-0.411303157400071 eV`, primal residual
`6.755e-11 eV`, and all five starts converge to the
same energy.

## What failed

The v1 target-free thresholds were intentionally reused rather than relaxed
after inspection.  The result is `fail`:

- rigid energy covariance at `lmax=8`: `1.237626e-05 eV` versus `1.0e-5 eV`;
- `lmax=8` versus `lmax=15`: energy `5.949639e-04 eV`, field relative `2.940215e-03`;
- laboratory-grid convergence is non-monotone, so the preregistered grid-ratio gate fails.

The target-free basis scan diagnoses a specific discretization issue: at
`lmax=12`, the energy error versus `lmax=15` falls to
`1.418267e-04 eV` and field relative error to
`8.199900e-04`.  This permits a separately
identified `lmax=12` diagnostic screen; it does **not** retroactively pass this
profile or prove any solvation-energy improvement.

`measurements-lmax12.json` freezes that separately executed diagnostic.  It
passes the root, replay, JVP/VJP, finite-field, passivity, local-root, source,
field, and `lmax=12` convergence gates.  It still fails the unchanged rigid
energy and laboratory-grid gates: the total rigid energy drift is
`1.2087694e-05 eV`, the polarization drift is `1.2084313e-05 eV`, and the
1202-point fixed-source field covariance error is `4.1326223e-05`.  These are
small chemical magnitudes but remain explicit finite laboratory-grid defects;
they are not silently relabelled as structural SO(3) covariance.

## Decision

Keep the POLAR-zero/MDP-alpha decomposition as an unadmitted mechanistic
candidate, increase its harmonic basis under a new content identity, and run a
small deterministic target-independent screen before paying for another 505
panel.  Do not alter the immutable v1 505 run, and do not tune against
experimental solvation targets.

Artifact SHA-256: `fa29d60fd3cd73ae9e85e297df37a35f8e0b22214a8f08dbf9529e94689f647e`
Embedded artifact SHA-256: `01b4955d8abbee78a0c047b02fba8e5553376c1955e125b71c22f861c4092cde`
Embedded measurement SHA-256: `c77b7e1424fb360bb1bb3024ffdd7a3fe4803ae917c9d775973cccbc5c5f2fc1`

Lmax-12 artifact SHA-256: `d6792f4e72abe61f95ef0c183edb340fcef91cea0809d47472e7cfe27c6d8703`
Lmax-12 embedded artifact SHA-256: `b1247a9174167874f490c23477ae8bf0352e2d1a83a2c2eb5c890a21788da998`
Lmax-12 embedded measurement SHA-256: `40a100e4aedf32d3a36d1ce05df91f4cedee58492a523f70a56590b150cf98ae`
