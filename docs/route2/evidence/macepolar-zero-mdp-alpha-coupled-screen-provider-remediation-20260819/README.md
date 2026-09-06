# Coupled-screen provider remediation

Date: 2026-08-19

The first invocation of the prospectively frozen twelve-case v3 coupled screen
failed before evaluator construction or any coupled-state evaluation.
`record-002-v1.json` records:

```text
failure_type:    ValueError
failure_message: Atomic-number identity changed.
```

The screen runner had reproduced the frozen array-identity header but omitted
the source contract's single `NUL` separator between that header and the array
bytes.  Consequently, the runner rejected the correct SPICE geometry after the
frozen backbones were loaded but before constructing the geometry-bound ADT/ddX
evaluator.  No coupled root, energy, or target was evaluated.

The only remediation is to make the runner's array SHA-256 byte-for-byte equal
to `run_mdp_mbis_pcm_source_gate._array_sha256` and add a regression against
that authoritative helper.  The twelve selected geometries, v3 profile,
checkpoints, `lmax=12`, 1202-point rule, cavity, solver, gates, and claim
boundary are unchanged.  A separately hashed v2 preregistration must disclose
and supersede the unusable v1 source binding before the scientific screen is
rerun.

Record file SHA-256:
`9aae1f40bba243c2440ebb6ef350ded1a53edd746c336e630cfc6ecbaaec467e`

Embedded record SHA-256:
`eaa83b2cfdb1fff38bef0ff199a3310b7d02df95b701147efdfa72780e946310`
