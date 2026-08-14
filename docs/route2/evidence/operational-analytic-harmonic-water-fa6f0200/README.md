# Analytic original-source operational harmonic water canary

This bundle retains two clean-process executions of the separately registered,
disabled operational scalar that couples the analytic isotropic
Gaussian-multipole MACE-POLAR evaluator to the original four-channel density
response and the smooth weighted harmonic-Galerkin continuum scalar.

## Bound identity

- execution commit: `fa6f02001b17356734bb801597a31cfe22027fae`
- official checkpoint SHA-256:
  `fab8b8713c832f31a2a853aaa22fd638be8a369cbf5095e6b3e982a18d10e93a`
- model evaluator:
  `graph-longrange-analytic-gaussian-multipole-realspace-v1`
- scalar:
  `route2-operational-macepolar-analytic-gaussian-multipole-smoothharmonicgalerkin-cpcm-v1`
- source identity:
  `original-four-channel-density-head-embedded-first-radial-block`
- measurement SHA-256 (both runs):
  `586024051151be3e3c63171e73bab34b6ce43459d4232c8070daadcd204b4547`
- primary JSON SHA-256:
  `15e1fa2b911e0c89130fd0d6df2ee2a807c579b09c4e6796d83bb0b3995ec19a`
- replay JSON SHA-256:
  `b0f1989e15e20bc2f1dbe51724580b3bf7327cdad5de6e674afa93b232325256`

The complete files differ only in execution metadata and wall-clock timings.
Their protocol, geometry, identities, roots, scalar ledger, force checks,
rotation measurements, decisions, and `measurement_sha256` are identical.

## Executed command

```bash
python tools/route2_release/run_operational_analytic_harmonic_water_canary.py \
  --checkpoint /home/axie/.cache/mace/MACEPOLAR1Mmodel \
  --device cuda \
  --output /tmp/route2-operational-analytic-harmonic-fa6f0200-run1.json
```

The replay used the same command with `run2.json` as the output.

## Narrow result

- center cold root: 31 Anderson iterations, actual unmixed reduced residual
  `1.7379968031740547e-10`;
- warm replay: zero iterations, zero source difference, and zero scalar-energy
  difference;
- vacuum scalar: `-2079.8637965936628 eV`;
- continuum scalar: `-0.017377834628868098 eV`;
- operational total scalar: `-2079.8811744282916 eV`;
- the identity
  `G_harm = 0.5 * <c,u>_Q` closed with exactly `0.0 eV` recorded error;
- the absent `sigma=3.0` source block remained exactly zero;
- net-force norm: `2.7755575615628914e-17 eV/Angstrom`;
- torque norm: `1.6752969150823915e-10 eV`.

All three re-solved operational implicit-adjoint central differences passed:

| step (Angstrom) | absolute error (eV/Angstrom) | relative error |
| ---: | ---: | ---: |
| `5e-4` | `3.5913535214593395e-6` | `3.5047466732816314e-5` |
| `2e-4` | `5.477295019762352e-7` | `5.345366748683474e-6` |
| `1e-4` | `8.616094082647408e-8` | `8.408601345778749e-7` |

For one deterministic proper rotation:

- harmonic-continuum energy error: exactly `0.0 eV`;
- harmonic-continuum coordinate-gradient maximum error:
  `1.8214596497756474e-17 eV/Angstrom`;
- complete operational-scalar energy error:
  `2.7694113668985665e-9 eV`;
- source relative covariance error: `8.025998472808204e-9`;
- field relative covariance error: `2.3417317587664313e-9`;
- force relative covariance error: `7.857995561759406e-8`;
- force maximum absolute covariance error:
  `2.2821200651446105e-8 eV/Angstrom`.

The continuum result is a positive structural/numerical canary for the new
coefficient-space construction. It does not repair or admit the legacy
laboratory-fixed Lebedev/SWiG discretization; that remains a distinct negative
route.

## Claim boundary

This is one water geometry, one deterministic coordinate direction, and one
rotation. It is not a solvation-accuracy result and does not establish
conformer or chemistry-domain coverage, global root uniqueness, closed-loop
work, Hessian/FREQ/MD readiness, sharp-union equivalence, nonpolar/CDS closure,
or a public solution-phase PES. The operational scalar deliberately excludes
the checkpoint field-conditioned energy difference and is not Tier V.

All public capability tiers remain closed:

```text
E = false
F = false
H = false
V = false
M = false
```
