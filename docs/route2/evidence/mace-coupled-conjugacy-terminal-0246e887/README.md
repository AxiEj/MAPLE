# Original-source coupling-active-space terminal audit

Two clean processes executed the exact command shape

```bash
python tools/route2_release/run_mace_coupled_conjugacy_terminal.py \
  --device cuda --output <external-json>
```

at source commit `0246e887de7cb749aab03cb3d118cf7e5cb2ed00` and tree
`99cd34fcf5798e52efc083f3052b693b8da91346`. Both reproduce measurement
SHA-256
`d9de1ad9e3af971e96e5566bc0ece207fe3e4a671fd353d4b1d5e7df028fb7c0`.

## Runtime semantics

- checkpoint SHA-256:
  `fab8b8713c832f31a2a853aaa22fd638be8a369cbf5095e6b3e982a18d10e93a`;
- original source: four components, width `1.5 A`;
- native receiver: eight components, widths `1.5 A` and `3.0 A`;
- live projection SHA-256:
  `6b529e271d99c63e73dd4ea45c0ed192c2a44ee4e916241867b076a32875d649`;
- separated harmonic dimensions: source `12`, boundary `12`, native field
  `24` for water.

## Result

| state | fixed-charge response norm | coupled curl relative defect | direct residual, sign `+1` | direct residual, sign `-1` |
| --- | ---: | ---: | ---: | ---: |
| zero field | `7.38e-17` | `0.2081259719` | `0.6196857587` | `0.9047304684` |
| continuum-active nonzero field | `8.66e-17` | `0.2085410719` | `0.6195909807` | `0.9047211912` |

The fixed-total-charge tangent is closed to floating-point precision, so the
curl defect is not a leaking-charge artifact. The frozen tolerance is `1e-9`;
the roughly `0.208` defect is a terminal counterexample to integrability of the
original source response on this actual continuum-active space. This closes
the last quotient-space loophole for the unchanged source/checkpoint Tier-V
route at the tested profile identity.

The direct energy/source residuals are retained but not used for the terminal
decision because the consumed intrinsic-energy hook has not been verified as
the complete external enthalpy with a frozen sign. The operational response
route remains open. Every `E/F/H/V/M` capability remains false.

## Files

- [`run1.json`](run1.json), SHA-256
  `9bb7acdb0b8f53026796636a28c4d758cb148b80d0605a6ecc6944c95c94a898`;
- [`run2.json`](run2.json), SHA-256
  `ea2e1c1da7e3cb0cb4cd69c950cfaf018fa0e8ea5dca5577c7a5e1f744e03ffc`.
