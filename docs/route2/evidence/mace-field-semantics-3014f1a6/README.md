# MACE-POLAR field-energy semantics terminal audit

Two clean CUDA processes executed

```bash
python tools/route2_release/run_mace_field_semantics.py \
  --device cuda \
  --checkpoint ~/.cache/mace/MACEPOLAR1Mmodel \
  --output <external-json>
```

at source commit `3014f1a63cc966ac8186c1af78cde327b7519456` and tree
`50e8f3a638fcfad40822852855a817288868bef7`. Both reproduce measurement
SHA-256
`e77ab89951e8eefec3480571fcc8df03026090f06aa15b817c464d033f230439`
and field-semantics manifest SHA-256
`3b425ff8ae41d0817a1b1f06e275b3f09478c4a27793f7d53a649cee24f446c8`.

## Result

The official uniform-field branch and MAPLE's direct native-eight-channel
injection produce bitwise-identical density/source coefficients and dipole for
the audited water state. They do **not** expose the same energy or force:

| quantity | value |
| --- | ---: |
| source maximum absolute difference | `0.0` |
| dipole maximum absolute difference | `0.0 e A` |
| upstream minus native raw energy | `3.8534993350367586e-3 eV` |
| explicit upstream field work | `3.853499334970462e-3 eV` |
| work-identity absolute error | `6.629636076227463e-14 eV` |
| selected explicit-work sign | `+1` |
| force maximum absolute difference | `3.5332428619208045e-3 eV/A` |
| directional-derivative difference | `4.9284166038887633e-1 e A` |

Thus the native injection consumed by separated Route 2 omits the explicit
uniform-field work term present in the upstream branch. It is correctly named
`E_conditioned_raw`, not complete external enthalpy.

The native raw graph itself is internally differentiable:

- zero-field energy and force match vacuum exactly;
- reverse AD versus forward JVP differs by
  `1.5265566588595902e-16 eV`;
- the best central finite-difference error is
  `2.62157989761036e-10 eV`;
- the radial charging identity error is
  `9.840447955187376e-14 eV`.

These facts do not rescue original-source Tier V: the separately retained
coupling-active curl counterexample remains terminal. They also do not decide
whether the original four-channel source is a quantitative PCM source or
whether `Phi0` is physically accurate. A matched QM boundary-RHS/surface-MEP
and electrostatic-component reference is still required.

Every `E/F/H/V/M` capability remains false. `Phi1Delta` remains an unadmitted
diagnostic ledger and cannot be described as a complete external-enthalpy
ledger for this native-injection branch.

## Files

- [`run1.json`](run1.json), SHA-256
  `99427334621fddf6384a86afacb4793684239ab32eb667292ad81232d9be61fc`;
- [`run2.json`](run2.json), SHA-256
  `e6f2db148012650703c0d299a64fe182ed3c9b50efbbcb578b2533201318cdbf`.
