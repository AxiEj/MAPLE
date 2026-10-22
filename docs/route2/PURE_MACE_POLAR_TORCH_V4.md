# Pure MACE-POLAR Torch v4: FP32 / lmax 5 / Lebedev 194

## Configuration

The experimental v4 profile changes the registered pure-Torch numerical
configuration to:

- MACE-POLAR model parameters and returned energy/source: `torch.float32`
- ddPCM: `torch.float32`, `lmax=5`, 194-point Lebedev grid, `eta=0.1`
- legacy SMD CDS and DAREAL: `torch.float32`
- ddPCM residual assessment: `100 * eps(float32)`
  (`1.1920928955078125e-5` in the verified runtime)

The upstream MACE implementation retains explicit float64 reductions in some
intermediate operations. Therefore, `FP32` describes the public model,
component inputs, parameters and outputs; it is not an all-operation-FP32
claim. MAPLE does not patch the installed MACE dependency.

V4 is an experimental execution profile. It has no scientific or release
admission. It does not inherit the v3 derivative or accuracy evidence.

## Molecular charge

V4 accepts explicit integer molecular charges, including cations and anions.
It checks the MACE source monopole sum against the requested charge rather
than requiring zero. Electron parity uses `sum(Z) - charge`; singlet-only,
element, mass, connectedness, non-periodic and numerical checks remain.
MOL2 charged atom types are allowed on this V4 path. Historical V3 and other
providers keep their existing neutral-only scope.

Charge and multiplicity are part of the ASE result-cache identity. A charge
change at fixed coordinates invalidates previously calculated results. This
execution change is not experimental ionic-solvation accuracy validation.

## Public input

For the pure Torch provider, omitting `profile` selects the device-appropriate
v4 profile. This default does not apply to any other implicit-solvent provider.

CPU default:

```text
#model=macepolm
#device=cpu
#solv(method=smd,provider=torch,implicit=water,experimental=true)
#sp(verbose=1)
```

For an archival input, select the same identity explicitly:

```text
#model=macepolm
#device=cpu
#solv(method=smd,provider=torch,profile=pure-macepolar-frozen-point-l1-ddpcm-smd-torch-cpu-v4,implicit=water,response=frozen,experimental=true)
#sp(verbose=1)
```

For CUDA, use `#device=cuda:0` and profile
`pure-macepolar-frozen-point-l1-ddpcm-smd-torch-cuda-v4`.

Direct calls to `build_smd_mace_polar_torch_pes()` also select the
device-appropriate v4 profile when `profile` is omitted. Passing an explicit
v3 profile preserves the historical FP64 / 15 / 1202 configuration.

See `examples/solvation/pure_torch/sp-cpu-default-fp32.inp` for the default and
`sp-cpu-v4-fp32.inp` for the explicit archival form. V4 retains the bounded SP, OPT and MW-FREQ task surface (no TS, IRC, MD or path
workflows) and now accepts all 179 named PySCF 2.13.1 SMD solvent parameterizations.
See `SMD_SOLVENT_CATALOG.md` for parser-safe names. This parameter coverage is
not a claim that every solute geometry passes numerical/topology checks.

## Validation

See `FP32_L5_N194_STATUS.md` for current real-entry replay, dtype-aware numerical
policies and known historical source-evidence mismatch. V4 evidence is limited
to engineering checks, not experimental accuracy or full-population admission.

## Topology grading

FP32 uses `ddx-v0.8.0-fp32-complementary-quintic-partition-v1`: stable paired
quintic values, structural single-sided exposure/normalization, nonblocking
warnings for proven smooth near-boundary cases, and hard rejection of genuine
multi-interior ambiguity. This is not a blanket guard disable. See
`TOPOLOGY_GRADING_STATUS.md` for exact-input before/after replay.
