# Private CHA Torch FP32 backend

This local experimental backend evaluates prepared-input CHA-GB/ALPB polar
energy and PBSA cavity/dispersion contributions. It supports CPU and CUDA with
one shared Torch FP32 numerical chain. It does not provide forces, optimization,
frequencies, or molecular dynamics.

## MAPLE integration

```text
InputReader: shared MAPLE device resolution
    → SetCalculator / CalcABC
    → ImplicitSolvationCorrection
    → PrivateTorchChaGB: prepared-input and source identity
    → shared Torch stages: setup, SES, boundary, R6, polar, cavity, dispersion
    → named components and execution provenance
```

Device selection follows MAPLE's existing rules:

| Input | Execution |
| --- | --- |
| No `#device` | CPU |
| `#device=cpu` | CPU |
| `#device=cuda:0` | CUDA device 0 |
| GPU aliases or explicit `auto` | MAPLE's shared resolver determines the initial device |

The private provider accepts resolved CPU or CUDA devices. Explicit unavailable
CUDA, unsupported devices, and execution failures raise an error. They do not
trigger a device, provider, or precision retry. Unrelated MAPLE device defaults
are unchanged.

## Prepared single-point input

The current profile is bounded to neutral, closed-shell, nonperiodic water
single points with zero salt and fixed charges. A prepared topology and its
matching original MOL2 file are required; arbitrary MOL2 input is not an
automatic parameterization interface.

```text
#model=aimnet2025(hessian=numerical,d4=false,model_path=<existing-model-file>)
#sp
#charge(source=prepared,mode=fixed,geometry=keep)
#solv(implicit=water,method=gb,model=chagb,nonpolar=cavity-dispersion,dtype=float32,prepared_topology=<topology.json>,prepared_topology_sha256=<file-sha256>,experimental=true)
0 1
MOL2 <matching-original.mol2>
```

This example uses CPU because it omits `#device`. Add `#device=cuda:0` to run
the model and solvent correction on CUDA. Frozen source charges are loaded from
the validated prepared topology; no charge generation or renormalization occurs.
The runtime does not call AmberTools. Preparation provenance remains explicit.

## Boundaries and maintenance

- `torch_runtime.py` owns device validation, workspace/chunk policy, metadata
  transfer accounting, and device-specific telemetry. It contains no fitted
  physical parameters.
- `prepared.py` validates and hashes host input, then binds one selected-device
  FP32 view. Scientific inputs are not changed by device selection.
- The scientific stages share formulas, ordering, and parameters across CPU and
  CUDA. There is no second legacy CPU implementation or fallback chain.
- CPU reports host-local metadata access and no CUDA peak-memory measurement.
  CUDA reports actual indexed device, FP32, workspace, and allocator peaks.
- SP output distinguishes gas energy, named solvent components, their combined
  value, and the energy-only capability. The combined value is not a complete
  thermochemical Gibbs free energy.

The source-derived engine remains an ignored, separately installed private
package with its GPL/LGPL notices and lineage map. It is not bundled into the
public MAPLE wheel. This integration does not grant public redistribution
approval. Historical implementation comparisons are not new experimental
accuracy qualification.

See [local validation and handoff](PRIVATE_CHA_TORCH_FP32_HANDOFF.md) for the
installation boundary, evidence requirements, and maintenance procedure.
