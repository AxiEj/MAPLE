# AIMNet2-2025 with independent OBC-II

## Scope and identity

This is a separate backend, not a renamed legacy `aimnet2` TorchScript model.
`aimnet2025`, `aimnet2-2025` and `aimnet2_2025` select the same official
B97-3c AIMNet2-2025 member 0. Legacy `aimnet2` and `aimnet2nse` keep their
previous identities. Other ensemble members, NSE, Pd and RXN are not inherited.

The exact member-0 checkpoint SHA256 is
`043ed5418a104e31f79462f8e5ebeca64a2d24422174f5d29f894d32271981b5`.
The backend checks that digest before loading an explicit file or official
cache entry. The MAPLE adapter is cache/explicit-file only and never downloads weights.
An offline missing file fails; another checkpoint is never selected.

The original potential includes learned energy/atomic shifts, the embedded
short-range Coulomb subtraction, full nonperiodic simple Coulomb, and native
D3BJ. No term is omitted and no external D4 is added.

## Installation and input

The optional 2025 runtime needs Python >=3.11. Core MAPLE retains Python >=3.10.
The initial exact-cell runtime is intentionally pinned: AIMNet 0.2.0,
Torch 2.12.0+cu130, nvalchemi-toolkit-ops 0.3.1 and Warp 1.13.0. Install the
matching official Torch build before the optional MAPLE extra:

```bash
python -m pip install 'torch==2.12.0+cu130' --index-url https://download.pytorch.org/whl/cu130
python -m pip install '.[implicit-gb,aimnet2025]'
# Prefetch the exact official member once, or supply model_path explicitly:
python -c "from aimnet.calculators import AIMNet2Calculator; AIMNet2Calculator('aimnet2_2025', device='cpu')"
```

The pin describes the tested runtime, not universal support for other Torch,
CUDA, platform or source builds. The CPU cell uses this same installed runtime;
a CPU-only Torch wheel is not silently treated as the same qualified build.
Critical upstream source and D3-table hashes are checked, not just version labels.

```text
#model=aimnet2025(hessian=analytic,d4=false)
#freq(method=mw,ilowfreq=2)
#device=gpu0
#charge(source=maple,method=am1bcc)
#solv(implicit=water,method=gb,model=obc2,dtype=float32,nonpolar=ace,experimental=true)

0 1
XYZ molecule.xyz
```

Use `#device=cpu` for the separate CPU cell. An explicit checkpoint can be
selected with `model_path=/absolute/path/aimnet2_2025_b973c_d3_0.pt` in the model
options. Fixed charged MOL2 input with `#charge(source=mol2)` remains supported.
AmberTools is required only when generating charges/topology; it does not replace
the gas-model runtime. XYZ preparation preserves original calculation coordinates.

Analytic examples require the **current sealed exact-cell receipt**. A missing,
stale or mismatched receipt fails with `ANALYTIC_CELL_UNQUALIFIED`; the syntax
alone does not constitute analytical qualification. Candidate testing is not
installed-console admission, and no public bypass exists.

Admission status is determined by the packaged receipt and its pointer, not by
this document or the presence of an adapter. Any source change requires a fresh
qualification; syntax and unit tests alone are not installed analytical admission.

## One scalar, native precision

Every E/F/H/direct-HVP request uses the same official differentiable energy
interface. A requires-grad coordinate leaf selects the official Torch D3 energy
path even for energy-only requests. This avoids using the fast first-derivative
D3 kernel for E/F and a different path only when curvature is requested.
No official dense Hessian is computed merely to get a differentiable scalar.

The neural network and coordinates retain native float32. Atomic shifts and
energy accumulation retain the upstream float64 representation. The gas AD
class is therefore **N32**, not D64; a float64 result container does not change
that fact. OBC-II also computes in float32 on CPU and CUDA. The shared MAPLE result
containers promote completed derivatives before composition; that promotion does
not make either derivative kernel D64. A conflicting OBC dtype is rejected for
this backend. Legacy ANI/OBC cells keep their separately identified FP64 profile. Direct HVP does not construct H.

The joint FP32 engineering profile is an explicitly user-authorized revision
after failures of the earlier N32 thresholds, not a preregistered success. The full force-FD ladder is 0.002/0.001/0.0005 A; both finest steps
and their pairwise stability must pass. The revised FD limits are max 0.002 Ha/A^2, RMS 0.0001 for full H and
RMS 0.0003 for directional HVP, with an added
0.5% relative L2 bound and a 0.002 Ha/A^2 symmetric-eigenvalue bound.
Source, graph, exact composition, raw symmetry, translation and direct-HVP
consistency gates remain strict. Original OBC-only and ANI D64 gates remain unchanged. Strict D64-reference feasibility failures remain recorded as
failures; they are not relabeled as successful native qualification.

## Source identity and runtime cost

The adapter binds the complete upstream source inventory and hashes at load
time. Every runtime boundary still checks file identities, directory entries,
module origins and configuration. Unchanged origins are validated once and
then checked by metadata and path-ancestry identities; no repeated recursive
filesystem traversal or path resolution is needed. Directory-entry changes
trigger a complete inventory check before refreshing only the directory cache.
This also detects additions within one filesystem timestamp tick. Ordinary
bytecode-cache churn is allowed; new helpers, symlinks and module replacement
are rejected. There is no polling interval or switch that disables checking.

Full neighbor-graph evidence and the original tensor/configuration checks remain
in place. Performance improvements do not change formulas, precision, native
D3 execution, or qualification tolerances.

## Limits

- Independent Torch OBC-II only for this analytic combination; no inheritance
  of legacy OpenMM analytic admission.
- Water, zero salt, mbondi2, ACE/NONE, fixed external charges and frozen topology.
- Default implicit domain remains one connected, nonperiodic, neutral closed-shell
  molecule with explicit hydrogens and an explicit `0 1` declaration. The wider
  upstream element/charge domain does not expand the solvent qualification.
- Analytic H/HVP is refused within the unchanged 0.003 A guard around radial,
  embedded-SR and D3 switching/cutoff boundaries. Runtime input coordinates are
  never nudged or transformed to evade that guard.
- No float64/import-order conversion, half precision, autocast, TF32,
  `torch.compile`, PBC, alternative Coulomb method, or external D4 admission.
- FREQ is mass-weighted; experimental TS is P-RFO or Dimer. A short execution
  replay does not establish a converged chemical transition state or reaction
  connectivity. NVE/NVT availability does not establish long-time MD stability.
- Numerical modes remain explicit alternatives with output warnings, never
  fallbacks for rejected analytic requests.
- Formula/derivative/workflow engineering checks do not certify experimental
  solvation accuracy or general reaction/barrier accuracy.

## Primary model references

- [Official AIMNet2-2025 model guide](https://isayevlab.github.io/aimnetcentral/models/aimnet2_2025/)
- [Executing official v0.2.0 source](https://github.com/isayevlab/aimnetcentral/tree/b1b0fd8c8b8d8acb8811eb646f9ba7b15d72da55)
- [Official model card](https://huggingface.co/isayevlab/aimnet2-2025)

## Relation to MAPLE's earlier AIMNet family

| MAPLE name | Identity | Difference relevant to this integration |
| --- | --- | --- |
| `aimnet2` | Existing local TorchScript; 36 shared tensors match official wB97M-D3 member 0 exactly | Preserved unchanged; not silently renamed to 2025 |
| `aimnet2nse` | Separate spin-aware family | Its legacy local checkpoint was absent in this comparison; no new execution or solvent admission claimed |
| `aimnet2025` | Official B97-3c 2025 member 0, new format/native runtime | Dedicated complete-potential AD boundary and separate joint-FP32 qualification |

These are not interchangeable reference levels. The upstream families share the
same 14-element list; 2025 is not an elemental-coverage expansion. Tensor lineage
is not a full cross-serialization parity claim. NSE's spin support is not
inherited by this neutral closed-shell solvent cell.

A six-molecule paired CPU diagnostic on the same frozen base geometries measured
old-vs-2025 gas-force RMS differences of 0.1692 (water), 0.1068 (methanol),
0.0613 (ethane), 0.0304 (benzene), 0.1379 (methyl-hexanoate) and 0.5121
(nitralin) eV/A. These are **model disagreement**, not errors against DFT or
experiment, and not an accuracy ranking. Absolute energies from different
reference levels must not be ranked against each other.

Changing the gas model does not change the OBC equation, fixed external charges
or radii at an identical geometry. Changing OBC64 to OBC32 separately introduces
rounding differences. Gas forces, optimized geometries and sampled populations
can change substantially; better solvation, barrier or MD accuracy is not
established by the model name or by these local perturbations.

See the official [AIMNet2](https://isayevlab.github.io/aimnetcentral/models/aimnet2/),
[2025](https://isayevlab.github.io/aimnetcentral/models/aimnet2_2025/) and
[NSE](https://isayevlab.github.io/aimnetcentral/models/aimnet2nse/) guides.
