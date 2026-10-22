# Torch v4: 179 named SMD solvent parameterizations

## Delivered scope

Default FP32 / lmax 5 / Lebedev 194 remains unchanged. Torch v4 now accepts
all 179 named valid entries from the PySCF 2.13.1 SMD descriptor table, not just
water/hexane. It reuses the published dielectric, optical, hydrogen-bonding,
surface-tension and composition descriptors. SMD acidity-dependent Coulomb
radii and aqueous/non-aqueous CDS formulas remain unchanged. No new fit or
solvation-network training is introduced.

The empty all-zero upstream sentinel is excluded. Names/aliases are normalized
without merging cis/trans isomers or mixtures. Runtime lookup is data-only and
does not import PySCF. Source version, source SHA256 and normalized parameter
snapshot SHA256 are recorded in `maple/function/smd_solvent_data.py`.

Full names/parameter reference: `SMD_SOLVENT_CATALOG.md`.
Example public input (full geometry must follow):

```text
#model=macepolm
#device=cpu
#solv(method=smd,provider=torch,implicit=acetone,experimental=true)
#sp(verbose=1)
```

Historical profile scopes are not silently broadened: Torch v3 still supports
water/hexane, and other legacy Route-2 profiles retain their original registered
scopes (water-only or the historical 11-solvent panel). Only those historical
11 have audited MNSol dataset name mappings; the
other 168 entries explicitly record `mnsol_name=None`. PySCF naming alone must
not be treated as a validated MNSol record join.

## Verification summary

- Initial multisolvency regression: **678 passed, 1 skipped**; current topology
  grading regression is recorded in `TOPOLOGY_GRADING_STATUS.md`.
- Multisolvent coverage and current-source evidence contracts: **13 passed**.
- The regression includes all 179 upstream CDS energy/analytic-gradient parity
  cases using unchanged existing tolerances.
- AST parsing, targeted pyflakes and `git diff --check` pass.

## Actual execution and limitations

- Fixed neutral water geometry, full 179-solvent CPU and CUDA sweeps:
  **179 strict E/F successes per device after topology grading**.
  Total: **358 attempted and recorded; 358 successes, zero skips**.
  Each device records warnings for the former 10 rejected combinations.
- The source graph is computed once per geometry/device and reused because
  this frozen-source model is solvent-independent. ddPCM and CDS are rebuilt
  for every solvent. These are cached-component checks, not 358 independent
  public-entry runs and not timing comparisons.
- Eight deterministic HVP representatives are attempted per device; all eight
  now succeed, including acetonitrile.
- Initial public `.inp` controls completed 10 of 11 selections; acetonitrile
  was rejected. After the numerical fix, its ordinary CPU and CUDA inputs both
  complete with a nonblocking warning.
- Independent upstream CDS parity is tested for all 179 solvents using the
  existing FP64 energy/analytic-gradient tolerances. This checks parameter
  reuse, not experimental solvation accuracy or FP32 derivative qualification.

The ten formerly rejected fixed-geometry combinations are: 1-hexyne, 2-butanol,
2-methyl-2-propanol, 2-propanol, acetonitrile, aniline, benzylalcohol,
cyclopentanol, N-methylformamide E/Z mixture and Z-1,2-dichloroethene.
These all have valid SMD parameters. The old blanket distance guard rejected
single-interior/C2 cases, while rounded FP32 switch values could also hide a
real derivative artifact. The active FP32 policy now evaluates complementary
quintics with one centered coordinate and selects exposure/normalization from
structural radial support. Proven smooth cases warn and continue. Genuine
multi-interior near-one ambiguity still fails closed; matrix/resource/finite
checks remain unchanged. No geometry, physical parameter, tolerance, or source
charge was tuned to improve the count. See `TOPOLOGY_GRADING_STATUS.md`.

## Evidence

- `evidence/route2-pure-macepolar-torch-fp32-v4-multisolvent-component-smoke-v1.json`
  binds full catalog values, geometry/source hashes, component identities and
  every success/failure to the source files.
- `evidence/route2-pure-macepolar-torch-fp32-v4-engineering-tests.json` was
  genuinely replayed on current code; its previous two-solvent artifact is
  preserved byte-for-byte in `evidence/history/`.
- Local public-entry logs and complete accounting:
  `.omx/research/multisolvent-v4-20261017/`.

Reproduce the catalog sweep in the configured environment:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
python -m tools.route2_release.run_pure_mace_polar_multisolvent_smoke \
  --device cpu --device cuda:0 --output new-multisolvent-evidence.json
```

The runner refuses overwrites and exits nonzero when any case fails, even if
all requested cases were recorded. Original 338/358-success evidence is archived
byte-for-byte under evidence/history; the new all-success record comes from
a fresh strict execution with current source hashes, not relabeling old rows. This is broad parameter integration,
not full-population scientific admission or a new experimental MAE result.
