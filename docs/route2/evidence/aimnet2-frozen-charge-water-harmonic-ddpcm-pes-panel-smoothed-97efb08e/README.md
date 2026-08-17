# Frozen AIMNet2 charge + water harmonic-ddPCM PES panel after smooth DFT-D3 repair

This bundle retains two independent clean-process executions of the exact
seventeen-shard H/C/N/O distorted-geometry panel after replacing the noisy
ordinary embedded DFT-D3 energy derivative path with the source-bound upstream
smooth differentiable DFT-D3 path. Every primary/replay shard has an identical
scientific measurement SHA-256, and both aggregates reproduce the same raw-
record digest and panel summary.

No PES threshold, step, molecule, geometry, continuum equation, cavity
parameter, source definition, or scalar formula changed. The earlier runtime-v2
negative panel remains retained separately. This runtime-v3 panel isolates the
numerical forward-path repair while preserving all public capabilities as
false.

The JSON artifacts are deterministic `gzip -9 -n` streams. `SHA256SUMS` binds
the compressed files; `RAW_SHA256SUMS` binds the exact decompressed JSON bytes.

## Bound identity

- execution commit: `97efb08ecbeff1a30a8876f3f83f01b40e10ad0e`
- execution tree: `af0cc177b4965fbed4c967f7bb595ac698ee698e`
- runtime: `aimnet-reconstructed-float64-runtime-v3`
- unchanged checkpoint SHA-256:
  `85ba59d8c78eb4d3185f6b1614df79706427f7ca72f53f2d90e365a1723d953d`
- aggregate contract:
  `route2-aimnet2-frozen-charge-water-harmonic-ddpcm-pes-panel-contract-v1`
- model source: one zero-field AIMNet2 NQE charge evaluation per geometry; no
  continuum field is supplied to AIMNet2 and no electronic SCF is performed
- continuum: finite-dielectric water (`epsilon = 78.355`) smooth harmonic
  ddPCM electrostatics with the exact registered SMD Coulomb-radii and cavity
  discretization identity
- first-order graph: embedded DFT-D3 removed from a frozen copy of the exact
  Python reconstruction and reapplied through the same source-bound upstream
  `hessian=True` path
- ordinary-forward parity gates on every evaluated geometry: energy `1e-6 eV`,
  charge `1e-10 e`, intrinsic gradient `1e-7 eV/Angstrom`, charge VJP
  `1e-10 eV/Angstrom`
- bound committed Python sources per shard: `135`
- aggregate raw-record measurement SHA-256, both processes:
  `a219082dbe88097923fd18b39fff83a613f1a7a02d57d7e126df2e549708c42b`
- primary aggregate JSON SHA-256:
  `7a92e85b5c9c7f5844dbd94586ce161733a7d37c06fc31050042eb1c1c8202b4`
- replay aggregate JSON SHA-256:
  `a1f3388d25d017cf498130b3853b22d90f51fe70dc7ca1629a875d614adc2afb`

## Result

The aggregate covers 17 molecules, 51 frozen geometries, 153 directional
records, 459 three-step records, and 918 displaced scalar-energy evaluations.
Twelve molecules pass every shard gate:

`water`, `ethanol`, `acetone`, `acetonitrile`, `benzene`, `trans-butane`,
`formic-acid`, `acetaldehyde`, `acetamide`, `pyridine`, `nitromethane`, and
`hydrogen-peroxide`.

All seventeen molecules now pass the frozen three-step refinement/convergence
criterion. The maximum force finite-difference error falls from the runtime-v2
panel's `6.4958e-4 eV/Angstrom` to `5.1489e-4 eV/Angstrom`; the latter is one
methane sample inside an independently failed cavity-event guard.

Five molecules remain fail-closed for topology reasons:

- methanol, methane, dimethyl ether, and acetic acid fail the frozen
  `0.02 Angstrom` point/source-shell event-distance guard;
- ethylamine fails the independent `0.02 Angstrom` sphere-tangency guard;
- methane also contains one individual numerical-threshold failure, while its
  three-step convergence test passes.

Every panel-wide stationarity, KKT/autograd, energy-cotangent, reciprocity,
apply/adjoint, charge-direction FD, charge-gauge VJP, deterministic replay,
hard-neighbor, and same-stratum gate passes. Aggregate extrema include:

- maximum directional absolute error: `5.148876215019804e-4 eV/Angstrom`;
- minimum hard-neighbor cutoff margin: `0.03499453808830477 Angstrom`;
- minimum point/source-shell margin: `0.002235739521015301 Angstrom`;
- minimum sphere-tangency margin: `0.011178247440158717 Angstrom`;
- maximum stationary surface condition number: `1943.3160873049574`;
- maximum KKT-vs-autograd relative error: `6.71025801204701e-14`.

## Exact execution pattern

Each index `0..16` was run in a separate clean process:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python tools/route2_release/run_aimnet2_geometry_mediated_pes_panel.py \
  --checkpoint /path/to/aimnet2.pt --device cpu \
  --continuum harmonic-ddpcm-water --molecule-index INDEX \
  --output /tmp/aimnet2-smoothed-RUN/shard-INDEX.json
```

The seventeen shards were aggregated with one `--shard` argument per index and
`--continuum harmonic-ddpcm-water`. The fail-closed aggregator wrote the
negative artifact and exited `2` in both runs.

## Claim boundary

This repair establishes broad numerical force consistency only inside the
sampled smooth cavity strata. The full preregistered domain still fails its
independent cavity-event guards, so it does not admit public E/F, OPT, FREQ,
TS, IRC, MD, or ASE integration. It is not fixed-R mutual polarization, an
electronic SCF method, complete solvation free energy, nonpolar/CDS treatment,
chemical-accuracy evidence, or a global `C1`/`C2` proof.

All public gates remain closed: `E/F/H/V/M = false`, `OPT = false`,
`FREQ/TS/IRC = false`, and `MD = false`.
