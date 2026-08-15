# AIMNet2 geometry-mediated float64 H/C/N/O PES panel v2: negative aggregate

This bundle retains two independent clean-process executions of the exact
seventeen-shard H/C/N/O distorted-geometry panel. Every primary/replay shard
has an identical scientific measurement SHA-256, and independent aggregation
reproduces the same raw-record digest and panel summary. The complete panel is
negative and all public capabilities remain disabled.

The JSON artifacts are stored as deterministic `gzip -9 -n` streams to avoid
expanding about 42 MiB of highly repetitive JSON in the checkout.
`SHA256SUMS` binds the compressed files; `RAW_SHA256SUMS` binds the exact JSON
bytes obtained with `gzip -dc`.

## Bound identity

- execution commit: `cd8769d4cdad439f9aa861429da4069535092c4a`
- execution tree: `3a14e042cfc94cfcce892e8ae24ee3af838c18e5`
- contract: `route2-aimnet2-geometry-mediated-pes-panel-contract-v2`
- aggregate schema: `route2-aimnet2-geometry-mediated-pes-panel-aggregate-v2`
- local checkpoint SHA-256:
  `85ba59d8c78eb4d3185f6b1614df79706427f7ca72f53f2d90e365a1723d953d`
- runtime: reconstructed official `aimnet==0.2.0` Python graph, unchanged
  checkpoint weights, CPU float64
- thread environment: `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`,
  `OPENBLAS_NUM_THREADS=1`
- continuum: fixed-dimensional smooth-harmonic point-monopole conductor
  reference; no finite-dielectric solvent identity
- bound committed Python sources per shard: `133`
- aggregate raw-record measurement SHA-256, both processes:
  `b0796da42ebd729f0bd33cc3d98ea3a0f0b3711d3d2d3819086b0df5e860e3b4`
- primary aggregate JSON SHA-256:
  `3da87ce2cda2a3eb8665158be7cf9e2741bb78402c2415c7a8a97d5707d64523`
- replay aggregate JSON SHA-256:
  `b6e9ac1babad58432a2477eae0e699812b0cd187b31e0933f4e50c0cc49145d3`

## Exact execution pattern

Each index `0..16` was executed in a separate process from the same clean tree:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python tools/route2_release/run_aimnet2_geometry_mediated_pes_panel.py \
  --checkpoint /home/axie/MAPLE/MAPLE-implicitsolv-route1/maple/function/calculator/model/aimnet2.pt \
  --molecule-index INDEX --device cpu \
  --output /tmp/maple-route2-aimnet2-pes-panel-cd8769d4-RUN/shard-INDEX.json
```

Both complete sets were then passed, with one `--shard` argument per index, to:

```bash
python tools/route2_release/aggregate_aimnet2_geometry_mediated_pes_panel.py \
  --shard ... \
  --output /tmp/maple-route2-aimnet2-pes-panel-cd8769d4-RUN/panel.json
```

The aggregator wrote the negative artifact and exited with status `2` in both
processes, as required by the fail-closed contract.

## Result

The aggregate covers 17 molecules, 51 frozen geometries, 153 directional
records, 459 three-step records, and 918 displaced scalar-energy evaluations.
Only `water` and `hydrogen-peroxide` pass every v2 diagnostic gate. The other
fifteen molecule IDs are explicitly retained in `failed_molecule_ids`.

Panel-wide gates that pass:

- all deterministic replays;
- all stationary continuum residual/conditioning gates;
- all metric reciprocity, apply/adjoint, charge-direction FD, and charge-gauge
  VJP audits;
- all hard-neighbor cutoff guards;
- all sampled stencils remain in their exact model/cavity stratum.

Observed negative boundaries:

- methanol, methane, dimethyl ether, and acetic acid fail the frozen
  `0.02 Angstrom` point/source-shell event-distance guard;
- ethylamine fails the independent `0.02 Angstrom` sphere-tangency guard;
- fourteen molecules have at least one directional scan without a demonstrated
  refinement/low-error plateau over the frozen `4e-4`, `2e-4`, `1e-4
  Angstrom` sequence; five of those also contain at least one individual
  numerical-threshold failure;
- no neighbor topology changes occur, so these failures cannot be hidden as a
  neighbor-list artifact.

Aggregate extrema:

- maximum directional absolute error:
  `6.495763803818777e-4 eV/Angstrom`;
- minimum hard-neighbor cutoff margin: `0.03499453808830477 Angstrom`;
- minimum point/source-shell margin: `0.002235739521015301 Angstrom`;
- minimum sphere-tangency margin: `0.011178247440158717 Angstrom`;
- maximum stationary surface condition number: `1943.3160873049574`.

The small-step convergence failures are evidence that the v2 three-step window
is insufficient for much of this float64 real stack, not permission to relabel
one favorable step as converged. A future broader step scan must be a new
versioned diagnostic. It cannot erase the independently observed topology-event
failures.

## Claim boundary

This bundle maps the current conductor-reference diagnostic domain. It is not
finite-dielectric solvent validation, chemical-accuracy evidence, a global
`C1`/`C2` proof, fixed-geometry mutual polarization, or admission of single-
point E/F, OPT, FREQ, TS, IRC, MD, or public ASE integration.

All public gates remain closed: `E/F/H/V/M = false`, `OPT = false`,
`FREQ/TS/IRC = false`, and `MD = false`.
