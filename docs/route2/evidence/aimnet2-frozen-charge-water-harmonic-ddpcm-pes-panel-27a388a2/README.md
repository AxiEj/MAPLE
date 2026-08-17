# Frozen AIMNet2 charge + water harmonic-ddPCM PES panel: negative aggregate

This bundle retains two independent clean-process executions of the exact
seventeen-shard H/C/N/O distorted-geometry panel for the registered
water-bound frozen-charge candidate. Every primary/replay shard has an
identical scientific measurement SHA-256. Independent aggregation reproduces
the same raw-record digest and panel summary. The complete panel is negative;
all public capabilities remain disabled.

The JSON artifacts are stored as deterministic `gzip -9 -n` streams to avoid
expanding roughly 40 MiB of repetitive JSON in the checkout. `SHA256SUMS`
binds the compressed files; `RAW_SHA256SUMS` binds the exact JSON bytes
obtained with `gzip -dc`.

## Bound identity

- execution commit: `27a388a23ea6cb8c54e96a36938b81eae486c45b`
- execution tree: `7c67b5148c3d42853d3f0e1b10efc32b7e1c3d7d`
- shard contract:
  `route2-aimnet2-frozen-charge-water-harmonic-ddpcm-pes-shard-contract-v1`
- aggregate contract:
  `route2-aimnet2-frozen-charge-water-harmonic-ddpcm-pes-panel-contract-v1`
- local checkpoint SHA-256:
  `85ba59d8c78eb4d3185f6b1614df79706427f7ca72f53f2d90e365a1723d953d`
- runtime: reconstructed official `aimnet==0.2.0` Python graph, unchanged
  checkpoint weights, CPU float64
- model source: one AIMNet2 NQE charge evaluation per geometry; no continuum
  field is supplied to AIMNet2 and there is no electronic SCF iteration
- continuum: finite-dielectric water (`epsilon = 78.355`) smooth harmonic
  ddPCM electrostatics with the exact registered SMD Coulomb-radii and cavity
  discretization identity
- thread environment: `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`,
  `OPENBLAS_NUM_THREADS=1`
- bound committed Python sources per shard: `135`
- aggregate raw-record measurement SHA-256, both processes:
  `b53dacb5d14c46d621b1900c8734e396ec939b8398a10a28f2cecb450ed16d72`
- primary aggregate JSON SHA-256:
  `25eea4aac0ff834490ce9fbeaf4a78cdfb114ed9147fcaf1e38688394c819c69`
- replay aggregate JSON SHA-256:
  `66418008fe5ca35e9ba1bd2cf6fa60cc5c9a95c56012a170fe3850120d991e68`

## Exact execution pattern

Each index `0..16` was executed in a separate process from the same clean tree:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python tools/route2_release/run_aimnet2_geometry_mediated_pes_panel.py \
  --checkpoint /home/axie/MAPLE/MAPLE-fork/maple/function/calculator/model/aimnet2.pt \
  --device cpu --continuum harmonic-ddpcm-water \
  --molecule-index INDEX \
  --output /tmp/aimnet2-frozen-water-ddpcm-pes-RUN/shard-INDEX.json
```

Both complete sets were then passed, with one `--shard` argument per index, to:

```bash
python tools/route2_release/aggregate_aimnet2_geometry_mediated_pes_panel.py \
  --continuum harmonic-ddpcm-water \
  --shard ... \
  --output /tmp/aimnet2-frozen-water-ddpcm-pes-RUN/panel.json
```

The aggregator wrote the negative artifact and exited with status `2` in both
processes, as required by the fail-closed contract.

## Result

The aggregate covers 17 molecules, 51 frozen geometries, 153 directional
records, 459 three-step records, and 918 displaced scalar-energy evaluations.
Only `water` and `hydrogen-peroxide` pass every diagnostic gate. The other
fifteen molecule IDs are explicitly retained in `failed_molecule_ids`.

Panel-wide gates that pass:

- all deterministic replays;
- all stationary ddPCM primal/transpose/KKT residual and conditioning gates;
- all energy-cotangent, metric reciprocity, apply/adjoint, charge-direction FD,
  and charge-gauge VJP audits;
- exclusion of primal response values from the field supplied to the model;
- all hard-neighbor cutoff guards;
- all sampled stencils remain in their exact model/cavity stratum.

Observed negative boundaries:

- methanol, methane, dimethyl ether, and acetic acid fail the frozen
  `0.02 Angstrom` point/source-shell event-distance guard;
- ethylamine fails the independent `0.02 Angstrom` sphere-tangency guard;
- fourteen molecules have at least one directional scan without a demonstrated
  refinement/low-error plateau over the frozen `4e-4`, `2e-4`, `1e-4
  Angstrom` sequence; five of those also contain an individual numerical-
  threshold failure;
- no hard-neighbor topology changes occur, so these failures cannot be hidden
  as a neighbor-list artifact.

Aggregate extrema:

- maximum directional absolute error:
  `6.495762976865826e-4 eV/Angstrom`;
- minimum hard-neighbor cutoff margin: `0.03499453808830477 Angstrom`;
- minimum point/source-shell margin: `0.002235739521015301 Angstrom`;
- minimum sphere-tangency margin: `0.011178247440158717 Angstrom`;
- maximum stationary surface condition number: `1943.3160873049574`;
- maximum KKT-vs-autograd relative error: `6.71025801204701e-14`.

The small-step convergence failures are evidence that the registered three-step
window does not demonstrate broad force convergence for this real float64
stack. They are not permission to relabel one favorable step as converged. A
future broader step scan must be a new versioned diagnostic and cannot erase
the independently observed topology-event failures.

## Claim boundary

This bundle maps the current electrostatic PES diagnostic domain for the exact
one-shot frozen-charge water candidate. It is not fixed-geometry mutual
polarization, an electronic SCF method, complete solvation free energy,
nonpolar/CDS treatment, solvent calibration, chemical-accuracy evidence, a
global `C1`/`C2` proof, or admission of single-point E/F, OPT, FREQ, TS, IRC,
MD, or public ASE integration.

All public gates remain closed: `E/F/H/V/M = false`, `OPT = false`,
`FREQ/TS/IRC = false`, and `MD = false`.
