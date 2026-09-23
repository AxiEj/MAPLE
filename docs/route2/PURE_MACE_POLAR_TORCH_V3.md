# Pure MACE-POLAR: explicit Torch analytic-derivative reference

## Scientific identity

This experimental route evaluates one coordinate-connected scalar:

```
E(R) = E_vac(MACE-POLAR; R) + G_ddPCM(R, c0(R)) + G_legacy-SMD-CDS(R)
c0(R) = MACE-POLAR(R, external field = 0)
```

Forces, Hessian-vector products and the raw Cartesian Hessian are generated
by Torch differentiation of this sum. In particular, **frozen source means
zero solvent-field response, not frozen coordinate derivatives**: `dc0/dR`,
`d2c0/dR2` and mixed continuum/source terms remain on the graph. There is no
production finite-difference, pyddx-solver or PySCF-CDS fallback. Old v1/v2
profiles are preserved as independent reference paths.

The rewrite retains the official checkpoint, float64, point-l<=1 source,
ddPCM lmax=15/Lebedev-1202/eta=0.1, dielectric and Coulomb radii, and the
published legacy SMD-CDS coefficients/radii. It neither fits parameters nor
changes the MNSol/FreeSolv energy ledger or experimental accuracy claims.

The equations are bound to [ddX 0.8.0](https://github.com/ddsolvation/ddX/tree/4d79e3d9caeae5e602683572a71cb550414f9b09)
and [PySCF 2.13.1 legacy SMD](https://github.com/pyscf/pyscf/blob/f3754ed5baad778280dba5ad3f4982a8bed0ec2f/pyscf/lib/solvent/mnsol.F).
The latter uses **analytical union-of-spheres accessible areas (DAREAL)**,
not the separate PySCF experimental 590-point SWIG surface. The SMD physical
parameterization is described by [Marenich, Cramer and Truhlar, 2009](https://doi.org/10.1021/jp810292n).

## Separation of responsibilities

| Module | Responsibility |
| --- | --- |
| `models/mace_polar_torch.py` | Existing model's live zero-field energy/source graph |
| `surfaces/lebedev.py` | Neutral immutable quadrature constants; no SWIG semantics |
| `continuum/torch_ddpcm.py` | Exact ddPCM operators and two dense linear solves |
| `surfaces/legacy_dareal.py` | Analytical sphere-union area and local topology certificate |
| `nonpolar/legacy_smd_cds.py` | Legacy CDS coefficient algebra and units |
| `experimental/mace_polar_torch.py` | Sum once, then differentiate the full PES |
| `derivatives/analytic.py` | Immutable analytic result, distinct from Richardson evidence |
| `function/calculator/route2/_mace_polar_torch_calculator.py` | Explicit ASE/factory boundary |

## Explicit selection

```
#model=macepolm
#device=cuda:0
#solv(method=smd,provider=torch,profile=pure-macepolar-frozen-point-l1-ddpcm-smd-torch-cuda-v3,implicit=water,response=frozen,experimental=true)
#freq(method=mw,treat_imag_as_real=false)
```

For CPU change **both** device to `cpu` and profile suffix to `torch-cpu-v3`.
Examples are in [`examples/solvation/pure_torch`](../../examples/solvation/pure_torch).
The explicitly opened task surface is SP, OPT and MW-FREQ, with water and
hexane solvents. TS, IRC, MD and path/scan workflows remain closed for this
new route. No historical v1/v2 convergence/admission evidence is inherited.

`opt-cpu.inp` is a near-stationary **one-step interface smoke** from the old
frozen canary, not a claim that v3 can converge arbitrary starting geometries.
`opt-topology-boundary-cpu.inp` preserves an observed negative: its first
LBFGS trial reaches a genuine `f_i≈1` cavity branch boundary and fails closed.
Do not remove the guard or relabel this failure as optimization success.

This v3 route requires PyTorch >=2.2's public graph-inspection API and was
verified with the existing Torch 2.12.0 runtime. Older runtimes fail with an
explicit message; the project's broader legacy Torch minimum is unchanged.

Run from this checkout with the pinned environment:

```bash
PYTHONPATH="$PWD" LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}" \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -m maple.main examples/solvation/pure_torch/sp-cpu.inp /new/path/sp.out
```

The live energy/derivative tensor chain uses the requested Torch device.
Constant-data initialization and discrete geometry-topology control may use
the host; they do not supply detached continuous derivatives. CUDA's serial
deterministic transaction spans forward **and both backward passes**. A CUDA
failure does not trigger CPU or legacy fallback. Molecular-mode projection
and its NumPy eigensolver remain **CPU**, reported separately from the model,
continuum and CDS device.

## Important limits

- This is a **bounded dense reference**, not a scalable replacement for all
  original 16–500 Da molecules. With the current 1 GB forward/first-backward
  preflight, the registered discretization allows at most **five atoms**;
  six atoms are rejected before continuum grid/operator allocation. Full
  second backward has a separate preflight: a four-times-first-order
  continuum estimate (4 GB operation cap), plus a 2 GiB CPU or 512 MiB CUDA
  model/runtime reserve. CUDA also requires 2 GiB of host headroom. Host and
  device estimates are checked before building the graph, not conflated.
  This is an engineering estimate, not an allocator upper-bound proof;
  measured memory is also reported. Do not bypass these guards to claim
  large-system efficiency.
- Legacy ddPCM and DAREAL contain piecewise geometry branches. Analytic
  derivatives are local to a regular branch, not a claim of global C2
  smoothness. Ambiguous intersections, genuine exposure-boundary crossings
  and legacy radius-perturbation cases fail explicitly; radii are not changed
  to force a result. Stable buried ddPCM plateaus are not misclassified as
  singular boundaries.
- An analytic Hessian still has floating-point, solver, discretization and
  model errors. Its uncertainty is **unavailable**, not zero. Raw signed
  frequencies may be shown, but resolved minimum/index-one certification
  remains false without an independent error bound.
- CPU/GPU implementation parity and derivative checks do not establish
  experimental geometry, frequency, barrier or solution-free-energy accuracy.
  No release/physical-accuracy admission is opened.

## Reproducible before/after audit

```bash
PYTHONPATH="$PWD" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python tools/route2_release/run_pure_mace_polar_torch_canary.py \
  --case water-water --device cpu \
  --checkpoint /path/to/MACEPOLAR1Mmodel \
  --output-dir /new/path/torch-water-audit --legacy-hessian
```

The runner preserves old/new energies and forces, optional legacy numerical
Hessian results (including failures), raw Torch Hessian, independent legacy
force-difference HVP checks inside matching discrete topologies, timings,
memory observations and source/checkpoint hashes. It never imports
experimental dataset labels. Use fresh output directories; failed evidence
must not be overwritten.

A later fixed ten-species CPU audit is retained in
[`route2-pure-macepolar-torch-v3-ten-species-audit.json`](evidence/route2-pure-macepolar-torch-v3-ten-species-audit.json):
six passed all frozen gates, three failed the independent legacy-force
finite-difference HVP gate, and HCN failed closed at the ddPCM topology guard.
The nine evaluable cases passed old/new energy and force limits. This negative
panel is not experimental solvation or physical Hessian accuracy evidence.

Five-dataset CDS fitting is a separate deferred task. Framework migration
alone does not justify changing fitting objectives, parameters or test splits.
