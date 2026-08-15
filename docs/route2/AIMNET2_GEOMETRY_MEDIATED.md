# AIMNet2 geometry-mediated Route-2 candidate

## Scientific identity

This disabled diagnostic covers a direct geometry-adaptive charge path,

```text
c_AIMNet2(R) = [q_NQE(R), 0, 0, 0]
E_gm(R) = E_AIMNet2(R) + 0.5 <c_AIMNet2(R), P_R c_AIMNet2(R)>_Q
G_np(R) = 0
```

where `P_R` is the pyddx ddPCM reaction map rebuilt at the current geometry.
AIMNet2 receives coordinates, elements, total charge, and multiplicity, but no
reaction field. Therefore this path contains the outer
`R -> q_NQE(R) -> ddPCM` response and **does not contain fixed-geometry
electronic mutual polarization**.

On one fixed neighbor/cavity-topology stratum, this is an ordinary composite
scalar. AIMNet2's energy and charge heads do not need to be conjugate here:
`q_NQE(R)` is an explicit differentiable function inserted into the PCM
scalar, not an independently optimized electronic coordinate. This statement
is local to the stratum; no global `C1` claim is made across hard neighbor-list
or cavity active-set events.

The model choice is grounded in the published AIMNet2 neural charge
equilibration construction and geometry-dependent charge output
([Chemical Science 2025](https://doi.org/10.1039/D4SC08572H)). The maintained
upstream implementation and model documentation live in
[AIMNetCentral](https://github.com/isayevlab/aimnetcentral). The local
TorchScript asset used by MAPLE is bound by SHA256, but its original upstream
release/commit has not been recovered; that provenance gap is recorded in the
checkpoint contract and blocks admission.

The continuum term follows the ddX half-coupling and adjoint derivative
construction: polarization energy is one half of the source/reaction-field
pairing, and analytic nuclear derivatives use the corresponding forward and
adjoint states ([ddX theory documentation](https://ddsolvation.github.io/ddX/md_docs_theory.html)).
The factor `1/2` alone is not a reciprocity proof. For the reaction field itself
to be the charge-energy cotangent, the discrete operator must satisfy

```text
<u, P_R w>_Q = <w, P_R u>_Q
```

in the registered metric. The implementation therefore checks deterministic
fixed-total-charge tangent probes, apply/adjoint dot products, three charge
finite-difference steps, and `J_q^T 1 = 0`; checking only
`G = 0.5 <q, Pq>` would be blind to an antisymmetric operator component. The
finite exposed Lebedev discretization is not treated as a structural global
`SO(3)` proof.

## Complete first derivative

For `G_pcm(R,c) = 0.5 <c, P_R c>_Q`, the implemented gradient is

```text
dE_gm/dR = dE_AIMNet2/dR
          + partial_R G_pcm(R,c) |_c
          + (D_R c)^T grad_c G_pcm(R,c).
```

Each term has one owner:

- `models/aimnet2.py` binds the checkpoint, model contract, neighbor topology,
  intrinsic energy/forces, and charge-position VJP;
- the historical negative-control arm delegates those primitives to the
  existing TorchScript `AIMNet2Calculator`, while the optional precision arm
  in `maple/function/calculator/aimnet/_aimnet2_float64_source.py` rebuilds the
  official `aimnet==0.2.0` Python architecture and loads the same unchanged
  checkpoint state dictionary;
- `continuum/atomic_l1_pyddx.py` delegates field, source JVP/VJP, and coordinate
  VJP to the existing `PyDDXPCMReactionFieldLinearMap`;
- `coupling/geometry_mediated.py` owns the half-coupling ledger and composes the
  three gradient terms.

No PCM equation or chain-rule term is reimplemented in a second monolithic
objective. The precision arm reuses the maintained upstream AIMNet2 modules
rather than forking the network implementation. It is pinned to the exact
package version, official wB97M-D3 YAML, and SHA256 of every AIMNet package
source/configuration/data file used by the dense molecular forward; external
runtime package versions are recorded separately. Any bound-file drift fails
closed.
No dependency was added to MAPLE: this is an optional research runtime and is
available only when the exact upstream package is already installed.

Synthetic and real-stack full-Cartesian finite differences lock the complete
scalar and include negative tests for an antisymmetric reaction operator, a
charge projection with nonzero gauge VJP, incomplete Cartesian coverage, and
nearby continuum event surfaces. The Cartesian wrapper reuses MAPLE's existing
PES convergence primitive; it does not introduce a second set of numerical
thresholds.

## Precision-controlled checkpoint reconstruction

The serialized legacy TorchScript graph calls `torch.to(coord, 6)` in its
input preparation; Torch dtype enum `6` is `float32`. Calling `double()` on the
loaded module therefore does not produce a double-precision coordinate graph:
each forward converts the coordinates back to float32 before distances are
formed. This is why the legacy arm is retained explicitly as a negative
control rather than hidden by selecting a favorable finite-difference step.

`AIMNet2ReconstructedFloat64SourceCalculator` instead:

1. verifies `aimnet==0.2.0` and content-addresses its configuration and source;
2. builds the official `aimnet2_dftd3_wb97m.yaml` architecture;
3. loads the unchanged local checkpoint state dictionary, allowing only its
   known unused misspelled sentinel buffer;
4. converts the complete Python forward and coordinate contract to float64;
5. exposes only the same-forward energy/charge state and `J_q^T v` research
   primitives.

It is CPU-only, not registered as an ASE calculator, rejects public
`calculate()`, and supplies no Hessian/HVP. It changes numerical precision and
execution graph, not weights, input semantics, neighbor policy, or charge
projection. Agreement with the historical float32 graph is tested separately;
that agreement is a reconstruction check, not chemical-accuracy evidence.

The opt-in real-checkpoint test uses the SHA256-bound AIMNet2 asset and
`pyddx==0.8.0`. It verifies the reciprocity/metric/gauge audit, then deliberately
applies the stricter three-step directional, complete `3N` Cartesian, and
three-rotation fail-closed harness. The current water diagnostic keeps one
cavity active set over the small coordinate stencil, but the legacy float32
energy differences do not form an admissible convergence window; the
laboratory-frame Lebedev active set also changes under all three frozen
rotations. These are negative admission results, not a tuned threshold pass:

```bash
export MAPLE_ROUTE2_REAL_AIMNET2=1
export MAPLE_ROUTE2_AIMNET2_CHECKPOINT=/absolute/path/to/aimnet2.pt
python -m pytest -q -s \
  tests/route2_vnext/test_aimnet2_real_geometry_mediated.py
```

A clean-tree, source-bound JSON runner writes outside the checkout:

```bash
python tools/route2_release/run_aimnet2_geometry_mediated_canary.py \
  --checkpoint /absolute/path/to/aimnet2.pt \
  --aimnet-runtime legacy-jit-float32 \
  --continuum ddpcm \
  --device cpu \
  --output /absolute/path/outside/the/repository/aimnet2-gm-water.json
```

The runner records the checkpoint, selected precision runtime, upstream source
hashes for the reconstruction arm, loaded committed sources, registered metric,
model hard-neighbor graph, exact exposed sphere/Lebedev candidate set,
three-step directional measurements, all `3N` Cartesian energy stencils,
rotation measurements, and the fact that this pyddx provider exposes a
requested solver tolerance but not a measured post-solve algebraic residual.
The pyddx topology record likewise has exact sampled active-set identity but no
continuous distance-to-active-set-event value; the artifact marks that event
margin as unavailable/not applicable instead of inventing one. It always
leaves all capabilities false.

## Structurally rotational harmonic branch

The separately registered smooth-harmonic diagnostic replaces the laboratory
Lebedev surface with MAPLE's existing fixed-dimensional weighted harmonic
Galerkin assembly and an analytic point-monopole boundary map. It keeps the
same unmodified AIMNet2 charge function and supplies structural `SO(3)`
intertwiners rather than tuning a larger angular grid.

The real water canary preserves the continuum stratum under all frozen
rotations and passes the complete rigid-rotation gate. The legacy float32 arm
remains negative for both the directional and full-Cartesian energy-difference
gates. The source-bound float64 arm passes those local one-water derivative
gates with central-difference refinement, while using the same checkpoint
weights and topology. This is a precision-isolation result, not force or task
admission: broader geometry/chemistry, closed-loop, event, HVP, and release
panels remain absent. The branch is also a conductor reference without
finite-dielectric solvent parameterization, so it must not be presented as an
admitted water ddPCM replacement. See
[`AIMNET2_POINT_HARMONIC.md`](AIMNET2_POINT_HARMONIC.md) for the addition
theorem, parent functional, topology events, executed tests, and claim
boundary.

## Preregistered distorted-geometry H/C/N/O panel

The smooth-harmonic float64 branch now reuses MAPLE's frozen twenty-molecule
PES asset instead of defining a favorable AIMNet2-only molecule set.  The
local checkpoint contract admits exactly the seventeen asset entries composed
of H/C/N/O; thiophene, methanethiol, and chloroform remain explicit S/Cl
exclusions rather than being silently evaluated outside the declared domain.

Each source-bound shard is fixed to one stable molecule index and contains:

- the reference, deterministic bond-compressed, and bond-stretched geometries;
- the seeded-internal, radial-internal, and bond-stretch unit directions;
- central differences at `4e-4`, `2e-4`, and `1e-4` angstrom;
- two independent center evaluations, the full reciprocity/metric/charge-gauge
  record, measured stationary residual and condition number, and raw displaced
  scalar energies;
- exact hard-neighbor and point/source-shell topology identities plus their
  minimum event distances over every stencil.

`summarize_aimnet2_geometry_mediated_pes_shard` recomputes all 27 central
differences (54 displaced scalar energies) for one molecule.  The full-panel
summarizer accepts only all seventeen raw shards in frozen order (51 geometries,
153 directional records, 459 step records); it cannot aggregate a cherry-picked
subset.  Both layers hard-code all public capabilities, OPT, FREQ/TS/IRC, and
MD to false.

```bash
python tools/route2_release/run_aimnet2_geometry_mediated_pes_panel.py \
  --checkpoint /absolute/path/to/aimnet2.pt \
  --molecule-index 0 \
  --device cpu \
  --output /absolute/path/outside/the/repository/aimnet2-gm-water-pes.json
```

The runner is deliberately limited to the source-bound float64 runtime and
the smooth harmonic conductor reference.  A passing shard is only local
distorted-geometry evidence; the complete chemistry panel requires seventeen
separately captured clean-tree shards and still does not establish a physical
finite-dielectric solvation model or chemical accuracy.

The water shard has been captured twice from execution commit `5244de8c` with
identical measurement SHA.  All 27 central differences pass; the maximum
absolute error is `5.866424167422224e-05 eV/angstrom`, the minimum
point/source-shell margin is `0.11773221132261891 angstrom`, and the maximum
stationary condition number is `150.593715831781`.  This is one positive shard,
not the seventeen-molecule panel.  Raw measurements and the cold replay are in
[`evidence/aimnet2-geometry-mediated-pes-water-5244de8c/`](evidence/aimnet2-geometry-mediated-pes-water-5244de8c/README.md).

## Fixed-geometry response no-go

At fixed `R`, the unmodified deterministic model always returns the same
`q_NQE(R)`, hence `partial q_NQE / partial sigma = 0`. Feeding PCM state into a
modified neighbor graph, virtual coordinates, hidden state, output correction,
or any other changed forward rule defines a new function even if checkpoint
weights are unchanged. This branch therefore forbids `sigma -> graph -> q`
and any description such as PCM-conditioned or mutually polarized AIMNet2.

## Bound domain and capability status

The first checkpoint contract is deliberately narrower than the published
model domain: neutral closed-shell, nonperiodic H/C/N/O molecules with fixed
atom identity and order. This is the subset exercised by retained MAPLE
AIMNet2/ddPCM work, not a claim that other published AIMNet2 elements are
unsupported upstream.

| item | status |
| --- | --- |
| local fixed-stratum scalar/gradient | implemented; diagnostic only |
| metric reciprocity / charge-direction / gauge audit | implemented; real water passes locally |
| pyddx three-step real coordinate admission gate | fails current water diagnostic |
| pyddx rigid rotation / exact cavity-active-set gate | fails current laboratory-grid diagnostic |
| harmonic-point structural continuum rotation | passes synthetic and real water canaries |
| harmonic-point full rigid-rotation gate | passes current real water canary |
| harmonic-point directional/full-Cartesian derivative gates | legacy float32 fails; source-bound float64 passes the local one-water canary |
| harmonic-point distorted-geometry PES harness | exact 17-shard H/C/N/O contract implemented; water passes two clean processes, remaining 16 shards absent |
| source-bound float64 reconstruction | optional CPU research primitive; unchanged weights; upstream source hashes recorded |
| fixed-geometry electronic mutual polarization | absent by model interface |
| SMD-CDS/nonpolar and standard-state terms | excluded |
| public single-point E/F | disabled |
| solution-phase OPT/NEB/TS | disabled pending full PES/release gates |
| Hessian/HVP/FREQ | absent |
| MD/NVE | absent |
| strict variational tier | not applicable/proven |

The original SMD CDS model is a separately parameterized solvent-accessible
surface term ([SMD paper](https://doi.org/10.1021/jp810292n)). It cannot be
appended here until one versioned nonpolar provider supplies the same-scalar
coordinate derivative and passes cavity/profile compatibility gates.

## Remaining admission gates

1. Recover the exact upstream release identity of the local checkpoint.
2. Execute all seventeen preregistered distorted-geometry H/C/N/O shards, then
   continue through bidirectional closed-loop and explicit
   cutoff/source-shell/sphere-tangency trial-step panels. One equilibrium-water
   float64 directional and Cartesian panel is positive, but it is not a domain
   or workflow gate. The legacy float32 arm remains a negative control, and the
   pyddx arm still has both derivative and laboratory-grid rotation failures.
3. Bind solvent dielectric, radii, grid, and solver choices to separately named
   physical-configuration profiles instead of the current unbound diagnostic.
4. Add a same-scalar nonpolar provider before making total solvation-free-energy
   or multi-solvent claims.
5. Add force-domain, optimization, Hessian/FREQ, and NVE evidence before any
   corresponding MAPLE workflow is enabled.
6. For Hessian/FREQ/TS/IRC, implement the complete HVP including the contracted
   AIMNet2 charge Hessian `D_R[J_q^T v][h]`, PCM `RR/Rq/qR/qq` blocks, mixed
   adjointness, and a full `C2` event-free displacement neighborhood.

No fitting, radius tuning, response tempering, calibration, or experimental
label use is part of this candidate.
