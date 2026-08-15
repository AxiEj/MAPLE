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
5. exposes the same-forward energy/charge state and `J_q^T v`, plus a separate
   diagnostic second-order graph for `J_q h`, `H_E h`, and the fixed-cotangent
   contracted charge Hessian `D_R[J_q^T v][h]`.

It is CPU-only, not registered as an ASE calculator, rejects public
`calculate()`, and rejects the inherited public Hessian/HVP entry points. The
second-order method is a Route-2 research primitive only. It deep-copies the
source-bound upstream graph, replaces the embedded first-order-only DFT-D3
call by identity, and reapplies the same upstream DFT-D3 module with
`hessian=True`. Every call must reproduce the ordinary graph's energy,
charges, intrinsic gradient, and charge VJP within fixed tolerances before any
second-order tensor is returned. It changes numerical precision and execution
graph, not weights, input semantics, neighbor policy, or charge projection.
Agreement with the historical float32 graph is tested separately; that
agreement is a reconstruction check, not chemical-accuracy evidence.

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
weights and topology. A separate source-bound water HVP canary now closes the
complete local weak-scalar HVP ledger, finite-difference response, bilinear
symmetry, and all three translational zero modes. A second source-bound water
canary now finds one guarded stationary point, assembles all nine Cartesian
HVP columns, closes all-column gradient finite differences, and verifies three
stationary rotational zero modes plus the correctly mass-weighted three-mode
vibrational subspace. These remain precision/implementation results, not force
or task admission: broader geometry/chemistry, a complete event-free `C2`
domain, and workflow panels are absent. The branch is also a conductor
reference without finite-dielectric solvent parameterization, so it must not
be presented as an admitted water ddPCM replacement. See
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
  minimum event distances over every stencil;
- exact unordered sphere-pair `nested`/`intersecting`/`separated` strata and
  minimum distances to both internal and external tangency surfaces.

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

The shard/panel contract is now `v2`: a harmonic record without its independent
sphere-tangency topology and margin fails closed. Earlier `v1` evidence remains
source-bound to its execution commit and does not retroactively satisfy this
stronger event contract.

The earlier water v1 shard remains immutable under
[`evidence/aimnet2-geometry-mediated-pes-water-5244de8c/`](evidence/aimnet2-geometry-mediated-pes-water-5244de8c/README.md).
Water has now also been rerun twice under v2 with identical raw measurements;
all 27 central differences and the independent sphere-tangency gates pass
([v2 bundle](evidence/aimnet2-geometry-mediated-pes-water-v2-02c21b52/README.md)).

The next frozen shard, methanol, is retained as a reproducible negative v2
result
([bundle](evidence/aimnet2-geometry-mediated-pes-methanol-v2-02c21b52/README.md)).
All numerical derivative, reciprocity, stationarity, replay, neighbor, sphere-
tangency, and same-stratum checks pass, but the bond-stretched geometry lies
only `0.015511399564898554 angstrom` from a point/source-shell event, below the
preregistered `0.02 angstrom` guard. The reducer therefore fails all three
bond-stretched direction records closed. No threshold or radius was changed.
This event-distance result blocks a pass of the current-profile v2 full panel;
it is not evidence that the local analytic force is mismatched.

## Bidirectional loop and straight-segment event contract

The water reference geometry also has a separate explicit-scalar loop harness.
It uses the frozen seeded-internal direction and a Gram-Schmidt-orthogonalized
radial direction, amplitudes `(0.02, 0.02) angstrom`, four Simpson-compatible
subdivisions per edge, and independent forward and reverse evaluations. It
does not reuse the cold/warm implicit-root semantics of mutually responsive
Route-2 models because this candidate has no outer charge root.

The pure reducer recomputes both force-work integrals and requires:

- forward and reverse loop-work gates plus `|W_forward + W_reverse| <= 1e-10 eV`;
- energy, charge-source, and force replay at every identical forward/reverse
  coordinate and at each traversal closure;
- stationarity and reciprocity/metric/charge-gauge gates at every point;
- one model and continuum stratum over the loop;
- a conservative certificate for every straight segment. The certificate
  subtracts the maximum pair-relative atomic displacement from the endpoint
  neighbor-cutoff, point/source-shell, and sphere-tangency margins. It can
  therefore reject a segment even when both endpoints separately exceed the
  guard, preventing an endpoint-only "cross twice and return" loophole.

```bash
python tools/route2_release/run_aimnet2_geometry_mediated_water_loop.py \
  --checkpoint /absolute/path/to/aimnet2.pt \
  --device cpu \
  --output /absolute/path/outside/the/repository/aimnet2-gm-water-loop.json
```

This is a water-only local diagnostic and cannot by itself open OPT or MD.
The release reducer recomputes every bilinear reciprocity, apply/adjoint, and
charge-direction finite-difference error from the raw operands; it rejects a
boolean gate that disagrees with those measurements.

Two independent clean-process executions from commit `e0a347f5` have identical
raw forward/reverse records and measurement SHA. The forward and reverse
Simpson works are `0.00014855129592190874` and
`-0.0001485512959219093 eV`, with a sum of
`-5.692061405548898e-19 eV`; all same-coordinate replay errors are exactly
zero. The largest recomputed reciprocity and charge-direction finite-difference
errors are `3.1086244689504383e-15 eV` and
`4.294398170401337e-12 eV/e`. Every straight segment remains above the frozen
hard-neighbor, point/source-shell, and sphere-tangency guards. The retained raw
artifacts, exact commands, hashes, and claim boundary are in
[`evidence/aimnet2-geometry-mediated-water-loop-e0a347f5/`](evidence/aimnet2-geometry-mediated-water-loop-e0a347f5/README.md).
This closes only the frozen water-loop diagnostic, not a force or workflow
admission gate.

## Complete local weak-scalar HVP

For the smooth-harmonic arm, the sealed scalar now exposes the diagnostic
action

```text
H_F h = H_E h
      + (G_RR h + G_Rc J_c h)
      + J_c^T (G_cR h + G_cc J_c h)
      + D_R[J_c^T v][h],
v = G_c(R,c_A(R)).
```

The continuum `RR/Rc/cR/cc` action is produced by one Torch HVP of the exact
sealed `G(R,c)` graph. The final AIMNet2 term holds the center cotangent `v`
fixed; the geometry dependence of `v` is already owned by the continuum joint
HVP and its `J_c^T` pullback. This prevents both omission and double counting.
The pyddx arm has no equivalent sealed joint-scalar HVP interface and therefore
fails closed instead of substituting coordinate finite differences or an
independent second derivative.

The source-bound runner freezes the water-loop center, two orthonormal internal
directions, all three rigid translations, and central steps `(4e-4, 2e-4,
1e-4) angstrom`. Its pure reducer recomputes:

- the four-term component ledger;
- `J_q h` from displaced charges;
- `D_R[J_q^T v][h]` from displaced VJPs with the center `v` fixed;
- the complete HVP from finite differences of the total scalar gradient;
- `u^T H v = v^T H u`;
- the three translational charge-JVP and total-HVP zero modes;
- stationarity, reciprocity, gauge, topology identity, and conservative
  center-to-stencil event guards.

Two independent clean processes at commit `2b119022` reproduce scientific
measurement SHA-256
`01009ee3f5f829da5906cf051f4bc6b110d8c28300ff31467a6dc663943b88a6`.
The three smallest-step errors are `2.203106592738926e-8 e/angstrom` for the
charge JVP, `5.08769444399133e-7 eV/angstrom^2` for the fixed-cotangent charge
Hessian, and `3.391279860294818e-5 eV/angstrom^2` for the complete HVP. The
bilinear-symmetry error is `3.552713678800501e-15 eV/angstrom^2`; all three
translation HVP norms are exactly zero. Raw operands, exact commands, hashes,
and the admission boundary are retained in
[`evidence/aimnet2-geometry-mediated-hvp-water-2b119022/`](evidence/aimnet2-geometry-mediated-hvp-water-2b119022/README.md).

Analytical PCM Hessians and response corrections are established methodology
([Garcia-Rates et al., 2019](https://onlinelibrary.wiley.com/doi/abs/10.1002/jcc.25833)),
but that literature precedent does not admit this implementation. Recent
surface-point-charge PCM Hessian work also identifies discretization-induced
solvation-potential discontinuities as a stability problem and derives a
Gaussian-charge alternative
([Hashimoto and Nakai, 2026](https://www.sciencedirect.com/science/article/abs/pii/S0009261426002198)).
That result reinforces the event/source-regularity boundary here; it does not
provide an AIMNet2 Gaussian width or justify silently changing the point-charge
model.

## Stationary-water dense Hessian canary

The research runner reuses SciPy's MINPACK hybrid root solver in the exact
three-dimensional internal-coordinate space of nonlinear water. It does not
introduce a MAPLE optimizer or a public calculator adapter. Every consecutive
root trial must retain model/cavity topology and pass the same conservative
straight-segment event certificate used elsewhere on this branch.

At the converged geometry, the reducer reconstructs a `9 x 9` Hessian from the
complete weak-scalar HVP, checks every column against total-gradient central
differences at `(8e-4, 4e-4, 2e-4, 1e-4) angstrom`, and requires an explicit
second-order window before the smallest step approaches the numerical floor.
It then forms

```text
H_mw = M^-1/2 H_cart M^-1/2
```

before constructing the translation/rotation projector in the same
mass-weighted space. This ordering and the public `eV/angstrom^2` input unit are
explicit. The retained canary does not use the public MAPLE FREQ workflow as
evidence: at that source revision the historical driver projected before
mass-weighting, its normal-mode class had an ambiguous direct-call unit
boundary, and its main Dispatcher path instead supplied a private Hartree
view. The current public molecular FREQ driver has since migrated to the shared
ASE-unit `normal_modes.py` kernel. That infrastructure repair does not alter
this source-bound artifact and does not open any Route-2 capability flag.

Two clean processes at source commit `f79d5051` reproduce measurement SHA-256
`126327853eb0caadcf5e98b41992030789bbdf0f7ddf23cec9c4c31ed98b99a4`.
The Cartesian gradient norm is `5.375312708523219e-12 eV/angstrom`, Hessian
symmetry error is `1.2434497875801753e-14 eV/angstrom^2`, the largest
stationary rotation-HVP norm is `6.719144789488478e-9 eV/angstrom^2`, and the
smallest-step all-column FD Frobenius error is
`4.074765566486307e-5 eV/angstrom^2`. The three implementation frequencies are
`1638.2321451446662`, `2629.6268042951565`, and
`2831.3349242547506 cm^-1`. They are not physical solvent predictions or an
accuracy panel. Raw operands and hashes are retained in
[`evidence/aimnet2-geometry-mediated-frequency-water-f79d5051/`](evidence/aimnet2-geometry-mediated-frequency-water-f79d5051/README.md).

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
| harmonic-point distorted-geometry PES harness | exact v2 17-shard H/C/N/O contract implemented; water passes two clean processes, methanol reproducibly fails the frozen point/source-shell event guard, remaining 15 shards absent |
| harmonic sphere-pair tangency identity/margin | implemented and required by the v2 shard and loop contracts |
| harmonic water bidirectional loop/event harness | passes two clean source-bound processes on the frozen local water path; no task admission |
| source-bound float64 reconstruction | optional CPU research primitive; unchanged weights; upstream source hashes recorded |
| fixed-geometry electronic mutual polarization | absent by model interface |
| SMD-CDS/nonpolar and standard-state terms | excluded |
| public single-point E/F | disabled |
| solution-phase OPT/NEB/TS | disabled pending full PES/release gates |
| complete local weak-scalar HVP | implemented for the sealed harmonic-point research arm; one source-bound water canary passes; diagnostic only |
| stationary-water dense Hessian / local normal modes | one source-bound guarded water canary passes symmetry, all-column gradient FD, six rigid modes, and a three-mode mass-weighted subspace; diagnostic only |
| Tier H / FREQ/TS/IRC | disabled; broad event-free `C2`, multi-stationary-point, TS/IRC, physical-solvent, and workflow evidence absent |
| MD/NVE | absent |
| strict variational tier | not applicable/proven |

The original SMD CDS model is a separately parameterized solvent-accessible
surface term ([SMD paper](https://doi.org/10.1021/jp810292n)). It cannot be
appended here until one versioned nonpolar provider supplies the same-scalar
coordinate derivative and passes cavity/profile compatibility gates.

## Remaining admission gates

1. Recover the exact upstream release identity of the local checkpoint.
2. Treat the reproducible methanol point/source-shell guard failure as a block
   on the current-profile v2 full panel. The remaining fifteen shards may be
   run to map the diagnostic domain, but cannot erase that failure. Any smooth
   source/cavity replacement must be a newly derived, versioned profile with
   fresh evidence rather than a relaxed threshold. Also expand explicit
   cutoff/source-shell/sphere-tangency trial-step panels beyond the retained
   water loop. The legacy float32 arm remains a negative control, and the
   pyddx arm still has both derivative and laboratory-grid rotation failures.
3. Bind solvent dielectric, radii, grid, and solver choices to separately named
   physical-configuration profiles instead of the current unbound diagnostic.
4. Add a same-scalar nonpolar provider before making total solvation-free-energy
   or multi-solvent claims.
5. Add force-domain, optimization, broad Hessian/FREQ, and NVE evidence before
   any corresponding MAPLE workflow is enabled.
6. For Hessian/FREQ/TS/IRC, extend the implemented complete local HVP from the
   water canaries to a broad event-free `C2` domain; add mixed-block
   adjointness, multi-molecule/stationary-point force-FD/HVP closure,
   conditioning, and FREQ/TS/IRC path panels. The local water Hessian,
   translational/rotational modes, and mass-weighted subspace now pass, but the
   current methanol event-guard failure remains a profile-level block.

No fitting, radius tuning, response tempering, calibration, or experimental
label use is part of this candidate.
