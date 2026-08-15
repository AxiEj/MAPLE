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

- `models/aimnet2.py` delegates intrinsic energy/forces and the charge-position
  VJP to the existing `AIMNet2Calculator` methods;
- `continuum/atomic_l1_pyddx.py` delegates field, source JVP/VJP, and coordinate
  VJP to the existing `PyDDXPCMReactionFieldLinearMap`;
- `coupling/geometry_mediated.py` owns the half-coupling ledger and composes the
  three gradient terms.

No model inference, ddPCM equation, or chain-rule term is reimplemented in a
second monolithic objective. Synthetic full-Cartesian finite differences lock
the complete scalar and include negative tests for an antisymmetric reaction
operator and a charge projection with nonzero gauge VJP.

The opt-in real-checkpoint test uses the SHA256-bound AIMNet2 asset and
`pyddx==0.8.0`. It verifies the reciprocity/metric/gauge audit, then deliberately
applies the stricter three-step coordinate and three-rotation fail-closed
harness. The current water diagnostic keeps one cavity active set over the
small coordinate stencil, but the float32 energy differences do not form an
admissible convergence window; the laboratory-frame Lebedev active set also
changes under all three frozen rotations. These are negative admission results,
not a tuned threshold pass:

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
  --continuum ddpcm \
  --device cpu \
  --output /absolute/path/outside/the/repository/aimnet2-gm-water.json
```

The runner records the checkpoint, loaded committed sources, registered metric,
model hard-neighbor graph, exact exposed sphere/Lebedev candidate set,
three-step directional measurements, rotation measurements, and the fact that
this pyddx provider exposes a requested solver tolerance but not a measured
post-solve algebraic residual. It always leaves all capabilities false.

## Structurally rotational harmonic branch

The separately registered smooth-harmonic diagnostic replaces the laboratory
Lebedev surface with MAPLE's existing fixed-dimensional weighted harmonic
Galerkin assembly and an analytic point-monopole boundary map. It keeps the
same unmodified AIMNet2 charge function and supplies structural `SO(3)`
intertwiners rather than tuning a larger angular grid.

The real water canary now preserves the continuum stratum under all frozen
rotations and passes the complete rigid-rotation gate. Its full three-step
coordinate directional gate remains negative because the float32 AIMNet2
energy differences do not show the required convergence window. This is an
isolation result, not force admission. The branch is also a conductor
reference without finite-dielectric solvent parameterization, so it must not
be presented as an admitted water ddPCM replacement. See
[`AIMNET2_POINT_HARMONIC.md`](AIMNET2_POINT_HARMONIC.md) for the addition
theorem, parent functional, topology events, executed tests, and claim
boundary.

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
| harmonic-point three-step full-energy direction | fails current float32 AIMNet2 water canary |
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
2. Continue the structurally controlled harmonic branch through
   distorted-geometry, full Cartesian, closed-loop, cutoff/source-shell/sphere-
   tangency, and broader chemistry panels. Its rotation gate is now positive,
   but its real full-energy directional gate remains negative. The pyddx arm
   still has both directional and laboratory-grid rotation failures.
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
