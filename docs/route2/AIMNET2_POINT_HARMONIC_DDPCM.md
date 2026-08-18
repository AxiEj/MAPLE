# AIMNet2 point-charge smooth-harmonic finite-dielectric ddPCM diagnostic

## Scientific boundary

This disabled branch extends the weak geometry-mediated AIMNet2 scalar with a
finite-dielectric **PCM** response.  It does not feed solvent state into
AIMNet2, change the AIMNet2 graph, or create fixed-geometry electronic mutual
polarization:

```text
c_A(R) = [q_AIMNet2(R), 0, 0, 0]
F(R) = E_AIMNet2(R) + G_ddPCM(R, c_A(R))
```

The finite dielectric is not implemented by multiplying the conductor energy
by one empirical number.  The PCM boundary equation contains a nonlocal
double-layer operator whose response is harmonic-mode dependent.  MAPLE uses
the integral equations documented by ddX and derived for ddPCM:

```text
R_epsilon Phi_epsilon = R_infinity Phi
S sigma = -Phi_epsilon

R_epsilon = 2*pi*(epsilon_s + 1)/(epsilon_s - 1) I - D
R_infinity = 2*pi I - D
```

See the [ddX theory manual](https://ddsolvation.github.io/ddX/md_docs_theory.html),
the original systematic ddPCM discretization
([Stamm et al., JCP 144, 054101 (2016)](https://doi.org/10.1063/1.4940136)),
and the analytic-force derivation
([Gatto, Lipparini, Stamm, JCP 147, 224108 (2017)](https://doi.org/10.1063/1.5008329)).

## Fixed-dimensional weighted Galerkin equations

Let `E(R)` embed the retained smooth exposed basis into the complete harmonic
product bandwidth, `K(R)` be the Coulomb single-layer matrix, `D(R)` the
Laplace double-layer principal-value matrix, and `V(R)` the analytic point
source boundary map.  Define

```text
M = E.T E
A = E.T K E
D_w = E.T D E
S = E.T V
b = S c

M f = b
[2*pi*g(epsilon_s) M - D_w] phi_epsilon
    = [2*pi M - D_w] f
A x = M phi_epsilon

G_ddPCM(R,c) = -1/2 b.T x
g(epsilon_s) = (epsilon_s + 1)/(epsilon_s - 1)
```

`f` is the weighted-basis projection of the vacuum boundary potential and
`x` is the sign-reversed apparent surface-charge coefficient vector.  The
reported energy is the charging-work scalar.  All source, coordinate, mixed,
and Hessian-vector derivatives are generated from this one Torch graph.

For one fully exposed sphere, the double-layer principal-value spectrum is

```text
D_lm,lm = -2*pi/(2*l+1),
```

so the finite-dielectric transfer factor is

```text
[1 + 1/(2*l+1)] / [g(epsilon_s) + 1/(2*l+1)].
```

Its dependence on `l` is an explicit regression against replacing ddPCM by a
uniform COSMO energy scale.  As `epsilon_s -> infinity`, the two `R` operators
coincide and the existing conductor scalar is recovered.  As
`epsilon_s -> 1+`, the solvent response vanishes.

## Stationarity, adjoints, and reciprocity

The dense research implementation reports the absolute residual, the true
RHS-relative residual, and a MAPLE `max(||rhs||,1)` scaled residual for all
three primal solves:

```text
M f - b = 0
A x - M phi_epsilon = 0
R_epsilon phi_epsilon - R_infinity f = 0
```

It also solves and audits the corresponding transpose chain.  This is needed
because the double-layer operator is not generally symmetric.  In particular,
the finite-dimensional primal source response

```text
P_primal = -S.T A^-1 M R_epsilon^-1 R_infinity M^-1 S
```

is not assumed to be self-adjoint.  MAPLE therefore does **not** expose
`P_primal c` as the provider field.  The KKT/transpose chain gives the actual
energy cotangent

```text
v_E = dG_ddPCM/dc = sym(P_primal) c,
```

which is the response sent into the AIMNet2 charge-position VJP.  This is the
dense analogue of ddX solving separate transpose systems for analytic energy
derivatives; it is not an ad-hoc replacement of the primal solve.

The same three residual measures are reported for every transpose solve, with
potential-equation residuals labelled `eV/e` and charge-adjoint residuals
labelled `e`.  The stationarity artifact keeps the response distinction
observable.  It reports the
primal-map asymmetry and primal/cotangent mismatch as diagnostic evidence,
then separately gates:

- self-adjointness of the energy-cotangent operator in the registered source
  metric, including the fixed-total-charge monopole tangent space;
- equality of the KKT cotangent and the sealed-scalar autograd derivative;
- `v_E = sym(P_primal)c` and `G_ddPCM = 1/2 <c,v_E>`;
- every primal and transpose residual.

Random bilinear reciprocity and charge-direction finite differences in the
geometry-mediated wrapper likewise test `v_E`, never the nonsymmetric primal
map.  Only the energy-conjugate response is called the discrete reaction
potential in MAPLE.

The discrete equation is a Galerkin analogue of the ddPCM primal/adjoint
system rather than a claim that MAPLE reproduces ddX's domain-decomposition
matrix entry by entry.  ddX v0.8.0 source commit
`4d79e3d9caeae5e602683572a71cb550414f9b09` is the pinned upstream equation
reference (`src/ddx_operators.f90`, `src/ddx_pcm.f90`).

## Reuse and regularity

The implementation reuses the existing:

- smooth exposure and rectangular harmonic embedding;
- analytic single-layer and point-source kernels;
- real-Wigner SO(3) representation machinery;
- scalar-first JVP/VJP/HVP implementation;
- point/source-shell and sphere-tangency topology guards.

Only the dimensionless double-layer kernel and the finite-dielectric solve are
new.  No laboratory-fixed cavity grid, graph modification, CDS term, or new
electronic response variable is introduced.

The double-layer cross blocks use a pair-axis, intersection-split
Gauss-Legendre contraction with an exact retained-band azimuthal projection.
Its order is part of configuration/provenance, and the unit suite requires an
energy/drive/coordinate-gradient refinement plateau rather than treating one
chosen order as self-validating.

The registered profile/configuration contract is deliberately named a
**parameterized diagnostic**.  `epsilon_s` is included in the immutable runtime
configuration SHA, but this generic profile can never itself become an
admission identity.  Any future solvent-specific admission must use a new
profile that binds the exact dielectric, radii, cavity rule, bandlimits,
quadrature orders, implementation tree, and evidence artifacts.

The fixed coefficient dimension does not prove global smoothness.  Point
sources on a sphere, sphere tangencies, rank loss, and AIMNet2 hard-neighbour
events remain fail-closed.  The provider and registry entries remain disabled;
E/F/H/V/M and OPT/FREQ/TS/IRC/MD admission require new real-checkpoint evidence
and are not implied by unit tests.

From a clean committed tree, the source-bound real-checkpoint diagnostic is
generated outside the checkout with:

```bash
python tools/route2_release/run_aimnet2_geometry_mediated_canary.py \
  --checkpoint "$MAPLE_ROUTE2_AIMNET2_CHECKPOINT" \
  --aimnet-runtime reconstructed-python-float64 \
  --continuum harmonic-ddpcm \
  --device cpu \
  --output /absolute/path/outside/the/repository/aimnet2-harmonic-ddpcm-water.json
```

The runner rejects a dirty tree and leaves every capability false even when
its local derivative, reciprocity, stationarity, rotation, and event checks
pass.  A retained artifact must additionally bind both executions, the exact
source tree, checkpoint bytes, raw operands, and an independently recomputed
summary before it can support any later admission discussion.

## Retained real-checkpoint water result

Two independent clean-tree executions at commit `4dca73d7` now reproduce
scientific measurement SHA256
`7ec7732e79aacfd1d9502d7c19de8fd49b0b1975b764ef1cf59a413eaaf5cbbf`.
The reconstructed float64 AIMNet2 graph keeps the checkpoint weights unchanged
and supplies one field-independent NQE charge source per geometry.  At the
registered water dielectric and SMD Coulomb radii:

- the adaptive directional analytic/numerical gradient error is
  `1.8257224045914455e-6 eV/A`;
- the terminal complete-Cartesian maximum and RMS errors are
  `2.826476747208595e-5` and `1.5826934760464932e-5 eV/A`;
- three rigid rotations have zero recorded energy drift and maximum relative
  force-covariance error `5.5951040041648464e-11`;
- every recorded primal/adjoint residual, condition, half-coupling,
  reciprocity, charge-gauge, topology, and event-clearance gate passes.

The raw operands, both clean-process records, exact source binding, and
independent reducers are retained under
[`evidence/aimnet2-geometry-mediated-harmonic-ddpcm-water-4dca73d7/`](evidence/aimnet2-geometry-mediated-harmonic-ddpcm-water-4dca73d7/README.md).
This closes a real-stack local force diagnostic for the finite-dielectric
implementation.  It does not open a capability.  The subsequent distinct
water-bound candidate locks `epsilon=78.355`, SMD Coulomb radii, the `1/2`
surface/exposure orders, both order-32 radial quadratures, and the source-bound
float64 `aimnet2-polarizable-v1` model profile.  It still uses exactly one NQE
charge evaluation per geometry, supplies no continuum field to AIMNet2, and
runs no electronic SCF. Two clean processes under that exact identity reproduce
scientific SHA256
`29ad72cd84a137b6bc7b9079ce6979000e7dab58983e2c41d6d2e3ce82fa08e5`
and are retained under
[`evidence/aimnet2-frozen-charge-water-harmonic-ddpcm-0c19ede4/`](evidence/aimnet2-frozen-charge-water-harmonic-ddpcm-0c19ede4/README.md).
Broader distorted-PES evidence, a compatible nonpolar scalar, and task-level
validation remain required.

## Literature-to-implementation boundary

The AIMNet2 paper establishes geometry-dependent partial charges produced by
Neural Charge Equilibration and a model trained for energies and forces
([Anstine, Zubatyuk, and Isayev, 2025](https://doi.org/10.1039/D4SC08572H)).
It does not establish a continuum-field input for the checkpoint used here.
The one-shot frozen-charge design is therefore a property of MAPLE's bound
model interface and local checkpoint contract, not a claim that the published
AIMNet2 architecture cannot ever be extended.

The ddX theory manual derives parameter derivatives through a transpose
adjoint solve, and published ddPCM implementations compute solvation forces
([ddX theory](https://ddsolvation.github.io/ddX/md_docs_theory.html);
[Nottoli et al., 2022](https://doi.org/10.1063/5.0104536)). This establishes
that analytic ddPCM force methodology exists. It does not validate MAPLE's
particular smooth weighted-harmonic discretization, its point-source event
domain, or its AIMNet2 charge-chain composition; those are tested separately
by the retained source-bound artifacts.

SMD defines total solvation free energy as bulk electrostatics plus a
cavity-dispersion-solvent-structure contribution
([Marenich, Cramer, and Truhlar, 2009](https://doi.org/10.1021/jp810292n)).
Consequently the exact differentiable candidate in this document, for which
`G_np=0`, is not a complete SMD solvation free energy and cannot inherit SMD's
published accuracy or workflow claims.

## Water-bound distorted-PES result

The exact water-bound candidate has now been run twice over the frozen
17-molecule H/C/N/O panel: 51 geometries, 153 directions, and 459 central-
difference samples per process. The source-bound float64 runtime keeps the
ordinary AIMNet2 forward as a per-geometry parity oracle while evaluating the
same upstream DFT-D3 term through its smooth `hessian=True` graph for
first/second coordinate derivatives. Across all 51 centers, ordinary versus
decomposed energy and intrinsic-gradient discrepancies are bounded by
`1.327271093e-7 eV` and `3.81574774e-8 eV/A`, respectively; checkpoint weights
and charge outputs are unchanged.

All 17 molecules now pass every frozen three-step convergence requirement.
Twelve also pass the topology/event gates: water, ethanol, acetone,
acetonitrile, benzene, trans-butane, formic acid, acetaldehyde, acetamide,
pyridine, nitromethane, and hydrogen peroxide. Methanol, methane, dimethyl
ether, and acetic acid fail the preregistered `0.02 A` point/source-shell
margin; ethylamine fails the independent sphere-tangency margin. Methane also
has one terminal numerical sample failure inside its already failed event
domain. No threshold, radius, molecule list, or public capability was changed.
The two aggregates reproduce measurement SHA256
`a219082dbe88097923fd18b39fff83a613f1a7a02d57d7e126df2e549708c42b`
and are retained under
[`evidence/aimnet2-frozen-charge-water-harmonic-ddpcm-pes-panel-smoothed-97efb08e/`](evidence/aimnet2-frozen-charge-water-harmonic-ddpcm-pes-panel-smoothed-97efb08e/README.md).

This is a materially stronger force-domain result, but the five topology
failures keep the complete current-profile panel negative. The `0.02 A` guard
must not be lowered to convert those failures into passes.

## Total harmonic-ddPCM plus SMD-CDS PES result

The same frozen 17-molecule panel has now been repeated with the separately
differentiable PySCF 2.13.1 SMD-CDS scalar included in the exact total energy.
Primary and replay executions at commit `20f9c65c` reproduce aggregate
measurement SHA256
`61512993a63cc7bf04d4a3d15a288c806eabc96be104e33373b802f88f62322e`
and identical independently reduced summaries. The evidence is retained under
[`evidence/aimnet2-frozen-charge-water-harmonic-ddpcm-smdcds-pes-panel-20f9c65c/`](evidence/aimnet2-frozen-charge-water-harmonic-ddpcm-smdcds-pes-panel-20f9c65c/README.md).

The total scalar again passes only 12/17 molecules. Methanol, methane,
dimethyl ether, and acetic acid fail the unchanged continuum event margin;
ethylamine fails the unchanged sphere-tangency margin. The maximum directional
absolute error among the 12 accepted diagnostic strata is
`5.552903772709783e-05 eV/A`, but the complete-panel maximum is
`0.23707138702659591 eV/A` inside already event-rejected methanol. Adding the
nonpolar term therefore closes the former `G_np=0` composition gap but does
not repair the topology-domain failure. Public E/F/OPT and every H/V/M,
FREQ/TS/IRC, and MD capability remain false.

## Accuracy pilot

The preregistered ten-record MNSol-v2012 frozen-source pilot was replayed twice
from the current source tree. With fixed AIMNet2 charges, full-resolution
`lmax=15`/1202-point pyddx electrostatics, and SMD-CDS, the ddPCM arm reports
MAE/RMSE/max/MSE of `1.1430625`, `1.3515630`, `2.2322398`, and
`+1.0333127 kcal/mol`; the separately named scaled-ddCOSMO arm reports
`1.0120976`, `1.2743164`, `2.1932768`, and `+0.8300035 kcal/mol`. On the same
ten records, the tracked frozen-source MACE-POLAR arm reports ddPCM MAE
`0.8635007 kcal/mol` and ddCOSMO MAE `0.9154388 kcal/mol`.

These values are descriptive only: ten records do not establish a source
ranking or solvent generalization. More importantly, this accuracy pilot uses
SMD-CDS and full-resolution pyddx, so it is not the same scalar or cavity
numerics as the current differentiable `harmonic-ddpcm-water` force candidate.
The aggregate-only evidence is retained under
[`evidence/aimnet2-frozen-charge-mnsol-pilot-replay-b758aede/`](evidence/aimnet2-frozen-charge-mnsol-pilot-replay-b758aede/README.md).

## Complete 653-record MNSol matrix

The pilot has now been superseded by the complete neutral absolute MNSol-v2012
matrix inside the frozen Route-2 domain: 653/653 records, 395 unique
geometries, 505 development records, and 148 confirmation records. No
experimental value or model output selected, fitted, calibrated, or tempered
the method. The public aggregate is retained as
[`route2-mnsol-aimnet2-full-frozen-charge-matrix-v1.json`](../implicit-solvation/benchmarks/route2-mnsol-aimnet2-full-frozen-charge-matrix-v1.json)
with measurement SHA256
`10b68d1fbbbc9eede7feefcb64daa2cb7be1bd0ab4a00186586eaf8965361039`.

Across all 653 records, full-resolution `lmax=15`/1202-point pyddx plus
PySCF SMD-CDS gives:

- ddPCM MAE/RMSE/median/max/MSE of `2.5443`, `3.3020`, `1.9718`,
  `10.2917`, and `+2.4983 kcal/mol`;
- separately named scaled ddCOSMO MAE/RMSE/median/max/MSE of `2.4393`,
  `3.2244`, `1.8466`, `10.1798`, and `+2.3377 kcal/mol`;
- on the untouched 102-record confirmation subset, ddPCM MAE/RMSE of
  `2.9075`/`3.6416 kcal/mol` and ddCOSMO MAE/RMSE of
  `2.7733`/`3.5580 kcal/mol`;
- on the 387 water records, ddPCM MAE/MSE of
  `3.3860`/`+3.3836 kcal/mol` and ddCOSMO MAE/MSE of
  `3.3430`/`+3.3397 kcal/mol`.

The positive signed water errors expose systematic under-solvation by this
one-shot frozen gas-charge protocol. They must not be hidden by calibrating the
charges, radii, dielectric, CDS term, or record selection. This matrix measures
the high-resolution pyddx energy protocol only. It does not validate the
lower-order smooth harmonic force scalar and does not open E/F/H/V/M, OPT,
FREQ/TS/IRC, or MD support.

## Water daily-task diagnostics

Three finite-dielectric, exact-scalar water workflows now pass in two clean
processes each at source commit `76d4d097`:

- bidirectional closed-loop work and conservative straight-segment event
  certificates, measurement SHA256
  `4c70022f570e2736e4b2f16312009971058461cf2a78776e3ba45e01738cd26f`;
- the complete weak-scalar HVP ledger, central response checks, bilinear
  symmetry, and all three translational zero modes, measurement SHA256
  `60159b0b31ec0ae1ae308dd03175121d9c467078882965cbf9a72670d4d39dfb`;
- an internal-coordinate stationary solve, all nine Cartesian HVP columns,
  four-step total-gradient finite differences, Hessian symmetry, six rigid
  modes, and the mass-weighted three-mode vibrational subspace, measurement
  SHA256
  `2cb5aceee9d22c84bfe458bf9e9967d5c5cb61c9536f90a450246d97730c258c`.

The finite-dielectric stationary search uses a componentwise open-bound
`tanh` parameterization of the same frozen internal-coordinate bounds and
MINPACK hybrid root equations. It changes neither the scalar nor any gate; it
prevents a solver trial from leaving the declared domain. The final Cartesian
maximum gradient is `6.744916582722416e-15 eV/A`, the dense-Hessian symmetry
error is `1.126172990825879e-14 eV/A^2`, and the smallest-step FD Frobenius
error is `3.524831549078756e-5 eV/A^2`.

The full primary/replay records and independent reducers are retained under
[`evidence/aimnet2-frozen-charge-water-ddpcm-daily-tasks-76d4d097/`](evidence/aimnet2-frozen-charge-water-ddpcm-daily-tasks-76d4d097/README.md).
They establish local implementation readiness for one event-safe water
stratum. They do not open public F/OPT/FREQ/MD support: the broad PES panel is
still negative, the total SMD-CDS panel remains 12/17, and the reported
frequencies are not physical solvent predictions.
