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
