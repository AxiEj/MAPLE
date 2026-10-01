# CHA continuum: unregistered analytic foundation

## Status and model identity

This is an **internal research implementation**, not an enabled MAPLE solvent
provider. It does not enable CHA OPT, FREQ, transition-state searches, or MD.
The existing AmberTools `chagb-bondi-pbsa-inp2` single-point endpoint is unchanged.

The separate `chagb-r6-pbsa-continuum-v1` input identity evaluates a continuous
surface-integral model. It is not a pointwise replacement for the historical
discrete Amber grid/sampling result and does not inherit its accuracy scores.
No experimental labels or fitting enter this implementation.

The owned scalar is:

```text
G = G_CHA(R6_SES(R), q_fixed) + G_SAV(R) + G_dispersion(R)
```

Coordinate derivatives are obtained from its complete live Torch graph. Born
radii are coordinate-dependent; they are not imported and frozen during AD.
Finite differences occur only in the independent validation runner.

## Module boundaries

| Module | Responsibility |
|---|---|
| `continuum_chagb_inputs.py` | Externally pinned topology, atom order, fixed charges and parameter identity |
| `torch_continuum_chagb_domain.py` | Conservative three-site R6 **geometric point** scope and typed rejections |
| `torch_continuum_ses_geometry.py` | Restored local rolling-probe geometry primitives |
| `torch_continuum_r6_patches.py` | Restored contact/toroidal/probe surface integrals; not a general SES provider |
| `torch_continuum_chagb.py` | Complete scalar composition, CHA branch diagnostics and quadrature identity |
| `torch_continuum_sav.py` | Independent SAV volume/cavity functional |
| `torch_continuum_dispersion.py` | Distinct sigma-split dispersion functional |
| `benchmarks/cha_continuum_reference.py` | Independent NumPy/SciPy reference, not a production fallback |

`CalcABC`, public provider selection, parser options and optimizers are not
modified by this foundation. There is no new public `#solv(...)` selector.

### Radius identity

These arrays are not interchangeable:

- **R6 SES and CHA polar size/algebra:** CHA-remapped native `radi`, including
  `Rs`. For the pinned water input this is approximately `[1.88, 1.04, 1.04] Å`.
- **Cavity:** GAFF2 LJ `rmin + 1.3 Å`.
- **Dispersion boundary:** GAFF2 LJ `rmin + 0.557 Å`, with the frozen separate
  TIP3P-oxygen LJ mixing and sigma split.
- **Bondi:** retained as input provenance, not consumed by this scalar.

A source-bound regression distinguishes water's Bondi `[1.5, 1.2, 1.2] Å`
from its CHA-mapped vector. Using the former in the new R6 path is incorrect.

## Supported internal geometry and rejected cases

The point-domain certificate requires exactly three distinct sites, regular
expanded-sphere intersections, no exposed triple-probe centers, only complete
or fully occluded pair circles, non-spindle active tori, and a conservative
separation bound between active rolling-probe circles. Ambiguous or unresolved
ownership is rejected rather than integrated as a partial molecular surface.

The exterior-connectivity proof is deliberately limited: at most three ball
centers lie in an affine plane. From any point outside the balls, a plane-normal
ray directed away from that plane increases every center distance and reaches
infinity without entering a ball. This proves probe-center exterior
connectivity here; it does not prove arbitrary-molecule SES ownership.

The certificate describes **R6 geometry at one point only**. CHA sign/size
branch diagnostics are separate. Neither object certifies a neighborhood,
optimization segment, or any trajectory. Methanol remains unsupported: its
source-bound six-site geometry includes exposed concave-probe centers about
`1.1381980149 Å` apart, below `2p = 1.76 Å`, requiring additional ownership and
overlap treatment.

## Hard CHA branches

The original CHA algebra contains a geometry-dependent weighted-charge sign.
Frozen charges do not make that sign coordinate-independent. At the preserved
water center, pointwise AD exists, but a prescribed small-displacement stencil
crosses a sign boundary and sees an energy jump. That is recorded as an event,
not repaired by smoothing or by freezing the sign.

Typed branch failures preserve raw margins. If the frozen polar helper rejects
an AD request before returning diagnostics, an error-only evaluation of the
same algebra on identical detached values can recover those diagnostics. It
then raises; it never returns a detached energy or force fallback.

## Validation and resource contracts

- Only explicit float64 coordinate tensors are accepted. The caller supplies
  the immutable topology's external content pin.
- Production refinement studies retain orders `24/48/64`. The internal API
  rejects orders above `128` before dense quadrature construction. Lower
  diagnostic orders are not accuracy qualifications.
- The existing dispersion AD six-site limit remains unchanged; it is not
  expanded by this three-site work.
- The independent R6 reference uses separate cap-axis geometry and SciPy
  integration. Near-boundary complementary caps are integrated directly to
  avoid subtracting large nearly equal fluxes.
- Reference refinement retains `64/96/128`. R6 quadrature-error estimates have
  units `Å^-3`; analytic surface-normal closure has units `Å^2`. Neither can
  be compared directly to an energy-error threshold.
- Reference estimates are explicitly non-rigorous. The runner distinguishes
  observed numerical agreement from unresolved uncertainty and from public
  admission; it never promotes a numerical estimate into a certified bound.
- The five preregistered O-H scales are `0.96, 0.98, 1.00, 1.02, 1.04`, with
  unchanged topology/charges and all failures/events retained. The original
  `1.00` event denominator is not replaced by a successful geometry.

The preparation tool uses existing fixed charges and verified GAFF2 types with
topology-only `parmchk2`/`tleap`. It does not run Antechamber, SQM, recharge, or
model downloads. Candidate evaluation consumes the resulting arrays and does
not require Amber or OpenMM at runtime.

## Before real OPT can be enabled

A subsequent reviewed design must establish whole-segment event certification,
public scalar-versus-force admission, rollback/cache semantics and bounded
step acceptance. Endpoints with matching signs are not proof that a segment
avoided an event. Passing this point study does not complete those obligations.

Current source-bound evidence is retained in the owning worktree's session
artifacts under `route1-cha-analytic-v1-20260930`; old exact-SP evidence remains
unaltered. Use the final report and hashes, not test counts alone, to assess
which claims were actually established.
