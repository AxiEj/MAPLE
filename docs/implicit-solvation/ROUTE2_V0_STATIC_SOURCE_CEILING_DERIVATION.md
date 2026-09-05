# Route-2 V0: deriving the static-source ceiling from a ΔG_solv tolerance

**Date:** 2026-09-04
**Status:** derivation and re-adjudication of an existing frozen result.
Adds no result, admits no source, changes no frozen artifact.
**Executable:** [`benchmarks/derive_static_source_ceiling.py`](benchmarks/derive_static_source_ceiling.py)
(standard library only, reads frozen artifacts, writes nothing).

---

## 1. The problem with the current gate

The zero-field permanent-source gate in
[`route2-v0-gfn2-molden-static-mep-acetone-v2.json`](benchmarks/route2-v0-gfn2-molden-static-mep-acetone-v2.json)
uses ceilings `0.20` (relative Frobenius) and `0.30` (relative max-abs). Those
two numbers are carried over unchanged from the **response**-source gate, where
they were applied to `∂V/∂E`. Nothing in either preregistration derives them
from a solvation-energy tolerance.

`V` and `∂V/∂E` are different quantities that enter the polarization scalar
differently. Reusing one ceiling for the other is not conservative in either
direction — it is simply undetermined until the propagation is done.

This document does the propagation.

## 2. Derivation

At fixed geometry the V0-RK common scalar is

$$
G(x;v_0)=\tfrac12x^{\mathsf T}C^{+}x
+\tfrac12(v_0+Bx)^{\mathsf T}Q(v_0+Bx)+x^{\mathsf T}f,
\qquad Nx=0 .
$$

Let $G^\*(v_0)=\min_x G(x;v_0)$ with stationary point $x^\*$. By the envelope
theorem the leading sensitivity of the **minimum value** to the permanent
source is the partial derivative evaluated at $x^\*$:

$$
\frac{\mathrm dG^\*}{\mathrm dv_0}=Q\,(v_0+Bx^\*)=q^\* ,
$$

the converged induced surface charge. Hence for a source error $\delta v_0$,

$$
\boxed{\;\delta G^\*=\langle q^\*,\delta v_0\rangle+O(\lVert\delta v_0\rVert^2)\;}
\tag{1}
$$

Cauchy–Schwarz bounds this by the two norms the gate actually reports:

$$
|\delta G^\*|\le\lVert q^\*\rVert\,\lVert\delta v_0\rVert
=\varepsilon_F\,\lVert q^\*\rVert\,\lVert v_0\rVert ,
\qquad \varepsilon_F=\frac{\lVert\delta v_0\rVert}{\lVert v_0\rVert}.
\tag{2}
$$

With $\Delta G_{\rm pol}=\tfrac12\langle v^\*,q^\*\rangle$ and the alignment
factor

$$
\kappa=\frac{\lVert v^\*\rVert\lVert q^\*\rVert}{|\langle v^\*,q^\*\rangle|}\ge1
\qquad(\kappa=1\iff q^\*\parallel v^\*),
$$

we get $\lVert q^\*\rVert\lVert v^\*\rVert=2\kappa|\Delta G_{\rm pol}|$ and

$$
|\delta G^\*|\le 2\kappa\rho\,\varepsilon_F\,|\Delta G_{\rm pol}| ,
\qquad \rho=\frac{\lVert v_0\rVert}{\lVert v^\*\rVert},
\tag{3}
$$

so for a declared tolerance $\tau$ on the solvation energy,

$$
\boxed{\;\varepsilon_F^{\max}=\frac{\tau}{2\kappa\rho\,|\Delta G_{\rm pol}|}\;}
\tag{4}
$$

$\kappa\ge1$ always, and $\rho\ge1$ whenever the induced surface potential
screens the permanent one — the ordinary case for a neutral solute. **Setting
$\kappa=\rho=1$ therefore gives the most permissive ceiling this derivation can
license**, and a candidate failing there fails for every alignment, with no
knowledge of $q^\*$ required.

## 3. Inputs, all from frozen artifacts

The 516 gate points are geometrically a van der Waals cavity surface, and they
are the same 516 points the PCM operator uses
(`fixed-cavity-gto-galerkin-v1`, `surface_point_count: 516` in
[`route2-gto-pcm-energy-projection-acetone-v1.json`](benchmarks/route2-gto-pcm-energy-projection-acetone-v1.json)).
Grouping each point by its nearest atom gives exact per-element radii —
C at $1.8500$ Å with zero spread, H minimum $1.2000$ Å, O minimum $1.5200$ Å.
So the gate's norm is measured on exactly the surface where the energy integral
lives; there is no representation gap to correct for.

The zero-field static MEP is recovered from the frozen finite-field records
without new QM: $V(+E)+V(-E)=2V(0)+O(E^2)$. The six independent
(direction, step) reconstructions agree to $2.3\times10^{-6}$ hartree/e. Adding
the exact nuclear potential gives

$$
\lVert v_0\rVert = 0.752896\ \text{hartree/e},
$$

from a near-cancellation of $\lVert V_{\rm nuc}\rVert=153.3304$ against
$\lVert V_{\rm el}\rVert=153.1618$. That cancellation matters: a small relative
error in either component is a large relative error in the total, which is why
a source can match a molecular dipole well (GFN2: $0.1040$) and still miss the
surface potential badly.

Acetone reference: $\Delta G_{\rm pol}=-6.6433389624$ kcal/mol.

## 4. Result — the inherited ceiling is too loose

At $\tau=1.5$ kcal/mol, i.e. spending the route's **entire** all-record budget
on this single error term:

| $\kappa$ ($\rho=1$) | $\varepsilon_F^{\max}$ |
| --- | --- |
| **1** | **0.1129** ← most permissive licensable |
| 2 | 0.0564 |
| 5 | 0.0226 |
| 10 | 0.0113 |

* inherited ceiling `0.2000` versus most permissive derived `0.1129`
  — **1.77× too loose**;
* a source sitting exactly at the inherited ceiling could carry
  **2.657 kcal/mol** of first-order polarization error, which by itself
  exceeds the 1.5 kcal/mol all-record gate.

This is the opposite of what one might assume from a near-miss rejection. The
gate was not too strict; **it was admitting sources that cannot meet the
route's own accuracy target.**

A realistic error allocation makes it much tighter still: at $\tau=0.5$ the
$\kappa=1$ ceiling is $0.0376$, and $\kappa>1$ shrinks it proportionally.

The ceiling is also **per-molecule**: it scales as $1/|\Delta G_{\rm pol}|$.
Acetone is moderately polar, so a panel criterion must take the worst-case
$|\Delta G_{\rm pol}|$ over the panel, not acetone's value.

## 5. Re-adjudication of the GFN2 MOLDEN candidate

| quantity | value |
| --- | --- |
| observed $\varepsilon_F$ | `0.2199987983` |
| worst-case $\Delta G_{\rm pol}$ error ($\kappa=\rho=1$) | **2.923 kcal/mol** |
| versus derived ceiling `0.1129` | **REJECT**, by 1.95× |

**The original verdict `reject-gfn2-molden-permanent-source` stands.** It is now
supported by a derived criterion rather than a borrowed number, and the
rejection is robust for every $\kappa\ge1,\rho\ge1$ — it does not depend on the
unknown alignment of the error with $q^\*$.

No threshold was weakened and no source was admitted by this document.

## 6. What the protocol should freeze instead

Equation (1) says the deciding quantity is a **single inner product**,
$\langle q^\*,\delta v_0\rangle$ — one dot product over 516 points. The frozen
artifact preserves only two scalar norms. Those norms *bound* the inner product
across a factor-of-$\kappa$ range, but they cannot evaluate it, so the gate as
recorded cannot decide the question it was built to decide; it can only reject
candidates that fail the loosest bound.

For every future $v_0$ candidate, register and store:

1. $\langle q^\*,\delta v_0\rangle$ — the first-order energy shift itself;
2. $\lVert q^\*\rVert$ and $\kappa$ — so the bound and the realised value can be
   compared;
3. the declared $\tau$ and the resulting per-record $\varepsilon_F^{\max}$ from
   equation (4), fixed **before** execution.

`derive_static_source_ceiling.py --induced-charge q_star.json` computes (1) and
(2) once the continuum solve supplies $q^\*$. It reports $\kappa=1.000$ exactly
for a $q^\*$ parallel to $v_0$, as the algebra requires.

## 7. Consequence for the V0-RK pipeline

[`route2_v0_full_response_kkt.py`](../../maple/function/calculator/extra_correction/implicit/route2_v0_full_response_kkt.py)
remains one declared input ($v_0$) from a complete variational pipeline, and
the response side is already sourced and validated (V0-AIPR: `0.0967` Frobenius,
`0.1101` worst-direction on the same 516 points). What changes is the target
the $v_0$ search must hit: **roughly 2× tighter than the criterion previously
in use**, and tighter still under a realistic error allocation.

Any $v_0$ candidate previously screened against `0.20` was screened against a
criterion that could not deliver the route's accuracy target. That screen
should be re-run against equation (4) before any further permanent-source work.

---

## 8. Addendum (2026-09-05): the gate is unpassable by construction

Running the MLIP sources on this surface returned a decisive negative
(`route2-v0-mace-zero-field-static-mep-acetone-516-v1`, harness validated by
reproducing the GFN2 numbers exactly):

| source | eps_F | eps_maxabs |
| --- | --- | --- |
| MACE-POLAR-1-M `l<=1` | 0.3224217790 | 0.5313041398 |
| MACE-EF auxiliary density | 0.2954775346 | 0.3140626053 |
| MACE-EF energy-conjugate | 0.2777210207 | 0.3414157680 |
| GFN2 MOLDEN (recorded, rejected) | 0.2199987983 | 0.3760559034 |

All three MLIP sources score worse than the rejected QM source, and all miss
the derived `0.1129` ceiling by 2.5-2.9x. Every arm nevertheless reproduced the
molecular dipole well (`0.037 e*bohr` for two of them), repeating the pattern
already recorded for the response source: a correct molecular moment does not
imply a correct cavity-surface MEP.

Before concluding anything about model quality, the **representation floor** was
measured by fitting the best possible atom-centred source directly to the QM
reference
([`benchmarks/measure_static_source_representation_floor.py`](benchmarks/measure_static_source_representation_floor.py)):

| best achievable fit | eps_F | sum q | max abs q |
| --- | --- | --- | --- |
| `l<=1`, free enclosed charge | 0.076329 | +0.0448 e | 0.757 e |
| `l<=1`, charge-neutral | **0.246061** | 0 | 1.184 e |
| `l<=2`, free enclosed charge | **0.022590** | +0.0437 e | 0.913 e |
| `l<=2`, charge-neutral | 0.218312 | 0 | 6.694 e |

**The charge-neutral floor `0.2461` exceeds the derived ceiling `0.1129`.** In
this representation no source of any kind — MLIP, semi-empirical, or
all-electron DFT — can pass the gate. The gate has been rejecting candidates
for a property none of them could have had.

### Why: charge outside the cavity

Imposing the enclosed charge and scanning it gives a sharp minimum:

```
Q = +0.0000 e -> eps_F = 0.246061
Q = +0.0437 e -> eps_F = 0.076545      <- minimum
Q = +0.1000 e -> eps_F = 0.298133
```

The preferred value is stable across basis size (`+0.0448` at `l<=1`,
`+0.0437` at `l<=2`, agreeing to 2%), it is a single well-determined degree of
freedom, and the charge-neutral residual is spatially diffuse rather than
concentrated at the closest points (mean absolute residual by nearest-atom
distance quartile: `0.0075 / 0.0084 / 0.0050 / 0.0069`).

That is the signature of **outlying charge**: roughly `0.044 e` of acetone's 32
electrons — about `0.14%` — lies outside this cavity. An atom-centred source
constrained to zero net enclosed charge cannot represent a reference that
includes it, and forcing neutrality drives the coefficients to unphysical
values (`max|q| = 6.694 e` at `l<=2`). This is the effect COSMO addresses with
an explicit outlying-charge correction; it is a property of the cavity and the
representation, not of the source model.

### Consequence

`eps_F` as currently measured conflates two things:

1. genuine source-model error, and
2. an outlying-charge artifact that no interior source can avoid.

Only (1) is what the gate intends to measure. The gate must therefore either
allow the enclosed charge as a fitted degree of freedom, apply an
outlying-charge correction, or move the evaluation surface outward — and then
re-measure every candidate. Until that is done, no static-MEP rejection on this
surface, **including the GFN2 rejection upheld in section 5 of this document**,
distinguishes a bad source from a good source measured badly.

Section 5's arithmetic is unchanged; its scientific weight is not. The derived
ceiling of section 4 stands, because it is a statement about energy
propagation, not about this measurement.

### What this does not say

This is one molecule, one cavity, and one reference. It does not establish that
any MLIP source is adequate; the corrected comparison has not been run. It says
the existing comparison cannot answer the question. The immediate next step is
to re-measure all four recorded candidates with the enclosed charge free, which
requires only the already-computed candidate MEP vectors and no new model run.

---

## 9. Corrections (2026-09-05, after independent review)

An independent re-measurement found two errors in this document. Both are
corrected here rather than edited away.

### 9.1 Section 5 inverts the Cauchy-Schwarz bound (logic error)

Equation (2) is an **upper** bound:
$|\delta G^\*|\le\varepsilon_F\cdot2\kappa\rho|\Delta G_{\rm pol}|$. Therefore

* $\varepsilon_F\le\varepsilon_F^{\max}\;\Rightarrow\;|\delta G^\*|\le\tau$ —
  the source is **certified admissible**;
* $\varepsilon_F>\varepsilon_F^{\max}\;\Rightarrow$ the bound exceeds $\tau$ and
  says **nothing** about $|\delta G^\*|$.

Section 5 read the second case as a rejection ("REJECT, by 1.95x"). That does
not follow. Exceeding the ceiling means the bound **cannot certify** the source,
not that its energy error exceeds the tolerance. Equation (4) is a *sufficient
admission criterion only*; it is not a rejection criterion.

**Therefore section 5's claim that it re-supports
`reject-gfn2-molden-permanent-source` is withdrawn.** The original rejection
still stands on its own preregistered terms; this document does not add to it,
and never could have. This also removes an internal contradiction: section 6
already says the norms cannot decide the question and only
$\langle q^\*,\delta v_0\rangle$ can, which section 5 then ignored.

The derivation itself (sections 2-4) is unaffected. What changes is what may be
concluded from a candidate sitting *above* the ceiling: nothing, without $q^\*$.

### 9.2 Section 8 misattributes the released degree of freedom

Section 8 called the extra degree of freedom **outlying charge**, estimated at
`+0.044 e`. That attribution is wrong.

On this surface the enclosed-charge mode
$g_k=\sum_a|s_k-\mathbf R_a|^{-1}$ is nearly collinear with a constant:
$\cos(g,\mathbf 1)=0.996127$ (independently confirmed). Separating the two:

| `l<=1` fit | eps_F | fitted constant |
| --- | --- | --- |
| charge-neutral, no constant | 0.246061 | — |
| charge-neutral, **free constant** | **0.076842** | `+0.008905` hartree/e |
| free enclosed charge, no constant | 0.076329 | — |

**A constant alone recovers 99.7% of the gap.** The `+0.0437 e` of section 8 was
the least-squares route to a near-uniform offset, not a measured enclosed
charge. Adding both parameters together improves only `0.076842 -> 0.075950`
while driving the coefficients into a sign fight (`Q=+0.128 e`, `c=-0.0166`),
which is the expected signature of a degenerate pair.

The physical reading changes accordingly. A near-uniform positive offset of
`~0.0089 hartree/e` at a surface that sits `1.20-1.85` Å from the nearest
nucleus is the signature of **charge penetration** — the points lie inside the
tail of the electron density, so an atom-centred point-multipole source
over-screens the nuclei — rather than of charge outside the cavity, which would
appear as a decaying $Q/r$ shape and does not win the fit.

This data cannot fully separate charge penetration from a difference in the
zero convention between the two potentials. The discriminating test is cheap:
penetration decays as the evaluation surface moves outward, a gauge constant
does not. That test has not been run, so **the mechanism is stated as the
leading explanation, not as a result.**

### 9.3 What survives

The correction is to the *mechanism and its name*, not to the structural
conclusion:

* the charge-neutral `l<=1` representation floor is `0.246061`, and it still
  **exceeds** the derived ceiling `0.1129`;
* releasing one degree of freedom still collapses it to `0.0768` (`l<=1`) and
  `0.0226` (`l<=2`, free charge);
* so the gate as constructed still cannot be passed by any atom-centred source,
  MLIP or QM, and one extra degree of freedom still fixes it.

### 9.4 Candidates re-measured with the degree of freedom released

From `route2-v0-static-source-charge-release-acetone-516-v1` (no new QM; MACE
raw vectors were regenerated by one forward pass each, raw eps_F reproducing to
`5e-7`):

| candidate | eps_F raw | + charge mode | + charge and constant |
| --- | --- | --- | --- |
| GFN2 MOLDEN | 0.219999 | 0.155776 | 0.139841 |
| MACE-POLAR-1-M `l<=1` | 0.322422 | 0.231777 | 0.217447 |
| MACE-EF-v2 auxiliary | 0.295477 | **0.113387** | **0.111590** |
| MACE-EF-v2 energy-conjugate | 0.277721 | 0.231687 | 0.157891 |

Every candidate remains above the `0.076329` achievable `l<=1` floor, by 1.5x
to 2.9x. **So there is genuine source-model error beyond the representation
issue**, and the earlier framing — that the floor explained the failures — was
too generous to the models. The best corrected candidate, MACE-EF auxiliary at
`0.1116`, is the checkpoint's diagnostic density output, which
`mace_polar_ef.py` explicitly does not treat as energy-conjugate; the
energy-conjugate source is worse (`0.1579`). None of these are admissions:
the parameters were fitted against the frozen QM reference, which is an oracle
diagnostic, not a production procedure.
