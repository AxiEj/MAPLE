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
