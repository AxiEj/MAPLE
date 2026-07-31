# Route-2 V0: frozen-source KKT boundary

## Narrow purpose

This note distinguishes a **V0-FrozenSource-KKT** calculation from a claim of
a complete stationary electronic functional.  It adds no model parameter,
training, experimental solvation label, cavity choice, or accuracy result.

At fixed nuclear geometry, the existing full-response KKT primitive permits

\[
\Omega_{\mathbf R}(x)=E_{\rm gas}(\mathbf R)
+\frac12x^\mathsf T C_{\mathbf R}^{+}x
+\frac12\left[v_0(\mathbf R)+B_{\mathbf R}x\right]^\mathsf T
Q_{\mathbf R}\left[v_0(\mathbf R)+B_{\mathbf R}x\right]
+G_{\rm CDS}(\mathbf R),
\]

subject to the exact support constraints on the induced-response coefficients
\(x\).  The KKT stationarity is only with respect to \(x\) (and the
continuum state after its elimination).  Consequently this is a common scalar
for the **induced response plus continuum**, not proof that the frozen
permanent source is the minimizer of a complete electronic functional.

## The required representation relation

The permanent source and the induced response do **not** have to share the
same coefficient basis.  They do have to meet in one declared surface-MEP
pairing:

\[
v_{\rm total}=v_0+B x,
\qquad
q=Qv_{\rm total}.
\]

Here \(v_0\) may be supplied directly by a permanent-source representation
(for example, a frozen point-multipole source) while \(x\) can remain in the
AIPR transition-density basis.  The non-negotiable duality requirement is

\[
q^\mathsf T Bx=x^\mathsf T B^\mathsf Tq,
\]

with the forward and transpose induced maps constructed from the same
integrals.  The permanent contribution needs no artificial projection into
the AIPR coefficient space: its energy pairing is directly
\(v_0^\mathsf TQv_0\) and its cross term is
\(v_0^\mathsf TQB x\).

This correction does **not** overturn the permanent-reference
identifiability result.  A response covariance alone still cannot identify a
permanent electronic density.  It only says that a separately proven
permanent **surface potential** can support a frozen-source scalar without
being falsely relabelled as an AIPR coefficient vector or as a complete
stationary electronic state.

## Admission requirements for a permanent surface source

Before a candidate \(v_0(\mathbf R)\) may enter a physical-continuum or
force calculation, it must be frozen before solvation results are read and
carry all of the following evidence:

1. an unambiguous semantic ledger: it is either the complete net solute
   electrostatic source or an electronic-only source paired with the nuclear
   potential exactly once; the two conventions must never be mixed;
2. total-charge, total-dipole, translation, and rotation certificates in its
   declared convention;
3. a frozen static QM exterior-MEP gate, including the continuum-energy norm
   when a reciprocal \(Q\) is available;
4. a coordinate VJP for \(v_0\), plus the coordinate derivatives of
   \(B,Q,C\), cavity, and every retained non-electrostatic term before an
   envelope-force or PES claim; and
5. a reciprocal/passive smooth continuum and the existing KKT/support gates.

The zero-field MACE point-\(l\leq1\) output is currently only a candidate
for item 1--3.  It has not passed the required static-QM-MEP and coordinate
derivative gates, so it is not admitted as a physical V0 permanent source.
No FreeSolv/MNSol score may select, repair, or validate it.

## Current code boundary

`route2_v0_full_response_kkt.py` already accepts
`permanent_surface_potential_hartree_per_e` as this explicit \(v_0\) input.
`route2_v0_response_kernel.py` supplies only the induced covariance and its
support curvature.  The two modules therefore support the frozen-source KKT
algebra without asserting a same-basis permanent density.  They remain
fixed-geometry structural primitives until the admissions above, a smooth
continuum coordinate derivative, forces/PES, and the frozen multi-molecule QM
physics gates pass.
