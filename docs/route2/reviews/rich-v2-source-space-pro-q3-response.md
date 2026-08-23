# Verdict

**APPROVE**, provided the schema encodes the representation and provenance constraints below as machine-checkable contracts rather than relying on comments or naming conventions.

The proposed correction fixes the central category error: the physical source is not an (N\times 9) tensor and is not one (N\times 4) tensor acted on by a universal kernel. It is a **kernel-tagged direct sum of two (N\times 4) coefficient branches**.

The remaining corrections are semantic and evidentiary rather than changes to the frozen mathematics.

---

## 1. The category-correct source representation is (N\times 4)

Let

[
\mathcal C_4(R)
===============

\mathbb R^{N\times 4}
]

denote the registered atomic (l\leq 1) coefficient space, with columns

[
(q,;c_{10}^{\mathrm{real}},;c_{11}^{\mathrm{real}},;c_{1,-1}^{\mathrm{real}}).
]

Both

[
c_{\mathrm{MDP}}\in\mathcal C_4(R)
]

and

[
\Delta c_{\mathrm{POLAR}}
=========================

## M_{\mathrm{POLAR}}(R,u^*)

M_{\mathrm{POLAR}}(R,0)
\in\mathcal C_4(R)
]

are therefore (N\times 4) arrays.

The continuum field belongs to a different space,

[
u\in\mathcal U_8(R)=\mathbb R^{N\times 8},
]

whose eight channels are receiver or field coordinates, not additional source components.

Therefore an (N\times 9) source requirement is a **category error** for this exact profile. It would necessarily confuse the source coefficient axis with something else, most plausibly:

* one monopole plus eight receiver channels;
* nine Cartesian force-panel positions or stencil entries;
* or an invented padded source representation.

None is valid. The nine-component Cartesian panel is a **measurement-occurrence structure**, not an atomic source-channel structure.

The physical source should be represented conceptually as

[
\mathscr S(R,u^*)
=================

\bigl(\mathrm{point},c_{\mathrm{MDP}}\bigr)
\oplus
\bigl(\mathrm{GTO}*{1.5\text{\AA}},\Delta c*{\mathrm{POLAR}}\bigr),
]

and its boundary action is

[
b
=

B_{\mathrm{point}}c_{\mathrm{MDP}}
+
B_{\mathrm{GTO}}\Delta c_{\mathrm{POLAR}}.
]

The two coefficient arrays are isomorphic as representation-space objects, but their physical source objects remain distinct because their kernel tags differ.

**Conclusion for Question 1:** (N\times4) is correct. (N\times9) is unequivocally wrong for this profile.

---

## 2. The order is non-Cartesian; the registered convention must be hash-bound

The order

[
(\texttt{net_monopole},
\texttt{real_l1_m0},
\texttt{real_l1_m1},
\texttt{real_l1_m_minus1})
]

is an exact index order, but the labels alone do not mathematically determine every convention. Different real-spherical-harmonic libraries can differ in:

* normalization;
* phase and sign conventions;
* the real/complex harmonic transformation;
* the association of real (m=\pm1) combinations with laboratory axes;
* handedness and rotation action;
* coefficient units and scaling.

In particular, the last three columns must not be relabelled casually as ((x,y,z)), ((z,x,y)), or any other Cartesian tuple. An (l=1) representation may be linearly related to Cartesian components, but that relation is part of the registered convention, not something inferable from column names.

The correct schema solution is not a second mutable free-text basis field. It is one authoritative binding such as:

```text
source_space_id =
    maple.route2.atomic-l1-source-space.v1

source_space_contract_digest =
    H(canonical registered source-space specification)
```

The canonical specification must include the component order, normalization, phase, units, and rotation convention.

If `maple.route2.atomic-l1-source-space.v1` is already immutable and content-addressed with all those semantics, no additional competing basis field is needed. If it is merely a versioned string whose definition could change externally, then its canonical contract digest is required before schema freeze.

The same requirement applies independently to the (N\times8) receiver:

```text
receiver_space_id
receiver_space_contract_digest
```

Its two radial-channel ordering, angular ordering, normalization, and units must be bound. Saying only “(N\times8), two radial channels” is not enough to interpret a raw residual vector independently.

**Conclusion for Question 2:** the order is non-Cartesian. It is sufficient only together with the immutable registered convention and its digest.

---

## 3. The coefficient-wise sum is mathematically valid but is not a physical total source

Define

[
c_{\mathrm{audit}}
==================

c_{\mathrm{MDP}}
+
\Delta c_{\mathrm{POLAR}}.
]

This addition is mathematically legitimate because both operands belong to the same registered coefficient space, use the same atom order, and have the same coefficient units.

For example, the global monopole functional

[
Q(c)=\sum_A c_{A,0}
]

is linear, so

[
Q(c_{\mathrm{audit}})
=====================

Q(c_{\mathrm{MDP}})
+
Q(\Delta c_{\mathrm{POLAR}}).
]

Thus the sum can support checks such as:

[
Q_{\mathrm{induced}}
====================

\sum_A \Delta q_A,
]

[
Q_{\mathrm{combined}}
=====================

Q_{\mathrm{permanent}}+Q_{\mathrm{induced}},
]

and any frozen charge-conservation requirement.

However, it is not a physical source that can be propagated through one kernel. Suppose there were a common operator (B) satisfying

[
B(c_p+c_i)
==========

B_{\mathrm{point}}c_p+B_{\mathrm{GTO}}c_i
]

for all (c_p,c_i).

Setting (c_i=0) gives

[
B=B_{\mathrm{point}},
]

while setting (c_p=0) gives

[
B=B_{\mathrm{GTO}}.
]

Therefore such a common (B) can exist only if

[
B_{\mathrm{point}}=B_{\mathrm{GTO}},
]

which is false for the frozen exterior point-multipole and 1.5-Å Gaussian kernels.

The coefficient sum is consequently a **lossy projection**:

[
(c_{\mathrm{MDP}},\Delta c_{\mathrm{POLAR}})
\longmapsto
c_{\mathrm{MDP}}+\Delta c_{\mathrm{POLAR}},
]

because it discards the branch identity required to reconstruct the physical boundary source.

### Required naming and type contract

`total_source4` is too easy to misunderstand as an operational total source. A safer name is:

```text
audit_coefficient_sum4
```

or

```text
nonoperational_source_coefficient_sum4
```

Its schema type should explicitly state:

```text
semantic_role       = algebraic_audit_projection
physical_kernel     = none
operator_dispatch   = forbidden
derived_from        = [permanent_point_source4,
                       induced_gto_source4]
```

The operational source should instead be serialized as two typed branches:

```text
permanent_point_source4
induced_gto_1p5A_source4
```

For a charge audit, the strongest design is to derive scalar charge quantities from the first column. Calling the complete (N\times4) array a “charge” object is imprecise because its three (l=1) columns are not charges.

**Conclusion for Question 3:** the coefficient sum is scientifically legitimate as a non-operational algebraic audit object. It must not be named or typed as a physical total source.

---

## 4. The decomposition neither double counts nor omits the frozen source terms

The decomposition is

[
c_{\mathrm{perm}}=c_{\mathrm{MDP}},
]

[
c_{\mathrm{zero}}=M_{\mathrm{POLAR}}(R,0),
]

[
c_{\mathrm{final}}=M_{\mathrm{POLAR}}(R,u^*),
]

[
\Delta c_{\mathrm{POLAR}}
=========================

c_{\mathrm{final}}-c_{\mathrm{zero}},
]

followed by

[
b
=

B_{\mathrm{point}}c_{\mathrm{perm}}
+
B_{\mathrm{GTO}}\Delta c_{\mathrm{POLAR}}.
]

This is correct.

`M_POLAR(R,0)` is the checkpoint’s zero-field baseline. It is removed so that the responsive branch contains only the induced change relative to that baseline. The permanent branch is supplied independently by (c_{\mathrm{MDP}}).

Adding `M_POLAR(R,0)` again would double count a zero-field source contribution. Failing to subtract it would have the same effect implicitly. Neither occurs in the stated formula.

There is also no omission within the frozen two-branch profile:

* the complete declared permanent contribution is (c_{\mathrm{MDP}});
* the complete declared responsive contribution is the nonlinear difference;
* the zero-field checkpoint output is a subtraction reference, not an additional physical branch.

For raw-evidence auditability, the rich leaf should retain or content-addressedly reference both

```text
polar_final_source4
polar_zero_reference_source4
```

and derive

```text
induced_source4 =
    polar_final_source4 - polar_zero_reference_source4
```

under the frozen dtype and arithmetic contract. Storing only an asserted induced array would allow its value to be checked as data but would not independently demonstrate that the required subtraction was performed.

The zero reference must be evaluated at the same geometry, atom ordering, checkpoint, and exactly canonical zero (u).

**Conclusion for Question 4:** no double count and no source omission are present.

---

## 5. The per-start leaves are sufficient for endpoint gate replay, not trajectory reconstruction

For each start (s), the proposed raw fields include:

[
u_s^*\in\mathbb R^{N\times8},
]

[
r_s^*
=====

T(u_s^*)-u_s^*
\in\mathbb R^{N\times8},
]

the final polarization energy (E_{\mathrm{pol},s}), initial-state identity, iteration count, and convergence flag.

These data are sufficient to recompute all **endpoint-defined** root and convergence quantities without trusting stored scalar summaries, including:

[
|r_s^*|,
]

[
|u_{\mathrm{cold}}^*-u_{\mathrm{wide}}^*|,
]

[
|E_{\mathrm{pol,cold}}-E_{\mathrm{pol,wide}}|,
]

cross-replay field, residual, and energy comparisons, and any maxima derived from the raw arrays.

The validator must obtain the norm, channel weighting, absolute/relative tolerances, comparison direction, accumulation precision, and NaN/Inf policy from the sealed convergence contract. It should recompute the convergence decision and compare it with the recorded flag rather than treating the flag as authoritative.

The leaves are not sufficient to reconstruct:

* the entire fixed-point iteration path;
* DIIS or Anderson histories;
* intermediate residuals;
* an independently verified iteration count;
* convergence gates involving the previous iteration’s energy or field change, unless those quantities are separately recorded.

Therefore the evidence should be described as supporting **final-state or endpoint gate replay**, not full solver-trajectory replay.

If the frozen gate is based only on the final actual residual, cross-start root agreement, and final energies, the proposed fields are sufficient. If any gate uses an iterative quantity such as

[
|E_k-E_{k-1}|,
\qquad
|u_k-u_{k-1}|,
]

or a monotonicity condition, that raw final-step quantity must also be retained.

The initial-state digest is sufficient for reproducibility only if it resolves to archived bytes or to a deterministic start generator whose contract and inputs are sealed. An opaque digest with no recoverable preimage proves identity but does not permit a replay.

### Energy recomputation

For geometry/state (j) and start (s), the required invariant is

[
E_{\mathrm{total},j,s}
======================

E_{\mathrm{vac}}(R_j)
+
E_{\mathrm{pol},j,s}.
]

The matching state-specific vacuum energy must be added exactly once.

“Same frozen vacuum energy” must mean the same (E_{\mathrm{vac}}(R_j)) is used across cold and wide starts for the same geometry. It must not mean that the central-geometry vacuum energy is reused for displaced geometries. Every displaced state requires its own frozen vacuum energy evaluated at that displaced geometry.

No parent, panel builder, force extractor, or root summary may add the vacuum energy a second time.

**Conclusion for Question 5:** yes for final-state convergence/root gates and force-energy reconstruction; no for exact iteration-history replay.

---

## 6. Remaining freeze conditions and possible blockers

Under the stated facts, there is no inherent source double count, no inherent hash cycle, and no necessary energy double count. The following nevertheless must be encoded before freeze.

### A. Type the two physical branches separately

The schema should make the invalid common-kernel operation unrepresentable:

```text
PermanentPointSource4
InducedGTO15Source4
AuditCoefficientSum4
NativeField8
```

A generic `Source4` object with a caller-selectable kernel would leave the central physical error open.

### B. Bind both representation conventions

Both the (N\times4) source convention and (N\times8) receiver convention require immutable IDs and contract digests. Array shape alone is not scientific semantics.

### C. Canonically serialize numerical leaves

Every content digest must bind at least:

* object type and schema version;
* shape;
* dtype and endianness;
* memory/index order;
* units;
* atom order and geometry digest;
* channel-space contract digest;
* exact array bytes;
* policy for NaN, infinity, and signed zero.

Type-domain separation is required so that identical byte strings cannot be interpreted interchangeably as a source, field, residual, or energy object.

### D. Preserve an acyclic hash graph

A safe dependency order is:

[
h_F
===

H(\text{force-panel contract}),
]

[
h_P
===

H(\text{prepared input payload},h_F),
]

[
h_C
===

H(\text{computation contract},h_F,h_P),
]

[
h_L
===

H(\text{raw leaf payload},h_C),
]

[
h_R
===

H(\text{ordered occurrence ledger of leaf references}).
]

No leaf may include (h_R), and neither (h_F) nor (h_P) may depend on a measurement result. A rich leaf digest must exclude its own digest field.

The fact that the computation seal binds both (h_F) and (h_P), while (h_P) itself includes (h_F), is redundant binding, not circularity.

The validator must recompute (h_P) from the captured and parsed input objects. Comparing two caller-supplied strings that happen to be equal does not establish provenance.

### E. Separate content identity from solve occurrence identity

Content-addressed storage and experimental occurrence accounting are different categories.

Two distinct solve occurrences can produce byte-identical state content. Conversely, one deliberately shared center solve can be referenced multiple times by a Cartesian stencil.

A robust construction is:

```text
state_content_digest = H(raw state content)

occurrence_record = {
    occurrence_id,
    ledger_position,
    panel_id,
    start_id,
    state_content_digest
}
```

The ordered occurrence ledger must retain all 141 occurrence records.

The clean rule is:

* every actual solve occurrence appears exactly once in the solve-occurrence ledger;
* a panel-reference graph may reference the designated center occurrence more than once;
* the historical second explicit center remains a distinct occurrence even if its geometry or numerical result matches another center;
* storage deduplication may deduplicate bytes but may never collapse occurrence records.

This formulation is clearer than saying an occurrence is “consumed more than once.”

### F. Keep raw-evidence and derived-summary authority separate

For revalidation, the validator must recompute:

* residual norms;
* cross-start root distances;
* energy differences;
* finite-difference forces;
* force errors and maxima;
* ledger completeness.

Legacy or newly stored scalar summaries may be checked for consistency, but they must not be the authoritative evidence.

Iteration count remains recorded metadata unless an iteration transcript exists. It must not be advertised as independently reconstructible from the endpoint leaf.

### G. Enforce the state-specific energy identity

The schema should validate

[
E_{\mathrm{total},j,s}
----------------------

# E_{\mathrm{pol},j,s}

E_{\mathrm{vac}}(R_j)
]

for every start, with the same right-hand side across starts at fixed geometry.

A second addition of (E_{\mathrm{vac}}) at the force-panel or aggregate layer is a freeze-blocking defect.

### H. Prevent archive evidence from entering H1 admission

The real H0 object, synthetic projection fixture, H1 raw leaves, and capability/admission state require separate namespaces and gate labels. There must be no logical rule of the form

[
\text{H0 compatibility}
;\lor;
\text{two clean H1 replays}.
]

H1 admission must continue to require the two clean sealed replays.

---

## 7. Final schema verdict

**APPROVE.**

The proposed correction is scientifically and mathematically sound under the following required contract refinements:

1. Bind the exact source-space and receiver-space conventions by immutable digests.
2. Rename or strictly type `total_source4` as a non-operational coefficient audit projection.
3. Keep permanent-point and induced-GTO branches separately kernel-tagged.
4. Retain or reference both final and zero-reference POLAR source arrays so the induced difference is independently checkable.
5. State explicitly that the raw leaves support endpoint-gate replay, not reconstruction of the solver trajectory.
6. Bind the matching state-specific vacuum energy and enforce exactly one addition.
7. Separate state-content hashes from ordered solve-occurrence identities.
8. Use a canonical acyclic hash DAG in which trusted validation recomputes input manifests from actual captured objects.

If the audit coefficient sum remains dispatchable through a physical kernel, if the (N\times8) receiver convention remains unidentified, or if a single central vacuum energy is reused across displaced geometries, the verdict changes to **BLOCK**.

---

## 8. The H0 replacement is scientifically preferable and may satisfy only the archive gate

Let

[
P:X_{\mathrm{rich}}\rightarrow Y_{\mathrm{v1}}
]

be the archive projection from a rich raw-evidence object to the legacy H0 v1 representation.

Because v1 lacks displaced energies, per-start fields, actual residual vectors, and per-state sources, (P) is many-to-one. Given the historical object

[
y_{\mathrm{H0}}\in Y_{\mathrm{v1}},
]

there is no identifiable historical

[
x_{\mathrm{H0}}\in X_{\mathrm{rich}}
]

such that

[
P(x_{\mathrm{H0}})=y_{\mathrm{H0}}.
]

Many invented rich objects could project to the same legacy summaries. Selecting one would create unsupported raw evidence and falsely attribute it to the historical run.

The proposed replacement is therefore the correct scientific approach:

1. Pin the real historical H0 v1 bytes and digest exactly.
2. Preserve its actual derived forces, errors, maxima, and legacy identifiers as archive facts.
3. Do not manufacture missing displaced energies, fields, residual vectors, or source arrays.
4. Test the deterministic rich-to-v1 projection adapter on an explicitly labelled synthetic rich golden whose complete leaves are known by construction.

The synthetic golden can legitimately prove propositions such as:

* the adapter reads the rich schema correctly;
* occurrence ordering is preserved;
* center-reference sharing is handled correctly;
* source and field objects are projected or omitted according to contract;
* derived v1 fields are produced deterministically.

It cannot prove that the historical H0 execution possessed those rich leaves, and it cannot transfer H0 admission or scientific validity to H1.

Yes, it may satisfy an **archive-only compatibility gate**, provided that gate is explicitly defined as adapter/projection compatibility and remains logically independent of H1 admission. H1 still requires two clean sealed replays and cannot be opened by the H0 object, its legacy admission, or the synthetic fixture.

This is strictly preferable to fabricating a historical rich fixture because it preserves the distinction between:

* observed historical evidence;
* deterministic software-adapter evidence;
* and new H1 scientific replay evidence.

MAPLE RICH V2 SOURCE SPACE AUDIT COMPLETE
