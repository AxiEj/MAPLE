# Positive Bernstein parent Pro audit — 2026-08-17

## Scope

This audit concerns only the target-blind geometry descriptor shared by the
candidate smooth CDS area and a future harmonic Galerkin continuum.  No MNSol
target, hybrid prediction, confirmation record, CDS fit, or accuracy result was
provided to or opened by the feature computation described here.

The predecessor independently projected every smooth pair factor to
`l <= 4`, multiplied those projections, and projected the product to `l <= 4`.
A frozen development geometry exposed a genuine truncation artifact: one
nearly buried carbon had `-0.1057693635 A^2`, while independent integration of
the positive unprojected product gave `+0.00848878963 A^2`.  Clipping was and
remains forbidden.

## Verified Pro consultation

The retained Chrome workflow was used with all five fail-closed checks:

1. the composer displayed **Pro** before submission;
2. the power control displayed **Pro, 5 of 5**;
3. the exact prompt was visibly present after submission;
4. the same conversation control remained bound throughout generation; and
5. the answer was extracted only after generation controls disappeared, the
   final decision marker appeared, and **Copy response** was present.

Evidence:

| Item | Value |
|---|---|
| Conversation | `https://chatgpt.com/c/6a8217be-0fa0-83e8-8572-f6747fcb032d` |
| Conversation control | `conversation-options-WEB:c3a436a0-a5a5-4baf-9a89-a456ad518ac5` |
| Prompt SHA-256 | `9bcf28b31a0ed1eb62de31445f473c9fa1b52c6fec50b962e0645d5b83fb1992` |
| Submission-evidence path | `C:\Users\29860\AppData\Local\Temp\maple_smooth_area_pro_q11_submission.txt` |
| Complete UIA SHA-256 | `6f92aaa7ab7f47ee60db107db6ff2bdc4f85100903048deb2a8beb0df002b8c4` |
| Clean answer SHA-256 | `20668f9c800462173f5c0630abcc8122703aa184a10e5adf31fc22558d31347c` |
| Reported reasoning time | `19m 12s` |
| Final marker | `USE POSITIVE BERNSTEIN PARENT` |

The external answer is advisory evidence, not an admission by itself.

## Adopted mathematical contract

For each pair, the compact smooth switch is sampled at the five fixed nodes
`t_k = -1 + k/2`, then represented by the degree-four Bernstein polynomial

```text
p_ij(x) = sum_k s(z_ij(t_k)/w) C(4,k) x^k (1-x)^(4-k),
x = (1 + u dot d_hat_ij)/2.
```

Every control value and Bernstein basis function lies in `[0,1]`, so
`0 <= p_ij <= 1` and the finite parent

```text
e_i(u) = product_j p_ij(u)
```

obeys `0 <= e_i <= 1` structurally.  Consequently
`0 <= a_i^2 integral e_i dOmega <= 4 pi a_i^2` without clipping.

The low-band harmonic reconstruction is **not** the physical mask.  The
continuum-facing object is the exact parent moment matrix

```text
W_i = P_L M_{e_i} P_L,
(W_i)_{ab} = integral e_i Y_a Y_b dOmega.
```

Both `W_i` and `I-W_i` are positive semidefinite in exact arithmetic.  The CDS
area and `W_i` are contracted from the same finite parent graph.  A fixed
positive-weight sphere rule may be used only as an exact backend for the
declared finite polynomial degree; it is not a laboratory-fixed hard mask.

The following choices are frozen before any target or hybrid output is joined:

* pair degree: `4`;
* maximum transition factors per atom: `30`;
* maximum exact integrand degree: `128`;
* production candidate surface band for this feature lane: `L = 2`;
* required parent moment band: `2L = 4`;
* no pointwise reconstruction, clipping, eigenvalue repair, active-factor
  truncation, or target-selected degree.

At the geometry-only frozen panel maximum of 27 transition factors, the exact
matrix integrand degree is `4*27 + 2*2 = 112`, below the bound.

## Admission boundary

All 306 target-blind geometry rows must be regenerated from row zero under a
new content-addressed profile.  Before any fit, the run must fail closed unless:

* every atom respects the factor and algebraic-degree caps;
* every area lies in its certified numerical enclosure of
  `[0, 4*pi*a_i^2]`;
* `W_i` and `I-W_i` are PSD within a forward-error bound;
* rigid rotation, translation, and label permutation tests pass;
* the same-parent coordinate VJP passes multi-step finite differences; and
* no code path evaluates the retained low moments as a physical point mask.

Degree four defines a new positive cavity parent.  It is not claimed to be the
`L2`-optimal projection or to reproduce the old smooth-step product exactly.
Representation fidelity and downstream solvation accuracy remain separate,
unpassed gates.

