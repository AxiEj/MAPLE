# Route-2 salvage audit (2026-09-04)

## Purpose

The [exploration register](ROUTE2_V0_EXPLORATION_REGISTER.md) records *what was
rejected*. It does not distinguish two very different reasons for rejection:

* **physics** — the construction is wrong, and no protocol change recovers it;
* **protocol** — the construction was never propagated to the quantity the
  route actually cares about, or was stopped by an implementation policy.

This audit makes that split explicit for the results with the largest sunk
cost. It **adds no result, changes no threshold, and modifies no frozen
artifact.** Every number below is either read from an existing immutable
artifact or derived from one by
[`benchmarks/analyze_confirmation123_subsample_bias.py`](benchmarks/analyze_confirmation123_subsample_bias.py),
which uses only the standard library and writes nothing.

Ranking is by *recoverable value per unit of remaining work*, not by age.

---

## S1 — V0-RK full-response KKT: one declared input from a complete pipeline

**Status: recoverable. Highest value in the repository.**

The project's own stated central bottleneck was that a molecular
polarizability fixes only a 3x3 projection of the density susceptibility, so an
MLIP cannot supply the spatial response a continuum needs
([`ROUTE2_V0_ACCURACY_BOTTLENECKS_20260731.md`](ROUTE2_V0_ACCURACY_BOTTLENECKS_20260731.md)).

**That problem is solved, without fitting, and the evidence is frozen:**

| source | rel. Frobenius (ceiling 0.20) | worst direction (ceiling 0.30) |
| --- | --- | --- |
| V0-ADT atomic displacement tangent | 0.1484314509 | 0.1611208415 |
| V0-AIPR atomic independent-particle | **0.0966763794** | **0.1100924547** |

Against all 516 frozen exterior acetone QM-MEP points, no fitted width, no
experimental label.

[`route2_v0_full_response_kkt.py`](../../maple/function/calculator/extra_correction/implicit/route2_v0_full_response_kkt.py)
already assembles the rest of a *variational* pipeline in one coefficient/dual
space:

* `C` — the frozen PSD response covariance and its support curvature `C+`;
* `B` — the exact GTO transition-density surface map, whose reaction dual is
  `B^T` **by construction** (both come from the same dense AO-integral matrix,
  so surface map and dual cannot silently disagree);
* `Q` — a declared reciprocal, passive continuum response;
* a symmetric KKT solve of one scalar
  `G(x) = ½ xᵀC⁺x + ½ (v₀+Bx)ᵀQ(v₀+Bx) + xᵀf` subject to `Nx = 0`.

The only missing term is **`v₀`, the permanent surface potential** — and the
theory document states plainly that `v₀` "is intentionally an explicit input".

### What actually blocks `v₀`, and why it is a protocol block

The nearest candidate, the zero-field GFN2-xTB MOLDEN permanent source, was
rejected in
[`route2-v0-gfn2-molden-static-mep-acetone-v2.json`](benchmarks/route2-v0-gfn2-molden-static-mep-acetone-v2.json):

```
static_dipole_relative_frobenius : 0.10399614  (max 0.2)   PASS
static_mep_relative_frobenius    : 0.21999880  (max 0.2)   FAIL
static_mep_relative_max_abs      : 0.37605590  (max 0.3)   FAIL
```

Every representation and QM-binding check passed. The rejection is on two
numbers that miss by 10% and 25%.

**The ceilings 0.20 / 0.30 were carried over from the *response* source gate
and applied unchanged to a *static* source.** They are not derived anywhere
from a solvation-energy tolerance. `∂V/∂E` and `V` are different quantities
that enter the polarization scalar differently, so reusing one ceiling for the
other leaves the criterion undetermined until the propagation is done.

> **Correction (2026-09-04, same day).** An earlier revision of this section
> implied the borrowed ceiling was probably too strict and that GFN2 might be
> recoverable by re-adjudication. **That was wrong.** The propagation has now
> been carried out in
> [`ROUTE2_V0_STATIC_SOURCE_CEILING_DERIVATION.md`](ROUTE2_V0_STATIC_SOURCE_CEILING_DERIVATION.md)
> and gives the opposite result: by the envelope theorem
> `δG* = ⟨q*, δv₀⟩`, so `ε_F^max = τ / (2κρ|ΔG_pol|)` with `κ, ρ ≥ 1`. Even at
> `κ=ρ=1` and with the route's entire 1.5 kcal/mol budget spent on this one
> term, the derived ceiling is **0.1129** — the inherited `0.20` is **1.77×
> too loose**, and a source sitting exactly at it could carry 2.657 kcal/mol
> of polarization error on its own.
>
> `reject-gfn2-molden-permanent-source` therefore **stands**, now on a derived
> criterion and robustly for every `κ ≥ 1, ρ ≥ 1` (GFN2 exceeds the derived
> ceiling by 1.95×). The salvage in S1 is not GFN2. It is that the `v₀` search
> has been screening against a criterion ~2× too permissive to reach the
> route's own target, so any candidate cleared at `0.20` was never actually
> cleared.
>
> The deeper finding is that the gate cannot decide its own question: the
> deciding quantity is the single inner product `⟨q*, δv₀⟩`, and the frozen
> artifact stores only two norms, which bound it across a factor-of-κ range
> but cannot evaluate it.

Until 2026-09-04 nobody had propagated any `v₀` candidate through `Q` to a
kcal/mol number, so the route had never tested the thing it says it cares
about. That propagation is now done and registered.

**Status of the recovery (steps 1-2 complete):**

1. ~~Derive the static-source ceiling from a declared `ΔG_solv` tolerance.~~
   **Done** — equation (4) of
   [`ROUTE2_V0_STATIC_SOURCE_CEILING_DERIVATION.md`](ROUTE2_V0_STATIC_SOURCE_CEILING_DERIVATION.md),
   executable as
   [`benchmarks/derive_static_source_ceiling.py`](benchmarks/derive_static_source_ceiling.py).
2. ~~Re-adjudicate the recorded GFN2 norms.~~ **Done — rejection upheld**,
   by 1.95× against the most permissive licensable ceiling. No new QM was run;
   the static MEP was recovered from the frozen ± field records
   (`V(+E)+V(-E) = 2V(0)+O(E²)`, six reconstructions agreeing to 2.3e-6
   hartree/e).
3. **Open:** re-screen every other `v₀` candidate against equation (4), and
   register `⟨q*, δv₀⟩` rather than norms for all future candidates. The `v₀`
   search continues — against a target roughly 2× tighter than the one that
   has been in use.

The salvage here is not a recovered candidate. It is that the search criterion
was quantitatively wrong in the permissive direction, which means the search
had no chance of delivering the route's accuracy target regardless of how many
candidates it screened.

---

## S2 — the 46 fail-closed confirmation records: a cavity-policy artifact

**Status: recoverable, and the published aggregate needs a disclosure.**

[`route2-legacy-exact-gto-confirmation123-diagnostic-v1.json`](benchmarks/route2-legacy-exact-gto-confirmation123-diagnostic-v1.json)
publishes `mae_kcal_mol = 1.2432` over 77 records and records that 46 of 123
were fail-closed:

> PCMSolver emitted a warning for every cavity attempt under policy
> `intrinsic-smd-probe0-noaddsph-v1`; Route 2 refused to publish a numerically
> suspect result.

**New finding: those 46 rejections are not independent of solute class.**
Reproduce with `python3 benchmarks/analyze_confirmation123_subsample_bias.py`:

```
size_bin       chi2 = 7.20  dof=2  crit(5%)=5.991  -> dependent
element_class  chi2 = 8.85  dof=2  crit(5%)=5.991  -> dependent

              fail_rate   surviving_MAE
  small          0.26         0.786
  medium         0.36         1.408
  large          0.65         1.881

  halogen        0.13         1.177
  heteroatom     0.39         1.290
  hydrocarbon    0.54         1.165
```

The fail-closed rate rises monotonically with size **and** the surviving MAE
rises monotonically with size. The dropped records are concentrated in exactly
the bins where the model does worst, so the surviving 77 are a subsample
enriched in the easy cases.

This is the signature of the declared cavity policy, not of physics: `probe=0`
with no added spheres leaves GePol unable to fill inter-sphere crevices, and
crevice count grows with molecular size and chain flexibility. Hydrocarbons —
long chains of similar-radius spheres — fail most (0.54).

A conservative stratified re-estimate, imputing each fail-closed record the
mean error of the *survivors in its own bin*, gives `MAE ~ 1.30` by size bin.
The true value is likely worse: a solute that defeats the cavity builder is
not an average member of its bin.

**Two consequences:**

1. The published `1.2432` is a subsample statistic and should not be quoted as
   the panel's MAE without this disclosure. (The frozen artifact is immutable
   and correctly stated its failure count; the disclosure belongs here.)
2. The confirmation gate requires `failure_rate == 0`. At 37.4% this run fails
   on failure rate alone, **before accuracy is considered** — and that failure
   rate is a meshing policy, not a model property.

**Recovery action:** re-run the panel on the pyddx ddCOSMO/ddPCM path. ddX does
not build a GePol tessellation at all, so this specific crevice failure mode
cannot arise there; consistent with that, the frozen-source direct-PCM run
published 12/12 records with zero provider rejections
([`route2-frozen-source-direct-pcm-freesolv12-ddpcm-ddcosmo-v1-execution-ae427ea7.json`](benchmarks/route2-frozen-source-direct-pcm-freesolv12-ddpcm-ddcosmo-v1-execution-ae427ea7.json),
`ddcosmo MAE 1.0723`, `max 1.9549`) — a 12-record panel, so this is supporting
evidence rather than proof of full coverage. Re-running confirmation-123 there
recovers the 46 records and gives the first unbiased estimate on that panel.

---

## S3 — FC-aSWIG force-v3: sound, verified, and buried

**Status: nothing to repair. This is a packaging problem.**

From [`VALIDATION_STATUS.md`](VALIDATION_STATUS.md) and
[`route2-fc-aswig-force-v3-release-evidence-v1.json`](benchmarks/route2-fc-aswig-force-v3-release-evidence-v1.json):

* worst central-difference force error `5.42e-6 eV/angstrom`;
* rigid rotation / translation discrepancies `1.37e-14` / `1.74e-14 eV/angstrom`;
* closed-loop work `-5.48e-12 eV`; 20-atom torsion two-coordinate loop
  `-2.84e-12 eV`;
* NVE drifts `8.99e-7 / 2.24e-7 / 5.59e-8 eV` at `0.1/0.05/0.025 fs`, ratios
  `0.2496` and `0.2490` — clean second order;
* a clean current-head replay through the public
  `CommandControl -> SetCalculator -> ASE get_forces()` path.

That is a conservative force, demonstrated to the standard the route set for
itself. Its scope is correctly stated as water-only and per-geometry gated.
The waste is that it reads, in the surrounding prose, like another rejection.
It is the route's only shipped force capability and should be presented as a
bounded working feature rather than a caveat.

---

## S4 — `cosmors_torch`: the relative-quantity gate is already written down

**Status: recoverable; the escape hatch exists and is unimplemented.**

The Torch COSMO-RS layer reproduces upstream `openCOSMO-RS_py@3db6614` against
two frozen oracles and is differentiable end to end. Its declared blocker is
that MACE-EF generates the surface, whereas open24a was parameterized on
BP86/def2-TZVPD COSMO surfaces — "an executable new model identity, not
numerical openCOSMO-RS-24a equivalence".

But [`MACE_EF_COSMO.md`](MACE_EF_COSMO.md) already states the resolution:

> those terms cancel from a fixed-geometry relative KSE

Systematic surface-convention error is common-mode in `ΔΔG_solv` and in
`ln(k_target/k_reference)`. The KSE application only ever needed the relative
quantity. There is no validator for it, so the layer is scored against an
absolute gate it was never going to pass and is marked `diagnostic_only`.

**Recovery action:** implement a relative-quantity (T1) validator —
`ΔΔG_transfer` and relative KSE against published transfer data — and score
`cosmors_torch` there. This is the shortest path in the repository to a real,
defensible number.

---

## D1 — legacy exact-GTO accuracy: not recoverable, close it

**Status: dead. No post-hoc correction helps. Recorded so it is not retried.**

On the 77 published confirmation-123 records:

* mean signed error `+0.0373 kcal/mol` — there is no offset to remove;
* `corr(signed_err, cds) = -0.163`, `corr(signed_err, electrostatic) = +0.198`,
  `corr(signed_err, cavity_tesserae) = -0.089` — no exploitable structure;
* the best linear correction in tesserae (a surface-area proxy) moves RMSE from
  `1.5701` to `1.5634` and leaves `max|resid| = 3.88`.

The error is essentially isotropic scatter of ~1.57 kcal/mol RMSE. No scaling,
offset, or area-linear correction rescues it — which is also why proposing one
would have been target fitting for no gain.

Experimental uncertainty does not explain the failures either. Of the 24
records at or above 1.5 kcal/mol, exactly **1** lies within `2*sigma_exp`
(62 of 77 records carry `sigma_exp = 0.6`). The gate sits at roughly
`2.5 sigma_exp`, which is defensible; the misses are real misses.

Conclusion: the legacy nonvariational point-multipole-source / exact-GTO-receiver
fixed point should be retired as an accuracy candidate rather than re-tuned.
Its value from here is as a control, which is how the register already treats it.

---

## Summary

| item | blocked by | recoverable | next action |
| --- | --- | --- | --- |
| S1 V0-RK KKT (`v₀`) | protocol — borrowed ceiling was 1.77× too loose | **criterion fixed; search open** | ceiling derived and GFN2 rejection upheld; re-screen other candidates, register `⟨q*, δv₀⟩` |
| S2 46 fail-closed records | implementation — GePol `probe=0` policy | **yes** | re-run panel on the ddX path; disclose subsample bias |
| S3 FC-aSWIG force-v3 | presentation | **n/a — already works** | present as a bounded feature |
| S4 `cosmors_torch` | protocol — no relative-quantity gate exists | **yes** | implement the T1 relative/KSE validator |
| D1 legacy exact-GTO accuracy | physics — isotropic ~1.57 RMSE scatter | **no** | retire as accuracy candidate; keep as control |

Three of the four recoverable items are blocked by a gate that was never
derived from the quantity the route cares about. None requires new training,
fitting, or a weakened threshold.

**S1 is now closed as a criterion problem and reopened as a search problem.**
Deriving the ceiling did not recover a candidate — it showed the gate had been
1.77× too permissive, so the rejection stands and the target is tighter than
anyone was aiming at. That is still the most valuable of the four: it is the
difference between a `v₀` search that can succeed and one that cannot.
