# Positive-parent PH1.0 Pro contract audit — 2026-08-17

## Scope

This review freezes the last development-only `MAPLE-CDS-W1` estimator for the
positive Bernstein parent area. It does not reopen the terminal stock-area M3
or M4 lanes, and it does not open the MNSol confirmation partition.

## Verified external review

The retained Windows Chrome surface was checked before submission:

```text
composer = Pro
power    = Pro, 5 of 5.
```

Generation was observed in the new conversation and completed before response
extraction.

```text
conversation:
  https://chatgpt.com/c/6a822a40-5ee4-83e8-84b8-8d2bbd86b973
conversation control:
  conversation-options-WEB:175b2fb8-1189-4217-8010-0aa995d191e6
prompt SHA256:
  c4a09e46142a8aef205c209ed0e16c48d60574e5c9212753f8ae8f63e5d18073
submission evidence SHA256:
  3db4077018f2468b528164a90e2b84bd9aa8f711d3ceaf1a8a927eaa399998f5
answer SHA256:
  c76a2eca4300f6019acbbb1c16801c46904bd652eda905919a756fe1b62cc965
answer marker:
  FREEZE PH1.0 CONTRACT
```

The prompt disclosed the complete target-blind `306 x 18` positive-parent
matrix spectrum and the prior terminal rules. It did not disclose private
checkpoint contents or confirmation values.

## Accepted mathematics

For each fold, only training geometries define the diagonal column metric, the
stock direction, and the leading two-dimensional complement projector. A
basis-independent spectral projector is converted to a deterministic frame by
choosing the lexicographically first tied maximum `2 x 2` principal minor and
using its positive-diagonal Cholesky factor. The same construction is frozen on
all 306 geometries for a possible deployment refit.

The pseudo-Huber loss remains fixed at `delta = 1 kcal/mol`, with three
parameters, no intercept, and no regularizer. Solver status is not an
acceptance condition. A result is accepted only when two deterministic
Newton-Cholesky/Armijo replays independently satisfy the strict minimizer-ball
certificate

```text
||gradient||_2 < 0.5 * m * R
```

where `m` is a conservative Hessian lower bound throughout the prediction ball
and `R` gives at most `1e-6 kcal/mol` row prediction uncertainty. The
zero-start endpoint is authoritative; the SVD least-squares start is a replay
check, not a model selector.

## Local critical adoption

The coefficient geometry, projector construction, ball proof, and deterministic
Newton contract were independently rederived before implementation. Target-free
canaries on the new matrix show every fold has all 18 active columns, a well
separated second geometry mode, a nonzero second-to-third spectral gap, and a
three-column optimizer condition number near four.

The Pro answer specified the water grouped-OOF gate. MAPLE additionally retains
the already documented project-level mixed-505 gate: water uses the PH1.0 OOF
prediction and nonaqueous records retain frozen M1. Both conservative numerical
MAE upper bounds must be at most `1.5 kcal/mol`. This is a prospective
strengthening inherited from `GOAL.md` and the earlier M4 contract, not a rule
selected after seeing PH1.0 targets.

The external answer is advisory. Repository code, frozen preregistration,
source hashes, exact row/fold binding, local unit tests, and the first real run
remain authoritative.

## Terminal boundary

Any pre-target subspace failure, fold certificate failure, replay failure,
representation mismatch, incomplete OOF coverage, water or mixed-505 MAE upper
bound above `1.5`, or deployment refit failure terminates this hybrid accuracy
route. No additional delta, modes, intercept, regularizer, feature edit, fold
edit, or replacement CDS candidate is permitted. A pass also closes
development; the aspirational `1.0 kcal/mol` target does not authorize another
candidate.
