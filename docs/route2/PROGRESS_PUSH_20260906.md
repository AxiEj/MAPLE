# Hybrid research progress snapshot — 2026-09-06

This snapshot publishes the current hybrid implementation, tests, research
protocols, numerical assets and terminal research findings on
`research/route2-mdp-polar-hybrid-v1`. It is not a release/accuracy admission.

## Scope

- Experimental daily ddX SP, first-order OPT, vibrational-only FREQ and
  E/F-only NEB/CINEB, with explicit domain, units, derivative and endpoint
  checks. See `USER_GUIDE.md` and `evidence/hybrid-daily-workflows-20260905/`.
- Analytic hybrid derivative work, source/response representation studies,
  ADT diagnostics, scalar-first pilots, response-mode selection and CPKS
  protocol/coverage tooling. Failed candidates remain failed; this publication
  does not promote a replacement source, force model or accuracy claim.
- Existing checkpoint weights and the local research-run cache are not
  redistributed as part of this snapshot.

## Validation at publication

The complete selected development sweep on the live checkout, with
`PYTHONPATH` explicitly set to that checkout, returned:

```text
8 failed, 1156 passed, 7 skipped
```

All eight failures are in historical W1 CDS feature/stock-area aggregation
tests. The preregistered `GOAL.md` source binding no longer matches the evolved
working document. The frozen source hashes were **not** refreshed to hide this
failure. This snapshot therefore must not be described as a fully green
repository or a clean scientific admission replay.

Formatting checks also report retained CRLF/trailing whitespace in frozen
advisor transcripts and seven pre-existing extra EOF blank lines in bound
test/runner files. Those bytes were not normalized as a publishing side effect.

Command (from the repository root):

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD" OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m pytest -q -p no:cacheprovider tests/route2_vnext tests/calculator/test_mace_mdp_polar_hybrid_calculator.py tests/test_frequency_vibrational_only.py tests/test_hybrid_neb_workflow.py
```

The earlier scoped daily-workflow validation was 266 passed; its distinct
scope and real-Hessian/final-frequency-replay limitations remain recorded in
the 2026-09-05 evidence bundle. Neither test result asserts new experimental
solvation, frequency or reaction-barrier accuracy.

## Deliberate publication exclusions

- The file
  `evidence/hybrid-mdp-atomic-partition-gauge-20260817/full505-component-diagnostic-m0-m1-v1.json`
  explicitly declares `do_not_commit=true` and remains local.
- Raw browser/accessibility captures, account/sidebar-bearing capture state,
  browser submit receipts and local UI automation scripts remain local.
  Mathematical prompts/answers, decisions and numerical results are retained
  where safe to publish.
- Bytecode, top-level scratch output and `.omx` runtime/session files remain
  local. Nothing in this exclusion process deletes the original files.

Historical manifests retain the original hashes; they have not been rewritten
to make an incomplete public copy look like complete evidence. In particular,
the public tree does not contain the private `complete-uia.tsv` or its companion
browser-mode captures. The associated
`tests/route2_vnext/test_hidden_feature_decoder_pro_artifact.py` also remains
local rather than shipping a forensic check without its required private
inputs. Full browser-mode/forensic verification requires those separately held
originals. The live-checkout test counts above include that local check and
are not a claim that every artifact test passes from this public subset alone.
