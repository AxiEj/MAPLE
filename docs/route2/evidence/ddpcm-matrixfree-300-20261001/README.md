# Exact matrix-free ddPCM research checkpoint

This is a source-and-evidence checkpoint, **not production integration or scientific admission**. It preserves the research performed against baseline commit `57effbb51c3442d82ca6a77d1542c490d5fee595` without changing the original runtime, frozen receipts, or numerical gates.

## Status

- Completed: exact CPU FP64 continuum `L`, `D`, and their transpose actions for **300 synthetic centres**, at `lmax=15`, 1202 Lebedev points, `eta=0.1`, source tile 8, with no FMM.
- Final source-bound run: **424.2577 s**, peak process RSS **985,907,200 bytes**, maximum transpose-pairing relative error **4.4943e-15**. This is an operator benchmark, not a complete energy/force calculation.
- Current research suite: 144 passed; unchanged operator/response/topology suite: 112 passed. Fresh pre-push repetitions are included separately.
- Still unavailable: 300-centre solve certification, deterministic replacement stability certificate, full 300-atom MACE-POLAR/ddPCM/CDS energy, forces, HVP, Hessian, chemical-domain expansion, and a public input profile.
- **Adaptive memory scheduling is discussed but not implemented.** The archived candidate retains fixed research budgets. Optional compiler experiments are not qualified as the selected backend.

Read the [full report](snapshot/REPORT.md), [independent endpoint review](snapshot/FINAL_REVIEW.md), and [final candidate audit](snapshot/accounting-final-audit.json). Historical failures and earlier candidates are intentionally retained and are not relabelled as current passes.

## Archive contract

`snapshot/` contains byte-identical copies of selected text/source evidence from `.omx/research/ddpcm-matrixfree-300-20261001T062415Z`. `plans/` contains the original bounded roadmap, test specification, and native review record. `ARCHIVE_MANIFEST.json` records each included relative path, SHA256, size, and original location.

The original `.omx` directory is untouched. Model weights, compiler caches, bytecode, binary profiling data, duplicate driver logs, credentials, and general OMX session/runtime files are not included. Historical absolute paths and local environment/version details in scientific receipts are preserved as provenance, not rewritten to pretend that the archive ran elsewhere.

The scripts and receipts deliberately pin the baseline HEAD and source manifest. **Do not run the archived scripts directly from this documentation directory or change the pinned baseline to the checkpoint commit.** Publishing this checkpoint changes HEAD; that does not invalidate historical evidence, but it means direct replay must use the original baseline in a separate worktree. Future implementation work needs a new versioned candidate and fresh source-bound receipts.

## Restore for independent replay

Prerequisites: an already provisioned compatible `maple` Python environment (including Torch and pyddx); Linux user systemd/cgroup v2 for the guarded resource runner; an independently provisioned official local checkpoint for the full V2 identity replay. No model is redistributed here and these commands do not download one.

From a checkout containing this archive:

```bash
ARCHIVE="$PWD/docs/route2/evidence/ddpcm-matrixfree-300-20261001"
REPLAY="$(mktemp -d /tmp/maple-ddpcm-replay-XXXXXX)"
git worktree add --detach "$REPLAY" 57effbb51c3442d82ca6a77d1542c490d5fee595
mkdir -p "$REPLAY/.omx/research" "$REPLAY/.omx/plans"
cp -a "$ARCHIVE/snapshot" \
  "$REPLAY/.omx/research/ddpcm-matrixfree-300-20261001T062415Z"
cp -a "$ARCHIVE/plans/." "$REPLAY/.omx/plans/"
cd "$REPLAY"
D="$PWD/.omx/research/ddpcm-matrixfree-300-20261001T062415Z"
export PYTHONPATH="$PWD:$D"
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

python "$D/research_receipt.py" \
  --test-file test_streamed_ddpcm.py \
  --test-file test_fused_value_operator.py \
  --test-file test_native_source_adjoint.py \
  --test-file test_research_identity.py \
  --test-file test_cgroup_probe.py \
  --output "$D/independent-replay-tests-NEW.json"
```

Use a new output name for every replay. Run expensive scale probes only after tests and the preceding scale/time preflight pass, under the documented OS cap and watchdog. See the full report for the guarded probe command and its operator-only interpretation. The numerical receipts' original paths and timestamps remain historical; newly generated receipts constitute new evidence.
