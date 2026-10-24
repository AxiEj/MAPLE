# Retired AniSolv records (2026-10-24)

These two cards are byte-for-byte historical records, outside the runtime
package. AniSolv compact and its sealed UMA wrapper are retired, not repaired
or aliases for AnySolv-MoE. Neither old selector is executable in current MAPLE.
Use the [current AnySolv instructions](../../CONSOLV_ANYSOLV.md).

| Historical card | SHA256 |
| --- | --- |
| [anisolv-compact.yaml](anisolv-compact.yaml) | `be9bb6994e1b2e8d313fdd1896d492c33d629435e7091b70692ebd9f5b20fdf8` |
| [anisolv-uma.yaml](anisolv-uma.yaml) | `098181a96a2685af888806d8d9c5be53bc60d45630b0df9330dfc1c223cc997f` |

Frozen evidence is retained unchanged:
- [Six-row upstream sample](../../benchmarks/anisolv-compact-upstream-sample-audit-2026-07-30.json)
- [Force-consistency rejection](../../benchmarks/anisolv-compact-force-consistency-audit-2026-07-30.json)
- [GPU no-loss rejection](../../benchmarks/anisolv-compact-gpu-no-loss-audit-2026-07-30.json)

For historical replay, use a separate checkout of MAPLE source commit
`8c27ed38` and its original `docs/pretrained-solvation-hub/README.md` commands
for `run_anisolv_compact_{sample,force_consistency,gpu_no_loss}_audit.py`.
That source retains the runners, runtime and dependency/authorized-checkpoint
requirements. Replay is not current scientific admission. No checkpoints are
copied here; compact checkpoints must never be remapped to MoE.
