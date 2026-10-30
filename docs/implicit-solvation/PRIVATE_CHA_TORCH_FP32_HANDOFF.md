# Validate and hand off the private CHA FP32 route

Use this procedure for a local experimental installation. It does not authorize
public distribution, automatic merging, or use outside prepared neutral
closed-shell, nonperiodic water single points with zero salt.

## Keep the installation separate

Install a matched MAPLE wheel and private-engine wheel into the existing local
test environment. Verify their hashes against the candidate build record before
installing. Stop every process using that environment before replacing a wheel.
Do not use an editable private-engine installation or import its checkout with
`PYTHONPATH`.

The private package must retain `NOTICE.md`, `SOURCE_MAP.json`, and its GPL and
LGPL license files. Keep model weights, prepared inputs, private source, and raw
validation artifacts outside Git. Neither the private engine nor model weights
belong in the public MAPLE wheel.

Run the input example in [the backend reference](PRIVATE_CHA_TORCH_FP32.md).
The output must identify `PrivateTorchChaGB`, the selected device,
`torch.float32`, `torch-fp32-sp-v1`, named energy components, and the energy-only
capability. Omitting `#device` must select CPU. Explicit unavailable CUDA must
fail without computing the result on CPU.

## Verify the frozen candidate

1. Preserve the existing worktree changes and historical artifact hashes.
2. Run focused source regressions and inspect all skips and failures. Record
   source tests separately from installed-wheel execution.
3. Freeze one reviewed, clean commit. Record the public and private source
   inventories, model profile, wheel hashes, and input identities.
4. Test installed CPU and CUDA SP inputs, device aliases, explicit `auto`,
   unavailable CUDA, unsupported devices, and invalid device syntax.
5. Evaluate every case in the frozen 30-case panel. Check polar, cavity,
   dispersion, and total separately against the native reference with the
   unchanged `0.005 kcal/mol` implementation-compatibility bound.
6. Check fresh-process repeats and deliberate small chunks against the unchanged
   `1e-5 kcal/mol` stability bound. A failure stays visible even when the ordinary
   implementation-compatibility check passes.
7. Profile the installed CUDA computation separately from timing. Require zero
   unallowlisted device-to-host transfers and scalar synchronizations.
8. Measure the same installed engine and frozen inputs in alternating CPU and
   CUDA blocks. Record cold startup separately from warm prepared-input calls,
   synchronize CUDA, check every output, and report background machine load.
9. Review the complete candidate independently. Preserve unresolved findings
   and keep activation closed rather than changing tolerances or adding fallbacks.

The local campaign uses
`.omx/checkpoints/route1-torch-fp32-dual-device-20261030/` under the outer Route 1
workspace. Its `closeout-20261030/` directory contains commands, raw logs,
installed checks, and the terminal handoff report. These local artifacts are not
part of the wheel. A handoff to another machine must include the permitted
artifacts separately and must not imply permission to redistribute private code.

## Preserve existing OBC-II analytic admission

The CHA addition changes shared runtime source files. An old OBC-II analytic
qualification receipt cannot authorize the changed installation.

For a clean candidate commit, generate fresh ANI exact-cell evidence, complete
OBC-II core validation, and AIMNet2025 CPU and CUDA qualification evidence. Use
the existing candidate runners and validators under `tests/solvation/` and
`docs/implicit-solvation/benchmarks/`. Their denominators and tolerances remain
unchanged. Seal a new receipt only when all artifacts bind the same source
snapshot. Rebuild the wheel and verify installed admission after sealing.

An absent or stale receipt must leave analytic admission closed. Copying the
historical receipt, restoring its pointer, or bypassing the source inventory is
not a repair. Qualification of OBC-II does not qualify the private CHA route for
forces or advanced tasks.

## Keep later parameter changes identifiable

`torch_runtime.ExecutionPolicy` owns execution budgets and chunks.
`exact_sp_inputs.ExactSpProfile` and the prepared topology describe scientific
inputs. Changing execution settings must not silently change physical constants.
Changing a physical parameter requires a new identified candidate and fresh
validation. Do not reuse a release seal or accuracy claim from the old parameter
set.

Preserve comments that document equations, units, assumptions, and provenance.
Do not duplicate the CPU and CUDA scientific chains to accommodate tuning.

## CHAFP32-001: strict chunk-stability discrepancy

Candidate-v4 completed all 30 GPU cases, but its deliberate small-chunk run
changed polar and total energies by up to `1.52587890625e-5 kcal/mol`. The frozen
stability limit is `1e-5 kcal/mol`. The original result is retained in
`GPU_V4_COMPATIBILITY.json` in the local campaign.

This is a numerical implementation-stability issue, not a measured experimental
accuracy loss. Its computational cause is not established by that comparison
alone. The integration owner must retain the failed cases and exact chunks,
rerun the unchanged gate after any justified repair, and report the terminal
disposition in the local handoff report. Passing a broader compatibility bound
does not close this issue.
