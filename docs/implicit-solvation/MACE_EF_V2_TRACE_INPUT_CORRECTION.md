# Existing V2: source-only external-potential correction candidate

## Status

The user requested handling the existing `macepol-ef-v2.pt` directly. This
follow-up no longer requires recovering the missing original eager object
before performing a **modified-traced-program** experiment. It does not claim
eager restoration, the original exporter invocation, or production admission.

The released checkpoint and all prior artifacts remain unchanged. No training,
parameter fitting, cavity change, response rescaling, output-gradient
projection, or production-adapter/spec change was performed.

## Exact change and identity

Only the archive member
`best_ep9-traced/code/__torch__/mace/modules/extensions.py` is changed. The other
**770 member payloads**, including tensors and constants, remain byte-identical.
Loaded parameters/buffers have the same digest before and after execution.

Both internal scalar projection sites receive

\[
V_i^\circ=V_i-\overline V_{\mathrm{graph}},
\]

instead of the field norm. The existing matrix, gradient-slot ordering, spin
half-factor and feature normalization are retained. The terminal `q_i V_i`
term still uses **raw** V. If the auxiliary charge constraint holds, this gives
`E(V+C)=E(V)+Q*C`; autograd differentiates the complete altered energy. This is
an input-gauge convention, not a post-hoc repair of the conjugate source.

The graphwise mean and all five arms were specified before altered GPU
measurements. The ordinary centroid-based affine potential already has mean
zero. General batched/ghost execution and nuclear-coordinate derivatives are
not qualified by these molecular tests.

- Parent checkpoint SHA-256: `4f820d381d06bbb37b02574c38da7203e5d429407fa2a512231fc08cdbb69b6b`.
- Primary candidate SHA-256: `0cac472aa566363a00f5f3fbc41e7fb8bc5e1f5bb8aa8cba654ebac3fe4dc902`.
- Shared loaded-state digest: `d31ed9f905dbb4a169e15732b9d1c233d9ca5063827c208e578035a9ade25b7c`.
- Local candidate: `.omx/artifacts/mace-ef-v2-trace-inputs-20260905/final/centered_v.pt`.

This file must not overwrite the released model or be presented using its old
checkpoint identity. Existing public loaders remain unchanged and will reject
the candidate under the original V2 hash contract.

## Five-arm CUDA result

[Protocol](benchmarks/mace-ef-v2-trace-input-ablation-prereg-v1.json),
[driver](benchmarks/audit_mace_ef_v2_trace_inputs.py), and
[raw result](benchmarks/mace-ef-v2-trace-input-ablation-v1.json).

The no-op ZIP repack changed no member payload. Its maximum energy discrepancy
was zero; maximum source and density discrepancies were about `1.79e-7`.
All no-op controls passed before any altered-arm GPU evaluations. Every
derived-program load verifies the complete bytes against its receipt and then
loads from immutable `BytesIO`, rather than trusting a mutable file path.

| Arm | Water/methanol sampled affine curvature | Charge/gauge checks | Interpretation |
| --- | --- | --- | --- |
| original | positive and approximately 1/h at zero | pass | unchanged negative control |
| roundtrip | reproduces original | pass | packaging control |
| zero scalar | negative, step-stable | pass | identifies the norm substitution's role; still no internal V response |
| raw V | negative, step-stable | **fail** conjugate charge | direct absolute-V injection is insufficient |
| centered V | negative, step-stable | pass | preregistered primary correction candidate |

At the smallest affine step `1.25e-4 eV/(e Angstrom)`, the primary candidate's
energy-Hessian spectra are approximately `[-0.1099,-0.1086,-0.1048]` for water
and `[-0.2244,-0.1688,-0.1654]` for methanol, versus positive values around
`30.01` and `71.92` for the unchanged model. Raw matrices, gradient sides and
antisymmetric residuals are retained; no eigenvalue was clipped.

The raw-V arm's maximum conjugate-charge errors are about `0.00389 e`,
`0.00913 e`, and `0.7093 e` for water, methanol, and the water-cation gauge
canary. Centering reduces these measured errors to at most `1.20e-7 e`.
Only scalar V was perturbed in a separate probe; the centered candidate's
auxiliary density changes by about `1.85e-6` and `3.22e-6` for the neutral
molecules. The true potential now enters the internal response, not just qV.
The charged canary tests `Q*C` covariance only, not ionic solvation or TS accuracy.

## Actual conductor reaction-potential direction

[Protocol](benchmarks/mace-ef-v2-cosmo-direction-prereg-v1.json),
[driver](benchmarks/audit_mace_ef_v2_cosmo_direction.py), and
[raw result](benchmarks/mace-ef-v2-cosmo-direction-v1.json).

For each neutral geometry, the frozen FreeSolv-20 v3 conductor configuration is
matched exactly. Its reaction jet generated from the released zero-field
energy-conjugate source is held fixed for both programs. The continuum
half-coupling identities agree to below `1e-15 eV`.

Along `field(t)=t*reference_reaction_jet`, the centered candidate's curvature is
negative and resolution-stable near both t=0 and t=1. Near zero it is about
`-0.1562 eV` for water and `-0.08065 eV` for methanol; the unchanged program's
zero-centered estimate grows positive as the step shrinks. These are **eV per
dimensionless t squared**, not the affine Hessian units. No affine numerical
threshold is reused for this different quantity.

All 14 center/finite-difference states per arm are stored and included in the
charge check. This is one real nonuniform-potential ray, not a complete
response-space or joint-stability qualification. No new stationary SCRF root,
full COSMO-RS panel, force/PES, or chemical-accuracy claim follows from it.

## Reproduction and remaining work

The trusted original V2 checkpoint and the existing CUDA project environment
are required. Use new output/work directories; neither driver overwrites:

```bash
PYTHONNOUSERSITE=1 python docs/implicit-solvation/benchmarks/audit_mace_ef_v2_trace_inputs.py --checkpoint ./macepol-ef-v2.pt --preregistration docs/implicit-solvation/benchmarks/mace-ef-v2-trace-input-ablation-prereg-v1.json --workdir /tmp/maple-v2-input-replay --output /tmp/maple-v2-input-replay.json
```

This archive transformation is deliberately specific to the pinned model. It
is not a supported general-purpose TorchScript conversion API. PyTorch's
[loader](https://docs.pytorch.org/docs/2.9/generated/torch.jit.load.html)
accepts saved programs; the
[tracing limitations](https://docs.pytorch.org/docs/2.9/generated/torch.jit.trace.html)
still apply to inherited control flow. Trace debug metadata is inherited, not
evidence of recovered eager code. Current measurements use PyTorch 2.12/CUDA
13.0; other runtimes have not been qualified.

Next: broader physically generated field directions and joint SCRF stability,
then an identity-bound paired FreeSolv-20/COSMO-RS comparison. Keep all current
scientific/public gates closed until the corresponding evidence exists.

Final targeted verification: **103 passed**, including 20 new trace/direction
tests, the actual released-checkpoint CUDA tests, prior P0 artifact checks,
conductor/COSMO-RS protocol tests and frozen profile replay. Black, Pyflakes,
Python compilation, both CLI help smokes and `git diff --check` passed. LSP/type
checking was unavailable; no whole-repository test claim is made. All previously
dirty files, prior frozen records, the released checkpoint and branch HEAD were
checked unchanged. No commit or push had been performed at measurement time;
later publication does not alter that execution provenance or admission scope.
