# Harmonic Green-operator external-advisor audit — 2026-08-14

## Status and scope

This is an external mathematical design cross-check, not release evidence and
not a capability-admission artifact. Every accepted conclusion was rederived
locally and locked by repository tests before it was used in code.

## Invocation evidence

```text
advisor: ChatGPT in the retained Windows Chrome session
power selector: Pro
power menu: Pro, 5 of 5.
verified UTC: 2026-08-14T05:10:16.3116298Z
submitted UTC: 2026-08-14T05:11:11.8584965Z
Pro thinking observed: true
completion control: Worked for 45m 4s
final precise UI status: Stop answering=0, Response actions=2
```

Artifact digests from the executed session:

```text
prompt SHA256:
2cf216002811322cf8fc59ef6580126c1899be05c0e2ec2aaba2e801b15fc69d

verification record SHA256:
4e21aff38a9af57950c45a880b3c70030d6fde660cc897af1d1c08092ced2cf0

submission record SHA256:
f871189f42ba3a45f3e408e9e5d9c74b5108021b4834629c3989c24e373ce035

raw UI-Automation extraction SHA256:
a4c6d2fbb2b29b08e2b14e1e2bb463d3f33b8721edfef8f0a6e56ea22381654a
```

The raw UI extraction contained browser controls and duplicated accessibility
text, so it was used only to verify completeness and the mathematical sections;
it is not treated as a clean reference implementation.

## Decisions independently confirmed

1. The coherent weighted-shell conductor discretization is

   ```text
   A = E.T K E
   S = E.T V
   receiver = S.T
   ```

   where `E` is a rectangular exposure-product embedding into a larger complete
   harmonic bandwidth, `K` is the bare full-sphere Coulomb single layer, and
   `V` is the raw solute boundary-potential map.

2. An invertible square exposure sandwich cancels from the stationary energy
   and cannot be the physical exposure mechanism.

3. The exact self-sphere charge-per-solid-angle eigenvalue is
   `k_e * 4*pi / (a*(2*l+1))`.

4. For intersecting spheres, one source shell can be integrated analytically;
   the remaining pair-axis polar integral is finite and split at the
   intersection cosine. The canonical block commutes with the `SO(2)`
   stabilizer, so transverse pair-frame gauge cancels from the value.

5. For coordinate derivatives, a future implementation must use a gauge-free
   bipolar/STF reconstruction or an equivalent global intertwiner. It must not
   differentiate through an arbitrary pair-axis section.

6. Bare `K` is positive on independent physical shell densities. The weighted
   `A` is positive definite exactly when the weighted basis retains full
   physical rank. Full burial/rank changes and near-coincidence fail closed;
   no diagonal jitter or eigenvalue clipping is permitted.

7. Exact moving delta-shell Coulomb blocks are `C1` but not generally `C2` at
   internal/external tangency. A frequency/HVP domain must therefore guard
   tangencies even when the exposure field is `C-infinity`.

8. The resulting model is a regularized weighted multi-shell conductor model,
   not the exact sharp union-of-spheres PCM boundary.

## Local evidence that supersedes advisor trust

The implementation and tests named below are authoritative:

- `harmonic_exposure.py`: exact finite-band rectangular product embedding;
- `harmonic_gaussian_source.py`: `S=E.T V` and exact `S.T` receiver;
- `harmonic_single_layer.py`: exact self spectrum and all-overlap pair blocks;
- `harmonic_weighted_galerkin.py`: `A=E.T K E` stationary assembly;
- the four corresponding `test_harmonic_*.py` modules.

The advisor answer does not enable E/F/H/V/M, prove physical accuracy, or
replace the real-checkpoint/model-side conjugacy gates.
