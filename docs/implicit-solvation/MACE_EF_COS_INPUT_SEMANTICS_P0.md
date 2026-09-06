# MACE-EF-COS P0: input semantics and unchanged-checkpoint response

## Scope and status (2026-09-05)

This is a diagnostic milestone on the existing `MACE-EF-COS` branch, not a
repaired-model result. No checkpoint, production adapter, continuum, COSMO-RS
parameter, public capability, or old frozen artifact was changed.

The experiment is preregistered in
[`mace-ef-cos-input-semantics-prereg-v1.json`](benchmarks/mace-ef-cos-input-semantics-prereg-v1.json).
The executable is
[`audit_mace_ef_cos_input_semantics.py`](benchmarks/audit_mace_ef_cos_input_semantics.py),
and the measured result is
[`mace-ef-cos-input-semantics-v2.json`](benchmarks/mace-ef-cos-input-semantics-v2.json).
The initial v1 result is retained unchanged, but v2 supersedes its imprecise
archive-layout labeling: the trace's scalar slot is zero then norm-overridden,
not a real `V` input. V2 explicitly distinguishes the inspected gradient slots
from the candidate's inserted `V`, without changing the preregistered experiment.
The new diagnostic records exact source-file hashes; its recorded base Git HEAD
does **not** imply that the new driver was committed at execution. It is not an
admission replay.

## What actually ran

- The exact existing 34,581,570-byte checkpoint, SHA-256
  `4f820d381d06bbb37b02574c38da7203e5d429407fa2a512231fc08cdbb69b6b`.
- Float32 on CUDA device 0, with neutral singlet water and methanol. Water uses
  the existing audit geometry. Methanol uses the committed FreeSolv-20 v3
  geometry, not a newly optimized conformer or an error-selected structure.
- Both parts of coherent affine potentials, `V_i=g dot (R_i-centroid)` and
  `grad(V)_i=g`, with the adapter's existing vector-sign convention.
- Five gradient-difference steps, from `2e-3` to `1.25e-4 eV/(e Angstrom)`,
  about zero and about the preregistered nonzero vector
  `[0.0013, -0.0007, 0.0009] eV/(e Angstrom)`.
- Raw gradient-difference matrices, both sampled gradient sides, diagonal
  derivative jumps, symmetric eigenvalues, and antisymmetric residuals.

The archive itself contains **two** per-atom field-norm computations and scalar
projection overrides. Its model-code member is
`best_ep9-traced/code/__torch__/mace/modules/extensions.py`. Thus the proxy
feature is confirmed in the actual deployed bytes, not only inferred from a
moving upstream source tree.

## Observations

### Zero-field curvature does not converge over the measured steps

The water baseline reproduces the previously recorded three-step ranges.
Extending to smaller steps gives approximately:

| Molecule | Symmetric eigenvalue range at h=5e-4 | Range at h=1.25e-4 |
| --- | --- | --- |
| water | 7.429–7.436 | 30.008–30.015 |
| methanol | 17.805–17.860 | 71.888–71.944 |

Units are `e^2 Angstrom^2/eV`; these are external-potential response curvatures,
not nuclear-coordinate Hessians. The axis derivative jumps approach about
`0.0075 e Angstrom` for water and `0.0180 e Angstrom` for methanol rather than
shrinking toward zero over this range. This supports a derivative-jump
interpretation; it does not identify a corrected polarizability, prove that the
norm is the only defect, or admit the model.

### Nonzero-field tests matter too

At the nonzero center and smallest tested step, the symmetric eigenvalues are
approximately `[-0.088, 2.077, 2.083]` for water and
`[-0.174, 5.028, 5.048]` for methanol. Positive curvature is therefore not
restricted to reporting a finite difference exactly across the zero-field
point. The tested finite-field response remains unsuitable as passivity proof.

The antisymmetric/symmetric Frobenius ratio at coarse nonzero-centered steps
reaches about 4–5%, falling to about 0.1% at the smallest step. These are raw
finite-difference errors, not proof of a nonconservative analytic energy:
the decrease is consistent with finite-step truncation effects. The result
shows why symmetrization must not erase the original matrix or replace a
resolution study.

### Projection equivalence is established only at the projection layer

The diagnostic reuses the installed native
`DisplacedGTOExternalFieldBlock.forward` and the matrix stored in the actual
checkpoint. Each local affine jet is supplied at its own origin. Archive
inspection separately verifies the gradient order `[z,x,y]`, the initial
**zero** scalar slot, and constant `c2=0.5`. The trace subsequently overwrites
the scalar feature with a field norm. The diagnostic candidate instead inserts
`V` into that scalar slot. Its local projection is compared against this
**archive-layout candidate linear convention**, independently of the installed
forward method. That comparison is not a claim that the trace already uses V;
no channel ordering is guessed from labels.

The installed forward's uniform-potential and local-affine projections agree
exactly in these two float32 canaries. This equality establishes the
local-origin/batching algebra under that implementation, not an independent
physical validation of Gaussian normalization or a full eager-model match.
The reconstructed archived proxy differs
by about `0.00494` and `0.00659` in raw half-projection features, respectively;
its scalar part is even under field reversal. These feature differences are
not errors in energy units and are not full-model eager comparisons.

The actual checkpoint's auxiliary density and conjugate charges change by at
most roughly `1e-7` in the measured nonuniform scalar-potential-only test; this
small numerical difference is not evidence of a physical potential response.
Constant-potential tests on these **neutral**
molecules show zero resolved energy shift. They do not qualify a charged-model
gauge domain. Total-energy subtraction is explicitly recorded as float32 and
is not used as a tight finite-difference certificate.

## Remaining asset gate: no proven original eager model

The checked local model locations contain the production trace and foundation
MACE-POLAR models, but no established matching eager EF model. The
[pinned upstream README](https://github.com/ClickFF/MACEPOL-EF/blob/e2b0aeed27c2822790a6994c9a64327e5f131158/README.md)
does not distribute its fine-tuned EF weights. A
[production evaluation example](https://github.com/ClickFF/MACEPOL-EF/blob/e2b0aeed27c2822790a6994c9a64327e5f131158/scripts/eval_macepol_0428.py#L29-L45)
points to a private `best_ep1.model`, which is not evidence that it produced
this archive with a `best_ep9-traced` prefix. The
[public exporter](https://github.com/ClickFF/MACEPOL-EF/blob/e2b0aeed27c2822790a6994c9a64327e5f131158/scripts/convert_polar_to_pt.py)
does not bind this local output checksum to an original eager checksum and
export invocation.

The required next asset is the **actual original `.model` (or equivalent
complete eager checkpoint/config) used to export this `.pt`**, together with
its export/source information. A name alone is insufficient; weights and
available buffers must be matched and baseline eager/trace parity must pass.
The ordinary MACE-POLAR-S checkpoint is not an acceptable substitute.

Accordingly, this milestone does **not** modify the serialized graph, claim
recovery of original Python control flow, export a repaired model, run new
conductor surfaces, or reopen COSMO-RS accuracy claims. Once the asset is
available, resume the planned same-weight eager A/B/C comparison; only after
that can the repaired input be evaluated on actual conductor reaction fields.

## Reproduction

Use the current project environment with its existing CUDA/PyTorch and native
GTO dependencies. The command refuses an existing output path:

```bash
PYTHONNOUSERSITE=1 python docs/implicit-solvation/benchmarks/audit_mace_ef_cos_input_semantics.py --checkpoint ./macepol-ef-v2.pt --preregistration docs/implicit-solvation/benchmarks/mace-ef-cos-input-semantics-prereg-v1.json --output /tmp/mace-ef-cos-input-semantics-replay.json
```

The new CPU regression tests cover smooth and cusp response examples,
antisymmetry preservation, native/local projection equivalence, potential
derivatives, invalid inputs, output non-overwrite, and the frozen artifact's
arithmetic/provenance boundary. They are separate from the actual CUDA evidence.

Final targeted verification: **83 passed** (including the 16 new audit tests,
the existing CUDA checkpoint tests, COSMO protocol tests, and profile-bundle
replay tests). Black check, Pyflakes, Python compilation, CLI help smoke, and
`git diff --check` also passed. No dedicated LSP/Pyright/mypy type check was
available; static validation used the checks above. The engineering test result
does not mean the checkpoint's physical passivity gate passed.

At the end of this measurement run, the three preexisting unrelated untracked
files matched their preflight SHA-256 values. No branch/worktree was created or
switched, and no commit or push had yet been performed. Later publication does
not turn this historical run into a clean admission replay.
