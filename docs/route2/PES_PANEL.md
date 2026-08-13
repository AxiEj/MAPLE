# Route 2 preregistered PES panel

This document freezes and records the first fixed-box40/CPCM590 real-stack PES
panel. It does **not** admit Tier E/F/H/V/M and it does not claim chemical
accuracy or a complete experimental solvation free energy.

## Scalar and domain

Every point uses the single registry entry
`route2-operational-cpcm-fixedtopology-electrostatic-v1`:

```text
E_op(R) = E_vac(R) + 0.5 <c*(R), P_R c*(R)>_Q
G_np = 0
```

The checkpoint adapter declares neutral singlets only. Consequently the panel
contains 20 neutral singlets and **no charged molecule is fabricated**. A
charged case remains required if a future checkpoint profile explicitly
supports it.

The geometry asset is
`tools/route2_release/data/fixedbox590_pes_panel_v1.json`, SHA256
`ecaa309cb468f17449a78a16e6a79acd0a5a16cf982eb52b3fb8d5d1dcd80ca3`.
Its source is the ASE 3.27.0 G2 collection, copied to exact arrays so execution
does not query or regenerate external geometry data.

## Frozen coverage

The 20 molecules are water, methanol, ethanol, acetone, acetonitrile, benzene,
methane, trans-butane, dimethyl ether, formic acid, acetic acid, acetaldehyde,
acetamide, ethylamine, pyridine, nitromethane, hydrogen peroxide, thiophene,
methanethiol, and chloroform.

Each molecule has three preregistered geometries: reference, one deterministic
covalent-bond compression, and the corresponding stretch. Each geometry is
tested along three deterministic translation-free directions at central
steps `4e-4`, `2e-4`, and `1e-4 Angstrom`.

Two actual coordinate paths—not metadata labels—are additionally frozen:

1. trans-butane C0-C1-C2-C3 torsion at 180, 150, 120, 90, and 60 degrees;
   the 60-degree endpoint supplies the flexible close-contact control;
2. hydrogen-peroxide O-O bond changes of -0.16, -0.08, 0.00, +0.08, +0.16,
   and +0.32 Angstrom; this is a nonstationary reaction-coordinate surrogate
   containing compressed, stretched, and TS-like geometries. It is **not** an
   optimized transition state and cannot satisfy the TS canary by itself.

Each path point is also checked with the same three Cartesian step sizes along
its normalized coordinate tangent, so the preregistered force error remains in
`eV/Angstrom`. The unnormalized tangent and independent generalized derivative
(`eV/radian` for the torsion, `eV/Angstrom` for O-O stretch) are retained in the
raw artifact and are never mislabeled as Cartesian force errors.

## Frozen gates and execution

Thresholds are exactly those in `VALIDATION_PROTOCOL.md`: directional absolute
error at most `5e-4 eV/Angstrom`, relative error at most `2e-3` away from the
`1e-3 eV/Angstrom` floor, primal residual at most `1e-12`, adjoint residual at
most `1e-10`, and cold/warm energy/source differences at most `1e-8`.

The sharded clean-tree runner is:

```bash
python tools/route2_release/run_fixedbox590_pes_panel.py \
  --molecule-start 0 --molecule-stop 1 \
  --output /absolute/path/pes-panel-shard-00-01.json
```

Each artifact binds Git head/tree, loaded-source hashes, checkpoint hash,
runtime, warning ledger, identities, raw roots, energies, forces, path points,
and topology hashes. A shard cannot aggregate or enable a capability. Only a
separate complete-panel verifier may interpret all 20 molecules and both
paths. The verifier recomputes analytic projections from raw forces, displaced
geometry hashes from the frozen asset, finite differences from raw energies,
root/residual gates, and topology coverage; it also requires one exact Git
tree, checkpoint, device/dtype, package, accelerator, and thread signature.
Failed gates must be preserved rather than retuned.

## Executed result

All 20 source-bound CUDA/float64 shards were executed on clean Git head
`f7f681657435ed114c34d57b43ab792162f825b9`. The independent aggregator
recomputed every analytic projection from raw forces, every central difference
from raw displaced energies, all frozen-geometry hashes, cold/warm root gates,
primal/adjoint residual gates, and topology coverage. It returned `status=pass`.

The aggregate covers 20 molecules, 60 reference/compressed/stretched
geometries, 180 directional records (540 step samples), and 11 additional path
geometries (33 step samples). The worst base-geometry errors were
`2.98256e-5 eV/Angstrom` absolute and `1.25795e-3` relative where the relative
gate applied. The worst path error was `2.40732e-5 eV/Angstrom`. Maximum primal
and adjoint residuals were `9.98924e-13` and `2.46415e-12`; maximum cold/warm
energy and normalized-source differences were `9.09495e-13 eV` and
`4.29641e-12`.

The immutable evidence bundle is
`evidence/fixedbox590-pes-panel-f7f68165/`; its independent aggregate file has
SHA256 `e087f6297d709f3d383b3bb8bfff7fd75dbcb4e4022020364d6a273f744c3775`.

This closes the preregistered **directional** panel only. The required
component-resolved Cartesian FD panel, remaining symmetry/loop scope,
component physics, Hessian/FREQ/TS/HVP/NVE, and public admission gates remain
open. Its separate pre-execution contract is frozen in `CARTESIAN_PANEL.md`.
OPT, NEB, TS, FREQ, and MD therefore remain unavailable.
