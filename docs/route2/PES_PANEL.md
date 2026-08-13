# Route 2 preregistered PES panel

This document freezes the first fixed-box40/CPCM590 real-stack PES panel before
its clean execution. It does **not** admit Tier E/F/H/V/M and it does not claim
chemical accuracy or a complete experimental solvation free energy.

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
paths; failed gates must be preserved rather than retuned.

## Current status

The contract and runner are implemented; the complete panel has not yet been
executed and no result is reported here. Until complete clean evidence passes,
OPT, NEB, TS, FREQ, and MD remain unavailable for this profile.
