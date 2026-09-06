# Iodine role-separated ADT full replay

This artifact closes one mechanism-level provider failure found by the
immutable canonical-ADT 505 development run. The v1 canonical free-atom
density registry did not contain iodine (`Z=53`), so development index 386
failed before a continuum state could be built.

The v2 role-separated contract uses the frozen 25-electron def2-ECP valence
pseudo-density only as the iodine **ADT response shape**. It does not use that
shape as a neutral penetration density. The geometry comes from the frozen
geometry-only artifact
`mace-mdp-polar-iodine-adt-pro-20260819/iodine_operator_geometry.json`; this
runner never loads MNSol or an experimental solvation target.

## Reproduction

```bash
python tools/route2_release/run_mace_mdp_polar_role_separated_adt_iodine_canary.py \
  --output /tmp/route2-iodine-role-separated-adt-full-replay-v1.json \
  --polar-device cpu \
  --cold-replays 2
```

The frozen output is [`full_replay_output.json`](full_replay_output.json),
SHA-256:

```text
da829c3e7238843b721145c1882ba1c6a03ef3766a0615a303ae777cfe0fb7e0
```

## Result

```text
profile:
  route2-research-mace-mdp-point-polar-residual-role-separated-adt-ddx-operational-v2
cold replays:                          2
five-start iterations per replay:     19--21
maximum primal residual:               8.5258e-11 eV
maximum cold-replay field difference:  0.0 eV
maximum cold-replay energy difference: 0.0 eV
permanent total charge:                0.0 e
residual total charge:                 9.80e-17 e
polarization energy:                  -0.10298980279231838 eV
```

This proves real-checkpoint iodine provider coverage and deterministic
multi-start root replay for this one geometry. It is not an accuracy, force,
Hessian, common-functional, or public-admission result. The immutable v1 505
run remains untouched; a complete v2 panel requires a new profile-bound run.
