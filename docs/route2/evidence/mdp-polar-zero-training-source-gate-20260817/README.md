# MDP/POLAR zero-training permanent-source gate

This directory freezes the complete 60-molecule, target-independent source and
kernel factorial for the hybrid Route-2 line.  The run used the two unchanged
official checkpoints, the frozen SPICE/MBIS evaluation-only selection, and one
consistent CUDA process.  It read no experimental solvation targets and fit no
parameters.

Frozen decision:

```text
zero-training-projected-polar-gaussian-source-rejected-under-frozen-gates
```

The native 1.5 A Gaussian permanent-source kernel is rejected.  The diagnostic
MACE-POLAR zero-field **point** source is the only strong candidate in this
factorial (fixed-source ddPCM MAE 1.345980 kcal/mol; 60/60 improvements versus
the latent MDP point partition), but it was not the preregistered primary and
must be tested prospectively on a disjoint selection before promotion.  The
Gaussian-Coulomb projection to the MACE-MDP molecular dipole is also rejected:
it increases the point-source MAE to 2.778987 kcal/mol and creates a 17.753922
kcal/mol maximum error.

File SHA256 of `aggregate.json`:

```text
ac41ab8a7df04347e02d4726379e1e0fd0da4a3a52981f8928f9074ffb26cd50
```

Canonical payload `aggregate_sha256`:

```text
2bf5f2face6225d23fb9005a774ad285fcf9d23b3af0b6d3327284cfa8e3af73
```

No public E/F/H/V/M capability follows from this evidence.
