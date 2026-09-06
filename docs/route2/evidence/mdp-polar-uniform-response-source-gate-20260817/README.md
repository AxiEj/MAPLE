# MDP/POLAR uniform-response source gate

This directory freezes the complete prospective 60-molecule held-out test of a
strictly zero-training source candidate for the hybrid Route-2 line.  The
selection is molecule-disjoint from the earlier frozen 60-molecule source
factorial and was fixed without reading source, energy, MNSol, FreeSolv, or
experimental solvation results.  The run used the two unchanged official
checkpoints and one consistent CUDA process.  No parameter was fit.

Frozen decision:

```text
reject-both-polar-point-source-candidates
```

The MACE-POLAR zero-field point source is close to the primary mean-error goal
but fails the preregistered tail gates:

```text
MAE   = 1.460958 kcal/mol
q95   = 3.670878 kcal/mol
max   = 4.530936 kcal/mol
```

Solving on the checkpoint's native two-width uniform-field response manifold
closes the MACE-MDP molecular dipole to numerical precision and remains well
conditioned, but does not materially improve the chemistry:

```text
MAE   = 1.428327 kcal/mol
q95   = 3.545288 kcal/mol
max   = 4.493630 kcal/mol
paired improvements = 32 / 60
relative MAE reduction versus zero field = 2.23%
```

The response-manifold source is therefore rejected rather than integrated into
the hybrid runtime.  The evidence localizes the remaining tail away from
molecular-dipole closure, hidden-field magnitude, or response-solver
conditioning and toward local near-field topology, omitted higher multipoles,
and/or continuum-consistent density penetration physics.

File SHA256 of `aggregate.json`:

```text
d2a278188ae20649b0797cd70da62e7330bc7766fb9c65533dfd43ec0a8746ed
```

Canonical payload `aggregate_sha256`:

```text
cfc4f6569cbe3841db7b63fc0cdfd660de09291c2c5febfc18cd9b9682ca7417
```

Preregistration SHA256:

```text
834c8a9f06ed2974467049c58485cf3d4809ff77ba93aaf0193b07ba697d0ef5
```

No public E/F/H/V/M capability and no permission to post-train follows from
this evidence.
