# Route 2 force-admission contract

## Status

**One bounded profile is released; all other Route-2 profiles remain
energy-only.**  The separately versioned water profile

```text
smd-cpcm-fc-aswig-jgp94-d2-mace-aqueous-pcm-half-coupling-force-v3
```

exposes an analytic force only after a per-geometry
`ForceAdmissionCertificate` passes.  The PCMSolver and pyddx profiles still
publish energy only; their single-point derivatives remain labelled research
evidence and cannot acquire force capability through a global switch.

This document accepts the useful part of the force/PES proposal: keep the
unmixed fixed-point residual, matrix-free JVP/VJP, GMRES adjoint, MACE
fixed-field partials, density-position VJP, and same-provider continuum
coordinate derivative.  It rejects two unsafe shortcuts:

1. promoting the existing variable-cardinality PySCF-SWIG or pyddx
   experiments directly to a force profile; and
2. treating an iterative largest-feedback-gain estimate as a proof that the
   fixed-point residual is well conditioned.

Neither shortcut creates a smooth, unique operational-scalar PES.  The
released profile instead uses a distinct fixed-cardinality amplitude-CPCM and
fixed-topology aqueous SMD-CDS scalar.  It does **not** establish a common
stationary MACE--PCM electronic free-energy functional.

## What the code records

`route2_force_admission.py` defines `ForceAdmissionCertificate`.  It records
all of the following before a profile can claim force admission:

- nominal, rather than finite-resolution, SCF root;
- separate monopole and dipole fixed-point residuals;
- relative GMRES-adjoint residual;
- same-continuum half-coupling identity error;
- a local fixed-point condition screen for
  \(A = I - J_{\mathrm M}J_{\mathrm P}\);
- continuum same-energy derivative, fixed-node-topology, and geometry-path
  smoothness evidence;
- multi-start root agreement; and
- closure of the operational-energy/common-energy semantics.

For reduced dimensions up to the declared bound (default 128), the screen
materializes the complete fixed-charge tangent operator and obtains a dense
floating-point SVD.  This is a local numerical gate, not a global uniqueness
proof.  Above that bound, the current ARPACK gain calculation is retained as
diagnostics only: it **cannot pass** the gate because a Ritz estimate does not
supply a certified upper bound on \(\lVert J_{\mathrm M}J_{\mathrm P}\rVert_2\)
or a lower bound on \(\sigma_{\min}(A)\).

Every pyddx profile is registered with
`PYDDX_HARD_ACTIVE_SET_SMOOTHNESS_CONTRACT`; it necessarily fails on node
topology and geometry-path smoothness. Thus it cannot become an ASE force
route by accident. The released FC-aSWIG profile supplies a different
fixed-topology smoothness contract and still fails closed when its local root,
conditioning, or geometry-domain checks do not pass.

The complete, source-bound admission record is
[`benchmarks/route2-fc-aswig-force-v3-release-evidence-v1.json`](benchmarks/route2-fc-aswig-force-v3-release-evidence-v1.json).
It retains the full numerical payloads for component finite differences,
rigid translation/rotation, a Cartesian path, two closed loops, a flexible
torsion, and a three-step short NVE refinement.  It also contains a clean-tree
replay through `CommandControl -> SetCalculator -> ASE get_forces()` on the
current force-capability metadata.

## SCF audit quantities

Every accepted/rejected SCF record now carries four independently interpretable
quantities rather than a single mixed-unit norm:

| quantity | stored fields | role |
| --- | --- | --- |
| density residual | `monopole_residual_e`, `dipole_residual_e_angstrom` | unmixed fixed-point residual |
| reaction-field change | `reaction_potential_change_ev`, `reaction_gradient_change_ev_per_angstrom` | current field versus prior accepted field |
| energy change | `energy_residual_ev`, `energy_residual_source` | selected, versioned ledger stability |
| charge conservation | `root_total_charge_e`, `raw_response_total_charge_e`, `response_charge_projection_max_e` | affine total-charge invariant and projection audit |

The field-change values are `None` for the first iteration because no prior
accepted field exists.  They are monitors, not a substitute for the residual
or for a continuum smoothness proof.

## JGP94 and SWIG disposition

The pure nondegenerate JGP94 transform and its reverse VJP live in
`maple/function/calculator/extra_correction/implicit/route2_body_frame.py`.
The old benchmark import remains a compatibility shim, so the public profile
and its canaries use identical math.

JGP94 by itself remains insufficient: existing pyddx and PySCF-SWIG/ISWIG
preflights change retained surface-point counts or ownership under geometry
or orientation changes. A moving rigid frame can eliminate laboratory-frame
rotation drift; it cannot make a variable-topology surface differentiable.
Accordingly, `PYSCF_SWIG_VARIABLE_SURFACE_SMOOTHNESS_CONTRACT` remains
fail-closed. The admitted profile combines JGP94 with MAPLE's distinct
fixed-cardinality amplitude-CPCM construction; every candidate remains
allocated and its physical charge vanishes continuously with exposure.

## Implemented narrow boundary and remaining work

The fixed-topology force profile implemented all of the following on the same
declared operational scalar:

1. fixed degrees of freedom and no geometry-dependent node deletion;
2. continuous coordinate dependence through the surface/operator/CDS terms;
3. a matching forward map, adjoint, scalar energy, and full coordinate VJP;
4. a profile-specific multi-start root study and condition screen; and
5. component-resolved finite differences, rigid rotation/translation, path
   continuity/closed work, and short NVE evidence.

Its admitted scope remains neutral, closed-shell, connected, non-periodic
16--500 Da molecules in water, local-jet response, a nondegenerate JGP94
frame, and the dense fixed-charge conditioning dimension bound. Broader
chemical classes, relaxed flexible paths, longer NVE trajectories, ions,
other solvents, and degenerate molecular frames remain open gates.

A frame-free smooth spherical-harmonic Galerkin continuum remains a plausible
long-term generalization, not an implementation claim. No experimental
solvation-energy fit, response-scale adjustment, radius tuning, or MACE
checkpoint change is part of this force-admission work.
