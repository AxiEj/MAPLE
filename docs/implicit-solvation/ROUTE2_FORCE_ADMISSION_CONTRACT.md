# Route 2 force-admission contract

## Status

**Implemented boundary, not a released force profile.**  Route 2 continues to
expose only energy through its public provider API.  The pyddx path may return
explicitly labelled single-point derivative evidence, but that evidence is
now accompanied by a machine-readable force-admission record and remains
fail-closed.

This document accepts the useful part of the force/PES proposal: keep the
unmixed fixed-point residual, matrix-free JVP/VJP, GMRES adjoint, MACE
fixed-field partials, density-position VJP, and same-provider continuum
coordinate derivative.  It rejects two unsafe shortcuts:

1. promoting the existing JGP94 + PySCF-SWIG experiments directly to a force
   profile; and
2. treating an iterative largest-feedback-gain estimate as a proof that the
   fixed-point residual is well conditioned.

Neither shortcut creates a smooth, unique, common-energy PES.

## What the code records

`route2_force_admission.py` defines `ForceAdmissionCertificate`.  It records
all of the following before a future profile can claim force admission:

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

The current pyddx profile is registered with
`PYDDX_HARD_ACTIVE_SET_SMOOTHNESS_CONTRACT`; it necessarily fails on node
topology, geometry-path smoothness, multi-start agreement (not yet measured),
and common-energy semantics.  Thus it cannot become an ASE force route by
accident.

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
The old benchmark import remains a compatibility shim, so a future profile and
its canary use identical math.

This does **not** promote JGP94 to a production continuum frame.  Existing
preflight evidence found pyddx active ownership changes under small coordinate
moves.  Existing PySCF-SWIG/ISWIG evidence also has variable retained
surface-point counts/parents across geometry or orientation.  A moving rigid
frame can eliminate laboratory-frame rotation drift; it cannot make a
variable-topology surface differentiable.  Accordingly,
`PYSCF_SWIG_VARIABLE_SURFACE_SMOOTHNESS_CONTRACT` is also fail-closed.

## Next implementation boundary

A force-capable continuum must be a genuinely new profile with all of the
following, demonstrated on the same scalar energy:

1. fixed degrees of freedom and no geometry-dependent node deletion;
2. continuous coordinate dependence through the surface/operator/CDS terms;
3. a matching forward map, adjoint, scalar energy, and full coordinate VJP;
4. a profile-specific multi-start root study and condition screen; and
5. component-resolved finite differences, rigid rotation/translation, path
   continuity/closed work, and short NVE evidence.

A smooth spherical-harmonic Galerkin ddPCM is a plausible **research target**
for this profile, but is not claimed as implemented merely because the desired
variational equations can be written down.  No experimental solvation-energy
fit, response-scale adjustment, radius tuning, or MACE checkpoint change is
part of this force-admission work.
