# Route 2 P−1/M0 MLIP plug-in and common-functional contract

## Status

This document defines the new M0 boundary. The content-hash-bound P0 artifact
certifies one narrow MACE-POLAR field interface for neutral H/O response-only
work. It does not promote historical accuracy results or certify a force
surface, an SCF energy ledger, or a common variational functional.

| Item | Current evidence | Status |
|---|---|---|
| capability-layered plug-in API | source/field/coupling contracts and synthetic non-MACE tests | implemented engineering contract |
| explicit factory registration | `Route2PluginRegistry`; MACE-POLAR is the first registered factory | implemented |
| reuse of existing SCF/PCM | `AtomicL1PluginEngineAdapter` translates only state/field names | implemented for local atomic `l<=1` |
| P−1 profile | water, pyddx/ddPCM primary, PCMSolver audit, frozen cavity identity, no CDS, response-only | implemented as an admission contract |
| MACE-POLAR P0 field convention | `route2-mace-p0-field-contract-v1.json`; selected mapping error `6.39e-14`, opposite-sign error `1.33e-4`, energy identity error `4.39e-13 eV` | **potential-gradient; response-only** |
| MACE-POLAR training optimizer coverage | exact training graph/code is unavailable in this checkout | **unattestable** |
| P1 outer residual and scalar ledger | dimensioned max/RMS source residuals, raw/projected ΔQ, total Δμ, ΔE, full pairing, PCM half coupling, CDS, standard state, and checked final scalar | implemented and synthetic-tested |
| P1 live water audit | `route2-mace-p1-water-operational-audit-v1.json`; outer root and scalar ledger pass, energy/source conjugacy fails, inner solve is tolerance-only | **P1 incomplete; fail-closed** |
| P1 inner continuum residual | pyddx provider records requested tolerance but exposes no actual algebraic residual through the current public provider contract | **tolerance-only; P1 admission blocked** |
| P2/P3 water electrostatics panel | requires a residual-certified P1 state and a frozen cavity/backend protocol | **not run** |
| common electronic functional | no admitted plug-in supplies it | **not implemented** |
| operational or variational forces through P−1 | P−1 permits response-only energy diagnostics | **blocked by profile** |

The frozen, uncommitted SMD/CDS solvent-profile work is a separate branch of
work. P−1 does not import it: `include_cds=False` is invariant.

## Why the old all-method protocol is now only a compatibility surface

The historical `Route2ElectronicModel` required every adapter to expose state,
response, field-energy, force, and coordinate-VJP methods, even when a model
could only provide energy and a source. Unsupported methods then existed only
to raise an exception. M0 instead separates four monotonically stronger
contracts:

1. `ElectronicSourceProvider`: zero-field/fixed-field scalar state and source;
2. `FieldResponsiveModel`: source-response JVP/VJP linearization;
3. `DifferentiableElectronicModel`: fixed-field force, source-position VJP,
   and feature VJP;
4. `VariationalElectronicModel`: one electronic functional and its source
   gradient/HVP in addition to the complete differentiable layer.

The old protocol remains unchanged for released profiles. The explicit
`AtomicL1PluginEngineAdapter` is the sole bridge into it. The bridge contains
no continuum equation, SCF iteration, energy ledger, or force algorithm; those
remain in the established engine.

## Source, dual field, and continuum coupling

For a geometry \(R\), a plug-in declares an electronic source space
\(\mathcal C_R\) and a field dual \(\mathcal F_R\), including representation,
shape, units, gauge, and bilinear pairing. The current compatibility space is
an atom-centred raw real-spherical \(l\leq1\) block. Its pairing reuses the
existing `MACE_POLAR_L1_PAIRING`, including the nontrivial Cartesian/raw
component permutation; it is not an elementwise dot-product invention.

A continuum coupling declares

\[
 B_R:\mathcal C_R\rightarrow\mathcal S_R,
 \qquad
 B_R^*:\mathcal S_R\rightarrow\mathcal F_R,
\]

and must numerically verify

\[
 \langle B_R c,\sigma\rangle_{\mathcal S}
 =\langle c,B_R^*\sigma\rangle_{\mathcal C,\mathcal F}.
\]

A failed dot test is retained as response-only diagnostic evidence. It is a
hard failure for a variational admission. The contract also records surface
units, gauge, and the total-charge constraint rather than deriving them from
array shapes.

This follows the variational PCM requirement that the surface-charge operator
and its weak/adjoint form belong to one energy functional. It does not assume
that an arbitrary discretized PCM operator is symmetric in an unweighted
Euclidean dot product.

## Canonical local reaction field

`LocalReactionField` carries only

\[
 [\phi,\nabla\phi]
\]

with explicit gauge and units. A bare `[N,3]` physical electric field is not a
production API because \(\mathbf E=-\nabla\phi\) and the sign changes the energy
conjugacy. The only `[N,3]` conversion is labelled as unledgered debugging and
requires exactly neutral total charge.

This distinction is essential for MACE-POLAR. The upstream public API accepts
a homogeneous `external_field`, while MAPLE installs nodewise potential and
potential-gradient features into an internal projector. Source inspection
alone establishes neither uniform-field equivalence nor the correct Born-sign
convention. `MACEPolarPlugin` therefore remains
`field_convention="unattested"` unless an accepted, content-addressed P0
certificate is supplied. An old or differently configured calculator cannot
inherit the new conclusion by class name.

## P0 result and exact admission boundary

The accepted-current artifact is
`benchmarks/route2-mace-p0-field-contract-v1.json`; the separate disposition
registry binds its SHA-256 and leaves the two historical artifacts unchanged.
The audit used the official MACE-POLAR-1-M checkpoint in float64, the existing
`graph-longrange-molecular-realspace-v1` evaluator, and the existing
`jgp94-d2-canonical-v1` coordinate wrapper. Its required interface gates were:

- upstream homogeneous-field versus nodewise local-jet equivalence;
- the identity \(E_{\rm upstream}-E_{\rm local}=f\cdot\mu\);
- canonical translation/gauge and rotation covariance;
- two-graph batch isolation; and
- the analytic central-charge Born sign and half-coupling identity.

All five passed. The positive mapping is uniquely selected:

\[
 f_{\rm upstream}=\nabla\phi,
 \qquad
 \mathbf E_{\rm physical}=-\nabla\phi=-f_{\rm upstream}.
\]

This is not a common-energy pass. The same artifact reports, rather than
hiding, three blocking results:

- \(\partial E/\partial f\) differs from the reported dipole by relative
  \(L_2=9.91\times10^{-3}\), although the autograd derivative itself matches
  central differences;
- the induced-dipole tensor is passive in the tested sign but is rank
  deficient and fails the fixed reciprocity tolerance; and
- the full node-source susceptibility passes its JVP/VJP implementation dot
  test but fails electrostatic reciprocity and nonpositive passivity (largest
  symmetrized eigenvalue `6.61e-3 eV`).

Consequently the certified construction is explicit:

```python
certificate = load_mace_p0_certificate(repo_root)
plugin = DEFAULT_ROUTE2_PLUGIN_REGISTRY.create(
    "mace-polar-1",
    calculator,
    p0_certificate=certificate,
)
```

The certificate rechecks artifact bytes, every repository source hash listed
by the audit, the checkpoint SHA, the composite inference-code SHA, coordinate
frame, and field evaluator. It admits only neutral H/O response work. Omitting
it leaves the field convention unattested; changing code, weights, frame, or
evaluator invalidates replay.

## P1 residual and energy-accounting contract

`route2_p1_audit.py` adds an audit layer without duplicating the existing SCF,
PCM, or energy-composition algorithms. Every accepted outer iteration now
records the following quantities in their native units:

- monopole max and RMS residuals in `e`;
- atom-centred dipole-component max and RMS residuals in `e Angstrom`;
- a dimensionless source RMS after scaling each channel by its own tolerance;
- raw response ΔQ before fixed-charge projection and projected total-charge
  residual after projection;
- the total molecular Δμ vector and its (L_2) norm, including charge
  transport between atomic centres; and
- the selected-ledger ΔE.

P1 strict nominal admission requires all of these gates at once, including an
observed ΔE from at least two energy samples. Rejected Anderson attempts remain
in the history, roll back to the prior accepted root, and cannot be mistaken
for convergence. A maximum-iteration failure carries the complete dimensioned
history. The pre-existing finite-resolution stagnation route remains
compatibility evidence, but it is explicitly
`nominal_residual_gate_passed=False` and cannot satisfy P1.

The immutable operational ledger separately stores

\[
 E_{\rm model}(0),\quad E_{\rm model}(f),\quad
 E_{\rm model}(f)-E_{\rm model}(0),\quad
 \langle c,f\rangle,\quad \tfrac12\langle c,f\rangle,
\]

plus the provider PCM energy, CDS, standard-state term, selected ledger, and
final scalar. `Route2EnergyLedger` independently closes the Hartree leaves and
derived totals. The full pairing is retained rather than reconstructing it
from the half-coupling energy.

Inner continuum evidence is deliberately fail-closed:

- `residual-certified` requires an observed residual, its definition, a
  requested tolerance, and a completed solve;
- `tolerance-only` means the backend accepted a requested tolerance but did not
  expose the resulting algebraic residual; and
- `unattested` means neither quantity is available.

Only the first status can pass the P1 inner gate. The current pyddx wrapper
therefore remains `tolerance-only`; Route 2 does not reconstruct undocumented
ddX matrices merely to manufacture a residual number.

Finally, `audit_plugin_energy_source_conjugacy` compares a central finite-field
energy derivative against the declared source/dual pairing. A failed measured
test is retained and labels the scalar
`response-conditioned-operational-prediction`; an unrun test blocks P1. A pass
still does not create the common functional required by the variational route.

## Live P1 water result

The content-hash-bound artifact
`benchmarks/route2-mace-p1-water-operational-audit-v1.json` records one CUDA
run of the official MACE-POLAR-1-M checkpoint with pyddx 0.8.0. The exact
profile is neutral water, H/O only, fixed cavity, ddPCM with
`dielectric=78.39`, `lmax=7`, 302 Lebedev points, solver tolerance `1e-12`, no
CDS, and the PCM-half-coupling ledger. It is not the P2/P3 molecular panel.

The outer safeguarded-Anderson loop reached the strict nominal root in seven
accepted iterations. Its final dimensioned evidence was:

- monopole max/RMS: `1.811e-15 / 1.280e-15 e`;
- dipole-component max/RMS: `5.967e-16 / 2.443e-16 e Angstrom`;
- raw response ΔQ / projected total-charge residual:
  `-1.041e-17 / 3.469e-18 e`;
- total molecular Δμ norm: `2.008e-15 e Angstrom`; and
- selected-ledger ΔE: `8.152e-13 eV`.

The energy audit retained, without folding terms together,

- `E_model(0) = -2079.8624817231876 eV`;
- `E_model(f) = -2079.8526149545987 eV`;
- full pairing `-0.9119381351131642 eV`;
- PCM half coupling `-0.4559690675565821 eV`;
- provider PCM energy `-0.4559690675565623 eV`;
- CDS and standard-state terms both zero; and
- final scalar `-0.4559690675565623 eV`.

This closes the selected scalar ledger to the stated tolerance. It does not
make the reported electronic energy variational in the reported source. Three
dimensionless central scaling steps (`2e-3`, `1e-3`, and `5e-4`) applied to the
complete converged local jet gave stable finite-difference energy derivatives
near `6.349e-3 eV`, versus the full declared source-pairing derivative of
`-0.911938 eV`; all three conjugacy probes failed. The result is therefore
labelled `response-conditioned-operational-prediction`.

More importantly for the staged gate, pyddx reports the requested `1e-12`
solver tolerance through the current provider provenance but not the achieved
algebraic residual. The inner evidence is therefore `tolerance-only`, not
`residual-certified`. The artifact sets `p1_complete=false` with the sole P1
blocker `inner-continuum-algebraic-residual-not-certified`; by construction,
P2/P3, PCMSolver comparison, operational forces, and the variational route
were not run or promoted.

### Upstream analysis of the remaining inner-solve blocker

The pyddx 0.8.0 binding and the corresponding ddX `v0.8.0` source commit
`4d79e3d9caeae5e602683572a71cb550414f9b09` were inspected before deciding
that this blocker could not be repaired locally without crossing the provider
boundary. The Fortran state internally stores iteration histories such as
`phieps_rel_diff` and `xs_rel_diff`, but the C/Python state exposes only the
solution, an iteration count, and solved flags. Moreover, the Jacobi/DIIS stop
quantity is the relative change between successive iterates, not an explicitly
reported algebraic residual such as

\[
 \frac{\lVert A x-b\rVert}{\max(\lVert b\rVert,\epsilon)}.
\]

For ddPCM there are two forward linear stages (`R_epsilon` followed by the
ddCOSMO `L` solve), while the public `x_n_iter` accessor reports only the latter
stage. Consequently, neither `is_solved`, `x_n_iter`, parsing a human-readable
log, nor replaying the requested tolerance is sufficient for the P1 residual
claim. Reconstructing `R_epsilon` and `L` in MAPLE would duplicate ddX and was
therefore rejected.

The minimal maintainable resolution belongs upstream: a structured solver
report for every internal stage, with the operator/RHS identity, convergence
flag, requested tolerance, stopping-metric definition/value, and preferably a
post-solve algebraic residual in a documented norm. MAPLE can consume that
report through `route2_continuum_solve_evidence` and use the maximum certified
forward-stage residual. Until such an API exists, the correct status remains
`tolerance-only`.

## Admission routes

### Response-only (P−1 default and only allowed route)

The fixed-point object

\[
 c=M_\theta(B_R^*\sigma(c))
\]

is an operational response equation. Convergence proves neither stationarity
nor a free energy. If finite-field integration does not show that the reported
source is conjugate to the reported model energy, results must be labelled
`response-conditioned operational prediction`.

P−1 fixes:

- solvent: water;
- nonpolar arm: disabled;
- primary continuum: pyddx/ddPCM;
- independent continuum audit: PCMSolver/IEFPCM;
- cavity identity: frozen and versioned;
- default execution: SCF;
- allowed scientific route: response-only;
- outputs: energy diagnostics only.

The ddX project already supplies ddPCM/ddCOSMO/ddLPB solvers, electrostatic
energies, forces, and host-code interfaces including polarizable MM. Route 2
therefore adapts the existing ddX provider rather than implementing a new PCM
solver.

### Operational force

A later profile may allow this route only if one exact scalar \(E_{op}\) owns
all terms and the plug-in supplies the complete differentiable layer. The
existing fixed-point adjoint can then be reused. Force admission still needs
fixed topology, cavity/gauge derivatives, a unique tracked root, analytic-vs-
central-difference agreement, rigid-motion tests, path/loop work, and narrow-
domain NVE evidence. A pyddx active-set or surface-topology change prevents a
claim of a globally smooth PES.

### Common variational functional

Only `VariationalElectronicModel` may enter

\[
\mathcal L(R,c,\sigma,\lambda)=
 F_\theta(R,c)
 +\tfrac12\sigma^T A_R\sigma
 +\sigma^T B_Rc
 +\lambda(q^Tc-Q)
 +G_{CDS}(R).
\]

Admission requires the complete differentiable layer, the electronic
functional/gradient/HVP, a source total-charge functional, and a passing
\(B/B^*\) check. Subsequent KKT work must still prove energy-source conjugacy,
Jacobian reciprocity, passivity, reduced-Hessian/Schur positivity, KKT
residuals, and analytic-vs-FD forces. The learned MACE response may be used as
an initializer or preconditioner; glue code cannot promote it to
\(F_\theta\).

The energy-first rule is supported by differentiable electric-response work:
polarization, Born charges, and polarizability are derivatives of one
generalized energy, which is what enforces conservation and reciprocity.

## Required plug-in provenance

Every plug-in records, without omission:

- checkpoint SHA-256;
- training-code SHA and inference-code SHA, or the literal `unattestable`;
- representation;
- supported elements and total-charge interval, or an unattestable domain;
- field convention and units;
- model field evaluator and coordinate-frame policy;
- energy semantics;
- derivative capability declaration;
- optimizer-coverage status.

An unattestable field convention blocks field-conditioned execution. An
unattestable element or charge domain blocks the corresponding domain canary.
MACE-POLAR's upstream claim of variable charge capability does not waive the
Route 2 adapter/checkpoint-specific charge canary.

Historical artifacts are never rewritten. The byte-bound registry
`benchmarks/route2-mace-p0-artifact-dispositions-v1.json` records their current
interpretation. It now contains one `accepted-current` P0 artifact and two
byte-identical `legacy-unresolved` artifacts; the historical JSON was not
rewritten.

## Tests and claim boundary

`tests/solvation/test_route2_plugin_contract.py` covers:

- source/field shapes, units, gauge, and immutable provenance;
- canonical component permutation;
- passing/failing \(B/B^*\) dot tests;
- explicit factory registration and identity substitution rejection;
- a non-MACE synthetic response plug-in;
- capability downgrade and missing-method failures;
- P−1 route restriction;
- element/charge domain checks;
- reuse of the established SCF/PCM engine through the atomic-l1 bridge;
- the analytic Born sign and fail-closed field-convention selector;
- accepted-current P0 certificate/source/checkpoint binding;
- historical artifact SHA binding without JSON rewrites.

`tests/solvation/test_route2_p1_audit.py` additionally covers dimensioned
max/RMS/ΔQ/Δμ gates, finite-field conjugacy pass/fail cases, complete scalar
ledger closure, `tolerance-only` rejection, and a synthetic
`residual-certified` admission.

These tests establish architecture, scalar-ledger closure, and fail-closed
behavior only. They do not establish a live pyddx inner residual, chemical
accuracy, a variational functional, solution-phase forces, production MD,
arbitrary elements/charges/solvents, or experimental validation. The P0 live
artifact establishes only the exact field semantics and evaluation policy
named above.

## Primary references and upstream implementations

- MACE-POLAR-1 preprint: <https://arxiv.org/abs/2602.19411>
- Official MACE-POLAR guide and public external-field interface:
  <https://mace-docs.readthedocs.io/en/latest/guide/polar_mace.html>
- MACE reference implementation: <https://github.com/ACEsuit/mace>
- ddX reference implementation: <https://github.com/ddsolvation/ddX>
- ddX documentation: <https://ddsolvation.github.io/ddX/>
- ddX v0.8.0 Python state binding:
  <https://github.com/ddsolvation/ddX/blob/v0.8.0/src/pyddx_classes.cpp>
- ddX v0.8.0 Jacobi/DIIS stopping metric:
  <https://github.com/ddsolvation/ddX/blob/v0.8.0/src/ddx_solvers.f90>
- Lipparini et al., *A variational formulation of the polarizable continuum
  model*, J. Chem. Phys. 133, 014106 (2010), DOI 10.1063/1.3454683.
- Falletta et al., *Unified differentiable learning of electric response*,
  Nature Communications 16, 4607 (2025), DOI 10.1038/s41467-025-59304-1.
