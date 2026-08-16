# Pure MACE-POLAR frozen-source ddX/SMD PES

## Scope

[`mace_polar_frozen_ddx.py`](../../maple/solvation/experimental/mace_polar_frozen_ddx.py)
defines the pure MACE-POLAR + solvent lane. It has no MACE-MDP source and no
coupled ML/continuum fixed point. The execution surface exposes E, analytic F,
a molecular virial, numerical HVP, and a full numerical Hessian for one declared
geometry scalar. The registry now marks those five operations as experimentally
available for the exact profile while keeping periodic stress and every
workflow/release admission false.

## Registered accuracy scalar

The quantitative profile is
`route2-profile-experimental-pure-macepolar-frozen-point-l1-ddpcm-smd-v1`:

\[
E(\mathbf R)=E_\mathrm{vac}^{\mathrm{MACE\mbox{-}POLAR}}(\mathbf R)
+G_\mathrm{ddPCM}[\mathbf R,c_0(\mathbf R)]
+G_\mathrm{SMD\mbox{-}CDS}(\mathbf R),
\qquad c_0(\mathbf R)=M_\mathrm{POLAR}(\mathbf R,u=0).
\]

Its frozen identity is:

- official unmodified MACE-POLAR-1-M radial-GTO adapter on CPU/float64 at zero
  external field, with its release contract, checkpoint, evaluator, runtime,
  provenance, and neutral-singlet domain checked fail-closed;
- the learned first `(q,l=1)` block mapped without fitting to ddX point
  multipoles; the unused second radial block is exactly zero;
- `pyddx==0.8.0` ddPCM, `lmax=15`, `n_lebedev=1202`,
  `solver_tolerance=1e-12`, `eta=0.1`, `n_proc=1`;
- SMD solvent radii/dielectric and official PySCF `2.13.1` legacy SMD-CDS;
- neutral-singlet fixed geometries and no response calibration or fitting.

Only that exact model/runtime binding and those ddX settings receive the
registered scalar ID
`route2-experimental-pure-macepolar-frozen-point-l1-ddpcm-smd-v1`.
Model impersonation and `lmax`, `n_lebedev`, `solver_tolerance`, `eta`, or
`n_proc` overrides remain callable only through the generic unregistered
frozen-source scalar contract. The constructor also rejects attempts to attach
the registered ID to another model, continuum, or CDS term.

The older radial-GTO ddPCM/194 + smooth Fibonacci/SWIG-inspired CDS water path
remains a distinct low-cost derivative canary. It is not the 505-row accuracy
identity and cannot inherit the registered point-profile MAE.

## Complete force ledger

The implemented force is the complete chain rule of the same scalar:

\[
\mathbf F=\mathbf F_\mathrm{vac}
-\left.\partial_{\mathbf R}G_\mathrm{ddPCM}\right|_{c_0}
-\left(\frac{\partial c_0}{\partial\mathbf R}\right)^\mathsf T
  \frac{\partial G_\mathrm{ddPCM}}{\partial c_0}
-\partial_{\mathbf R}G_\mathrm{SMD\mbox{-}CDS}.
\]

The vacuum, fixed-source continuum-coordinate, MACE-POLAR source-VJP, and CDS
leaves are stored separately. A supplied central energy or force cache is
accepted only after replay against the current provider, configuration,
geometry, and complete production state digest. A caller cannot change an
energy leaf, rebuild a self-consistent digest, and reuse the stale cache for F,
virial, HVP, or H.

## Virial, HVP, Hessian, and topology

- The virial is a declared-origin molecular affine derivative in eV. It is not
  periodic stress and not the registry's historical `V=variational` tier.
- HVP/H are error-estimated Richardson derivatives of the same replayed
  conservative force. Policy SHA, actual steps, retry count, force hashes,
  raw antisymmetry, and the symmetrized Hessian remain visible.
- ddX exposes a discrete cavity topology fingerprint. PySCF SMD-CDS does not
  expose libsolvent's internal surface active set, so point-ddX + PySCF-SMD has
  `partial` topology observation. HVP/H therefore reject by default. The
  explicit `observed-components-only-experimental-v1` policy permits a
  diagnostic result labelled `partial-experimental`; it is not fully topology
  fail-closed and does not admit FREQ, OPT, MD, or Tier H.

## Development energy evidence

The frozen preregistered 505-row MNSol development panel (10 solvents, fixed
record/geometry/profile identity) measured:

- MAE: `1.2850369252161231 kcal/mol` — passes the hard `<=1.5` target;
- RMSE: `1.8120165083234037 kcal/mol`;
- maximum absolute error: `7.243453829909983 kcal/mol`;
- records with absolute error `>=1.5`: `159/505`;
- water subset: `306` records, MAE `1.5787238393184055 kcal/mol`.

This is an aggregate development-panel result, not a per-record or per-solvent
guarantee. The sealed confirmation partition remains unopened. The exact
source-bound replay artifact must accompany integration; an interrupted or
path-unbound rerun is not evidence.

## Derivative evidence boundary

Earlier same-scalar development diagnostics found maximum errors of about
`6.23e-6 eV/A` for a force directional check, `1.55e-5 eV` for an affine
virial check, `3.59e-5 eV/A^2` for an HVP versus force-FD check, and
`7.99e-6 eV/A^2` for `H @ direction` versus HVP on the selected small panel.
These are numerical consistency checks, not independent physical force,
stress, or Hessian references.

Energy MAE does not bound force, virial, or Hessian error. Broad covariance,
distorted-PES, independent derivative-reference, frequency, optimization, MD,
and periodic-stress evidence remain open. Accordingly the registry/profile
have an enabled `experimental_execution` surface for E/F/molecular-virial/HVP/H,
but no admitted E/F/H/V/M tiers. `V` remains the historical variational tier;
it is not the molecular virial. Periodic stress is explicitly unavailable.

## Literature and upstream boundaries

- A nonconjugate fixed-point derivative is mathematically admissible only for
  a smooth, locally unique residual root with nonsingular state Jacobian; see
  Christianson's work on reverse accumulation and implicit functions. This
  frozen-source route is simpler because it has no coupled ML/continuum root.
- ddX `0.8.0` exposes energy, adjoint/source derivatives, and force ingredients,
  but no public nuclear-Hessian API. Combined HVP/H therefore differentiate the
  complete MAPLE force rather than claiming an upstream ddX Hessian.
- MACE `0.3.16` supports PolarMACE E/F/stress and model-energy autograd Hessians.
  Those APIs do not establish derivatives of the added ddPCM + SMD-CDS scalar;
  MAPLE owns and verifies that combined chain rule.
