# MAPLE implicit solvation: Route 2 research/innovation route

This branch contains only the MACE-POLAR + SMD continuum route. It does not
contain the fixed-charge PB/GB implementation. The default
PCMSolver--IEFPCM/GePol profile remains an energy proof-of-concept. Separate,
explicit pyddx ddPCM profiles retain single-point derivative evidence behind a
research-only API; their public Route-2 result remains energy-only. The one
exception is the explicitly versioned, water-only FC-aSWIG C-PCM `force-v3`
profile: it exposes forces only after every geometry passes the bounded root,
conditioning, and force-admission certificate. It is not a generic
solution-phase PES or a common variational MACE--PCM electronic functional. A separate
PCMSolver profile changes only the ASC-to-MACE reaction-field projection from
the historical local first-order jet to the checkpoint-native \(l\le1\) GTO
integrals. A newer, still non-default profile combines that receiver with the
SMD intrinsic Coulomb-sphere electrostatic cavity (`probe=0`, no added
spheres, explicit water dielectric). No profile is a complete MAPLE solution-phase PES for generic molecules.
All calculations therefore require `experimental=true`.

Two deliberately separate diagnostic profiles connect the exact local
`macepol-ef-v2.pt` TorchScript checkpoint to MAPLE's fixed-dimensional smooth
PCM. The original v1 profile remains water-only and energy-only. The v2
profile supports all 11 registered SMD solvents, analytic first derivatives,
finite-difference Hessians from those derivatives, `SP`, L-BFGS `OPT`, `FREQ`,
and P-RFO `TS`. Both remain CUDA-device-0-only and never default selected. The
v2 input requires both acknowledgements:

```text
#model=mace-polar-ef-v2(model_path=/absolute/path/macepol-ef-v2.pt)
#sp(verbose=1)
#device=gpu0
#solv(implicit=methanol,method=smd,provider=torch-smooth-pcm,profile=mace-polar-ef-v2-smooth-ddpcm-l3-p6-r96-128-128-multisolv-derivatives-known-nonpassive-v2,response=scf,standard_state=1m,experimental=true,acknowledge_known_nonpassive=true,acknowledge_unvalidated_derivatives=true)
```

Replace the task line with `#opt(method=lbfgs)`, `#freq`, or
`#ts(method=prfo)` as needed. Registered solvents are water, methanol,
ethanol, acetonitrile, DMSO, DMF, THF, chloroform, dichloromethane, toluene,
and hexane. Audit ledgers distinguish diagnostic derivatives from formally
admitted forces.

The no-training Route-2 V0 scalar-response alternative is also frozen as a
rejected diagnostic, not a new profile.  Its preregistered one-water CUDA
canary preserved the gas density at zero field and matched finite differences
of its anchored scalar, but failed the base-field total-charge gate
(`-5.61e-4 e`) and the neutral-response passivity gate
(`lambda_max=1.97e-2 eV`, threshold `1e-6 eV`).  The result changes no public
energy, response, provider, or schema and does not authorize V1, post-training,
or fine-tuning.  The immutable evidence is
[`route2-v0-scalar-response-water-v1.json`](benchmarks/route2-v0-scalar-response-water-v1.json).

The separate no-training **V0-FD** research contract freezes the solute density
at zero field and varies only an energy-conjugate continuum state.  It is not a
reinterpretation of the failed scalar-response V0 canary and does not change a
public profile.  Its mathematical boundary, custom-solvent information
requirements, and strict per-record blind-test gates are recorded in
[`ROUTE2_V0_FD_THEORY.md`](ROUTE2_V0_FD_THEORY.md) and
[`route2-v0-fd-multisolvent-prereg-v1.json`](benchmarks/route2-v0-fd-multisolvent-prereg-v1.json).

The distinct no-training **V0-Q** kernel adds an induced density only through
an explicit quadratic electronic scalar and the same-basis reciprocal GTO PCM
operator.  It never invokes or repairs the rejected MACE field update.  It has
no default physical curvature and therefore cannot report chemistry yet; its
KKT stability, charge, reciprocity, and passivity gates are recorded in
[`ROUTE2_V0_VARIATIONAL_QUADRATIC_THEORY.md`](ROUTE2_V0_VARIATIONAL_QUADRATIC_THEORY.md)
and
[`route2-v0-variational-quadratic-prereg-v1.json`](benchmarks/route2-v0-variational-quadratic-prereg-v1.json).
Its first frozen physical control is a published-QEq-hardness, same-MACE-GTO,
**monopole-only** tangent; it freezes every dipole response channel exactly and
remains structural/fixed-geometry only.  It is explicitly not a complete QEq
or solvation model.  Its source-bound acetone canary now passes the frozen
charge, reciprocal-operator, KKT stationarity, positive-curvature, passive
response, and same-basis half-coupling gates using an archived unmodified
MACE-POLAR gas density and the same PCMSolver cavity.  It contains no
experimental solvation metric or total-solvation claim; the immutable evidence
is [`route2-v0-qeq-monopole-acetone-v1.json`](benchmarks/route2-v0-qeq-monopole-acetone-v1.json).
That structural pass did **not** validate the physical curvature.  A separately
frozen ωB97M-V/def2-TZVPD acetone finite-field screen passed every QM numerical
gate but rejected the fixed QEq-monopole response: its polarizability trace is
`1.9768` times the QM trace, its relative tensor mismatch is `1.4295`, and its
largest principal-value relative error is `2.2244`.  The preregistered rule
forbids rescaling or changing the hardnesses after this result.  V0-Q therefore
retains a structurally valid variational kernel but has no accepted physical
curvature.  Evidence:
[`route2-v0-qeq-acetone-qm-field-v1.json`](benchmarks/route2-v0-qeq-acetone-qm-field-v1.json).

The active no-training **V0-AQ-C** research line instead keeps MACE as the
untouched gas branch and adds only a stationary auxiliary-electronic/
implicit-continuum free-energy difference.  Its first two code blocks now
lock a symmetric diffuse reaction scalar and a nonlocal iso-density-product
cavity whose chain-rule contribution is included in the electronic derivative.
They are structural mathematics only: no source-bound solvent electron-density
kernel/overlap rule, auxiliary electronic minimization, nonpolar/dispersion
functional, standard-state term, public profile, force/PES certificate, or
accuracy value exists yet.  It neither requires nor invokes GROMACS,
molecular-liquid trajectories, 3D-RISM, or MDFT.  The complete boundary and
admission tests are in [`ROUTE2_V0_AQ_THEORY.md`](ROUTE2_V0_AQ_THEORY.md).

The first provider-feasibility experiment is frozen separately in
[`ROUTE2_PROVIDER_CANARY.md`](ROUTE2_PROVIDER_CANARY.md). Its one-shot
fixed-density acetone scalar canary passed at commit `da89ab8`: the
laboratory-frame rotation span was `1.6528e-4 kcal/mol`, while the JGP94
molecule-following candidate reduced it to `4.8644e-14 kcal/mol` with a
`1.4473e-4 kcal/mol` identity-profile shift. This is not a force, PES, or
chemical-accuracy result; it authorizes only a separately pre-registered
analytic frame-VJP canary.

That derivative stage is frozen in
[`ROUTE2_FRAME_VJP_CANARY.md`](ROUTE2_FRAME_VJP_CANARY.md). Its zero-solve
preflight validates all 30 coordinate and 30 dipole components of the pure
frame VJP, but rejects the provider experiment: the eight prescribed acetone
displacements produce four pyddx cavity active-set signatures and 4992/4993
active pairs. No continuum state was solved and no one-shot lock was opened.
The JGP94 frame therefore repairs rigid-rotation covariance but is not, by
itself, a smooth-cavity solution for a MAPLE PES.

The resulting implementation boundary is recorded separately in
[`ROUTE2_CONTINUUM_ENGINE.md`](ROUTE2_CONTINUUM_ENGINE.md). That stage only
extracts the existing ML--SCF/adjoint/energy-ledger orchestration behind an
internal provider-neutral interface. It does not add a continuum provider,
change a formula or numerical setting, or promote the current force candidate.

The separate experimental reconstructed-density **rho-DROP CPCM** provider is
documented in
[`ROUTE2_RHODROP_CPCM.md`](ROUTE2_RHODROP_CPCM.md). It reuses that engine with
a source-dependent MOIST surface and a local nonlinear reaction-map JVP/VJP,
but is admitted only as electrostatics-only research energy. Its fixed
half-coupling ledger is enforced at the engine boundary; CDS, analytic forces,
OPT/FREQ/MD, chemical accuracy, and the metadata-only full-functional drive
remain closed.

The electronic side of that engine is now isolated behind the explicit
[`ROUTE2_ELECTRONIC_MODEL_ADAPTER.md`](ROUTE2_ELECTRONIC_MODEL_ADAPTER.md)
contract.  Current released profiles remain MACE-POLAR-bound, but a new
field/charge-aware MLIP can be added through one adapter plus one new
versioned profile without changing PCM, COSMO, CDS, SCF, or adjoint modules.
The current common source space is still only atom-centred monopole plus
dipole `(N,4)`; GTO, quadrupole, and other representations require distinct
source-space implementations rather than shape-compatible shortcuts.

## Route-2 contract: explicit electronic response + MLIP--continuum coupling

Route 2 is separate from the fixed-charge PB/GB path. It couples the official
pretrained MACE-POLAR-1-M coarse-grained charge moments either to the default
external PCMSolver IEFPCM reaction field plus MAPLE's native aqueous SMD CDS
term, or to an explicitly selected pyddx ddPCM reaction field plus the matching
official PySCF aqueous SMD CDS energy/gradient pair. At the dielectric
boundary, the moments use their cavity-exterior point-multipole expansion
rather than extending MACE's internal 1.5 A GTO smearing across the cavity. It
does not train or fine-tune a model, consume MOL2 partial charges, invoke a
quantum-chemistry executable, or use a solvation-trained MLIP.

The default PCMSolver profile feeds the atom-centred reaction potential and
gradient through the upstream affine-field matrix (`local-jet`) in the
continuum zero-at-infinity gauge. Two non-default energy-only profiles use the
same atomic-centre-mean model-field gauge: the matched
`smd-iefpcm-point-l1-local-jet-atomic-mean-v1` control retains `local-jet`,
whereas `smd-iefpcm-point-l1-exact-gto-v1` analytically integrates the point
ASC reaction potential against every receiver GTO channel exposed by the
loaded checkpoint. Only a comparison between those two matched-gauge profiles
isolates the reaction-field projector. Comparing the historical default
directly with exact GTO changes both projector and model-field gauge.
The exact path fails closed unless the runtime is
`graph-longrange==0.4.0`. Route 2 also resolves and hashes the complete
official MACE-POLAR-1-M checkpoint before loading, then fails closed unless
its release URL, byte size, and SHA-256 match the frozen contract. Its audit
records that full checkpoint identity plus the versioned receiver-feature
layout and live projection-matrix fingerprint.

The exact-GTO profile remains an **energy-only operational diagnostic**, not a
variational SCRF route: it keeps a point-\(l\le1\) multipole source while
driving MACE through finite-width GTO receiver features. Those maps are known
not to be a common transpose pair, so neither a common stationary free energy,
conservative solution-phase forces, nor a PES claim follows. Each public
result now writes this source/receiver contract explicitly. The replacement
physics path is the fixed-geometry same-basis GTO Galerkin operator followed
by a common electronic scalar/KKT admission—not empirical correction of the
legacy exact-GTO output.

The research objective remains mutual polarization: the MACE-POLAR
representation generates the solute electrostatic potential, PCM returns a
reaction field, and that reaction field is fed back through the unmodified
MACE-POLAR field-response path until convergence.  The currently more robust
no-training energy baseline is nevertheless an explicit Route-2 response mode:
`response=frozen` evaluates the electronic model at zero field and solves the
continuum once.  This is not PB/GB and not a fitted correction; it is the
zero-response member of the same source/continuum/ledger architecture.

The public v1 input is:

```text
#model=macepol-m
#sp
#solv(implicit=water,method=smd,provider=pcmsolver,profile=smd-iefpcm,response=scf,standard_state=1m,experimental=true)

0 1
MOL2 molecule.mol2
```

The provider and versioned profile are explicit so that a result's continuum
equation and cavity contract cannot be selected by a hidden default.
`response=scf` remains the backward-compatible public default.
`response=frozen` is an explicit, separately audited zero-field source mode:
it performs no field-conditioned model evaluation, has no learned fixed point,
and reports no synthetic SCF residual.  On pyddx it is admitted only for a
direct PCM half-coupling profile. Canonical element-radius profiles also accept
XYZ/inline geometry when the explicit `0 1` domain metadata is present;
only a profile whose identity includes a GAFF/GAFF2 radius override requires
MOL2 atom types. The omitted cavity default is
`cavity_policy=warning-fallback`: it may retry a warned primary GePol cavity
and is fixed-conformer energy infrastructure only. The explicit research
option `cavity_policy=fixed-stability-branch` instead selects
`AREA=0.28 A^2, MINRADIUS=0.30 A` before evaluation and fails closed on the
native `PCMSolver warning.` stderr marker. `PEDRA.OUT` warning lines are
retained separately in `route2-result.json`; they are diagnostics, not a
cavity-branch selector. This removes geometry-dependent **policy selection**,
but does not make the GePol surface differentiable, prove topology continuity,
certify the tessellation, or enable forces. There is no public mock or ddX
backend on this default path.

The model-native reaction-field projection experiment must be selected
exactly:

```text
#model=macepol-m
#sp(verbose=1)
#solv(implicit=water,method=smd,provider=pcmsolver,profile=smd-iefpcm-point-l1-exact-gto-v1,response=scf,standard_state=1m,cavity_policy=fixed-stability-branch,experimental=true)

0 1
MOL2 molecule.mol2
```

Its complete scientific identity is
`MACE-POLAR coarse residual point-(l<=1) source + PCMSolver/IEFPCM +
exact point-ASC-to-receiver-GTO projection + atomic-center-mean model-field
gauge + native water SMD-CDS`.
“Exact” refers only to the analytic receiver-GTO projection for the discrete
ASC; it does not mean full electron density, original SMD equivalence,
variational free-energy consistency, a complete coordinate derivative, or
chemical-accuracy certification. The profile is energy-only and non-default.
The matching projector control replaces only the profile name with
`smd-iefpcm-point-l1-local-jet-atomic-mean-v1`.

The corrected intrinsic-cavity/exact-GTO combination must be selected exactly:

```text
#model=macepol-m
#sp(verbose=1)
#solv(implicit=water,method=smd,provider=pcmsolver,profile=macepolar-mlpcm-smdcds-iefpcm-intrinsic-cavity-exact-gto-v1,response=scf,standard_state=1m,experimental=true)

0 1
MOL2 molecule.mol2
```

This versioned profile owns its cavity policy: it uses the SMD Coulomb radii
with zero electrostatic probe radius, disables added spheres, and uses an
explicit `78.355` water dielectric. It therefore rejects the legacy
`cavity_policy` option. The matched intrinsic-cavity local-jet control is
`macepolar-mlpcm-smdcds-iefpcm-intrinsic-cavity-v1`. Neither profile changes
the historical default.

A pre-registered ten-class FreeSolv development panel gives MAEs of `1.084`,
`1.752`, and `0.915 kcal/mol` for fixed \(l\le1\), intrinsic-cavity local-jet
SCF, and intrinsic-cavity exact-GTO SCF, respectively. Exact GTO wins 7/10
paired comparisons and reduces acetone's error to `0.933 kcal/mol`, but its
maximum error is still `3.247 kcal/mol` for acetic acid and it regresses
methanol, phenol, and chloroethane relative to local jet. The pre-registered
maximum-error gate therefore fails, so this profile remains an explicit
energy-only experiment rather than a default replacement. The row-level
experimental provenance and component ledger are frozen in
[`route2-pcmsolver-intrinsic-exact-gto-freesolv-ten-v1.json`](benchmarks/route2-pcmsolver-intrinsic-exact-gto-freesolv-ten-v1.json).

The separately named pyddx research profile must be selected exactly for its
energy result:

```text
#model=macepol-m
#sp
#solv(implicit=water,method=smd,provider=pyddx,profile=smd-ddpcm-l15-n1202-v1,response=scf,standard_state=1m,experimental=true)

0 1
MOL2 molecule.mol2
```

This profile fixes `lmax=15`, 1202 Lebedev points per sphere, `mixing=1.0`,
the documented ddPCM and adjoint tolerances, and one same-energy derivative
evidence chain. It does not accept the PCMSolver-specific `cavity_policy`.
Its public result does **not** expose `forces`; explicit
`evaluate_single_point_derivative_evidence()` calls are retained solely for
falsification and finite-difference validation, not ASE, optimization, MD, or
a solution-phase PES. The profile name records a bounded research candidate,
not a universal grid or accuracy certification. The separately named
`smd-ddpcm-l15-n1202-gaff2-o-v1` profile applies the already documented
GAFF/GAFF2 `o` carbonyl-oxygen radius change to the same numerical candidate;
it is not the canonical default or a broad-accuracy claim.

The multi-solvent direct-PCM comparison has two **parallel, explicit**
profiles.  They share the MACE-POLAR source, selected response mode,
SMD-derived cavity radii, solvent dielectric, standard state, and PySCF
SMD-CDS term; only the continuum equation changes.

```text
# Route 2A: ddPCM + SMD-CDS
#model=macepol-m
#sp(verbose=1)
#solv(implicit=acetonitrile,method=smd,provider=pyddx,profile=smd-ddpcm-l15-n1202-multisolv-pcm-half-coupling-v2,response=frozen,standard_state=1m,experimental=true)

# Route 2B: scaled ddCOSMO + the same SMD-CDS comparison term
#model=macepol-m
#sp(verbose=1)
#solv(implicit=acetonitrile,method=smd,provider=pyddx,profile=smd-ddcosmo-l15-n1202-multisolv-pcm-half-coupling-v2,response=frozen,standard_state=1m,experimental=true)
```

Changing only `response=frozen` to `response=scf` requests the learned
field-conditioned fixed point while retaining the direct PCM energy ledger.
The response mode is part of the cache, run-lock, and artifact identity, so
the two states cannot be silently mixed.

Both profiles register 11 solvents from the tested PySCF 2.13.1 SMD solvent
database: water, methanol, ethanol, acetonitrile, dimethyl sulfoxide,
dimethylformamide, tetrahydrofuran, chloroform, dichloromethane, toluene, and
hexane.  Each uses that solvent's tabulated dielectric,
solvent-acidity-dependent SMD oxygen radius, tested PySCF element-radius
mapping, and matching PySCF SMD CDS energy/gradient.  Water-only profiles
retain their original `78.39` dielectric but use the corrected
atomic-number-indexed SMD/SMD18 P/S/Cl radii (`2.12/2.49/2.38 angstrom`).
Artifacts generated with the former shifted mapping remain bound to their old
execution commits and are stale evidence for the corrected profile.

`ddCOSMO` here means the finite-dielectric, scaled ddX COSMO continuum
equation.  It is **not** COSMO-RS and it is **not** Route 1 ALPB.  Its
nonpolar arm is intentionally the same frozen PySCF SMD-CDS contribution as
the ddPCM arm, so a paired result changes one mathematical object only: the
electrostatic continuum operator.  This is a controlled
``ddCOSMO + SMD-CDS`` experimental comparator.  The ddPCM member is therefore
an SMD-CDS-augmented MACE-POLAR/ddPCM hybrid, not a claim that the original
QM-density SMD parametrization was re-fit for MACE residual multipoles or for
COSMO.

The direct ledger for both arms is

\[
\Delta G_{\rm solv}=\frac12\langle c_{\rm MACE-POLAR},f_{\rm reac}\rangle
+G_{\rm SMD-CDS}.
\]

Here $c_{\rm MACE-POLAR}=c_0$ for `response=frozen` and is the converged
learned source $c^*$ for `response=scf`.  The former makes no response or
stationarity claim; the latter is a fixed point, not a proved common
electronic-energy stationary state.

It deliberately excludes `E_MACE[V_reac]-E_MACE[gas]`: the present checkpoint
does not establish `dE_MACE/df = c`.  The complete scientific identity is
printed in provenance as `electrostatics_model=ddpcm` or `ddcosmo`,
`solute_source=point-multipole-l1`, `reaction_field_projector=local-jet`,
`nonpolar_model=pyscf-smd-cds`, and
`strict_original_smd_equivalence=false`.  Registration and successful
execution are usability evidence, not multi-solvent accuracy certification.
The corresponding all-solvent, two-equation runtime record is
[`route2-live-multisolv-ddpcm-ddcosmo-v2-execution-1bbef280.json`](benchmarks/route2-live-multisolv-ddpcm-ddcosmo-v2-execution-1bbef280.json):
one fixed acetone geometry completed all **11 x 2** rows, with at most 13 SCF
iterations and a worst half-coupling identity error of
`2.220446049250313e-16 eV`.  It deliberately contains no experimental labels
or error statistics and does not open forces for either pyddx profile.

The frozen-source experimental audits are separately source-bound:

* the ten-solvent/ten-chemistry MNSol pilot gives MAE/max errors of
  `0.8635/1.6454 kcal/mol` for ddPCM and `0.9154/1.6451 kcal/mol` for ddCOSMO;
* the historical-worst FreeSolv-12 panel gives `1.0843/1.9910 kcal/mol` for
  ddPCM and `1.0723/1.9549 kcal/mol` for ddCOSMO.

Both panels therefore fail the immutable requirement that every record be
strictly below `1.5 kcal/mol`.  The smaller averages do not certify accuracy,
and the mixed paired outcomes do not select a universally preferred equation.
See the source-bound artifacts in the benchmark README for the exact dataset,
geometry, checkpoint, profile, and claim boundaries.

For reproducibility only, the earlier field-conditioned ddPCM multi-solvent
experiment remains selectable as
`profile=smd-ddpcm-l15-n1202-multisolv-v1`.  It has a different ledger and is
not interchangeable with either direct half-coupling `v2` arm above.

### Bounded public conservative-force profile

The fixed-topology water profile has a separately versioned public-force
identity; it is deliberately **not** an upgrade of the energy-only `v1` or
`v2` profiles:

```text
#model=macepol-m
#sp(verbose=1)
#solv(implicit=water,method=smd,provider=fc-aswig,profile=smd-cpcm-fc-aswig-jgp94-d2-mace-aqueous-pcm-half-coupling-force-v3,response=scf,standard_state=1m,experimental=true)
```

It differentiates the direct operational scalar

\[
\Delta G_{\rm solv}=\frac12\langle c_{\rm MACE-POLAR},f_{\rm reac}\rangle
+G_{\rm fixed\mbox{-}topology\ aqueous\ SMD-CDS},
\]

using the un-mixed fixed-point adjoint and one fixed-cardinality amplitude
C-PCM/SMD-CDS geometry. It does **not** add the unproven
`E_MACE[V_reac]-E_MACE[gas]` term and does **not** claim a common stationary
electronic free energy. The profile is fail-closed to neutral, closed-shell,
connected 16--500 Da molecules in water, a nondegenerate JGP94 frame, the
four-branch D2-canonical local-jet MACE operator, a dense fixed-charge
conditioning screen, nominal SCF/adjoint residuals, and three-start local root
agreement. If any gate fails at a new geometry, no force is returned.

The release evidence was generated on a clean `e724cf5a` source tree: acetone
passed component finite differences, translation, rotation, a Cartesian path,
a closed coordinate loop, and short NVE conservation with three time-step
refinement; an
independent 20-atom 2-acetoxyethyl-acetate torsion and two-coordinate closed
loop also passed at a constant 1720 surface candidates. This is a bounded
conservative operational-scalar force capability, not proof for all chemical
classes, every conformation, long MD, or universal solvation accuracy.
The complete payloads, source hashes, and a clean current-head public-call plus
kinematic replay are frozen in
[`benchmarks/route2-fc-aswig-force-v3-release-evidence-v1.json`](benchmarks/route2-fc-aswig-force-v3-release-evidence-v1.json).

One additional profile isolates the rigid-rotation defect of MACE-POLAR's
default molecular long-range evaluator:

```text
#model=macepol-m
#sp(verbose=1)
#solv(implicit=water,method=smd,provider=pyddx,profile=smd-ddpcm-l15-n1202-gaff2-o-mace-kspace40-v1,response=scf,standard_state=1m,experimental=true)
```

This profile keeps the same checkpoint, ddPCM equations, GAFF/GAFF2
carbonyl-oxygen radius, PySCF CDS functional, SCF policy, and force derivative.
It changes only the MACE long-range evaluation operator: an FFF molecular graph
is arithmetic-mean centred in a fixed 40 Å cubic helper box and evaluated with
graph_longrange's forced periodic reciprocal-space path by setting
`use_pbc_evaluator=True`, while `pbc=False` remains unchanged. It is locked to
`graph_longrange==0.4.0`; a narrow dtype
bridge casts only the reciprocal molecular-correction field into the
float64 projection dtype. The evaluator also owns the exact chain rule for
its coordinate transform: every model force or density-position VJP is
projected by \(I-\mathbf 1\mathbf 1^\mathsf T/N\), and any Cartesian Hessian
is projected on both sides. No learned weight is changed.

The combined profile is indivisible: free-form `evaluator`, box-length, and
centering options are rejected, and a molecule that does not fit the 40 Å box
with the MACE cutoff fails closed. This is a non-default experimental operator
variant, not proof that it is equivalent to the default real-space evaluator.
Its results remain box-dependent and require convergence validation.

An execution-only variant is also available:

```text
#solv(implicit=water,method=smd,provider=pyddx,profile=smd-ddpcm-l15-n1202-gaff2-o-mace-kspace40-omp4-v1,response=scf,standard_state=1m,experimental=true)
```

It is identical to
`smd-ddpcm-l15-n1202-gaff2-o-mace-kspace40-v1` in checkpoint, cavity,
equations, grids, tolerances, SCF policy, CDS functional, and MACE long-range
operator. The only change is requesting and auditing four OpenMP threads for
the upstream pyddx solver instead of one. The original profile remains fixed at
one thread. Parallel speedup is not guaranteed: it depends on molecule, cavity,
grid, hardware load, and the fraction of wall time spent inside pyddx. Treat
the OMP4 profile as a reproducible performance candidate, not a different
physical model or a general performance claim.

### Route-2 runtime

Create one Python environment for MAPLE, MACE, pyddx, and PySCF.  Do **not**
put arbitrary complete `site-packages` directories in `PYTHONPATH`: that can
silently replace NumPy/SciPy beneath compiled Torch or continuum extensions.
The Route-2 extra supplies the exact version-locked continuum pair; MACE stays
separate because the user must select a CPU/CUDA-compatible Torch runtime.

```bash
python -m venv .venv-maple-route2
. .venv-maple-route2/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[route2-pyddx]' 'mace-torch==0.3.16'
python -c 'import pyddx, pyscf; print(pyddx.__version__, pyscf.__version__)'
```

The final command must print `0.8.0 2.13.1`.  MAPLE's `polar-1-m` loader then
populates the upstream MACE cache on its first use.  This installs an explicit
research optional dependency; it does not make the pyddx profiles a default or
production PES provider.

Install PCMSolver separately.  MAPLE neither vendors nor builds PCMSolver and
requires both the matching official Python input parser and the v1.1.12-style
`libpcm.so` C API:

```bash
export PCMSOLVER_LIBRARY=/absolute/path/to/libpcm.so
export PCMSOLVER_PYTHON_PATH=/absolute/path/to/pcmsolver/lib/python
```

When explicit paths are used, MAPLE treats `PCMSOLVER_LIBRARY` as
authoritative and never falls back to another soname. The parser must resolve
under the same installation prefix as that library; mixed installations fail
before cavity construction.

The pyddx derivative-evidence backend lazily requires exactly `pyddx==0.8.0`
and `pyscf==2.13.1`. They are an optional `route2-pyddx` extra rather than core
MAPLE dependencies, and MAPLE fails closed when the exact versions or their
compiled solvent libraries are unavailable. pyddx owns the ddPCM scalar energy,
forward/adjoint maps, and complete coordinate VJP; PySCF owns both the SMD CDS
scalar energy and analytic gradient. Energy from one continuum definition is
never combined with a derivative from another.

The model is loaded as `polar-1-m` in float64. MAPLE uses the pretrained
model's existing GTO field-response path and changes no learned weight. The
historical profiles supply a different reaction potential/gradient at each
atom through the upstream local affine-field projector. The exact-GTO
experiment instead supplies the complete checkpoint-native feature tensor
computed from the same discrete ASC. The official weights remain subject to
the upstream Academic Software License; MAPLE does not bundle or mirror them.

### Route-2 result and domain

MAPLE reports three distinct quantities:

```text
Gas-phase MLIP energy
Solvation free-energy correction (Delta G_solv, 1M(gas)->1M(solution))
Combined E_MLIP(gas)+Delta G_solv
```

The public result ledger is split into immutable **leaf** components
`solute_polarization`, `pcm_polarization`, `cds`, and `standard_state`, plus
checked **derived** totals `electrostatic` and `delta_g_solv`. The legacy
flat `components_hartree` view is read-only compatibility data; it must never
be summed over because it contains both leaves and their derived totals.

ASE's `free_energy` field is the same combined electronic-plus-solvation value,
not a thermochemical Gibbs free energy with vibrational or thermal terms. The
route uses a 1 M gas to 1 M solution convention, so it does **not** add the
commonly used 1 atm to 1 M `1.89 kcal/mol` correction.

The current research route is deliberately fail-closed:

- one conformer and single-point evaluation only; canonical element-radius
  profiles accept XYZ/inline/MOL2, while GAFF/GAFF2 radius profiles require
  MOL2 atom types;
- explicitly declared neutral closed-shell molecules (`0 1`), one connected
  component, no radicals or periodic cells;
- H/C/N/O/F/P/S/Cl/Br/I and molecular mass from 16 through 500 Da;
- the default and historical profiles remain water-only; only the explicitly
  named multi-solvent pyddx profile accepts its 11 registered solvents;
  optimization, Hessian, scan, TS search, and MD remain disabled;
- exact official MACE-POLAR-1-M only; `#charge`, D4, other models, and
  unlisted continuum providers are rejected.

The MOL2 path additionally rejects the formally charged Tripos types currently
covered by the Route-2 domain screen, including the registered salt/zwitterion
markers. XYZ and inline coordinates contain no per-atom formal-charge or
bond-order contract, so accepting them for canonical element-radius profiles does
**not** prove that a latent zwitterionic assignment is absent. A study that
requires strict zwitterion exclusion must supply topology-aware formal-charge
metadata; MAPLE does not infer formal charges from Cartesian distances.

The derivative capability boundary is likewise explicit:

| Continuum path | Energy | Complete same-energy coordinate VJP | Public capability |
| --- | --- | --- | --- |
| PCMSolver--GePol | yes | no; fails closed | energy only |
| PCMSolver--GePol + exact receiver-GTO field projection | yes | no; feature-space adjoint/VJP absent | explicit non-default energy experiment |
| pyddx ddPCM `l15/n1202` + PySCF SMD CDS | yes | research evidence only | energy only |
| pyddx ddPCM multi-solvent parameters + PySCF SMD CDS | yes | research evidence only | energy only; accuracy not certified |
| pyddx/GAFF2 + MACE reciprocal fixed-box40 | yes | research evidence only | energy only; non-default operator variant |
| FC-aSWIG C-PCM + fixed-topology aqueous SMD-CDS `force-v3` | yes | yes; fixed-cardinality same scalar | bounded public forces; water-only, per-geometry fail closed |
| synthetic contract oracle | test only | yes | no |
| external PySCF SWIG investigation | separate canary only | incomplete Route-2 integration | no |

`continuum_coupled_solvation_coordinate_gradient()` accepts the complete
coordinate VJP only from the same reaction-field object used for its
forward/adjoint maps. The current PCMSolver map does not implement that
contract and remains energy-only. The explicitly named pyddx/PySCF profile
implements a complete single-point correction derivative for research
evidence, but deliberately does not expose `forces`. It remains blocked from
PES tasks by rotation, continuity, energy-conservation, and variational
stationarity gates.

Every PCMSolver evaluation retains `manifest.json`, the human and parsed
PCMSolver inputs, PCMSolver/PEDRA cavity side files, `route2-state.npz`, and
`route2-result.json` under `<output>.implicit/`. The pyddx research provider
retains `manifest.json`, `route2-ddpcm-state.npz`, and
`route2-ddpcm-result.json` there. Legacy provider files remain contained rather
than written into the launch directory. Both provenance records label the
model output correctly as a coarse-grained net charge density rather than a QM
electron density. The common wrapper also writes
`route2-public-result-ledger.json` with immutable leaf terms and their checked
derived totals. Each evaluation additionally receives a distinct immutable
`route2-public-results/<run-id>.json` record containing the **current** ASE
geometry, its SHA-256, the selected-profile digest, evaluated-property scope,
and linked manifest/content digests. A force-bearing record additionally embeds
the per-geometry fail-closed `force_admission` certificate; an energy-only
record carries `forces_evaluated=false` and a null certificate. The
`route2-public-result-ledger.json` file is only the latest-record alias and must
not be used to erase earlier evaluation evidence.

FreeSolv remains a secondary energy diagnostic; it does not define Route 2 and
cannot certify a solution-phase PES. The retained **pyddx** derivative evidence
is not a public force capability. A bounded
three-fixed-geometry QM/experiment comparison and four-geometry electronic
conformer panel are frozen in
[`route2-qm-fidelity-v1.json`](benchmarks/route2-qm-fidelity-v1.json);
component-resolved differences must be read with the total because polarization
errors currently cancel. All three fixed-conformer energies have been
reconfirmed in clean worktrees at `5746f24`, `f3e9892`, and `e34abc5`, with no
tracked `maple/` runtime-source change among those heads. One clean
source-geometry analytic-force canary for 2-acetoxyethyl acetate at `d72dfba`
reproduces the historical correction force within
`2.66e-14 eV/angstrom`, converges its adjoint below the configured tolerance,
and emits no PCMSolver or legacy `primary` warning. Its direct
`5e-4`-angstrom central-difference pair was then rerun against the same runtime:
the largest analytic component is `0.7570000052 eV/angstrom`, the finite
difference is `0.7570030893 eV/angstrom`, and their absolute difference is
`3.084e-6 eV/angstrom`. Both displaced energies converged in 18 root
iterations with zero PCMSolver or `primary` warnings. A separately
pre-registered central C--C torsion pair at \(\pm0.5^\circ\) was also rerun at
`d72dfba`: its analytic and central-difference generalized forces are
`0.0452064347` and `0.0451054120 eV/rad`, an absolute error of
`1.01023e-4 eV/rad` (`0.2235%`). The same locked coordinate was then evaluated
at \(\pm1.0^\circ\). Its central difference is
`0.0450446898 eV/rad`, with an absolute error of `1.61745e-4 eV/rad` and a
relative error of `0.3578%`. Refining from \(1.0^\circ\) to \(0.5^\circ\)
therefore reduces the absolute error by `6.07221e-5 eV/rad`, to `0.62458` of
the coarse-step error. All four displaced energy roots converged in 18
iterations with zero PCMSolver or legacy `primary` warnings. The
\(1.0^\circ\) pair took `151.35 s` total wall time, but this contended single
run is not a speed claim. The original \(0.5^\circ\) evidence runner exited
only while serializing a NumPy Boolean after writing both immutable point
records; a read-only finalizer verified those records without rerunning either
energy.

A third pre-registered pair at \(\pm0.25^\circ\) then converged in 18
iterations per point with correction energies `-0.3396924890` and
`-0.3400866505 eV`. Its finite-difference generalized force is
`0.0451675753 eV/rad`, only `3.88594e-5 eV/rad` (`0.0860%`) from the analytic
value. Nevertheless, the pre-registered smooth second-order refinement test
**fails**: the finite-difference drift is `6.21634e-5 eV/rad`, slightly larger
than the preceding `6.07221e-5 eV/rad`, and the observed order is `-0.0338`
rather than the locked `1.5--2.5` interval. The nonzero runner exit records
those failed gates after both immutable results were written; it is not an
execution or serialization failure. PCMSolver and legacy `primary` warning
counts remain zero, and the `148.50 s` wall time is not a speed claim.

Component central differences localize the failure: gas MACE and CDS show
orders `2.008` and `2.001`, while solvent-intrinsic MACE and PCM polarization
show only `0.544` and `0.212`; their coupled total has order `-0.034`. Thus the
non-asymptotic behavior lies in the self-consistent electrostatic coupling
block, not the gas MACE geometry response or CDS term.

A subsequent pre-registered diagnostic then reused the six immutable torsion
states and performed exactly one independent \(+0.25^\circ\) energy replicate.
The replicate agrees to `2.22e-16 eV` in energy and `2.22e-16 e` in density,
all 18 validity gates pass, and no PCMSolver or legacy `primary` warning
occurs. Across the
\(0.5^\circ\to0.25^\circ\) interval, formal sphere/Lebedev active-set
comparisons change by 19 points on the minus side and 27 on the plus side.
Frozen density carries `73.8%` of the fine-drift L1 norm; within that block,
fixed-center-density PCM carries `67.6%` and reaction-map-through-MACE carries
`31.9%`. The valid result therefore associates the failed refinement with
active-set-associated explicit continuum geometry response. It does **not**
prove that active-set change alone is causal, nor split operator-only from
cavity-only response, because pyddx 0.8.0 exposes only their combined
provider-consistent coordinate response.

No finer step is being used to overwrite the failed asymptotic gate, and this
diagnosis does not authorize a production change for the old pyddx profile.
The four-geometry flexible panel, closed-loop panel, and second-molecule force
evidence in this section remain historical. The newer FC-aSWIG force-v3
evidence is described above; broader flexible/relaxed-path continuity,
additional chemical classes, complete conformer thermochemistry, and long-NVE
conservation remain open. Optimization or MD with all other Route-2 profiles
remains out of scope, and force-v3 still fails closed outside its stated
per-geometry domain.

See [FORMULAS_AND_REFERENCES.md](FORMULAS_AND_REFERENCES.md) for equations and
the literature ledger, and [VALIDATION_STATUS.md](VALIDATION_STATUS.md) for the
passing engineering checks and still-open scientific gates. The concrete
derivation and implementation sequence is in
[ROUTE2_FORCE_ROADMAP.md](ROUTE2_FORCE_ROADMAP.md).
