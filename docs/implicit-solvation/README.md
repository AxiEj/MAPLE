# MAPLE implicit solvation: Route 2 research/innovation route

This branch contains only the MACE-POLAR + SMD continuum route. It does not
contain the fixed-charge PB/GB implementation. The default
PCMSolver--IEFPCM/GePol profile remains an energy proof-of-concept. Separate,
explicit pyddx ddPCM profiles expose single-point research force candidates,
including one versioned multi-solvent parameter profile. A separate
PCMSolver profile changes only the ASC-to-MACE reaction-field projection from
the historical local first-order jet to the checkpoint-native \(l\le1\) GTO
integrals. A newer, still non-default profile combines that receiver with the
SMD intrinsic Coulomb-sphere electrostatic cavity (`probe=0`, no added
spheres, explicit water dielectric). None is yet a complete MAPLE solution-phase PES.
All calculations therefore require `experimental=true`.

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

## Route-2 contract: self-consistent polarizable MLIP--PCM/SMD coupling

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

The research objective is mutual polarization: the MACE-POLAR representation
generates the solute electrostatic potential, PCM returns a reaction field, and
that reaction field is fed back through the unmodified MACE-POLAR field-response
path until convergence. Fixed-charge PB/GB and post hoc one-way polarization
are separate routes.

The public v1 input is:

```text
#model=macepol-m
#sp
#solv(implicit=water,method=smd,response=scf,standard_state=1m,experimental=true)

0 1
MOL2 molecule.mol2
```

The omitted locked defaults are `provider=pcmsolver` and
`profile=smd-iefpcm`.  `response=scf` is the public default;
`response=frozen` is retained only as a diagnostic that solves PCM once from
the gas-phase density. Canonical element-radius profiles also accept
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

The separately named single-point force candidate must be selected exactly:

```text
#model=macepol-m
#sp(verbose=1)
#solv(implicit=water,method=smd,provider=pyddx,profile=smd-ddpcm-l15-n1202-v1,response=scf,standard_state=1m,experimental=true)

0 1
MOL2 molecule.mol2
```

This profile fixes `lmax=15`, 1202 Lebedev points per sphere, `mixing=1.0`,
the documented ddPCM and adjoint tolerances, and one same-energy derivative
chain. It does not accept the PCMSolver-specific `cavity_policy`. The profile
name records a bounded research candidate, not a universal grid or accuracy
certification. The separately named
`smd-ddpcm-l15-n1202-gaff2-o-v1` profile applies the already documented
GAFF/GAFF2 `o` carbonyl-oxygen radius change to the same numerical candidate;
it is not the canonical default or a broad-accuracy claim.

The multi-solvent parameter profile must be selected explicitly:

```text
#model=macepol-m
#sp(verbose=1)
#solv(implicit=acetonitrile,method=smd,provider=pyddx,profile=smd-ddpcm-l15-n1202-multisolv-v1,response=scf,standard_state=1m,experimental=true)
```

It registers 11 solvents from the tested PySCF 2.13.1 SMD solvent database:
water, methanol, ethanol, acetonitrile, dimethyl sulfoxide,
dimethylformamide, tetrahydrofuran, chloroform, dichloromethane, toluene, and
hexane. The profile uses each solvent's tabulated dielectric, the
solvent-acidity-dependent SMD oxygen radius, the tested PySCF element-radius
mapping, and the matching PySCF SMD CDS energy/gradient. Water-only profiles
retain their original `78.39` dielectric but now use the corrected
atomic-number-indexed SMD/SMD18 P/S/Cl radii (`2.12/2.49/2.38 angstrom`).
Artifacts generated with the former shifted mapping remain bound to their old
execution commits and are stale evidence for the corrected profile.

The complete scientific identity is printed in provenance:
`electrostatics_model=ddpcm`, `solute_source=point-multipole-l1`,
`reaction_field_projector=local-jet`, and
`nonpolar_model=pyscf-smd-cds`, with
`strict_original_smd_equivalence=false`. Thus `method=smd` is a compatible
input label for an **SMD-CDS-augmented MACE-POLAR/ddPCM hybrid**, not a claim
that the coarse residual multipoles reproduce original electron-density SMD.
Registration and successful execution are not multi-solvent accuracy
validation.

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

Install the exact MACE release used by the contract and let its official
foundation-model loader populate the upstream cache:

```bash
pip install 'mace-torch==0.3.16'
```

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

The explicit force candidate instead lazily requires exactly `pyddx==0.8.0`
and `pyscf==2.13.1`. MAPLE does not declare either optional research runtime as
a core dependency and fails closed when the exact versions or their compiled
solvent libraries are unavailable. pyddx owns the ddPCM scalar energy,
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

| Continuum path | Energy | Complete same-energy coordinate VJP | Public |
| --- | --- | --- | --- |
| PCMSolver--GePol | yes | no; fails closed | energy only |
| PCMSolver--GePol + exact receiver-GTO field projection | yes | no; feature-space adjoint/VJP absent | explicit non-default energy experiment |
| pyddx ddPCM `l15/n1202` + PySCF SMD CDS | yes | yes | explicit single-point research force candidate |
| pyddx ddPCM multi-solvent parameters + PySCF SMD CDS | yes | yes | explicit capability candidate; accuracy not certified |
| pyddx/GAFF2 + MACE reciprocal fixed-box40 | yes | yes | explicit non-default operator-variant candidate |
| synthetic contract oracle | test only | yes | no |
| external PySCF SWIG investigation | separate canary only | incomplete Route-2 integration | no |

`continuum_coupled_solvation_coordinate_gradient()` accepts the complete
coordinate VJP only from the same reaction-field object used for its
forward/adjoint maps. The current PCMSolver map does not implement that
contract and remains energy-only. The explicitly named pyddx/PySCF profile
implements the complete single-point correction derivative and exposes
`forces`, but remains blocked from PES tasks by rotation, continuity, and
energy-conservation gates.

Every PCMSolver evaluation retains `manifest.json`, the human and parsed
PCMSolver inputs, PCMSolver/PEDRA cavity side files, `route2-state.npz`, and
`route2-result.json` under `<output>.implicit/`. The pyddx force candidate
retains `manifest.json`, `route2-ddpcm-state.npz`, and
`route2-ddpcm-result.json` there. Legacy provider files remain contained rather
than written into the launch directory. Both provenance records label the
model output correctly as a coarse-grained net charge density rather than a QM
electron density.

FreeSolv remains a secondary energy diagnostic; it does not define Route 2 and
cannot certify a solution-phase PES. The first public single-point
energy-consistent force candidate is now wired through MAPLE. A bounded
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
diagnosis does not authorize a production change. The next bounded stage is an
upstream-backed smooth cavity/operator research profile, not a local switching
patch or a mixing/radius retune. The four-geometry flexible panel, closed-loop
panel, and second-molecule force evidence remain historical. Broader
flexible/relaxed-path continuity, additional chemical classes, complete
conformer thermochemistry, and short NVE conservation remain open.
Optimization, scans, transition states, and MD remain out of scope.

See [FORMULAS_AND_REFERENCES.md](FORMULAS_AND_REFERENCES.md) for equations and
the literature ledger, and [VALIDATION_STATUS.md](VALIDATION_STATUS.md) for the
passing engineering checks and still-open scientific gates. The concrete
derivation and implementation sequence is in
[ROUTE2_FORCE_ROADMAP.md](ROUTE2_FORCE_ROADMAP.md).
