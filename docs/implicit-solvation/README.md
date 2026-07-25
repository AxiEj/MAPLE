# Route 1: additive fixed-charge PB/GB implicit solvation

This is MAPLE's baseline/product implicit-solvation route. It composes any
registered gas-phase molecular MLIP that passes the Route 1 composition
capability gate with an additive, auditable fixed-charge solvent correction:

```text
E_solution(R) = E_MLIP,gas(R)
              + G_polar(R,q_fixed)
              + G_nonpolar(R)
```

No gas-phase MM energy, retraining, hydration-label residual model, or
chemistry-specific fitted selector is part of Route 1. Innovation is expected
at the integration boundary: common MLIP composition, charge/radius/nonpolar
providers, provenance, force consistency, compatibility checks,
conformer-aware evaluation, and explicit-inner/implicit-outer workflows.
Innovation does not need to alter the PB/GB equations, but it must preserve the
fixed-charge additive formula above.

Registration alone is not treated as compatibility. `CalcABC` backends inherit
the shared composition path; a non-`CalcABC` backend must explicitly declare
and implement the equivalent capability or fail before construction.

The core product surface is fast SP, OPT, and SCAN/PES. Explicit numerical
FREQ, MD, MBAR/TI, conformer-aware evaluation, and prebuilt
explicit-inner/implicit-outer composition are optional extensions over the
same additive potential; they do not redefine Route 1 or replace its core
FreeSolv, provider-parity, and complete-potential force checks.

The default validation domain is one neutral, closed-shell, connected organic
molecule in water, supplied as a Tripos MOL2 file with explicit bonds and
hydrogens. An experimental prebuilt-cluster extension is described below; it
does not change the single-solute default or create an absolute hydration
free-energy claim.

The frozen product semantics, claim boundaries, gates, and speed terminology
are collected in [`ROUTE1_PRODUCT_SPEC.md`](ROUTE1_PRODUCT_SPEC.md).

All implicit-solvation providers currently require `experimental=true`.  This
flag records that the implementation gates pass locally while the public
scientific benchmark gate remains open; it is not a request to mix arbitrary
parameter profiles.

## Input contract

MOL2 fixed charges (RESP, RESP2, or another documented charge set):

```text
#model=aimnet2
#sp(verbose=1)
#charge(source=mol2,label=resp2)
#solv(implicit=water,method=gb,model=obc2,nonpolar=ace,experimental=true)

0 1
MOL2 molecule.mol2
```

MAPLE-orchestrated charge generation:

```text
#charge(source=maple)
#charge(source=maple,method=am1bcc,geometry=keep)
#charge(source=maple,method=abcg2,geometry=keep)
#charge(source=maple,method=qeq-gto,mode=fixed)
#charge(source=maple,method=qeq-gto,mode=polarizable)
```

`#charge(source=maple)` defaults to AM1-BCC with fixed charges and
`geometry=keep`. ABCG2 requires explicit `method=abcg2` selection and is never
chosen as an automatic fallback. Omitting the entire `#charge(...)` directive
still fails closed for implicit solvation.

Fixed QEq-GTO and polarizable CQEq-GTO/GB are frozen experimental research
controls. They require explicit `method=qeq-gto` selection, are never selected
as defaults or provider fallbacks, and are not accuracy-certified.
Polarizable CQEq-GTO/GB is not a Route 1 fixed-charge product profile. Current
Route 1 certification work is limited to MOL2 fixed charges, AM1-BCC, and
explicitly selected ABCG2.

`qeq-gto` performs the full published hydrogen SCF update for both the
idempotential and screening exponent in fixed mode.  `mode=polarizable` does
not reuse that nonvariational fixed-point equation: it switches to the
consistent-QEq (CQEq) derivative, solves the nonlinear charge-constrained
minimum of CQEq plus the GB polar energy, and applies the envelope theorem only
after KKT and projected-Hessian minimum gates pass.

AmberTools is an optional executable provider.  It can be kept outside the
main MAPLE environment to avoid dependency conflicts:

```bash
conda create -n maple-ambertools --override-channels -c conda-forge python=3.11 'ambertools=26.0'
```

Either activate that environment before running MAPLE or point the charge
provider at its Antechamber wrapper explicitly:

```text
#charge(source=maple,method=am1bcc,geometry=keep,executable=/path/to/maple-ambertools/bin/antechamber)
#charge(source=maple,method=abcg2,geometry=keep,executable=/path/to/maple-ambertools/bin/antechamber)
```

The optional energy-only CHA-GB/cavity-dispersion single-point provider uses
the same AmberTools installation:

```text
#model=aimnet2
#sp
#charge(source=maple,method=am1bcc,geometry=keep,executable=/path/to/maple-ambertools/bin/antechamber)
#solv(implicit=water,method=gb,provider=ambertools,model=chagb,profile=chagb-bondi-pbsa-inp2,nonpolar=cavity-dispersion,executable=/path/to/maple-ambertools/bin/gbnsr6,experimental=true)

0 1
MOL2 gaff-typed-molecule.mol2
```

`gbnsr6`, `pbsa`, `parmchk2`, and `tleap` must be siblings in that AmberTools
`bin` directory. The input atom types must be GAFF/GAFF2-compatible; MAPLE
preserves them instead of silently accepting Antechamber retyping. Unknown
nonbonded types fail closed. GBNSR6 calls are serialized in-process and across
POSIX processes because concurrent AmberTools 26 runs showed allocator
failures while identical serial runs passed.

`geometry=keep` always preserves the submitted MOL2 geometry.  Antechamber
outputs, commands, stdout, stderr, atom mapping, and charge sums are retained in
`<output>.implicit/`.  `geometry=provider` explicitly adopts the geometry
written by the provider.  Fixed charges are generated/read once and remain
frozen through SP, OPT, SCAN/PES, explicit numerical FREQ, and admitted
NVE/NVT MD.

MOL2 charges are never silently normalized.  Their sum must match the declared
molecular charge within `1e-4 e`.

## Complete-potential numerical frequency

Force-capable OpenMM GB compositions admit an explicit numerical FREQ task:

```text
#model=ani2x(hessian=numerical)
#freq(method=mw,ilowfreq=2)
#charge(source=mol2,label=am1bcc-frozen)
#solv(implicit=water,method=gb,provider=openmm,model=obc2,profile=obc2-mbondi2,nonpolar=ace,experimental=true)

0 1
MOL2 molecule.mol2
```

The opt-in `hessian=numerical` is mandatory because MAPLE central-differences
the **complete reported MLIP-plus-GB force**. It takes two force evaluations
per movable Cartesian degree of freedom. A backend analytic Hessian contains
only the gas MLIP term and therefore remains fail-closed with implicit
solvation. Energy-only APBS PB and AmberTools CHA-GB providers cannot enter
this path. The admitted implicit-solvent frequency method is mass-weighted
`method=mw`, with `ilowfreq` restricted to `0`, `1`, `2`, or `3`;
non-mass-weighted and unimplemented dual-mode requests fail during parsing.
The low-frequency choices are explicit physical approximations: `0` is
harmonic RRHO, `1` applies the Truhlar entropy cap, `2` applies Grimme's
entropy-only harmonic/free-rotor interpolation, and `3` applies the
Otlyotov--Minenkov extension to both entropy and complete vibrational internal
energy, including ZPE. The rotor entropy uses a finite effective inertia built
from the mean molecular moment of inertia, rather than the divergent
frequency-only free-rotor limit. MAPLE's default switching frequency remains
`100 cm-1`; alternative thresholds are sensitivity choices, not fit
parameters.
Constrained FREQ also fails closed: MAPLE does not yet implement the reduced
active-coordinate Hessian, mass matrix, and rigid-body projection required for
ASE `FixAtoms` or `FixInternals`. Silently projecting a constrained full-space
Hessian would produce incorrect modes.
The same checks run in the Python computation API: the correction must declare
fixed charges and force support, the calculator must declare Route 1
composition support, the Hessian mode must be numerical, and the atoms must be
non-periodic. Missing capability metadata is rejected rather than inferred.

This task supplies local vibrational curvature of the same composed potential
used by SP/OPT/SCAN. The frequency dispatcher still reports its existing
ideal-gas translational/rotational RRHO correction and prints an explicit
boundary warning. That correction is not a solution-standard-state Gibbs
energy, not an absolute hydration free energy, and not evidence that
multi-structure RRHO improves FreeSolv accuracy. Gas/solution ensembles,
conformer populations, standard-state handling, and uncertainty remain
separate scientific gates.

The low-frequency formulas follow Grimme
(DOI `10.1002/chem.201200497`), the conformer-ensemble treatment discussed by
Pracht and Grimme (DOI `10.1039/D1SC00621E`), and the internally consistent
energy extension of Otlyotov and Minenkov (DOI `10.1002/jcc.27129`). These
references support the thermochemistry formulas, not an accuracy claim for the
Route 1 hydration estimator.

## Prebuilt explicit inner / implicit outer

Route 1 can apply its same additive correction to a user-supplied
solute-plus-inner-solvent cluster:

```text
#model=maceoff23m
#opt
#charge(source=mol2,label=prebuilt-fixed-cluster)
#solv(implicit=water,inner=prebuilt,method=gb,model=obc2,nonpolar=ace,experimental=true)

0 1
MOL2 solute-water-cluster.mol2
```

The MOL2 must contain at least two disconnected, internally bonded components
and one fixed charge per atom for the **complete** cluster. MAPLE creates one
synthetic OpenMM residue per connected component, records component count and
charge sums, and applies the outer PB/GB and nonpolar terms to all cluster
atoms. `inner=prebuilt` accepts only `#charge(source=mol2,mode=fixed,
geometry=keep)`. It never invokes AM1-BCC or another charge generator on a
disconnected cluster. The runtime endpoint is restricted to OpenMM GB with
ACE or LCPO. The prepared correction binds charges, radii, and MOL2 topology to
a stable per-atom identity array; atom deletion, element substitution, or atom
reordering fails closed, including reorderings between atoms of the same
element.

This mode deliberately does not reuse the existing coordinate-only
`explicit=water` generator. `explicit=...` and `implicit=...` remain mutually
exclusive because generated solvent coordinates do not yet carry the complete
bond, residue, charge, and sampling contract required by the continuum
provider.

For the whole cluster coordinate \(R\), the runtime quantity is

\[
E_{\mathrm{cluster,outer}}(R)=E_{\mathrm{MLIP,gas}}(\mathrm{cluster};R)
 +G_{\mathrm{polar,outer}}(R,q_{\mathrm{fixed}})
 +G_{\mathrm{nonpolar,outer}}(R).
\]

It is a **fixed-shell cluster-continuum configurational potential**, usable for
GB SP, OPT, and SCAN when both the MLIP and outer provider supply the requested
derivative. “Fixed shell” means that component membership and coordination
number are fixed by the input; atomic coordinates remain variables in OPT and
SCAN. The result is not printed as `Delta G_solv` and must not be compared
directly with FreeSolv. Absolute solvation free energy additionally requires
a cluster-formation/occupancy free energy, solvent-cluster or solvent chemical
potential reference, standard-state corrections, and an ensemble over cluster
conformations and coordination numbers.

The frozen engineering smoke runs real MAPLE SP, two-iteration OPT, and
three-point rigid SCAN jobs for MACE-OFF23m, AIMNet2, and ANI2x. All nine jobs
pass the topology, energy-closure, output-semantics, and audit checks on one
illustrative methanol-plus-water cluster. One combined-potential force
component per MLIP also matches a centered `0.003 A` energy difference within
the frozen `5e-5 Hartree/A` tolerance. This is task-plumbing evidence, not
cluster sampling or hydration-accuracy validation. See
[`benchmarks/prebuilt_inner_outer_protocol.json`](benchmarks/prebuilt_inner_outer_protocol.json)
and
[`benchmarks/route1-prebuilt-inner-outer-smoke-2026-07-24.json`](benchmarks/route1-prebuilt-inner-outer-smoke-2026-07-24.json).

## GB methods

| MAPLE model | Amber selector | locked radii profile | provider |
|---|---:|---|---|
| `hct` | `igb=1` | `hct-mbondi` | OpenMM `GBSAHCTForce` |
| `obc1` | `igb=2` | `obc1-mbondi2` | OpenMM `GBSAOBC1Force` |
| `obc2` | `igb=5` | `obc2-mbondi2` | OpenMM `GBSAOBC2Force` |
| `gbn` | `igb=7` | `gbn-bondi` | OpenMM `GBSAGBnForce` |
| `gbn2` | `igb=8` | `gbn2-mbondi3` | OpenMM `GBSAGBn2Force` |
| `chagb` | CHA-GB/ALPB | `chagb-bondi-pbsa-inp2` | AmberTools GBNSR6 `EGB` + PBSA `ECAVITY+EDISPER` (SP energy only) |

OBC-II is the Route 1 development default.  `nonpolar=ace` is the default;
`nonpolar=lcpo` requires OpenMM 8.5 or newer.  `nonpolar=none` is diagnostic
only and requires `experimental=true`. The product OpenMM platform default is
CPU with one thread and deterministic forces; `platform=Reference` remains an
explicit correctness/parity control. OpenMM returns both correction energy and
conservative correction force, so fixed-charge GB is available to SP, OPT,
SCAN/PES, explicit numerical FREQ, and non-periodic NVE/NVT MD. NPT remains
rejected because this implicit-solvent release is non-periodic.
Polarizable-QEq MD and `inner=prebuilt` MD remain fail-closed pending separate
conservative sampling and fixed-shell occupancy contracts.

A minimal admitted canonical sampling input is:

```text
#model=maceoff23m
#md(ensemble=nvt,steps=10000,timestep=0.1,temperature=298.15)
#charge(source=mol2,label=am1bcc-frozen)
#solv(implicit=water,method=gb,model=obc2,nonpolar=ace,experimental=true)

0 1
MOL2 molecule.mol2
```

This runs dynamics on the same combined potential that MAPLE reports. It does
not by itself turn a trajectory into a hydration free energy; a validated
partition-function estimator, equilibration/convergence protocol, overlap
diagnostics, and uncertainty analysis are still required.

`nonpolar=ace` and Amber `gbsa=1` are not aliases: Amber's latter selector is
LCPO.  MAPLE uses LCPO only for independent Amber complete-energy/force parity
and keeps ACE as the predeclared experimental-accuracy profile.  OpenMM 8.5.2
does not expose Amber's phosphorus-specific GBn2 alpha/beta/gamma parameters;
MAPLE therefore rejects `model=gbn2` for P-containing molecules instead of
silently using the generic fallback. Its generic GBn2 expression also differs
from Amber's signed near-pair descreening branch for sulfur's negative
screening radius, so S-containing GBn2 inputs fail closed as well. The other
four GB models remain available for those elements.

Radius assignment and nonpolar construction are first-class provider objects,
not undocumented switches inside the composition layer. The OpenMM radius
provider supplies the exact upstream standard-parameter array used to build
the GB force. For synthetic generic small-molecule `MOL` residues, the GBN2
provider preserves Amber's `1.50 A` neutral carbonyl-oxygen `mbondi3` radius
instead of accepting OpenMM's carboxylate-like connectivity classification.
The LCPO provider starts from OpenMM's table but uses submitted GAFF `o`/`o2`
types to distinguish sp2/nitro oxygen from carboxylate oxygen. ACE/LCPO/none
providers configure the actual OpenMM force term, and every adjustment is
retained in the nested solvation provenance.

LCPO remains an explicit compatibility/parity option, not an accuracy
replacement for ACE. It is parameterized for only 454/526 development
molecules; on that identical supported subset its MAE is
`2.250 kcal/mol`, versus `1.801 kcal/mol` for ACE. The paired
ACE-minus-LCPO gain is `-0.449 kcal/mol` with 95% interval
`[-0.581, -0.318]`, while the shared OBC-II polar component is exactly
unchanged. This is a post-hoc development screen, not an independent or
label-blind confirmation.

A label-blind development benchmark also evaluates AM1-BCC with AmberTools
CHA-GB/GBNSR6 and with PBSA cavity plus dispersion. On the frozen 526-molecule
development partition, MAE/RMSE improve from `1.760/2.537` for OBC-II/ACE to
`1.322/1.854 kcal/mol` for CHA-GB/cavity-dispersion. The exact endpoint is now
available as the explicit SP-only `provider=ambertools,model=chagb` profile.
It reads only `EGB`, `ECAVITY`, and `EDISPER`; no gas or bonded MM energy enters
the Route 1 target. It has not replaced the runtime default, and a
force-consistent OPT/SCAN path plus independent confirmation remain open gates.

A Route 1 component-attribution audit then crossed the already frozen polar
and nonpolar components instead of assuming which term caused that gain.
OBC-II/PBSA cavity-dispersion worsens to `2.103/3.011 kcal/mol` MAE/RMSE, and
CHA-GB/ACE gives `1.819/2.412`; only the joint CHA-GB/PBSA pairing retains
`1.322/1.854`. A descriptive endpoint Shapley allocation is
`0.361/0.077 kcal/mol` for the polar/nonpolar swaps, but its large interaction
means it does not prove microscopic causality. Therefore neither isolated
component swap is promoted, and the next conservative candidate must justify
the polar, radii, cavity, and dispersion parameterization together. The
526-record evidence is
`benchmarks/route1-chagb-component-attribution-2026-07-25.json`.

A separate pre-registered evaluation has now computed the legacy 116-case
FreeSolv reserve without calling it blind confirmation: its labels were already
present in the older preparation artifact. The energy phase used a newly
frozen label-free source manifest and completed 116/116 cases before sealing
an artifact with no experimental fields. Only then were the labels read.
OBC-II/ACE gives MAE/RMSE/max `1.791/2.464/9.622 kcal/mol`, while the coupled
CHA-GB/PBSA endpoint gives `1.301/1.846/6.376`. The paired MAE gain is
`0.490 kcal/mol` with 95% interval `[0.215, 0.782]`, passing the frozen
materiality, interval, RMSE, maximum-error, and coverage gates. Therefore the
coupled endpoint is retained as an explicit SP-only fixed-geometry accuracy
profile; OBC-II/ACE remains the force-capable default, and no tuning or
derivative claim is opened. The protocol and artifacts are
`benchmarks/route1_freesolv_reserve_protocol.json`,
`benchmarks/route1-freesolv-reserve-energy-2026-07-25.json`, and
`benchmarks/route1-freesolv-reserve-score-2026-07-25.json`.

The remaining GBr6 analytical candidate was also screened on all 526
development molecules without fitting. Combined with the same PBSA
cavity/dispersion term, it gives MAE/RMSE `2.257/3.451 kcal/mol`, worse than
both OBC-II/ACE and CHA-GB/cavity-dispersion. Its pinned released program has
no force/gradient output interface, so no GBr6 runtime provider is added.

The maintained ddX implementation was then audited as an external conservative
PCM candidate. An isolated `pyddx 0.8.0` build passes its shipped Python tests,
and the zero-salt ddPCM polar derivative passes all 69 methyl-hexanoate
Cartesian finite differences (primary-step RMSE/maximum
`0.000404/0.001504 kJ/mol/A`). A label-free 20-case numerical check also
passes. The product decision is nevertheless negative: the full
AM1-BCC/ddPCM/mbondi2/ACE development endpoint gives MAE/RMSE
`1.782/2.881 kcal/mol`, versus `1.760/2.537` for OBC-II/ACE, and its local
polar force call is roughly `360x` slower than the current OpenMM correction.
The expanded-radius cavity is much worse (`5.829/6.706 kcal/mol`). ddX remains
an external reference audit; `pyddx` is not a MAPLE dependency or runtime
provider.

## PB methods

```text
#charge(source=mol2,label=resp2)
#solv(implicit=water,method=pb,model=lpb,provider=apbs,profile=generic-mbondi2,nonpolar=apbs,experimental=true)
```

The APBS provider writes PQR, evaluates documented solvated/reference LPBE
blocks, subtracts their electrostatic energies, and adds an APBS APOLAR term.
The locked generic nonpolar profile is the APBS-style SASA reduction
`gamma*A` with `gamma=0.105 kJ mol^-1 A^-2`, zero pressure, and zero bulk
solvent density.  The default `97^3` grid at `0.33 A` follows the APBS solvation
example; alternative `grid_points` must have the nlev=4 form `c*32+1`, and
MAPLE refuses grids that do not enclose the molecular surface plus the SPL2
boundary margin.
PB is SP-energy-only. Its separately tested spline-force path fails the
independent all-component, cross-grid, and complete polar-plus-nonpolar force
gates described below.

The APBS PQR radii and APOLAR input block are likewise produced by explicit
radius and nonpolar providers. This extraction changes no PB mathematics and
does not imply that profiles can be mixed outside their locked pairing.

A frozen development screen tested the only remaining label-free PB
composition suggested by the provider evidence:
APBS molecular-surface LPBE polar energy plus the OpenMM OBC-II/mbondi2 ACE
nonpolar component.  It reads no gas-phase MM energy and applies no fit or
residual.  Across all 526 development molecules, replacing APBS APOLAR with
ACE reduces MAE/RMSE from `4.415/4.984` to `1.658/2.535 kcal/mol`, but the MAE
gain over OBC-II/ACE is only `0.102 kcal/mol`, below the prospectively frozen
`0.15` materiality gate.  More importantly, the preselected 20-case
`97^3 @ 0.33 A` versus `129^3 @ 0.25 A` polar-energy check changes by as much
as `0.671 kcal/mol` (P90 `0.455`).  The candidate fails its numerical-stability
gate, remains worse than CHA-GB/cavity-dispersion (`1.322/1.854`), and is not
added to the runtime.

A post hoc, label-free numerical follow-up showed that the two finer grids do
agree: `129^3 @ 0.25 A` versus `161^3 @ 0.20 A` has maximum/P90 changes
`0.175/0.148 kcal/mol`.  The full 526-molecule `129^3` endpoint was therefore
screened separately.  It improves PB/ACE to MAE/RMSE `1.629/2.462`, but its
paired MAE gain over OBC-II/ACE is still only `0.131 kcal/mol`, below the same
frozen `0.15` materiality threshold.  Numerical convergence does not rescue
the accuracy/product decision: the fine-grid candidate is also rejected.

APBS can print electrostatic forces only for spline-based surfaces; the pinned
3.4.1 executable aborts a force request for the product's molecular surface.
An isolated `srfm=spl4` probe therefore checked all 69 force components on the
same `129^3` and `161^3` grids. The fine-grid force-to-energy finite-difference
RMSE/maximum are `0.199/1.103 kJ/mol/A`, while coarse-to-fine force
RMSE/maximum are `0.483/2.579 kJ/mol/A`; the fine-grid net-force norm is
`2.336 kJ/mol/A`. APBS also warns that spline surfaces require radius/
force-field reparameterization, which generic mbondi2 does not provide.
The locked APBS `gamma*SASA` force separately misses its matching-step energy
finite difference with RMSE/maximum `1.114/4.871 kJ/mol/A`. The two sealed
artifacts reject both derivative candidates, so no SPL4/ACE runtime pairing or
APBS force support is added.

The `amber-pbsa/abcg2-pbsa-2023` configuration is parsed and pairing-locked but
execution deliberately stops at an evidence gate.  The paper reports optimized
GAFF2 atom-type radii (including new `on`, `oi`, `hn1`, `hn2`, `hn3` types) and
a refitted nonpolar model; those exact redistributable parameter artifacts have
not been obtained from an authoritative upstream package.  MAPLE will not
invent or approximate them under the certified profile name.

## Product capability boundary

| endpoint | SP energy | SP gradient | OPT/SCAN | current role |
|---|---:|---:|---:|---|
| OpenMM Amber GB + ACE/LCPO | yes | yes | yes | Route 1 executable baseline |
| prebuilt cluster + OpenMM GB + ACE/LCPO | yes | yes | yes | fixed-shell cluster potential; not absolute solvation free energy |
| APBS LPB + APOLAR | yes | no | no | PB energy-only provider |
| APBS molecular-surface LPB + OpenMM ACE | benchmark only | no | no | rejected: coarse grid unstable; converged fine grid still misses material-gain gate |
| APBS SPL4 LPB force / APBS APOLAR force | benchmark only | no | no | rejected: complete derivative and grid gates fail |
| external ddX/ddPCM + OpenMM ACE audit | benchmark only | benchmark-validated polar derivative | no | rejected: no accuracy gain and local correction is roughly 360x slower |
| AmberTools CHA-GB + cavity/dispersion | yes | no | no | explicit AM1-BCC SP-only accuracy provider; not runtime default |

OPT and relaxed SCAN always request the derivative of the same combined
potential used for the energy. Providers that expose only an energy fail
closed; MAPLE never optimizes with gas-only MLIP forces while reporting a
solvated energy.

The frozen multi-MLIP compatibility trace exercises this exact product
construction with both MACE-OFF23m and AIMNet2. On the same fixed AM1-BCC
geometry, both adapters pass additive energy/force closure, all-`3N`-component
finite differences of the combined potential, and a two-step combined BFGS
smoke. Three manually displaced geometries also return finite combined
energies, but they do not execute MAPLE's SCAN task path. That evidence
establishes adapter compatibility for the two named checkpoints, not universal
MLIP accuracy.

A separate task-matrix trace closes that manual-SCAN limitation by executing
MAPLE's actual `engine` and task dispatcher. MACE-OFF23m, AIMNet2, and ANI2x
each complete SP, a two-iteration MAPLE LBFGS OPT, a three-point rigid MAPLE
SCAN, and a deterministic four-step NVT trajectory with the same normalized
AM1-BCC/OBC-II/ACE input. All twelve jobs retain the structured solvation
decomposition and audit manifest. This is real product-path plumbing evidence,
not broad optimization/scan/MD stability, equilibrated sampling, or
chemical-accuracy certification.

Conformer-aware calculations are an optional evaluation layer over the same
additive potential, not a replacement for the product SP/OPT/SCAN/MD interface.
The reusable
`maple.function.free_energy.analyze_discrete_conformer_ensemble` function
combines same-state gas MLIP energies with fixed-charge solvent corrections
without selecting a particular MLIP or conformer generator. Its default
state-count, effective-count, dominant-weight, and gas/solution-overlap gates
fail closed on concentrated submitted state sets. A sealed replay reproduces
all legacy fields from 40 historical MACE-OFF23m records (2,596 states)
exactly, while only 14/20 AM1-BCC and 14/20 ABCG2 cases pass those weight
diagnostics. That proves the generic analysis boundary, not conformer
completeness, multi-MLIP accuracy, or a hydration-free-energy result; no public
`#solvfe` task is opened.

The next frozen run supplies real named multi-MLIP evidence. Checkpoint-derived
element domains select 19 common cases and 1,270 states for MACE-OFF23m,
AIMNet2, and ANI2x; the same fixed AM1-BCC/OBC-II/ACE correction is used for
all three. Median/p90 cross-model ranges of the final correction are
`0.03597/0.11969 kcal/mol`, but alachlor reaches `1.41020 kcal/mol`.
Population agreement is much weaker: weight gates pass only `13/19`, `4/19`,
and `12/19` records, and AIMNet2/ANI2x median gas/solution weight overlaps are
`0.2141/0.2013`. The prospective `1e-6 kcal/mol` GPU repeat gate also remains
failed rather than being relaxed after the run.

Separate development scoring changes fixed-geometry MAE `1.930 kcal/mol` to
`1.910`, `1.969`, and `1.993 kcal/mol` for AIMNet2, MACE-OFF23m, and ANI2x.
All paired MAE-gain bootstrap intervals include zero. Thus the generic
conformer interface and automatic element compatibility are demonstrated, but
no accuracy promotion or public free-energy claim is made.

The current energy-only Metropolis/TI probe can use an MLIP plus an
energy-only solvent endpoint without fabricating solvent forces, but its small
development result is negative and it is not enabled as a default workflow.
A separate force-consistent five-window OBC-II/ACE TI probe ran the same rigid,
limited, and flexible development cases with MACE-OFF23m, AIMNet2, and ANI2x.
All three MLIPs slightly worsen the fixed-geometry three-case MAE
(`+0.047` to `+0.083 kcal/mol`). The full reduced-potential matrices are
stored. MAPLE now exposes a reusable PyMBAR-backed analysis API through the
optional `implicit-free-energy` extra; it delegates equilibration detection,
decorrelation, MBAR, effective sample counts, and overlap to upstream PyMBAR
rather than implementing a local estimator. Post hoc label-free analysis of
all nine records finds connected adjacent overlap (`minimum=0.145`), agreement
with TI within `0.112 kcal/mol`, and repeat differences below
`0.239 kcal/mol`. However, only `9-19` decorrelated samples survive per state
against the diagnostic minimum of 20, and 30-fs windows cannot prove
equilibrium or conformer mixing. The result therefore remains fail-closed.
Separate development scoring gives MBAR MAE changes of `+0.056`, `+0.061`, and
`+0.069 kcal/mol` for AIMNet2, ANI2x, and MACE-OFF23m, respectively: the
MLIP-dependent ensemble term is real but does not repair OBC-II/ACE accuracy.

The remaining provider-accuracy experiment keeps those same gas MLIPs and
force-capable OBC-II/ACE solution ensembles, then applies a label-free
one-sided Zwanzig correction to the energy-only
AM1-BCC/CHA-GB/PBSA-cavity-dispersion endpoint. The gas-leg correction is
exactly zero: no MM gas energy, residual, retraining, or MLIP-specific solvent
term is introduced. Across 720 high-endpoint evaluations, the minimum
effective-weight fraction is `0.911` and the largest normalized weight is
`0.136`, but the 30-fs source trajectories leave only `4-13` decorrelated
samples per replicate. All `9/9` records therefore fail the frozen numerical
gate and retain the explicit no-equilibrium claim. Development-only three-case
MAEs improve to `1.725`, `1.646`, and `1.777 kcal/mol` for AIMNet2, ANI2x, and
MACE-OFF23m, but the gain is dominated by one molecule and cannot be promoted.
This is evidence that an energy-only provider post-correction is
computationally feasible, not evidence of target coverage, general accuracy,
or speed versus bare MM.

That three-case signal is superseded by a prospective common-state screen over
19 cases, 1,270 frozen states, and all three MLIPs. The label-free
[`route1-multi-mlip-chagb-discrete-conformer-2026-07-25.json`](benchmarks/route1-multi-mlip-chagb-discrete-conformer-2026-07-25.json)
contains 2,540 CHA-GB/PBSA evaluations, two exact repeats per state, completed
in `145.51 s` locally. Separate scoring in
[`route1-multi-mlip-chagb-discrete-conformer-score-2026-07-25.json`](benchmarks/route1-multi-mlip-chagb-discrete-conformer-score-2026-07-25.json)
compares the finite-state result with the same high endpoint at the reference
geometry. Fixed CHA-GB/PBSA MAE is `1.700 kcal/mol`; AIMNet2, ANI2x, and
MACE-OFF23m discrete MAEs are `1.699`, `1.720`, and `1.709 kcal/mol`.
All paired intervals versus fixed high-endpoint geometry cross zero, only
`6/19`, `7/19`, and `7/19` cases improve, and the weight diagnostics are not
universally valid. The frozen decision is
`long_sampling_candidate_not_supported`: on this construction the provider
change improves the low-endpoint result, whereas MLIP-dependent conformer
weighting does not demonstrate additional accuracy. A large long-MD campaign
is therefore not justified by this screen.

A second optional experiment uses GAFF2/OBC-II/ACE only as a cheap sampling
reference, evaluates the unchanged Route 1 MLIP/OBC-II/ACE target sparsely,
and combines gas and solution corrections through an exact indirect
thermodynamic cycle. It exercises the desired force-free acceleration
mechanism: 400 target energy calls and zero target force calls per model/case.
However, endpoint overlap is inadequate. Across the nine records, minimum
gas/solution BAR overlap is `0.015/0.057`, while the stricter minimum
directional MBAR overlap is only `0.005/0.016`. Every forward/reverse agreement
check fails, only `7/9` records are free of a final PyMBAR solver
nonconvergence, and reference-only estimates differ from bidirectional MBAR by
up to `2.514 kcal/mol`. Separate development MAEs worsen to `3.999`, `3.514`,
and `3.025 kcal/mol` for AIMNet2, ANI2x, and MACE-OFF23m. The MM energy never
enters the target formula; this is a negative reference-sampling diagnostic,
not a default or product free-energy path.

The corresponding force-based bridge was also tested with a prospectively
frozen linear MM/GB-to-MLIP/GB nonequilibrium protocol. The same gas-derived
constant shifts both phase corrections and cancels exactly; no MM or residual
energy enters the target. Across the same 3x3 matrix, 20-fs switching reduces
the maximum difference from the short direct-target MBAR control to
`0.492 kcal/mol`, but 5-fs-to-20-fs cycle drift reaches
`1.364 kcal/mol`, `0/9` complete numerical gates pass, and endpoint
equilibrium is not established. The overall development MAE is
`2.330 kcal/mol`, worse than the `2.150 kcal/mol` fixed-geometry baseline
although better than the failed `2.665 kcal/mol` endpoint correction. The
experiment also consumes `29,376` target energy+force evaluations, so it
establishes neither chemical improvement nor acceleration and remains an
optional analysis primitive.

A two-level OBC-II/ACE optimization followed by CHA-GB/PBSA final single
points was also tested with both potentials disclosed. It worsens all six
cases relative to the fixed-geometry high-level baseline, so it remains a
benchmark-only negative result.

## Result composition

For fixed charges:

```text
E_solution(R) = E_MLIP,gas(R) + G_polar(R,q_fixed) + G_nonpolar(R)
```

For `qeq-gto,mode=fixed`, QEq is evaluated once at the submitted reference
geometry and those charges remain fixed during SP, OPT, SCAN/PES, and admitted
NVE/NVT MD, matching the lifecycle used for AM1-BCC, ABCG2, and MOL2-provided
charges.

For `qeq-gto,mode=polarizable`, MAPLE uses the literature-defined consistent
QEq derivative at every geometry:

```text
q_vac  = argmin_q E_CQEq(R,q),                         sum(q)=Q
q_solv = argmin_q [E_CQEq(R,q) + G_GB,polar(R,q)],     sum(q)=Q
DeltaE = E_CQEq(R,q_solv) - E_CQEq(R,q_vac)
       + G_GB,polar(R,q_solv) + G_nonpolar(R)
```

The polarizable profile remains experimental: the mathematics and forces are
consistent, but the QEq and Amber GB parameters were not jointly fit.
