# MAPLE implicit solvation: Route 2 research/innovation route

This branch contains only the MACE-POLAR + SMD continuum route. It does not
contain the fixed-charge PB/GB implementation. The default
PCMSolver--IEFPCM/GePol profile remains an energy proof-of-concept. A separate,
explicit pyddx ddPCM profile now exposes a single-point research force
candidate. Neither profile is yet a complete MAPLE solution-phase PES. All
calculations therefore require `experimental=true`.

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
the gas-phase density. The omitted cavity default is
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
float64 projection dtype. No learned weight is changed.

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
model's existing GTO field-response path, supplies a different reaction
potential/gradient at each atom, and changes no learned weight.  The official
weights remain subject to the upstream Academic Software License; MAPLE does
not bundle or mirror them.

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

- one Tripos MOL2 conformer; single-point evaluation only;
- neutral closed-shell molecules (`0 1`), no salts, zwitterions, radicals, or
  periodic cells;
- H/C/N/O/F/P/S/Cl/Br/I and molecular mass from 16 through 500 Da;
- water only; the explicit pyddx profile may return a single-point force, but
  optimization, Hessian, scan, TS search, and MD remain disabled;
- exact official MACE-POLAR-1-M only; `#charge`, D4, other models, and
  unlisted continuum providers are rejected.

The derivative capability boundary is likewise explicit:

| Continuum path | Energy | Complete same-energy coordinate VJP | Public |
| --- | --- | --- | --- |
| PCMSolver--GePol | yes | no; fails closed | energy only |
| pyddx ddPCM `l15/n1202` + PySCF SMD CDS | yes | yes | explicit single-point research force candidate |
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
tracked `maple/` runtime-source change among those heads. The four-geometry
flexible panel and the local flexible-coordinate/two-torsion force evidence
were generated at earlier commits and remain explicitly historical until
rerun. Broader flexible/relaxed-path continuity, additional chemical classes,
and complete conformer thermochemistry remain open without changing the named
profile; the short NVE conservation gate is still unrun. Until those gates
pass, optimization, scans, transition states, and MD remain out of scope.

See [FORMULAS_AND_REFERENCES.md](FORMULAS_AND_REFERENCES.md) for equations and
the literature ledger, and [VALIDATION_STATUS.md](VALIDATION_STATUS.md) for the
passing engineering checks and still-open scientific gates. The concrete
derivation and implementation sequence is in
[ROUTE2_FORCE_ROADMAP.md](ROUTE2_FORCE_ROADMAP.md).
