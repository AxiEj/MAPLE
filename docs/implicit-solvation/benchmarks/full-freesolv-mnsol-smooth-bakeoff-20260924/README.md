# Full FreeSolv/MNSol smooth-cavity energy diagnostic

**Research only; not an admitted model.** [Machine-readable aggregate](aggregate.json).
The original execution was on clean commit
`c08ede0c7fb520c1dd088a5c3cc4e6f4ba6ae490`; this documentation commit
publishes only aggregate statistics and cryptographic commitments to private
row-level evidence. No model, checkpoint, radius, dielectric, SMD tension,
MNSol archive, or production calculator was changed or copied here.

## Scope and method

- [FreeSolv v0.52](https://github.com/MobleyLab/FreeSolv), source commit
  `6c7d19b4b565537365ffd22006aa2cd4643200c6`: all **642/642** records
  attempted, checked against the pinned database and GAFF MOL2 archive.
- [MNSol v2012](https://comp.chem.umn.edu/mnsol/): all **3037/3037** original
  records audited; **653/653** neutral absolute solvation records eligible
  under the frozen 10-solvent/element/geometry domain were calculated. The
  mutually exclusive, priority-ordered audit is 653 eligible, 1874 outside
  the solvent panel, 363 nonneutral, 144 transfer records, and three other
  geometry/element/mass exclusions. **3037 audited does not mean 3037
  predicted.**
- Old baseline: unmodified MACE-POLAR-1-M zero-field point-`l<=1` source,
  pyddx 0.8 ddPCM `lmax=15/nleb=1202/eta=0.1`, stock PySCF 2.13.1
  SMD-CDS/DAREAL. This is the scalable registered v2 *energy* scalar, not a
  claim that Torch v3 can deliver analytic Hessians for 6–46 atoms.
- Candidate: the same MACE source and original SMD parameters with PySCF
  SWIG/ISWIG Gaussian C-PCM polarization and/or MOIST SvdW-DROP per-atom
  CDS areas, each using 302 Lebedev points per atom. This **changes the
  polar/area model**; differences are not same-scalar implementation errors.
  No fitting or post-result parameter selection occurred.

The observable is one fixed-geometry `E_polar + E_CDS` value per record,
compared diagnostically with experimental solvation free energy; it does not
include conformer/tautomer ensembles, solution optimization, or a complete
thermal free-energy treatment.

## Aggregate results (MAE, kcal/mol)

| Method | FreeSolv 642 | MNSol eligible 653 |
|---|---:|---:|
| ddPCM + DAREAL baseline | 1.5028 | 1.2745 |
| SWIG + original DAREAL | 1.4950 | 1.2778 |
| ISWIG + original DAREAL | 1.4921 | 1.2758 |
| ddPCM + MOIST-CDS | 1.4909 | 1.2601 |
| SWIG + MOIST-CDS | **1.4852** | **1.2472** |
| ISWIG + MOIST-CDS | **1.4821** | **1.2452** |

Pooled MAE decreases for the combined candidate, but this is **not a
no-degradation guarantee**. With SWIG+MOIST, 236/642 FreeSolv and 264/653
MNSol records worsen in absolute error; their worst absolute errors rise
`8.6384→9.5620` and `7.2435→7.7834` kcal/mol, respectively. In MNSol,
hexane MAE rises `0.5330→0.6298`; ethanol, dichloromethane and DMSO also
worsen. The unweighted 10-solvent macro-MAE improvement is only about
`0.0020` kcal/mol. A post-result diagnostic attributes the largest shared
chemical-family regressions predominantly to changed CDS area assignment,
which persisted in a 590-point spot check; this was **not** used to tune a
candidate.

A previously locked 20-molecule FreeSolv sample showed a tiny opposite MAE
change (+0.0084), demonstrating why this full-size result must not be
extrapolated from that pilot. FreeSolv and MNSol contain related chemistry;
their mean improvements are not two independent blind confirmations.

## Integrity and limits

The original FreeSolv-642 attempt retained one Torch DAREAL **derivative**
topology-guard failure and did **not** report successful-subset MAE. A
separately versioned, frozen energy-only replay recovered that one row using
stock SMD atomic tensions without requesting the Torch DAREAL derivative.
All 20 previously locked controls agreed **exactly** across all six energy
methods; independent reconstruction then verified complete 642/642 paired
results. MNSol's separate energy-only tension path similarly passed eight
clean stock-formula controls and completed 653/653. All three long processes
exited zero. Existing targeted tests: **61 passed**; private runners passed
Black, Pyflakes and bytecode compilation. The private row/output commitments
in `aggregate.json` bind the local evidence without redistributing MNSol
row-level data.

The original MNSol 148-row confirmation partition and all FreeSolv labels
were opened by this full-test request; neither dataset remains an untouched
sealed holdout. The required **five-dataset, 3-fit/2-sealed-test** accuracy
gate, global smooth total-PES force/analytic-Hessian qualification, and final
Astra max approval remain **open**. No new runtime capability is admitted.
