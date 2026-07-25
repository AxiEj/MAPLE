# Route A literature evidence matrix

Status: evidence freeze for the `research-qct-hybrid` claim in `.omx/plans/prd-route-a-qct.md`.

This file records what the literature **does** support for Route A,
what it **does not** support, and where MAPLE still needs internal
validation before making broader claims.

## Scope

Route A is the planned MAPLE workflow

```text
#solvfe(method=qct, ...)
```

for neutral closed-shell solutes in liquid water, combining:

1. periodic explicit-water discovery,
2. nonperiodic restrained `X(H2O)n` supermolecule ensembles,
3. sequential alchemical water association free energies,
4. one whole-cluster outer continuum correction, and
5. a multi-`n` QCT log-sum with explicit standard-state, symmetry, and water-reference terms.

The approved product claim remains intentionally narrow:

> a versioned, auditable **QCT-hybrid** hydration free-energy estimator,
> not an exact QCT implementation and not a scientifically validated production method yet.

## A. Supermolecule / cluster-continuum support and limitations

### A1. What the local authoritative review supports

Primary local source:

- Cramer and Truhlar, *Implicit Solvation Models: Equilibria, Structure, Spectra, and Dynamics*, Chem. Rev. 1999, 99, 2161-2200.
- Local file used for review: `/home/axie/MAPLE/theory/implicit solvent/Implicit Solvation Models Equilibria, Structure, Spectra, and Dynamics.pdf`
- Local checksum: `01edbfe6575df2f179e34f5358549bd82b3f8451c17613275f381a72cf4b7df1`
- DOI: <https://doi.org/10.1021/cr960149m>

The review explicitly supports the **supermolecule + continuum** idea:

- treat solute plus some explicit first-shell waters as one cluster;
- place that cluster in an outer implicit solvent model;
- use this as a practical compromise when directional H-bonding or specific first-shell structure matters.

The same review also gives the key limitation that drove the Route A design:

- the **number and orientation** of nonbulk waters are not unique;
- averaging over those states is the hard part;
- satisfactory averaging generally requires explicit-solvent sampling extending beyond only one hand-picked first shell;
- a calculation with only a few explicit waters can still be informative, but it is scientifically fragile if promoted to a bulk free-energy claim without additional ensemble treatment.
- adding explicit hydrogen-bonded waters does not automatically improve a
  continuum result: the review warns that the supermolecule method must itself
  reach roughly the accuracy of the continuum baseline (often a fraction of a
  kcal/mol) before the added microscopic detail can improve free energies.

### A2. Route A implication

This literature supports:

- the existence of a legitimate **supermolecule/cluster-continuum modeling family**;
- using explicit first-shell waters when local chemistry matters.

From that literature plus MAPLE's own implementation boundary, **MAPLE infers** that current Route 3 is a valid **static potential-energy composition**. That Route 3 statement is a MAPLE interpretation, not a direct literature claim.

This literature does **not** support:

- fixed-`n`, single-geometry microsolvation as a defensible absolute hydration free energy;
- choosing one water count or one orientation without an ensemble argument;
- calling a few-water cluster calculation a validated bulk-solvation method.

### A3. Consequence for MAPLE

This is why the PRD rejects Route A = “just Route 3 plus one implicit term”.
Route A must resolve fluctuating inner-shell composition and ensemble averaging,
not only evaluate one cluster geometry.

### A4. Outer continuum component: what SMD / IEFPCM / PCMSolver do and do not prove

Primary sources checked for this subsection:

- A. V. Marenich, C. J. Cramer, and D. G. Truhlar, *Universal Solvation Model Based on Solute Electron Density and on a Continuum Model of the Solvent Defined by the Bulk Dielectric Constant and Atomic Surface Tensions*, *J. Phys. Chem. B* **113**, 6378-6396 (2009). DOI: <https://doi.org/10.1021/jp810292n>
- PubMed record confirming the title, abstract, and DOI: <https://pubmed.ncbi.nlm.nih.gov/19366259/>
- PCMSolver official documentation: <https://pcmsolver.readthedocs.io/en/stable/>
- PCMSolver user input docs for the implemented solver choices (`IEFPCM` / `CPCM`): <https://pcmsolver.readthedocs.io/en/stable/users/input.html>
- PCMSolver code reference for the `IEFSolver`: <https://pcmsolver.readthedocs.io/en/stable/code-reference/solvers.html>
- PCMSolver official repository: <https://github.com/pcmsolver/pcmsolver>
- Di Remigio et al., *PCMSolver: An open-source library for solvation modeling*, *Int. J. Quantum Chem.* **119**, e25685 (2019). DOI: <https://doi.org/10.1002/qua.25685>

These sources support a narrower claim than “one SMD single-point equals hydration free energy”.
The original SMD paper defines the observable solvation free energy as two pieces: a
bulk electrostatic contribution from an IEF-PCM reaction-field treatment and a
cavity-dispersion-solvent-structure (CDS) contribution for short-range first-shell effects.
That supports using SMD as a **continuum free-energy model for a fixed solute or fixed
cluster geometry**. It does not, by itself, supply the occupancy averaging over fluctuating
`X(H2O)n` states that Route A needs.

The official PCMSolver documentation independently supports the electrostatic implementation
class that Route A plans to call: the library is an API for PCM electrostatics, documents
`IEFPCM` and `CPCM` solver choices, and exposes an `IEFSolver` implementation. That is
support for the **outer electrostatic component**, not for the full Route A thermodynamic
claim and not for SMD's CDS parametrization.

Therefore, a single `ΔG_outer(XW_n)` evaluation, or an SMD/PCM delta on one restrained
cluster snapshot, is only one term in the Route A ledger. Protocol v1 wrote
the unconditioned raw ledger as
`G_n = Σ_i G_assoc,i - nRT ln(rho_W_over_C0) + ΔG_outer(XW_n) - nΔG_outer(W)`.
The real pilot showed that this is not a promotable multi-`n`
hydration-free-energy formula: it omits the pure-water empty-volume packing
probability and treats an ordinary bare-solute Route 2 value as the `n=0`
state. Protocol v2 instead uses the conditioned cluster-QCT identity
`mu_X_ex = -RT ln p_0(lambda_s) - RT ln Σ_n exp[-beta A_n_tilde(lambda_s)]`,
with an empty-shell-conditioned `n=0` ensemble and one preregistered
whole-supermolecule outer-cavity construction rule.
Literature supports the **components** of that construction
(cluster-continuum treatment, outer continuum electrostatics, and QCT-style occupancy
bookkeeping), but we did **not** find a primary source that validates the exact
OMOL + PolarMACE + PCMSolver hybrid as an end-to-end scientific method.

## B. QCT, standard-state, and symmetry foundations

### B1. Primary foundations

Route A borrows the **organizational logic** of quasi-chemical theory, not the right to claim exactness by name.
Relevant primary references used in the PRD are:

- Pratt and Rempe, J. Chem. Phys. 2000. DOI: <https://doi.org/10.1063/1.1301528>
- Rogers and Beck, J. Chem. Phys. 2008. DOI: <https://doi.org/10.1063/1.2985613>
- Chaudhari, Pratt, and Rempe, J. Chem. Phys. 2017. DOI: <https://doi.org/10.1063/1.4986244>
- Asthagiri, Paulaitis, and Pratt, J. Phys. Chem. B 2021. DOI: <https://doi.org/10.1021/acs.jpcb.1c04182>
- Gomez et al., Acc. Chem. Res. 2022. DOI: <https://doi.org/10.1021/acs.accounts.2c00078>

Related cluster-continuum / standard-state references:

- Pliego and Riveros, J. Phys. Chem. A 2001. DOI: <https://doi.org/10.1021/jp004192w>
- Bryantsev, Diallo, and Goddard, J. Phys. Chem. B 2008. DOI: <https://doi.org/10.1021/jp802665d>
- Pliego and Riveros review, WIREs Comput. Mol. Sci. 2020. DOI: <https://doi.org/10.1002/wcms.1440>
- Lehmann, Jameel, and Kaupp, *Systematic Evaluation of a Cluster-Continuum-Model Workflow to Compute the Free Energies of Solvation of Ions in Different Solvents*, J. Phys. Chem. A 2026. DOI: <https://doi.org/10.1021/acs.jpca.6c00886>

### B2. What these references justify

They justify the need to track, explicitly and separately:

- inner-shell occupancy `n`, including `n = 0`;
- standard-state conversion terms;
- water-reservoir / water-density terms;
- symmetry / permutation counting terms that may be required, depending on the chosen ledger definition;
- outer-cluster and outer-water reference terms;
- shell-consistency and tail-convergence diagnostics.

In the QCT literature, the exact bookkeeping is framed in terms of gas-phase cluster
equilibrium / density factors together with an excess-chemical-potential difference of the
form `μ_XWn^ex - n μ_W^ex`. That supports the **structure** of a multi-`n` ledger with an
outer-water reference term. It does **not** make MAPLE's current multi-`n` MLIP/continuum
realization automatically exact.

That logic is the reason the authoritative Route A v2 thermodynamic contract is
frozen in `docs/solvation/route-a/thermodynamics-v2.md` instead of being
hidden inside one calculator. `thermodynamics.md` is retained as the
historical v1 contract only. The exact `+RT ln n` implementation used by
MAPLE remains a Route A ledger choice that must be justified by the chosen
partition-function convention; it is not presented here as a direct theorem
proved by every cited QCT paper.

### B2a. What the 2026 automated cluster-continuum ions paper adds

The 2026 JPCA ions study does **not** validate the exact Route A neutral-water stack, but it is directly relevant to workflow design. Primary article page: <https://pubs.acs.org/doi/full/10.1021/acs.jpca.6c00886>. It supports:

- partially automated construction of embedded microsolvated clusters;
- systematic scans and averaging over a **range of cluster sizes** rather than trusting one hand-picked cluster;
- explicit treatment of standard-state corrections and embedding-model dependence;
- a solvent-**cluster** reference cycle as a more realistic error-cancellation strategy than a monomer cycle for cluster-continuum workflows.

The same ACS article also draws an important thermodynamic distinction: the solvent-cluster cycle, referenced to a solvated solvent cluster `S_n`, is described as giving better continuum-error cancellation and more realistic results, whereas the monomer cycle is reported as not converging properly with increasing cluster size in cited prior studies. Current Route A instead uses sequential monomer association steps together with the QCT-style outer reference `-n ΔG_outer(W)`. That is a deliberate research hypothesis for MAPLE's neutral-solute workflow, not something this 2026 ions paper validates; it should eventually be tested by ablation against a solvent-cluster reference construction.

Its demonstrated domain is primarily **single-ion** solvation across different solvents with cluster-continuum machinery such as QCG growth, DFT reoptimization, higher-level energies, conformer selection, and implicit embedding such as COSMO-RS. That makes it a strong methodological precedent for automation, systematic size-dependence studies, and embedding-sensitivity audits, but not a direct validation of MAPLE's neutral-solute, MLIP-based, PCM-corrected, multi-`n` Route A stack.

Route A therefore hash-binds
`reference-cycle-ablation-contract-v1.json`. The solvent-cluster cycle is a
serious accuracy candidate, but the ion-focused literature is not enough to
select it post hoc on MAPLE's final neutral-solute holdout. Both cycles must be
run with the same molecules, `n` values and non-reference controls; selection,
if any, is made on development evidence before the holdout is opened.

### B2b. MLIP-accelerated QCT is feasible, but transfer is not automatic

Bonnet and Marzari, *Solvation Free Energies from Machine Learning Molecular
Dynamics*, J. Chem. Theory Comput. 2024. DOI:
<https://doi.org/10.1021/acs.jctc.4c00116>; open record:
<https://pubmed.ncbi.nlm.nih.gov/38771939/>.

This is the closest successful precedent found for the combination
`ML potential + explicit inner region + QCT organization + analytic outer
environment`.  The authors trained an equivariant ML potential on
first-principles energies and forces, ran about `200 ps` of ML molecular
dynamics, and reported convergence of their tested alkaline and alkaline-earth
ion solvation energies to `0.04 eV`.  It directly supports using an MLIP to make
the ensemble part of a QCT-like hybrid calculation long enough for statistical
testing.

It does **not** validate MAPLE's current OMOL model, neutral-organic domain,
sequential association path, or PCM outer correction: the paper used a
system-specific first-principles training workflow, an ionic benchmark, and a
different analytic outer treatment.  Route A must therefore demonstrate
model-domain accuracy and convergence independently rather than inheriting the
paper's result.

Wu and Kieffer, *New Hybrid Method for the Calculation of the Solvation Free
Energy of Small Molecules in Aqueous Solutions*, J. Chem. Theory Comput. 2019.
DOI: <https://doi.org/10.1021/acs.jctc.8b00615>.

Their cluster-continuum workflow supplies two additional design precedents:
determine the minimum explicit-water count by a convergence study, and compare
optimized clusters with thermally disordered sampled configurations.  Their
reported counterion sensitivity also reinforces that the method cannot be
transferred outside its charge/reference convention without a new
thermodynamic audit.  MAPLE adopts the convergence and ensemble lessons, not
their numerical claims.

### B2c. Restraint and outer-delta boundary: literature versus MAPLE inference

Additional primary references:

- Boresch et al., *Absolute Binding Free Energies: A Quantitative Approach for
  Their Calculation*, J. Phys. Chem. B 2003. DOI:
  <https://doi.org/10.1021/jp0217839>
- General, *A Note on the Standard State's Binding Free Energy*, J. Chem.
  Theory Comput. 2010. DOI: <https://doi.org/10.1021/ct100255z>

These sources support keeping the bound-state/restraint definition,
standard-state conversion and restraint free energy in one explicit
thermodynamic cycle. They do **not** state that MAPLE's particular
signed-distance restraint has a universal zero release.

`restraint-contract-v1.json` therefore labels zero release as a versioned MAPLE
choice: `R_n` itself is retained in both interacting endpoints, and V1 adds no
separate placement/orientation bias. If that changes, the release is no longer
zero and the contract must be versioned.

Likewise, the requirement that vacuum and outer endpoints share the exact
atom-list, restraint, measure and boundary-condition hashes is an engineering
invariant inferred from thermodynamic-cycle consistency, not a sentence copied
from a QCT paper. It is deliberately stronger than informal prose because it
prevents the outer continuum, PCM half-coupling or CDS term from being counted
twice.

### B3. What these references do **not** certify

They do **not** certify that the specific MAPLE stack

```text
OFF24 discovery -> OMOL scorer -> MACE-POLAR/PCM outer term -> multi-n log-sum
```

is already validated by the literature.

They support the **bookkeeping structure**.
They do not validate the chosen MLIP Hamiltonians or the continuum approximation.

## C. Successful explicit-solvent MLIP alchemical free-energy precedents

### C1. Architecture-independent NNP decoupling (JPC Lett. 2025)

Primary paper:

- Picha et al., *Architecture-Independent Absolute Solvation Free Energy Calculations with Neural Network Potentials*, J. Phys. Chem. Lett. 2025, 16, 12080-12086.
- DOI: <https://doi.org/10.1021/acs.jpclett.5c02980>
- Open text used here: PMC mirror <https://pmc.ncbi.nlm.nih.gov/articles/PMC12641469/>

What it establishes:

- alchemical decoupling can be implemented for fully NNP-described systems by manipulating the neighbor list rather than relying on a force-field-specific soft-core form;
- this is **architecture-independent** at the NNP level;
- explicit-solvent absolute solvation free-energy calculations with MACE-style models are feasible;
- the MACE ecosystem now also documents this path at the user-guide level through the official OpenMM interface docs, which point to `mace-md` and a public ASL `MACE-OFF23-SC` example model as a softcore-trained route for alchemical simulations: <https://mace-docs.readthedocs.io/en/latest/guide/openmm.html>, <https://github.com/jharrymoore/mace-md>, <https://github.com/jharrymoore/MACE-OFF23-SC>.

Verified quantitative evidence from the PMC text/tables:

- cycle-closure errors: `0.0 ± 0.4` and `-0.2 ± 0.2 kcal/mol` for the model cycles reported in Table 1;
- reported MACE-OFF23(S) hydration free energies vs experiment (kcal/mol):
  - water: `-6.5 ± 0.1` vs `-6.3`
  - methane: `+2.0 ± 0.0` vs `+2.3`
  - ethane: `+2.3 ± 0.1` vs `+1.8`
  - methanol: `-4.8 ± 0.1` vs `-5.1`
  - ethanol: `-4.3 ± 0.1` vs `-5.0`
  - toluene: `-1.3 ± 0.1` vs `-0.9`
  - phenol: `-6.0 ± 0.1` vs `-6.6`
- the same article shows that MACE-OFF23(M) can be materially worse than MACE-OFF23(S) on this task, so larger or newer is **not automatically** better.

Route A implication:

- this is strong evidence that **explicit-solvent MLIP alchemical FE is technically viable**;
- it supports using MLIPs in rigorous FE components of Route A;
- it provides a reproducible **OFF23-SC-style control route** that MAPLE can compare against;
- it does **not** validate Route A's cluster/continuum partition, because that paper is a full explicit-solvent alchemical protocol, not a supermolecule + continuum hybrid.

### C2. MACE-OFF24-SC hydration FE (JACS 2026)

Primary paper:

- *Computing Solvation Free Energies of Small Molecules with Experimental Accuracy*, J. Am. Chem. Soc. 2026.
- DOI: <https://doi.org/10.1021/jacs.5c10940>
- Open text used here: PMC mirror <https://pmc.ncbi.nlm.nih.gov/articles/PMC12903862/>

What it establishes:

- MACE-OFF24-SC is a working explicit-solvent MLIP alchemical hydration protocol in the paper's own reported benchmark;
- the authors selected a **36-compound neutral subset** from FreeSolv with broad functional-group coverage, explicitly filtered for compatibility with a model trained on neutral molecules only;
- the reported production protocol used **16 OpenMM HREX replicas** on that subset;
- the method can outperform common classical baselines on that benchmark.

Verified quantitative evidence from the PMC text/tables:

- summary hydration-free-energy errors (kcal/mol) on the reported subset:
  - `MACE-OFF24-SC`: `MAE 0.69`, `RMSE 0.80`
  - `GAFF`: `MAE 1.09`, `RMSE 1.24`
  - `OpenFF 2.1`: `MAE 0.98`, `RMSE 1.15`

Route A implication:

- a modern explicit-solvent MACE route can beat mainstream classical baselines on a real hydration benchmark;
- this is the strongest current precedent that an MLIP-centered MAPLE route can be worth pursuing scientifically;
- it still does **not** validate the exact Route A hybrid because the paper is not a first-shell-explicit + outer-continuum QCT stack;
- a public **OFF24 medium** artifact exists in the MACE ecosystem, but this review did **not** verify a public artifact matching the paper's exact **OFF24-SC** model; therefore the JACS route should still be treated as a literature precedent, not yet as a pinned reproducible upstream dependency for MAPLE.

### C3. MBAR overlap is not a substitute for time-series independence

Primary method and official implementation guidance:

- Shirts and Chodera, J. Chem. Phys. 2008. DOI:
  <https://doi.org/10.1063/1.2978177>
- PyMBAR 4.0.3 MBAR documentation:
  <https://pymbar.readthedocs.io/en/4.0.3/mbar.html>
- PyMBAR time-series documentation:
  <https://pymbar.readthedocs.io/en/4.0.0/timeseries.html>

The MBAR documentation explicitly assumes uncorrelated samples and directs
users to identify equilibration, estimate statistical inefficiency, and
subsample correlated trajectories before constructing the estimator. Its
`compute_effective_sample_number()` value measures reweighting support at a
thermodynamic state; it is not the number of time-independent configurations
in each sampled trajectory.

Route A must therefore keep two gates separate:

1. state-space overlap / MBAR reweighting effective sample number; and
2. per-window time-series equilibration plus decorrelated sample count.

Half-trajectory agreement and independent replicas remain additional
nonredundant checks. A small formal MBAR uncertainty from correlated frames
cannot certify convergence.

## D. Reproducible baselines: ReSolv and FreeSolv

### D1. ReSolv

Primary paper / code:

- Röcken, Burnet, and Zavadlav, *Predicting solvation free energies with an implicit solvent machine learning potential*, J. Chem. Phys. 2024.
- DOI: <https://doi.org/10.1063/5.0235189>
- arXiv: <https://arxiv.org/abs/2406.00183>
- code: <https://github.com/tummfm/ReSolv>

What it establishes:

- a solvent-conditioned implicit ML potential can be trained directly against hydration free-energy targets;
- on FreeSolv, the authors report a mean absolute error **close to average experimental uncertainty** and a speedup of **four orders of magnitude** relative to their explicit-solvent ML potential baseline;
- the upstream repository is public and the code path is reproducible in principle.

What it does **not** establish for Route A:

- ReSolv is not an inner-shell-explicit method;
- its solvent conditioning already folds solvent response into the model output;
- therefore ReSolv is a **baseline and control**, not an additive term inside Route A.

### D2. FreeSolv

Primary dataset / repository:

- MobleyLab FreeSolv repository: <https://github.com/MobleyLab/FreeSolv>
- database paper: Mobley and Guthrie, JCAMD 2014. DOI: <https://doi.org/10.1007/s10822-014-9747-x>

What it establishes:

- a curated, versioned hydration-free-energy benchmark with experimental values, uncertainties, structure files, and prior calculated baselines;
- a common evaluation target for Route 2, Route 3, ReSolv, and future Route A frozen holdouts.

Route A implication:

- FreeSolv is a benchmark substrate, not a proof of method correctness;
- any Route A accuracy claim must specify the exact split, experimental uncertainties, and whether each molecule is in-distribution or effectively training-adjacent for the selected model family.

## E. Licensing and upstream artifact boundary

The literature path and the deployable artifact path are different issues.

Verified upstream boundary used by the PRD:

- **MACE code** is MIT-licensed: <https://github.com/ACEsuit/mace>
- **MACE-OFF weights** are released under an Academic Software License for academic/noncommercial use: <https://github.com/ACEsuit/mace-off>
- the public **MACE-OFF23-SC** example model referenced by the official MACE OpenMM docs is also ASL: <https://github.com/jharrymoore/MACE-OFF23-SC>
- **MACE-POLAR foundation weights** are likewise distributed under ASL terms: <https://github.com/ACEsuit/mace-foundations/releases>
- OMOL-related released weights in the MACE ecosystem are treated here as **model artifacts with separate upstream terms**, not as MIT by default.

Route A consequence:

- MAPLE can integrate against upstream interfaces and locally available weights;
- MAPLE must not blur code license and model-weight license;
- any future release notes must keep that distinction explicit.

## F. Exact-stack evidence gap

### F1. What we found

We found strong evidence for **parts** of the Route A stack:

- supermolecule/cluster-continuum as a recognized modeling family;
- QCT-style occupancy bookkeeping and standard-state logic;
- explicit-solvent MLIP alchemical hydration FE as a viable modern baseline;
- reproducible external baselines and benchmark datasets.

### F2. What we did **not** find

We did **not** find a fully pinned public implementation that already matches the exact MAPLE Route A stack:

```text
periodic OFF24 discovery
-> nonperiodic restrained X(H2O)n ensembles
-> sequential water association FE
-> whole-cluster PCM/SMD outer correction
-> multi-n QCT log-sum
```

In particular, no public source found in this review simultaneously pins:

- the periodic first-shell discovery stage,
- the restrained nonperiodic cluster stage,
- the explicit symmetry/water-density ledger,
- the same outer continuum correction together with the matching `-n ΔG_outer(W)` water reference term,
- and a released end-to-end implementation with fixed hashes and benchmark scripts.

The closest reproducible public control found here is the **OFF23-SC / mace-md** explicit-solvent path documented by the MACE project itself. That is valuable as an explicit-solvent comparator, but it is still a different method class from Route A's supermolecule + continuum QCT-hybrid design.

So Route A is a **research assembly problem**, not a direct “wire together the known public recipe” problem.

## G. Evidence matrix

| Claim | Supported by | Support level | What it supports | What it does not support |
|---|---|---:|---|---|
| Supermolecule + continuum is a legitimate modeling pattern | Cramer-Truhlar review | High | Route 3-style cluster + outer continuum composition, as a MAPLE inference from that model family | absolute hydration FE from one hand-picked cluster |
| SMD / IEFPCM / PCMSolver support an outer continuum term | Marenich-Cramer-Truhlar 2009 + official PCMSolver docs/reference | High | a fixed-geometry cluster-level outer correction and the implementation class behind `ΔG_outer` | end-to-end validation of the OMOL + PolarMACE + PCMSolver Route A hybrid or an ensemble hydration FE from one single-point |
| Fixed-`n` microsolvation alone is insufficient | Cramer-Truhlar review | High | need for ensemble treatment and occupancy logic | promotion of single geometry to bulk FE |
| QCT-style occupancy ledger is physically motivated | Pratt/Rempe/Rogers/Asthagiri line | High | explicit `n`, symmetry/permutation counting when required, standard-state, water-reference bookkeeping | exactness of MAPLE's chosen MLIP + PCM approximation |
| Explicit-solvent MLIP alchemical FE can work | JPC Lett. 2025; JACS 2026 | High | MLIP-based FE components in Route A are plausible | first-shell-explicit + continuum hybrid correctness |
| MBAR requires equilibrated, effectively independent samples | Shirts-Chodera + official PyMBAR docs | High | separate time-series decorrelation and state-overlap gates | treating MBAR reweighting ESS or a small correlated-frame error bar as convergence |
| Modern MACE explicit FE can beat classical baselines | JACS 2026 | High | Route A is worth testing against Route 2/GAFF/OpenFF baselines | Route A already beats Route 2 |
| ReSolv is a valid high-quality external baseline | JCP 2024 + public repo | High | comparison/control route | additive inner-shell term inside Route A |
| FreeSolv is a valid benchmark substrate | FreeSolv repo + paper | High | frozen benchmark design | scientific validation by itself |
| Automated / semiautomated cluster construction and systematic cluster-size studies are viable cluster-continuum workflow ideas | JPCA 2026 ions workflow | High | automated microsolvated cluster construction, size-dependence scans/averaging, and standard-state plus embedding sensitivity audits | exact validation of the Route A neutral-water MLIP + PCM stack, Route A's exact plateau thresholds, or its monomer-reference choice |
| The exact Route A stack already exists as a pinned public implementation | no matching source found | None | none | cannot claim prior end-to-end external validation |

## H. Evidence hierarchy for MAPLE decisions

Use the following hierarchy when judging Route A progress:

1. **Primary literature + upstream artifact provenance**
   - establishes whether the scientific move is even defensible.
2. **Frozen thermodynamic contract**
   - `docs/solvation/route-a/thermodynamics-v2.md`
   - prevents hidden sign, symmetry, and standard-state drift.
3. **Engineering reproducibility**
   - hashes, restartability, deterministic artifacts, mesh reproducibility.
   - necessary, but not scientific validation.
4. **Internal scientific gates**
   - shell plateau, tail omission, overlap diagnostics, held-out subsets.
5. **External accuracy comparisons**
   - Route A vs Route 2 vs explicit reference subset vs experiment.

A green engineering test without levels 4-5 is **not** a validated solvation method.

## I. Current scientific status

Current status for Route A should be stated exactly as follows:

- **scientifically motivated** by primary literature;
- **architecturally justified** by the approved PRD;
- **partially implemented**, including the real OMOL alchemical sampler,
  warning-fatal whole-supermolecule PCM outer backend, and protocol-v3
  one-measure soft-membership, packing analysis, joint-covariance ledger and
  exact finite-enumeration closure primitives;
- **not yet scientifically validated** as a production hydration-free-energy method;
- **not yet demonstrated** to outperform Route 2 on a frozen, held-out benchmark;
- **not licensed or evidenced** as a simple repackaging of one public upstream route.

That is the right standard for the current phase.
