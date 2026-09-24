# PySCF SMD profile extension: MNSol energy audit

**Research-only fixed-geometry comparison; not a solvent registration, fitted
model, smooth-force qualification, or analytic-Hessian admission.** The
[machine-readable aggregate](aggregate.json) contains no MNSol record labels,
coordinates, or source-table rows. The licensed source archive and row-level
results remain outside Git.

## What changed

The earlier 653-record [MNSol benchmark](../full-freesolv-mnsol-smooth-bakeoff-20260924/README.md)
was restricted by MAPLE's 11-solvent Route-2 registry, **not** by the absence
of upstream solvent parameters. This follow-up uses frozen
[PySCF 2.13.1 SMD profiles](https://pyscf.org/_modules/pyscf/solvent/smd.html)
directly in an isolated research runner. It does not mutate the production
registry or silently treat a profile match as chemical-identity proof.

MNSol's descriptor columns and PySCF's eight-component profile have different
order/content: MNSol contains `eps,n,alpha,beta,gamma,phi²,psi²,beta²`,
whereas PySCF contains `n,n25,alpha,beta,gamma,eps,phi,psi`. The values were
matched by named descriptors and reported-precision tolerances, never copied
positionally. See the [MNSol v2012 manual](https://comp.chem.umn.edu/mnsol/MNSol-v2012_Manual.pdf)
and [PySCF source](https://pyscf.org/_modules/pyscf/solvent/smd.html).

## Coverage and results

All **3,037** MNSol records were audited. The prior **653** registered,
neutral, absolute records and **1,843** additional records with uniquely
matched PySCF profiles give **2,496 descriptor-matched, geometry-eligible**
records. An additional **27 decalin records** were calculated separately with
PySCF's `decalin (cis/trans mixture)` profile, whose refractive index
differs from the MNSol decalin entry. The remaining 514 rows are 363
nonneutral, 144 transfer free energies, and seven outside the frozen geometry,
element, or mass domain. The new run completed **1,870/1,870** records with
zero calculation failures.

| Neutral absolute MNSol panel | ddPCM + legacy DAREAL MAE | ISWIG + MOIST MAE | Paired change |
|---|---:|---:|---:|
| New matched PySCF profiles, 1,843 records | 0.8431 | 0.8288 | −0.0143 |
| Prior registered panel, 653 records | 1.2745 | 1.2452 | −0.0293 |
| Combined descriptor-matched panel, 2,496 records | 0.9559 | 0.9377 | −0.0182 |

All energies are kcal/mol fixed-gas-geometry `E_polar + E_CDS` proxies against
experimental solvation free energies, not full solution thermodynamics. The
baseline retains frozen MACE-POLAR-1-M point-`l≤1` source, pyddx 0.8 ddPCM
`lmax=15/nleb=1202/eta=0.1`, and PySCF 2.13.1 legacy SMD-CDS. The candidate
uses the **same frozen source and SMD tensions** but PySCF ISWIG Gaussian
C-PCM and MOIST SvdW-DROP areas, each with 302 points per atom. This changes
the electrostatic/area model; it is not an implementation-equivalent rewrite
of ddPCM/DAREAL. No parameter was fitted or adjusted after seeing labels.

**Pooled improvement is not a non-degradation guarantee.** In the new
1,843-record panel, 922 records and 31/80 solvent groups worsen. The paired
MAE difference's 10,000-resample, 317-geometry-cluster bootstrap interval is
`[−0.0487, +0.0188]` kcal/mol (candidate minus baseline), which includes
worsening. The 2,496-record combined descriptive interval is
`[−0.0489, +0.0103]` kcal/mol. The decalin profile-mismatch stratum is **not**
pooled with descriptor-matched records; its separate 27-record MAE is
`0.7171→0.5300` kcal/mol under the PySCF mixture profile.

## Verification and open gates

- All 80 off-panel matched solvents passed an independent MNSol-vs-PySCF
  named-descriptor audit; decalin was the single declared mismatch.
- All 11 registered profiles matched PySCF descriptors exactly. Ten independent
  old-panel energy controls retained identical MACE source hashes; PySCF CDS
  matched exactly after using the registered Hartree conversion, and maximum
  ddPCM energy difference was `5.74×10⁻¹⁰ eV`.
- Ten off-panel PySCF-CDS vs translated stock-tension/DAREAL checks differed
  by at most `2.95×10⁻⁷ kcal/mol`. A fresh complete-summary validation
  reproduced its SHA256 in `aggregate.json`.

This study does **not** establish whole-total-PES smooth forces, an analytic
total Hessian, per-solvent non-inferiority, or the required five-dataset
3-fit/2-sealed-test gate. Final Astra max approval and runtime capability
admission remain open. No restricted MNSol row-level evidence is published.
