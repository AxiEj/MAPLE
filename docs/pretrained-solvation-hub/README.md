# Pretrained solvation-aware model hub

This directory records the scientific and deployment contracts for Route 4.
Route 4 is a registry of pretrained potentials, solvent backends, and
free-energy protocol adapters.  It is not a claim that an MLIP alone computes
a binding free energy.

## Free-energy boundary

A binding free-energy calculation requires a potential, solvent environment,
thermodynamic path, sampling, restraints/standard-state corrections, and an
estimator.  MAPLE therefore distinguishes:

1. ordinary potential-energy surfaces;
2. native solution-phase potentials;
3. additive solvent PMFs;
4. alchemical free-energy protocol backends; and
5. scalar property predictors.

Only the first four may have executable backends, and each task is gated by a
machine-readable capability card.  Scalar predictors are benchmark baselines,
not ASE calculators.

ASE `results["free_energy"]` remains an energy-like calculator result.  It is
not thermochemical Gibbs energy, absolute solvation free energy, or binding
free energy.  Thermodynamic results must be emitted by a dedicated protocol
with sampling, estimator, standard-state, and uncertainty metadata.

## Incremental architecture

The existing `register_calculator`/`SetCalculator`/`CalcABC` path remains the
potential registry and composition boundary.  Route 4 adds:

- explicit model capabilities and immutable provenance cards;
- a formal solvent-backend protocol over the existing GB/PB/SMD dispatcher;
- fail-closed model/task/solvent combination validation;
- a lightweight topology provider; and
- dedicated protocol adapters for models such as LSNN.

Route 4 does not duplicate the Route 3 `#solvfe` sampling engine and does not add a public `#bindfe` task.

## Initial engineering integration lanes

This milestone implements architecture, pinned provenance, adapters, and
fail-closed unit contracts.  It does **not** complete the four scientific
verticals: no real-checkpoint NVE certification, FreeSolv/MNSol accuracy run,
SAMPL host--guest calculation, or protein--ligand free-energy calculation is
claimed here.

| Lane | Backend | Engineering status |
| --- | --- | --- |
| Conservative PES | MACE-OFF24(M), AceFF 2.0 | Adapter SP/force/numerical-Hessian contracts are unit-tested; live-checkpoint MD/NVE certification remains pending |
| Native solution PES | AIMNet2-CPCMS v2 | Neutral-singlet mechanics probe only; MD is disabled until real-checkpoint stability evidence and solvent/license/reference metadata close |
| Additive/reference solvent PMF | OpenFF + GNNIS | Sealed reference-Hamiltonian contract; live upstream runtime/MD validation remains pending and absolute solvation free energy is forbidden |
| Alchemical solvation | LSNN-v1 | Scaffold-only water TI/MBAR protocol contract; default execution remains disabled until the model domain and runtime are audited |

GNNIS is an additive solvent PMF physically, but the executable
`gnnis-reference` adapter seals it to the upstream OpenFF-2.0.0 vacuum
Hamiltonian.  Its capability card therefore treats the resulting composition
as a native/reference potential so a second solvent term cannot be added.

MACE-POLAR-1 already uses the upstream `mace_polar()` loader and is audited,
not reimplemented.  The public MACE-OFF23-SC checkpoint is pinned in this
branch as an explicit-solvent alchemical control, but its optional
OpenMM-ML/OpenMMTools replica-exchange bridge remains pending.  The published
MACE-OFF24-SC result is not assigned to that public OFF23-SC checkpoint.

## Benchmark rules

- FreeSolv and MNSol are independent panels and are never ranked against one
  another.
- A paired comparison may differ in potential/model identity, but requires
  identical records, geometry/conformer policy, solvent protocol/backend,
  cavity/PMF, sampling, estimator, standard-state, and experimental provenance.
- Training overlap defaults to `overlap_unknown`.  Absence of evidence is not a
  strict holdout.
- The frozen MNSol split is referenced by
  `mnsol-partition-reference.json`.  MNSol row-level data and labels are not
  redistributed.
- Exact public checkpoint identities, sizes, hashes, unit contracts, and
  license unknowns are frozen in `upstream-artifacts.json`.
- Numerical target bands such as 1.0 or 1.5 kcal/mol are internal research
  targets, not universal literature-derived integration gates.

## Upstream sources

- AIMNet2-CPCMS artifact context: <https://github.com/isayevlab/LoQI>
- GNNIS: <https://github.com/rinikerlab/GNNImplicitSolvent>
- LSNN-v1: <https://github.com/Popov-Lab-UNC/LSNN-v1>
- MACE-OFF: <https://github.com/ACEsuit/mace-off>
- AceFF 2.0: <https://huggingface.co/Acellera/AceFF-2.0>
- OpenMM-ML sampling bridge: <https://github.com/openmm/openmm-ml>
