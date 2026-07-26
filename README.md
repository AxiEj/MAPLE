# **MA**chine-learning **P**otential for **L**andscape **E**xploration (**MAPLE**)

![MAPLE Concept](./maple.jpg)

MAPLE is a machine-learning-potential-native computational chemistry toolkit for
geometry optimization, transition-state search, reaction-path analysis, molecular
dynamics, and related post-processing workflows.

## Core Capabilities

| Category | Methods |
|----------|---------|
| **Optimization** | L-BFGS, RFO, SD, CG, SD-CG, GDIIS |
| **Transition State** | NEB, CI-NEB, P-RFO, Dimer, String/GSM, AutoNEB |
| **Reaction Path** | IRC with GS, LQA, HPC, EulerPC |
| **Dynamics** | NVE, NVT, NPT |
| **Analysis** | Frequency, PES Scan, Single Point |
| **ML Potentials** | ANI, AIMNet2, MACE, MACEPol, UMA |
| **Extras** | D4 dispersion, explicit solvent cluster builder, experimental supermolecule–continuum solvation (GB-polar SP or optional TBLite/ALPB energy+forces), UMA/FAIR-Chem-backed PBC, restart files, DCD output |

## Installation

### Requirements

- Python >= 3.10
- CUDA-capable GPU recommended for production workloads

MAPLE separates dependencies into three groups:

| Group | Installed by `pip install -e .` | Purpose |
|-------|----------------------------------|---------|
| Core | Yes | Base MAPLE runtime and general scientific I/O |
| Optional tools | Only when explicitly requested | Plotting and development tools |
| External ML runtimes | No | User-selected PyTorch/CUDA and FAIR-Chem stacks |

Core dependencies declared by MAPLE:

| Package | Minimum version | Notes |
|---------|-----------------|-------|
| `ase` | `>=3.22` | Atomic structures, calculators, I/O |
| `numpy` | `>=1.20` | Numerical arrays |
| `scipy` | `>=1.7` | Scientific routines |

External runtime dependencies that users install manually:

| Package | Version boundary | Required for | Why MAPLE does not auto-install it |
|---------|------------------|--------------|------------------------------------|
| `torch` | `>=2.0` | ANI, AIMNet2, MACE-OFF, MACE-O-MOL, MACE-Polar, UMA | PyTorch wheels must match the user's CUDA/CPU runtime and should be selected from the official PyTorch index. |
| `fairchem-core` | FAIR-Chem release with UMA support; tested locally with `2.19.0` | UMA and FAIR-Chem-backed/PBC workflows | FAIR-Chem may impose its own compatible PyTorch/runtime constraints, so install it after the matching PyTorch wheel. |
| `tblite` with its Python/ASE interface | Upstream release supporting `tblite.ase.TBLite` | Route 3 `method=alpb,provider=tblite` | It is optional and may require platform-specific compiled libraries; MAPLE fails with an install hint when it is absent. |

### Install MAPLE

```bash
git clone https://github.com/ClickFF/MAPLE.git
cd MAPLE
pip install -e .
```

### Install Dependencies

The `pip install -e .` command above installs only MAPLE's core dependencies.
Install PyTorch separately for your hardware. Examples:

```bash
# PyTorch example: CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118

# CPU-only PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Install FAIR-Chem only if you need UMA or FAIR-Chem-backed/PBC workflows:

```bash
pip install fairchem-core
```

Model checkpoint boundary:

- MAPLE auto-downloads only the model files hosted at https://huggingface.co/Wayne7815/MAPLE_models.
- Auto-downloads use a pinned HuggingFace revision by default; set `MAPLE_MODEL_REVISION` only when intentionally refreshing model assets.
- Backend-specific or local checkpoints, such as MACE-Polar `.pt` files, must be present in `maple/function/calculator/model/` or supplied through an explicit model path.
- UMA checkpoints are resolved through an explicit path, a local `maple/function/calculator/model/uma-*.pt` file, or FAIR-Chem's official model-loading path.

PBC boundary:

- PBC support is currently available through UMA/FAIR-Chem-backed workflows only.
- ANI, AIMNet2, MACE-OFF, MACE-O-MOL, and MACE-Polar are molecular no-PBC wrappers in MAPLE and fail fast when periodic atoms are supplied.
- AIMNet2 `coulomb_method=ewald` is disabled until validated cell/PBC/MIC inputs and reference tests exist; use `simple` or `dsf`.
- UMA stress/virial requests are rejected until MAPLE validates stress-unit conversion.

## Quick Start

### Command Line

```bash
maple input.inp
maple input.inp output.out
maple --version
maple md nve
```

### Minimal Example

```text
#model=uma(size=uma-s-1p1)
#opt(method=lbfgs)
#device=gpu0

0 1
C   -0.748   0.014   0.025
C    0.748  -0.014  -0.025
O    1.170   0.016   1.330
H   -1.155  -0.888  -0.460
H   -1.096   0.888  -0.530
H   -1.155   0.049   1.065
H    1.148  -0.912   0.457
H    1.096   0.869   0.513
H    0.802   0.842   1.742
```
### External coordinates:
```
#model=uma(size=uma-s-1p1)
#opt(method=lbfgs)
#device=gpu0

XYZ 0 1 /path/to/molecule.xyz
```
TIPS:  Charge and spin multiplicity are supported only in the **OMOL task** mode of the **UMA** model and in the **AIMNet2 / AIMNet2-NSE** models.
## Input Overview

### Header Keywords

```text
#model=<model>
#<task>(options)
#device=<device>
```

### Common Tasks

| Header | Description |
|--------|-------------|
| `#opt(method=lbfgs)` | Geometry optimization |
| `#sp` | Single-point energy |
| `#ts(method=neb)` | Transition-state search |
| `#freq` | Frequency analysis |
| `#irc(method=gs)` | Intrinsic reaction coordinate |
| `#scan(method=lbfgs)` | PES scan |
| `#md(mdp=nvt.mdp)` | Molecular dynamics |

### UMA Options

`#model=uma(...)` accepts the following keys (all optional):

| Key | Values | Default | Notes |
|-----|--------|---------|-------|
| `size` | `uma-s-1p1`, `uma-s-1p2`, `uma-m-1p1` | `uma-s-1p1` | Checkpoint variant |
| `task` | `omol`, `omat`, `oc20`, `odac`, `omc`, `oc22`, `oc25` | `omol` for non-periodic systems | Periodic UMA requires an explicit periodic task such as `omat`, `oc20`, `oc22`, `oc25`, `omc`, or `odac` |
| `inference` | `default`, `turbo` | `default` | `turbo` accelerates fixed-composition GPU workloads (NEB / TS / freq); ignored on CPU |

### Coordinates

Inline coordinates:

```text
#model=uma(size=uma-s-1p1,task=omol,inference=default)
#sp
#device=gpu0

0 1
C   0.000   0.000   0.000
H   1.089   0.000   0.000
...
```

External coordinates:

```text
#model=uma(size=uma-s-1p1,task=omol,inference=default)
#sp
#device=gpu0

XYZ 0 1 /path/to/molecule.xyz
```

Multi-structure jobs such as NEB accept multiple `XYZ` records.<br>
TIPS:  Charge and spin multiplicity are supported only in the **OMOL task** mode of the **UMA** model

MAPLE supports custom explicit-solvent PDB templates; see the
[solvent documentation](https://www.maplechem.org/functions/solvent.html)
for usage guidance.

### Experimental supermolecule–continuum solvation (Route 3)

Following Cramer and Truhlar's terminology, Route 3 is a **supermolecule
approach**: the complete `solute + first-shell solvent` supermolecule is
evaluated with the selected MAPLE MLIP and surrounded by an outer continuum.
This is a specific member of the broader cluster–continuum/discrete-SCRF
family:

```text
E_total = E_inner_MLIP(cluster) + DeltaE_outer(cluster)
```

Energy-only engineering probe using MAPLE's existing GB-polar/QEq correction:

```text
#solv(explicit=water,number=4,implicit=water,method=experimental-gb-polar,provider=maple-qeq-heuristic,experimental=true)
```

Force-capable optional route using
`GFN2-xTB/ALPB - GFN2-xTB(vacuum)` through TBLite:

```text
#solv(explicit=water,number=4,implicit=water,method=alpb,provider=tblite,experimental=true)
```

See [docs/route3-cluster-continuum.md](docs/route3-cluster-continuum.md) for
the capability boundary, prebuilt-cluster workflow, examples, and scientific
limitations.

### Research hydration-free-energy workflow (Route A)

`#solvfe(method=qct, ...)` is a separate research workflow for **ensemble**
hydration free energies. It does not reinterpret one Route 3 cluster energy
as a bulk free energy. The current implementation provides:

- versioned thermodynamic and state-space contracts;
- sequential OMOL alchemical insertion with rigid-water enhanced sampling;
- one complementary soft-cutoff membership for packing and fixed-`n`
  association, with hash-bound observation-volume and packing schedules;
- overlap, BAR/MBAR, effective-sample-size, and restart diagnostics; and
- a frozen Route 2/MACE-POLAR adapter that evaluates every sampled
  `solute + first-shell water` frame in one scaled whole-supermolecule PCM
  cavity, with zero-warning and zero-frame-deletion gates.

The workflow is **partially implemented and not scientifically validated**.
Protocol v3 removes v2's hard-packing/soft-association measure mismatch and
implements the soft-membership, association, packing-field, explicit-schedule,
two-estimator packing analysis, full `p̃(n)/x̃(n)/A_n` covariance ledger, and
finite exact-enumeration closure. The toy ledger deliberately fails if water
density or `n!` is counted twice. The pure-water Hamiltonian and real packing,
`n=0`, multi-occupancy and cluster replicas are not yet frozen or run.
Independent-replica convergence and the frozen blind-holdout comparison
against Route 2 remain promotion gates. A fixed-`n=1` development diagnostic
must not be reported as the final Route A hydration free energy.

See
[docs/solvation/route-a/thermodynamics-v3.md](docs/solvation/route-a/thermodynamics-v3.md)
and
[docs/solvation/route-a/literature-evidence.md](docs/solvation/route-a/literature-evidence.md)
for the scientific contract and evidence boundary.

## Documentation

- Website: https://www.maplechem.org/
- Release history: https://github.com/ClickFF/MAPLE/releases
- Architecture notes: [ARCHITECTURE.md](ARCHITECTURE.md)
- Authoring a calculator backend: [maple/function/calculator/AUTHORING.md](maple/function/calculator/AUTHORING.md)
- Route 3 supermolecule–continuum solvation: [docs/route3-cluster-continuum.md](docs/route3-cluster-continuum.md)
- Route A research hydration free energy: [docs/solvation/route-a/thermodynamics-v3.md](docs/solvation/route-a/thermodynamics-v3.md)

## Citation

```text
https://github.com/ClickFF/MAPLE
```

## Contributing

1. Fork the repository.
2. Create a feature branch.
3. Make changes with clear commits.
4. Open a pull request.

## Acknowledgments

- [ASE](https://wiki.fysik.dtu.dk/ase/)
- [PyTorch](https://pytorch.org/)
- [AIMNet2](https://github.com/isayevlab/AIMNet2)
- [FAIR-Chem](https://github.com/FAIR-Chem/fairchem)

**Version**: 0.1.4<br>
**Status**: Active Development<br>
**Updated**: May 2026<br>
