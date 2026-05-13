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
| **Extras** | D4 dispersion, GBSA solvation, PBC, restart files, DCD output |

## Installation

### Requirements

- Python >= 3.9
- PyTorch >= 2.0
- CUDA-capable GPU recommended for production workloads

### Install MAPLE

```bash
git clone https://github.com/ClickFF/MAPLE.git
cd MAPLE
pip install -e .
```

### Install Dependencies

```bash
# Core scientific stack
pip install numpy scipy matplotlib ase

# PyTorch example: CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118

# CPU-only PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cpu

# UMA / FAIR-Chem models
pip install fairchem-core

# Optional periodic-boundary backends
pip install -e ".[pbc-aimnet]"   # AIMNet2 PBC, stress, DSF/Ewald/PME
pip install -e ".[pbc-mace]"     # official MACE foundation PBC + stress
pip install -e ".[pbc]"          # both PBC backend families
```

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
#model=uma(size=uma-s-1p2)
#opt(method=lbfgs)
#device=gpu0

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
| `#md(ensemble=nvt,mdp=nvt.mdp)` | Molecular dynamics |

### Periodic Boundary Conditions

Use `#pbc(...)` to define a cell. MAPLE accepts 2, 3, or 6 values:

```text
#pbc(a,b)                         # 2D slab: c defaults to 1000 Å
#pbc(a,b,c)                       # orthorhombic cell
#pbc(a,b,c,alpha,beta,gamma)      # full cell parameters
```

PBC backend selection is explicit. The non-PBC model names keep their original
local `.pt` behavior and are not automatically switched when `#pbc` is present.

| Backend family | Model names | Extra | Notes |
|----------------|-------------|-------|-------|
| UMA | `uma(task=omat)` or `uma` | `fairchem-core` | Main built-in PBC/NPT route; `task=omol` rejects PBC. |
| AIMNet2 official PBC | `aimnet2-pbc`, `aimnet2nse-pbc` | `pbc-aimnet` | Supports PBC/stress and `coulomb=dsf|ewald|pme`. |
| MACE official PBC | `mace-mp-pbc`, `mace-omat-pbc`, `mace-matpes-pbc`, `mace-mh-pbc` | `pbc-mace` | Supports PBC/stress through official MACE foundation calculators. |

PBC backends are intended for MD/SP workflows. They currently fail fast for
frequency/Hessian requests instead of silently falling back to a different
backend. Stress is kept in ASE-native `eV/Å³` for the NPT pressure path.

Example NPT header:

```text
#model=mace-omat-pbc(default_dtype=float32)
#md(ensemble=npt,steps=1000,timestep=0.5,temperature=300,pressure=1.0)
#device=gpu0
#pbc(10.0,10.0,10.0)
```

### Coordinates

Inline coordinates:

```text
#model=uma
#sp

C   0.000   0.000   0.000
H   1.089   0.000   0.000
...
```

External coordinates:

```text
XYZ /path/to/molecule.xyz
```

Multi-structure jobs such as NEB accept multiple `XYZ` records.

## Documentation

- Website: https://www.maplechem.org/
- Release history: https://github.com/ClickFF/MAPLE/releases
- Architecture notes: [ARCHITECTURE.md](ARCHITECTURE.md)

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

**Version**: 0.1.2  
**Status**: Active Development  
**Updated**: April 2026
