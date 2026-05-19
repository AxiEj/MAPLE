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

# ML potentials
pip install fairchem-core
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

### UMA Options

`#model=uma(...)` accepts the following keys (all optional):

| Key | Values | Default | Notes |
|-----|--------|---------|-------|
| `size` | `uma-s-1p1`, `uma-s-1p2`, `uma-m-1p1` | `uma-s-1p1` | Checkpoint variant |
| `task` | `omol`, `omat`, `oc20`, `odac`, `omc`, `oc22`, `oc25` | inferred from PBC | `omol` for molecules, `omat` for periodic |
| `inference` | `default`, `turbo` | `default` | `turbo` accelerates fixed-composition GPU workloads (NEB / TS / freq); ignored on CPU |

### PBC Backends

For periodic workloads (PBC OPT / SCAN / TS / IRC / MD), choose an explicit
backend rather than relying on a legacy local wrapper:

| Model name | Range | Long-range Coulomb | Optional extra | Reference |
|------------|-------|--------------------|----------------|-----------|
| `aimnet2-pbc`        | short + long | yes (Ewald / DSF) | `pbc-aimnet` | Anstine et al., ChemSci 2025, DOI 10.1039/D4SC08572H |
| `aimnet2nse-pbc`     | short + long | yes (Ewald / DSF) | `pbc-aimnet` | (no-self-energy variant of the same backend)            |
| `mace-mp-pbc-small`  | short-range  | no                | `pbc-mace`   | MACE-MP-0, Batatia et al., arXiv:2401.00096 (2023)      |
| `mace-mp-pbc-medium` | short-range  | no                | `pbc-mace`   | MACE-MP-0, Batatia et al., arXiv:2401.00096 (2023)      |
| `mace-mp-pbc-large`  | short-range  | no                | `pbc-mace`   | MACE-MP-0, Batatia et al., arXiv:2401.00096 (2023)      |
| `macepol-pbc-small`  | short + long | yes (Ewald + real-space erf) | `pbc-macepol` | MACE-POLAR-1, Baldwin et al., arXiv:2602.19411 (2026) |
| `macepol-pbc-medium` | short + long | yes (Ewald + real-space erf) | `pbc-macepol` | MACE-POLAR-1, Baldwin et al., arXiv:2602.19411 (2026) |
| `macepol-pbc-large`  | short + long | yes (Ewald + real-space erf) | `pbc-macepol` | MACE-POLAR-1, Baldwin et al., arXiv:2602.19411 (2026) |

Install the optional extras as needed:

```bash
pip install -e .[pbc-aimnet]   # pulls aimnet[ase]
pip install -e .[pbc-mace]     # pulls mace-torch>=0.3.14,<0.4
pip install -e .[pbc-macepol]  # pulls mace-torch>=0.3.16 plus graph_longrange from git
```

**Physics note.** Use `aimnet2-pbc` / `aimnet2nse-pbc` for polar or charged
periodic systems where the long-range Coulomb sum matters (electrolytes,
zeolites with explicit charges, polar surfaces). Use `macepol-pbc-*` as an
additional periodic polar-molecular option with charge/spin support via
`atoms.info["charge"]` and `atoms.info["mult"]`. Use `mace-mp-pbc-*` for
solid-state foundation-model coverage where the short-range materials
potential is sufficient (metals, semiconductors, neutral oxides).

**Design note.** Backend choice is encoded in the model name, not in a flag.
`#model=aimnet2` always means the legacy local wrapper; `#model=aimnet2-pbc`
always means the official ASE-backed PBC adapter. MAPLE does not implicitly
switch between the two based on whether `#pbc(...)` is present, so absolute
energies and reproducibility provenance stay tied to the input.
`#model=macepols/m/l` stays the legacy local-wrapper non-PBC path;
`#model=macepol-pbc-*` selects the official ASE-backed PolarMACE PBC adapter.

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
