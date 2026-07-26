# Route A bulk-water Hamiltonian validation

Status: **engineering preflight passed; Hamiltonian not frozen**.

This document is deliberately outside the hash-bound Route A v3 protocol.
Its job is to collect evidence needed to choose and freeze the periodic
pure-water Hamiltonian. It does not change the v3 thermodynamic cycle and it
does not authorize a public `#solvfe` result.

## Why this is a separate gate

Route A uses two distinct MLIP roles:

1. a periodic pure-water Hamiltonian for packing and reference occupancies;
2. a nonperiodic solute-water Hamiltonian for fixed-occupancy association.

Loading both models successfully is not enough. The packing Hamiltonian must
also produce stable liquid-water dynamics, a credible density and credible
intermolecular structure under the exact periodic ensemble used by the
packing calculation.

MACE-OFF24(M) is the current candidate because its upstream model targets
organic molecules and molecular liquids, includes water clusters in its
SPICE-v2 training data, and exposes periodic energy, forces and virial stress.
The model remains a short-range MLIP without an explicit long-range
electrostatic term. Periodic liquid water is therefore a validation target,
not something MAPLE may infer from successful cluster tests. See the
[primary MACE-OFF24 paper](https://pubs.acs.org/doi/10.1021/jacs.4c07099),
the [official model repository](https://github.com/ACEsuit/mace-off), and the
[official MACE calculator documentation](https://mace-docs.readthedocs.io/en/latest/guide/ase.html).

## Immutable inputs

The initial 64-water box is read from the MACE authors' `mace-md` example
repository, but it is not vendored into MAPLE.

| Field | Value |
|---|---|
| source commit | `e19729524fc91920169d4e193e4edd55bc4c5707` |
| source path | `examples/example_data/waterbox.xyz` |
| source SHA256 | `a052257f5f9c068884ec7527d6dd41d05a7c3705b729e7d9703543890931ec61` |
| topology | 64 contiguous O-H-H molecules, 192 atoms |
| cubic cell | 12.442877769470215 Å |
| initial density | 0.9938042121 g mL⁻¹ |

The runner requires the exact byte hash and validates full three-dimensional
PBC, molecule order, O-H distances, H-O-H angles, atom count, volume and the
maximum safe minimum-image RDF radius before loading the MLIP.

The current checkpoint is:

| Field | Value |
|---|---|
| provider | `MACEOff24Provider` |
| model | MACE-OFF24(M) |
| checkpoint SHA256 | `e5ccf5837f685899811a68754e7c994393bfd1a81720393b03c643b46c70bc69` |
| `mace-torch` | 0.3.16 |
| arithmetic | float64 |

## Runner contract

`examples/solvation/route_a/validate_bulk_water.py` runs three distinct NVT
stages:

1. **thermalization** — strong Langevin coupling removes the structural
   relaxation heat of an external starting box;
2. **equilibration** — weaker coupling tests whether the target temperature
   remains stable;
3. **production** — weak coupling generates the frames used for diagnostics.

Every sampled state records temperature, potential/kinetic/total energy,
force RMS and maximum, pressure including the kinetic contribution, volume
and density. Production frames additionally retain coordinates, cells and
velocities.

Independently of the sampling interval, every MD step is checked for finite
energy/forces/velocities, the configured temperature and force emergency
limits, and preservation of the original O-H-H molecular identity using
minimum-image O-H distances and H-O-H angles. A violation terminates with a
nonzero exit status; a completed artifact records the step count and observed
geometry extrema. Constraints and non-default atomic masses are rejected.

The analysis computes intermolecular O-O, O-H and H-H RDFs with triclinic
minimum-image distances. Pair normalization uses the exact eligible
intermolecular pair count and instantaneous cell volume. Contiguous block RDFs
provide a diagnostic standard error; this is not treated as an independent-
sample uncertainty when the trajectory is too short.

The result directory contains:

- `arrays.npz`: observations, production frames and RDF arrays;
- `summary.json`: inputs, diagnostics, gates and semantic array hashes;
- `manifest.json`: result hash, raw `arrays.npz` byte hash, canonical
  `summary.json` hash and semantic hashes for every array.

An existing output path is never overwritten. A fixed RNG seed constrains the
stochastic path, but bitwise reproducibility across different CUDA and library
stacks is explicitly not claimed.

## Final v2 staged preflight

Command shape:

```bash
python examples/solvation/route_a/validate_bulk_water.py \
  --waterbox /path/to/hash-verified/waterbox.xyz \
  --checkpoint /path/to/MACE-OFF24_medium.model \
  --checkpoint-sha256 e5ccf5837f685899811a68754e7c994393bfd1a81720393b03c643b46c70bc69 \
  --output /path/to/new/result \
  --device cuda \
  --timestep-fs 0.5 \
  --thermalization-steps 100 \
  --thermalization-friction-per-fs 0.05 \
  --equilibration-steps 100 \
  --equilibration-friction-per-fs 0.005 \
  --production-steps 100 \
  --production-friction-per-fs 0.001 \
  --sample-interval-steps 10
```

The final v2 artifact was generated from clean implementation commit
`b172a59a62b36303ec64d90987d838f7453ec1fe`. It binds the executing
repository root and exact hashes of the runner, analysis module, MACE provider
and canonical hashing module.

| Artifact field | Recorded value |
|---|---|
| summary schema | `maple-route-a-bulk-water-validation-summary-v2` |
| manifest schema | `maple-route-a-bulk-water-validation-artifact-v2` |
| result hash | `3fda5c6b6d4dfe259d9db21ac11d0ebc50f898a52572bfb92675875d415b5f6b` |
| implementation Git HEAD | `b172a59a62b36303ec64d90987d838f7453ec1fe` |
| implementation worktree | clean |
| runtime | Python 3.11.14; `mace-torch` 0.3.16; ASE 3.27.0; PyTorch 2.12.0+cu130 |
| hardware | NVIDIA GeForce RTX 4060 Laptop GPU |
| protocol | 100 thermalization + 100 equilibration + 100 production steps |
| integration step | 0.5 fs |
| production evidence | 0.05 ps; 10 sampled frames; 5 contiguous RDF blocks |
| required minimum diagnostic | 10 ps; 500 sampled frames |

The full every-step stability monitor reported:

| Check | Result |
|---|---|
| unique states checked | 301 / 301 |
| every expected state checked | yes |
| maximum temperature over every state | 507.986590 K |
| maximum sampled production temperature | 314.204110 K |
| maximum force over every state | 5.662289 eV Å⁻¹ |
| maximum sampled production force | 4.512084 eV Å⁻¹ |
| O-H distance range | 0.884722–1.086703 Å |
| H-O-H angle range | 88.753567–126.047075° |
| original O-H-H identity preserved | yes |

The two temperature maxima cover different evidence sets: the first covers
the complete staged path, while the second covers sampled production frames.
The artifact does not retain the stage and step at which the global maximum
occurred, so no timing or causal interpretation is assigned to the 507.99 K
excursion. It does show why the full staged path must be monitored rather than
judged from the sampled production table alone. The run is accepted as an
engineering preflight and rejected as equilibrated production evidence.

The sampled production diagnostics were:

| Diagnostic | Result |
|---|---|
| mean temperature | 296.651705 K |
| relative temperature-centering error | 0.0050253 |
| fixed-cell density | 0.993804 g mL⁻¹ |
| mean pressure | 1940.496683 bar |
| potential-energy slope | 0.109098 eV water⁻¹ ps⁻¹ |
| O-O first peak | 2.775 Å, height 3.673392 |
| O-O first minimum | 3.475 Å |
| apparent coordination at first minimum | 5.10625 |

Neither the pressure nor the RDF features are equilibrium observables here.
The density is inherited from the fixed input cell, while the pressure and RDF
statistics come from only ten strongly correlated frames. The high positive
mean pressure is a warning that makes an independently equilibrated NPT
density calculation mandatory; it is not a 1 bar density result. Likewise,
the apparent coordination number is recorded only to demonstrate that the
RDF pipeline executes and must not be compared quantitatively with experiment.

The gates deliberately remained fail closed:

| Gate | Result | Reason |
|---|---|---|
| engineering stability | pass | all states finite, below emergency limits and topology-preserving |
| minimum NVT diagnostic | fail | 0.05 ps and 10 frames are below 10 ps and 500 frames |
| Hamiltonian freeze | fail | no independent long NVT replicas, NPT density, finite-size series or external reference |
| Route A scientific claim | fail | no frozen Hamiltonian and no blind hydration-free-energy comparison |

Exact copies of the result
[`summary.json`](evidence/bulk-water/maceoff24m-64w-staged-20260726-c/summary.json)
and
[`manifest.json`](evidence/bulk-water/maceoff24m-64w-staged-20260726-c/manifest.json)
are retained with this document. `arrays.npz` remains in the non-overwriting,
hash-bound local artifact directory and is not copied into Git; the manifest
records its raw byte SHA256 as
`f716e6c387cf890c30856243d2f0e95afe3a6f4241e981f65c7a3ad0b9ca9116`
and also records semantic hashes for every array.

Two earlier exploratory runs established that staged temperature control was
promising, but they used the superseded v1 summary shape and predated the
final stepwise topology/provenance guards. They are retained only in the local
research log and are not promotion evidence. In particular, their numerical
RDF and pressure values must not be migrated into a v2 result.

## Promotion gates still open

MACE-OFF24(M) must not be called the frozen Route A water Hamiltonian until all
of the following are complete:

1. **precision gate** — compare float32 and float64 energies, forces, stress,
   trajectories and RDF observables over multiple decorrelated frames before
   selecting the production dtype;
2. **fixed-density NVT gate** — run independent seeds long enough for block
   convergence of O-O, O-H and H-H RDFs and coordination statistics;
3. **NPT density gate** — obtain an equilibrated 1 bar density distribution
   with a stress-aware integrator, separately from the fixed-cell packing
   ensemble;
4. **finite-size gate** — repeat the structural/stability checks for 64, 128,
   256 and 512 waters;
5. **reference gate** — compare against numerical experimental or high-level
   water benchmarks with cited temperature, isotope and uncertainty;
6. **Route A gate** — only after freezing the periodic Hamiltonian, run real
   v3 packing/occupancy replicas and the blind hydration-free-energy holdout
   against the already frozen Route 2 baseline.

Passing the bulk-water gates is necessary but not sufficient for claiming that
Route A is more accurate than Route 2.
