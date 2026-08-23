# Route 2 conservative-vNext user guide

## Current public status

No conservative-vNext profile is currently admitted. The profile

```text
route2-profile-experimental-macemdppoint-macepolarinduced-
smoothharmonicgalerkin-electrostatic-v1
```

has a replicated historical `E/F` record at commit `4cf8db40`, where its force
was the runtime-guarded fourth-order Richardson gradient of the same registered
electrostatic scalar. The current provenance-bearing provider implementation
differs from that frozen binding, so its profile, scalar, and state are disabled
and `MACE_MDPPolarHybridSmoothHarmonicPES.evaluate()` cannot publish a current
`Route2Result`. Internal `solve/sample` diagnostics remain available.

The historical result was restricted to
the frozen checkpoint/adaptor identity on `float64` CUDA and to neutral-singlet
conductor-limit electrostatics with SMD-water Coulomb radii;
chemical accuracy, complete solvation free energy, finite-dielectric solvent
transfer, nonpolar terms, analytic force, OPT/FREQ/MD, and Tier V are not
admitted, and none of those claims carry forward. See
[HYBRID_HARMONIC_EXPERIMENTAL.md](HYBRID_HARMONIC_EXPERIMENTAL.md).

The older radial-GTO water path remains an internal validation candidate and is
deliberately unavailable through MAPLE's public calculator/result API.

Its single-water equilibrium 32/40/48/56-Angstrom box audit passes the frozen
tail thresholds. Its preregistered 20-molecule directional PES panel also
passes, as do the 20-reference-geometry, 465-component Cartesian and separated
residual-refinement panels. None of these results admits energy or force for
public use because all-panel symmetry/loop, physical-component, multi-geometry
box, and public workflow gates are still open.

The stable public legacy Route-2 inputs remain experimental energy-only paths
with their historical contracts. They must not be interpreted as the vNext
same-scalar PES described here.

## Pure MACE-POLAR frozen-source developer API

The new pure route is callable for direct E/F/virial/HVP/H validation without
pretending it is a stable public calculator profile:

```python
from maple.solvation.experimental import build_water_mace_polar_frozen_ddx_pes
from maple.solvation.models import build_official_mace_polar_1_m_radial_gto_adapter

model = build_official_mace_polar_1_m_radial_gto_adapter(device="cuda")
pes = build_water_mace_polar_frozen_ddx_pes(
    model,
    atoms.get_chemical_symbols(),
)

energy_eV = pes.get_potential_energy(atoms)
solvation_energy_eV = pes.get_solvation_energy(atoms)
forces_eV_per_A = pes.get_forces(atoms)
virial = pes.molecular_virial(atoms)  # eV; not periodic stress
hvp = pes.hessian_vector_product(atoms, direction)
hessian_eV_per_A2 = pes.get_hessian(atoms)
```

This scalar is exactly
`E_vac(MACE-POLAR) + G_ddX[R,c0(R)] + G_CDS(R)`, where `c0` is the
unmodified zero-field MACE-POLAR source. It has no MACE-MDP dependency and no
coupled self-consistent root. The force contains vacuum, moving-cavity,
source-coordinate-VJP, and CDS leaves. HVP/H are numerical derivatives of
that same conservative force and fail closed on ddX exposed-node topology
changes or error-budget violations.

The bundled CDS is the existing differentiable Fibonacci/SWIG-inspired
candidate, not exact published Lebedev-SWIG. The default and every override
are configuration-hashed. Existing pilot accuracy numbers from another CDS,
continuum, evaluator, or panel do not transfer to this identity. See
[PURE_MACE_POLAR_FROZEN_DDX.md](PURE_MACE_POLAR_FROZEN_DDX.md).

## Original radial-GTO vNext scalar (internal validation only)

The original radial-GTO operational electrostatic scalar is

```text
E_op(R) = E_vac(R) + 0.5 * <c*(R), P_R(c*(R))>_Q
G_np    = 0
```

where `c*(R)` is the converged root of
`route2-constrained-mutual-polarization-root-v1`. Field-conditioned MACE energy
and SMD/CDS are not part of this scalar.

An internal result/evidence record must identify at least:

```text
profile_id
scalar_id
state_equation_id
checkpoint/provenance SHA256
continuum configuration contract
root hash and primal residual
adjoint residual (for derivative evidence)
topology hash
scalar leaves in eV
capability admission status
```

No caller may turn this evidence into a public result by setting a capability
flag manually; capability identity is registry-derived.

## Executing the real validation canaries

These are developer audits, not user calculations:

```bash
export MAPLE_ROUTE2_REAL_MACEPOL=1
export MAPLE_ROUTE2_MACE_DEVICE=cuda
export MAPLE_ROUTE2_METHANE_MOL2=/absolute/path/to/mobley_9055303.mol2
python -m pytest -q -s --disable-warnings \
  tests/route2_vnext/test_mace_polar_real_checkpoint.py \
  tests/route2_vnext/test_mace_polar_real_operational_scalar.py \
  tests/route2_vnext/test_mace_polar_frozen_ddx_real.py

python tools/route2_release/run_mace_polar_frozen_ddx_water_canary.py \
  --device cuda \
  --checkpoint "$HOME/.cache/mace/MACEPOLAR1Mmodel" \
  --output /tmp/pure-mace-polar-frozen-ddx-water-canary.json
```

The checkpoint and MOL2 bytes must match the evidence contract. Missing
advertised assets fail the real-stack job rather than producing a green skip.

## Unsupported examples (fail closed)

The following vNext requests are intentionally unsupported today:

```text
general stable-calculator conservative-force single point
solution-phase OPT
NEB/CINEB, PRFO, Dimer, TS, IRC
stable-workflow numerical FREQ or HVP
MD
fixed-topology SMD-derived CDS
multi-solvent vNext profiles
rho-DROP forces
strict common variational MACE-continuum functional
```

They remain closed because the historical experimental Tier-F record is not a
chemical-accuracy, full-solvation, analytic-force, workflow, Tier-H, or Tier-M
release artifact. The original radial-water canary's local
derivative does not override its rotation/torque failures. The distinct
fixed-box590 diagnostic now has one-equilibrium-geometry Cartesian and
orientation evidence plus a clean-commit seven-geometry/63-direction and
bidirectional cold/warm loop audit, plus a passing equilibrium
32/40/48/56-Angstrom box-tail audit. It also has a passing source-bound
20-molecule, 60-base-geometry and 11-path-point directional panel, followed by
a passing 20-reference-geometry, 465-component Cartesian panel. Those results
and residual-refinement evidence do not substitute for all-panel
symmetry/loop coverage, multi-geometry box convergence, Hessian, NVE,
physical-component validation, or public workflow admission.

## Solvation-energy interpretation

The current radial-GTO methane value is an **electrostatic C-PCM component**.
It excludes nonpolar/CDS and standard-state terms and therefore is not
`Delta G_solv`. Do not compare it directly with the FreeSolv experimental total
and do not append a CDS term from an incompatible cavity/profile.

The historical experimental harmonic hybrid record covers only an electrostatic
component and must not be compared directly with experimental total
`Delta G_solv`. It is not a current callable admission. Complete
stable-calculator inputs will be added only after fresh provenance-bound E/F,
physical-component, accuracy, and workflow admission. OPT/NEB/TS/FREQ examples
will be added only after their corresponding workflow/H tiers pass; placeholder
commands that bypass admission are forbidden.
