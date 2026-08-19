# Route 2 conservative-vNext user guide

## Current public status

There is one narrowly admitted **experimental programmatic** Route-2 profile:

```text
route2-profile-experimental-macemdppoint-macepolarinduced-
smoothharmonicgalerkin-electrostatic-v1
```

It exposes `E=true` and `F=true` through
`MACE_MDPPolarHybridSmoothHarmonicPES.evaluate()`. Its force is the
runtime-guarded fourth-order Richardson gradient of the same registered
electrostatic scalar. `H/V/M` remain false, and every other conservative-vNext
profile remains disabled.

This is not yet a stable general MAPLE calculator feature. It is restricted to
the frozen checkpoint/adaptor identity on `float64` CUDA and to neutral-singlet
conductor-limit electrostatics with SMD-water Coulomb radii;
chemical accuracy, complete solvation free energy, finite-dielectric solvent
transfer, nonpolar terms, analytic force, OPT/FREQ/MD, and Tier V are not
admitted. See
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

## Experimental named-solvent hybrid daily surface

Accuracy development no longer blocks direct exploratory use of the frozen
operational scalar.  The explicit calculator identity
`macemdppolarhybridddx` composes MACE-MDP permanent q/p, zero-anchored
MACE-POLAR response, pyddx ddPCM, and the existing named-solvent SMD-CDS term:

```python
calc = SetCalculator(
    device="cuda",
    model="macemdppolarhybridddx",
    solvent="water",                 # any registered Route-2 solvent
    model_options={
        "module": (
            "maple.function.calculator.route2."
            "_mace_mdp_polar_hybrid_ddx_calculator"
        ),
        "route2_solvent": "water",  # any registered Route-2 solvent
        "hessian": "numerical",
    },
    output="maple.out",
    atoms=atoms,
).set_calculator()
atoms.calc = calc

energy_eV = atoms.get_potential_energy()
forces_eV_per_A = atoms.get_forces()
hessian_eV_per_A2 = calc.get_hessian(atoms)
molecular_virial_eV = calc.get_molecular_virial(atoms)
```

MAPLE `SP` and first-order `LBFGS`/`SD`/`SDCG`/`CG` optimization are enabled.
The explicit `module` keeps this experimental calculator outside the stable
builtin registry while making it available through the normal MAPLE factory.
For a normal MAPLE input file, use the same options in the model header; this
profile owns its solvent ledger, so do not add a second `#solv` correction:

```text
#model=macemdppolarhybridddx(module=maple.function.calculator.route2._mace_mdp_polar_hybrid_ddx_calculator,route2_solvent=water,hessian=numerical)
#sp(verbose=1)
```
The MACE-POLAR graph runs on CUDA; the audited MACE-MDP coefficient adapter
remains CPU/float64.  The calculator owns its ddX and SMD-CDS terms, so
`implicit=none` is mandatory and D4 composition is rejected rather than double
counted.

This is an availability decision, not a claim that the current solvation
accuracy is satisfactory.  The frozen 505-development baseline is
`1.696313 kcal/mol` MAE with `14.904481 kcal/mol` maximum absolute error.  H is
the Richardson derivative of the same analytic E/F scalar under an explicit
observed-components-only policy for the opaque PySCF SMD surface topology.
Periodic stress, strict Tier V, FREQ/TS/IRC/scan, and MD remain closed.  The
existing FREQ driver adds gas-phase translational/rotational thermochemistry,
so exposing it unchanged would be chemically incorrect for this solution-phase
scalar.

In the historical `E/F/H/V/M` notation, `V` is strict common variational
functionality; it does not mean virial.  The molecular virial above is
available even though strict `V` remains false.

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

Topology coverage is recorded separately from the topology projection hash.
The PySCF SMD-CDS adapter cannot observe libsolvent's internal surface topology,
so point-ddX + PySCF-SMD states are labelled `partial`. Their HVP/H path rejects
by default; an explicitly supplied
`observed-components-only-experimental-v1` Richardson policy runs only as a
diagnostic and records the unobservable component, policy SHA256, and retry count.
Such a result is not fully topology fail-closed and does not change the `H=no`
workflow admission.

The bundled CDS is the existing differentiable Fibonacci/SWIG-inspired
candidate, not exact published Lebedev-SWIG. The default and every override
are configuration-hashed. Existing pilot accuracy numbers from another CDS,
continuum, evaluator, or panel do not transfer to this identity. See
[PURE_MACE_POLAR_FROZEN_DDX.md](PURE_MACE_POLAR_FROZEN_DDX.md).

## MACE-MDP + MACE-POLAR separated-ddX developer API

The separated-source hybrid ddX surface is also callable without claiming a
common conjugate functional or registry admission:

```python
from maple.solvation.experimental import MACE_MDPPolarHybridDDXPES
from maple.solvation.derivatives import RichardsonScalarHessian

pes = MACE_MDPPolarHybridDDXPES(
    hybrid=hybrid_model,
    symbols=atoms.get_chemical_symbols(),
    cavity_radii_angstrom=coulomb_radii,
    continuum_model="pcm",
    dielectric=dielectric,
    lmax=15,
    n_lebedev=1202,
    hessian_backend=RichardsonScalarHessian(),
)

state = pes.solve(atoms)
energy_eV = pes.get_potential_energy(atoms)
force = pes.evaluate_forces(atoms, central_state=state)
forces_eV_per_A = force.total_forces_ev_per_angstrom
virial = pes.molecular_virial(atoms, force_evaluation=force)
hvp = pes.hessian_vector_product(atoms, direction, central_force=force)
hessian_eV_per_A2 = pes.get_hessian(atoms)
```

The operational ledger is exactly `E_vac + E_ddX,pol`: permanent point
multipoles come from MACE-MDP and the induced increment comes from MACE-POLAR.
F is the block implicit adjoint of that ledger.  The molecular virial is its
homogeneous molecular-deformation derivative, not periodic stress or the
historical Tier-V common-functional flag.  HVP/H are error-estimated Richardson
derivatives of the same replayed analytic force; the derivative-policy SHA,
actual step, topology status, error estimate, and raw Hessian antisymmetry are
recorded.

Successful local root replay and adjoint convergence do not prove global root
uniqueness or a nonsingular state Jacobian.  The ddX exposed-node topology guard
also does not prove global smoothness or SO(3) covariance.  This identity is
electrostatic-only and currently has no CDS/nonpolar completion, independent
force/virial/Hessian accuracy panel, or FREQ/OPT/MD admission.  Energy MAE from
another identity cannot fill those derivative gates.

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

They remain closed because the admitted experimental Tier-F record is not a
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

The admitted experimental harmonic hybrid still provides only an electrostatic
component and must not be compared directly with experimental total
`Delta G_solv`. Complete stable-calculator inputs will be added only after
physical-component, accuracy, and workflow admission. OPT/NEB/TS/FREQ examples
will be added only after their corresponding workflow/H tiers pass;
placeholder commands that bypass admission are forbidden.
