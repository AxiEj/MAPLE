# Route 2 conservative-vNext user guide

## Current public status

There is one narrowly admitted **experimental programmatic** Route-2 profile:

```text
route2-profile-experimental-macemdppoint-macepolarinduced-
smoothharmonicgalerkin-electrostatic-v1
```

It exposes `E=true` and `F=true` through
`MACE_MDPPolarHybridSmoothHarmonicPES.evaluate()`. Its force is the
matrix-free implicit-adjoint total derivative of the same registered
electrostatic scalar; the re-solved fourth-order Richardson path remains a
diagnostic oracle. `H/V/M` remain false, and every other conservative-vNext
profile remains disabled.

It is now reachable through MAPLE's normal ASE calculator boundary under the
explicit experimental model name `macemdppolarhybrid`:

```python
from maple.function.calculator.set_calculator import SetCalculator

calc = SetCalculator(
    device="cuda",
    model="macemdppolarhybrid",
    output="maple.out",
    atoms=atoms,
).set_calculator()
atoms.calc = calc
energy_eV = atoms.get_potential_energy()
forces_eV_per_A = atoms.get_forces()
```

This is not yet a stable general MAPLE workflow feature. It is restricted to
the frozen checkpoint/adaptor identity on `float64` CUDA and to neutral-singlet
conductor-limit electrostatics with SMD-water Coulomb radii. Chemical accuracy,
complete solvation free energy, finite-dielectric solvent transfer, nonpolar
terms, H/V, FREQ/MD, and Tier V are not admitted. Normal MAPLE
SP and first-order `LBFGS`/`SD`/`SDCG`/`CG` OPT are enabled as experimental E/F
workflows; RFO and every H/TS/IRC/MD/scan path are rejected. The calculator
also rejects Hessian, stress, and any second-solvent request rather than
silently returning an unsupported result. See
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

The separated pyddx-ddPCM plus stock SMD-CDS scalar is now reachable through
the explicit model name `macemdppolarhybridddx`:

```text
#model=macemdppolarhybridddx(solvent=water,hessian=numerical)
#device=cuda
#sp
```

The model owns its continuum and solvent ledger, so this input deliberately
does **not** add a separate `#solv(...)` directive.  Other registered solvents
are selected with the model-local `solvent=` option, for example
`solvent=ethanol`.

```python
calc = SetCalculator(
    device="cuda",
    model="macemdppolarhybridddx",
    solvent="water",
    model_options={"hessian": "numerical"},
    output="maple.out",
    atoms=atoms,
).set_calculator()
atoms.calc = calc

energy_eV = atoms.get_potential_energy()
forces_eV_per_A = atoms.get_forces()
hessian_eV_per_A2 = calc.get_hessian(atoms)
molecular_virial_eV = calc.get_molecular_virial(atoms)
```

MAPLE SP and first-order `LBFGS`/`SD`/`SDCG`/`CG` OPT are enabled for this
model; H and the molecular virial are direct programmatic calls.  The
MACE-POLAR graph runs on CUDA; the frozen
MACE-MDP coefficient adapter intentionally remains CPU/float64 because that is
its audited inference identity.  The calculator owns both ddX and the solvent
term, so `implicit=none` is mandatory and prevents accidental double counting.

This is an availability decision, not an accuracy admission. Its frozen
505-development result is `1.696313 kcal/mol` MAE with a `14.904481 kcal/mol`
maximum absolute error. The Hessian is a Richardson derivative of the total
same-scalar analytic force, but the internal PySCF-SMD surface topology is not
observable. Its backend uses the explicit
`observed-components-only-experimental-v1` policy; H diagnostics and FREQ output
report the actual steps, error estimates, `partial-experimental` guard status,
and unobservable component list. Up to six topology-preserving step halvings
are allowed without relaxing error or antisymmetry bounds. Experimental `FREQ` now
selects vibrational-only output for this calculator. It does not compute
gas-phase translational/rotational thermochemistry or a thermochemical Gibbs
correction. Explicit requests for gas thermochemistry or hiding small
imaginary frequencies are rejected. The same guarded numerical Hessian is
used; an error or topology failure is not silently repaired.

Use these job headers with the `macemdppolarhybridddx` model header above:

```text
#sp
#opt(method=lbfgs)
#freq(method=mw,thermochemistry=none)
```

Run each job as usual; FREQ should follow a converged optimization. Small
near-zero frequencies are numerically uncertain, not proof of a saddle.
The `1.5 kcal/mol` accuracy target does not prevent these experimental jobs.

For atom-order-matched endpoints optimized under the same model and solvent:

```text
#ts(method=neb)
#ts(method=neb,refine=cineb)
```

The second form enables climbing-image refinement. These E/F-only paths do
not call a Hessian, HVP, or automatic frequency analysis. Their highest-energy
or climbing images are **TS candidates**, not certified first-order saddles.
Path images must share the same content-bound PES identity. A max-iteration
termination is not convergence. Chemical barrier accuracy requires independent
reference evidence, beyond a converged numerical path.

Both endpoints must satisfy maximum/RMS Cartesian force-component limits of
`1.0e-3` / `5.0e-4 Hartree/Angstrom`. Caller settings can tighten, not loosen,
these hybrid limits. Use converged endpoint files or request
`#ts(method=neb,initial_opt=true)`; failed endpoint optimization does not proceed
to the hybrid path search.

Periodic stress, strict Tier V, RFO/PRFO, `refine=nebts`, Dimer, IRC/MD,
charged/open-shell chemistry, solution-phase thermochemistry, and production
capability flags remain closed. The harmonic `macemdppolarhybrid` profile is
unchanged by this separate ddX availability extension.

In the Route-2 `E/F/H/V/M` notation, `V` means a strict common variational
functional. It is not an abbreviation for virial. The nonperiodic molecular
virial above is callable now; strict `V` remains false for the unchanged
MACE-MDP + MACE-POLAR checkpoint pair.

This daily surface is also distinct from the running canonical-ADT accuracy
candidate.  Canonical ADT currently has E and fixed-geometry response
derivatives only; its complete moving-geometry coordinate VJP is not yet
implemented, so ADT results cannot silently replace this calculator's force or
Hessian.

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
solution-phase OPT outside the explicit experimental
  macemdppolarhybrid/macemdppolarhybridddx first-order lane
NEB/CINEB outside the explicit macemdppolarhybridddx experimental lane
PRFO, Dimer, NEBTS, IRC
production-admitted numerical FREQ or HVP
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
