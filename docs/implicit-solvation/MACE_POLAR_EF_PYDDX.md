# MACE-POLAR-EF v2 + PCM adapters

## Status: explicit known-nonpassive diagnostic exposure

The standard `pyddx==0.8.0` ddPCM PyTorch bridge is implemented and retains
the upstream cavity, operator, solve, adjoint source derivative, and analytic
coordinate derivative. The supplied `macepol-ef-v2.pt` checkpoint is also
connected to the smooth PCM and registered under explicit diagnostic
profiles. Its electronic response is known to fail the
required passivity test after correcting the spin input to singlet
multiplicity `1`.

The v2 multi-solvent derivative profile requires `experimental=true`,
`acknowledge_known_nonpassive=true`, and
`acknowledge_unvalidated_derivatives=true`. It exposes analytic first
derivatives, numerical Hessians formed from those derivatives, L-BFGS OPT,
FREQ, and P-RFO TS. Default selection and chemical-accuracy admission remain
closed.

### Explicit input

```text
#model=mace-polar-ef-v2(model_path=/absolute/path/macepol-ef-v2.pt)
#sp(verbose=1)
#device=gpu0
#solv(implicit=methanol,method=smd,provider=torch-smooth-pcm,profile=mace-polar-ef-v2-smooth-ddpcm-l3-p6-r96-128-128-multisolv-derivatives-known-nonpassive-v2,response=scf,standard_state=1m,experimental=true,acknowledge_known_nonpassive=true,acknowledge_unvalidated_derivatives=true)

0 1
O   0.000000   0.000000   0.000000
H   0.957200   0.000000   0.000000
H  -0.239987   0.927297   0.000000
```

Every normal output prints `KNOWN-NONPASSIVE DIAGNOSTIC` and
`scientifically_valid=false` before the energy decomposition.

The same profile accepts water, methanol, ethanol, acetonitrile, DMSO, DMF,
THF, chloroform, dichloromethane, toluene, and hexane. Replace the task line
with `#opt(method=lbfgs)`, `#freq`, or `#ts(method=prfo)` for those workflows.

## Connection to `torch-smooth-pcm-v1`

The EF adapter is now connected to the private fixed-dimensional smooth PCM
through one backend-neutral stationary core:

```text
mace_polar_ef_stationary.py    common scalar, SCF, passivity, envelope gradient
mace_polar_ef_pyddx.py         standard pyddx/ddPCM adapter
mace_polar_ef_smooth_pcm.py    torch-smooth-pcm-v1 adapter
```

Both continuum paths use the same equations and safeguarded-Anderson
implementation:

```text
L(R,c,f) = E_MACE-EF(R,f) + U_PCM(R,c) - <c,f>
c = Q dE_MACE-EF/df
f = Q^T dU_PCM/dc
```

The smooth path has the distinct private identity
`maple.route2.coupling.mace-polar-ef-v2-torch-smooth-pcm-stationary-first-order.v1`.
It preserves the EF float32 geometry as the common coordinate identity and
promotes those exact values to CPU float64 for the smooth PCM.

The private default policy stops before PCM iteration:

```text
MACEPolarEFPassivityError:
MACE-POLAR-EF-v2 failed the required external-potential
concavity/passivity gate; refusing torch-smooth-pcm-v1 coupling.
```

The explicit public diagnostic profile instead uses the profile-owned policy
`record-known-failure-diagnostic`: it records the failed eigenvalues and
continues only because the input contains the second acknowledgement above.
This is not an admission bypass—provenance fixes `scientifically_valid=false`,
`accuracy_certified=false`, `solution_phase_pes=false`, and
`release_admitted=false`.

## What the EF wrapper actually adds

The exact checkpoint is bound by SHA-256:

```text
macepol-ef-v2.pt
4f820d381d06bbb37b02574c38da7203e5d429407fa2a512231fc08cdbb69b6b
```

It accepts an independent local electrostatic jet for every atom:

```text
f_i = [V_i, grad_x V_i, grad_y V_i, grad_z V_i].
```

MAPLE preserves the complete `N x 4` array. Standard ddPCM therefore supplies
the potential and gradient at each atomic centre without averaging them into
one molecular field. The checkpoint wrapper API expects the physical electric
field, so MAPLE converts `grad(V)` to `E=-grad(V)` before the call; the wrapper
then applies its built-in negation and the underlying model receives
`grad(V)`.

There is an important narrower limitation in these checkpoint bytes:

- `external_potential[i]` enters the explicit `q_i V_i` energy term;
- the auxiliary density head is unchanged when only `external_potential`
  changes;
- `external_field[i]` enters the learned field-response path and can differ
  between atoms.
- the traced polar head embeds `cuda:0`, so this exact asset is not a portable
  CPU or arbitrary-GPU checkpoint.

Thus the checkpoint supports atomwise nonuniform drives, but it does not prove
that arbitrary atomwise potentials form a conservative electronic response.

## Correct spin convention

The traced model applies `total_spin - 1` when constructing spin-up and
spin-down populations. Its input is therefore spin **multiplicity**, not
`2S` and not the number of unpaired electrons. Neutral water must use
`spin_multiplicity=1`. Earlier diagnostic water and FreeSolv-methanol numbers
obtained with the old default `0` are invalid and must not be used as accuracy
evidence.

## Why the common-scalar candidate is closed

The candidate source was defined from the checkpoint energy,

```text
c_raw = Q dE_MACE(R,f)/df,
Q [V,gx,gy,gz] = [V,gy,gz,gx],
```

with the prospective common scalar

```text
L(R,c,f) = E_MACE(R,f) + G_ddPCM(R,c) - <c,f>.
```

Autograd consistency establishes only that this scalar differentiates its own
implementation. A physical ground-state energy must also be concave in an
applied electrostatic potential, because the static polarizability is
`alpha = -d2E/dE2`.

For water at singlet multiplicity `1`, the audit supplies both parts of a
coherent affine external potential,

```text
V_i = g dot (R_i - centroid)
grad(V)_i = g,
|g| step = 1e-3 eV/(e Angstrom).
```

The measured energy-Hessian eigenvalue ranges remain positive across a
three-step resolution check:

```text
step 5e-4: 7.429 to 7.436
step 1e-3: 3.666 to 3.673
step 2e-3: 1.785 to 1.792
```

They are positive rather than non-positive. The corresponding polarizability
would have the wrong sign. Reversing the field convention cannot repair a
Hessian sign. At a diagnostic ddPCM root, the full local field Hessian is also
indefinite. Standard/private admission therefore fails; only the explicitly
acknowledged known-invalid smooth-PCM diagnostic is allowed to continue.

The adapter additionally:

- passes raw energy gradients through a charge-drift gate instead of silently
  projecting them onto neutrality;
- rejects public second-order autograd;
- records that pyddx's requested tolerance is known while its achieved
  algebraic residual is unavailable;
- loads the exact verified checkpoint bytes from memory, so path replacement
  cannot create hybrid provenance.

## Standard ddPCM PyTorch boundary

`torch_pyddx.py` remains independently usable as a first-order autograd scalar:

- positions: float64, angstrom;
- raw point-multipole source: float64 `[q,y,z,x]`;
- output energy: eV;
- backward position gradient: eV/angstrom;
- backward source gradient: raw source-dual order;
- double backward: deliberately unavailable.

## Reproducible checkpoint audit

```bash
python docs/implicit-solvation/benchmarks/audit_mace_polar_ef_v2_water.py --checkpoint "$PWD/macepol-ef-v2.pt" --output /tmp/mace-polar-ef-v2-water-audit.json
```

The default coupling code remains behind the passivity gate so that a
corrected, provenance-bound checkpoint can reuse the same stationary core.
The exposed diagnostic records rather than conceals the failed gate.

## Larger `macepol-ef-L.pt` candidate

The local L checkpoint was independently inspected rather than substituted
under the v2 identity:

```text
size:   130965540 bytes
sha256: 4a665c27e5e4e5a2bfe80c69705c18ef2ba418986e5113a6eafe956207993484
class:  PolarMACETraceable
```

It accepts an `N x 3` atomwise electric-field tensor and responds to a
zero-mean nonuniform field, but exposes no atomwise scalar-potential input.
Its charge-position energy term uses the molecular mean field, so it cannot
represent the complete ddPCM local jet `[V_i, grad(V)_i]` without changing the
model energy definition.

The same singlet-water passivity scan also fails:

```text
step 5e-4: 11.523 to 11.539
step 1e-3:  5.706 to  5.718
step 2e-3:  2.794 to  2.808
```

At zero field, its energy-derived dipole is approximately
`[0.3953, 0.5113, 0.0] e Angstrom`, while the moment reconstructed from its
reported density is `[0.1797, 0.2323, 0.0] e Angstrom`. Therefore the L
checkpoint is neither a drop-in atomwise-potential replacement nor admitted
as a common electronic functional for ddPCM.

## Medium `macepol-ef-M.pt` candidate

The local M checkpoint is another distinct old-interface identity:

```text
size:         69326774 bytes
sha256:       066e362a11081c5d07218b4007052ca96dafca1e12358536881e96ab6834e196
class:        PolarMACETraceable
interactions: 2
```

Like L, it accepts an `N x 3` atomwise electric field but has no atomwise
scalar-potential input. A zero-mean nonuniform field changes its water energy
by `0.008545 eV` and its reported density by up to `6.56e-4`, confirming that
the vector rows are not collapsed. Nevertheless, its singlet-water energy
curvature also has the wrong sign:

```text
step 5e-4: 15.952 to 16.036
step 1e-3:  7.872 to  7.954
step 2e-3:  3.831 to  3.913
```

Its zero-field energy-derived dipole (`[0.3948, 0.5107, 0.0] e Angstrom`)
also disagrees with the moment reconstructed from the reported density
(`[0.1974, 0.2553, 0.0] e Angstrom`). M is therefore not admitted for ddPCM.

## References

- MACE-POLAR-1 defines molecular polarizability as the negative second energy
  derivative with respect to an applied field:
  <https://arxiv.org/html/2602.19411>.
- ddX v0.8.0 theory and implementation reference, commit
  `4d79e3d9caeae5e602683572a71cb550414f9b09`.
- Stamm et al., *J. Chem. Phys.* **144**, 054101 (2016),
  <https://doi.org/10.1063/1.4940136>.
- Gatto, Lipparini, and Stamm, *J. Chem. Phys.* **147**, 224108 (2017),
  <https://doi.org/10.1063/1.5008329>.
