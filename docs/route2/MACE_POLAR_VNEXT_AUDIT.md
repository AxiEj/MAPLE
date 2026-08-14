# MACE-POLAR conservative-vNext audit

## Current result

The vNext contracts, constrained state equation, implicit adjoint,
fixed-topology C-PCM backend, official MACE-POLAR-1-M adapter, and a conjugate
two-width radial-GTO `B/B*` path are implemented. **No Route-2 capability is
admitted.** A source/model/runtime-bound real-checkpoint canary now formally
rules out retaining both the original intrinsic field-conditioned energy and
the original four-channel source as one common scalar on the full eight-channel
field space. The separately identified fixed-box40/CPCM590 diagnostic passes
the preregistered directional and component-resolved same-scalar derivative
panels and residual-refinement gate. It still lacks all-panel symmetry/loop, workflow,
and physical-component gates, and its methane electrostatic component is much
smaller than older diagnostic profiles.

The one production-target scalar remains

\[
E_{\rm op}(R)=E_{\rm vac}(R)
+\tfrac12\langle c^*(R),P_R(c^*(R))\rangle_Q,
\]

where \(c^*=c_{\rm ref}+Ty^*\) and \(y^*\) is the selected root of

\[
T^+\left[c_{\rm ref}+Ty-
\Pi_qM_\theta\!\left(R,P_R(c_{\rm ref}+Ty)\right)\right]=0.
\]

The field-conditioned MACE energy difference and every nonpolar/CDS term are
excluded. Its total derivative is computed only by the matching implicit
adjoint. There is no independently coded force formula.

## Radial-GTO construction and claim boundary

The official checkpoint exposes a learned one-width source
`sigma=(1.5 Angstrom), l<=1` with four values per atom and an eight-feature
two-width receiver `sigma=(1.5, 3.0 Angstrom), l<=1`. The vNext candidate does
not pretend those native objects are already one four-dimensional conjugate
space. Instead it defines an explicit physical eight-channel radial-GTO space:

- the four learned coefficients occupy the `sigma=1.5 Angstrom` source block;
- the independent `sigma=3.0 Angstrom` source coefficients are zero;
- exact surface evaluation defines `B` in that same physical space;
- its exact discrete transpose defines `B*`;
- both physical field-width blocks are transformed to the checkpoint's eight
  receiver features through the content-addressed upstream projection.

This is a mathematically explicit operational surrogate construction. It is
not proof that the learned coefficients are a calibrated finite-width charge
density, a strict common variational MACE-continuum functional, or an accurate
solvation model.

The water operational-candidate profile additionally freezes dielectric
`78.39`, SMD-water Coulomb radii, and 194 Lebedev nodes per atom. Alternate
dielectric, radii, order, or injected surfaces fail profile binding.

A second, separately identified disabled diagnostic now combines the
fixed-40-A reciprocal MACE-POLAR evaluator with 590 nodes per atom.  Its one-
water Cartesian and six-orientation checks materially improve the local
symmetry/force result; exact measurements and limits are recorded in
`FIXED_BOX590_WATER_DIAGNOSTIC.md`.  It does not replace or silently change the
194-node profile. It remains unadmitted even after its directional and full
reference-geometry Cartesian and residual-refinement panels pass, because
all-panel symmetry/loop, physical-component, multi-geometry box, and public
workflow gates remain open.

## Executed official-checkpoint evidence

Environment used in the local audit: Python 3.11, Torch 2.12.0+cu130, CUDA,
`mace-torch==0.3.16`, `graph-longrange==0.4.0`, float64, official `polar-1-m`
checkpoint SHA256
`fab8b8713c832f31a2a853aaa22fd638be8a369cbf5095e6b3e982a18d10e93a`.

The earlier source/linearization canary remains recorded in
`docs/route2/evidence/vnext-mace-polar-gate-19eea1ca/`:

| check | value |
| --- | ---: |
| uniform local-field source parity | max absolute error `0.0` |
| nonuniform-field source change | L2 `1.5563605803939204e-4` |
| source JVP/VJP dot error | `2.554920623752243e-13` |
| source-position VJP directional error | `2.4787356743549704e-7` |
| total source charge | `2.7755575615628914e-17 e` |
| local versus upstream field-conditioned energy branch | `-0.0038528325721927104 eV` |

The last value is negative conjugacy evidence: it is not added to the
operational half-coupling scalar.

## Real-checkpoint common-scalar no-go

At clean execution commit
`d17c35acb3de71e9780741d57b476de3f589d9d7`, the official checkpoint was
audited at zero field and at a deterministic nonzero eight-channel field. The
exact command was:

```bash
python tools/route2_release/run_mace_conjugacy_nogo.py \
  --checkpoint /home/axie/.cache/mace/MACEPOLAR1Mmodel \
  --device cuda \
  --output /tmp/route2-mace-conjugacy-nogo-d17c35ac-run2.json
```

The original four-channel learned source is embedded in the first-width block;
the second-width source block is identically zero. The intrinsic-energy field
gradient nevertheless has a material component in the missing source-dual
subspace:

| measurement | zero field | nonzero field |
| --- | ---: | ---: |
| intrinsic-energy gradient L2 | `0.4232215784` | `0.4230873918` |
| full missing-block witness L2 | `0.3143613123` | `0.3142884952` |
| full relative witness | `0.7427818625` | `0.7428453347` |
| gauge-reduced witness L2 | `0.3096925107` | `0.3096212958` |
| gauge-reduced relative witness | `0.7317502851` | `0.7318140455` |
| numerical-zero threshold | `5.2322e-10` | `5.2309e-10` |
| reduced reciprocity defect | `1.2573947688` | `1.2569246697` |

Both signs fail the direct conjugacy and intrinsic-stationarity identities.
Forward/reverse AD directional checks agree to `1.4236e-10 eV` and
`6.4483e-12 eV`, respectively. The original frozen central-difference gate
fails because differences of the roughly 2-keV total energy are
cancellation-limited; that result remains recorded as a failure and is not
relabelled.

The complete `protocol`, `states`, `decision`, and measurement digest replayed
exactly in a second cold process. The primary measurement digest is
`d89b620ed31cfdc0dfd7a89d03551251008458fe62e60c4eacda020255ab6909`.
The source/checkpoint/runtime-bound artifacts and warnings are retained under
[`evidence/mace-conjugacy-nogo-d17c35ac/`](evidence/mace-conjugacy-nogo-d17c35ac/README.md).

One material counterexample is sufficient to disprove the global claim that
the original energy and original source can both be retained unchanged in one
scalar. It does not admit the separately defined eight-channel energy-gradient
effective source, conservative nuclear forces, or any chemical-accuracy claim.

## Changed-source common-stationarity canary

The separately named complete eight-channel energy-gradient source was then
tested with the fixed reciprocal CPCM194 scalar at clean execution commit
`576550e9cfe1011519c7bd93e7ee256972ff2560`:

```bash
python tools/route2_release/run_variational_common_water_canary.py \
  --checkpoint /home/axie/.cache/mace/MACEPOLAR1Mmodel \
  --device cuda \
  --output /tmp/route2-variational-common-water-576550e9-run1.json
```

The common state converged in 32 Anderson iterations to actual unmixed reduced
residual `1.104397210579806e-10`. A warm evaluation from the stored root had
zero source and stationary-energy difference. For one deterministic
translation-free coordinate direction, the stationary-envelope derivative was
`-0.1039157969636657 eV/Angstrom`. Three independently re-solved central
differences gave:

| step (Angstrom) | absolute error (eV/Angstrom) | relative error |
| ---: | ---: | ---: |
| `5e-4` | `2.7915015345103544e-7` | `2.686303831808301e-6` |
| `2e-4` | `4.157446151231703e-9` | `4.000783588933413e-8` |
| `1e-4` | `2.9895738047208686e-6` | `2.8769194791107433e-5` |

An independent cold process reproduced the complete protocol, geometry,
identity, roots, envelope measurement, and decision exactly. Both executions
have measurement SHA-256
`62d43c63868270fc74254cf0ddbc182b33af0d158c3fbca1ef6493cdf52ee9fc`.
Raw source/checkpoint/runtime-bound evidence and unfiltered runtime warnings are
retained under
[`evidence/variational-common-water-576550e9/`](evidence/variational-common-water-576550e9/README.md).

This closes only a one-geometry implementation canary for the *changed model
identity*. The original density head is the zero-field anchor and a diagnostic,
not the returned source. The continuum still uses a conventional
laboratory-fixed 194-point Lebedev assembly, which has no structural global
`SO(3)` guarantee. Passivity, root uniqueness over a declared domain, combined
Hessian stability, harmonic moving-geometry derivatives, full symmetry/PES
panels, and chemical validation remain open. Therefore `E/F/H/V/M` all remain
false and no public force was admitted.

## Real water same-scalar force audit

Command:

```bash
export MAPLE_ROUTE2_REAL_MACEPOL=1
export MAPLE_ROUTE2_MACE_DEVICE=cuda
python -m pytest -q -s --disable-warnings \
  tests/route2_vnext/test_mace_polar_real_operational_scalar.py
```

For one water geometry, the root and adjoint residuals were
`3.8119194931281014e-13` and `1.5496456060009448e-13`. Cold/warm energy agreed
exactly and source L2 difference was `9.339417195847146e-13`. The scalar leaves
were:

| leaf | value |
| --- | ---: |
| vacuum | `-2079.863671296707 eV` |
| C-PCM electrostatic half coupling | `-0.017148810303582215 eV` |
| total operational scalar | `-2079.8808201070105 eV` |

For one normalized internal direction, the analytic derivative was
`-0.361775005453125 eV/Angstrom`. Central-difference errors were:

| step (Angstrom) | absolute error (eV/Angstrom) | relative error |
| ---: | ---: | ---: |
| `4e-4` | `1.5643755e-5` | `4.32398e-5` |
| `2e-4` | `1.9200484e-6` | `5.30727e-6` |
| `1e-4` | `2.7169932e-6` | `7.51012e-6` |

That local same-scalar direction passes. It is not Tier F: the executed
symmetry audit found rotation-force relative error
`2.5151882742290786e-4` (threshold `1e-4`) and torque
`1.0777102111375557e-4 eV` (threshold `1e-4 eV`). Translation energy, net
force, and topology checks passed. The rotation defect includes contributions
from the vacuum model and continuum quadrature, so it must not be hidden by
loosening the gate.

## Methane solvation-component comparison

For the pinned FreeSolv `mobley_9055303` MOL2 geometry, the 194-point radial-GTO
water candidate produced:

| quantity | result |
| --- | ---: |
| root residual | `6.406575415816992e-13` |
| C-PCM electrostatic component | `-0.056741561355549835 kcal/mol` |
| frozen gas-source component | `-0.056180124331638936 kcal/mol` |
| root minus gas source L2 | `9.165360565771093e-4` |

The magnitude collapse is already present at the frozen gas source; it is not
caused by failure of the outer fixed-point solve.

Comparisons must retain method identity:

| older record | electrostatic (kcal/mol) | relation to new result |
| --- | ---: | --- |
| vNext local-jet, 86 nodes/atom | `-1.483425546707858` | different source kernel/grid; new magnitude is `3.825%` |
| legacy point-source / exact-GTO receiver / PCMSolver | `-1.1554017711374178` | known nonconjugate source/receiver and different backend/cavity; new magnitude is `4.911%` |
| rho-DROP source-dependent cavity | `-0.239490996880803` | different cavity and incomplete coordinate derivative; not parity evidence |

The legacy record also had a separate CDS term `+2.764796099834587` and total
ledger `+1.609394328697169 kcal/mol`, versus the experimental total
`+2.0 kcal/mol`. Those CDS/cavity semantics are incompatible with the new
electrostatics-only profile. Therefore `-0.05674` cannot be appended to that CDS,
cannot be scored against `+2.0`, and does **not** show an accuracy improvement.
It is negative physical-component evidence requiring source-normalization and
matched QM/C-PCM investigation.

## Capability decision

| capability | status | reason |
| --- | --- | --- |
| scalar energy E | closed | callable internal scalar lacks release and physical-component admission |
| conservative force F | closed | fixed-box diagnostic passes directional, Cartesian, and residual-refinement panels, but all-panel symmetry/loop and public workflow admission remain open |
| Hessian/FREQ H | closed | depends on admitted F and raw-Hessian gates |
| strict variational V | closed | original-energy/original-source route formally ruled out by a real-checkpoint counterexample; the changed-source candidate remains unadmitted |
| MD M | closed | depends on admitted F plus path/loop/NVE gates |
| OPT/NEB/TS/IRC | closed | no Tier-F profile |

Next work is not solver tuning or another attempt to retain the original source
inside a common functional. The optional strict branch must use the separately
identified energy-gradient effective source and pass its sign/gauge,
passivity/root, envelope-coordinate, and release gates. The operational branch
still requires a matched source-normalization/basis audit, component-level
QM/C-PCM comparison, and the remaining all-panel symmetry/workflow gates.
