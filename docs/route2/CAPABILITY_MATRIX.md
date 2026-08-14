# Route 2 capability matrix

Capabilities are evidence-admitted, not inferred from a callable method or a
CLI restriction.

| tier | meaning |
| --- | --- |
| E | scalar energy |
| F | conservative force of that same scalar |
| H | force-derived numerical Hessian/HVP admitted for workflows |
| V | strict common variational electronic-continuum functional |
| M | public MD release gate |

## Conservative-vNext registry

| scalar/profile | E | F | H | V | M | current disposition |
| --- | :---: | :---: | :---: | :---: | :---: | --- |
| `route2-operational-cpcm-fixedtopology-electrostatic-v1` | no | no | no | no | no | scalar/state kernel implemented; legacy-width profile remains unadmitted |
| `route2-profile-operational-cpcm-fixedtopology-radialgto-electrostatic-v1` | no | no | no | no | no | real same-scalar derivative candidate; rotation/torque and physical-component gates failed |
| `route2-operational-macepolar-analytic-gaussian-multipole-smoothharmonicgalerkin-cpcm-v1` | no | no | no | no | no | separately registered operational PES candidate: analytic isotropic MACE evaluator, original four-channel density response embedded in the first radial block, and scalar-first smooth harmonic continuum; the contiguous frozen rigid-panel range `[0,12)` passes in two clean processes per molecule, but physical-component, accuracy, PES/domain, Hessian/FREQ/MD, and release gates remain absent |
| `route2-profile-diagnostic-fixedbox40-cpcm590-radialgto-electrostatic-v1` | no | no | no | no | no | earlier derivative/path panels pass, but the frozen all-panel water canary fails rotation energy and force covariance; retained as negative evidence, not admissible |
| `route2-profile-diagnostic-fixedbox{32,48,56}-cpcm590-radialgto-electrostatic-v1` | no | no | no | no | no | preregistered box controls passed at one equilibrium water geometry; distinct identities, no adaptive selection, no public capability |
| `route2-profile-diagnostic-fixedbox48-cpcm1202-radialgto-electrostatic-v1` | no | no | no | no | no | separately versioned higher-order candidate; same scalar and unchanged symmetry thresholds, no executed release evidence yet |
| `route2-profile-diagnostic-pairframe-cpcm110-radialgto-electrostatic-v1` | no | no | no | no | no | distinct ordered-pair-frame ensemble discretization; algebra/continuum tests pass and one preliminary unbound real methanol engineering run meets local thresholds, but clean source-bound PES/symmetry/accuracy evidence is absent |
| `route2-profile-diagnostic-ddx-ddpcm194-radialgto-electrostatic-v1` | no | no | no | no | no | full eight-channel joint `(psi,phi)` map is derived from one ddPCM scalar and passes local derivative tests; ddX finite-grid rotation drift and missing achieved algebraic-residual report keep it diagnostic |
| `route2-profile-diagnostic-cpcm-injectedgrid-radialgto-electrostatic-v1` | no | no | no | no | no | synthetic injected-grid diagnostic only |
| `route2-diagnostic-localjet-cpcm-fixedtopology-electrostatic-v1` | no | no | no | no | no | implemented diagnostic; exact-GTO mismatch prevents admission |
| `route2-operational-cpcm-fixedtopology-smdcds-v1` | no | no | no | no | no | blocked until electrostatic F/H and same-scalar CDS force pass |
| `route2-variational-common-functional-v1` | no | no | no | no | no | the current checkpoint's original intrinsic energy plus original four-channel source instantiation is formally ruled out by a source-bound real-checkpoint counterexample; the generic scalar identity remains disabled rather than being reassigned to a changed model |
| `route2-variational-macepolar-energygradient-fixedcavity-cpcm-v1` | no | no | no | no | no | scalar-first complete eight-channel effective-source candidate, fixed reciprocal C-PCM scalar, and their common constrained state/envelope kernel are implemented; a source-bound real-checkpoint water canary passes cold/warm replay and one three-step envelope FD, but the pinned molecular-realspace MACE scalar has a structural `SO(3)` counterexample in its upstream fixed-axis finite-difference long-range operator |
| `route2-variational-macepolar-energygradient-fixedcavity-harmonicgalerkin-cpcm-v1` | no | no | no | no | no | complete-irrep coefficient action, exact-adjoint fixed-snapshot scalar, and common-state integration are implemented; the external coefficient snapshot is geometry independent and intentionally has zero continuum coordinate partial; every release gate remains missing |
| `route2-variational-macepolar-energygradient-smoothharmonicgalerkin-cpcm-v1` | no | no | no | no | no | distinct moving smooth weighted-overlap scalar; its Torch graph reassembles `E`, `K`, `V`, `A=E.T K E`, and `S=E.T V`, and a real-checkpoint water run passes root replay plus a three-step envelope FD; the harmonic continuum alone rotates at roundoff, but the full scalar fails because the current MACE molecular-realspace evaluator is not an exact `SO(3)` intertwiner |
| `route2-variational-macepolar-analytic-gaussian-multipole-energygradient-smoothharmonicgalerkin-cpcm-v1` | no | no | no | no | no | separately identified changed-inference candidate; one clean water common-state canary passes replay, envelope FD, and one rotation, and five starts find the same local root, but the source-bound reduced susceptibility is indefinite and singular, so passivity, local invertibility, feedback-sign, and combined-Hessian Tier-V gates fail |

The first release target is the operational electrostatic profile. Tier V is
not required for it and must remain false unless the model energy/source
identity, reciprocity, stability, invertibility, and full coordinate derivative
are independently proven.

The decisive current negative artifact is
[`evidence/mace-conjugacy-nogo-d17c35ac/`](evidence/mace-conjugacy-nogo-d17c35ac/README.md).
At clean commit `d17c35ac`, both tested field states have a gauge-reduced
missing-radial witness relative magnitude above `0.731`, versus numerical-zero
thresholds near `5.23e-10`; both source/energy signs fail. This closes only the
"retain original energy and original source" route. The separately named
eight-channel energy-gradient source is a changed model identity and has no
admitted tier.

The model-level rotational counterexample is retained separately under
[`evidence/mace-realspace-so3-nogo-6da676cd/`](evidence/mace-realspace-so3-nogo-6da676cd/README.md).
At exact zero external field, the anchored checkpoint scalar drifts by
`8.888361298886593e-5 eV`, its energy-gradient source has relative covariance
error `1.573371208574198e-4`, and its fixed-field coordinate gradient has
relative covariance error `1.2075108741020767e-3`. With the checkpoint source
rotated exactly, the isolated upstream real-space feature operator still has
relative covariance error `2.1280691879188778e-2`. Therefore neither changing
the continuum nor changing the eight-channel field transform can admit the
current model profile.

The corresponding positive-but-narrow changed-source implementation canary is
[`evidence/variational-common-water-576550e9/`](evidence/variational-common-water-576550e9/README.md).
It proves that one real water state of the new scalar converges and that its
stationary-envelope derivative matches three re-solved finite differences. It
does not override the `no` entries above: the sampled Lebedev continuum has no
structural global `SO(3)` guarantee, and the required passivity, uniqueness,
Hessian, domain, PES, and chemical gates are absent.

The separately versioned analytic Gaussian-multipole inference candidate is
retained under
[`evidence/variational-analytic-harmonic-water-50809803/`](evidence/variational-analytic-harmonic-water-50809803/README.md).
For one water geometry, its common state converged, cold/warm replay was exact,
all three stationary-envelope finite differences passed, the harmonic
continuum rotated at float64 roundoff, and the complete stationary scalar had
rotation errors `2.799424692057073e-9 eV` in energy and
`8.198584915195558e-8` relative in the coordinate gradient. Both clean
processes have identical scientific measurement SHA-256
`7ce9e9c07f40552ea513e0bbd4f29f4e4a5aa6887d7b5888c5647756525fe500`.
This is evidence for one changed model candidate, not checkpoint parity or a
global Tier-V theorem; every capability remains `no`.

The subsequent preregistered local-stability canary is retained under
[`evidence/variational-analytic-stability-water-604ecfa2/`](evidence/variational-analytic-stability-water-604ecfa2/README.md).
Five deterministic starts agree within the frozen root thresholds and the
independent residual factorization has relative error
`4.8093797227417603e-17`. However, the 23-dimensional model susceptibility has
5 significantly negative, 12 significantly positive, and 6 near-zero
eigenvalues. It therefore fails passivity and local invertibility; the
feedback-sign and combined-Hessian gates also fail. Both processes reproduce
measurement SHA-256
`56e7b23760fbd45aadccc9d1579d3d8647ccba9879a30db72f5fff49d9a5e1ff`.
This is negative Tier-V evidence; it does not affect the separate operational
conservative-PES target, and every capability remains `no`.

The separate original-source operational target now has positive but narrow
real-checkpoint evidence under
[`evidence/operational-analytic-harmonic-water-fa6f0200/`](evidence/operational-analytic-harmonic-water-fa6f0200/README.md).
For one water geometry, its exact operational scalar converges and replays,
all three implicit-force finite differences pass, and the second radial source
block remains exactly zero. The coefficient-space continuum has zero recorded
rotation-energy error and `1.8214596497756474e-17 eV/Angstrom` maximum
coordinate-gradient covariance error. The complete model/root/force chain has
`2.7694113668985665e-9 eV` rotation-energy error and
`7.857995561759406e-8` relative force-covariance error. Both clean processes
reproduce measurement SHA-256
`586024051151be3e3c63171e73bab34b6ce43459d4232c8070daadcd204b4547`.
This validates one canary, not a public PES or solvation model; every
capability remains `no`, and the legacy laboratory-grid route remains failed.

The preregistered methanol rigid-symmetry shard is retained under
[`evidence/operational-analytic-harmonic-rigid-methanol-93c98598/`](evidence/operational-analytic-harmonic-rigid-methanol-93c98598/README.md).
Across three proper rotations its maximum energy error is
`2.8617250791285187e-9 eV`, maximum relative force-covariance error is
`5.000913075374153e-8`, source-covariance error is
`2.592598170552474e-9`, and field-covariance error is
`4.269092690650618e-9`. Translation and identical-atom permutation gates also
pass, and an independent clean replay reproduces the scientific digest
`2bd5b02b6376faa39789cd9b081985f6805903a9ce227044842bf57af666ccfe`.
Relative to the failed CPCM1202 methanol record, the measured energy, force,
and source rotation errors are smaller by factors of approximately `1775`,
`3575`, and `653`. This comparison changes both model-evaluator and continuum
identities, so it supports the replacement route only. It is not evidence that
the old finite Lebedev/SWiG route was repaired, and every capability remains
`no` while the broader frozen panel is incomplete.

The next preregistered ethanol shard is retained under
[`evidence/operational-analytic-harmonic-rigid-ethanol-538f9f4d/`](evidence/operational-analytic-harmonic-rigid-ethanol-538f9f4d/README.md).
Its two clean processes reproduce scientific digest
`575420ae56c17f8dc37d28536ba86c9cba0f13559552f6afdc7934c4a2945937`.
Across three rotations its maximum energy, relative force, source, and field
covariance errors are `1.9072103896178305e-9 eV`,
`4.7353129839838976e-8`, `3.2020656091357397e-9`, and
`2.3729140073528977e-9`, respectively. This is the third equilibrium molecule,
not completion of the frozen 20-molecule or distorted-geometry panels; every
capability remains `no`.

The fourth preregistered equilibrium shard, acetone, is retained under
[`evidence/operational-analytic-harmonic-rigid-acetone-371a2b60/`](evidence/operational-analytic-harmonic-rigid-acetone-371a2b60/README.md).
Both clean runs reproduce scientific digest
`38efae6ff40ac225b6f1992bcfb00a62249ac695a607bbdb35a7915b3212d387`.
Its maximum rotation-energy, relative force, source, and field covariance
errors are `6.936716090422124e-9 eV`, `2.0250564002461316e-8`,
`6.218784064316717e-9`, and `3.212899614059117e-8`, respectively. All frozen
local gates pass, but the remaining sixteen molecules and every broader PES
gate remain unexecuted; every capability remains `no`.

The fifth equilibrium shard, acetonitrile, is retained under
[`evidence/operational-analytic-harmonic-rigid-acetonitrile-cf43e050/`](evidence/operational-analytic-harmonic-rigid-acetonitrile-cf43e050/README.md).
Its maximum rotation-energy, relative force, source, and field covariance
errors are `1.7384991224389523e-8 eV`, `1.7558870386345935e-8`,
`3.395228677265498e-9`, and `8.92400703128298e-9`; both clean runs reproduce
scientific digest
`2f01bb45ffa23f1dba09ee15123e71dbae0f742b607ff231758423487cfde616`.
All local gates pass, but this remains an incomplete equilibrium-only panel and
does not change any capability.

Water has also been rerun under this exact three-rotation shard contract, rather
than being inferred from the earlier single-rotation force-directional canary.
The two clean runs are retained under
[`evidence/operational-analytic-harmonic-rigid-water-bea47120/`](evidence/operational-analytic-harmonic-rigid-water-bea47120/README.md)
and reproduce scientific digest
`d8802924939ad40814eb81986aef4e76a76ba03fde88880597b5724b8f45480c`.
Consequently the contiguous frozen rigid-panel range `[0,5)` is now executed
twice: five of twenty equilibrium molecules, with fifteen still missing.

Benzene extends this range to `[0,6)`. Its two clean runs are retained under
[`evidence/operational-analytic-harmonic-rigid-benzene-97552baa/`](evidence/operational-analytic-harmonic-rigid-benzene-97552baa/README.md)
and reproduce scientific digest
`8f16ef863f5299eb3c7d4ace639c42d804058749740fe6713ced7477cff2f527`.
Its maximum rotation-energy and relative force-covariance errors are
`4.2611645767465234e-8 eV` and `5.356119997094117e-8`. Six of twenty
equilibrium molecules are now complete under this contract; fourteen remain,
and every public capability stays closed.

Methane extends the contiguous range to `[0,7)`. Its two clean runs are
retained under
[`evidence/operational-analytic-harmonic-rigid-methane-817501b5/`](evidence/operational-analytic-harmonic-rigid-methane-817501b5/README.md)
and reproduce scientific digest
`c0977ba027f2a94ef2e71d2db8c9384937c4df0ac9c2d48ea2c531bb77b382c6`.
Its maximum rotation-energy and relative force-covariance errors are
`3.5061020753346384e-10 eV` and `8.977094991729323e-9`. Seven of twenty
equilibrium molecules are complete; thirteen remain and no capability changes.

Trans-butane extends the contiguous range to `[0,8)`. Its two clean runs are
retained under
[`evidence/operational-analytic-harmonic-rigid-trans-butane-77abe42b/`](evidence/operational-analytic-harmonic-rigid-trans-butane-77abe42b/README.md)
and reproduce scientific digest
`44102bdffb58e49e95694fe2e7993f661b2178aa23136c92208f2f8b818a3734`.
Its maximum rotation-energy and relative force-covariance errors are
`2.6966517907567322e-9 eV` and `3.591746985795435e-8`. Eight of twenty
equilibrium molecules are complete; twelve remain and no capability changes.

Dimethyl ether extends the contiguous range to `[0,9)`. Its two clean runs are
retained under
[`evidence/operational-analytic-harmonic-rigid-dimethyl-ether-8a3aecce/`](evidence/operational-analytic-harmonic-rigid-dimethyl-ether-8a3aecce/README.md)
and reproduce scientific digest
`23f177cb0a70fbd513b68951231205e241d2a92eecb2b0457cee7a1804caad5d`.
Its maximum rotation-energy and relative force-covariance errors are
`2.2464519133791327e-9 eV` and `3.798391250201632e-8`. Nine of twenty
equilibrium molecules are complete; eleven remain and no capability changes.

Formic acid extends the contiguous range to `[0,10)`. Its two clean runs are
retained under
[`evidence/operational-analytic-harmonic-rigid-formic-acid-cd734f73/`](evidence/operational-analytic-harmonic-rigid-formic-acid-cd734f73/README.md)
and reproduce scientific digest
`5440c145c1e616d5ca180ed67d51662c1f47e3d430d6bfaa1e9585ecf6c266a5`.
Its maximum rotation-energy and relative force-covariance errors are
`1.4257238944992423e-8 eV` and `3.227108575652941e-8`. Ten of twenty
equilibrium molecules are complete; ten remain and no capability changes.

Acetic acid extends the contiguous range to `[0,11)`. Its two clean runs are
retained under
[`evidence/operational-analytic-harmonic-rigid-acetic-acid-c51d7154/`](evidence/operational-analytic-harmonic-rigid-acetic-acid-c51d7154/README.md)
and reproduce scientific digest
`efc0a12bf086eaac2cc6485379bc1843cf826614b3f1caee574c88d14f4153fa`.
Its maximum rotation-energy and relative force-covariance errors are
`1.2804775906261057e-8 eV` and `2.5678550041588242e-8`. Eleven of twenty
equilibrium molecules are complete; nine remain and no capability changes.

Acetaldehyde extends the contiguous range to `[0,12)`. Its two clean runs are
retained under
[`evidence/operational-analytic-harmonic-rigid-acetaldehyde-857af1b8/`](evidence/operational-analytic-harmonic-rigid-acetaldehyde-857af1b8/README.md)
and reproduce scientific digest
`76ae1cb0abbec4aa64ba9f492aed8c4a2bf4fe65f3af051aac78afd9897f45a6`.
Its maximum rotation-energy and relative force-covariance errors are
`5.841684469487518e-9 eV` and `2.376488400904626e-8`. Twelve of twenty
equilibrium molecules are complete; eight remain and no capability changes.

## Legacy baseline

At baseline `15777aad`, the old registry contains 20 experimental energy-only
profiles and one bounded experimental force profile. Those declarations do not
satisfy the new scalar/state/provenance and full-panel admission contract, so
their vNext tiers are all false. The exact per-profile snapshot is
`evidence/baseline-15777aad/capability-matrix.json`.

Separate paths remain fail-closed:

| path | vNext status |
| --- | --- |
| pyddx/ddX | same-scalar eight-channel diagnostic implemented; finite-grid rotation and achieved-residual gates remain failed/open |
| PCMSolver | independent energy/operator audit backend |
| source-dependent \(\rho\)-DROP | energy-only; coordinate VJP and Gates B/C missing |
| local-jet receiver | research diagnostic under a distinct identity |
| public MD | disabled until Tier M |

This file will be updated only when the corresponding evidence artifact is
source/model/runtime bound and passes every preregistered gate.

The callable water radial-GTO path is deliberately absent from public result
admission. At the current real-water canary it passed the three central
directional differences but missed rotation covariance (`2.515e-4` versus
`1e-4`) and torque (`1.078e-4 eV` versus `1e-4 eV`). It is therefore not a
conservative-force capability for MAPLE workflows.
