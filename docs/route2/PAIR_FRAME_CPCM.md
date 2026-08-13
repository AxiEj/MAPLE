# Disabled ordered-pair-frame C-PCM candidate

## Purpose and truth boundary

Laboratory-fixed Lebedev quadratures left measurable rotation anisotropy in the
fixed-box590 and fixed-box1202 candidates. Increasing a single grid was not a
general remedy. This candidate instead averages one fixed-topology
amplitude-SWIG C-PCM map over molecule-derived body frames. It retains the
canonical Route-2 scalar

```text
E_op(R) = E_vac(R) + 0.5 <c*(R), P_R c*(R)>_Q
G_np = 0
```

and differentiates that exact scalar. It is **not** conventional single-cavity
C-PCM and is not SMD or a complete solvation free energy.

## Frame construction

For every ordered atom pair, the first axis follows the pair,
the third follows its cross product with the nuclear-charge centroid offset,
and the second completes a right-handed frame. All ordered frames enter with
smooth weight `|d cross v|^8`. The normalized weight derivative, centroid
motion, frame derivative, source/field rotations, amplitude-SWIG surface,
Gaussian source/receiver kernels, and C-PCM operator are all included in
`coordinate_vjp()`.

The implementation keeps all `N*(N-1)` ordered members and 110 candidate nodes
per atom in each member. Exactly singular members use the smooth zero-weight
extension: their arbitrary bounded fallback orientation cannot affect the map
or its first derivative because both the weight and its derivative vanish.
Thus neither members nor surface nodes are inserted or deleted. The geometry,
fixed member-topology hash, numerical activity hash, source, field, and scalar
are bound into the immutable state. Fully collinear molecules fail closed
because the normalized ensemble has zero total weight.

## Current evidence

The production tests cover half coupling, source JVP/VJP transpose and finite
difference, coordinate VJP finite difference, translation, proper rotations,
atom permutation, immutable state/provenance, and a collinear negative case.
In a preliminary dirty-tree engineering run using the real fixed-box40
MACE-POLAR checkpoint, the methanol canary measured:

- maximum rotation energy error: `4.6625382311e-7 eV`;
- maximum force covariance relative error: `5.7147134759e-6`;
- maximum source covariance relative error: `8.1428852560e-7`;
- directional same-scalar error at `h=1e-4 A`: `1.1084336815e-7 eV/A`;
- base torque residual: `5.7263511161e-7 eV`;
- primal/adjoint residual maxima: `3.117e-13` / `2.165e-15`.

These numbers are not source-bound release evidence. They pass the local
preregistered numerical thresholds, but do not replace
the frozen 20-molecule PES, Cartesian, symmetry/loop, component-accuracy,
Hessian/FREQ/TS, NVE, performance, and clean-tree evidence gates. The profile
therefore remains `E=F=H=V=M=false` and cannot enter public MAPLE workflows.
