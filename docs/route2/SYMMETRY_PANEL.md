# Route 2 preregistered symmetry and closed-loop panel

This document freezes the first all-panel rigid-symmetry and closed-loop gate
for the disabled fixed-box40/CPCM590 electrostatic profile. It does not enable
E/F/H/V/M and does not change the scalar.

## Frozen scope

Every one of the 20 reference geometries in
`fixedbox590_pes_panel_v1.json` uses the same operational scalar:

```text
E_op(R) = E_vac(R) + 0.5 <c*(R), P_R c*(R)>_Q
G_np = 0
```

For each molecule the clean-tree runner evaluates:

1. the base root, energy, force, source, and topology;
2. rigid translation by exactly `(1.7, -0.8, 0.5) Angstrom`;
3. three molecule-ID-bound proper rotations generated from seed `20260813`;
4. one deterministic swap of the first pair of identical atoms;
5. a closed rectangular loop in the molecule-specific seeded-internal and
   orthogonalized radial-internal coordinates, with amplitudes `(0.02, 0.02)
   Angstrom` and four equal subintervals per edge;
6. the loop in cold forward/reverse and sequential-warm forward/reverse modes.

The contract ID is `route2-fixedbox590-symmetry-panel-contract-v1`. Rotation
matrices, permutation, coordinates, force/source arrays, roots, work values,
topology hashes, and warnings are retained as raw measurements. A separate
aggregator recomputes the gates on the exact source tree.

## Frozen gates

- rigid translation and every rotation/permutation energy change `<= 1e-6 eV`;
- base and translated net force `<= 1e-5 eV/Angstrom`;
- translation force difference norm `<= 1e-5 eV/Angstrom`;
- base torque `<= 1e-4 eV`;
- rotation/permutation force covariance relative error `<= 1e-4`;
- rotation/permutation source covariance relative error `<= 1e-4`;
- primal residual `<= 1e-12` and adjoint residual `<= 1e-10`;
- identical topology within each molecule;
- every loop traversal obeys
  `abs(W) <= max(1e-5 eV, 1e-3 sum(abs(F dot dR)))`;
- cold forward/reverse work antisymmetry `<= 1e-10 eV`;
- warm forward/reverse work antisymmetry `<= 1e-8 eV`;
- cold/warm roots and warm forward/reverse repeats meet the frozen `1e-8`
  source/energy equivalence gates.

Any missing molecule, changed deterministic transform, nonconverged root,
topology change, or failed scalar/force gate is retained as a failure. No
threshold may be changed after observing the real panel.

## Execution

After this contract is committed on a clean tree:

```bash
python tools/route2_release/run_fixedbox590_symmetry_panel.py \
  --device cuda --molecule-start 0 --molecule-stop 1 \
  --output /absolute/path/symmetry-shard-00-01.json

python tools/route2_release/aggregate_fixedbox590_symmetry_panel.py \
  --shard /absolute/path/symmetry-shard-00-01.json \
  ... \
  --output /absolute/path/symmetry-aggregate.json
```

This gate covers conservative-PES symmetry/path behavior, not matched
QM/C-PCM component physics, complete solvation free energy, multi-geometry box
convergence, workflow integration, Hessian/FREQ/TS/HVP/NVE, or MD.
