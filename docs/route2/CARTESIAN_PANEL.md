# Route 2 preregistered Cartesian force panel

This contract complements, but does not replace, the already executed
20-molecule directional panel. It is frozen before its first real-stack run
and cannot enable E/F/H/V/M by itself.

## Scope

Every one of the 20 frozen neutral-singlet molecules in
`fixedbox590_pes_panel_v1.json` contributes its exact `reference` geometry.
For every atom and Cartesian axis, the runner evaluates the same registered
scalar at positive and negative displacements of

```text
4e-4, 2e-4, and 1e-4 Angstrom.
```

This produces 465 Cartesian components and 1395 component/step comparisons.
The previously executed directional panel retains the compressed, stretched,
torsional, close-contact, and reaction-coordinate coverage; this panel adds
full component resolution without redefining those geometries or their scalar.

## Scalar and force

The only scalar remains

```text
E_op(R) = E_vac(R) + 0.5 <c*(R), P_R c*(R)>_Q
G_np = 0
```

and the analytic force is only the implicit-adjoint total derivative of that
same scalar. The independent aggregator recomputes every finite-difference
component from raw displaced energies and every analytic gradient from the raw
stored forces.

## Frozen gates

At each of the three step sizes:

- Cartesian RMS error `<= 5e-4 eV/Angstrom`;
- maximum component error `<= 2e-3 eV/Angstrom`.

The step sequence must also satisfy the versioned
`central-order-or-ten-percent-error-plateau-v1` rule. It passes only if the
first-to-last observed order is at least 1.5 with no terminal divergence, or
if all RMS and maximum errors are already below ten percent of their release
budgets and their RMS spread is at most 1.5. This prevents a systematic
first-order trend from being hidden by a loose absolute threshold while still
allowing an explicit numerical plateau.

Every displaced root must have normalized primal residual `<= 1e-12`; every
base adjoint true residual must be `<= 1e-10`; cold/warm source and energy
differences must be `<= 1e-8`; topology and ownership must remain fixed.

## Clean source-bound execution

One shard is run with:

```bash
env CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    PYTHONHASHSEED=0 \
    OMP_NUM_THREADS=10 MKL_NUM_THREADS=10 OPENBLAS_NUM_THREADS=10 \
    CUDA_VISIBLE_DEVICES=0 \
python tools/route2_release/run_fixedbox590_cartesian_panel.py \
    --device cuda \
    --molecule-start 0 --molecule-stop 1 \
    --output /absolute/path/cartesian-shard-00-01.json
```

Only `aggregate_fixedbox590_cartesian_panel.py`, run at the exact shard Git
head/tree, may combine all 20 source/checkpoint/runtime-identical shards. A
failed shard or aggregate exits nonzero and is preserved as negative evidence.

## Claim boundary

Before execution this document contains no numerical result. Even a passing
aggregate will establish only the component-resolved same-scalar derivative
gate for this frozen electrostatic candidate. It will not establish complete
solvation free energy, chemical accuracy, Hessian/FREQ/TS/HVP/NVE, a strict
common variational functional, original SMD equivalence, or public admission.
