# ConSolv / AnySolv SP examples

Run from the MAPLE root with the authorized shared checkpoints installed:

```bash
python -m maple.main examples/pretrained_solvation/gas.inp
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 python -m maple.main examples/pretrained_solvation/anysolv_water.inp
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 python -m maple.main examples/pretrained_solvation/anysolv_toluene.inp
python -m maple.main examples/pretrained_solvation/consolv_ethanol.inp
python -m maple.main examples/pretrained_solvation/consolv_toluene.inp
```

The AnySolv examples name this installation's pinned source checkout; change
only its location if needed. ConSolv examples print gradients. AnySolv prints
only scalar gas/correction/combined energies. All energies use Hartree and
gradients use Hartree/angstrom. These are fixed-geometry SP mechanics examples.
See [domains, exact identities, and default paths](../../docs/pretrained-solvation-hub/CONSOLV_ANYSOLV.md).

AnySolv and ConSolv inputs now use CUDA/FP32 by default. `#device` can be omitted;
these examples spell it out. Optimized inference is explicit: add
`inference=optimized` inside AnySolv's `#solv(...)` or ConSolv's `#model(...)`.
It is signature-bound and can have a slow first compilation. See the linked
usage document for the hardware/evidence gate and FP32 force-audit boundary.
The standalone gas reference example keeps its own existing CPU setting.
