# AnySolv-MoE mechanics audit (2026-10-23)

This is a CPU/reference, float64 mechanics audit of the authorized official
`model_moe.pt`. It calls the pinned upstream public API unchanged. It is not an
experimental-accuracy benchmark or an absolute-solvation-free-energy result.

## Identity

- source revision: `1778e1f18384e5f59de1475182765733a7401e39`
- checkpoint SHA256: `77b7d883ac1a01fc13d78b8cf5530e5709af679e6fbdec53cb086ae0a90bd251`
- inference: `default`, CPU, float64

## Result

- vacuum gate: exact zero energy and force
- water correction: `-0.3203827202246117 eV`
- toluene correction: `-0.1289282932072227 eV`
- rotation energy difference: `0 eV`
- rotation force covariance maximum: `0.0009850333356048319 eV/A`
- pole-orientation full-FD maximum: `0.00134793491373286 eV/A` at `h=0.001 A`
- rotated-orientation full-FD maximum: `1.92249158703639e-07 eV/A` at `h=0.001 A`

The stable pole-orientation FD discrepancy and nonzero rotation-force covariance
error reject force exposure. MAPLE may expose only the scalar geometry-level
correction for SP; OPT/scan/TS/IRC/FREQ/MD remain closed.

## Reproduce

```bash
PYTHONPATH=/home/axie/.cache/maple/anysolv:$PWD \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
/home/axie/miniconda3/envs/maple/bin/python \
  docs/pretrained-solvation-hub/benchmarks/anysolv-moe-mechanics-2026-10-23/probe.py \
  --source-root /home/axie/.cache/maple/anysolv/anysolv \
  --checkpoint maple/function/calculator/model/anysolv/model_moe.pt \
  --output docs/pretrained-solvation-hub/benchmarks/anysolv-moe-mechanics-2026-10-23/result.json
```

- `probe.py` SHA256: `69adb02b9cdd42e74a5a1c6ceb83fba1d07ba01569f93716599be803b2bbe0fc`
- `result.json` SHA256: `da7f96e1fd399098d1d16499cc2b134389d5cd8f50e8aa5332baa63a9f4fbc78`
