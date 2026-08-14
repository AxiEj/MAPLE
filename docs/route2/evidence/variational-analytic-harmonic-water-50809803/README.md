# Analytic Gaussian multipole + harmonic common-scalar water canary

This bundle retains two clean-process executions of the disabled, separately
identified MACE-POLAR analytic Gaussian-multipole inference candidate coupled
to the smooth weighted harmonic-Galerkin continuum scalar.

## Bound identity

- execution commit: `508098034faaad28beceb5d7ee6243aa04bddca1`
- official checkpoint SHA-256:
  `fab8b8713c832f31a2a853aaa22fd638be8a369cbf5095e6b3e982a18d10e93a`
- model evaluator:
  `graph-longrange-analytic-gaussian-multipole-realspace-v1`
- scalar:
  `route2-variational-macepolar-analytic-gaussian-multipole-energygradient-smoothharmonicgalerkin-cpcm-v1`
- measurement SHA-256 (both runs):
  `7ce9e9c07f40552ea513e0bbd4f29f4e4a5aa6887d7b5888c5647756525fe500`
- primary JSON SHA-256:
  `c75ebdee104244937c74dc13f49640e5c393a0c742ab1515b9e0e9336f7e1328`
- replay JSON SHA-256:
  `2a72a853f0c0b07c510b2afe1cf3802182209c3974b7bd63d837be34207a74d7`

The two complete files differ only in execution metadata (output path, UTC time,
and wall-clock timings). The protocol, geometry, identities, roots, energies,
derivatives, rotation measurements, decisions, and `measurement_sha256` are
identical.

## Executed command

```bash
LD_LIBRARY_PATH=/home/axie/miniconda3/envs/maple/lib:${LD_LIBRARY_PATH:-} \
python tools/route2_release/run_variational_harmonic_water_canary.py \
  --checkpoint /home/axie/.cache/mace/MACEPOLAR1Mmodel \
  --device cuda \
  --model-evaluator-profile \
    graph-longrange-analytic-gaussian-multipole-realspace-v1 \
  --output /tmp/route2-variational-analytic-harmonic-50809803-run1.json
```

The replay used the same command with `run2.json` as the output.

## Narrow result

- center cold root: 31 Anderson iterations, actual unmixed reduced residual
  `1.720697910222882e-10`;
- warm replay: zero iterations, zero source difference, zero stationary-energy
  difference;
- stationary total scalar: `-2079.8808921464733 eV`;
- continuum scalar: `-0.01700524306563489 eV`;
- all three re-solved stationary-envelope central differences passed:

| step (Angstrom) | absolute error (eV/Angstrom) | relative error |
| ---: | ---: | ---: |
| `5e-4` | `1.5331551746211591e-7` | `1.4783485298357486e-6` |
| `2e-4` | `2.5758885538462728e-8` | `2.483806675042099e-7` |
| `1e-4` | `6.432123125788003e-9` | `6.202190048427979e-8` |

For one deterministic proper rotation:

- harmonic-continuum energy error: exactly `0.0 eV`;
- harmonic-continuum coordinate-gradient maximum error:
  `7.806255641895632e-18 eV/Angstrom`;
- complete stationary-scalar energy error: `2.799424692057073e-9 eV`;
- source relative covariance error: `7.89339570981596e-9`;
- field relative covariance error: `2.7812389718986214e-9`;
- total coordinate-gradient relative covariance error:
  `8.198584915195558e-8`;
- total coordinate-gradient maximum error:
  `1.9196180839342603e-8 eV/Angstrom`.

This is a positive implementation canary for one water geometry, one envelope
direction, and one rotation. It also demonstrates a large reduction of the
previous fixed-axis model-side rotation defect after changing the inference
operator. It is not a checkpoint-parity result and it does not establish
physical accuracy.

## Claim boundary

The analytic evaluator changes the checkpoint inference operator and the strict
candidate uses the complete eight-channel energy-gradient effective source.
The original four-channel density head is only a zero-field anchor and
diagnostic observable. This bundle does not prove global passivity, uniqueness,
combined-Hessian stability, all-geometry SO(3) covariance, physical source
quality, solvation accuracy, PES smoothness, Hessians, frequencies, or MD.

All public capability tiers remain closed:

```text
E = false
F = false
H = false
V = false
M = false
```
