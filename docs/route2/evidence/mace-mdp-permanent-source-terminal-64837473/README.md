# MACE-MDP unchanged permanent-source terminal audit

## Frozen candidate

- Execution commit: `64837473053d39bfae5b61788d7a4edce5d4502b`
- Checkpoint: `/home/axie/.cache/mace/MACE-MDP.model`
- Checkpoint SHA-256: `126f8d1602549e6fa0df775c701a5119ddeb0e3738202af8e7aa736de6c2b692`
- Candidate profile: `mace-mdp-route2-independent-variational-polarization-candidate-v1`
- Permanent source: the checkpoint's unchanged atom-resolved charges and
  Cartesian dipoles
- Surface map: exterior point multipoles through `l <= 1`; no Gaussian width,
  radial mixture, charge/dipole rescaling, clipping, or fit
- Reference: the same four frozen total-QM surface MEPs, intrinsic PCMSolver
  cavities, dielectric convention, and `1 kcal/mol` per-case fixed-source
  budget used by the earlier source audit
- Preregistration:
  [`../../preregistrations/mace-mdp-permanent-source-pcmsolver-four-v1.json`](../../preregistrations/mace-mdp-permanent-source-pcmsolver-four-v1.json)

The earlier MACE-MDP evidence admits only frozen response coefficients,
atom-resolved moment partitions, and their coordinate derivatives as research
coefficients. MACE-MDP is not an energy model, and this audit does not create a
common scalar or a solvation-energy ledger.

## Two independent cold executions

Both runs loaded the checkpoint independently on CPU and reproduced the same
scientific measurement SHA-256:

```text
9c3801fb87c5bc7e3fe1396bf31afad41cc397288fe6b9e6f2824ebf6910664c
```

The complete JSON, stdout, and stderr records are retained as `run1.*` and
`run2.*`. The different whole-file JSON hashes are caused by expected runtime
metadata and output-path differences; the scientific measurement payload is
identical.

## Frozen four-case result

| compound | QM fixed-source PCM (kcal/mol) | MACE-MDP source (kcal/mol) | absolute error (kcal/mol) | area-weighted surface-MEP relative error | case gate |
| --- | ---: | ---: | ---: | ---: | :---: |
| acetic acid | -10.9834322111 | -10.2193759695 | 0.7640562416 | 0.2563584249 | pass |
| benzene | -3.1363028647 | -4.0309361648 | 0.8946333001 | 0.5947660375 | pass |
| 2-acetoxyethyl acetate | -12.7473572143 | -10.9234945816 | **1.8238626327** | 0.4052114585 | **fail** |
| acetone | -6.6433389624 | -6.3582569053 | 0.2850820571 | 0.3540637979 | pass |

Aggregate diagnostics:

- case pass count: `3/4`;
- mean fixed-source energy absolute error: `0.9419085579 kcal/mol`;
- maximum fixed-source energy absolute error: `1.8238626327 kcal/mol`;
- maximum dipole relative error: `0.0361038680`;
- maximum area-weighted surface-MEP relative error: `0.5947660375`.

The preregistered rule requires every case to meet the unchanged
`1 kcal/mol` budget. Therefore `3/4` is a failure; neither the mean error nor
the comparatively good dipoles may override the per-case gate.

## Terminal decision

The unchanged MACE-MDP permanent charge/dipole source is rejected for
quantitative PCM. This exact untrained independent-polarization candidate is
closed:

- no broader static-MEP panel is authorized;
- no independent variational-polarization KKT construction is authorized;
- no radial fit, charge/dipole rescaling, or solvation-energy tuning is
  authorized;
- no `A/B/L` force, ledger, or public-admission work proceeds on this identity;
- `E/F/H/V/M` remain false.

The positive MACE-MDP polarizability evidence remains useful only as a research
coefficient or possible distillation target. A successor requires a newly
trained, content-addressed scalar-first permanent-source and passive
polarization head, or another independently source-validated variational
model. It must receive a new profile and rerun all source, scalar, root, force,
and physical-component gates.
