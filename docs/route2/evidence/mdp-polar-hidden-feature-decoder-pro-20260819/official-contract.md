# Official output-contract facts frozen before Pro review

- MACE-POLAR shared config: `max_ell=3`, `atomic_multipoles_max_l=1`, source width `1.5 A`, `field_feature_max_l=1`, receiver widths `1.5/3.0 A`, and `quadrupole_feature_corrections=False`.
- Official calculator docs expose only `(n_atoms,4)` `density_coefficients` (q plus Cartesian dipoles) and warn atomwise q/p are not unique.
- MACE Foundations lists MACE-MDP as dipole/polarizability-only and not an energy/force model.
- Installed runtime: `mace-torch==0.3.16`, `graph-longrange==0.4.0`, `torch==2.12.0+cu130`.

Primary sources:
- https://raw.githubusercontent.com/ACEsuit/mace-foundations/main/mace_polar_1/config-mace-polar-1.yaml
- https://mace-docs.readthedocs.io/en/latest/guide/polar_mace.html
- https://github.com/ACEsuit/mace-foundations

## Relevant 2026 primary literature

- Cain, *Gauge Freedom and Metric Dependence in Neural Representation Spaces*,
  arXiv:2603.06774, proves the basic hidden-coordinate gauge: an invertible
  hidden transformation with an inverse downstream transformation leaves the
  network function unchanged. This supports treating an unbound hidden-channel
  readout as gauge dependent rather than physical.
  https://arxiv.org/abs/2603.06774
- Bogojeski et al., *Enhancing molecular dynamics with equivariant
  machine-learned densities*, arXiv:2604.24563, obtains physical density
  coefficients by explicitly training an SE(3)-equivariant density predictor in
  a declared atom-centred Gaussian basis. It is evidence for a separately
  supervised density-first successor, not for a zero-training hidden readout.
  https://arxiv.org/abs/2604.24563
