# Hybrid ddPCM nonuniform-work terminal audit

Two clean CUDA processes ran the target-independent command

```bash
PYTHONPATH="$PWD" MAPLE_ROUTE2_MACE_DEVICE=cuda \
python tools/route2_release/run_hybrid_ddx_nonuniform_work_audit.py \
  --polar-device cuda --output <external-json>
```

Both runs have measurement SHA-256
`d91edf0e5a7e29ec20708add487c70583d3ad811447e7ef735e2d5948aca8593`
and status
`pass-outcome-c-no-upstream-nonuniform-work-endpoint`.

## Decisive measurements

| quantity | value |
| --- | ---: |
| converged root residual | `4.3452669630021775e-13 eV` |
| nonuniform component of reaction field | `0.5799277131715507` relative |
| ddPCM polarization energy | `-0.26709875725093374 eV` |
| correct direct-sum energy-dual work | `-0.5341975145018621 eV` |
| `work - 2 G_cont` | `5.329070518200751e-15 eV` |
| naive `<embedded permanent + induced, model_field>` | `-0.11908391080137502 eV` |
| naive endpoint mismatch from correct work | `0.4151136037004871 eV` |
| `||model_field - energy_gradient||_2` | `1.2632168333582887` |
| raw conditioned energy difference | `-0.0075003197475780325 eV` |
| raw charging-line-integral error | `4.590008928495592e-13 eV` |
| endpoint `grad E dot u` minus raw energy difference | `3.3677853492443226e-4 eV` |

The installed upstream MACE source is SHA-256
`5ce5372251097f9d6fd17f69f6c63738a6d32278683586522fc70be4d9010e06`.
Its `forward` signature contains only the uniform three-vector
`external_field`; it constructs `external_potential` from that vector, adds
half to each spin channel, and adds the explicit uniform
`external_potential[:,1:] * total_dipole` work.  No official arbitrary
nonuniform explicit-work parameter or endpoint is present.

## Decision

1. The heterogeneous point-permanent plus Gaussian-induced ddPCM scalar is
   internally exact: its declared direct-sum energy cotangents satisfy
   `2 G_cont = c_p dot grad_p G + c_g dot grad_g G` to `5.4e-15 eV`.
2. The external-MEP field used to drive MACE-POLAR is a different operator.  A
   naive source/native-field endpoint pairing is numerically false by
   `0.415 eV` on the real water root and is not an admissible work correction.
3. The reaction field has a large component outside the upstream uniform-field
   subspace.  The prior `+E dot mu` audit therefore cannot be extrapolated to
   this state.
4. The same-graph raw energy remains a valid scalar; its charging line integral
   reproduces its endpoint energy.  One endpoint gradient contraction does not.
5. This selects Pro-review Outcome C.  `Phi0` and `Phi_raw` remain distinct
   operational scalar candidates.  Neither becomes a unique physical ledger or
   Tier V from this audit.

## Claim boundary

No experimental hydration/solvation target was read.  This evidence does not
choose a ledger by accuracy, certify MACE-MDP moments as a quantitative PCM
source, or admit public `E/F/H/V/M`.  It removes an invalid nonuniform-work
repair and leaves a finite next chemistry task: compare preregistered scalar
components against matched, target-independent electronic-distortion and
same-equation continuum references.

## Files

- [`run1.json`](run1.json), file SHA-256
  `3e18b100d7668ea0a6e13c47382de6f0396280efb149d1a4ee5e8e46f4e1b0e1`;
- [`run2.json`](run2.json), file SHA-256
  `b6ee4dcc13df022a6da7c3f5c551b0c530cee8df8de6dd388029fa85ec4a3692`.
