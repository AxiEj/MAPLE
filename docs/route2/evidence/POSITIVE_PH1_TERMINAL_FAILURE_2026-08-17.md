# Positive-parent PH1.0 terminal development result — 2026-08-17

## Frozen identity

The final smooth-CDS development candidate was frozen before target access:

```text
profile:
  maple-cds-w1-positive-parent-target-blind-3d-ph1-water-v1
source Git HEAD:
  617437ec5d92c578dce00572b99f53c33804a599
feature matrix SHA256:
  feda6475c5df7faf0e511c9e4af3b152dc5b05019f3289ecaa6fc781e473626b
preregistration file SHA256:
  73221560d3e2e6ee5ee73a0f34671ac65f4091903b3dfb5d7849e1ca1102b437
preregistration self SHA256:
  9379f00dd4acf8824fd5ec561120a06d3a14459539a735d1b406ee2a36c1b849
fold-subspaces SHA256:
  857c379fb5120398d79b93a9146b5dbf505891f91eef85690b636f4ea7f894ac
deployment-subspace SHA256:
  8bceffce1f478aad5c2855dcdeabb84a021ed9babc3255e3327eb40e0b2e8ab6
```

The external preregistration and result live under:

```text
~/.local/share/maple/route2/maple-cds-w1-positive-parent-ph1-v1/
```

The 148-record confirmation partition was not opened.

## First and only target-joined result

All ten grouped-OOF pseudo-Huber minimizers independently passed the frozen
two-start minimizer-ball certificate.  The maximum difference between the
physical 18-parameter representation and the three-coordinate optimizer
representation was `4.4408921e-15 kcal/mol`.

The result was nevertheless a development failure:

```text
water M0 electrostatics-only MAE:                    2.9793111 kcal/mol
water M1 stock-CDS MAE:                             2.0958073 kcal/mol
water positive-parent PH1 OOF endpoint MAE:         1.6969962 kcal/mol
water exact-minimizer MAE upper bound:               1.6969972 kcal/mol  FAIL
mixed-505 endpoint MAE:                              1.4546576 kcal/mol
mixed-505 exact-minimizer MAE upper bound:            1.4546582 kcal/mol  PASS
required prospective rule:                           both <= 1.5 kcal/mol
clustered water OOF q95 diagnostic:                  1.8984089 kcal/mol
```

Result binding:

```text
result file SHA256:
  4c1b89af7909700f256b18b700175cced2c546134d013e2818e1bdecb3b86be8
result self SHA256:
  425984414993fc8bc8c296a930ec9b397583bea0d7de375408080a6fece36545
status:
  development-fail
```

The water failure margin is `0.1969972 kcal/mol`, far above the frozen
`1e-6 kcal/mol` numerical prediction allowance.  It is a near miss in an
informal descriptive sense, but it is not a pass.

## Terminal boundary

No all-306 deployment fit was produced.  The following are now forbidden for
this lane:

- changing the `1.5 kcal/mol` gate;
- adding or changing CDS modes or features;
- changing the pseudo-Huber delta, folds, loss, intercept, or regularizer;
- opening confirmation to choose a replacement;
- fitting another CDS residual model.

The failure closes the CDS-capacity branch for the current electrostatic
profile.  It does not by itself prove that heterogeneous MDP-point plus
POLAR-Gaussian electrostatics is invalid.
