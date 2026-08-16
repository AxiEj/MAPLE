# MAPLE-CDS-W1 M3 terminal development failure — 2026-08-16

## Frozen identity

The target-blind M3 preregistration was generated from clean Git commit
`44ce23189fe0494876dafcee61a0d591e0a63773` and published read-only outside
the checkout.

```text
preregistration file SHA256:
0302635aa2186f326cf73ffccf284b0f0ba07aa7d51c4432c13cc03e80ba365b

preregistration self SHA256:
a79d551ac7b8040444a7a4168c3e872b43bcd84f2b481695c7a77fd371c26092
```

The artifact froze 202 exact element-count families, maximum family size 7,
fold loads `[31,31,31,31,31,31,30,30,30,30]`, an 18-rank geometry design,
and a rank-3 reduced design whose largest standardized fold condition number
was `4.65674171711236`.

## Terminal result

The first target-visible grouped-OOF run exited before producing any OOF
prediction or accuracy metric:

```text
M3FitError: Primary LAD coefficient 0 is numerically non-unique.
```

This is the preregistered Q8 failure condition. The coefficient-range tolerance,
LAD objective cap, folds, subspace, or solver were not relaxed after observing
the failure. M3 is therefore terminally rejected as an estimable pure-LAD
profile; it has no development MAE and no deployable coefficient.

The external terminal evidence is:

```text
failure JSON file SHA256:
0085d34463e73e9ae39314853a57e3ee515463e208706d4c942aa57f38a5f24d

failure JSON self SHA256:
e1ae0588b10b550ea38b68884dc64972a6c5871f24239a59ff78d7527b205570
```

The MNSol confirmation partition remained sealed. A later estimator must use a
new profile and preregistration, explicitly disclose that it was proposed after
this M3 failure, and must not be described as a rescue or result of M3.
