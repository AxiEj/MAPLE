# MDP/POLAR planar-rank obstruction

This target-free audit rejects the restricted arithmetic susceptibility
candidate on the general molecular domain.  It does **not** reject the full
MDP + MACE-POLAR hybrid, and it does not admit an alternative response model.

## Restricted candidate

Let

```text
B   = J_P U
C_P = C B
K   = B A G,   G U = I.
```

Every corrected uniform molecular response has the form

```text
C (J_P + K) U = C_P (I + A).
```

Consequently its rank cannot exceed `rank(C_P)`.  If `C_P` has rank two and
the target `-alpha_MDP` has rank three, no choice of `A` can satisfy the
closure.  A pseudoinverse cannot enlarge the range and is not an admissible
repair.

## Real-checkpoint result

The audit uses a fixed water geometry and an independently rotated/translated
copy.  It reads neither experimental nor QM/MBIS/solvation targets and performs
no fitting.

| case | singular values of `C_P` | rank `C_P` | rank `alpha_MDP` | minimum relative Frobenius residual |
| --- | --- | ---: | ---: | ---: |
| reference | `5.9202383e-2, 2.5244887e-2, 0` | 2 | 3 | `0.55974756` |
| rotated/translated | `5.9202383e-2, 2.5244887e-2, 1.51e-17` | 2 | 3 | `0.55974756` |

The right null vector aligns with the molecular-plane normal to numerical
precision in both cases.  The production constructor fails closed rather than
using a pseudoinverse.

## Decision

```text
reject-arithmetic-span-tangent-on-general-domain
```

The next zero-training question is whether leaving the original POLAR uniform
source span is physically justified by either:

1. the already content-addressed MACE-MDP atomwise polarizability partition; or
2. the source-bound free-atom density-translation tangent (V0-ADT).

Neither alternative may inherit accuracy, force, or variational admission from
the rejected candidate.

## Reproduction

```bash
python tools/route2_release/audit_mdp_polar_planar_rank.py \
  --device cuda \
  --output /tmp/mdp-polar-planar-rank.json
```

Evidence file SHA-256:

```text
08327f39e95ab8f3a77603fedd9ac87ae14c39b539a366d5d744205fad7914da  record.json
```
