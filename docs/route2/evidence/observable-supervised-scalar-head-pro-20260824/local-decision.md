# Local decision

Critically adopted:

1. The failed stockholder coefficient-label gate remains failed. Its small
   actual energy error cannot replace the rigorous reaction bound.
2. No coefficient, partition, MACE source, ADT, PCM energy, or experimental
   solvation target may enter the training objective.
3. The electronic scalar is a constrained convex dual:

       E(R,v) = E_vac^MDP(R)
                + min_{a^T z=Q}[A_theta(R,z)-eta_R(v)^T z]
                - min_{a^T z=Q} A_theta(R,z).

4. `A_theta` is strongly convex on the fixed-charge tangent space. This defines
   a unique internal coefficient gauge without claiming that coefficients are
   observables or using a molecular pseudoinverse.
5. Permanent source is `B z*(R,0)`; induced source is
   `B[z*(R,v)-z*(R,0)]`. Charge conservation, reciprocity, passivity, and
   concavity follow from the same scalar.
6. Frozen MACE-MDP and zero-field MACE-POLAR latent geometry features may enter
   the convex functional. Original MDP q/p, MACE-POLAR source, tangent, and ADT
   remain excluded from the physical source.
7. PCMSolver is a held-out transfer evaluator only. The training contract now
   explicitly admits nonuniform field energy and exterior-MEP response targets.
