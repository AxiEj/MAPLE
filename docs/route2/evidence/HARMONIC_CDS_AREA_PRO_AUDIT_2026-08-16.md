# Harmonic CDS area: verified Pro review and local decision

## Scope

This evidence freezes one mathematical choice only: whether the SMD-like
atomic area derived from the harmonic cavity field `e_i(R,u)` is proportional
to `integral e_i` or `integral e_i^2`. It does not admit a solvation-energy
profile, fit a coefficient, or establish chemical accuracy.

## Verified Chrome Pro invocation

The browser call followed the repository-authorized fail-closed workflow.

```text
advisor: ChatGPT in retained Windows Chrome
power_verified: Pro, 5 of 5.
verified_utc: 2026-08-16T12:41:39.6470908Z
submitted_utc: 2026-08-16T12:41:41.3786572Z
pro_thinking_observed: true
answer_complete_verified_utc: 2026-08-16T13:05:04.8368363Z
answer_start: Decision
answer_end: Final verdict
response_action_present: Copy response
generation_control_present_at_completion: false
prompt_sha256: 688d8baf073554dc959d3867f2f51939669faf37ece30cb4fd858b8295888275
submission_evidence_sha256: dddde5d6f7143af34757d2535aba885a2569b01f332db439e6b4ebc7578a7fdc
raw_uia_answer_sha256: c18409ec16028e810a788c96c2c9db308b6e5432aeaa26014700925ebda5f7c0
completion_evidence_sha256: 9e923b8070f1b8e4a134270b10bec32731647d79fe9588321acb8bf55ea1b2e4
```

The raw UIA capture remains a local audit artifact because it includes browser
chrome and unrelated sidebar labels. Completeness was checked independently:
the generation controls had disappeared, the answer contained its `Decision`
and `Final verdict` endpoints, and the stable response actions were present.

## Pro conclusion used as a challenge, not as authority

The answer distinguished two scientifically different parent objects:

```text
e = relaxed exposed-area fraction  ->  A = a^2 integral e dOmega
e = square-root exposure amplitude ->  A = a^2 integral e^2 dOmega
```

It also showed that the two occurrences of `e` in `E.T @ K @ E` arise from
the trial and test functions and do not, by themselves, define a local
`e^2 dS` geometric measure. For a centered smooth Heaviside interpreted as an
exposure fraction, squaring introduces a first-order inward area bias in the
transition belt. Both choices can be differentiable and SO(3)-invariant, so
those structural properties do not select the physical meaning.

## Independent local re-derivation

The repository implementation resolves the semantic ambiguity without using a
solvation benchmark:

1. `harmonic_exposure.smooth_flat_step()` maps signed overlap directly to a
   value in `[0,1]` with exposed-fraction semantics.
2. Pair values are projected and composed into the single coefficient object
   consumed by the harmonic electrostatic basis. There is no square-root map.
3. Installed PySCF 2.13.1 `pyscf.solvent.pcm.gen_surface()` independently uses
   `area = w * r_vdw**2 * swf`, linear in its SWIG switching function.
4. For the declared coefficient object `c_i`, orthonormality gives

   ```text
   A_i = a_i^2 sqrt(4*pi) c_i,00.
   ```

5. Its differential at fixed radii is

   ```text
   dA_i = a_i^2 sqrt(4*pi) dc_i,00.
   ```

   If radii later become geometry-dependent, both the explicit
   `2*a_i*da_i` term and every indirect radius contribution to `dc_i` must be
   included. The current candidate freezes radii.

The adopted contract is therefore
`maple.route2.cds-area.smooth-harmonic-exposure-integral.v1`. A quadratic norm
of the coefficients is not silently retained as an alternative area.

## Local verification

The implementation and tests establish:

- isolated-sphere `4*pi*a^2` and zero coordinate gradient;
- exact containment and a bounded, monotone near-tangency sweep;
- triple-overlap agreement with an independent unprojected smooth-product
  oracle across transition widths, harmonic orders, and radial orders;
- equality with the NumPy harmonic cavity coefficients;
- equality of the coefficient `l=0` contraction and an independent exact
  finite-band quadrature;
- strict rejection, without clipping, outside the positive-parent bounds;
- multistep central-difference convergence of the coordinate VJP;
- rigid translation, rotation, and atom-label permutation behavior;
- complete CDS derivatives containing both area and environment-tension terms;
- exact cavity-identity checks against the harmonic continuum;
- dependency-light import and immutable/content-addressed configuration.

Targeted result after the decision: `36 passed` (the final fresh count is
recorded in the commit verification). Chemical accuracy and any public
capability remain separate gates.

## References

- Marenich, Cramer, and Truhlar, SMD, J. Phys. Chem. B 2009,
  DOI `10.1021/jp810292n`.
- Lange and Herbert, smooth/SWIG PCM, J. Phys. Chem. Lett. 2010,
  DOI `10.1021/jz900282c`, including its supporting information.
