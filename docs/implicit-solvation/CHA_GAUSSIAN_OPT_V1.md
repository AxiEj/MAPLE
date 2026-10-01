# Gaussian-sign CHA optimization v1

## Identity and scope

`chagb-r6-pbsa-gaussian-sign-v1` is a separately versioned, opt-in research
scalar. It replaces only the legacy discontinuous CHA sign with

\[
g_\sigma(S_e)=\operatorname{erf}\!\left(\frac{S_e}{\sqrt{2}\sigma_e}\right),
\qquad \sigma_e>0.
\]

There is deliberately no default or calibrated value for `sigma_e`. The width
is an immutable constructor parameter in electrons. The implementation is
limited to CPU `torch.float64`, exactly-three-site geometries admitted by the
existing R6 point-domain certificate, and electrostatic size strictly below
9.5 A. A scale-aware float64 roundoff guard fails closed at mathematical
equality. Positive widths whose float64 reciprocal is nonfinite are also
rejected rather than permitting nonfinite automatic derivatives. The model
preserves the legacy hard 10 A size-shift formula, which is inactive inside
this narrower domain.

The complete solvent scalar is evaluated from live coordinates as

`R6 -> inverse Born -> Gaussian-CHA polar + cavity + dispersion`.

Forces are `-dG/dR` from that same complete Torch scalar. Finite differences
are validation-only and are never a runtime force path. Diagnostics expose the
weighted effective charges `S_e`, their dimensionless ratios `S_e/sigma_e`,
the erf factors, direct erfc/erfcx tails, Born radii, CHA factors, size, and the R6
point-domain certificate.

## Programmatic use only

Construct `GaussianChaCorrection` explicitly with an immutable
`ContinuumChaTopology`, its externally pinned SHA-256, a mandatory `sigma_e`,
and the quadrature order. Attach it explicitly to an already constructed
gas-phase calculator through `calculator.solvent_correction`. The correction
returns `SolvationResult` in Hartree and Hartree/A and reports truthful CPU
placement through `correction.provider.provenance`. Its structured result is a
model-predicted Gaussian-CHA solvation correction, not the independent
numerical reference, an explicit-cluster continuum quantity, or a
thermochemical Gibbs free energy. Provenance explicitly keeps production
admission, accuracy certification, and physical-accuracy claims false.

This tranche does **not** register a public provider or ordinary `.inp`
selector. Public input integration, arbitrary molecules, global smoothness,
physical-accuracy qualification, Hessian/FREQ/TS/MD admission, deployment, and
publication remain closed. The bounded real MAPLE LBFGS optimization matrix and
independent reference/finite-difference evidence are separate qualification
artifacts; the presence of this API alone does not establish them.
