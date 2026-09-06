# Local decision after Pro 5/5 architecture review

## Accepted

1. The first transferable response model is the direct reduced-field quadratic
   scalar, not an inner hardness/source minimization:

   `E = E_vac + <s0,eta> - 1/2 ||B_theta(R) eta||^2`.

   It represents the identifiable reference and first-order response in the
   current one-amplitude dataset, permits exact zero-susceptibility modes, and
   avoids a local-hardness inverse that would recreate ordinary QEq's global
   equalization and size-scaling risks.
2. The response factor is variable-N and sparse: one radial-null scalar row per
   atom, two Cartesian vector rows per atom, and local antisymmetric edge rows.
   `maple/solvation/models/sparse_passive_factor.py` now implements this algebra;
   gauge invariance, rotation/edge-orientation invariance, Hessian symmetry,
   negative semidefiniteness, and the gauge zero mode pass.
3. Training remains scalar/observable-only. No per-molecule source coefficient,
   MDP atomic partition, old POLAR response, PCM quantity, or experimental
   solvation value is a target. Dense fit probes enter gradients; the rotated
   audit frame does not.
4. Frozen backbone weights do not imply detached coordinates. MDP and optional
   zero-field POLAR feature graphs must remain differentiable with respect to
   nuclear positions.

## Permanent-source amendment

After the main answer, one frozen sulfur validation record violated the radial8
reference-MEP gate (`6.91% > 5%`) while its response-MEP audit remained `1.27% <
3%`. The follow-up Pro review and local algebra agree on the minimum theoretical
repair: a P13/R8 split-order scalar with geometry-only permanent l=2 and
radial8-only passive response. A nonzero radial-l2 cross response is forbidden
when the l2 diagonal susceptibility is exactly zero, because PSD would be
violated.

The orthonormal STF point-quadrupole kernel and rotation representation are now
implemented in `maple/solvation/coupling/point_quadrupole.py`. On the opened
sulfur record, radial8 plus permanent point-l2 gives `1.64%` zero-field and
`1.15%` response audit error with a positive Hessian-extension Gram spectrum.

## Evidence-driven rejection retained

A separately preregistered, overdetermined 86-point fit/audit P13 point-source
gate subsequently failed: all 32 absolute audit errors were below `1.56%`, but
8/32 records exceeded the frozen audit/fit ratio of 2.0. This favorable absolute
error does not override the registered generalization failure. The point-P13
campaign is closed under that identity.

The next representation named before opening those results is fixed-width
Gaussian l=2 permanent source, still with exactly zero induced l=2 response.
Pilot Gaussian-l2 results improve the sulfur and bromine ratio but one HCNO case
remains above 2.0 under an unconstrained per-molecule fit. Therefore no Gaussian
P13 claim is made yet. The ratio gate belongs decisively to the eventual shared
transferable head; a new Gaussian model identity must be preregistered before
training, and no fitted quadrupole coefficients may be retained.

The completed batch added two further corrections. First, the full radial8
response gate has one bromine absolute failure (`0.00340 > 0.002 Ha/e`) despite
`2.48% < 3%` relative error; induced Gaussian l2 at 1.5 A reduces it to
`0.000837 Ha/e` and `0.630%`. The zero-induced-l2 P13/R8 candidate is therefore
also closed; the next passive factor is full P13, initially block diagonal.
Second, both allowed shared permanent heads failed without opening validation:
the 364-parameter MDP Q-A head gives `21.1%` mean and `39.2%` maximum audit MEP,
while the 398-parameter Q-B head with two target-free POLAR l2 carriers gives
`19.4%` mean and `32.9%` maximum. The permanent-head capacity sequence stops.
The operational next target is the demonstrated main error: replace the old
46%-error induced response with the new block-passive P13 response while
retaining the existing MDP point permanent source as the affine term of the
same scalar.

## Still open

- a preregistered block-passive P13 induced-response head;
- grouped train-CV, validation and blind electrostatic accuracy;
- reciprocal ddPCM spectral margin, root, force, rotation and final solvation
  gates.

No MAPLE capability or chemical-accuracy claim follows from this review.
