# Hybrid harmonic analytic-force canary

Artifact ID:

```text
route2-mace-mdp-polar-hybrid-harmonic-analytic-force-canary-v1
```

## Decision

The default force for the already-admitted experimental harmonic operational
scalar may use the matrix-free implicit adjoint. The former fourth-order,
fully re-solved Richardson force remains an independent diagnostic oracle.

This is a force-implementation decision, not a chemical-accuracy admission.
It does not add a CDS/nonpolar term, finite-dielectric solvent transfer, H/V/M,
FREQ/TS/IRC/MD, or a strict MACE-continuum common functional.

## Independent mathematical audit

The prompt was submitted through the retained Windows Chrome ChatGPT session
after verifying both the composer label `Pro` and the menu label `Pro, 5 of 5`.
The complete answer was captured from the Windows UI Automation accessibility
tree because the browser copy action did not replace the clipboard.

The Pro answer found:

- the Euclidean adjoint formula correct as written;
- the factor of two and all residual/coordinate signs correct;
- the resulting derivative conservative for the registered operational scalar;
- no implication of Tier V or source-energy conjugacy; and
- a conditional `GO`, subject to root/adjoint/coordinate/loop fail-closed gates.

Raw prompt, mode proof, submit state, and answer are retained in this directory.

## Real-checkpoint checks

Official MACE-MDP and MACE-POLAR-1M checkpoints were evaluated in float64 on
CPU without consuming the four ongoing development GPUs.

| check | result |
| --- | ---: |
| water primal residual | `1.4484e-14 eV` |
| water adjoint residual / iterations | `9.4998e-11 eV` / `2` |
| water net force | `2.7756e-17 eV/A` |
| water all 9 components vs stored GPU Richardson | `9.5916e-9 eV/A` max abs |
| water directional Richardson-extrapolated mismatch | `6.417e-9 eV/A` |
| benzene adjoint residual / iterations | `1.9267e-12 eV` / `4` |
| benzene component vs stored GPU Richardson | `6.0357e-10 eV/A` |
| translation force relative error | `2.2427e-14` |
| rotation force relative error | `3.2848e-8` |
| midpoint closed-loop absolute work | `2.1360e-9 eV` |

The water directional calculation was rerun after the public-default
integration and reproduced the earlier force and residual values exactly.

See `manifest.json` for raw-file hashes, source hashes, and exact claim
boundaries.
