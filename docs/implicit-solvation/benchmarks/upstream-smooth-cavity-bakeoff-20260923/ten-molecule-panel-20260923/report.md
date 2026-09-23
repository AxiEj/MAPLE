# Ten-species Torch v3 analytic-Hessian parity panel — 2026-09-23

## Question and claim boundary

Ten chemically different, neutral closed-shell solutes were attempted at
fixed ASE 3.27.0 `ase.build.molecule` geometries in water. These are geometry
fixtures from ASE's G2/extra database, **not** experimental solution data or
optimized solution structures. This panel tests the same frozen scalar's
Torch-v3 versus legacy energy/force implementation agreement and local
second-derivative consistency; it cannot measure physical/experimental
accuracy. No fitting, molecule replacement, parameter change, or threshold
change occurred after the preregistration in `run-1/protocol.json`.

The v3 Hessian is **analytic by Torch double autograd** of the total energy.
The independent finite difference below is only an *audit* of the legacy
implementation's force at displaced geometries. It never computes a v3
returned Hessian and is not a production fallback.

Source manifest `527b7aebf3623c6cc0e7ada61fa1aca508d29b2c772c2dd908f3fdffadfa6d99`,
checkpoint `fab8b8713c832f31a2a853aaa22fd638be8a369cbf5095e6b3e982a18d10e93a`,
single CPU device, water solvent, fixed geometries and charge/multiplicity
are frozen in `run-1/protocol.json`. All ten fresh-process raw results are
in `run-1/<name>/result.json`; source and wrapper hashes remained unchanged.

## Raw result synopsis

`ΔE` and `ΔF` are maximum absolute Torch-v3 versus legacy differences;
`H·v` is the Torch Hessian matrix versus a separate Torch directional
autograd HVP, **not** independent physical accuracy. The FD column is the
largest discrepancy between v3 `H·v` and a centered difference of legacy
forces at the preregistered `2e-5`, `1e-5`, `5e-6 Å` steps. The frozen gate
uses the latter two steps and requires each to be at most `1e-4 eV/Å²`, plus
their mutual stability; the energy and force limits are `1e-7 eV` and
`1e-6 eV/Å`.

| Solute (category) | ΔE (eV) | max ΔF (eV/Å) | H·v internal (eV/Å²) | legacy-force FD max (eV/Å²) | Frozen gate |
|---|---:|---:|---:|---:|---|
| H₂O (bent oxide) | 6.49e-9 | 8.26e-9 | 3.92e-10 | 9.33e-5 | pass |
| CH₄ (saturated hydrocarbon) | 4.10e-9 | 1.71e-9 | 1.59e-10 | 1.46e-5 | pass |
| NH₃ (nitrogen hydride) | 1.17e-8 | 2.69e-8 | 1.91e-10 | 2.75e-5 | pass |
| CO (heterodiatomic) | 2.70e-8 | 1.10e-9 | 2.05e-10 | 2.19e-4 | **FD gate fail** |
| CO₂ (linear oxide) | 2.40e-8 | 1.51e-9 | 1.31e-9 | 2.55e-4 | **FD gate fail** |
| HCN (nitrile) | — | — | — | — | **ddPCM topology fail-closed** |
| H₂CO (carbonyl) | 1.40e-8 | 3.29e-9 | 2.90e-10 | 9.42e-5 | pass |
| H₂O₂ (peroxide) | 6.40e-9 | 2.20e-9 | 8.83e-10 | 1.05e-4 | **FD gate fail** |
| C₂H₂ (alkyne) | 2.37e-8 | 8.65e-9 | 1.73e-10 | 3.67e-5 | pass |
| HF (hydrogen halide) | 3.67e-9 | 3.07e-10 | 8.69e-11 | 3.25e-5 | pass |

**Six of ten** pass the entire frozen audit. Nine of ten produce an analytic
Hessian; all nine pass E/F identity limits and `H·v` internal consistency.
Across those nine, maximum ΔE is `2.699152901e-8 eV`, maximum ΔF
`2.686196374e-8 eV/Å`, maximum internal `H·v` discrepancy
`1.310183961e-9 eV/Å²`, and maximum raw Hessian antisymmetry
`3.959996775e-9 eV/Å²`.

CO, CO₂, and H₂O₂ miss the preregistered independent legacy-force-difference
gate. In CO and H₂O₂ the finest step is worse than the coarser steps;
that behavior is consistent with differentiation-amplified reference noise,
but the current evidence does **not** prove that this is the only cause.
For CO₂ both fine steps exceed the gate. No threshold/step was changed to
relabel any of these as passing. HCN's original ASE geometry lies within
the ddPCM certified branch-change margin, so no Torch Hessian is returned;
the failure is not replaced by a nearby geometry.

`run-1/summary.json` SHA256:
`0a7c13ca0242c1e7c2b72a47bd3562cdde5f509a7b55bdb2e3fa5455ce11b312`.
The shell/panel exit code is 1 because four preregistered cases did not pass,
not because execution stopped before ten attempts.

## Unresolved boundaries

- The independent audit uses a **single deterministic direction** per
  molecule, not a full Cartesian reference Hessian. No independent analytic
  reference Hessian exists in the legacy route.
- The method is a dense N≤5 experimental reference, not a test of the five
  requested solvation datasets or broader chemical/atom-count coverage.
- Analytic differentiation removes coordinate-step truncation from the
  delivered Hessian, but not floating-point, solver, quadrature, topology,
  model, or physical-parameter error.
