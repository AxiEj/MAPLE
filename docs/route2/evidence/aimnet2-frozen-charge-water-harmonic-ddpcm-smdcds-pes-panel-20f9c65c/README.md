# AIMNet2 frozen-charge water harmonic-ddPCM plus SMD-CDS PES panel

This directory retains two clean-tree executions of commit
`20f9c65c3fc238089bb4ea47512e8c136664bcf3`. Both use the exact
SHA256-bound AIMNet2 checkpoint, one field-independent NQE charge evaluation
per geometry, the water-bound smooth harmonic ddPCM scalar, and the PySCF
2.13.1 SMD-CDS scalar. No continuum field is supplied to AIMNet2 and no
electronic SCF or fixed-point iteration is introduced.

The primary and replay aggregates have different file hashes because runtime
and path metadata are retained, but reproduce the exact scientific aggregate
measurement SHA256
`61512993a63cc7bf04d4a3d15a288c806eabc96be104e33373b802f88f62322e`.
Their independently reduced `panel_summary` payloads are identical.

## Result

- 17 H/C/N/O molecules, 51 geometries, 153 directions, and 459 directional
  finite-difference samples per execution;
- 12/17 molecules pass every registered diagnostic gate;
- the maximum directional absolute error among those 12 passing molecules is
  `5.552903772709783e-05 eV/angstrom`;
- methanol, methane, dimethyl ether, and acetic acid fail the preregistered
  `0.02 angstrom` continuum point/source event margin;
- ethylamine fails the independent `0.02 angstrom` sphere-tangency margin;
- the full-panel minimum continuum-event and sphere-tangency margins are
  `0.002235739521015301` and `0.011178247440158717 angstrom`;
- the full-panel maximum directional absolute error is
  `0.23707138702659591 eV/angstrom`, reached by already event-rejected
  methanol;
- every E/F/H/V/M capability and OPT/FREQ/TS/IRC/MD admission remains false.

This is negative admission evidence. The event threshold, cavity radii,
dielectric, molecule list, or SMD-CDS parameters were not changed to convert
the five failures into passes. The smooth harmonic electrostatic scalar is
also not the full-resolution pyddx scalar used by the separate 653-record
MNSol accuracy matrix, so this panel cannot inherit that matrix's accuracy.
