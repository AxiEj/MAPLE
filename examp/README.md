# Route 2 command-line examples

The top level contains exactly two **single-point acetone** examples.  They
share the same MACE-POLAR checkpoint, ddPCM profile, SMD-CDS term, and direct
PCM energy ledger; only the electronic response strategy changes.

Run them from this directory so `MOL2 acetone.mol2` resolves correctly:

```bash
cd /home/axie/MAPLE/MAPLE-implicitsolv-route2/examp
export PYTHONPATH="$(cd .. && pwd):${PYTHONPATH:-}"
maple 01_direct_pcm_frozen_ddpcm.inp 01_direct_pcm_frozen_ddpcm.out
maple 02_direct_pcm_scf_ddpcm.inp 02_direct_pcm_scf_ddpcm.out
```

Activate an environment that supplies `mace`, `pyddx==0.8.0`, and `pyscf`.
The normal user-facing command is `maple INPUT.inp OUTPUT.out`; setting
`PYTHONPATH` above ensures that it imports this checkout rather than another
installed MAPLE checkout.

| File | Electronic source | Meaning |
| --- | --- | --- |
| `01_direct_pcm_frozen_ddpcm.inp` | `response=frozen` | Evaluates MACE-POLAR once at zero external field, sends its point-`l<=1` source directly to ddPCM, and reports `0.5*<c0,Pc0> + SMD-CDS`.  There is no field-conditioned MACE call and no fixed point.  This is the current no-training energy baseline. |
| `02_direct_pcm_scf_ddpcm.inp` | `response=scf` | Feeds ddPCM's reaction field back through the MACE-POLAR local-jet interface until the unmixed learned fixed point converges, but reports the same direct PCM ledger.  This retains mutual polarization as a research diagnostic without adding the nonconjugate MACE field-energy change. |

For the parallel Route-2B equation, replace `ddpcm` by `ddcosmo` in the
versioned profile name and choose one of its registered solvents.  The two
`v2` profiles are a paired comparison that changes only the continuum
equation.  Route 2B is scaled finite-dielectric ddCOSMO plus the same frozen
PySCF SMD-CDS comparison term; it is not COSMO-RS or Route-1 ALPB.

Both profiles are **single-point energy-only**.  They do not expose Route-2
forces, optimization, scans, or MD, and neither proves a common stationary
electronic free energy.  In particular, convergence of `response=scf` proves
only the learned fixed-point equation, not Maxwell reciprocity or a
variational MACE--PCM functional.

## Where to read the output

For either prefix `NN_direct_pcm_*_ddpcm`:

* `PREFIX.out` is the human-readable single-point result.  `Energy` is the
  gas MLIP energy plus the solvation correction exposed to MAPLE.
* `PREFIX.out.implicit/route2-ddpcm-result.json` is the full scientific audit.
  It records `response_mode`, whether a fixed point applies, the exact energy
  ledger, provider identity, and the structured state archive.
* `PREFIX.out.implicit/route2-public-result-ledger.json` contains the public
  component ledger.  `solute_polarization` is exactly zero for the direct PCM
  ledger, and `electrostatic` equals `pcm_polarization`; derived totals must
  not be summed again.

The older field-conditioned-MACE-plus-PCM ledger is preserved under
`_historical_legacy_ledger/`, and the earlier exact-GTO materials remain under
`_historical_exact_gto/`.  They are provenance records, not additional
recommended invocation examples.
