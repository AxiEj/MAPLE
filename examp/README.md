# Route 2 command-line examples

These are deliberately small **single-point acetone** examples. Run them from
this directory so `MOL2 acetone.mol2` resolves correctly:

```bash
cd /home/axie/MAPLE/MAPLE-implicitsolv-route2/examp
export PYTHONPATH="$(cd .. && pwd):${PYTHONPATH:-}"
maple 01_legacy_scf_ddpcm.inp 01_legacy_scf_ddpcm.out
maple 02_direct_pcm_scf_ddpcm.inp 02_direct_pcm_scf_ddpcm.out
```

The normal user-facing command is always `maple INPUT.inp OUTPUT.out`.
Activate the MAPLE environment that supplies `mace`, `pyddx==0.8.0`, and
`pyscf`. Setting `PYTHONPATH` first ensures that the command imports this
checkout rather than another installed MAPLE checkout.

| File | What it calls | Meaning |
| --- | --- | --- |
| `01_legacy_scf_ddpcm.inp` | pyddx / ddPCM, `response=scf` | A current, directly runnable legacy Route 2 mutual-polarization calculation. The matching `.out` and structured audit files were regenerated on this checkout. |
| `02_direct_pcm_scf_ddpcm.inp` | pyddx / ddPCM, `response=scf`, `pcm-half-coupling-v1` | The direct-PCM ledger: MACE-POLAR still supplies the self-consistent density source, but the reported electrostatic term is only `0.5*<c, f_reac>` from the same ddPCM operator. The field-conditioned MACE energy change is retained in the audit only and is not added to `Delta G_solv`. The matching output is a fresh acetone run on this checkout. |

Both are **legacy fixed-point Route 2 diagnostics**, not the proposed
V0-FrozenSource-KKT method and not a solution-phase force/PES interface. The
current checkout has no public end-to-end `#solv(...)` CLI route for
V0-FrozenSource-KKT yet, so neither example should be described as V0. The
direct-PCM ledger removes the ambiguous direct addition of
`E_MACE[V_reac]-E_MACE[gas]`; it **does not** make the MACE fixed point a
stationary common electronic free energy. An SCF residual only proves
convergence of that fixed-point equation; it does not establish reciprocal
response, conservative forces, or chemical-accuracy certification.

## Where to read the output

* `01_legacy_scf_ddpcm.out` — human-readable single-point output; it prints
  `Energy`, gas MLIP energy, `Delta G_solv`, and their combined value.
* `01_legacy_scf_ddpcm.out.implicit/route2-ddpcm-result.json` — actual SCF
  history and backend provenance from this example run.
* `01_legacy_scf_ddpcm.out.implicit/route2-public-result-ledger.json` —
  component ledger (`pcm_polarization`, `solute_polarization`, `cds`, and
  standard-state term). Derived totals must not be summed again.
* `02_direct_pcm_scf_ddpcm.out.implicit/route2-ddpcm-result.json` — the
  direct-PCM SCF history. Its `energy_residual_source` is
  `pcm-half-coupling-v1`, and the excluded MACE field-energy difference is
  written as `field_conditioned_mace_energy_change_hartree` for audit.
* `02_direct_pcm_scf_ddpcm.out.implicit/route2-public-result-ledger.json` —
  the public direct ledger. Its `solute_polarization` leaf is exactly zero;
  the reported electrostatic total equals `pcm_polarization`.

Each new calculation writes a sibling `OUTPUT.out.implicit/` directory. The
older exact-GTO/PCMSolver example materials are preserved under
`_historical_exact_gto/`; they are not a third supported invocation example.

`Energy:` is the value exposed to the single-point driver. The printed
`Solvation free-energy correction` and `Combined E_MLIP(gas)+Delta G_solv`
lines are the quantities relevant to this energy-only interface; they are not
a thermochemical Gibbs free energy or an experimental validation.
