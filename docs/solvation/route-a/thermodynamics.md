# Route A thermodynamic specification (Phase 0 rebuild)

## 1) Units and standards

- Energies: kcal/mol
- Distances: Å
- Temperature: K
- Temperature is frozen to 298.15 K.
- `R = 1.987204258640831e-3 kcal/(mol*K)`.
- Standard-state reference volume is `V0 = 1660.5390671738467 Å^3` **per molecule**.
- Molecular number-density constant is
  `C0 = 1/V0 = 0.000602214076 Å^-3`, corresponding to `1 mol/L`.
- Bulk water concentration is `55.345 mol/L`.
- `rho_W_number_per_A3 = water_bulk_concentration_molar * C0 / solution_reference_concentration_molar = 0.03332953803622 Å^-3`.
- `rho_W_over_C0 = rho_W_number_per_A3 / C0 = water_bulk_concentration_molar / solution_reference_concentration_molar = 55.345`.

Only the dimensionless `rho_W_over_C0` is allowed inside a logarithm.

## 2) Shell, measure, and restraint mathematics

### Occupancy indicator

For a water oxygen site `O_j` and all solute atoms `i`:

`d_s(O_j) = min_i (||r_Oj - r_i|| - R_i^vdW)`.

A water contributes only if its oxygen satisfies

`n(q, λ_s) = Σ_j 1[d_s(O_j) <= λ_s]`.

Waters are retained as complete molecules once their oxygen passes the criterion.

### Signed distance and effective volume

For a point `x` and solute atom `i`:

`s_i(x) = ||x - r_i|| - R_i^vdW`.

For insertion edge `i`, the flat-bottom soft-restraint volume is the
decoupled-reference-ensemble average

`V_eff,i = < ∫ exp[-β r_i(q_X,r_O)] d^3r_O >_{pi_base,i-1}`,

where

`pi_base,i-1(q_XW_(i-1)) ∝ exp{-β[U_OMOL_vac(XW_(i-1))+R_(i-1)]}`

on the nonperiodic solute-center-of-mass-reduced measure and
`R_i=R_(i-1)+r_i`. The per-base-frame translational integral is averaged
**before** applying `-RT ln(C0 V_eff,i)`, with block bootstrap over base frames.
It is **not** replaced by a hard-indicator volume, an integral at one frozen
solute geometry, or an interacting-endpoint average when the harmonic exterior
is part of the target restraint.

The finite auxiliary overlap core on the `D -> R` leg is

`U_rep(r)=epsilon exp[-(r/sigma)^2]`,

with force `F_i=(2 U_rep/sigma^2)(r_i-r_j)`. It has a finite positive
coincident barrier and a separating near-contact force. Its insertion and
removal belong to `DeltaG_alch_labeled`; it is not a restraint-release term.

### Frozen nonperiodic measure

The cluster measure removes only global translation by the solute center of
mass. Solute rotation, water rotation, relative translation, and all internal
coordinates remain integrated. The one-water reference removes only its
center-of-mass translation.

The restraint Hamiltonians are:

- `R_0(q)=0`;
- `R_n(q)=Σ_{j=1..n} 0.5*k*max(d_s(O_j)-λ_s,0)^2` for `n` labeled complete
  waters;
- `R_W(q)=0` on the water center-of-mass-reduced measure.

Here `k=2*buffer_height_kBT*kBT/buffer_width_A^2`. Version 1 introduces no
additional placement or orientation restraint. Therefore

`ΔG_release(i)=F_target_i_interacting-F_biased_i_interacting=0`

for every insertion edge. The zero term stays in the ledger so a future,
separately versioned restraint contract cannot silently alter the cycle.

## 3) QCT row and thermodynamic corrections

For insertion edge `i`:

`G_assoc,i^labeled = ΔG_alch_labeled(i) + ΔG_vol(i) + RT ln(i) + ΔG_release(i)`.

Row `n` free energy is

`G_n = Σ_{i=1..n}[ΔG_alch_labeled(i)+ΔG_vol(i)+RT ln(i)+ΔG_release(i)] - n RT ln(ρ_W_over_C0) + ΔG_outer(XW_n) - nΔG_outer(W)`,

with `n=0` as bare-solute outer free energy `G_0 = ΔG_outer(X)`.

The density/volume identity is

`ΔG_vol(i) - RT ln(rho_W_over_C0) = -RT ln(rho_W_number_per_A3 * V_eff,i)`

for each edge, and the corresponding sum is

`Σ_i ΔG_vol(i) - n RT ln(rho_W_over_C0) = -RT Σ_i ln(rho_W_number_per_A3 * V_eff,i)`.

Hydration free energy is

`ΔG_hyd = -RT ln(Σ_n exp(-βG_n))`,

with the sum over included rows only.

## 4) Outer correction and no-double-counting contract

The whole supermolecule `XW_n` is one atom list. By construction,

`ΔG_outer(XW_n)=F[U_OMOL_vac+δ_outer,R_n]-F[U_OMOL_vac,R_n]`,

`δ_outer_kcal(q)=Hartree_to_kcal_mol*((E_intrinsic_Polar[V_reac](q)-E_intrinsic_Polar[0](q))/Hartree_eV+E_PCM_pol_Ha(q)+G_CDS_Ha(q))`,

`E_PCM_pol_Ha(q)=0.5*dot(MEP_Ha_per_e(q),ASC_e(q))`,

and

`ΔG_outer(W)=F[U_OMOL_vac+δ_outer,R_W]-F[U_OMOL_vac,R_W]`.

`U_OMOL_vac` is counted once. `δ_outer` is an additive correction counted once.
It is forbidden to add a complete Route 2 solvated total energy, a second PCM
polarization term, a second CDS term, or another half-coupling term.

The Polar model's required `(N,4)` `density_coefficients` are learned
atom-centered coefficients used to evaluate a coarse point-multipole MEP. They
are not claimed to be a quantum-mechanical electron density, and a fixed-charge
fallback is forbidden.

Each vacuum/outer endpoint pair must have identical coordinates,
`atom_list_hash`, `restraint_hash`, `measure_id`, and boundary conditions.
MACE-OFF24 periodic frames are discovery/proposal data only: they are not
production reduced-potential rows and cannot be directly reweighted into the
nonperiodic OMOL measure.

## 5) Tail, plateau, and reference-cycle gates

- Tail omission uses a geometric-exponential conservative ceiling. With
  `w_n=exp(-beta G_n)` and a preregistered deterministic ratio ceiling
  `q_u=0.55` respected by at least two consecutive computed high-`n` ratios, the
  unseen weight after the last computed state `N` is bounded by
  `w_N q_u/(1-q_u)`. The ceiling is an explicit modeling assumption, **not** a
  statistical 95% confidence bound. The model, ceiling, minimum ratio count,
  ratio floor, and extension policy are frozen before production high-`n`
  results in `protocol-v1.json::tail_envelope_contract`; closure evidence must
  carry its matching SHA.
- The gate applies to the resulting **upper bounds**:
  `P_tail,upper < 0.005` and
  `ln[(Z_included+W_tail,upper)/Z_included]/beta < 0.1 kcal/mol`.
  Equality, an unresolved `q_u>=1`, insufficient consecutive ratios, a missing
  occupancy index, or a revealed ratio above `q_u` fails closed and extends
  the computed occupancy support.
- Occupancy plateau spread: `<= 0.5 kcal/mol` with non-empty common CI overlap.

Final scientific status also requires a hash-bound, same-holdout ablation of
`sequential-monomer-water-reference-v1` against
`solvent-cluster-reference-v1`. Both cycles must use the same `n` values and
every non-reference control. The frozen holdout is sensitivity evidence only
and may not reselect the reference cycle post hoc.

## 6) Cavity and shell geometry gate

For every cross-fragment overlapping protected-sphere pair `(i,j)` satisfying
`R_i+R_j-d_ij>0`:

- `s_i(t)=||t-r_i||-R_i`;
- `s_j(t)=||t-r_j||-R_j`;
- `p_ij(t)=min(-s_i(t),-s_j(t))`.

Fail when `max_{t,(i,j)} p_ij(t) > 1e-6 Å`.

For the water-dimer fixture the exhaustive overlap pairs are
`[(0,3), (1,3), (1,5)]`. The synthetic point `[1.4,0,0]` must fail
deterministically while the real mesh passes.

## 7) Scientific claim boundary

The current stack is **OMOL + PolarMACE-SMD + QCT-hybrid**. It is not an
externally validated production method yet. Engineering execution, cavity
stability, and single-molecule canaries do not establish Route A accuracy or
superiority over Route 2.
