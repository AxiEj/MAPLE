# Verdict

**The CPKS acceleration is mathematically valid and does not change the scientific target**, provided the intended target is the strict (q\to0) static derivative of the same frozen discretized (\omega)B97M‑V/def2‑TZVPD DFRKS problem.

As written, however, the implementation contract is not yet fail-closed. It must additionally freeze the resolved density-fitting configuration, both numerical grids, the zero-density-pruning policy, and checkpoint/Fock consistency; explicitly validate the solved CPKS residual; and pass the existing two-step finite-field oracle. The checkpoint tuple ((C,\mathrm{occ},\epsilon,E_0)) alone is necessary but not sufficient to identify the discrete response problem.

---

# A. Exact static closed-shell CPKS equations

## A.1 Ground-state conventions

Let the fixed AO basis have overlap matrix (S), and let the spin-summed RKS density matrix be

[
D(q)=C(q),n,C(q)^\dagger
=2C_o(q)C_o(q)^\dagger ,
]

where (C_o) contains the occupied spatial orbitals and every occupied orbital has occupation 2.

For point-charge mode (k),

[
F[D(q);q]
=========

h_0+q h'*k+V*{\mathrm{eff}}[D(q)],
]

with

[
h'_k=-J_k,\qquad
J_k\equiv \operatorname{int1e_rinv}(\mathbf s_k).
]

At (q=0), the checkpoint orbitals must satisfy the canonical generalized eigenvalue equation

[
F_0 C = S C \epsilon,
\qquad
C^\dagger S C=I.
]

Define the zero-field linear response kernel

[
\mathcal K[D^{(1)}]
\equiv
\left.
\frac{\partial V_{\mathrm{eff}}[D]}{\partial D}
\right|_{D_0}
[D^{(1)}].
]

For (\omega)B97M‑V this contains, consistently with PySCF’s discretized energy:

[
\mathcal K[D^{(1)}]
===================

J[D^{(1)}]
+
f_{\mathrm{xc}}^{\mathrm{semi}}[D^{(1)}]
+
f_{\mathrm{NLC}}[D^{(1)}]
-\frac12
\left[
\mathrm{hyb},K[D^{(1)}]
+
(\alpha-\mathrm{hyb})K_{\mathrm{LR},\omega}[D^{(1)}]
\right],
]

with the special coefficient branches handled internally by PySCF.

## A.2 Definition and sign of the orbital-response amplitudes

Use the fixed-overlap gauge

[
C_i(q)
======

C_i
+
q\sum_a C_a U^{(k)}_{ai}
+
O(q^2),
]

so that

[
U^{(k)}_{ai}
============

C_a^\dagger S
\left.\frac{\partial C_i}{\partial q_k}\right|_{0}.
]

Here:

* (i) indexes occupied orbitals;
* (a) indexes virtual orbitals;
* (U^{(k)}) has shape ((n_{\mathrm{vir}},n_{\mathrm{occ}}));
* all four modes together have shape
  [
  (4,n_{\mathrm{vir}},n_{\mathrm{occ}}).
  ]

The perturbation block is

[
h^{(k)}_{ai}
============

\left(C_v^\dagger h'*k C_o\right)*{ai},
]

with the same shape as (U^{(k)}).

**Do not negate (h_{vo}) before passing it to `cphf.solve`.** Since (h'_k=-J_k), the physical minus sign is already inside (h'_k). PySCF’s solver itself constructs the uncoupled response

[
U^{(0)}_{ai}
============

-\frac{h_{ai}}{\epsilon_a-\epsilon_i}.
]

Differentiating (FC_i=SC_i\epsilon_i), using (S'=0), and projecting with (C_a^\dagger) gives

[
(\epsilon_a-\epsilon_i)U^{(k)}*{ai}
+
h^{(k)}*{ai}
+
v^{(k)}_{ai}
=0,
]

where

[
v^{(k)}_{ai}
============

\left[
C_v^\dagger
\mathcal K[D^{(1)}*k]
C_o
\right]*{ai}.
]

Equivalently,

[
\boxed{
\left(\Delta+\mathcal K_{vo}\right)U^{(k)}
==========================================

-h^{(k)}_{vo}
}
]

with

[
\Delta_{ai}=\epsilon_a-\epsilon_i>0
]

for an ordinary gapped stable state.

PySCF 2.13.1’s `solve_nos1` explicitly expects `h1` with shape `(..., nvir, nocc)`, forms `-h1/(eps_a-eps_i)`, and solves ((I+\Delta^{-1}\mathcal K)U=-\Delta^{-1}h). 

## A.3 Every closed-shell factor of two

The zero-order spin-summed density is

[
D_0=2C_oC_o^\dagger.
]

Its derivative is

[
\boxed{
D^{(1)}_k
=========

2\left(
C_vU^{(k)}C_o^\dagger
+
C_oU^{(k)\dagger}C_v^\dagger
\right)
}
]

for a fixed AO basis.

A correct implementation is therefore

```python
half = Cvir @ (2.0 * U) @ Cocc.conj().T
dD = half + half.conj().T
```

The accounting is:

1. The factor `2.0` is the closed-shell spatial-orbital occupation.
2. `half + half.T` adds the bra and ket orbital derivatives.
3. There is **no second occupancy factor after symmetrization**.
4. There is **no factor of two on the returned (vo) potential block**.
5. There is **no factor of two multiplying (h_{vo})**.

Thus `fvind` must implement

[
U
\longmapsto
D^{(1)}[U]
\longmapsto
\mathcal K[D^{(1)}[U]]
\longmapsto
C_v^\dagger\mathcal K[D^{(1)}[U]]C_o.
]

PySCF’s own restricted Hessian `gen_vind` uses exactly `x*2` for double occupation, followed by `dm + dm.T`, and then returns the induced (vo) block without another factor. 

## A.4 (vo) block versus full (n_{\mathrm{mo}}\times n_{\mathrm{occ}}) block

Because:

* the nuclei are fixed;
* the AO basis is fixed;
* the overlap derivative is exactly (S'=0);

the call must use

[
h_1.shape=(n_{\mathrm{rhs}},n_{\mathrm{vir}},n_{\mathrm{occ}})
]

and `s1=None`.

A full ((n_{\mathrm{mo}},n_{\mathrm{occ}})) block is used by PySCF’s `solve_withs1` only when the basis/overlap changes, such as nuclear-coordinate perturbations. It is not the correct interface here. 

Also distinguish two unrelated arguments:

* `mf.gen_response(..., hermi=1)` is correct because (D^{(1)}) is Hermitian.
* `cphf.solve(..., hermi=False)` must remain false; `solve_nos1` explicitly rejects `hermi=True`.

---

# B. Observable contractions and signs

All expressions below are per unit positive source charge (q), in atomic units.

## B.1 Electronic exterior MEP

At exterior point (\mathbf r),

[
V_e(\mathbf r)
==============

-\operatorname{Tr}[D J(\mathbf r)],
\qquad
J(\mathbf r)=\operatorname{int1e_rinv}(\mathbf r).
]

Therefore

[
\boxed{
\frac{\partial V_e(\mathbf r)}{\partial q_k}
============================================

-\operatorname{Tr}
\left[
D^{(1)}_kJ(\mathbf r)
\right]
}
]

with no factor (1/2) and no additional closed-shell factor.

At the perturbing source point itself,

[
h'_k=-J_k,
]

so

[
\frac{\partial V_e(\mathbf s_k)}{\partial q_k}
==============================================

\operatorname{Tr}[D^{(1)}_k h'_k].
]

The point coordinates supplied to `with_rinv_origin` must be in Bohr. ([PySCF][1])

## B.2 Total molecular dipole

With a fixed origin,

[
\boldsymbol\mu
==============

## \boldsymbol\mu_{\mathrm{nuc}}

\operatorname{Tr}[D,\mathbf r].
]

Since the nuclei and origin do not move,

[
\boxed{
\frac{\partial\boldsymbol\mu}{\partial q_k}
===========================================

-\operatorname{Tr}[D^{(1)}_k,\mathbf r]
}
]

componentwise.

The external test charge is not part of the molecular dipole defined here. Do not add (q\mathbf s_k).

This sign agrees with PySCF’s molecular dipole implementation, which evaluates nuclear dipole minus the electronic density contraction with `int1e_r`. 

Electron-number conservation,

[
\operatorname{Tr}[S D^{(1)}_k]=0,
]

also makes the induced dipole origin-independent.

## B.3 Energy curvature

For a stationary self-consistent solution,

[
\frac{dE}{dq}
=============

\operatorname{Tr}[D(q)h'].
]

If a classical source–nuclear term

[
q\sum_A\frac{Z_A}{|\mathbf R_A-\mathbf s|}
]

is included in the reported total energy, it contributes to (E'(0)) but not to (E''(0)).

Therefore,

[
\boxed{
E''_k(0)
========

\operatorname{Tr}[D^{(1)}_k h'_k]
}
]

and, using the (vo) amplitudes,

[
\boxed{
E''_k(0)
========

4,\operatorname{Re}
\sum_{ai}
h_{k,ai}^{\prime *}U^{(k)}_{ai}
}
]

for a closed-shell spatial-orbital density.

These two evaluations must agree to solver precision:

[
\operatorname{Tr}[D^{(1)}h']
============================

4\operatorname{Re}\langle h_{vo},U\rangle.
]

The factor (1/2) occurs only in the Taylor expansion,

[
E(q)
====

E_0+qE'(0)+\frac12q^2E''(0)+O(q^3),
]

not in the definition or contraction for (E'').

Because (h'=-J_s),

[
\boxed{
E''_k(0)
========

# -\operatorname{Tr}[D^{(1)}_kJ_k]

\frac{\partial V_e(\mathbf s_k)}{\partial q_k}.
}
]

## B.4 Why the curvature is non-positive

For a fixed variational electronic model,

[
E(q)
====

\min_{\Phi}
\left[
\mathcal E_0[\Phi]
+
q\langle\Phi|h'|\Phi\rangle
\right]
]

is the lower envelope of affine functions of (q), and is therefore concave.

Equivalently, on a stable closed-shell branch,

[
U=-L^{-1}h_{vo},
]

where (L) is the positive static singlet orbital Hessian. Hence

[
E''
===

# 4h_{vo}^\dagger U

-4h_{vo}^\dagger L^{-1}h_{vo}
\le 0.
]

For the uncoupled limit,

[
E''
===

-4\sum_{ai}
\frac{|h'_{ai}|^2}{\epsilon_a-\epsilon_i}
<0.
]

A significantly positive curvature means at least one of:

* a wrong sign or factor;
* an inconsistent energy and response kernel;
* convergence to an unstable stationary point;
* a branch-changing finite-field calculation;
* a numerically failed or nearly singular response solve.

For all four perturbations together, define

[
\chi_{\ell k}
=============

# \frac{\partial V_e(\mathbf s_\ell)}{\partial q_k}

\operatorname{Tr}[D^{(1)}*k h'*\ell].
]

Then an exact conservative response requires

[
\boxed{\chi=\chi^T}
]

and passivity requires

[
\boxed{\chi\preceq0}.
]

Negative diagonal elements alone are weaker than this full negative-semidefinite condition.

---

# C. PySCF 2.13.1 implementation audit

## C.1 Is the proposed `gen_response` call correct?

Yes:

```python
vresp = mf.gen_response(
    C,
    occ,
    singlet=None,
    hermi=1,
    with_nlc=True,
)
```

is the correct PySCF ground-state RKS orbital-Hessian response for this purpose.

In PySCF 2.13.1:

* `singlet=None` selects the ground-state orbital Hessian/CPHF path;
* it uses the unhalved RKS XC kernel;
* it adds Coulomb and the appropriate hybrid/range-separated exchange response;
* `with_nlc=True` adds the nonlocal-correlation response when `mf.do_nlc()` is true.

Using `singlet=True` would select the TDDFT singlet path and alter the XC-kernel scaling; it is not interchangeable with ground-state CPKS. 

For (\omega)B97M‑V, `with_nlc=True` is mandatory. PySCF’s NLC response routine explicitly uses `mf.nlcgrids`. Its source also contains a cautionary TODO concerning grid choice, so agreement with the completed finite-field oracle is a release requirement rather than an optional check. 

## C.2 Is density fitting retained?

Yes, provided `mf` is genuinely reconstructed as the same DFRKS object, for example:

```python
mf = dft.RKS(mol, xc="wb97m_v").density_fit(
    auxbasis=resolved_auxbasis,
    only_dfj=False,
)
```

`gen_response` calls `mf.get_j`, `mf.get_k`, or `mf.get_jk`; the density-fitting mixin overrides those methods, so the response J/K contractions retain the DF approximation. Since the AO basis and nuclei are fixed, no derivative of the auxiliary basis or DF metric is needed with respect to (q). 

However, **“density fitting” alone is not a complete identity**. Freeze:

* the resolved auxiliary basis, not merely `auxbasis=None`;
* `only_dfj`;
* integral-screening configuration if nondefault;
* PySCF version;
* AO basis ordering and spherical/Cartesian choice.

PySCF’s default auxiliary-basis resolution depends on the AO basis and XC functional, using a JK-fit choice for hybrid/RSH methods when available. 

## C.3 Can this be done without rerunning a zero-field SCF?

Yes. Do not call `mf.kernel()`.

Assign the checkpoint state:

```python
mf.mo_coeff = C
mf.mo_occ = occ
mf.mo_energy = eps
mf.e_tot = E0
mf.converged = True
```

Of these:

* (C) and occupations define the zero-order density and XC/NLC kernels;
* (\epsilon) supplies the CPHF denominators;
* (E_0) is not used by `gen_response`, but must be assigned and checked for identity;
* `converged=True` is bookkeeping, not a substitute for consistency checks.

A single zero-field Fock and energy reconstruction is nevertheless mandatory:

[
D_0=2C_oC_o^\dagger,
]

[
F_0=h_0+V_{\mathrm{eff}}[D_0],
]

followed by

[
R_{\mathrm{GS}}=F_0C-SC\epsilon.
]

This is not an SCF rerun. It is a deterministic checkpoint-identity check.

Fail if any of the following is not within the original SCF tolerances:

[
|C^\dagger SC-I|,
]

[
\frac{|F_0C-SC\epsilon|_F}
{|F_0C|_F+|SC\epsilon|_F},
]

[
|C_v^\dagger F_0C_o|,
]

or

[
|E_{\mathrm{reconstructed}}-E_0|.
]

Also require occupations to be exactly (2/0). Partially occupied or smeared states are outside this hard-coded closed-shell density construction.

## C.4 Grid handling

Set both grids explicitly:

```python
mf.grids.level = 3
mf.nlcgrids.level = 3
mf.small_rho_cutoff = 0.0
```

Then build both once from the zero-field density:

```python
mf.initialize_grids(mol, dm0)
```

PySCF can lazily build the semilocal grid inside `NumInt.block_loop`, and the NLC response can lazily build `mf.nlcgrids`, so explicit construction is not required merely to avoid an exception. It is required for a reproducible evidence contract: build once, hash coordinates and weights, and reuse the same objects for all four right-hand sides. 

Do not rely only on `level=3`. Also freeze:

* radial-grid method;
* Becke partition;
* pruning scheme;
* atomic-grid overrides;
* radii adjustment;
* grid cutoff;
* `small_rho_cutoff`;
* coordinates and weights, ideally by hash.

A density-dependent grid-pruning policy that produces different point sets for (+q), (-q), and (q=0) is not differentiable in the required sense and is fail-closed.

## C.5 Four right-hand sides in one solve

Pass

```python
h_vo.shape == (4, nvir, nocc)
```

to one call:

```python
U, _ = cphf.solve(
    fvind,
    eps,
    occ,
    h_vo,
    s1=None,
    tol=1e-11,
    max_cycle=100,
    hermi=False,
    level_shift=0.0,
)
```

PySCF reshapes the leading dimensions into an arbitrary number of Krylov right-hand sides. Therefore `fvind` must accept an arbitrary leading batch size, not assume it will always receive exactly four trial vectors. 

## C.6 Direct residual checks

`cphf.solve` does not return a trustworthy final physical residual report. Recompute

[
R_k
===

(\epsilon_a-\epsilon_i)U^{(k)}*{ai}
+
h^{(k)}*{ai}
+
[\mathrm{fvind}(U)]^{(k)}_{ai}.
]

Use both

[
r_{F,k}
=======

\frac{|R_k|_F}
{|\Delta U_k|_F+|h_k|_F+|v_k|_F}
]

and

[
r_{\infty,k}
============

\frac{|R_k|*\infty}
{\max\left(
|\Delta U_k|*\infty,
|h_k|*\infty,
|v_k|*\infty
\right)}.
]

A reasonable frozen internal gate is

[
\max_k r_{F,k}\le10^{-10},
\qquad
\max_k r_{\infty,k}\le10^{-9}.
]

These internal tolerances are deliberately much tighter than the (10^{-4})-level finite-field truncation budget.

Also require:

[
\frac{|D^{(1)}-D^{(1)\dagger}|_F}
{|D^{(1)}|_F}
\le10^{-12},
]

[
\operatorname{Tr}[SD^{(1)}]\approx0,
]

and

[
\operatorname{Tr}[D^{(1)}h']
============================

4\operatorname{Re}\langle h_{vo},U\rangle.
]

As a batching unit test, solve each of the four modes separately and require agreement with the block solve at the direct-residual scale.

---

# PySCF-style pseudocode

```python
from __future__ import annotations

import numpy as np
from pyscf import dft
from pyscf.scf import cphf
from pyscf.scf import _response_functions  # Registers mf.gen_response


def rinv_matrix(mol, point_bohr: np.ndarray) -> np.ndarray:
    point_bohr = np.asarray(point_bohr, dtype=float)
    if point_bohr.shape != (3,):
        raise ValueError("Point must have shape (3,) in Bohr.")
    with mol.with_rinv_origin(point_bohr):
        return mol.intor_symmetric("int1e_rinv")


def build_static_cphf(
    mol,
    C: np.ndarray,
    occ: np.ndarray,
    eps: np.ndarray,
    E0: float,
    source_points_bohr: np.ndarray,
    resolved_auxbasis,
    dipole_origin_bohr: np.ndarray | None = None,
):
    # ----- Exact method reconstruction; do not call mf.kernel() -----
    mf = dft.RKS(mol, xc="wb97m_v").density_fit(
        auxbasis=resolved_auxbasis,
        only_dfj=False,
    )

    mf.grids.level = 3
    mf.nlcgrids.level = 3
    mf.small_rho_cutoff = 0.0

    mf.mo_coeff = np.asarray(C)
    mf.mo_occ = np.asarray(occ)
    mf.mo_energy = np.asarray(eps)
    mf.e_tot = float(E0)
    mf.converged = True

    if not np.all((mf.mo_occ == 0) | (mf.mo_occ == 2)):
        raise RuntimeError("Not an exact closed-shell 2/0 occupation pattern.")

    occ_mask = mf.mo_occ > 0
    vir_mask = mf.mo_occ == 0

    Cocc = mf.mo_coeff[:, occ_mask]
    Cvir = mf.mo_coeff[:, vir_mask]
    eps_occ = mf.mo_energy[occ_mask]
    eps_vir = mf.mo_energy[vir_mask]

    nocc = Cocc.shape[1]
    nvir = Cvir.shape[1]
    nao = C.shape[0]

    # Spin-summed zero-order density
    dm0 = (Cocc * 2.0) @ Cocc.conj().T

    # Build and freeze both grids once.
    mf.initialize_grids(mol, dm0)

    # ----- Checkpoint consistency: one Fock build, not an SCF rerun -----
    S = mf.get_ovlp(mol)
    h0 = mf.get_hcore(mol)
    veff0 = mf.get_veff(mol, dm0)
    F0 = h0 + veff0

    ortho_res = C.conj().T @ S @ C - np.eye(C.shape[1])
    gs_res = F0 @ C - (S @ C) * eps[None, :]
    fock_vo = Cvir.conj().T @ F0 @ Cocc
    E0_rebuilt = mf.energy_tot(dm=dm0, h1e=h0, vhf=veff0)

    # These must be checked against frozen SCF tolerances.
    consistency = {
        "orthonormality_max": np.max(np.abs(ortho_res)),
        "ground_state_residual_fro": np.linalg.norm(gs_res),
        "fock_vo_max_hartree": np.max(np.abs(fock_vo)),
        "energy_difference_hartree": float(E0_rebuilt - E0),
    }

    # ----- External one-electron perturbations -----
    source_points_bohr = np.asarray(source_points_bohr, dtype=float)
    if source_points_bohr.shape != (4, 3):
        raise ValueError("Expected four source points with shape (4, 3).")

    Jsrc = np.stack([
        rinv_matrix(mol, point)
        for point in source_points_bohr
    ])

    # Positive source q: electron Hamiltonian derivative is -1/|r-s|.
    h1ao = -Jsrc

    # Shape: (4, nvir, nocc). Do not negate again.
    h_vo = np.einsum(
        "pa,kpq,qi->kai",
        Cvir.conj(),
        h1ao,
        Cocc,
        optimize=True,
    )

    # Ground-state RKS response kernel, including VV10/NLC.
    vresp = mf.gen_response(
        C,
        occ,
        singlet=None,
        hermi=1,
        with_nlc=True,
    )

    def amplitudes_to_dm1(U_batch: np.ndarray) -> np.ndarray:
        U_batch = np.asarray(U_batch)
        U_batch = U_batch.reshape(-1, nvir, nocc)

        # One factor 2 for closed-shell occupation.
        half = np.einsum(
            "pa,xai,qi->xpq",
            Cvir,
            2.0 * U_batch,
            Cocc.conj(),
            optimize=True,
        )

        # Add bra/ket orbital derivatives; no extra occupation factor.
        return half + half.swapaxes(-1, -2).conj()

    def fvind(U_in: np.ndarray) -> np.ndarray:
        original_shape = np.asarray(U_in).shape
        U_batch = np.asarray(U_in).reshape(-1, nvir, nocc)

        dm1 = amplitudes_to_dm1(U_batch)
        v1ao = np.asarray(vresp(dm1))

        v1vo = np.einsum(
            "pa,xpq,qi->xai",
            Cvir.conj(),
            v1ao,
            Cocc,
            optimize=True,
        )
        return v1vo.reshape(original_shape)

    U, _ = cphf.solve(
        fvind,
        eps,
        occ,
        h_vo,
        s1=None,
        tol=1e-11,
        max_cycle=100,
        hermi=False,   # Distinct from gen_response(hermi=1)
        level_shift=0.0,
    )

    U = np.asarray(U).reshape(4, nvir, nocc)
    dD = amplitudes_to_dm1(U)

    # ----- Direct CPKS equation residual -----
    delta = eps_vir[:, None] - eps_occ[None, :]
    v_vo = np.asarray(fvind(U)).reshape(4, nvir, nocc)
    residual = delta[None, :, :] * U + h_vo + v_vo

    residual_rel = np.array([
        np.linalg.norm(residual[k]) /
        max(
            np.linalg.norm(delta * U[k])
            + np.linalg.norm(h_vo[k])
            + np.linalg.norm(v_vo[k]),
            np.finfo(float).tiny,
        )
        for k in range(4)
    ])

    # ----- Observable contractions -----
    # dVe[k, ell] = response at source/probe ell to source mode k.
    dVe_sources = -np.einsum(
        "kpq,lqp->kl",
        dD,
        Jsrc,
        optimize=True,
    ).real

    # chi[ell, k] = dVe_sources[k, ell]
    chi = dVe_sources.T

    if dipole_origin_bohr is None:
        dipole_origin_bohr = np.zeros(3)

    with mol.with_common_origin(np.asarray(dipole_origin_bohr, dtype=float)):
        r_ao = mol.intor_symmetric("int1e_r", comp=3)

    dmu = -np.einsum(
        "kpq,xqp->kx",
        dD,
        r_ao,
        optimize=True,
    ).real

    Epp_ao = np.einsum(
        "kpq,kqp->k",
        dD,
        h1ao,
        optimize=True,
    ).real

    Epp_vo = 4.0 * np.einsum(
        "kai,kai->k",
        h_vo.conj(),
        U,
        optimize=True,
    ).real

    dN = np.einsum(
        "pq,kqp->k",
        S,
        dD,
        optimize=True,
    ).real

    checks = {
        "cpks_residual_relative": residual_rel,
        "density_hermiticity_max": np.max(
            np.abs(dD - dD.swapaxes(-1, -2).conj()),
            axis=(1, 2),
        ),
        "electron_number_derivative": dN,
        "energy_curvature_ao": Epp_ao,
        "energy_curvature_vo": Epp_vo,
        "energy_identity_error": Epp_ao - Epp_vo,
        "reciprocity_antisymmetry": chi - chi.T,
        "passivity_eigenvalues": np.linalg.eigvalsh(
            0.5 * (chi + chi.T)
        ),
    }

    return {
        "mf": mf,
        "U_vo": U,
        "dD_ao": dD,
        "dVe_sources": dVe_sources,
        "dmu": dmu,
        "Epp": Epp_ao,
        "chi": chi,
        "consistency": consistency,
        "checks": checks,
    }
```

---

# D. Decisive finite-field validation

Let

[
h_s=3\times10^{-4},
\qquad
h_l=10^{-3}.
]

For a first-derivative quantity (X),

[
X_h
===

\frac{X(+h)-X(-h)}{2h}.
]

For the energy curvature,

[
K_h
===

\frac{E(+h)-2E_0+E(-h)}{h^2}.
]

Use the (O(h^2)) Richardson estimate

[
X_R
===

\frac{h_l^2X_{h_s}-h_s^2X_{h_l}}
{h_l^2-h_s^2}
]

or numerically

[
X_R
===

## 1.0989010989,X_{h_s}

0.0989010989,X_{h_l}.
]

For any normed object, define the absolute and symmetric relative discrepancies

[
a(X,Y)=|X-Y|,
]

[
\rho(X,Y)
=========

\frac{2|X-Y|}
{|X|+|Y|},
]

with the relative metric omitted only when both norms are below a pre-frozen zero-scale floor. Near-zero cases are governed by the absolute gate, never by a post hoc relaxed denominator.

Before opening CPKS results, derive from the completed oracle, separately for each observable category,

[
B_X^{\mathrm{abs}}
==================

\max_{\mathrm{oracle}}
a(X_{h_s},X_{h_l}),
]

[
B_X^{\mathrm{rel}}
==================

\max_{\mathrm{oracle}}
\rho(X_{h_s},X_{h_l}).
]

Then require every CPKS case to satisfy

[
a(X_{\mathrm{CPKS}},X_R)
\le
B_X^{\mathrm{abs}},
]

and

[
\rho(X_{\mathrm{CPKS}},X_R)
\le
B_X^{\mathrm{rel}}.
]

No category’s relative budget may exceed the observed overall maximum

[
\boxed{5.8\times10^{-4}}.
]

Use the category-specific value when it is smaller. Do not replace the existing (1.3\times10^{-4})–(5.8\times10^{-4}) oracle envelope with a generic (10^{-3}) tolerance.

## D.1 Density response

A raw AO Frobenius norm is representation-dependent. Use the Löwdin-transformed density

[
\widetilde D^{(1)}
==================

S^{1/2}D^{(1)}S^{1/2}
]

and define

[
a_D
===

\left|
S^{1/2}
\left(
D^{(1)}_{\mathrm{CPKS}}-D^{(1)}_R
\right)
S^{1/2}
\right|_F,
]

[
\rho_D
======

\frac{
2a_D
}{
|\widetilde D^{(1)}_{\mathrm{CPKS}}|_F
+
|\widetilde D^{(1)}_R|_F
}.
]

Also record the maximum element in the Löwdin basis as an absolute diagnostic.

**A direct (D^{(1)}) validation requires the archived finite-field density matrices (D(+h)) and (D(-h)).** Agreement of MEP, dipole, and energy contractions cannot certify the full density response. If those finite-field densities were not retained, only the observable outputs—not (D^{(1)}) as a released label—are validated.

## D.2 Induced exterior MEP

For the fixed probe vector (\mathbf v_k),

[
a_V
===

|\mathbf v_{k,\mathrm{CPKS}}-\mathbf v_{k,R}|_\infty,
]

and

[
\rho_V
======

\frac{
2|\mathbf v_{k,\mathrm{CPKS}}-\mathbf v_{k,R}|*2
}{
|\mathbf v*{k,\mathrm{CPKS}}|*2+
|\mathbf v*{k,R}|_2
}.
]

Use vector-level relative errors rather than componentwise ratios because some probe responses may be near zero.

## D.3 Induced dipole

For each mode,

[
a_\mu
=====

|
\boldsymbol\mu'_{\mathrm{CPKS}}
-------------------------------

\boldsymbol\mu'_R
|_2,
]

[
\rho_\mu
========

\frac{
2a_\mu
}{
|\boldsymbol\mu'_{\mathrm{CPKS}}|_2+
|\boldsymbol\mu'_R|_2
}.
]

Also record the maximum absolute Cartesian-component error.

## D.4 Energy curvature

For each mode,

[
a_E
===

|E''_{\mathrm{CPKS}}-E''_R|,
]

[
\rho_E
======

\frac{
2|E''_{\mathrm{CPKS}}-E''*R|
}{
|E''*{\mathrm{CPKS}}|+|E''_R|
}.
]

Internally require

[
E''_{\mathrm{CPKS}}
===================

# \operatorname{Tr}[D^{(1)}h']

# 4\operatorname{Re}\langle h_{vo},U\rangle

\chi_{kk}
]

to a relative tolerance of approximately (10^{-10}).

## D.5 Reciprocity

Construct the complete (4\times4) matrix

[
\chi_{\ell k}
=============

-\operatorname{Tr}[D^{(1)}*kJ*\ell].
]

Use

[
a_{\mathrm{rec}}
================

|\chi-\chi^T|_\infty,
]

[
\rho_{\mathrm{rec}}
===================

\frac{
|\chi-\chi^T|_F
}{
\max(|\chi|_F,\text{frozen zero floor})
}.
]

The CPKS reciprocity error should be at the direct CPKS residual scale, preferably

[
\rho_{\mathrm{rec}}\le10^{-9},
]

and must not exceed the finite-field reciprocity budget.

If the oracle did not record the MEP at every source point under every source mode, it does not contain the full cross-response matrix and cannot validate reciprocity in the four-dimensional perturbation subspace.

## D.6 Electron-number conservation

The correct AO invariant is

[
n'_k=\operatorname{Tr}[S D^{(1)}_k],
]

not (\operatorname{Tr}D^{(1)}_k).

Use

[
a_N=|n'_k|,
]

and the normalized measure

[
\rho_N
======

\frac{
|n'_k|
}{
\max\left(
|S^{1/2}D^{(1)}_kS^{1/2}|_F,
1
\right)
}.
]

For the CPKS construction above, both should normally be near roundoff; a hard gate around (10^{-10}) per unit source charge is appropriate. Finite-field density differences may be noisier because SCF residuals are divided by (h), but they must still lie within their independently frozen numerical budget.

## D.7 Passivity

Require both:

[
E''_k\le0
\quad\text{for every }k,
]

and the stronger combined-mode condition

[
\lambda_{\max}
\left[
\frac{\chi+\chi^T}{2}
\right]
\le0
]

within numerical tolerance.

Report

[
a_{\mathrm{pass}}
=================

\max\left(
0,
\lambda_{\max}
\left[
\frac{\chi+\chi^T}{2}
\right]
\right),
]

and

[
\rho_{\mathrm{pass}}
====================

\frac{
a_{\mathrm{pass}}
}{
\max(|\chi|_2,\text{frozen zero floor})
}.
]

A positive eigenvalue larger than the direct residual or oracle uncertainty is fail-closed even if all four diagonal curvatures are negative.

## D.8 Legitimate sources of finite-field/CPKS disagreement

Small disagreement of (O(h^2)) is expected and is exactly what the two-step budget measures. Larger disagreement can legitimately occur when the two methods are no longer differentiating the same mathematical branch:

* an SCF occupation or symmetry branch changes between (+q), (-q), and zero;
* the zero-field solution is close to an internal instability or has a nearly singular response Hessian;
* the checkpoint orbitals and orbital energies do not correspond to the rebuilt Fock operator;
* finite-field and CPKS calculations use different DF auxiliary bases or exact-exchange approximations;
* semilocal or NLC grids, pruning, coordinates, or weights differ;
* `with_nlc=False` or a different NLC kernel is used;
* the finite-field code and CPKS code use opposite signs or different units for `int1e_rinv`;
* one route includes a (q^2) source self-energy not present in the other;
* finite-field SCFs are not converged tightly enough for a second energy difference;
* a symmetry-constrained SCF prevents the point-charge perturbation from accessing the full response;
* the response solve is ill-conditioned or its Krylov residual is not actually converged.

These explanations do not authorize selecting the more favorable result. Any such case fails the batch-backend gate until the identity mismatch is resolved.

---

# E. Hidden category errors and fail-closed scope

## E.1 Four point-charge modes are not a complete density response

Each (D^{(1)}_k) is a complete finite-AO induced one-particle density **for that particular perturbation direction**.

Four such matrices do not identify the full susceptibility operator

[
\frac{\delta D}{\delta v_{\mathrm{ext}}(\mathbf r)}
]

or its AO supermatrix

[
\frac{\partial D_{\mu\nu}}{\partial h_{\lambda\sigma}}.
]

They span at most a four-dimensional subspace of the external-potential space. Therefore they are correctly described as:

* four directional response labels;
* four observable-training probes;
* a probe-subspace test of reciprocity and passivity.

They are not a complete molecular density-response label and do not establish response to arbitrary cavity fields, uniform fields, or higher-spatial-frequency perturbations.

The AO matrices are also basis-representation-specific. Their observable contractions are the representation-invariant scientific quantities.

## E.2 Does CPKS change the scientific target?

No, under the stated goal.

The finite-field oracle estimates

[
\left.\frac{dX}{dq}\right|_{q=0}
]

or

[
\left.\frac{d^2E}{dq^2}\right|_{q=0}
]

by symmetric finite differences. Static CPKS computes those same derivatives directly by differentiating the same self-consistent KS equations.

Replacing 16 finite-field SCFs with one four-right-hand-side CPKS solve therefore:

* removes (O(q^2)) finite-step truncation;
* removes finite-amplitude nonlinear contamination;
* preserves the frozen QM functional, basis, DF approximation, and grids;
* does not introduce PCM, a cavity, solvation data, or an empirical correction.

It would change the target only if the original labels were intentionally defined as finite-amplitude responses at (q=3\times10^{-4}) or (10^{-3}), rather than as approximations to the zero-field derivative.

## E.3 Fail-closed conditions

Do not use the CPKS backend for batch data if any of the following occurs:

1. The exact DF auxiliary basis, `only_dfj`, XC/NLC settings, or both grids are not reproducibly identified.
2. The checkpoint fails orthonormality, canonical-Fock, (vo)-stationarity, or (E_0) reconstruction checks.
3. Occupations are not exact closed-shell (2/0).
4. `gen_response` cannot provide the second derivative of the selected XC functional or NLC is omitted.
5. The direct CPKS residual fails.
6. Electron-number conservation or density Hermiticity fails.
7. The two independent energy-curvature contractions disagree.
8. Reciprocity fails beyond the numerical/oracle budget.
9. The four-mode susceptibility has a significant positive eigenvalue.
10. CPKS and the finite-field oracle disagree beyond the frozen category-specific two-step budgets or the overall (5.8\times10^{-4}) relative ceiling.
11. A finite-field branch change or near instability prevents a unique smooth (q=0) derivative.
12. Finite-field densities were not archived but (D^{(1)}) itself is proposed as a validated released label.
13. The four directional probes are represented as a complete response-density target.

---

# Implementation checklist

* Reconstruct the exact DFRKS object; do not call `kernel()`.
* Freeze the resolved DF auxiliary basis and `only_dfj`.
* Assign `mo_coeff`, `mo_occ`, `mo_energy`, `e_tot`, and `converged`.
* Set and explicitly build both level-3 grids once; set `small_rho_cutoff=0`.
* Hash grid coordinates and weights.
* Rebuild one zero-field Fock and verify (F_0C=SC\epsilon) and (E_0).
* Import `_response_functions` so `gen_response` is registered.
* Use `singlet=None`, `hermi=1`, `with_nlc=True`.
* Pass (h_{vo}=C_v^\dagger(-J_k)C_o) with shape `(4,nvir,nocc)`.
* Construct (D^{(1)}=2(C_vUC_o^\dagger+\mathrm{h.c.})).
* Return only (C_v^\dagger v^{(1)}C_o) from `fvind`.
* Call `cphf.solve(..., s1=None, hermi=False)`.
* Recompute the physical residual (\Delta U+h_{vo}+\mathrm{fvind}(U)).
* Verify (E''=\operatorname{Tr}(D^{(1)}h')=4h_{vo}:U=\chi_{kk}).
* Verify (\operatorname{Tr}(SD^{(1)})=0), reciprocity, and negative-semidefinite passivity.
* Compare every released quantity with the Richardson-extrapolated finite-field oracle under the frozen absolute and relative two-step budgets.

[1]: https://pyscf.org/pyscf_api_docs/pyscf.gto.html?utm_source=chatgpt.com "pyscf.gto package"
