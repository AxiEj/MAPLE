## Decision: **GO once for a prospectively frozen local-whitened ETB-β2 architecture.**

This is **not** a reversal of the failed ladder. The ladder remains **FAIL** because its registered object was the globally fitted molecular coefficient vector, whose constrained Coulomb metric exceeded (10^{10}).

The results separate two distinct facts:

[
\text{small reaction error}
\quad\Longrightarrow\quad
\text{adequate physical span},
]

whereas

[
\kappa!\left(Z^{T}D^{-1/2}JD^{-1/2}Z\right)\gg 10^{10}
\quad\Longrightarrow\quad
\text{nonidentifiable global coefficients}.
]

ETB-β2 therefore appears physically expressive enough on the registered panel, but overlapping atom-centred functions admit large coefficient changes producing almost the same density and exterior potential. The worsening sequence ETB-β2 (\rightarrow) AutoAux (\rightarrow) ETB-β1.5 is consistent with added representational redundancy, not deficient physical capacity. Nonorthogonality of off-centre auxiliary functions is known to couple nominally atomic density components globally, while molecular Löwdin-type orthogonalization makes the coordinates configuration-dependent. ([arXiv][1])

## 1. Exact fixed local transform

For element (z), angular momentum (l), radial indices (n,n'), and real spherical component (m), write the ETB-β2 functions as

[
\phi_{zlnm}(\mathbf r)=R_{zln}(r)Y_{lm}(\widehat{\mathbf r}).
]

Define the **isolated-atom Coulomb radial metric**

[
[K_{zl}]_{nn'}
==============

\iint
\frac{
\phi_{zlnm}(\mathbf r),
\phi_{zln'm}(\mathbf r')
}{
|\mathbf r-\mathbf r'|
},d\mathbf r,d\mathbf r'.
]

Spherical symmetry makes this independent of (m), with the full block

[
K_{zl}\otimes I_{2l+1}.
]

Freeze the canonical symmetric whitening

[
W_{zl}=K_{zl}^{-1/2},
]

or equivalently, for an unpivoted Cholesky (K_{zl}=L_{zl}L_{zl}^{T}),

[
W_{zl}=L_{zl}^{-T}.
]

No eigenvalue removal, clipping, regularization, pivot selection, or molecular information is permitted. Symmetric whitening is preferable because it does not depend on an arbitrary radial ordering. PySCF explicitly supplies the standard even-tempered (\beta=2) auxiliary construction and identifies this machinery as (J)-metric density fitting. ([PySCF][2])

To simplify exact constraints, apply a further fixed radial orthogonal rotation (O_{zl}):

* for (l=0), its first column aligns with the whitened charge-integral vector;
* for (l=1), its first column aligns with the whitened local first-moment vector;
* for (l\ge2), take (O_{zl}=I).

For molecule (R),

[
T_R
===

\bigoplus_{A,l}
\left[
W_{Z_A l}O_{Z_A l}\otimes I_{2l+1}
\right],
\qquad
c=T_Ra .
]

Then

[
\rho(\mathbf r)=\phi_R(\mathbf r)^Tc
=\phi_R(\mathbf r)^TT_Ra,
]

and the molecular Coulomb metric in the new coordinates is

[
\widetilde J_R=T_R^T J_R T_R.
]

Because (T_R) is invertible,

[
\operatorname{span}(\phi_RT_R)=\operatorname{span}(\phi_R).
]

Thus **physical representation capacity is exactly unchanged**. This is a fixed coefficient gauge and preconditioner, not an electrostatic repair.

## 2. Exact charge and first-moment constraints

Let the selected (l=0) carrier on atom (A) produce charge

[
q_A=\gamma_{Z_A}a_{Aq},
]

and the selected (l=1) carrier produce local first moment

[
\mathbf p_A=\eta_{Z_A}\mathbf a_{Ap},
]

with the fixed real-spherical-to-Cartesian map absorbed into (\mathbf a_{Ap}).

Choose the translation-covariant origin

[
\mathbf o_R
===========

\frac{\sum_A\gamma_{Z_A}^{,2}\mathbf R_A}
{\sum_A\gamma_{Z_A}^{,2}}.
]

Construct the (4\times n) constraint matrix (A_R) so that

[
A_Ra=
\begin{pmatrix}
\sum_A q_A[2mm]
\sum_A\left[(\mathbf R_A-\mathbf o_R)q_A+\mathbf p_A\right]
\end{pmatrix}.
]

For target electron count (N_e) and electronic first moment (\mathbf M_e),

[
b_R=
\begin{pmatrix}
N_e\
\mathbf M_e-\mathbf o_RN_e
\end{pmatrix}.
]

Given unconstrained network output (\widehat a), apply

[
\boxed{
a^\star
=======

\widehat a+
A_R^T(A_RA_R^T)^{-1}
\left(b_R-A_R\widehat a\right)
}
]

using a direct (4\times4) SPD solve.

This is permissible. It is an exact minimum-local-norm conservation layer, not a pseudoinverse of the molecular Coulomb metric. It makes no rank decisions and removes no density directions. Because the (l=1) carriers contribute

[
\sum_A\eta_{Z_A}^2 I_3
]

to the moment block, (A_RA_R^T) is structurally full rank whenever the registered elements have nonzero charge and dipole carriers.

## 3. SO(3) requirement

The transform must satisfy

[
T_{zl}=T^{\rm radial}*{zl}\otimes I*{2l+1},
]

so it commutes with every (D^{(l)}(Q)). There may be:

* no mixing between different (l);
* no (m)-dependent radial transform;
* no geometry-defined local frames;
* no pivot ordering based on the molecule.

The final density is a scalar field, but the coefficient head is **not scalar-only**. It must output

[
a_{Anl}(QR)
===========

D^{(l)}(Q),a_{Anl}(R).
]

Scalar outputs may provide radial mixing weights, but all (l>0) coefficients must be genuine equivariant irreps. Symmetry-adapted atom-centred density models use precisely this covariant coefficient structure. ([arXiv][1])

## 4. Frozen acceptance gate

For the new architecture, register all of the following before molecular evaluation:

1. **Atomic gate:** every (K_{zl}) is positive definite, with no truncation and
   [
   \kappa_2(K_{zl})\le10^{10}.
   ]

2. **Constraint gate:** after scaling the three moment rows by one fixed length unit,
   [
   \kappa_2(A_RA_R^T)\le10^{10}
   ]
   for every geometry, with scaled constraint residual at most (10^{-10}).

3. **Global identifiability gate:** let (Z_R) be any Euclidean-orthonormal basis of (\ker A_R), used only for auditing. Require
   [
   H_R=Z_R^T\widetilde J_RZ_R,
   \qquad
   \boxed{\kappa_2(H_R)\le10^{10}}
   ]
   for **every** registered geometry. Retain the original cap; no mean criterion and no benzene/aniline exception.

4. **Forward stability gate:** binary64 versus high-precision replay must change the PCMSolver reaction quantity by at most
   [
   10^{-4}\ {\rm kcal/mol},
   ]
   while all existing reciprocity, charge, moment, SO(3), and mean/max reaction-bound gates remain unchanged.

Failure of any one item closes the route. Do not proceed to another basis, cutoff, pivot tolerance, or whitening variant.

## Architecture comparison

| Choice                                 |                     Decision | Reason                                                                                                                                                                                                                                                                                         |
| -------------------------------------- | ---------------------------: | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Fixed per-element ETB-β2 whitening** |                  **GO once** | Cavity-independent, smooth, transferable, span-preserving, and SO(3)-compatible. It removes atomic radial scale disparities but honestly leaves molecular overlap dependencies for the global gate to detect.                                                                                  |
| **Molecular global pivoted Cholesky**  |                   **REJECT** | Without truncation it is merely a geometry-dependent coordinate trick that can make conditioning tautologically good; pivot swaps make learned coordinates nonlocal and potentially nonsmooth. With truncation it becomes prohibited geometry-dependent pruning and changes the physical span. |
| **Close auxiliary-density route now**  | Premature, but next fallback | The excellent ETB-β2 reaction bound justifies exactly one fixed-local-coordinate attempt. If global constrained conditioning still fails, the overlap nonidentifiability is intrinsic enough that the route should close.                                                                      |

The new architecture becomes an **impermissible rescue** if it merely retransforms the old global fitted coefficients and announces a new condition number. It is a **principled new architecture** only when the isolated-atom transforms are frozen and hashed in advance, the old ladder remains recorded as FAIL, no molecular (J^{-1}), (J^{-1/2}), SVD cutoff, or pivoted projection is used to create coefficient labels, and supervision is applied directly to independent QM density, Coulomb-field, or exterior-potential observables.

[1]: https://arxiv.org/html/2206.14087v1 "Electronic-structure properties from atom-centered predictions of the electron density"
[2]: https://pyscf.org/pyscf_api_docs/pyscf.df.html "pyscf.df package — PySCF"
