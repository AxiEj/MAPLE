Terminal decision
The proposition is true under the stated facts. Choose B.
The hidden 
l\ge 2
 tensors are equivariant latent coordinates, not identified physical multipoles or density coefficients. Exact closure of 
Q
, 
\boldsymbol\mu
, and 
\boldsymbol\alpha
 leaves an infinite family of equivariant, PCM-visible source perturbations unconstrained. A decoder attached to one serialized checkpoint can be perfectly deterministic and content-addressed, but it is still a new, representation-dependent source model rather than a checkpoint-native physical observable.
￼
1. Non-identifiability theorem and explicit counterexample
Let 
x=(R,E)
 denote geometry and applied field, and let 
h(x)\in H
 be the intercepted hidden representation. Let
D:H\longrightarrow Y
be a proposed equivariant decoder. Define the public closure operator
\mathcal C[y](R) = \left( Q[y(R,0)], \boldsymbol\mu[y(R,0)], \left.\frac{\partial\boldsymbol\mu[y(R,E)]}{\partial E}\right|_{E=0} \right).
The last component is the molecular polarizability closure.
Let 
B_R:Y\to\mathbb R^M
 be the linear map from the decoded source to the discrete ddPCM right-hand side, including whatever boundary-potential representation is used. For fixed cavity geometry,
A_R\sigma=b,\qquad b=B_Ry,
and the electrostatic polarization energy has the quadratic form
G_R(y) = -\frac f2\, (B_Ry)^{T}A_R^{-1}(B_Ry), \qquad A_R\succ0.
Theorem: moment closure does not identify the PCM source
Assume:
1. H
 contains at least one nonzero hidden 
V_l
 channel with 
l\ge2
;
2. Y
 admits a corresponding physical 
l\ge2
 density or boundary-potential channel;
3. at least one such channel is visible to the continuum, meaning 
B_Ry\neq0
 for some geometry and source;
4. the only physical closure conditions are 
Q
, 
\boldsymbol\mu
, and 
\boldsymbol\alpha
.
Then either no decoder satisfies the closure conditions, or infinitely many distinct equivariant decoders satisfy them. Among the latter are decoders producing different cavity-surface potentials and different ddPCM energies.
Proof
Consider one atom 
A
. Write its hidden 
l=2
 coefficients as
h_{A,2}\in \mathbb R^{m_2}\otimes V_2.
Choose any nonzero multiplicity contraction 
u\in\mathbb R^{m_2}
, giving
z_A=u^{T}h_{A,2}\in V_2.
Choose a smooth compactly supported radial function 
g\in C_c^\infty(0,a)
 such that
M_4=\int_0^a r^4g(r)\,dr\neq0.
Using real spherical harmonics, define the equivariant density perturbation
[N_u(h)](\mathbf r) = \sum_A\sum_{m=-2}^{2} (z_A)_m\, g(r_A)\, Y_{2m}(\widehat{\mathbf r-\mathbf R_A}),
where 
r_A=|\mathbf r-\mathbf R_A|
.
This is rotationally equivariant because 
z_A
 and the 
Y_{2m}
 basis both transform under 
V_2
. It can be made permutation-equivariant by applying the same construction atomwise.
For every hidden state,
Q[N_u(h)]=0.
This follows because every 
Y_{2m}
 has zero angular average. Likewise,
\boldsymbol\mu[N_u(h)]=0.
The contribution from 
\mathbf R_A
 vanishes because the perturbation has zero total charge, while the local first moment vanishes by orthogonality of 
l=1
 and 
l=2
 spherical harmonics.
These statements hold for every applied field 
E
, regardless of how 
h_{A,2}(R,E)
 depends on 
E
. Consequently,
\frac{\partial}{\partial E} \boldsymbol\mu[N_u(h(R,E))] =0.
Thus 
N_u
 changes neither 
Q
, nor 
\boldsymbol\mu
, nor 
\boldsymbol\alpha
.
But its electrostatic potential is not zero. For a point 
\mathbf s
 outside the support around atom 
A
,
\delta V_A(\mathbf s) = \frac{4\pi M_4}{5} \sum_{m=-2}^{2} (z_A)_m \frac{ Y_{2m}(\widehat{\mathbf s-\mathbf R_A}) }{ |\mathbf s-\mathbf R_A|^3 },
up to the chosen electrostatic unit convention. This is a nonzero quadrupolar cavity potential whenever 
z_A\neq0
.
Now suppose 
D_0
 is any decoder satisfying the public closures. For every scalar 
t
, define
D_t=D_0+tN_u.
Then
\mathcal C[D_t(h)]=\mathcal C[D_0(h)]
for every 
t
, geometry, and applied field.
For a geometry where
\delta b=B_RN_u(h)\neq0,
the ddPCM energy becomes
G_R(D_t) = -\frac f2 (b_0+t\delta b)^TA_R^{-1}(b_0+t\delta b).
Hence
G_R(D_t)-G_R(D_0) = -ft\,\delta b^TA_R^{-1}b_0 -\frac f2t^2\delta b^TA_R^{-1}\delta b.
Since 
A_R\succ0
 and 
\delta b\neq0
,
\delta b^TA_R^{-1}\delta b>0.
Therefore 
G_R(D_t)
 is not constant in 
t
. The decoders have exactly the same 
Q/\mu/\alpha
 closure but different PCM energies. ∎
The exact identifiability criterion
The PCM-relevant source would be identifiable from the closure data only if every closure-null perturbation were also continuum-silent:
\boxed{ \ker\mathcal C \subseteq \bigcap_R \ker B_R }
on the declared admissible source class.
For identification of the actual interior density, the stronger condition would be
\ker\mathcal C=\{0\}.
The explicit 
l=2
 construction lies in 
\ker\mathcal C
 but not in 
\ker B_R
. Therefore even the weaker PCM-equivalence class is not identified.
Why polarizability does not cure this
Although an anisotropic polarizability tensor itself contains rotational 
l=0
 and 
l=2
 pieces, it remains the derivative of the dipole moment with respect to field:
\alpha_{ij}=\frac{\partial\mu_i}{\partial E_j}.
That does not make it a measurement of the spatial 
l=2
 density. A spatial quadrupolar density perturbation whose dipole is identically zero for every field contributes zero to 
\alpha
, even when its quadrupole coefficient changes with the field.
Radial non-identifiability exists independently
The ambiguity is not limited to 
l\ge2
. Suppose a new decoder introduces multiple radial 
l=0
 channels. Let
\gamma_a(\mathbf r) = (2\pi a^2)^{-3/2} e^{-r^2/(2a^2)}
and choose 
a\neq b
. Then
\eta_{ab}=\gamma_a-\gamma_b
has
\int\eta_{ab}\,d^3r=0, \qquad \int\mathbf r\,\eta_{ab}\,d^3r=0.
Nevertheless its finite-distance potential is
V_{\eta_{ab}}(s) = \frac{ \operatorname{erf}\!\left(s/\sqrt{2}a\right) - \operatorname{erf}\!\left(s/\sqrt{2}b\right) }{s},
which is nonzero at finite cavity distance. Multiplication by any hidden scalar feature therefore supplies another infinite family of equivariant, 
Q/\mu/\alpha
-silent but PCM-visible decoder changes.
The same construction can be made with two 
l=1
 radial functions normalized to have the same total dipole moment. Their difference has zero dipole but a different near-field potential.
The only radial loophole would be a theorem showing that all admissible radial differences are fully enclosed and exactly electrostatically silent on every relevant cavity. No such theorem follows from the checkpoint facts, and Gaussian tails make that especially inapplicable at finite cavity distance.
￼
2. The hidden-channel gauge issue
Equivariance fixes the angular representation type, not the physical meaning of multiplicity coordinates.
For one angular degree 
l
,
H_l=\mathbb R^{m_l}\otimes V_l.
By Schur-type reasoning, an equivariant linear decoder has the form
D_l=A_l\otimes I_{V_l}, \qquad A_l\in\mathbb R^{n_l\times m_l}.
Symmetry leaves the entire matrix 
A_l
 undetermined. It says that all magnetic components of a given irrep must be treated consistently; it does not specify:
• which multiplicity direction is physical;
• the sign or normalization of that direction;
• its units;
• a radial basis;
• an electron-versus-charge convention;
• whether it means density, multipole, MEP, or an abstract learned feature.
Function-preserving hidden reparameterization
Factor the public checkpoint function through a hidden layer:
F=F_{>}\circ F_{<}, \qquad h=F_{<}(x).
For
T_G = \bigoplus_l(G_l\otimes I_{V_l}), \qquad G_l\in GL(m_l),
define the functionally equivalent representation
\widetilde F_{<}=T_GF_{<}, \qquad \widetilde F_{>}=F_{>}T_G^{-1}.
Then
\widetilde F_{>}\circ\widetilde F_{<} = F_{>}\circ F_{<} = F.
The checkpoint’s declared input-output function is unchanged, while its hidden coordinates become
\widetilde h_l=(G_l\otimes I)h_l.
A decoder with numerical block 
A_l
 now produces
A_l\widetilde h_l = A_lG_lh_l,
which generally differs from 
A_lh_l
.
To preserve the decoded output, the decoder would have to be changed simultaneously to
\widetilde A_l=A_lG_l^{-1}.
That transformation law demonstrates that 
A_l
 is an additional model object. It is not determined by the public checkpoint function.
Indeed, a fixed nonzero linear decoder cannot be invariant under arbitrary hidden basis changes. Requiring
A_lG_l=A_l \qquad \text{for every }G_l\in GL(m_l)
implies 
A_l=0
: take 
G_l=2I
. Even invariance under sign reversal alone gives the same result.
Content-addressed reproducibility is weaker
A decoder tied to:
• an exact checkpoint hash;
• an exact software commit;
• a named hook location;
• a fixed tensor ordering;
• a fixed radial basis and units;
can be deterministic and bitwise reproducible.
That establishes engineering provenance. It does not establish either representation invariance or physical identifiability.
The distinctions are:
Property
What it requires
Hidden hand-decoder
Engineering reproducibility
Same bytes and code give the same result
Yes
Representation/gauge invariance
Functionally equivalent hidden parameterizations give the same source
No, unless the decoder co-transforms
Checkpoint-native status
Source is part of the released model’s declared output map
No
Physical identifiability
All physically admissible models matching public observables give the same source or PCM potential
No
Copy table
Channel permutations, sign changes, rescalings, arbitrary invertible mixing among equal irreps, and compensating transformations in adjacent weights can therefore change the hidden decoder while leaving the public 
Q/\mu/\alpha
 function unchanged.
Packaging the decoder and its transformation rule with the checkpoint would produce a reproducible new composite model. It would not retrospectively make the source an observable of the original release.
￼
3. Canonical optimization does not create physical identification
A strictly convex selector can make an underdetermined inverse problem have a unique numerical answer. That is not the same as showing the answer is physically identified.
Consider the finite-dimensional analogue
y=(y_1,y_2),\qquad y_1=q,
where 
y_2
 is an unresolved PCM-visible mode.
Euclidean minimum norm gives
\arg\min_{y_1=q}(y_1^2+y_2^2) = (q,0).
But for any 
|c|<1
, the equally strictly convex quadratic
J_c(y) = y_1^2+2cy_1y_2+y_2^2
gives
\arg\min_{y_1=q}J_c(y) = (q,-cq).
Every choice is unique. Every choice satisfies the observable constraint. The unresolved physical prediction changes with the metric.
The same distinction applies to the proposed canonical principles:
• Euclidean minimum norm in hidden coordinates is directly gauge-dependent under 
G_l
.
• Coulomb minimum norm in density space may be coordinate-invariant after the density representation is specified, but it introduces a physical prior not contained in 
Q/\mu/\alpha
.
• Least action is physical only when the action is independently derived and the source is proved to be its stationary solution. Merely writing down an action is a new model assumption.
• Maximum entropy requires a support, reference measure or prior, positivity convention, and selected constraints. Different legitimate priors produce different densities.
• Minimum self-work uniquely selects a member of a predeclared ansatz only relative to the chosen ansatz and metric.
Thus a canonical selector can define a reproducible source profile, but cannot prove that the selected profile is the density represented by the hidden features.
This is also the correct interpretation of the already-running ADT construction. Its Coulomb-self-work criterion can make its atomic translation allocation mathematically unique within that explicit physical ansatz. It does not identify hidden MACE channels, and it does not acquire truth merely from uniqueness. It must succeed or fail as the separately specified ADT source model.
￼
4. Narrow exceptions
There are genuine exceptions, but none is present for the proposed hidden-feature route.
Exception 1: a released frozen physical decoder
Zero-training deployment would be defensible if the checkpoint already included a frozen head
D_{\mathrm{released}}:H\to Y
trained and documented to output physical density coefficients, complete multipoles, or cavity MEP values with fixed units, basis functions, widths, and conventions.
Under a hidden gauge transformation, that trained head would transform together with the rest of the checkpoint, preserving the complete declared output map.
The existing released 
4N
 monopole/dipole head is such an operational output only for its declared 
l\le1
, fixed-width representation. It does not authorize an 
l\ge2
 or new radial completion. The documented nonuniqueness also means its atomwise decomposition should not be reinterpreted as a unique quantum density.
No released 
l\ge2
, compact-density, or cavity-MEP head exists in the supplied facts.
Exception 2: an exact analytic semantic theorem
A hidden channel could be used if an analytic theorem established that it equals a specified physical coefficient. Such a theorem would need to fix, at minimum,
\text{angular normalization, radial basis, units, origin, sign, and gauge transformation law}.
For example, an architecture that explicitly constructs an atom-centered density basis and defines a named hidden tensor to be its expansion coefficients could provide such semantics.
“Transforms as 
V_2
” is not that theorem. It proves only tensor character, not quadrupole units or meaning.
No such analytic link is present here.
Exception 3: complete supervised boundary-potential output
If the released model directly output the complete reaction-driving boundary potential
V_\Gamma(\mathbf s)
for every relevant cavity point, geometry, and field condition, then the PCM right-hand side would be defined without reconstructing an interior density.
That would identify the PCM-relevant equivalence class, although many interior densities could still produce the same boundary potential. Complete Dirichlet data determine the exterior electrostatic solution, not generally a unique interior charge density.
No such output is present.
Exception 4: an injective predeclared admissible source class
Low-order observables could identify a source if the source class had been fixed in advance so that
\ker\mathcal C=\{0\},
or at least
\ker\mathcal C\subseteq\ker B_R
for every relevant cavity.
An example would be a source class containing exactly one fixed radial monopole basis and one fixed radial dipole basis, with no atomwise redistribution freedom and no higher multipoles.
The proposed hidden 
l\ge2
/radial completion explicitly violates this condition. Its purpose is to introduce precisely the modes that 
Q/\mu/\alpha
 do not determine.
Therefore no narrow identifiable exception remains under the stated facts.
￼
5. Model identity and provenance
Intercepting hidden 
l\ge2
 tensors and attaching a new hand-designed decoder does not preserve the original physical source-model identity.
It preserves:
• the original backbone parameter values;
• the original public 
Q/\mu/\alpha
 outputs;
• the fact that no new coefficients were statistically fitted.
It changes:
• the overall input-to-source map;
• the source representation;
• the cavity potential;
• the PCM energy;
• the claimed physical semantics.
“Zero-training” describes how the new decoder parameters were chosen. It does not mean that the model is unchanged.
The resulting construction would have to be named and versioned as something like
\text{MACE-POLAR-1 backbone} + \text{hand-decoder source profile v1},
with a separate specification of the checkpoint hash, intercepted tensor, hidden gauge representative, decoder equations, radial functions, units, source sign conventions, and continuum interface.
It must not be called:
• a recovered MACE-POLAR density;
• the checkpoint’s latent quadrupole;
• a native MACE-MDP density;
• a physical source implied by the original training.
The original model is being used as a deterministic feature generator for a new source model.
￼
6. Terminal recommendation
Choose B.
Close the zero-training hidden-feature-decoder loophole now.
The empirical 
q/p
-only panel and response-manifold results support that governance decision, but they are not the mathematical reason for it. The mathematical reason is stronger: hidden completion fails identifiability before any accuracy result is inspected. Exact low-order closure cannot distinguish infinitely many PCM-visible equivariant completions, and hidden-channel coordinates have no invariant physical calibration.
Finish the already-running ADT candidate because it is the previously defined, independently motivated zero-training source construction. Treat it as its own physical ansatz, with its own target-independent gates—not as evidence that hidden tensors have acquired density semantics.
If ADT fails, terminate the current zero-training source-expansion program. The next admissible route is a separately named scalar-first, density, or cavity-MEP head trained on independent QM electrostatic information. That would be a new source model with explicit supervision and provenance, not a residual correction and not solvation-target fitting.
Option A is mathematically indefensible as an identified continuation of MACE-MDP/MACE-POLAR. Option C would require the exact kernel inclusion theorem
\ker\mathcal C\subseteq\bigcap_R\ker B_R
or an equivalent released semantic decoder theorem. The explicit quadrupolar and radial counterexamples show that condition is false for the contemplated source class.
HIDDEN FEATURES DO NOT IDENTIFY A PHYSICAL PCM SOURCE
