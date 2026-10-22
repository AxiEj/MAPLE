# SMD solvent catalog (PySCF 2.13.1)

The default Torch v4 profile accepts all 179 named entries below. The empty upstream all-zero sentinel is excluded. Parameters are reused verbatim, not fitted here. This is parameter coverage, not an assertion of experimental accuracy for every solute/solvent combination.

Use the parser-safe name in `.inp`, for example:

```text
#solv(method=smd,provider=torch,implicit=acetone,experimental=true)
```

The original PySCF name and common aliases resolve to the same canonical identity. Cis/trans isomers and explicitly named mixtures remain distinct; bare `decalin` and `xylene` are rejected rather than guessed. Historical v3 profiles remain water/hexane only; other legacy profiles retain their original scope.

Parameter ordering in `SMDSolventDescriptors.as_pyscf_tuple()` is n, n(25C), acidity, basicity, surface tension, dielectric, aromatic-carbon fraction, electronegative-halogen fraction. The vendored snapshot includes upstream source/table SHA256 and Apache-2.0 attribution.

| Parser-safe name | PySCF SMD name | Dielectric ε |
|---|---|---:|
| `1-1-1-trichloroethane` | 1,1,1-trichloroethane | 7.0826 |
| `1-1-2-trichloroethane` | 1,1,2-trichloroethane | 7.1937 |
| `1-2-4-trimethylbenzene` | 1,2,4-trimethylbenzene | 2.3653 |
| `1-2-dibromoethane` | 1,2-dibromoethane | 4.9313 |
| `1-2-dichloroethane` | 1,2-dichloroethane | 10.125 |
| `1-2-ethanediol` | 1,2-ethanediol | 40.245 |
| `1-4-dioxane` | 1,4-dioxane | 2.2099 |
| `1-bromo-2-methylpropane` | 1-bromo-2-methylpropane | 7.7792 |
| `1-bromooctane` | 1-bromooctane | 5.0244 |
| `1-bromopentane` | 1-bromopentane | 6.269 |
| `1-bromopropane` | 1-bromopropane | 8.0496 |
| `1-butanol` | 1-butanol | 17.332 |
| `1-chlorohexane` | 1-chlorohexane | 5.9491 |
| `1-chloropentane` | 1-chloropentane | 6.5022 |
| `1-chloropropane` | 1-chloropropane | 8.3548 |
| `1-decanol` | 1-decanol | 7.5305 |
| `1-fluorooctane` | 1-fluorooctane | 3.89 |
| `1-heptanol` | 1-heptanol | 11.321 |
| `1-hexanol` | 1-hexanol | 12.51 |
| `1-hexene` | 1-hexene | 2.0717 |
| `1-hexyne` | 1-hexyne | 2.615 |
| `1-iodobutane` | 1-iodobutane | 6.173 |
| `1-iodohexadecane` | 1-iodohexadecane | 3.5338 |
| `1-iodopentane` | 1-iodopentane | 5.6973 |
| `1-iodopropane` | 1-iodopropane | 6.9626 |
| `1-nitropropane` | 1-nitropropane | 23.73 |
| `1-nonanol` | 1-nonanol | 8.5991 |
| `1-octanol` | 1-octanol | 9.8629 |
| `1-pentanol` | 1-pentanol | 15.13 |
| `1-pentene` | 1-pentene | 1.9905 |
| `1-propanol` | 1-propanol | 20.524 |
| `2-2-2-trifluoroethanol` | 2,2,2-trifluoroethanol | 26.726 |
| `2-2-4-trimethylpentane` | 2,2,4-trimethylpentane | 1.9358 |
| `2-4-dimethylpentane` | 2,4-dimethylpentane | 1.8939 |
| `2-4-dimethylpyridine` | 2,4-dimethylpyridine | 9.4176 |
| `2-6-dimethylpyridine` | 2,6-dimethylpyridine | 7.1735 |
| `2-bromopropane` | 2-bromopropane | 9.361 |
| `2-butanol` | 2-butanol | 15.944 |
| `2-chlorobutane` | 2-chlorobutane | 8.393 |
| `2-heptanone` | 2-heptanone | 11.658 |
| `2-hexanone` | 2-hexanone | 14.136 |
| `2-methoxyethanol` | 2-methoxyethanol | 17.2 |
| `2-methyl-1-propanol` | 2-methyl-1-propanol | 16.777 |
| `2-methyl-2-propanol` | 2-methyl-2-propanol | 12.47 |
| `2-methylpentane` | 2-methylpentane | 1.89 |
| `2-methylpyridine` | 2-methylpyridine | 9.9533 |
| `2-nitropropane` | 2-nitropropane | 25.654 |
| `2-octanone` | 2-octanone | 9.4678 |
| `2-pentanone` | 2-pentanone | 15.2 |
| `2-propanol` | 2-propanol | 19.264 |
| `2-propen-1-ol` | 2-propen-1-ol | 19.011 |
| `3-methylpyridine` | 3-methylpyridine | 11.645 |
| `3-pentanone` | 3-pentanone | 16.78 |
| `4-heptanone` | 4-heptanone | 12.257 |
| `4-methyl-2-pentanone` | 4-methyl-2-pentanone | 12.887 |
| `4-methylpyridine` | 4-methylpyridine | 11.957 |
| `5-nonanone` | 5-nonanone | 10.6 |
| `a-chlorotoluene` | a-chlorotoluene | 6.7175 |
| `acetic-acid` | acetic acid | 6.2528 |
| `acetone` | acetone | 20.493 |
| `acetonitrile` | acetonitrile | 35.688 |
| `acetophenone` | acetophenone | 17.44 |
| `aniline` | aniline | 6.8882 |
| `anisole` | anisole | 4.2247 |
| `benzaldehyde` | benzaldehyde | 18.22 |
| `benzene` | benzene | 2.2706 |
| `benzonitrile` | benzonitrile | 25.592 |
| `benzylalcohol` | benzylalcohol | 12.457 |
| `bromobenzene` | bromobenzene | 5.3954 |
| `bromoethane` | bromoethane | 9.01 |
| `bromoform` | bromoform | 4.2488 |
| `butanal` | butanal | 13.45 |
| `butanoic-acid` | butanoic acid | 2.9931 |
| `butanone` | butanone | 18.246 |
| `butanonitrile` | butanonitrile | 24.291 |
| `butylamine` | butylamine | 4.6178 |
| `butylethanoate` | butylethanoate | 4.9941 |
| `butylether` | butylether | 3.0473 |
| `carbon-disulfide` | carbon disulfide | 2.6105 |
| `carbon-tetrachloride` | carbon tetrachloride | 2.228 |
| `chlorobenzene` | chlorobenzene | 5.6968 |
| `chloroform` | chloroform | 4.7113 |
| `cis-1-2-dimethylcyclohexane` | cis-1,2-dimethylcyclohexane | 2.06 |
| `cis-decalin` | cis-decalin | 2.2139 |
| `cyclohexane` | cyclohexane | 2.0165 |
| `cyclohexanone` | cyclohexanone | 15.619 |
| `cyclopentane` | cyclopentane | 1.9608 |
| `cyclopentanol` | cyclopentanol | 16.989 |
| `cyclopentanone` | cyclopentanone | 13.58 |
| `decalin-cis-trans-mixture` | decalin (cis/trans mixture) | 2.196 |
| `dibromomethane` | dibromomethane | 7.2273 |
| `dichloromethane` | dichloromethane | 8.93 |
| `diethylamine` | diethylamine | 3.5766 |
| `diethylether` | diethylether | 4.24 |
| `diethylsulfide` | diethylsulfide | 5.723 |
| `diiodomethane` | diiodomethane | 5.32 |
| `diisopropyl-ether` | diisopropyl ether | 3.38 |
| `dimethyldisulfide` | dimethyldisulfide | 9.6 |
| `dimethylformamide` | N,N-dimethylformamide | 37.219 |
| `dimethylsulfoxide` | dimethylsulfoxide | 46.826 |
| `diphenylether` | diphenylether | 3.73 |
| `dipropylamine` | dipropylamine | 2.9112 |
| `e-1-2-dichloroethene` | E-1,2-dichloroethene | 2.14 |
| `e-2-pentene` | E-2-pentene | 2.051 |
| `ethanethiol` | ethanethiol | 6.667 |
| `ethanol` | ethanol | 24.852 |
| `ethylbenzene` | ethylbenzene | 2.4339 |
| `ethylethanoate` | ethylethanoate | 5.9867 |
| `ethylmethanoate` | ethylmethanoate | 8.331 |
| `ethylphenylether` | ethylphenylether | 4.1797 |
| `fluorobenzene` | fluorobenzene | 5.42 |
| `formamide` | formamide | 108.94 |
| `formicacid` | formicacid | 51.1 |
| `hexane` | n-hexane | 1.8819 |
| `hexanoicacid` | hexanoicacid | 2.6 |
| `iodobenzene` | iodobenzene | 4.547 |
| `iodoethane` | iodoethane | 7.6177 |
| `iodomethane` | iodomethane | 6.865 |
| `isopropylbenzene` | isopropylbenzene | 2.3712 |
| `m-cresol` | m-cresol | 12.44 |
| `m-xylene` | m-xylene | 2.3478 |
| `mesitylene` | mesitylene | 2.265 |
| `methanol` | methanol | 32.613 |
| `methylbenzoate` | methylbenzoate | 6.7367 |
| `methylbutanoate` | methylbutanoate | 5.5607 |
| `methylcyclohexane` | methylcyclohexane | 2.024 |
| `methylethanoate` | methylethanoate | 6.8615 |
| `methylmethanoate` | methylmethanoate | 8.8377 |
| `methylpropanoate` | methylpropanoate | 6.0777 |
| `n-butylbenzene` | n-butylbenzene | 2.36 |
| `n-decane` | n-decane | 1.9846 |
| `n-dodecane` | n-dodecane | 2.006 |
| `n-heptane` | n-heptane | 1.9113 |
| `n-hexadecane` | n-hexadecane | 2.0402 |
| `n-methylaniline` | N-methylaniline | 5.96 |
| `n-methylformamide-e-zmixture` | N-methylformamide(E/Zmixture) | 181.56 |
| `n-n-dimethylacetamide` | N,N-dimethylacetamide | 37.781 |
| `n-nonane` | n-nonane | 1.9605 |
| `n-octane` | n-octane | 1.9406 |
| `n-pentadecane` | n-pentadecane | 2.0333 |
| `n-pentane` | n-pentane | 1.8371 |
| `n-undecane` | n-undecane | 1.991 |
| `nitrobenzene` | nitrobenzene | 34.809 |
| `nitroethane` | nitroethane | 28.29 |
| `nitromethane` | nitromethane | 36.562 |
| `o-chlorotoluene` | o-chlorotoluene | 4.6331 |
| `o-cresol` | o-cresol | 6.76 |
| `o-dichlorobenzene` | o-dichlorobenzene | 9.9949 |
| `o-nitrotoluene` | o-nitrotoluene | 25.669 |
| `o-xylene` | o-xylene | 2.5454 |
| `p-isopropyltoluene` | p-isopropyltoluene | 2.2322 |
| `p-xylene` | p-xylene | 2.2705 |
| `pentanal` | pentanal | 10 |
| `pentanoic-acid` | pentanoic acid | 2.6924 |
| `pentyl-ethanoate` | pentyl ethanoate | 4.7297 |
| `pentylamine` | pentylamine | 4.201 |
| `perfluorobenzene` | perfluorobenzene | 2.029 |
| `propanal` | propanal | 18.5 |
| `propanoic-acid` | propanoic acid | 3.44 |
| `propanonitrile` | propanonitrile | 29.324 |
| `propyl-ethanoate` | propyl ethanoate | 5.5205 |
| `propylamine` | propylamine | 4.9912 |
| `pyridine` | pyridine | 12.978 |
| `sec-butylbenzene` | sec-butylbenzene | 2.3446 |
| `tert-butylbenzene` | tert-butylbenzene | 2.3447 |
| `tetrachloroethene` | tetrachloroethene | 2.268 |
| `tetrahydrofuran` | tetrahydrofuran | 7.4257 |
| `tetrahydrothiophene-s-s-dioxide` | tetrahydrothiophene-S,S-dioxide | 43.962 |
| `tetralin` | tetralin | 2.771 |
| `thiophene` | thiophene | 2.727 |
| `thiophenol` | thiophenol | 4.2728 |
| `toluene` | toluene | 2.3741 |
| `trans-decalin` | trans-decalin | 2.1781 |
| `tributylphosphate` | tributylphosphate | 8.1781 |
| `trichloroethene` | trichloroethene | 3.422 |
| `triethylamine` | triethylamine | 2.3832 |
| `water` | water | 78.355 |
| `xylene-mixture` | xylene (mixture) | 2.3879 |
| `z-1-2-dichloroethene` | Z-1,2-dichloroethene | 9.2 |
