# Matrix-free ddPCM research — working evidence

**Superseded working snapshot:** see `REPORT.md` and final candidate-bound receipts for current results. This earlier checklist is retained as history; its pending items and counts are not current status.

Not a completed 300-atom PES implementation. All code remains ignored research only.

- Untouched source baseline: 112 tests passed. Tracked source freeze has 312 records and HEAD57effbb5.
- Original streaming prototype: 9 transpose tests caught diagonal overwriting prior off-diagonal accumulations; fixed with accumulation, no tolerance change.
- Hardening: full topologycertificate parity, mutation rejection, RHS-width/tile limits, adversarialgeometry and full productionresolution panel: 52 tests passed (iteration-5).
- Historical eager CPU50 atom fullL/Dforward+transpose:50.79s, peakRSS0.995GiB;100 atom274.09s,peakRSS1.0195GiB. Both no retained BxB or globalpair-grid arrays. These are engineering observations, not formal speedupqualification.
- Unique-degree radial-power candidate:50 atom165.42s vsold50.79s in recordedruns; retained negative and reverted. No speed claim or acceptance by tolerance relaxation.
- Direct torch.compile of oldlazyimport helper failed: Dynamo cannot trace __import__. Oldruntime untouched.
- Closed value recurrence compile succeeded. Pure real-sectoral polynomial candidate:eagerbasisdifference5.766e-15; compiledmicrotiming medians0.1533s→0.03638s, compilation203.78s. This is a value-only microbenchmark, not fulloperator/Hessian speedup.
- First integratedcompiledoperator test hit300s coldcompile watchdog without output. Negative receipt retained. A separately recorded600s initialization probe normalizes inputstrides and processdefaultFP64; pending. Numericerror/memorygates unchanged.
- Native pyddx source-adjoint reference:tiny fullresolutionEerror0,sourceVJPerror4.44e-16eV,gradientrepeaterror2.57e-9eV/A. This is nativefixed-source reference, notTorchfullPES. Native300baseline ladder approved by Architect under8GiB/1200s percase; not launched yet.
- Actual nativeFP64V2water model/CDS/source/checkpointidentity ledger completed; no download. Actualmaple.main .inp rejects newresearchprofile. Initialnegativeobserveromitted must-be-one-of wording;rawfailurepreserved and separately audited as expectedprofile rejection.
- Native Planner→Architect→Criticapproved ONLY Gate1–2 research execution. This does not certify Gatecompletion, Gate3 stability,full300PES,chemistrydomain or publicentry.

Outstanding: candidate-bound final receipts, integratedcompiledoperator parity andresource ladder, independent implementationreview; laternewsolverstability/analyticresponse/MACE+CDS300/fullH/gates.
