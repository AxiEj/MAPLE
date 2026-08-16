# Smooth-area terminal lane Pro decision — 2026-08-17

## Trigger

The preregistered stock-area M4 pseudo-Huber candidate terminated on its first
fold certificate before producing any OOF prediction or MAE.  The completed
stock-area M2 value remained `1.626595 kcal/mol`, above the frozen
`1.5 kcal/mol` development gate.  A new decision was therefore required before
opening any further candidate.

## Verified Pro consultation

The retained Chrome session visibly showed **Pro** in the composer and
**Pro, 5 of 5** in the power control.  The answer was retained only after the
same conversation finished, its final marker and **Copy response** appeared,
and generation controls disappeared.

| Item | Value |
|---|---|
| Conversation | `https://chatgpt.com/c/6a8213a3-f084-83e8-8d16-48fb32357f35` |
| Conversation control | `conversation-options-WEB:9974c1d2-e6a0-4c2c-b37e-5d21e4f08108` |
| Prompt SHA-256 | `872ccc40ed15e7e9d04629427057ca1de6463b0554d59047339d0494e35a69fd` |
| Submission evidence SHA-256 | `b90a174ccc9295e3b29f8034110aaff7fdbef54040f52589d6128b6e8ab7ee9e` |
| Complete UIA SHA-256 | `38ef2101c41a838c6006a3578b83dbcefa054d2af636b1a645927864876e6431` |
| Clean answer SHA-256 | `c156fb7bcefc889f1430986b715225291031125657bd754dd31be9360a82ec6f` |
| Reported reasoning time | `12m 1s` |

The external review is advisory.  Repository mathematics, code, and tests
remain the authority.

## Adopted decision

The stock-area three-dimensional lane is terminal.  M4 is not re-solved under
a different optimizer or relaxed replay rule, and M2 remains historical rather
than admitted.

Exactly one final CDS accuracy lane is allowed:

1. use a production-consistent, positive, SO(3)-structured smooth area frozen
   before any new target-visible fit;
2. regenerate all 306 water geometry rows under that new content-addressed
   area definition;
3. use the same exact-family ten-fold OOF split;
4. retain pseudo-Huber `delta=1.0 kcal/mol` without a delta search;
5. use a deterministic three-dimensional subspace and a safeguarded solver
   whose scientific certificate is based on a strong-convexity minimizer ball,
   not only an optimizer success flag;
6. pass only if OOF `MAE + 1e-6 <= 1.5 kcal/mol` and every structural/numerical
   certificate passes; and
7. if it fails, terminate the current hybrid accuracy route rather than opening
   another CDS candidate.

The Pro answer suggested constructing the two geometry directions within each
training fold.  That detail is not accepted automatically: it must be frozen
as a deterministic training-only transformation and proved not to use held-out
targets.  Confirmation remains sealed.

