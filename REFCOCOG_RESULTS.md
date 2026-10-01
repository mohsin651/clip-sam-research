# Frozen CASR on RefCOCOg UMD test

CASR is the existing frozen CLIP-aware SAM refinement (the saved `frozen_final` row): refined CLIP attribution, three positive points, SAM ViT-B, smart selector and density fallback. No method changes, fitting, recalibration, negative points, prompt rewriting or difficulty exclusions occurred.

**CASR improves instance-mask quality on standard RefCOCOg test and the strict image-non-overlap cohort.** Instance disambiguation remains a limitation, despite a modest improvement over naive SAM.

## Dataset and overlap

Original [REFER research release](https://github.com/lichengunc/refer), `refs(umd).p`; UMD is recommended by the research API and provides train/val/test. The same archive also contains Google annotations, whose test split is not released; no convention was selected by performance. Original COCO train2014 RGB. The unavailable UNC archive host was accessed through an archived original; checksums and resolved URL are in `outputs/refcocog_frozen/dataset_provenance.json`.

| UMD split | Images | Targets | References | Expressions |
| --- | --- | --- | --- | --- |
| train | 21,899 | 42,224 | 42,226 | 80,512 |
| val | 1,300 | 2,573 | 2,573 | 4,896 |
| test (evaluated) | 2,600 | 5,023 | 5,023 | 9,602 |

UMD train/val/test image sets were checked disjoint. Only test was evaluated. Every test expression is verbatim and accepted by the unchanged tokenizer.

Prior-image intersections recorded before inference: COCO development 0, COCO heldout 0, negative-point VAL development 39, RefCOCO tests 118, RefCOCO+ tests 118. The two test intersections are the same images; union overlap is **157**. Standard scores retain them. Strict scores exclude the union: **2,443 images, 8,894 expressions, 4,648 targets**. Strict and standard are nested, not independent replications. Exact IDs are retained.

## Primary full-image instance results

Rates are fractions. Expression-macro IoU, not detection AP.

| Method | mIoU | Dice | P@0.5 | P@0.7 | Harm vs CLIP | Correct instance |
| --- | --- | --- | --- | --- | --- | --- |
| CLIP | 0.2173 | 0.3343 | 0.0507 | 0.0015 | 0 | 0.6412 |
| Naive SAM | 0.3183 | 0.4037 | 0.2861 | 0.1638 | 0.4370 | 0.6245 |
| Smart | 0.3451 | 0.4358 | 0.3147 | 0.1829 | 0.3641 | 0.6287 |
| CASR final | **0.3472** | **0.4420** | 0.3080 | 0.1786 | **0.3013** | **0.6343** |
| Center control | 0.0963 | 0.1176 | 0.0900 | 0.0669 | 0.8523 | 0.3682 |

CASR mIoU 95% CI: [0.3396, 0.3544]. Final precision/recall/pixel accuracy: 0.4241 / 0.5831 / 0.8538. Full per-method metrics and intervals are in the complete report.

Strict mIoU: CLIP **0.2172**, naive **0.3199**, smart **0.3473**, CASR **0.3492** [0.3414, 0.3573], center **0.0948**. Strict CASR-minus-naive: **+0.0293 [0.0249, 0.0337]**; smart-minus-naive: **+0.0274 [0.0234, 0.0313]**. This supports transfer to images absent from all five prior cohorts.

## Paired evidence

| Standard test comparison | mIoU difference | Paired 95% CI |
| --- | --- | --- |
| Naive minus CLIP | +0.1010 | [+0.0944, +0.1071] |
| Smart minus naive | +0.0268 | [+0.0231, +0.0306] |
| CASR minus naive | +0.0289 | [+0.0247, +0.0330] |
| CASR minus CLIP | +0.1298 | [+0.1244, +0.1355] |
| CASR minus smart | +0.0021 | [+0.0006, +0.0036] |

2,000 paired whole-image bootstrap resamples, seed 2026, expression-macro weighting; no multiplicity adjustment. All expressions in each resampled image stay together.

## Harm, instance ambiguity and fallback tradeoffs

Naive harm versus CLIP is **43.70%**, smart **36.41%**, CASR **30.13%**. CASR-minus-naive harm reduction is **13.57 percentage points [12.61, 14.53]**. Fallback alone reduces harm by **6.28 points [5.63, 6.95]**, alongside the small positive IoU interval above. It lowers P@0.5 versus smart by **0.68 points [0.47, 0.89]**.

CASR versus CLIP: improved 60.10%, worsened 30.13%, unchanged 9.77%; median delta +0.0729, conditional positive/negative means +0.2709/-0.1094, worst-decile mean -0.2200. Directly versus naive: improved **30.77%**, worsened **16.07%**, unchanged **53.16%**. These direct rates are different from harm versus CLIP.

All 9,602 expressions have same-category competitors. CASR correct-instance **63.43%**, wrong-instance **34.18%**, tie **2.38%**. Final-minus-naive correct-instance gain is **+0.99 percentage points [0.40, 1.57]**. Smart-minus-naive +0.43 points has an interval spanning zero. CLIP's correct-instance rate is 64.12%; CASR-minus-CLIP -0.69 points [-1.54, +0.13] does not establish a gain. Naive SAM lowers this proxy relative to CLIP. These are overlap-based proxies, not manual semantic verification.

## Language analysis

Mean expression length **8.23 words**, median **8**; bins and lexicons are unchanged.

| Length | Expressions | CASR mIoU | P@0.5 | P@0.7 | Correct instance | Harm |
| --- | --- | --- | --- | --- | --- | --- |
| short <=3 | 736 | 0.4307 | 0.4239 | 0.2541 | 0.6984 | 0.2174 |
| medium 4-6 | 2,779 | 0.3641 | 0.3213 | 0.1914 | 0.6434 | 0.2904 |
| long >=7 | 6,087 | 0.3293 | 0.2878 | 0.1636 | 0.6225 | 0.3164 |

Long expressions have lower observed means than short/medium within this test. This is descriptive, not a causal length effect or proof the weakness becomes more severe than on another dataset. Spatial mIoU is **0.2844** (2,780 expressions); attribute **0.3772** (4,723); color **0.3794** (4,086); clothing **0.3801** (1,522). These overlapping groups confound object size/category and scene difficulty; all complementary groups and methods are retained.

## Candidate and fallback diagnostics

No candidate reaches IoU >=0.5 for **58.80%** of expressions. Candidate oracle mIoU **0.4104**, selector regret **0.0653**, candidate-plus-CLIP oracle **0.4436**, selected-mask-plus-perfect-fallback oracle **0.3858**. Oracles are GT-only diagnostic bounds, not deployable performance.

Fallback prevents harm in **6.28%** (603 expressions), rejects a useful mask in **2.21%** (212), and accepts harm in **30.13%**. Candidate failures may reflect guidance, crop limits, SAM, or combinations; no single causal bottleneck is established. All 128 crop-invisible targets remain in evaluation; secondary crop scores are separate.

## Cross-dataset context only

| Dataset | CLIP | Naive SAM | Smart | CASR final |
| --- | --- | --- | --- | --- |
| Dense heldout COCO | 0.2699 | 0.3949 | 0.4291 | 0.4335 |
| RefCOCO | 0.2034 | 0.2872 | 0.3081 | 0.3071 |
| RefCOCO+ | 0.2079 | 0.3016 | 0.3224 | 0.3216 |
| RefCOCOg UMD test | 0.2173 | 0.3183 | 0.3451 | 0.3472 |

Raw scores are not directly comparable: COCO uses category unions in the center crop; referring-expression experiments use full-image instances and differ in language, targets and overlap. Selector gains and harm reductions are qualitatively consistent; this is not proof of a causal language improvement.

The evidence supports stopping additional closely related referring-expression evaluations and, after user review, moving to external comparisons and paper preparation. It does not establish competitive performance or solve instance disambiguation. No external baseline, method change or additional dataset was run after this evaluation.

## Artifacts and runtime

- [Full report and thirteen research answers](outputs/refcocog_frozen/REFCOCOG_RESULTS.md)
- [Thirty representative panels](outputs/refcocog_frozen/examples.html), three per requested diagnostic group; selection is post-hoc and does not estimate prevalence
- [Protocol](REFCOCOG_PLAN.md) and [reproduction notes](REFCOCOG_REPRODUCTION.md)
- [Final integrity audit](outputs/refcocog_frozen/audit.json)

Final audit passed: all 102,753 prior output files unchanged, all 21,804 prediction files verified, all decisions replayed without GT, and 118 overlapping-image center controls reproduced exactly. 61 tests passed; exactly 20 smoke expressions reused without GT scoring. Inference including hashing/loading/saving and smoke: **1,567.4 seconds**; evaluation including prediction verification: **164.1 seconds**. Download/preparation, implementation, reporting, visualization and final audit are additional. All expressions retained; zero/constant maps 0/0, empty candidates 0. No automatic GitHub push. Stop for review.
