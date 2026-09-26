# Frozen RefCOCO+ evaluation

Completed official UNC testA + testB: **1,500 images, 10,615 expressions, 3,773 referred targets**. Original expressions and positive-only CLIP attribution / SAM / smart selector / density fallback are unchanged. No fitting, negative points, tuning, or exclusions.

**All 1,500 images overlap the prior RefCOCO test evaluation.** Overlap with COCO development, COCO heldout and negative-point validation development is zero. The union-excluded strict cohort is empty; strict results are not estimable. This is not image-disjoint generalization.

## Primary full-image instance results

| Method | mIoU | Dice | P@0.5 | Harm versus CLIP | Correct instance |
| --- | --- | --- | --- | --- | --- |
| CLIP | 0.2079 | 0.3232 | 0.0366 | 0 | 0.5387 |
| Naive SAM | 0.3016 | 0.3879 | 0.2560 | 0.4463 | 0.5435 |
| Smart | 0.3224 | 0.4142 | 0.2768 | 0.3761 | 0.5449 |
| Frozen final | 0.3216 | 0.4157 | 0.2698 | 0.3308 | 0.5472 |
| Center control | 0.0925 | 0.1152 | 0.0858 | 0.8476 | 0.3112 |

All rates are fractions. Final mIoU 95% CI: [0.3129, 0.3298]. These are expression-macro instance scores, not official COCO AP.

| Split | Expressions | CLIP mIoU | Naive mIoU | Smart mIoU | Final mIoU |
| --- | --- | --- | --- | --- | --- |
| testA | 5,726 | 0.2008 | 0.2927 | 0.3205 | 0.3220 |
| testB | 4,889 | 0.2161 | 0.3119 | 0.3247 | 0.3211 |

## Paired evidence and interpretation

- Smart minus naive mIoU: **+0.0209 [0.0165, 0.0251]**. The frozen selector improves mask quality.
- Final minus naive mIoU: **+0.0200 [0.0154, 0.0244]**.
- Final minus CLIP mIoU: **+0.1137 [0.1075, 0.1195]**.
- Final minus smart mIoU: **-0.0009 [-0.0023, +0.0005]**. An IoU benefit from fallback is not established.
- Final reduces harm versus CLIP from naive's **44.63% to 33.08%**; paired reduction **11.55 percentage points [10.56, 12.49]**. Fallback alone reduces harm by 4.53 points, but also lowers P@0.5 by 0.70 points [0.49, 0.92].
- Final correct-instance success is **54.72%**. Final minus naive: **+0.38 percentage points [-0.22, +0.97]**. Improved instance disambiguation is not established. Wrong-instance preference is 44.10%; ties are 1.18%.

Intervals use 2,000 paired image-level resamples, seed 2026, retaining all expressions per image; no multiplicity adjustment. Harm contrasts above compare each method with CLIP. Directly versus naive SAM, final improves 29.48%, worsens 18.44% and leaves 52.09% unchanged; these are different denominators of comparison from harm versus CLIP. Full distributions are in direct_vs_naive.csv.

## Remaining limitations

60.40% of expressions have no candidate reaching IoU 0.5. Candidate oracle mIoU is 0.4015, candidate-plus-CLIP oracle 0.4312, and selected-mask-plus-perfect-fallback oracle 0.3595. These are analysis-only upper bounds. They do not identify a unique causal failure of CLIP versus SAM or guarantee recoverable practical gains.

Final mIoU is 0.2060 for long expressions, 0.2590 for medium and 0.3770 for short. Attribute/color/clothing and spatial groups use unchanged lexicons; these overlapping descriptive comparisons are confounded. They do not establish language effects. The difference from earlier RefCOCO final mIoU 0.3071 is also not a controlled language intervention.

All 152 expressions whose target is outside the CLIP crop remain. Primary evaluation uses the full original image; secondary crop scores remain separate. CLIP sees its original crop, SAM sees the original RGB image. No geometry repair occurred.

## Artifacts and reproduction

- [Complete tables and research answers](outputs/refcocoplus_frozen/REFCOCOPLUS_RESULTS.md)
- [30 representative panels](outputs/refcocoplus_frozen/examples.html), selected by disclosed outcome groups; not a frequency estimate
- [Reproduction notes](REFCOCOPLUS_REPRODUCTION.md)
- [Predeclared protocol](REFCOCOPLUS_PLAN.md)
- [Final integrity audit](outputs/refcocoplus_frozen/audit.json)

Final audit passed: 79,966 prior files unchanged; all 22,730 prediction files verified; all decisions replayed; same-image center masks reproduced exactly. 56 tests passed. Exactly 20 technical smoke predictions were reused. All 10,615 expressions were retained; zero/constant maps and empty candidates: zero. Inference including smoke, hashing/loading/saving took 1,290.2 seconds; evaluation including prediction verification took 215.8 seconds. Reporting, visualization, download and implementation are additional.

The previous negative-point STOP decision remains unchanged. **RefCOCOg was not run. Stop for user review.** New artifacts are local and have not been pushed to GitHub; datasets, weights and large arrays are excluded from Git.
