# Frozen RefCOCO generalization: completed

The unchanged smart selector improves instance-mask IoU on natural referring expressions. Reliable improvement in correct-instance selection is not established. The fallback reduces harm but does not establish an additional IoU gain.

## Dataset and integrity

Original REFER RefCOCO UNC testA + testB: 1,500 images, 3,785 target instances, 10,752 original referring expressions. testA has 5,657 expressions; testB has 5,095. Zero images overlap either prior 500-image COCO cohort, so the strict non-overlap result is identical to the standard result, not an independent replication.

All valid expressions were retained, including 163 whose target is invisible in the CLIP crop. Every target has another same-category non-crowd instance in the full image; the single-instance subgroup is empty. Expressions were passed verbatim to the existing tokenizer. No fitting, tuning, prompt rewriting or component changes occurred.

## Primary full-original-image results

IoU and Dice are expression-macro instance-mask scores. Percentage columns use all expressions. Harm means lower IoU than the matching CLIP baseline.

| Method | mIoU | Dice | P@0.5 | Improved | Worsened | Correct instance |
| --- | --- | --- | --- | --- | --- | --- |
| CLIP baseline | 0.2034 | 0.3172 | 3.50% | 0% | 0% | 50.80% |
| Naive SAM-score | 0.2872 | 0.3711 | 24.00% | 52.05% | 46.30% | 50.74% |
| Frozen smart selector | 0.3081 | 0.3973 | 26.04% | 58.86% | 39.50% | 51.15% |
| Frozen final + fallback | 0.3071 | 0.3992 | 25.23% | 55.64% | 34.11% | 51.15% |
| Center SAM control | 0.0907 | 0.1132 | 8.39% | 13.15% | 84.81% | 30.74% |

Final precision: 0.3919; recall: 0.5200; pixel accuracy: 0.8353; P@0.7: 14.39%. Final mIoU 95% CI: [0.2985, 0.3157]. Final testA mIoU: 0.3078; testB: 0.3064. Secondary CLIP-crop results are retained separately. Earlier COCO category-union crop mIoU 0.4335 is not a directly comparable benchmark.

## Paired comparisons

2,000 whole-image bootstrap resamples, seed 2026, preserve all expressions from each sampled image. Intervals are not multiplicity-adjusted.

| Comparison | mIoU difference | 95% CI |
| --- | --- | --- |
| Naive minus CLIP | +0.0838 | [+0.0773, +0.0905] |
| Smart minus naive | +0.0209 | [+0.0167, +0.0252] |
| Final minus naive | +0.0199 | [+0.0155, +0.0244] |
| Final minus CLIP | +0.1037 | [+0.0978, +0.1095] |
| Final minus smart | -0.0010 | [-0.0025, +0.0005] |

Final harm is 12.18 percentage points below naive (95% CI [-13.19, -11.14]) and 5.39 points below smart alone. Versus CLIP, final improves 55.64%, worsens 34.11%, and leaves 10.25% unchanged; mean delta +0.1037, median +0.0380, conditional positive mean +0.2463, conditional negative mean -0.0977, worst-decile mean -0.2048.

Directly versus naive, final improves 29.50%, worsens 18.63%, and leaves 51.87% unchanged; mean delta +0.0199, median 0, conditional positive mean +0.1766, conditional negative mean -0.1729, worst-decile mean -0.2981. Direct worsening versus naive differs from the harm rate measured against CLIP.

## Instance identity and language difficulty

Correct-instance selection requires target IoU to exceed every other same-category instance IoU. Final succeeds on 51.15%, favors another instance on 47.77%, and ties on 1.08%. Final versus naive accuracy difference is +0.41 percentage points, 95% CI [-0.25, +1.09]; improvement is not established. This overlap proxy does not guarantee a clean mask containing only the requested instance.

| Expression group | Expressions | Final mIoU |
| --- | --- | --- |
| Short, <=3 words | 6,452 | 0.3496 |
| Medium, 4-6 words | 3,401 | 0.2536 |
| Long, >=7 words | 899 | 0.2044 |
| Spatial-language lexicon match | 5,893 | 0.2398 |
| No spatial-language match | 4,859 | 0.3887 |
| Attribute-language match | 2,749 | 0.3609 |
| No attribute-language match | 8,003 | 0.2886 |

These are overlapping descriptive lexicon groups, with category and image-content confounding; they do not isolate causal language effects. No subgroup informed tuning.

## Remaining failures and conclusion

63.27% of expressions have no candidate reaching target IoU 0.5. Candidate oracle mIoU is 0.3808; candidate-plus-CLIP oracle is 0.4128. The latter leaves 0.1057 above final. The fallback prevents harm in 579 expressions and rejects a useful SAM mask in 347. Inadequate candidates, candidate choice and fallback errors remain; these measurements cannot isolate CLIP semantic failure from SAM proposal failure as the dominant cause.

The method transfers in mask quality and harm reduction, while instance disambiguation remains limited. It is technically ready for unchanged evaluation on RefCOCO+ and RefCOCOg, but neither was run. Stop here for user review; do not repair or tune the frozen method on these inspected test results.

## Saved evidence and reproduction

- [Full report, including all nine research answers](outputs/refcoco_frozen/REFCOCO_RESULTS.md)
- [Thirty post-hoc visual panels](outputs/refcoco_frozen/examples.html), with exact selections in [example_manifest.json](outputs/refcoco_frozen/example_manifest.json)
- [Paired statistics](outputs/refcoco_frozen/paired_comparisons.json), [per-expression scores](outputs/refcoco_frozen/per_expression_results.csv), [integrity audit](outputs/refcoco_frozen/audit.json)
- [Reproduction instructions and source provenance](REFCOCO_REPRODUCTION.md), [frozen protocol](REFCOCO_FROZEN_PLAN.md)

Exactly 20 technical smoke samples preceded full inference, without aggregate GT evaluation. All 46 tests passed. Final audit verified unchanged frozen components and all 49,467 prior output files, all prediction hashes, and GT-free replay of every decision. No expressions were excluded. Zero/constant attribution maps: 0/0; empty candidate masks: 0.

Inference-stage wall time was 1,596.9 seconds, including initial preservation hashing, loading, verification and saving; it is not pure GPU time. Evaluation took 269.1 seconds. Final audit and report rendering are additional. New RefCOCO artifacts are local; the earlier GitHub snapshot does not yet include this experiment.
