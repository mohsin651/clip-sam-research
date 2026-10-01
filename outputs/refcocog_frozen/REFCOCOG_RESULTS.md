# Frozen RefCOCOg generalization results

Evaluation-only, positive-point-only. The original frozen RefCOCO infer_one function, CLIP/SAM checkpoints, features, ranker, fallback and preprocessing are reused unchanged. Raw expressions are verbatim. No negative points, fitting, calibration, filtering by performance, or method repair. Primary metrics are original-image referred-instance expression-macro scores.

## Dataset and provenance

Original [REFER research release](https://github.com/lichengunc/refer), UMD train/val/test split, recommended by REFER; Google test is not released. Original host unavailable; archived original RefCOCOg ZIP used. Official COCO train2014 images, reused or downloaded with dimensions/decoding/hash checks. Exact archive URL and checksums: dataset_provenance.json. All split fields preserved.

| split | images | target_instances | references | expressions |
| --- | --- | --- | --- | --- |
| test | 2600 | 5023 | 5023 | 9602 |
| train | 21899 | 42224 | 42226 | 80512 |
| val | 1300 | 2573 | 2573 | 4896 |

Evaluated counts: {"images": 2600, "expressions": 9602, "target_instances": 5023, "references": 5023}. Complete competitor annotation IDs verified against official COCO annotations.

## Overlap, recorded before inference

| prior_cohort | overlap_images |
| --- | --- |
| COCO development | 0 |
| COCO heldout | 0 |
| RefCOCO negative-point validation | 39 |
| RefCOCO testA/testB | 118 |
| RefCOCO+ testA/testB | 118 |

Union overlap: 157 images. Strict remaining cohort: {"images": 2443, "expressions": 8894, "target_instances": 4648, "references": 4648}. Per-split counts and exact IDs are in overlap_audit.json. Prior-cohort counts may overlap and must not be added.

## Standard UMD test primary results

| method | iou | dice | p_at_05 | p_at_07 | worsened | correct_instance |
| --- | --- | --- | --- | --- | --- | --- |
| clip_baseline | 0.2173 | 0.3343 | 0.0507 | 0.0015 | 0.0000 | 0.6412 |
| sam_score | 0.3183 | 0.4037 | 0.2861 | 0.1638 | 0.4370 | 0.6245 |
| smart | 0.3451 | 0.4358 | 0.3147 | 0.1829 | 0.3641 | 0.6287 |
| frozen_final | 0.3472 | 0.4420 | 0.3080 | 0.1786 | 0.3013 | 0.6343 |
| center_score | 0.0963 | 0.1176 | 0.0900 | 0.0669 | 0.8523 | 0.3682 |

Rates are fractions; worsened is harm versus CLIP. P@0.5/P@0.7 are target IoU success fractions, not detection AP.


## Strict image-non-overlap results

| split | expressions | images | method | iou | dice | p_at_05 | p_at_07 | worsened | correct_instance |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| test | 8894 | 2443 | clip_baseline | 0.2172 | 0.3339 | 0.0514 | 0.0015 | 0.0000 | 0.6456 |
| test | 8894 | 2443 | sam_score | 0.3199 | 0.4054 | 0.2883 | 0.1658 | 0.4342 | 0.6296 |
| test | 8894 | 2443 | smart | 0.3473 | 0.4381 | 0.3175 | 0.1859 | 0.3604 | 0.6341 |
| test | 8894 | 2443 | frozen_final | 0.3492 | 0.4440 | 0.3109 | 0.1817 | 0.2994 | 0.6394 |
| test | 8894 | 2443 | center_score | 0.0948 | 0.1159 | 0.0879 | 0.0652 | 0.8544 | 0.3674 |

### Strict paired comparisons

| comparison | metric | difference | ci_low | ci_high |
| --- | --- | --- | --- | --- |
| sam_score minus clip_baseline | iou | 0.1027 | 0.0957 | 0.1094 |
| sam_score minus clip_baseline | harm_rate | 0.4342 | 0.4213 | 0.4484 |
| sam_score minus clip_baseline | correct_instance | -0.0160 | -0.0260 | -0.0059 |
| sam_score minus clip_baseline | p_at_05 | 0.2369 | 0.2251 | 0.2486 |
| smart minus sam_score | iou | 0.0274 | 0.0234 | 0.0313 |
| smart minus sam_score | harm_rate | -0.0739 | -0.0816 | -0.0662 |
| smart minus sam_score | correct_instance | 0.0045 | -0.0003 | 0.0095 |
| smart minus sam_score | p_at_05 | 0.0292 | 0.0217 | 0.0369 |
| frozen_final minus sam_score | iou | 0.0293 | 0.0249 | 0.0337 |
| frozen_final minus sam_score | harm_rate | -0.1348 | -0.1451 | -0.1247 |
| frozen_final minus sam_score | correct_instance | 0.0098 | 0.0039 | 0.0157 |
| frozen_final minus sam_score | p_at_05 | 0.0226 | 0.0149 | 0.0305 |
| frozen_final minus clip_baseline | iou | 0.1320 | 0.1260 | 0.1381 |
| frozen_final minus clip_baseline | harm_rate | 0.2994 | 0.2871 | 0.3117 |
| frozen_final minus clip_baseline | correct_instance | -0.0062 | -0.0154 | 0.0024 |
| frozen_final minus clip_baseline | p_at_05 | 0.2595 | 0.2490 | 0.2706 |
| frozen_final minus smart | iou | 0.0019 | 0.0004 | 0.0034 |
| frozen_final minus smart | harm_rate | -0.0609 | -0.0678 | -0.0545 |
| frozen_final minus smart | correct_instance | 0.0053 | 0.0016 | 0.0089 |
| frozen_final minus smart | p_at_05 | -0.0066 | -0.0087 | -0.0045 |
| sam_score minus clip_baseline | correct_instance_multiple_only | -0.0160 | -0.0260 | -0.0059 |
| smart minus sam_score | correct_instance_multiple_only | 0.0045 | -0.0003 | 0.0095 |
| frozen_final minus sam_score | correct_instance_multiple_only | 0.0098 | 0.0039 | 0.0157 |
| frozen_final minus clip_baseline | correct_instance_multiple_only | -0.0062 | -0.0154 | 0.0024 |

### Strict mIoU intervals

| method | iou | iou_ci_low | iou_ci_high |
| --- | --- | --- | --- |
| clip_baseline | 0.2172 | 0.2128 | 0.2215 |
| sam_score | 0.3199 | 0.3115 | 0.3281 |
| smart | 0.3473 | 0.3393 | 0.3555 |
| frozen_final | 0.3492 | 0.3414 | 0.3573 |
| center_score | 0.0948 | 0.0885 | 0.1016 |

## Binary quality and intervals

| method | iou | iou_ci_low | iou_ci_high | dice | precision | recall | pacc | p_at_05 | p_at_07 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| clip_baseline | 0.2173 | 0.2134 | 0.2212 | 0.3343 | 0.2670 | 0.6401 | 0.7738 | 0.0507 | 0.0015 |
| sam_score | 0.3183 | 0.3104 | 0.3254 | 0.4037 | 0.4188 | 0.4977 | 0.8394 | 0.2861 | 0.1638 |
| smart | 0.3451 | 0.3376 | 0.3525 | 0.4358 | 0.4210 | 0.5604 | 0.8484 | 0.3147 | 0.1829 |
| frozen_final | 0.3472 | 0.3396 | 0.3544 | 0.4420 | 0.4241 | 0.5831 | 0.8538 | 0.3080 | 0.1786 |
| center_score | 0.0963 | 0.0900 | 0.1026 | 0.1176 | 0.2079 | 0.1146 | 0.8096 | 0.0900 | 0.0669 |

## Paired comparisons

| comparison | metric | difference | ci_low | ci_high |
| --- | --- | --- | --- | --- |
| sam_score minus clip_baseline | iou | 0.1010 | 0.0944 | 0.1071 |
| sam_score minus clip_baseline | harm_rate | 0.4370 | 0.4247 | 0.4492 |
| sam_score minus clip_baseline | correct_instance | -0.0168 | -0.0266 | -0.0075 |
| sam_score minus clip_baseline | p_at_05 | 0.2354 | 0.2244 | 0.2461 |
| smart minus sam_score | iou | 0.0268 | 0.0231 | 0.0306 |
| smart minus sam_score | harm_rate | -0.0729 | -0.0801 | -0.0654 |
| smart minus sam_score | correct_instance | 0.0043 | -0.0006 | 0.0093 |
| smart minus sam_score | p_at_05 | 0.0286 | 0.0217 | 0.0356 |
| frozen_final minus sam_score | iou | 0.0289 | 0.0247 | 0.0330 |
| frozen_final minus sam_score | harm_rate | -0.1357 | -0.1453 | -0.1261 |
| frozen_final minus sam_score | correct_instance | 0.0099 | 0.0040 | 0.0157 |
| frozen_final minus sam_score | p_at_05 | 0.0219 | 0.0144 | 0.0292 |
| frozen_final minus clip_baseline | iou | 0.1298 | 0.1244 | 0.1355 |
| frozen_final minus clip_baseline | harm_rate | 0.3013 | 0.2900 | 0.3124 |
| frozen_final minus clip_baseline | correct_instance | -0.0069 | -0.0154 | 0.0013 |
| frozen_final minus clip_baseline | p_at_05 | 0.2572 | 0.2469 | 0.2677 |
| frozen_final minus smart | iou | 0.0021 | 0.0006 | 0.0036 |
| frozen_final minus smart | harm_rate | -0.0628 | -0.0695 | -0.0563 |
| frozen_final minus smart | correct_instance | 0.0056 | 0.0021 | 0.0093 |
| frozen_final minus smart | p_at_05 | -0.0068 | -0.0089 | -0.0047 |
| sam_score minus clip_baseline | correct_instance_multiple_only | -0.0168 | -0.0266 | -0.0075 |
| smart minus sam_score | correct_instance_multiple_only | 0.0043 | -0.0006 | 0.0093 |
| frozen_final minus sam_score | correct_instance_multiple_only | 0.0099 | 0.0040 | 0.0157 |
| frozen_final minus clip_baseline | correct_instance_multiple_only | -0.0069 | -0.0154 | 0.0013 |

2,000 paired whole-image resamples, seed 2026, preserving expressions per image and expression weighting. Intervals are not multiplicity-adjusted. All split and strict contrasts are in paired_comparisons.json. Harm contrasts compare each method with CLIP; direct final-versus-naive worsening is different.

## Harm versus CLIP

| method | improved | worsened | unchanged | delta_iou | median_delta | mean_positive_delta | mean_negative_delta | worst_decile_delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| clip_baseline | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| sam_score | 0.5501 | 0.4370 | 0.0129 | 0.1010 | 0.0383 | 0.2887 | -0.1324 | -0.2974 |
| smart | 0.6231 | 0.3641 | 0.0128 | 0.1278 | 0.0832 | 0.2704 | -0.1118 | -0.2413 |
| frozen_final | 0.6010 | 0.3013 | 0.0977 | 0.1298 | 0.0729 | 0.2709 | -0.1094 | -0.2200 |
| center_score | 0.1338 | 0.8523 | 0.0139 | -0.1210 | -0.1429 | 0.3436 | -0.1959 | -0.4607 |

## Smart/final directly versus naive

| method | improved | worsened | unchanged | delta_iou | median_delta | mean_positive_delta | mean_negative_delta | worst_decile_delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| smart | 0.2655 | 0.1569 | 0.5776 | 0.0268 | 0.0000 | 0.1890 | -0.1489 | -0.2280 |
| frozen_final | 0.3077 | 0.1607 | 0.5316 | 0.0289 | 0.0000 | 0.1857 | -0.1759 | -0.2712 |

## Instance disambiguation

| ambiguity | method | expressions | images | correct_instance | correct_instance_ci_low | correct_instance_ci_high | wrong_instance | instance_tie |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| multiple | clip_baseline | 9602 | 2600 | 0.6412 | 0.6306 | 0.6519 | 0.3574 | 0.0014 |
| multiple | sam_score | 9602 | 2600 | 0.6245 | 0.6134 | 0.6344 | 0.3427 | 0.0328 |
| multiple | smart | 9602 | 2600 | 0.6287 | 0.6179 | 0.6390 | 0.3428 | 0.0284 |
| multiple | frozen_final | 9602 | 2600 | 0.6343 | 0.6229 | 0.6448 | 0.3418 | 0.0238 |
| multiple | center_score | 9602 | 2600 | 0.3682 | 0.3573 | 0.3797 | 0.3592 | 0.2727 |

Target IoU must strictly exceed every other non-crowd same-category instance IoU. Ties are failures; single-instance cases, if any, require positive overlap. Empty groups are absent, not filtered. Category annotations can overlap: this is an overlap proxy, not manual semantic confirmation.

## Expression subgroups

| family | group | expressions | iou | correct_instance | p_at_05 | p_at_07 | worsened |
| --- | --- | --- | --- | --- | --- | --- | --- |
| length_group | long | 6087 | 0.3293 | 0.6225 | 0.2878 | 0.1636 | 0.3164 |
| length_group | medium | 2779 | 0.3641 | 0.6434 | 0.3213 | 0.1914 | 0.2904 |
| length_group | short | 736 | 0.4307 | 0.6984 | 0.4239 | 0.2541 | 0.2174 |
| spatial | False | 6822 | 0.3727 | 0.6771 | 0.3445 | 0.2106 | 0.2927 |
| spatial | True | 2780 | 0.2844 | 0.5295 | 0.2183 | 0.1000 | 0.3223 |
| attribute | False | 4879 | 0.3181 | 0.6011 | 0.2673 | 0.1400 | 0.3216 |
| attribute | True | 4723 | 0.3772 | 0.6686 | 0.3500 | 0.2185 | 0.2803 |
| color | False | 5516 | 0.3232 | 0.6071 | 0.2739 | 0.1467 | 0.3205 |
| color | True | 4086 | 0.3794 | 0.6711 | 0.3539 | 0.2217 | 0.2753 |
| clothing | False | 8080 | 0.3409 | 0.6223 | 0.2968 | 0.1670 | 0.2991 |
| clothing | True | 1522 | 0.3801 | 0.6984 | 0.3673 | 0.2405 | 0.3127 |

All methods, standard/strict and split-specific results are in subgroup_results.csv. Exactly the previous predefined word-boundary lexicons and length bins are used. Groups overlap and confound target size/category/scene difficulty; they do not establish causal language effects.

## Candidate and fallback diagnostics

| metric | mean | ci_low | ci_high |
| --- | --- | --- | --- |
| candidate_oracle | 0.4104 | 0.4025 | 0.4184 |
| candidate_fallback_oracle | 0.4436 | 0.4362 | 0.4511 |
| selected_fallback_oracle | 0.3858 | 0.3788 | 0.3927 |
| no_candidate_at_05 | 0.5880 | 0.5756 | 0.5996 |
| selector_regret | 0.0653 | 0.0624 | 0.0685 |
| fallback_rejects_useful | 0.0221 | 0.0189 | 0.0255 |
| fallback_prevents_harm | 0.0628 | 0.0563 | 0.0695 |
| fallback_accepts_harm | 0.3013 | 0.2900 | 0.3124 |

GT-only oracles do not enter inference. Poor candidates may reflect CLIP guidance, crop limits or SAM; these statistics do not identify a unique causal bottleneck. Fallback labels compare smart-selected SAM to CLIP before fallback.

Expression length: mean 8.23 words; median 8.0 words. Bins unchanged: short <=3, medium 4-6, long >=7.

## Thirteen research answers

1. Naive minus CLIP mIoU: +0.1010 [+0.0944, +0.1071]; positive change established by this interval.
2. Smart minus naive mIoU: +0.0268 [+0.0231, +0.0306]; positive change established by this interval.
3. Fallback final-minus-smart harm: -0.0628 [-0.0695, -0.0563]; final-minus-smart mIoU: +0.0021 [+0.0006, +0.0036]. Naive/final harm versus CLIP: 0.4370/0.3013.
4. No RefCOCOg training or tuning occurred. Final minus naive: +0.0289 [+0.0247, +0.0330]; positive change established by this interval. Standard and strict results above delimit generalization claims. CASR's selector/fallback were fitted earlier on COCO development.
5. CASR final mIoU: 0.3472, 95% CI [0.3396, 0.3544]. Final minus CLIP: +0.1298 [+0.1244, +0.1355].
6. Multiple-instance correct selection: 0.6343; wrong 0.3418; tie 0.0238, over 9602 expressions. All-expression correct selection: 0.6343.
7. Long-expression CASR mIoU 0.3293; correct-instance 0.6225, over 6087 expressions.
8. Within-test length-group means: long: 0.3293 (n=6087); medium: 0.3641 (n=2779); short: 0.4307 (n=736). Differences are descriptive, with confounded scene/target difficulty; they do not establish a causal length effect or a controlled worsening versus earlier datasets.
9. No SAM candidate reaches IoU 0.5 in 0.5880 of expressions.
10. Smart-selector regret: 0.0653; candidate oracle 0.4104. These are post-hoc upper bounds, not deployable gains.
11. Ambiguous-only failure/tie fraction: 0.3657. Correct-instance is an overlap proxy, not manual semantic verification.
12. Earlier smart-minus-naive gains were about +0.0209 on both RefCOCO and RefCOCO+; here +0.0268 [+0.0231, +0.0306]. Current final-minus-naive correct-instance contrast: +0.0099 [+0.0040, +0.0157]. Interpret alongside harm, length groups and overlap; cross-dataset raw means are not controlled comparisons.
13. This completes the requested three referring-expression evaluations. It is reasonable to stop adding closely related datasets and next assess external baselines and paper framing, subject to review; these evaluations alone do not establish competitive performance or solve instance disambiguation. No external baseline, method change or new dataset was run.

## Cross-dataset descriptive context

| dataset | CLIP | Naive SAM | Smart | CASR final |
| --- | --- | --- | --- | --- |
| Dense heldout COCO | 0.2699 | 0.3949 | 0.4291 | 0.4335 |
| RefCOCO | 0.2034 | 0.2872 | 0.3081 | 0.3071 |
| RefCOCO+ | 0.2079 | 0.3016 | 0.3224 | 0.3216 |
| RefCOCOg UMD test | 0.2173 | 0.3183 | 0.3451 | 0.3472 |

Raw mIoU values are not directly comparable: dense COCO uses category unions in a center crop, whereas referring-expression evaluation uses full-image instances; language, target distributions and overlap differ. This is not a causal language comparison.

## Secondary unchanged CLIP crop

| method | iou | dice | p_at_05 | p_at_07 | worsened | correct_instance |
| --- | --- | --- | --- | --- | --- | --- |
| clip_baseline | 0.2311 | 0.3501 | 0.0711 | 0.0052 | 0.0000 | 0.6436 |
| sam_score | 0.3200 | 0.4046 | 0.2889 | 0.1662 | 0.4444 | 0.6171 |
| smart | 0.3482 | 0.4375 | 0.3209 | 0.1875 | 0.3705 | 0.6218 |
| frozen_final | 0.3513 | 0.4451 | 0.3151 | 0.1831 | 0.3067 | 0.6297 |
| center_score | 0.0982 | 0.1195 | 0.0923 | 0.0690 | 0.8540 | 0.3574 |

Crop-invisible target expressions: 128; retained, with secondary empty-target IoU/Dice defined as zero. SAM sees original RGB; CLIP/fallback is limited to its original crop.

## Runtime, integrity and reproduction

Inference including smoke/hashing/loading/saving: 1567.4 seconds; evaluation: 164.1 seconds. Not pure GPU runtime. All 9602 expressions retained. Zero/constant attribution maps 0/0; empty candidates 0. Exactly 20 technical smoke expressions reused; no smoke GT scoring. Final hash and decision replay audit: audit.json. [Representative panels](examples.html).

Commands: download_refcocog.py; prepare_refcocog.py; run_refcocog.py --smoke; run_refcocog.py --resume; evaluate_refcocog.py; report_refcocog.py; visualize_refcocog.py; audit_refcocog.py. Use existing pinned environment. All scripts are under scripts/. No completed experiment may be overwritten. Protocol: ../../REFCOCOG_PLAN.md; reproduction notes: ../../REFCOCOG_REPRODUCTION.md.
