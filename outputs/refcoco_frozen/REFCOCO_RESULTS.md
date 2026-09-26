# Frozen RefCOCO generalization results

This evaluation uses the unchanged COCO-trained selector and density fallback, unchanged CLIP attribution and SAM, and original referring expressions verbatim. No RefCOCO fitting, tuning, exclusions by difficulty or method repair occurred. Primary metrics use the specific referred instance over the full original image.

## Dataset, sources and overlap

Original RefCOCO UNC annotations from the [REFER project](https://github.com/lichengunc/refer), retrieved from the archived official ZIP linked in its [download issue](https://github.com/lichengunc/refer/issues/14#issuecomment-1258318183). Images are official COCO train2014 RGB files. The original host was unavailable; no different dataset was substituted. Complete split counts:

| split | images | target_instances | references | expressions |
| --- | --- | --- | --- | --- |
| testA | 750 | 1975 | 1975 | 5657 |
| testB | 750 | 1810 | 1810 | 5095 |
| train | 16994 | 42404 | 42404 | 120624 |
| val | 1500 | 3811 | 3811 | 10834 |

Evaluated testA + testB: 1500 images, 10752 expressions, 3785 target instances. testA and testB remain separate official splits; combined is their expression-weighted aggregate. Annotation coverage for competitors was verified against complete official COCO annotation ID sets.

Prior development overlap: 0 images; prior heldout overlap: 0 images. Strict cohort: {'images': 1500, 'expressions': 10752, 'target_instances': 3785}. The overlap audit was saved before inference. When overlap is zero, strict scores are identical and are not an independent replication.

## Primary combined-test results

| method | iou | dice | p_at_05 | improved | worsened | correct_instance |
| --- | --- | --- | --- | --- | --- | --- |
| clip_baseline | 0.2034 | 0.3172 | 0.0350 | 0.0000 | 0.0000 | 0.5080 |
| sam_score | 0.2872 | 0.3711 | 0.2400 | 0.5205 | 0.4630 | 0.5074 |
| smart | 0.3081 | 0.3973 | 0.2604 | 0.5886 | 0.3950 | 0.5115 |
| frozen_final | 0.3071 | 0.3992 | 0.2523 | 0.5564 | 0.3411 | 0.5115 |
| center_score | 0.0907 | 0.1132 | 0.0839 | 0.1315 | 0.8481 | 0.3074 |

All rates are fractions. P@0.5 is the expression fraction with instance IoU >=0.5. Correct-instance selection requires target IoU greater than every other non-crowd same-category instance IoU. Single-instance images require positive overlap; ambiguous-image results appear below.

## Official testA and testB results

| split | method | iou | dice | p_at_05 | improved | worsened | correct_instance |
| --- | --- | --- | --- | --- | --- | --- | --- |
| testA | clip_baseline | 0.1967 | 0.3080 | 0.0309 | 0.0000 | 0.0000 | 0.5570 |
| testA | sam_score | 0.2782 | 0.3567 | 0.2413 | 0.4867 | 0.4925 | 0.5483 |
| testA | smart | 0.3066 | 0.3866 | 0.2809 | 0.5496 | 0.4296 | 0.5586 |
| testA | frozen_final | 0.3078 | 0.3914 | 0.2740 | 0.5257 | 0.3732 | 0.5595 |
| testA | center_score | 0.0782 | 0.1006 | 0.0633 | 0.1174 | 0.8565 | 0.3081 |
| testB | clip_baseline | 0.2109 | 0.3275 | 0.0395 | 0.0000 | 0.0000 | 0.4536 |
| testB | sam_score | 0.2972 | 0.3871 | 0.2385 | 0.5580 | 0.4302 | 0.4620 |
| testB | smart | 0.3098 | 0.4093 | 0.2377 | 0.6320 | 0.3566 | 0.4593 |
| testB | frozen_final | 0.3064 | 0.4079 | 0.2283 | 0.5904 | 0.3056 | 0.4583 |
| testB | center_score | 0.1046 | 0.1271 | 0.1068 | 0.1472 | 0.8389 | 0.3066 |

## Strict non-overlap, combined tests

| method | iou | dice | p_at_05 | improved | worsened | correct_instance |
| --- | --- | --- | --- | --- | --- | --- |
| clip_baseline | 0.2034 | 0.3172 | 0.0350 | 0.0000 | 0.0000 | 0.5080 |
| sam_score | 0.2872 | 0.3711 | 0.2400 | 0.5205 | 0.4630 | 0.5074 |
| smart | 0.3081 | 0.3973 | 0.2604 | 0.5886 | 0.3950 | 0.5115 |
| frozen_final | 0.3071 | 0.3992 | 0.2523 | 0.5564 | 0.3411 | 0.5115 |
| center_score | 0.0907 | 0.1132 | 0.0839 | 0.1315 | 0.8481 | 0.3074 |

## Binary metrics and mIoU intervals

| method | iou | iou_ci_low | iou_ci_high | dice | precision | recall | pacc | p_at_07 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| clip_baseline | 0.2034 | 0.1988 | 0.2078 | 0.3172 | 0.2545 | 0.5591 | 0.7595 | 0.0016 |
| sam_score | 0.2872 | 0.2786 | 0.2955 | 0.3711 | 0.3897 | 0.4577 | 0.8301 | 0.1354 |
| smart | 0.3081 | 0.2993 | 0.3167 | 0.3973 | 0.3919 | 0.5107 | 0.8348 | 0.1472 |
| frozen_final | 0.3071 | 0.2985 | 0.3157 | 0.3992 | 0.3919 | 0.5200 | 0.8353 | 0.1439 |
| center_score | 0.0907 | 0.0840 | 0.0975 | 0.1132 | 0.2056 | 0.1099 | 0.8223 | 0.0549 |

## Paired comparisons, original image

| comparison | metric | difference | ci_low | ci_high |
| --- | --- | --- | --- | --- |
| sam_score minus clip_baseline | iou | 0.0838 | 0.0773 | 0.0905 |
| sam_score minus clip_baseline | harm_rate | 0.4630 | 0.4491 | 0.4769 |
| sam_score minus clip_baseline | correct_instance | -0.0006 | -0.0103 | 0.0088 |
| smart minus sam_score | iou | 0.0209 | 0.0167 | 0.0252 |
| smart minus sam_score | harm_rate | -0.0680 | -0.0767 | -0.0598 |
| smart minus sam_score | correct_instance | 0.0041 | -0.0013 | 0.0097 |
| frozen_final minus sam_score | iou | 0.0199 | 0.0155 | 0.0244 |
| frozen_final minus sam_score | harm_rate | -0.1218 | -0.1319 | -0.1114 |
| frozen_final minus sam_score | correct_instance | 0.0041 | -0.0025 | 0.0109 |
| frozen_final minus clip_baseline | iou | 0.1037 | 0.0978 | 0.1095 |
| frozen_final minus clip_baseline | harm_rate | 0.3411 | 0.3283 | 0.3543 |
| frozen_final minus clip_baseline | correct_instance | 0.0035 | -0.0047 | 0.0116 |
| frozen_final minus smart | iou | -0.0010 | -0.0025 | 0.0005 |
| frozen_final minus smart | harm_rate | -0.0539 | -0.0599 | -0.0479 |
| frozen_final minus smart | correct_instance | 0.0000 | -0.0034 | 0.0036 |
| sam_score minus clip_baseline | correct_instance_multiple_only | -0.0006 | -0.0103 | 0.0088 |
| smart minus sam_score | correct_instance_multiple_only | 0.0041 | -0.0013 | 0.0097 |
| frozen_final minus sam_score | correct_instance_multiple_only | 0.0041 | -0.0025 | 0.0109 |
| frozen_final minus clip_baseline | correct_instance_multiple_only | 0.0035 | -0.0047 | 0.0116 |

Harm-rate contrasts compare each method's fraction with IoU below CLIP; they are not the fraction with final IoU below naive SAM. All contrasts are paired by sentence and bootstrapped by image. Separate official split and strict-cohort contrasts are in paired_comparisons.json.

## Harm versus CLIP

| method | improved | worsened | unchanged | delta_iou | median_delta | mean_positive_delta | mean_negative_delta | worst_decile_delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| clip_baseline | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| sam_score | 0.5205 | 0.4630 | 0.0166 | 0.0838 | 0.0124 | 0.2662 | -0.1183 | -0.2698 |
| smart | 0.5886 | 0.3950 | 0.0164 | 0.1047 | 0.0562 | 0.2447 | -0.0996 | -0.2199 |
| frozen_final | 0.5564 | 0.3411 | 0.1025 | 0.1037 | 0.0380 | 0.2463 | -0.0977 | -0.2048 |
| center_score | 0.1315 | 0.8481 | 0.0204 | -0.1127 | -0.1360 | 0.3302 | -0.1841 | -0.4257 |

## Final directly versus naive SAM

| split | improved | worsened | unchanged | delta_iou | median_delta | mean_positive_delta | mean_negative_delta | worst_decile_delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| combined | 0.2950 | 0.1863 | 0.5187 | 0.0199 | 0.0000 | 0.1766 | -0.1729 | -0.2981 |
| testA | 0.2959 | 0.1628 | 0.5413 | 0.0296 | 0.0000 | 0.1902 | -0.1643 | -0.2598 |
| testB | 0.2940 | 0.2124 | 0.4936 | 0.0091 | 0.0000 | 0.1613 | -0.1803 | -0.3321 |

## Instance disambiguation

| ambiguity | method | expressions | images | iou | correct_instance | correct_instance_ci_low | correct_instance_ci_high | wrong_instance | instance_tie |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| multiple | clip_baseline | 10752 | 1500 | 0.2034 | 0.5080 | 0.4964 | 0.5194 | 0.4918 | 0.0002 |
| multiple | sam_score | 10752 | 1500 | 0.2872 | 0.5074 | 0.4948 | 0.5199 | 0.4797 | 0.0128 |
| multiple | smart | 10752 | 1500 | 0.3081 | 0.5115 | 0.4989 | 0.5240 | 0.4764 | 0.0121 |
| multiple | frozen_final | 10752 | 1500 | 0.3071 | 0.5115 | 0.4989 | 0.5238 | 0.4777 | 0.0108 |
| multiple | center_score | 10752 | 1500 | 0.0907 | 0.3074 | 0.2964 | 0.3179 | 0.5330 | 0.1596 |

Competitors are all other non-crowd instances of the same category, including instances not selected as referring targets. Ties, including zero-overlap ties, are not successes. Same-category instance counts are measured in the full image. Crowd regions are not individual-instance competitors. This is a mask-overlap proxy, not manual confirmation of language understanding.

## Expression subgroups

| family | group | method | expressions | iou | p_at_05 | correct_instance | worsened |
| --- | --- | --- | --- | --- | --- | --- | --- |
| length_group | short | clip_baseline | 6452 | 0.2286 | 0.0496 | 0.5648 | 0.0000 |
| length_group | short | sam_score | 6452 | 0.3236 | 0.2784 | 0.5626 | 0.4375 |
| length_group | short | smart | 6452 | 0.3498 | 0.3081 | 0.5640 | 0.3628 |
| length_group | short | frozen_final | 6452 | 0.3496 | 0.3004 | 0.5663 | 0.3111 |
| length_group | short | center_score | 6452 | 0.1082 | 0.1018 | 0.3554 | 0.8354 |
| length_group | medium | clip_baseline | 3401 | 0.1721 | 0.0156 | 0.4325 | 0.0000 |
| length_group | medium | sam_score | 3401 | 0.2425 | 0.1905 | 0.4369 | 0.4896 |
| length_group | medium | smart | 3401 | 0.2551 | 0.1988 | 0.4455 | 0.4331 |
| length_group | medium | frozen_final | 3401 | 0.2536 | 0.1905 | 0.4431 | 0.3781 |
| length_group | medium | center_score | 3401 | 0.0667 | 0.0582 | 0.2440 | 0.8650 |
| length_group | long | clip_baseline | 899 | 0.1413 | 0.0033 | 0.3860 | 0.0000 |
| length_group | long | sam_score | 899 | 0.1953 | 0.1513 | 0.3782 | 0.5451 |
| length_group | long | smart | 899 | 0.2094 | 0.1513 | 0.3849 | 0.4816 |
| length_group | long | frozen_final | 899 | 0.2044 | 0.1413 | 0.3771 | 0.4171 |
| length_group | long | center_score | 899 | 0.0559 | 0.0523 | 0.2024 | 0.8754 |
| spatial | False | clip_baseline | 4859 | 0.2439 | 0.0593 | 0.6454 | 0.0000 |
| spatial | False | sam_score | 4859 | 0.3574 | 0.3229 | 0.6452 | 0.4093 |
| spatial | False | smart | 4859 | 0.3889 | 0.3651 | 0.6528 | 0.3311 |
| spatial | False | frozen_final | 4859 | 0.3887 | 0.3562 | 0.6563 | 0.2892 |
| spatial | False | center_score | 4859 | 0.1277 | 0.1146 | 0.4174 | 0.8129 |
| spatial | True | clip_baseline | 5893 | 0.1700 | 0.0149 | 0.3947 | 0.0000 |
| spatial | True | sam_score | 5893 | 0.2294 | 0.1716 | 0.3939 | 0.5072 |
| spatial | True | smart | 5893 | 0.2414 | 0.1741 | 0.3950 | 0.4476 |
| spatial | True | frozen_final | 5893 | 0.2398 | 0.1666 | 0.3922 | 0.3840 |
| spatial | True | center_score | 5893 | 0.0602 | 0.0585 | 0.2167 | 0.8771 |
| attribute | False | clip_baseline | 8003 | 0.2009 | 0.0339 | 0.4912 | 0.0000 |
| attribute | False | sam_score | 8003 | 0.2666 | 0.2158 | 0.4732 | 0.4971 |
| attribute | False | smart | 8003 | 0.2890 | 0.2377 | 0.4786 | 0.4253 |
| attribute | False | frozen_final | 8003 | 0.2886 | 0.2294 | 0.4789 | 0.3637 |
| attribute | False | center_score | 8003 | 0.0940 | 0.0882 | 0.3173 | 0.8428 |
| attribute | True | clip_baseline | 2749 | 0.2109 | 0.0382 | 0.5569 | 0.0000 |
| attribute | True | sam_score | 2749 | 0.3473 | 0.3103 | 0.6071 | 0.3638 |
| attribute | True | smart | 2749 | 0.3638 | 0.3267 | 0.6075 | 0.3067 |
| attribute | True | frozen_final | 2749 | 0.3609 | 0.3190 | 0.6064 | 0.2754 |
| attribute | True | center_score | 2749 | 0.0809 | 0.0713 | 0.2786 | 0.8636 |
| color | False | clip_baseline | 8409 | 0.2028 | 0.0362 | 0.4949 | 0.0000 |
| color | False | sam_score | 8409 | 0.2700 | 0.2196 | 0.4791 | 0.4941 |
| color | False | smart | 8409 | 0.2939 | 0.2445 | 0.4858 | 0.4207 |
| color | False | frozen_final | 8409 | 0.2934 | 0.2362 | 0.4858 | 0.3606 |
| color | False | center_score | 8409 | 0.0953 | 0.0892 | 0.3199 | 0.8415 |
| color | True | clip_baseline | 2343 | 0.2058 | 0.0307 | 0.5548 | 0.0000 |
| color | True | sam_score | 2343 | 0.3491 | 0.3128 | 0.6090 | 0.3513 |
| color | True | smart | 2343 | 0.3592 | 0.3175 | 0.6039 | 0.3026 |
| color | True | frozen_final | 2343 | 0.3562 | 0.3103 | 0.6039 | 0.2714 |
| color | True | center_score | 2343 | 0.0741 | 0.0649 | 0.2625 | 0.8720 |
| clothing | False | clip_baseline | 9826 | 0.2027 | 0.0353 | 0.5020 | 0.0000 |
| clothing | False | sam_score | 9826 | 0.2826 | 0.2346 | 0.4957 | 0.4729 |
| clothing | False | smart | 9826 | 0.3031 | 0.2532 | 0.4999 | 0.4040 |
| clothing | False | frozen_final | 9826 | 0.3020 | 0.2446 | 0.4999 | 0.3475 |
| clothing | False | center_score | 9826 | 0.0930 | 0.0874 | 0.3114 | 0.8465 |
| clothing | True | clip_baseline | 926 | 0.2109 | 0.0313 | 0.5713 | 0.0000 |
| clothing | True | sam_score | 926 | 0.3368 | 0.2970 | 0.6317 | 0.3575 |
| clothing | True | smart | 926 | 0.3608 | 0.3369 | 0.6350 | 0.2991 |
| clothing | True | frozen_final | 926 | 0.3614 | 0.3348 | 0.6350 | 0.2732 |
| clothing | True | center_score | 926 | 0.0659 | 0.0464 | 0.2646 | 0.8650 |

Length groups: short <=3 words, medium 4-6, long >=7. Spatial and attribute groups use the predeclared word-boundary lexicons in REFCOCO_FROZEN_PLAN.md. These overlapping descriptive groups can confound expression length, object size and ambiguity; they do not establish causal effects.

## Post-hoc proposal/selection/fallback diagnostics

| metric | mean | ci_low | ci_high |
| --- | --- | --- | --- |
| candidate_oracle | 0.3808 | 0.3710 | 0.3903 |
| candidate_fallback_oracle | 0.4128 | 0.4035 | 0.4219 |
| selected_fallback_oracle | 0.3474 | 0.3393 | 0.3555 |
| no_candidate_at_05 | 0.6327 | 0.6192 | 0.6467 |
| selector_regret | 0.0727 | 0.0693 | 0.0764 |
| fallback_rejects_useful | 0.0323 | 0.0275 | 0.0370 |
| fallback_prevents_harm | 0.0539 | 0.0479 | 0.0599 |
| fallback_accepts_harm | 0.3411 | 0.3283 | 0.3543 |

No-candidate cases can reflect incorrect CLIP guidance, SAM proposal failure, missing instance coverage, or their combination. These diagnostics cannot causally assign every failure to CLIP versus SAM. Oracle values are analysis-only and never affect inference.

## Secondary unchanged CLIP crop

| method | iou | dice | p_at_05 | improved | worsened | correct_instance |
| --- | --- | --- | --- | --- | --- | --- |
| clip_baseline | 0.2163 | 0.3328 | 0.0503 | 0.0000 | 0.0000 | 0.5103 |
| sam_score | 0.2882 | 0.3713 | 0.2448 | 0.5098 | 0.4684 | 0.5047 |
| smart | 0.3112 | 0.3992 | 0.2705 | 0.5807 | 0.3973 | 0.5090 |
| frozen_final | 0.3116 | 0.4029 | 0.2632 | 0.5517 | 0.3410 | 0.5104 |
| center_score | 0.0924 | 0.1148 | 0.0866 | 0.1290 | 0.8490 | 0.3031 |

163 expressions have empty GT inside the center crop; none was removed. Their secondary IoU/Dice are defined as zero even when the prediction is empty. Primary full-image GT is nonempty. SAM sees original RGB; CLIP sees only its frozen crop. CLIP/fallback masks are zero outside that crop. This geometric limitation is retained, not repaired.

## Nine research answers

1. With raw referring expressions, the frozen final method reaches full-image instance mIoU 0.3071, versus CLIP 0.2034; final-minus-CLIP +0.1037 [95% CI +0.0978, +0.1095]. This measures useful localization, not proof of complete expression understanding.
2. Smart selection versus SAM-score changes mIoU by +0.0209 [95% CI +0.0167, +0.0252]. A positive interval excluding zero supports transfer of candidate selection; an interval spanning zero does not establish a gain.
3. Naive harm versus CLIP is 46.30%, smart harm 39.50%, final harm 34.11%. Final-minus-naive harm-rate contrast: -0.1218 [95% CI -0.1319, -0.1114]; final-minus-smart mIoU -0.0010 [95% CI -0.0025, +0.0005].
4. On images with multiple same-category instances, final correct-instance selection is 51.15% over 10752 expressions. Wrong-instance preference is 47.77%; zero-overlap/other ties remain failures.
5. Earlier COCO mIoU 0.4335 evaluated category unions in a center crop; this experiment evaluates a referred instance in a full image. The raw difference is not a controlled estimate of task difficulty. This study changes language and target granularity together and does not include a category-prompt intervention on the same samples.
6. Among predefined length groups, the lowest final mean IoU is for long expressions (0.2044). Spatial, attribute and ambiguity rows above describe other differences without post-hoc tuning or causal claims.
7. The oracle and failure-proxy table separates unavailable adequate candidates, selection regret and fallback mistakes. It does not cleanly separate CLIP semantic failures from SAM proposal failures; claiming a single proven cause would exceed this evidence.
8. Standard and strict non-overlap results are identical because no test image overlaps either prior cohort.
9. The unchanged method can be evaluated on RefCOCO+ and RefCOCOg as further generalization tests, with dataset-specific integrity checks. This does not imply competitive performance or authorize those runs. This experiment stops at RefCOCO pending user review.

## Integrity, examples and reproduce

Exactly 20 samples passed technical smoke checks without aggregate GT scoring. Their predictions were reused. Full inference completed before GT evaluation. All 10752 expressions were retained; zero/constant maps: 0/0; empty candidates: 0. Frozen source and parameter hashes match the original COCO freeze. Runtime: inference including smoke 1596.9s, evaluation 269.1s.

Intervals use 2,000 image-level paired resamples, seed 2026, expression-macro weighting and no multiplicity adjustment. The frozen method was fitted on COCO category data, so this is transfer of that fitted component, not a claim of an entirely untrained method.

[Representative examples](examples.html). Dataset hashes, manifests, overlap audit, saved candidates and raw expression text are retained.

Commands: `python scripts/prepare_refcoco_frozen.py`; `python -m pytest -q`; `python scripts/run_refcoco_frozen.py --smoke`; `python scripts/run_refcoco_frozen.py --resume`; `python scripts/evaluate_refcoco_frozen.py`; `python scripts/report_refcoco_frozen.py`; `python scripts/visualize_refcoco_frozen.py`. Completed artifacts must not be overwritten; use a separate destination for reproduction.
