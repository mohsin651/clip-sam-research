# Frozen smart SAM: image-disjoint heldout results

## Primary method comparison

| method | iou | delta_iou | improved | worsened | semantic_success | precise_wrong_object |
| --- | --- | --- | --- | --- | --- | --- |
| clip_baseline | 0.2699 | 0.0000 | 0.0000 | 0.0000 | 0.6400 | 0.0150 |
| sam_score | 0.3949 | 0.1250 | 0.5880 | 0.4110 | 0.7320 | 0.0780 |
| coverage | 0.3776 | 0.1078 | 0.5930 | 0.4060 | 0.6810 | 0.1060 |
| density | 0.3907 | 0.1209 | 0.6020 | 0.3970 | 0.7560 | 0.0600 |
| smart | 0.4291 | 0.1592 | 0.6580 | 0.3410 | 0.7340 | 0.0830 |
| sam_score+logistic | 0.3963 | 0.1264 | 0.4320 | 0.1270 | 0.7200 | 0.0450 |
| smart+logistic | 0.4196 | 0.1498 | 0.4930 | 0.1030 | 0.7260 | 0.0440 |
| frozen_final | 0.4335 | 0.1636 | 0.6370 | 0.2620 | 0.7370 | 0.0740 |
| center_score | 0.1019 | -0.1680 | 0.1230 | 0.8770 | 0.3520 | 0.1040 |

Values improved/worsened and semantic/wrong-object columns are fractions. IoU is pair-macro foreground IoU on the unchanged CLIP crop, not official COCO AP. All failures and unchanged fallback cases remain.

## Binary quality and image-bootstrap intervals

| method | iou | iou_ci_low | iou_ci_high | dice | precision | recall | pacc |
| --- | --- | --- | --- | --- | --- | --- | --- |
| clip_baseline | 0.2699 | 0.2568 | 0.2836 | 0.3880 | 0.3395 | 0.7705 | 0.7378 |
| sam_score | 0.3949 | 0.3745 | 0.4158 | 0.4830 | 0.5746 | 0.5472 | 0.8201 |
| coverage | 0.3776 | 0.3592 | 0.3972 | 0.4685 | 0.4800 | 0.6280 | 0.7910 |
| density | 0.3907 | 0.3714 | 0.4115 | 0.4833 | 0.6021 | 0.5212 | 0.8506 |
| smart | 0.4291 | 0.4092 | 0.4496 | 0.5194 | 0.5713 | 0.6075 | 0.8419 |
| sam_score+logistic | 0.3963 | 0.3765 | 0.4164 | 0.4992 | 0.5367 | 0.6910 | 0.8068 |
| smart+logistic | 0.4196 | 0.4003 | 0.4392 | 0.5232 | 0.5407 | 0.7148 | 0.8192 |
| frozen_final | 0.4335 | 0.4136 | 0.4542 | 0.5293 | 0.5764 | 0.6407 | 0.8469 |
| center_score | 0.1019 | 0.0884 | 0.1156 | 0.1284 | 0.2804 | 0.1225 | 0.7522 |

## Paired comparisons

| comparison | metric | difference | ci_low | ci_high |
| --- | --- | --- | --- | --- |
| smart minus sam_score | iou | 0.0342 | 0.0236 | 0.0444 |
| smart minus sam_score | semantic_success | 0.0020 | -0.0130 | 0.0160 |
| smart minus sam_score | precise_wrong_object | 0.0050 | -0.0060 | 0.0160 |
| smart minus sam_score | worsened | -0.0700 | -0.0890 | -0.0500 |
| smart+logistic minus smart | iou | -0.0094 | -0.0183 | -0.0008 |
| smart+logistic minus smart | semantic_success | -0.0080 | -0.0280 | 0.0120 |
| smart+logistic minus smart | precise_wrong_object | -0.0390 | -0.0520 | -0.0260 |
| smart+logistic minus smart | worsened | -0.2380 | -0.2640 | -0.2100 |
| sam_score+logistic minus sam_score | iou | 0.0014 | -0.0080 | 0.0105 |
| sam_score+logistic minus sam_score | semantic_success | -0.0120 | -0.0330 | 0.0080 |
| sam_score+logistic minus sam_score | precise_wrong_object | -0.0330 | -0.0450 | -0.0220 |
| sam_score+logistic minus sam_score | worsened | -0.2840 | -0.3130 | -0.2560 |
| frozen_final minus smart | iou | 0.0044 | 0.0006 | 0.0082 |
| frozen_final minus smart | semantic_success | 0.0030 | -0.0070 | 0.0140 |
| frozen_final minus smart | precise_wrong_object | -0.0090 | -0.0150 | -0.0040 |
| frozen_final minus smart | worsened | -0.0790 | -0.0960 | -0.0620 |
| frozen_final minus sam_score | iou | 0.0386 | 0.0274 | 0.0497 |
| frozen_final minus sam_score | semantic_success | 0.0050 | -0.0120 | 0.0220 |
| frozen_final minus sam_score | precise_wrong_object | -0.0040 | -0.0150 | 0.0070 |
| frozen_final minus sam_score | worsened | -0.1490 | -0.1740 | -0.1240 |
| frozen_final minus clip_baseline | iou | 0.1636 | 0.1483 | 0.1799 |
| frozen_final minus clip_baseline | semantic_success | 0.0970 | 0.0710 | 0.1230 |
| frozen_final minus clip_baseline | precise_wrong_object | 0.0590 | 0.0440 | 0.0750 |
| frozen_final minus clip_baseline | worsened | 0.2620 | 0.2350 | 0.2900 |

## Harm

| method | improved | worsened | unchanged | mean_positive_delta | mean_negative_delta | median_delta | worst_decile_delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| clip_baseline | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| sam_score | 0.5880 | 0.4110 | 0.0010 | 0.3108 | -0.1404 | 0.0618 | -0.3315 |
| coverage | 0.5930 | 0.4060 | 0.0010 | 0.2561 | -0.1086 | 0.0527 | -0.2530 |
| density | 0.6020 | 0.3970 | 0.0010 | 0.2972 | -0.1462 | 0.0843 | -0.3293 |
| smart | 0.6580 | 0.3410 | 0.0010 | 0.3013 | -0.1146 | 0.1168 | -0.2542 |
| sam_score+logistic | 0.4320 | 0.1270 | 0.4410 | 0.3397 | -0.1603 | 0.0000 | -0.1964 |
| smart+logistic | 0.4930 | 0.1030 | 0.4040 | 0.3263 | -0.1079 | 0.0000 | -0.1111 |
| frozen_final | 0.6370 | 0.2620 | 0.1010 | 0.3033 | -0.1130 | 0.1044 | -0.2221 |
| center_score | 0.1230 | 0.8770 | 0.0000 | 0.2611 | -0.2282 | -0.1455 | -0.5912 |

Worst-decile delta is the mean of the lowest 10% of per-target IoU changes; positive/negative means are conditional on improvement/harm. The underlying CSV also retains confidence intervals for harm and improvement fractions and all secondary mean metrics.

## Frozen baseline subgroups

| method | group | n_pairs | baseline_iou | iou | delta_iou | delta_ci_low | delta_ci_high | iou_ci_low | iou_ci_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| clip_baseline | A_correct_target_coarse_mask | 363 | 0.2950 | 0.2950 | 0.0000 | 0.0000 | 0.0000 | 0.2814 | 0.3080 |
| clip_baseline | B_wrong_or_unconfirmed_target | 486 | 0.1439 | 0.1439 | 0.0000 | 0.0000 | 0.0000 | 0.1331 | 0.1556 |
| clip_baseline | C_good_localization | 151 | 0.6149 | 0.6149 | 0.0000 | 0.0000 | 0.0000 | 0.6030 | 0.6264 |
| sam_score | A_correct_target_coarse_mask | 363 | 0.2950 | 0.4876 | 0.1927 | 0.1610 | 0.2241 | 0.4566 | 0.5190 |
| sam_score | B_wrong_or_unconfirmed_target | 486 | 0.1439 | 0.2442 | 0.1003 | 0.0778 | 0.1244 | 0.2200 | 0.2712 |
| sam_score | C_good_localization | 151 | 0.6149 | 0.6572 | 0.0423 | -0.0016 | 0.0875 | 0.6123 | 0.7017 |
| coverage | A_correct_target_coarse_mask | 363 | 0.2950 | 0.4810 | 0.1860 | 0.1586 | 0.2136 | 0.4511 | 0.5114 |
| coverage | B_wrong_or_unconfirmed_target | 486 | 0.1439 | 0.1944 | 0.0505 | 0.0327 | 0.0693 | 0.1744 | 0.2175 |
| coverage | C_good_localization | 151 | 0.6149 | 0.7189 | 0.1039 | 0.0704 | 0.1396 | 0.6845 | 0.7515 |
| density | A_correct_target_coarse_mask | 363 | 0.2950 | 0.4486 | 0.1536 | 0.1226 | 0.1846 | 0.4199 | 0.4791 |
| density | B_wrong_or_unconfirmed_target | 486 | 0.1439 | 0.2531 | 0.1093 | 0.0878 | 0.1336 | 0.2300 | 0.2796 |
| density | C_good_localization | 151 | 0.6149 | 0.6944 | 0.0794 | 0.0350 | 0.1227 | 0.6483 | 0.7397 |
| smart | A_correct_target_coarse_mask | 363 | 0.2950 | 0.5239 | 0.2290 | 0.1994 | 0.2588 | 0.4947 | 0.5536 |
| smart | B_wrong_or_unconfirmed_target | 486 | 0.1439 | 0.2526 | 0.1087 | 0.0867 | 0.1315 | 0.2279 | 0.2813 |
| smart | C_good_localization | 151 | 0.6149 | 0.7692 | 0.1542 | 0.1230 | 0.1851 | 0.7363 | 0.8014 |
| sam_score+logistic | A_correct_target_coarse_mask | 363 | 0.2950 | 0.4729 | 0.1779 | 0.1503 | 0.2077 | 0.4450 | 0.5013 |
| sam_score+logistic | B_wrong_or_unconfirmed_target | 486 | 0.1439 | 0.2477 | 0.1038 | 0.0839 | 0.1243 | 0.2247 | 0.2729 |
| sam_score+logistic | C_good_localization | 151 | 0.6149 | 0.6902 | 0.0753 | 0.0384 | 0.1128 | 0.6532 | 0.7279 |
| smart+logistic | A_correct_target_coarse_mask | 363 | 0.2950 | 0.5027 | 0.2077 | 0.1798 | 0.2357 | 0.4751 | 0.5308 |
| smart+logistic | B_wrong_or_unconfirmed_target | 486 | 0.1439 | 0.2475 | 0.1036 | 0.0850 | 0.1235 | 0.2259 | 0.2716 |
| smart+logistic | C_good_localization | 151 | 0.6149 | 0.7738 | 0.1589 | 0.1335 | 0.1840 | 0.7460 | 0.7997 |
| frozen_final | A_correct_target_coarse_mask | 363 | 0.2950 | 0.5245 | 0.2295 | 0.2013 | 0.2591 | 0.4961 | 0.5540 |
| frozen_final | B_wrong_or_unconfirmed_target | 486 | 0.1439 | 0.2615 | 0.1176 | 0.0963 | 0.1401 | 0.2375 | 0.2885 |
| frozen_final | C_good_localization | 151 | 0.6149 | 0.7683 | 0.1533 | 0.1224 | 0.1838 | 0.7349 | 0.8002 |
| center_score | A_correct_target_coarse_mask | 363 | 0.2950 | 0.1216 | -0.1733 | -0.1974 | -0.1481 | 0.0981 | 0.1480 |
| center_score | B_wrong_or_unconfirmed_target | 486 | 0.1439 | 0.0348 | -0.1091 | -0.1245 | -0.0946 | 0.0233 | 0.0484 |
| center_score | C_good_localization | 151 | 0.6149 | 0.2702 | -0.3448 | -0.3988 | -0.2887 | 0.2151 | 0.3265 |

## Post-hoc oracle context

| oracle | iou | ci_low | ci_high |
| --- | --- | --- | --- |
| candidate_oracle | 0.4878 | 0.4672 | 0.5097 |
| selected_sam_or_clip_oracle | 0.4681 | 0.4497 | 0.4873 |
| naive_sam_or_clip_oracle | 0.4526 | 0.4340 | 0.4726 |
| candidate_plus_fallback_oracle | 0.5181 | 0.4984 | 0.5377 |

| selector | best_candidate_accuracy |
| --- | --- |
| sam_score | 0.4390 |
| coverage | 0.5030 |
| density | 0.5380 |
| smart | 0.5810 |

The selected-SAM-or-CLIP oracle uses the frozen selector before fallback; the naive oracle uses SAM-score selection. These upper bounds do not affect the method.

## Ten research answers

1. The frozen smart selector changes mIoU by +0.0342 [95% CI +0.0236, +0.0444] on 500 unseen images. This is the prospective generalization comparison; prior development results are not test evidence.
2. Naive SAM achieves 0.3949 on this cohort; smart selection achieves 0.4291. The earlier 0.4005 came from different images and is not a paired comparator.
3. The frozen density-threshold fallback changes mIoU by +0.0044 [95% CI +0.0006, +0.0082]. Frozen final mIoU is 0.4335. The separately retained logistic fallback changes it by -0.0094 [95% CI -0.0183, -0.0008].
4. Naive harm is 41.1%; frozen final harm is 26.2%, a reduction of 14.9 percentage points. Unchanged fallback cases count in the denominator.
5. Semantic success changes from 73.2% to 73.7%; paired difference +0.0050 [95% CI -0.0120, +0.0220]. The interval includes zero: a semantic-success improvement over naive SAM is not established.
6. Precise-wrong-object proxy changes from 7.8% to 7.4%; paired difference -0.0040 [95% CI -0.0150, +0.0070]. The interval includes zero: a reduction relative to naive SAM is not established. The final rate remains higher than CLIP's 1.5%. These are category-union overlap proxies, not manual semantic labels.
7. 49.2% of pairs have no candidate reaching IoU 0.5. Other remaining failures include wrong-object selection, partial instance coverage, and rejecting useful SAM masks. Outcome-selected panels illustrate these mechanisms; their frequency is not inferred from the panels.
8. The combined oracle is 0.5181, leaving 0.0846 mIoU above frozen final. Candidate-only oracle is 0.4878; selected-mask fallback oracle is 0.4681.
9. Candidate-selection headroom above smart selection is 0.0588; perfect fallback headroom above smart selection is 0.0391. Candidate choice remains the larger of these two decision gaps, but inadequate candidate masks are also a major limitation. These overlapping oracle gains do not add and do not prove a practical model can recover them.
10. The heldout evidence supports presenting this frozen small selector/fallback as a promising proposed method, with the reported tradeoffs and limitations. Broader datasets and stronger comparisons remain necessary for a paper-level claim.

## Group benefit and retained failures

A_correct_target_coarse_mask: naive SAM mIoU 0.4876 -> final 0.5245; harm 32.0% -> 20.9%.

B_wrong_or_unconfirmed_target: naive SAM mIoU 0.2442 -> final 0.2615; harm 49.4% -> 32.7%.

C_good_localization: naive SAM mIoU 0.6572 -> final 0.7683; harm 36.4% -> 17.9%.

Group A retains its large spatial-refinement gain. Group C improves substantially relative to naive SAM and has fewer harmed pairs, although harm is not eliminated. Logistic fallback achieves lower harm than the frozen density fallback but lower mIoU; it was not substituted after seeing test results.

Inference runtime: 210.6 seconds; evaluation: 15.8 seconds. All 1,000 pairs retained; zero maps: 0; empty candidates: 0. These times exclude development, initial imports and report rendering.

## Reproducibility and limitations

All 500 test images are disjoint from development. Models and inference source hashes were frozen before cohort selection. No heldout labels entered features, calibration, candidate choice or fallback. Inference completed before evaluation loaded GT; all saved prediction hashes were checked. The original CLIP and SAM checkpoints and implementations are unchanged. Only the small linear ranker, comparison logistic gates and density thresholds were fitted on development. No heldout tuning, exclusions or category filtering occurred.

Intervals use 2,000 paired whole-image resamples, preserving both prompts; subgroup denominators retain pair weighting. Intervals are not multiplicity-adjusted. This is one selected dense-scene COCO cohort. Category unions include all instances; a point-prompted SAM candidate may cover one instance. Overlapping annotations can affect semantic proxies. All binary quality metrics use the same crop and threshold; no artificial continuous SAM AP is reported.

[Representative examples](examples.html). Frozen method: [FROZEN_METHOD.md](../frozen/FROZEN_METHOD.md). Full per-target predictions, candidate metrics, model parameters, hashes and audit files are retained.
