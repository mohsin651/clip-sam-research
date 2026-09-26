# Development grouped cross-validation

These 500 images / 1,000 pairs have been inspected previously and are development data only. Estimates use five outer image folds, four inner folds for cross-fitted selector examples used to train fallback, and no hyperparameter search. The final gate fit uses all outer out-of-fold selected examples. Reported learned-method scores are out-of-fold, not full-fit training scores. Earlier feature exploration used this cohort, so these estimates are still exploratory.

## Candidate selection and fallback validation

| selector | fallback | selection_accuracy | iou | auc | balanced_accuracy | precision | recall |
| --- | --- | --- | --- | --- | --- | --- | --- |
| sam_score | always | 0.4620 | 0.4005 | nan | nan | nan | nan |
| sam_score | density_threshold | 0.4620 | 0.4146 | 0.7776 | 0.6762 | 0.6780 | 0.8911 |
| sam_score | logistic | 0.4620 | 0.4114 | 0.8039 | 0.7270 | 0.7627 | 0.7518 |
| coverage | always | 0.4990 | 0.3947 | nan | nan | nan | nan |
| density | always | 0.5280 | 0.3940 | nan | nan | nan | nan |
| smart | always | 0.5870 | 0.4419 | nan | nan | nan | nan |
| smart | density_threshold | 0.5870 | 0.4465 | 0.7602 | 0.6081 | 0.6994 | 0.9424 |
| smart | logistic | 0.5870 | 0.4383 | 0.7875 | 0.7348 | 0.8243 | 0.7601 |
| candidate_oracle | always | 1.0000 | 0.5021 | nan | nan | nan | nan |

Always-SAM classification control has constant score, AUC 0.5, balanced accuracy 0.5, recall 1.0 and precision equal to the improvement fraction. Full classification rows are retained in fallback_validation.csv. Density-threshold AUC uses continuous log-density; its binary decisions use fold-specific training-only thresholds.

## Primary method comparison

| method | iou | delta_iou | improved | worsened | semantic_success | precise_wrong_object |
| --- | --- | --- | --- | --- | --- | --- |
| clip_baseline | 0.2846 | 0.0000 | 0.0000 | 0.0000 | 0.6850 | 0.0130 |
| sam_score | 0.4005 | 0.1158 | 0.5600 | 0.4400 | 0.7580 | 0.0710 |
| sam_score+density_threshold | 0.4146 | 0.1299 | 0.4990 | 0.2370 | 0.7660 | 0.0520 |
| sam_score+logistic | 0.4114 | 0.1268 | 0.4210 | 0.1310 | 0.7520 | 0.0410 |
| coverage | 0.3947 | 0.1101 | 0.5900 | 0.4100 | 0.7150 | 0.1040 |
| density | 0.3940 | 0.1093 | 0.5690 | 0.4310 | 0.7870 | 0.0510 |
| smart | 0.4419 | 0.1573 | 0.6420 | 0.3580 | 0.7700 | 0.0750 |
| smart+density_threshold | 0.4465 | 0.1619 | 0.6050 | 0.2600 | 0.7710 | 0.0620 |
| smart+logistic | 0.4383 | 0.1537 | 0.4880 | 0.1040 | 0.7630 | 0.0460 |

Values improved/worsened and semantic/wrong-object columns are fractions. IoU is pair-macro foreground IoU on the unchanged CLIP crop, not official COCO AP. All failures and unchanged fallback cases remain.

## Binary quality and image-bootstrap intervals

| method | iou | iou_ci_low | iou_ci_high | dice | precision | recall | pacc |
| --- | --- | --- | --- | --- | --- | --- | --- |
| clip_baseline | 0.2846 | 0.2732 | 0.2965 | 0.4087 | 0.3602 | 0.7770 | 0.7430 |
| sam_score | 0.4005 | 0.3790 | 0.4217 | 0.4875 | 0.5819 | 0.5469 | 0.8007 |
| sam_score+density_threshold | 0.4146 | 0.3950 | 0.4337 | 0.5152 | 0.5872 | 0.6450 | 0.8234 |
| sam_score+logistic | 0.4114 | 0.3926 | 0.4305 | 0.5175 | 0.5513 | 0.7086 | 0.8053 |
| coverage | 0.3947 | 0.3757 | 0.4144 | 0.4881 | 0.4977 | 0.6436 | 0.7834 |
| density | 0.3940 | 0.3745 | 0.4152 | 0.4863 | 0.6173 | 0.5191 | 0.8393 |
| smart | 0.4419 | 0.4213 | 0.4642 | 0.5327 | 0.5861 | 0.6167 | 0.8325 |
| smart+density_threshold | 0.4465 | 0.4266 | 0.4672 | 0.5436 | 0.5889 | 0.6583 | 0.8402 |
| smart+logistic | 0.4383 | 0.4190 | 0.4586 | 0.5433 | 0.5585 | 0.7309 | 0.8180 |

## Paired comparisons

| comparison | metric | difference | ci_low | ci_high |
| --- | --- | --- | --- | --- |
| smart minus sam_score | iou | 0.0414 | 0.0299 | 0.0527 |
| smart minus sam_score | semantic_success | 0.0120 | -0.0020 | 0.0260 |
| smart minus sam_score | precise_wrong_object | 0.0040 | -0.0080 | 0.0150 |
| smart minus sam_score | worsened | -0.0820 | -0.1050 | -0.0600 |
| smart+density_threshold minus smart | iou | 0.0046 | -0.0001 | 0.0091 |
| smart+density_threshold minus smart | semantic_success | 0.0010 | -0.0110 | 0.0130 |
| smart+density_threshold minus smart | precise_wrong_object | -0.0130 | -0.0200 | -0.0070 |
| smart+density_threshold minus smart | worsened | -0.0980 | -0.1170 | -0.0790 |
| smart+logistic minus smart | iou | -0.0036 | -0.0121 | 0.0050 |
| smart+logistic minus smart | semantic_success | -0.0070 | -0.0270 | 0.0130 |
| smart+logistic minus smart | precise_wrong_object | -0.0290 | -0.0410 | -0.0180 |
| smart+logistic minus smart | worsened | -0.2540 | -0.2820 | -0.2260 |
| sam_score+logistic minus sam_score | iou | 0.0109 | 0.0002 | 0.0215 |
| sam_score+logistic minus sam_score | semantic_success | -0.0060 | -0.0290 | 0.0150 |
| sam_score+logistic minus sam_score | precise_wrong_object | -0.0300 | -0.0420 | -0.0190 |
| sam_score+logistic minus sam_score | worsened | -0.3090 | -0.3380 | -0.2780 |

## Harm

| method | improved | worsened | unchanged | mean_positive_delta | mean_negative_delta | median_delta | worst_decile_delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| clip_baseline | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| sam_score | 0.5600 | 0.4400 | 0.0000 | 0.3247 | -0.1500 | 0.0466 | -0.3348 |
| sam_score+density_threshold | 0.4990 | 0.2370 | 0.2640 | 0.3382 | -0.1637 | 0.0000 | -0.2906 |
| sam_score+logistic | 0.4210 | 0.1310 | 0.4480 | 0.3446 | -0.1395 | 0.0000 | -0.1761 |
| coverage | 0.5900 | 0.4100 | 0.0000 | 0.2610 | -0.1072 | 0.0578 | -0.2438 |
| density | 0.5690 | 0.4310 | 0.0000 | 0.3093 | -0.1547 | 0.0529 | -0.3390 |
| smart | 0.6420 | 0.3580 | 0.0000 | 0.3097 | -0.1161 | 0.1234 | -0.2545 |
| smart+density_threshold | 0.6050 | 0.2600 | 0.1350 | 0.3161 | -0.1130 | 0.1015 | -0.2154 |
| smart+logistic | 0.4880 | 0.1040 | 0.4080 | 0.3367 | -0.1022 | 0.0000 | -0.1061 |

Worst-decile delta is the mean of the lowest 10% of per-target IoU changes; positive/negative means are conditional on improvement/harm. The underlying CSV also retains confidence intervals for harm and improvement fractions and all secondary mean metrics.

## Frozen baseline subgroups

| method | group | n_pairs | baseline_iou | iou | delta_iou | delta_ci_low | delta_ci_high | iou_ci_low | iou_ci_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| clip_baseline | A_correct_target_coarse_mask | 398 | 0.3111 | 0.3111 | 0.0000 | 0.0000 | 0.0000 | 0.2986 | 0.3236 |
| clip_baseline | B_wrong_or_unconfirmed_target | 455 | 0.1567 | 0.1567 | 0.0000 | 0.0000 | 0.0000 | 0.1451 | 0.1690 |
| clip_baseline | C_good_localization | 147 | 0.6091 | 0.6091 | 0.0000 | 0.0000 | 0.0000 | 0.5969 | 0.6219 |
| sam_score | A_correct_target_coarse_mask | 398 | 0.3111 | 0.4925 | 0.1814 | 0.1509 | 0.2126 | 0.4611 | 0.5239 |
| sam_score | B_wrong_or_unconfirmed_target | 455 | 0.1567 | 0.2369 | 0.0802 | 0.0559 | 0.1051 | 0.2107 | 0.2642 |
| sam_score | C_good_localization | 147 | 0.6091 | 0.6577 | 0.0486 | 0.0069 | 0.0894 | 0.6144 | 0.6992 |
| sam_score+density_threshold | A_correct_target_coarse_mask | 398 | 0.3111 | 0.4922 | 0.1811 | 0.1534 | 0.2098 | 0.4637 | 0.5201 |
| sam_score+density_threshold | B_wrong_or_unconfirmed_target | 455 | 0.1567 | 0.2644 | 0.1077 | 0.0860 | 0.1290 | 0.2385 | 0.2892 |
| sam_score+density_threshold | C_good_localization | 147 | 0.6091 | 0.6694 | 0.0604 | 0.0188 | 0.1005 | 0.6276 | 0.7099 |
| sam_score+logistic | A_correct_target_coarse_mask | 398 | 0.3111 | 0.4982 | 0.1872 | 0.1606 | 0.2141 | 0.4701 | 0.5262 |
| sam_score+logistic | B_wrong_or_unconfirmed_target | 455 | 0.1567 | 0.2399 | 0.0832 | 0.0646 | 0.1020 | 0.2170 | 0.2628 |
| sam_score+logistic | C_good_localization | 147 | 0.6091 | 0.7072 | 0.0982 | 0.0672 | 0.1293 | 0.6749 | 0.7393 |
| coverage | A_correct_target_coarse_mask | 398 | 0.3111 | 0.5092 | 0.1982 | 0.1723 | 0.2249 | 0.4815 | 0.5385 |
| coverage | B_wrong_or_unconfirmed_target | 455 | 0.1567 | 0.1904 | 0.0337 | 0.0165 | 0.0518 | 0.1692 | 0.2133 |
| coverage | C_good_localization | 147 | 0.6091 | 0.7171 | 0.1080 | 0.0738 | 0.1427 | 0.6801 | 0.7532 |
| density | A_correct_target_coarse_mask | 398 | 0.3111 | 0.4623 | 0.1512 | 0.1225 | 0.1818 | 0.4340 | 0.4931 |
| density | B_wrong_or_unconfirmed_target | 455 | 0.1567 | 0.2431 | 0.0864 | 0.0633 | 0.1105 | 0.2185 | 0.2675 |
| density | C_good_localization | 147 | 0.6091 | 0.6759 | 0.0668 | 0.0233 | 0.1095 | 0.6295 | 0.7214 |
| smart | A_correct_target_coarse_mask | 398 | 0.3111 | 0.5533 | 0.2422 | 0.2136 | 0.2711 | 0.5244 | 0.5818 |
| smart | B_wrong_or_unconfirmed_target | 455 | 0.1567 | 0.2415 | 0.0848 | 0.0626 | 0.1078 | 0.2163 | 0.2680 |
| smart | C_good_localization | 147 | 0.6091 | 0.7606 | 0.1515 | 0.1210 | 0.1813 | 0.7284 | 0.7929 |
| smart+density_threshold | A_correct_target_coarse_mask | 398 | 0.3111 | 0.5485 | 0.2375 | 0.2099 | 0.2651 | 0.5207 | 0.5767 |
| smart+density_threshold | B_wrong_or_unconfirmed_target | 455 | 0.1567 | 0.2557 | 0.0990 | 0.0780 | 0.1217 | 0.2303 | 0.2816 |
| smart+density_threshold | C_good_localization | 147 | 0.6091 | 0.7609 | 0.1519 | 0.1216 | 0.1805 | 0.7301 | 0.7921 |
| smart+logistic | A_correct_target_coarse_mask | 398 | 0.3111 | 0.5368 | 0.2257 | 0.1996 | 0.2514 | 0.5097 | 0.5642 |
| smart+logistic | B_wrong_or_unconfirmed_target | 455 | 0.1567 | 0.2495 | 0.0928 | 0.0748 | 0.1128 | 0.2264 | 0.2734 |
| smart+logistic | C_good_localization | 147 | 0.6091 | 0.7563 | 0.1472 | 0.1194 | 0.1741 | 0.7261 | 0.7852 |
