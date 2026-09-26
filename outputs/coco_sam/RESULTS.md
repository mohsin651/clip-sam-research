# First frozen CLIP-attribution + SAM experiment

All 500 frozen images / 1,000 target pairs evaluated. Official SAM ViT-B, original RGB images, 3 candidates per prompt, no training. Inference used no COCO masks, boxes, areas or GT metrics. Existing baseline files were preserved and hash-verified.

## Research questions

1. Overall: the highest observed guided mIoU is 0.4005 for `sam_topk_points/sam_score`, versus the frozen baseline 0.2846; paired delta +0.1158, 95% CI [+0.0983, +0.1339]. Ranking is descriptive; no strategy was retuned or selected for deployment.
2. Group A: for the highest overall row, IoU increases from 0.3111 to 0.4925 (delta +0.1814, CI [+0.1509, +0.2126]). B gains +0.0802; C gains +0.0486. Group A has the largest observed gain. The highest Group-A row separately is `sam_top1_point/attribution_coverage` at 0.5362, delta +0.2251.
3. Best observed prompt/selector combination: `sam_topk_points/sam_score`. All alternatives and the center control are retained in the tables.
4. Coverage does not uniformly outperform SAM score. Mean IoU differences (coverage minus score): sam_top1_point: +0.0861; sam_topk_points: -0.0058; sam_attribution_box: +0.0239; sam_point_box: +0.0097. Top-3's interval includes zero; top1, box and point+box intervals favor coverage. For top1, coverage improves IoU but lowers semantic success by 4.6 percentage points; for top3, by 4.3 points. Both semantic decreases have paired intervals below zero. Coverage rewards containing attribution and can favor oversized masks.
5. Yes, CLIP top1 guidance outperforms the fully nonsemantic center+SAM-score control: IoU difference +0.1956, CI [+0.1712, +0.2201]. Center with coverage also uses CLIP at candidate selection and is not fully nonsemantic.
6. Precise wrong-object proxy for the highest observed guided row: 71/1000 (7.1%). Every method has its own rate below. This uses other-category union IoU>=0.5 and greater than target IoU, not manual confirmation of object identity.
7. The largest gain is in Group A, supporting spatial refinement as the main benefit. There is also partial recovery in B: 87/455 pairs reach target IoU>=0.5 and semantic success for the highest overall row. Its semantic success is 75.8%, versus baseline 68.5%. SAM receives no target text, so this is geometric refinement/candidate resolution driven by CLIP, not independent language understanding. 44.0% of pairs still lose IoU.
8. Yes, these results justify further controlled research: an untuned guided method improves overall and Group-A IoU and clearly beats the nonsemantic control. They do not justify claiming semantic localization is solved or that one strategy is universally best. This is one fixed cohort; rankings are exploratory and multiple comparison intervals are not multiplicity-adjusted. No further tuning was performed.

## Primary results: same CLIP crop as baseline

Rows are pair-macro averages. Improved/unchanged/worsened and semantic-success columns are fractions (multiply by 100 for percentages). Delta intervals use 2,000 whole-image bootstrap resamples, seed 2026. Baseline groups are frozen; no SAM results change membership.

| method | iou | delta_iou | median_delta_iou | improved | unchanged | worsened | semantic_success | precise_wrong_object | delta_ci_low | delta_ci_high |
|---|---|---|---|---|---|---|---|---|---|---|
| clip_baseline | 0.2846 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.6850 | 0.0130 | 0.0000 | 0.0000 |
| sam_top1_point/sam_score | 0.3004 | 0.0158 | -0.0549 | 0.3870 | 0.0000 | 0.6130 | 0.7660 | 0.0460 | -0.0081 | 0.0401 |
| sam_top1_point/attribution_coverage | 0.3865 | 0.1019 | 0.0346 | 0.5380 | 0.0000 | 0.4620 | 0.7200 | 0.0770 | 0.0828 | 0.1225 |
| sam_topk_points/sam_score | 0.4005 | 0.1158 | 0.0466 | 0.5600 | 0.0000 | 0.4400 | 0.7580 | 0.0710 | 0.0983 | 0.1339 |
| sam_topk_points/attribution_coverage | 0.3947 | 0.1101 | 0.0578 | 0.5900 | 0.0000 | 0.4100 | 0.7150 | 0.1040 | 0.0965 | 0.1246 |
| sam_attribution_box/sam_score | 0.3273 | 0.0426 | -0.0223 | 0.4680 | 0.0000 | 0.5320 | 0.6820 | 0.0580 | 0.0240 | 0.0613 |
| sam_attribution_box/attribution_coverage | 0.3512 | 0.0666 | 0.0125 | 0.5140 | 0.0000 | 0.4860 | 0.6970 | 0.0620 | 0.0489 | 0.0836 |
| sam_point_box/sam_score | 0.3645 | 0.0798 | 0.0305 | 0.5460 | 0.0000 | 0.4540 | 0.7120 | 0.0590 | 0.0613 | 0.0985 |
| sam_point_box/attribution_coverage | 0.3742 | 0.0896 | 0.0540 | 0.5860 | 0.0000 | 0.4140 | 0.7110 | 0.0670 | 0.0731 | 0.1062 |
| center_point_sam/sam_score | 0.1048 | -0.1798 | -0.1713 | 0.1200 | 0.0000 | 0.8800 | 0.3600 | 0.1160 | -0.1947 | -0.1647 |
| center_point_sam/attribution_coverage | 0.1723 | -0.1123 | -0.1149 | 0.2110 | 0.0000 | 0.7890 | 0.3880 | 0.2240 | -0.1282 | -0.0971 |

## Binary segmentation quality

| method | iou | iou_ci_low | iou_ci_high | pacc | dice | precision | recall | target_overlap_preferred |
|---|---|---|---|---|---|---|---|---|
| clip_baseline | 0.2846 | 0.2732 | 0.2965 | 0.7430 | 0.4087 | 0.3602 | 0.7770 | 0.6160 |
| sam_top1_point/sam_score | 0.3004 | 0.2778 | 0.3238 | 0.8255 | 0.3637 | 0.6866 | 0.3240 | 0.7870 |
| sam_top1_point/attribution_coverage | 0.3865 | 0.3656 | 0.4088 | 0.8203 | 0.4705 | 0.5832 | 0.5089 | 0.7210 |
| sam_topk_points/sam_score | 0.4005 | 0.3790 | 0.4217 | 0.8007 | 0.4875 | 0.5819 | 0.5469 | 0.7520 |
| sam_topk_points/attribution_coverage | 0.3947 | 0.3757 | 0.4144 | 0.7834 | 0.4881 | 0.4977 | 0.6436 | 0.6910 |
| sam_attribution_box/sam_score | 0.3273 | 0.3064 | 0.3488 | 0.8388 | 0.4070 | 0.5357 | 0.4360 | 0.6750 |
| sam_attribution_box/attribution_coverage | 0.3512 | 0.3317 | 0.3729 | 0.8414 | 0.4402 | 0.5495 | 0.4781 | 0.6900 |
| sam_point_box/sam_score | 0.3645 | 0.3433 | 0.3867 | 0.8427 | 0.4528 | 0.5636 | 0.5138 | 0.6990 |
| sam_point_box/attribution_coverage | 0.3742 | 0.3548 | 0.3962 | 0.8432 | 0.4687 | 0.5528 | 0.5493 | 0.6920 |
| center_point_sam/sam_score | 0.1048 | 0.0915 | 0.1192 | 0.7586 | 0.1333 | 0.2946 | 0.1286 | 0.3860 |
| center_point_sam/attribution_coverage | 0.1723 | 0.1558 | 0.1889 | 0.7245 | 0.2127 | 0.2840 | 0.2401 | 0.4060 |

## Group A: correct-target/coarse-mask proxy (398 pairs)

| method | baseline_iou | iou | delta_iou | delta_ci_low | delta_ci_high |
|---|---|---|---|---|---|
| clip_baseline | 0.3111 | 0.3111 | 0.0000 | 0.0000 | 0.0000 |
| sam_top1_point/sam_score | 0.3111 | 0.3771 | 0.0661 | 0.0297 | 0.1043 |
| sam_top1_point/attribution_coverage | 0.3111 | 0.5362 | 0.2251 | 0.1953 | 0.2548 |
| sam_topk_points/sam_score | 0.3111 | 0.4925 | 0.1814 | 0.1509 | 0.2126 |
| sam_topk_points/attribution_coverage | 0.3111 | 0.5092 | 0.1982 | 0.1723 | 0.2249 |
| sam_attribution_box/sam_score | 0.3111 | 0.3694 | 0.0584 | 0.0258 | 0.0907 |
| sam_attribution_box/attribution_coverage | 0.3111 | 0.4008 | 0.0898 | 0.0604 | 0.1201 |
| sam_point_box/sam_score | 0.3111 | 0.4512 | 0.1401 | 0.1098 | 0.1718 |
| sam_point_box/attribution_coverage | 0.3111 | 0.4600 | 0.1489 | 0.1217 | 0.1772 |
| center_point_sam/sam_score | 0.3111 | 0.1043 | -0.2068 | -0.2288 | -0.1835 |
| center_point_sam/attribution_coverage | 0.3111 | 0.1821 | -0.1289 | -0.1551 | -0.1016 |

## All baseline subgroups

| frame | method | group | n_pairs | baseline_iou | iou | delta_iou | delta_ci_low | delta_ci_high | iou_ci_low | iou_ci_high |
|---|---|---|---|---|---|---|---|---|---|---|
| clip_crop | clip_baseline | A_correct_target_coarse_mask | 398 | 0.3111 | 0.3111 | 0.0000 | 0.0000 | 0.0000 | 0.2986 | 0.3236 |
| clip_crop | clip_baseline | B_wrong_or_unconfirmed_target | 455 | 0.1567 | 0.1567 | 0.0000 | 0.0000 | 0.0000 | 0.1451 | 0.1690 |
| clip_crop | clip_baseline | C_good_localization | 147 | 0.6091 | 0.6091 | 0.0000 | 0.0000 | 0.0000 | 0.5969 | 0.6219 |
| clip_crop | sam_top1_point/sam_score | A_correct_target_coarse_mask | 398 | 0.3111 | 0.3771 | 0.0661 | 0.0297 | 0.1043 | 0.3440 | 0.4112 |
| clip_crop | sam_top1_point/sam_score | B_wrong_or_unconfirmed_target | 455 | 0.1567 | 0.1883 | 0.0315 | -0.0000 | 0.0656 | 0.1612 | 0.2182 |
| clip_crop | sam_top1_point/sam_score | C_good_localization | 147 | 0.6091 | 0.4400 | -0.1690 | -0.2319 | -0.1090 | 0.3760 | 0.5028 |
| clip_crop | sam_top1_point/attribution_coverage | A_correct_target_coarse_mask | 398 | 0.3111 | 0.5362 | 0.2251 | 0.1953 | 0.2548 | 0.5082 | 0.5651 |
| clip_crop | sam_top1_point/attribution_coverage | B_wrong_or_unconfirmed_target | 455 | 0.1567 | 0.1782 | 0.0215 | -0.0058 | 0.0510 | 0.1535 | 0.2055 |
| clip_crop | sam_top1_point/attribution_coverage | C_good_localization | 147 | 0.6091 | 0.6260 | 0.0169 | -0.0298 | 0.0634 | 0.5782 | 0.6740 |
| clip_crop | sam_topk_points/sam_score | A_correct_target_coarse_mask | 398 | 0.3111 | 0.4925 | 0.1814 | 0.1509 | 0.2126 | 0.4611 | 0.5239 |
| clip_crop | sam_topk_points/sam_score | B_wrong_or_unconfirmed_target | 455 | 0.1567 | 0.2369 | 0.0802 | 0.0559 | 0.1051 | 0.2107 | 0.2642 |
| clip_crop | sam_topk_points/sam_score | C_good_localization | 147 | 0.6091 | 0.6577 | 0.0486 | 0.0069 | 0.0894 | 0.6144 | 0.6992 |
| clip_crop | sam_topk_points/attribution_coverage | A_correct_target_coarse_mask | 398 | 0.3111 | 0.5092 | 0.1982 | 0.1723 | 0.2249 | 0.4815 | 0.5385 |
| clip_crop | sam_topk_points/attribution_coverage | B_wrong_or_unconfirmed_target | 455 | 0.1567 | 0.1904 | 0.0337 | 0.0165 | 0.0518 | 0.1692 | 0.2133 |
| clip_crop | sam_topk_points/attribution_coverage | C_good_localization | 147 | 0.6091 | 0.7171 | 0.1080 | 0.0738 | 0.1427 | 0.6801 | 0.7532 |
| clip_crop | sam_attribution_box/sam_score | A_correct_target_coarse_mask | 398 | 0.3111 | 0.3694 | 0.0584 | 0.0258 | 0.0907 | 0.3382 | 0.4010 |
| clip_crop | sam_attribution_box/sam_score | B_wrong_or_unconfirmed_target | 455 | 0.1567 | 0.1936 | 0.0369 | 0.0131 | 0.0621 | 0.1696 | 0.2200 |
| clip_crop | sam_attribution_box/sam_score | C_good_localization | 147 | 0.6091 | 0.6268 | 0.0178 | -0.0249 | 0.0584 | 0.5823 | 0.6706 |
| clip_crop | sam_attribution_box/attribution_coverage | A_correct_target_coarse_mask | 398 | 0.3111 | 0.4008 | 0.0898 | 0.0604 | 0.1201 | 0.3715 | 0.4310 |
| clip_crop | sam_attribution_box/attribution_coverage | B_wrong_or_unconfirmed_target | 455 | 0.1567 | 0.2076 | 0.0509 | 0.0287 | 0.0754 | 0.1842 | 0.2328 |
| clip_crop | sam_attribution_box/attribution_coverage | C_good_localization | 147 | 0.6091 | 0.6613 | 0.0523 | 0.0187 | 0.0860 | 0.6247 | 0.6978 |
| clip_crop | sam_point_box/sam_score | A_correct_target_coarse_mask | 398 | 0.3111 | 0.4512 | 0.1401 | 0.1098 | 0.1718 | 0.4218 | 0.4803 |
| clip_crop | sam_point_box/sam_score | B_wrong_or_unconfirmed_target | 455 | 0.1567 | 0.1902 | 0.0334 | 0.0091 | 0.0577 | 0.1673 | 0.2147 |
| clip_crop | sam_point_box/sam_score | C_good_localization | 147 | 0.6091 | 0.6692 | 0.0601 | 0.0200 | 0.1001 | 0.6253 | 0.7130 |
| clip_crop | sam_point_box/attribution_coverage | A_correct_target_coarse_mask | 398 | 0.3111 | 0.4600 | 0.1489 | 0.1217 | 0.1772 | 0.4322 | 0.4876 |
| clip_crop | sam_point_box/attribution_coverage | B_wrong_or_unconfirmed_target | 455 | 0.1567 | 0.1947 | 0.0379 | 0.0152 | 0.0606 | 0.1729 | 0.2172 |
| clip_crop | sam_point_box/attribution_coverage | C_good_localization | 147 | 0.6091 | 0.6978 | 0.0887 | 0.0531 | 0.1251 | 0.6597 | 0.7383 |
| clip_crop | center_point_sam/sam_score | A_correct_target_coarse_mask | 398 | 0.3111 | 0.1043 | -0.2068 | -0.2288 | -0.1835 | 0.0840 | 0.1264 |
| clip_crop | center_point_sam/sam_score | B_wrong_or_unconfirmed_target | 455 | 0.1567 | 0.0558 | -0.1009 | -0.1174 | -0.0831 | 0.0418 | 0.0717 |
| clip_crop | center_point_sam/sam_score | C_good_localization | 147 | 0.6091 | 0.2579 | -0.3512 | -0.4093 | -0.2929 | 0.2002 | 0.3194 |
| clip_crop | center_point_sam/attribution_coverage | A_correct_target_coarse_mask | 398 | 0.3111 | 0.1821 | -0.1289 | -0.1551 | -0.1016 | 0.1550 | 0.2108 |
| clip_crop | center_point_sam/attribution_coverage | B_wrong_or_unconfirmed_target | 455 | 0.1567 | 0.0871 | -0.0696 | -0.0885 | -0.0495 | 0.0693 | 0.1068 |
| clip_crop | center_point_sam/attribution_coverage | C_good_localization | 147 | 0.6091 | 0.4094 | -0.1996 | -0.2581 | -0.1398 | 0.3496 | 0.4729 |

Subgroup bootstrap resamples whole images and divides resampled subgroup sums by counts. This preserves pair weighting when a sampled image contributes one or two subgroup cases.

## Candidate selection and guidance controls

| comparison | IoU difference [95% CI] | Semantic-success difference [95% CI] |
|---|---|---|
| sam_top1_point: coverage minus score | 0.0861 [0.0668, 0.1055] | -0.0460 [-0.0660, -0.0270] |
| sam_topk_points: coverage minus score | -0.0058 [-0.0205, 0.0093] | -0.0430 [-0.0610, -0.0250] |
| sam_attribution_box: coverage minus score | 0.0239 [0.0159, 0.0317] | 0.0150 [0.0000, 0.0310] |
| sam_point_box: coverage minus score | 0.0097 [0.0013, 0.0181] | -0.0010 [-0.0150, 0.0120] |
| center_point_sam: coverage minus score | 0.0675 [0.0564, 0.0780] | 0.0280 [0.0150, 0.0410] |
| top1 minus center: sam_score | 0.1956 [0.1712, 0.2201] | 0.4060 [0.3790, 0.4350] |
| top1 minus center: attribution_coverage | 0.2142 [0.1910, 0.2377] | 0.3320 [0.3020, 0.3600] |

| strategy | selector_agreement |
|---|---|
| sam_top1_point | 0.1960 |
| sam_topk_points | 0.3800 |
| sam_attribution_box | 0.6110 |
| sam_point_box | 0.6630 |
| center_point_sam | 0.2030 |

## Failure proxies

| method | failure_proxy | count | percent |
|---|---|---|---|
| center_point_sam/attribution_coverage | A_correct_coarse_improved | 91 | 9.1000 |
| center_point_sam/attribution_coverage | A_correct_coarse_worsened | 307 | 30.7000 |
| center_point_sam/attribution_coverage | B_precise_wrong_object | 128 | 12.8000 |
| center_point_sam/attribution_coverage | B_target_recovered | 36 | 3.6000 |
| center_point_sam/attribution_coverage | B_unresolved | 291 | 29.1000 |
| center_point_sam/attribution_coverage | C_already_good_improved | 53 | 5.3000 |
| center_point_sam/attribution_coverage | C_already_good_worsened | 94 | 9.4000 |
| center_point_sam/sam_score | A_correct_coarse_improved | 47 | 4.7000 |
| center_point_sam/sam_score | A_correct_coarse_worsened | 351 | 35.1000 |
| center_point_sam/sam_score | B_precise_wrong_object | 61 | 6.1000 |
| center_point_sam/sam_score | B_target_recovered | 20 | 2.0000 |
| center_point_sam/sam_score | B_unresolved | 374 | 37.4000 |
| center_point_sam/sam_score | C_already_good_improved | 31 | 3.1000 |
| center_point_sam/sam_score | C_already_good_worsened | 116 | 11.6000 |
| sam_attribution_box/attribution_coverage | A_correct_coarse_improved | 219 | 21.9000 |
| sam_attribution_box/attribution_coverage | A_correct_coarse_worsened | 179 | 17.9000 |
| sam_attribution_box/attribution_coverage | B_precise_wrong_object | 49 | 4.9000 |
| sam_attribution_box/attribution_coverage | B_target_recovered | 78 | 7.8000 |
| sam_attribution_box/attribution_coverage | B_unresolved | 328 | 32.8000 |
| sam_attribution_box/attribution_coverage | C_already_good_improved | 95 | 9.5000 |
| sam_attribution_box/attribution_coverage | C_already_good_worsened | 52 | 5.2000 |
| sam_attribution_box/sam_score | A_correct_coarse_improved | 204 | 20.4000 |
| sam_attribution_box/sam_score | A_correct_coarse_worsened | 194 | 19.4000 |
| sam_attribution_box/sam_score | B_precise_wrong_object | 47 | 4.7000 |
| sam_attribution_box/sam_score | B_target_recovered | 73 | 7.3000 |
| sam_attribution_box/sam_score | B_unresolved | 335 | 33.5000 |
| sam_attribution_box/sam_score | C_already_good_improved | 87 | 8.7000 |
| sam_attribution_box/sam_score | C_already_good_worsened | 60 | 6.0000 |
| sam_point_box/attribution_coverage | A_correct_coarse_improved | 263 | 26.3000 |
| sam_point_box/attribution_coverage | A_correct_coarse_worsened | 135 | 13.5000 |
| sam_point_box/attribution_coverage | B_precise_wrong_object | 53 | 5.3000 |
| sam_point_box/attribution_coverage | B_target_recovered | 60 | 6.0000 |
| sam_point_box/attribution_coverage | B_unresolved | 342 | 34.2000 |
| sam_point_box/attribution_coverage | C_already_good_improved | 109 | 10.9000 |
| sam_point_box/attribution_coverage | C_already_good_worsened | 38 | 3.8000 |
| sam_point_box/sam_score | A_correct_coarse_improved | 248 | 24.8000 |
| sam_point_box/sam_score | A_correct_coarse_worsened | 150 | 15.0000 |
| sam_point_box/sam_score | B_precise_wrong_object | 45 | 4.5000 |
| sam_point_box/sam_score | B_target_recovered | 63 | 6.3000 |
| sam_point_box/sam_score | B_unresolved | 347 | 34.7000 |
| sam_point_box/sam_score | C_already_good_improved | 99 | 9.9000 |
| sam_point_box/sam_score | C_already_good_worsened | 48 | 4.8000 |
| sam_top1_point/attribution_coverage | A_correct_coarse_improved | 291 | 29.1000 |
| sam_top1_point/attribution_coverage | A_correct_coarse_worsened | 107 | 10.7000 |
| sam_top1_point/attribution_coverage | B_precise_wrong_object | 65 | 6.5000 |
| sam_top1_point/attribution_coverage | B_target_recovered | 70 | 7.0000 |
| sam_top1_point/attribution_coverage | B_unresolved | 320 | 32.0000 |
| sam_top1_point/attribution_coverage | C_already_good_improved | 83 | 8.3000 |
| sam_top1_point/attribution_coverage | C_already_good_worsened | 64 | 6.4000 |
| sam_top1_point/sam_score | A_correct_coarse_improved | 186 | 18.6000 |
| sam_top1_point/sam_score | A_correct_coarse_worsened | 212 | 21.2000 |
| sam_top1_point/sam_score | B_precise_wrong_object | 35 | 3.5000 |
| sam_top1_point/sam_score | B_target_recovered | 81 | 8.1000 |
| sam_top1_point/sam_score | B_unresolved | 339 | 33.9000 |
| sam_top1_point/sam_score | C_already_good_improved | 60 | 6.0000 |
| sam_top1_point/sam_score | C_already_good_worsened | 87 | 8.7000 |
| sam_topk_points/attribution_coverage | A_correct_coarse_improved | 282 | 28.2000 |
| sam_topk_points/attribution_coverage | A_correct_coarse_worsened | 116 | 11.6000 |
| sam_topk_points/attribution_coverage | B_precise_wrong_object | 79 | 7.9000 |
| sam_topk_points/attribution_coverage | B_target_recovered | 47 | 4.7000 |
| sam_topk_points/attribution_coverage | B_unresolved | 329 | 32.9000 |
| sam_topk_points/attribution_coverage | C_already_good_improved | 106 | 10.6000 |
| sam_topk_points/attribution_coverage | C_already_good_worsened | 41 | 4.1000 |
| sam_topk_points/sam_score | A_correct_coarse_improved | 257 | 25.7000 |
| sam_topk_points/sam_score | A_correct_coarse_worsened | 141 | 14.1000 |
| sam_topk_points/sam_score | B_precise_wrong_object | 63 | 6.3000 |
| sam_topk_points/sam_score | B_target_recovered | 87 | 8.7000 |
| sam_topk_points/sam_score | B_unresolved | 305 | 30.5000 |
| sam_topk_points/sam_score | C_already_good_improved | 95 | 9.5000 |
| sam_topk_points/sam_score | C_already_good_worsened | 52 | 5.2000 |

A/C improved/worsened are IoU changes, not verified boundary-only changes. B recovered requires target IoU>=0.5 and semantic success. B precise wrong requires other-category IoU>=0.5 and greater than target IoU. Remaining B cases are unresolved. No exhaustive manual classification was performed.

## Original-image evaluation (secondary)

| method | iou | delta_iou | median_delta_iou | improved | unchanged | worsened | semantic_success | precise_wrong_object | delta_ci_low | delta_ci_high |
|---|---|---|---|---|---|---|---|---|---|---|
| clip_baseline | 0.2609 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.6920 | 0.0080 | 0.0000 | 0.0000 |
| sam_top1_point/sam_score | 0.2914 | 0.0305 | -0.0523 | 0.3860 | 0.0000 | 0.6140 | 0.7680 | 0.0460 | 0.0073 | 0.0540 |
| sam_top1_point/attribution_coverage | 0.3754 | 0.1145 | 0.0511 | 0.5530 | 0.0000 | 0.4470 | 0.7290 | 0.0770 | 0.0961 | 0.1347 |
| sam_topk_points/sam_score | 0.3884 | 0.1275 | 0.0630 | 0.5630 | 0.0000 | 0.4370 | 0.7580 | 0.0670 | 0.1097 | 0.1452 |
| sam_topk_points/attribution_coverage | 0.3826 | 0.1217 | 0.0730 | 0.5980 | 0.0000 | 0.4020 | 0.7210 | 0.0950 | 0.1081 | 0.1367 |
| sam_attribution_box/sam_score | 0.3011 | 0.0402 | -0.0224 | 0.4620 | 0.0000 | 0.5380 | 0.6900 | 0.0500 | 0.0230 | 0.0575 |
| sam_attribution_box/attribution_coverage | 0.3220 | 0.0611 | 0.0073 | 0.5070 | 0.0000 | 0.4930 | 0.7040 | 0.0540 | 0.0450 | 0.0776 |
| sam_point_box/sam_score | 0.3386 | 0.0777 | 0.0278 | 0.5460 | 0.0000 | 0.4540 | 0.7170 | 0.0540 | 0.0605 | 0.0948 |
| sam_point_box/attribution_coverage | 0.3463 | 0.0854 | 0.0465 | 0.5840 | 0.0000 | 0.4160 | 0.7190 | 0.0560 | 0.0699 | 0.1010 |
| center_point_sam/sam_score | 0.1004 | -0.1605 | -0.1635 | 0.1250 | 0.0000 | 0.8750 | 0.3610 | 0.1080 | -0.1742 | -0.1463 |
| center_point_sam/attribution_coverage | 0.1647 | -0.0962 | -0.1072 | 0.2140 | 0.0000 | 0.7860 | 0.3950 | 0.2130 | -0.1113 | -0.0818 |

These compare original SAM masks with original category unions. The matching CLIP baseline is its binary crop mask mapped back and zero outside the visible crop. Its IoU differs from 0.2846 because the evaluation field and target extent differ. Do not mix these rows with the primary table.

## Continuous CLIP attribution context

| metric | mean |
|---|---|
| pg | 0.7110 |
| epg | 0.3931 |
| ap | 0.5690 |
| pacc | 0.7430 |
| iou | 0.2846 |

PG/EPG/AP above are the unchanged continuous CLIP attribution metrics. SAM outputs are binary segmentation masks; no artificial continuous SAM PG/EPG/AP was computed.

## Geometry, models and limitations

The exact resize/crop inverse uses original/resized dimension ratios, pixel centers for points and pixel edges for boxes. Boxes are clamped to image edges. Synthetic geometry tests and a 50-pair technical smoke test passed before the full run. The tiny floating-point bounds correction and archived initial attempt are documented in TECHNICAL_NOTES.md; no GT evaluation occurred before full inference.
SAM works on original-resolution RGB images via its official longest-side-1024 preprocessing. Primary masks are projected back using the exact nearest-neighbor CLIP crop transform. Saved geometry panels are in geometry_debug/.
Semantic success means target category-union IoU is strictly greater than every other category-union IoU. COCO category masks can overlap/nest; this proxy can be ambiguous. Target GT unions all instances, while point/box prompting often extracts only one instance. The box uses only the strongest connected attribution component. Top-3 positives can cross object boundaries. Coverage selection can favor large masks. No GT-based candidate filtering, parameter fitting or method selection was performed.
SAM source: https://github.com/facebookresearch/segment-anything, commit `dca509fe793f601edb92606367a655c15ac00fdf`, package 1.0; checkpoint `data\checkpoints\sam_vit_b_01ec64.pth`, SHA256 `ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912`. Peak allocated CUDA memory 2.73 GiB.

## Visual examples

[Browse 50 distinct representative cases](examples.html). Each panel shows every candidate and both selected masks; display-selector IoU is SAM-score, fixed before evaluation. Exact IDs and reasons: example_manifest.json. All 15,000 candidate masks are saved as packed original-resolution arrays.

## Runtime and integrity

| stage | seconds |
|---|---|
| preparation_seconds | 22.0626 |
| inference_seconds_including_smoke | 213.5030 |
| technical_restart_attempt_seconds | 14.0810 |
| evaluation_seconds | 49.0160 |
| report_and_panels_seconds | 20.2404 |
| total_pipeline_seconds | 318.9029 |
Audit: {'images': 500, 'pairs': 1000, 'rows': 22000, 'candidate_masks': 15000, 'empty_candidate_masks': 0, 'baseline_files_unchanged': True, 'baseline_scores_reproduced': True, 'all_candidate_hashes_verified': True, 'gt_loaded_only_after_complete_inference': True, 'evaluation_seconds': 49.015955900000336}. 34 unit tests passed. Pipeline times exclude implementation and manual inspection. The attempted technical run is included in runtime, and final inference includes its successful smoke stage.

## Reproduce

`python scripts/prepare_sam.py`; `python -m pytest -q`; `python scripts/run_coco_sam.py --smoke`; inspect geometry; `python scripts/run_coco_sam.py --resume`; `python scripts/evaluate_coco_sam.py`; `python scripts/report_coco_sam.py`.
Frozen choices: ../../COCO_SAM_PLAN.md. Inference and evaluation are separate processes. Baseline hashes, source signatures, checkpoint hash and per-candidate-file hashes are retained. No further optimization was performed.
