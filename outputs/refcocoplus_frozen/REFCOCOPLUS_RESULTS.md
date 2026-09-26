# Frozen RefCOCO+ generalization results

Evaluation-only, positive-point-only. The original frozen RefCOCO infer_one function, CLIP/SAM checkpoints, features, ranker, fallback and preprocessing are reused unchanged. Raw expressions are verbatim. No negative points, fitting, calibration, filtering by performance, or method repair. Primary metrics are original-image referred-instance expression-macro scores.

## Dataset and provenance

Original [REFER research release](https://github.com/lichengunc/refer), UNC split. Original host unavailable; archived original RefCOCO+ ZIP used. Official COCO train2014 images, reused or downloaded with dimensions/decoding/hash checks. Exact archive URL and checksums: dataset_provenance.json. All split fields preserved.

| split | images | target_instances | references | expressions |
| --- | --- | --- | --- | --- |
| testA | 750 | 1975 | 1975 | 5726 |
| testB | 750 | 1798 | 1798 | 4889 |
| train | 16992 | 42278 | 42278 | 120191 |
| val | 1500 | 3805 | 3805 | 10758 |

Evaluated counts: {"images": 1500, "expressions": 10615, "target_instances": 3773, "references": 3773}. Complete competitor annotation IDs verified against official COCO annotations.

## Overlap, recorded before inference

| prior_cohort | overlap_images |
| --- | --- |
| COCO development | 0 |
| COCO heldout | 0 |
| RefCOCO negative-point validation | 0 |
| RefCOCO testA/testB | 1500 |

Union overlap: 1500 images. Strict remaining cohort: {"images": 0, "expressions": 0, "target_instances": 0, "references": 0}. Per-split counts and exact IDs are in overlap_audit.json. Prior-cohort counts may overlap and must not be added.

## Standard combined primary results

| method | iou | dice | p_at_05 | p_at_07 | worsened | correct_instance |
| --- | --- | --- | --- | --- | --- | --- |
| clip_baseline | 0.2079 | 0.3232 | 0.0366 | 0.0016 | 0.0000 | 0.5387 |
| sam_score | 0.3016 | 0.3879 | 0.2560 | 0.1433 | 0.4463 | 0.5435 |
| smart | 0.3224 | 0.4142 | 0.2768 | 0.1537 | 0.3761 | 0.5449 |
| frozen_final | 0.3216 | 0.4157 | 0.2698 | 0.1515 | 0.3308 | 0.5472 |
| center_score | 0.0925 | 0.1152 | 0.0858 | 0.0565 | 0.8476 | 0.3112 |

Rates are fractions; worsened is harm versus CLIP. P@0.5/P@0.7 are target IoU success fractions, not detection AP.

## Standard testA and testB

| split | method | iou | dice | p_at_05 | p_at_07 | worsened | correct_instance |
| --- | --- | --- | --- | --- | --- | --- | --- |
| testA | clip_baseline | 0.2008 | 0.3140 | 0.0299 | 0.0014 | 0.0000 | 0.5840 |
| testA | sam_score | 0.2927 | 0.3749 | 0.2538 | 0.1463 | 0.4743 | 0.5791 |
| testA | smart | 0.3205 | 0.4049 | 0.2904 | 0.1769 | 0.4046 | 0.5873 |
| testA | frozen_final | 0.3220 | 0.4097 | 0.2848 | 0.1750 | 0.3535 | 0.5934 |
| testA | center_score | 0.0780 | 0.1006 | 0.0629 | 0.0382 | 0.8596 | 0.3086 |
| testB | clip_baseline | 0.2161 | 0.3341 | 0.0446 | 0.0018 | 0.0000 | 0.4856 |
| testB | sam_score | 0.3119 | 0.4031 | 0.2585 | 0.1397 | 0.4134 | 0.5017 |
| testB | smart | 0.3247 | 0.4252 | 0.2608 | 0.1264 | 0.3426 | 0.4952 |
| testB | frozen_final | 0.3211 | 0.4228 | 0.2522 | 0.1240 | 0.3042 | 0.4931 |
| testB | center_score | 0.1095 | 0.1323 | 0.1127 | 0.0779 | 0.8335 | 0.3142 |

## Strict image-non-overlap results

Not estimable: no test images remain after excluding all previously used images. No substitute split is evaluated.

Empty split/cohort combinations (no metrics or CIs claimed): [{"cohort": "strict_nonoverlap", "split": "combined"}, {"cohort": "strict_nonoverlap", "split": "testA"}, {"cohort": "strict_nonoverlap", "split": "testB"}]

## Binary quality and intervals

| method | iou | iou_ci_low | iou_ci_high | dice | precision | recall | pacc | p_at_05 | p_at_07 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| clip_baseline | 0.2079 | 0.2033 | 0.2122 | 0.3232 | 0.2626 | 0.5576 | 0.7650 | 0.0366 | 0.0016 |
| sam_score | 0.3016 | 0.2932 | 0.3097 | 0.3879 | 0.4167 | 0.4661 | 0.8400 | 0.2560 | 0.1433 |
| smart | 0.3224 | 0.3138 | 0.3309 | 0.4142 | 0.4181 | 0.5183 | 0.8433 | 0.2768 | 0.1537 |
| frozen_final | 0.3216 | 0.3129 | 0.3298 | 0.4157 | 0.4183 | 0.5258 | 0.8436 | 0.2698 | 0.1515 |
| center_score | 0.0925 | 0.0856 | 0.0996 | 0.1152 | 0.2093 | 0.1112 | 0.8229 | 0.0858 | 0.0565 |

## Paired comparisons

| comparison | metric | difference | ci_low | ci_high |
| --- | --- | --- | --- | --- |
| sam_score minus clip_baseline | iou | 0.0937 | 0.0868 | 0.1006 |
| sam_score minus clip_baseline | harm_rate | 0.4463 | 0.4331 | 0.4604 |
| sam_score minus clip_baseline | correct_instance | 0.0048 | -0.0060 | 0.0151 |
| sam_score minus clip_baseline | p_at_05 | 0.2193 | 0.2077 | 0.2313 |
| smart minus sam_score | iou | 0.0209 | 0.0165 | 0.0251 |
| smart minus sam_score | harm_rate | -0.0702 | -0.0783 | -0.0619 |
| smart minus sam_score | correct_instance | 0.0014 | -0.0038 | 0.0067 |
| smart minus sam_score | p_at_05 | 0.0208 | 0.0126 | 0.0288 |
| frozen_final minus sam_score | iou | 0.0200 | 0.0154 | 0.0244 |
| frozen_final minus sam_score | harm_rate | -0.1155 | -0.1249 | -0.1056 |
| frozen_final minus sam_score | correct_instance | 0.0038 | -0.0022 | 0.0097 |
| frozen_final minus sam_score | p_at_05 | 0.0138 | 0.0053 | 0.0220 |
| frozen_final minus clip_baseline | iou | 0.1137 | 0.1075 | 0.1195 |
| frozen_final minus clip_baseline | harm_rate | 0.3308 | 0.3183 | 0.3433 |
| frozen_final minus clip_baseline | correct_instance | 0.0086 | -0.0010 | 0.0182 |
| frozen_final minus clip_baseline | p_at_05 | 0.2332 | 0.2217 | 0.2449 |
| frozen_final minus smart | iou | -0.0009 | -0.0023 | 0.0005 |
| frozen_final minus smart | harm_rate | -0.0453 | -0.0510 | -0.0401 |
| frozen_final minus smart | correct_instance | 0.0024 | -0.0007 | 0.0055 |
| frozen_final minus smart | p_at_05 | -0.0070 | -0.0092 | -0.0049 |

2,000 paired whole-image resamples, seed 2026, preserving expressions per image and expression weighting. Intervals are not multiplicity-adjusted. All split and strict contrasts are in paired_comparisons.json. Harm contrasts compare each method with CLIP; direct final-versus-naive worsening is different.

## Harm versus CLIP

| method | improved | worsened | unchanged | delta_iou | median_delta | mean_positive_delta | mean_negative_delta | worst_decile_delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| clip_baseline | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| sam_score | 0.5360 | 0.4463 | 0.0177 | 0.0937 | 0.0246 | 0.2745 | -0.1198 | -0.2725 |
| smart | 0.6067 | 0.3761 | 0.0172 | 0.1146 | 0.0677 | 0.2500 | -0.0987 | -0.2133 |
| frozen_final | 0.5795 | 0.3308 | 0.0898 | 0.1137 | 0.0529 | 0.2511 | -0.0961 | -0.1984 |
| center_score | 0.1318 | 0.8476 | 0.0206 | -0.1153 | -0.1390 | 0.3328 | -0.1878 | -0.4283 |

## Smart/final directly versus naive

| method | improved | worsened | unchanged | delta_iou | median_delta | mean_positive_delta | mean_negative_delta | worst_decile_delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| smart | 0.2647 | 0.1736 | 0.5617 | 0.0209 | 0.0000 | 0.1834 | -0.1595 | -0.2642 |
| frozen_final | 0.2948 | 0.1844 | 0.5209 | 0.0200 | 0.0000 | 0.1803 | -0.1798 | -0.3072 |

## Instance disambiguation

| ambiguity | method | expressions | images | correct_instance | correct_instance_ci_low | correct_instance_ci_high | wrong_instance | instance_tie |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| multiple | clip_baseline | 10615 | 1500 | 0.5387 | 0.5267 | 0.5505 | 0.4612 | 0.0001 |
| multiple | sam_score | 10615 | 1500 | 0.5435 | 0.5319 | 0.5558 | 0.4411 | 0.0154 |
| multiple | smart | 10615 | 1500 | 0.5449 | 0.5334 | 0.5573 | 0.4415 | 0.0137 |
| multiple | frozen_final | 10615 | 1500 | 0.5472 | 0.5357 | 0.5594 | 0.4410 | 0.0118 |
| multiple | center_score | 10615 | 1500 | 0.3112 | 0.3002 | 0.3217 | 0.5304 | 0.1585 |

Target IoU must strictly exceed every other non-crowd same-category instance IoU. Ties are failures; single-instance cases, if any, require positive overlap. Empty groups are absent, not filtered. Category annotations can overlap: this is an overlap proxy, not manual semantic confirmation.

## Expression subgroups

| family | group | expressions | iou | correct_instance | p_at_05 | worsened |
| --- | --- | --- | --- | --- | --- | --- |
| length_group | long | 989 | 0.2060 | 0.3600 | 0.1355 | 0.4166 |
| length_group | medium | 3549 | 0.2589 | 0.4553 | 0.1922 | 0.3866 |
| length_group | short | 6077 | 0.3770 | 0.6314 | 0.3370 | 0.2842 |
| spatial | False | 9699 | 0.3352 | 0.5659 | 0.2858 | 0.3172 |
| spatial | True | 916 | 0.1772 | 0.3493 | 0.1004 | 0.4738 |
| attribute | False | 6519 | 0.3066 | 0.5243 | 0.2511 | 0.3519 |
| attribute | True | 4096 | 0.3454 | 0.5837 | 0.2996 | 0.2971 |
| color | False | 7163 | 0.3118 | 0.5302 | 0.2598 | 0.3490 |
| color | True | 3452 | 0.3417 | 0.5826 | 0.2906 | 0.2929 |
| clothing | False | 9032 | 0.3191 | 0.5370 | 0.2673 | 0.3365 |
| clothing | True | 1583 | 0.3354 | 0.6058 | 0.2843 | 0.2982 |

All methods, standard/strict and split-specific results are in subgroup_results.csv. Exactly the previous predefined word-boundary lexicons and length bins are used. Groups overlap and confound target size/category/scene difficulty; they do not establish causal language effects.

## Candidate and fallback diagnostics

| metric | mean | ci_low | ci_high |
| --- | --- | --- | --- |
| candidate_oracle | 0.4015 | 0.3920 | 0.4111 |
| candidate_fallback_oracle | 0.4312 | 0.4222 | 0.4398 |
| selected_fallback_oracle | 0.3595 | 0.3514 | 0.3673 |
| no_candidate_at_05 | 0.6040 | 0.5897 | 0.6179 |
| selector_regret | 0.0791 | 0.0754 | 0.0832 |
| fallback_rejects_useful | 0.0272 | 0.0233 | 0.0314 |
| fallback_prevents_harm | 0.0453 | 0.0401 | 0.0510 |
| fallback_accepts_harm | 0.3308 | 0.3183 | 0.3433 |

GT-only oracles do not enter inference. Poor candidates may reflect CLIP guidance, crop limits or SAM; these statistics do not identify a unique causal bottleneck. Fallback labels compare smart-selected SAM to CLIP before fallback.

## Twelve research answers

1. Naive SAM minus CLIP mIoU: +0.0937 [+0.0868, +0.1006]; positive change established by this interval. Final minus CLIP: +0.1137 [+0.1075, +0.1195].
2. Smart minus naive mIoU: +0.0209 [+0.0165, +0.0251]; positive change established by this interval. Final minus naive: +0.0200 [+0.0154, +0.0244].
3. Harm: naive 0.4463, smart 0.3761, final 0.3308. Final minus smart harm: -0.0453 [-0.0510, -0.0401]; final minus naive harm: -0.1155 [-0.1249, -0.1056]. Fallback IoU change: -0.0009 [-0.0023, +0.0005].
4. No RefCOCO+ tuning occurred. Frozen final mIoU is 0.3216; its paired gains/limitations are those above, not a fitted dataset-specific result.
5. Final correct-instance rate 0.5472, wrong-instance 0.4410, tie 0.0118. Final minus naive correct-instance: +0.0038 [-0.0022, +0.0097].
6. Instance ambiguity remains unresolved in a fraction 0.4528; see ambiguous-only denominator above. This is a major observed limitation, not proof of one causal failure source.
7. Long-expression final mIoU 0.2060, correct-instance 0.3600, over 989 expressions.
8. Final attribute/color/clothing results: attribute: mIoU 0.3454; color: mIoU 0.3417; clothing: mIoU 0.3354. All comparison methods and counts are retained in subgroup_results.csv.
9. No candidate reaches IoU 0.5 in 0.6040 of expressions.
10. Mean smart-selector regret is 0.0791; candidate oracle mIoU 0.4015.
11. Earlier RefCOCO CLIP/final mIoU was 0.2034/0.3071; here it is 0.2079/0.3216. Raw cross-dataset differences are descriptive, not controlled attribution degradation: expression and target distributions differ and image overlap is disclosed. Earlier selector gain was +0.0209; current smart-minus-naive is +0.0209 [+0.0165, +0.0251].
12. The same frozen pipeline is technically reusable for a separately authorized RefCOCOg evaluation. These results and their limitations inform that decision; no method repair or automatic RefCOCOg run follows. Stop for user review.

## Secondary unchanged CLIP crop

| method | iou | dice | p_at_05 | p_at_07 | worsened | correct_instance |
| --- | --- | --- | --- | --- | --- | --- |
| clip_baseline | 0.2211 | 0.3393 | 0.0516 | 0.0026 | 0.0000 | 0.5407 |
| sam_score | 0.3023 | 0.3880 | 0.2610 | 0.1455 | 0.4525 | 0.5383 |
| smart | 0.3257 | 0.4163 | 0.2866 | 0.1602 | 0.3803 | 0.5371 |
| frozen_final | 0.3259 | 0.4193 | 0.2802 | 0.1572 | 0.3332 | 0.5416 |
| center_score | 0.0943 | 0.1169 | 0.0886 | 0.0579 | 0.8479 | 0.3066 |

Crop-invisible target expressions: 152; retained, with secondary empty-target IoU/Dice defined as zero. SAM sees original RGB; CLIP/fallback is limited to its original crop.

## Runtime, integrity and reproduction

Inference including smoke/hashing/loading/saving: 1290.2 seconds; evaluation: 215.8 seconds. Not pure GPU runtime. All 10615 expressions retained. Zero/constant attribution maps 0/0; empty candidates 0. Exactly 20 technical smoke expressions reused; no smoke GT scoring. Final hash and decision replay audit: audit.json. [Representative panels](examples.html).

Commands: download_refcocoplus.py; prepare_refcocoplus.py; run_refcocoplus.py --smoke; run_refcocoplus.py --resume; evaluate_refcocoplus.py; report_refcocoplus.py; visualize_refcocoplus.py; audit_refcocoplus.py. Use existing pinned environment. All scripts are under scripts/. No completed experiment may be overwritten. Protocol: ../../REFCOCOPLUS_PLAN.md; reproduction notes: ../../REFCOCOPLUS_REPRODUCTION.md.
