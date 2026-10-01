# Official Grad-ECLIP and frozen CASR transfer: dense COCO

With the frozen smart selector, mIoU increases, with this paired interval above zero: change +0.0393 [95% CI +0.0283, +0.0502]. Frozen fallback mIoU change +0.0046 [95% CI +0.0011, +0.0077]; own-map harm change -0.0570 [95% CI -0.0710, -0.0430]. No fitting, tuning or method repair.

Exactly the original 500 images / 1,000 category targets. This cohort is already inspected; it is not a newly untouched benchmark. All prior results/predictions are reused. Grad-ECLIP full-image preprocessing differs from original CDA crop and CS512 warp, so cross-source differences do not isolate attribution formulas. Within GE, all SAM variants share the exact same maps, POS3 points and three ViT-B candidates.

Official [Grad-ECLIP repository](https://github.com/Cyang-Zhao/Grad-Eclip), commit `e370e6cb194faf2020f5d1ed268f9d57e91a38e6`. Official ViT-B/16 image notebook definitions executed verbatim, last layer n=1; original dog/car/traffic-light demo passed finite/nonconstant, gradient and prompt-sensitivity checks. No license file or requirements file found. No algorithm edits. **Grad-ECLIP does not provide an official SAM refinement baseline in this inspected release.** GE_POS3 variants are controlled common-protocol experiments.

## Cross-attribution common-protocol comparison

| source | map_iou | naive_iou | smart_iou | casr_iou | naive_harm | final_harm |
| --- | --- | --- | --- | --- | --- | --- |
| CDA-inspired | 0.2699 | 0.3949 | 0.4291 | 0.4335 | 0.4110 | 0.2620 |
| CLIP Surgery | 0.2997 | 0.4295 | 0.4648 | 0.4684 | 0.4060 | 0.1460 |
| Grad-ECLIP | 0.2683 | 0.3972 | 0.4365 | 0.4411 | 0.4120 | 0.2650 |

Harm uses each source's own binary map baseline, so cross-source harm rates have different references. Official CS Text2Points/ViT-H mIoU 0.5000 is separate context and is not the controlled comparator. Original smart mIoU is retained even where earlier summary prompts omitted it.

## Attribution maps

| method | pg | epg | ap | pacc | iou |
| --- | --- | --- | --- | --- | --- |
| clip_baseline | 0.6680 | 0.3725 | 0.5422 | 0.7378 | 0.2699 |
| CS_MAP | 0.6690 | 0.2706 | 0.6438 | 0.6985 | 0.2997 |
| GE_MAP | 0.6930 | 0.3806 | 0.5445 | 0.7371 | 0.2683 |

Native GE maps retain raw nonnegative gradient scale; no CDA or CS normalization. Existing mean-threshold binary evaluation, unchanged target masks/crop. AP is per-target pixel AP, not COCO detection AP. EPG depends on map scale/zero conventions. CIs for all map metrics are in attribution_summary.csv. No continuous metrics are invented for SAM masks.

## Grad-ECLIP binary quality

| method | iou | iou_ci_low | iou_ci_high | dice | precision | recall | pacc |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GE_MAP | 0.2683 | 0.2549 | 0.2817 | 0.3845 | 0.3413 | 0.7518 | 0.7371 |
| GE_POS3_SAM | 0.3972 | 0.3766 | 0.4187 | 0.4847 | 0.5762 | 0.5432 | 0.8255 |
| GE_POS3_SMART | 0.4365 | 0.4160 | 0.4577 | 0.5275 | 0.5786 | 0.6080 | 0.8465 |
| GE_POS3_CASR | 0.4411 | 0.4206 | 0.4619 | 0.5366 | 0.5842 | 0.6288 | 0.8508 |

Pair-macro foreground category-union IoU; all failures/fallback cases retained.

## Primary transfer and all paired comparisons

| comparison | metric | difference | ci_low | ci_high |
| --- | --- | --- | --- | --- |
| GE_MAP minus clip_baseline | iou | -0.0016 | -0.0055 | 0.0020 |
| GE_MAP minus clip_baseline | dice | -0.0035 | -0.0079 | 0.0006 |
| GE_MAP minus clip_baseline | own_baseline_harm | 0.0000 | 0.0000 | 0.0000 |
| GE_MAP minus clip_baseline | semantic_success | -0.0090 | -0.0250 | 0.0080 |
| GE_MAP minus clip_baseline | precise_wrong_object | 0.0080 | 0.0000 | 0.0160 |
| GE_MAP minus CS_MAP | iou | -0.0314 | -0.0385 | -0.0241 |
| GE_MAP minus CS_MAP | dice | -0.0238 | -0.0313 | -0.0161 |
| GE_MAP minus CS_MAP | own_baseline_harm | 0.0000 | 0.0000 | 0.0000 |
| GE_MAP minus CS_MAP | semantic_success | -0.0090 | -0.0270 | 0.0100 |
| GE_MAP minus CS_MAP | precise_wrong_object | -0.0230 | -0.0360 | -0.0110 |
| GE_POS3_SAM minus GE_MAP | iou | 0.1290 | 0.1102 | 0.1480 |
| GE_POS3_SAM minus GE_MAP | dice | 0.1002 | 0.0812 | 0.1201 |
| GE_POS3_SAM minus GE_MAP | own_baseline_harm | 0.4120 | 0.3800 | 0.4420 |
| GE_POS3_SAM minus GE_MAP | semantic_success | 0.1110 | 0.0830 | 0.1380 |
| GE_POS3_SAM minus GE_MAP | precise_wrong_object | 0.0610 | 0.0430 | 0.0800 |
| GE_POS3_SAM minus sam_score | iou | 0.0023 | -0.0154 | 0.0204 |
| GE_POS3_SAM minus sam_score | dice | 0.0017 | -0.0171 | 0.0211 |
| GE_POS3_SAM minus sam_score | own_baseline_harm | 0.0010 | -0.0330 | 0.0340 |
| GE_POS3_SAM minus sam_score | semantic_success | 0.0100 | -0.0170 | 0.0370 |
| GE_POS3_SAM minus sam_score | precise_wrong_object | 0.0060 | -0.0120 | 0.0260 |
| GE_POS3_SAM minus CS_POS3_SAM | iou | -0.0323 | -0.0511 | -0.0114 |
| GE_POS3_SAM minus CS_POS3_SAM | dice | -0.0398 | -0.0601 | -0.0176 |
| GE_POS3_SAM minus CS_POS3_SAM | own_baseline_harm | 0.0060 | -0.0310 | 0.0410 |
| GE_POS3_SAM minus CS_POS3_SAM | semantic_success | -0.0810 | -0.1080 | -0.0520 |
| GE_POS3_SAM minus CS_POS3_SAM | precise_wrong_object | 0.0370 | 0.0170 | 0.0560 |
| GE_POS3_SMART minus GE_POS3_SAM | iou | 0.0393 | 0.0283 | 0.0502 |
| GE_POS3_SMART minus GE_POS3_SAM | dice | 0.0428 | 0.0322 | 0.0537 |
| GE_POS3_SMART minus GE_POS3_SAM | own_baseline_harm | -0.0900 | -0.1110 | -0.0680 |
| GE_POS3_SMART minus GE_POS3_SAM | semantic_success | 0.0050 | -0.0100 | 0.0200 |
| GE_POS3_SMART minus GE_POS3_SAM | precise_wrong_object | -0.0040 | -0.0190 | 0.0100 |
| GE_POS3_CASR minus GE_POS3_SMART | iou | 0.0046 | 0.0011 | 0.0077 |
| GE_POS3_CASR minus GE_POS3_SMART | dice | 0.0091 | 0.0048 | 0.0132 |
| GE_POS3_CASR minus GE_POS3_SMART | own_baseline_harm | -0.0570 | -0.0710 | -0.0430 |
| GE_POS3_CASR minus GE_POS3_SMART | semantic_success | 0.0070 | -0.0010 | 0.0140 |
| GE_POS3_CASR minus GE_POS3_SMART | precise_wrong_object | -0.0060 | -0.0110 | -0.0020 |
| GE_POS3_CASR minus GE_POS3_SAM | iou | 0.0439 | 0.0323 | 0.0544 |
| GE_POS3_CASR minus GE_POS3_SAM | dice | 0.0519 | 0.0402 | 0.0632 |
| GE_POS3_CASR minus GE_POS3_SAM | own_baseline_harm | -0.1470 | -0.1710 | -0.1220 |
| GE_POS3_CASR minus GE_POS3_SAM | semantic_success | 0.0120 | -0.0050 | 0.0280 |
| GE_POS3_CASR minus GE_POS3_SAM | precise_wrong_object | -0.0100 | -0.0250 | 0.0040 |
| GE_POS3_CASR minus frozen_final | iou | 0.0076 | -0.0076 | 0.0236 |
| GE_POS3_CASR minus frozen_final | dice | 0.0072 | -0.0091 | 0.0241 |
| GE_POS3_CASR minus frozen_final | own_baseline_harm | 0.0030 | -0.0290 | 0.0330 |
| GE_POS3_CASR minus frozen_final | semantic_success | 0.0170 | -0.0080 | 0.0420 |
| GE_POS3_CASR minus frozen_final | precise_wrong_object | 0.0000 | -0.0160 | 0.0160 |
| GE_POS3_CASR minus CS_POS3_CASR | iou | -0.0273 | -0.0436 | -0.0090 |
| GE_POS3_CASR minus CS_POS3_CASR | dice | -0.0362 | -0.0532 | -0.0177 |
| GE_POS3_CASR minus CS_POS3_CASR | own_baseline_harm | 0.1190 | 0.0870 | 0.1500 |
| GE_POS3_CASR minus CS_POS3_CASR | semantic_success | -0.0360 | -0.0610 | -0.0110 |
| GE_POS3_CASR minus CS_POS3_CASR | precise_wrong_object | 0.0290 | 0.0120 | 0.0450 |
| GE_POS3_SMART minus GE_POS3_SAM | best_candidate_accuracy | 0.1330 | 0.0960 | 0.1700 |
| GE_POS3_SMART minus GE_POS3_SAM | selector_regret | -0.0393 | -0.0502 | -0.0283 |

2,000 paired whole-image bootstrap resamples, seed 2026, both prompts grouped; no multiplicity adjustment. Cross-source own-baseline harm contrasts compare different references.

## Harm against GE_MAP

| method | improved | worsened | unchanged | delta_iou | median_delta | mean_positive_delta | mean_negative_delta | worst_decile_delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GE_MAP | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| GE_POS3_SAM | 0.5850 | 0.4120 | 0.0030 | 0.1290 | 0.0685 | 0.3227 | -0.1451 | -0.3399 |
| GE_POS3_SMART | 0.6750 | 0.3220 | 0.0030 | 0.1683 | 0.1310 | 0.3052 | -0.1173 | -0.2538 |
| GE_POS3_CASR | 0.6540 | 0.2650 | 0.0810 | 0.1728 | 0.1187 | 0.3094 | -0.1115 | -0.2247 |

Conditional positive/negative means and mean lowest10% delta are shown. Full CIs for fractions/mean deltas are in method_summary.csv.

## Semantic overlap proxies

| method | semantic_success | semantic_success_ci_low | semantic_success_ci_high | precise_wrong_object | precise_wrong_object_ci_low | precise_wrong_object_ci_high |
| --- | --- | --- | --- | --- | --- | --- |
| GE_MAP | 0.6310 | 0.6040 | 0.6570 | 0.0230 | 0.0140 | 0.0320 |
| GE_POS3_SAM | 0.7420 | 0.7160 | 0.7680 | 0.0840 | 0.0670 | 0.1010 |
| GE_POS3_SMART | 0.7470 | 0.7220 | 0.7720 | 0.0800 | 0.0640 | 0.0970 |
| GE_POS3_CASR | 0.7540 | 0.7290 | 0.7800 | 0.0740 | 0.0580 | 0.0900 |

Smart-minus-naive semantic success: +0.0050 [95% CI -0.0100, +0.0200]; fallback-minus-smart: +0.0070 [95% CI -0.0010, +0.0140]. Target-union IoU must exceed every other category-union IoU. Precise wrong object requires other-union IoU>=.5 and greater than target. These proxies are not manual object identity; overlaps/instance unions can affect them.

## Candidate choice and oracles

| source | sam_score_accuracy | smart_accuracy | candidate_oracle | no_candidate_at_05 | sam_score_regret | smart_regret |
| --- | --- | --- | --- | --- | --- | --- |
| CDA-inspired | 0.4390 | 0.5810 | 0.4878 | 0.4920 | 0.0929 | 0.0588 |
| CLIP Surgery | 0.4780 | 0.5440 | 0.5273 | 0.4350 | 0.0977 | 0.0625 |
| Grad-ECLIP | 0.4400 | 0.5730 | 0.4941 | 0.4800 | 0.0969 | 0.0576 |

Primary GE accuracy difference +0.1330 [95% CI +0.0960, +0.1700]; regret difference -0.0393 [95% CI -0.0502, -0.0283]. Tied optima within1e-8 count as correct; strict index accuracy and intervals are retained in candidate_analysis.json. Oracles use GT only after complete inference and are not deployable.

## Thirteen research answers

1. Yes: official image-demo method cells and example ran successfully, including enabled gradients, text sensitivity and finite/nonconstant maps. This reproduces released code under our protocol, not published benchmark numbers.
2. GE map: PG 0.6930, EPG 0.3806, AP 0.5445, PAcc 0.7371, mIoU 0.2683.
3. Naive GE SAM versus its map: +0.1290 [95% CI +0.1102, +0.1480]; mIoU increases, with this paired interval above zero.
4. Smart-selector mIoU increases, with this paired interval above zero.
5. Primary paired mIoU gain: +0.0393 [95% CI +0.0283, +0.0502]; Dice difference +0.0428 [95% CI +0.0322, +0.0537].
6. Best-candidate accuracy increases, with this paired interval above zero: +0.1330 [95% CI +0.0960, +0.1700].
7. Selector regret decreases, with this paired interval below zero: -0.0393 [95% CI -0.0502, -0.0283].
8. Fallback harm decreases, with this paired interval below zero: -0.0570 [95% CI -0.0710, -0.0430].
9. Fallback mean IoU increases, with this paired interval above zero: +0.0046 [95% CI +0.0011, +0.0077].
10. Smart semantic success has an interval spanning zero; a change is not established; fallback semantic success has an interval spanning zero; a change is not established. Full rates and negative findings are retained above.
11. The positive GE selector result, alongside original CDA and CS results, supports frozen candidate-selection transfer across these three tested sources on this fixed cohort. Fallback and semantic effects must be stated separately; this is not universal transfer.
12. One already-inspected selected COCO cohort, category unions, overlapping annotations, different upstream views/resolutions, raw attribution scaling and exploratory unadjusted comparisons remain limitations. Candidate oracles do not prove recoverable practical gains; no semantic understanding claim follows from IoU.
13. Stop external-attribution runs as requested. The current results are enough to draft a bounded evidence-based research narrative, not to claim comprehensive superiority or paper acceptance. Paper preparation may summarize these findings and limitations; new experiments require separate authorization.

## Integrity, runtime and reproduction

Inference including 20-pair smoke, initialization, hashes/loading/saving: 332.9s; evaluation: 12.3s; peak allocated CUDA 3.10 GiB. Excludes implementation/demo/reporting. All 1,000 targets retained; zero/constant maps 0/0, empty candidates 0. 69 tests passed before inference. No GT entered features or decisions; evaluation started only after complete inference and verified predictions. Source/freeze/previous-artifact audit: audit.json.

[Representative panels](outputs/external_baselines/grad_eclip/examples.html), [reproduction notes](outputs/external_baselines/grad_eclip/GRAD_ECLIP_REPRODUCTION.md), root GRAD_ECLIP_PLAN.md, saved raw maps/candidates/features/points and exact source/checkpoint hashes are retained. Examples follow predeclared outcome-based selection and do not estimate prevalence. No previous experiment was overwritten; no additional method/dataset or paper experiment follows automatically.
