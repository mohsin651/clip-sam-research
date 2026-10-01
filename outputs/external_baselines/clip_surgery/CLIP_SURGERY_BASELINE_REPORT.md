# Official CLIP Surgery and frozen CASR transfer on dense heldout COCO

All 500 original heldout images / 1,000 category targets, unchanged masks and crop, all outcomes retained. No fitting or tuning. This cohort had already been inspected in prior studies; this is a fixed external comparison, not a newly untouched benchmark.

Official CLIP Surgery + SAM has higher observed mIoU (0.5000) than our existing frozen CASR (0.4335); paired difference +0.0665 [95% CI +0.0473, +0.0865]. Its ViT-H and variable positive/negative prompts differ from our ViT-B/POS3 protocol. We cannot claim overall superiority over the official external pipeline. Within the common protocol, the unchanged selector transfers: +0.0352 [95% CI +0.0238, +0.0479]. Fallback harm change is -0.1880 [95% CI -0.2130, -0.1640], while its mIoU change is +0.0036 [95% CI -0.0067, +0.0140].

Tradeoff: semantic-success overlap proxy decreases from 0.823 (naive) to 0.809 (smart) and 0.790 (fallback). Smart-minus-naive: -0.0140 [95% CI -0.0270, -0.0010]; fallback-minus-smart: -0.0190 [95% CI -0.0370, -0.0030]. The mIoU/harm benefit does not establish better semantic selection.

## Official implementation and sanity

Official [CLIP Surgery repository](https://github.com/xmed-lab/clip_surgery), commit `d4696d47f49cfe70f49140afe5eb94f94c5f59bc`. Entire unmodified demo completed, 19 figures saved, finite/nonconstant maps and working Text2Points/SAM; source hashes unchanged. No license or requirements file was present. The released demo is reproduced under our protocol; published paper scores are not reproduced. See [reproduction notes](CLIP_SURGERY_REPRODUCTION.md) and `official_repo_info.json`.

CS_OFFICIAL_SAM uses official single-target feature surgery, 85-template ensemble, empty-text redundant feature, full-image 512 warp, official variable positive/negative Text2Points (t=0.8, downsample=2), SAM ViT-H and highest predicted SAM score. CS_POS3_* are our controlled hybrids: same official map, unchanged POS3 rule, SAM ViT-B, frozen candidate features/ranker/density fallback. Original CASR uses its existing attribution. Different image views and text ensembles remain upstream confounders; official SAM additionally changes backbone and prompting.

## Continuous map comparison

| method | pg | epg | ap | pacc | iou |
| --- | --- | --- | --- | --- | --- |
| clip_baseline | 0.6680 | 0.3725 | 0.5422 | 0.7378 | 0.2699 |
| CS_MAP | 0.6690 | 0.2706 | 0.6438 | 0.6985 | 0.2997 |

PG/EPG/AP apply only to continuous maps; AP is per-target pixel average precision, not official COCO detection AP. CS uses official token min-max normalization, no CDA normalization. EPG depends on this zero point and is not calibrated across attribution methods. Binary maps use the unchanged mean threshold and exact original target masks/crop. Full CIs are in attribution_summary.csv.

## All nine methods

| method | iou | iou_ci_low | iou_ci_high | dice | precision | recall | pacc | semantic_success | precise_wrong_object |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| clip_baseline | 0.2699 | 0.2568 | 0.2836 | 0.3880 | 0.3395 | 0.7705 | 0.7378 | 0.6400 | 0.0150 |
| sam_score | 0.3949 | 0.3745 | 0.4158 | 0.4830 | 0.5746 | 0.5472 | 0.8201 | 0.7320 | 0.0780 |
| frozen_final | 0.4335 | 0.4136 | 0.4542 | 0.5293 | 0.5764 | 0.6407 | 0.8469 | 0.7370 | 0.0740 |
| center_score | 0.1019 | 0.0884 | 0.1156 | 0.1284 | 0.2804 | 0.1225 | 0.7522 | 0.3520 | 0.1040 |
| CS_MAP | 0.2997 | 0.2837 | 0.3158 | 0.4083 | 0.3395 | 0.8902 | 0.6985 | 0.6400 | 0.0460 |
| CS_POS3_SAM | 0.4295 | 0.4089 | 0.4501 | 0.5245 | 0.6210 | 0.5675 | 0.8360 | 0.8230 | 0.0470 |
| CS_POS3_SMART | 0.4648 | 0.4443 | 0.4852 | 0.5596 | 0.6129 | 0.6387 | 0.8455 | 0.8090 | 0.0570 |
| CS_POS3_CASR | 0.4684 | 0.4491 | 0.4878 | 0.5728 | 0.5965 | 0.7564 | 0.8272 | 0.7900 | 0.0450 |
| CS_OFFICIAL_SAM | 0.5000 | 0.4789 | 0.5216 | 0.5972 | 0.6909 | 0.6260 | 0.8736 | 0.8290 | 0.0600 |

IoU is pair-macro foreground category-union IoU. Original clip_baseline, sam_score, frozen_final and center_score are reused predictions/results. SAM-score and frozen CASR are the original paired comparators, not earlier development scores. Semantic success is target-union IoU greater than every other category-union IoU; precise wrong object requires other-union IoU>=0.5 and greater than target. These are overlap proxies, not manual semantic labels.

## Harm relative to each source map

| method | improved | worsened | unchanged | delta_iou | median_delta | mean_positive_delta | mean_negative_delta | worst_decile_delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| clip_baseline | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| sam_score | 0.5880 | 0.4110 | 0.0010 | 0.1250 | 0.0618 | 0.3108 | -0.1404 | -0.3315 |
| frozen_final | 0.6370 | 0.2620 | 0.1010 | 0.1636 | 0.1044 | 0.3033 | -0.1130 | -0.2221 |
| center_score | 0.1230 | 0.8770 | 0.0000 | -0.1680 | -0.1455 | 0.2611 | -0.2282 | -0.5912 |
| CS_MAP | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| CS_POS3_SAM | 0.5930 | 0.4060 | 0.0010 | 0.1299 | 0.0895 | 0.3573 | -0.2019 | -0.4849 |
| CS_POS3_SMART | 0.6650 | 0.3340 | 0.0010 | 0.1651 | 0.1319 | 0.3322 | -0.1671 | -0.4067 |
| CS_POS3_CASR | 0.5460 | 0.1460 | 0.3080 | 0.1687 | 0.0536 | 0.3485 | -0.1479 | -0.2067 |
| CS_OFFICIAL_SAM | 0.7040 | 0.2950 | 0.0010 | 0.2003 | 0.1456 | 0.3388 | -0.1297 | -0.2905 |

Fractions use all 1,000 pairs, including fallback/unchanged cases. Original methods and center use original CLIP; CS methods use CS_MAP. Cross-source harm contrasts have different baselines. Conditional mean changes and worst 10% mean are included; CIs for fractions/deltas are in method_summary.csv.

## Paired whole-image differences

| comparison | metric | difference | ci_low | ci_high |
| --- | --- | --- | --- | --- |
| CS_MAP minus clip_baseline | iou | 0.0298 | 0.0228 | 0.0364 |
| CS_MAP minus clip_baseline | semantic_success | 0.0000 | -0.0190 | 0.0200 |
| CS_MAP minus clip_baseline | precise_wrong_object | 0.0310 | 0.0190 | 0.0440 |
| CS_MAP minus clip_baseline | own_baseline_harm | 0.0000 | 0.0000 | 0.0000 |
| CS_OFFICIAL_SAM minus CS_MAP | iou | 0.2003 | 0.1804 | 0.2192 |
| CS_OFFICIAL_SAM minus CS_MAP | semantic_success | 0.1890 | 0.1630 | 0.2160 |
| CS_OFFICIAL_SAM minus CS_MAP | precise_wrong_object | 0.0140 | -0.0040 | 0.0310 |
| CS_OFFICIAL_SAM minus CS_MAP | own_baseline_harm | 0.2950 | 0.2660 | 0.3240 |
| CS_POS3_SAM minus sam_score | iou | 0.0346 | 0.0141 | 0.0537 |
| CS_POS3_SAM minus sam_score | semantic_success | 0.0910 | 0.0620 | 0.1220 |
| CS_POS3_SAM minus sam_score | precise_wrong_object | -0.0310 | -0.0500 | -0.0120 |
| CS_POS3_SAM minus sam_score | own_baseline_harm | -0.0050 | -0.0380 | 0.0310 |
| CS_POS3_SMART minus CS_POS3_SAM | iou | 0.0352 | 0.0238 | 0.0479 |
| CS_POS3_SMART minus CS_POS3_SAM | semantic_success | -0.0140 | -0.0270 | -0.0010 |
| CS_POS3_SMART minus CS_POS3_SAM | precise_wrong_object | 0.0100 | -0.0010 | 0.0210 |
| CS_POS3_SMART minus CS_POS3_SAM | own_baseline_harm | -0.0720 | -0.0920 | -0.0520 |
| CS_POS3_CASR minus CS_POS3_SMART | iou | 0.0036 | -0.0067 | 0.0140 |
| CS_POS3_CASR minus CS_POS3_SMART | semantic_success | -0.0190 | -0.0370 | -0.0030 |
| CS_POS3_CASR minus CS_POS3_SMART | precise_wrong_object | -0.0120 | -0.0230 | -0.0010 |
| CS_POS3_CASR minus CS_POS3_SMART | own_baseline_harm | -0.1880 | -0.2130 | -0.1640 |
| CS_POS3_CASR minus CS_POS3_SAM | iou | 0.0388 | 0.0234 | 0.0542 |
| CS_POS3_CASR minus CS_POS3_SAM | semantic_success | -0.0330 | -0.0530 | -0.0130 |
| CS_POS3_CASR minus CS_POS3_SAM | precise_wrong_object | -0.0020 | -0.0160 | 0.0120 |
| CS_POS3_CASR minus CS_POS3_SAM | own_baseline_harm | -0.2600 | -0.2900 | -0.2310 |
| CS_POS3_CASR minus frozen_final | iou | 0.0349 | 0.0174 | 0.0508 |
| CS_POS3_CASR minus frozen_final | semantic_success | 0.0530 | 0.0280 | 0.0790 |
| CS_POS3_CASR minus frozen_final | precise_wrong_object | -0.0290 | -0.0470 | -0.0120 |
| CS_POS3_CASR minus frozen_final | own_baseline_harm | -0.1160 | -0.1480 | -0.0830 |
| CS_POS3_SAM minus frozen_final | iou | -0.0039 | -0.0248 | 0.0152 |
| CS_POS3_SAM minus frozen_final | semantic_success | 0.0860 | 0.0570 | 0.1140 |
| CS_POS3_SAM minus frozen_final | precise_wrong_object | -0.0270 | -0.0460 | -0.0080 |
| CS_POS3_SAM minus frozen_final | own_baseline_harm | 0.1440 | 0.1090 | 0.1810 |
| CS_OFFICIAL_SAM minus sam_score | iou | 0.1050 | 0.0844 | 0.1254 |
| CS_OFFICIAL_SAM minus sam_score | semantic_success | 0.0970 | 0.0660 | 0.1280 |
| CS_OFFICIAL_SAM minus sam_score | precise_wrong_object | -0.0180 | -0.0380 | 0.0020 |
| CS_OFFICIAL_SAM minus sam_score | own_baseline_harm | -0.1160 | -0.1490 | -0.0800 |
| CS_OFFICIAL_SAM minus frozen_final | iou | 0.0665 | 0.0473 | 0.0865 |
| CS_OFFICIAL_SAM minus frozen_final | semantic_success | 0.0920 | 0.0640 | 0.1200 |
| CS_OFFICIAL_SAM minus frozen_final | precise_wrong_object | -0.0140 | -0.0330 | 0.0050 |
| CS_OFFICIAL_SAM minus frozen_final | own_baseline_harm | 0.0330 | -0.0020 | 0.0670 |

2,000 whole-image paired bootstrap resamples, seed 2026, both target prompts grouped; pair-macro weighting, no multiplicity adjustment. Official versus original methods is descriptive and confounded by SAM backbone/prompt/view/template differences.

## Candidate choice: post-hoc diagnostics

| method | selected_iou | candidate_oracle | selector_regret | best_candidate_accuracy | strict_argmax_accuracy | selector_regret_ci_low | selector_regret_ci_high | best_candidate_accuracy_ci_low | best_candidate_accuracy_ci_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CS_POS3_SAM | 0.4295 | 0.5273 | 0.0977 | 0.4780 | 0.4610 | 0.0880 | 0.1083 | 0.4470 | 0.5070 |
| CS_POS3_SMART | 0.4648 | 0.5273 | 0.0625 | 0.5440 | 0.5280 | 0.0540 | 0.0714 | 0.5130 | 0.5750 |
| CS_OFFICIAL_SAM | 0.5000 | 0.5708 | 0.0708 | 0.4950 | 0.4740 | 0.0617 | 0.0805 | 0.4650 | 0.5260 |

| protocol | candidate_oracle | no_candidate_at_05 | candidate_plus_fallback_oracle | ci_low | ci_high |
| --- | --- | --- | --- | --- | --- |
| common | 0.5273 | 0.4350 | 0.5759 | 0.5077 | 0.5467 |
| official | 0.5708 | 0.3680 | 0.5928 | 0.5504 | 0.5900 |

Best-candidate accuracy counts tied optima within 1e-8 as correct; strict index argmax is also shown. Oracles are evaluation-only and cannot be deployed. Common variants share exactly the same three candidates.

## Ten research answers

1. Yes: the released official demo and its SAM path completed unchanged, and the benchmark uses its imported code. This supports a released-code external baseline, not an exact paper evaluation reproduction.
2. CS_MAP reaches PG 0.6690, EPG 0.2706, AP 0.6438, binary mIoU 0.2997. Versus original map: +0.0298 [95% CI +0.0228, +0.0364].
3. Official CS Text2Points + SAM ViT-H reaches mIoU 0.5000; versus CS_MAP: +0.2003 [95% CI +0.1804, +0.2192].
4. Shared POS3/ViT-B: CS mIoU 0.4295, original attribution 0.3949; CS minus original +0.0346 [95% CI +0.0141, +0.0537], a positive difference supported by this interval. This compares upstream packages, not just attribution equations.
5. Frozen smart selection changes CS mIoU by +0.0352 [95% CI +0.0238, +0.0479]: a positive difference supported by this interval. Own-map harm change: -0.0720 [95% CI -0.0920, -0.0520].
6. Frozen fallback changes CS mIoU by +0.0036 [95% CI -0.0067, +0.0140]; harm change -0.1880 [95% CI -0.2130, -0.1640]. Final harm is 0.1460, naive harm 0.4060.
7. The selector result supports transfer of the frozen selector on this cohort. Fallback is a separate tradeoff reported above; feature compatibility alone does not establish performance transfer or universality.
8. Against original naive SAM, official CS difference is +0.1050 [95% CI +0.0844, +0.1254]; common CS difference is +0.0346 [95% CI +0.0141, +0.0537]. Both protocols are retained; neither is substituted for the other.
9. Against existing final CASR, common CS naive difference is -0.0039 [95% CI -0.0248, +0.0152], an interval spanning zero, so superiority is not established; common CS with frozen CASR +0.0349 [95% CI +0.0174, +0.0508], a positive difference supported by this interval. Official CS difference is +0.0665 [95% CI +0.0473, +0.0865], with additional confounders.
10. Using CLIP-derived points to prompt SAM is already demonstrated by CLIP Surgery and is not our novelty. The narrower candidate-selection result supports transfer of the frozen selector on this cohort; any defensible contribution must be framed around measured selection/harm tradeoffs, not universal superiority or solved semantics. In fact, the final-versus-naive semantic proxy changes by -0.0330 [95% CI -0.0530, -0.0130]. No further baseline or method change follows this study.

## Runtime and integrity

Benchmark inference including smoke, initialization, hashing/loading/saving: 854.1s; evaluation: 24.1s. Peak CUDA allocated: 6.55 GiB. Runtime excludes implementation, downloads, demo and reporting. All 1,000 pairs retained; zero/constant maps 0/0, empty common/official candidates 0/0. Old baseline scores reproduced exactly. GT loaded only after complete inference and prediction hash verification. See final `audit.json` for previous-output preservation and decision replay.

## Examples and reproduction

[Representative paired examples](examples.html) cover the eight predeclared groups where eligible; empty groups are disclosed. Panels are outcome-selected illustrations, not prevalence estimates. [Reproduction notes](CLIP_SURGERY_REPRODUCTION.md), root CLIP_SURGERY_PLAN.md, raw predictions, features, prompts, source/checkpoint signatures and full CSV/JSON metrics are retained. Original CASR freeze is unchanged. No RefCOCO or Grad-ECLIP evaluation was run.
