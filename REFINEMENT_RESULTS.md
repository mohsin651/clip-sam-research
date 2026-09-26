# Source-informed refinement results

**The pathological map behavior is substantially corrected, but this is a CDA-inspired refinement, not an exact reproduction.**

## What changed

1. Min-max normalize the structural cosine prior before adding the unchanged nonnegative query-gradient constraint. This prevents negative spatial weights from reversing channel contributions.
2. Use the per-image mean attribution as the binary-mask threshold. This is a separate evaluation change, not a localization-method gain.
3. Keep full-channel attention, the same checkpoint, original ImageNet text targets, crop geometry and all 500 fixed images. No model training or GT-mask-dependent attribution is used.

The choices were frozen after the 20-image diagnostic, before the refined 500-image run. Native attention and projected Q/K alternatives were explored and are disclosed in outputs/refinement_diagnostics. The final minmax/raw-QK choice is not asserted to be the authors' unpublished implementation.

## Fair method comparison: same target masks and mean threshold

| variant | pg | epg | pacc | ap | iou |
|---|---|---|---|---|---|
| baseline | 0.7320 | 0.4668 | 0.7246 | 0.6531 | 0.4039 |
| refined full | 0.7700 | 0.5303 | 0.7640 | 0.6799 | 0.4233 |
| gcls | 0.7340 | 0.4668 | 0.7246 | 0.6531 | 0.4039 |
| old literal full | 0.1460 | 0.1463 | 0.5347 | 0.2742 | 0.0482 |
| normalized prior only | 0.7800 | 0.5357 | 0.7659 | 0.6841 | 0.4261 |

The old literal maps above are re-evaluated under the same new threshold. Original fixed-0.5 results remain in RESULTS.md. This separates changes to attribution from changes to binarization.

## Paired differences under that same protocol

| comparison | metric | mean_delta | 95% CI |
|---|---|---|---|
| refined - literal_full | pg | 0.6240 | [+0.58000, +0.67000] |
| refined - literal_full | epg | 0.3840 | [+0.35992, +0.40830] |
| refined - literal_full | pacc | 0.2292 | [+0.21261, +0.24658] |
| refined - literal_full | ap | 0.4058 | [+0.38335, +0.42915] |
| refined - literal_full | iou | 0.3751 | [+0.35344, +0.39623] |
| refined - baseline | pg | 0.0380 | [+0.01000, +0.06605] |
| refined - baseline | epg | 0.0635 | [+0.05758, +0.06955] |
| refined - baseline | pacc | 0.0393 | [+0.03237, +0.04645] |
| refined - baseline | ap | 0.0268 | [+0.01904, +0.03436] |
| refined - baseline | iou | 0.0194 | [+0.01141, +0.02727] |
| refined - prior_only | pg | -0.0100 | [-0.02200, +0.00000] |
| refined - prior_only | epg | -0.0054 | [-0.00688, -0.00406] |
| refined - prior_only | pacc | -0.0019 | [-0.00305, -0.00092] |
| refined - prior_only | ap | -0.0041 | [-0.00552, -0.00285] |
| refined - prior_only | iou | -0.0028 | [-0.00413, -0.00141] |

Intervals use 2,000 image-level paired bootstrap resamples, seed 2026. Prior-only is the normalized structural prior without the semantic R term; it directly tests whether that term adds value.
Under the primary protocol, refined full improves all five metrics over baseline, with paired 95% intervals excluding zero. However, adding R to the normalized prior slightly lowers all five means: EPG, PAcc, AP and IoU have negative paired intervals excluding zero; PG reaches zero at its upper endpoint. Thus the spatial-normalization repair is supported, while the claimed extra benefit of the semantic-gradient constraint is not reproduced.

## Protocol sensitivity, all 500 images

| variant | mask | threshold_rule | pg | epg | pacc | ap | iou |
|---|---|---|---|---|---|---|---|
| baseline | all_labeled | fixed | 0.8240 | 0.5391 | 0.7687 | 0.7324 | 0.4121 |
| baseline | all_labeled | mean | 0.8240 | 0.5391 | 0.7322 | 0.7324 | 0.4562 |
| baseline | target | fixed | 0.7320 | 0.4668 | 0.7857 | 0.6531 | 0.3780 |
| baseline | target | mean | 0.7320 | 0.4668 | 0.7246 | 0.6531 | 0.4039 |
| full | all_labeled | fixed | 0.8660 | 0.6088 | 0.7121 | 0.7628 | 0.2370 |
| full | all_labeled | mean | 0.8660 | 0.6088 | 0.7671 | 0.7628 | 0.4757 |
| full | target | fixed | 0.7700 | 0.5303 | 0.7502 | 0.6799 | 0.2210 |
| full | target | mean | 0.7700 | 0.5303 | 0.7640 | 0.6799 | 0.4233 |
| gcls | all_labeled | fixed | 0.8240 | 0.5391 | 0.7687 | 0.7324 | 0.4121 |
| gcls | all_labeled | mean | 0.8240 | 0.5391 | 0.7322 | 0.7324 | 0.4562 |
| gcls | target | fixed | 0.7340 | 0.4668 | 0.7857 | 0.6531 | 0.3780 |
| gcls | target | mean | 0.7340 | 0.4668 | 0.7246 | 0.6531 | 0.4039 |
| literal_full | all_labeled | fixed | 0.1920 | 0.1886 | 0.6377 | 0.3295 | 0.0084 |
| literal_full | all_labeled | mean | 0.1920 | 0.1886 | 0.5000 | 0.3295 | 0.0631 |
| literal_full | target | fixed | 0.1460 | 0.1463 | 0.6909 | 0.2742 | 0.0059 |
| literal_full | target | mean | 0.1460 | 0.1463 | 0.5347 | 0.2742 | 0.0482 |
| prior_only | all_labeled | fixed | 0.8700 | 0.6146 | 0.7129 | 0.7672 | 0.2402 |
| prior_only | all_labeled | mean | 0.8700 | 0.6146 | 0.7692 | 0.7672 | 0.4787 |
| prior_only | target | fixed | 0.7800 | 0.5357 | 0.7514 | 0.6841 | 0.2246 |
| prior_only | target | mean | 0.7800 | 0.5357 | 0.7659 | 0.6841 | 0.4261 |

The all-labeled-foreground protocol treats every valid annotated object as foreground. It is a different task from original-target-only localization. It cannot be substituted silently to obtain more attractive AP/IoU values.

## Paper context, not an exact-match target

| variant | pg | epg | pacc | ap | iou |
|---|---|---|---|---|---|
| paper full evaluation reference | 0.7585 | 0.3800 | 0.6341 | 0.7657 | 0.4465 |

Our primary target-only result: PG 0.7700, EPG 0.5303, PAcc 0.7640, AP 0.6799, mIoU 0.4233.
Secondary all-foreground/mean result: PG 0.8660, EPG 0.6088, PAcc 0.7671, AP 0.7628, mIoU 0.4757.
Differences in masks, crop, threshold and metric aggregation remain unresolved in the paper. Similar magnitudes do not establish equal protocols or exact reproduction.

## Development versus remaining samples

| partition | variant | mask | threshold_rule | pg | epg | pacc | ap | iou |
|---|---|---|---|---|---|---|---|---|
| development20 | baseline | target | mean | 0.5000 | 0.4269 | 0.7114 | 0.5590 | 0.3744 |
| development20 | full | target | mean | 0.6000 | 0.4930 | 0.7694 | 0.6174 | 0.4138 |
| development20 | gcls | target | mean | 0.5000 | 0.4269 | 0.7114 | 0.5590 | 0.3744 |
| development20 | literal_full | target | mean | 0.1000 | 0.1367 | 0.5646 | 0.2625 | 0.0495 |
| development20 | prior_only | target | mean | 0.6500 | 0.4986 | 0.7726 | 0.6248 | 0.4180 |
| remaining480 | baseline | target | mean | 0.7417 | 0.4685 | 0.7252 | 0.6571 | 0.4051 |
| remaining480 | full | target | mean | 0.7771 | 0.5319 | 0.7637 | 0.6825 | 0.4237 |
| remaining480 | gcls | target | mean | 0.7438 | 0.4685 | 0.7252 | 0.6571 | 0.4051 |
| remaining480 | literal_full | target | mean | 0.1479 | 0.1467 | 0.5335 | 0.2747 | 0.0482 |
| remaining480 | prior_only | target | mean | 0.7854 | 0.5373 | 0.7656 | 0.6865 | 0.4264 |

The 20 development images selected the implementation choices. The remaining 480 were evaluated after freezing these alternatives, but their earlier literal-method results were already seen. This is not an independent held-out benchmark.

## Validation

14 unit tests passed; the refined 20-image smoke test passed finite/nonconstant-map and prompt-sensitivity checks. The completed study has 1,500 primary rows and 10,000 disclosed method/protocol rows. Zero numerical failures; all expected artifacts and source hashes verified.
Study runtime: 367.1 seconds. Baseline/G_cls remain theoretically proportional under single-head attention; the published distinct ablation rows are still not explained.

## Sources

- Spatial normalization precedent: [official Grad-ECLIP code](https://github.com/Cyang-Zhao/Grad-Eclip/blob/e370e6cb194faf2020f5d1ed268f9d57e91a38e6/generate_emap.py).
- Mean threshold precedent: [official Transformer Explainability segmentation evaluator](https://github.com/hila-chefer/Transformer-Explainability/blob/main/baselines/ViT/imagenet_seg_eval.py).
- Exact new equations and selection history: REFINEMENT_PLAN.md; configuration: config_refined.yaml.

Dashboard: `python scripts/run_refined_dashboard.py` (port 8052); original results: `python scripts/run_dashboard.py` (port 8050).
