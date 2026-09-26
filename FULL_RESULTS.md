# Full ImageNet-S919 validation results

Evaluated **all 12,419 validation images** with the frozen CDA-inspired refinement. Valid metric images: **12419**; explicit annotation failures: **0**.
No method, checkpoint, prompts, threshold rule or subset-dependent parameter was retuned for this evaluation. This is the refined method, not the literal paper equations.

## Full refined-method numbers

| mask | pg | epg | pacc | ap | iou |
|---|---|---|---|---|---|
| all_labeled | 0.8685 | 0.6196 | 0.7624 | 0.7714 | 0.4807 |
| target | 0.7812 | 0.5463 | 0.7603 | 0.6950 | 0.4327 |

Both rows use the same attribution maps and mean-score threshold. `target` means original ImageNet target-class pixels; `all_labeled` means every valid annotated foreground object. They are different evaluation tasks.

## 95% bootstrap intervals

| mask | metric | mean | n | 95% CI |
|---|---|---|---|---|
| target | pg | 0.7812 | 12419 | [0.77413, 0.78863] |
| target | epg | 0.5463 | 12419 | [0.54097, 0.55144] |
| target | pacc | 0.7603 | 12419 | [0.75821, 0.76260] |
| target | ap | 0.6950 | 12419 | [0.69007, 0.70018] |
| target | iou | 0.4327 | 12419 | [0.42858, 0.43670] |
| all_labeled | pg | 0.8685 | 12419 | [0.86255, 0.87455] |
| all_labeled | epg | 0.6196 | 12419 | [0.61518, 0.62389] |
| all_labeled | pacc | 0.7624 | 12419 | [0.76020, 0.76475] |
| all_labeled | ap | 0.7714 | 12419 | [0.76779, 0.77508] |
| all_labeled | iou | 0.4807 | 12419 | [0.47726, 0.48393] |

2,000 image-level bootstrap resamples, seed 2026. These describe variability across images; all finite-validation-set images have been evaluated. They are not subset-sampling uncertainty about the already-complete validation set.

## All controls, identical protocols

| mask | variant | pg | epg | pacc | ap | iou |
|---|---|---|---|---|---|---|
| all_labeled | baseline | 0.8162 | 0.5546 | 0.7350 | 0.7448 | 0.4675 |
| all_labeled | full | 0.8685 | 0.6196 | 0.7624 | 0.7714 | 0.4807 |
| all_labeled | gcls | 0.8158 | 0.5546 | 0.7350 | 0.7448 | 0.4675 |
| all_labeled | literal_full | 0.1957 | 0.1936 | 0.4915 | 0.3369 | 0.0635 |
| all_labeled | prior_only | 0.8718 | 0.6256 | 0.7647 | 0.7765 | 0.4840 |
| target | baseline | 0.7358 | 0.4865 | 0.7270 | 0.6714 | 0.4187 |
| target | full | 0.7812 | 0.5463 | 0.7603 | 0.6950 | 0.4327 |
| target | gcls | 0.7356 | 0.4865 | 0.7270 | 0.6714 | 0.4187 |
| target | literal_full | 0.1577 | 0.1563 | 0.5253 | 0.2870 | 0.0506 |
| target | prior_only | 0.7848 | 0.5517 | 0.7625 | 0.6997 | 0.4357 |

## Paired primary-protocol comparisons

| comparison | metric | mean_delta | 95% CI |
|---|---|---|---|
| target/full_minus_baseline | pg | 0.0454 | [+0.03929, +0.05186] |
| target/full_minus_baseline | epg | 0.0598 | [+0.05861, +0.06090] |
| target/full_minus_baseline | pacc | 0.0332 | [+0.03184, +0.03456] |
| target/full_minus_baseline | ap | 0.0236 | [+0.02207, +0.02508] |
| target/full_minus_baseline | iou | 0.0140 | [+0.01253, +0.01554] |
| target/full_minus_prior_only | pg | -0.0035 | [-0.00596, -0.00105] |
| target/full_minus_prior_only | epg | -0.0054 | [-0.00565, -0.00515] |
| target/full_minus_prior_only | pacc | -0.0023 | [-0.00247, -0.00205] |
| target/full_minus_prior_only | ap | -0.0047 | [-0.00497, -0.00439] |
| target/full_minus_prior_only | iou | -0.0031 | [-0.00332, -0.00280] |

The refined full method improves all five primary metrics over baseline, with paired 95% intervals excluding zero. However, prior-only remains better than full on all five metrics, now also with paired intervals excluding zero. The full validation set supports the spatial-normalization repair but does not support an additional benefit from the semantic-gradient R term.

## Prior 500 versus complete validation

| cohort | mask | pg | epg | pacc | ap | iou |
|---|---|---|---|---|---|---|
| prior 500 | all_labeled | 0.8660 | 0.6088 | 0.7671 | 0.7628 | 0.4757 |
| prior 500 | target | 0.7700 | 0.5303 | 0.7640 | 0.6799 | 0.4233 |
| all 12419 | all_labeled | 0.8685 | 0.6196 | 0.7624 | 0.7714 | 0.4807 |
| all 12419 | target | 0.7812 | 0.5463 | 0.7603 | 0.6950 | 0.4327 |

## Prior versus additional images

| partition | mask | variant | pg | epg | pacc | ap | iou |
|---|---|---|---|---|---|---|---|
| additional11919 | all_labeled | baseline | 0.8158 | 0.5553 | 0.7351 | 0.7453 | 0.4679 |
| additional11919 | all_labeled | full | 0.8686 | 0.6200 | 0.7622 | 0.7718 | 0.4809 |
| additional11919 | all_labeled | gcls | 0.8155 | 0.5553 | 0.7351 | 0.7453 | 0.4679 |
| additional11919 | all_labeled | literal_full | 0.1959 | 0.1938 | 0.4912 | 0.3372 | 0.0635 |
| additional11919 | all_labeled | prior_only | 0.8719 | 0.6260 | 0.7645 | 0.7768 | 0.4842 |
| additional11919 | target | baseline | 0.7360 | 0.4873 | 0.7271 | 0.6722 | 0.4193 |
| additional11919 | target | full | 0.7817 | 0.5469 | 0.7601 | 0.6956 | 0.4331 |
| additional11919 | target | gcls | 0.7357 | 0.4873 | 0.7271 | 0.6722 | 0.4193 |
| additional11919 | target | literal_full | 0.1582 | 0.1567 | 0.5249 | 0.2875 | 0.0507 |
| additional11919 | target | prior_only | 0.7850 | 0.5524 | 0.7624 | 0.7004 | 0.4361 |
| prior500 | all_labeled | baseline | 0.8240 | 0.5391 | 0.7322 | 0.7324 | 0.4562 |
| prior500 | all_labeled | full | 0.8660 | 0.6088 | 0.7671 | 0.7628 | 0.4757 |
| prior500 | all_labeled | gcls | 0.8240 | 0.5391 | 0.7322 | 0.7324 | 0.4562 |
| prior500 | all_labeled | literal_full | 0.1920 | 0.1886 | 0.5000 | 0.3295 | 0.0631 |
| prior500 | all_labeled | prior_only | 0.8700 | 0.6146 | 0.7692 | 0.7672 | 0.4787 |
| prior500 | target | baseline | 0.7320 | 0.4668 | 0.7246 | 0.6531 | 0.4039 |
| prior500 | target | full | 0.7700 | 0.5303 | 0.7640 | 0.6799 | 0.4233 |
| prior500 | target | gcls | 0.7340 | 0.4668 | 0.7246 | 0.6531 | 0.4039 |
| prior500 | target | literal_full | 0.1460 | 0.1463 | 0.5347 | 0.2742 | 0.0482 |
| prior500 | target | prior_only | 0.7800 | 0.5357 | 0.7659 | 0.6841 | 0.4261 |

The additional 11,919 images were not used to choose the refinement. The original 500 and all their artifacts remain preserved.

## Paper reference

| evaluation | pg | epg | pacc | ap | iou |
|---|---|---|---|---|---|
| paper reference, original full evaluation | 0.7585 | 0.3800 | 0.6341 | 0.7657 | 0.4465 |

Dataset size now matches the paper description, but the method adds min-max spatial normalization and the paper's exact mask/threshold/aggregation protocol remains unresolved. Similar numbers do not establish an exact reproduction.

## Integrity and reproducibility

Raw patch maps saved: 62,095; per-image method/protocol rows: 124,190. Zero maps: 19; constant maps: 19. These flags are retained in the metrics, not filtered silently.
Original target absent from full annotation: 1227; absent after crop: 1243. These samples were retained.
Every mask was checked pixel-for-pixel against the official archive. All official IDs, source class labels, image/mask dimensions and pinned mirror shard hashes were verified. The image bytes remain from the disclosed unofficial mirror.
The first 20 newly generated maps match the saved refinement smoke maps within FP32 tolerance. All recorded source/metadata hashes and all 62,095 saved patch maps were checked.
All 5,000 repeated method/protocol rows for the original 500 images match the prior saved scores within numerical tolerance; maximum differences are recorded in artifact_audit.json.
Inference/evaluation wall time (including checkpointing and first-20 panels): 1318.5 seconds. Statistical aggregation time is additional.

Configuration: config_refined.yaml. Frozen full-run procedure: FULL_EVALUATION_PLAN.md.
Detailed outputs: outputs/refined_all/{run.json,protocol_results.csv,summary.json,all_protocol_intervals.json,controlled_comparisons.json,artifact_audit.json}.
Commands: `python scripts/prepare_full_dataset.py`, `python scripts/run_full_evaluation.py`, `python scripts/report_full_evaluation.py`. Add `--resume` to resume verified per-image checkpoints.
