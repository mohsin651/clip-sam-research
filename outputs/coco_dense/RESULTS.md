# Frozen COCO multi-object localization results

Under the documented joint target-point/energy criterion, 54.5% of pairs select the requested category; 39.8% fall in the coarse-mask group. PG alone is 71.1%, and target energy exceeds distractor energy in 69.0% of pairs. Both requested categories gain energy under their own prompt in 84.2% of images. A majority meets the joint selection criterion, with the coarse-mask group identifying potential boundary-refinement cases. These diagnostics do not demonstrate that SAM would solve the remaining errors. Group A may include partial coverage as well as boundary spill; group B includes background misses and ambiguous energy preference, not only confirmed wrong-object selections. No SAM was run.

## Main results

| Metric | Result |
|---|---|
| Number of images | 500 |
| Number of image-target pairs | 1000 |
| PG | 0.7110 [0.6810, 0.7400] |
| EPG | 0.3931 [0.3782, 0.4079] |
| AP | 0.5690 [0.5527, 0.5864] |
| mIoU | 0.2846 [0.2732, 0.2965] |
| Mean target ratio | 0.6513 [0.6360, 0.6658] |

Intervals: 2,000 bootstrap resamples of images, preserving both target prompts; seed 2026. PG/EPG/AP/IoU are per-target averages, not official COCO detection AP.

## Dataset and selection

Official validation images verified: 5,000. Images with at least two nonempty original categories: 3927. Crop-visible eligible images: 2925. Selected with seed 42: 500; exactly two targets each.
Decoded annotations: {'decoded': 36781, 'empty': 0, 'crowds': 446, 'malformed': 0}. Selected targets cover 76 categories; mean annotated categories per image: 3.76; mean instance annotations: 8.95.
Each eligible target union has at least 502 crop pixels and retains at least 25% of its resized uncropped area. Valid crowd masks are included. All categories remain distractors even if too small to qualify as targets. Selected IDs and all selection decisions were saved before inference.

## Target versus distractor

| Metric | Mean [95% CI] |
|---|---|
| target_energy_share | 0.3931 [0.3782, 0.4079] |
| distractor_energy_share | 0.2000 [0.1878, 0.2125] |
| background_energy_share | 0.4070 [0.3877, 0.4258] |
| target_preferred | 0.6900 [0.6650, 0.7160] |
| uniform_target_ratio | 0.4296 [0.4167, 0.4424] |
| ratio_above_uniform | 0.9000 [0.8820, 0.9170] |
| density_ratio | 0.7825 [0.7729, 0.7914] |
| target_density_greater | 0.9000 [0.8820, 0.9170] |

Target/distractor energies are sums of raw positive attribution. Distractors are the union of other categories minus target pixels, preventing double counting at overlap. Inclusive other-category and overlap energies are also saved. Unannotated/background pixels contribute to total energy but not the target ratio denominator. Thus target ratio measures preference among annotated objects and can be inflated by target size; uniform-area and density diagnostics expose this limitation.
Annotation overlap: 811 pairs have some target/other-category mask overlap; 127 have more than half of target pixels overlapping another category. The primary disjoint convention assigns overlap to the requested target, so nested/supporting-object annotations can make this diagnostic easier. The original category masks are preserved without correction.

## Prompt switching

| Metric | Mean [95% CI] |
|---|---|
| pearson | 0.1405 [0.1130, 0.1678] |
| cosine | 0.4137 [0.3939, 0.4331] |
| unit_energy_l1 | 1.1961 [1.1673, 1.2259] |
| mean_own_minus_cross_energy_share | 0.2296 [0.2162, 0.2440] |
| both_targets_switch_correctly | 0.8420 [0.8080, 0.8740] |
| maps_changed | 1.0000 [1.0000, 1.0000] |
Undefined Pearson correlations (constant maps): 0. Bidirectional success requires each category's energy share under its own prompt to exceed that under the other prompt by >1e-8. A changing heatmap alone is not evidence of correct switching.

## Objective diagnostic groups

| Group | Count | Percent | Fraction [95% CI] |
|---|---|---|---|
| A_correct_target_coarse_mask | 398 | 39.8000 | 0.3980 [0.3700, 0.4270] |
| B_wrong_or_unconfirmed_target | 455 | 45.5000 | 0.4550 [0.4270, 0.4830] |
| C_good_localization | 147 | 14.7000 | 0.1470 [0.1260, 0.1700] |

C: PG=1, target_ratio>0.5 and IoU>=0.5. A: PG=1 and target_ratio>0.5 but IoU<0.5. B: all remaining pairs, including wrong-object and unconfirmed localization. Rules were frozen before inference; these are diagnostic proxies, not manually verified semantic labels.

| Subtype | Count | Percent |
|---|---|---|
| target_preferred | 545 | 54.5000 |
| peak_in_background | 194 | 19.4000 |
| target_peak_distractor_energy | 166 | 16.6000 |
| peak_in_distractor | 95 | 9.5000 |

## Visual examples

[Browse all selected examples](examples.html). Random, best (PG then IoU), worst, and highest-confusion (lowest target ratio) sets each contain 20 pairs; groups may overlap. Exact selection is in example_manifest.json. All 500 paired-prompt panels are in prompt_switch/.
Representative examples below are selected by distance to median IoU within each nonempty diagnostic group, with stable ID tie breaks:

### A_correct_target_coarse_mask: dining table
PG 1; IoU 0.3284; target ratio 0.6233.
![dining table](examples/representative/000000575970_67.png)
[Two-prompt comparison](prompt_switch/000000575970.png)

### B_wrong_or_unconfirmed_target: hot dog
PG 1; IoU 0.1167; target ratio 0.2917.
![hot dog](examples/representative/000000513567_58.png)
[Two-prompt comparison](prompt_switch/000000513567.png)

### C_good_localization: cat
PG 1; IoU 0.5993; target ratio 0.8748.
![cat](examples/representative/000000416330_17.png)
[Two-prompt comparison](prompt_switch/000000416330.png)

## Runtime and integrity

| Stage | Seconds |
|---|---|
| download_and_verification_seconds | 129.6472 |
| selection_seconds | 11.1690 |
| inference_and_evaluation_seconds | 32.7531 |
| reporting_and_visualizations_seconds | 26.9059 |
| total_pipeline_seconds | 200.4753 |
Numerical failures: 0. Zero maps: 0; constant maps: 0; retained in scores. All 1000 saved maps, selected IDs, frozen source/config/checkpoint hashes and annotation hashes verified. Existing ImageNet experiment sources were not changed.
Pipeline runtime excludes implementation, environment repair and manual inspection. Download/verification, selection, inference/evaluation and report/panel generation times are included above.

## Reproduce

`python scripts/prepare_coco.py` → `python scripts/select_coco_dense.py` → `python scripts/run_coco_dense.py` → `python scripts/report_coco_dense.py`.
Selection refuses to overwrite its frozen manifest. Inference supports `--resume` with signature checks. Install requirements-coco.txt into the existing pinned environment. Full protocol: ../../COCO_DENSE_PLAN.md (project root).
Dataset source: [official COCO downloads](https://cocodataset.org/#download). Archive hashes and image verification are in data/coco/provenance.json. Per-target metrics, prompt-switch metrics, all maps and manifest files are retained in this output folder.
