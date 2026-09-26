# Frozen COCO multi-object attribution findings

Completed on 2026-09-23. The ImageNet refinement source, checkpoint and parameters were unchanged. No SAM was implemented or run.

Official COCO 2017 validation: 5,000 images verified, 36,781 annotations decoded without errors. Of 2,925 crop-visible eligible images, seed 42 selected 500 images and two categories per image: 1,000 target pairs, 76 categories. Selection preceded attribution inference.

| Metric | Mean | Image-cluster bootstrap 95% CI |
|---|---:|---|
| PG | 0.7110 | [0.6810, 0.7400] |
| EPG | 0.3931 | [0.3782, 0.4079] |
| AP | 0.5690 | [0.5527, 0.5864] |
| mIoU | 0.2846 | [0.2732, 0.2965] |
| Target ratio | 0.6513 | [0.6360, 0.6658] |

Mean attribution energy: 39.3% target, 20.0% other annotated categories, 40.7% background/unannotated. Target energy exceeds distractor energy in 69.0% of pairs. Its per-pixel energy density exceeds distractor density in 90.0%.

Both targets gain energy under their own versus the other prompt in 84.2% of images (95% CI 80.8–87.4%). All 500 maps change under prompt switching; mean Pearson correlation is 0.1405. This supports text-conditioned behavior, but map changes alone are not proof of correctness.

| Frozen diagnostic rule | Count | Percent |
|---|---:|---:|
| A: PG=1, target ratio>0.5, IoU<0.5 | 398 | 39.8% |
| B: all other wrong/unconfirmed cases | 455 | 45.5% |
| C: PG=1, target ratio>0.5, IoU>=0.5 | 147 | 14.7% |

The joint target-point/energy criterion succeeds on 54.5% of pairs. There is evidence of correct but coarse localization in many scenes, but the claim that remaining errors are merely boundary errors is too strong. B includes 95 distractor peaks, 194 background peaks and 166 target peaks whose energy is dominated by distractors. The latter can include small objects correctly highlighted with substantial spill. Overlapping COCO category masks and target size also affect energy-based diagnostics.

18 tests passed; zero numerical failures, zero constant/empty attribution maps. All saved-map scores were recomputed and source/checkpoint hashes verified. Inference and evaluation took 32.8 seconds; full preparation/reporting stage timings are in the detailed report.

- [Full results, confidence intervals and runtimes](outputs/coco_dense/RESULTS.md)
- [Frozen procedure and selection criteria](COCO_DENSE_PLAN.md)
- [Visual example gallery](outputs/coco_dense/examples.html)
- [Representative visual review and limitations](outputs/coco_dense/VISUAL_REVIEW.md)
- [Per-target measurements](outputs/coco_dense/per_target_results.csv)
- [Prompt-switch measurements](outputs/coco_dense/prompt_switch_results.csv)

Data source: [official COCO downloads](https://cocodataset.org/#download). All raw maps, selected IDs and the 500 paired-prompt panels are saved under outputs/coco_dense/.
