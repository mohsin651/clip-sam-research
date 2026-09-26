# Smart SAM: prospective held-out result

The frozen nine-feature linear candidate ranker plus density fallback generalizes to 500 new COCO images / 1,000 target pairs. Final mIoU is **0.4335**, versus **0.3949** for naive top-three/SAM-score selection: paired gain **+0.0386**, 95% CI **[+0.0274, +0.0497]**. Harm falls from **41.1% to 26.2%**. All previous experiments are preserved.

## Development ablation (old 500 images, out-of-fold)

| Method | mIoU | Harm | Semantic success |
|---|---:|---:|---:|
| CLIP | 0.2846 | 0.0% | 68.5% |
| SAM score, always SAM | 0.4005 | 44.0% | 75.8% |
| SAM score + logistic fallback | 0.4114 | 13.1% | 75.2% |
| Smart selector, always SAM | 0.4419 | 35.8% | 77.0% |
| Smart selector + logistic fallback | 0.4383 | 10.4% | 76.3% |
| **Smart selector + density fallback (chosen)** | **0.4465** | **26.0%** | **77.1%** |

The density fallback was chosen for highest development mIoU. Logistic fallback was more conservative but sacrificed useful improvements. Five outer image folds and four inner folds kept both prompts and all candidates grouped by image. Development estimates remain exploratory because these images informed prior feature design.

## Frozen method

Original refined CLIP attribution -> unchanged three positive points -> unchanged SAM ViT-B candidates -> standardized pairwise logistic ranker -> use SAM if log density ratio >= **0.72536122868084263**, otherwise use the exact original CLIP binary mask. Neither CLIP nor SAM was trained. Feature definitions, coefficients, thresholds, checkpoints and source hashes were frozen before selecting the new cohort. Heldout seed: 2026; image overlap: zero.

## Held-out comparison

| Method | mIoU | Harm vs CLIP | Semantic success | Precise wrong-object proxy |
|---|---:|---:|---:|---:|
| CLIP | 0.2699 | 0.0% | 64.0% | 1.5% |
| SAM score | 0.3949 | 41.1% | 73.2% | 7.8% |
| Coverage | 0.3776 | 40.6% | 68.1% | 10.6% |
| Density ratio | 0.3907 | 39.7% | 75.6% | 6.0% |
| Smart selector | 0.4291 | 34.1% | 73.4% | 8.3% |
| SAM score + logistic fallback | 0.3963 | 12.7% | 72.0% | 4.5% |
| Smart + logistic fallback | 0.4196 | 10.3% | 72.6% | 4.4% |
| **Frozen smart + density fallback** | **0.4335** | **26.2%** | **73.7%** | **7.4%** |
| Center point / SAM score | 0.1019 | 87.7% | 35.2% | 10.4% |

Smart selection alone adds **0.0342** mIoU over naive SAM [0.0236, 0.0444]. Frozen fallback adds **0.0044** over smart selection [0.0006, 0.0082]. Final absolute mIoU CI: **[0.4136, 0.4542]**. Harm reduction versus naive SAM: **14.9 percentage points**, CI **[12.4, 17.4]**. Final improved/unchanged/worsened fractions are **63.7% / 10.1% / 26.2%**. Worst-decile mean delta improves from **-0.3315 to -0.2221**.

The semantic-success change versus naive SAM is only +0.5 points [-1.2, +2.2]; wrong-object change is -0.4 points [-1.5, +0.7]. Neither establishes improvement. Wrong-object rate remains substantially higher than CLIP's 1.5%. This is a segmentation-quality and harm-reduction result, with semantic limitations still present.

## Held-out subgroups

| Frozen CLIP group | Pairs | CLIP mIoU | Naive SAM mIoU | Final mIoU | Naive harm | Final harm |
|---|---:|---:|---:|---:|---:|---:|
| A: correct target / coarse mask | 363 | 0.2950 | 0.4876 | 0.5245 | 32.0% | 20.9% |
| B: wrong or unconfirmed target | 486 | 0.1439 | 0.2442 | 0.2615 | 49.4% | 32.7% |
| C: already-good localization | 151 | 0.6149 | 0.6572 | 0.7683 | 36.4% | 17.9% |

Group A retains its large benefit, and Group C suffers less harm than under naive SAM. Groups are GT-derived post-hoc diagnostics and never enter inference.

## Held-out oracles

| Analysis-only upper bound | mIoU |
|---|---:|
| Best of three candidates | 0.4878 |
| Perfect fallback after frozen smart selection | 0.4681 |
| Perfect fallback after naive SAM-score selection | 0.4526 |
| Best candidate + perfect fallback | 0.5181 |

The final-to-combined-oracle gap is **0.0846**. Smart selection chooses a GT-best candidate in **58.1%** of pairs versus **43.9%** for SAM score. Candidate-choice headroom remains larger than fallback headroom, but **49.2%** of pairs have no candidate reaching IoU 0.5. Better decision rules alone cannot resolve every failure.

## Evidence and reproduction

- [Complete heldout report](outputs/smart_sam/heldout/HELDOUT_REPORT.md): all metrics, paired intervals, harm, subgroups and ten research answers.
- [Development report](outputs/smart_sam/development/DEVELOPMENT_REPORT.md): grouped CV, gate classification metrics and all alternatives.
- [Frozen method](outputs/smart_sam/frozen/FROZEN_METHOD.md), [parameters](outputs/smart_sam/frozen/model_files/models.json), [integrity hashes](outputs/smart_sam/frozen/integrity.json).
- [24 representative panels](outputs/smart_sam/heldout/examples.html): selector fixes, correct fallback rejections, selector failures, rejected useful masks, successful dense-scene masks and precise wrong-object failures.
- [Reproduction notes](SMART_SAM_REPRODUCTION.md), [protocol](SMART_SAM_PLAN.md), [final audit](outputs/smart_sam/audit.json).

41 tests passed. All old artifacts and frozen hashes verified; decisions replayed without GT; all 1,000 heldout pairs retained. No test-based tuning or method substitution occurred. Confidence intervals use 2,000 whole-image paired bootstrap samples and are not multiplicity-adjusted.

This is promising evidence for a proposed small semantic candidate selector plus fallback, with demonstrated mIoU and harm benefits on one prospective dense-scene cohort. It does not establish improved semantic identity selection or solve inadequate candidate generation. Broader comparisons and datasets remain necessary for stronger paper claims.
