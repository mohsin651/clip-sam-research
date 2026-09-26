# Saved-artifact SAM reliability diagnostics

Written before extracting feature/outcome associations. Diagnostic analysis only: no CLIP/SAM inference, no deployable gate, no new mask selection, no changes to existing outputs.

## Cohort, labels and leakage boundary

Same 500 images / 1,000 targets. Primary outcome: crop-matched top-3 SAM-score IoU minus saved CLIP baseline IoU. Positive=improved, negative=worsened; exact zeros remain in continuous analysis and are excluded from binary prediction. GT IoUs, category IDs/names, diagnostic groups and semantic labels are never predictors. Feature extraction is a separate module that reads only the GT-free manifest, attribution NPZs, saved prompt coordinates and SAM candidate arrays/scores. IDs join rows and define groups only. Keep features.csv and labels.csv separate; record a strict feature allowlist. Snapshot all baseline and SAM output hashes and verify them at completion.

## Feature definitions fixed before analysis

- CLIP resized 224x224 map: raw maximum/mean/population SD; Shannon entropy of unit-sum positive scores divided by log(224*224); mass at largest 1 and 3 pixels and largest ceil(5%/10%) pixels. Also report top-1/top-3 mass on the native 14x14 patch map to make the scale explicit. Raw magnitude depends on gradient scale.
- Points: use the three saved top-k positive prompts, not new maxima. Original-image pairwise distances, mean/max and distances divided by original image diagonal; enclosing box area in pixels and as image fraction. Also mean/max crop-coordinate distances normalized by crop diagonal. No GT spatial data.
- Connectedness: unchanged mean-attribution threshold, 8-connected components in the crop; count, largest size, largest/positive fraction, positive area fraction, and whether all three crop points lie in the same nonzero component.
- SAM top-3 strategy only: three native scores, ordered high/second/third, top margin, population SD/variance. Pairwise IoUs of candidate masks on the original image and mean/min/max agreement. Three pixel areas and image fractions, selected area/fraction, area variance/SD and fractional variance.
- Consistency: recompute original-coordinate attribution coverage for each candidate using existing geometry and verify saved coverage. Selected coverage, raw inside/outside mean densities, their ratio with denominator epsilon 1e-12, and a log density ratio (same fixed epsilon). Count/fraction of saved positive points inside selected original mask and whether first point is inside. Rasterize points with floor(x+0.5), clamped to image bounds.
- Prompt switch: recompute Pearson, cosine and unit-energy L1 from the two attribution maps only. These three features are shared by the two targets; no target-mask-based switch scores.
- Undefined diagnostics (e.g. zero outside area or constant maps) become missing, with counts disclosed. No outcome-based outlier removal or clipping.

## Analyses and fixed predictive models

For every feature: improved/worsened count, mean, median, sample SD, pooled-SD Cohen d, and 2,000 image-cluster bootstrap CIs of mean difference (seed 2026). Spearman association with continuous delta and raw-direction AUC predicting improvement. AUC direction-reversed magnitude=max(AUC,1-AUC) is a descriptive full-cohort ranking, not a held-out score. Constant features get undefined correlation and AUC=0.5; retain and flag them. No predictive feature selection from these rankings.

Fit exactly two diagnostic models, both with ALL permitted features and no search: logistic regression L2 C=1, class_weight=balanced, lbfgs, max_iter=3000; decision tree max_depth=3, min_samples_leaf=25, class_weight=balanced. Five-fold StratifiedGroupKFold(shuffle=True, random_state=42), grouped by image. Median imputation is fit on each training fold only; LR standardization also training-only. No hyperparameter or decision-threshold search; probability cutoff 0.5. Save all out-of-fold predictions and image fold assignments; report pooled OOF and per-fold AUC, balanced accuracy, precision, recall. Conditional 2,000-image-bootstrap CIs on fixed OOF predictions do not include model-refitting uncertainty. No final full-data fit or gate application.

Three descriptive 2x2 interactions only, using feature medians without looking at outcomes: SAM max score + selected coverage; CLIP entropy + point spread; candidate agreement + selected coverage. Report all cells, counts, mean delta and improvement rates. These are not proposed decision rules.

Ten interpretable plots: entropy, top-10% mass, max SAM score, score margin, mean agreement, normalized point spread, selected coverage, selected area fraction, log density ratio, prompt Pearson versus delta. Scatter plus feature-quantile summaries. Ranking does not change model inputs. Save PNG/PDF artifacts.

## GT-only explanatory/oracle work

After features are frozen, join saved outcomes. Group A/B/C, precise-wrong-object status, recovered-target status and baseline/SAM IoU are explanatory labels only. Examine all feature distributions by these labels. Assess prompt-map similarity versus semantic success/precise-wrong labels separately from improvement; do not claim improvement prediction establishes semantic correctness.

Compute primary-crop oracle means: baseline, SAM-score, coverage, max(baseline,SAM-score), max(top-3 strategy's three candidate GT IoUs), and max(baseline,all three candidate IoUs). Read COCO masks only in this oracle/evaluation stage. Bootstrap image CIs and paired headroom. These are unattainable GT oracles; never write selected masks or deploy a rule. Also report the fraction for which even the candidate oracle stays below IoU 0.5. Gate and candidate-selection headroom overlap and must not be added as independent gains.

Examples: up to three each from high-confidence improve/fail, low-confidence improve/fail, high-agreement improve, low-agreement fail, high-point-spread fail and low-point-spread improve. High/low means label-blind upper/lower feature quartiles; random seed 42 within pools. These panels are explanatory, not method selection. Show GT, original, attribution, points, all candidates, selected mask and fixed IoUs.

Report all results and negative findings. Associations, plots and model interpretation are exploratory on a previously inspected cohort; no confirmatory p-value claims and no superiority claim based on selecting the best CV score. Recommend one next experiment with independent validation, but do not implement it.
