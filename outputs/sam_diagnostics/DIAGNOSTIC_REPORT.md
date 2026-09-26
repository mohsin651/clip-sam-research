# Diagnostic analysis of saved CLIP + SAM results

**Diagnostic only: no CLIP/SAM inference, no deployed gate, no new mask-selection method.** Same 500 images / 1,000 target pairs; 560 improve, 440 worsen, zero ties. Delta IoU uses the original top-3/SAM-score result on the common CLIP crop.

## Answers to the twelve research questions


1. Strongest improvement signal: higher selected-mask attribution density inside versus outside (raw or log ratio), univariate AUC 0.7776 and Spearman +0.4628. Raw/log forms are the same ordering, not independent findings. Higher inside density (AUC 0.7272), smaller candidate-mask area and lower area variability also show useful associations.

2. Degradation is associated with lower inside/outside attribution density, larger selected/candidate masks, more scattered prompt points and weaker candidate agreement. These are associations, not reliable rejection rules; no threshold was optimized.

3. Scattered top-3 points are a modest warning: normalized mean spread is 0.2359 in worsened versus 0.2061 in improved cases; Spearman −0.1845, direction-reversed AUC 0.5889. Scattering alone is insufficient.

4. SAM predicted quality is a weak predictor of incremental benefit: maximum score AUC 0.5173, score-margin AUC 0.5402. The improved-minus-worsened mean-score and margin intervals include zero. A geometrically confident mask can still be the wrong object or worse than the existing CLIP mask.

5. Candidate agreement helps modestly: mean pairwise IoU is 0.5243 for improvements versus 0.4671 for degradations; AUC 0.5848. Agreement can also mean all candidates agree on the same wrong region.

6. Attribution coverage alone is weak (AUC 0.5746). Density relative to the outside region is substantially more informative. Coverage rewards large masks, whereas density accounts for mask area; large masks can collect much energy without isolating the intended object.

7. Prompt sensitivity barely predicts improvement: Pearson raw-direction AUC 0.4911 and Spearman +0.0164. It has a different, modest association with semantics: lower Pearson predicts semantic success with AUC 0.6268; higher Pearson predicts precise-wrong-object cases with AUC 0.6078. Improvement and semantic correctness are distinct outcomes.

8. Yes, improvement is predictable above chance within grouped CV: logistic OOF AUC 0.8160 [0.7890, 0.8419], shallow-tree AUC 0.7823 [0.7513, 0.8115]. These fixed diagnostic models use all 69 features. No feature selection, tuning, full-cohort refit or deployed gate was performed.

9. Unattainable GT SAM-or-CLIP oracle mIoU: 0.4665, compared with always-SAM 0.4005 and CLIP 0.2846. Perfect fallback therefore has +0.0660 mean-IoU headroom over the current selected mask.

10. Unattainable GT candidate-selection oracle mIoU: 0.5021, versus SAM-score 0.4005 and attribution coverage 0.3947. Perfect candidate choice has +0.1016 headroom. Combining candidate choice with CLIP fallback gives 0.5324, +0.1319 over always-SAM.

11. The largest measured single opportunity is WHICH candidate (+0.1016), larger than WHEN to retain CLIP (+0.0660); their combination has the highest ceiling. These headrooms overlap and must not be added. Even the candidate oracle has IoU<0.5 for 472/1000 pairs, leaving prompt/proposal quality and category-union versus instance limitations; this analysis cannot isolate which causes the residual.

12. Proposed next experiment, not implemented: a preregistered 2×2 study of candidate selection and fallback, with this cohort used only for development and a fresh image-disjoint COCO cohort for final evaluation. Freeze a compact scorer using density contrast, candidate area, point spread and agreement; fit/calibrate only on development data. Compare always-SAM, fallback-only, selector-only and combined, plus CLIP. Report mean IoU, paired gains, harm rate/magnitude and precise-wrong-object rate. Choose any thresholds solely on development data and never revise them after held-out results. AUC alone does not optimize mean IoU: large harms and small gains have different utility.

## Feature signals

| feature | spearman_delta | auc_higher_predicts_improved | direction_free_auc | improvement_direction | cohen_d | mean_difference | difference_ci_low | difference_ci_high |
|---|---|---|---|---|---|---|---|---|
| consistency_log_density_ratio | 0.4628 | 0.7776 | 0.7776 | higher | 1.1128 | 0.8734 | 0.7684 | 0.9802 |
| area_selected_fraction | -0.2266 | 0.3769 | 0.6231 | lower | -0.5150 | -0.0959 | -0.1231 | -0.0691 |
| spatial_spread_mean_norm | -0.1845 | 0.4111 | 0.5889 | lower | -0.3088 | -0.0298 | -0.0422 | -0.0173 |
| agreement_mean | 0.1721 | 0.5848 | 0.5848 | higher | 0.2962 | 0.0572 | 0.0342 | 0.0826 |
| clip_top10pct_mass | 0.1732 | 0.5806 | 0.5806 | higher | 0.2604 | 0.0342 | 0.0187 | 0.0497 |
| consistency_selected_coverage | 0.1162 | 0.5746 | 0.5746 | higher | 0.2268 | 0.0459 | 0.0189 | 0.0727 |
| clip_entropy | -0.1247 | 0.4361 | 0.5639 | lower | -0.1844 | -0.0061 | -0.0102 | -0.0023 |
| sam_score_margin | 0.0954 | 0.5402 | 0.5402 | higher | 0.1225 | 0.0041 | -0.0002 | 0.0084 |
| sam_score_max | 0.1164 | 0.5173 | 0.5173 | higher | 0.0858 | 0.0045 | -0.0022 | 0.0113 |
| switch_pearson | 0.0164 | 0.4911 | 0.5089 | lower | -0.0011 | -0.0003 | -0.0421 | 0.0388 |

All univariate results are exploratory full-cohort associations, not out-of-sample classifier performance. Direction-free AUC flips orientation using the same cohort and is descriptive. Many features are correlated or equivalent transforms; ranking does not establish independent effects. CIs resample images with both prompts; no multiplicity-adjusted significance claims are made.

### Improved versus worsened descriptive statistics

| feature | improved_mean | worsened_mean | improved_median | worsened_median | improved_std | worsened_std |
|---|---|---|---|---|---|---|
| clip_entropy | 0.9189 | 0.9250 | 0.9223 | 0.9311 | 0.0330 | 0.0337 |
| clip_top10pct_mass | 0.4871 | 0.4529 | 0.4747 | 0.4363 | 0.1326 | 0.1299 |
| spatial_spread_mean_norm | 0.2061 | 0.2359 | 0.1846 | 0.2227 | 0.0933 | 0.1005 |
| sam_score_max | 0.9277 | 0.9232 | 0.9373 | 0.9348 | 0.0504 | 0.0557 |
| sam_score_margin | 0.0386 | 0.0344 | 0.0281 | 0.0239 | 0.0345 | 0.0330 |
| agreement_mean | 0.5243 | 0.4671 | 0.5033 | 0.4429 | 0.1926 | 0.1939 |
| area_selected_fraction | 0.1341 | 0.2300 | 0.0801 | 0.1348 | 0.1450 | 0.2282 |
| consistency_selected_coverage | 0.4233 | 0.3773 | 0.3981 | 0.3406 | 0.1978 | 0.2085 |
| consistency_log_density_ratio | 1.9524 | 1.0791 | 1.9550 | 1.1613 | 0.6895 | 0.8916 |
| switch_pearson | 0.1403 | 0.1407 | 0.0537 | 0.1053 | 0.3237 | 0.3041 |

Every one of the 69 features, including pairwise distances/scores/agreements, is retained in feature_summary.csv and correlations.csv. Definitions and units are in ../../SAM_DIAGNOSTICS_PLAN.md; feature_schema.json is the exact predictor allowlist. IDs and all GT columns are excluded. No missing or constant features occurred.

## Simple diagnostic predictors

| model | roc_auc | balanced_accuracy | precision | recall | AUC 95% CI |
|---|---|---|---|---|---|
| logistic_regression | 0.8160 | 0.7429 | 0.7825 | 0.7518 | [0.7890, 0.8419] |
| shallow_tree | 0.7823 | 0.7410 | 0.7744 | 0.7661 | [0.7513, 0.8115] |

Five StratifiedGroupKFold splits by image, seed 42. Both target prompts stay together. Median imputation and LR scaling are fit on training folds only. LR: L2 C=1, balanced class weights. Tree: depth 3, minimum leaf 25, balanced class weights. All 69 features, fixed probability threshold 0.5, no search. OOF predictions and per-fold metrics are saved. Precision/recall refer to the improved class (prevalence 56%).
The image-bootstrap CIs are conditional on saved OOF predictions and omit model-refitting uncertainty. This previously examined cohort is exploratory, not an independent final validation set. No gate-selected IoU or training-set predictor score is reported. No final model was fitted on all examples.
Many strong signals require SAM candidate masks, so they can decide whether to retain a SAM output only AFTER SAM runs. They cannot save SAM computation as a pre-SAM gate. Pure CLIP-map and point features are available earlier.

## Three fixed, interpretable interactions

| feature_a | feature_b | median_a | median_b | a_high | b_high | n | mean_delta | improved_fraction | delta_ci_low | delta_ci_high |
|---|---|---|---|---|---|---|---|---|---|---|
| sam_score_max | consistency_selected_coverage | 0.9360 | 0.3715 | False | False | 330 | 0.0763 | 0.5091 | 0.0445 | 0.1092 |
| sam_score_max | consistency_selected_coverage | 0.9360 | 0.3715 | False | True | 170 | 0.1017 | 0.6412 | 0.0706 | 0.1334 |
| sam_score_max | consistency_selected_coverage | 0.9360 | 0.3715 | True | False | 170 | 0.1570 | 0.5059 | 0.0997 | 0.2120 |
| sam_score_max | consistency_selected_coverage | 0.9360 | 0.3715 | True | True | 330 | 0.1414 | 0.5970 | 0.1131 | 0.1698 |
| clip_entropy | spatial_spread_mean_norm | 0.9259 | 0.2041 | False | False | 278 | 0.1813 | 0.6727 | 0.1440 | 0.2168 |
| clip_entropy | spatial_spread_mean_norm | 0.9259 | 0.2041 | False | True | 222 | 0.1072 | 0.5495 | 0.0756 | 0.1398 |
| clip_entropy | spatial_spread_mean_norm | 0.9259 | 0.2041 | True | False | 222 | 0.1499 | 0.5856 | 0.1055 | 0.1928 |
| clip_entropy | spatial_spread_mean_norm | 0.9259 | 0.2041 | True | True | 278 | 0.0302 | 0.4353 | -0.0003 | 0.0587 |
| agreement_mean | consistency_selected_coverage | 0.4767 | 0.3715 | False | False | 290 | 0.0573 | 0.4552 | 0.0190 | 0.0945 |
| agreement_mean | consistency_selected_coverage | 0.4767 | 0.3715 | False | True | 210 | 0.0930 | 0.5429 | 0.0618 | 0.1227 |
| agreement_mean | consistency_selected_coverage | 0.4767 | 0.3715 | True | False | 210 | 0.1680 | 0.5810 | 0.1244 | 0.2137 |
| agreement_mean | consistency_selected_coverage | 0.4767 | 0.3715 | True | True | 290 | 0.1531 | 0.6621 | 0.1238 | 0.1825 |

High/low partitions use feature medians without looking at labels. All cells are shown. Low entropy + concentrated points improves in 67.3% of cases versus 43.5% for high entropy + scattered points. High agreement + high coverage improves in 66.2% versus 45.5% for low/low. High confidence + high coverage reaches 59.7%, providing no universal safe region. These descriptive partitions are not a gate.

## GT oracles: unattainable, never deployable

| analysis | mIoU | 95% CI |
|---|---|---|
| baseline | 0.2846 | [0.2732, 0.2965] |
| always_sam_score | 0.4005 | [0.3790, 0.4217] |
| always_attribution_coverage | 0.3947 | [0.3757, 0.4144] |
| oracle_sam_or_clip | 0.4665 | [0.4483, 0.4854] |
| oracle_candidate_selection | 0.5021 | [0.4820, 0.5240] |
| oracle_candidate_or_clip | 0.5324 | [0.5145, 0.5523] |

| headroom | mean | ci_low | ci_high |
|---|---|---|---|
| oracle_sam_or_clip minus always_sam_score | 0.0660 | 0.0592 | 0.0730 |
| oracle_candidate_selection minus always_sam_score | 0.1016 | 0.0902 | 0.1130 |
| oracle_candidate_or_clip minus always_sam_score | 0.1319 | 0.1207 | 0.1433 |
| oracle_candidate_or_clip minus oracle_sam_or_clip | 0.0659 | 0.0576 | 0.0746 |
| oracle_candidate_or_clip minus oracle_candidate_selection | 0.0303 | 0.0259 | 0.0350 |

Candidate oracle refers to the three candidates from the existing top-3 prompt strategy, not a search over other prompt strategies. GT candidate IoUs were computed only after freezing features and only for this analysis. No oracle-selected mask was written. Oracle and fallback gains overlap.

## Failure-type explanations, GT labels only

| group | feature | n | mean | median | std |
|---|---|---|---|---|---|
| C_good_localization | spatial_spread_mean_norm | 147 | 0.2042 | 0.1887 | 0.0887 |
| C_good_localization | sam_score_max | 147 | 0.9451 | 0.9538 | 0.0435 |
| C_good_localization | agreement_mean | 147 | 0.5477 | 0.5162 | 0.1884 |
| C_good_localization | consistency_selected_coverage | 147 | 0.5728 | 0.5964 | 0.1981 |
| C_good_localization | consistency_log_density_ratio | 147 | 1.9919 | 1.9673 | 0.5915 |
| C_good_localization | switch_pearson | 147 | 0.0315 | -0.0625 | 0.3443 |
| B_wrong_or_unconfirmed_target | spatial_spread_mean_norm | 455 | 0.2375 | 0.2228 | 0.1035 |
| B_wrong_or_unconfirmed_target | sam_score_max | 455 | 0.9133 | 0.9228 | 0.0601 |
| B_wrong_or_unconfirmed_target | agreement_mean | 455 | 0.4666 | 0.4455 | 0.1905 |
| B_wrong_or_unconfirmed_target | consistency_selected_coverage | 455 | 0.3514 | 0.3016 | 0.1974 |
| B_wrong_or_unconfirmed_target | consistency_log_density_ratio | 455 | 1.2962 | 1.3576 | 0.9194 |
| B_wrong_or_unconfirmed_target | switch_pearson | 455 | 0.2023 | 0.1475 | 0.3155 |
| A_correct_target_coarse_mask | spatial_spread_mean_norm | 398 | 0.2038 | 0.1849 | 0.0902 |
| A_correct_target_coarse_mask | sam_score_max | 398 | 0.9328 | 0.9395 | 0.0427 |
| A_correct_target_coarse_mask | agreement_mean | 398 | 0.5185 | 0.4937 | 0.1970 |
| A_correct_target_coarse_mask | consistency_selected_coverage | 398 | 0.3994 | 0.3844 | 0.1781 |
| A_correct_target_coarse_mask | consistency_log_density_ratio | 398 | 1.7226 | 1.8098 | 0.8668 |
| A_correct_target_coarse_mask | switch_pearson | 398 | 0.1101 | 0.0326 | 0.2874 |
| precise_wrong_object | spatial_spread_mean_norm | 71 | 0.2423 | 0.2336 | 0.1053 |
| precise_wrong_object | sam_score_max | 71 | 0.9268 | 0.9377 | 0.0456 |
| precise_wrong_object | agreement_mean | 71 | 0.5108 | 0.5126 | 0.1753 |
| precise_wrong_object | consistency_selected_coverage | 71 | 0.4753 | 0.4565 | 0.2178 |
| precise_wrong_object | consistency_log_density_ratio | 71 | 1.5188 | 1.4941 | 0.8053 |
| precise_wrong_object | switch_pearson | 71 | 0.2785 | 0.2378 | 0.3667 |
| recovered_target | spatial_spread_mean_norm | 87 | 0.1954 | 0.1695 | 0.0884 |
| recovered_target | sam_score_max | 87 | 0.9298 | 0.9377 | 0.0414 |
| recovered_target | agreement_mean | 87 | 0.4668 | 0.4234 | 0.1888 |
| recovered_target | consistency_selected_coverage | 87 | 0.3124 | 0.2930 | 0.1464 |
| recovered_target | consistency_log_density_ratio | 87 | 2.0405 | 1.9981 | 0.6203 |
| recovered_target | switch_pearson | 87 | 0.1839 | 0.1252 | 0.3139 |
| improved | spatial_spread_mean_norm | 560 | 0.2061 | 0.1846 | 0.0933 |
| improved | sam_score_max | 560 | 0.9277 | 0.9373 | 0.0504 |
| improved | agreement_mean | 560 | 0.5243 | 0.5033 | 0.1926 |
| improved | consistency_selected_coverage | 560 | 0.4233 | 0.3981 | 0.1978 |
| improved | consistency_log_density_ratio | 560 | 1.9524 | 1.9550 | 0.6895 |
| improved | switch_pearson | 560 | 0.1403 | 0.0537 | 0.3237 |
| worsened | spatial_spread_mean_norm | 440 | 0.2359 | 0.2227 | 0.1005 |
| worsened | sam_score_max | 440 | 0.9232 | 0.9348 | 0.0557 |
| worsened | agreement_mean | 440 | 0.4671 | 0.4429 | 0.1939 |
| worsened | consistency_selected_coverage | 440 | 0.3773 | 0.3406 | 0.2085 |
| worsened | consistency_log_density_ratio | 440 | 1.0791 | 1.1613 | 0.8916 |
| worsened | switch_pearson | 440 | 0.1407 | 0.1053 | 0.3041 |

These GT group labels never enter the model matrix. Full distributions for all features are in failure_feature_summary.csv. Group B remains wrong OR unconfirmed, not exclusively wrong-object cases. Recovered means the previous frozen recovery proxy; precise-wrong is the previous other-category IoU proxy.

### Prompt sensitivity and semantics

| feature | label | auc_higher_predicts_label | mean_label_true | mean_label_false | difference_ci_low | difference_ci_high |
|---|---|---|---|---|---|---|
| switch_pearson | semantic_success | 0.3732 | 0.1054 | 0.2505 | -0.1919 | -0.0992 |
| switch_pearson | precise_wrong_object | 0.6078 | 0.2785 | 0.1299 | 0.0636 | 0.2374 |
| switch_cosine | semantic_success | 0.3761 | 0.3894 | 0.4898 | -0.1328 | -0.0654 |
| switch_cosine | precise_wrong_object | 0.6331 | 0.5192 | 0.4056 | 0.0561 | 0.1720 |
| switch_l1 | semantic_success | 0.6215 | 1.2307 | 1.0875 | 0.0912 | 0.1915 |
| switch_l1 | precise_wrong_object | 0.3571 | 1.0424 | 1.2078 | -0.2479 | -0.0792 |

Semantic-success and precise-wrong labels are explanatory outcomes only. Shared per-image prompt similarities are cluster-bootstrapped by image. Lower map similarity supports modest semantic discrimination here, but does not predict the relative IoU gain from SAM.

## Plots

Ten predeclared interpretable scatter plots with image-bootstrap quintile summaries; PNG and PDF versions are saved in plots/. No plot-guided feature/model tuning was performed.

![Density contrast](plots/consistency_log_density_ratio.png)
![SAM predicted quality](plots/sam_score_max.png)
![Point spread](plots/spatial_spread_mean_norm.png)
![Candidate agreement](plots/agreement_mean.png)

## Representative examples

[Browse all eight diagnostic categories](examples.html). Three distinct random examples per category, seed 42, using feature upper/lower quartiles without label-based threshold optimization. Selection within each diagnostic outcome category is illustrative; panels include GT only for post-hoc explanation.

## Reproducibility

`python scripts/sam_diagnostic_features.py`; `python scripts/analyze_sam_diagnostics.py`; `python scripts/sam_diagnostic_oracles.py`; `python scripts/report_sam_diagnostics.py`.
Feature calculations never parse GT annotations or outcome CSVs. Protected outputs are also read as bytes for integrity hashing only. features.csv contains identifiers and allowed features only; labels.csv contains outcomes. The model matrix uses feature_schema.json explicitly. Baseline and SAM output hashes are verified before and after. No old output was overwritten.
Method references: [scikit-learn grouped folds](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html) and [training-only preprocessing guidance](https://scikit-learn.org/stable/common_pitfalls.html).
