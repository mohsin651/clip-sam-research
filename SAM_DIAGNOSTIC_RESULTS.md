# Saved-artifact SAM reliability findings

Completed diagnostic analysis of all 500 frozen images / 1,000 targets on 2026-09-23. No CLIP or SAM inference, no deployable gate, no new candidate selection. All old output files were hash-verified unchanged.

69 inference-time features were extracted before joining GT outcomes. Outcome: top-3/SAM-score IoU minus CLIP IoU on the same crop; 560 improved, 440 worsened. GT IoUs, areas, groups and semantic labels were excluded from predictors; IDs were used only for joins and image grouping.

## Strongest signals

| Signal | Improvement association | Exploratory univariate AUC |
|---|---|---:|
| Attribution density inside selected mask / outside | Higher is better | 0.7776 |
| Raw attribution density inside | Higher is better | 0.7272 |
| Selected mask fraction of image | Smaller is better | 0.6231 |
| Mean top-3 point spread / image diagonal | More concentrated is better | 0.5889 |
| Mean candidate IoU agreement | Higher is better | 0.5848 |
| Selected attribution coverage | Higher is modestly better | 0.5746 |
| SAM maximum predicted score | Almost chance | 0.5173 |

AUC directions above are oriented using this full cohort and are descriptive, not held-out classifier results. Correlated area features and raw/log density transforms are not independent findings. Log density contrast has Spearman +0.4628 with delta IoU and Cohen d=1.1128 between improved/worsened groups.

Prompt Pearson similarity barely predicts improvement (raw AUC 0.4911), but lower similarity modestly predicts semantic success (AUC 0.6268). Thus semantic reliability and incremental segmentation benefit are different questions.

## Fixed diagnostic models, grouped five-fold CV

| Model | OOF AUC | AUC 95% CI | Balanced accuracy | Precision | Recall |
|---|---:|---|---:|---:|---:|
| Logistic regression | 0.8160 | [0.7890, 0.8419] | 0.7429 | 0.7825 | 0.7518 |
| Shallow tree | 0.7823 | [0.7513, 0.8115] | 0.7410 | 0.7744 | 0.7661 |

Both prompts from each image remain in the same fold. All 69 permitted features, fixed hyperparameters, no feature selection or search. Preprocessing is fitted on training folds only. CIs are conditional on OOF predictions and omit refitting uncertainty. No final full-data model or gate was fitted/applied.

## Unattainable GT oracles

| Analysis | mIoU |
|---|---:|
| CLIP baseline | 0.2846 |
| Always top-3 SAM, score selection | 0.4005 |
| Always top-3 SAM, coverage selection | 0.3947 |
| Oracle SAM-score mask or CLIP | 0.4665 |
| Oracle among three top-3 SAM candidates | 0.5021 |
| Oracle among those candidates and CLIP | 0.5324 |

Candidate selection has more measured headroom (+0.1016) than fallback alone (+0.0660); together the ceiling is +0.1319 over current SAM. These gains overlap and cannot be added. Even the candidate oracle remains below IoU 0.5 on 472/1,000 pairs, so proposal/prompt/model and category-union limitations remain.

## Proposed next experiment — not implemented

Use this cohort only for development. Preregister a fresh image-disjoint evaluation of a compact post-SAM reliability scorer, with a 2x2 comparison of candidate selection and CLIP fallback. Freeze features, fitting and calibration on development data only. Compare always-SAM, fallback-only, selector-only, combined, and CLIP on untouched images; report IoU, degradation frequency/magnitude and wrong-object rates, not just AUC.

The strongest features require SAM masks. They can decide which output to keep after SAM, but cannot avoid SAM computation through a pre-SAM gate.

- [Full twelve-question report](outputs/sam_diagnostics/DIAGNOSTIC_REPORT.md)
- [Frozen diagnostic plan](SAM_DIAGNOSTICS_PLAN.md)
- [All feature comparisons](outputs/sam_diagnostics/feature_summary.csv)
- [Cross-validation results](outputs/sam_diagnostics/predictive_results.json)
- [Oracle estimates and confidence intervals](outputs/sam_diagnostics/oracle_results.json)
- [24 diagnostic example panels](outputs/sam_diagnostics/examples.html)
- [Density-contrast plot](outputs/sam_diagnostics/plots/consistency_log_density_ratio.png)

This remains exploratory evidence from a previously examined cohort. No new segmentation method was implemented.
