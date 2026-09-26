# RefCOCO negative-point development: STOP

CLIP-derived negative prompting did not provide reliable benefit under the six predeclared rules. Do not add these rules to the proposed method, refit the selector to rescue them, or rerun testA/testB in this task. Preserve the original POS3 method. No negative strategy passed the predeclared continuation gate, so no new method was frozen.

## Cohort and controlled comparison

500 RefCOCO UNC validation images, 3,689 original expressions, 1,296 target instances; seed 2026. Images were selected from 1,483 eligible images using annotation metadata only, then every expression in selected images was retained. Zero overlap with prior COCO cohorts or RefCOCO tests. All 3,689 expressions have same-category competitors, so the dedicated ambiguous subset equals the full development cohort. These development scores must not be compared directly with the earlier 10,752-expression test scores as paired evidence.

CLIP baseline: mIoU 0.1948, correct-instance selection 50.69%. POS3 with SAM-score: 0.2821 / 51.15%; POS3 with smart selection: 0.2955 / 51.53%. The table below uses the SAME frozen smart selector and SAME frozen fallback for every row. All 22 system results, including SAM-score and smart without fallback, are disclosed in the full report.

| Prompting, frozen smart + fallback | mIoU | Correct instance | Wrong instance | P@0.5 | Harm vs CLIP | Candidate oracle |
| --- | --- | --- | --- | --- | --- | --- |
| POS3 | 0.2950 | 51.40% | 47.57% | 22.96% | 32.88% | 0.3628 |
| NEG_LOW1 | 0.2936 | 51.23% | 47.87% | 22.99% | 33.61% | 0.3519 |
| NEG_LOW2 | 0.2858 | 51.10% | 47.93% | 21.28% | 34.18% | 0.3372 |
| NEG_CONTRAST1 | 0.2886 | 50.83% | 48.25% | 22.15% | 34.32% | 0.3523 |
| NEG_CONTRAST2 | 0.2855 | 50.75% | 48.31% | 21.47% | 33.51% | 0.3405 |
| NEG_REGION1 | 0.2925 | 51.12% | 47.82% | 22.74% | 33.56% | 0.3580 |
| NEG_REGION2 | 0.2909 | 51.02% | 48.01% | 22.23% | 33.97% | 0.3541 |

Candidate oracles use the three candidate masks before selection/fallback. Correct-instance selection is the strict target-versus-other-same-category IoU proxy; ties are separate failures.

## Ten development answers

1. **No reliable mIoU improvement.** Every negative system has a lower mean than matched POS3. The closest final variant, NEG_LOW1, changes mIoU by -0.0014, paired 95% CI [-0.0046, +0.0020]. This is not an established gain. All other final variants have negative IoU intervals excluding zero.
2. **No reliable correct-instance improvement.** None of the 18 negative system/selector combinations has a positive lower paired confidence bound for this metric. All six final variants have lower mean accuracy than POS3/final. Even the largest observed increase, NEG_LOW2/SAM-score, is only +0.054 percentage points, CI [-1.026, +1.076].
3. **No strategy qualifies for continuation.** NEG_LOW1/final has the highest negative-variant final mIoU (0.2936), but is still below POS3/final (0.2950). A descriptive least-bad row is not a selected new method.
4. **NEG_CONTRAST does not improve same-category ambiguity.** For one negative with frozen final selection, mIoU change is -0.0064 [-0.0096, -0.0033]; correct-instance change is -0.569 percentage points [-1.363, +0.163]. Two negatives also lower mIoU: -0.0095 [-0.0135, -0.0057]. The ambiguous subset contains every expression.
5. **Competitor-point precision is limited.** Among emitted points, true same-category competitor hits are about 16.8% for NEG_LOW, 37.7-42.2% for NEG_CONTRAST, and 36.4-40.1% for NEG_REGION. These are point-level rates conditional on availability, not accuracy over all expressions.
6. **Bad target cues are frequent for contrast/region rules.** About 4.5-4.6% of NEG_LOW points, 14.1-14.3% of NEG_CONTRAST points, and 29.4-29.8% of NEG_REGION points hit the target. A hit is an erroneous negative cue, not proof that SAM necessarily removed that pixel. Point and expression denominators are retained separately.
7. **Every fixed negative strategy worsens candidate-oracle quality.** All six paired candidate-oracle IoU intervals are strictly below zero. Thus these negative prompts fail to improve the average quality of their three candidates, independently of selector mistakes. Pooling POS3 and all negative candidates gives an oracle of 0.3962 versus POS3's 0.3628; this shows some complementary masks exist, but uses GT to choose among more options and is not a deployable gain.
8. **No consistent expression-subgroup benefit.** For final systems, spatial-expression mIoU falls under every negative variant (POS3 0.2292; negative variants 0.2235-0.2285). Long expressions show a small NEG_LOW2 mean increase, 0.1982 to 0.2016, but its correct-instance interval spans zero. Attribute/color/clothing results contain isolated tiny gains and many losses; none justifies changing the overall STOP decision. All subgroup rows and paired intervals are saved; no subgroup was used to tune.
9. **Final-system harm increases for every rule.** POS3/final harm is 32.88%; negatives range 33.51-34.32%. NEG_CONTRAST1 increases harm by 1.437 percentage points [0.530, 2.393]. NEG_LOW2 and both NEG_REGION variants also have positive harm intervals excluding zero. NEG_LOW1 and NEG_CONTRAST2 harm intervals include zero. Direct improvement/harm relative to matched POS3 is separately reported.
10. **STOP.** The proposed negative mechanisms do not have a large or consistent benefit and do not meet the predeclared joint gate. This is evidence against these specific simple rules, not a proof that every possible negative-prompting method must fail. No formula revision, refitting, new freeze, or test run followed these results.

## Negative-point accuracy and availability

| Rule | Available expressions | Unavailable | Emitted points | True competitor | Target hit | Different category | Background |
| --- | --- | --- | --- | --- | --- | --- | --- |
| NEG_LOW1 | 98.51% | 1.49% | 3,634 | 16.76% | 4.54% | 14.78% | 63.73% |
| NEG_LOW2 | 95.55% | 4.45% | 7,050 | 16.82% | 4.58% | 15.93% | 62.34% |
| NEG_CONTRAST1 | 81.24% | 18.76% | 2,997 | 42.24% | 14.15% | 14.28% | 29.00% |
| NEG_CONTRAST2 | 75.63% | 24.37% | 5,580 | 37.71% | 14.27% | 13.41% | 34.00% |
| NEG_REGION1 | 16.24% | 83.76% | 599 | 36.39% | 29.38% | 11.35% | 22.20% |
| NEG_REGION2 | 13.26% | 86.74% | 978 | 40.08% | 29.75% | 9.82% | 19.73% |

Point classes above leave a small separately recorded same-category crowd fraction, so displayed columns need not sum to 100%. Exclusive class priority is target, individual same-category competitor, different-category, same-category crowd, background. Overlapping membership flags are saved. If fewer than the requested one/two negatives existed, the exact POS3 masks and decisions were used; no expressions were dropped.

NEG_LOW explicitly used SAM proposals rather than the global minimum attribution pixel. Nevertheless, most chosen points fall on background/unannotated regions. NEG_REGION's low availability reflects the limited spatial distinctness of the three existing POS3 candidates. These are observed shortcomings of the implemented rules, not grounds for an outcome-driven redesign in this experiment.

## Candidate oracle paired evidence

| Strategy vs POS3 | Oracle IoU difference | Image-bootstrap 95% CI |
| --- | --- | --- |
| NEG_LOW1 | -0.0109 | [-0.0142, -0.0077] |
| NEG_LOW2 | -0.0256 | [-0.0306, -0.0209] |
| NEG_CONTRAST1 | -0.0105 | [-0.0135, -0.0077] |
| NEG_CONTRAST2 | -0.0223 | [-0.0267, -0.0180] |
| NEG_REGION1 | -0.0049 | [-0.0070, -0.0029] |
| NEG_REGION2 | -0.0087 | [-0.0114, -0.0060] |

2,000 paired whole-image bootstrap resamples, seed 2026, expression-macro weighting. Intervals are exploratory and not multiplicity-adjusted. The metadata-selected cohort favors visible ambiguity. No random-negative or matched-compute control was added; isolated recoveries therefore cannot establish semantic specificity of the negative cue.

## Evidence, examples and continuity

- [Full development report: all 22 systems and statistics](outputs/refcoco_negative_points/development/DEVELOPMENT_REPORT.md)
- [24 representative panels across all eight requested groups](outputs/refcoco_negative_points/development/examples.html)
- [Exact point rules and predeclared decision](NEGATIVE_POINTS_PLAN.md)
- [Reproduction and technical changes](NEGATIVE_POINTS_REPRODUCTION.md)
- [Paired comparisons](outputs/refcoco_negative_points/development/paired_comparisons.csv), [subgroups](outputs/refcoco_negative_points/development/subgroup_results.csv), [point diagnostics](outputs/refcoco_negative_points/development/diagnostic_summary.csv), [oracle results](outputs/refcoco_negative_points/development/oracle_results.json)

Example groups use objective post-hoc overlap criteria, not exhaustive manual semantic labels. Counts in example_group_counts.json count expression-strategy rows, not distinct images or expressions. Images within each displayed group are distinct; groups may overlap. Successful examples coexist with the negative average result and do not overturn it.

Exactly 20 technical smoke samples preceded complete inference. All 3,689 expressions and 81,158 method rows were retained; zero/constant maps: 0/0; empty candidates: 0. All 52 tests passed. Inference-stage wall time including smoke, hashing, loading and saving: 1,025.6 seconds; evaluation: 216.4 seconds. These exclude implementation and final reporting/audit.

A reporting-only bug in the oracle comparator name (sam_score parsed as score) was corrected after inference; initial source and both hashes are preserved in reporting_correction.json. No prediction, metric definition, bootstrap design or continuation criterion changed. No earlier experiment was overwritten. Final preservation and geometry verification is recorded in outputs/refcoco_negative_points/development/audit.json. New results are local; not yet pushed to GitHub.
Final audit PASSED: all 72,528 prior output files unchanged; all 7,378 new prediction files verified; all 20,838 emitted negative points passed region-membership, boundary-distance, separation and coordinate checks; every selector/fallback decision replayed. All 8,100 unavailable strategy-expression cases matched POS3 exactly. Zero new test predictions. No negative-method frozen directory was created.
