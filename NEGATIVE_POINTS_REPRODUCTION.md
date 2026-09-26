# Negative-point experiment: reproduction notes

This adds an explicitly exploratory RefCOCO VAL development experiment. Frozen prior code, checkpoints, ranker, fallback and all older results remain unchanged. The current protocol is in NEGATIVE_POINTS_PLAN.md, sealed by the new inference signature before the smoke run. TestA/testB receive no new inference.

Changes: a metadata-only cohort builder; three inference-only negative-support rules; one/two-negative SAM calls retaining the original positive coordinates; separate post-hoc instance evaluation, grouped statistics, point-hit diagnostics, oracle comparisons and visualizations. Existing CLIP/SAM functions, candidate features, selectors, fallback and instance metrics are imported unchanged. Positive-point feature calculations still receive ONLY the original positives; negative points do not silently change learned feature definitions.

NEG_LOW adds a fixed 3x3 grid of SAM object proposals. This has extra computation and is not identical in cost to POS3. NEG_REGION uses existing POS3 candidates, which may overlap too much to supply a competitor. NEG_CONTRAST uses category-name metadata as an allowed generic text cue and one fixed normalized difference support rule. None guarantees an actual same-category competitor. Post-hoc GT point labels expose this weakness. No labels enter any inference-time choice.

Cohort: seed 2026, 500 images chosen from 1,483 eligible RefCOCO UNC validation images; all 3,689 original expressions and 1,296 targets in selected images retained. Eligibility uses only annotation visibility/instance metadata. Zero prior COCO or RefCOCO test overlap. Complete competitor annotation IDs verified against official COCO train+val2017 annotations. Images downloaded from official COCO train2014 host by the existing verified downloader. Existing original RefCOCO annotation hashes are retained in manifest.json.

Commands, existing Python environment:

```powershell
python scripts/prepare_negative_points.py
python -m pytest -q -p no:cacheprovider --basetemp=<new-writable-temporary-directory>
python scripts/run_negative_points.py --smoke
python scripts/visualize_negative_points.py --technical
# Inspect inferred-region geometry, without GT scoring.
python scripts/run_negative_points.py --resume
python scripts/evaluate_negative_points.py
python scripts/report_negative_points.py
python scripts/visualize_negative_points.py
python scripts/audit_negative_points.py
```

The scripts use a new fixed output directory, outputs/refcoco_negative_points/development. Preparation can resume image acquisition before inference; inference can resume after the exact 20-expression smoke stage with source/input hashes. Completed inference refuses overwriting. To reproduce anew, use a separate copy/output destination and preserve original run signatures. Report generation may create a frozen/ directory only if the predeclared development gate passes; it must not be rerun to overwrite a selected freeze.

Technical environment notes: initial sandboxed pytest ran 50 tests successfully but two older dataset fixtures failed on Windows temporary-directory access. Re-running with authorized filesystem access passed all 52 tests. Initial sandboxed image download preparation stalled under restricted network access and was interrupted; the same saved cohort manifest was reused with authorized network access. No GT performance was computed during these technical corrections and no method parameter changed.

Statistics are exploratory: 2,000 image-grouped resamples, seed 2026, no multiplicity adjustment. Selected development images favor visible ambiguity and are not a population-uniform RefCOCO estimate. Full-image instance metrics retain the original CLIP crop limitation. The old RefCOCO tests were already inspected for the prior method; this task does not claim they are virgin test data. No test rerun is performed.
Evaluation-only clarification before any GT scoring: a negative point on a same-category crowd region, without target/individual-competitor/different-category overlap, is recorded in a separate same_category_crowd category. It is neither an individual-instance competitor nor unannotated background. This does not alter prompts, thresholds, the predeclared continuation gate, or any prediction.
Post-inference reporting correction: the first report attempt split the selector name sam_score into score for a strategy-oracle comparator, creating unmatched rows/NaN and stopping JSON serialization. Corrected to match the complete known selector suffix. The initial script is archived as development/report_negative_points_initial.py; reporting_correction.json preserves initial/corrected hashes. No predictions, metric definitions, bootstrap procedure or continuation criterion changed. The initial analysis_source_hashes.json is retained; the audit verifies this documented correction explicitly.
