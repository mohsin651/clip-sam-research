# Frozen final cross-product experiment

Read CROSS_ATTRIBUTION_PLAN.md and the eventual outputs/cross_attribution_generalization/CROSS_ATTRIBUTION_REPORT.md. This extension only changes the dataset loop and reporting. It imports the completed CS/GE map adapters and frozen CASR helpers without modifying them.

## Sources and text

- CS: d4696d47f49cfe70f49140afe5eb94f94c5f59bc, CS-ViT-B/16, 512 full-image warp. Official encode_text_with_prompt_ensemble receives the whole expression unchanged and retains all 85 templates; the redundant embedding remains the official ensemble of empty text. Official signed feature subtraction and map interpolation/min-max normalization are unchanged.
- GE: e370e6cb194faf2020f5d1ed268f9d57e91a38e6. Existing load_official executes image notebook method cells verbatim. Raw expression tokenization, last layer n=1, native half precision, full-image patch-compatible preprocessing and native ReLU attribution scale are unchanged.
- Neither official checkout is updated or copied into this repository. Their previous signatures and all tracked/previously signed source bytes are verified.

All expressions fit the original context limit. CS's conservative upper bounds including template/special tokens are 44/36/45 for RefCOCO/+/g, below 77; GE raw bounds are 33/25/34. No truncation, category substitution, text simplification or filtering. Tokenizer-internal behavior is unchanged.

## Downstream geometry and evaluation

The external full-image map is projected to the established 224 crop through the exact prior adapter. The same frozen POS3 points are mapped to the original image, SAM ViT-B sees the original RGB, and three masks are produced. All 18 diagnostic features, the ranker's nine-feature ordering/scaler/coefficients, and log-density fallback threshold 0.72536122868084263 are unchanged. No GT enters those computations. Numeric epsilon and scale conventions are retained.

The source MAP and fallback binary masks use the existing mean threshold inside that crop and are lifted with zero outside. Thus original-image MAP/fallback is crop-limited, preserving the existing referring evaluation. Full-image instance IoU is primary; secondary crop IoU retains empty cropped targets as zero. PG/EPG/AP are retained separately for lifted original-image attribution and the established crop field, as predeclared in the plan; no continuous SAM metrics are invented. The first evaluator draft omitted the planned lifted-map continuous metrics. This reporting omission was corrected before any new GT evaluation; evaluation_protocol_hashes_v2.json signs that correction, and the first hash record is retained. No inference code, features, decisions or binary masks changed.

Each expression has a cached native map, raw crop map, all three packed full-image SAM masks, SAM scores, original text, geometry, points, exact computed features and choices. MAP, NAIVE, SMART and FINAL are all evaluated. Harm always references that source's MAP, never another attribution source. Saved CDA and dense COCO scores are read directly, with no old prediction reruns.

## Run and resume

Use C:/CREMI/trdp/.venv/Scripts/python.exe in the project root. A clean reproduction must use a separate output destination/checkout, since preparation and completed stages refuse overwrite.

1. `python -m pytest -q` (Windows may require an accessible `--basetemp`).
2. `python scripts/prepare_cross_attribution.py` verifies manifests, prior source signatures and model hashes, then hashes all earlier output files.
3. `python scripts/run_cross_attribution_matrix.py` runs exactly 20 technical expressions per source/dataset, then resumes the complete manifest. It uses separate processes and sequential GPU access. Each source/dataset can be resumed with `python scripts/run_cross_attribution.py DATASET SOURCE --resume`; a completed run is never overwritten. Cached maps and prediction hashes are verified before reuse.
4. The matrix runner invokes `evaluate_cross_attribution.py DATASET SOURCE` only after all six inference cells are terminal. Annotation masks are loaded only by this separate evaluation process. Failed cells remain disclosed and are not scored on a favorable partial subset.
5. `python scripts/report_cross_attribution.py` assembles all twelve cells, unchanged official splits and strict subsets, paired 2,000-image-bootstrap CIs, harm/candidate/instance matrices and fixed language subgroups.
6. `python scripts/visualize_cross_attribution.py` writes at most 48 outcome-selected panels with exact selection manifests.
7. `python scripts/audit_cross_attribution.py` verifies protected historical files, every prediction hash, native map projection, POS3 extraction, all feature values and selected/fallback decisions.

`manifest_audit.json` retains exact original manifest hashes/counts and overlap. `freeze_audit.json` signs the new inference adapter plus old official/CASR provenance before inference. Logs and timing are per cell. Caches are per dataset/source; no expression is removed. Statistics are expression-macro, with whole-image resampling, seed 2026, no multiplicity adjustment. RefCOCO+ has no historical strict remaining images; RefCOCOg's strict subset is nested, not an independent replication. All these test cohorts were already inspected.

No new methods, calibration, training or datasets are authorized after this final matrix. Stop for review.
