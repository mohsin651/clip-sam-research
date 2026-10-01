# Frozen CASR RefCOCOg reproduction notes

CASR is the existing positive-only CLIP-aware SAM refinement. This task adds only dataset routing, integrity checks, evaluation/reporting and visualizations for RefCOCOg. Original source files, model weights, tokenizer, attribution, points, geometry, features, fitted ranker and density threshold remain unchanged. run_refcocog.py imports run_refcoco_frozen.infer_one directly; tests assert function identity. Negative-point code is not used.

The original REFER RefCOCOg archive contains both refs(google).p and refs(umd).p. UMD is chosen before inference because the research API recommends its train/val/test convention; Google has no released test. Original source: https://github.com/lichengunc/refer . The unavailable UNC host is accessed through an archived original; exact requested/resolved URLs and hashes are retained in dataset_provenance.json. ZIP SHA256: 3d1f7e5b2ff2205940bf59de55f861f5f2cc1403fb980669933a7f9af1aa8211. No alternate referring-expression dataset is substituted.

UMD test: 2,600 images, 9,602 raw expressions, 5,023 referred targets. Before inference, overlap was 0 COCO development, 0 COCO heldout, 39 negative-point VAL development, 118 RefCOCO test and 118 RefCOCO+ test images. The last two overlap sets coincide. Their union has 157 images; strict remaining cohort is 2,443 images / 8,894 expressions / 4,648 targets. Standard retains overlap, strict excludes all five prior cohorts. These nested cohorts are not independent replications.

Image acquisition reuses the unchanged helper and data/refcoco/images by original COCO train2014 filename. New annotations remain separately in data/refcocog. Images are verified by dimensions, decoding and SHA256. Restricted unpickling prevents executable pickle globals; target masks are decoded and competitor annotation IDs checked against full official COCO annotation ID sets. The stripped inference manifest contains only image_id, sent_id, expression and image_path. Every expression passes the unchanged tokenizer; no truncation, rewriting or exclusion.

## Commands

Use the existing pinned environment at C:/CREMI/trdp/.venv (Python 3.11.16). On this session its uv launcher junction was missing again; restored cpython-3.11-windows-x86_64-none to the already installed cpython-3.11.16-windows-x86_64-none. No packages or weights changed.

```powershell
python scripts/download_refcocog.py
python scripts/prepare_refcocog.py
python -m pytest -q -p no:cacheprovider --basetemp=<new writable temporary directory>
python scripts/run_refcocog.py --smoke
python scripts/run_refcocog.py --resume
python scripts/evaluate_refcocog.py
python scripts/report_refcocog.py
python scripts/visualize_refcocog.py
python scripts/audit_refcocog.py
```

Exactly 20 technical smoke expressions precede full inference without aggregate GT scoring. Valid smoke predictions are reused. Frozen original source/checkpoint hashes and prior adapter signatures are verified. Inference finishes and prediction hashes are checked before evaluation accesses GT. Complete runs must not be overwritten; use a separately configured destination for reproduction. Resume checks exact signatures.

Metrics, image bootstrap and language lexicons are imported unchanged. 2,000 paired image resamples, seed 2026, expression-macro weighting; no multiplicity adjustment. Only actual UMD test is evaluated. Oracles, fallback error labels and GT panels are post-hoc diagnostics. Primary full-image and secondary crop metrics remain separate. All saved predictions remain in denominators.

No external baselines, model changes, new datasets or GitHub push follow automatically. Stop for review after this evaluation. Large datasets, weights and NPZ arrays remain local and excluded from Git.

## Completed evaluation and preservation audit

Completed every test expression: 9,602 expressions, 2,600 images, 5,023 targets; strict 8,894 expressions and 2,443 images. Final standard mIoU 0.3472, strict 0.3492. Complete metrics, paired intervals, language groups, oracles and all thirteen research answers are in outputs/refcocog_frozen/REFCOCOG_RESULTS.md; human interpretation is REFCOCOG_RESULTS.md. Saved row counts: 96,020 method rows (five methods, two frames).

61 tests passed. Exactly 20 technical smoke predictions were reused. Zero exclusions, zero/constant attribution maps 0/0 and empty candidates 0; 128 expressions with crop-invisible targets remain. UMD train/val/test image sets were checked disjoint before full inference.

Final audit passed: all 102,753 previously recorded output files unchanged; all 21,804 new prediction files hash-verified; frozen original components and adapter source signatures verified; every candidate-selection and fallback decision replayed without GT. All raw expressions preserved. All 118 overlapping-image center-control masks match the earlier RefCOCO masks bitwise, with predicted scores within 1e-6. Audit file: outputs/refcocog_frozen/audit.json.

Thirty panels (three per each of ten predeclared diagnostic groups) were generated; three were manually inspected for readable expression, overlays, prompts and candidate/selection labels. Complete group sizes and sample IDs are retained. Examples are outcome-selected and do not estimate prevalence.

Inference including smoke, preservation hashing, imports/loading and saving: 1567.4 seconds. Evaluation including prediction/source verification: 164.1 seconds. Preparation was 144.2 seconds; download, implementation, tests, reporting, visualization and final audit are additional. No pure-GPU speed claim. New adapters and reporting choices were frozen before the smoke stage, with no post-inference method changes.

Environment-only recovery: the first command failed before creating files because the uv Python launcher junction was absent. Restoring the documented junction to the existing Python 3.11.16 resolved it. No experiment implementation failure or performance-based repair followed.

Human summary and START_HERE/README were updated. Old notes/results are preserved. New RefCOCOg artifacts have not been pushed to GitHub. No external baseline or additional dataset was run after RefCOCOg; CASR remains unchanged. Stop for user review.
