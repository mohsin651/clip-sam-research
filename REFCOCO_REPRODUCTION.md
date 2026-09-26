# Frozen RefCOCO transfer: reproduction notes

This task adds a dataset adapter, evaluation and reporting. It does not alter or refit the existing CLIP attribution, SAM, nine-feature ranker or density fallback. Original component verification uses the same frozen integrity manifest as the COCO experiment. No RefCOCO+ or RefCOCOg inference is included.

Data: the original REFER RefCOCO UNC annotation archive, obtained from the Internet Archive snapshot recommended in the official repository's download issue because the original UNC server was unavailable. Archive SHA256 `7f924bb7ed8dc4568058e4ff626281918d56e5206f4c868c5a80f088f38c8bf0`. The original pickle is decoded with a restricted unpickler that rejects executable globals. Only UNC reference records and original instance JSON are extracted; the Google split is not evaluated.

Official testA: 750 images, 1,975 targets, 5,657 expressions. Official testB: 750 images, 1,810 targets, 5,095 expressions. Combined: 1,500 distinct images, 3,785 targets, 10,752 expressions. All original `raw` strings are preserved, including capitalization and colloquial spelling. Normal CLIP tokenization succeeds without truncation or rewriting.

UNC split definitions come directly from each reference's original `split` field. testA is predominantly people: in this exact archive 5,273 of 5,657 testA expressions refer to person annotations, while 384 refer to other categories. testB has 5,095 non-person expressions and no person targets. Do not reinterpret testA as an exclusively-person filter; all official records are retained. The full archive contains 142,210 expressions over 50,000 referred targets in 19,994 images, including training and validation records that are not evaluated or used for fitting here.

Original images are fetched individually from the official COCO train2014 image host and validated by dimensions, decoding and SHA256. No full 13 GB image archive is necessary. Competitor annotation completeness is checked by exact annotation-ID sets against the previously downloaded official COCO train2017 + val2017 annotation archive. This is a coverage check, not a replacement of the referred-instance GT.

No RefCOCO test image overlaps either of the previous 500-image cohorts. The requested strict subset therefore equals the standard combined test. This fact is recorded before predictions; strict results do not provide a second independent replication.

Primary metrics evaluate each specific referred instance in the full original image. Inference retains the old CLIP center crop and the original SAM RGB preprocessing. CLIP/fallback binary masks are inverse-mapped using the existing transform and zero outside the crop. Secondary crop results are explicitly separate. Crop-invisible targets remain in both analyses; secondary empty-GT IoU/Dice are zero, not rewarded as perfect empty predictions. Thus raw mIoU is not directly comparable to the old category-union crop benchmark.

The first exactly 20 sorted image/sentence records form a technical smoke stage. No aggregate GT metrics are calculated during that stage. Their predictions are reused for the complete inference run. The evaluation process requires an `inference_complete` marker, verifies every prediction hash, and only then opens GT. The adapter/protocol source hashes are recorded before the smoke stage. No subsequent model changes or outcome-based reruns are allowed.

Reproduce in a separate output destination, restoring excluded data/checkpoints first:

```powershell
python scripts/download_refcoco_annotations.py
python scripts/prepare_refcoco_frozen.py
python -m pytest -q
python scripts/run_refcoco_frozen.py --smoke
python scripts/run_refcoco_frozen.py --resume
python scripts/evaluate_refcoco_frozen.py
python scripts/report_refcoco_frozen.py
python scripts/visualize_refcoco_frozen.py
python scripts/audit_refcoco_frozen.py
```

The original COCO annotation archive is required by preparation. Existing completed manifests/runs refuse overwrite. Inference resume is hash-checked and intended to continue the 20-sample smoke run, not to select favorable outcomes. Saved masks, scores, features and final decisions are retained per expression. Large image/array files remain local and are excluded from Git.

Outputs: `outputs/refcoco_frozen/REFCOCO_RESULTS.md`, `summary.json`, `paired_comparisons.json`, `per_expression_results.csv`, `instance_analysis.csv`, `expression_analysis.csv`, `overlap_audit.json`, `dataset_provenance.json`, `examples.html` and integrity audits. The protocol and exact post-hoc language lexicons are in `REFCOCO_FROZEN_PLAN.md`. Instance comparisons use every other non-crowd same-category instance, require a strict IoU advantage, and report ambiguous images separately. No-competing-instance cases still require positive target overlap. Confidence intervals use 2,000 image-level paired bootstrap samples, seed 2026.

Runtime interpretation: `run.json` seconds sum smoke and full-run wall time, including the initial preservation-hash audit of roughly 49,000 prior output files, checkpoint verification, model loading, feature computation and saving. This is not pure GPU inference time. The initial preservation audit took several minutes on this Windows filesystem; the final read-only hash audit parallelizes independent file checks. That audit implementation does not enter inference or change any prediction.
