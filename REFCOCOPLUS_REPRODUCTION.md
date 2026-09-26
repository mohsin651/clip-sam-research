# RefCOCO+ frozen evaluation reproduction notes

The new RefCOCO+ adapter imports run_refcoco_frozen.infer_one directly; a regression test checks function identity. Original COCO frozen source/checkpoint hashes and all original RefCOCO adapter hashes are checked before inference. All earlier source files are unchanged. Changes are limited to dataset paths, overlap accounting, separate output loops, summaries and visualization groups. No negative-point code is imported by the inference adapter.

Original RefCOCO+ UNC archive: https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco+.zip, listed by https://github.com/lichengunc/refer. The original host does not resolve. An archived original was downloaded; resolved source: https://web.archive.org/web/20220413011656id_/http://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco+.zip . ZIP SHA256: 5f1238112d63199e68da54a28f201471909b21ded7ed79a57d51b4c1443c6b45. Original instance JSON SHA256: 35cfcbe8cd12a7927e7940453ed7d7ea05c5000b0ab1106d1e08eed1b934e441. Original UNC reference pickle SHA256: 7de24c182449758b9d9ae87d968b1f25831236a54d38d45408e2e0fdd318e802. Restricted unpickling rejects executable globals. No alternate dataset is substituted.

Standard testA: 750 images, 1,975 references/targets, 5,726 expressions. Standard testB: 750 images, 1,798 references/targets, 4,889 expressions. Combined: 1,500 images, 3,773 targets, 10,615 original expressions. All valid targets were nonempty and all text tokenized normally. Original raw strings are preserved without rewriting.

Overlap recorded before inference: original COCO development 0; COCO heldout 0; RefCOCO negative-point VAL development 0; RefCOCO testA/testB 1,500. The strict union-excluded cohort is EMPTY, for both testA and testB. Strict scores are not estimable. This is evaluation of new dataset expressions on previously evaluated images, not image-disjoint generalization. No prior experiment trained the frozen method on these test images; nevertheless, prior exposure is disclosed.

Images reuse data/refcoco/images by original COCO train2014 filename; dimensions, decode and SHA256 are verified. The RefCOCO+ original annotations are kept separately under data/refcocoplus. Competitor annotation coverage is verified against complete official COCO train2017+val2017 ID sets; original RefCOCO+ masks remain the GT source. Shared image files are not modified.

Predeclared protocol: REFCOCOPLUS_PLAN.md. New outputs: outputs/refcocoplus_frozen/. Primary original-image instance metrics and secondary crop metrics use the old evaluation functions and lexicons. Exactly five systems: CLIP, naive SAM, smart, frozen final and unchanged center control. All unavailable/crop-invisible/poor predictions remain. A strict empty cohort produces no fabricated rows or intervals.

```powershell
python scripts/download_refcocoplus.py
python scripts/prepare_refcocoplus.py
python -m pytest -q -p no:cacheprovider --basetemp=<new writable temporary directory>
python scripts/run_refcocoplus.py --smoke
python scripts/run_refcocoplus.py --resume
python scripts/evaluate_refcocoplus.py
python scripts/report_refcocoplus.py
python scripts/visualize_refcocoplus.py
python scripts/audit_refcocoplus.py
```

Use the existing pinned virtual environment. 56 tests passed before smoke. Smoke uses exactly 20 expressions, verifies saved masks/geometry/feature decisions without GT aggregate scoring, and is reused in full inference. Evaluation requires full completion and verifies prediction hashes before opening GT. Existing completed runs refuse overwrite; resume checks hashes. Reproduction requires a separately configured destination/copy to preserve this completed run.

Source/config/input signatures, all old output hashes, new prediction hashes and raw expressions are retained. Inference wall time includes initial prior-file preservation hashing, checkpoint checks, imports/loading, saving and smoke; not pure GPU time. Final audit/report/visualization are additional stages. Datasets/weights/large arrays remain local and excluded from Git.

No RefCOCOg run is authorized by these notes. Stop after RefCOCO+ for user review.

## Completed run and final audit

All 10,615 expressions / 1,500 images / 3,773 referred targets completed. Primary final mIoU 0.3215518426; smart 0.3224268243; naive 0.3015644776. Full results: REFCOCOPLUS_RESULTS.md and outputs/refcocoplus_frozen/REFCOCOPLUS_RESULTS.md.

Final audit passed: all 79,966 previously recorded output files unchanged, all 22,730 prediction files hash-verified, original frozen components and adapter sources verified, all selection/fallback decisions replayed without GT, raw expressions preserved, zero exclusions. Center-control masks reproduce the earlier same-image RefCOCO masks bitwise, with scores matching within 1e-6. Strict cohort remains empty and no strict metric rows were fabricated. All 30 panels exist; two panels were visually inspected for layout and overlays. 56 tests passed before inference.

Inference including smoke, hashing, loading and saving: 1290.2 seconds. Evaluation including prediction verification: 215.8 seconds. These exclude download, implementation, reporting, visualizations and final audit. Zero/constant maps: 0/0; empty candidates: 0. Exactly 20 technical smoke predictions reused without aggregate GT scoring.

The audit is outputs/refcocoplus_frozen/audit.json. Human summary and handoff notes were updated after completion; frozen inference/evaluation source files were not edited. New artifacts have not been pushed to GitHub. No RefCOCOg execution or negative-point follow-up occurred.
