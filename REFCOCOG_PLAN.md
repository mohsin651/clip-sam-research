# Frozen CASR RefCOCOg evaluation protocol

Evaluation only. CASR is an internal shorthand for the existing frozen positive-only CLIP-aware SAM refinement; no old files are renamed. No changes to models, tokenizer, attribution, top-three point placement, SAM, features, ranker, density threshold, crop, preprocessing or geometry. No training, refitting, negative points, prompt rewriting, exclusions by performance or post-hoc method repair.

## Release and cohort

Use the original REFER RefCOCOg research archive and refs(umd).p. The official REFER README recommends UMD train/val/test; Google lacks a released test split. Evaluate the complete UMD test only, never artificial testA/testB. Keep every raw expression. Verify annotation IDs against full official COCO annotation sets and decode targets before inference. Verify images, dimensions and hashes. Any genuine frozen-tokenizer/component incompatibility stops the experiment without filtering or truncation.

Record source URLs, archive and extracted-file SHA256 and actual release split counts. Use original COCO train2014 RGB via the unchanged image acquisition helper, reusing already verified images. No different dataset substitute.

Before inference compare image IDs with COCO development, COCO heldout, RefCOCO negative-point VAL development, RefCOCO tests and RefCOCO+ tests. Save exact intersections and their union. Standard test retains overlap. A separate strict subset excludes the union; report counts and intervals if nonempty, flag small sample limitations, and never fabricate empty-subset results. Strict and standard are nested, not independent replications. Prior exposure is not model fitting, but must be disclosed.

## Frozen inference and protection

The adapter imports run_refcoco_frozen.infer_one directly. Verify original COCO freeze and earlier RefCOCO/RefCOCO+ inference source signatures. Preserve all previous output files with before/after hashes. Inference receives only image_id, sent_id, expression and image_path; annotation masks/categories, GT outcomes and subgroup labels do not enter inference. Write only outputs/refcocog_frozen and data/refcocog, plus new code/notes. Reused images are not overwritten.

Exactly 20 expressions from the deterministic image-ID/sentence-ID order form a technical smoke stage: loading, tokenization, geometry, attribution, mask saving/reloading and feature/selection replay only; no aggregate GT performance. Reuse valid predictions in full inference. Save all candidates, raw maps, points, decisions, signatures and timings. Evaluation is a separate process after full inference and hash verification. Resume requires exact signatures; completed runs refuse overwrite.

## Metrics and statistics

Five unchanged methods: CLIP baseline, top-three/SAM-score, smart selector, frozen final (CASR), center-point/SAM-score. Primary full original-image referred-instance expression-macro IoU. Secondary Dice, precision, recall, pixel accuracy, P@0.5 and P@0.7. Secondary unchanged CLIP-crop metrics are separate and retain crop-invisible targets with existing zero-score convention.

Reuse existing instance metrics and lexicons. Correct-instance for multiple non-crowd same-category cases requires strictly greater target IoU than every competitor; ties fail. Single-instance cases reported separately. Preserve all expressions including unavailable/poor maps and fallback cases.

Use 2,000 paired image bootstrap resamples, seed 2026, keeping expressions grouped and expression-macro weighting. No multiplicity adjustment. Report all standard and strict scores and naive-minus-CLIP, smart-minus-naive, final-minus-naive, final-minus-CLIP, final-minus-smart paired contrasts (IoU, P@0.5, harm versus CLIP, correct-instance). Report ambiguous-only correct-instance contrasts separately if single-instance cases exist.

Harm: fractions improved/worsened/unchanged versus CLIP, mean/median delta, conditional positive/negative means and worst-decile mean. Report direct smart/final versus naive separately. Existing tolerance and definitions remain unchanged.

Length bins unchanged: short <=3, medium 4-6, long >=7 words; report counts, IoU, both P@ thresholds, correct-instance, harm and overall mean/median word count. Spatial/attribute/color/clothing use exactly existing word-boundary lexicons. Groups are descriptive and confounded, not causal language effects.

Post-hoc oracles only: best of three candidates, none at IoU >=0.5, oracle-minus-smart regret, candidate+CLIP and selected+CLIP oracles; fallback prevents harm/rejects useful/accepts harm. None affects a decision.

## Reporting and visual selection

Write all requested JSON/CSV artifacts, complete REFCOCOG_RESULTS.md, root summary and reproduction notes. Cross-dataset table uses saved dense COCO/RefCOCO/RefCOCO+ values as descriptive context; crops, targets and language differ.

Select three distinct images nearest median final IoU in each of ten diagnostic groups, sentence-ID tie breaks: long success/failure; naive wrong to smart correct; fallback prevents harm; fallback rejects useful; correct/wrong same-category instance; spatial success/failure; CLIP IoU >=0.5 but all SAM candidates below 0.5. Success requires final IoU >=0.5 and correct-instance, failure IoU <0.2. Disclose empty groups and selection manifests. Panels show raw expression, original image, attribution, points, every candidate, both choices, final/fallback, GT and IoUs. Outcome-selected panels do not measure prevalence.

Answer all thirteen research questions with actual evidence and limitations. Stop after RefCOCOg. No external baseline, new dataset, method modification or automatic GitHub push.
